//! Quantization on the CPU: the layouts the GPU kernels read. Weights are converted once, when a model is pulled;
//! the same code is the reference the GPU tests check the kernels against.
//!
//! **Q4 weights.** A `rows × cols` matrix, row-major, `cols` a multiple of [`GROUP`]. Each group of 64 weights
//! along a row shares one f16 scale `d`; weight `w ≈ (n − 8)·d` with `n` in 0..=15. Packed: 32 bytes per group,
//! as 8 words of 4 bytes; word `m` holds weights `8m..8m+8`, byte `b` of it weight `8m+b` in the low nibble and
//! `8m+4+b` in the high one. Masking a word with `0x0F0F0F0F` then gives four weights that line up with four
//! consecutive activation bytes, ready for `dp4a`. Scales are a separate `rows × cols/64` array.
//!
//! **Q8 activations.** A vector, a multiple of [`BLOCK`] long, as `i8` values and, per block of 32, the pair
//! `[s, s·Σq]` with `x ≈ q·s`. The sum lets a kernel apply the weights' zero point of 8 once per block.
//!
//! **Sources.** [`Safetensors`] reads the checkpoints models ship as, rows at a time, as `f32`.

use crate::json::Json;
use crate::wpk::Dtype;
use crate::{Error, Result};
use std::os::unix::fs::FileExt;

/// Weights per Q4 scale.
pub const GROUP: usize = 64;
/// Activations per Q8 scale.
pub const BLOCK: usize = 32;

/// A Q4 matrix: packed weights and their f16 scales (as bits).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Q4 {
    /// Two weights per byte, in the layout described above.
    pub packed: Vec<u8>,
    /// One f16 per group of [`GROUP`] weights.
    pub scales: Vec<u16>,
}

/// Quantizes a row-major matrix whose rows are `cols` long.
///
/// # Panics
/// When `cols` is not a multiple of [`GROUP`], or `w` not a whole number of rows.
pub fn q4(w: &[f32], cols: usize) -> Q4 {
    assert!(cols % GROUP == 0 && w.len() % cols == 0, "rows are a whole number of {GROUP}-weight groups");
    let mut packed = vec![0u8; w.len() / 2];
    let mut scales = Vec::with_capacity(w.len() / GROUP);
    for (g, group) in w.chunks_exact(GROUP).enumerate() {
        let scale = f16_bits(group.iter().fold(0f32, |m, v| m.max(v.abs())) / 7.0);
        let d = f16_value(scale);
        let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
        for (j, v) in group.iter().enumerate() {
            let n = ((v * inv).round_ties_even() as i32).clamp(-8, 7) + 8;
            packed[g * 32 + j / 8 * 4 + j % 4] |= (n as u8) << (j % 8 / 4 * 4);
        }
        scales.push(scale);
    }
    Q4 { packed, scales }
}

/// Quantizes a vector to `i8` values and per-block `[s, s·Σq]`.
///
/// # Panics
/// When the vector is not a multiple of [`BLOCK`] long.
pub fn q8(x: &[f32]) -> (Vec<i8>, Vec<[f32; 2]>) {
    assert!(x.len() % BLOCK == 0, "a whole number of {BLOCK}-value blocks");
    let (mut q, mut s) = (Vec::with_capacity(x.len()), Vec::with_capacity(x.len() / BLOCK));
    for block in x.chunks_exact(BLOCK) {
        let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
        let (scale, inv) = (amax / 127.0, if amax > 0.0 { 127.0 / amax } else { 0.0 });
        let start = q.len();
        q.extend(block.iter().map(|v| (v * inv).round_ties_even() as i8));
        let sum: i32 = q[start..].iter().map(|&v| i32::from(v)).sum();
        s.push([scale, sum as f32 * scale]);
    }
    (q, s)
}

/// The IEEE half-precision bits nearest to `v`, ties to even.
pub fn f16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    let (sign, exp, man) = ((b >> 16) & 0x8000, ((b >> 23) & 0xff) as i32, b & 0x7f_ffff);
    let round = |kept: u32, shift: u32, bits: u32| {
        let (rem, half) = (bits & ((1 << shift) - 1), 1 << (shift - 1));
        kept + u32::from(rem > half || (rem == half && kept & 1 == 1))
    };
    let magnitude = match exp - 112 {
        _ if exp == 255 => 0x7c00 | u32::from(man != 0) << 9,
        31.. => 0x7c00,
        1..=30 => round(((exp - 112) as u32) << 10 | man >> 13, 13, man),
        -10..=0 => round((man | 0x80_0000) >> (126 - exp), (126 - exp) as u32, man | 0x80_0000),
        _ => 0,
    };
    (sign | magnitude) as u16
}

/// The value of half-precision bits `h`.
pub fn f16_value(h: u16) -> f32 {
    let (sign, exp, man) = (u32::from(h & 0x8000) << 16, u32::from(h >> 10) & 0x1f, u32::from(h) & 0x3ff);
    match exp {
        0 => f32::from_bits(sign) + (if sign == 0 { 1.0 } else { -1.0 }) * man as f32 * 2f32.powi(-24),
        31 => f32::from_bits(sign | 0x7f80_0000 | man << 13),
        _ => f32::from_bits(sign | (exp + 112) << 23 | man << 13),
    }
}

/// A model's `.safetensors` files: each a little-endian header length, a JSON header naming every tensor's type,
/// shape and byte range, then the data. Tensors are read on demand, so a checkpoint larger than memory converts.
pub struct Safetensors {
    files: Vec<std::fs::File>,
    tensors: std::collections::HashMap<String, (usize, Dtype, Vec<usize>, u64)>,
}

impl Safetensors {
    /// Opens every `.safetensors` file in `dir`.
    pub fn open(dir: &std::path::Path) -> Result<Safetensors> {
        let mut st = Safetensors { files: Vec::new(), tensors: std::collections::HashMap::new() };
        let mut paths: Vec<_> =
            std::fs::read_dir(dir)?.filter_map(|e| Some(e.ok()?.path())).filter(|p| p.extension().is_some_and(|e| e == "safetensors")).collect();
        paths.sort();
        for path in paths {
            let file = std::fs::File::open(&path)?;
            let (head, data) = header(&file)?;
            for (name, t) in head.obj().unwrap_or_default() {
                if let Some((dtype, shape, start)) = entry(t) {
                    st.tensors.insert(name.clone(), (st.files.len(), dtype, shape, data + start));
                }
            }
            st.files.push(file);
        }
        Ok(st)
    }

    /// Whether the checkpoint has tensor `name`.
    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// The prefix before `name` in this checkpoint: a multimodal one nests its decoder.
    pub fn prefix(&self, name: &str) -> &'static str {
        ["model.language_model.", "model.", ""].into_iter().find(|p| self.has(&format!("{p}{name}"))).unwrap_or_default()
    }

    /// Rows and columns of tensor `name`: the last dimension is a row; a scalar is one row of one.
    pub fn dims(&self, name: &str) -> Result<(usize, usize)> {
        let shape = &self.tensors.get(name).ok_or_else(|| bad(name))?.2;
        let cols = shape.last().copied().unwrap_or(1);
        Ok((shape.iter().product::<usize>() / cols.max(1), cols))
    }

    /// Rows `rows` of tensor `name` as `f32`.
    pub fn rows(&self, name: &str, rows: std::ops::Range<usize>) -> Result<Vec<f32>> {
        let (file, dtype, _, start) = self.tensors.get(name).ok_or_else(|| bad(name))?;
        let (size, cols) = (dtype.size() as usize, self.dims(name)?.1);
        let mut raw = vec![0; rows.len() * cols * size];
        self.files[*file].read_exact_at(&mut raw, start + (rows.start * cols * size) as u64)?;
        Ok(match dtype {
            Dtype::F32 => raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
            Dtype::F16 => raw.chunks_exact(2).map(|b| f16_value(u16::from_le_bytes([b[0], b[1]]))).collect(),
            _ => raw.chunks_exact(2).map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16)).collect(),
        })
    }
}

/// A file's JSON header, and where its data starts.
fn header(file: &std::fs::File) -> Result<(Json, u64)> {
    let mut len = [0; 8];
    file.read_exact_at(&mut len, 0)?;
    let mut head = vec![0; usize::try_from(u64::from_le_bytes(len)).map_err(|_| bad("header"))?];
    file.read_exact_at(&mut head, 8)?;
    Ok((Json::parse(&head)?, 8 + head.len() as u64))
}

/// A header entry's type, shape and data offset; `None` for `__metadata__` and types other than floats.
fn entry(t: &Json) -> Option<(Dtype, Vec<usize>, u64)> {
    let types = [("BF16", Dtype::BF16), ("F16", Dtype::F16), ("F32", Dtype::F32)];
    let dtype = types.into_iter().find(|&(n, _)| Some(n) == t.get("dtype").and_then(Json::str))?.1;
    let shape = t.get("shape")?.arr()?.iter().map(|d| d.num().unwrap_or(0.0) as usize).collect();
    Some((dtype, shape, t.get("data_offsets")?.arr()?.first()?.num()? as u64))
}

fn bad(name: &str) -> Error {
    Error::Format(format!("safetensors: {name} is missing or damaged"))
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // the conversions are exact, so the checks are too
mod tests {
    use super::*;

    #[test]
    fn f16_round_trips_every_value_and_rounds_to_even() {
        for h in (0..=u16::MAX).filter(|h| h & 0x7c00 != 0x7c00 || h.trailing_zeros() >= 10) {
            assert_eq!(f16_bits(f16_value(h)), h, "{h:#06x}");
        }
        let cases = [(1.0, 0x3c00), (65504.0, 0x7bff), (65520.0, 0x7c00), (2f32.powi(-24), 1), (2f32.powi(-25), 0), (0.1, 0x2e66), (-2.0, 0xc000)];
        for (v, h) in cases {
            assert_eq!(f16_bits(v), h, "{v}");
        }
        assert_eq!(f16_bits(1.0 + 2f32.powi(-11)), 0x3c00, "a tie rounds to the even neighbour");
    }

    #[test]
    fn q4_packs_weights_where_the_kernel_reads_them() {
        let w: Vec<f32> = (0..64).map(|j| (j % 15) as f32 - 7.0).collect();
        let q = q4(&w, 64);
        assert_eq!(f16_value(q.scales[0]), 1.0);
        let nibble = |j: usize| q.packed[j / 8 * 4 + j % 4] >> (j % 8 / 4 * 4) & 15;
        assert!((0..64).all(|j| f32::from(nibble(j)) - 8.0 == w[j]));
        assert_eq!(q.packed[0], 0x51, "weights 0 and 4 share byte 0");
    }

    #[test]
    fn q8_keeps_block_sums() {
        let x: Vec<f32> = (0..64).map(|i| i as f32 - 20.0).collect();
        let (q, s) = q8(&x);
        assert_eq!((q[0], q[20], q[63]), (-127, 0, 127));
        let sum: i32 = q[..32].iter().map(|&v| i32::from(v)).sum();
        assert_eq!(s[0], [20.0 / 127.0, sum as f32 * (20.0 / 127.0)]);
    }
}