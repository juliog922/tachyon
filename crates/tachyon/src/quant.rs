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
