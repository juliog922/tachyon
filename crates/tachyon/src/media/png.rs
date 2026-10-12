//! PNG to RGB, as PIL's `convert("RGB")` gives it: every color type and bit depth, the alpha channel dropped,
//! 16-bit samples cut to their high byte. Interlaced images are refused.

use super::{Image, bad, refuse};
use crate::Result;

/// The header's fields: width, height, bit depth, color type.
struct Header {
    width: usize,
    height: usize,
    depth: u8,
    color: u8,
}

impl Header {
    /// Samples per pixel.
    fn channels(&self) -> usize {
        match self.color {
            0 | 3 => 1,
            4 => 2,
            2 => 3,
            _ => 4,
        }
    }

    /// Bytes per row, before the filter byte.
    fn stride(&self) -> usize {
        (self.width * self.channels() * usize::from(self.depth)).div_ceil(8)
    }
}

/// Decodes PNG file `bytes`.
pub(super) fn decode(bytes: &[u8]) -> Result<Image> {
    let chunks = chunks(bytes)?;
    let find = |kind: &[u8]| chunks.iter().find(|c| c.0 == kind).map(|c| c.1);
    let h = parse_header(find(b"IHDR").ok_or_else(|| bad("a PNG without a header"))?)?;
    let idat: Vec<u8> = chunks.iter().filter(|c| c.0 == b"IDAT").flat_map(|c| c.1.iter().copied()).collect();
    let raw = super::inflate::zlib(&idat)?;
    if raw.len() < h.height * (h.stride() + 1) {
        return Err(bad("a PNG's image data is short"));
    }
    let rows = unfilter(&h, &raw)?;
    Ok(Image { width: h.width, height: h.height, rgb: to_rgb(&h, &rows, find(b"PLTE").unwrap_or(&[])) })
}

/// The chunks of a PNG, kind and data, up to `IEND`.
fn chunks(bytes: &[u8]) -> Result<Vec<(&[u8], &[u8])>> {
    let (mut at, mut out) = (8, Vec::new());
    loop {
        let len = bytes.get(at..at + 4).map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]) as usize).ok_or_else(|| bad("a PNG ends early"))?;
        let data = bytes.get(at + 8..(at + 8).saturating_add(len)).ok_or_else(|| bad("a PNG chunk ends early"))?;
        let kind = &bytes[at + 4..at + 8];
        if kind == b"IEND" {
            return Ok(out);
        }
        out.push((kind, data));
        at += 12 + len;
    }
}

fn parse_header(d: &[u8]) -> Result<Header> {
    if d.len() < 13 {
        return Err(bad("a short PNG header"));
    }
    let be = |i: usize| u32::from_be_bytes([d[i], d[i + 1], d[i + 2], d[i + 3]]) as usize;
    let h = Header { width: be(0), height: be(4), depth: d[8], color: d[9] };
    let valid = match h.color {
        0 => matches!(h.depth, 1 | 2 | 4 | 8 | 16),
        3 => matches!(h.depth, 1 | 2 | 4 | 8),
        2 | 4 | 6 => matches!(h.depth, 8 | 16),
        _ => false,
    };
    if !valid || d[10] != 0 || d[11] != 0 {
        return Err(bad("a PNG header with an invalid color type, depth, compression or filter"));
    }
    if d[12] != 0 {
        return Err(refuse("an interlaced PNG"));
    }
    super::check_size(h.width, h.height)?;
    Ok(h)
}

/// The rows of `raw` with their filters undone, back to back.
fn unfilter(h: &Header, raw: &[u8]) -> Result<Vec<u8>> {
    let (stride, bpp) = (h.stride(), (h.channels() * usize::from(h.depth)).div_ceil(8));
    let mut out = vec![0u8; h.height * stride];
    for y in 0..h.height {
        let (filter, src) = (raw[y * (stride + 1)], &raw[y * (stride + 1) + 1..(y + 1) * (stride + 1)]);
        let (done, row) = out.split_at_mut(y * stride);
        let (row, up) = (&mut row[..stride], if y > 0 { &done[(y - 1) * stride..] } else { &[][..] });
        for i in 0..stride {
            let (b, back) = (up.get(i).copied().unwrap_or(0), i.checked_sub(bpp));
            let (a, c) = back.map_or((0, 0), |j| (row[j], up.get(j).copied().unwrap_or(0)));
            row[i] = src[i].wrapping_add(predict(filter, a, b, c)?);
        }
    }
    Ok(out)
}

/// Filter `filter`'s prediction from the bytes left (`a`), above (`b`) and above left (`c`).
fn predict(filter: u8, a: u8, b: u8, c: u8) -> Result<u8> {
    Ok(match filter {
        0 => 0,
        1 => a,
        2 => b,
        3 => u16::midpoint(u16::from(a), u16::from(b)) as u8,
        4 => paeth(a, b, c),
        _ => return Err(bad("a PNG row filter past 4")),
    })
}

fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let p = i16::from(a) + i16::from(b) - i16::from(c);
    let (pa, pb, pc) = ((p - i16::from(a)).abs(), (p - i16::from(b)).abs(), (p - i16::from(c)).abs());
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

/// Each pixel's RGB: gray repeated (depths under 8 scaled to 0–255), palette entries (black past its end), the
/// alpha dropped, 16-bit samples' high byte.
fn to_rgb(h: &Header, rows: &[u8], palette: &[u8]) -> Vec<u8> {
    let (stride, depth, n) = (h.stride(), usize::from(h.depth), h.channels());
    let sample = |row: &[u8], i: usize| -> u8 {
        match depth {
            8 => row[i],
            16 => row[2 * i],
            _ => {
                let v = row[i * depth / 8] >> (8 - depth - i * depth % 8) & ((1 << depth) - 1);
                if h.color == 3 { v } else { v * (255 / ((1 << depth) - 1)) }
            }
        }
    };
    let mut rgb = Vec::with_capacity(h.width * h.height * 3);
    for row in rows.chunks(stride) {
        for x in 0..h.width {
            match h.color {
                0 | 4 => rgb.extend([sample(row, x * n); 3]),
                3 => {
                    let p = usize::from(sample(row, x)) * 3;
                    rgb.extend(palette.get(p..p + 3).unwrap_or(&[0; 3]));
                }
                _ => (0..3).for_each(|c| rgb.push(sample(row, x * n + c))),
            }
        }
    }
    rgb
}
