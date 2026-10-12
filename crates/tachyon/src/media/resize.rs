//! PIL's bicubic resampling of 8-bit images (`Resample.c`), bit for bit: per output pixel, a filter of support
//! `2 · max(1, scale)` normalized in `f64`, turned into 22-bit fixed point; rows first, then columns, each pass
//! rounding and clamping to 8 bits.

use super::Image;

/// Fraction bits of the fixed-point weights (`32 − 8 − 2`).
const BITS: u32 = 22;

/// PIL's bicubic kernel, `a = −0.5`.
fn bicubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

/// The weights of each of `out` outputs over `size` inputs: its first input and fixed-point weights.
fn weights(size: usize, out: usize) -> Vec<(usize, Vec<i32>)> {
    let scale = size as f64 / out as f64;
    let filter = scale.max(1.0);
    let (support, inverse) = (2.0 * filter, 1.0 / filter);
    (0..out)
        .map(|o| {
            let center = (o as f64 + 0.5) * scale;
            // PIL truncates toward zero after adding 0.5, then clamps.
            let lo = ((center - support + 0.5) as i64).max(0) as usize;
            let hi = ((center + support + 0.5) as i64).min(size as i64) as usize;
            let w: Vec<f64> = (lo..hi).map(|x| bicubic((x as f64 - center + 0.5) * inverse)).collect();
            let sum: f64 = w.iter().sum();
            let fixed = w
                .iter()
                .map(|&v| {
                    let v = if sum == 0.0 { v } else { v / sum } * f64::from(1 << BITS);
                    if v < 0.0 { (v - 0.5) as i32 } else { (v + 0.5) as i32 }
                })
                .collect();
            (lo, fixed)
        })
        .collect()
}

/// A weighted sum of 8-bit samples back to 8 bits, rounded and clamped.
fn clip(sum: i32) -> u8 {
    (sum >> BITS).clamp(0, 255) as u8
}

impl Image {
    /// This image resized to `width × height` as PIL's `resize(…, BICUBIC)` does it.
    #[must_use]
    pub fn resize(&self, width: usize, height: usize) -> Image {
        if (width, height) == (self.width, self.height) {
            return Image { width, height, rgb: self.rgb.clone() };
        }
        let (wx, wy) = (weights(self.width, width), weights(self.height, height));
        // Only the rows the vertical pass reads, as PIL does.
        let (first, last) = (wy[0].0, wy.last().map_or(0, |(lo, w)| lo + w.len()));
        let mut mid = vec![0u8; width * (last - first) * 3];
        let horizontal = width != self.width;
        for y in first..last {
            let (src, dst) = (&self.rgb[y * self.width * 3..], &mut mid[(y - first) * width * 3..]);
            for (x, (lo, w)) in wx.iter().enumerate() {
                for c in 0..3 {
                    dst[x * 3 + c] = if horizontal {
                        clip(w.iter().enumerate().fold(1 << (BITS - 1), |s, (k, &v)| s + i32::from(src[(lo + k) * 3 + c]) * v))
                    } else {
                        src[x * 3 + c]
                    };
                }
            }
        }
        if height == self.height {
            return Image { width, height, rgb: mid };
        }
        let mut rgb = vec![0u8; width * height * 3];
        for (y, (lo, w)) in wy.iter().enumerate() {
            for i in 0..width * 3 {
                let s = w.iter().enumerate().fold(1 << (BITS - 1), |s, (k, &v)| s + i32::from(mid[(lo - first + k) * width * 3 + i]) * v);
                rgb[y * width * 3 + i] = clip(s);
            }
        }
        Image { width, height, rgb }
    }
}
