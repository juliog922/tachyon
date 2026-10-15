//! Audio for Gemma 4's audio encoder: WAV files read as `soundfile` reads them, resampled to 16 kHz, and the
//! log-mel features of Gemma 4's feature extractor.
//!
//! - WAV: PCM of 8 (unsigned), 16, 24 and 32 bits and IEEE floats of 32 and 64 bits, plain or
//!   `WAVE_FORMAT_EXTENSIBLE`; integers become `v / 2^(bits−1)` (8 bits: `(v − 128) / 128`), channels their mean.
//! - Resampling: a windowed sinc (Kaiser, β 8, 16 zero crossings of the lower rate, cut at 94% of its Nyquist
//!   frequency), polyphase over the rates' ratio.
//! - Features: frames of 20 ms every 10 ms, the first centred at 0 (10 ms of zeros before), the audio padded with
//!   zeros to a multiple of 128 samples and cut at 30 s; a periodic Hann window, a 512-point FFT's magnitudes, 128
//!   triangular HTK mel filters from 0 to 8 kHz, and `ln(mel + 0.001)`, in `f64`, as `f32`. A frame whose last
//!   sample is padding is all zeros.

use super::{bad, refuse};
use crate::Result;

/// The rate the audio encoder reads.
pub const SAMPLE_RATE: u32 = 16_000;
/// Mel bins of a feature frame.
pub const MEL_BINS: usize = 128;
/// The most audio the encoder takes: 30 s.
pub const MAX_SAMPLES: usize = 30 * SAMPLE_RATE as usize;
/// Samples of a feature frame, and between frames.
const FRAME: usize = 320;
const HOP: usize = 160;
/// Points of the FFT.
const FFT: usize = 512;

/// Mono audio: samples in `[-1, 1]`, `rate` per second.
#[derive(Clone, Debug, PartialEq)]
pub struct Audio {
    /// Samples per second.
    pub rate: u32,
    /// The samples, channels averaged.
    pub samples: Vec<f32>,
}

/// A WAV file's format: its tag (1 integers, 3 floats), channels, rate and bits per sample.
struct Format {
    tag: u16,
    channels: usize,
    rate: u32,
    bits: usize,
}

/// Decodes a WAV file.
pub fn decode_wav(bytes: &[u8]) -> Result<Audio> {
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(bad("not a WAV file"));
    }
    let (mut at, mut format) = (12, None);
    while at + 8 <= bytes.len() {
        let size = u32::from_le_bytes([bytes[at + 4], bytes[at + 5], bytes[at + 6], bytes[at + 7]]) as usize;
        let body = &bytes[at + 8..bytes.len().min((at + 8).saturating_add(size))];
        match &bytes[at..at + 4] {
            b"fmt " => format = Some(fmt(body)?),
            b"data" => return format.as_ref().map(|f| samples(f, body)).ok_or_else(|| bad("WAV data before its format")),
            _ => {}
        }
        at = at.saturating_add(8 + size + size % 2);
    }
    Err(bad("WAV file without data"))
}

/// The `fmt ` chunk.
fn fmt(body: &[u8]) -> Result<Format> {
    if body.len() < 16 {
        return Err(bad("WAV format chunk too short"));
    }
    let word = |i: usize| u16::from_le_bytes([body[i], body[i + 1]]);
    let mut tag = word(0);
    if tag == 0xFFFE && body.len() >= 26 {
        tag = word(24);
    }
    let f = Format { tag, channels: usize::from(word(2)), rate: u32::from_le_bytes([body[4], body[5], body[6], body[7]]), bits: usize::from(word(14)) };
    match (f.tag, f.bits) {
        _ if f.channels == 0 || f.rate == 0 => Err(bad("WAV file of no channels or no rate")),
        (1, 8 | 16 | 24 | 32) | (3, 32 | 64) => Ok(f),
        _ => Err(refuse(&format!("WAV of format {} with {}-bit samples", f.tag, f.bits))),
    }
}

/// The samples of `data`, whole frames only, each frame's channels averaged.
fn samples(f: &Format, data: &[u8]) -> Audio {
    let width = f.bits / 8;
    let value = |b: &[u8]| -> f32 {
        match (f.tag, width) {
            (1, 1) => (f32::from(b[0]) - 128.0) / 128.0,
            (1, 2) => f32::from(i16::from_le_bytes([b[0], b[1]])) / 32_768.0,
            (1, 3) => (i32::from_le_bytes([0, b[0], b[1], b[2]]) >> 8) as f32 / 8_388_608.0,
            (1, _) => i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32 / 2_147_483_648.0,
            (_, 4) => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            _ => f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32,
        }
    };
    let frame = width * f.channels;
    let samples = data.chunks_exact(frame).map(|fr| fr.chunks_exact(width).map(value).sum::<f32>() / f.channels as f32).collect();
    Audio { rate: f.rate, samples }
}

/// Greatest common divisor.
fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// The zeroth-order modified Bessel function of the first kind, by its series.
fn bessel(x: f64) -> f64 {
    let (mut sum, mut term) = (1.0, 1.0);
    for k in 1..50 {
        term *= (x / (2.0 * f64::from(k))).powi(2);
        sum += term;
    }
    sum
}

impl Audio {
    /// This audio at `rate` samples per second, through a windowed sinc (see the module documentation). An error if
    /// the rates' ratio, in lowest terms, has a numerator over 4,096.
    pub fn resampled(&self, rate: u32) -> Result<Audio> {
        if rate == self.rate {
            return Ok(self.clone());
        }
        let g = gcd(self.rate, rate);
        let (up, down) = ((rate / g) as usize, (self.rate / g) as usize);
        if up > 4096 {
            return Err(refuse(&format!("resampling from {} Hz to {rate} Hz", self.rate)));
        }
        let cut = 0.5 * 0.94 * f64::from(rate.min(self.rate)) / f64::from(self.rate);
        let half = (16.0 / (2.0 * cut)).ceil() as usize;
        let table: Vec<f64> = (0..up)
            .flat_map(|p| (0..2 * half).map(move |j| (p, j)))
            .map(|(p, j)| {
                let tau = p as f64 / up as f64 + half as f64 - 1.0 - j as f64;
                let w = 1.0 - (tau / half as f64).powi(2);
                let sinc = if tau == 0.0 { 1.0 } else { (std::f64::consts::PI * 2.0 * cut * tau).sin() / (std::f64::consts::PI * 2.0 * cut * tau) };
                if w <= 0.0 { 0.0 } else { 2.0 * cut * sinc * bessel(8.0 * w.sqrt()) / bessel(8.0) }
            })
            .collect();
        let n = (self.samples.len() * up).div_ceil(down);
        let samples = (0..n)
            .map(|i| {
                let (k, p) = ((i * down) / up, (i * down) % up);
                let taps = &table[p * 2 * half..][..2 * half];
                let at = |j: usize| (k + j).checked_sub(half - 1).and_then(|s| self.samples.get(s)).map_or(0.0, |&v| f64::from(v));
                taps.iter().enumerate().map(|(j, t)| t * at(j)).sum::<f64>() as f32
            })
            .collect();
        Ok(Audio { rate, samples })
    }
}

/// Gemma 4's log-mel features of `samples` (16 kHz; past 30 s cut): `f32 [frames, MEL_BINS]`, frames as the
/// module documentation says.
pub fn mel(samples: &[f32]) -> Vec<f32> {
    let len = samples.len().min(MAX_SAMPLES);
    let mut padded = vec![0f32; HOP + len.next_multiple_of(128)];
    padded[HOP..HOP + len].copy_from_slice(&samples[..len]);
    let frames = padded.len().checked_sub(FRAME + 1).map_or(0, |r| r / HOP + 1);
    let window: Vec<f32> = (0..FRAME).map(|n| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / FRAME as f64).cos()) as f32).collect();
    let (filters, twiddles) = (filters(), twiddles());
    let mut out = Vec::with_capacity(frames * MEL_BINS);
    for f in 0..frames {
        // A frame reaching into the padding is all zeros, as the extractor's mask leaves it.
        if f * HOP + FRAME - HOP >= len {
            out.extend([0.0; MEL_BINS]);
            continue;
        }
        let mut re = [0f64; FFT];
        for (n, r) in re.iter_mut().take(FRAME).enumerate() {
            *r = f64::from(padded[f * HOP + n] * window[n]);
        }
        let mag = magnitudes(re, &twiddles);
        out.extend((0..MEL_BINS).map(|m| ((0..=FFT / 2).map(|k| mag[k] * filters[k * MEL_BINS + m]).sum::<f64>() + 1e-3).ln() as f32));
    }
    out
}

/// The triangular HTK mel filters, `[FFT/2 + 1, MEL_BINS]`, as `transformers`' `mel_filter_bank` makes them.
fn filters() -> Vec<f64> {
    let mel = |hz: f64| 2595.0 * (1.0 + hz / 700.0).log10();
    let top = mel(8000.0);
    let edges: Vec<f64> = (0..MEL_BINS + 2).map(|i| 700.0 * (10f64.powf(top * i as f64 / (MEL_BINS + 1) as f64 / 2595.0) - 1.0)).collect();
    (0..=FFT / 2)
        .flat_map(|k| (0..MEL_BINS).map(move |m| (k, m)))
        .map(|(k, m)| {
            let hz = 8000.0 * k as f64 / (FFT / 2) as f64;
            let down = -(edges[m] - hz) / (edges[m + 1] - edges[m]);
            let up = (edges[m + 2] - hz) / (edges[m + 2] - edges[m + 1]);
            down.min(up).max(0.0)
        })
        .collect()
}

/// `e^(−2πik/FFT)` for `k < FFT/2`.
fn twiddles() -> Vec<(f64, f64)> {
    (0..FFT / 2).map(|k| (-2.0 * std::f64::consts::PI * k as f64 / FFT as f64).sin_cos()).map(|(s, c)| (c, s)).collect()
}

/// The magnitudes of the FFT of the real frame `re`, bins `0..=FFT/2`: radix 2, in place, bits reversed first.
fn magnitudes(mut re: [f64; FFT], twiddles: &[(f64, f64)]) -> Vec<f64> {
    let mut im = [0f64; FFT];
    let bits = FFT.trailing_zeros();
    for i in 0..FFT {
        let j = i.reverse_bits() >> (usize::BITS - bits);
        if j > i {
            re.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= FFT {
        for start in (0..FFT).step_by(len) {
            for k in 0..len / 2 {
                let (c, s) = twiddles[k * (FFT / len)];
                let (a, b) = (start + k, start + k + len / 2);
                let (tr, ti) = (re[b] * c - im[b] * s, re[b] * s + im[b] * c);
                (re[b], im[b]) = (re[a] - tr, im[a] - ti);
                (re[a], im[a]) = (re[a] + tr, im[a] + ti);
            }
        }
        len *= 2;
    }
    (0..=FFT / 2).map(|k| re[k].hypot(im[k])).collect()
}
