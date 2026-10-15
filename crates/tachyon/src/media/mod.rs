//! Media inputs for Gemma 4's encoders. Images: PNG and JPEG decoded to RGB as PIL decodes them (JPEGs turned by
//! their EXIF orientation, as `transformers` loads them), then resized as PIL resizes them to the vision encoder's
//! patch grid. Audio: WAV files, resampled to 16 kHz, and their log-mel features ([`decode_wav`], [`mel`]).

mod audio;
mod inflate;
mod jpeg;
mod png;
mod resize;

pub use audio::{Audio, MAX_SAMPLES, MEL_BINS, SAMPLE_RATE, decode_wav, mel};

use crate::{Error, Result};

/// The most pixels an image may have: 64 Mi, 192 MiB of RGB.
const MAX_PIXELS: usize = 1 << 26;
/// Pixels on a side of the vision encoder's patches.
pub const PATCH: usize = 16;
/// Patches on a side of the square the vision encoder pools into one soft token.
pub const POOL: usize = 3;
/// The soft-token budgets an image may have, as Gemma 4 is trained.
pub const BUDGETS: [usize; 5] = [70, 140, 280, 560, 1120];

/// The soft-token budget an image gets unless asked otherwise, as Gemma 4's image processor's.
pub const DEFAULT_BUDGET: usize = 280;

/// An 8-bit RGB image, rows top to bottom.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    /// Pixels across.
    pub width: usize,
    /// Pixels down.
    pub height: usize,
    /// `width · height` pixels of red, green and blue.
    pub rgb: Vec<u8>,
}

/// Decodes a PNG or JPEG file.
pub fn decode(bytes: &[u8]) -> Result<Image> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        png::decode(bytes)
    } else if bytes.starts_with(&[0xFF, 0xD8]) {
        jpeg::decode(bytes).map(|(image, orientation)| image.oriented(orientation))
    } else {
        Err(refuse("an image that is neither PNG nor JPEG"))
    }
}

impl Image {
    /// This image turned as EXIF orientation `o` (1 to 8) says, as PIL's `exif_transpose` does.
    #[must_use]
    pub fn oriented(self, o: u16) -> Image {
        if !(2..=8).contains(&o) {
            return self;
        }
        let (w, h) = (self.width, self.height);
        let (ow, oh) = if o >= 5 { (h, w) } else { (w, h) };
        let mut rgb = vec![0u8; w * h * 3];
        for (y, x) in (0..oh).flat_map(|y| (0..ow).map(move |x| (y, x))) {
            let (sx, sy) = source(o, w, h, x, y);
            rgb[(y * ow + x) * 3..][..3].copy_from_slice(&self.rgb[(sy * w + sx) * 3..][..3]);
        }
        Image { width: ow, height: oh, rgb }
    }

    /// The size Gemma 4's image processor gives this image for `budget` soft tokens: the largest with the same
    /// aspect ratio, in whole squares of [`POOL`] × [`PATCH`] pixels, of at most `budget · POOL²` patches.
    pub fn fitted_size(&self, budget: usize) -> Result<(usize, usize)> {
        if !BUDGETS.contains(&budget) {
            return Err(Error::Media(format!("{budget} soft tokens per image; Gemma 4 takes {BUDGETS:?}")));
        }
        let (side, target) = (POOL * PATCH, (budget * POOL * POOL * PATCH * PATCH) as f64);
        let factor = (target / (self.width * self.height) as f64).sqrt();
        let fit = |v: usize| (factor * v as f64 / side as f64).floor() as usize * side;
        let (mut h, mut w) = (fit(self.height), fit(self.width));
        let most = budget * side;
        if h == 0 && w == 0 {
            return Err(Error::Media(format!("a {}×{} image too thin to fit any patch", self.width, self.height)));
        }
        if h == 0 {
            (h, w) = (side, (self.width / self.height * side).min(most));
        } else if w == 0 {
            (w, h) = (side, (self.height / self.width * side).min(most));
        }
        Ok((w, h))
    }

    /// The soft tokens the vision tower makes of this image for `budget`: one per square of [`POOL`] × [`POOL`]
    /// patches of its [`Image::fitted_size`].
    pub fn soft_tokens(&self, budget: usize) -> Result<usize> {
        let (w, h) = self.fitted_size(budget)?;
        Ok(w / (PATCH * POOL) * (h / (PATCH * POOL)))
    }

    /// This image at its [`Image::fitted_size`] for `budget` soft tokens.
    pub fn fitted(&self, budget: usize) -> Result<Image> {
        self.fitted_size(budget).map(|(w, h)| self.resize(w, h))
    }
}

/// The pixel of a `w × h` image that lands at `(x, y)` once turned as EXIF orientation `turn` (2 to 8) says.
fn source(turn: u16, w: usize, h: usize, x: usize, y: usize) -> (usize, usize) {
    match turn {
        2 => (w - 1 - x, y),
        3 => (w - 1 - x, h - 1 - y),
        4 => (x, h - 1 - y),
        5 => (y, x),
        6 => (y, h - 1 - x),
        7 => (w - 1 - y, h - 1 - x),
        _ => (w - 1 - y, x),
    }
}

/// An error for a corrupt file.
fn bad(what: &str) -> Error {
    Error::Media(format!("{what}: the file is damaged or not what it claims"))
}

/// An error for a valid file Tachyon does not decode.
fn refuse(what: &str) -> Error {
    Error::Media(format!("{what}, which tachyon does not decode"))
}

/// An error unless a `width × height` image is non-empty and within [`MAX_PIXELS`].
fn check_size(width: usize, height: usize) -> Result<()> {
    if width == 0 || height == 0 || width.saturating_mul(height) > MAX_PIXELS {
        return Err(Error::Media(format!("a {width}×{height} image: at most {MAX_PIXELS} pixels, none empty")));
    }
    Ok(())
}
