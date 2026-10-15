//! Images against PIL: every file of `tests/fixtures/media/` (written by `scripts/media.py images`) decoded to the
//! same RGB as PIL's `exif_transpose(open(…)).convert("RGB")`, and resized to the same size and pixels as Gemma 4's
//! image processor gives it for each soft-token budget; the files PIL reads that Tachyon does not, refused. Audio
//! (`scripts/media.py audio`): WAV files of every sample format decoded to the same samples, the log-mel features of
//! Gemma 4's feature extractor, and resampling to 16 kHz against the tones it should keep.

use tachyon::json::Json;
use tachyon::media::{self, Image};

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/media");

/// FNV-1a 64, as the fixture's hashes.
fn fnv(data: &[u8]) -> String {
    format!("{:016x}", data.iter().fold(0xCBF2_9CE4_8422_2325_u64, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01B3)))
}

fn cases() -> Vec<Json> {
    std::fs::read_to_string(format!("{DIR}/images.ref")).unwrap().lines().map(|l| Json::parse(l.as_bytes()).unwrap()).collect()
}

fn file(case: &Json) -> (&str, Vec<u8>) {
    let name = case.get("file").and_then(Json::str).unwrap();
    (name, std::fs::read(format!("{DIR}/{name}")).unwrap())
}

/// What is wrong with Tachyon's decoding of `case`, if anything.
fn mismatch(case: &Json) -> Option<String> {
    let (name, bytes) = file(case);
    let refused = case.get("refused").is_some() || case.get("error").is_some();
    match (media::decode(&bytes), refused) {
        (Err(e), false) => Some(format!("{name}: {e}")),
        (Ok(_), true) => Some(format!("{name}: decoded, but Tachyon refuses it")),
        (Err(_), true) => None,
        (Ok(img), false) => {
            let size = [case.get("width"), case.get("height")].map(|v| v.and_then(Json::num).unwrap() as usize);
            let same = fnv(&img.rgb) == case.get("fnv").and_then(Json::str).unwrap() || case.get("clipped").is_some();
            ([img.width, img.height] != size || !same).then(|| format!("{name}: {}×{} decoded, its pixels differ from PIL's", img.width, img.height))
        }
    }
}

#[test]
fn images_decode_as_pil_decodes_them() {
    let cases = cases();
    let wrong: Vec<String> = cases.iter().filter_map(mismatch).collect();
    assert!(wrong.is_empty(), "{wrong:#?}");
    assert!(cases.iter().filter(|c| c.get("fnv").is_some()).count() >= 45, "too few files");
}

#[test]
fn images_resize_as_gemma_4_processor_does() {
    for case in cases().iter().filter(|c| c.get("fitted").is_some() && c.get("refused").is_none() && c.get("clipped").is_none()) {
        let (name, bytes) = file(case);
        let img = media::decode(&bytes).unwrap();
        for (budget, want) in case.get("fitted").and_then(Json::obj).unwrap() {
            let want = want.arr().unwrap();
            let (w, h) = img.fitted_size(budget.parse().unwrap()).unwrap();
            assert_eq!([w, h], [0, 1].map(|i| want[i].num().unwrap() as usize), "{name} at {budget}: size");
            let out = img.resize(w, h);
            assert_eq!(fnv(&out.rgb), want[2].str().unwrap(), "{name} at {budget}: pixels differ from PIL's bicubic");
        }
    }
}

#[test]
fn orientations_turn_the_image() {
    let img = Image { width: 3, height: 2, rgb: (0..18).collect() };
    let px = |i: &Image, x: usize, y: usize| i.rgb[(y * i.width + x) * 3];
    let turned = img.clone().oriented(6);
    assert_eq!((turned.width, turned.height), (2, 3));
    // Clockwise: the bottom-left pixel comes to the top left.
    assert_eq!(px(&turned, 0, 0), px(&img, 0, 1));
    assert_eq!(img.clone().oriented(8).oriented(6), img);
    assert_eq!(img.clone().oriented(3).oriented(3), img);
}

#[test]
fn damaged_and_unknown_files_are_errors() {
    let (_, png) = file(&cases().into_iter().find(|c| c.get("file").and_then(Json::str) == Some("rgb8.png")).unwrap());
    for cut in [8, 30, png.len() / 2, png.len() - 13] {
        assert!(media::decode(&png[..cut]).is_err(), "a PNG cut at {cut} decoded");
    }
    assert!(media::decode(b"GIF89a").is_err());
    assert!(media::decode(&[]).is_err());
}

/// The audio cases of `audio.ref`.
fn audio_cases() -> Vec<Json> {
    std::fs::read_to_string(format!("{DIR}/audio.ref")).unwrap().lines().map(|l| Json::parse(l.as_bytes()).unwrap()).collect()
}

/// The audio of `case`.
fn audio(case: &Json) -> media::Audio {
    media::decode_wav(&std::fs::read(format!("{DIR}/{}", case.get("file").and_then(Json::str).unwrap())).unwrap()).unwrap()
}

#[test]
fn wav_files_decode_to_the_same_samples() {
    for case in audio_cases() {
        let (name, a) = (case.get("file").and_then(Json::str).unwrap(), audio(&case));
        let int = |k: &str| case.get(k).and_then(Json::num).unwrap() as usize;
        assert_eq!((a.rate as usize, a.samples.len()), (int("rate"), int("samples")), "{name}: rate and samples");
        let bytes: Vec<u8> = a.samples.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(fnv(&bytes), case.get("fnv").and_then(Json::str).unwrap(), "{name}: samples differ");
    }
}

#[test]
fn features_match_gemma_4s_extractor() {
    for case in audio_cases().iter().filter(|c| c.get("mel").is_some()) {
        let want: Vec<f32> = std::fs::read(format!("{DIR}/{}", case.get("mel").and_then(Json::str).unwrap()))
            .unwrap()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let got = media::mel(&audio(case).samples);
        assert_eq!(got.len(), want.len(), "{case:?}: frames");
        let err = got.iter().zip(&want).fold(0f32, |m, (g, w)| m.max((g - w).abs()));
        assert!(err <= 1e-4, "{case:?}: features differ by {err}");
    }
    assert!(media::mel(&[0.1; 100]).is_empty(), "too short for a frame");
    assert_eq!(media::mel(&vec![0.0; media::MAX_SAMPLES + 5000]).len(), 2999 * media::MEL_BINS, "cut at 30 s: 2,999 frames");
}

/// Tones of 440 Hz, 1 kHz and 3 kHz (and 6 kHz when the rate holds it) at `rate`, `seconds` long, sampled at `at`.
fn tones(rate: u32, at: u32, seconds: f64) -> Vec<f32> {
    let freqs: &[f64] = if rate > 12_000 { &[440.0, 1000.0, 3000.0, 6000.0] } else { &[440.0, 1000.0, 3000.0] };
    (0..(seconds * f64::from(at)) as usize)
        .map(|n| freqs.iter().map(|f| 0.2 * (2.0 * std::f64::consts::PI * f * n as f64 / f64::from(at)).sin()).sum::<f64>() as f32)
        .collect()
}

#[test]
fn resampling_keeps_the_tones_16_khz_holds_and_drops_the_rest() {
    for rate in [8000, 22_050, 32_000, 44_100, 48_000] {
        let a = media::Audio { rate, samples: tones(rate, rate, 1.0) };
        let (got, want) = (a.resampled(16_000).unwrap(), tones(rate, 16_000, 1.0));
        assert_eq!(got.samples.len(), 16_000, "{rate} Hz: length");
        // The ends see zeros past the edges; 100 ms in, the filter sees only the tones.
        let err = got.samples[1600..14_400].iter().zip(&want[1600..14_400]).fold(0f32, |m, (g, w)| m.max((g - w).abs()));
        assert!(err <= 2e-3, "{rate} Hz: the tones moved by {err}");
    }
    let high = media::Audio {
        rate: 44_100,
        samples: (0..44_100).map(|n| (0.5 * (2.0 * std::f64::consts::PI * 10_000.0 * f64::from(n) / 44_100.0).sin()) as f32).collect(),
    };
    let left = high.resampled(16_000).unwrap().samples[1600..14_400].iter().fold(0f32, |m, v| m.max(v.abs()));
    assert!(left <= 1e-3, "a 10 kHz tone left {left} at 16 kHz");
    assert!(high.resampled(44_099).is_err(), "a ratio of 44099/44100 is refused");
}

#[test]
fn damaged_and_unknown_wav_files_are_errors() {
    let wav = std::fs::read(format!("{DIR}/audio-s16.wav")).unwrap();
    for cut in [10, 20, 36] {
        assert!(media::decode_wav(&wav[..cut]).is_err(), "a WAV cut at {cut} decoded");
    }
    let mut adpcm = wav.clone();
    adpcm[20] = 2;
    assert!(media::decode_wav(&adpcm).is_err(), "ADPCM decoded");
    // A file cut inside its data keeps its whole frames, as a recording cut short.
    assert_eq!(media::decode_wav(&wav[..wav.len() - 1001]).unwrap().samples.len(), 3499);
}
