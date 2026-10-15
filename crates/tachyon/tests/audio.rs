//! Gemma 4's audio tower: its soft-token count, its kernels against the CPU, and on the real checkpoint, its soft
//! tokens against the float32 tower of Hugging Face (`scripts/media.py speech`, files `soft-clip-*.f16` of
//! `tests/fixtures/media/`).
// Kernel arguments and the maths read as their names: `x`, `w`, `g`, `h`, `n`.
#![allow(clippy::many_single_char_names)]

#[cfg(feature = "gpu")]
mod common;

#[cfg(feature = "gpu")]
use tachyon::quant::{f16_bits, f16_value};

/// The soft tokens of the two speech fixtures, as Hugging Face's processor counts them, and the edges.
#[test]
fn soft_tokens_are_counted_as_gemma_4_counts_them() {
    assert_eq!(tachyon::audio::soft_tokens(199_760), 312);
    assert_eq!(tachyon::audio::soft_tokens(158_400), 248);
    assert_eq!(tachyon::audio::soft_tokens(160), 0);
    assert_eq!(tachyon::audio::soft_tokens(161), 1);
    assert_eq!(tachyon::audio::soft_tokens(10 * tachyon::media::MAX_SAMPLES), 750);
}

/// Launches `kernel` with pointers, then 32-bit values (`u32`, or `f32` when marked `true`).
#[cfg(feature = "gpu")]
fn launch(gpu: &common::Gpu, kernel: &str, shape: ([u32; 3], u32), pointers: &[u64], values: &[(u32, bool)]) {
    use tachyon::cuda::arg;
    let (grid, threads) = shape;
    let floats: Vec<f32> = values.iter().map(|&(v, _)| f32::from_bits(v)).collect();
    let args: Vec<_> = pointers.iter().map(arg).chain(values.iter().zip(&floats).map(|(v, f)| if v.1 { arg(f) } else { arg(&v.0) })).collect();
    // SAFETY: each test passes the kernel's arguments in order, on buffers of the sizes `crate::ptx` documents.
    unsafe { gpu.stream.launch(&gpu.module.function(kernel).unwrap(), grid, [threads, 1, 1], 0, &args) }.unwrap();
    gpu.stream.sync().unwrap();
}

/// The subsampling convolution, layer norm and ReLU on the CPU: `input` `[rows, fin, cin]`, `out_rows` rows out.
#[cfg(feature = "gpu")]
fn subsample(input: &[f32], [rows, fin, cin, out_rows]: [usize; 4], w: &[f32], g: &[f32]) -> Vec<f32> {
    let cout = g.len();
    let mut out = Vec::new();
    for (to, fo) in (0..out_rows).flat_map(|t| (0..fin / 2).map(move |f| (t, f))) {
        let sums: Vec<f64> = (0..cout)
            .map(|co| {
                let mut acc = 0.0;
                for (kh, kw, ci) in (0..3).flat_map(|kh| (0..3).flat_map(move |kw| (0..cin).map(move |ci| (kh, kw, ci)))) {
                    let (ti, fi) = ((2 * to + kh).wrapping_sub(1), (2 * fo + kw).wrapping_sub(1));
                    if ti < rows && fi < fin {
                        acc += f64::from(w[((co * cin + ci) * 3 + kh) * 3 + kw]) * f64::from(input[(ti * fin + fi) * cin + ci]);
                    }
                }
                acc
            })
            .collect();
        let mean = sums.iter().sum::<f64>() / cout as f64;
        let var = sums.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / cout as f64;
        out.extend(sums.iter().zip(g).map(|(s, &gw)| ((s - mean) / (var + 1e-6).sqrt() * f64::from(gw)).max(0.0) as f32));
    }
    out
}

/// The subsampling convolution: 7 valid rows of 8 frequencies and 3 channels into 4 rows of 4 and 32 channels, as
/// `f32` and as `f16`.
#[cfg(feature = "gpu")]
#[test]
fn the_subsampling_convolution_matches_the_cpu() {
    let gpu = common::Gpu::new();
    let ([rows, fin, cin, cout], out_rows) = ([7, 8, 3, 32], 4);
    let (input, w, g) = (common::values(9 * fin * cin, 51), common::values(cout * cin * 9, 52), common::values(cout, 53));
    let want = subsample(&input, [rows, fin, cin, out_rows], &w, &g);
    let (ib, wb, gb, out) = (gpu.upload(&input), gpu.upload(&w), gpu.upload(&g), gpu.zeros(4 * want.len()));
    for half in [0u32, 1] {
        let values = [(rows as u32, false), (fin as u32, false), (cin as u32, false), (half, false), (1e-6f32.to_bits(), true)];
        launch(&gpu, tachyon::ptx::SUBSAMPLE, ([fin as u32 / 2, out_rows as u32, 1], cout as u32), &[ib.ptr(), wb.ptr(), gb.ptr(), out.ptr()], &values);
        let got: Vec<f32> = if half == 1 { gpu.halves(&out)[..want.len()].iter().map(|&b| f16_value(b)).collect() } else { gpu.floats(&out) };
        for (i, (&a, &b)) in got.iter().zip(&want).enumerate() {
            assert!((a - b).abs() <= 1e-4 + if half == 1 { 1e-3 * b.abs() } else { 1e-5 * b.abs() }, "half {half}, value {i}: {a} against {b}");
        }
    }
}

/// The audio attention on the CPU, for query `t` and head `hd`: logits over keys `t − left ..= t`, soft-capped.
#[cfg(feature = "gpu")]
fn attend(qkv: &[f32], rel: &[f32], scale: &[f32], at: [usize; 4], kscale: f64) -> Vec<f64> {
    let [t, hd, h, left] = at;
    let row = |s: usize, part: usize, d: usize| f64::from(qkv[s * 3 * h + part * h + hd * 128 + d]);
    let logits: Vec<(usize, f64)> = (t.saturating_sub(left)..=t)
        .map(|s| {
            let dot: f64 = (0..128).map(|d| row(t, 0, d) * f64::from(scale[d]) * (kscale * row(s, 1, d) + f64::from(rel[(t - s) * h + hd * 128 + d]))).sum();
            (s, 50.0 * (dot / 50.0).tanh())
        })
        .collect();
    let top = logits.iter().fold(f64::NEG_INFINITY, |m, l| m.max(l.1));
    let z: f64 = logits.iter().map(|l| (l.1 - top).exp()).sum();
    (0..128).map(|d| logits.iter().map(|&(s, l)| (l - top).exp() / z * row(s, 2, d)).sum()).collect()
}

/// The audio attention: 20 tokens of 2 heads, each query seeing its 12 past keys and itself.
#[cfg(feature = "gpu")]
#[test]
fn the_audio_attention_matches_the_cpu() {
    let gpu = common::Gpu::new();
    let (n, h, left, kscale) = (20, 256, 12, 1.894_6_f32);
    let qkv: Vec<f32> = common::values(n * 3 * h, 61).iter().map(|v| 3.0 * v).collect();
    let (rel, scale) = (common::values((left + 1) * h, 62), common::values(128, 63));
    let (qb, rb, sb, out) = (gpu.upload(&qkv), gpu.upload(&rel), gpu.upload(&scale), gpu.zeros(2 * n * h));
    let values = [(h as u32, false), (left as u32, false), (kscale.to_bits(), true), (50f32.to_bits(), true)];
    launch(&gpu, tachyon::ptx::ATTEND_AUDIO, ([n as u32, (h / 128) as u32, 1], 32), &[qb.ptr(), rb.ptr(), sb.ptr(), out.ptr()], &values);
    let got = gpu.halves(&out);
    for (t, hd) in (0..n).flat_map(|t| (0..h / 128).map(move |hd| (t, hd))) {
        for (d, want) in attend(&qkv, &rel, &scale, [t, hd, h, left], f64::from(kscale)).into_iter().enumerate() {
            let g = f64::from(f16_value(got[t * h + hd * 128 + d]));
            assert!((g - want).abs() <= 2e-3 * (1.0 + want.abs()), "token {t}, head {hd}, dimension {d}: {g} against {want}");
        }
    }
}

/// The light convolution's middle: 6 tokens of 128 channels, 5 taps, then the RMS norm and the `SiLU`.
#[cfg(feature = "gpu")]
#[test]
fn the_light_convolution_matches_the_cpu() {
    let gpu = common::Gpu::new();
    let (n, h, taps) = (6, 128, 5);
    let (bits, x) = halves(&common::values(n * h, 71));
    let (w, g) = (common::values(h * taps, 72), common::values(h, 73));
    let (xb, wb, gb, out) = (gpu.upload(&bits), gpu.upload(&w), gpu.upload(&g), gpu.zeros(2 * n * h));
    let values = [(h as u32, false), (taps as u32, false), (1e-6f32.to_bits(), true)];
    launch(&gpu, tachyon::ptx::LCONV, ([n as u32, 1, 1], h as u32 / 4), &[xb.ptr(), wb.ptr(), gb.ptr(), out.ptr()], &values);
    let got = gpu.halves(&out);
    for t in 0..n {
        let mut y: Vec<f64> = (0..h)
            .map(|c| (0..taps).filter(|j| t + 1 + j >= taps).map(|j| f64::from(w[c * taps + j]) * f64::from(x[(t + 1 + j - taps) * h + c])).sum())
            .collect();
        common::normalize(&mut y, Some(&g));
        for (c, v) in y.iter().enumerate() {
            let want = v / (1.0 + (-v).exp());
            let g = f64::from(f16_value(got[t * h + c]));
            assert!((g - want).abs() <= 1e-3 * (1.0 + want.abs()), "token {t}, channel {c}: {g} against {want}");
        }
    }
}

/// `v` rounded to `f16`, as bits and as the values they hold.
#[cfg(feature = "gpu")]
fn halves(v: &[f32]) -> (Vec<u16>, Vec<f32>) {
    let bits: Vec<u16> = v.iter().map(|&x| f16_bits(x)).collect();
    (bits.clone(), bits.iter().map(|&b| f16_value(b)).collect())
}

/// A tiny random tower (`scripts/media.py tiny-audio`: hidden 128, one head, two layers, clip bounds that bite)
/// converted and run on `speech-like.wav`, against `transformers`' float32 tower and embedder: no checkpoint
/// needed, every step of the conversion and the pipeline checked. The same bounds as the real tower's.
#[cfg(feature = "gpu")]
#[test]
fn a_tiny_tower_matches_transformers() {
    let fixtures = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    let (src, dst) = (std::path::Path::new(fixtures).join("tiny-audio"), std::env::temp_dir().join(format!("tachyon-tiny-audio-{}", std::process::id())));
    std::fs::create_dir_all(&dst).unwrap();
    let config = std::fs::read(src.join("config.json")).unwrap();
    tachyon::audio::convert(&tachyon::quant::Safetensors::open(&src).unwrap(), &config, &dst).unwrap();
    std::fs::write(dst.join("config.json"), &config).unwrap();
    let ctx = tachyon::cuda::Context::new(0).unwrap();
    let mut tower = tachyon::audio::Tower::open(&ctx, &dst).unwrap();
    let audio = tachyon::media::decode_wav(&std::fs::read(format!("{fixtures}/media/speech-like.wav")).unwrap()).unwrap();
    let got = tower.soft_tokens(&audio).unwrap();
    let want: Vec<f32> = std::fs::read(src.join("soft.f32")).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    std::fs::remove_dir_all(&dst).unwrap();
    assert_eq!(got.len(), want.len(), "soft tokens");
    let (cos, err) = compare(&got, &want, tower.width());
    println!("tiny tower: min cosine {cos:.6}, max error {err:.2e} of the largest");
    assert!(cos >= 0.999 && err <= 0.03, "min cosine {cos}, max error {err} of the largest");
}

/// Gemma 4 E4B's audio tower (in `$TACHYON_MODEL`, by default `~/.local/share/tachyon/gemma-4-e4b`, converted with
/// its `audio.wpk`) against Hugging Face's float32 tower on the two speech clips: every soft token within the f16
/// study's bounds, a cosine of at least 0.999 and a largest error of at most 3% of the largest value.
#[cfg(all(feature = "gpu", feature = "models"))]
#[test]
fn the_tower_matches_hugging_faces_soft_tokens() {
    use tachyon::json::Json;
    let dir = std::env::var("TACHYON_MODEL")
        .map_or_else(|_| std::path::PathBuf::from(std::env::var("HOME").unwrap()).join(".local/share/tachyon/gemma-4-e4b"), Into::into);
    let media = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/media");
    let ctx = tachyon::cuda::Context::new(0).unwrap();
    let mut tower = tachyon::audio::Tower::open(&ctx, &dir).unwrap();
    for line in std::fs::read_to_string(format!("{media}/speech.ref")).unwrap().lines() {
        let case = Json::parse(line.as_bytes()).unwrap();
        let (clip, soft) = (case.get("clip").and_then(Json::str).unwrap(), case.get("soft").and_then(Json::str).unwrap());
        let audio = tachyon::media::decode_wav(&std::fs::read(format!("{media}/{clip}")).unwrap()).unwrap();
        let start = std::time::Instant::now();
        let got = tower.soft_tokens(&audio).unwrap();
        let took = start.elapsed();
        let want: Vec<f32> = std::fs::read(format!("{media}/{soft}")).unwrap().chunks_exact(2).map(|b| f16_value(u16::from_le_bytes([b[0], b[1]]))).collect();
        assert_eq!(got.len(), want.len(), "{clip}: {} soft tokens against {}", got.len() / tower.width(), want.len() / tower.width());
        let (cos, err) = compare(&got, &want, tower.width());
        println!("{clip}: {} soft tokens in {took:.2?}, min cosine {cos:.6}, max error {err:.2e} of the largest", want.len() / tower.width());
        assert!(cos >= 0.999 && err <= 0.03, "{clip}: min cosine {cos}, max error {err} of the largest");
    }
}

/// The smallest cosine between rows of `got` and `want`, and the largest error relative to the largest of `want`.
#[cfg(feature = "gpu")]
fn compare(got: &[f32], want: &[f32], width: usize) -> (f64, f64) {
    let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum::<f64>();
    let cos = got.chunks(width).zip(want.chunks(width)).map(|(g, w)| dot(g, w) / (dot(g, g) * dot(w, w)).sqrt()).fold(1.0, f64::min);
    let top = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let err = got.iter().zip(want).fold(0.0f32, |m, (g, w)| m.max((g - w).abs()));
    (cos, f64::from(err / top))
}
