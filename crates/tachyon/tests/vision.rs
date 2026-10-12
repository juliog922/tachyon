//! Gemma 4's vision tower: its kernels against the CPU, and on the real checkpoint, its soft tokens against the
//! float32 tower of Hugging Face (`scripts/media.py vision`, files `soft-*.f16` of `tests/fixtures/media/`).
#![cfg(feature = "gpu")]
// Kernel arguments and the maths read as their names: `a`, `w`, `x`, `y`, `h`.
#![allow(clippy::many_single_char_names)]

mod common;

use tachyon::cuda::{DevBuf, arg};
use tachyon::quant::{f16_bits, f16_value};

/// `v` rounded to `f16`, as bits and as the values they hold.
fn halves(v: &[f32]) -> (Vec<u16>, Vec<f32>) {
    let bits: Vec<u16> = v.iter().map(|&x| f16_bits(x)).collect();
    (bits.clone(), bits.iter().map(|&b| f16_value(b)).collect())
}

/// Launches `kernel` with pointers then 32-bit values (`u32` or `f32` bits, as `kinds` says: `true` for `f32`).
fn launch(gpu: &common::Gpu, kernel: &str, shape: ([u32; 3], u32), pointers: &[u64], values: &[(u32, bool)]) {
    let (grid, threads) = shape;
    let floats: Vec<f32> = values.iter().map(|&(v, _)| f32::from_bits(v)).collect();
    let args: Vec<_> = pointers.iter().map(arg).chain(values.iter().zip(&floats).map(|(v, f)| if v.1 { arg(f) } else { arg(&v.0) })).collect();
    // SAFETY: each test passes the kernel's arguments in order, on buffers of the sizes `crate::ptx` documents.
    unsafe { gpu.stream.launch(&gpu.module.function(kernel).unwrap(), grid, [threads, 1, 1], 0, &args) }.unwrap();
    gpu.stream.sync().unwrap();
}

/// GELU in its tanh form, as Gemma's towers take it.
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x * x)).tanh())
}

/// The clipped product in modes 0 (`f32`), 2 (`f16`) and 4 (gated, GELU's tanh form) for 70 tokens, in blocks of
/// 64 and 128: inputs clamped to their bounds, each row's outputs to its own, tight enough that both bite.
#[test]
fn the_clipped_product_clamps_its_input_and_each_rows_output() {
    let gpu = common::Gpu::new();
    let (rows, cols, tokens) = (128, 192, 70);
    let raw: Vec<f32> = common::values(128 * cols, 11).iter().map(|v| 2.0 * v).collect();
    let ((a16, a), (w16, w)) = (halves(&raw), halves(&common::values(rows * cols, 12)));
    let (lo, hi) = (-0.75f32, 1.25f32);
    let bounds: Vec<[f32; 2]> = (0..rows).map(|r| [-1.5 - (r % 5) as f32, 2.0 + (r % 3) as f32]).collect();
    let clip: Vec<f32> = [lo, hi, 0.0, 0.0].into_iter().chain(bounds.iter().flatten().copied()).collect();
    let dot = |t: usize, r: usize| {
        let y: f32 = (0..cols).map(|k| a[t * cols + k].clamp(f16_value(f16_bits(lo)), f16_value(f16_bits(hi))) * w[r * cols + k]).sum();
        y.clamp(bounds[r][0], bounds[r][1])
    };
    let (ab, wb, cb) = (gpu.upload(&a16), gpu.upload(&w16), gpu.upload(&clip));
    for (mode, (kernel, blocks, threads)) in
        [0, 2, 4].into_iter().flat_map(|m| [(tachyon::ptx::GEMM_CLIP, 4, 128), (tachyon::ptx::GEMM_CLIP_128, 2, 256)].map(|k| (m, k)))
    {
        let y = gpu.zeros(4 * tokens * rows);
        let sizes = [rows, cols, tokens, mode as usize].map(|v| (v as u32, false));
        let (pointers, grid) = ([ab.ptr(), wb.ptr(), 0, y.ptr(), cb.ptr()], ([blocks, 1, 1], threads));
        // The clip pointer comes last, after the four sizes.
        let floats: Vec<_> = pointers[..4].iter().map(arg).chain(sizes.iter().map(|s| arg(&s.0))).chain([arg(&pointers[4])]).collect();
        // SAFETY: the kernel's arguments in order; `a` holds whole blocks of tokens, `y` room for all of them.
        unsafe { gpu.stream.launch(&gpu.module.function(kernel).unwrap(), grid.0, [grid.1, 1, 1], 0, &floats) }.unwrap();
        let (f32s, f16s) = (gpu.floats(&y), gpu.halves(&y));
        for (t, r) in (0..tokens).flat_map(|t| (0..rows).map(move |r| (t, r))) {
            let (got, want) = match mode {
                0 => (f32s[t * rows + r], dot(t, r)),
                2 => (f16_value(f16s[t * rows + r]), dot(t, r)),
                _ if r % 2 == 0 => (f16_value(f16s[t * rows / 2 + r / 2]), gelu(dot(t, r)) * dot(t, r + 1)),
                _ => continue,
            };
            assert!((got - want).abs() <= 1e-3 + 2e-3 * want.abs(), "{kernel} mode {mode}, token {t}, row {r}: {got} against {want}");
        }
    }
}

/// `1 / √(mean(v²) + eps)`, in `f64`.
fn rms(v: &[f32], eps: f64) -> f64 {
    common::rms(v.iter().map(|&x| f64::from(x)), eps)
}

/// The residual step and the norm: `x += w1 ⊙ rms(y)`, `a = w2 ⊙ rms(x)`; then without `y` or weights.
#[test]
fn the_rms_norm_adds_the_residual_and_normalizes() {
    let gpu = common::Gpu::new();
    let (n, h, eps) = (3, 256, 1e-6f32);
    let (x0, y, w1, w2) = (common::values(n * h, 21), common::values(n * h, 22), common::values(h, 23), common::values(h, 24));
    let (xb, yb, w1b, w2b, ab) = (gpu.upload(&x0), gpu.upload(&y), gpu.upload(&w1), gpu.upload(&w2), gpu.zeros(2 * n * h));
    let values = [(h as u32, false), (eps.to_bits(), true)];
    launch(&gpu, tachyon::ptx::RMS_F16, ([n as u32, 1, 1], h as u32 / 4), &[xb.ptr(), yb.ptr(), w1b.ptr(), w2b.ptr(), ab.ptr()], &values);
    let (x, a) = (gpu.floats(&xb), gpu.halves(&ab));
    for r in 0..n {
        let (row, yr) = (&x0[r * h..][..h], &y[r * h..][..h]);
        let sum: Vec<f32> = (0..h).map(|i| (f64::from(row[i]) + f64::from(w1[i]) * f64::from(yr[i]) * rms(yr, 1e-6)) as f32).collect();
        let scale = rms(&sum, 1e-6);
        for i in 0..h {
            assert!((x[r * h + i] - sum[i]).abs() <= 1e-5, "row {r}, value {i}: x {} against {}", x[r * h + i], sum[i]);
            let want = (f64::from(sum[i]) * scale * f64::from(w2[i])) as f32;
            assert!((f16_value(a[r * h + i]) - want).abs() <= 1e-3 * (1.0 + want.abs()), "row {r}, value {i}: a against {want}");
        }
    }
    launch(&gpu, tachyon::ptx::RMS_F16, ([n as u32, 1, 1], h as u32 / 4), &[xb.ptr(), 0, 0, 0, ab.ptr()], &values);
    let (again, a) = (gpu.floats(&xb), gpu.halves(&ab));
    assert_eq!(again, x, "without y, x stays");
    for (i, &v) in x.iter().enumerate() {
        let want = (f64::from(v) * rms(&x[i / h * h..][..h], 1e-6)) as f32;
        assert!((f16_value(a[i]) - want).abs() <= 1e-3 * (1.0 + want.abs()), "value {i}: unweighted a against {want}");
    }
}

/// A head of 64 as [`tachyon::ptx::HEADS_F16`] computes it, in `f64`: normalized, weighted (queries and keys), and
/// turned by `x` on its first half and `y` on its second (queries and keys).
fn head(v: &[f32], w: Option<&[f32]>, (x, y): (u32, u32), freq: &[f32]) -> Vec<f64> {
    let mut out: Vec<f64> = v.iter().map(|&a| f64::from(a)).collect();
    common::normalize(&mut out, w);
    if w.is_some() {
        common::rotate(&mut out[..32], freq, x);
        common::rotate(&mut out[32..], freq, y);
    }
    out
}

/// The per-head norms and 2-D RoPE, for 3 tokens of 2 heads.
#[test]
fn the_heads_are_normalized_and_turned_in_two_dimensions() {
    let gpu = common::Gpu::new();
    let (n, h, positions) = (3, 128, 40u32);
    let (bits, qkv) = halves(&common::values(n * 3 * h, 31).iter().map(|v| 3.0 * v).collect::<Vec<_>>());
    let (qw, kw) = (common::values(64, 32), common::values(64, 33));
    let (xs, ys) = (vec![0u32, 7, 39], vec![5u32, 0, 31]);
    let freq: Vec<f32> = (0..16).map(|j| 1.0 / 100f32.powf(j as f32 / 16.0)).collect();
    let table: Vec<f32> =
        (0..positions).flat_map(|p| freq.iter().flat_map(move |&f| [f64::from(p as f32 * f).cos() as f32, f64::from(p as f32 * f).sin() as f32])).collect();
    let bufs: Vec<DevBuf> = vec![gpu.upload(&bits), gpu.upload(&xs), gpu.upload(&ys), gpu.upload(&table), gpu.upload(&qw), gpu.upload(&kw)];
    let pointers: Vec<u64> = bufs.iter().map(DevBuf::ptr).collect();
    launch(&gpu, tachyon::ptx::HEADS_F16, ([3, n as u32, 1], 32 * h as u32 / 64), &pointers, &[(h as u32, false), (1e-6f32.to_bits(), true)]);
    let got = gpu.halves(&bufs[0]);
    for (t, part, hd) in (0..n).flat_map(|t| (0..3).flat_map(move |p| (0..h / 64).map(move |hd| (t, p, hd)))) {
        let at = t * 3 * h + part * h + hd * 64;
        let w = [Some(&qw[..]), Some(&kw[..]), None][part];
        let want = head(&qkv[at..at + 64], w, (xs[t], ys[t]), &freq);
        for (i, &v) in want.iter().enumerate() {
            let g = f64::from(f16_value(got[at + i]));
            assert!((g - v).abs() <= 2e-3 * (1.0 + v.abs()), "token {t}, part {part}, head {hd}, dimension {i}: {g} against {v}");
        }
    }
}

/// The 3 × 3 pooling of a grid 6 patches across and 3 down: two soft tokens, of 300 values (not a whole block).
#[test]
fn the_pool_averages_each_square_of_patches() {
    let gpu = common::Gpu::new();
    let (gw, gh, h) = (6, 3, 300);
    let x = common::values(gw * gh * h, 41);
    let (xb, out) = (gpu.upload(&x), gpu.zeros(4 * 2 * h));
    launch(&gpu, tachyon::ptx::POOL_F16, ([2, 2, 1], 256), &[xb.ptr(), out.ptr()], &[(gw as u32, false), (h as u32, false)]);
    let got = gpu.floats(&out);
    for (sq, i) in (0..2).flat_map(|sq| (0..h).map(move |i| (sq, i))) {
        let want: f32 = (0..9).map(|k| x[((k / 3) * gw + 3 * sq + k % 3) * h + i]).sum::<f32>() / 9.0;
        assert!((got[sq * h + i] - want).abs() <= 1e-6, "square {sq}, value {i}: {} against {want}", got[sq * h + i]);
    }
}

/// Gemma 4 E4B's tower (in `$TACHYON_MODEL`, by default `~/.local/share/tachyon/gemma-4-e4b`, converted with its
/// `vision.wpk`) against Hugging Face's float32 tower, at 70 soft tokens: every soft token within the f16 study's
/// bounds, a cosine of at least 0.999 and a largest error of at most 3% of the largest value.
#[cfg(feature = "models")]
#[test]
fn the_tower_matches_hugging_faces_soft_tokens() {
    use tachyon::json::Json;
    let dir = std::env::var("TACHYON_MODEL")
        .map_or_else(|_| std::path::PathBuf::from(std::env::var("HOME").unwrap()).join(".local/share/tachyon/gemma-4-e4b"), Into::into);
    let media = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/media");
    let ctx = tachyon::cuda::Context::new(0).unwrap();
    let mut tower = tachyon::vision::Vision::open(&ctx, &dir, 70).unwrap();
    for line in std::fs::read_to_string(format!("{media}/vision.ref")).unwrap().lines() {
        let case = Json::parse(line.as_bytes()).unwrap();
        let Some(soft) = case.get("soft").and_then(Json::str) else { continue };
        let image = tachyon::media::decode(&std::fs::read(format!("{media}/{}", case.get("image").and_then(Json::str).unwrap())).unwrap()).unwrap();
        let start = std::time::Instant::now();
        let got = tower.soft_tokens(&image, 70).unwrap();
        let took = start.elapsed();
        let want: Vec<f32> = std::fs::read(format!("{media}/{soft}")).unwrap().chunks_exact(2).map(|b| f16_value(u16::from_le_bytes([b[0], b[1]]))).collect();
        assert_eq!(got.len(), want.len(), "{soft}: {} soft tokens against {}", got.len() / tower.width(), want.len() / tower.width());
        let (cos, err) = compare(&got, &want, tower.width());
        println!("{soft}: {} soft tokens in {took:.2?}, min cosine {cos:.6}, max error {err:.2e} of the largest", want.len() / tower.width());
        assert!(cos >= 0.999 && err <= 0.03, "{soft}: min cosine {cos}, max error {err} of the largest");
    }
}

/// The smallest cosine between rows of `got` and `want`, and the largest error relative to the largest of `want`.
#[cfg(feature = "models")]
fn compare(got: &[f32], want: &[f32], width: usize) -> (f64, f64) {
    let cos = got
        .chunks(width)
        .zip(want.chunks(width))
        .map(|(g, w)| {
            let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum::<f64>();
            dot(g, w) / (dot(g, g) * dot(w, w)).sqrt()
        })
        .fold(1.0, f64::min);
    let top = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let err = got.iter().zip(want).fold(0.0f32, |m, (g, w)| m.max((g - w).abs()));
    (cos, f64::from(err / top))
}
