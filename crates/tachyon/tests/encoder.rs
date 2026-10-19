//! The ModernBERT encoder and Laya's decision head on the CPU, in float32, as plainly as the maths reads: the
//! reference the GPU encoder is tested against. It is checked here against PyTorch on the tiny random Laya of
//! `scripts/encoders.py tiny` (fixture `tiny-encoder/`): the `laya` package's own model on transformers' ModernBERT.
// The reference reads as the maths does: `q`, `k`, `h`, `n`, `x` and their kind.
#![allow(clippy::many_single_char_names)]

use std::path::Path;
use tachyon::encoder::Config;
use tachyon::encoder::laya::Head;
use tachyon::json::Json;
use tachyon::quant::Safetensors;

const TINY: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/tiny-encoder");

/// A checkpoint's tensors as `f32`, by name.
struct Weights {
    st: Safetensors,
    prefix: &'static str,
}

impl Weights {
    fn get(&self, name: &str) -> Vec<f32> {
        let name = format!("{}{name}", self.prefix);
        let rows = self.st.dims(&name).unwrap().0;
        self.st.rows(&name, 0..rows).unwrap()
    }
}

/// `x` (rows of `w.len() / out` values) times `wᵀ` (`out` rows), plus `bias`.
fn linear(x: &[f32], w: &[f32], bias: Option<&[f32]>, out: usize) -> Vec<f32> {
    let cols = w.len() / out;
    x.chunks(cols)
        .flat_map(|row| (0..out).map(move |o| row.iter().zip(&w[o * cols..][..cols]).map(|(a, b)| a * b).sum::<f32>() + bias.map_or(0.0, |b| b[o])))
        .collect()
}

/// Layer norm of each row of `x`, with an optional bias.
fn norm(x: &[f32], w: &[f32], bias: Option<&[f32]>, eps: f32) -> Vec<f32> {
    x.chunks(w.len())
        .flat_map(|row| {
            let n = row.len() as f32;
            let mean = row.iter().sum::<f32>() / n;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
            let r = 1.0 / (var + eps).sqrt();
            row.iter().enumerate().map(move |(i, v)| (v - mean) * r * w[i] + bias.map_or(0.0, |b| b[i]))
        })
        .collect()
}

/// The error function, to 1.5e-7 (Abramowitz and Stegun 7.1.26).
fn erf(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let p = t * (0.254_829_6 + t * (-0.284_496_74 + t * (1.421_413_7 + t * (-1.453_152_1 + t * 1.061_405_4))));
    (1.0 - p * (-x * x).exp()).copysign(x)
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x / std::f32::consts::SQRT_2))
}

/// Attention of each head over the rows of `q`, `k`, `v` (`n` rows of `heads × d`), each row seeing the rows within
/// `window` of it, or all of them; RoPE at base `theta` first, if any.
#[allow(dead_code)]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn attend(qkv: &[f32], n: usize, heads: usize, d: usize, window: Option<usize>, theta: Option<f32>) -> Vec<f32> {
    let h = heads * d;
    let rotate = |row: usize, part: usize| -> Vec<f32> {
        let mut x = qkv[row * 3 * h + part * h..][..h].to_vec();
        let Some(theta) = theta else { return x };
        for head in x.chunks_mut(d) {
            for j in 0..d / 2 {
                let angle = row as f32 * theta.powf(-2.0 * j as f32 / d as f32);
                let (c, s) = (angle.cos(), angle.sin());
                let (a, b) = (head[j], head[j + d / 2]);
                (head[j], head[j + d / 2]) = (a * c - b * s, b * c + a * s);
            }
        }
        x
    };
    let (q, k): (Vec<Vec<f32>>, Vec<Vec<f32>>) = (0..n).map(|r| (rotate(r, 0), rotate(r, 1))).unzip();
    let mut out = vec![0.0; n * h];
    for i in 0..n {
        for hd in 0..heads {
            let seen: Vec<usize> = (0..n).filter(|&j| window.is_none_or(|w| i.abs_diff(j) <= w)).collect();
            let scores: Vec<f32> = seen.iter().map(|&j| (0..d).map(|t| q[i][hd * d + t] * k[j][hd * d + t]).sum::<f32>() / (d as f32).sqrt()).collect();
            let top = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = scores.iter().map(|s| (s - top).exp()).collect();
            let sum: f32 = e.iter().sum();
            for (&j, p) in seen.iter().zip(&e) {
                for t in 0..d {
                    out[i * h + hd * d + t] += p / sum * qkv[j * 3 * h + 2 * h + hd * d + t];
                }
            }
        }
    }
    out
}

/// The encoder's last hidden states for `ids`.
fn encode(cfg: &Config, w: &Weights, ids: &[u32]) -> Vec<f32> {
    let (h, n, eps) = (cfg.hidden, ids.len(), cfg.eps);
    let table = w.get("embeddings.tok_embeddings.weight");
    let embedded: Vec<f32> = ids.iter().flat_map(|&t| table[t as usize * h..][..h].to_vec()).collect();
    let mut x = norm(&embedded, &w.get("embeddings.norm.weight"), None, eps);
    for (l, &global) in cfg.global.iter().enumerate() {
        let p = format!("layers.{l}.");
        let a = if l == 0 { x.clone() } else { norm(&x, &w.get(&format!("{p}attn_norm.weight")), None, eps) };
        let qkv = linear(&a, &w.get(&format!("{p}attn.Wqkv.weight")), None, 3 * h);
        let (window, theta) = if global { (None, cfg.theta[0]) } else { (Some(cfg.window), cfg.theta[1]) };
        let theta = Some(theta);
        let attn = linear(&attend(&qkv, n, cfg.heads, cfg.head_dim(), window, theta), &w.get(&format!("{p}attn.Wo.weight")), None, h);
        x.iter_mut().zip(attn).for_each(|(x, a)| *x += a);
        let m = norm(&x, &w.get(&format!("{p}mlp_norm.weight")), None, eps);
        let up = linear(&m, &w.get(&format!("{p}mlp.Wi.weight")), None, 2 * cfg.inter);
        let gated: Vec<f32> = up.chunks(2 * cfg.inter).flat_map(|r| (0..cfg.inter).map(move |i| gelu(r[i]) * r[cfg.inter + i])).collect();
        let down = linear(&gated, &w.get(&format!("{p}mlp.Wo.weight")), None, h);
        x.iter_mut().zip(down).for_each(|(x, d)| *x += d);
    }
    norm(&x, &w.get("final_norm.weight"), None, eps)
}

/// A Laya checkpoint: its encoder's shape, its head's settings, its tensors.
struct Laya<'a> {
    cfg: &'a Config,
    head: &'a Head,
    w: &'a Weights,
}

/// Laya's option logits: the type embedding added to the encoder's states, the head's pre-norm layers (ReLU,
/// `hidden / 64` heads), and the scorer at each marker.
fn decide(hidden: &[f32], laya: &Laya, markers: &[usize], kind: usize) -> Vec<f32> {
    let Laya { cfg, head, w } = *laya;
    let (h, n) = (cfg.hidden, hidden.len() / cfg.hidden);
    let kinds = w.get("type_emb.weight");
    let mut x: Vec<f32> = hidden.chunks(h).flat_map(|r| r.iter().zip(&kinds[kind * h..]).map(|(a, b)| a + b).collect::<Vec<_>>()).collect();
    for l in 0..head.layers {
        let g = |s: &str| w.get(&format!("head.layers.{l}.{s}"));
        let a = norm(&x, &g("norm1.weight"), Some(&g("norm1.bias")), 1e-5);
        let qkv = linear(&a, &g("self_attn.in_proj_weight"), Some(&g("self_attn.in_proj_bias")), 3 * h);
        let attn = attend(&qkv, n, h / 64, 64, None, None);
        let out = linear(&attn, &g("self_attn.out_proj.weight"), Some(&g("self_attn.out_proj.bias")), h);
        x.iter_mut().zip(out).for_each(|(x, a)| *x += a);
        let m = norm(&x, &g("norm2.weight"), Some(&g("norm2.bias")), 1e-5);
        let up: Vec<f32> = linear(&m, &g("linear1.weight"), Some(&g("linear1.bias")), 4 * h).into_iter().map(|v| v.max(0.0)).collect();
        let down = linear(&up, &g("linear2.weight"), Some(&g("linear2.bias")), h);
        x.iter_mut().zip(down).for_each(|(x, d)| *x += d);
    }
    let picked: Vec<f32> = markers.iter().flat_map(|&m| x[m * h..][..h].to_vec()).collect();
    let s = norm(&picked, &w.get("scorer.0.weight"), Some(&w.get("scorer.0.bias")), 1e-5);
    let s: Vec<f32> = linear(&s, &w.get("scorer.1.weight"), Some(&w.get("scorer.1.bias")), h).into_iter().map(gelu).collect();
    linear(&s, &w.get("scorer.3.weight"), Some(&w.get("scorer.3.bias")), 1)
}

fn numbers(v: Option<&Json>) -> Vec<f64> {
    v.and_then(Json::arr).unwrap().iter().map(|x| x.num().unwrap()).collect()
}

#[test]
fn the_reference_encoder_and_head_match_pytorch() {
    let dir = Path::new(TINY);
    let cfg = Config::from_json(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
    let head = Head::from_json(&std::fs::read(dir.join("rl_agent_config.json")).unwrap()).unwrap();
    assert_eq!((cfg.global.clone(), cfg.window, cfg.theta.map(|t| t as u32)), (vec![true, false, false], 8, [160_000, 10_000]));
    let enc = Weights { st: Safetensors::open(dir).unwrap(), prefix: "encoder." };
    let top = Weights { st: Safetensors::open(dir).unwrap(), prefix: "" };
    for (i, line) in std::fs::read_to_string(dir.join("outputs.ref")).unwrap().lines().enumerate() {
        let case = Json::parse(line.as_bytes()).unwrap();
        let ids: Vec<u32> = numbers(case.get("ids")).iter().map(|&v| v as u32).collect();
        let markers: Vec<usize> = numbers(case.get("markers")).iter().map(|&v| v as usize).collect();
        let hidden = encode(&cfg, &enc, &ids);
        let worst = hidden.iter().zip(numbers(case.get("hidden"))).map(|(a, b)| (f64::from(*a) - b).abs()).fold(0.0, f64::max);
        assert!(worst < 1e-4, "sequence {i} ({} tokens): hidden states off by {worst}", ids.len());
        let kind = case.get("qtype").and_then(Json::num).unwrap() as usize;
        let logits = decide(&hidden, &Laya { cfg: &cfg, head: &head, w: &top }, &markers, kind);
        let worst = logits.iter().zip(numbers(case.get("logits"))).map(|(a, b)| (f64::from(*a) - b).abs()).fold(0.0, f64::max);
        assert!(worst < 1e-4, "sequence {i}: logits off by {worst}: {logits:?}");
    }
}

/// The real checkpoints, as their Hugging Face repositories lay them out in `$TACHYON_MODELS`, against the
/// references of `scripts/encoders.py`: Laya multilingual on its three shortest cases (probabilities within 1e-4),
/// Granite Embedding 311M on short texts and on one of 251 tokens, past the 128-token window (cosine ≥ 0.99999).
/// Slow: the reference is plain loops.
#[cfg(feature = "models")]
#[test]
fn the_reference_matches_the_real_checkpoints() {
    let models = std::path::PathBuf::from(std::env::var("TACHYON_MODELS").expect("TACHYON_MODELS: the models' repositories"));
    let fixtures = Path::new(TINY).parent().unwrap();
    let dir = models.join("laya-multilingual");
    let cfg = Config::from_json(&std::fs::read(dir.join("encoder/config.json")).unwrap()).unwrap();
    let head = Head::from_json(&std::fs::read(dir.join("rl_agent_config.json")).unwrap()).unwrap();
    let (enc, top) = (Weights { st: Safetensors::open(&dir).unwrap(), prefix: "encoder." }, Weights { st: Safetensors::open(&dir).unwrap(), prefix: "" });
    let cases = std::fs::read_to_string(fixtures.join("laya-multilingual.ref")).unwrap();
    let mut cases: Vec<Json> = cases.lines().map(|l| Json::parse(l.as_bytes()).unwrap()).collect();
    cases.sort_by_key(|c| numbers(c.get("ids")).len());
    for case in &cases[..3] {
        let ids: Vec<u32> = numbers(case.get("ids")).iter().map(|&v| v as u32).collect();
        let markers: Vec<usize> = numbers(case.get("markers")).iter().map(|&v| v as usize).collect();
        let kind = case.get("qtype").and_then(Json::num).unwrap() as usize;
        let logits = decide(&encode(&cfg, &enc, &ids), &Laya { cfg: &cfg, head: &head, w: &top }, &markers, kind);
        let p = tachyon::encoder::laya::probabilities(&logits, 1.0);
        let worst = p.iter().zip(numbers(case.get("probs"))).map(|(a, b)| (f64::from(*a) - b).abs()).fold(0.0, f64::max);
        assert!(worst < 1e-4, "Laya, {} tokens: probabilities off by {worst}", ids.len());
    }
    let dir = models.join("granite-embedding-311m-multilingual-r2");
    let cfg = Config::from_json(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
    assert_eq!(cfg.theta.map(|t| t as u32), [150_000, 160_000], "Granite's bases, from transformers 4's keys");
    let st = Safetensors::open(&dir).unwrap();
    let prefix = st.prefix("embeddings.norm.weight");
    let w = Weights { st, prefix };
    let texts = std::fs::read_to_string(fixtures.join("granite-311m.ref")).unwrap();
    for i in [43, 0, 41] {
        let case = Json::parse(texts.lines().nth(i).unwrap().as_bytes()).unwrap();
        let ids: Vec<u32> = numbers(case.get("ids")).iter().map(|&v| v as u32).collect();
        let cls = encode(&cfg, &w, &ids)[..cfg.hidden].to_vec();
        let len = cls.iter().map(|v| v * v).sum::<f32>().sqrt();
        let cosine: f64 = cls.iter().zip(numbers(case.get("embedding"))).map(|(a, b)| f64::from(a / len) * b).sum();
        assert!(cosine > 0.999_99, "Granite, text {i} ({} tokens): cosine {cosine}", ids.len());
    }
}

#[cfg(feature = "gpu")]
mod common;

/// `v` rounded to `f16`, as bits and as the values they hold.
#[cfg(feature = "gpu")]
fn halves(v: &[f32]) -> (Vec<u16>, Vec<f32>) {
    let bits: Vec<u16> = v.iter().map(|&x| tachyon::quant::f16_bits(x)).collect();
    let back = bits.iter().map(|&b| tachyon::quant::f16_value(b)).collect();
    (bits, back)
}

/// The `f16` product in its four modes (store, add, `f16`, `f16` after ReLU), with a bias, for 70 tokens, in blocks
/// of 64 and 128 tokens (32 or 64 to a warp): the last block of tokens ends past them, and nothing is stored there.
#[cfg(feature = "gpu")]
#[test]
fn the_f16_product_matches_the_cpu_in_every_mode() {
    use tachyon::cuda::arg;
    let gpu = common::Gpu::new();
    let (rows, cols, tokens) = (128, 192, 70);
    let ((a16, a), (w16, w)) = (halves(&common::values(128 * cols, 1)), halves(&common::values(rows * cols, 2)));
    let (bias, base) = (common::values(rows, 3), common::values(128 * rows, 4));
    let want: Vec<f32> = (0..tokens)
        .flat_map(|t| (0..rows).map(move |r| (t, r)))
        .map(|(t, r)| (0..cols).map(|k| a[t * cols + k] * w[r * cols + k]).sum::<f32>() + bias[r])
        .collect();
    let (ab, wb, bb) = (gpu.upload(&a16), gpu.upload(&w16), gpu.upload(&bias));
    let kernels = [(tachyon::ptx::GEMM_F16, 4, 128), (tachyon::ptx::GEMM_F16_128, 2, 256), (tachyon::ptx::GEMM_F16_TALL, 1, 512)];
    for (mode, (kernel, blocks, threads)) in (0..4u32).flat_map(|m| kernels.map(|k| (m, k))) {
        let y = gpu.upload(&base);
        // The values live until the launch: `arg` takes their addresses.
        let (pointers, sizes) = ([ab.ptr(), wb.ptr(), bb.ptr(), y.ptr()], [rows as u32, cols as u32, tokens as u32, mode]);
        let args: Vec<_> = pointers.iter().map(arg).chain(sizes.iter().map(arg)).collect();
        // SAFETY: the kernel's arguments in order; `a` holds two tiles of tokens, `y` room for all of them.
        unsafe { gpu.stream.launch(&gpu.module.function(kernel).unwrap(), [blocks, 1, 1], [threads, 1, 1], 0, &args) }.unwrap();
        let got: Vec<f32> = if mode < 2 { gpu.floats(&y) } else { gpu.halves(&y).iter().map(|&h| tachyon::quant::f16_value(h)).collect() };
        for (i, &v) in want.iter().enumerate() {
            let expect = match mode {
                0 | 2 => v,
                1 => v + base[i],
                _ => v.max(0.0),
            };
            let slack = if mode < 2 { 1e-3 } else { 1e-3 + 2e-3 * expect.abs() };
            assert!((got[i] - expect).abs() <= slack, "{kernel} mode {mode}, token {}, row {}: {} against {expect}", i / rows, i % rows, got[i]);
        }
        let past: Vec<f32> = if mode < 2 { gpu.floats(&y)[tokens * rows..].to_vec() } else { Vec::new() };
        assert!(past.iter().zip(&base[tokens * rows..]).all(|(a, b)| a.to_bits() == b.to_bits()), "{kernel} mode {mode}: rows past the tokens were written");
    }
}

/// The `f16` product with the gated GELU and `SiLU` over interleaved rows (modes 4 and 5: `gelu(y[2j])·y[2j+1]` and
/// `silu(…)·…`, `f16`, half as wide), in each shape of block.
#[cfg(feature = "gpu")]
#[test]
fn the_f16_product_gates() {
    use tachyon::cuda::arg;
    let gpu = common::Gpu::new();
    let (rows, cols, tokens) = (128, 256, 70);
    let ((a16, a), (w16, w)) = (halves(&common::values(128 * cols, 5)), halves(&common::values(rows * cols, 6)));
    let bias = common::values(rows, 7);
    let dot = |t: usize, r: usize| (0..cols).map(|k| a[t * cols + k] * w[r * cols + k]).sum::<f32>() + bias[r];
    let (ab, wb, bb) = (gpu.upload(&a16), gpu.upload(&w16), gpu.upload(&bias));
    for (mode, (kernel, blocks, threads)) in [4u32, 5]
        .into_iter()
        .flat_map(|m| [(tachyon::ptx::GEMM_F16, 4, 128), (tachyon::ptx::GEMM_F16_128, 2, 256), (tachyon::ptx::GEMM_F16_TALL, 1, 512)].map(|k| (m, k)))
    {
        let y = gpu.upload(&vec![0f32; tokens * rows]);
        let (pointers, sizes) = ([ab.ptr(), wb.ptr(), bb.ptr(), y.ptr()], [rows as u32, cols as u32, tokens as u32, mode]);
        let args: Vec<_> = pointers.iter().map(arg).chain(sizes.iter().map(arg)).collect();
        // SAFETY: the kernel's arguments in order; `a` holds whole blocks of tokens, `y` room for all of them.
        unsafe { gpu.stream.launch(&gpu.module.function(kernel).unwrap(), [blocks, 1, 1], [threads, 1, 1], 0, &args) }.unwrap();
        let gated = gpu.halves(&y);
        for (t, r) in (0..tokens).flat_map(|t| (0..rows / 2).map(move |r| (t, 2 * r))) {
            let gate = if mode == 4 { gelu(dot(t, r)) } else { silu(dot(t, r)) };
            let (got, want) = (tachyon::quant::f16_value(gated[t * rows / 2 + r / 2]), gate * dot(t, r + 1));
            assert!((got - want).abs() <= 1e-3 + 2e-3 * want.abs(), "{kernel} mode {mode}, token {t}, row {r}: {got} against {want}");
        }
    }
}

/// Launches product `kernel` (its name, grid and threads) over `buffers` `[a, a's scales, w, w's scales, y]`
/// (Q8E) with `sizes` `[rows, cols, tokens, mode]`, `rows` being 128: two tiles of rows.
#[cfg(feature = "gpu")]
fn q8_product(gpu: &common::Gpu, kernel: (&str, u32, u32), buffers: [u64; 5], sizes: [u32; 4]) {
    use tachyon::cuda::arg;
    let (pointers, scales) = ([buffers[0], buffers[2], 0, buffers[4]], [buffers[1], buffers[3]]);
    let args: Vec<_> = pointers.iter().map(arg).chain(sizes.iter().map(arg)).chain(scales.iter().map(arg)).collect();
    // SAFETY: the kernel's arguments in order; `a` and its scales hold whole blocks of tokens, `y` room for all.
    unsafe { gpu.stream.launch(&gpu.module.function(kernel.0).unwrap(), [kernel.1, 1, 1], [kernel.2, 1, 1], 0, &args) }.unwrap();
}

/// `f16` rows rounded to Q8E on the GPU exactly as on the CPU; then the 8-bit product in every mode (store, add,
/// `f16`, ReLU, the gated GELU and `SiLU`) against the CPU's product of the rounded values, for 70 tokens, in each
/// shape of block.
#[cfg(feature = "gpu")]
#[test]
fn the_q8_product_matches_the_cpu_in_every_mode() {
    use tachyon::cuda::arg;
    use tachyon::quant::{f16_value, q8e};
    let gpu = common::Gpu::new();
    let (rows, cols, tokens) = (128, 192, 70);
    let ((a16, a), w) = (halves(&common::values(128 * cols, 8)), common::values(rows * cols, 9));
    let ((aq, sa), (wq, sw)) = (q8e(&a), q8e(&w));
    let (ab, qb, sb) = (gpu.upload(&a16), gpu.zeros(128 * cols), gpu.zeros(128 * cols / 16));
    let (pointers, width) = ([ab.ptr(), qb.ptr(), sb.ptr()], cols as u32);
    let args: Vec<_> = pointers.iter().map(arg).chain([arg(&width)]).collect();
    // SAFETY: QUANT_F16's arguments in order, on 128 rows of `cols`.
    unsafe { gpu.stream.launch(&gpu.module.function(tachyon::ptx::QUANT_F16).unwrap(), [1, 128, 1], [256, 1, 1], 0, &args) }.unwrap();
    assert!(gpu.download(&qb) == common::bytes(&aq) && gpu.floats(&sb) == sa, "the GPU's rounding differs from the CPU's");
    let value = |q: &[i8], s: &[f32], i: usize| f64::from(q[i]) * f64::from(s[i / 64]);
    let dot = |t: usize, r: usize| (0..cols).map(|k| value(&aq, &sa, t * cols + k) * value(&wq, &sw, r * cols + k)).sum::<f64>() as f32;
    let (wb, swb, base) = (gpu.upload(&wq), gpu.upload(&sw), common::values(128 * rows, 10));
    for (mode, kernel) in
        (0..6u32).flat_map(|m| [(tachyon::ptx::GEMM_Q8, 4, 128), (tachyon::ptx::GEMM_Q8_128, 2, 256), (tachyon::ptx::GEMM_Q8_TALL, 1, 512)].map(|k| (m, k)))
    {
        let y = gpu.upload(&base);
        q8_product(&gpu, kernel, [qb.ptr(), sb.ptr(), wb.ptr(), swb.ptr(), y.ptr()], [rows as u32, cols as u32, tokens as u32, mode]);
        let (f32s, f16s) = (gpu.floats(&y), gpu.halves(&y));
        let width = if mode >= 4 { rows / 2 } else { rows };
        for (t, r) in (0..tokens).flat_map(|t| (0..width).map(move |r| (t, r))) {
            let (got, want) = match mode {
                0 => (f32s[t * rows + r], dot(t, r)),
                1 => (f32s[t * rows + r], dot(t, r) + base[t * rows + r]),
                2 => (f16_value(f16s[t * rows + r]), dot(t, r)),
                3 => (f16_value(f16s[t * rows + r]), dot(t, r).max(0.0)),
                4 => (f16_value(f16s[t * width + r]), gelu(dot(t, 2 * r)) * dot(t, 2 * r + 1)),
                _ => (f16_value(f16s[t * width + r]), silu(dot(t, 2 * r)) * dot(t, 2 * r + 1)),
            };
            assert!((got - want).abs() <= 1e-3 + 2e-3 * want.abs(), "{} mode {mode}, token {t}, row {r}: {got} against {want}", kernel.0);
        }
    }
}

/// A tiny case: its tokens, markers, kind and the CPU reference's hidden states.
#[cfg(feature = "gpu")]
type Case = (Vec<u32>, Vec<usize>, usize, Vec<f32>);

/// The three cases of `outputs.ref`, with their CPU-reference hidden states.
#[cfg(feature = "gpu")]
fn tiny_cases() -> Vec<Case> {
    let dir = Path::new(TINY);
    let cfg = Config::from_json(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
    let enc = Weights { st: Safetensors::open(dir).unwrap(), prefix: "encoder." };
    let lines = std::fs::read_to_string(dir.join("outputs.ref")).unwrap();
    lines
        .lines()
        .map(|l| {
            let case = Json::parse(l.as_bytes()).unwrap();
            let ids: Vec<u32> = numbers(case.get("ids")).iter().map(|&v| v as u32).collect();
            let markers = numbers(case.get("markers")).iter().map(|&v| v as usize).collect();
            let hidden = encode(&cfg, &enc, &ids);
            (ids, markers, case.get("qtype").and_then(Json::num).unwrap() as usize, hidden)
        })
        .collect()
}

/// The GPU encoder and head on the tiny Laya, converted, against the CPU reference: hidden states and logits
/// within 2e-2 in `f16`. A capacity of 64 tokens splits the three sequences (7, 24, 40 tokens) into two batches.
#[cfg(feature = "gpu")]
#[test]
fn the_gpu_encoder_matches_the_reference_on_the_tiny_laya() {
    use tachyon::encoder::laya::Kind;
    use tachyon::encoder::{Encoder, Settings, convert};
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("tiny-encoder");
    convert(Path::new(TINY), &dir).unwrap();
    let ctx = tachyon::cuda::Context::new(0).unwrap();
    let mut enc = Encoder::open(&ctx, &dir, &Settings { tokens: 64 }).unwrap();
    let cases = tiny_cases();
    let seqs: Vec<&[u32]> = cases.iter().map(|c| c.0.as_slice()).collect();
    let hidden = enc.hidden(&seqs).unwrap();
    let want: Vec<f32> = cases.iter().flat_map(|c| c.3.clone()).collect();
    let worst = hidden.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    assert!(worst < 2e-2, "hidden states off by {worst}");
    let kinds = [Kind::Choice, Kind::Score, Kind::Noul];
    let questions: Vec<(&[u32], &[usize], Kind)> = cases.iter().map(|c| (c.0.as_slice(), c.1.as_slice(), kinds[c.2])).collect();
    let logits = enc.logits(&questions).unwrap();
    let lines = std::fs::read_to_string(Path::new(TINY).join("outputs.ref")).unwrap();
    for ((got, line), c) in logits.iter().zip(lines.lines()).zip(&cases) {
        let want = numbers(Json::parse(line.as_bytes()).unwrap().get("logits"));
        let worst = got.iter().zip(&want).map(|(a, b)| (f64::from(*a) - b).abs()).fold(0.0, f64::max);
        assert!(worst < 2e-2, "{} tokens: logits off by {worst}: {got:?} against {want:?}", c.0.len());
    }
}

/// The tiny Laya's encoder alone, converted as an embedding model with its layers' matrices in Q8 (too narrow to have
/// them by default): every
/// token's hidden state within a cosine of 0.99999 of the float32 reference's (a random model's layers add little).
#[cfg(feature = "gpu")]
#[test]
fn the_q8_encoder_follows_the_reference_on_the_tiny_model() {
    use tachyon::encoder::{Encoder, Settings, convert_with};
    let tmp = Path::new(env!("CARGO_TARGET_TMPDIR"));
    let (src, dir) = (tmp.join("tiny-embedder"), tmp.join("tiny-embedder-q8"));
    std::fs::create_dir_all(&src).unwrap();
    std::fs::copy(Path::new(TINY).join("config.json"), src.join("config.json")).unwrap();
    // The encoder's tensors under the names an embedding model's checkpoint gives them: `encoder.` dropped from the
    // header, whose offsets count from the data after it.
    let file = std::fs::read(Path::new(TINY).join("model.safetensors")).unwrap();
    let n = u64::from_le_bytes(file[..8].try_into().unwrap()) as usize;
    let header = String::from_utf8(file[8..8 + n].to_vec()).unwrap().replace("\"encoder.", "\"");
    std::fs::write(src.join("model.safetensors"), [&(header.len() as u64).to_le_bytes()[..], header.as_bytes(), &file[8 + n..]].concat()).unwrap();
    convert_with(&src, &dir, Some(true)).unwrap();
    let wpk = tachyon::wpk::Wpk::open(dir.join("weights.wpk")).unwrap();
    assert!(wpk.tensor("1.up").is_some_and(|t| t.dtype == tachyon::wpk::Dtype::I8) && wpk.tensor("1.up.scale").is_some());
    let ctx = tachyon::cuda::Context::new(0).unwrap();
    let mut enc = Encoder::open(&ctx, &dir, &Settings { tokens: 64 }).unwrap();
    let cases = tiny_cases();
    let seqs: Vec<&[u32]> = cases.iter().map(|c| c.0.as_slice()).collect();
    let (hidden, h) = (enc.hidden(&seqs).unwrap(), enc.config().hidden);
    let want: Vec<f32> = cases.iter().flat_map(|c| c.3.clone()).collect();
    let cosine = |a: &[f32], b: &[f32]| {
        let dot = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(p, q)| f64::from(*p) * f64::from(*q)).sum::<f64>();
        dot(a, b) / (dot(a, a) * dot(b, b)).sqrt()
    };
    let low = hidden.chunks(h).zip(want.chunks(h)).map(|(a, b)| cosine(a, b)).fold(1.0, f64::min);
    println!("tiny encoder in Q8: lowest cosine 1 − {:.2e}", 1.0 - low);
    assert!(low >= 0.99999, "lowest cosine {low:.6}");
}

/// The checkpoint `$TACHYON_MODELS/name`, converted once into the test's temporary directory.
#[cfg(all(feature = "gpu", feature = "models"))]
fn converted(name: &str) -> std::path::PathBuf {
    let dst = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    if !dst.join("host.wpk").exists() {
        let models = std::env::var("TACHYON_MODELS").expect("TACHYON_MODELS: the models' repositories");
        tachyon::encoder::convert(&Path::new(&models).join(name), &dst).unwrap();
    }
    dst
}

/// A Laya checkpoint (`$TACHYON_MODELS/name`) on the GPU against the float32 references of all 500 cases (fixture
/// `reference`): probabilities within `bounds` (on average, at most), the same answer in at least 499.
#[cfg(all(feature = "gpu", feature = "models"))]
fn laya_matches(name: &str, reference: &str, bounds: [f64; 2]) {
    use tachyon::encoder::laya::{Kind, probabilities};
    use tachyon::encoder::{Encoder, Settings};
    let ctx = tachyon::cuda::Context::new(0).unwrap();
    let mut enc = Encoder::open(&ctx, &converted(name), &Settings::default()).unwrap();
    let lines = std::fs::read_to_string(Path::new(TINY).parent().unwrap().join(reference)).unwrap();
    let cases: Vec<Json> = lines.lines().map(|l| Json::parse(l.as_bytes()).unwrap()).collect();
    let ids: Vec<Vec<u32>> = cases.iter().map(|c| numbers(c.get("ids")).iter().map(|&v| v as u32).collect()).collect();
    let marks: Vec<Vec<usize>> = cases.iter().map(|c| numbers(c.get("markers")).iter().map(|&v| v as usize).collect()).collect();
    let kinds = [Kind::Choice, Kind::Score, Kind::Noul];
    let kind = |c: &Json| kinds[c.get("qtype").and_then(Json::num).unwrap() as usize];
    let questions: Vec<(&[u32], &[usize], Kind)> = cases.iter().enumerate().map(|(i, c)| (ids[i].as_slice(), marks[i].as_slice(), kind(c))).collect();
    let logits = enc.logits(&questions).unwrap();
    let (mut sum, mut count, mut worst, mut same) = (0.0, 0, 0.0f64, 0);
    for (c, l) in cases.iter().zip(&logits) {
        let p = probabilities(l, enc.head().unwrap().temperature(kind(c), l.len()));
        let want = numbers(c.get("probs"));
        for (a, b) in p.iter().zip(&want) {
            let d = (f64::from(*a) - b).abs();
            (sum, count, worst) = (sum + d, count + 1, worst.max(d));
        }
        let top = |v: &[f64]| v.iter().enumerate().fold(0, |best, (i, x)| if *x > v[best] { i } else { best });
        same += usize::from(top(&p.iter().map(|&x| f64::from(x)).collect::<Vec<_>>()) == top(&want));
    }
    let mean = sum / f64::from(count);
    println!("{name} on the GPU: mean |Δp| {mean:.2e}, largest {worst:.2e}, same answer {same}/{}", cases.len());
    let [most_mean, most] = bounds;
    assert!(mean <= most_mean && worst <= most && same >= 499, "{name}: mean |Δp| {mean:.2e}, largest {worst:.2e}, same answer {same}/500");
}

#[cfg(all(feature = "gpu", feature = "models"))]
#[test]
fn laya_on_the_gpu_matches_the_reference() {
    laya_matches("laya-multilingual", "laya-multilingual.ref", [2e-4, 1e-2]);
}

/// Laya English: ModernBERT-large, its byte-level tokenizer, sequences of up to 512 tokens. `f16` rounding of its
/// matrices alone moves a probability by up to 2.4e-2 (1.4e-4 on average), so the bounds are those of the study
/// with room, still tighter than the `laya` package's own bf16 (6.5e-2, 1.1e-3).
#[cfg(all(feature = "gpu", feature = "models"))]
#[test]
fn laya_english_on_the_gpu_matches_the_reference() {
    laya_matches("laya", "laya-english.ref", [3e-4, 3e-2]);
}

#[cfg(all(feature = "gpu", feature = "models"))]
#[test]
fn laya_typed_decisions_on_the_gpu_matches_the_reference() {
    laya_matches("laya-typed-decisions", "laya-typed-decisions.ref", [2e-4, 1e-2]);
}

/// A Granite Embedding checkpoint (`$TACHYON_MODELS/name`) on the GPU against the 54 float32
/// references of fixture `reference`, its inputs as they are: cosine ≥ 0.9999 each.
#[cfg(all(feature = "gpu", feature = "models"))]
fn granite_matches(name: &str, reference: &str) {
    use tachyon::encoder::{Encoder, Settings};
    let ctx = tachyon::cuda::Context::new(0).unwrap();
    let mut enc = Encoder::open(&ctx, &converted(name), &Settings::default()).unwrap();
    let lines = std::fs::read_to_string(Path::new(TINY).parent().unwrap().join(reference)).unwrap();
    let cases: Vec<Json> = lines.lines().map(|l| Json::parse(l.as_bytes()).unwrap()).collect();
    let ids: Vec<Vec<u32>> = cases.iter().map(|c| numbers(c.get("ids")).iter().map(|&v| v as u32).collect()).collect();
    let seqs: Vec<&[u32]> = ids.iter().map(Vec::as_slice).collect();
    let got = enc.embeddings(&seqs).unwrap();
    let cosines: Vec<f64> = got.iter().zip(&cases).map(|(e, c)| e.iter().zip(numbers(c.get("embedding"))).map(|(a, b)| f64::from(*a) * b).sum()).collect();
    let (text, low) = cosines.iter().copied().enumerate().fold((0, 1.0), |a, (i, c)| if c < a.1 { (i, c) } else { a });
    println!("{name} on the GPU: lowest cosine {low:.6}, text {text}");
    assert!(low >= 0.9999, "{name}: lowest cosine {low:.6}, text {text}");
}

#[cfg(all(feature = "gpu", feature = "models"))]
#[test]
fn granite_on_the_gpu_matches_the_reference() {
    granite_matches("granite-embedding-311m-multilingual-r2", "granite-311m.ref");
}

/// Granite 97M: heads of 32, a gated `SiLU`, `<|startoftext|>` and `<|return|>` around each text; its layers in
/// `f16`, as it is too narrow for Q8.
#[cfg(all(feature = "gpu", feature = "models"))]
#[test]
fn granite_97m_on_the_gpu_matches_the_reference() {
    granite_matches("granite-embedding-97m-multilingual-r2", "granite-97m.ref");
}

/// Text in, vectors out: `Engine::embed` with both Granites on their 54 texts against the float32 references (cosine
/// ≥ 0.9999), each framed as its tokenizer frames it, and cut to 256 values, still of unit length; and the encoders'
/// VRAM and keep-alive as the engine reports and applies them.
#[cfg(all(feature = "gpu", feature = "models"))]
#[test]
fn the_engine_embeds_text() {
    use tachyon::engine::{Engine, Limits, Load};
    let mut engine = Engine::new(0, Limits::default()).unwrap();
    for (name, reference) in [("granite-embedding-311m-multilingual-r2", "granite-311m.ref"), ("granite-embedding-97m-multilingual-r2", "granite-97m.ref")] {
        engine.load(name, &converted(name), &Load::default()).unwrap();
        let lines = std::fs::read_to_string(Path::new(TINY).parent().unwrap().join(reference)).unwrap();
        let cases: Vec<Json> = lines.lines().map(|l| Json::parse(l.as_bytes()).unwrap()).collect();
        let texts: Vec<&str> = cases.iter().map(|c| c.get("text").and_then(Json::str).unwrap()).collect();
        for (i, (e, c)) in engine.embed(name, &texts, None).unwrap().iter().zip(&cases).enumerate() {
            let cosine: f64 = e.iter().zip(numbers(c.get("embedding"))).map(|(a, b)| f64::from(*a) * b).sum();
            assert!(cosine >= 0.9999, "{name}, text {i}: cosine {cosine}");
        }
        for e in engine.embed(name, &texts, Some(256)).unwrap() {
            let len: f32 = e.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!(e.len() == 256 && (len - 1.0).abs() < 1e-5, "{name}: {} values of length {len}", e.len());
        }
    }
    // Encoders are listed with their VRAM and follow their keep-alive: one kept for no time is gone at the next call.
    assert!(engine.models().iter().all(|m| m.vram > 0 && m.left.is_some()), "{:?}", engine.models());
    let name = "granite-embedding-97m-multilingual-r2";
    engine.load(name, &converted(name), &Load { keep_alive: std::time::Duration::ZERO, ..Load::default() }).unwrap();
    engine.expire().unwrap();
    assert_eq!(engine.tier(name), None, "an encoder whose keep-alive ran out is unloaded");
}
