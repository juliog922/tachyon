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
