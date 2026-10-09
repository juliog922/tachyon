//! A tiny Gemma 4 checkpoint, generated: the real model's structure (sliding and global layers, shared keys and
//! values, per-layer inputs) at a size the CPU reference and the PTX interpreter run in seconds. Every matrix is
//! on the Q4 grid (`n·2^-e` per group of 64, one weight at ±7), so converting it loses nothing.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

pub const HIDDEN: usize = 256;
pub const FFN: usize = 512;
pub const LAYERS: usize = 6;
pub const VOCAB: usize = 4000;
/// Layers 2 and 5 are global; 4 and 5 reuse the caches of 3 and 2.
pub const GLOBAL: [bool; LAYERS] = [false, false, true, false, false, true];

/// A deterministic stream of numbers.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform in `[lo, hi)`.
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// `v` rounded to bfloat16, as the checkpoint stores it.
pub fn bf16(v: f32) -> f32 {
    let b = v.to_bits();
    f32::from_bits((b + 0x7fff + (b >> 16 & 1)) & 0xffff_0000)
}

/// The tensors of the checkpoint, in order: name, shape, values.
pub fn tensors(seed: u64) -> Vec<(String, Vec<usize>, Vec<f32>)> {
    let mut rng = Rng(seed);
    let mut out = Vec::new();
    let matrix = |out: &mut Vec<_>, rng: &mut Rng, name: String, rows: usize, cols: usize, e: i32| {
        let mut w = Vec::with_capacity(rows * cols);
        for _ in 0..rows * cols / 64 {
            let d = 2f32.powi(-e - (rng.next() % 2) as i32);
            w.push(if rng.next() % 2 == 0 { 7.0 * d } else { -7.0 * d });
            w.extend((1..64).map(|_| ((rng.next() % 15) as f32 - 7.0) * d));
        }
        out.push((name, vec![rows, cols], w));
    };
    let vector = |out: &mut Vec<_>, rng: &mut Rng, name: String, len: usize, lo: f32, hi: f32| {
        out.push((name, vec![len], (0..len).map(|_| bf16(rng.range(lo, hi))).collect()));
    };
    let p = "model.language_model.";
    matrix(&mut out, &mut rng, format!("{p}embed_tokens.weight"), VOCAB, HIDDEN, 5);
    matrix(&mut out, &mut rng, format!("{p}embed_tokens_per_layer.weight"), VOCAB, LAYERS * 256, 5);
    matrix(&mut out, &mut rng, format!("{p}per_layer_model_projection.weight"), LAYERS * 256, HIDDEN, 6);
    vector(&mut out, &mut rng, format!("{p}per_layer_projection_norm.weight"), 256, 0.5, 1.5);
    vector(&mut out, &mut rng, format!("{p}norm.weight"), HIDDEN, 0.5, 1.5);
    for (i, &global) in GLOBAL.iter().enumerate() {
        let (l, d) = (format!("{p}layers.{i}."), if global { 512 } else { 256 });
        for (n, rows, cols) in [("self_attn.q_proj", 8 * d, HIDDEN), ("self_attn.k_proj", 2 * d, HIDDEN), ("self_attn.v_proj", 2 * d, HIDDEN)] {
            matrix(&mut out, &mut rng, format!("{l}{n}.weight"), rows, cols, 5);
        }
        matrix(&mut out, &mut rng, format!("{l}self_attn.o_proj.weight"), HIDDEN, 8 * d, 6);
        for (n, rows, cols) in [("mlp.gate_proj", FFN, HIDDEN), ("mlp.up_proj", FFN, HIDDEN), ("mlp.down_proj", HIDDEN, FFN)] {
            matrix(&mut out, &mut rng, format!("{l}{n}.weight"), rows, cols, 5);
        }
        matrix(&mut out, &mut rng, format!("{l}per_layer_input_gate.weight"), 256, HIDDEN, 5);
        matrix(&mut out, &mut rng, format!("{l}per_layer_projection.weight"), HIDDEN, 256, 5);
        vector(&mut out, &mut rng, format!("{l}self_attn.q_norm.weight"), d, 0.15, 0.35);
        vector(&mut out, &mut rng, format!("{l}self_attn.k_norm.weight"), d, 0.15, 0.35);
        for n in ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm", "post_per_layer_input_norm"] {
            vector(&mut out, &mut rng, format!("{l}{n}.weight"), HIDDEN, 0.5, 1.5);
        }
        vector(&mut out, &mut rng, format!("{l}layer_scalar"), 1, 0.5, 1.0);
    }
    out
}

/// Writes the checkpoint (`config.json`, the toy `tokenizer.json`, `model.safetensors` in bfloat16) to a fresh
/// directory under the system's temporary one, and returns it.
pub fn checkpoint(name: &str, seed: u64) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("tachyon-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    std::fs::copy(fixtures.join("toy.json"), dir.join("tokenizer.json")).unwrap();
    std::fs::write(dir.join("config.json"), config()).unwrap();
    let (mut head, mut data) = (String::from("{"), Vec::new());
    for (name, shape, values) in tensors(seed) {
        let start = data.len();
        data.extend(values.iter().flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes()));
        let _ = write!(head, "\"{name}\":{{\"dtype\":\"BF16\",\"shape\":{shape:?},\"data_offsets\":[{start},{}]}},", data.len());
    }
    head.pop();
    head.push('}');
    while head.len() % 8 != 0 {
        head.push(' ');
    }
    let file = [&(head.len() as u64).to_le_bytes()[..], head.as_bytes(), &data].concat();
    std::fs::write(dir.join("model.safetensors"), file).unwrap();
    dir
}

/// The checkpoint's `config.json`, shaped as Gemma 4's: the decoder's settings under `text_config`.
pub fn config() -> String {
    let types: Vec<String> = GLOBAL.iter().map(|&g| format!("\"{}\"", if g { "full_attention" } else { "sliding_attention" })).collect();
    format!(
        r#"{{"architectures":["Gemma4ForConditionalGeneration"],"text_config":{{"hidden_size":{HIDDEN},"intermediate_size":{FFN},
        "num_attention_heads":8,"num_key_value_heads":2,"head_dim":256,"global_head_dim":512,"hidden_size_per_layer_input":256,
        "vocab_size":{VOCAB},"vocab_size_per_layer_input":{VOCAB},"num_hidden_layers":{LAYERS},"layer_types":[{}],"sliding_window":8,
        "num_kv_shared_layers":2,"rms_norm_eps":1e-6,"final_logit_softcapping":30.0,"hidden_activation":"gelu_pytorch_tanh",
        "tie_word_embeddings":true,"attention_k_eq_v":false,"use_double_wide_mlp":false,"enable_moe_block":false,
        "model_type":"gemma4_text","rope_parameters":{{"sliding_attention":{{"rope_type":"default","rope_theta":10000.0}},
        "full_attention":{{"rope_type":"proportional","partial_rotary_factor":0.25,"rope_theta":1000000.0}}}}}}}}"#,
        types.join(",")
    )
}

/// Gemma 4's forward pass on the CPU in `f32`, as Hugging Face computes it, over the checkpoint of `seed`: the
/// logits (before soft-capping) after each of `tokens`.
pub fn reference(seed: u64, tokens: &[u32]) -> Vec<Vec<f32>> {
    let t: std::collections::HashMap<String, Vec<f32>> = tensors(seed).into_iter().map(|(n, _, v)| (n.replace("model.language_model.", ""), v)).collect();
    let mut cache: Vec<Vec<(Vec<f32>, Vec<f32>)>> = vec![Vec::new(); LAYERS];
    tokens
        .iter()
        .enumerate()
        .map(|(pos, &tok)| {
            let row = |name: &str, i: usize, n: usize| t[name][i * n..(i + 1) * n].to_vec();
            let mut x: Vec<f32> = row("embed_tokens.weight", tok as usize, HIDDEN).iter().map(|v| v * 16.0).collect();
            let ple = row("embed_tokens_per_layer.weight", tok as usize, LAYERS * 256);
            let proj: Vec<f32> = mv(&t["per_layer_model_projection.weight"], &x).iter().map(|v| v / (HIDDEN as f32).sqrt()).collect();
            let pli: Vec<f32> = proj
                .chunks(256)
                .zip(ple.chunks(256))
                .flat_map(|(p, e)| {
                    rms(p, Some(&t["per_layer_projection_norm.weight"]))
                        .iter()
                        .zip(e)
                        .map(|(a, b)| (a + b * 16.0) * std::f32::consts::FRAC_1_SQRT_2)
                        .collect::<Vec<_>>()
                })
                .collect();
            for i in 0..LAYERS {
                x = layer(&t, &mut cache, i, pos, &x, &pli[i * 256..(i + 1) * 256]);
            }
            mv(&t["embed_tokens.weight"], &rms(&x, Some(&t["norm.weight"])))
        })
        .collect()
}

fn layer(t: &std::collections::HashMap<String, Vec<f32>>, cache: &mut [Vec<(Vec<f32>, Vec<f32>)>], i: usize, pos: usize, x: &[f32], pli: &[f32]) -> Vec<f32> {
    let w = |n: &str| &t[&format!("layers.{i}.{n}")][..];
    let d = if GLOBAL[i] { 512 } else { 256 };
    let freq: Vec<f32> =
        (0..d / 2).map(|j| if GLOBAL[i] && j >= 64 { 0.0 } else { (if GLOBAL[i] { 1e6f32 } else { 1e4 }).powf(-((2 * j) as f32) / d as f32) }).collect();
    let h = rms(x, Some(w("input_layernorm.weight")));
    let mut q = mv(w("self_attn.q_proj.weight"), &h);
    for head in q.chunks_mut(d) {
        let n = rms(head, Some(w("self_attn.q_norm.weight")));
        head.copy_from_slice(&rotate(&n, pos, &freq));
    }
    let source = [0, 1, 2, 3, 3, 2][i];
    if source == i {
        let (k, v) = (mv(w("self_attn.k_proj.weight"), &h), mv(w("self_attn.v_proj.weight"), &h));
        let k: Vec<f32> = k.chunks(d).flat_map(|c| rotate(&rms(c, Some(w("self_attn.k_norm.weight"))), pos, &freq)).collect();
        cache[i].push((k, v.chunks(d).flat_map(|c| rms(c, None)).collect()));
    }
    let first = if GLOBAL[i] { 0 } else { (pos + 1).saturating_sub(8) };
    let out = attend(&q, &cache[source][first..], d);
    let add = |x: &[f32], y: &[f32], n: &str| -> Vec<f32> { x.iter().zip(rms(y, Some(w(n)))).map(|(a, b)| a + b).collect() };
    let x = add(x, &mv(w("self_attn.o_proj.weight"), &out), "post_attention_layernorm.weight");
    let h = rms(&x, Some(w("pre_feedforward_layernorm.weight")));
    let (g, u) = (mv(w("mlp.gate_proj.weight"), &h), mv(w("mlp.up_proj.weight"), &h));
    let a: Vec<f32> = g.iter().zip(&u).map(|(g, u)| gelu(*g) * u).collect();
    let x = add(&x, &mv(w("mlp.down_proj.weight"), &a), "post_feedforward_layernorm.weight");
    let a: Vec<f32> = mv(w("per_layer_input_gate.weight"), &x).iter().zip(pli).map(|(g, p)| gelu(*g) * p).collect();
    let x = add(&x, &mv(w("per_layer_projection.weight"), &a), "post_per_layer_input_norm.weight");
    x.iter().map(|v| v * w("layer_scalar")[0]).collect()
}

/// Each query head of `q` attends over `cache` through its key-value head, with scale 1.
fn attend(q: &[f32], cache: &[(Vec<f32>, Vec<f32>)], d: usize) -> Vec<f32> {
    let mut out = vec![0f32; q.len()];
    for (hq, o) in out.chunks_mut(d).enumerate() {
        let kv = hq / 4;
        let scores: Vec<f32> = cache.iter().map(|(k, _)| dot(&q[hq * d..(hq + 1) * d], &k[kv * d..(kv + 1) * d])).collect();
        let m = scores.iter().fold(f32::MIN, |a, &b| a.max(b));
        let e: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
        let sum: f32 = e.iter().sum();
        for ((_, v), e) in cache.iter().zip(&e) {
            for (o, v) in o.iter_mut().zip(&v[kv * d..(kv + 1) * d]) {
                *o += e / sum * v;
            }
        }
    }
    out
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `W·x` for a row-major `W`.
fn mv(w: &[f32], x: &[f32]) -> Vec<f32> {
    w.chunks(x.len()).map(|r| dot(r, x)).collect()
}

/// `x / √(mean(x²) + 1e-6)`, times `w` when given.
fn rms(x: &[f32], w: Option<&[f32]>) -> Vec<f32> {
    let r = 1.0 / (dot(x, x) / x.len() as f32 + 1e-6).sqrt();
    x.iter().enumerate().map(|(i, v)| v * r * w.map_or(1.0, |w| w[i])).collect()
}

/// RoPE as Hugging Face's `rotate_half`: pair `(j, j + d/2)` turns by `pos · freq[j]`.
fn rotate(x: &[f32], pos: usize, freq: &[f32]) -> Vec<f32> {
    let half = x.len() / 2;
    let mut out = x.to_vec();
    for (j, f) in freq.iter().enumerate() {
        let (s, c) = (pos as f32 * f).sin_cos();
        out[j] = x[j] * c - x[j + half] * s;
        out[j + half] = x[j + half] * c + x[j] * s;
    }
    out
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044_715 * x * x * x)).tanh())
}