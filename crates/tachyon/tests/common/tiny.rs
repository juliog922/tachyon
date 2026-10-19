//! A tiny Gemma 4 checkpoint, generated: the real model's structure (sliding and global layers, shared keys and
//! values, per-layer inputs) in E4B's shape or E2B's, or the larger models' ([`DENSE`]: no per-layer inputs, no
//! shared caches, two query heads per key-value head in sliding layers, one key-value head whose values are its keys
//! in global layers), or with a mixture of experts beside each MLP as the 26B A4B has ([`MOE`]: 64 experts of 192, 4
//! a token, beside an MLP of 320, both widths padded on conversion), at a size the CPU reference runs in seconds. Every matrix is on the Q4 grid (`n·2^-e` per
//! group of 64, one weight at ±7), so converting it loses nothing.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

pub const HIDDEN: usize = 256;
pub const FFN: usize = 512;
pub const LAYERS: usize = 6;
pub const VOCAB: usize = 4000;
/// Layers 2 and 5 are global; 4 and 5 reuse the caches of 3 and 2.
pub const GLOBAL: [bool; LAYERS] = [false, false, true, false, false, true];

/// Where Gemma 4's models differ: key-value heads (of 8 query heads) in sliding layers, whether the shared layers'
/// MLPs are twice as wide, and whether the model is laid out as the larger ones (see the module's documentation).
#[derive(Clone, Copy)]
pub struct Shape {
    pub kv: usize,
    pub wide: bool,
    pub dense: bool,
    pub moe: bool,
}

pub const E4B: Shape = Shape { kv: 2, wide: false, dense: false, moe: false };
pub const E2B: Shape = Shape { kv: 1, wide: true, dense: false, moe: false };
/// The 31B's layout: four key-value heads in sliding layers (two query heads each), one in global layers.
pub const DENSE: Shape = Shape { kv: 4, wide: false, dense: true, moe: false };
/// [`DENSE`] with experts, as the 26B A4B.
pub const MOE: Shape = Shape { kv: 4, wide: false, dense: true, moe: true };
/// Experts, experts a token takes, an expert's width; the MLP's width beside them.
pub const EXPERTS: [usize; 3] = [64, 4, 192];
pub const MOE_FFN: usize = 320;

impl Shape {
    /// Key-value heads of layer `i`.
    pub fn kv(self, i: usize) -> usize {
        if self.dense && GLOBAL[i] { 1 } else { self.kv }
    }

    /// The layer whose keys and values layer `i` attends over.
    pub fn source(self, i: usize) -> usize {
        if self.dense { i } else { [0, 1, 2, 3, 3, 2][i] }
    }

    /// Width of layer `i`'s MLP.
    pub fn ffn(self, i: usize) -> usize {
        if self.moe {
            MOE_FFN
        } else if self.wide && i >= 4 {
            2 * FFN
        } else {
            FFN
        }
    }
}

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

/// A checkpoint's tensors as they are made: name, shape, values.
type Tensors = Vec<(String, Vec<usize>, Vec<f32>)>;

/// A matrix on the Q4 grid: per group of 64, `n·2^-e` (`e` this or one more), one weight at ±7.
fn matrix(out: &mut Tensors, rng: &mut Rng, name: String, [rows, cols]: [usize; 2], e: i32) {
    let mut w = Vec::with_capacity(rows * cols);
    for _ in 0..rows * cols / 64 {
        let d = 2f32.powi(-e - (rng.next() % 2) as i32);
        w.push(if rng.next() % 2 == 0 { 7.0 * d } else { -7.0 * d });
        w.extend((1..64).map(|_| ((rng.next() % 15) as f32 - 7.0) * d));
    }
    out.push((name, vec![rows, cols], w));
}

/// A vector of bfloat16 values in `[lo, hi)`.
fn vector(out: &mut Tensors, rng: &mut Rng, name: String, len: usize, [lo, hi]: [f32; 2]) {
    out.push((name, vec![len], (0..len).map(|_| bf16(rng.range(lo, hi))).collect()));
}

/// The tensors of the checkpoint, in order: name, shape, values.
pub fn tensors(seed: u64, shape: Shape) -> Tensors {
    let mut rng = Rng(seed);
    let mut out = Vec::new();
    let p = "model.language_model.";
    matrix(&mut out, &mut rng, format!("{p}embed_tokens.weight"), [VOCAB, HIDDEN], 5);
    if !shape.dense {
        matrix(&mut out, &mut rng, format!("{p}embed_tokens_per_layer.weight"), [VOCAB, LAYERS * 256], 5);
        matrix(&mut out, &mut rng, format!("{p}per_layer_model_projection.weight"), [LAYERS * 256, HIDDEN], 6);
        vector(&mut out, &mut rng, format!("{p}per_layer_projection_norm.weight"), 256, [0.5, 1.5]);
    }
    vector(&mut out, &mut rng, format!("{p}norm.weight"), HIDDEN, [0.5, 1.5]);
    for i in 0..LAYERS {
        layer_tensors(&mut out, &mut rng, &format!("{p}layers.{i}."), (i, shape));
    }
    out
}

/// Layer `i`'s tensors, named from `l`.
fn layer_tensors(out: &mut Tensors, rng: &mut Rng, l: &str, (i, shape): (usize, Shape)) {
    let d = if GLOBAL[i] { 512 } else { 256 };
    let (kv, ffn) = (shape.kv(i) * d, shape.ffn(i));
    // The larger models' global layers have no value projection: their values are their keys.
    let values = !(shape.dense && GLOBAL[i]);
    let projections = [("self_attn.q_proj", 8 * d, HIDDEN), ("self_attn.k_proj", kv, HIDDEN), ("self_attn.v_proj", kv, HIDDEN)];
    for (n, rows, cols) in projections.into_iter().filter(|p| values || p.0 != "self_attn.v_proj") {
        matrix(out, rng, format!("{l}{n}.weight"), [rows, cols], 5);
    }
    matrix(out, rng, format!("{l}self_attn.o_proj.weight"), [HIDDEN, 8 * d], 6);
    for (n, rows, cols) in [("mlp.gate_proj", ffn, HIDDEN), ("mlp.up_proj", ffn, HIDDEN), ("mlp.down_proj", HIDDEN, ffn)] {
        matrix(out, rng, format!("{l}{n}.weight"), [rows, cols], 5);
    }
    if !shape.dense {
        matrix(out, rng, format!("{l}per_layer_input_gate.weight"), [256, HIDDEN], 5);
        matrix(out, rng, format!("{l}per_layer_projection.weight"), [HIDDEN, 256], 5);
    }
    vector(out, rng, format!("{l}self_attn.q_norm.weight"), d, [0.15, 0.35]);
    vector(out, rng, format!("{l}self_attn.k_norm.weight"), d, [0.15, 0.35]);
    let norms = ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm", "post_per_layer_input_norm"];
    for n in norms.into_iter().filter(|&n| !shape.dense || n != "post_per_layer_input_norm") {
        vector(out, rng, format!("{l}{n}.weight"), HIDDEN, [0.5, 1.5]);
    }
    if shape.moe {
        experts(out, rng, l);
    }
    vector(out, rng, format!("{l}layer_scalar"), 1, [0.5, 1.0]);
}

/// A layer's experts, named as the 26B A4B's checkpoint names them: the router's projection, scale (zero, see
/// below) and per-expert scales, every expert's gate and up stacked, every expert's down, and their three norms.
fn experts(out: &mut Tensors, rng: &mut Rng, l: &str) {
    let [e, _, w] = EXPERTS;
    matrix(out, rng, format!("{l}router.proj.weight"), [e, HIDDEN], 5);
    // A zero scale: every logit is 0, so every token takes experts 0..k with equal probabilities, on the GPU as in
    // the reference. Random routers put the k-th and the next expert within the GPU's Q8 rounding of each other
    // often enough that a token takes another expert within a few positions, and the comparison fails for a
    // rounding, not an error; `tests/moe.rs` checks the routing itself against the CPU.
    out.push((format!("{l}router.scale"), vec![HIDDEN], vec![0.0; HIDDEN]));
    vector(out, rng, format!("{l}router.per_expert_scale"), e, [0.5, 1.5]);
    matrix(out, rng, format!("{l}experts.gate_up_proj"), [e * 2 * w, HIDDEN], 5);
    matrix(out, rng, format!("{l}experts.down_proj"), [e * HIDDEN, w], 5);
    for n in ["pre_feedforward_layernorm_2", "post_feedforward_layernorm_1", "post_feedforward_layernorm_2"] {
        vector(out, rng, format!("{l}{n}.weight"), HIDDEN, [0.5, 1.5]);
    }
}

/// Writes the checkpoint (`config.json`, the toy `tokenizer.json`, `model.safetensors` in bfloat16) to a fresh
/// directory under the system's temporary one, and returns it.
pub fn checkpoint(name: &str, seed: u64, shape: Shape) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("tachyon-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // From this crate's tests, or tachyon-cli's (which include this file).
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("../tachyon/tests/fixtures");
    // The toy tokenizer, with Gemma 4's chat markers in place of three of its special tokens.
    let toy = std::fs::read_to_string(fixtures.join("toy.json")).unwrap();
    let toy = toy.replace("<start_of_turn>", "<|turn>").replace("<end_of_turn>", "<turn|>").replace("<pad>", "<|think|>");
    std::fs::write(dir.join("tokenizer.json"), toy).unwrap();
    std::fs::write(dir.join("config.json"), config(shape)).unwrap();
    let (mut head, mut data) = (String::from("{"), Vec::new());
    for (name, shape, values) in tensors(seed, shape) {
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
pub fn config(shape: Shape) -> String {
    let types: Vec<String> = GLOBAL.iter().map(|&g| format!("\"{}\"", if g { "full_attention" } else { "sliding_attention" })).collect();
    format!(
        r#"{{"architectures":["Gemma4ForConditionalGeneration"],"text_config":{{"hidden_size":{HIDDEN},"intermediate_size":{},
        "num_attention_heads":8,"num_key_value_heads":{},"num_global_key_value_heads":{},"head_dim":256,"global_head_dim":512,
        "hidden_size_per_layer_input":{},"vocab_size":{VOCAB},"num_experts":{},"top_k_experts":{},"moe_intermediate_size":{},"vocab_size_per_layer_input":{VOCAB},"num_hidden_layers":{LAYERS},
        "layer_types":[{}],"sliding_window":8,"num_kv_shared_layers":{},"rms_norm_eps":1e-6,"final_logit_softcapping":30.0,
        "hidden_activation":"gelu_pytorch_tanh","tie_word_embeddings":true,"attention_k_eq_v":{},"use_double_wide_mlp":{},
        "enable_moe_block":{},"model_type":"gemma4_text","rope_parameters":{{"sliding_attention":{{"rope_type":"default","rope_theta":10000.0}},
        "full_attention":{{"rope_type":"proportional","partial_rotary_factor":0.25,"rope_theta":1000000.0}}}}}}}}"#,
        shape.ffn(0),
        shape.kv,
        shape.kv(2),
        if shape.dense { 0 } else { 256 },
        EXPERTS[0],
        EXPERTS[1],
        EXPERTS[2],
        types.join(","),
        if shape.dense { 0 } else { 2 },
        shape.dense,
        shape.wide,
        shape.moe
    )
}

/// Gemma 4's forward pass on the CPU in `f32`, as Hugging Face computes it, over the checkpoint of `seed`: the
/// logits (before soft-capping) after each of `tokens`.
pub fn reference(seed: u64, shape: Shape, tokens: &[u32]) -> Vec<Vec<f32>> {
    let t: std::collections::HashMap<String, Vec<f32>> =
        tensors(seed, shape).into_iter().map(|(n, _, v)| (n.replace("model.language_model.", ""), v)).collect();
    let mut cache: Vec<Vec<(Vec<f32>, Vec<f32>)>> = vec![Vec::new(); LAYERS];
    tokens
        .iter()
        .enumerate()
        .map(|(pos, &tok)| {
            let row = |name: &str, i: usize, n: usize| t[name][i * n..(i + 1) * n].to_vec();
            let mut x: Vec<f32> = row("embed_tokens.weight", tok as usize, HIDDEN).iter().map(|v| v * 16.0).collect();
            if shape.dense {
                for i in 0..LAYERS {
                    x = layer(&t, &mut cache, [i, pos], shape, (&x, &[]));
                }
                return mv(&t["embed_tokens.weight"], &rms(&x, Some(&t["norm.weight"])));
            }
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
                x = layer(&t, &mut cache, [i, pos], shape, (&x, &pli[i * 256..(i + 1) * 256]));
            }
            mv(&t["embed_tokens.weight"], &rms(&x, Some(&t["norm.weight"])))
        })
        .collect()
}

/// Layer `i` at `pos` on `x`, `pli` its per-layer input (empty for [`DENSE`]).
fn layer(
    t: &std::collections::HashMap<String, Vec<f32>>,
    cache: &mut [Vec<(Vec<f32>, Vec<f32>)>],
    [i, pos]: [usize; 2],
    shape: Shape,
    (x, pli): (&[f32], &[f32]),
) -> Vec<f32> {
    let group = 8 / shape.kv(i);
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
    let source = shape.source(i);
    if source == i {
        let k = mv(w("self_attn.k_proj.weight"), &h);
        let v = if t.contains_key(&format!("layers.{i}.self_attn.v_proj.weight")) { mv(w("self_attn.v_proj.weight"), &h) } else { k.clone() };
        let k: Vec<f32> = k.chunks(d).flat_map(|c| rotate(&rms(c, Some(w("self_attn.k_norm.weight"))), pos, &freq)).collect();
        cache[i].push((k, v.chunks(d).flat_map(|c| rms(c, None)).collect()));
    }
    let first = if GLOBAL[i] { 0 } else { (pos + 1).saturating_sub(8) };
    let out = attend(&q, &cache[source][first..], d, group);
    let add = |x: &[f32], y: &[f32], n: &str| -> Vec<f32> { x.iter().zip(rms(y, Some(w(n)))).map(|(a, b)| a + b).collect() };
    let x = add(x, &mv(w("self_attn.o_proj.weight"), &out), "post_attention_layernorm.weight");
    let x = feed_forward(t, i, &x, shape.moe);
    if pli.is_empty() {
        return x.iter().map(|v| v * w("layer_scalar")[0]).collect();
    }
    let a: Vec<f32> = mv(w("per_layer_input_gate.weight"), &x).iter().zip(pli).map(|(g, p)| gelu(*g) * p).collect();
    let x = add(&x, &mv(w("per_layer_projection.weight"), &a), "post_per_layer_input_norm.weight");
    x.iter().map(|v| v * w("layer_scalar")[0]).collect()
}

/// Layer `i`'s MLP and, with `moe`, the experts beside it, on `x`, as Hugging Face's `Gemma4TextDecoderLayer`: the residual
/// plus `post_ff(post_ff_1(mlp) + post_ff_2(experts))`, the experts reading `pre_ff_2(x)`, routed from
/// `rmsnorm(x) · scale / √hidden`.
fn feed_forward(t: &std::collections::HashMap<String, Vec<f32>>, i: usize, x: &[f32], moe: bool) -> Vec<f32> {
    let w = |n: &str| &t[&format!("layers.{i}.{n}")][..];
    let add = |x: &[f32], y: &[f32], n: &str| -> Vec<f32> { x.iter().zip(rms(y, Some(w(n)))).map(|(a, b)| a + b).collect() };
    let mlp = |gate: &[f32], up: &[f32], down: &[f32], h: &[f32]| {
        let a: Vec<f32> = mv(gate, h).iter().zip(mv(up, h)).map(|(g, u)| gelu(*g) * u).collect();
        mv(down, &a)
    };
    let dense = mlp(w("mlp.gate_proj.weight"), w("mlp.up_proj.weight"), w("mlp.down_proj.weight"), &rms(x, Some(w("pre_feedforward_layernorm.weight"))));
    if !moe {
        return add(x, &dense, "post_feedforward_layernorm.weight");
    }
    let [e, k, width] = EXPERTS;
    let routed: Vec<f32> = rms(x, Some(w("router.scale"))).iter().map(|v| v / (HIDDEN as f32).sqrt()).collect();
    let logits = mv(w("router.proj.weight"), &routed);
    let top = logits.iter().fold(f32::MIN, |a, &b| a.max(b));
    let p: Vec<f32> = logits.iter().map(|l| (l - top).exp()).collect();
    let mut order: Vec<usize> = (0..e).collect();
    order.sort_by(|&a, &b| p[b].total_cmp(&p[a]).then(a.cmp(&b)));
    let chosen = &order[..k];
    let total: f32 = chosen.iter().map(|&j| p[j]).sum();
    let h = rms(x, Some(w("pre_feedforward_layernorm_2.weight")));
    let (gate_up, down) = (w("experts.gate_up_proj"), w("experts.down_proj"));
    let mut sum = vec![0f32; HIDDEN];
    for &j in chosen {
        let rows = &gate_up[j * 2 * width * HIDDEN..(j + 1) * 2 * width * HIDDEN];
        let out = mlp(&rows[..width * HIDDEN], &rows[width * HIDDEN..], &down[j * HIDDEN * width..(j + 1) * HIDDEN * width], &h);
        let weight = p[j] / total * w("router.per_expert_scale")[j];
        sum.iter_mut().zip(out).for_each(|(s, o)| *s += weight * o);
    }
    let both: Vec<f32> = rms(&dense, Some(w("post_feedforward_layernorm_1.weight")))
        .iter()
        .zip(rms(&sum, Some(w("post_feedforward_layernorm_2.weight"))))
        .map(|(a, b)| a + b)
        .collect();
    add(x, &both, "post_feedforward_layernorm.weight")
}

/// Each query head of `q` attends over `cache` through its key-value head, shared by `group` heads, with scale 1.
fn attend(q: &[f32], cache: &[(Vec<f32>, Vec<f32>)], d: usize, group: usize) -> Vec<f32> {
    let mut out = vec![0f32; q.len()];
    for (hq, o) in out.chunks_mut(d).enumerate() {
        let kv = hq / group;
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
