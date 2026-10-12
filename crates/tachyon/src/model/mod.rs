//! Gemma 4's decoder: its configuration, and the conversion of a Hugging Face checkpoint to the files the engine
//! loads.
//!
//! [`convert`] reads a checkpoint directory (`config.json`, `tokenizer.json`, `*.safetensors`) and writes a model
//! directory: `weights.wpk`, everything that lives in VRAM; `host.wpk`, the per-layer embedding table, which stays
//! in pinned host memory and is read one row per token over PCIe; `tokenizer.tok`; `config.json`, copied; and
//! `vision.wpk`, the vision tower, when the checkpoint has one ([`crate::vision::convert`]). Each
//! matrix becomes Q4 ([`crate::quant`]) as `{name}.w` and `{name}.s`; projections that read the same input are
//! fused into one matrix, rows stacked: query, key and value as `{i}.qkv`, gate and up as `{i}.gate_up`. Vectors
//! are `f32`.

mod decode;
mod prefill;

pub use decode::{Decoder, Parked, Settings, footprint};

use crate::json::Json;
use crate::quant::{Safetensors, q4};
use crate::token::Tokenizer;
use crate::wpk::{Dtype, Writer};
use crate::{Error, Result};
use std::path::Path;

/// What the engine needs of a Gemma 4 text decoder's `config.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// Width of the residual stream.
    pub hidden: usize,
    /// Width of the MLP; see [`Config::mlp`].
    pub ffn: usize,
    /// Query heads.
    pub heads: usize,
    /// Key-value heads.
    pub kv_heads: usize,
    /// Head size of sliding layers, and of global ones.
    pub head_dim: [usize; 2],
    /// Width of each layer's per-layer input.
    pub per_layer: usize,
    /// Tokens.
    pub vocab: usize,
    /// Whether each layer attends globally rather than over the sliding window.
    pub global: Vec<bool>,
    /// Positions a sliding layer sees, its own included.
    pub window: usize,
    /// The last this many layers reuse an earlier layer's keys and values.
    pub shared: usize,
    /// The shared layers' MLPs are twice as wide (E2B's `use_double_wide_mlp`).
    pub wide: bool,
    /// RoPE base of sliding layers, and of global ones.
    pub theta: [f32; 2],
    /// Share of a global head's dimensions RoPE turns.
    pub fraction: f32,
    /// RMS normalization epsilon.
    pub eps: f32,
    /// Final logits become `softcap · tanh(x / softcap)`.
    pub softcap: f32,
}

impl Config {
    /// Reads `config.json`; a multimodal one holds the decoder's under `text_config`.
    pub fn from_json(text: &[u8]) -> Result<Config> {
        const INTS: [&str; 10] = [
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "global_head_dim",
            "hidden_size_per_layer_input",
            "vocab_size",
            "sliding_window",
            "rms_norm_eps",
        ];
        let doc = Json::parse(text)?;
        let c = doc.get("text_config").unwrap_or(&doc);
        let n: Vec<f64> =
            INTS.iter().map(|k| c.get(k).and_then(Json::num).ok_or_else(|| Error::Format(format!("config.json: no {k}")))).collect::<Result<_>>()?;
        unsupported(c, &n)?;
        let rope = |kind: &str, k: &str, or: f64| c.get("rope_parameters").and_then(|r| r.get(kind)?.get(k)?.num()).unwrap_or(or) as f32;
        let types = c.get("layer_types").and_then(Json::arr).unwrap_or_default();
        Ok(Config {
            hidden: n[0] as usize,
            ffn: n[1] as usize,
            heads: n[2] as usize,
            kv_heads: n[3] as usize,
            head_dim: [n[4] as usize, n[5] as usize],
            per_layer: n[6] as usize,
            vocab: n[7] as usize,
            global: types.iter().map(|t| t.str() == Some("full_attention")).collect(),
            window: n[8] as usize,
            shared: c.get("num_kv_shared_layers").and_then(Json::num).unwrap_or(0.0) as usize,
            wide: c.get("use_double_wide_mlp").and_then(Json::bool).unwrap_or(false),
            theta: [rope("sliding_attention", "rope_theta", 1e4), rope("full_attention", "rope_theta", 1e6)],
            fraction: rope("full_attention", "partial_rotary_factor", 1.0),
            eps: n[9] as f32,
            softcap: c.get("final_logit_softcapping").and_then(Json::num).unwrap_or(0.0) as f32,
        })
    }

    /// Layers.
    pub fn layers(&self) -> usize {
        self.global.len()
    }

    /// Width of layer `i`'s MLP: twice [`Config::ffn`] in a shared layer of a [`Config::wide`] model.
    pub fn mlp(&self, i: usize) -> usize {
        self.ffn << usize::from(self.wide && self.shared < self.layers() && i >= self.layers() - self.shared)
    }

    /// Attention blocks per key-value head: each takes four query heads.
    pub fn group(&self) -> u32 {
        (self.heads / 4 / self.kv_heads) as u32
    }

    /// The layer whose keys and values layer `i` attends over: itself, or for a shared layer the last unshared
    /// layer of its kind.
    pub fn source(&self, i: usize) -> usize {
        let first = self.layers() - self.shared;
        if i < first { i } else { (0..first).rev().find(|&j| self.global[j] == self.global[i]).unwrap_or(i) }
    }
}

/// Refuses, as [`Error::Unsupported`], what Gemma 4's larger models add (step 12 brings it): `c` is the decoder's
/// configuration, `n` the numbers [`Config::from_json`] read from it.
fn unsupported(c: &Json, n: &[f64]) -> Result<()> {
    let on = |k: &str| c.get(k).and_then(Json::bool) == Some(true);
    let (global_kv, [heads, kv]) = (c.get("num_global_key_value_heads").and_then(Json::num), [n[2] as usize, n[3] as usize]);
    let features = [
        (on("enable_moe_block"), "a mixture-of-experts block (26B A4B)"),
        (on("attention_k_eq_v"), "values equal to keys in global layers (26B A4B, 31B)"),
        (n[6] == 0.0, "layers without per-layer inputs (26B A4B, 31B)"),
        (global_kv.is_some_and(|g| g as usize != kv), "a different number of key-value heads in global layers (26B A4B, 31B)"),
        (kv == 0 || heads % (4 * kv) != 0, "fewer than four query heads per key-value head (26B A4B, 31B)"),
    ];
    features.iter().find(|f| f.0).map_or(Ok(()), |f| Err(Error::Unsupported(f.1.into())))
}

/// Converts the checkpoint in `src` to a model directory `dst` (see the module documentation).
pub fn convert(src: &Path, dst: &Path) -> Result<()> {
    let text = std::fs::read(src.join("config.json"))?;
    let cfg = Config::from_json(&text)?;
    let st = Safetensors::open(src)?;
    let p = st.prefix("embed_tokens.weight");
    std::fs::create_dir_all(dst)?;
    weights(dst, &st, &cfg, p)?;
    host(dst, &st, p)?;
    crate::vision::convert(&st, dst)?;
    tokenizer(src, dst)?;
    Ok(std::fs::write(dst.join("config.json"), text)?)
}

/// Writes `weights.wpk` into `dst`; the checkpoint's names start with `p`.
fn weights(dst: &Path, st: &Safetensors, cfg: &Config, p: &str) -> Result<()> {
    let mut w = Writer::create(dst.join("weights.wpk"))?;
    for (name, source) in [("embed", "embed_tokens"), ("per_layer_proj", "per_layer_model_projection")] {
        matrix(&mut w, st, name, &[format!("{p}{source}.weight")])?;
    }
    for (name, source) in [("per_layer_norm", "per_layer_projection_norm"), ("norm", "norm")] {
        vector(&mut w, st, name, &format!("{p}{source}.weight"))?;
    }
    for i in 0..cfg.layers() {
        layer(&mut w, st, &format!("{p}layers.{i}."), i, cfg.source(i) == i)?;
    }
    w.finish()
}

/// Writes `host.wpk` into `dst`.
fn host(dst: &Path, st: &Safetensors, p: &str) -> Result<()> {
    let mut host = Writer::create(dst.join("host.wpk"))?;
    matrix(&mut host, st, "embed_per_layer", &[format!("{p}embed_tokens_per_layer.weight")])?;
    host.finish()
}

/// Stores the tokenizer of `src` in `dst`.
fn tokenizer(src: &Path, dst: &Path) -> Result<()> {
    let tok = Tokenizer::from_json(&std::fs::read(src.join("tokenizer.json"))?)?;
    Ok(std::fs::write(dst.join("tokenizer.tok"), tok.to_bytes())?)
}

/// Writes layer `i`, whose checkpoint names start with `l`; `kv` when it projects keys and values.
fn layer(w: &mut Writer, st: &Safetensors, l: &str, i: usize, kv: bool) -> Result<()> {
    let qkv: &[&str] = if kv { &["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"] } else { &["self_attn.q_proj"] };
    let fused = [("qkv", qkv), ("o", &["self_attn.o_proj"]), ("gate_up", &["mlp.gate_proj", "mlp.up_proj"]), ("down", &["mlp.down_proj"])];
    for (name, parts) in fused.into_iter().chain([("per_layer_gate", &["per_layer_input_gate"][..]), ("per_layer_proj", &["per_layer_projection"])]) {
        matrix(w, st, &format!("{i}.{name}"), &parts.iter().map(|n| format!("{l}{n}.weight")).collect::<Vec<_>>())?;
    }
    let norms = ["self_attn.q_norm", "self_attn.k_norm", "input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm"];
    for n in norms.into_iter().chain(["post_feedforward_layernorm", "post_per_layer_input_norm"]).filter(|n| st.has(&format!("{l}{n}.weight"))) {
        vector(w, st, &format!("{i}.{}", n.trim_start_matches("self_attn.")), &format!("{l}{n}.weight"))?;
    }
    vector(w, st, &format!("{i}.layer_scalar"), &format!("{l}layer_scalar"))
}

/// Writes the source matrices, rows stacked, as one Q4 matrix: `{name}.w` (`u8 [rows, cols/2]`) and `{name}.s`
/// (`f16 [rows, cols/64]`). Rows are read and quantized a slice at a time.
fn matrix(w: &mut Writer, st: &Safetensors, name: &str, parts: &[String]) -> Result<()> {
    let (mut packed, mut scales, mut cols) = (Vec::new(), Vec::new(), 0);
    for part in parts {
        let (rows, c) = st.dims(part)?;
        cols = c;
        let step = (1 << 22) / c.max(1) + 1;
        for start in (0..rows).step_by(step) {
            let q = q4(&st.rows(part, start..rows.min(start + step))?, c);
            packed.extend(q.packed);
            scales.extend(q.scales.iter().flat_map(|s| s.to_le_bytes()));
        }
    }
    let rows = packed.len() as u64 * 2 / cols.max(1) as u64;
    w.add(&format!("{name}.w"), Dtype::U8, &[rows, cols as u64 / 2], &packed)?;
    w.add(&format!("{name}.s"), Dtype::F16, &[rows, cols as u64 / 64], &scales)
}

/// Writes a vector (or scalar) as `f32`.
fn vector(w: &mut Writer, st: &Safetensors, name: &str, source: &str) -> Result<()> {
    let v = st.rows(source, 0..st.dims(source)?.0)?;
    w.add(name, Dtype::F32, &[v.len() as u64], &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
}
