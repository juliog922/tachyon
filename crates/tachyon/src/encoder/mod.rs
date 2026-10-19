//! ModernBERT encoders: Laya multilingual's decisions and Granite Embedding's vectors.
//!
//! An encoder embeds its tokens and layer-normalizes them, then runs pre-norm layers (the first without a norm
//! before attention): bidirectional attention with RoPE on the two halves of each head, global in every third layer
//! and within [`Config::window`] positions elsewhere, each with its own RoPE base; and a gated MLP,
//! `Wo · (gelu(a) ⊙ b)` with `[a, b] = Wi · x` (`SiLU` in place of GELU for Granite 97M). A final layer norm closes
//! it. No layer has a bias.

mod convert;
pub mod laya;
pub(crate) mod race;
pub mod route;
mod run;
mod text;

pub use convert::{convert, convert_with};
pub use run::{Encoder, footprint};

use crate::json::Json;
use crate::{Error, Result};

/// How an [`Encoder`] runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Settings {
    /// The most tokens encoded at once, up to 65535: a batch's sequences in all, and the longest sequence.
    pub tokens: usize,
}

impl Default for Settings {
    fn default() -> Settings {
        Settings { tokens: 4096 }
    }
}

/// A ModernBERT encoder's shape, from its `config.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// Width of the residual stream.
    pub hidden: usize,
    /// Attention heads, of `hidden / heads` dimensions each.
    pub heads: usize,
    /// Width of the MLP's gated activation.
    pub inter: usize,
    /// Tokens in the vocabulary.
    pub vocab: usize,
    /// Whether each layer attends globally; the others attend within [`Config::window`].
    pub global: Vec<bool>,
    /// How far a sliding layer looks either way: positions `i` and `j` see each other when `|i − j| ≤ window`.
    pub window: usize,
    /// RoPE bases: global layers', sliding layers'.
    pub theta: [f32; 2],
    /// Layer norms' epsilon.
    pub eps: f32,
    /// Whether the MLP's gate is a `SiLU` rather than GELU.
    pub silu: bool,
}

impl Config {
    /// Reads a Hugging Face `config.json` of a ModernBERT, in the layout of transformers 5 (`layer_types`,
    /// `rope_parameters`) or of 4 (`global_attn_every_n_layers`, `global_rope_theta`, `local_rope_theta`).
    pub fn from_json(json: &[u8]) -> Result<Config> {
        let c = Json::parse(json)?;
        supported(&c)?;
        let sizes = ["hidden_size", "num_attention_heads", "intermediate_size", "vocab_size", "num_hidden_layers"];
        let [hidden, heads, inter, vocab, layers] = sizes.map(|k| c.get(k).and_then(Json::num).map_or(0, |v| v as usize));
        let cfg = Config {
            hidden,
            heads,
            inter,
            vocab,
            global: global_layers(&c, layers),
            window: c.get("local_attention").and_then(Json::num).unwrap_or(128.0) as usize / 2,
            theta: [theta(&c, "full_attention", "global_rope_theta", 160_000.0), theta(&c, "sliding_attention", "local_rope_theta", 10_000.0)],
            eps: c.get("norm_eps").and_then(Json::num).unwrap_or(1e-5) as f32,
            silu: c.get("hidden_activation").and_then(Json::str) == Some("silu"),
        };
        let sound = heads > 0 && hidden % heads == 0 && (hidden / heads) % 2 == 0 && inter * vocab * layers > 0;
        if sound { Ok(cfg) } else { Err(Error::Format("config.json: an inconsistent ModernBERT shape".into())) }
    }

    /// Dimensions of each head.
    pub fn head_dim(&self) -> usize {
        self.hidden / self.heads
    }

    /// Number of layers.
    pub fn layers(&self) -> usize {
        self.global.len()
    }
}

/// An error unless config `c` is a ModernBERT this encoder runs: GELU or `SiLU`, no attention biases.
fn supported(c: &Json) -> Result<()> {
    if c.get("model_type").and_then(Json::str) != Some("modernbert") {
        return Err(Error::Unsupported("an encoder other than ModernBERT".into()));
    }
    let act = c.get("hidden_activation").and_then(Json::str);
    if !matches!(act, Some("gelu" | "silu")) || c.get("attention_bias").and_then(Json::bool) == Some(true) {
        return Err(Error::Unsupported("a ModernBERT with another activation or with biases".into()));
    }
    Ok(())
}

/// Whether each of `layers` layers attends globally: `layer_types`, else every `global_attn_every_n_layers`-th.
fn global_layers(c: &Json, layers: usize) -> Vec<bool> {
    let every = c.get("global_attn_every_n_layers").and_then(Json::num).unwrap_or(3.0) as usize;
    match c.get("layer_types").and_then(Json::arr) {
        Some(types) => types.iter().map(|t| t.str() == Some("full_attention")).collect(),
        None => (0..layers).map(|i| i % every.max(1) == 0).collect(),
    }
}

/// The RoPE base of `kind` layers: in `rope_parameters`, else under the `legacy` key, else `default`.
fn theta(c: &Json, kind: &str, legacy: &str, default: f64) -> f32 {
    let modern = c.get("rope_parameters").and_then(|r| r.get(kind)).and_then(|r| r.get("rope_theta")).and_then(Json::num);
    modern.or_else(|| c.get(legacy).and_then(Json::num)).unwrap_or(default) as f32
}
