//! Conversion of a Hugging Face encoder checkpoint into a model directory: `config.json` (the ModernBERT's),
//! `head.json` (a Laya checkpoint's `rl_agent_config.json`), `tokenizer.tok`, `weights.wpk` for VRAM and
//! `host.wpk` for pinned host memory.
//!
//! `weights.wpk`: `embed_norm`, per layer `{i}.attn_norm` (not layer 0), `{i}.qkv`, `{i}.o`, `{i}.mlp_norm`,
//! `{i}.up` (`Wi`), `{i}.down` (`Wo`), and `final_norm`; matrices `f16 [rows, cols]` as PyTorch's `Linear` holds
//! them, norms `f32`. An embedding model at least [`Q8_WIDTH`] wide takes 8-bit weights in its layers instead (spec,
//! step 10e): each matrix `i8` with its scales `{name}.scale`, `f32 [rows, cols / 64]` ([`crate::quant::q8e`]). A Laya checkpoint adds its head: `type_emb` (`f32 [3, h]`); per head layer `h{l}.norm1.w`,
//! `.b`, `h{l}.qkv.w`, `.b`, `h{l}.o.w`, `.b`, `h{l}.norm2.w`, `.b`, `h{l}.up.w`, `.b`, `h{l}.down.w`, `.b`; and
//! `score.norm.w`, `.b`, `score.1.w`, `.b`, `score.2.w`, `.b`, the last padded with zeros to 64 rows for the
//! matrix product. `host.wpk`: `embed`, the token embeddings, `f16 [vocab, h]`.
//!
//! A Laya repository holds `rl_agent_config.json`, `encoder/config.json`, `tokenizer/tokenizer.json` and one
//! `model.safetensors` whose encoder tensors start with `encoder.` (the encoder's `config.json` may also sit at the
//! top, as in the tests' tiny Laya); a Granite Embedding repository holds
//! `config.json`, `tokenizer.json` and the encoder's tensors.

use super::Config;
use super::laya::Head;
use crate::quant::{Safetensors, f16_bits, q8e};
use crate::token::Tokenizer;
use crate::wpk::{Dtype, Writer};
use crate::{Error, Result};
use std::path::Path;

/// The narrowest embedding model whose layers take 8-bit weights. The precision study (`scripts/encoders.py`) puts
/// Granite 311M (768 wide) at a lowest cosine of 0.999926 in Q8, within the tests' 0.9999, and Granite 97M (384) at
/// 0.998635, which stays in `f16` (1.000000).
const Q8_WIDTH: usize = 768;
/// Rows the last scorer layer is padded to.
const PAD: usize = 64;

/// Converts the encoder checkpoint in `src` (Laya or Granite Embedding, see the module documentation) to a model
/// directory `dst`.
pub fn convert(src: &Path, dst: &Path) -> Result<()> {
    convert_with(src, dst, None)
}

/// [`convert`], with 8-bit layers if `q8` says so; by default an embedding model at least 768 wide has them.
pub fn convert_with(src: &Path, dst: &Path, q8: Option<bool>) -> Result<()> {
    let laya = src.join("rl_agent_config.json").exists();
    let dir = if laya && src.join("encoder").is_dir() { src.join("encoder") } else { src.to_path_buf() };
    let st = Safetensors::open(src)?;
    let p = if laya { "encoder." } else { st.prefix("embeddings.norm.weight") };
    std::fs::create_dir_all(dst)?;
    let source = Source { config: dir.join("config.json"), prefix: p, head: laya.then(|| src.join("rl_agent_config.json")), q8 };
    write(dst, &st, &source)?;
    tokenizer(&src.join(if laya { "tokenizer/tokenizer.json" } else { "tokenizer.json" }), dst)
}

/// Writes into `dst` the encoder's `config.json`, `weights.wpk`, with Laya's head when the source has one (its
/// configuration copied as `head.json`), and `host.wpk`.
fn write(dst: &Path, st: &Safetensors, source: &Source) -> Result<()> {
    let text = std::fs::read(&source.config)?;
    std::fs::write(dst.join("config.json"), &text)?;
    let mut w = Writer::create(dst.join("weights.wpk"))?;
    let cfg = Config::from_json(&text)?;
    let p = source.prefix;
    encoder(&mut w, st, &cfg, (p, source.q8.unwrap_or(source.head.is_none() && cfg.hidden >= Q8_WIDTH)))?;
    if let Some(path) = &source.head {
        laya(&mut w, st, path, dst)?;
    }
    w.finish()?;
    host(dst, st, p)
}

/// Where a checkpoint holds an encoder's parts: its `config.json`, its tensors' names' prefix, Laya's head
/// configuration if it has one; and whether its layers take 8-bit weights (`None`: as [`Q8_WIDTH`] says).
struct Source<'a> {
    config: std::path::PathBuf,
    prefix: &'a str,
    head: Option<std::path::PathBuf>,
    q8: Option<bool>,
}

/// Writes `host.wpk`, the token embeddings, into `dst`.
fn host(dst: &Path, st: &Safetensors, p: &str) -> Result<()> {
    let mut host = Writer::create(dst.join("host.wpk"))?;
    matrix(&mut host, st, "embed", &format!("{p}embeddings.tok_embeddings.weight"), 0)?;
    host.finish()
}

/// Stores the tokenizer at `path`, if there is one, in `dst`.
fn tokenizer(path: &Path, dst: &Path) -> Result<()> {
    if !path.exists() {
        return Ok(());
    }
    Ok(std::fs::write(dst.join("tokenizer.tok"), Tokenizer::from_json(&std::fs::read(path)?)?.to_bytes())?)
}

/// The encoder's tensors, whose checkpoint names start with `p`; the layers' matrices in Q8E if `q8`.
fn encoder(w: &mut Writer, st: &Safetensors, cfg: &Config, (p, q8): (&str, bool)) -> Result<()> {
    vector(w, st, "embed_norm", &format!("{p}embeddings.norm.weight"))?;
    for i in 0..cfg.layers() {
        let l = format!("{p}layers.{i}.");
        if i > 0 {
            vector(w, st, &format!("{i}.attn_norm"), &format!("{l}attn_norm.weight"))?;
        }
        vector(w, st, &format!("{i}.mlp_norm"), &format!("{l}mlp_norm.weight"))?;
        for (name, source) in [("qkv", "attn.Wqkv"), ("o", "attn.Wo"), ("up", "mlp.Wi"), ("down", "mlp.Wo")] {
            let (name, source) = (format!("{i}.{name}"), format!("{l}{source}.weight"));
            if q8 { matrix_q8(w, st, &name, &source) } else { matrix(w, st, &name, &source, 0) }?;
        }
    }
    vector(w, st, "final_norm", &format!("{p}final_norm.weight"))
}

/// Laya's head and scorer, configured by `path`, which is copied into `dst` as `head.json`.
fn laya(w: &mut Writer, st: &Safetensors, path: &Path, dst: &Path) -> Result<()> {
    let json = std::fs::read(path)?;
    head_layers(w, st, Head::from_json(&json)?.layers)?;
    scorer(w, st)?;
    Ok(std::fs::write(dst.join("head.json"), json)?)
}

/// Laya's type embedding and head of `layers` layers.
fn head_layers(w: &mut Writer, st: &Safetensors, layers: usize) -> Result<()> {
    vector(w, st, "type_emb", "type_emb.weight")?;
    for l in 0..layers {
        let s = format!("head.layers.{l}.");
        let parts =
            [("norm1", "norm1."), ("qkv", "self_attn.in_proj_"), ("o", "self_attn.out_proj."), ("norm2", "norm2."), ("up", "linear1."), ("down", "linear2.")];
        for (name, source) in parts {
            let (weight, out) = (format!("{s}{source}weight"), format!("h{l}.{name}.w"));
            if name.starts_with("norm") { vector(w, st, &out, &weight) } else { matrix(w, st, &out, &weight, 0) }?;
            vector(w, st, &format!("h{l}.{name}.b"), &format!("{s}{source}bias"))?;
        }
    }
    Ok(())
}

/// Laya's scorer: a layer norm, a layer to `h`, and one to a logit, padded to [`PAD`] rows.
fn scorer(w: &mut Writer, st: &Safetensors) -> Result<()> {
    vector(w, st, "score.norm.w", "scorer.0.weight")?;
    vector(w, st, "score.norm.b", "scorer.0.bias")?;
    for (name, source, pad) in [("score.1", "scorer.1", 0), ("score.2", "scorer.3", PAD)] {
        matrix(w, st, &format!("{name}.w"), &format!("{source}.weight"), pad)?;
        let mut bias = st.rows(&format!("{source}.bias"), 0..st.dims(&format!("{source}.bias"))?.0)?;
        bias.resize(bias.len().max(pad), 0.0);
        w.add(&format!("{name}.b"), Dtype::F32, &[bias.len() as u64], &bias.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())?;
    }
    Ok(())
}

/// Writes `source` as `f16 [rows, cols]`, with zero rows up to `pad` rows; read a slice of rows at a time.
fn matrix(w: &mut Writer, st: &Safetensors, name: &str, source: &str, pad: usize) -> Result<()> {
    let (rows, cols) = st.dims(source)?;
    if cols == 0 {
        return Err(Error::Format(format!("{source}: no columns")));
    }
    let mut out = Vec::with_capacity(rows.max(pad) * cols * 2);
    let step = (1 << 22) / cols + 1;
    for start in (0..rows).step_by(step) {
        out.extend(st.rows(source, start..rows.min(start + step))?.iter().flat_map(|&x| f16_bits(x).to_le_bytes()));
    }
    out.resize(rows.max(pad) * cols * 2, 0);
    w.add(name, Dtype::F16, &[rows.max(pad) as u64, cols as u64], &out)
}

/// Writes `source` as Q8E: `i8 [rows, cols]` and its scales, `{name}.scale`; read a slice of rows at a time.
fn matrix_q8(w: &mut Writer, st: &Safetensors, name: &str, source: &str) -> Result<()> {
    let (rows, cols) = st.dims(source)?;
    if cols == 0 || cols % 64 != 0 {
        return Err(Error::Format(format!("{source}: {cols} columns, not a multiple of 64")));
    }
    let (mut values, mut scales) = (Vec::with_capacity(rows * cols), Vec::with_capacity(rows * cols / 16));
    let step = (1 << 22) / cols + 1;
    for start in (0..rows).step_by(step) {
        let (q, s) = q8e(&st.rows(source, start..rows.min(start + step))?);
        values.extend(q.iter().map(|&v| v.to_le_bytes()[0]));
        scales.extend(s.iter().flat_map(|v| v.to_le_bytes()));
    }
    w.add(name, Dtype::I8, &[rows as u64, cols as u64], &values)?;
    w.add(&format!("{name}.scale"), Dtype::F32, &[rows as u64, cols as u64 / 64], &scales)
}

/// Writes a vector as `f32`.
fn vector(w: &mut Writer, st: &Safetensors, name: &str, source: &str) -> Result<()> {
    let v = st.rows(source, 0..st.dims(source)?.0)?;
    w.add(name, Dtype::F32, &[v.len() as u64], &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
}
