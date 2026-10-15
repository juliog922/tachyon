//! Conversion of Gemma 4's audio tower into `audio.wpk`, beside the decoder's weights, when the checkpoint has one.
//! Matrices are `f16 [rows, cols]` with their clip bounds `{name}.clip`, as `vision.wpk`'s
//! ([`crate::vision::convert`]); everything else is `f32`.
//!
//! - `sub0`, `sub1`: the subsampling convolutions `[out, in, 3, 3]`, flattened; `sub0.norm`, `sub1.norm`.
//! - `in`: the projection of the subsampled features.
//! - Per layer `i`: for each feed-forward block `f` (`ffn1`, `ffn2`) `{i}.{f}.up`, `{i}.{f}.down`, `{i}.{f}.pre` and
//!   `{i}.{f}.post` (times the residual weight); `{i}.qkv` (stacked), `{i}.o`, `{i}.attn_pre`, `{i}.attn_post`,
//!   `{i}.scale` (each query dimension's scale, `d^−½ / ln 2 · softplus(per_dim_scale)`), `{i}.rel` (the relative
//!   positions' keys, `[left + 1, hidden]` by distance, the sinusoidal embedding through `relative_k_proj`);
//!   `{i}.conv_pre`, `{i}.start` (the GLU's halves' rows interleaved), `{i}.taps` (`[hidden, K]`), `{i}.conv_norm`,
//!   `{i}.end`; `{i}.out_norm`.
//! - `out` with `out.bias`, the output projection; `proj`, into the decoder's embeddings.

use crate::json::Json;
use crate::quant::Safetensors;
use crate::vision::convert::{floats, linear, read, write};
use crate::wpk::{Dtype, Writer};
use crate::{Error, Result};
use std::path::Path;

/// Where the tower's tensors are in a Gemma 4 checkpoint.
const P: &str = "model.audio_tower.";

/// What the conversion reads of `config.json`'s `audio_config`: the residual weight, the past positions attention
/// sees, and the head size.
struct Shape {
    residual: f32,
    /// The farthest past position a query sees.
    left: usize,
    head: usize,
}

/// Writes `audio.wpk` into `dst` if `st` has an audio tower; `config` is the checkpoint's `config.json`.
pub fn convert(st: &Safetensors, config: &[u8], dst: &Path) -> Result<()> {
    if !st.has(&format!("{P}subsample_conv_projection.input_proj_linear.weight")) {
        return Ok(());
    }
    let shape = Shape::from_json(config)?;
    let mut w = Writer::create(dst.join("audio.wpk"))?;
    subsample(&mut w, st)?;
    for i in (0..).take_while(|i| st.has(&format!("{P}layers.{i}.norm_out.weight"))) {
        layer(&mut w, st, i, &shape)?;
    }
    linear(&mut w, st, "out", &[format!("{P}output_proj")], false)?;
    vector(&mut w, st, "out.bias", &format!("{P}output_proj.bias"), 1.0)?;
    linear(&mut w, st, "proj", &["model.embed_audio.embedding_projection".into()], false)?;
    w.finish()
}

impl Shape {
    /// The shape `config.json` gives, Gemma 4's defaults where it gives none.
    fn from_json(config: &[u8]) -> Result<Shape> {
        let doc = Json::parse(config)?;
        let num = |k: &str, or: f64| doc.get("audio_config").and_then(|a| a.get(k)).and_then(Json::num).unwrap_or(or);
        let heads = (num("num_attention_heads", 8.0) as usize).max(1);
        Ok(Shape {
            residual: num("residual_weight", 0.5) as f32,
            left: num("attention_context_left", 13.0) as usize - 2,
            head: num("hidden_size", 1024.0) as usize / heads,
        })
    }
}

/// The subsampling convolutions, their norms, and the projection after them.
fn subsample(w: &mut Writer, st: &Safetensors) -> Result<()> {
    for k in 0..2 {
        let conv = format!("{P}subsample_conv_projection.layer{k}.");
        vector(w, st, &format!("sub{k}"), &format!("{conv}conv.weight"), 1.0)?;
        vector(w, st, &format!("sub{k}.norm"), &format!("{conv}norm.weight"), 1.0)?;
    }
    linear(w, st, "in", &[format!("{P}subsample_conv_projection.input_proj_linear")], false)
}

/// Layer `i`'s feed-forward blocks.
fn ffns(w: &mut Writer, st: &Safetensors, i: usize, residual: f32) -> Result<()> {
    let l = format!("{P}layers.{i}.");
    for (f, source) in [("ffn1", "feed_forward1"), ("ffn2", "feed_forward2")] {
        linear(w, st, &format!("{i}.{f}.up"), &[format!("{l}{source}.ffw_layer_1")], false)?;
        linear(w, st, &format!("{i}.{f}.down"), &[format!("{l}{source}.ffw_layer_2")], false)?;
        vector(w, st, &format!("{i}.{f}.pre"), &format!("{l}{source}.pre_layer_norm.weight"), 1.0)?;
        vector(w, st, &format!("{i}.{f}.post"), &format!("{l}{source}.post_layer_norm.weight"), residual)?;
    }
    Ok(())
}

/// Layer `i`'s tensors.
fn layer(w: &mut Writer, st: &Safetensors, i: usize, shape: &Shape) -> Result<()> {
    let l = format!("{P}layers.{i}.");
    ffns(w, st, i, shape.residual)?;
    let attn = |m: &str| format!("{l}self_attn.{m}");
    linear(w, st, &format!("{i}.qkv"), &[attn("q_proj"), attn("k_proj"), attn("v_proj")], false)?;
    linear(w, st, &format!("{i}.o"), &[attn("post")], false)?;
    relative(w, st, i, shape)?;
    let norms = [
        ("attn_pre", "norm_pre_attn"),
        ("attn_post", "norm_post_attn"),
        ("conv_pre", "lconv1d.pre_layer_norm"),
        ("conv_norm", "lconv1d.conv_norm"),
        ("out_norm", "norm_out"),
    ];
    for (name, source) in norms {
        vector(w, st, &format!("{i}.{name}"), &format!("{l}{source}.weight"), 1.0)?;
    }
    write(w, &format!("{i}.start"), &read(st, &format!("{l}lconv1d.linear_start"))?.halves(), true)?;
    vector(w, st, &format!("{i}.taps"), &format!("{l}lconv1d.depthwise_conv1d.weight"), 1.0)?;
    linear(w, st, &format!("{i}.end"), &[format!("{l}lconv1d.linear_end")], false)
}

/// Layer `i`'s query scale and relative positions' keys.
fn relative(w: &mut Writer, st: &Safetensors, i: usize, shape: &Shape) -> Result<()> {
    let attn = format!("{P}layers.{i}.self_attn.");
    let per_dim = st.rows(&format!("{attn}per_dim_scale"), 0..1)?;
    if per_dim.len() != shape.head {
        return Err(Error::Format(format!("audio layer {i}: per_dim_scale of {} for heads of {}", per_dim.len(), shape.head)));
    }
    let base = 1.0 / (shape.head as f64).sqrt() / std::f64::consts::LN_2;
    let scale: Vec<f32> = per_dim.iter().map(|&p| (base * f64::from(p).exp().ln_1p()) as f32).collect();
    w.add(&format!("{i}.scale"), Dtype::F32, &[scale.len() as u64], &floats(&scale))?;
    let name = format!("{attn}relative_k_proj.weight");
    let (rows, cols) = st.dims(&name)?;
    let proj = st.rows(&name, 0..rows)?;
    let half = cols / 2;
    let step = (10_000f64).ln() / (half.max(2) - 1) as f64;
    let mut rel = Vec::with_capacity((shape.left + 1) * rows);
    for d in 0..=shape.left {
        // The sinusoidal embedding of distance `d`: sines, then cosines, of `d` at timescales from 1 to 10⁴.
        let embed: Vec<f64> = (0..cols)
            .map(|k| {
                let a = d as f64 * (-((k % half) as f64) * step).exp();
                if k < half { a.sin() } else { a.cos() }
            })
            .collect();
        rel.extend((0..rows).map(|r| proj[r * cols..][..cols].iter().zip(&embed).map(|(&p, e)| f64::from(p) * e).sum::<f64>() as f32));
    }
    w.add(&format!("{i}.rel"), Dtype::F32, &[(shape.left + 1) as u64, rows as u64], &floats(&rel))
}

/// Writes tensor `source`, flattened, times `scale`, as `f32` `name`.
fn vector(w: &mut Writer, st: &Safetensors, name: &str, source: &str, scale: f32) -> Result<()> {
    let v: Vec<f32> = st.rows(source, 0..st.dims(source)?.0)?.iter().map(|x| x * scale).collect();
    w.add(name, Dtype::F32, &[v.len() as u64], &floats(&v))
}
