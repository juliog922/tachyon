//! Conversion of Gemma 4's vision tower into `vision.wpk`, beside the decoder's weights, when the checkpoint has
//! one. Matrices are `f16 [rows, cols]` as PyTorch's `Linear` holds them, each with its clip bounds `{name}.clip`
//! (`f32 [4 + 2·rows]`: the input's, two zeros, then each row's output bounds; infinite where the layer has none),
//! as [`crate::ptx::GEMM_CLIP`] reads them. Norms are `f32`.
//!
//! - `patch`: the patch embedding `[hidden, 3·16·16]`, a patch's pixels row by row, red, green and blue.
//! - `pos`: the position embeddings, `f16 [2·positions, hidden]`: the `x` table, then the `y` table.
//! - Per layer `i`: `{i}.qkv` (queries, keys and values stacked), `{i}.o`, `{i}.gate_up` (the gate's row `j`, then
//!   the up's, interleaved), `{i}.down`; `{i}.input_norm`, `{i}.post_attn_norm`, `{i}.pre_ffn_norm`,
//!   `{i}.post_ffn_norm`, `{i}.q_norm`, `{i}.k_norm`.
//! - `proj`: the projection of a soft token into the decoder's embeddings, `[text hidden, hidden]`.

use crate::quant::{Safetensors, f16_bits};
use crate::wpk::{Dtype, Writer};
use crate::{Error, Result};
use std::path::Path;

/// Where the tower's tensors are in a Gemma 4 checkpoint.
const P: &str = "model.vision_tower.";

/// A linear layer read from the checkpoint: rows, columns, weights, and its bounds (input then output).
pub(crate) struct Linear {
    rows: usize,
    cols: usize,
    w: Vec<f32>,
    bounds: [f32; 4],
}

impl Linear {
    /// Its first and second halves of rows, as two layers of its bounds (a GLU's two products).
    pub(crate) fn halves(self) -> [Linear; 2] {
        let (rows, cols, bounds) = (self.rows / 2, self.cols, self.bounds);
        let (a, b) = self.w.split_at(rows * cols);
        [a, b].map(|w| Linear { rows, cols, w: w.to_vec(), bounds })
    }
}

/// Writes `vision.wpk` into `dst` if `st` has a vision tower this engine runs.
pub fn convert(st: &Safetensors, dst: &Path) -> Result<()> {
    if !st.has(&format!("{P}patch_embedder.input_proj.weight")) {
        return Ok(());
    }
    // The larger models' towers standardize their soft tokens (`std_bias`, `std_scale`): not yet (step 12d), so
    // those models convert for text only.
    if st.has(&format!("{P}std_bias")) {
        return Ok(());
    }
    let mut w = Writer::create(dst.join("vision.wpk"))?;
    linear(&mut w, st, "patch", &[format!("{P}patch_embedder.input_proj")], false)?;
    positions(&mut w, st)?;
    for i in (0..).take_while(|i| st.has(&format!("{P}encoder.layers.{i}.input_layernorm.weight"))) {
        layer(&mut w, st, i)?;
    }
    linear(&mut w, st, "proj", &["model.embed_vision.embedding_projection".into()], false)?;
    w.finish()
}

/// The position embeddings, the `x` table then the `y` table.
fn positions(w: &mut Writer, st: &Safetensors) -> Result<()> {
    let table = format!("{P}patch_embedder.position_embedding_table");
    let (rows, cols) = st.dims(&table)?;
    w.add("pos", Dtype::F16, &[rows as u64, cols as u64], &halves(&st.rows(&table, 0..rows)?))
}

/// Layer `i`'s tensors.
fn layer(w: &mut Writer, st: &Safetensors, i: usize) -> Result<()> {
    let l = format!("{P}encoder.layers.{i}.");
    let attn = |m: &str| format!("{l}self_attn.{m}");
    linear(w, st, &format!("{i}.qkv"), &[attn("q_proj"), attn("k_proj"), attn("v_proj")], false)?;
    linear(w, st, &format!("{i}.o"), &[attn("o_proj")], false)?;
    linear(w, st, &format!("{i}.gate_up"), &[format!("{l}mlp.gate_proj"), format!("{l}mlp.up_proj")], true)?;
    linear(w, st, &format!("{i}.down"), &[format!("{l}mlp.down_proj")], false)?;
    let norms = [
        ("input_norm", "input_layernorm"),
        ("post_attn_norm", "post_attention_layernorm"),
        ("pre_ffn_norm", "pre_feedforward_layernorm"),
        ("post_ffn_norm", "post_feedforward_layernorm"),
        ("q_norm", "self_attn.q_norm"),
        ("k_norm", "self_attn.k_norm"),
    ];
    for (name, source) in norms {
        let source = format!("{l}{source}.weight");
        // A vector reads as one row: `dims` gives (1, length).
        let v = st.rows(&source, 0..st.dims(&source)?.0)?;
        w.add(&format!("{i}.{name}"), Dtype::F32, &[v.len() as u64], &floats(&v))?;
    }
    Ok(())
}

/// The linear layer `m` (clippable, with `linear.weight` and bounds, or plain, with `weight`).
pub(crate) fn read(st: &Safetensors, m: &str) -> Result<Linear> {
    let clipped = format!("{m}.linear.weight");
    let weight = if st.has(&clipped) { clipped } else { format!("{m}.weight") };
    let (rows, cols) = st.dims(&weight)?;
    let mut bounds = [f32::NEG_INFINITY, f32::INFINITY, f32::NEG_INFINITY, f32::INFINITY];
    for (b, k) in bounds.iter_mut().zip(["input_min", "input_max", "output_min", "output_max"]) {
        let name = format!("{m}.{k}");
        if st.has(&name) {
            *b = st.rows(&name, 0..1)?[0];
        }
    }
    Ok(Linear { rows, cols, w: st.rows(&weight, 0..rows)?, bounds })
}

/// Writes the linear layers `modules` as one matrix `name`, stacked or with their rows interleaved, and its bounds:
/// the layers must share their columns and input bounds.
pub(crate) fn linear(w: &mut Writer, st: &Safetensors, name: &str, modules: &[String], interleave: bool) -> Result<()> {
    write(w, name, &modules.iter().map(|m| read(st, m)).collect::<Result<Vec<_>>>()?, interleave)
}

/// Writes `parts` as one matrix `name`, as [`linear`] does.
pub(crate) fn write(w: &mut Writer, name: &str, parts: &[Linear], interleave: bool) -> Result<()> {
    let first = &parts[0];
    if parts.iter().any(|p| p.cols != first.cols || p.bounds[..2] != first.bounds[..2] || interleave && p.rows != first.rows) {
        return Err(Error::Format(format!("{name}: its layers differ in shape or input bounds")));
    }
    let order: Vec<(usize, usize)> = if interleave {
        (0..first.rows).flat_map(|r| (0..parts.len()).map(move |k| (k, r))).collect()
    } else {
        parts.iter().enumerate().flat_map(|(k, p)| (0..p.rows).map(move |r| (k, r))).collect()
    };
    let (mut data, mut clip) = (Vec::with_capacity(order.len() * first.cols), vec![first.bounds[0], first.bounds[1], 0.0, 0.0]);
    for &(k, r) in &order {
        data.extend_from_slice(&parts[k].w[r * first.cols..][..first.cols]);
        clip.extend_from_slice(&parts[k].bounds[2..]);
    }
    w.add(name, Dtype::F16, &[order.len() as u64, first.cols as u64], &halves(&data))?;
    w.add(&format!("{name}.clip"), Dtype::F32, &[clip.len() as u64], &floats(&clip))
}

/// `v` as little-endian `f16`.
pub(crate) fn halves(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|&x| f16_bits(x).to_le_bytes()).collect()
}

/// `v` as little-endian `f32`.
pub(crate) fn floats(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
