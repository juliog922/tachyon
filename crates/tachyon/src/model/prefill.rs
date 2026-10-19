//! Prefill: a prompt's tokens through the model many at a time, on tensor cores, filling the key-value caches.
//!
//! A chunk of up to [`super::Settings::chunk`] tokens at positions `pos..` (the step counter) runs as one
//! sequence of launches, each kernel over every token: the embeddings, then per layer the matrix products as
//! [`GEMM_Q4`] on Q8P activations, the keys and values stored by [`KV_256`] or [`KV_512`] and the queries by
//! [`QUERY_256`] or [`QUERY_512`] before [`FLASH_256`] or [`FLASH_512`] attends on tensor cores (its `f32` out made
//! Q8P by [`QUANT_Q8P`]), and the residual steps as [`NORM_Q8P`] and [`GEGLU_Q8P`].
//! A model without per-layer inputs skips their kernels, and its layers start and end in the same residual buffer.
//! Only the layers that write a cache run: the last [`super::Config::shared`] layers read earlier layers' caches
//! and add nothing a later token needs, so a prompt's tokens but its last skip them (18 of E4B's 42 layers). The
//! last token runs the decode step, which samples the first answer token. The chunk's kernels are launched
//! directly, not recorded: their count does not depend on the chunk's length, which the matrix products need.
//! Images' soft tokens replace their positions' embeddings right after the lookup ([`scatter`]), before the
//! per-layer projection reads them.

// Buffers and dimensions keep their one-letter names from the math, as in `decode`: x, q, y, d, h, l, n.
#![allow(clippy::many_single_char_names)]

use super::Config;
use super::decode::{A, Bufs, Mat, Op, Weights};
use crate::ptx::{
    EMBED_Q4, GATHER_ROWS, GEGLU_Q8P, GEMM_Q4, GEMM_ROWS, GEMM_TOKENS, KV_256, KV_512, MOE_ADD, NORM_Q8P, NORM_THREADS, QUANT_Q8P, QUERY_256, QUERY_512,
    ROUTE_TOPK, SUM_SLICES, flash::Shape,
};

/// Device addresses of prefill's buffers, each for a chunk of tokens: the residual stream (two), Q8P values and
/// scales, the query-key-value projections, the other products, the per-layer inputs (embedded, projected,
/// combined), attention's scratch and counters, the chunk's token IDs (pinned host memory), and the room in `y`
/// (floats per token) and the GPU's multiprocessors, for the products' slices.
pub(super) struct Chunk {
    pub(super) x: [u64; 2],
    pub(super) q: [u64; 2],
    pub(super) qkv: u64,
    pub(super) y: u64,
    pub(super) per_layer: [u64; 3],
    pub(super) attend: [u64; 2],
    /// The mixture of experts' buffers: zeros, two sums, the dense MLP's output, the experts' sum, one expert's
    /// output, the router's logits, the chosen experts and their weights, and the experts' Q8P input.
    pub(super) moe: [u64; 11],
    pub(super) ids: u64,
    pub(super) room: u32,
    pub(super) sms: u32,
}

/// Lays out prefill's buffers for `chunk` tokens with `take`, which returns the address of `len` new bytes.
pub(super) fn layout(cfg: &Config, chunk: usize, take: &mut impl FnMut(usize) -> u64) -> Chunk {
    let (d, pl, f) = (cfg.head_dim[1], cfg.layers() * cfg.per_layer, 4 * chunk.next_multiple_of(GEMM_TOKENS as usize));
    let widest = cfg.hidden.max(cfg.mlp(cfg.layers() - 1)).max(cfg.heads * d).max(cfg.per_layer);
    let x = [take(f * cfg.hidden), take(f * cfg.hidden)];
    let q = [take(f / 4 * widest), take(f / 4 * widest / 8)];
    let room = (2 * cfg.mlp(cfg.layers() - 1)).max(cfg.hidden).max(pl).max(cfg.heads * d);
    let (qkv, y) = (take(f * super::decode::qkv_width(cfg)), take(f * room));
    let per_layer = [take(f * pl), take(f * pl), take(f * pl)];
    let attend = [take(f * cfg.heads / 4 * (8 + 4 * d)), take(f * cfg.heads / 4)];
    let ([e, k, _], hm) = (cfg.experts, if cfg.moe() { cfg.hidden } else { 0 });
    let mut moe = [0; 11];
    for (m, len) in moe.iter_mut().zip([f * hm, f * hm, f * hm, f * hm, f * hm, f * hm, f * e, f * k, f * k, f / 4 * hm, f / 32 * hm]) {
        *m = take(len);
    }
    Chunk { x, q, qkv, y, per_layer, attend, moe, ids: 0, room: room as u32, sms: 0 }
}

/// `y = X·Mᵀ` over `n` tokens of Q8P input `q`; into `c.y` in slices of the columns where they fill the GPU
/// better (see [`crate::ptx::slices`]), added up after.
fn gemm(m: Mat, q: [u64; 2], y: u64, n: u32, c: &Chunk) -> Vec<Op> {
    let room = if y == c.y { c.room / m.rows } else { 1 };
    let slices = crate::ptx::slices(m.rows, m.cols, n, c.sms, room);
    let args = vec![A::P(m.w[0]), A::P(m.w[1]), A::P(q[0]), A::P(q[1]), A::P(y), A::U(m.rows), A::U(m.cols), A::U(n)];
    let len = n * m.rows;
    let sum = (SUM_SLICES, [len.div_ceil(1024), 1, 1], 256, vec![A::P(y), A::U(len), A::U(slices)]);
    std::iter::once((GEMM_Q4, [n.div_ceil(GEMM_TOKENS), m.rows / GEMM_ROWS, slices], 128, args)).chain((slices > 1).then_some(sum)).collect()
}

/// [`NORM_Q8P`] on `p = [h, h_out, y, w1, w2]` over `chunks` of `len`, `k = [eps, scale]`, Q8P to `q`.
fn norm(p: [u64; 5], q: [u64; 2], len: usize, chunks: u32, k: [f32; 2]) -> Op {
    let args = p.into_iter().chain(q).map(A::P).chain([A::U(len as u32), A::F(k[0]), A::F(k[1])]).collect();
    (NORM_Q8P, [len as u32 / NORM_THREADS, chunks, 1], NORM_THREADS, args)
}

/// Q8P of `gelu(a) · b` over `n` rows of `width`, `a` and `b` rows `stride[0]`, `stride[1]` floats apart, the output's
/// `stride[2]` (the reading matrix's padded columns).
fn geglu([a, b]: [u64; 2], q: [u64; 2], n: u32, width: usize, stride: [u32; 3]) -> Op {
    let len = n * width as u32;
    let args = [a, b, q[0], q[1]].map(A::P).into_iter().chain([len, width as u32, stride[0], stride[1], stride[2]].map(A::U)).collect();
    (GEGLU_Q8P, [len.div_ceil(256), 1, 1], 256, args)
}

/// Layer `i`'s experts over the chunk of `n` tokens, as [`super::decode`]'s: the router and [`ROUTE_TOPK`] for
/// every token, then every expert over the whole chunk, [`MOE_ADD`] keeping its output for the tokens that chose
/// it, weighted, expert after expert (a token's sum follows the experts' numbers, where a decode step's follows its
/// choice). A first form: it runs 16 times the experts' products a token needs.
fn experts(cfg: &Config, w: &Weights, c: &Chunk, [i, n]: [usize; 2]) -> Vec<Op> {
    let ([e, k, width], h, eps, nt) = (cfg.experts.map(|v| v as u32), cfg.hidden, cfg.eps, n as u32);
    let (at, x1) = (|name: &str| w.at(&format!("{i}.{name}")), super::decode::residual(cfg, c.x, i).1);
    let [.., ymoe, yx, logits, idx, wts, q2s, q2z] = c.moe;
    let (up, down) = (w.mat(&format!("{i}.experts.gate_up")), w.mat(&format!("{i}.experts.down")));
    let mut ops = vec![norm([x1, 0, 0, 0, at("router_scale")], c.q, h, nt, [eps, 1.0])];
    ops.extend(gemm(w.mat(&format!("{i}.router")), c.q, logits, nt, c));
    ops.push((ROUTE_TOPK, [nt, 1, 1], 128, [logits, at("expert_scale"), idx, wts].map(A::P).into_iter().chain([A::U(e), A::U(k)]).collect()));
    ops.push(norm([x1, 0, 0, 0, at("pre_feedforward_layernorm_2")], [q2s, q2z], h, nt, [eps, 1.0]));
    let expert = |m: Mat, j: u32| {
        let rows = m.rows / e;
        let skip = u64::from(j) * u64::from(rows) * u64::from(m.cols);
        Mat { w: [m.w[0] + skip / 2, m.w[1] + skip / 32], rows, cols: m.cols }
    };
    for j in 0..e {
        ops.extend(gemm(expert(up, j), [q2s, q2z], c.y, nt, c));
        ops.push(geglu([c.y, c.y + 4 * u64::from(width)], c.q, nt, width as usize, [2 * width, 2 * width, down.cols]));
        ops.extend(gemm(expert(down, j), c.q, yx, nt, c));
        let add = [ymoe, yx, idx, wts].map(A::P).into_iter().chain([j, k, h as u32, u32::from(j == 0)].map(A::U));
        ops.push((MOE_ADD, [(h as u32).div_ceil(256), nt, 1], 256, add.collect()));
    }
    ops
}

/// The soft tokens of `run` (its first row in the chunk, its rows) over the chunk's embeddings in `x[0]`, from
/// `soft` (pinned: the rows, at their place in the chunk, and the row numbers).
pub(super) fn scatter(cfg: &Config, c: &Chunk, soft: [u64; 2], (first, rows): (u32, u32)) -> Op {
    let (h, skip) = (cfg.hidden as u32, u64::from(first) * cfg.hidden as u64 * 4);
    (GATHER_ROWS, [h.div_ceil(256), rows, 1], 256, vec![A::P(soft[0] + skip), A::P(soft[1]), A::P(c.x[0] + skip), A::U(h)])
}

/// The launches of a chunk of `n` tokens, whose IDs are at `c.ids`; `scalars` holds each layer's `layer_scalar`.
pub(super) fn ops(cfg: &Config, [w, hw]: [&Weights; 2], b: &Bufs, (c, scalars): (&Chunk, &[f32]), n: u32) -> Vec<Op> {
    let (h, pl, l, eps) = (cfg.hidden, cfg.per_layer, cfg.layers(), cfg.eps);
    let [ple, pli, proj] = c.per_layer;
    let embed =
        |m: Mat, out, scale| (EMBED_Q4, [m.cols.div_ceil(256), n, 1], 256, vec![A::P(m.w[0]), A::P(m.w[1]), A::P(c.ids), A::P(out), A::U(m.cols), A::F(scale)]);
    let mut ops = vec![embed(w.mat("embed"), c.x[0], super::decode::bf16((h as f32).sqrt()))];
    if pl > 0 {
        ops.extend([embed(hw.mat("embed_per_layer"), ple, (pl as f32).sqrt()), norm([c.x[0], 0, 0, 0, 0], c.q, h, n, [eps, 1.0])]);
        ops.extend(gemm(w.mat("per_layer_proj"), c.q, proj, n, c));
        ops.push(norm([ple, pli, proj, w.at("per_layer_norm"), 0], [0, 0], pl, n * l as u32, [eps * h as f32, std::f32::consts::FRAC_1_SQRT_2]));
    }
    ops.push(norm([c.x[0], 0, 0, 0, w.at("0.input_layernorm")], c.q, h, n, [eps, 1.0]));
    for (i, &scalar) in scalars.iter().enumerate().take(l - cfg.shared) {
        ops.extend(attention(cfg, w, b, c, i, n));
        ops.extend(mlp(cfg, w, c, [i, n as usize], scalar));
    }
    ops
}

/// Layer `i`'s attention over the chunk, as [`super::decode`]'s, the keys and values stored first.
fn attention(cfg: &Config, w: &Weights, b: &Bufs, c: &Chunk, i: usize, n: u32) -> Vec<Op> {
    let (g, at) = (usize::from(cfg.global[i]), |name: &str| w.at(&format!("{i}.{name}")));
    let (d, kv, per) = (cfg.head_dim[g], cfg.kv_heads[g] as u32, cfg.per_block(g));
    let (cap, win) = if g == 1 { (b.context, b.context) } else { (b.ring, cfg.window as u32) };
    let (freq, kc, vc, pos) = (b.control[2 + g], b.kv[i][0], b.kv[i][1], b.control[0]);
    let (heads, row, shape) = (cfg.heads as u32, ((cfg.heads + 2 * cfg.kv_heads[g]) * d) as u32, Shape::of(d, per));
    let store = [c.qkv, at("k_norm"), freq, kc, vc, pos].map(A::P).into_iter().chain([A::U(cap), A::U(heads / kv)]).collect();
    let query = [c.qkv, at("q_norm"), freq, c.attend[0], pos].map(A::P).into_iter().chain([A::U(row)]).collect();
    let flash = [c.attend[0], kc, vc, pos].map(A::P).into_iter().chain([cap, win, cfg.group(g), n].map(A::U)).chain([A::P(c.y)]).collect();
    let len = n * heads * d as u32;
    let quant = vec![A::P(c.y), A::P(c.q[0]), A::P(c.q[1]), A::U(len)];
    let (x0, x1) = super::decode::residual(cfg, c.x, i);
    let mut ops = gemm(w.mat(&format!("{i}.qkv")), c.q, c.qkv, n, c);
    ops.push((if d == 256 { KV_256 } else { KV_512 }, [kv, n, 1], 32, store));
    ops.push((if d == 256 { QUERY_256 } else { QUERY_512 }, [heads, n, 1], 32, query));
    ops.push((shape.name(), [n.div_ceil(shape.tokens()), heads / per, 1], shape.threads(), flash));
    ops.push((QUANT_Q8P, [len.div_ceil(256), 1, 1], 256, quant));
    ops.extend(gemm(w.mat(&format!("{i}.o")), c.q, c.y, n, c));
    ops.push(norm([x0, x1, c.y, at("post_attention_layernorm"), at("pre_feedforward_layernorm")], c.q, cfg.hidden, n, [cfg.eps, 1.0]));
    ops
}

/// Layer `i`'s MLP and per-layer input over the chunk of `n` tokens, as [`super::decode`]'s, times `scalar`.
fn mlp(cfg: &Config, w: &Weights, c: &Chunk, [i, n]: [usize; 2], scalar: f32) -> Vec<Op> {
    let n = n as u32;
    let (h, eps, at) = (cfg.hidden, cfg.eps, |name: &str| w.at(&format!("{i}.{name}")));
    let ((x0, x1), (ffn, pl)) = (super::decode::residual(cfg, c.x, i), (cfg.mlp(i), cfg.per_layer));
    let next = if i + 1 < cfg.layers() { w.at(&format!("{}.input_layernorm", i + 1)) } else { w.at("norm") };
    let mut ops = gemm(w.mat(&format!("{i}.gate_up")), c.q, c.y, n, c);
    ops.push(geglu([c.y, c.y + 4 * ffn as u64], c.q, n, ffn, [2 * ffn as u32, 2 * ffn as u32, w.mat(&format!("{i}.down")).cols]));
    if cfg.moe() {
        let [zero, t, sum, dense, ymoe, ..] = c.moe;
        ops.extend(gemm(w.mat(&format!("{i}.down")), c.q, dense, n, c));
        ops.extend(experts(cfg, w, c, [i, n as usize]));
        ops.extend([
            norm([zero, t, dense, at("post_feedforward_layernorm_1"), 0], [0, 0], h, n, [eps, 1.0]),
            norm([t, sum, ymoe, at("post_feedforward_layernorm_2"), 0], [0, 0], h, n, [eps, 1.0]),
            norm([x1, x0, sum, at("post_feedforward_layernorm"), next], c.q, h, n, [eps, scalar]),
        ]);
        return ops;
    }
    ops.extend(gemm(w.mat(&format!("{i}.down")), c.q, c.y, n, c));
    if pl == 0 {
        ops.push(norm([x1, x0, c.y, at("post_feedforward_layernorm"), next], c.q, h, n, [eps, scalar]));
        return ops;
    }
    ops.push(norm([x1, x0, c.y, at("post_feedforward_layernorm"), 0], c.q, h, n, [eps, 1.0]));
    ops.extend(gemm(w.mat(&format!("{i}.per_layer_gate")), c.q, c.y, n, c));
    ops.push(geglu([c.y, c.per_layer[1] + 4 * (i * pl) as u64], c.q, n, pl, [pl as u32, (cfg.layers() * pl) as u32, pl as u32]));
    ops.extend(gemm(w.mat(&format!("{i}.per_layer_proj")), c.q, c.y, n, c));
    ops.push(norm([x0, x1, c.y, at("post_per_layer_input_norm"), next], c.q, h, n, [eps, scalar]));
    ops
}
