//! Prefill: a prompt's tokens through the model many at a time, on tensor cores, filling the key-value caches.
//!
//! A chunk of up to [`super::Settings::chunk`] tokens at positions `pos..` (the step counter) runs as one
//! sequence of launches, each kernel over every token: the embeddings, then per layer the matrix products as
//! [`GEMM_Q4`] on Q8P activations, the keys and values stored by [`KV_256`] or [`KV_512`] before
//! [`ATTEND_256P`] or [`ATTEND_512P`] reads them, and the residual steps as [`NORM_Q8P`] and [`GEGLU_Q8P`].
//! Only the layers that write a cache run: the last [`super::Config::shared`] layers read earlier layers' caches
//! and add nothing a later token needs, so a prompt's tokens but its last skip them (18 of E4B's 42 layers). The
//! last token runs the decode step, which samples the first answer token. The chunk's kernels are launched
//! directly, not recorded: their count does not depend on the chunk's length, which the matrix products need.

// Buffers and dimensions keep their one-letter names from the math, as in `decode`: x, q, y, d, h, l, n.
#![allow(clippy::many_single_char_names)]

use super::Config;
use super::decode::{A, Bufs, Mat, Op, Weights};
use crate::ptx::{ATTEND_256P, ATTEND_512P, ATTEND_WARPS, EMBED_Q4, GEGLU_Q8P, GEMM_Q4, GEMM_ROWS, GEMM_TOKENS, KV_256, KV_512, NORM_Q8P, NORM_THREADS};

/// Device addresses of prefill's buffers, each for a chunk of tokens: the residual stream (two), Q8P values and
/// scales, the query-key-value projections, the other products, the per-layer inputs (embedded, projected,
/// combined), attention's scratch and counters, and the chunk's token IDs (pinned host memory).
pub(super) struct Chunk {
    pub(super) x: [u64; 2],
    pub(super) q: [u64; 2],
    pub(super) qkv: u64,
    pub(super) y: u64,
    pub(super) per_layer: [u64; 3],
    pub(super) attend: [u64; 2],
    pub(super) ids: u64,
}

/// Lays out prefill's buffers for `chunk` tokens with `take`, which returns the address of `len` new bytes.
pub(super) fn layout(cfg: &Config, chunk: usize, take: &mut impl FnMut(usize) -> u64) -> Chunk {
    let (d, kv, pl, f) = (cfg.head_dim[1], cfg.kv_heads, cfg.layers() * cfg.per_layer, 4 * chunk.next_multiple_of(GEMM_TOKENS as usize));
    let widest = cfg.hidden.max(cfg.mlp(cfg.layers() - 1)).max(cfg.heads * d).max(cfg.per_layer);
    let x = [take(f * cfg.hidden), take(f * cfg.hidden)];
    let q = [take(f / 4 * widest), take(f / 4 * widest / 8)];
    let (qkv, y) = (take(f * (cfg.heads + 2 * kv) * d), take(f * (2 * cfg.mlp(cfg.layers() - 1)).max(cfg.hidden).max(pl)));
    let per_layer = [take(f * pl), take(f * pl), take(f * pl)];
    Chunk { x, q, qkv, y, per_layer, attend: [take(f * cfg.heads / 4 * (8 + 4 * d)), take(f * cfg.heads / 4)], ids: 0 }
}

/// `y = X·Mᵀ` over `n` tokens of Q8P input `q`.
fn gemm(m: Mat, q: [u64; 2], y: u64, n: u32) -> Op {
    let args = vec![A::P(m.w[0]), A::P(m.w[1]), A::P(q[0]), A::P(q[1]), A::P(y), A::U(m.rows), A::U(m.cols), A::U(n)];
    (GEMM_Q4, [n.div_ceil(GEMM_TOKENS), m.rows / GEMM_ROWS, 1], 128, args)
}

/// [`NORM_Q8P`] on `p = [h, h_out, y, w1, w2]` over `chunks` of `len`, `k = [eps, scale]`, Q8P to `q`.
fn norm(p: [u64; 5], q: [u64; 2], len: usize, chunks: u32, k: [f32; 2]) -> Op {
    let args = p.into_iter().chain(q).map(A::P).chain([A::U(len as u32), A::F(k[0]), A::F(k[1])]).collect();
    (NORM_Q8P, [len as u32 / NORM_THREADS, chunks, 1], NORM_THREADS, args)
}

/// Q8P of `gelu(a) · b` over `n` rows of `width`, `a` and `b` rows `stride` floats apart.
fn geglu([a, b]: [u64; 2], q: [u64; 2], n: u32, width: usize, stride: [usize; 2]) -> Op {
    let len = n * width as u32;
    let args = [a, b, q[0], q[1]].map(A::P).into_iter().chain([len, width as u32, stride[0] as u32, stride[1] as u32].map(A::U)).collect();
    (GEGLU_Q8P, [len.div_ceil(256), 1, 1], 256, args)
}

/// The launches of a chunk of `n` tokens, whose IDs are at `c.ids`; `scalars` holds each layer's `layer_scalar`.
pub(super) fn ops(cfg: &Config, [w, hw]: [&Weights; 2], b: &Bufs, (c, scalars): (&Chunk, &[f32]), n: u32) -> Vec<Op> {
    let (h, pl, l, eps) = (cfg.hidden, cfg.per_layer, cfg.layers(), cfg.eps);
    let [ple, pli, proj] = c.per_layer;
    let embed =
        |m: Mat, out, scale| (EMBED_Q4, [m.cols.div_ceil(256), n, 1], 256, vec![A::P(m.w[0]), A::P(m.w[1]), A::P(c.ids), A::P(out), A::U(m.cols), A::F(scale)]);
    let mut ops = vec![
        embed(w.mat("embed"), c.x[0], super::decode::bf16((h as f32).sqrt())),
        embed(hw.mat("embed_per_layer"), ple, (pl as f32).sqrt()),
        norm([c.x[0], 0, 0, 0, 0], c.q, h, n, [eps, 1.0]),
        gemm(w.mat("per_layer_proj"), c.q, proj, n),
        norm([ple, pli, proj, w.at("per_layer_norm"), 0], [0, 0], pl, n * l as u32, [eps * h as f32, std::f32::consts::FRAC_1_SQRT_2]),
        norm([c.x[0], 0, 0, 0, w.at("0.input_layernorm")], c.q, h, n, [eps, 1.0]),
    ];
    for (i, &scalar) in scalars.iter().enumerate().take(l - cfg.shared) {
        ops.extend(attention(cfg, w, b, c, i, n));
        ops.extend(mlp(cfg, w, c, [i, n as usize], scalar));
    }
    ops
}

/// Layer `i`'s attention over the chunk, as [`super::decode`]'s, the keys and values stored first.
fn attention(cfg: &Config, w: &Weights, b: &Bufs, c: &Chunk, i: usize, n: u32) -> Vec<Op> {
    let (g, at) = (usize::from(cfg.global[i]), |name: &str| w.at(&format!("{i}.{name}")));
    let (d, kv) = (cfg.head_dim[g], cfg.kv_heads as u32);
    let (cap, win) = if g == 1 { (b.context, b.context) } else { (b.ring, cfg.window as u32) };
    let (freq, kc, vc, pos) = (b.control[2 + g], b.kv[i][0], b.kv[i][1], b.control[0]);
    let store = [c.qkv, at("k_norm"), freq, kc, vc, pos].map(A::P).into_iter().chain([A::U(cap), A::U(cfg.group())]).collect();
    let ptrs = [c.qkv, at("q_norm"), at("k_norm"), freq, kc, vc, pos].map(A::P).into_iter();
    let args = ptrs.chain([cap, win, 0, cfg.group()].map(A::U)).chain([c.attend[0], c.attend[1], c.q[0], c.q[1]].map(A::P)).collect();
    let (x0, x1) = (c.x[i % 2], c.x[(i + 1) % 2]);
    vec![
        gemm(w.mat(&format!("{i}.qkv")), c.q, c.qkv, n),
        (if d == 256 { KV_256 } else { KV_512 }, [kv, n, 1], 32, store),
        (if d == 256 { ATTEND_256P } else { ATTEND_512P }, [cfg.heads as u32 / 4, 1, n], 32 * ATTEND_WARPS, args),
        gemm(w.mat(&format!("{i}.o")), c.q, c.y, n),
        norm([x0, x1, c.y, at("post_attention_layernorm"), at("pre_feedforward_layernorm")], c.q, cfg.hidden, n, [cfg.eps, 1.0]),
    ]
}

/// Layer `i`'s MLP and per-layer input over the chunk of `n` tokens, as [`super::decode`]'s, times `scalar`.
fn mlp(cfg: &Config, w: &Weights, c: &Chunk, [i, n]: [usize; 2], scalar: f32) -> Vec<Op> {
    let n = n as u32;
    let (h, eps, at) = (cfg.hidden, cfg.eps, |name: &str| w.at(&format!("{i}.{name}")));
    let (x0, x1, (ffn, pl)) = (c.x[i % 2], c.x[(i + 1) % 2], (cfg.mlp(i), cfg.per_layer));
    let next = if i + 1 < cfg.layers() { w.at(&format!("{}.input_layernorm", i + 1)) } else { w.at("norm") };
    vec![
        gemm(w.mat(&format!("{i}.gate_up")), c.q, c.y, n),
        geglu([c.y, c.y + 4 * ffn as u64], c.q, n, ffn, [2 * ffn, 2 * ffn]),
        gemm(w.mat(&format!("{i}.down")), c.q, c.y, n),
        norm([x1, x0, c.y, at("post_feedforward_layernorm"), 0], c.q, h, n, [eps, 1.0]),
        gemm(w.mat(&format!("{i}.per_layer_gate")), c.q, c.y, n),
        geglu([c.y, c.per_layer[1] + 4 * (i * pl) as u64], c.q, n, pl, [pl, cfg.layers() * pl]),
        gemm(w.mat(&format!("{i}.per_layer_proj")), c.q, c.y, n),
        norm([x0, x1, c.y, at("post_per_layer_input_norm"), next], c.q, h, n, [eps, scalar]),
    ]
}
