//! The encoder's kernels: ModernBERT and Laya's decision head in `f16` with `f32` sums, over sequences packed back
//! to back, without padding. Activations between kernels are `f32` rows (the residual stream, attention's input)
//! or `f16` rows (what a matrix product reads).
//!
//! - [`GEMM_F16`]: `y = a · wᵀ (+ bias)`, `a` `f16 [tokens, cols]`, `w` `f16 [rows, cols]`, on tensor cores
//!   (`mma.sync` `m16n8k16`, `f32` sums). `mode` 0 stores `f32 [tokens, rows]`, 1 adds into it (the residual), 2
//!   stores `f16`, 3 stores `f16` after a ReLU, 4 stores `gelu(y[2j])·y[2j+1]` as `f16 [tokens, rows/2]` (a gated
//!   GELU whose two halves' rows are interleaved). Arguments: `a, w, bias (or 0), y: u64`, `rows, cols, tokens,
//!   mode: u32`. `rows` is a multiple of [`GEMM_F16_TILE`], `cols` of 64; `a` holds whole blocks of tokens. Blocks
//!   of 64 tokens × 64 rows: launch `tokens.div_ceil(64) · rows / 64` blocks of 128 threads. [`GEMM_F16_128`] is
//!   the same in blocks of 128 tokens (256 threads), reading fewer bytes per operation: for batches that give it
//!   enough blocks.
//! - [`NORM_F16`]: layer norm of each row of `x` (`f32 [n, h]`): `γ ⊙ (x − mean) / √(var + eps) (+ β)` to `a`
//!   (`f16`), and to `xo` (`f32`, may be `x`) unless 0. Arguments: `x, γ, β (or 0), a, xo (or 0): u64`, `h: u32,
//!   eps: f32`. `h` is a multiple of 4, at most 4096. Launch `n` blocks of `(h/4).div_ceil(32)·32` threads.
//! - [`EMBED_F16`], [`ADD_ROWS`], [`GATHER_ROWS`]: row `r` of `out` (`f32 [n, h]`) becomes row `idx[r]` of
//!   `table`: converted from `f16` (a token embedding, read from pinned host memory), added (`f32`, a type
//!   embedding), or copied (`f32`, rows picked out). Arguments: `table, idx: *const u32, out: u64`, `h: u32`.
//!   Launch a grid of `(h.div_ceil(256), n)` blocks of 256.
//! - [`GELU_F16`]: `gelu(a)·b` to `f16 [n, width]`, GELU exact (through erf, to 1.5e-7), from `x`: with `gated`,
//!   rows of `[a, b]` (`f32 [n, 2·width]`); otherwise rows of `a` and `b = 1`. Arguments: `x, out: u64`, `width,
//!   len, gated: u32` (`len = n · width`). Launch `len.div_ceil(256)` blocks of 256.
//! - [`ROPE_F16`]: rotates the queries and keys of `qkv` (`f16 [n, 3h]`, rows `[q, k, v]`, heads of `d`) in place
//!   by each token's position `pos[t]`, the two halves of a head as pairs, with `table` (`f32 [positions, d/2, 2]`,
//!   cosine and sine). Arguments: `qkv, table, pos: u64`, `h, d: u32`. Launch a grid of `((h/2).div_ceil(256), n)`
//!   blocks of 256.
//! - [`ATTEND_F16`] (heads of 64) and [`ATTEND_F16_32`] (heads of 32): bidirectional attention of each sequence's
//!   tokens over its own, from `qkv` (`f16 [n, 3h]`) to `out` (`f16 [n, h]`), each query seeing the keys within
//!   `window` of it, or all when `window` is 0, on tensor cores. Tile `i` of `tiles` (`u32 [·, 3]`) is the
//!   sequence's first row, its length, and the first of the tile's up to 64 queries. `scale` is `log2(e)/√d`.
//!   Arguments: `qkv, out, tiles: u64`, `h, window: u32, scale: f32`. Launch a grid of `(tiles, h / d)` blocks of
//!   128.

use std::fmt::Write;

/// Name of the `f16` matrix product.
pub const GEMM_F16: &str = "gemm_f16";
/// Name of the `f16` matrix product in blocks of 128 tokens, 256 threads.
pub const GEMM_F16_128: &str = "gemm_f16_128";
/// Rows of weights a [`GEMM_F16`] block takes; batches of tokens are padded to a multiple of it.
pub const GEMM_F16_TILE: u32 = 64;
/// Name of the layer norm.
pub const NORM_F16: &str = "norm_f16";
/// Name of the token embedding lookup.
pub const EMBED_F16: &str = "embed_f16";
/// Name of the row-wise add of a table's rows.
pub const ADD_ROWS: &str = "add_rows";
/// Name of the row gather.
pub const GATHER_ROWS: &str = "gather_rows";
/// Name of the GELU, gated or not.
pub const GELU_F16: &str = "gelu_f16";
/// Name of the rotary embedding.
pub const ROPE_F16: &str = "rope_f16";
/// Name of the encoder attention, heads of 64.
pub const ATTEND_F16: &str = "attend_f16";
/// Name of the encoder attention, heads of 32.
pub const ATTEND_F16_32: &str = "attend_f16_32";

/// `v` as a PTX `f32` literal.
fn f(v: f32) -> String {
    format!("0f{:08X}", v.to_bits())
}

/// Bytes of a shared row of 32 `f16` (a `k` slice), padded so `ldmatrix` is free of bank conflicts.
const ROW: usize = 80;
/// One stage of a product taking `tokens` rows of activations: those, then 64 weight rows.
const fn stage(tokens: usize) -> usize {
    (tokens + 64) * ROW
}

/// Tiles of tokens a group of [`GEMM_F16`]'s blocks takes together.
const GROUP: usize = 8;

/// The shape of a product's block of `tokens` tokens: its warps (two across the 64 rows, the rest down the tokens,
/// 32 each), each warp's tokens, and the rows a pass of copies covers (four threads to a row).
const fn block(tokens: usize) -> (usize, usize, usize) {
    (tokens / 16, 32, tokens / 2)
}

/// Stages of a product's pipeline for blocks of `tokens`: four, or three where four would pass 48 KiB.
const fn stages(tokens: usize) -> usize {
    if tokens == 128 { 3 } else { 4 }
}

/// Queues stage `buf`'s copy of the slice `off` bytes past the current, under `guard`, for blocks of `tokens`.
fn copy(s: &mut String, tokens: usize, (buf, off): (usize, usize), guard: &str) {
    let (base, pass) = (buf * stage(tokens), block(tokens).2);
    let mut parts = Vec::new();
    for (dst, n, src) in [("%sa", tokens, ["%ga0", "%ga1"]), ("%sw", 64, ["%gw0", "%gw1"])] {
        parts.extend((0..n / pass).map(|p| (dst, p * pass * ROW, src[p])));
    }
    for (dst, at, src) in parts {
        let _ = writeln!(s, "    {guard}cp.async.cg.shared.global [{dst}+{}], [{src}+{off}], 16;", base + at);
    }
    s.push_str("    cp.async.commit_group;\n");
}

/// The products of stage `buf`: per 16 of `k`, an `ldmatrix` of activations per row tile of 16 tokens (two for
/// blocks of 64 tokens, one for 32), two of weights (four tiles of 8 rows), and an `mma` per pair of tiles.
fn products(s: &mut String, tokens: usize, buf: usize) {
    let base = buf * stage(tokens);
    for ks in 0..2 {
        for mi in 0..block(tokens).1 / 16 {
            let a = 4 * mi;
            let _ = writeln!(
                s,
                "    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%a{a}, %a{}, %a{}, %a{}}}, [%fa+{}];",
                a + 1,
                a + 2,
                a + 3,
                base + mi * 16 * ROW + 32 * ks
            );
        }
        for pair in 0..2 {
            let b = 4 * pair;
            let _ = writeln!(
                s,
                "    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%b{b}, %b{}, %b{}, %b{}}}, [%fb+{}];",
                b + 1,
                b + 2,
                b + 3,
                base + pair * 16 * ROW + 32 * ks
            );
        }
        for mi in 0..block(tokens).1 / 16 {
            for nt in 0..4 {
                let (c, a, b) = ((mi * 4 + nt) * 4, 4 * mi, 2 * nt);
                let _ = writeln!(
                    s,
                    "    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%c{c}, %c{}, %c{}, %c{}}}, {{%a{a}, %a{}, %a{}, %a{}}}, {{%b{b}, %b{}}}, {{%c{c}, %c{}, %c{}, %c{}}};",
                    c + 1,
                    c + 2,
                    c + 3,
                    a + 1,
                    a + 2,
                    a + 3,
                    b + 1,
                    c + 1,
                    c + 2,
                    c + 3
                );
            }
        }
    }
}

/// The block's tiles, copies' sources and destinations, and fragments' addresses. Blocks go through the tiles
/// [`GROUP`] tiles of tokens at a time, each group across every tile of rows, so the activations and the weights a
/// wave of blocks reads stay in L2. `%r6` the block's first token, `%r7` its first row; `%r12` the warp's first
/// token, `%r13` its first row.
const GEMM_HEAD: &str = "
    mov.u32 %r0, %tid.x;
    and.b32 %r1, %r0, 31;
    shr.u32 %r2, %r0, 5;
    ld.param.u32 %r3, [prows];
    ld.param.u32 %r4, [pcols];
    ld.param.u32 %r5, [ptok];
    ld.param.u32 %mode, [pmode];
    add.u32 %r6, %r5, TMASK;
    shr.u32 %r6, %r6, TSHIFT;
    shr.u32 %r7, %r3, 6;
    mul.lo.u32 %r8, %r7, GROUP;
    mov.u32 %r9, %ctaid.x;
    div.u32 %r10, %r9, %r8;
    mul.lo.u32 %r11, %r10, %r8;
    sub.u32 %r9, %r9, %r11;
    mul.lo.u32 %r10, %r10, GROUP;
    sub.u32 %r11, %r6, %r10;
    min.u32 %r11, %r11, GROUP;
    rem.u32 %r6, %r9, %r11;
    add.u32 %r6, %r6, %r10;
    div.u32 %r7, %r9, %r11;
    shl.b32 %r6, %r6, TSHIFT;
    shl.b32 %r7, %r7, 6;
    shr.u32 %r8, %r0, 2;
    and.b32 %r9, %r0, 3;
    shl.b32 %r9, %r9, 3;
    cvt.u64.u32 %rd1, %r9;
    add.u32 %r10, %r8, %r6;
    mul.wide.u32 %rd0, %r10, %r4;
    add.u64 %rd0, %rd0, %rd1;
    shl.b64 %rd0, %rd0, 1;
    ld.param.u64 %rd2, [pa];
    cvta.to.global.u64 %rd2, %rd2;
    add.u64 %ga0, %rd2, %rd0;
    mul.wide.u32 %rd3, %r4, PSTRIDE;
    add.u64 %ga1, %ga0, %rd3;
    add.u32 %r10, %r8, %r7;
    mul.wide.u32 %rd0, %r10, %r4;
    add.u64 %rd0, %rd0, %rd1;
    shl.b64 %rd0, %rd0, 1;
    ld.param.u64 %rd2, [pw];
    cvta.to.global.u64 %rd2, %rd2;
    add.u64 %gw0, %rd2, %rd0;
    add.u64 %gw1, %gw0, %rd3;
    mov.u32 %sm, smem;
    mad.lo.u32 %sa, %r8, ROW, %sm;
    shl.b32 %r9, %r9, 1;
    add.u32 %sa, %sa, %r9;
    add.u32 %sw, %sa, HALF;
    shr.u32 %r10, %r2, 1;
    shl.b32 %r12, %r10, WSHIFT;
    and.b32 %r11, %r1, 15;
    add.u32 %r11, %r11, %r12;
    mad.lo.u32 %fa, %r11, ROW, %sm;
    shr.u32 %r11, %r1, 4;
    mad.lo.u32 %fa, %r11, 16, %fa;
    and.b32 %r10, %r2, 1;
    shl.b32 %r13, %r10, 5;
    and.b32 %r11, %r1, 7;
    add.u32 %r11, %r11, %r13;
    shr.u32 %r10, %r1, 4;
    mad.lo.u32 %r11, %r10, 8, %r11;
    mad.lo.u32 %fb, %r11, ROW, %sm;
    add.u32 %fb, %fb, HALF;
    shr.u32 %r10, %r1, 3;
    and.b32 %r10, %r10, 1;
    mad.lo.u32 %fb, %r10, 16, %fb;
    add.u32 %r12, %r12, %r6;
    add.u32 %r13, %r13, %r7;
    shr.u32 %turns, %r4, 5;
";

/// The store's pointers and predicates: the bias, the mode.
const STORE_HEAD: &str = "    ld.param.u64 %rd4, [py];
    cvta.to.global.u64 %rd4, %rd4;
    ld.param.u64 %rd5, [pb];
    setp.ne.u64 %pb, %rd5, 0;
    cvta.to.global.u64 %rd5, %rd5;
    setp.eq.u32 %m1, %mode, 1;
    setp.ge.u32 %m2, %mode, 2;
    setp.lt.u32 %n2, %mode, 2;
    setp.eq.u32 %m3, %mode, 3;
    setp.eq.u32 %m4, %mode, 4;
    and.pred %m2, %m2, !%m4;
    shr.u32 %r10, %r1, 2;
    and.b32 %r11, %r1, 3;
    mad.lo.u32 %r13, %r11, 2, %r13;
    add.u32 %r12, %r12, %r10;
    mul.wide.u32 %rd6, %r13, 4;
    add.u64 %rd6, %rd5, %rd6;\n";

/// Stores accumulators `c`, `c + 1` (rows `8nt + 2t`, `+1` of the warp's) as the mode asks.
fn store_pair(s: &mut String, c: usize, nt: usize) {
    let _ = writeln!(s, "    add.f32 %v0, %c{c}, %bs{};\n    add.f32 %v1, %c{}, %bs{};", 2 * nt, c + 1, 2 * nt + 1);
    s.push_str("    @%m3 max.f32 %v0, %v0, 0f00000000;\n    @%m3 max.f32 %v1, %v1, 0f00000000;\n");
    let _ = writeln!(s, "    @%sadd ld.global.v2.f32 {{%o0, %o1}}, [%rd6+{}];", 32 * nt);
    s.push_str("    @%sadd add.f32 %v0, %v0, %o0;\n    @%sadd add.f32 %v1, %v1, %o1;\n");
    let _ = writeln!(s, "    @%s32 st.global.v2.f32 [%rd6+{}], {{%v0, %v1}};", 32 * nt);
    let _ = writeln!(s, "    cvt.rn.f16x2.f32 %h, %v1, %v0;\n    @%s16 st.global.b32 [%rd7+{}], %h;", 16 * nt);
    let _ =
        writeln!(s, "    mov.f32 %x, %v0;\n{}    mul.f32 %x, %x, %v1;\n    cvt.rn.f16.f32 %hg, %x;\n    @%s4 st.global.b16 [%rd2+{}], %hg;", gelu(), 8 * nt);
}

/// Stores the warp's tiles as `mode` asks: tokens `%r12 + 16mi + 8j + g`, rows `%r13 + 8nt + 2t, +1`; for mode 4
/// the pair's `gelu(even)·odd` as one `f16`, column `(%r13 + 8nt + 2t) / 2` of a row half as wide.
fn gemm_store(s: &mut String, tokens: usize) {
    s.push_str(STORE_HEAD);
    for nt in 0..4 {
        let b = 2 * nt;
        let _ = writeln!(
            s,
            "    mov.f32 %bs{b}, 0f00000000;\n    mov.f32 %bs{}, 0f00000000;\n    @%pb ld.global.v2.f32 {{%bs{b}, %bs{}}}, [%rd6+{}];",
            b + 1,
            b + 1,
            32 * nt
        );
    }
    for mi in 0..block(tokens).1 / 16 {
        for j in 0..2 {
            let _ = writeln!(s, "    add.u32 %r14, %r12, {};\n    setp.lt.u32 %pt, %r14, %r5;", 16 * mi + 8 * j);
            s.push_str("    and.pred %s32, %pt, %n2;\n    and.pred %sadd, %pt, %m1;\n    and.pred %s16, %pt, %m2;\n");
            s.push_str("    mad.lo.u32 %r14, %r14, %r3, %r13;\n    mul.wide.u32 %rd6, %r14, 4;\n    add.u64 %rd6, %rd4, %rd6;\n");
            s.push_str("    mul.wide.u32 %rd7, %r14, 2;\n    add.u64 %rd7, %rd4, %rd7;\n    and.pred %s4, %pt, %m4;\n    cvt.u64.u32 %rd2, %r14;\n    add.u64 %rd2, %rd4, %rd2;\n");
            for nt in 0..4 {
                store_pair(s, (mi * 4 + nt) * 4 + 2 * j, nt);
            }
        }
    }
}

/// [`GEMM_F16`] (blocks of 64 tokens) or [`GEMM_F16_128`]: [`stages`] of 32 of `k`, the next
/// slices' copies in flight while one multiplies, and one barrier per slice: the one that lets a slice in, after which
/// the stage multiplied before it is free to refill. Each warp takes 32 tokens × 32 rows.
pub(super) fn gemm_f16(name: &str, tokens: usize) -> String {
    let mut s = format!(
        ".visible .entry {name}(.param .u64 pa, .param .u64 pw, .param .u64 pb, .param .u64 py, .param .u32 prows, .param .u32 pcols, .param .u32 ptok, .param .u32 pmode)
    .maxntid {}, 1, 1
{{
    .reg .pred %p, %q, %pb, %m1, %m2, %n2, %m3, %m4, %pt, %s32, %sadd, %s16, %s4, %neg;
    .reg .b16 %hg;
    .reg .b32 %r<16>, %a<8>, %b<8>, %h, %sm, %sa, %sw, %fa, %fb, %turns, %mode;
    .reg .f32 %c<32>, %v<2>, %o<2>, %bs<8>, %x, %u, %w, %z;
    .reg .b64 %rd<8>, %ga<2>, %gw<2>;
    .shared .align 16 .b8 smem[{}];",
        32 * block(tokens).0,
        stages(tokens) * stage(tokens)
    );
    let (_, warp, pass) = block(tokens);
    s += &GEMM_HEAD
        .replace("HALF", &(tokens * ROW).to_string())
        .replace("ROW", &ROW.to_string())
        .replace("TSHIFT", &tokens.trailing_zeros().to_string())
        .replace("TMASK", &(tokens - 1).to_string())
        .replace("GROUP", &GROUP.to_string())
        .replace("PSTRIDE", &(2 * pass).to_string())
        .replace("WSHIFT", &warp.trailing_zeros().to_string());
    (0..warp).for_each(|c| _ = writeln!(s, "    mov.f32 %c{c}, 0f00000000;"));
    let n = stages(tokens);
    for p in 0..n - 1 {
        let _ = writeln!(s, "    setp.gt.u32 %q, %turns, {p};");
        copy(&mut s, tokens, (p, 64 * p), "@%q ");
    }
    s += "TURN:\n";
    for buf in 0..n {
        let _ = writeln!(s, "    cp.async.wait_group {};\n    bar.sync 0;\n    setp.gt.u32 %q, %turns, {};", n - 2, n - 1);
        copy(&mut s, tokens, ((buf + n - 1) % n, 64 * (n - 1)), "@%q ");
        products(&mut s, tokens, buf);
        for p in ["%ga0", "%ga1", "%gw0", "%gw1"] {
            let _ = writeln!(s, "    add.u64 {p}, {p}, 64;");
        }
        s += "    sub.u32 %turns, %turns, 1;\n    setp.eq.u32 %p, %turns, 0;\n    @%p bra STORE;\n";
    }
    s += "    bra TURN;\nSTORE:\n";
    gemm_store(&mut s, tokens);
    s + "    ret;\n}\n"
}

/// [`NORM_F16`]: a block per row, four values per thread; the sums of the values less the row's first and of
/// their squares in one pass, so a large mean costs no precision; then the row out.
pub(super) const NORM: &str = "
.visible .entry norm_f16(.param .u64 px, .param .u64 pg, .param .u64 pb, .param .u64 pa, .param .u64 pxo, .param .u32 ph, .param .f32 peps)
{
    .reg .pred %p, %ok, %hb, %ho;
    .reg .b32 %t, %i, %hh, %w, %nw, %k, %h0, %h1;
    .reg .f32 %x<4>, %d<4>, %g<4>, %b<4>, %sh, %s1, %s2, %u1, %u2, %mean, %r, %eps, %nh;
    .reg .b64 %row, %o, %ad, %rb;
    .shared .align 8 .f32 red[64];
    mov.u32 %t, %tid.x;
    shl.b32 %i, %t, 2;
    ld.param.u32 %hh, [ph];
    setp.lt.u32 %ok, %i, %hh;
    ld.param.f32 %eps, [peps];
    cvt.rn.f32.u32 %nh, %hh;
    mov.u32 %k, %ctaid.x;
    mul.wide.u32 %o, %k, %hh;
    ld.param.u64 %row, [px];
    cvta.to.global.u64 %row, %row;
    shl.b64 %rb, %o, 2;
    add.u64 %row, %row, %rb;
    ld.global.f32 %sh, [%row];
    mov.f32 %x0, %sh;\n    mov.f32 %x1, %sh;\n    mov.f32 %x2, %sh;\n    mov.f32 %x3, %sh;
    mul.wide.u32 %ad, %i, 4;
    add.u64 %ad, %row, %ad;
    @%ok ld.global.v4.f32 {%x0, %x1, %x2, %x3}, [%ad];
    sub.f32 %d0, %x0, %sh;\n    sub.f32 %d1, %x1, %sh;\n    sub.f32 %d2, %x2, %sh;\n    sub.f32 %d3, %x3, %sh;
    add.f32 %s1, %d0, %d1;\n    add.f32 %u1, %d2, %d3;\n    add.f32 %s1, %s1, %u1;
    mul.f32 %s2, %d0, %d0;\n    fma.rn.f32 %s2, %d1, %d1, %s2;\n    fma.rn.f32 %s2, %d2, %d2, %s2;\n    fma.rn.f32 %s2, %d3, %d3, %s2;
SUMS
    and.b32 %k, %t, 31;
    shr.u32 %w, %t, 5;
    setp.eq.u32 %p, %k, 0;
    mov.u32 %h0, red;
    mad.lo.u32 %h1, %w, 8, %h0;
    @%p st.shared.v2.f32 [%h1], {%s1, %s2};
    bar.sync 0;
    mov.u32 %nw, %ntid.x;
    shr.u32 %nw, %nw, 5;
    mov.f32 %s1, 0f00000000;
    mov.f32 %s2, 0f00000000;
WARPS:
    ld.shared.v2.f32 {%u1, %u2}, [%h0];
    add.f32 %s1, %s1, %u1;
    add.f32 %s2, %s2, %u2;
    add.u32 %h0, %h0, 8;
    sub.u32 %nw, %nw, 1;
    setp.ne.u32 %p, %nw, 0;
    @%p bra WARPS;
    @!%ok bra EXIT;
    div.rn.f32 %s1, %s1, %nh;
    div.rn.f32 %s2, %s2, %nh;
    mul.f32 %u1, %s1, %s1;
    sub.f32 %s2, %s2, %u1;
    max.f32 %s2, %s2, 0f00000000;
    add.f32 %mean, %sh, %s1;
    add.f32 %s2, %s2, %eps;
    sqrt.rn.f32 %s2, %s2;
    rcp.rn.f32 %r, %s2;
    mul.wide.u32 %ad, %i, 4;
    ld.param.u64 %o, [pg];
    cvta.to.global.u64 %o, %o;
    add.u64 %o, %o, %ad;
    ld.global.v4.f32 {%g0, %g1, %g2, %g3}, [%o];
    mov.f32 %b0, 0f00000000;\n    mov.f32 %b1, 0f00000000;\n    mov.f32 %b2, 0f00000000;\n    mov.f32 %b3, 0f00000000;
    ld.param.u64 %o, [pb];
    setp.ne.u64 %hb, %o, 0;
    cvta.to.global.u64 %o, %o;
    add.u64 %o, %o, %ad;
    @%hb ld.global.v4.f32 {%b0, %b1, %b2, %b3}, [%o];
OUT
    ld.param.u64 %o, [pxo];
    setp.ne.u64 %ho, %o, 0;
    cvta.to.global.u64 %o, %o;
    add.u64 %ad, %ad, %rb;
    add.u64 %o, %o, %ad;
    @%ho st.global.v4.f32 [%o], {%x0, %x1, %x2, %x3};
    cvt.rn.f16x2.f32 %h0, %x1, %x0;
    cvt.rn.f16x2.f32 %h1, %x3, %x2;
    ld.param.u64 %o, [pa];
    cvta.to.global.u64 %o, %o;
    shr.u64 %ad, %ad, 1;
    add.u64 %o, %o, %ad;
    st.global.v2.b32 [%o], {%h0, %h1};
EXIT:
    ret;
}
";

/// [`NORM_F16`]'s PTX.
pub(super) fn norm_f16() -> String {
    let sums = (0..5).fold(String::new(), |mut s, k| {
        for v in ["%s1", "%s2"] {
            let _ = writeln!(s, "    shfl.sync.bfly.b32 %u1, {v}, {}, 31, -1;\n    add.f32 {v}, {v}, %u1;", 16 >> k);
        }
        s
    });
    let out = (0..4).fold(String::new(), |mut s, e| {
        let _ = writeln!(s, "    sub.f32 %x{e}, %x{e}, %mean;\n    mul.f32 %x{e}, %x{e}, %r;\n    fma.rn.f32 %x{e}, %x{e}, %g{e}, %b{e};");
        s
    });
    NORM.replace("SUMS\n", &sums).replace("OUT\n", &out)
}

/// [`EMBED_F16`] (`half`), [`ADD_ROWS`] (`add`) or [`GATHER_ROWS`]: one value per thread.
pub(super) fn rows(name: &str, half: bool, add: bool) -> String {
    let (size, load) = if half { (2, "    ld.global.b16 %hx, [%src];\n    cvt.f32.f16 %x, %hx;\n") } else { (4, "    ld.global.f32 %x, [%src];\n") };
    let combine = if add { "    ld.global.f32 %y, [%dst];\n    add.f32 %x, %x, %y;\n" } else { "" };
    format!(
        ".visible .entry {name}(.param .u64 ptab, .param .u64 pidx, .param .u64 pout, .param .u32 ph)
{{
    .reg .pred %p;
    .reg .b16 %hx;
    .reg .b32 %i, %n, %t, %row, %hh, %id;
    .reg .f32 %x, %y;
    .reg .b64 %src, %dst, %o, %idx;
    mov.u32 %i, %ctaid.x;
    mov.u32 %n, %ntid.x;
    mov.u32 %t, %tid.x;
    mad.lo.u32 %i, %i, %n, %t;
    ld.param.u32 %hh, [ph];
    setp.ge.u32 %p, %i, %hh;
    @%p bra EXIT;
    mov.u32 %row, %ctaid.y;
    ld.param.u64 %idx, [pidx];
    cvta.to.global.u64 %idx, %idx;
    mul.wide.u32 %o, %row, 4;
    add.u64 %idx, %idx, %o;
    ld.global.u32 %id, [%idx];
    mul.wide.u32 %o, %id, %hh;
    cvt.u64.u32 %src, %i;
    add.u64 %o, %o, %src;
    mul.lo.u64 %o, %o, {size};
    ld.param.u64 %src, [ptab];
    cvta.to.global.u64 %src, %src;
    add.u64 %src, %src, %o;
    mul.wide.u32 %o, %row, %hh;
    cvt.u64.u32 %dst, %i;
    add.u64 %o, %o, %dst;
    shl.b64 %o, %o, 2;
    ld.param.u64 %dst, [pout];
    cvta.to.global.u64 %dst, %dst;
    add.u64 %dst, %dst, %o;
{load}{combine}    st.global.f32 [%dst], %x;
EXIT:
    ret;
}}
"
    )
}

/// `%x = gelu(%x)` through erf to 1.5e-7 (Abramowitz and Stegun 7.1.26); `%u`, `%w`, `%z` are scratch.
fn gelu() -> String {
    let p = [0.254_829_6, -0.284_496_74, 1.421_413_7, -1.453_152_1, 1.061_405_4].map(f);
    format!(
        "    mul.f32 %z, %x, {};
    abs.f32 %u, %z;
    fma.rn.f32 %u, %u, {}, 0f3F800000;
    rcp.rn.f32 %u, %u;
    mul.f32 %w, %u, {};
    add.f32 %w, %w, {};
    fma.rn.f32 %w, %w, %u, {};
    fma.rn.f32 %w, %w, %u, {};
    fma.rn.f32 %w, %w, %u, {};
    mul.f32 %w, %w, %u;
    mul.f32 %u, %z, %z;
    mul.f32 %u, %u, {};
    ex2.approx.f32 %u, %u;
    mul.f32 %w, %w, %u;
    sub.f32 %w, 0f3F800000, %w;
    setp.lt.f32 %neg, %z, 0f00000000;
    @%neg neg.f32 %w, %w;
    add.f32 %w, %w, 0f3F800000;
    mul.f32 %w, %w, %x;
    mul.f32 %x, %w, 0f3F000000;
",
        f(std::f32::consts::FRAC_1_SQRT_2),
        f(0.327_591_1),
        p[4],
        p[3],
        p[2],
        p[1],
        p[0],
        f(-std::f32::consts::LOG2_E),
    )
}

/// [`GELU_F16`].
pub(super) fn gelu_f16() -> String {
    format!(
        ".visible .entry gelu_f16(.param .u64 px, .param .u64 pout, .param .u32 pwidth, .param .u32 plen, .param .u32 pgated)
{{
    .reg .pred %p, %gated, %neg;
    .reg .b16 %hx;
    .reg .b32 %i, %n, %t, %wd, %len, %row, %col, %stride;
    .reg .f32 %x, %y, %u, %z, %w;
    .reg .b64 %a, %o;
    mov.u32 %i, %ctaid.x;
    mov.u32 %n, %ntid.x;
    mov.u32 %t, %tid.x;
    mad.lo.u32 %i, %i, %n, %t;
    ld.param.u32 %len, [plen];
    setp.ge.u32 %p, %i, %len;
    @%p bra EXIT;
    ld.param.u32 %wd, [pwidth];
    ld.param.u32 %t, [pgated];
    setp.ne.u32 %gated, %t, 0;
    div.u32 %row, %i, %wd;
    rem.u32 %col, %i, %wd;
    selp.u32 %stride, 2, 1, %gated;
    mul.lo.u32 %stride, %stride, %wd;
    mad.lo.u32 %t, %row, %stride, %col;
    mul.wide.u32 %o, %t, 4;
    ld.param.u64 %a, [px];
    cvta.to.global.u64 %a, %a;
    add.u64 %a, %a, %o;
    ld.global.f32 %x, [%a];
    mov.f32 %y, 0f3F800000;
    mul.wide.u32 %o, %wd, 4;
    add.u64 %o, %a, %o;
    @%gated ld.global.f32 %y, [%o];
{}    mul.f32 %x, %x, %y;
    cvt.rn.f16.f32 %hx, %x;
    mul.wide.u32 %o, %i, 2;
    ld.param.u64 %a, [pout];
    cvta.to.global.u64 %a, %a;
    add.u64 %a, %a, %o;
    st.global.b16 [%a], %hx;
EXIT:
    ret;
}}
",
        gelu()
    )
}

/// [`ROPE_F16`]: one rotation pair of a query and a key per thread, a token per block row.
pub(super) const ROPE: &str = "
.visible .entry rope_f16(.param .u64 pqkv, .param .u64 ptab, .param .u64 ppos, .param .u32 ph, .param .u32 pd)
{
    .reg .pred %p;
    .reg .b16 %ha, %hb;
    .reg .b32 %i, %t, %hh, %d, %half, %tok, %head, %j, %pos, %e;
    .reg .f32 %c, %s, %a, %b, %x, %y;
    .reg .b64 %q, %tab, %o, %w;
    mov.u32 %i, %ctaid.x;
    mov.u32 %t, %tid.x;
    mad.lo.u32 %i, %i, 256, %t;
    ld.param.u32 %hh, [ph];
    ld.param.u32 %d, [pd];
    shr.u32 %t, %hh, 1;
    setp.ge.u32 %p, %i, %t;
    @%p bra EXIT;
    mov.u32 %tok, %ctaid.y;
    shr.u32 %half, %d, 1;
    div.u32 %head, %i, %half;
    rem.u32 %j, %i, %half;
    ld.param.u64 %o, [ppos];
    cvta.to.global.u64 %o, %o;
    mul.wide.u32 %q, %tok, 4;
    add.u64 %o, %o, %q;
    ld.global.u32 %pos, [%o];
    mad.lo.u32 %e, %pos, %half, %j;
    mul.wide.u32 %o, %e, 8;
    ld.param.u64 %tab, [ptab];
    cvta.to.global.u64 %tab, %tab;
    add.u64 %tab, %tab, %o;
    ld.global.v2.f32 {%c, %s}, [%tab];
    mul.lo.u32 %e, %tok, 3;
    mul.lo.u32 %e, %e, %hh;
    mad.lo.u32 %e, %head, %d, %e;
    add.u32 %e, %e, %j;
    mul.wide.u32 %o, %e, 2;
    ld.param.u64 %q, [pqkv];
    cvta.to.global.u64 %q, %q;
    add.u64 %q, %q, %o;
    mul.wide.u32 %w, %half, 2;
TURN%q
    mul.wide.u32 %o, %hh, 2;
    add.u64 %q, %q, %o;
TURN%k
EXIT:
    ret;
}
";

/// [`ROPE_F16`]'s PTX: the pair `(a, b)` at `%q` and `%q + %w` turned by `(cos, sin)`, for the query then the key.
pub(super) fn rope_f16() -> String {
    let turn = "    add.u64 %o, %q, %w;
    ld.global.b16 %ha, [%q];
    ld.global.b16 %hb, [%o];
    cvt.f32.f16 %a, %ha;
    cvt.f32.f16 %b, %hb;
    mul.f32 %x, %a, %c;
    neg.f32 %y, %s;
    fma.rn.f32 %x, %b, %y, %x;
    mul.f32 %y, %b, %c;
    fma.rn.f32 %y, %a, %s, %y;
    cvt.rn.f16.f32 %ha, %x;
    cvt.rn.f16.f32 %hb, %y;
    st.global.b16 [%q], %ha;
    st.global.b16 [%o], %hb;\n";
    ROPE.replace("TURN%q\n", turn).replace("TURN%k\n", turn)
}

/// [`ATTEND_F16`]'s registers; its tile, the keys it sees (`%lo..%hi`), and its shared-memory addresses: `%cd` this
/// thread's copy slot, `%fq`, `%fk`, `%fv` its fragments' rows of queries, keys and values.
const ATTEND_HEAD: &str = ".visible .entry NAME(.param .u64 pqkv, .param .u64 pout, .param .u64 ptiles, .param .u32 ph, .param .u32 pwin, .param .f32 pscale)
    .maxntid 128, 1, 1
{
    .reg .pred %p, %v, %glob, %out, %full;
    .reg .b32 %t, %lane, %wp, %head, %hh, %h3, %win, %start, %len, %qoff, %lo, %hi, %kb, %kq, %qr<2>, %key, %df;
    .reg .b32 %hp<3>, %cr, %cd, %fq, %fk, %fv, %r<3>, %a<DQ>, %pa<4>, %kf<4>, %vf<4>;
    .reg .f32 %s<32>, %o<DO>, %m<2>, %l<2>, %mx, %mu, %corr, %e, %scale;
    .reg .b64 %base, %gb, %rd, %ad;
    .shared .align 16 .b8 smem[SIZE];
    mov.u32 %t, %tid.x;
    and.b32 %lane, %t, 31;
    shr.u32 %wp, %t, 5;
    mov.u32 %head, %ctaid.y;
    ld.param.u32 %hh, [ph];
    mul.lo.u32 %h3, %hh, 3;
    ld.param.u32 %win, [pwin];
    setp.eq.u32 %glob, %win, 0;
    ld.param.f32 %scale, [pscale];
    ld.param.u64 %ad, [ptiles];
    cvta.to.global.u64 %ad, %ad;
    mov.u32 %r0, %ctaid.x;
    mul.wide.u32 %rd, %r0, 12;
    add.u64 %ad, %ad, %rd;
    ld.global.u32 %start, [%ad];
    ld.global.u32 %len, [%ad+4];
    ld.global.u32 %qoff, [%ad+8];
    mov.u32 %lo, 0;
    mov.u32 %hi, %len;
    @%glob bra RANGE;
    sub.u32 %lo, %qoff, %win;
    setp.gt.u32 %p, %win, %qoff;
    @%p mov.u32 %lo, 0;
    add.u32 %hi, %qoff, 64;
    add.u32 %hi, %hi, %win;
    min.u32 %hi, %hi, %len;
RANGE:
    mul.lo.u32 %hp0, %head, DIM;
    add.u32 %hp1, %hp0, %hh;
    add.u32 %hp2, %hp1, %hh;
    shr.u32 %cr, %t, SHIFT;
    and.b32 %r0, %t, PIECES;
    shl.b32 %r0, %r0, 4;
    ld.param.u64 %base, [pqkv];
    cvta.to.global.u64 %base, %base;
    cvt.u64.u32 %gb, %r0;
    add.u64 %gb, %gb, %base;
    mov.u32 %r1, smem;
    mad.lo.u32 %cd, %cr, RS, %r1;
    add.u32 %cd, %cd, %r0;
    and.b32 %r0, %lane, 15;
    mad.lo.u32 %fv, %r0, RS, %r1;
    shr.u32 %r2, %lane, 4;
    mad.lo.u32 %fv, %r2, 16, %fv;
    shl.b32 %r0, %wp, 4;
    mad.lo.u32 %fq, %r0, RS, %fv;
    add.u32 %fv, %fv, KEYS2;
    and.b32 %r0, %lane, 7;
    mad.lo.u32 %r0, %r2, 8, %r0;
    mad.lo.u32 %fk, %r0, RS, %r1;
    shr.u32 %r0, %lane, 3;
    and.b32 %r0, %r0, 1;
    mad.lo.u32 %fk, %r0, 16, %fk;
    add.u32 %fk, %fk, KEYS;
    shr.u32 %r0, %lane, 2;
    shl.b32 %r1, %wp, 4;
    add.u32 %r0, %r0, %r1;
    add.u32 %qr0, %qoff, %r0;
    add.u32 %qr1, %qr0, 8;
    and.b32 %r0, %lane, 3;
    shl.b32 %kq, %r0, 1;
    mov.f32 %m0, 0fFF800000;
    mov.f32 %m1, 0fFF800000;
    mov.f32 %l0, 0f00000000;
    mov.f32 %l1, 0f00000000;
";

/// Queues this thread's copies of 64 rows of `f16` heads of `d` into the shared rows at `at`: row `%cr + p·step`
/// of the tile is row `first` + it of the sequence, part `hp` (queries, keys or values) of `qkv`, or zeros from
/// `limit` on.
fn attend_copy(s: &mut String, d: usize, at: usize, part: [&str; 3]) {
    let ([first, limit, hp], step, rs) = (part, 1024 / d, 2 * d + 16);
    for p in 0..d / 16 {
        let _ = writeln!(
            s,
            "    add.u32 %r0, %cr, {};\n    add.u32 %r0, %r0, {first};\n    setp.lt.u32 %v, %r0, {limit};\n    selp.u32 %r1, 16, 0, %v;
    selp.u32 %r0, %r0, 0, %v;\n    add.u32 %r0, %r0, %start;\n    mad.lo.u32 %r0, %r0, %h3, {hp};\n    mul.wide.u32 %rd, %r0, 2;
    add.u64 %rd, %rd, %gb;\n    cp.async.cg.shared.global [%cd+{}], [%rd], 16, %r1;",
            p * step,
            at + p * step * rs
        );
    }
    s.push_str("    cp.async.commit_group;\n");
}

/// The chunk's scores `S = Q·Kᵀ`: each warp's 16 queries against the 64 keys, eight tiles of 8 keys.
fn attend_scores(s: &mut String, d: usize) {
    let rs = 2 * d + 16;
    (0..32).for_each(|i| _ = writeln!(s, "    mov.f32 %s{i}, 0f00000000;"));
    for np in 0..4 {
        for kk in 0..d / 16 {
            let _ = writeln!(s, "    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%kf0, %kf1, %kf2, %kf3}}, [%fk+{}];", np * 16 * rs + kk * 32);
            for (n, b) in [(2 * np, 0), (2 * np + 1, 2)] {
                let (c, a) = (4 * n, 4 * kk);
                let _ = writeln!(
                    s,
                    "    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%s{c}, %s{}, %s{}, %s{}}}, {{%a{a}, %a{}, %a{}, %a{}}}, {{%kf{b}, %kf{}}}, {{%s{c}, %s{}, %s{}, %s{}}};",
                    c + 1,
                    c + 2,
                    c + 3,
                    a + 1,
                    a + 2,
                    a + 3,
                    b + 1,
                    c + 1,
                    c + 2,
                    c + 3
                );
            }
        }
    }
}

/// Scales the scores to base 2; outside the keys `%lo..%hi` or the window, `−∞`. A chunk all inside skips the tests.
fn attend_mask(s: &mut String) {
    s.push_str("    add.u32 %r0, %kb, 64;\n    setp.le.u32 %full, %r0, %hi;\n    and.pred %full, %full, %glob;\n    add.u32 %kq, %kq, %kb;\n");
    (0..32).for_each(|i| _ = writeln!(s, "    mul.f32 %s{i}, %s{i}, %scale;"));
    s.push_str("    @%full bra MASKED;\n");
    for i in 0..32 {
        let (n, e) = (i / 4, i % 4);
        let _ = writeln!(
            s,
            "    add.u32 %key, %kq, {};\n    setp.ge.u32 %out, %key, %hi;\n    sub.u32 %df, %qr{r}, %key;\n    setp.lt.u32 %p, %qr{r}, %key;
    @%p sub.u32 %df, %key, %qr{r};\n    setp.gt.u32 %p, %df, %win;\n    and.pred %p, %p, !%glob;\n    or.pred %out, %out, %p;\n    @%out mov.f32 %s{i}, 0fFF800000;",
            8 * n + (e & 1),
            r = e >> 1
        );
    }
    s.push_str("MASKED:\n    sub.u32 %kq, %kq, %kb;\n");
}

/// The online softmax of row `r` (`0`: the thread's first query, `1`: its second): the running maximum, the scale
/// of what came before, the probabilities and the running sum.
fn attend_softmax(s: &mut String, d: usize, r: usize) {
    let _ = writeln!(s, "    mov.f32 %mx, %m{r};");
    for n in 0..8 {
        let _ = writeln!(s, "    max.f32 %mx, %mx, %s{};\n    max.f32 %mx, %mx, %s{};", 4 * n + 2 * r, 4 * n + 2 * r + 1);
    }
    for k in [1, 2] {
        let _ = writeln!(s, "    shfl.sync.bfly.b32 %e, %mx, {k}, 31, -1;\n    max.f32 %mx, %mx, %e;");
    }
    let _ = writeln!(
        s,
        "    setp.eq.f32 %p, %mx, 0fFF800000;\n    selp.f32 %mu, 0f00000000, %mx, %p;\n    sub.f32 %corr, %m{r}, %mu;\n    ex2.approx.f32 %corr, %corr;\n    mov.f32 %m{r}, %mx;\n    mul.f32 %l{r}, %l{r}, %corr;"
    );
    for n in 0..8 {
        for j in 0..2 {
            let i = 4 * n + 2 * r + j;
            let _ = writeln!(s, "    sub.f32 %s{i}, %s{i}, %mu;\n    ex2.approx.f32 %s{i}, %s{i};\n    add.f32 %l{r}, %l{r}, %s{i};");
        }
    }
    for n in 0..d / 8 {
        let i = 4 * n + 2 * r;
        let _ = writeln!(s, "    mul.f32 %o{i}, %o{i}, %corr;\n    mul.f32 %o{}, %o{}, %corr;", i + 1, i + 1);
    }
}

/// `O += P·V`: the probabilities as `f16` fragments, sixteen keys at a time, against the values.
fn attend_values(s: &mut String, d: usize) {
    let rs = 2 * d + 16;
    for kk in 0..4 {
        for (j, n) in [(0, 2 * kk), (2, 2 * kk + 1)] {
            let c = 4 * n;
            let _ = writeln!(s, "    cvt.rn.f16x2.f32 %pa{j}, %s{}, %s{c};\n    cvt.rn.f16x2.f32 %pa{}, %s{}, %s{};", c + 1, j + 1, c + 3, c + 2);
        }
        for dp in 0..d / 16 {
            let _ = writeln!(s, "    ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {{%vf0, %vf1, %vf2, %vf3}}, [%fv+{}];", kk * 16 * rs + dp * 32);
            for (n, b) in [(2 * dp, 0), (2 * dp + 1, 2)] {
                let c = 4 * n;
                let _ = writeln!(
                    s,
                    "    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%o{c}, %o{}, %o{}, %o{}}}, {{%pa0, %pa1, %pa2, %pa3}}, {{%vf{b}, %vf{}}}, {{%o{c}, %o{}, %o{}, %o{}}};",
                    c + 1,
                    c + 2,
                    c + 3,
                    b + 1,
                    c + 1,
                    c + 2,
                    c + 3
                );
            }
        }
    }
}

/// Each valid query's values divided by its sum, as `f16`, into `out`.
fn attend_store(s: &mut String, d: usize) {
    s.push_str("    ld.param.u64 %base, [pout];\n    cvta.to.global.u64 %base, %base;\n");
    for r in 0..2 {
        let _ = writeln!(
            s,
            "    shfl.sync.bfly.b32 %e, %l{r}, 1, 31, -1;\n    add.f32 %l{r}, %l{r}, %e;\n    shfl.sync.bfly.b32 %e, %l{r}, 2, 31, -1;\n    add.f32 %l{r}, %l{r}, %e;
    setp.ge.u32 %p, %qr{r}, %len;\n    @%p bra STORED{r};\n    rcp.rn.f32 %l{r}, %l{r};\n    add.u32 %r0, %qr{r}, %start;\n    mad.lo.u32 %r0, %r0, %hh, %hp0;
    add.u32 %r0, %r0, %kq;\n    mul.wide.u32 %rd, %r0, 2;\n    add.u64 %rd, %rd, %base;"
        );
        for n in 0..d / 8 {
            let i = 4 * n + 2 * r;
            let _ = writeln!(
                s,
                "    mul.f32 %mx, %o{i}, %l{r};\n    mul.f32 %mu, %o{}, %l{r};\n    cvt.rn.f16x2.f32 %r1, %mu, %mx;\n    st.global.b32 [%rd+{}], %r1;",
                i + 1,
                16 * n
            );
        }
        let _ = writeln!(s, "STORED{r}:");
    }
}

/// [`ATTEND_F16`] for heads of `d` (32 or 64), on tensor cores: a block of 4 warps takes a tile of 64 queries of one
/// head, each warp 16; keys and values come through shared memory 64 at a time, with an online softmax between the
/// two products, as flash attention does.
pub(super) fn attend_f16(name: &str, d: usize) -> String {
    let rs = 2 * d + 16;
    let mut s = ATTEND_HEAD
        .replace("NAME", name)
        .replace("DQ", &(d / 4).to_string())
        .replace("DO", &(d / 2).to_string())
        .replace("SIZE", &(3 * 64 * rs).to_string())
        .replace("DIM", &d.to_string())
        .replace("SHIFT", &(d / 8).trailing_zeros().to_string())
        .replace("PIECES", &(d / 8 - 1).to_string())
        .replace("KEYS2", &(128 * rs).to_string())
        .replace("KEYS", &(64 * rs).to_string())
        .replace("RS", &rs.to_string());
    (0..d / 2).for_each(|i| _ = writeln!(s, "    mov.f32 %o{i}, 0f00000000;"));
    attend_copy(&mut s, d, 0, ["%qoff", "%len", "%hp0"]);
    s.push_str("    cp.async.wait_group 0;\n    bar.sync 0;\n");
    for kk in 0..d / 16 {
        let a = 4 * kk;
        let _ = writeln!(s, "    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%a{a}, %a{}, %a{}, %a{}}}, [%fq+{}];", a + 1, a + 2, a + 3, 32 * kk);
    }
    s.push_str("    mov.u32 %kb, %lo;\nCHUNK:\n    setp.ge.u32 %p, %kb, %hi;\n    @%p bra DONE;\n    bar.sync 0;\n");
    attend_copy(&mut s, d, 64 * rs, ["%kb", "%hi", "%hp1"]);
    attend_copy(&mut s, d, 128 * rs, ["%kb", "%hi", "%hp2"]);
    s.push_str("    cp.async.wait_group 0;\n    bar.sync 0;\n");
    attend_scores(&mut s, d);
    attend_mask(&mut s);
    attend_softmax(&mut s, d, 0);
    attend_softmax(&mut s, d, 1);
    attend_values(&mut s, d);
    s.push_str("    add.u32 %kb, %kb, 64;\n    bra CHUNK;\nDONE:\n");
    attend_store(&mut s, d);
    s + "    ret;\n}\n"
}
