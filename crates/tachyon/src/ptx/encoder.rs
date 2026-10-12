//! The encoder's kernels: ModernBERT and Laya's decision head in `f16` with `f32` sums, over sequences packed back
//! to back, without padding. Activations between kernels are `f32` rows (the residual stream, attention's input)
//! or `f16` rows (what a matrix product reads); a buffer of `f16` rows holds whole tiles of [`GEMM_F16_TILE`] rows.
//!
//! - [`GEMM_F16`]: `y = a · wᵀ (+ bias)`, `a` `f16 [tokens, cols]`, `w` `f16 [rows, cols]`, on tensor cores
//!   (`mma.sync` `m16n8k16`, `f32` sums). `mode` 0 stores `f32 [tokens, rows]`, 1 adds into it (the residual), 2
//!   stores `f16`, 3 stores `f16` after a ReLU. Arguments: `a, w, bias (or 0), y: u64`, `rows, cols, tokens,
//!   mode: u32`. `rows` and `tokens`' buffer are multiples of [`GEMM_F16_TILE`], `cols` of 64. Launch a grid of
//!   `(tokens.div_ceil(64), rows / 64)` blocks of 128 threads.
//! - [`NORM_F16`]: layer norm of each row of `x` (`f32 [n, h]`): `γ ⊙ (x − mean) / √(var + eps) (+ β)` to `a`
//!   (`f16`), and to `xo` (`f32`, may be `x`) unless 0. Arguments: `x, γ, β (or 0), a, xo (or 0): u64`, `h, n: u32,
//!   eps: f32`. Launch `n.div_ceil(4)` blocks of 128: a warp per row.
//! - [`EMBED_F16`], [`ADD_ROWS`], [`GATHER_ROWS`]: row `r` of `out` (`f32 [n, h]`) becomes row `idx[r]` of
//!   `table`: converted from `f16` (a token embedding, read from pinned host memory), added (`f32`, a type
//!   embedding), or copied (`f32`, rows picked out). Arguments: `table, idx: *const u32, out: u64`, `h: u32`.
//!   Launch a grid of `(h.div_ceil(256), n)` blocks of 256.
//! - [`GELU_F16`]: `gelu(a)·b` to `f16 [n, width]`, GELU exact (through erf, to 1.5e-7), from `x`: with `gated`,
//!   rows of `[a, b]` (`f32 [n, 2·width]`); otherwise rows of `a` and `b = 1`. Arguments: `x, out: u64`, `width,
//!   len, gated: u32` (`len = n · width`). Launch `len.div_ceil(256)` blocks of 256.
//! - [`ROPE_F32`]: rotates the queries and keys of `qkv` (`f32 [n, 3h]`, rows `[q, k, v]`, heads of `d`) in place
//!   by each token's position `pos[t]`, the two halves of a head as pairs, with `table` (`f32 [positions, d/2, 2]`,
//!   cosine and sine). Arguments: `qkv, table, pos: u64`, `h, d, n: u32`. Launch `(n·h/2).div_ceil(256)` blocks of
//!   256.
//! - [`ATTEND_F16`] (heads of 64) and [`ATTEND_F16_32`] (heads of 32): bidirectional attention of each sequence's
//!   tokens over its own, from `qkv` (`f32 [n, 3h]`) to `out` (`f16 [n, h]`), each query seeing the keys within
//!   `window` of it, or all when `window` is 0. Tile `i` of `tiles` (`u32 [·, 3]`) is the sequence's first row,
//!   its length, and the first of the tile's up to 64 queries. `scale` is `log2(e)/√d`. Arguments: `qkv, out,
//!   tiles: u64`, `h, window: u32, scale: f32`. Launch a grid of `(tiles, h / d)` blocks of 64: a query per thread,
//!   keys staged in shared memory 64 at a time.

use std::fmt::Write;

/// Name of the `f16` matrix product.
pub const GEMM_F16: &str = "gemm_f16";
/// Rows of tokens a [`GEMM_F16`] block takes, and of weights.
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
pub const ROPE_F32: &str = "rope_f32";
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
/// One stage: 64 activation rows, then 64 weight rows.
const STAGE: usize = 2 * 64 * ROW;

/// Queues stage `buf`'s copy of the slice `off` bytes past the current, under `guard`.
fn copy(s: &mut String, buf: usize, off: usize, guard: &str) {
    let base = buf * STAGE;
    for (dst, at, src) in [("%sa", 0, "%ga0"), ("%sa", 32 * ROW, "%ga1"), ("%sw", 0, "%gw0"), ("%sw", 32 * ROW, "%gw1")] {
        let _ = writeln!(s, "    {guard}cp.async.cg.shared.global [{dst}+{}], [{src}+{off}], 16;", base + at);
    }
    s.push_str("    cp.async.commit_group;\n");
}

/// The products of stage `buf`: per 16 of `k`, two `ldmatrix` of activations (two row tiles of 16 tokens), two of
/// weights (four tiles of 8 rows), eight `mma`.
fn products(s: &mut String, buf: usize) {
    let base = buf * STAGE;
    for ks in 0..2 {
        for mi in 0..2 {
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
        for mi in 0..2 {
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

/// Copies' sources and destinations, and fragments' addresses. `%r6` the block's first token, `%r7` its first
/// row; `%r12` the warp's first token, `%r13` its first row.
const GEMM_HEAD: &str = "
    mov.u32 %r0, %tid.x;
    and.b32 %r1, %r0, 31;
    shr.u32 %r2, %r0, 5;
    ld.param.u32 %r3, [prows];
    ld.param.u32 %r4, [pcols];
    ld.param.u32 %r5, [ptok];
    ld.param.u32 %mode, [pmode];
    mov.u32 %r6, %ctaid.x;
    shl.b32 %r6, %r6, 6;
    mov.u32 %r7, %ctaid.y;
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
    mul.wide.u32 %rd3, %r4, 64;
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
    shl.b32 %r12, %r10, 5;
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
    shr.u32 %turns, %r4, 6;
";

/// Stores the warp's tiles as `mode` asks: tokens `%r12 + 16mi + 8j + g`, rows `%r13 + 8nt + 2t, +1`.
fn gemm_store(s: &mut String) {
    s.push_str(
        "    ld.param.u64 %rd4, [py];
    cvta.to.global.u64 %rd4, %rd4;
    ld.param.u64 %rd5, [pb];
    setp.ne.u64 %pb, %rd5, 0;
    cvta.to.global.u64 %rd5, %rd5;
    setp.eq.u32 %m1, %mode, 1;
    setp.ge.u32 %m2, %mode, 2;
    setp.lt.u32 %n2, %mode, 2;
    setp.eq.u32 %m3, %mode, 3;
    shr.u32 %r10, %r1, 2;
    and.b32 %r11, %r1, 3;
    mad.lo.u32 %r13, %r11, 2, %r13;
    add.u32 %r12, %r12, %r10;
    mul.wide.u32 %rd6, %r13, 4;
    add.u64 %rd6, %rd5, %rd6;\n",
    );
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
    for mi in 0..2 {
        for j in 0..2 {
            let _ = writeln!(s, "    add.u32 %r14, %r12, {};\n    setp.lt.u32 %pt, %r14, %r5;", 16 * mi + 8 * j);
            s.push_str("    and.pred %s32, %pt, %n2;\n    and.pred %sadd, %pt, %m1;\n    and.pred %s16, %pt, %m2;\n");
            s.push_str("    mad.lo.u32 %r14, %r14, %r3, %r13;\n    mul.wide.u32 %rd6, %r14, 4;\n    add.u64 %rd6, %rd4, %rd6;\n");
            s.push_str("    mul.wide.u32 %rd7, %r14, 2;\n    add.u64 %rd7, %rd4, %rd7;\n");
            for nt in 0..4 {
                let c = (mi * 4 + nt) * 4 + 2 * j;
                let _ = writeln!(s, "    add.f32 %v0, %c{c}, %bs{};\n    add.f32 %v1, %c{}, %bs{};", 2 * nt, c + 1, 2 * nt + 1);
                s.push_str("    @%m3 max.f32 %v0, %v0, 0f00000000;\n    @%m3 max.f32 %v1, %v1, 0f00000000;\n");
                let _ = writeln!(s, "    @%sadd ld.global.v2.f32 {{%o0, %o1}}, [%rd6+{}];", 32 * nt);
                s.push_str("    @%sadd add.f32 %v0, %v0, %o0;\n    @%sadd add.f32 %v1, %v1, %o1;\n");
                let _ = writeln!(s, "    @%s32 st.global.v2.f32 [%rd6+{}], {{%v0, %v1}};", 32 * nt);
                let _ = writeln!(s, "    cvt.rn.f16x2.f32 %h, %v1, %v0;\n    @%s16 st.global.b32 [%rd7+{}], %h;", 16 * nt);
            }
        }
    }
}

/// [`GEMM_F16`]: two stages of 32 of `k`, the next slice's copy in flight while this one multiplies.
pub(super) fn gemm_f16() -> String {
    let mut s = String::from(
        ".visible .entry gemm_f16(.param .u64 pa, .param .u64 pw, .param .u64 pb, .param .u64 py, .param .u32 prows, .param .u32 pcols, .param .u32 ptok, .param .u32 pmode)
    .maxntid 128, 1, 1
{
    .reg .pred %p, %q, %pb, %m1, %m2, %n2, %m3, %pt, %s32, %sadd, %s16;
    .reg .b32 %r<16>, %a<8>, %b<8>, %h, %sm, %sa, %sw, %fa, %fb, %turns, %mode;
    .reg .f32 %c<32>, %v<2>, %o<2>, %bs<8>;
    .reg .b64 %rd<8>, %ga<2>, %gw<2>;\n",
    );
    let _ = writeln!(s, "    .shared .align 16 .b8 smem[{}];", 2 * STAGE);
    s += &GEMM_HEAD.replace("HALF", &(64 * ROW).to_string()).replace("ROW", &ROW.to_string());
    (0..32).for_each(|c| _ = writeln!(s, "    mov.f32 %c{c}, 0f00000000;"));
    copy(&mut s, 0, 0, "");
    s += "TURN:\n";
    copy(&mut s, 1, 64, "");
    s += "    cp.async.wait_group 1;\n    bar.sync 0;\n";
    products(&mut s, 0);
    s += "    bar.sync 0;\n    setp.gt.u32 %q, %turns, 1;\n";
    copy(&mut s, 0, 128, "@%q ");
    s += "    cp.async.wait_group 1;\n    bar.sync 0;\n";
    products(&mut s, 1);
    s += "    bar.sync 0;\n";
    for p in ["%ga0", "%ga1", "%gw0", "%gw1"] {
        let _ = writeln!(s, "    add.u64 {p}, {p}, 128;");
    }
    s += "    sub.u32 %turns, %turns, 1;\n    setp.ne.u32 %p, %turns, 0;\n    @%p bra TURN;\n";
    gemm_store(&mut s);
    s + "    ret;\n}\n"
}

/// The sum of `x` over the warp, left in every lane; `t` is scratch.
fn warp_sum(x: &str, t: &str) -> String {
    (0..5).fold(String::new(), |mut s, k| {
        let _ = writeln!(s, "    shfl.sync.bfly.b32 {t}, {x}, {}, 31, -1;\n    add.f32 {x}, {x}, {t};", 16 >> k);
        s
    })
}

/// A loop over this lane's values of the row, `%i` from the lane to `%hh` by 32, `%e` the element's byte offset
/// from the row (4 bytes each); `body` in between.
fn lane_loop(label: &str, body: &str) -> String {
    format!(
        "    mov.u32 %i, %lane;\n{label}:\n    setp.ge.u32 %p, %i, %hh;\n    @%p bra {label}_END;\n    mul.wide.u32 %e, %i, 4;\n{body}    add.u32 %i, %i, 32;\n    bra {label};\n{label}_END:\n"
    )
}

/// [`NORM_F16`]'s registers and pointers: `%row`, `%orow` and `%arow` the row in `x`, `xo` and `a`.
const NORM_HEAD: &str = ".visible .entry norm_f16(.param .u64 px, .param .u64 pg, .param .u64 pb, .param .u64 pa, .param .u64 pxo, .param .u32 ph, .param .u32 pn, .param .f32 peps)
{
    .reg .pred %p, %hb, %ho;
    .reg .b16 %hx;
    .reg .b32 %lane, %rowi, %hh, %n, %i;
    .reg .f32 %s, %t, %x, %y, %z, %mean, %r, %eps, %nh;
    .reg .b64 %row, %orow, %arow, %g, %b, %ad, %e, %o;
    mov.u32 %lane, %tid.x;
    shr.u32 %rowi, %lane, 5;
    and.b32 %lane, %lane, 31;
    mov.u32 %i, %ctaid.x;
    mad.lo.u32 %rowi, %i, 4, %rowi;
    ld.param.u32 %n, [pn];
    setp.ge.u32 %p, %rowi, %n;
    @%p bra EXIT;
    ld.param.u32 %hh, [ph];
    ld.param.f32 %eps, [peps];
    cvt.rn.f32.u32 %nh, %hh;
    mul.wide.u32 %o, %rowi, %hh;
    shl.b64 %o, %o, 2;
    ld.param.u64 %row, [px];
    cvta.to.global.u64 %row, %row;
    add.u64 %row, %row, %o;
    ld.param.u64 %orow, [pxo];
    setp.ne.u64 %ho, %orow, 0;
    cvta.to.global.u64 %orow, %orow;
    add.u64 %orow, %orow, %o;
    ld.param.u64 %arow, [pa];
    cvta.to.global.u64 %arow, %arow;
    shr.u64 %o, %o, 1;
    add.u64 %arow, %arow, %o;
    ld.param.u64 %g, [pg];
    cvta.to.global.u64 %g, %g;
    ld.param.u64 %b, [pb];
    setp.ne.u64 %hb, %b, 0;
    cvta.to.global.u64 %b, %b;
    mov.f32 %s, 0f00000000;
";

/// [`NORM_F16`]: three passes over the row (mean, variance, output), a warp per row.
pub(super) fn norm_f16() -> String {
    let load = "    add.u64 %ad, %row, %e;\n    ld.global.f32 %x, [%ad];\n";
    let mean = format!("{load}    add.f32 %s, %s, %x;\n");
    let var = format!("{load}    sub.f32 %x, %x, %mean;\n    fma.rn.f32 %s, %x, %x, %s;\n");
    let out = format!(
        "{load}    sub.f32 %x, %x, %mean;\n    mul.f32 %x, %x, %r;\n    add.u64 %ad, %g, %e;\n    ld.global.f32 %y, [%ad];
    mov.f32 %z, 0f00000000;\n    add.u64 %ad, %b, %e;\n    @%hb ld.global.f32 %z, [%ad];\n    fma.rn.f32 %x, %x, %y, %z;
    @%ho add.u64 %ad, %orow, %e;\n    @%ho st.global.f32 [%ad], %x;\n    cvt.rn.f16.f32 %hx, %x;\n    shr.u64 %ad, %e, 1;\n    add.u64 %ad, %arow, %ad;\n    st.global.b16 [%ad], %hx;\n"
    );
    let variance = "    div.rn.f32 %s, %s, %nh;\n    add.f32 %s, %s, %eps;\n    sqrt.rn.f32 %s, %s;\n    rcp.rn.f32 %r, %s;\n";
    let mut s = String::from(NORM_HEAD) + &lane_loop("MEAN", &mean) + &warp_sum("%s", "%t") + "    div.rn.f32 %mean, %s, %nh;\n    mov.f32 %s, 0f00000000;\n";
    s += &(lane_loop("VAR", &var) + &warp_sum("%s", "%t") + variance + &lane_loop("OUT", &out));
    s + "EXIT:\n    ret;\n}\n"
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

/// [`ROPE_F32`]: one rotation pair of a query and a key per thread.
pub(super) const ROPE: &str = "
.visible .entry rope_f32(.param .u64 pqkv, .param .u64 ptab, .param .u64 ppos, .param .u32 ph, .param .u32 pd, .param .u32 pn)
{
    .reg .pred %p;
    .reg .b32 %i, %n, %t, %hh, %d, %half, %tok, %r, %head, %j, %pos, %e;
    .reg .f32 %c, %s, %a, %b, %x, %y;
    .reg .b64 %q, %k, %tab, %o, %w;
    mov.u32 %i, %ctaid.x;
    mov.u32 %n, %ntid.x;
    mov.u32 %t, %tid.x;
    mad.lo.u32 %i, %i, %n, %t;
    ld.param.u32 %hh, [ph];
    ld.param.u32 %d, [pd];
    ld.param.u32 %n, [pn];
    shr.u32 %t, %hh, 1;
    mul.lo.u32 %e, %t, %n;
    setp.ge.u32 %p, %i, %e;
    @%p bra EXIT;
    div.u32 %tok, %i, %t;
    rem.u32 %r, %i, %t;
    shr.u32 %half, %d, 1;
    div.u32 %head, %r, %half;
    rem.u32 %j, %r, %half;
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
    mul.wide.u32 %o, %e, 4;
    ld.param.u64 %q, [pqkv];
    cvta.to.global.u64 %q, %q;
    add.u64 %q, %q, %o;
    mul.wide.u32 %o, %hh, 4;
    add.u64 %k, %q, %o;
    mul.wide.u32 %w, %half, 4;
    ld.global.f32 %a, [%q];
    add.u64 %o, %q, %w;
    ld.global.f32 %b, [%o];
    neg.f32 %y, %s;
    mul.f32 %x, %a, %c;
    fma.rn.f32 %x, %b, %y, %x;
    mul.f32 %y, %b, %c;
    fma.rn.f32 %y, %a, %s, %y;
    st.global.f32 [%q], %x;
    st.global.f32 [%o], %y;
    ld.global.f32 %a, [%k];
    add.u64 %o, %k, %w;
    ld.global.f32 %b, [%o];
    neg.f32 %y, %s;
    mul.f32 %x, %a, %c;
    fma.rn.f32 %x, %b, %y, %x;
    mul.f32 %y, %b, %c;
    fma.rn.f32 %y, %a, %s, %y;
    st.global.f32 [%k], %x;
    st.global.f32 [%o], %y;
EXIT:
    ret;
}
";

/// [`ATTEND_F16`] for heads of `d`: each thread one query, its scaled values and its running sums in registers;
/// keys and values of 64 tokens at a time in shared memory, rows padded by 4 against bank conflicts.
pub(super) fn attend_f16(name: &str, d: usize) -> String {
    let stride = 4 * (d + 4);
    attend_start(name, d, stride) + &attend_stage(d) + &attend_key(d, stride) + &attend_end(d, stride)
}

/// [`ATTEND_F16`]'s registers, its tile and its first query's row.
const ATTEND_HEAD: &str = ".visible .entry NAME(.param .u64 pqkv, .param .u64 pout, .param .u64 ptiles, .param .u32 ph, .param .u32 pwin, .param .f32 pscale)
    .maxntid 64, 1, 1
{
    .reg .pred %p, %valid, %glob, %more, %take, %ok;
    .reg .b16 %h0, %h1;
    .reg .b32 %t, %head, %hh, %win, %start, %len, %qoff, %qi, %lo, %hi, %kb, %kk, %nk, %j, %key, %diff, %sm, %sk, %sv, %hp;
    .reg .f32 %q<DIM>, %acc<DIM>, %scale, %m, %l, %s, %pr, %corr, %k<4>, %v<4>;
    .reg .b64 %base, %row, %o;
    .shared .align 16 .b8 kv[SIZE];
    mov.u32 %t, %tid.x;
    mov.u32 %head, %ctaid.y;
    ld.param.u32 %hh, [ph];
    ld.param.u32 %win, [pwin];
    setp.eq.u32 %glob, %win, 0;
    ld.param.f32 %scale, [pscale];
    ld.param.u64 %o, [ptiles];
    cvta.to.global.u64 %o, %o;
    mov.u32 %qi, %ctaid.x;
    mul.wide.u32 %row, %qi, 12;
    add.u64 %o, %o, %row;
    ld.global.u32 %start, [%o];
    ld.global.u32 %len, [%o+4];
    ld.global.u32 %qoff, [%o+8];
    add.u32 %qi, %qoff, %t;
    setp.lt.u32 %valid, %qi, %len;
    ld.param.u64 %base, [pqkv];
    cvta.to.global.u64 %base, %base;
    mul.lo.u32 %hp, %head, DIM;
    add.u32 %j, %start, %qi;
    mul.lo.u32 %j, %j, 3;
    mul.lo.u32 %j, %j, %hh;
    add.u32 %j, %j, %hp;
    mul.wide.u32 %o, %j, 4;
    add.u64 %row, %base, %o;
";

/// The range of keys a tile sees: all of its sequence, or its queries' with `window` either side.
const ATTEND_RANGE: &str = "    mov.f32 %m, 0fFF800000;
    mov.f32 %l, 0f00000000;
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
    mov.u32 %sm, kv;
    mad.lo.u32 %sk, %t, STRIDE, %sm;
    add.u32 %sv, %sk, VALUES;
    mov.u32 %kb, %lo;
CHUNK:
    setp.ge.u32 %p, %kb, %hi;
    @%p bra DONE;
    add.u32 %kk, %kb, %t;
    setp.lt.u32 %take, %kk, %hi;
    add.u32 %j, %start, %kk;
    mul.lo.u32 %j, %j, 3;
    mad.lo.u32 %j, %j, %hh, %hp;
    add.u32 %j, %j, %hh;
    mul.wide.u32 %o, %j, 4;
    add.u64 %row, %base, %o;
    mul.wide.u32 %o, %hh, 4;
";

/// The tile's queries, scaled, and the range of keys they see.
fn attend_start(name: &str, d: usize, stride: usize) -> String {
    let mut s = ATTEND_HEAD.replace("NAME", name).replace("DIM", &d.to_string()).replace("SIZE", &(2 * 64 * stride).to_string());
    for c in (0..d).step_by(4) {
        let _ = writeln!(s, "    @%valid ld.global.v4.f32 {{%q{c}, %q{}, %q{}, %q{}}}, [%row+{}];", c + 1, c + 2, c + 3, 4 * c);
    }
    for c in 0..d {
        let _ = writeln!(s, "    mul.f32 %q{c}, %q{c}, %scale;\n    mov.f32 %acc{c}, 0f00000000;");
    }
    s + &ATTEND_RANGE.replace("STRIDE", &stride.to_string()).replace("VALUES", &(64 * stride).to_string())
}

/// A chunk of 64 keys and values into shared memory, each thread one token.
fn attend_stage(d: usize) -> String {
    let mut s = String::new();
    for c in (0..d).step_by(4) {
        let _ = writeln!(
            s,
            "    @%take ld.global.v4.f32 {{%k0, %k1, %k2, %k3}}, [%row+{}];\n    @%take st.shared.v4.f32 [%sk+{}], {{%k0, %k1, %k2, %k3}};",
            4 * c,
            4 * c
        );
    }
    s += "    add.u64 %row, %row, %o;\n";
    for c in (0..d).step_by(4) {
        let _ = writeln!(
            s,
            "    @%take ld.global.v4.f32 {{%v0, %v1, %v2, %v3}}, [%row+{}];\n    @%take st.shared.v4.f32 [%sv+{}], {{%v0, %v1, %v2, %v3}};",
            4 * c,
            4 * c
        );
    }
    s
}

/// Each key of the chunk within the query's window: its score, the running maximum and sum, the values.
fn attend_key(d: usize, stride: usize) -> String {
    let mut s = String::new();
    let _ = write!(
        s,
        "    bar.sync 0;
    sub.u32 %nk, %hi, %kb;
    min.u32 %nk, %nk, 64;
    mov.u32 %j, 0;
    @!%valid bra NEXT;
KEY:
    add.u32 %key, %kb, %j;
    sub.u32 %diff, %qi, %key;
    setp.lt.u32 %p, %qi, %key;
    @%p sub.u32 %diff, %key, %qi;
    setp.le.u32 %ok, %diff, %win;
    or.pred %ok, %ok, %glob;
    @!%ok bra SKIP;
    mad.lo.u32 %sk, %j, {stride}, %sm;
    mov.f32 %s, 0f00000000;\n"
    );
    for c in (0..d).step_by(4) {
        let _ = writeln!(s, "    ld.shared.v4.f32 {{%k0, %k1, %k2, %k3}}, [%sk+{}];", 4 * c);
        for e in 0..4 {
            let _ = writeln!(s, "    fma.rn.f32 %s, %q{}, %k{e}, %s;", c + e);
        }
    }
    s += "    setp.gt.f32 %p, %s, %m;\n    @!%p bra STAY;\n    sub.f32 %corr, %m, %s;\n    ex2.approx.f32 %corr, %corr;\n    mul.f32 %l, %l, %corr;\n";
    for c in 0..d {
        let _ = writeln!(s, "    mul.f32 %acc{c}, %acc{c}, %corr;");
    }
    let _ = writeln!(
        s,
        "    mov.f32 %m, %s;\nSTAY:\n    sub.f32 %pr, %s, %m;\n    ex2.approx.f32 %pr, %pr;\n    add.f32 %l, %l, %pr;\n    add.u32 %sv, %sk, {};",
        64 * stride
    );
    for c in (0..d).step_by(4) {
        let _ = writeln!(s, "    ld.shared.v4.f32 {{%v0, %v1, %v2, %v3}}, [%sv+{}];", 4 * c);
        for e in 0..4 {
            let _ = writeln!(s, "    fma.rn.f32 %acc{}, %pr, %v{e}, %acc{};", c + e, c + e);
        }
    }
    s
}

/// The next chunk, and at the end each query's values divided by its sum, as `f16`.
fn attend_end(d: usize, stride: usize) -> String {
    let mut s = String::new();
    let _ = write!(
        s,
        "SKIP:
    add.u32 %j, %j, 1;
    setp.lt.u32 %more, %j, %nk;
    @%more bra KEY;
NEXT:
    bar.sync 0;
    mad.lo.u32 %sk, %t, {stride}, %sm;
    add.u32 %sv, %sk, {};
    add.u32 %kb, %kb, 64;
    bra CHUNK;
DONE:
    @!%valid bra EXIT;
    rcp.rn.f32 %l, %l;
    ld.param.u64 %base, [pout];
    cvta.to.global.u64 %base, %base;
    add.u32 %j, %start, %qi;
    mad.lo.u32 %j, %j, %hh, %hp;
    mul.wide.u32 %o, %j, 2;
    add.u64 %row, %base, %o;\n",
        64 * stride
    );
    for c in (0..d).step_by(2) {
        let _ = writeln!(
            s,
            "    mul.f32 %k0, %acc{c}, %l;\n    mul.f32 %k1, %acc{}, %l;\n    cvt.rn.f16x2.f32 %sk, %k1, %k0;\n    st.global.b32 [%row+{}], %sk;",
            c + 1,
            2 * c
        );
    }
    s + "EXIT:\n    ret;\n}\n"
}
