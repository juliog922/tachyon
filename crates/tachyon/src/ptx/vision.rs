//! Gemma 4's vision tower's own kernels, beside the encoder's products, attention and row lookups
//! ([`super::encoder`]). Rows of the residual stream are `f32`; what a product reads, `f16`.
//!
//! - [`RMS_F16`]: per row of `x` (`f32 [n, h]`), with `y` (`f32 [n, h]`) unless 0, first the residual step
//!   `x += w1 ⊙ rms(y)`, written back; then `w2 ⊙ rms(x)` as `f16` to `a` and as `f32` to `xo` (may be `x`), each
//!   unless 0. `rms(v) = v / √(mean(v²) + eps)`; a weight of 0 is all ones. Arguments: `x, y, w1, w2, a, xo: u64`,
//!   `h: u32, eps: f32`. `h` is a multiple of 128, at most 4096. Launch `n` blocks of `h / 4` threads.
//! - [`HEADS_F16`]: Gemma 4's per-head norms and 2-D RoPE over `qkv` (`f16 [n, 3h]`, rows `[q, k, v]`, heads of
//!   64), in place: each head's values to `rms`, queries times `qw`, keys times `kw` (`f32 [64]`); then the queries
//!   and keys turned, dimensions `j, j + 16` (`j < 16`) by the token's `x` position, `32 + j, 48 + j` by its `y`,
//!   at the angle `pos · θ^(−j/16)` whose cosine and sine are `table` (`f32 [positions, 16, 2]`). Positions are
//!   `xs, ys` (`u32 [n]`). Arguments: `qkv, xs, ys, table, qw, kw: u64`, `h: u32, eps: f32`. Launch a grid of `(3,
//!   n)` blocks of `32 · h / 64` threads: a warp per head, a block row per token.
//! - [`POOL_F16`]: the mean of each square of [`POOL`] × [`POOL`] rows of `x` (`f32`, the patch grid `width`
//!   rows across), rows of `out` (`f32`) in the squares' order, row by row. Arguments: `x, out: u64`, `width, h:
//!   u32`. Launch a grid of `(h.div_ceil(256), squares)` blocks of 256.

use super::warp;
use crate::media::POOL;
use std::fmt::Write;

/// Name of the RMS norm with its residual step.
pub const RMS_F16: &str = "rms_f16";
/// Name of the per-head norms and 2-D RoPE.
pub const HEADS_F16: &str = "heads_f16";
/// Name of the pooling of the patch grid.
pub const POOL_F16: &str = "pool_f16";

/// `%r = 1 / √(mean(v²) + eps)` over the block's row, the thread's `v` in `%v0..%v3`; through `red`, label `L`.
pub(super) fn rms(v: &str, label: &str) -> String {
    let mut s = format!("    mul.f32 %s, {v}0, {v}0;\n");
    (1..4).for_each(|e| _ = writeln!(s, "    fma.rn.f32 %s, {v}{e}, {v}{e}, %s;"));
    s += &warp("add.f32", "%s", "%u");
    let _ = write!(
        s,
        "    @%lane0 st.shared.f32 [%slot], %s;
    bar.sync 0;
    mov.f32 %s, 0f00000000;
    mov.u32 %k, red;
    mov.u32 %n, %nw;
{label}:
    ld.shared.f32 %u, [%k];
    add.f32 %s, %s, %u;
    add.u32 %k, %k, 4;
    sub.u32 %n, %n, 1;
    setp.ne.u32 %p, %n, 0;
    @%p bra {label};
    bar.sync 0;
    div.rn.f32 %s, %s, %nh;
    add.f32 %s, %s, %eps;
    sqrt.rn.f32 %s, %s;
    rcp.rn.f32 %r, %s;
"
    );
    s
}

/// The thread's four weights of `param` at `%wo` into `%w0..%w3`, ones if it is 0.
fn weights(param: &str) -> String {
    let mut s = (0..4).fold(String::new(), |mut s, e| {
        let _ = writeln!(s, "    mov.f32 %w{e}, 0f3F800000;");
        s
    });
    let _ = writeln!(s, "    ld.param.u64 %ad, [{param}];\n    setp.ne.u64 %p, %ad, 0;\n    cvta.to.global.u64 %ad, %ad;\n    add.u64 %ad, %ad, %wo;");
    s + "    @%p ld.global.v4.f32 {%w0, %w1, %w2, %w3}, [%ad];\n"
}

/// [`RMS_F16`]'s start: the thread's slot for its warp's sum, its four values of `x` (at `%xa`), and of `y`
/// unless there is none.
const RMS_HEAD: &str =
    ".visible .entry rms_f16(.param .u64 px, .param .u64 py, .param .u64 pw1, .param .u64 pw2, .param .u64 pa, .param .u64 pxo, .param .u32 ph, .param .f32 peps)
{
    .reg .pred %p, %lane0;
    .reg .b32 %t, %k, %n, %nw, %hh, %slot, %h0, %h1;
    .reg .f32 %x<4>, %y<4>, %w<4>, %s, %u, %r, %eps, %nh;
    .reg .b64 %xa, %ad, %o, %wo, %fo, %wa;
    .shared .align 4 .f32 red[32];
    mov.u32 %t, %tid.x;
    and.b32 %k, %t, 31;
    setp.eq.u32 %lane0, %k, 0;
    shr.u32 %k, %t, 5;
    mov.u32 %slot, red;
    mad.lo.u32 %slot, %k, 4, %slot;
    mov.u32 %nw, %ntid.x;
    shr.u32 %nw, %nw, 5;
    ld.param.u32 %hh, [ph];
    ld.param.f32 %eps, [peps];
    cvt.rn.f32.u32 %nh, %hh;
    mul.wide.u32 %wo, %t, 16;
    mov.u32 %k, %ctaid.x;
    mul.wide.u32 %o, %k, %hh;
    shl.b64 %o, %o, 2;
    add.u64 %o, %o, %wo;
    mov.u64 %fo, %o;
    ld.param.u64 %xa, [px];
    cvta.to.global.u64 %xa, %xa;
    add.u64 %xa, %xa, %o;
    ld.global.v4.f32 {%x0, %x1, %x2, %x3}, [%xa];
    ld.param.u64 %ad, [py];
    setp.eq.u64 %p, %ad, 0;
    @%p bra NORM;
    cvta.to.global.u64 %ad, %ad;
    add.u64 %ad, %ad, %o;
    ld.global.v4.f32 {%y0, %y1, %y2, %y3}, [%ad];
";

/// [`RMS_F16`]'s end: the four values of `a` stored as `f16`.
const RMS_STORE: &str = "    ld.param.u64 %ad, [pxo];
    setp.ne.u64 %p, %ad, 0;
    cvta.to.global.u64 %ad, %ad;
    add.u64 %ad, %ad, %fo;
    @%p st.global.v4.f32 [%ad], {%x0, %x1, %x2, %x3};
    setp.eq.u64 %p, %o, 0;
    @%p bra EXIT;
    cvt.rn.f16x2.f32 %h0, %x1, %x0;
    cvt.rn.f16x2.f32 %h1, %x3, %x2;
    mov.u32 %k, %ctaid.x;
    mul.wide.u32 %ad, %k, %hh;
    shl.b64 %ad, %ad, 1;
    shr.u64 %wo, %wo, 1;
    add.u64 %ad, %ad, %wo;
    cvta.to.global.u64 %o, %o;
    add.u64 %o, %o, %ad;
    st.global.v2.b32 [%o], {%h0, %h1};
EXIT:
    ret;
}
";

/// [`RMS_F16`].
pub(super) fn rms_f16() -> String {
    let mut s = String::from(RMS_HEAD);
    s += &rms("%y", "SUMY");
    s += &weights("pw1");
    (0..4).for_each(|e| _ = writeln!(s, "    mul.f32 %y{e}, %y{e}, %r;\n    fma.rn.f32 %x{e}, %y{e}, %w{e}, %x{e};"));
    s += "    st.global.v4.f32 [%xa], {%x0, %x1, %x2, %x3};\nNORM:\n    ld.param.u64 %o, [pa];\n    ld.param.u64 %ad, [pxo];\n    or.b64 %wa, %o, %ad;\n    setp.eq.u64 %p, %wa, 0;\n    @%p bra EXIT;\n";
    s += &rms("%x", "SUMX");
    s += &weights("pw2");
    (0..4).for_each(|e| _ = writeln!(s, "    mul.f32 %x{e}, %x{e}, %r;\n    mul.f32 %x{e}, %x{e}, %w{e};"));
    s + RMS_STORE
}

/// [`HEADS_F16`]: lane `l` holds dimensions `l` and `l + 32` of its head; the pair partner of `l` is lane `l ^ 16`.
pub(super) const HEADS: &str = "
.visible .entry heads_f16(.param .u64 pqkv, .param .u64 pxs, .param .u64 pys, .param .u64 ptab, .param .u64 pqw, .param .u64 pkw, .param .u32 ph, .param .f32 peps)
{
    .reg .pred %p, %lo;
    .reg .b16 %ha, %hb;
    .reg .b32 %t, %lane, %part, %tok, %hh, %k, %pos;
    .reg .f32 %a, %b, %s, %u, %r, %eps, %c, %sn, %pa, %pb;
    .reg .b64 %q, %o, %ad, %tab;
    mov.u32 %t, %tid.x;
    and.b32 %lane, %t, 31;
    shr.u32 %k, %t, 5;
    mov.u32 %part, %ctaid.x;
    mov.u32 %tok, %ctaid.y;
    ld.param.u32 %hh, [ph];
    ld.param.f32 %eps, [peps];
    mul.lo.u32 %k, %k, 64;
    mad.lo.u32 %k, %part, %hh, %k;
    mul.lo.u32 %t, %hh, 3;
    mad.lo.u32 %k, %tok, %t, %k;
    add.u32 %k, %k, %lane;
    mul.wide.u32 %o, %k, 2;
    ld.param.u64 %q, [pqkv];
    cvta.to.global.u64 %q, %q;
    add.u64 %q, %q, %o;
    ld.global.b16 %ha, [%q];
    ld.global.b16 %hb, [%q+64];
    cvt.f32.f16 %a, %ha;
    cvt.f32.f16 %b, %hb;
    mul.f32 %s, %a, %a;
    fma.rn.f32 %s, %b, %b, %s;
SUM    mul.f32 %s, %s, 0f3C800000;
    add.f32 %s, %s, %eps;
    sqrt.rn.f32 %s, %s;
    rcp.rn.f32 %r, %s;
    mul.f32 %a, %a, %r;
    mul.f32 %b, %b, %r;
    setp.eq.u32 %p, %part, 2;
    @%p bra STORE;
    setp.eq.u32 %p, %part, 0;
    ld.param.u64 %ad, [pkw];
    ld.param.u64 %o, [pqw];
    @%p mov.u64 %ad, %o;
    cvta.to.global.u64 %ad, %ad;
    mul.wide.u32 %o, %lane, 4;
    add.u64 %ad, %ad, %o;
    ld.global.f32 %u, [%ad];
    mul.f32 %a, %a, %u;
    ld.global.f32 %u, [%ad+128];
    mul.f32 %b, %b, %u;
    shfl.sync.bfly.b32 %pa, %a, 16, 31, -1;
    shfl.sync.bfly.b32 %pb, %b, 16, 31, -1;
    setp.lt.u32 %lo, %lane, 16;
    @%lo neg.f32 %pa, %pa;
    @%lo neg.f32 %pb, %pb;
    ld.param.u64 %tab, [ptab];
    cvta.to.global.u64 %tab, %tab;
TURN%a%pa%pxs
TURN%b%pb%pys
STORE:
    cvt.rn.f16.f32 %ha, %a;
    cvt.rn.f16.f32 %hb, %b;
    st.global.b16 [%q], %ha;
    st.global.b16 [%q+64], %hb;
    ret;
}
";

/// [`HEADS_F16`]'s PTX: `v ← v·cos + partner·sin` at the token's position from `pos`, for each half of the head.
pub(super) fn heads_f16() -> String {
    let turn = |v: &str, partner: &str, pos: &str| {
        format!(
            "    ld.param.u64 %o, [{pos}];
    cvta.to.global.u64 %o, %o;
    mul.wide.u32 %ad, %tok, 4;
    add.u64 %o, %o, %ad;
    ld.global.u32 %pos, [%o];
    and.b32 %k, %lane, 15;
    mad.lo.u32 %k, %pos, 16, %k;
    mul.wide.u32 %o, %k, 8;
    add.u64 %o, %tab, %o;
    ld.global.v2.f32 {{%c, %sn}}, [%o];
    mul.f32 {v}, {v}, %c;
    fma.rn.f32 {v}, {partner}, %sn, {v};
"
        )
    };
    HEADS
        .replace("SUM", &warp("add.f32", "%s", "%u"))
        .replace("TURN%a%pa%pxs\n", &turn("%a", "%pa", "pxs"))
        .replace("TURN%b%pb%pys\n", &turn("%b", "%pb", "pys"))
}

/// [`POOL_F16`]'s start: the thread's value, the first row of its square at `%a`, `%step` bytes a row of squares
/// down, `%hs` a patch across.
const POOL_HEAD: &str = ".visible .entry pool_f16(.param .u64 px, .param .u64 pout, .param .u32 pwidth, .param .u32 ph)
{
    .reg .pred %p;
    .reg .b32 %i, %t, %hh, %wd, %sq, %bx, %by, %row;
    .reg .f32 %s, %u;
    .reg .b64 %a, %b, %o, %step, %hs;
    mov.u32 %i, %ctaid.x;
    mov.u32 %t, %tid.x;
    mad.lo.u32 %i, %i, 256, %t;
    ld.param.u32 %hh, [ph];
    setp.ge.u32 %p, %i, %hh;
    @%p bra EXIT;
    ld.param.u32 %wd, [pwidth];
    div.u32 %t, %wd, POOL;
    mov.u32 %sq, %ctaid.y;
    rem.u32 %bx, %sq, %t;
    div.u32 %by, %sq, %t;
    mul.lo.u32 %row, %by, POOL;
    mul.lo.u32 %row, %row, %wd;
    mad.lo.u32 %row, %bx, POOL, %row;
    mad.lo.u32 %row, %row, %hh, %i;
    mul.wide.u32 %o, %row, 4;
    ld.param.u64 %a, [px];
    cvta.to.global.u64 %a, %a;
    add.u64 %a, %a, %o;
    mul.wide.u32 %step, %wd, %hh;
    shl.b64 %step, %step, 2;
    mul.wide.u32 %hs, %hh, 4;
    mov.f32 %s, 0f00000000;
";

/// [`POOL_F16`]'s end: the mean stored.
const POOL_STORE: &str = "    mul.f32 %s, %s, SCALE;
    mov.u32 %t, %ctaid.y;
    mad.lo.u32 %row, %t, %hh, %i;
    mul.wide.u32 %o, %row, 4;
    ld.param.u64 %a, [pout];
    cvta.to.global.u64 %a, %a;
    add.u64 %a, %a, %o;
    st.global.f32 [%a], %s;
EXIT:
    ret;
}
";

/// [`POOL_F16`]: one value per thread, the sum of its square's rows times `1 / POOL²`.
pub(super) fn pool_f16() -> String {
    let mut s = POOL_HEAD.replace("POOL", &POOL.to_string());
    for dy in 0..POOL {
        s += "    mov.u64 %b, %a;\n";
        (0..POOL).for_each(|_| s += "    ld.global.f32 %u, [%b];\n    add.f32 %s, %s, %u;\n    add.u64 %b, %b, %hs;\n");
        if dy + 1 < POOL {
            s += "    add.u64 %a, %a, %step;\n";
        }
    }
    s + &POOL_STORE.replace("SCALE", &super::encoder::f(1.0 / (POOL * POOL) as f32))
}
