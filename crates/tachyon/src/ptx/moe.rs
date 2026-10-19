//! The mixture of experts of Gemma 4 26B A4B: routing, and the sums of the experts each token takes.
//!
//! - [`ROUTE_TOPK`] turns a token's router logits into its experts: a softmax over all of them, the `k` most likely
//!   (ties to the lower number), their probabilities divided by their sum and times each expert's scale, as Hugging
//!   Face's `Gemma4TextRouter` computes them. Arguments: `logits: *const f32` (`[tokens][experts]`), `scale: *const
//!   f32` (`[experts]`), `idx: *mut u32`, `w: *mut f32` (each `[tokens][k]`, in the order picked), `experts, k: u32`
//!   (`experts ≤ 128`, `k ≤ 16`). Launch a block of 128 threads per token.
//! - [`MOE_ADD`] adds expert `e`'s output to the tokens that took it: `y[t][i] += w[t][j] · x[t][i]` where
//!   `idx[t][j] = e`, for prefill, which runs every expert over the whole chunk; with `fresh` (the first expert),
//!   `y` starts from zero for every token. Arguments: `y, x: *mut f32` (`[tokens][len]`), `idx: *const u32`,
//!   `w: *const f32`, `e, k, len, fresh: u32`. Launch a grid of `(len.div_ceil(256), tokens)` blocks of 256.
//!
//! A decode step's experts run as [`super::GEMV_Q4_MOE`]: one launch for every chosen expert's gate and up, one for
//! their downs, each weighted, then [`super::SUM_SLICES`] adds them in order.

use std::fmt::Write;

/// Name of the router's top-k.
pub const ROUTE_TOPK: &str = "route_topk";
/// Name of prefill's accumulation of one expert's output.
pub const MOE_ADD: &str = "moe_add";

/// `{reg}` reduced over the block of four warps with `op` through the shared `rv`; every thread gets the result.
fn reduce(op: &str, reg: &str) -> String {
    let mut s = String::new();
    for k in [16, 8, 4, 2, 1] {
        _ = write!(s, "    shfl.sync.bfly.b32 %o, {reg}, {k}, 31, -1;\n    {op} {reg}, {reg}, %o;\n");
    }
    _ = write!(s, "    @%lane0 st.shared.f32 [%rw], {reg};\n    bar.sync 0;\n    ld.shared.v4.f32 {{%r0, %r1, %r2, %r3}}, [rv];\n    bar.sync 0;\n");
    s + &format!("    {op} {reg}, %r0, %r1;\n    {op} %r2, %r2, %r3;\n    {op} {reg}, {reg}, %r2;\n")
}

/// Keeps in `%bv`, `%bi` the better of them and `(%ov, %oi)`: the larger value, or the lower number at a tie.
const BETTER: &str = "    setp.gt.f32 %p, %ov, %bv;
    setp.eq.f32 %q, %ov, %bv;
    setp.lt.u32 %u, %oi, %bi;
    and.pred %q, %q, %u;
    or.pred %p, %p, %q;
    @%p mov.f32 %bv, %ov;
    @%p mov.u32 %bi, %oi;
";

/// [`ROUTE_TOPK`]'s head: the token's logit of this thread's expert in `%v`, `-∞` past the last.
const ROUTE_HEAD: &str = "
.visible .entry route_topk(.param .u64 pl, .param .u64 pes, .param .u64 pidx, .param .u64 pw, .param .u32 pe, .param .u32 pk)
{
    .shared .align 16 .f32 rv[4];
    .shared .align 16 .u32 ri[4];
    .shared .align 4 .f32 wk[16];
    .shared .align 4 .u32 ik[16];
    .reg .pred %p, %q, %u, %in, %lane0;
    .reg .b32 %th, %lane, %warp, %t, %e, %k, %j, %bi, %oi, %x, %rw, %ri;
    .reg .f32 %v, %m, %o, %s, %bv, %ov, %tot, %r<4>;
    .reg .b64 %a, %b;
    mov.u32 %th, %tid.x;
    and.b32 %lane, %th, 31;
    shr.u32 %warp, %th, 5;
    setp.eq.u32 %lane0, %lane, 0;
    mov.u32 %rw, rv;
    mad.lo.u32 %rw, %warp, 4, %rw;
    mov.u32 %ri, ri;
    mad.lo.u32 %ri, %warp, 4, %ri;
    mov.u32 %t, %ctaid.x;
    ld.param.u32 %e, [pe];
    ld.param.u32 %k, [pk];
    setp.lt.u32 %in, %th, %e;
    ld.param.u64 %a, [pl];
    cvta.to.global.u64 %a, %a;
    mad.lo.u32 %x, %t, %e, %th;
    mul.wide.u32 %b, %x, 4;
    add.u64 %a, %a, %b;
    mov.f32 %v, 0fFF800000;
    @%in ld.global.f32 %v, [%a];
    mov.f32 %m, %v;
";

/// [`ROUTE_TOPK`]'s tail: thread `j < k` writes pick `j`, its probability over the picks' sum, times its expert's
/// scale.
const ROUTE_TAIL: &str = "    setp.ge.u32 %p, %th, %k;
    @%p bra EXIT;
    mov.u32 %x, wk;
    mad.lo.u32 %x, %th, 4, %x;
    ld.shared.f32 %v, [%x];
    mov.u32 %x, ik;
    mad.lo.u32 %x, %th, 4, %x;
    ld.shared.u32 %bi, [%x];
    div.rn.f32 %v, %v, %tot;
    ld.param.u64 %a, [pes];
    cvta.to.global.u64 %a, %a;
    mul.wide.u32 %b, %bi, 4;
    add.u64 %a, %a, %b;
    ld.global.f32 %s, [%a];
    mul.f32 %v, %v, %s;
    mad.lo.u32 %x, %t, %k, %th;
    mul.wide.u32 %b, %x, 4;
    ld.param.u64 %a, [pidx];
    cvta.to.global.u64 %a, %a;
    add.u64 %a, %a, %b;
    st.global.u32 [%a], %bi;
    ld.param.u64 %a, [pw];
    cvta.to.global.u64 %a, %a;
    add.u64 %a, %a, %b;
    st.global.f32 [%a], %v;
EXIT:
    ret;
}
";

/// One pick of [`ROUTE_TOPK`]: the block's best expert, out of the running; thread 0 keeps it in `wk`, `ik`.
fn pick(s: &mut String) {
    *s += "    mov.f32 %bv, %v;\n    mov.u32 %bi, %th;\n";
    for k in [16, 8, 4, 2, 1] {
        _ = write!(s, "    shfl.sync.bfly.b32 %ov, %bv, {k}, 31, -1;\n    shfl.sync.bfly.b32 %oi, %bi, {k}, 31, -1;\n{BETTER}");
    }
    *s += "    @%lane0 st.shared.f32 [%rw], %bv;\n    @%lane0 st.shared.u32 [%ri], %bi;\n    bar.sync 0;\n";
    *s += "    ld.shared.f32 %bv, [rv];\n    ld.shared.u32 %bi, [ri];\n";
    for w in 1..4 {
        _ = write!(s, "    ld.shared.f32 %ov, [rv+{}];\n    ld.shared.u32 %oi, [ri+{}];\n{BETTER}", 4 * w, 4 * w);
    }
    *s += "    bar.sync 0;\n    setp.eq.u32 %p, %th, %bi;\n    @%p mov.f32 %v, 0fBF800000;\n    add.f32 %tot, %tot, %bv;\n";
    *s += "    setp.eq.u32 %p, %th, 0;\n    mov.u32 %x, wk;\n    mad.lo.u32 %x, %j, 4, %x;\n    @%p st.shared.f32 [%x], %bv;\n";
    *s += "    mov.u32 %x, ik;\n    mad.lo.u32 %x, %j, 4, %x;\n    @%p st.shared.u32 [%x], %bi;\n";
}

/// [`ROUTE_TOPK`].
fn route_topk() -> String {
    let mut s = String::from(ROUTE_HEAD);
    s += &reduce("max.f32", "%m");
    s += "    sub.f32 %v, %v, %m;\n    mul.f32 %v, %v, 0f3FB8AA3B;\n    ex2.approx.f32 %v, %v;\n    mov.f32 %s, %v;\n";
    s += &reduce("add.f32", "%s");
    // Experts past the last are never picked: their probability is below every real one.
    s += "    div.rn.f32 %v, %v, %s;\n    @!%in mov.f32 %v, 0fBF800000;\n    mov.f32 %tot, 0f00000000;\n    mov.u32 %j, 0;\nPICK:\n";
    pick(&mut s);
    s += "    add.u32 %j, %j, 1;\n    setp.lt.u32 %p, %j, %k;\n    @%p bra PICK;\n    bar.sync 0;\n";
    s + ROUTE_TAIL
}

/// Every kernel of this module, and [`super::GEMV_Q4_MOE`].
pub(super) fn kernels() -> String {
    route_topk() + MOE_ADD_PTX + &super::gemv::gemv_q4(super::UNROLL, 32, true)
}

/// [`MOE_ADD`].
const MOE_ADD_PTX: &str = "
.visible .entry moe_add(.param .u64 py, .param .u64 px, .param .u64 pidx, .param .u64 pw, .param .u32 pe, .param .u32 pk, .param .u32 plen,
    .param .u32 pfresh)
{
    .reg .pred %p, %fresh;
    .reg .b32 %i, %t, %e, %k, %j, %len, %x, %c;
    .reg .f32 %w, %a, %b;
    .reg .b64 %pi, %pw, %o, %y, %s;
    mov.u32 %i, %ctaid.x;
    mov.u32 %x, %tid.x;
    mad.lo.u32 %i, %i, 256, %x;
    ld.param.u32 %len, [plen];
    setp.ge.u32 %p, %i, %len;
    @%p bra EXIT;
    ld.param.u32 %x, [pfresh];
    setp.ne.u32 %fresh, %x, 0;
    mov.u32 %t, %ctaid.y;
    ld.param.u32 %e, [pe];
    ld.param.u32 %k, [pk];
    ld.param.u64 %pi, [pidx];
    cvta.to.global.u64 %pi, %pi;
    ld.param.u64 %pw, [pw];
    cvta.to.global.u64 %pw, %pw;
    mul.lo.u32 %x, %t, %k;
    mul.wide.u32 %o, %x, 4;
    add.u64 %pi, %pi, %o;
    add.u64 %pw, %pw, %o;
    mov.f32 %w, 0f00000000;
    mov.u32 %j, 0;
SLOT:
    setp.ge.u32 %p, %j, %k;
    @%p bra MISSED;
    ld.global.u32 %c, [%pi];
    setp.eq.u32 %p, %c, %e;
    @%p ld.global.f32 %w, [%pw];
    @%p bra FOUND;
    add.u64 %pi, %pi, 4;
    add.u64 %pw, %pw, 4;
    add.u32 %j, %j, 1;
    bra SLOT;
MISSED:
    @!%fresh bra EXIT;
FOUND:
    mad.lo.u32 %x, %t, %len, %i;
    mul.wide.u32 %o, %x, 4;
    ld.param.u64 %y, [py];
    cvta.to.global.u64 %y, %y;
    add.u64 %y, %y, %o;
    ld.param.u64 %s, [px];
    cvta.to.global.u64 %s, %s;
    add.u64 %s, %s, %o;
    mov.f32 %a, 0f00000000;
    @!%fresh ld.global.f32 %a, [%y];
    ld.global.f32 %b, [%s];
    fma.rn.f32 %a, %w, %b, %a;
    st.global.f32 [%y], %a;
EXIT:
    ret;
}
";
