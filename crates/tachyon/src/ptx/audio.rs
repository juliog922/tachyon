//! Gemma 4's audio tower's own kernels, beside the clipped products ([`super::encoder`]) and the vision tower's
//! RMS norm ([`super::vision`]). Rows of the residual stream are `f32`; what a product reads, `f16`.
//!
//! - [`SUBSAMPLE`]: a 3 × 3 convolution of stride 2 and padding 1 over time and frequency, then a layer norm over
//!   the channels (weight `g`, no bias) and a ReLU: `input` is `f32 [rows, fin, cin]` (time, frequency, channel),
//!   rows from `rows` on read as zeros; `w` is `f32 [cout, cin, 3, 3]`; the output, `[time, fin / 2, cout]`, is `f32`
//!   or with `half` `f16`. Arguments: `input, w, g, out: u64`, `rows, fin, cin, half: u32, eps: f32`. `fin` is
//!   even. Launch a grid of `(fin / 2, output rows)` blocks of `cout` threads, a multiple of 32.
//! - [`ATTEND_AUDIO`]: Gemma 4's audio attention, heads of 128: query `t` sees keys `t − left ..= t`, its logit
//!   `cap · tanh(Σ q·s ⊙ (kscale·k + rel[t − s]) / cap)`, with the queries' scale `s` (`f32 [128]`) and a relative
//!   position's key `rel` (`f32 [left + 1, h]`, by distance). `qkv` is `f32 [n, 3h]` (rows `[q, k, v]`); `out`
//!   `f16 [n, h]`. Arguments: `qkv, rel, scale, out: u64`, `h, left: u32, kscale, cap: f32`. Launch a grid of `(n,
//!   h / 128)` blocks of 32: a warp per token and head, four dimensions a lane.
//! - [`LCONV`]: the light convolution's middle: per channel a causal convolution of `K` taps (`w`, `f32 [h, K]`)
//!   over `x` (`f16 [n, h]`), then an RMS norm with weight `g` and a `SiLU`, to `out` (`f16 [n, h]`). Arguments:
//!   `x, w, g, out: u64`, `h, taps: u32, eps: f32`. Launch `n` blocks of `h / 4` threads, a multiple of 32.

use super::vision::rms;
use super::warp;
use std::fmt::Write;

/// Name of the subsampling convolution.
pub const SUBSAMPLE: &str = "subsample";
/// Name of the audio attention.
pub const ATTEND_AUDIO: &str = "attend_audio";
/// Name of the light convolution's middle.
pub const LCONV: &str = "lconv";

/// `−log2(e)` and `log2(e)`, as PTX literals.
const LOG2E_NEG: &str = "0fBFB8AA3B";
const LOG2E: &str = "0f3FB8AA3B";

/// `v` summed over the block into every thread; through `red`, label `label` (needs the registers of
/// [`SUBSAMPLE`]'s head).
fn block_sum(v: &str, label: &str) -> String {
    let mut s = warp("add.f32", v, "%u");
    let _ = write!(
        s,
        "    @%lane0 st.shared.f32 [%slot], {v};
    bar.sync 0;
    mov.f32 {v}, 0f00000000;
    mov.u32 %k, red;
    mov.u32 %n, %nw;
{label}:
    ld.shared.f32 %u, [%k];
    add.f32 {v}, {v}, %u;
    add.u32 %k, %k, 4;
    sub.u32 %n, %n, 1;
    setp.ne.u32 %p, %n, 0;
    @%p bra {label};
    bar.sync 0;
"
    );
    s
}

/// The registers and shared slots of a block reduction, in [`SUBSAMPLE`] and [`LCONV`].
const REDUCE: &str = "    .reg .pred %p, %lane0;
    .reg .b32 %k, %n, %nw, %slot;
    .reg .f32 %u;
    .shared .align 4 .f32 red[32];
    mov.u32 %k, %tid.x;
    and.b32 %k, %k, 31;
    setp.eq.u32 %lane0, %k, 0;
    mov.u32 %k, %tid.x;
    shr.u32 %k, %k, 5;
    mov.u32 %slot, red;
    mad.lo.u32 %slot, %k, 4, %slot;
    mov.u32 %nw, %ntid.x;
    shr.u32 %nw, %nw, 5;
";

/// [`SUBSAMPLE`]'s start: the thread's channel `%co`, its output row `%to` and frequency `%fo`, and the sum `%acc`.
const SUBSAMPLE_HEAD: &str = ".visible .entry subsample(.param .u64 pin, .param .u64 pw, .param .u64 pg, .param .u64 pout, .param .u32 prows, .param .u32 pfin, .param .u32 pcin, .param .u32 phalf, .param .f32 peps)
{
    .reg .pred %ok, %q;
    .reg .b16 %hv;
    .reg .b32 %co, %cout, %to, %fo, %rows, %fin, %cin, %ti, %fi, %ci, %e;
    .reg .f32 %acc, %x, %y, %mean, %var, %eps;
    .reg .b64 %in, %w, %a, %b, %o;
REDUCE    mov.u32 %co, %tid.x;
    mov.u32 %cout, %ntid.x;
    mov.u32 %fo, %ctaid.x;
    mov.u32 %to, %ctaid.y;
    ld.param.u32 %rows, [prows];
    ld.param.u32 %fin, [pfin];
    ld.param.u32 %cin, [pcin];
    ld.param.f32 %eps, [peps];
    ld.param.u64 %in, [pin];
    cvta.to.global.u64 %in, %in;
    ld.param.u64 %w, [pw];
    cvta.to.global.u64 %w, %w;
    mov.f32 %acc, 0f00000000;
";

/// The taps of window position `(kh, kw)`: input row `2·to − 1 + kh`, frequency `2·fo − 1 + kw`, every channel.
fn tap(kh: u32, kw: u32) -> String {
    format!(
        "    shl.b32 %ti, %to, 1;
    add.u32 %ti, %ti, {kh};
    sub.u32 %ti, %ti, 1;
    shl.b32 %fi, %fo, 1;
    add.u32 %fi, %fi, {kw};
    sub.u32 %fi, %fi, 1;
    setp.lt.u32 %ok, %ti, %rows;
    setp.lt.u32 %q, %fi, %fin;
    and.pred %ok, %ok, %q;
    @!%ok bra SKIP{kh}{kw};
    mad.lo.u32 %e, %ti, %fin, %fi;
    mul.lo.u32 %e, %e, %cin;
    mul.wide.u32 %a, %e, 4;
    add.u64 %a, %in, %a;
    mul.lo.u32 %e, %co, %cin;
    mul.lo.u32 %e, %e, 9;
    add.u32 %e, %e, {};
    mul.wide.u32 %b, %e, 4;
    add.u64 %b, %w, %b;
    mov.u32 %ci, %cin;
TAPS{kh}{kw}:
    ld.global.f32 %x, [%a];
    ld.global.f32 %y, [%b];
    fma.rn.f32 %acc, %x, %y, %acc;
    add.u64 %a, %a, 4;
    add.u64 %b, %b, 36;
    sub.u32 %ci, %ci, 1;
    setp.ne.u32 %q, %ci, 0;
    @%q bra TAPS{kh}{kw};
SKIP{kh}{kw}:
",
        3 * kh + kw
    )
}

/// [`SUBSAMPLE`]'s end: the layer norm over the block's channels, the ReLU, and the store.
const SUBSAMPLE_NORM: &str = "    cvt.rn.f32.u32 %y, %cout;
    mov.f32 %mean, %acc;
SUM_MEAN    div.rn.f32 %mean, %mean, %y;
    sub.f32 %x, %acc, %mean;
    mul.f32 %var, %x, %x;
SUM_VAR    div.rn.f32 %var, %var, %y;
    add.f32 %var, %var, %eps;
    sqrt.rn.f32 %var, %var;
    div.rn.f32 %x, %x, %var;
    ld.param.u64 %a, [pg];
    cvta.to.global.u64 %a, %a;
    mul.wide.u32 %b, %co, 4;
    add.u64 %a, %a, %b;
    ld.global.f32 %y, [%a];
    mul.f32 %x, %x, %y;
    max.f32 %x, %x, 0f00000000;
    shr.u32 %e, %fin, 1;
    mad.lo.u32 %e, %to, %e, %fo;
    mad.lo.u32 %e, %e, %cout, %co;
    ld.param.u64 %o, [pout];
    cvta.to.global.u64 %o, %o;
    ld.param.u32 %ti, [phalf];
    setp.ne.u32 %q, %ti, 0;
    @%q bra HALF;
    mul.wide.u32 %b, %e, 4;
    add.u64 %o, %o, %b;
    st.global.f32 [%o], %x;
    ret;
HALF:
    mul.wide.u32 %b, %e, 2;
    add.u64 %o, %o, %b;
    cvt.rn.f16.f32 %hv, %x;
    st.global.b16 [%o], %hv;
    ret;
}
";

/// [`SUBSAMPLE`]: a thread per output channel sums its 3 × 3 window over every input channel; the block's
/// channels are one position's, normalized together.
pub(super) fn subsample() -> String {
    let taps: String = (0..3).flat_map(|kh| (0..3).map(move |kw| tap(kh, kw))).collect();
    let norm = SUBSAMPLE_NORM.replace("SUM_MEAN", &block_sum("%mean", "MEAN")).replace("SUM_VAR", &block_sum("%var", "VAR"));
    SUBSAMPLE_HEAD.replace("REDUCE", REDUCE) + &taps + &norm
}

/// [`ATTEND_AUDIO`]: the thread's query at `%q0..%q3` (scaled), an online softmax over the keys `%s = lo..=t`.
const ATTEND_HEAD: &str = ".visible .entry attend_audio(.param .u64 pqkv, .param .u64 prel, .param .u64 pscale, .param .u64 pout, .param .u32 ph, .param .u32 pleft, .param .f32 pks, .param .f32 pcap)
{
    .reg .pred %p;
    .reg .b32 %t, %lane, %head, %hh, %left, %s, %e, %h0, %h1;
    .reg .f32 %q<4>, %k<4>, %v<4>, %r<4>, %o<4>, %ks, %cap, %l, %u, %m, %mn, %c, %sum, %pw;
    .reg .b64 %qkv, %base, %a, %rel;
    mov.u32 %lane, %tid.x;
    mov.u32 %t, %ctaid.x;
    mov.u32 %head, %ctaid.y;
    ld.param.u32 %hh, [ph];
    ld.param.u32 %left, [pleft];
    ld.param.f32 %ks, [pks];
    ld.param.f32 %cap, [pcap];
    shl.b32 %e, %head, 7;
    shl.b32 %h0, %lane, 2;
    add.u32 %e, %e, %h0;
    cvt.u64.u32 %base, %e;
    ld.param.u64 %qkv, [pqkv];
    cvta.to.global.u64 %qkv, %qkv;
    ld.param.u64 %rel, [prel];
    cvta.to.global.u64 %rel, %rel;
    mul.lo.u32 %h1, %hh, 3;
    mul.wide.u32 %a, %t, %h1;
    add.u64 %a, %a, %base;
    shl.b64 %a, %a, 2;
    add.u64 %a, %qkv, %a;
    ld.global.v4.f32 {%q0, %q1, %q2, %q3}, [%a];
    ld.param.u64 %a, [pscale];
    cvta.to.global.u64 %a, %a;
    mul.wide.u32 %rel, %h0, 4;
    add.u64 %a, %a, %rel;
    ld.global.v4.f32 {%k0, %k1, %k2, %k3}, [%a];
    ld.param.u64 %rel, [prel];
    cvta.to.global.u64 %rel, %rel;
    mov.f32 %m, 0fFF800000;
    mov.f32 %sum, 0f00000000;
";

/// One key `%s` of [`ATTEND_AUDIO`]: its logit, then the online softmax's step.
const ATTEND_KEY: &str = "KEY:
    mul.lo.u32 %h1, %hh, 3;
    mul.wide.u32 %a, %s, %h1;
    add.u64 %a, %a, %base;
    cvt.u64.u32 %rel, %hh;
    add.u64 %a, %a, %rel;
    shl.b64 %a, %a, 2;
    add.u64 %a, %qkv, %a;
    ld.global.v4.f32 {%k0, %k1, %k2, %k3}, [%a];
    shl.b64 %rel, %rel, 2;
    add.u64 %a, %a, %rel;
    ld.global.v4.f32 {%v0, %v1, %v2, %v3}, [%a];
    sub.u32 %e, %t, %s;
    mul.wide.u32 %a, %e, %hh;
    add.u64 %a, %a, %base;
    shl.b64 %a, %a, 2;
    ld.param.u64 %rel, [prel];
    cvta.to.global.u64 %rel, %rel;
    add.u64 %a, %rel, %a;
    ld.global.v4.f32 {%r0, %r1, %r2, %r3}, [%a];
DOT    div.rn.f32 %l, %l, %cap;
    add.f32 %u, %l, %l;
    mul.f32 %u, %u, LOG2E;
    ex2.approx.f32 %u, %u;
    add.f32 %u, %u, 0f3F800000;
    rcp.rn.f32 %u, %u;
    fma.rn.f32 %l, %u, 0fC0000000, 0f3F800000;
    mul.f32 %l, %l, %cap;
    max.f32 %mn, %m, %l;
    sub.f32 %c, %m, %mn;
    mul.f32 %c, %c, LOG2E;
    ex2.approx.f32 %c, %c;
    sub.f32 %pw, %l, %mn;
    mul.f32 %pw, %pw, LOG2E;
    ex2.approx.f32 %pw, %pw;
    mov.f32 %m, %mn;
    fma.rn.f32 %sum, %sum, %c, %pw;
ACC    add.u32 %s, %s, 1;
    setp.le.u32 %p, %s, %t;
    @%p bra KEY;
";

/// [`ATTEND_AUDIO`]: every lane's part of the logit summed over the warp, so each lane holds it.
pub(super) fn attend_audio() -> String {
    let mut dot = String::from("    mov.f32 %l, 0f00000000;\n");
    (0..4).for_each(|e| _ = writeln!(dot, "    fma.rn.f32 %r{e}, %k{e}, %ks, %r{e};\n    fma.rn.f32 %l, %q{e}, %r{e}, %l;"));
    dot += &warp("add.f32", "%l", "%u");
    let acc = (0..4).fold(String::new(), |mut s, e| {
        let _ = writeln!(s, "    mul.f32 %o{e}, %o{e}, %c;\n    fma.rn.f32 %o{e}, %pw, %v{e}, %o{e};");
        s
    });
    let mut s = String::from(ATTEND_HEAD);
    (0..4).for_each(|e| _ = writeln!(s, "    mul.f32 %q{e}, %q{e}, %k{e};\n    mov.f32 %o{e}, 0f00000000;"));
    s += "    sub.u32 %s, %t, %left;\n    setp.gt.u32 %p, %left, %t;\n    @%p mov.u32 %s, 0;\n";
    s += &ATTEND_KEY.replace("DOT", &dot).replace("ACC", &acc).replace("LOG2E", LOG2E);
    s += "    rcp.rn.f32 %sum, %sum;\n";
    (0..4).for_each(|e| _ = writeln!(s, "    mul.f32 %o{e}, %o{e}, %sum;"));
    s + "    cvt.rn.f16x2.f32 %h0, %o1, %o0;
    cvt.rn.f16x2.f32 %h1, %o3, %o2;
    mul.wide.u32 %a, %t, %hh;
    add.u64 %a, %a, %base;
    shl.b64 %a, %a, 1;
    ld.param.u64 %qkv, [pout];
    cvta.to.global.u64 %qkv, %qkv;
    add.u64 %a, %qkv, %a;
    st.global.v2.b32 [%a], {%h0, %h1};
    ret;
}
"
}

/// [`LCONV`]'s start: the thread's four channels' convolution into `%x0..%x3`.
const LCONV_HEAD: &str =
    ".visible .entry lconv(.param .u64 px, .param .u64 pw, .param .u64 pg, .param .u64 pout, .param .u32 ph, .param .u32 ptaps, .param .f32 peps)
{
    .reg .pred %q;
    .reg .b16 %hv<4>;
    .reg .b32 %t, %c, %hh, %taps, %j, %e, %h0, %h1;
    .reg .f32 %x<4>, %w<4>, %g<4>, %s, %r, %eps, %nh, %y;
    .reg .b64 %in, %wt, %a, %b;
REDUCE    mov.u32 %t, %ctaid.x;
    mov.u32 %c, %tid.x;
    shl.b32 %c, %c, 2;
    ld.param.u32 %hh, [ph];
    ld.param.u32 %taps, [ptaps];
    ld.param.f32 %eps, [peps];
    cvt.rn.f32.u32 %nh, %hh;
    ld.param.u64 %in, [px];
    cvta.to.global.u64 %in, %in;
    ld.param.u64 %wt, [pw];
    cvta.to.global.u64 %wt, %wt;
    mov.f32 %x0, 0f00000000;\n    mov.f32 %x1, 0f00000000;\n    mov.f32 %x2, 0f00000000;\n    mov.f32 %x3, 0f00000000;
    mov.u32 %j, 0;
TAP:
    add.u32 %e, %t, %j;
    add.u32 %e, %e, 1;
    setp.lt.u32 %q, %e, %taps;
    @%q bra NEXT;
    sub.u32 %e, %e, %taps;
    mad.lo.u32 %e, %e, %hh, %c;
    mul.wide.u32 %a, %e, 2;
    add.u64 %a, %in, %a;
    ld.global.v4.b16 {%hv0, %hv1, %hv2, %hv3}, [%a];
";

/// [`LCONV`]'s end: the RMS norm with its weight, the `SiLU`, the store.
const LCONV_TAIL: &str = "NEXT:
    add.u32 %j, %j, 1;
    setp.lt.u32 %q, %j, %taps;
    @%q bra TAP;
SUMS    ld.param.u64 %a, [pg];
    cvta.to.global.u64 %a, %a;
    mul.wide.u32 %b, %c, 4;
    add.u64 %a, %a, %b;
    ld.global.v4.f32 {%g0, %g1, %g2, %g3}, [%a];
ACT    cvt.rn.f16x2.f32 %h0, %x1, %x0;
    cvt.rn.f16x2.f32 %h1, %x3, %x2;
    mad.lo.u32 %e, %t, %hh, %c;
    mul.wide.u32 %a, %e, 2;
    ld.param.u64 %b, [pout];
    cvta.to.global.u64 %b, %b;
    add.u64 %a, %b, %a;
    st.global.v2.b32 [%a], {%h0, %h1};
    ret;
}
";

/// [`LCONV`]: tap `j` reads row `t + 1 + j − taps`, channel `c`'s weight `w[c·taps + j]`.
pub(super) fn lconv() -> String {
    let mut s = LCONV_HEAD.replace("REDUCE", REDUCE);
    for e in 0..4 {
        let _ = writeln!(
            s,
            "    add.u32 %e, %c, {e};\n    mad.lo.u32 %e, %e, %taps, %j;\n    mul.wide.u32 %b, %e, 4;\n    add.u64 %b, %wt, %b;\n    ld.global.f32 %w{e}, [%b];\n    cvt.f32.f16 %y, %hv{e};\n    fma.rn.f32 %x{e}, %y, %w{e}, %x{e};"
        );
    }
    let act = (0..4).fold(String::new(), |mut a, e| {
        let _ = writeln!(
            a,
            "    mul.f32 %x{e}, %x{e}, %r;\n    mul.f32 %x{e}, %x{e}, %g{e};\n    mul.f32 %y, %x{e}, {LOG2E_NEG};\n    ex2.approx.f32 %y, %y;\n    add.f32 %y, %y, 0f3F800000;\n    div.rn.f32 %x{e}, %x{e}, %y;"
        );
        a
    });
    s + &LCONV_TAIL.replace("SUMS", &rms("%x", "SUMX")).replace("ACT", &act)
}
