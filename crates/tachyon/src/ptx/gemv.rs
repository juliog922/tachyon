//! [`GEMV_Q4`], the decode matrix-vector product: Q4 weights times Q8 activations, `f32` out.
//!
//! One warp per row, [`GEMV_ROWS`] rows per block. Each lane takes chunks of 32 weights (16 bytes)
//! `lane, lane + 32, …`, with [`UNROLL`] chunks in flight per loop turn and the rest in one final round whose
//! loads are predicated, so even a short row issues all its loads at once. Each chunk is 8 `dp4a` integer dot
//! products, then one scale step in `f32`.
//!
//! [`GEMV_Q4_NARROW`] is the same product for rows of at most 256 weights (8 chunks): 8 lanes per row, so a warp
//! takes 4 rows and a block 16, and a matrix of such rows runs in a quarter of the blocks. Its rows are a multiple
//! of 4. Launch `rows.div_ceil(16)` blocks of 128 threads.
//!
//! Three variants end with a step that needs the whole product, run by the block that finishes last (each block
//! counts itself in on `count`, a zero word it leaves at zero), so the step costs no launch of its own:
//! - [`GEMV_Q4_GEGLU`] then writes Q8 of `gelu(y)·b` over its rows, as [`super::GEGLU_Q8`]. Extra arguments:
//!   `count, b: u64`, Q8 out. Rows a multiple of 32.
//! - [`GEMV_Q4_NORM`] and [`GEMV_Q4_NARROW_NORM`] then take [`super::NORM_Q8`]'s step with `y` the product:
//!   `h_out = (h + w1 ⊙ rmsnorm(y))·scale`, then Q8 of `w2 ⊙ rmsnorm(h_out)` (of `h_out` when `w2` is 0). Extra
//!   arguments: `count, h, h_out, w1, w2: u64`, Q8 out, `eps, scale: f32`. Rows a multiple of 128, at most
//!   [`NORM_ROWS`].

use super::{BLOCK, Q8, block, q8};
use std::fmt::Write;

/// Name of the Q4 × Q8 matrix-vector product.
pub const GEMV_Q4: &str = "gemv_q4";
/// Rows of [`GEMV_Q4`] per block, one warp each.
pub const GEMV_ROWS: u32 = 4;
/// Name of [`GEMV_Q4`] for rows of at most 256 weights.
pub const GEMV_Q4_NARROW: &str = "gemv_q4_narrow";
/// Chunks of 32 weights each [`GEMV_Q4`] lane has in flight per loop turn.
pub const UNROLL: usize = 4;
/// Name of [`GEMV_Q4`] followed by the gated GELU of its rows.
pub const GEMV_Q4_GEGLU: &str = "gemv_q4_geglu";
/// Name of [`GEMV_Q4`] followed by the residual step of [`super::NORM_Q8`].
pub const GEMV_Q4_NORM: &str = "gemv_q4_norm";
/// Name of [`GEMV_Q4_NARROW`] followed by the residual step of [`super::NORM_Q8`].
pub const GEMV_Q4_NARROW_NORM: &str = "gemv_q4_narrow_norm";
/// Most rows of [`GEMV_Q4_NORM`]: its last block of 128 threads holds 24 values per thread.
pub const NORM_ROWS: u32 = 128 * PER as u32;
/// Values per thread of the norm step.
const PER: usize = 24;

/// What a product does after its rows: nothing, the gated GELU, or the residual step.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Tail {
    None,
    Geglu,
    Norm,
}

/// Registers: `%rd0..3` walk this lane's weights, scales, Q8 values and Q8 scales; `%r10` is its chunk index,
/// `%r7` the row's chunk count, `%r4` the row, `%r1` the lane.
const GEMV_HEAD: &str = "
    mov.u32 %r0, %tid.x;
    and.b32 %r1, %r0, LMASK;
    shr.u32 %r2, %r0, LSHIFT;
    mov.u32 %r3, %ctaid.x;
    mad.lo.u32 %r4, %r3, ROWS, %r2;
    ld.param.u32 %r5, [prows];
    setp.ge.u32 %p, %r4, %r5;
    @%p bra DONE;
    ld.param.u32 %r6, [pcols];
    shr.u32 %r7, %r6, 5;
    ld.param.u64 %rd0, [pw];
    ld.param.u64 %rd1, [pws];
    ld.param.u64 %rd2, [pq];
    ld.param.u64 %rd3, [ps];
    cvta.to.global.u64 %rd0, %rd0;
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;
    cvta.to.global.u64 %rd3, %rd3;
    shr.u32 %r8, %r6, 1;
    mul.wide.u32 %rd4, %r4, %r8;
    add.u64 %rd0, %rd0, %rd4;
    mul.wide.u32 %rd4, %r4, %r7;
    add.u64 %rd1, %rd1, %rd4;
    mul.wide.u32 %rd4, %r1, 16;
    add.u64 %rd0, %rd0, %rd4;
    shr.u32 %r9, %r1, 1;
    mul.wide.u32 %rd4, %r9, 2;
    add.u64 %rd1, %rd1, %rd4;
    mul.wide.u32 %rd4, %r1, 32;
    add.u64 %rd2, %rd2, %rd4;
    mul.wide.u32 %rd4, %r1, 8;
    add.u64 %rd3, %rd3, %rd4;
    mov.u32 %r10, %r1;
    mov.f32 %sum, 0f00000000;
";

const GEMV_TAIL: &str = "
    setp.ne.u32 %p, %r1, 0;
    @%p bra DONE;
    ld.param.u64 %rd5, [py];
    cvta.to.global.u64 %rd5, %rd5;
    mul.wide.u32 %rd6, %r4, 4;
    add.u64 %rd5, %rd5, %rd6;
    st.global.f32 [%rd5], %sum;
FENCE
DONE:
";

/// Counts the block in, with fences on both sides (each row's lane 0 fenced its store), and lets only the last
/// block of the grid go on, its `%r5` still the rows.
const COUNT: &str = "
    bar.sync 0;
    setp.ne.u32 %p, %r0, 0;
    @%p bra COUNTED;
    ld.param.u64 %rd5, [pcount];
    cvta.to.global.u64 %rd5, %rd5;
    mov.u32 %r11, %nctaid.x;
    sub.u32 %r11, %r11, 1;
    fence.acq_rel.gpu;
    atom.global.add.u32 %r12, [%rd5], 1;
    setp.eq.u32 %last, %r12, %r11;
    @%last st.global.u32 [%rd5], 0;
    fence.acq_rel.gpu;
    selp.u32 %r12, 1, 0, %last;
    st.shared.u32 [tflag], %r12;
COUNTED:
    bar.sync 0;
    ld.shared.u32 %r12, [tflag];
    setp.eq.u32 %p, %r12, 0;
    @%p bra EXIT;
    ld.param.u64 %rd5, [py];
    cvta.to.global.u64 %rd5, %rd5;
    ld.param.u64 %qp, [pqo];
    cvta.to.global.u64 %qp, %qp;
    ld.param.u64 %sp, [pso];
    cvta.to.global.u64 %sp, %sp;
";

/// The gated GELU of [`GEMV_Q4_GEGLU`]: thread `i` takes rows `i, i + 128, …`.
fn geglu() -> String {
    format!(
        "    ld.param.u64 %rd6, [pb];\n    cvta.to.global.u64 %rd6, %rd6;\n    mov.u32 %r10, %r0;
GATE:
    setp.ge.u32 %p, %r10, %r5;
    @%p bra EXIT;
    mul.wide.u32 %rd7, %r10, 4;
    add.u64 %rd4, %rd5, %rd7;
    ld.global.cg.f32 %x, [%rd4];
    add.u64 %rd4, %rd6, %rd7;
    ld.global.f32 %t, [%rd4];
    mul.f32 %u, %x, %x;
    fma.rn.f32 %u, %u, 0f3D372713, 0f3F800000;
    mul.f32 %u, %u, %x;
    mul.f32 %u, %u, 0fC0135761;
    ex2.approx.f32 %u, %u;
    add.f32 %u, %u, 0f3F800000;
    div.rn.f32 %x, %x, %u;
    mul.f32 %x, %x, %t;
{}    add.u32 %r10, %r10, 128;
    bra GATE;
",
        q8("%x", "%r10")
    )
}

/// The residual step of [`GEMV_Q4_NORM`] in one block, as [`super::NORM_Q8`] takes it: value `k` of thread `i` is
/// row `i + 128k`, `%h`, `%y` and `%a` (`w1 ⊙ y`) loaded at once, the four sums reduced together.
fn norm() -> String {
    let mut s = String::from(
        "    ld.param.f32 %eps, [peps];\n    ld.param.f32 %scale, [pscale];\n    cvt.rn.f32.u32 %n, %r5;
    ld.param.u64 %hp, [ph];\n    cvta.to.global.u64 %hp, %hp;\n    ld.param.u64 %ho, [pho];\n    cvta.to.global.u64 %ho, %ho;
    ld.param.u64 %wa, [pw1];\n    cvta.to.global.u64 %wa, %wa;\n    ld.param.u64 %wb, [pw2];\n    setp.ne.u64 %hw, %wb, 0;\n    cvta.to.global.u64 %wb, %wb;
    mul.wide.u32 %rd7, %r0, 4;\n    add.u64 %hp, %hp, %rd7;\n    add.u64 %ho, %ho, %rd7;\n    add.u64 %rd5, %rd5, %rd7;\n    add.u64 %wa, %wa, %rd7;\n    add.u64 %wb, %wb, %rd7;
",
    );
    for k in 0..PER {
        let at = 512 * k;
        let _ = writeln!(
            s,
            "    add.u32 %r10, %r0, {};\n    setp.lt.u32 %v{k}, %r10, %r5;\n    mov.f32 %h{k}, 0f00000000;\n    mov.f32 %y{k}, 0f00000000;",
            128 * k
        );
        let _ = writeln!(s, "    mov.f32 %a{k}, 0f00000000;\n    @%v{k} ld.global.f32 %h{k}, [%hp+{at}];\n    @%v{k} ld.global.cg.f32 %y{k}, [%rd5+{at}];");
        let _ = writeln!(s, "    @%v{k} ld.global.f32 %a{k}, [%wa+{at}];");
    }
    (0..PER).for_each(|k| _ = writeln!(s, "    mul.f32 %a{k}, %a{k}, %y{k};"));
    for (sum, a, b) in [("ss", "y", "y"), ("sh", "h", "h"), ("su", "h", "a"), ("uu", "a", "a")] {
        let _ = writeln!(s, "    mul.f32 %{sum}, %{a}0, %{b}0;");
        (1..PER).for_each(|k| _ = writeln!(s, "    fma.rn.f32 %{sum}, %{a}{k}, %{b}{k}, %{sum};"));
    }
    s += &block("add.f32", &["%ss", "%sh", "%su", "%uu"], 4);
    s += "    div.rn.f32 %r, %ss, %n;\n    add.f32 %r, %r, %eps;\n    rsqrt.approx.f32 %r, %r;\n";
    for k in 0..PER {
        let _ = writeln!(s, "    fma.rn.f32 %h{k}, %a{k}, %r, %h{k};\n    mul.f32 %h{k}, %h{k}, %scale;\n    @%v{k} st.global.f32 [%ho+{}], %h{k};", 512 * k);
    }
    s += "    add.f32 %su, %su, %su;\n    fma.rn.f32 %su, %r, %uu, %su;\n    fma.rn.f32 %sh, %r, %su, %sh;\n    mul.f32 %sh, %sh, %scale;\n    mul.f32 %sh, %sh, %scale;\n";
    s += "    div.rn.f32 %r, %sh, %n;\n    add.f32 %r, %r, %eps;\n    rsqrt.approx.f32 %r, %r;\n    @!%hw mov.f32 %r, 0f3F800000;\n";
    for k in 0..PER {
        // Rows are a multiple of 128, so a value past them is past them in every thread: the warps stay whole.
        let _ = writeln!(s, "    @!%v{k} bra EXIT;\n    mul.f32 %x, %h{k}, %r;\n    mov.f32 %t, 0f3F800000;\n    @%hw ld.global.f32 %t, [%wb+{}];", 512 * k);
        let _ = writeln!(s, "    mul.f32 %x, %x, %t;\n    add.u32 %r10, %r0, {};", 128 * k);
        s += &q8("%x", "%r10");
    }
    s
}

/// Loads chunk `i` of the current turn: 16 bytes of weights, its scale, 32 Q8 values and their scales. A guarded
/// load runs only under predicate `%q{i}`, after zeroing what makes a missing chunk add nothing.
fn load(ptx: &mut String, i: usize, guarded: bool, lanes: usize) {
    let guard = if guarded {
        let _ = writeln!(ptx, "    mov.f32 %sx{i}, 0f00000000;\n    mov.f32 %ss{i}, 0f00000000;\n    mov.b16 %h{i}, 0;");
        format!("@%q{i} ")
    } else {
        String::new()
    };
    let (wr, ar, l) = (4 * i, 8 * i, lanes * i);
    let _ = writeln!(ptx, "    {guard}ld.global.cs.v4.u32 {{%w{wr}, %w{}, %w{}, %w{}}}, [%rd0+{}];", wr + 1, wr + 2, wr + 3, 16 * l);
    let _ = writeln!(ptx, "    {guard}ld.global.nc.v4.u32 {{%a{ar}, %a{}, %a{}, %a{}}}, [%rd2+{}];", ar + 1, ar + 2, ar + 3, 32 * l);
    let _ = writeln!(ptx, "    {guard}ld.global.nc.v4.u32 {{%a{}, %a{}, %a{}, %a{}}}, [%rd2+{}];", ar + 4, ar + 5, ar + 6, ar + 7, 32 * l + 16);
    let _ = writeln!(ptx, "    {guard}ld.global.nc.v2.f32 {{%sx{i}, %ss{i}}}, [%rd3+{}];", 8 * l);
    let _ = writeln!(ptx, "    {guard}ld.global.nc.b16 %h{i}, [%rd1+{l}];");
}

/// Adds chunk `i` to `%sum`: `d · (s · Σ n·q − 8 · s·Σq)`, the integer sum from 8 `dp4a`.
fn dot(ptx: &mut String, i: usize) {
    for word in 0..4 {
        let (wr, ar) = (4 * i + word, 8 * i + 2 * word);
        let acc = if word == 0 { "0".to_string() } else { format!("%acc{i}") };
        let _ = writeln!(ptx, "    and.b32 %r12, %w{wr}, 0x0F0F0F0F;\n    dp4a.u32.s32 %acc{i}, %r12, %a{ar}, {acc};");
        let _ = writeln!(ptx, "    shr.u32 %r13, %w{wr}, 4;\n    and.b32 %r13, %r13, 0x0F0F0F0F;\n    dp4a.u32.s32 %acc{i}, %r13, %a{}, %acc{i};", ar + 1);
    }
    let _ = writeln!(ptx, "    cvt.f32.f16 %sw{i}, %h{i};\n    cvt.rn.f32.s32 %f{i}, %acc{i};\n    mul.f32 %f{i}, %f{i}, %sx{i};");
    let _ = writeln!(ptx, "    fma.rn.f32 %f{i}, %ss{i}, 0fC1000000, %f{i};\n    fma.rn.f32 %sum, %f{i}, %sw{i}, %sum;");
}

/// The extra parameters and registers of `tail`.
fn extra(tail: Tail) -> (&'static str, String) {
    let regs = format!(
        "{Q8}{BLOCK}
    .shared .align 4 .u32 tflag;
    .reg .pred %last, %hw, %v<{PER}>;
    .reg .f32 %x, %t, %u, %r, %n, %eps, %scale, %ss, %su, %uu, %hh<{PER}>, %y<{PER}>, %aa<{PER}>;
    .reg .b64 %hp, %ho, %wa, %wb;"
    );
    match tail {
        Tail::None => ("", String::new()),
        Tail::Geglu => (", .param .u64 pcount, .param .u64 pb, .param .u64 pqo, .param .u64 pso", regs),
        Tail::Norm => (
            ", .param .u64 pcount, .param .u64 ph, .param .u64 pho, .param .u64 pw1, .param .u64 pw2, .param .u64 pqo, .param .u64 pso, .param .f32 peps, .param .f32 pscale",
            regs,
        ),
    }
}

/// The kernel of `lanes` per row ending with `tail`.
fn name(lanes: usize, tail: Tail) -> &'static str {
    match (lanes, tail) {
        (32, Tail::None) => GEMV_Q4,
        (32, Tail::Geglu) => GEMV_Q4_GEGLU,
        (32, Tail::Norm) => GEMV_Q4_NORM,
        (_, Tail::Norm) => GEMV_Q4_NARROW_NORM,
        _ => GEMV_Q4_NARROW,
    }
}

/// [`GEMV_Q4`] with `u` chunks in flight per loop turn, or with `lanes` 8, [`GEMV_Q4_NARROW`]; with `tail`, the
/// variant that ends with it.
pub(super) fn gemv_q4(u: usize, lanes: usize, tail: Tail) -> String {
    let (name, (params, regs)) = (name(lanes, tail), extra(tail));
    let mut s = format!(
        ".visible .entry {name}(.param .u64 pw, .param .u64 pws, .param .u64 pq, .param .u64 ps, .param .u64 py, .param .u32 prows, .param .u32 pcols{params}){}
{{
    .reg .pred %p, %q<{u}>;
    .reg .b16 %h<{u}>;
    .reg .b32 %r<16>, %w<{}>, %a<{}>, %acc<{u}>;
    .reg .f32 %sx<{u}>, %ss<{u}>, %sw<{u}>, %f<{u}>, %sum, %o, %sh;
    .reg .b64 %rd<8>;{regs}",
        // The tails' registers would cost the product blocks per multiprocessor: they spill instead, once a call.
        if tail == Tail::None { "" } else { "\n    .maxnreg 72" },
        4 * u,
        8 * u
    );
    let head = GEMV_HEAD.replace("LMASK", &(lanes - 1).to_string()).replace("LSHIFT", &lanes.trailing_zeros().to_string());
    s += &head.replace("ROWS", &(128 / lanes).to_string());
    let _ = writeln!(s, "TURN:\n    add.u32 %r11, %r10, {};\n    setp.ge.u32 %p, %r11, %r7;\n    @%p bra LAST;", lanes * (u - 1));
    (0..u).for_each(|i| load(&mut s, i, false, lanes));
    (0..u).for_each(|i| dot(&mut s, i));
    let l = lanes * u;
    let _ = writeln!(s, "    add.u64 %rd0, %rd0, {};\n    add.u64 %rd1, %rd1, {l};\n    add.u64 %rd2, %rd2, {};", 16 * l, 32 * l);
    let _ = writeln!(s, "    add.u64 %rd3, %rd3, {};\n    add.u32 %r10, %r10, {l};\n    bra TURN;\nLAST:", 8 * l);
    for i in 0..u {
        let _ = writeln!(s, "    add.u32 %r11, %r10, {};\n    setp.lt.u32 %q{i}, %r11, %r7;", lanes * i);
        load(&mut s, i, true, lanes);
    }
    (0..u).for_each(|i| dot(&mut s, i));
    let mut k = lanes / 2;
    while k > 0 {
        let _ = writeln!(s, "    shfl.sync.bfly.b32 %o, %sum, {k}, 31, -1;\n    add.f32 %sum, %sum, %o;");
        k /= 2;
    }
    s += &GEMV_TAIL.replace("FENCE\n", if tail == Tail::None { "" } else { "    fence.acq_rel.gpu;\n" });
    s += &match tail {
        Tail::None => String::new(),
        Tail::Geglu => COUNT.to_string() + &geglu(),
        Tail::Norm => {
            COUNT.to_string() + &norm().replace("%h", "%hh").replace("%a", "%aa").replace("%hhp", "%hp").replace("%hho", "%ho").replace("%hhw", "%hw")
        }
    };
    s + "EXIT:\n    ret;\n}\n"
}
