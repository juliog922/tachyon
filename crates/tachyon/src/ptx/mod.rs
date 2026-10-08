//! The GPU kernels, generated as PTX text and compiled by the driver for the GPU at hand.
//!
//! [`module`] targets `sm_80` with PTX ISA 7.1, so one text runs on every supported GPU and driver; the driver
//! compiles it to machine code once and caches it. Generating the text lets one template serve every unroll
//! depth and head size without the CUDA Toolkit.
//!
//! Every kernel is listed with its arguments, in order, and its launch shape. Pointers are device addresses
//! (`u64`); "Q8 out" means two pointers, `q: *mut i8` and `s: *mut [f32; 2]`, in the layout of [`crate::quant`].
//! A decode step runs, per layer: [`NORM_Q8`] → [`GEMV_Q4`] (query, key, value) → [`ATTEND_256`] or
//! [`ATTEND_512`] → [`GEMV_Q4`] (output) → [`NORM_Q8`] → [`GEMV_Q4`] (gate and up) → [`GEGLU_Q8`] →
//! [`GEMV_Q4`] (down) → [`NORM_Q8`] → … Every activation a GEMV reads is written as Q8 by the kernel before it.
//!
//! - [`QUANT_Q8`] quantizes an `f32` vector to Q8. Arguments: `x`, Q8 out, `len: u32`.
//!   Launch `len.div_ceil(256)` blocks of 256.
//! - [`GEGLU_Q8`] quantizes `gelu(a[i]) · b[i]`, GELU in its tanh form, to Q8. Arguments: `a`, `b`, Q8 out,
//!   `len: u32`. Launch as [`QUANT_Q8`].
//! - [`NORM_Q8`] is the residual step: when `y` is not null, `h' = (h + w1 ⊙ rmsnorm(y)) · scale` is written to
//!   `h_out` (another buffer: every block reads all of `h`); then Q8 out of `w2 ⊙ rmsnorm(h')`, or of `h'` when
//!   `w2` is null, where `h'` is `h` when `y` is null; no Q8 when `q` is null. `rmsnorm(x) = x / √(mean(x²) + eps)`.
//!   Arguments: `h, h_out, y, w1, w2: *f32`, Q8 out, `len: u32, eps: f32, scale: f32`. `len` is a multiple of
//!   [`NORM_THREADS`] up to 16 times it; vectors may hold several chunks of `len`, each normalized on its own with
//!   the same weights. Launch a grid of `(len / NORM_THREADS, chunks)` blocks of [`NORM_THREADS`]: every block
//!   reduces its whole chunk and writes one slice of it, so a vector spreads over several multiprocessors.
//! - [`EMBED_Q4`] dequantizes row `*token` of a Q4 table: `out[i] = w[token, i] · scale`. Arguments:
//!   `packed, scales, token: *const u32, out, cols: u32, scale: f32`. The table may be pinned host memory, read
//!   over PCIe. Launch `cols.div_ceil(256)` blocks of 256.
//! - [`GEMV_Q4`]: see [`gemv`]. [`ATTEND_256`], [`ATTEND_512`]: see [`attend`]. [`SAMPLE`]: see [`sample`].
//!
//! Lengths are multiples of 32.

pub mod attend;
pub mod gemv;
pub mod sample;

pub use attend::{ATTEND_256, ATTEND_512, ATTEND_WARPS, HEADS_PER_KV, rope};
pub use gemv::{GEMV_Q4, GEMV_ROWS, UNROLL};
pub use sample::{SAMPLE, SAMPLE_THREADS, Sampling};

use std::fmt::Write;

/// Name of the activation quantizer.
pub const QUANT_Q8: &str = "quant_q8";
/// Name of the GELU-gated quantizer.
pub const GEGLU_Q8: &str = "geglu_q8";
/// Name of the residual and normalization step.
pub const NORM_Q8: &str = "norm_q8";
/// Name of the embedding lookup.
pub const EMBED_Q4: &str = "embed_q4";
/// Threads per block of [`NORM_Q8`].
pub const NORM_THREADS: u32 = 256;

/// Every kernel, as one PTX module.
pub fn module() -> String {
    let kernels = [quant(QUANT_Q8, ""), quant(GEGLU_Q8, GELU), norm(), EMBED.into(), gemv::gemv_q4(UNROLL), attend::attend(256), attend::attend(512)];
    kernels.iter().fold(String::from(".version 7.1\n.target sm_80\n.address_size 64\n"), |m, k| m + "\n" + k) + &sample::sample()
}

/// Five butterfly steps that leave `op` of `x` over the warp in every lane; `t` is scratch.
fn warp(op: &str, x: &str, t: &str) -> String {
    (0..5).fold(String::new(), |mut s, k| {
        let _ = writeln!(s, "    shfl.sync.bfly.b32 {t}, {x}, {}, 31, -1;\n    {op} {x}, {x}, {t};", 16 >> k);
        s
    })
}

/// The sums (`add.f32`) or another `op` of each of `xs` (up to four) over a block of `warps` warps, left in every
/// thread; needs [`BLOCK`]. With fewer than 32 warps the missing ones count as 0, so `op` must be a sum.
fn block(op: &str, xs: &[&str], warps: u32) -> String {
    let lanes: String = xs.iter().map(|x| warp(op, x, "%bt")).collect();
    let mut s = lanes.clone();
    xs.iter().enumerate().for_each(|(i, x)| _ = writeln!(s, "    @%lane0 st.shared.f32 [%rw+{}], {x};", 128 * i));
    s += "    bar.sync 0;\n";
    if warps < 32 {
        let _ = writeln!(s, "    setp.lt.u32 %bp, %lane, {warps};");
    }
    for (i, x) in xs.iter().enumerate() {
        let guard = if warps < 32 { format!("    mov.f32 {x}, 0f00000000;\n    @%bp ") } else { "    ".into() };
        let _ = writeln!(s, "{guard}ld.shared.f32 {x}, [%rl+{}];", 128 * i);
    }
    s + &lanes + "    bar.sync 0;\n"
}

/// Thread, lane and warp, and the shared slots [`block`] reduces through.
const BLOCK: &str = "
    .shared .align 4 .f32 red[128];
    .reg .pred %lane0, %bp;
    .reg .b32 %th, %lane, %warp, %rw, %rl;
    .reg .f32 %bt;
    mov.u32 %th, %tid.x;
    and.b32 %lane, %th, 31;
    shr.u32 %warp, %th, 5;
    setp.eq.u32 %lane0, %lane, 0;
    mov.u32 %rw, red;
    mad.lo.u32 %rl, %lane, 4, %rw;
    mad.lo.u32 %rw, %warp, 4, %rw;
";

/// Registers of [`q8`]; `%qp` and `%sp` hold the Q8 output pointers.
const Q8: &str = "
    .reg .pred %qz;
    .reg .b32 %qn, %qm, %qt;
    .reg .f32 %qa, %qb, %qd, %qi, %qv, %qf;
    .reg .b64 %qo, %qp, %sp;
";

/// Stores `x`, value `i` of a vector, as Q8. Each warp holds 32 consecutive values, one block of the layout:
/// scale `amax/127`, values rounded to nearest even, and the block's `[s, s·Σq]` written by its first lane.
fn q8(x: &str, i: &str) -> String {
    format!(
        "    abs.f32 %qa, {x};\n{}    div.rn.f32 %qd, %qa, 0f42FE0000;
    setp.gt.f32 %qz, %qa, 0f00000000;
    div.rn.f32 %qi, 0f42FE0000, %qa;
    selp.f32 %qi, %qi, 0f00000000, %qz;
    mul.rn.f32 %qv, {x}, %qi;
    cvt.rni.s32.f32 %qn, %qv;
    cvt.u64.u32 %qo, {i};
    add.u64 %qo, %qp, %qo;
    st.global.s8 [%qo], %qn;\n{}    and.b32 %qt, {i}, 31;
    setp.eq.u32 %qz, %qt, 0;
    cvt.rn.f32.s32 %qf, %qn;
    mul.rn.f32 %qf, %qf, %qd;
    shr.u32 %qt, {i}, 5;
    mul.wide.u32 %qo, %qt, 8;
    add.u64 %qo, %sp, %qo;
    @%qz st.global.v2.f32 [%qo], {{%qd, %qf}};\n",
        warp("max.f32", "%qa", "%qb"),
        warp("add.s32", "%qn", "%qm")
    )
}

/// One value per thread: `%x = a[i]`, with `%qo` the byte offset of `i`; `EXIT` for threads past the end.
const QUANT: &str = "
    .reg .pred %p;
    .reg .b32 %i, %n, %len;
    .reg .f32 %x, %b, %t;
    .reg .b64 %a;
    mov.u32 %i, %ctaid.x;
    mov.u32 %n, %ntid.x;
    mov.u32 %len, %tid.x;
    mad.lo.u32 %i, %i, %n, %len;
    ld.param.u32 %len, [plen];
    setp.ge.u32 %p, %i, %len;
    @%p bra EXIT;
    ld.param.u64 %qp, [pq];
    ld.param.u64 %sp, [ps];
    cvta.to.global.u64 %qp, %qp;
    cvta.to.global.u64 %sp, %sp;
    ld.param.u64 %a, [px];
    cvta.to.global.u64 %a, %a;
    mul.wide.u32 %qo, %i, 4;
    add.u64 %a, %a, %qo;
    ld.global.f32 %x, [%a];
";

/// `%x ← gelu(%x) · b[i]`, with `gelu(x) = x / (1 + e^(−2u))`, `u = √(2/π)·(x + 0.044715·x³)`: the tanh form.
const GELU: &str = "
    ld.param.u64 %a, [pb];
    cvta.to.global.u64 %a, %a;
    add.u64 %a, %a, %qo;
    ld.global.f32 %b, [%a];
    mul.f32 %t, %x, %x;
    fma.rn.f32 %t, %t, 0f3D372713, 0f3F800000;
    mul.f32 %t, %t, %x;
    mul.f32 %t, %t, 0fC0135761;
    ex2.approx.f32 %t, %t;
    add.f32 %t, %t, 0f3F800000;
    div.rn.f32 %x, %x, %t;
    mul.f32 %x, %x, %b;
";

/// [`QUANT_Q8`], or with [`GELU`] as `gate`, [`GEGLU_Q8`].
fn quant(name: &str, gate: &str) -> String {
    let b = if gate.is_empty() { "" } else { ".param .u64 pb, " };
    format!(
        ".visible .entry {name}(.param .u64 px, {b}.param .u64 pq, .param .u64 ps, .param .u32 plen)\n{{{Q8}{QUANT}{gate}{}EXIT:\n    ret;\n}}\n",
        q8("%x", "%i")
    )
}

/// Values of [`NORM_Q8`] each thread reads: rows of up to `16 · NORM_THREADS`.
const PER_THREAD: usize = 16;

/// [`NORM_Q8`]. Every block reads its whole chunk, `LOADS` putting value `k` of this thread in `%h{k}`, `%y{k}`
/// and `%a{k}` (`w1 ⊙ y`), and `SUM` reduces Σy², Σh², Σh·a and Σa² at once: the new `h` is `(h + r·a)·scale`,
/// so its squares sum to `scale²·(Σh² + 2r·Σh·a + r²·Σa²)`. Each block stores only its own slice of the new `h` and
/// of the Q8 output.
const NORM: &str = "
.visible .entry norm_q8(.param .u64 ph, .param .u64 pho, .param .u64 py, .param .u64 pw1, .param .u64 pw2, .param .u64 pq, .param .u64 ps,
                        .param .u32 plen, .param .f32 peps, .param .f32 pscale)
{
    .reg .pred %hy, %hw, %hq, %t, %p<16>;
    .reg .b32 %len, %base, %slice, %own, %gi;
    .reg .f32 %ss, %sh, %su, %uu, %r, %x, %w, %eps, %scale, %n, %h<16>, %y<16>, %a<16>;
    .reg .b64 %hp, %ho, %yp, %w1, %w2, %ah, %ao, %ay, %aw, %ad;
BLOCK Q8
    ld.param.u32 %len, [plen];
    ld.param.f32 %eps, [peps];
    ld.param.f32 %scale, [pscale];
    cvt.rn.f32.u32 %n, %len;
    mov.u32 %slice, %ctaid.x;
    mov.u32 %base, %ctaid.y;
    mul.lo.u32 %base, %base, %len;
    mad.lo.u32 %own, %slice, 256, %th;
    mul.wide.u32 %ad, %base, 4;
    ld.param.u64 %hp, [ph];
    cvta.to.global.u64 %hp, %hp;
    add.u64 %hp, %hp, %ad;
    ld.param.u64 %ho, [pho];
    cvta.to.global.u64 %ho, %ho;
    add.u64 %ho, %ho, %ad;
    ld.param.u64 %yp, [py];
    setp.ne.u64 %hy, %yp, 0;
    cvta.to.global.u64 %yp, %yp;
    add.u64 %yp, %yp, %ad;
    ld.param.u64 %w1, [pw1];
    cvta.to.global.u64 %w1, %w1;
    ld.param.u64 %w2, [pw2];
    setp.ne.u64 %hw, %w2, 0;
    cvta.to.global.u64 %w2, %w2;
    ld.param.u64 %qp, [pq];
    setp.ne.u64 %hq, %qp, 0;
    cvta.to.global.u64 %qp, %qp;
    ld.param.u64 %sp, [ps];
    cvta.to.global.u64 %sp, %sp;
    mul.wide.u32 %ad, %th, 4;
    add.u64 %ah, %hp, %ad;
    add.u64 %ao, %ho, %ad;
    add.u64 %ay, %yp, %ad;
    add.u64 %aw, %w1, %ad;
LOADS
SUM
    div.rn.f32 %r, %ss, %n;
    add.f32 %r, %r, %eps;
    rsqrt.approx.f32 %r, %r;
    @!%hy bra ADDED;
RESIDUAL
    add.f32 %su, %su, %su;
    fma.rn.f32 %su, %r, %uu, %su;
    fma.rn.f32 %sh, %r, %su, %sh;
    mul.f32 %sh, %sh, %scale;
    mul.f32 %sh, %sh, %scale;
ADDED:
    @!%hq bra EXIT;
    div.rn.f32 %r, %sh, %n;
    add.f32 %r, %r, %eps;
    rsqrt.approx.f32 %r, %r;
    @!%hw mov.f32 %r, 0f3F800000;
OWN
    mul.f32 %x, %x, %r;
    mov.f32 %w, 0f3F800000;
    mul.wide.u32 %ad, %own, 4;
    add.u64 %ad, %w2, %ad;
    @%hw ld.global.f32 %w, [%ad];
    mul.f32 %x, %x, %w;
    add.u32 %gi, %base, %own;
Q8
EXIT:
    ret;
}
";

/// `%{sum} ← Σ %{a}k·%{b}k` over this thread's values.
fn dot(sum: &str, a: &str, b: &str) -> String {
    (1..PER_THREAD).fold(format!("    mul.f32 %{sum}, %{a}0, %{b}0;\n"), |mut s, k| {
        let _ = writeln!(s, "    fma.rn.f32 %{sum}, %{a}{k}, %{b}{k}, %{sum};");
        s
    })
}

fn norm() -> String {
    let (mut loads, mut loads2, mut residual, mut own) = (String::new(), String::new(), String::new(), String::new());
    for k in 0..PER_THREAD {
        let (at, t) = (1024 * k, 256 * k);
        let _ =
            writeln!(loads, "    add.u32 %gi, %th, {t};\n    setp.lt.u32 %p{k}, %gi, %len;\n    mov.f32 %h{k}, 0f00000000;\n    mov.f32 %y{k}, 0f00000000;");
        let _ = writeln!(loads, "    @%p{k} ld.global.f32 %h{k}, [%ah+{at}];\n    and.pred %t, %p{k}, %hy;\n    @%t ld.global.f32 %y{k}, [%ay+{at}];");
        let _ = writeln!(loads, "    mov.f32 %a{k}, 0f00000000;\n    @%t ld.global.f32 %a{k}, [%aw+{at}];");
        let _ = writeln!(loads2, "    mul.f32 %a{k}, %a{k}, %y{k};");
        let _ = writeln!(residual, "    fma.rn.f32 %h{k}, %a{k}, %r, %h{k};\n    mul.f32 %h{k}, %h{k}, %scale;");
        let _ = writeln!(residual, "    setp.eq.u32 %t, %slice, {k};\n    @%t st.global.f32 [%ao+{at}], %h{k};");
        let _ = writeln!(own, "    setp.eq.u32 %t, %slice, {k};\n    @%t mov.f32 %x, %h{k};");
    }
    NORM.replace("BLOCK Q8", &(BLOCK.to_string() + Q8))
        .replace("LOADS\n", &(loads + &loads2 + &dot("ss", "y", "y") + &dot("sh", "h", "h") + &dot("su", "h", "a") + &dot("uu", "a", "a")))
        .replace("SUM\n", &block("add.f32", &["%ss", "%sh", "%su", "%uu"], NORM_THREADS / 32))
        .replace("RESIDUAL\n", &residual)
        .replace("OWN\n", &own)
        .replace("Q8\n", &q8("%x", "%gi"))
}

/// [`EMBED_Q4`]: thread `i` decodes weight `i` of row `token`, as [`crate::quant`] packs it.
const EMBED: &str = "
.visible .entry embed_q4(.param .u64 pw, .param .u64 ps, .param .u64 ptok, .param .u64 py, .param .u32 pcols, .param .f32 pscale)
{
    .reg .pred %p;
    .reg .b32 %i, %t, %cols, %tok, %g, %k, %n;
    .reg .b16 %h;
    .reg .f32 %d, %x, %scale;
    .reg .b64 %a, %b, %o;
    mov.u32 %i, %ctaid.x;
    mov.u32 %t, %ntid.x;
    mov.u32 %n, %tid.x;
    mad.lo.u32 %i, %i, %t, %n;
    ld.param.u32 %cols, [pcols];
    setp.ge.u32 %p, %i, %cols;
    @%p bra EXIT;
    ld.param.u64 %a, [ptok];
    cvta.to.global.u64 %a, %a;
    ld.global.u32 %tok, [%a];
    shr.u32 %g, %i, 6;
    shr.u32 %t, %cols, 6;
    mul.wide.u32 %o, %tok, %t;
    cvt.u64.u32 %b, %g;
    add.u64 %o, %o, %b;
    shl.b64 %o, %o, 1;
    ld.param.u64 %a, [ps];
    cvta.to.global.u64 %a, %a;
    add.u64 %a, %a, %o;
    ld.global.b16 %h, [%a];
    cvt.f32.f16 %d, %h;
    shr.u32 %t, %cols, 1;
    mul.wide.u32 %o, %tok, %t;
    and.b32 %k, %i, 63;
    shr.u32 %t, %k, 3;
    shl.b32 %t, %t, 2;
    and.b32 %n, %k, 3;
    add.u32 %t, %t, %n;
    shl.b32 %n, %g, 5;
    add.u32 %t, %t, %n;
    cvt.u64.u32 %b, %t;
    add.u64 %o, %o, %b;
    ld.param.u64 %a, [pw];
    cvta.to.global.u64 %a, %a;
    add.u64 %a, %a, %o;
    ld.global.u8 %n, [%a];
    shr.u32 %t, %k, 2;
    and.b32 %t, %t, 1;
    shl.b32 %t, %t, 2;
    shr.u32 %n, %n, %t;
    and.b32 %n, %n, 15;
    sub.s32 %n, %n, 8;
    cvt.rn.f32.s32 %x, %n;
    mul.f32 %x, %x, %d;
    ld.param.f32 %scale, [pscale];
    mul.f32 %x, %x, %scale;
    ld.param.u64 %a, [py];
    cvta.to.global.u64 %a, %a;
    mul.wide.u32 %o, %i, 4;
    add.u64 %a, %a, %o;
    st.global.f32 [%a], %x;
EXIT:
    ret;
}
";

#[cfg(test)]
mod tests {
    /// Writes the module to `target/ptx/kernels.ptx`, where `ci.sh` checks it with `ptxas`.
    #[test]
    fn module_is_written_for_ptxas() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(2).unwrap().join("target/ptx");
        std::fs::create_dir_all(&dir).unwrap();
        let ptx = super::module();
        for name in [super::QUANT_Q8, super::GEGLU_Q8, super::NORM_Q8, super::EMBED_Q4, super::GEMV_Q4, super::ATTEND_256, super::ATTEND_512, super::SAMPLE] {
            assert!(ptx.contains(&format!(".entry {name}(")), "{name} is missing");
        }
        std::fs::write(dir.join("kernels.ptx"), ptx).unwrap();
    }
}