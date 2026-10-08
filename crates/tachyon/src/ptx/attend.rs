//! Decode attention: one new query token against the key-value cache, for one layer.
//!
//! [`ATTEND_256`] and [`ATTEND_512`] differ only in the head size `D`. Each key-value head serves
//! [`HEADS_PER_KV`] query heads. The kernel takes the projected query, key and value of the new token as `f32`
//! (`q[H][D]`, then `k[Hkv][D]`, then `v[Hkv][D]`, with `H = 4·Hkv`) and:
//!
//! 1. RMS-normalizes each query head and scales it by `qn`; with `fresh`, does the same to the key with `kn`
//!    and normalizes the value without a scale (Gemma 4's `q_norm`, `k_norm`, `v_norm`, ε = 1e-6);
//! 2. rotates query and key by RoPE at position `*pos`: pairs `(i, i + D/2)` turn by `pos · freq[i]`, so a
//!    zero frequency leaves a pair alone, which is how [`rope`] expresses Gemma 4's proportional RoPE;
//! 3. with `fresh`, stores the key and value as `f16` in the cache at slot `pos % cap`. Without it (a layer that
//!    shares another layer's cache) the cache already holds them;
//! 4. attends over the last `min(pos + 1, cap)` positions with scale 1 and writes the result as Q8: a cache of
//!    `cap = window` slots is a sliding window, a cache as long as the context is global attention.
//!
//! The cache is `f16 [Hkv][cap][D]`, keys and values in separate buffers. Arguments: `qkv, qn, kn, freq:
//! *const f32, k_cache, v_cache: *mut f16, pos: *const u32, cap: u32, fresh: u32, partial: *mut f32, count:
//! *mut u32`, Q8 out (`H × D` values). Launch a grid of `(Hkv, S)` blocks of `32 · ATTEND_WARPS` threads, `S ≤
//! 256`: the positions in the cache are split evenly over the `S` blocks of each key-value head, whatever the
//! context, and the block that finishes last for its head merges the others' partial results. Blocks of
//! [`ATTEND_256`] fit two per multiprocessor and those of [`ATTEND_512`] one, so one wave is `S = 2·SMs / Hkv`
//! or `SMs / Hkv`. `partial` holds
//! `Hkv · S · (8 + 4·D)` floats of scratch; `count` holds `Hkv` counters, zero before the first launch and left
//! at zero by every launch.

use super::{Q8, q8, warp};
use std::fmt::Write;

/// Name of the attention kernel for heads of 256.
pub const ATTEND_256: &str = "attend_256";
/// Name of the attention kernel for heads of 512.
pub const ATTEND_512: &str = "attend_512";
/// Warps per attention block.
pub const ATTEND_WARPS: u32 = 8;
/// Query heads per key-value head.
pub const HEADS_PER_KV: u32 = 4;

/// RoPE frequencies for heads of `dim`: `theta^(−2i/dim)` for the first `fraction · dim / 2` pairs, zero for the
/// rest. `fraction` 1 is standard RoPE; Gemma 4's global layers use 0.25 ("proportional").
pub fn rope(theta: f32, dim: usize, fraction: f32) -> Vec<f32> {
    let rotated = (fraction * dim as f32) as usize / 2;
    (0..dim / 2).map(|i| if i < rotated { 1.0 / theta.powf((2 * i) as f32 / dim as f32) } else { 0.0 }).collect()
}

const LOG2E: &str = "0f3FB8AA3B";
const NEG_INF: &str = "0fFF800000";

/// `n` registers `%{reg}{first}..` loaded as `f32` from `[{addr}+{off}..]`.
fn ld32(s: &mut String, reg: &str, first: usize, addr: &str, off: usize, n: usize) {
    for m in (0..n).step_by(4) {
        let r = first + m;
        let _ = writeln!(s, "    ld.global.v4.f32 {{%{reg}{r}, %{reg}{}, %{reg}{}, %{reg}{}}}, [{addr}+{}];", r + 1, r + 2, r + 3, off + 4 * m);
    }
}

/// `e` registers `%{reg}0..` loaded from `f16` at `[{addr}]`.
fn ld16(s: &mut String, reg: &str, addr: &str, e: usize) {
    for m in (0..e).step_by(8) {
        let _ = writeln!(s, "    ld.global.v4.b32 {{%pk0, %pk1, %pk2, %pk3}}, [{addr}+{}];", 2 * m);
        for j in 0..8 {
            let _ = writeln!(s, "    mov.b32 {{%hk0, %hk1}}, %pk{};\n    cvt.f32.f16 %{reg}{}, %hk{};", j / 2, m + j, j % 2);
        }
    }
}

/// `e` registers `%{reg}0..` stored as `f16` at `[{addr}]`.
fn st16(s: &mut String, reg: &str, addr: &str, e: usize) {
    for m in (0..e).step_by(8) {
        for p in 0..4 {
            let (a, b) = (m + 2 * p, m + 2 * p + 1);
            let _ = writeln!(s, "    cvt.rn.f16.f32 %hk0, %{reg}{a};\n    cvt.rn.f16.f32 %hk1, %{reg}{b};\n    mov.b32 %pk{p}, {{%hk0, %hk1}};");
        }
        let _ = writeln!(s, "    st.global.v4.b32 [{addr}+{}], {{%pk0, %pk1, %pk2, %pk3}};", 2 * m);
    }
}

/// RMS-normalizes the `e` registers `%{reg}{first}..` across the warp (a head of `32·e`), then, when `scaled`,
/// multiplies them by the weights in `%w`.
fn norm(s: &mut String, reg: &str, first: usize, e: usize, scaled: bool) {
    let _ = writeln!(s, "    mul.f32 %f, %{reg}{first}, %{reg}{first};");
    (1..e).for_each(|j| _ = writeln!(s, "    fma.rn.f32 %f, %{reg}{}, %{reg}{}, %f;", first + j, first + j));
    *s += &warp("add.f32", "%f", "%ft");
    let _ = writeln!(s, "    div.rn.f32 %f, %f, 0f{:08X};\n    add.f32 %f, %f, 0f358637BD;\n    rsqrt.approx.f32 %f, %f;", ((32 * e) as f32).to_bits());
    (0..e).for_each(|j| _ = writeln!(s, "    mul.f32 %{reg}{}, %{reg}{}, %f;", first + j, first + j));
    if scaled {
        (0..e).for_each(|j| _ = writeln!(s, "    mul.f32 %{reg}{}, %{reg}{}, %w{j};", first + j, first + j));
    }
}

/// Rotates the `e` registers `%{reg}{first}..` by RoPE: each lane's partner holds the other half of its pairs.
fn rotate(s: &mut String, reg: &str, first: usize, e: usize) {
    for j in 0..e {
        let x = format!("%{reg}{}", first + j);
        let _ = writeln!(s, "    shfl.sync.bfly.b32 %ft, {x}, 16, 31, -1;\n    mul.f32 %ft, %ft, %sn{j};\n    mul.f32 %ft, %ft, %sgn;");
        let _ = writeln!(s, "    fma.rn.f32 {x}, {x}, %c{j}, %ft;");
    }
}

/// Cosines `%c` and sines `%sn` of this lane's `e` angles `pos · freq`, the angle first reduced to [−π, π].
fn angles(s: &mut String, e: usize) {
    *s += "    cvt.rn.f32.u32 %pf, %pos;\n    and.b32 %x, %lane, 15;\n";
    let _ = writeln!(s, "    mul.wide.u32 %o, %x, {};\n    ld.param.u64 %a, [pfreq];\n    cvta.to.global.u64 %a, %a;\n    add.u64 %a, %a, %o;", 4 * e);
    ld32(s, "c", 0, "%a", 0, e);
    for j in 0..e {
        let _ = writeln!(s, "    mul.rn.f32 %f, %pf, %c{j};\n    mul.f32 %r, %f, 0f3E22F983;\n    cvt.rni.f32.f32 %r, %r;");
        let _ = writeln!(s, "    fma.rn.f32 %f, %r, 0fC0C90000, %f;\n    fma.rn.f32 %f, %r, 0fBAFDAA22, %f;");
        let _ = writeln!(s, "    sin.approx.f32 %sn{j}, %f;\n    cos.approx.f32 %c{j}, %f;");
    }
    *s += "    setp.lt.u32 %p, %lane, 16;\n    selp.f32 %sgn, 0fBF800000, 0f3F800000, %p;\n";
}

/// Normalized, rotated query heads `4g..4g+4` into `%q`, and the running maxima, sums and accumulators cleared.
fn queries(s: &mut String, d: usize, e: usize) {
    let _ = writeln!(s, "    shl.b32 %x, %g, 2;\n    mul.lo.u32 %x, %x, {d};\n    mad.lo.u32 %x, %lane, {e}, %x;\n    mul.wide.u32 %o, %x, 4;");
    let _ = writeln!(s, "    add.u64 %a, %qkv, %o;\n    ld.param.u64 %b, [pqn];\n    cvta.to.global.u64 %b, %b;\n    mul.wide.u32 %o, %lane, {};", 4 * e);
    *s += "    add.u64 %b, %b, %o;\n";
    ld32(s, "w", 0, "%b", 0, e);
    (0..4).for_each(|h| ld32(s, "q", h * e, "%a", 4 * h * d, e));
    for h in 0..4 {
        norm(s, "q", h * e, e, true);
        rotate(s, "q", h * e, e);
        let _ = writeln!(s, "    mov.f32 %m{h}, {NEG_INF};\n    mov.f32 %l{h}, 0f00000000;");
    }
    (0..4 * e).for_each(|j| _ = writeln!(s, "    mov.f32 %acc{j}, 0f00000000;"));
}

/// The new token's key and value, normalized, the key rotated, stored to the cache and left in `%k`, `%v`.
fn fresh(s: &mut String, d: usize, e: usize) {
    let _ = writeln!(s, "    shl.b32 %x, %nkv, 2;\n    add.u32 %x, %x, %g;\n    mul.lo.u32 %x, %x, {d};\n    mad.lo.u32 %x, %lane, {e}, %x;");
    *s += "    mul.wide.u32 %o, %x, 4;\n    add.u64 %a, %qkv, %o;\n";
    ld32(s, "k", 0, "%a", 0, e);
    let _ = writeln!(s, "    mul.lo.u32 %x, %nkv, {};\n    cvt.u64.u32 %o, %x;\n    add.u64 %a, %a, %o;", 4 * d);
    ld32(s, "v", 0, "%a", 0, e);
    let _ = writeln!(s, "    ld.param.u64 %b, [pkn];\n    cvta.to.global.u64 %b, %b;\n    mul.wide.u32 %o, %lane, {};\n    add.u64 %b, %b, %o;", 4 * e);
    ld32(s, "w", 0, "%b", 0, e);
    norm(s, "k", 0, e, true);
    rotate(s, "k", 0, e);
    norm(s, "v", 0, e, false);
    slot(s, "%cur", d, e);
    st16(s, "k", "%a", e);
    *s += "    add.u64 %a, %vc, %o;\n";
    st16(s, "v", "%a", e);
}

/// `%o`: byte offset of this lane's values of position `t` in the cache; `%a`: their address in the key cache.
fn slot(s: &mut String, t: &str, d: usize, e: usize) {
    let _ = writeln!(s, "    mad.lo.u32 %x, %g, %cap, {t};\n    mul.lo.u32 %x, %x, {d};\n    mad.lo.u32 %x, %lane, {e}, %x;");
    *s += "    mul.wide.u32 %o, %x, 2;\n    add.u64 %a, %kc, %o;\n";
}

/// One position for every head: score `q·k`, then the online softmax update of `%m`, `%l` and `%acc` with `v`.
fn step(s: &mut String, e: usize) {
    for h in 0..4 {
        let _ = writeln!(s, "    mul.f32 %sc, %q{}, %k0;", h * e);
        (1..e).for_each(|j| _ = writeln!(s, "    fma.rn.f32 %sc, %q{}, %k{j}, %sc;", h * e + j));
        *s += &warp("add.f32", "%sc", "%ft");
        let _ = writeln!(s, "    max.f32 %mn, %m{h}, %sc;\n    sub.f32 %f, %m{h}, %mn;\n    mul.f32 %f, %f, {LOG2E};\n    ex2.approx.f32 %f, %f;");
        let _ = writeln!(s, "    sub.f32 %ft, %sc, %mn;\n    mul.f32 %ft, %ft, {LOG2E};\n    ex2.approx.f32 %ft, %ft;");
        let _ = writeln!(s, "    fma.rn.f32 %l{h}, %l{h}, %f, %ft;\n    mov.f32 %m{h}, %mn;");
        for j in 0..e {
            let _ = writeln!(s, "    mul.f32 %acc{0}, %acc{0}, %f;\n    fma.rn.f32 %acc{0}, %ft, %v{j}, %acc{0};", h * e + j);
        }
    }
}

/// Merges the warps' maxima, sums and accumulators into the block's partial result in shared memory.
fn merge(s: &mut String, d: usize, e: usize) {
    *s += "    mov.u32 %x, sm;\n    mad.lo.u32 %x, %warp, 16, %x;\n    @%lane0 st.shared.v4.f32 [%x], {%m0, %m1, %m2, %m3};\n    bar.sync 0;\n";
    (0..4).for_each(|h| _ = writeln!(s, "    mov.f32 %M{h}, {NEG_INF};"));
    for w in 0..ATTEND_WARPS {
        let _ = writeln!(s, "    ld.shared.v4.f32 {{%cr0, %cr1, %cr2, %cr3}}, [sm+{}];", 16 * w);
        (0..4).for_each(|h| _ = writeln!(s, "    max.f32 %M{h}, %M{h}, %cr{h};"));
    }
    for h in 0..4 {
        let _ = writeln!(s, "    sub.f32 %f, %m{h}, %M{h};\n    mul.f32 %f, %f, {LOG2E};\n    ex2.approx.f32 %cr{h}, %f;");
        let _ = writeln!(s, "    setp.eq.f32 %p, %M{h}, {NEG_INF};\n    @%p mov.f32 %cr{h}, 0f00000000;");
    }
    for h in 0..4 {
        (0..e).for_each(|j| _ = writeln!(s, "    mul.f32 %acc{0}, %acc{0}, %cr{h};", h * e + j));
        let _ = writeln!(s, "    mul.f32 %l{h}, %l{h}, %cr{h};");
    }
    // A tree over the warps through four shared slots: 4-7 into 0-3, then 2-3 into 0-1, then 1 into 0. A warp
    // only overwrites a slot it read itself, so each level needs one barrier.
    for (half, store, read) in [(4, 4, 0), (2, 0, 2), (1, 0, 1)] {
        let _ =
            writeln!(s, "    setp.lt.u32 %p, %warp, {half};\n    setp.ge.u32 %ps, %warp, {};\n    or.pred %p, %p, %ps;\n    @%p bra STORED{half};", 2 * half);
        let _ = writeln!(s, "    sub.u32 %x, %warp, {store};");
        slot_of(s, d, e);
        sums(s, d, e, false);
        let _ = writeln!(s, "STORED{half}:\n    bar.sync 0;\n    setp.ge.u32 %p, %warp, {half};\n    @%p bra ADDED{half};\n    add.u32 %x, %warp, {read};");
        slot_of(s, d, e);
        sums(s, d, e, true);
        let _ = writeln!(s, "ADDED{half}:");
    }
}

/// `%x ← sacc` slot `%x`, this lane's values.
fn slot_of(s: &mut String, d: usize, e: usize) {
    let _ = writeln!(s, "    mov.u32 %t, sacc;\n    mad.lo.u32 %x, %x, {}, %t;\n    mad.lo.u32 %x, %lane, {}, %x;", 4 * (4 * d + 4), 4 * e);
}

/// Stores this warp's sums to the slot at `%x`, or with `add`, adds the slot's to them.
fn sums(s: &mut String, d: usize, e: usize, add: bool) {
    let regs = |first: usize, name: &str| (0..4).map(|i| format!("%{name}{}", first + i)).collect::<Vec<_>>().join(", ");
    let mut one = |r: String, at: usize, guard: &str| {
        if add {
            let _ = writeln!(s, "    {guard}ld.shared.v4.f32 {{%k0, %k1, %k2, %k3}}, [%x+{at}];");
            r.split(", ").enumerate().for_each(|(i, x)| _ = writeln!(s, "    add.f32 {x}, {x}, %k{i};"));
        } else {
            let _ = writeln!(s, "    {guard}st.shared.v4.f32 [%x+{at}], {{{r}}};");
        }
    };
    for h in 0..4 {
        (0..e).step_by(4).for_each(|j| one(regs(h * e + j, "acc"), 4 * (h * d + j), ""));
    }
    one(regs(0, "l"), 16 * d, "@%lane0 ");
}

/// Head of every attention kernel, after the declarations: positions and this block's range.
const SETUP: &str = "
    mov.u32 %th, %tid.x;
    and.b32 %lane, %th, 31;
    shr.u32 %warp, %th, 5;
    setp.eq.u32 %lane0, %lane, 0;
    mov.u32 %g, %ctaid.x;
    mov.u32 %spl, %ctaid.y;
    mov.u32 %nkv, %nctaid.x;
    mov.u32 %nsp, %nctaid.y;
    ld.param.u64 %a, [ppos];
    cvta.to.global.u64 %a, %a;
    ld.global.u32 %pos, [%a];
    ld.param.u32 %cap, [pcap];
    ld.param.u32 %x, [pfresh];
    setp.ne.u32 %fresh, %x, 0;
    add.u32 %n, %pos, 1;
    min.u32 %n, %n, %cap;
    rem.u32 %cur, %pos, %cap;
    add.u32 %span, %n, %nsp;
    sub.u32 %span, %span, 1;
    div.u32 %span, %span, %nsp;
    selp.b32 %skip, %cur, -1, %fresh;
    mul.lo.u32 %start, %spl, %span;
    setp.ge.u32 %p, %start, %n;
    @%p bra EXIT;
    add.u32 %end, %start, %span;
    min.u32 %end, %end, %n;
    add.u32 %active, %n, %span;
    sub.u32 %active, %active, 1;
    div.u32 %active, %active, %span;
    ld.param.u64 %qkv, [pqkv];
    cvta.to.global.u64 %qkv, %qkv;
    ld.param.u64 %kc, [pkc];
    cvta.to.global.u64 %kc, %kc;
    ld.param.u64 %vc, [pvc];
    cvta.to.global.u64 %vc, %vc;
    ld.param.u64 %part, [ppart];
    cvta.to.global.u64 %part, %part;
    ld.param.u64 %qp, [pq];
    cvta.to.global.u64 %qp, %qp;
    ld.param.u64 %sp, [ps];
    cvta.to.global.u64 %sp, %sp;
";

/// The cache loop: warp `w` takes positions `start + w`, `start + w + 8`, …; the fresh token goes first.
const LOOP: &str = "
    setp.ne.u32 %infresh, %th, %th;
    @!%dofresh bra POSITIONS;
FRESH
    setp.eq.u32 %infresh, %th, %th;
    bra STEP;
POSITIONS:
    setp.ne.u32 %infresh, %th, %th;
    add.u32 %t, %start, %warp;
LOOP:
    setp.ge.u32 %p, %t, %end;
    @%p bra DONE;
PREFETCH
    setp.eq.u32 %p, %t, %skip;
    @%p bra NEXT;
LOAD
STEP:
SCORE
    @%infresh bra POSITIONS;
NEXT:
    add.u32 %t, %t, 8;
    bra LOOP;
DONE:
";

/// Warp 0 writes the block's partial result (`STORE`: its sums, from registers); thread 0 counts the block in, with
/// fences on both sides, as grid-wide synchronization does; the last block of its key-value head goes on to merge.
const PARTIAL: &str = "
    mad.lo.u32 %x, %g, %nsp, %spl;
    mul.lo.u32 %x, %x, STRIDE;
    mul.wide.u32 %o, %x, 4;
    add.u64 %a, %part, %o;
    setp.ne.u32 %p, %warp, 0;
    @%p bra WRITTEN;
    @%lane0 st.global.v4.f32 [%a], {%M0, %M1, %M2, %M3};
    @%lane0 st.global.v4.f32 [%a+16], {%l0, %l1, %l2, %l3};
    mul.wide.u32 %o, %lane, LANE;
    add.u64 %b, %a, %o;
STORE
WRITTEN:
    bar.sync 0;
    setp.ne.u32 %p, %th, 0;
    @%p bra COUNTED;
    ld.param.u64 %b, [pcount];
    cvta.to.global.u64 %b, %b;
    mul.wide.u32 %o, %g, 4;
    add.u64 %b, %b, %o;
    fence.acq_rel.gpu;
    atom.global.add.u32 %old, [%b], 1;
    sub.u32 %x, %active, 1;
    setp.eq.u32 %last, %old, %x;
    @%last st.global.u32 [%b], 0;
    fence.acq_rel.gpu;
    selp.u32 %x, 1, 0, %last;
    st.shared.u32 [flag], %x;
COUNTED:
    bar.sync 0;
    ld.shared.u32 %x, [flag];
    setp.eq.u32 %p, %x, 0;
    @%p bra EXIT;
";

/// The last block of key-value head `g`: weights per active block and head in shared memory, `w = e^(M − M*) / Σ
/// e^(M − M*)·L`, then each output as `Σ w · ACC` with eight blocks' values in flight per thread.
fn combine(s: &mut String, d: usize) {
    let (stride, outs) = (8 + 4 * d, 4 * d / 256);
    let _ = writeln!(s, "    mul.lo.u32 %x, %g, %nsp;\n    mul.lo.u32 %x, %x, {stride};\n    mul.wide.u32 %o, %x, 4;\n    add.u64 %part, %part, %o;");
    let _ = writeln!(s, "    setp.lt.u32 %p, %th, %active;\n    mul.lo.u32 %x, %th, {stride};\n    mul.wide.u32 %o, %x, 4;\n    add.u64 %a, %part, %o;");
    (0..4).for_each(|h| _ = writeln!(s, "    mov.f32 %cr{h}, {NEG_INF};\n    mov.f32 %l{h}, 0f00000000;"));
    *s += "    @%p ld.global.cg.v4.f32 {%cr0, %cr1, %cr2, %cr3}, [%a];\n    @%p ld.global.cg.v4.f32 {%l0, %l1, %l2, %l3}, [%a+16];\n";
    (0..4).for_each(|h| _ = writeln!(s, "    mov.f32 %M{h}, %cr{h};"));
    across(s, "max.f32", "M", NEG_INF);
    for h in 0..4 {
        let _ = writeln!(s, "    sub.f32 %f, %cr{h}, %M{h};\n    mul.f32 %f, %f, {LOG2E};\n    ex2.approx.f32 %cr{h}, %f;\n    mul.f32 %l{h}, %l{h}, %cr{h};");
    }
    across(s, "add.f32", "l", "0f00000000");
    (0..4).for_each(|h| _ = writeln!(s, "    div.rn.f32 %cr{h}, %cr{h}, %l{h};"));
    *s += "    mov.u32 %x, sacc;\n    mad.lo.u32 %x, %th, 16, %x;\n    st.shared.v4.f32 [%x], {%cr0, %cr1, %cr2, %cr3};\n    bar.sync 0;\n";
    (0..outs).for_each(|k| _ = writeln!(s, "    mov.f32 %acc{k}, 0f00000000;"));
    *s += "    mul.wide.u32 %o, %th, 4;\n    add.u64 %b, %part, %o;\n    mov.u32 %s, 0;\n    mov.u32 %x, sacc;\nSUMS:\n";
    *s += "    setp.ge.u32 %p, %s, %active;\n    @%p bra SUMMED;\n";
    for u in 0..8 {
        let _ = writeln!(s, "    add.u32 %j, %s, {u};\n    setp.lt.u32 %p, %j, %active;");
        for k in 0..outs {
            let _ = writeln!(s, "    mov.f32 %q{0}, 0f00000000;\n    @%p ld.global.cg.f32 %q{0}, [%b+{1}];", u * outs + k, 4 * (u * stride + 8 + 256 * k));
        }
    }
    for u in 0..8 {
        let _ = writeln!(s, "    ld.shared.v4.f32 {{%k0, %k1, %k2, %k3}}, [%x+{}];", 16 * u);
        (0..outs).for_each(|k| _ = writeln!(s, "    fma.rn.f32 %acc{k}, %k{}, %q{}, %acc{k};", 256 * k / d, u * outs + k));
    }
    let _ = writeln!(s, "    add.u64 %b, %b, {};\n    add.u32 %x, %x, 128;\n    add.u32 %s, %s, 8;\n    bra SUMS;\nSUMMED:", 32 * stride);
    for k in 0..outs {
        let _ = writeln!(s, "    mad.lo.u32 %gi, %g, {}, %th;\n    add.u32 %gi, %gi, {};", 4 * d, 256 * k);
        *s += &q8(&format!("%acc{k}"), "%gi");
    }
    *s += "EXIT:\n    ret;\n}\n";
}

/// `op` of each head's `%{reg}0..3` over the block's warps, through `sm`; `init` is `op`'s identity.
fn across(s: &mut String, op: &str, reg: &str, init: &str) {
    (0..4).for_each(|h| *s += &warp(op, &format!("%{reg}{h}"), "%ft"));
    let _ = writeln!(
        s,
        "    mov.u32 %x, sm;\n    mad.lo.u32 %x, %warp, 16, %x;\n    @%lane0 st.shared.v4.f32 [%x], {{%{reg}0, %{reg}1, %{reg}2, %{reg}3}};\n    bar.sync 0;"
    );
    (0..4).for_each(|h| _ = writeln!(s, "    mov.f32 %{reg}{h}, {init};"));
    for w in 0..ATTEND_WARPS {
        let _ = writeln!(s, "    ld.shared.v4.f32 {{%k0, %k1, %k2, %k3}}, [sm+{}];", 16 * w);
        (0..4).for_each(|h| _ = writeln!(s, "    {op} %{reg}{h}, %{reg}{h}, %k{h};"));
    }
    *s += "    bar.sync 0;\n";
}

/// The attention kernel for heads of `d`.
pub(super) fn attend(d: usize) -> String {
    let e = d / 32;
    let mut s = format!(
        ".visible .entry attend_{d}(.param .u64 pqkv, .param .u64 pqn, .param .u64 pkn, .param .u64 pfreq, .param .u64 pkc, .param .u64 pvc,
    .param .u64 ppos, .param .u32 pcap, .param .u32 pfresh, .param .u64 ppart, .param .u64 pcount, .param .u64 pq, .param .u64 ps)
{}{{
    .shared .align 16 .f32 sm[32];
    .shared .align 16 .f32 sacc[{}];
    .shared .align 4 .u32 flag;
    .reg .pred %p, %ps, %fresh, %dofresh, %infresh, %last, %lane0;
    .reg .b32 %th, %lane, %warp, %g, %spl, %nkv, %nsp, %pos, %cap, %span, %n, %cur, %skip, %start, %end, %active, %t, %x, %j, %s, %h, %old, %gi;
    .reg .b32 %pk<4>;
    .reg .b16 %hk<2>;
    .reg .f32 %q<{}>, %acc<{}>, %k<{e}>, %v<{e}>, %c<{e}>, %sn<{e}>, %w<{e}>, %m<4>, %l<4>, %cr<4>, %M<4>;
    .reg .f32 %pf, %sgn, %sc, %f, %ft, %mn, %r, %num, %den, %ls, %as;
    .reg .b64 %qkv, %kc, %vc, %part, %a, %b, %o;{Q8}{}",
        // Two blocks per multiprocessor for the smaller heads: register use capped at 128.
        if d == 256 { ".maxntid 256, 1, 1\n.minnctapersm 2\n" } else { "" },
        4 * (4 * d + 4),
        4 * e,
        4 * e,
        SETUP
    );
    angles(&mut s, e);
    queries(&mut s, d, e);
    s += "    setp.eq.u32 %dofresh, %spl, 0;\n    and.pred %dofresh, %dofresh, %fresh;\n    setp.eq.u32 %p, %warp, 0;\n    and.pred %dofresh, %dofresh, %p;\n";
    let (mut fresh_code, mut load, mut score) = (String::new(), String::new(), String::new());
    fresh(&mut fresh_code, d, e);
    slot(&mut load, "%t", d, e);
    ld16(&mut load, "k", "%a", e);
    load += "    add.u64 %a, %vc, %o;\n";
    ld16(&mut load, "v", "%a", e);
    step(&mut score, e);
    let mut fetch = format!("    add.u32 %j, %t, {};\n    setp.lt.u32 %p, %j, %end;\n    @!%p bra FETCHED;\n", 4 * ATTEND_WARPS);
    slot(&mut fetch, "%j", d, e);
    fetch += "    prefetch.global.L2 [%a];\n    add.u64 %a, %vc, %o;\n    prefetch.global.L2 [%a];\nFETCHED:\n";
    s += &LOOP.replace("PREFETCH\n", &fetch).replace("FRESH\n", &fresh_code).replace("LOAD\n", &load).replace("SCORE\n", &score);
    merge(&mut s, d, e);
    let stride = 8 + 4 * d;
    let fill = |t: &str| t.replace("STRIDE", &stride.to_string()).replace("LANE", &(4 * e).to_string());
    let mut sums = String::new();
    for (h, j) in (0..4).flat_map(|h| (0..e).step_by(4).map(move |j| (h, j))) {
        let r = h * e + j;
        let _ = writeln!(sums, "    st.global.v4.f32 [%b+{}], {{%acc{r}, %acc{}, %acc{}, %acc{}}};", 32 + 4 * (h * d + j), r + 1, r + 2, r + 3);
    }
    s += &fill(PARTIAL).replace("STORE\n", &sums);
    combine(&mut s, d);
    s
}