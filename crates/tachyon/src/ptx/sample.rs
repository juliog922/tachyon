//! Sampling the next token from the logits, on the GPU.
//!
//! [`SAMPLE`] reads a [`Sampling`] from device memory, so one recorded decode step serves every setting. Each
//! logit `x` becomes `softcap · tanh(x / softcap)`; then −∞ when `allowed` excludes it; then `x / penalty` (or
//! `x · penalty` when negative) when the token was seen before; then `x / temperature`. With temperature 0 the
//! result is the largest (lowest index on ties). Otherwise the token is drawn from `softmax(x)` restricted to the
//! tokens that pass every filter, each a set of the most likely tokens:
//!
//! - `top_k`: fewer than `top_k` tokens are more likely;
//! - `top_p`: the tokens more likely than it hold less than `top_p` of the probability;
//! - `min_p`: its probability is at least `min_p` times the largest.
//!
//! The draw is exact. Every block takes a share of the vocabulary, writes it transformed to `work` and draws its own
//! candidate by the Gumbel-max trick (`argmax x + G`, `G = −ln(−ln U)` from a hash of seed, step and token);
//! the last block to finish merges the blocks' results. With filters, the last block checks the candidate
//! against them and, if it fails, draws again among the tokens more likely than it: the final token follows the
//! filtered distribution exactly. A draw fails as often as the filters cut probability away: with a language
//! model's peaked logits one check is usual; a flat distribution under a small `top_k` takes several rounds of two
//! passes over the logits, and after 32 failed draws the kernel falls back to the largest.
//!
//! Arguments: `logits: *const f32, work: *mut f32, vocab: u32, cfg: *const Sampling, state: *mut [u32; 2],
//! seen: *mut u32, out: *mut u32, partial: *mut u32, count: *mut u32`. The logits are left unchanged; `work`
//! holds `vocab` floats. `state` is `[step, token]`: the kernel writes the token and advances the step, which
//! seeds the next draw. `seen` is a bitmap of `vocab` bits; the kernel sets the token's bit. `out`, when not null,
//! also receives the token (pinned host memory, read without a copy). `partial` holds 8 words per block; `count`
//! is zero before the first launch and left at zero. `vocab` is a multiple of 4. Launch at most
//! `min(1024, vocab.div_ceil(4 · SAMPLE_THREADS))` blocks of [`SAMPLE_THREADS`]; one per multiprocessor is enough.

use super::{BLOCK, block};
use std::fmt::Write;

/// Name of the sampler.
pub const SAMPLE: &str = "sample";
/// Threads per block of [`SAMPLE`].
pub const SAMPLE_THREADS: u32 = 1024;

/// How [`SAMPLE`] picks a token; laid out as the kernel reads it.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sampling {
    /// Logit soft cap; 0 for none.
    pub softcap: f32,
    /// 0 picks the most likely token.
    pub temperature: f32,
    /// Nucleus size; 1 for no limit.
    pub top_p: f32,
    /// Least probability relative to the most likely token; 0 for no limit.
    pub min_p: f32,
    /// Repetition penalty on seen tokens; 1 for none.
    pub penalty: f32,
    /// Most tokens kept; 0 for no limit.
    pub top_k: u32,
    /// Device address of a bitmap of the tokens allowed; 0 allows every token.
    pub allowed: u64,
    /// Seed of the draws.
    pub seed: u64,
}

impl Default for Sampling {
    /// Greedy: the most likely token, nothing else applied.
    fn default() -> Sampling {
        Sampling { softcap: 0.0, temperature: 0.0, top_p: 1.0, min_p: 0.0, penalty: 1.0, top_k: 0, allowed: 0, seed: 0 }
    }
}

impl Sampling {
    /// The bytes the kernel reads.
    pub fn bytes(&self) -> [u8; 40] {
        let words = [self.softcap.to_bits(), self.temperature.to_bits(), self.top_p.to_bits(), self.min_p.to_bits(), self.penalty.to_bits(), self.top_k];
        let mut out = [0; 40];
        for (chunk, w) in out.chunks_exact_mut(4).zip(words) {
            chunk.copy_from_slice(&w.to_ne_bytes());
        }
        out[24..32].copy_from_slice(&self.allowed.to_ne_bytes());
        out[32..].copy_from_slice(&self.seed.to_ne_bytes());
        out
    }
}

/// Murmur3's finalizer on `%h`.
const FMIX: &str = "    shr.b32 %hs, %h, 16;\n    xor.b32 %h, %h, %hs;\n    mul.lo.u32 %h, %h, 0x85EBCA6B;
    shr.b32 %hs, %h, 13;\n    xor.b32 %h, %h, %hs;\n    mul.lo.u32 %h, %h, 0xC2B2AE35;\n    shr.b32 %hs, %h, 16;\n    xor.b32 %h, %h, %hs;\n";

/// `%base`: the hash of seed, step and draw that every token's noise starts from.
fn base() -> String {
    format!(
        "    mad.lo.u32 %h, %ctr, 0x9E3779B9, %seedhi;\n    mad.lo.u32 %h, %round, 0x85EBCA77, %h;\n{FMIX}    xor.b32 %h, %h, %seedlo;\n{FMIX}    mov.u32 %base, %h;\n"
    )
}

/// Keeps `(v, i)` in `(bv, bi)` when larger, or equal with a lower index.
fn better(v: &str, i: &str, bv: &str, bi: &str) -> String {
    format!(
        "    setp.gt.f32 %p, {v}, {bv};\n    setp.eq.f32 %q, {v}, {bv};\n    setp.lt.u32 %ok, {i}, {bi};\n    and.pred %q, %q, %ok;
    or.pred %p, %p, %q;\n    @%p mov.f32 {bv}, {v};\n    @%p mov.u32 {bi}, {i};\n"
    )
}

/// `(v, i)` with the largest `v` over the block, in every thread.
fn argmax(v: &str, i: &str) -> String {
    let lanes = (0..5).fold(String::new(), |mut s, k| {
        let _ = writeln!(s, "    shfl.sync.bfly.b32 %ov, {v}, {}, 31, -1;\n    shfl.sync.bfly.b32 %oi, {i}, {}, 31, -1;", 16 >> k, 16 >> k);
        s + &better("%ov", "%oi", v, i)
    });
    format!(
        "{lanes}    @%lane0 st.shared.f32 [%rw], {v};\n    @%lane0 st.shared.u32 [%rwi], {i};\n    bar.sync 0;\n    ld.shared.f32 {v}, [%rl];\n    ld.shared.u32 {i}, [%rli];\n{lanes}    bar.sync 0;\n"
    )
}

/// `%t ← x + G(i)` when `x > %thr`, else −∞: this token's Gumbel key.
fn gumbel(x: &str, i: &str) -> String {
    format!(
        "    mul.lo.u32 %h, {i}, 0x9E3779B9;\n    xor.b32 %h, %h, %base;\n{FMIX}    shr.u32 %h, %h, 9;\n    cvt.rn.f32.u32 %u, %h;
    add.f32 %u, %u, 0f3F000000;\n    mul.f32 %u, %u, 0f34000000;\n    lg2.approx.f32 %u, %u;\n    mul.f32 %u, %u, 0fBF317218;
    lg2.approx.f32 %u, %u;\n    mul.f32 %u, %u, 0fBF317218;\n    add.f32 %t, {x}, %u;\n    setp.gt.f32 %p, {x}, %thr;\n    @!%p mov.f32 %t, 0fFF800000;\n"
    )
}

/// Transforms logit `%x{k}` (token `%e + k`) and folds it into the thread's maximum, sum and Gumbel candidate.
fn element(k: usize) -> String {
    let x = format!("%x{k}");
    let mut s = format!(
        "    @%hcap mul.f32 %t, {x}, %isc;\n    @%hcap ex2.approx.f32 %t, %t;\n    @%hcap add.f32 %t, %t, 0f3F800000;
    @%hcap rcp.approx.f32 %t, %t;\n    @%hcap fma.rn.f32 {x}, %t, %sc2, %sc;\n    add.u32 %bits, %sh, {k};
    shr.b32 %h, %aw, %bits;\n    and.b32 %h, %h, 1;\n    setp.eq.u32 %p, %h, 0;\n    @%p mov.f32 {x}, 0fFF800000;
    shr.b32 %h, %sw, %bits;\n    and.b32 %h, %h, 1;\n    setp.ne.u32 %p, %h, 0;\n    setp.gt.f32 %q, {x}, 0f00000000;
    selp.f32 %t, %ipen, %pen, %q;\n    @%p mul.f32 {x}, {x}, %t;\n    mul.f32 {x}, {x}, %itemp;\n    add.u32 %oi, %e, {k};
    max.f32 %mn, %m, {x};\n    sub.f32 %t, %m, %mn;\n    mul.f32 %t, %t, 0f3FB8AA3B;\n    ex2.approx.f32 %t, %t;
    sub.f32 %u, {x}, %mn;\n    mul.f32 %u, %u, 0f3FB8AA3B;\n    ex2.approx.f32 %u, %u;\n    setp.neu.f32 %p, {x}, 0fFF800000;
    @%p fma.rn.f32 %z, %z, %t, %u;\n    @%p mov.f32 %m, %mn;\n"
    );
    s += &better(&x, "%oi", "%bv", "%bi");
    let _ = writeln!(s, "    @%greedy bra SKIP{k};");
    s += &gumbel(&x, "%oi");
    s += &better("%t", "%oi", "%kv", "%ki");
    let _ = writeln!(s, "SKIP{k}:");
    s
}

/// Loads four logits at `%e` into `%x0..3` from `%lg`.
const FOUR: &str = "    mul.wide.u32 %o, %e, 4;\n    add.u64 %a, %lg, %o;\n    ld.global.v4.f32 {%x0, %x1, %x2, %x3}, [%a];\n";

/// Loads the transformed logits `%e + 4096·g + i` into `%x{4g+i}`, all sixteen loads in flight, −∞ past the end.
fn sixteen() -> String {
    let mut s = String::from("    mul.wide.u32 %o, %e, 4;\n    add.u64 %a, %wk, %o;\n");
    for g in 0..4 {
        let r = 4 * g;
        let _ = writeln!(s, "    add.u32 %oi, %e, {};\n    setp.lt.u32 %g{g}, %oi, %vocab;", 4096 * g);
        (r..r + 4).for_each(|k| _ = writeln!(s, "    mov.f32 %x{k}, 0fFF800000;"));
        let _ = writeln!(s, "    @%g{g} ld.global.cg.v4.f32 {{%x{r}, %x{}, %x{}, %x{}}}, [%a+{}];", r + 1, r + 2, r + 3, 16_384 * g);
    }
    s
}

const HEAD: &str = "
.visible .entry sample(.param .u64 plog, .param .u64 pwork, .param .u32 pvocab, .param .u64 pcfg, .param .u64 pstate, .param .u64 pseen, .param .u64 pout,
                       .param .u64 ppart, .param .u64 pcount)
{
    .shared .align 4 .u32 redi[32];
    .shared .align 4 .u32 flag;
    .reg .pred %p, %q, %ok, %greedy, %filter, %hmask, %hcap, %last, %g<4>;
    .reg .b32 %rwi, %rli, %vocab, %e, %stride, %aw, %sw, %bits, %sh, %bi, %ki, %oi, %amax, %tok, %topk, %ctr, %round, %base;
    .reg .b32 %h, %hs, %x, %nb, %seedlo, %seedhi;
    .reg .f32 %x<16>, %t, %u, %bv, %kv, %ov, %m, %z, %mn, %sc, %sc2, %isc, %itemp, %temp, %pen, %ipen;
    .reg .f32 %topp, %minp, %kf, %thr, %lj, %cnt, %mass;
    .reg .b64 %lg, %wk, %cfg, %st, %seen, %allow, %part, %a, %o, %seed;
BLOCK
    mov.u32 %rwi, redi;
    mad.lo.u32 %rli, %lane, 4, %rwi;
    mad.lo.u32 %rwi, %warp, 4, %rwi;
    ld.param.u32 %vocab, [pvocab];
    ld.param.u64 %lg, [plog];
    cvta.to.global.u64 %lg, %lg;
    ld.param.u64 %wk, [pwork];
    cvta.to.global.u64 %wk, %wk;
    ld.param.u64 %cfg, [pcfg];
    cvta.to.global.u64 %cfg, %cfg;
    ld.param.u64 %st, [pstate];
    cvta.to.global.u64 %st, %st;
    ld.param.u64 %seen, [pseen];
    cvta.to.global.u64 %seen, %seen;
    ld.param.u64 %part, [ppart];
    cvta.to.global.u64 %part, %part;
    ld.global.f32 %sc, [%cfg];
    ld.global.f32 %temp, [%cfg+4];
    ld.global.f32 %topp, [%cfg+8];
    ld.global.f32 %minp, [%cfg+12];
    ld.global.f32 %pen, [%cfg+16];
    ld.global.u32 %topk, [%cfg+20];
    ld.global.u64 %allow, [%cfg+24];
    ld.global.u64 %seed, [%cfg+32];
    mov.b64 {%seedlo, %seedhi}, %seed;
    ld.global.u32 %ctr, [%st];
    setp.gt.f32 %hcap, %sc, 0f00000000;
    rcp.rn.f32 %isc, %sc;
    mul.f32 %isc, %isc, 0f4038AA3B;
    mul.f32 %sc2, %sc, 0fC0000000;
    setp.le.f32 %greedy, %temp, 0f00000000;
    rcp.rn.f32 %itemp, %temp;
    @%greedy mov.f32 %itemp, 0f3F800000;
    rcp.rn.f32 %ipen, %pen;
    setp.ne.u64 %hmask, %allow, 0;
    cvta.to.global.u64 %allow, %allow;
    setp.ne.u32 %filter, %topk, 0;
    setp.lt.f32 %p, %topp, 0f3F800000;
    or.pred %filter, %filter, %p;
    setp.gt.f32 %p, %minp, 0f00000000;
    or.pred %filter, %filter, %p;
    mov.u32 %round, 0;
BASE
    mov.u32 %x, %ctaid.x;
    mov.u32 %nb, %nctaid.x;
    mad.lo.u32 %e, %x, 1024, %th;
    shl.b32 %e, %e, 2;
    shl.b32 %stride, %nb, 12;
    mov.f32 %bv, 0fFF800000;
    mov.u32 %bi, -1;
    mov.f32 %kv, 0fFF800000;
    mov.u32 %ki, -1;
    mov.f32 %m, 0fFF800000;
    mov.f32 %z, 0f00000000;
    mov.f32 %thr, 0fFF800000;
TRANSFORM:
    setp.ge.u32 %p, %e, %vocab;
    @%p bra TRANSFORMED;
FOUR
    shr.u32 %x, %e, 5;
    mul.wide.u32 %o, %x, 4;
    add.u64 %o, %seen, %o;
    ld.global.u32 %sw, [%o];
    mov.u32 %aw, -1;
    @!%hmask bra MASKED;
    mul.wide.u32 %o, %x, 4;
    add.u64 %o, %allow, %o;
    ld.global.u32 %aw, [%o];
MASKED:
    and.b32 %sh, %e, 31;
ELEMENTS
    mul.wide.u32 %o, %e, 4;
    add.u64 %a, %wk, %o;
    st.global.v4.f32 [%a], {%x0, %x1, %x2, %x3};
    add.u32 %e, %e, %stride;
    bra TRANSFORM;
TRANSFORMED:
";

/// Every block's maximum, sum and candidate to `partial`; the last block reads them all back and merges them.
const MERGE: &str = "
ARGMAX_B
    sub.f32 %t, %m, %bv;
    mul.f32 %t, %t, 0f3FB8AA3B;
    ex2.approx.f32 %t, %t;
    mul.f32 %z, %z, %t;
    setp.eq.f32 %p, %m, 0fFF800000;
    @%p mov.f32 %z, 0f00000000;
SUM_Z
ARGMAX_K
    mov.u32 %x, %ctaid.x;
    mul.wide.u32 %o, %x, 32;
    add.u64 %a, %part, %o;
    setp.eq.u32 %p, %th, 0;
    @%p st.global.v4.b32 [%a], {%bv, %bi, %z, %kv};
    @%p st.global.u32 [%a+16], %ki;
    bar.sync 0;
    setp.ne.u32 %p, %th, 0;
    @%p bra COUNTED;
    ld.param.u64 %a, [pcount];
    cvta.to.global.u64 %a, %a;
    fence.acq_rel.gpu;
    atom.global.add.u32 %x, [%a], 1;
    sub.u32 %h, %nb, 1;
    setp.eq.u32 %last, %x, %h;
    @%last st.global.u32 [%a], 0;
    fence.acq_rel.gpu;
    selp.u32 %x, 1, 0, %last;
    st.shared.u32 [flag], %x;
COUNTED:
    bar.sync 0;
    ld.shared.u32 %x, [flag];
    setp.eq.u32 %p, %x, 0;
    @%p bra EXIT;
    mov.f32 %bv, 0fFF800000;
    mov.u32 %bi, -1;
    mov.f32 %z, 0f00000000;
    mov.f32 %kv, 0fFF800000;
    mov.u32 %ki, -1;
    mul.wide.u32 %o, %th, 32;
    add.u64 %a, %part, %o;
    setp.lt.u32 %q, %th, %nb;
    @%q ld.global.cg.v4.b32 {%bv, %bi, %z, %kv}, [%a];
    @%q ld.global.cg.u32 %ki, [%a+16];
    mov.f32 %m, %bv;
ARGMAX_B
    sub.f32 %t, %m, %bv;
    mul.f32 %t, %t, 0f3FB8AA3B;
    ex2.approx.f32 %t, %t;
    mul.f32 %z, %z, %t;
    setp.eq.f32 %p, %m, 0fFF800000;
    @%p mov.f32 %z, 0f00000000;
SUM_Z
ARGMAX_K
    mov.u32 %amax, %bi;
    mov.u32 %tok, %bi;
    @%greedy bra FINISH;
    mov.u32 %tok, %ki;
    @!%filter bra FINISH;
    lg2.approx.f32 %minp, %minp;
    fma.rn.f32 %minp, %minp, 0f3F317218, %bv;
    setp.ge.f32 %p, %topp, 0f3F800000;
    @%p mov.f32 %topp, 0f7F800000;
    mul.f32 %topp, %topp, %z;
    cvt.rn.f32.u32 %kf, %topk;
    setp.eq.u32 %p, %topk, 0;
    @%p mov.f32 %kf, 0f7F800000;
";

/// The last block's checks: count and mass above the candidate, then a fresh draw above it if it fails.
const CHECK: &str = "
CHECK:
    mul.wide.u32 %o, %tok, 4;
    add.u64 %a, %wk, %o;
    ld.global.cg.f32 %lj, [%a];
    mov.f32 %cnt, 0f00000000;
    mov.f32 %mass, 0f00000000;
    shl.b32 %e, %th, 2;
COUNT:
    setp.ge.u32 %p, %e, %vocab;
    @%p bra TALLIED;
SIXTEEN
TALLY
    add.u32 %e, %e, 16384;
    bra COUNT;
TALLIED:
SUMS
    setp.lt.f32 %ok, %cnt, %kf;
    setp.lt.f32 %p, %mass, %topp;
    setp.eq.f32 %q, %mass, 0f00000000;
    or.pred %p, %p, %q;
    and.pred %ok, %ok, %p;
    setp.ge.f32 %p, %lj, %minp;
    and.pred %ok, %ok, %p;
    @%ok bra FINISH;
    mov.f32 %thr, %lj;
    add.u32 %round, %round, 1;
    mov.u32 %tok, %amax;
    setp.ge.u32 %p, %round, 32;
    @%p bra FINISH;
BASE
    mov.f32 %kv, 0fFF800000;
    mov.u32 %ki, -1;
    shl.b32 %e, %th, 2;
DRAW:
    setp.ge.u32 %p, %e, %vocab;
    @%p bra DRAWN;
SIXTEEN
KEYS
    add.u32 %e, %e, 16384;
    bra DRAW;
DRAWN:
ARGMAX_K
    mov.u32 %tok, %ki;
    bra CHECK;
FINISH:
    setp.ne.u32 %p, %th, 0;
    @%p bra EXIT;
    st.global.u32 [%st+4], %tok;
    add.u32 %x, %ctr, 1;
    st.global.u32 [%st], %x;
    shr.u32 %x, %tok, 5;
    mul.wide.u32 %o, %x, 4;
    add.u64 %a, %seen, %o;
    and.b32 %x, %tok, 31;
    mov.u32 %h, 1;
    shl.b32 %x, %h, %x;
    atom.global.or.b32 %h, [%a], %x;
    ld.param.u64 %a, [pout];
    setp.eq.u64 %p, %a, 0;
    @%p bra EXIT;
    cvta.to.global.u64 %a, %a;
    st.global.u32 [%a], %tok;
EXIT:
    ret;
}
";

/// The sampler.
pub(super) fn sample() -> String {
    let elements: String = (0..4).map(element).collect();
    let (mut tally, mut keys) = (String::new(), String::new());
    tally += "    setp.ne.u32 %q, %th, %th;\n";
    for k in 0..16 {
        let _ = writeln!(tally, "    setp.gt.f32 %p, %x{k}, %lj;\n    @%p add.f32 %cnt, %cnt, 0f3F800000;\n    or.pred %q, %q, %p;");
    }
    // Only tokens above the candidate add mass; with a model's peaked logits most warps have none.
    tally += "    vote.sync.any.pred %q, %q, -1;\n    @!%q bra MASSED;\n";
    for k in 0..16 {
        let _ = writeln!(tally, "    setp.gt.f32 %p, %x{k}, %lj;\n    sub.f32 %t, %x{k}, %bv;\n    mul.f32 %t, %t, 0f3FB8AA3B;\n    ex2.approx.f32 %t, %t;");
        let _ = writeln!(tally, "    @%p add.f32 %mass, %mass, %t;");
    }
    tally += "MASSED:\n";
    for k in 0..16 {
        let _ = writeln!(keys, "    add.u32 %oi, %e, {};", 4096 * (k / 4) + k % 4);
        keys += &gumbel(&format!("%x{k}"), "%oi");
        keys += &better("%t", "%oi", "%kv", "%ki");
    }
    let head = HEAD.replace("BLOCK\n", BLOCK).replace("BASE\n", &base()).replace("FOUR\n", FOUR).replace("ELEMENTS\n", &elements);
    let merge =
        MERGE.replace("ARGMAX_B\n", &argmax("%bv", "%bi")).replace("ARGMAX_K\n", &argmax("%kv", "%ki")).replace("SUM_Z\n", &block("add.f32", &["%z"], 32));
    let check = CHECK
        .replace("BASE\n", &base())
        .replace("SIXTEEN\n", &sixteen())
        .replace("TALLY\n", &tally)
        .replace("KEYS\n", &keys)
        .replace("SUMS\n", &block("add.f32", &["%cnt", "%mass"], 32))
        .replace("ARGMAX_K\n", &argmax("%kv", "%ki"));
    head + &merge + &check
}