//! The decode kernels around the GEMV, each timed per launch on every Gemma 4 E4B shape, and one token's worth of
//! them against a decode step whose GEMVs stream at the measured VRAM ceiling, at contexts of 1K, 4K and 16K tokens.
//! The share is reported, not gated: as separate launches these kernels pay a fixed cost each, which step 6 removes
//! by fusing them into the GEMVs and overlapping them with weight prefetching, and step 6's end-to-end gate judges
//! the result. Each kernel's time is checked against the machine's baseline.
//!
//! `cargo bench -p tachyon --features gpu --bench decode`
//!
//! Each kernel runs as a graph of back-to-back launches. Attention cycles through copies of its cache that add up
//! to 16 times the L2 cache, so the cache comes from VRAM, as in a decode step.

// Buffers are named as the kernels name them (h, y, w, q, s), and sit beside their pointers (part, ppart).
#![allow(clippy::many_single_char_names, clippy::similar_names)]

mod harness;

use harness::Suite;
use harness::gpu::{Gpu, token_bytes};
use std::process::ExitCode;
use std::time::Duration;
use tachyon::cuda::{DevBuf, Stream, arg};
use tachyon::ptx::{ATTEND_256, ATTEND_512, ATTEND_WARPS, EMBED_Q4, GEGLU_Q8, NORM_Q8, NORM_THREADS, SAMPLE, SAMPLE_THREADS, Sampling, rope};

/// Launches per timed graph, at least.
const LAUNCHES: usize = 64;
/// Contexts reported, in tokens.
const CONTEXTS: [u32; 3] = [1024, 4096, 16_384];
/// E4B's key-value heads.
const KV: usize = 2;
/// E4B's sliding window, in positions.
const WINDOW: u32 = 512;

/// Time of one launch: a graph of `n` launches of `queue(stream, i)`, divided by `n`.
fn per_launch(gpu: &Gpu, suite: &mut Suite, name: &str, n: usize, queue: impl Fn(&Stream, usize) -> tachyon::Result<()>) -> Duration {
    let graph = gpu.stream.capture(|s| (0..n).try_for_each(|i| queue(s, i))).unwrap();
    let t = gpu.replay(suite, &format!("{name} ×{n}"), &graph) / n as u32;
    suite.rate(t.as_secs_f64() * 1e6, "µs each");
    t
}

/// [`NORM_Q8`] with a residual input and a quantized output, over `chunks` blocks of `len`.
fn norm(gpu: &Gpu, suite: &mut Suite, len: u32, chunks: u32) -> Duration {
    let n = (len * chunks) as usize;
    let (h, ho, y, w, q, s) =
        (gpu.filled(4 * n, 0x3c), gpu.filled(4 * n, 0x3c), gpu.filled(4 * n, 0x3d), gpu.filled(4 * len as usize, 0x3f), gpu.filled(n, 0), gpu.filled(n / 4, 0));
    let [ph, pho, py, pw, pq, ps] = [&h, &ho, &y, &w, &q, &s].map(DevBuf::ptr);
    let (eps, scale, f) = (1e-6f32, 1f32, gpu.module.function(NORM_Q8).unwrap());
    per_launch(gpu, suite, &format!("norm {len}×{chunks}"), LAUNCHES, |st, _| {
        let args = [arg(&ph), arg(&pho), arg(&py), arg(&pw), arg(&pw), arg(&pq), arg(&ps), arg(&len), arg(&eps), arg(&scale)];
        // SAFETY: ten arguments of the kernel's types; every buffer holds `len × chunks` values.
        unsafe { st.launch(&f, [len / NORM_THREADS, chunks, 1], [NORM_THREADS, 1, 1], 0, &args) }
    })
}

/// [`GEGLU_Q8`] over `len` values.
fn geglu(gpu: &Gpu, suite: &mut Suite, len: u32) -> Duration {
    let n = len as usize;
    let (a, q, s) = (gpu.filled(8 * n, 0x3c), gpu.filled(n, 0), gpu.filled(n / 4, 0));
    let (pa, pb, pq, ps, f) = (a.ptr(), a.ptr() + 4 * n as u64, q.ptr(), s.ptr(), gpu.module.function(GEGLU_Q8).unwrap());
    per_launch(gpu, suite, &format!("geglu {len}"), LAUNCHES, |st, _| {
        // SAFETY: five arguments of the kernel's types; every buffer holds `len` values.
        unsafe { st.launch(&f, [len.div_ceil(256), 1, 1], [256, 1, 1], 0, &[arg(&pa), arg(&pb), arg(&pq), arg(&ps), arg(&len)]) }
    })
}

/// [`EMBED_Q4`] of a row of `cols`, from VRAM or from pinned host memory.
fn embed(gpu: &Gpu, suite: &mut Suite, cols: u32, host: bool) -> Duration {
    let rows = 1024usize;
    let bytes = rows * cols as usize * 17 / 32;
    let (table, pinned) = (gpu.filled(if host { 1 } else { bytes }, 0x5a), gpu.ctx.alloc_host(if host { bytes } else { 1 }).unwrap());
    let base = if host { pinned.device_ptr().unwrap() } else { table.ptr() };
    let (pw, ps, tok, out) = (base, base + (rows * cols as usize / 2) as u64, gpu.filled(4, 0), gpu.filled(4 * cols as usize, 0));
    let (pt, po, scale, f) = (tok.ptr(), out.ptr(), 1f32, gpu.module.function(EMBED_Q4).unwrap());
    per_launch(gpu, suite, &format!("embed {cols} from {}", if host { "host" } else { "vram" }), LAUNCHES, |st, _| {
        // SAFETY: six arguments of the kernel's types; the table holds `rows × cols` weights.
        unsafe { st.launch(&f, [cols.div_ceil(256), 1, 1], [256, 1, 1], 0, &[arg(&pw), arg(&ps), arg(&pt), arg(&po), arg(&cols), arg(&scale)]) }
    })
}

/// [`ATTEND_256`] or [`ATTEND_512`] over a cache of `cap` slots with the new token at position `ctx − 1`.
fn attend(gpu: &Gpu, suite: &mut Suite, d: usize, cap: u32, ctx: u32) -> Duration {
    let half = KV * cap as usize * d * 2;
    let copies = gpu.uncached().div_ceil(2 * half).max(1);
    let (k, v) = (gpu.filled(copies * half, 0x3c), gpu.filled(copies * half, 0x3c));
    let freq = rope(if d == 256 { 10_000.0 } else { 1e6 }, d, if d == 256 { 1.0 } else { 0.25 });
    let freq = gpu.ctx.alloc(4 * freq.len()).and_then(|b| upload(gpu, b, &freq)).unwrap();
    // Sliding: 16 blocks per head, two per multiprocessor; global: one block per multiprocessor, in one wave.
    let splits = if d == 256 { 16 } else { (gpu.ctx.info().sms / KV as u32).clamp(1, 256) };
    let (qkv, w, pos) = (gpu.filled(4 * 6 * KV * d, 0x3c), gpu.filled(4 * d, 0x3f), gpu.ctx.alloc(4).and_then(|b| upload(gpu, b, &[ctx - 1])).unwrap());
    let (part, count, q, s) = (gpu.filled(KV * splits as usize * (8 + 4 * d) * 4, 0), gpu.filled(4 * KV, 0), gpu.filled(4 * KV * d, 0), gpu.filled(KV * d, 0));
    let [pqkv, pw, pf, pp, ppart, pc, pq, ps] = [&qkv, &w, &freq, &pos, &part, &count, &q, &s].map(DevBuf::ptr);
    let f = gpu.module.function(if d == 256 { ATTEND_256 } else { ATTEND_512 }).unwrap();
    let name = format!("attend {d}, {} of {cap} positions", ctx.min(cap));
    per_launch(gpu, suite, &name, copies.max(LAUNCHES), |st, i| {
        let (pk, pv, fresh) = (k.ptr() + (i % copies * half) as u64, v.ptr() + (i % copies * half) as u64, 1u32);
        let args = [arg(&pqkv), arg(&pw), arg(&pw), arg(&pf), arg(&pk), arg(&pv), arg(&pp), arg(&cap), arg(&fresh), arg(&ppart), arg(&pc), arg(&pq), arg(&ps)];
        // SAFETY: thirteen arguments of the kernel's types, every buffer sized as the kernel's documentation gives.
        unsafe { st.launch(&f, [KV as u32, splits, 1], [32 * ATTEND_WARPS, 1, 1], 0, &args) }
    })
}

/// `data` copied into `buf`.
fn upload<T: Copy>(gpu: &Gpu, buf: DevBuf, data: &[T]) -> tachyon::Result<DevBuf> {
    // SAFETY: plain numbers viewed as bytes; `data` outlives the copy, which `sync` completes.
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), size_of_val(data)) };
    // SAFETY: as above.
    unsafe { gpu.stream.upload(&buf, 0, bytes) }?;
    gpu.stream.sync().map(|()| buf)
}

/// [`SAMPLE`] over E4B's 262 144 logits, shaped like a language model's: a normal-looking bulk, and 64 tokens
/// 20 to 24 above it that hold nearly all the probability.
fn sample(gpu: &Gpu, suite: &mut Suite, name: &str, cfg: Sampling) -> Duration {
    let vocab = 262_144u32;
    let mut x = 0x2545_F491_4F6C_DD1Du64;
    let mut logits: Vec<f32> = (0..vocab)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (0..4).map(|k| (x >> (16 * k) & 0xffff) as f32 / 65536.0).sum::<f32>() * 6.0 - 12.0
        })
        .collect();
    (7..vocab as usize).step_by(4096).for_each(|i| logits[i] += 20.0 + (i % 5) as f32);
    let blocks = vocab.div_ceil(4 * SAMPLE_THREADS).min(gpu.ctx.info().sms);
    let (logits, work) = (gpu.ctx.alloc(4 * vocab as usize).and_then(|b| upload(gpu, b, &logits)).unwrap(), gpu.filled(4 * vocab as usize, 0));
    let cfg = gpu.ctx.alloc(40).and_then(|b| upload(gpu, b, &cfg.bytes())).unwrap();
    let (state, seen, part, count) = (gpu.filled(8, 0), gpu.filled(vocab as usize / 8, 0), gpu.filled(32 * blocks as usize, 0), gpu.filled(4, 0));
    let [pl, pw, pc, pst, pse, ppa, pco] = [&logits, &work, &cfg, &state, &seen, &part, &count].map(DevBuf::ptr);
    let f = gpu.module.function(SAMPLE).unwrap();
    per_launch(gpu, suite, &format!("sample {vocab}, {name}"), LAUNCHES, |st, _| {
        let args = [arg(&pl), arg(&pw), arg(&vocab), arg(&pc), arg(&pst), arg(&pse), arg(&0u64), arg(&ppa), arg(&pco)];
        // SAFETY: nine arguments of the kernel's types; every buffer is sized for `vocab`.
        unsafe { st.launch(&f, [blocks, 1, 1], [SAMPLE_THREADS, 1, 1], 0, &args) }
    })
}

/// Time of one token's non-GEMV kernels other than attention: per-layer steps times 42, per-token steps once.
fn fixed(gpu: &Gpu, suite: &mut Suite) -> Duration {
    let layers = 42;
    let hidden = norm(gpu, suite, 2560, 1);
    let per_layer = hidden * 3 + geglu(gpu, suite, 10_240) + geglu(gpu, suite, 256);
    let per_token = hidden * 2 + norm(gpu, suite, 256, 42) + embed(gpu, suite, 2560, false) + embed(gpu, suite, 10_752, true);
    sample(gpu, suite, "greedy", Sampling { softcap: 30.0, ..Sampling::default() });
    let draw = Sampling { softcap: 30.0, temperature: 1.0, top_k: 64, top_p: 0.95, min_p: 0.05, penalty: 1.1, ..Sampling::default() };
    per_layer * layers + per_token + sample(gpu, suite, "top-k, top-p, min-p", draw)
}

fn main() -> ExitCode {
    let gpu = Gpu::new();
    let mut suite = Suite::new("decode");
    let roof = gpu.ceiling(&mut suite);
    let gemv = Duration::from_secs_f64(token_bytes() / (roof * 1e9));
    let fixed = fixed(&gpu, &mut suite);
    let sliding = attend(&gpu, &mut suite, 256, WINDOW, WINDOW) * 35;
    println!("{} · read ceiling {roof:.1} GB/s · one token's GEMVs at the ceiling: {gemv:.2?}", gpu.ctx.info().name);
    for ctx in CONTEXTS {
        let attention = sliding + attend(&gpu, &mut suite, 512, ctx, ctx) * 7;
        let others = fixed + attention;
        let share = others.as_secs_f64() / (gemv + others).as_secs_f64();
        println!("context {ctx}: non-GEMV kernels {others:.2?} per token (attention {attention:.2?}), {:.1}% of the step", 100.0 * share);
    }
    suite.finish()
}