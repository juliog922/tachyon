//! The decode GEMV, Q4 weights × Q8 activations, on every Gemma 4 E4B projection the engine runs, against the
//! GPU's measured VRAM read bandwidth. Checks the step-3 exit gate: one E4B token's GEMVs, weighted by the bytes
//! each reads, stream at ≥ 90% of that bandwidth. Each shape is reported too; small ones fall short because every
//! kernel pays a fixed cost (a DRAM round trip and the gap between kernels), which step 6 hides by prefetching.
//!
//! `cargo bench -p tachyon --features gpu --bench gemv`
//!
//! Each shape runs as a graph of back-to-back launches, as decode does, over distinct copies of its weights that
//! add up to at least 16 times the L2 cache, so every byte comes from VRAM, as in a decode step.

mod harness;

use harness::Suite;
use std::process::ExitCode;
use std::time::Duration;
use tachyon::cuda::{Context, DevBuf, Event, Module, Stream, arg};
use tachyon::ptx::{GEMV_Q4, GEMV_ROWS, module};

const PROBE: &str = include_str!("../tests/kernels/probe.ptx");
/// Fewest launches per timed graph.
const LAUNCHES: usize = 64;

/// A decode projection and how many times one token runs it.
struct Shape {
    name: &'static str,
    rows: u32,
    cols: u32,
    per_token: u32,
}

const fn shape(name: &'static str, rows: u32, cols: u32, per_token: u32) -> Shape {
    Shape { name, rows, cols, per_token }
}

/// Gemma 4 E4B: hidden 2560, MLP 10240, vocabulary 262144; 8 query heads and 2 key-value heads of 256, or 512 in
/// the 7 global layers of 42; the last 18 layers reuse earlier layers' keys and values, so they project queries
/// only. Query, key and value are fused, and so are gate and up.
const SHAPES: [Shape; 9] = [
    shape("q, sliding, shared kv", 2048, 2560, 15),
    shape("qkv, sliding", 3072, 2560, 20),
    shape("q, global, shared kv", 4096, 2560, 3),
    shape("qkv, global", 6144, 2560, 4),
    shape("o, sliding", 2560, 2048, 35),
    shape("o, global", 2560, 4096, 7),
    shape("gate + up", 20480, 2560, 42),
    shape("down", 2560, 10240, 42),
    shape("lm head", 262_144, 2560, 1),
];

struct Gpu {
    ctx: Context,
    stream: Stream,
    module: Module,
    start: Event,
    end: Event,
}

impl Gpu {
    /// GPU time of the work `queue` puts on the stream.
    fn time(&self, queue: impl FnOnce(&Stream)) -> Duration {
        self.stream.record(&self.start).unwrap();
        queue(&self.stream);
        self.stream.record(&self.end).unwrap();
        self.end.sync().unwrap();
        Duration::from_secs_f64(f64::from(self.end.since(&self.start).unwrap()) / 1e3)
    }

    fn filled(&self, len: usize, byte: u8) -> DevBuf {
        let buf = self.ctx.alloc(len).unwrap();
        self.stream.fill(&buf, byte).unwrap();
        buf
    }

    /// Bytes that overflow the L2 cache 16 times, within half the free memory.
    fn uncached(&self) -> usize {
        (16 * self.ctx.info().l2 as usize).min(self.ctx.memory().unwrap().0 / 2)
    }
}

/// The best VRAM read bandwidth of the probe kernel, in GB/s: the ceiling every shape is measured against.
fn ceiling(gpu: &Gpu, suite: &mut Suite) -> f64 {
    let probe = gpu.ctx.load(PROBE).unwrap();
    let read = probe.function("xor_read").unwrap();
    let (len, sms) = (gpu.uncached().max(1 << 30) & !0xffff, gpu.ctx.info().sms);
    let (src, out) = (gpu.filled(len, 1), gpu.ctx.alloc(sms as usize * 8 * 512 * 4).unwrap());
    let (ptr, count, sink) = (src.ptr(), (len / 16) as u64, out.ptr());
    let mut best = 0f64;
    for per_sm in [4, 8] {
        let t = suite.run(&format!("vram read ceiling, {per_sm} blocks/SM"), 20, || {
            gpu.time(|s| {
                // SAFETY: three arguments of the kernel's types; `out` holds a word for each of the grid's threads.
                unsafe { s.launch(&read, [sms * per_sm, 1, 1], [512, 1, 1], 0, &[arg(&ptr), arg(&count), arg(&sink)]) }.unwrap();
            })
        });
        best = best.max(len as f64 / t.as_secs_f64() / 1e9);
        suite.rate(len as f64 / t.as_secs_f64() / 1e9, "GB/s");
    }
    best
}

/// Bytes of `s`'s packed weights and of their f16 scales.
fn sizes(s: &Shape) -> (usize, usize) {
    let weights = s.rows as usize * s.cols as usize;
    (weights / 2, weights / 32)
}

/// Bytes one launch of `s` reads.
fn bytes(s: &Shape) -> usize {
    sizes(s).0 + sizes(s).1
}

/// GB/s at which one shape streams its weights and scales.
fn measure(gpu: &Gpu, suite: &mut Suite, s: &Shape) -> f64 {
    let ((packed, scales), len, (rows, cols)) = (sizes(s), bytes(s), (s.rows, s.cols));
    let copies = gpu.uncached().div_ceil(len).max(1);
    let matrices: Vec<(DevBuf, DevBuf)> = (0..copies).map(|_| (gpu.filled(packed, 0x5a), gpu.filled(scales, 0x3c))).collect();
    let (q, xs, y) = (gpu.filled(cols as usize, 1), gpu.filled(cols as usize / 4, 0), gpu.ctx.alloc(rows as usize * 4).unwrap());
    let (gemv, launches) = (gpu.module.function(GEMV_Q4).unwrap(), copies.max(LAUNCHES));
    let graph = gpu
        .stream
        .capture(|st| {
            for (w, d) in matrices.iter().cycle().take(launches) {
                let (pw, pd, pq, ps, py) = (w.ptr(), d.ptr(), q.ptr(), xs.ptr(), y.ptr());
                let args = [arg(&pw), arg(&pd), arg(&pq), arg(&ps), arg(&py), arg(&rows), arg(&cols)];
                // SAFETY: seven arguments of the kernel's types; every buffer has the size its layout gives for `rows × cols`.
                unsafe { st.launch(&gemv, [rows.div_ceil(GEMV_ROWS), 1, 1], [32 * GEMV_ROWS, 1, 1], 0, &args) }?;
            }
            Ok(())
        })
        .unwrap();
    // SAFETY: the module and buffers the graph uses outlive every replay.
    let t = suite.run(&format!("{}: {rows}×{cols} ×{launches}", s.name), 20, || gpu.time(|st| unsafe { st.replay(&graph) }.unwrap()));
    let gbs = len as f64 * launches as f64 / t.as_secs_f64() / 1e9;
    suite.rate(gbs, "GB/s");
    gbs
}

/// The step-3 gate: one token's GEMVs, each timed at its measured rate, against the ceiling.
fn gate(roof: f64, rates: &[f64]) -> bool {
    let (mut total, mut seconds) = (0.0, 0.0);
    for (s, gbs) in SHAPES.iter().zip(rates) {
        let token_bytes = f64::from(s.per_token) * bytes(s) as f64;
        (total, seconds) = (total + token_bytes, seconds + token_bytes / (gbs * 1e9));
        println!("{} {}×{} ×{}: {gbs:.1} GB/s, {:.0}% of the ceiling", s.name, s.rows, s.cols, s.per_token, 100.0 * gbs / roof);
    }
    let token = total / seconds / 1e9;
    let passed = token >= 0.9 * roof;
    println!(
        "gate: one E4B token's GEMVs read {:.2} GB at {token:.1} GB/s, {:.0}% of the ceiling (≥ 90%): {}",
        total / 1e9,
        100.0 * token / roof,
        if passed { "PASS" } else { "FAIL" }
    );
    passed
}

fn main() -> ExitCode {
    let ctx = Context::new(0).expect("GPU 0 with compute capability 8.0 or newer");
    let (stream, module) = (ctx.stream().unwrap(), ctx.load(&module()).unwrap());
    let (start, end) = (ctx.event(true).unwrap(), ctx.event(true).unwrap());
    let gpu = Gpu { ctx, stream, module, start, end };
    let mut suite = Suite::new("gemv");
    let roof = ceiling(&gpu, &mut suite);
    let rates: Vec<f64> = SHAPES.iter().map(|s| measure(&gpu, &mut suite, s)).collect();
    println!("{} · read ceiling {roof:.1} GB/s · {} MiB of weights per shape to defeat the L2 cache", gpu.ctx.info().name, gpu.uncached() >> 20);
    let passed = gate(roof, &rates);
    let code = suite.finish();
    if passed { code } else { ExitCode::FAILURE }
}