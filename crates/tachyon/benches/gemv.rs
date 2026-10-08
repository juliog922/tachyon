//! The decode GEMV, Q4 weights × Q8 activations, on every Gemma 4 E4B projection the engine runs, against the
//! GPU's measured VRAM read bandwidth. Checks the step-3 exit gate: every shape streams its weights and scales at
//! ≥ 90% of that bandwidth.
//!
//! `cargo bench -p tachyon --features gpu --bench gemv`
//!
//! Each shape runs as a graph of back-to-back launches, as decode does, cycling through enough copies of its
//! weights to overflow the L2 cache: in a decode step no weight is read twice, so every byte comes from VRAM.

mod harness;

use harness::Suite;
use std::process::ExitCode;
use std::time::Duration;
use tachyon::cuda::{Context, DevBuf, Event, Module, Stream, arg};
use tachyon::ptx::{GEMV_Q4, GEMV_ROWS, module};

const PROBE: &str = include_str!("../tests/kernels/probe.ptx");
/// Launches per timed graph.
const LAUNCHES: usize = 64;
/// Gemma 4 E4B decode projections as (name, rows, cols): hidden 2560, 8 query heads and 2 key-value heads of 256
/// (512 in global layers), MLP 10240, vocabulary 262144. Query, key and value are fused, and so are gate and up.
const SHAPES: [(&str, u32, u32); 8] = [
    ("q, layers sharing kv", 2048, 2560),
    ("qkv, sliding", 3072, 2560),
    ("qkv, global", 6144, 2560),
    ("o, sliding", 2560, 2048),
    ("o, global", 2560, 4096),
    ("gate + up", 20480, 2560),
    ("down", 2560, 10240),
    ("lm head", 262_144, 2560),
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
}

/// The best VRAM read bandwidth of the probe kernel, in GB/s: the ceiling every shape is measured against.
fn ceiling(gpu: &Gpu, suite: &mut Suite) -> f64 {
    let probe = gpu.ctx.load(PROBE).unwrap();
    let read = probe.function("xor_read").unwrap();
    let (len, sms) = ((gpu.ctx.memory().unwrap().0 / 4).min(1 << 30) & !0xffff, gpu.ctx.info().sms);
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

/// GB/s at which one shape streams its weights and scales.
fn shape(gpu: &Gpu, suite: &mut Suite, spec: (&str, u32, u32)) -> f64 {
    let (name, rows, cols) = spec;
    let (weights, scales) = (rows as usize * cols as usize / 2, rows as usize * cols as usize / 32);
    let copies = (3 * gpu.ctx.info().l2 as usize).div_ceil(weights + scales);
    let matrices: Vec<(DevBuf, DevBuf)> = (0..copies).map(|_| (gpu.filled(weights, 0x5a), gpu.filled(scales, 0x3c))).collect();
    let (q, s, y) = (gpu.filled(cols as usize, 1), gpu.filled(cols as usize / 4, 0), gpu.ctx.alloc(rows as usize * 4).unwrap());
    let gemv = gpu.module.function(GEMV_Q4).unwrap();
    let graph = gpu
        .stream
        .capture(|st| {
            for (w, d) in matrices.iter().cycle().take(LAUNCHES) {
                let (pw, pd, pq, ps, py) = (w.ptr(), d.ptr(), q.ptr(), s.ptr(), y.ptr());
                let args = [arg(&pw), arg(&pd), arg(&pq), arg(&ps), arg(&py), arg(&rows), arg(&cols)];
                // SAFETY: seven arguments of the kernel's types; every buffer has the size its layout gives for `rows × cols`.
                unsafe { st.launch(&gemv, [rows.div_ceil(GEMV_ROWS), 1, 1], [32 * GEMV_ROWS, 1, 1], 0, &args) }?;
            }
            Ok(())
        })
        .unwrap();
    // SAFETY: the module and buffers the graph uses outlive every replay.
    let t = suite.run(&format!("{name}: {rows}×{cols} ×{LAUNCHES}"), 20, || gpu.time(|st| unsafe { st.replay(&graph) }.unwrap()));
    let gbs = (weights + scales) as f64 * LAUNCHES as f64 / t.as_secs_f64() / 1e9;
    suite.rate(gbs, "GB/s");
    gbs
}

fn main() -> ExitCode {
    let ctx = Context::new(0).expect("GPU 0 with compute capability 8.0 or newer");
    let (stream, module) = (ctx.stream().unwrap(), ctx.load(&module()).unwrap());
    let (start, end) = (ctx.event(true).unwrap(), ctx.event(true).unwrap());
    let gpu = Gpu { ctx, stream, module, start, end };
    let mut suite = Suite::new("gemv");
    let roof = ceiling(&gpu, &mut suite);
    let rates: Vec<f64> = SHAPES.iter().map(|&s| shape(&gpu, &mut suite, s)).collect();
    println!("{} · read ceiling {roof:.1} GB/s", gpu.ctx.info().name);
    let mut passed = true;
    for ((name, rows, cols), gbs) in SHAPES.iter().zip(&rates) {
        let ok = *gbs >= 0.9 * roof;
        passed &= ok;
        println!(
            "gate: {name} {rows}×{cols} streams at {gbs:.1} GB/s, {:.0}% of the ceiling (≥ 90%): {}",
            100.0 * gbs / roof,
            if ok { "PASS" } else { "FAIL" }
        );
    }
    let code = suite.finish();
    if passed { code } else { ExitCode::FAILURE }
}
