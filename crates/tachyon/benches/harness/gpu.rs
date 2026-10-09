//! What the GPU benches share: timing with events, the VRAM read ceiling, and the decode projections of Gemma 4 E4B.

use super::Suite;
use std::time::Duration;
use tachyon::cuda::{Context, DevBuf, Event, Graph, Module, Stream, arg};

/// A null pointer or zero length: the prefetch arguments of a launch that prefetches nothing.
pub static NONE: u64 = 0;

const PROBE: &str = include_str!("../../tests/kernels/probe.ptx");

/// GPU 0 with every kernel loaded, a stream, and two timing events.
pub struct Gpu {
    pub ctx: Context,
    pub stream: Stream,
    pub module: Module,
    start: Event,
    end: Event,
}

impl Gpu {
    pub fn new() -> Gpu {
        let ctx = Context::new(0).expect("GPU 0 with compute capability 8.0 or newer");
        let (stream, module) = (ctx.stream().unwrap(), ctx.load(&tachyon::ptx::module()).unwrap());
        let (start, end) = (ctx.event(true).unwrap(), ctx.event(true).unwrap());
        Gpu { ctx, stream, module, start, end }
    }

    /// GPU time of the work `queue` puts on the stream.
    pub fn time(&self, queue: impl FnOnce(&Stream)) -> Duration {
        self.stream.record(&self.start).unwrap();
        queue(&self.stream);
        self.stream.record(&self.end).unwrap();
        self.end.sync().unwrap();
        Duration::from_secs_f64(f64::from(self.end.since(&self.start).unwrap()) / 1e3)
    }

    /// GPU time of one replay of `graph`, the median of `runs`, recorded in `suite` as `name`.
    pub fn replay(&self, suite: &mut Suite, name: &str, graph: &Graph) -> Duration {
        // SAFETY: callers keep every buffer and module the graph uses alive.
        suite.run(name, 20, || self.time(|s| unsafe { s.replay(graph) }.unwrap()))
    }

    /// `len` bytes set to `byte`.
    pub fn filled(&self, len: usize, byte: u8) -> DevBuf {
        let buf = self.ctx.alloc(len).unwrap();
        self.stream.fill(&buf, byte).unwrap();
        buf
    }

    /// Bytes that overflow the L2 cache 16 times, within half the free memory.
    pub fn uncached(&self) -> usize {
        (16 * self.ctx.info().l2 as usize).min(self.ctx.memory().unwrap().0 / 2)
    }

    /// The best VRAM read bandwidth of the probe kernel, in GB/s: the ceiling every rate is measured against.
    pub fn ceiling(&self, suite: &mut Suite) -> f64 {
        let probe = self.ctx.load(PROBE).unwrap();
        let read = probe.function("xor_read").unwrap();
        let (len, sms) = (self.uncached().max(1 << 30) & !0xffff, self.ctx.info().sms);
        let (src, out) = (self.filled(len, 1), self.ctx.alloc(sms as usize * 8 * 512 * 4).unwrap());
        let (ptr, count, sink) = (src.ptr(), (len / 16) as u64, out.ptr());
        let mut best = 0f64;
        for per_sm in [4, 8] {
            let t = suite.run(&format!("vram read ceiling, {per_sm} blocks/SM"), 20, || {
                self.time(|s| {
                    // SAFETY: three arguments of the kernel's types; `out` holds a word for each of the grid's threads.
                    unsafe { s.launch(&read, [sms * per_sm, 1, 1], [512, 1, 1], 0, &[arg(&ptr), arg(&count), arg(&sink)]) }.unwrap();
                })
            });
            best = best.max(len as f64 / t.as_secs_f64() / 1e9);
            suite.rate(len as f64 / t.as_secs_f64() / 1e9, "GB/s");
        }
        best
    }
}

/// A decode projection and how many times one token runs it.
pub struct Shape {
    pub name: &'static str,
    pub rows: u32,
    pub cols: u32,
    pub per_token: u32,
}

const fn shape(name: &'static str, rows: u32, cols: u32, per_token: u32) -> Shape {
    Shape { name, rows, cols, per_token }
}

/// Gemma 4 E4B: hidden 2560, MLP 10240, vocabulary 262144; 8 query heads and 2 key-value heads of 256, or 512 in
/// the 7 global layers of 42; the last 18 layers reuse earlier layers' keys and values, so they project queries
/// only. Query, key and value are fused, and so are gate and up. Each layer also gates a 256-wide per-layer input
/// and projects it back, and each token projects its embedding to all 42 per-layer inputs.
pub const SHAPES: [Shape; 12] = [
    shape("q, sliding, shared kv", 2048, 2560, 15),
    shape("qkv, sliding", 3072, 2560, 20),
    shape("q, global, shared kv", 4096, 2560, 3),
    shape("qkv, global", 6144, 2560, 4),
    shape("o, sliding", 2560, 2048, 35),
    shape("o, global", 2560, 4096, 7),
    shape("gate + up", 20480, 2560, 42),
    shape("down", 2560, 10240, 42),
    shape("per-layer gate", 256, 2560, 42),
    shape("per-layer projection", 2560, 256, 42),
    shape("per-layer inputs", 10752, 2560, 1),
    shape("lm head", 262_144, 2560, 1),
];

impl Shape {
    /// Bytes of the packed weights and of their f16 scales.
    pub fn sizes(&self) -> (usize, usize) {
        let weights = self.rows as usize * self.cols as usize;
        (weights / 2, weights / 32)
    }

    /// Bytes one launch reads.
    pub fn bytes(&self) -> usize {
        self.sizes().0 + self.sizes().1
    }
}

/// Bytes of weights one E4B token reads.
pub fn token_bytes() -> f64 {
    SHAPES.iter().map(|s| f64::from(s.per_token) * s.bytes() as f64).sum()
}