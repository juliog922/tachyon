//! Machine ceilings, the measured limits every later target is a fraction of:
//! VRAM read bandwidth (the decode roofline), PCIe in both directions, and the
//! CPU cost of queueing work. Also checks the step-1 exit gate.
//!
//! `cargo bench -p tachyon --features gpu --bench machine`

mod harness;

use harness::{Suite, cpu};
use std::process::ExitCode;
use std::time::Duration;
use tachyon::cuda::{Context, DeviceInfo, Event, Module, Stream, arg};

const PTX: &str = include_str!("../tests/kernels/probe.ptx");

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
}

fn gbs(bytes: usize, t: Duration) -> f64 {
    bytes as f64 / t.as_secs_f64() / 1e9
}

fn main() -> ExitCode {
    let ctx = Context::new(0).expect("GPU 0 with compute capability 8.0 or newer");
    let (stream, module) = (ctx.stream().unwrap(), ctx.load(PTX).unwrap());
    let (start, end) = (ctx.event(true).unwrap(), ctx.event(true).unwrap());
    let gpu = Gpu { ctx, stream, module, start, end };
    let info = gpu.ctx.info().clone();
    let version = tachyon::cuda::driver_version().unwrap();
    println!(
        "{} · sm_{}{} · {} SMs · {:.1} GiB · {:.0} GB/s datasheet · CUDA {}.{} driver",
        info.name,
        info.compute.0,
        info.compute.1,
        info.sms,
        info.vram as f64 / f64::from(1u32 << 30),
        info.nominal_bandwidth() / 1e9,
        version / 1000,
        version % 1000 / 10
    );
    let mut suite = Suite::new("machine");
    let read = vram(&gpu, &mut suite);
    let h2d = pcie(&gpu, &mut suite);
    let graph = launches(&gpu, &mut suite);
    println!("decode roofline: {read:.1} GB/s measured read bandwidth ({:.0}% of datasheet)", 100.0 * read * 1e9 / info.nominal_bandwidth());
    let passed = gates(&info, h2d, graph);
    let code = suite.finish();
    if passed { code } else { ExitCode::FAILURE }
}

/// VRAM bandwidth: the read kernel at several grid sizes, and the driver's device copy. Returns the best read rate.
fn vram(gpu: &Gpu, suite: &mut Suite) -> f64 {
    let len = (gpu.ctx.memory().unwrap().0 / 4).min(1 << 30) & !0xffff;
    let (src, dst) = (gpu.ctx.alloc(len).unwrap(), gpu.ctx.alloc(len).unwrap());
    gpu.stream.fill(&src, 1).unwrap();
    let (read, sms) = (gpu.module.function("xor_read").unwrap(), gpu.ctx.info().sms);
    let out = gpu.ctx.alloc(sms as usize * 32 * 512 * 4).unwrap();
    let (ptr, count, sink) = (src.ptr(), (len / 16) as u64, out.ptr());
    let mut best = 0.0f64;
    for per_sm in [4, 8, 16, 32] {
        let name = format!("vram read {} MiB, {per_sm} blocks/SM", len >> 20);
        let t = suite.run(&name, 30, || {
            gpu.time(|s| {
                // SAFETY: three arguments of the kernel's types; `out` holds a word for each of the grid's threads.
                unsafe { s.launch(&read, [sms * per_sm, 1, 1], [512, 1, 1], 0, &[arg(&ptr), arg(&count), arg(&sink)]) }.unwrap();
            })
        });
        best = best.max(gbs(len, t));
        suite.rate(gbs(len, t), "GB/s");
    }
    let t = suite.run(&format!("vram copy {} MiB (driver)", len >> 20), 30, || gpu.time(|s| s.copy(&dst, &src).unwrap()));
    suite.rate(gbs(2 * len, t), "GB/s");
    best
}

/// PCIe: pinned copies both ways timed on the GPU, and a pageable upload timed on the CPU. Returns pinned host→device GB/s.
fn pcie(gpu: &Gpu, suite: &mut Suite) -> f64 {
    const LEN: usize = 256 << 20;
    let (mut host, dev) = (gpu.ctx.alloc_host(LEN).unwrap(), gpu.ctx.alloc(LEN).unwrap());
    let t = suite.run("host→device 256 MiB, pinned", 20, || {
        gpu.time(|s| {
            // SAFETY: this copy, like each below, completes when `time` or `sync` returns, before its host memory is touched again.
            unsafe { s.upload(&dev, 0, &host) }.unwrap();
        })
    });
    let h2d = gbs(LEN, t);
    suite.rate(h2d, "GB/s");
    let t = suite.run("device→host 256 MiB, pinned", 20, || {
        gpu.time(|s| {
            // SAFETY: as above.
            unsafe { s.download(&mut host, &dev, 0) }.unwrap();
        })
    });
    suite.rate(gbs(LEN, t), "GB/s");
    let pageable = vec![1u8; LEN];
    let t = suite.run("host→device 256 MiB, pageable", 10, || {
        // SAFETY: as above.
        cpu(|| unsafe { gpu.stream.upload(&dev, 0, &pageable) }.and_then(|()| gpu.stream.sync()).unwrap())
    });
    suite.rate(gbs(LEN, t), "GB/s");
    h2d
}

/// CPU cost of queueing work: single launches, and one replay of a 100-kernel graph. Returns the replay's CPU time.
fn launches(gpu: &Gpu, suite: &mut Suite) -> Duration {
    let noop = gpu.module.function("noop").unwrap();
    // SAFETY: `noop` takes no arguments and touches no memory.
    let launch = |s: &Stream| unsafe { s.launch(&noop, [1, 1, 1], [32, 1, 1], 0, &[]) }.unwrap();
    let t = suite.run("launch ×1000 (CPU)", 50, || {
        let t = cpu(|| (0..1000).for_each(|_| launch(&gpu.stream)));
        gpu.stream.sync().unwrap();
        t
    });
    suite.rate(t.as_secs_f64() * 1e3, "µs/launch");
    let graph = gpu
        .stream
        .capture(|s| {
            (0..100).for_each(|_| launch(s));
            Ok(())
        })
        .unwrap();
    // SAFETY: the module the graph launches is alive.
    let replay = || unsafe { gpu.stream.replay(&graph) }.unwrap();
    let cpu_time = suite.run("graph of 100 kernels, replay (CPU)", 200, || {
        let t = cpu(replay);
        gpu.stream.sync().unwrap();
        t
    });
    suite.rate(cpu_time.as_secs_f64() * 1e6, "µs");
    // Decode queues the next step while the GPU runs the current one: replays back to back are the cost that counts.
    let queued = suite.run("graph of 100 kernels, 10 replays queued (CPU)", 200, || {
        let t = cpu(|| (0..10).for_each(|_| replay()));
        gpu.stream.sync().unwrap();
        t
    }) / 10;
    suite.rate(queued.as_secs_f64() * 1e6, "µs/replay");
    let t = suite.run("graph of 100 kernels (GPU)", 200, || gpu.time(|_| replay()));
    suite.rate(t.as_secs_f64() * 1e4, "µs/kernel");
    queued
}

/// Whether this is WSL2, where every GPU submission goes through Windows' paravirtualized driver.
fn wsl() -> bool {
    std::fs::read_to_string("/proc/sys/kernel/osrelease").is_ok_and(|r| r.to_ascii_lowercase().contains("microsoft"))
}

/// The step-1 exit gate: CPU cost of a queued graph replay, gated on native Linux and reported under WSL2,
/// whose paravirtualized driver adds submission cost. The pinned upload rate is reported as the machine's PCIe
/// ceiling, not gated: one driver call moves the data, so the platform sets it; step 2's loader is gated against it.
fn gates(info: &DeviceInfo, h2d: f64, graph: Duration) -> bool {
    let graph_ok = graph <= Duration::from_micros(10);
    let graph_us = graph.as_secs_f64() * 1e6;
    if wsl() {
        println!("gate: a queued graph replay costs {graph_us:.1} µs of CPU; not gated under WSL2, whose driver path adds cost");
    } else {
        println!("gate: a queued graph replay costs {graph_us:.1} µs of CPU (≤ 10 µs): {}", if graph_ok { "PASS" } else { "FAIL" });
    }
    match info.pcie() {
        Some(link) => println!(
            "ceiling: pinned host→device {h2d:.1} GB/s, {:.0}% of PCIe {} GT/s ×{} ({:.1} GB/s); step 2's loader is measured against it",
            h2d * 1e9 / link.bandwidth() * 100.0,
            link.max_speed,
            link.width,
            link.bandwidth() / 1e9
        ),
        None => println!("ceiling: pinned host→device {h2d:.1} GB/s; the PCIe link is not visible here; step 2's loader is measured against it"),
    }
    graph_ok || wsl()
}
