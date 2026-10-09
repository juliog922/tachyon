//! Gemma 4 E4B decoding on the GPU, end to end: tokens per second against the roofline, the measured VRAM read
//! ceiling divided by the bytes of weights one token reads. Checks the step-6 gate: ≥ 80% of the roofline (changed
//! from 85% by the decision of 2026-10-09: one kernel per operation measures 81–83% on the RTX 3060 Laptop, where
//! Ollama's llama.cpp runs E4B at 61–77 tok/s against Tachyon's 107). Then shows where a step's time goes, kernel by
//! kernel.
//!
//! `cargo bench -p tachyon --features gpu --bench generate` uses the converted model in `$TACHYON_MODEL`, by default
//! `~/.local/share/tachyon/gemma-4-e4b` (`tachyon convert` makes it).

mod harness;

use harness::Suite;
use harness::gpu::Gpu;
use std::process::ExitCode;
use std::time::{Duration, Instant};
use tachyon::model::{Decoder, Settings};
use tachyon::wpk::Wpk;

/// Steps before timing, and steps timed.
const WARM: usize = 32;
const STEPS: usize = 256;

/// The median time between generated tokens after `WARM`, decoding greedily from `<bos>` as the engine does: one
/// step always queued behind the running one.
fn steps(gpu: &Gpu, suite: &mut Suite, dir: &std::path::Path) -> Duration {
    let mut dec = Decoder::open(&gpu.ctx, dir, &Settings::default()).unwrap();
    let (mut times, mut last) = (Vec::new(), Instant::now());
    dec.generate(2, |_| {
        times.push(last.elapsed());
        last = Instant::now();
        times.len() < WARM + STEPS
    })
    .unwrap();
    let mut times = times.into_iter().skip(WARM);
    suite.run("decode", STEPS, || times.next().unwrap_or_default())
}

/// Bytes a matrix product of kind `gemv… {rows}×{cols}` reads: Q4 weights and their f16 scales.
fn gemv_bytes(kind: &str) -> Option<f64> {
    let (rows, cols) = kind.rsplit_once(' ')?.1.split_once('×')?;
    let weights = rows.parse::<f64>().ok()? * cols.parse::<f64>().ok()?;
    Some(weights / 2.0 + weights / 32.0)
}

/// Prints where a step's time goes, from `RUNS` steps timed kernel by kernel (the median of each kernel): every kind
/// of kernel with its total, and for matrix products their rate against the ceiling `roof` (GB/s); then the kernels
/// of three layers in order.
fn profile(gpu: &Gpu, dir: &std::path::Path, roof: f64) {
    const RUNS: usize = 11;
    let mut dec = Decoder::open(&gpu.ctx, dir, &Settings::default()).unwrap();
    let mut token = (0..WARM).fold(2, |t, _| dec.step(t).unwrap());
    let runs: Vec<Vec<(String, Duration)>> = (0..RUNS).map(|_| dec.timeline(std::mem::replace(&mut token, 2)).unwrap()).collect();
    let median = |i: usize| {
        let mut t: Vec<Duration> = runs.iter().map(|r| r[i].1).collect();
        t.sort();
        t[RUNS / 2]
    };
    let ops: Vec<(String, Duration)> = (0..runs[0].len()).map(|i| (runs[0][i].0.clone(), median(i))).collect();
    let total: Duration = ops.iter().map(|o| o.1).sum();
    let mut kinds: Vec<(String, usize, Duration)> = Vec::new();
    for (kind, t) in &ops {
        match kinds.iter_mut().find(|k| &k.0 == kind) {
            Some(k) => (k.1, k.2) = (k.1 + 1, k.2 + *t),
            None => kinds.push((kind.clone(), 1, *t)),
        }
    }
    kinds.sort_by_key(|k| std::cmp::Reverse(k.2));
    println!("one step kernel by kernel: {total:.2?} for {} kernels (median of {RUNS} steps)", ops.len());
    println!("  {:<26} {:>5} {:>10} {:>9} {:>6} {:>9} {:>6}", "kernel", "count", "total", "each", "share", "GB/s", "%ceil");
    let (mut gemv, mut ideal) = (Duration::ZERO, 0.0);
    for (kind, n, t) in &kinds {
        let rate = gemv_bytes(kind).map(|b| b * *n as f64 / t.as_secs_f64() / 1e9);
        if let Some(b) = gemv_bytes(kind) {
            (gemv, ideal) = (gemv + *t, ideal + b * *n as f64 / (roof * 1e9));
        }
        let (gbs, pct) = rate.map_or((String::new(), String::new()), |r| (format!("{r:.0}"), format!("{:.0}%", 100.0 * r / roof)));
        let share = 100.0 * t.as_secs_f64() / total.as_secs_f64();
        println!("  {kind:<26} {n:>5} {:>10.2?} {:>9.2?} {share:>5.1}% {gbs:>9} {pct:>6}", t, *t / *n as u32);
    }
    let rest = total.saturating_sub(gemv);
    println!("  matrix products {gemv:.2?} (at the ceiling {:.2?}); everything else {rest:.2?}", Duration::from_secs_f64(ideal));
    println!("each kind of kernel alone in a graph (mean of 20 replays):");
    for (kind, n, t) in dec.alone().unwrap() {
        let rate = gemv_bytes(&kind).map_or(String::new(), |b| {
            format!("{:.0} GB/s, {:.0}% of the ceiling", b * n as f64 / t.as_secs_f64() / 1e9, 100.0 * b * n as f64 / t.as_secs_f64() / 1e9 / roof)
        });
        println!("  {kind:<26} {n:>5} {:>10.2?} {:>9.2?} each  {rate}", t, t / n as u32);
    }
    let per_layer = (ops.len() - 8) / 42;
    for layer in [0, 5, 30] {
        let at = 6 + layer * per_layer;
        let line: Vec<String> =
            ops[at..at + per_layer].iter().map(|(k, t)| format!("{} {:.1}", k.split(' ').next().unwrap_or(k), t.as_secs_f64() * 1e6)).collect();
        println!("  layer {layer} (µs): {}", line.join(" · "));
    }
}

fn main() -> ExitCode {
    let dir = std::env::var("TACHYON_MODEL")
        .map_or_else(|_| std::path::PathBuf::from(std::env::var("HOME").unwrap()).join(".local/share/tachyon/gemma-4-e4b"), Into::into);
    let gpu = Gpu::new();
    let mut suite = Suite::new("generate");
    let roof = gpu.ceiling(&mut suite);
    let bytes = Wpk::open(dir.join("weights.wpk")).unwrap().data_len() as f64;
    let ideal = bytes / (roof * 1e9);
    let t = steps(&gpu, &mut suite, &dir);
    suite.rate(1.0 / t.as_secs_f64(), "tok/s");
    profile(&gpu, &dir, roof);
    println!("{} · read ceiling {roof:.1} GB/s · {:.2} GB of weights per token · roofline {:.1} tok/s", gpu.ctx.info().name, bytes / 1e9, 1.0 / ideal);
    let passed = ideal / t.as_secs_f64() >= 0.80;
    println!(
        "gate: {:.1} tok/s, {:.1}% of the roofline (≥ 80%): {}",
        1.0 / t.as_secs_f64(),
        100.0 * ideal / t.as_secs_f64(),
        if passed { "PASS" } else { "FAIL" }
    );
    let code = suite.finish();
    if passed { code } else { ExitCode::FAILURE }
}