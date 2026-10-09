//! Gemma 4 E4B decoding on the GPU, end to end: tokens per second against the roofline, the measured VRAM read
//! ceiling divided by the bytes of weights one token reads. Checks the step-6 gate: ≥ 85% of the roofline. The
//! same steps run with each prefetch budget (the bytes of the next matrix every small kernel fetches into L2), the
//! default last; the gate judges the default.
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

/// The median time of a step after `WARM`, decoding greedily from `<bos>`.
fn steps(gpu: &Gpu, suite: &mut Suite, dir: &std::path::Path, prefetch: Option<usize>) -> Duration {
    let mut dec = Decoder::open(&gpu.ctx, dir, &Settings { context: 4096, prefetch }).unwrap();
    let mut token = (0..WARM).fold(2, |t, _| dec.step(t).unwrap());
    let name = prefetch.map_or("decode, default prefetch".into(), |b| format!("decode, prefetch {} KiB", b >> 10));
    suite.run(&name, STEPS, || {
        let start = Instant::now();
        token = dec.step(token).unwrap();
        start.elapsed()
    })
}

fn main() -> ExitCode {
    let dir = std::env::var("TACHYON_MODEL")
        .map_or_else(|_| std::path::PathBuf::from(std::env::var("HOME").unwrap()).join(".local/share/tachyon/gemma-4-e4b"), Into::into);
    let gpu = Gpu::new();
    let mut suite = Suite::new("generate");
    let roof = gpu.ceiling(&mut suite);
    let bytes = Wpk::open(dir.join("weights.wpk")).unwrap().data_len() as f64;
    let ideal = bytes / (roof * 1e9);
    let l2 = gpu.ctx.info().l2 as usize;
    let mut t = Duration::ZERO;
    for budget in [Some(0), Some(1 << 20), Some(l2), None] {
        t = steps(&gpu, &mut suite, &dir, budget);
        suite.rate(1.0 / t.as_secs_f64(), "tok/s");
        println!("{:?}: {t:.2?} per token, {:.1} tok/s, {:.1}% of the roofline", budget, 1.0 / t.as_secs_f64(), 100.0 * ideal / t.as_secs_f64());
    }
    println!("{} · read ceiling {roof:.1} GB/s · {:.2} GB of weights per token · roofline {:.1} tok/s", gpu.ctx.info().name, bytes / 1e9, 1.0 / ideal);
    let passed = ideal / t.as_secs_f64() >= 0.85;
    println!(
        "gate: {:.1} tok/s, {:.1}% of the roofline (≥ 85%): {}",
        1.0 / t.as_secs_f64(),
        100.0 * ideal / t.as_secs_f64(),
        if passed { "PASS" } else { "FAIL" }
    );
    let code = suite.finish();
    if passed { code } else { ExitCode::FAILURE }
}