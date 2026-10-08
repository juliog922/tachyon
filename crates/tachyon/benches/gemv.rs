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
use harness::gpu::{Gpu, SHAPES, Shape, token_bytes};
use std::process::ExitCode;
use tachyon::cuda::{DevBuf, arg};
use tachyon::ptx::{GEMV_Q4, GEMV_ROWS};

/// Fewest launches per timed graph.
const LAUNCHES: usize = 64;

/// GB/s at which one shape streams its weights and scales.
fn measure(gpu: &Gpu, suite: &mut Suite, s: &Shape) -> f64 {
    let ((packed, scales), len, (rows, cols)) = (s.sizes(), s.bytes(), (s.rows, s.cols));
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
    let t = gpu.replay(suite, &format!("{}: {rows}×{cols} ×{launches}", s.name), &graph);
    let gbs = len as f64 * launches as f64 / t.as_secs_f64() / 1e9;
    suite.rate(gbs, "GB/s");
    gbs
}

/// The step-3 gate: one token's GEMVs, each timed at its measured rate, against the ceiling.
fn gate(roof: f64, rates: &[f64]) -> bool {
    let mut seconds = 0.0;
    for (s, gbs) in SHAPES.iter().zip(rates) {
        seconds += f64::from(s.per_token) * s.bytes() as f64 / (gbs * 1e9);
        println!("{} {}×{} ×{}: {gbs:.1} GB/s, {:.0}% of the ceiling", s.name, s.rows, s.cols, s.per_token, 100.0 * gbs / roof);
    }
    let total = token_bytes();
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
    let gpu = Gpu::new();
    let mut suite = Suite::new("gemv");
    let roof = gpu.ceiling(&mut suite);
    let rates: Vec<f64> = SHAPES.iter().map(|s| measure(&gpu, &mut suite, s)).collect();
    println!("{} · read ceiling {roof:.1} GB/s · {} MiB of weights per shape to defeat the L2 cache", gpu.ctx.info().name, gpu.uncached() >> 20);
    let passed = gate(roof, &rates);
    let code = suite.finish();
    if passed { code } else { ExitCode::FAILURE }
}