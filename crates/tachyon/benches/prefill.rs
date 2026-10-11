//! Prefill on tensor cores, against the measured integer `mma` peak: each Gemma 4 E4B projection as one
//! [`GEMM_Q4`] over prompts of 256, 512 and 2048 tokens, and all of them weighted by how often a token runs each:
//! the share of the peak a prompt's matrix products reach, and the tokens per second they allow.
//!
//! `cargo bench -p tachyon --features gpu --bench prefill`

mod harness;

use harness::Suite;
use harness::gpu::{Gpu, SHAPES, Shape};
use std::process::ExitCode;
use tachyon::cuda::arg;
use tachyon::ptx::{GEMM_Q4, GEMM_ROWS, GEMM_TOKENS};

/// TOPS of one projection over `tokens` tokens.
fn measure(gpu: &Gpu, suite: &mut Suite, s: &Shape, tokens: u32) -> f64 {
    let ((packed, scales), (rows, cols)) = (s.sizes(), (s.rows, s.cols));
    let sizes = [packed, scales, (tokens * cols) as usize, (tokens * cols / 8) as usize, (tokens * rows * 4) as usize];
    let bufs = sizes.map(|len| gpu.filled(len, 0x11));
    let (gemm, ptrs) = (gpu.module.function(GEMM_Q4).unwrap(), bufs.each_ref().map(tachyon::cuda::DevBuf::ptr));
    let t = suite.run(&format!("{}: {rows}×{cols} × {tokens} tokens", s.name), 20, || {
        gpu.time(|st| {
            let args = [arg(&ptrs[0]), arg(&ptrs[1]), arg(&ptrs[2]), arg(&ptrs[3]), arg(&ptrs[4]), arg(&rows), arg(&cols), arg(&tokens)];
            // SAFETY: eight arguments of the kernel's types; every buffer has the size its layout gives.
            unsafe { st.launch(&gemm, [tokens.div_ceil(GEMM_TOKENS), rows / GEMM_ROWS, 1], [128, 1, 1], 0, &args) }.unwrap();
        })
    });
    let tops = 2.0 * f64::from(rows) * f64::from(cols) * f64::from(tokens) / t.as_secs_f64() / 1e12;
    suite.rate(tops, "TOPS");
    tops
}

/// Every projection over `tokens` tokens (printed when `detail`), and their weighted rate.
fn report(gpu: &Gpu, suite: &mut Suite, peak: f64, tokens: u32, detail: bool) {
    let (mut ops, mut seconds) = (0.0, 0.0);
    // The LM head runs on the last token only, as a GEMV.
    for s in SHAPES.iter().filter(|s| s.name != "lm head") {
        let tops = measure(gpu, suite, s, tokens);
        let o = 2.0 * f64::from(s.rows) * f64::from(s.cols) * f64::from(s.per_token);
        (ops, seconds) = (ops + o, seconds + o / (tops * 1e12));
        if detail {
            println!("  {:<24} {:>6}×{:<6} {tops:>6.1} TOPS  {:>3.0}% of the peak", s.name, s.rows, s.cols, 100.0 * tops / peak);
        }
    }
    let rate = ops / seconds / 1e12;
    println!("{tokens:>5} tokens: matrix products at {rate:.1} TOPS, {:.0}% of the peak, {:.0} tok/s", 100.0 * rate / peak, 1.0 / seconds);
}

fn main() -> ExitCode {
    let gpu = Gpu::new();
    let mut suite = Suite::new("prefill");
    let peak = gpu.mma_peak(&mut suite);
    println!("{} · int8 mma peak {peak:.1} TOPS", gpu.ctx.info().name);
    for tokens in [512, 256, 2048] {
        report(&gpu, &mut suite, peak, tokens, tokens == 512);
    }
    suite.finish()
}