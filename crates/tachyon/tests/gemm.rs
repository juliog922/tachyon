//! The prefill matrix product on a real GPU, against the CPU: `cargo test --features gpu --test gemm`.
#![cfg(feature = "gpu")]
// The names follow the math of the kernel (w, x, q, s, y), as in its documentation.
#![allow(clippy::many_single_char_names)]

mod common;

use common::{Gpu, values};
use tachyon::cuda::arg;
use tachyon::ptx::{GEMM_Q4, GEMM_ROWS, GEMM_TOKENS, QUANT_Q8P, SUM_SLICES};
use tachyon::quant::{Q4, f16_value, q4, q8p};

/// A product's rows, columns and tokens.
type Shape = (usize, usize, usize);

/// One tile, part of a tile, several tiles of each kind, and the widest E4B input.
const SHAPES: [Shape; 5] = [(64, 128, 1), (128, 2560, 200), (256, 256, 130), (64, 10240, 37), (192, 384, 128)];

/// `x` quantized to Q8P on the GPU, in buffers of whole token tiles of `cols`: the device buffers and the values
/// and scales.
fn gpu_q8p(gpu: &Gpu, x: &[f32], cols: usize) -> (tachyon::cuda::DevBuf, tachyon::cuda::DevBuf, Vec<u8>, Vec<f32>) {
    let padded = (x.len() / cols).div_ceil(GEMM_TOKENS as usize) * GEMM_TOKENS as usize * cols;
    let (xd, q, s) = (gpu.upload(x), gpu.zeros(padded), gpu.zeros(padded / 8));
    let (px, pq, ps, len) = (xd.ptr(), q.ptr(), s.ptr(), x.len() as u32);
    // SAFETY: four arguments of the kernel's types; `q` and `s` hold `len` bytes and `len / 64` pairs.
    unsafe {
        gpu.stream.launch(&gpu.module.function(QUANT_Q8P).unwrap(), [len.div_ceil(256), 1, 1], [256, 1, 1], 0, &[arg(&px), arg(&pq), arg(&ps), arg(&len)])
    }
    .unwrap();
    let (qv, sv) = (gpu.download(&q), gpu.floats(&s));
    (q, s, qv, sv)
}

/// `Y = X·Wᵀ` for the Q8P tokens in `q, s` on the GPU, the columns in `slices` added up by [`SUM_SLICES`].
fn gpu_gemm(gpu: &Gpu, w: &Q4, q8p: [&tachyon::cuda::DevBuf; 2], shape: Shape, slices: u32) -> Vec<f32> {
    let ((rows, cols, tokens), [q, s]) = (shape, q8p);
    let (wd, wsd, y) = (gpu.upload(&w.packed), gpu.upload(&w.scales), gpu.zeros(slices as usize * tokens * rows * 4));
    let (pw, pws, pq, ps, py) = (wd.ptr(), wsd.ptr(), q.ptr(), s.ptr(), y.ptr());
    let (r, c, t) = (rows as u32, cols as u32, tokens as u32);
    let args = [arg(&pw), arg(&pws), arg(&pq), arg(&ps), arg(&py), arg(&r), arg(&c), arg(&t)];
    // SAFETY: eight arguments of the kernel's types; every buffer has the size its layout gives, the Q8P one for
    // whole tiles of tokens.
    unsafe { gpu.stream.launch(&gpu.module.function(GEMM_Q4).unwrap(), [t.div_ceil(GEMM_TOKENS), r / GEMM_ROWS, slices], [128, 1, 1], 0, &args) }.unwrap();
    let len = t * r;
    // SAFETY: three arguments of the kernel's types; `y` holds `slices` of `len` floats.
    unsafe { gpu.stream.launch(&gpu.module.function(SUM_SLICES).unwrap(), [len.div_ceil(1024), 1, 1], [256, 1, 1], 0, &[arg(&py), arg(&len), arg(&slices)]) }
        .unwrap();
    gpu.floats(&y)[..tokens * rows].to_vec()
}

/// What the kernel computes for one token, in `f64`, from its Q8P values `q` and block scales `s`: per row, Σ over
/// groups of `d · s · Σ (n − 8)·q`; and the error allowed: a hundred-thousandth of the terms' magnitudes plus two
/// units of each group's integer sum (`d · s`), what the rounding of `z` costs (`ptx::gemm`).
fn reference(w: &Q4, q: &[u8], s: &[f32], rows: usize) -> Vec<(f64, f64)> {
    let cols = q.len();
    let nibble = |r: usize, j: usize| {
        let (g, k) = ((r * cols + j) / 64, j % 64);
        f64::from(w.packed[g * 32 + k / 8 * 4 + k % 4] >> (k % 8 / 4 * 4) & 15) - 8.0
    };
    (0..rows)
        .map(|r| {
            (0..cols / 64).fold((0.0, 0.0), |(sum, bound), g| {
                let dot: f64 = (0..64).map(|j| nibble(r, g * 64 + j) * f64::from(q[g * 64 + j] as i8)).sum();
                let unit = f64::from(f16_value(w.scales[(r * cols) / 64 + g])) * f64::from(s[2 * g]);
                (sum + unit * dot, bound + 1e-5 * (unit * dot).abs() + 2.0 * unit)
            })
        })
        .collect()
}

#[test]
fn quant_q8p_matches_the_cpu_exactly() {
    let gpu = Gpu::new();
    let mut x = values(4096, 7);
    x[128..192].fill(0.0);
    x[300] = 1e-30;
    let (_, _, q, s) = gpu_q8p(&gpu, &x, 4096);
    let (cq, cs) = q8p(&x);
    assert!(q.iter().zip(&cq).all(|(&g, &c)| g as i8 == c), "Q8P values differ");
    assert_eq!(&s[..cs.len() * 2], cs.concat(), "Q8P scales differ");
}

/// Within the bound of [`reference`], far below the Q8P rounding of the activations themselves: whole, and with
/// the columns in three slices (as many as there are, if fewer: uneven shares).
#[test]
fn gemm_matches_the_reference_on_every_shape() {
    let gpu = Gpu::new();
    for (i, &(rows, cols, tokens), split) in SHAPES.iter().enumerate().flat_map(|(i, s)| [1, 3].map(|k| (i, s, k))) {
        let (wf, x) = (values(rows * cols, 21 + i as u64), values(tokens * cols, 77 + i as u64));
        let w = q4(&wf, cols);
        let (qd, sd, q, s) = gpu_q8p(&gpu, &x, cols);
        let slices = split.min(cols as u32 / 128);
        let y = gpu_gemm(&gpu, &w, [&qd, &sd], (rows, cols, tokens), slices);
        for (n, got) in y.chunks_exact(rows).enumerate() {
            let want = reference(&w, &q[n * cols..(n + 1) * cols], &s[n * cols / 32..(n + 1) * cols / 32], rows);
            for (r, (&g, (want, bound))) in got.iter().zip(want).enumerate() {
                assert!((f64::from(g) - want).abs() <= bound, "{rows}×{cols}×{tokens} in {slices}: token {n} row {r}: {g} vs {want}");
            }
        }
    }
}
