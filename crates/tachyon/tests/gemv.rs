//! The decode kernels on a real GPU, against the CPU: `cargo test --features gpu`.
#![cfg(feature = "gpu")]
// The names follow the math of the kernel (q, s, d, n, k), as in its documentation.
#![allow(clippy::many_single_char_names)]

mod common;

use common::{Gpu, values};
use tachyon::cuda::{DevBuf, arg};
use tachyon::ptx::{GEMV_Q4, GEMV_Q4_NARROW, GEMV_ROWS, QUANT_Q8};
use tachyon::quant::{Q4, f16_value, q4, q8};

/// Every column count of a Gemma 4 E4B decode projection (only `cols` changes the kernel's path; rows only add
/// warps), and edge shapes: rows not a multiple of the block, one group, and fewer chunks than lanes. Rows of at most
/// 256 weights, in a multiple of 4, run on [`GEMV_Q4_NARROW`].
const SHAPES: [(usize, usize); 9] = [(512, 2560), (512, 2048), (512, 4096), (256, 10240), (37, 2560), (5, 64), (3, 192), (2560, 256), (36, 64)];

/// Quantizes `x` on the GPU; returns the device buffers and their contents.
fn gpu_q8(gpu: &Gpu, x: &[f32]) -> (DevBuf, DevBuf, Vec<u8>, Vec<f32>) {
    let (xd, q, s) = (gpu.upload(x), gpu.zeros(x.len()), gpu.zeros(x.len() / 4));
    let (px, pq, ps, len) = (xd.ptr(), q.ptr(), s.ptr(), x.len() as u32);
    // SAFETY: four arguments of the kernel's types; `q` and `s` hold `len` bytes and `len / 32` pairs.
    unsafe {
        gpu.stream.launch(&gpu.module.function(QUANT_Q8).unwrap(), [len.div_ceil(256), 1, 1], [256, 1, 1], 0, &[arg(&px), arg(&pq), arg(&ps), arg(&len)])
    }
    .unwrap();
    let (qv, sv) = (gpu.download(&q), gpu.floats(&s));
    (q, s, qv, sv)
}

/// `y = W·x` on the GPU.
fn gpu_gemv(gpu: &Gpu, w: &Q4, x: &[f32], rows: usize) -> Vec<f32> {
    let (q, s, _, _) = gpu_q8(gpu, x);
    let (wd, sd, y) = (gpu.upload(&w.packed), gpu.upload(&w.scales), gpu.zeros(rows * 4));
    let (pw, pws, pq, ps, py, n, k) = (wd.ptr(), sd.ptr(), q.ptr(), s.ptr(), y.ptr(), rows as u32, x.len() as u32);
    let args = [arg(&pw), arg(&pws), arg(&pq), arg(&ps), arg(&py), arg(&n), arg(&k)];
    let (f, grid) = if k <= 256 && n % 4 == 0 { (GEMV_Q4_NARROW, n.div_ceil(16)) } else { (GEMV_Q4, n.div_ceil(GEMV_ROWS)) };
    // SAFETY: seven arguments of the kernel's types; every buffer has the size the layout gives for `n × k`.
    unsafe { gpu.stream.launch(&gpu.module.function(f).unwrap(), [grid, 1, 1], [32 * GEMV_ROWS, 1, 1], 0, &args) }.unwrap();
    gpu.floats(&y)
}

/// What the kernel computes, in `f64`: per row, Σ over chunks of `d · (s · Σ n·q − 8 · s·Σq)`; and the bound on its
/// rounding error, a small multiple of the terms' magnitudes.
fn reference(w: &Q4, x: &[f32], rows: usize) -> Vec<(f64, f64)> {
    let (q, s) = q8(x);
    let cols = x.len();
    let nibble = |r: usize, j: usize| {
        let (g, k) = ((r * cols + j) / 64, j % 64);
        f64::from(w.packed[g * 32 + k / 8 * 4 + k % 4] >> (k % 8 / 4 * 4) & 15)
    };
    (0..rows)
        .map(|r| {
            (0..cols / 32).fold((0.0, 0.0), |(sum, bound), c| {
                let dot: f64 = (0..32).map(|j| nibble(r, c * 32 + j) * f64::from(q[c * 32 + j])).sum();
                let d = f64::from(f16_value(w.scales[(r * cols + c * 32) / 64]));
                let term = d * (f64::from(s[c][0]) * dot - 8.0 * f64::from(s[c][1]));
                (sum + term, bound + term.abs() + d * f64::from(s[c][0]) * dot.abs())
            })
        })
        .collect()
}

#[test]
fn quant_q8_matches_the_cpu_exactly() {
    let gpu = Gpu::new();
    let mut x = values(4096, 7);
    x[64..96].fill(0.0);
    x[100] = 1e-30;
    let (_, _, q, s) = gpu_q8(&gpu, &x);
    let (cq, cs) = q8(&x);
    assert!(q.iter().zip(&cq).all(|(&g, &c)| g as i8 == c), "Q8 values differ");
    assert_eq!(s, cs.concat(), "Q8 scales differ");
}

#[test]
fn gemv_matches_the_reference_on_every_shape() {
    let gpu = Gpu::new();
    for (i, &(rows, cols)) in SHAPES.iter().enumerate() {
        let (wf, x) = (values(rows * cols, 11 + i as u64), values(cols, 99 + i as u64));
        let w = q4(&wf, cols);
        let y = gpu_gemv(&gpu, &w, &x, rows);
        for (r, (&got, (want, bound))) in y.iter().zip(reference(&w, &x, rows)).enumerate() {
            assert!((f64::from(got) - want).abs() <= 1e-5 * bound + 1e-6, "{rows}×{cols} row {r}: {got} vs {want}");
        }
    }
}

#[test]
fn gemv_stays_close_to_the_unquantized_product() {
    let gpu = Gpu::new();
    let (rows, cols) = (512, 2560);
    let (wf, x) = (values(rows * cols, 3), values(cols, 4));
    let y = gpu_gemv(&gpu, &q4(&wf, cols), &x, rows);
    let exact: Vec<f64> = wf.chunks_exact(cols).map(|row| row.iter().zip(&x).map(|(a, b)| f64::from(a * b)).sum()).collect();
    let err: f64 = y.iter().zip(&exact).map(|(&g, e)| (f64::from(g) - e).powi(2)).sum::<f64>().sqrt();
    let norm: f64 = exact.iter().map(|e| e * e).sum::<f64>().sqrt();
    assert!(err / norm < 0.15, "relative error {:.3}: the layout is wrong, not just coarse", err / norm);
}