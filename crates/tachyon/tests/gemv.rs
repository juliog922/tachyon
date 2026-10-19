//! The decode kernels on a real GPU, against the CPU: `cargo test --features gpu`.
#![cfg(feature = "gpu")]
// The names follow the math of the kernel (q, s, d, n, k), as in its documentation.
#![allow(clippy::many_single_char_names)]

mod common;

use common::{Gpu, product, values};
use tachyon::cuda::{DevBuf, arg};
use tachyon::ptx::{GEGLU_Q8, GEMV_Q4, GEMV_Q4_GEGLU, GEMV_Q4_NARROW, GEMV_Q4_NARROW_NORM, GEMV_Q4_NORM, GEMV_ROWS, NORM_Q8, NORM_THREADS, QUANT_Q8};
use tachyon::quant::{Q4, q4, q8};

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
        for (r, (&got, (want, bound))) in y.iter().zip(product(&w, &x, rows)).enumerate() {
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

/// Launches `kernel` on `grid` blocks of 128 threads with `args` (pointers then the sizes, as `A` holds them).
fn launch(gpu: &Gpu, kernel: &str, grid: u32, args: &[A]) {
    let ptrs: Vec<_> = args
        .iter()
        .map(|a| match a {
            A::P(v) => arg(v),
            A::U(v) => arg(v),
            A::F(v) => arg(v),
        })
        .collect();
    let threads = if kernel == NORM_Q8 { NORM_THREADS } else { 128 };
    // SAFETY: the callers pass each kernel's arguments in order, with buffers of the sizes its layout gives.
    unsafe { gpu.stream.launch(&gpu.module.function(kernel).unwrap(), [grid, 1, 1], [threads, 1, 1], 0, &ptrs) }.unwrap();
}

/// A kernel argument.
#[derive(Clone, Copy)]
enum A {
    P(u64),
    U(u32),
    F(f32),
}

/// The product of `rows × cols` followed by the residual step, fused ([`GEMV_Q4_NORM`], or [`GEMV_Q4_NARROW_NORM`]
/// for rows of 256) and as two kernels ([`GEMV_Q4`] then [`NORM_Q8`]): the same new `h` within float rounding, and Q8
/// within a quantum, with and without `w2`; run twice, as the counter must be back at zero.
#[test]
fn the_product_then_the_norm_matches_the_two_kernels() {
    let gpu = Gpu::new();
    for (i, &(rows, cols, scaled)) in [(2560usize, 2048usize, true), (2560, 256, false), (1536, 2560, true)].iter().enumerate() {
        let seed = 40 + 10 * i as u64;
        let w = q4(&values(rows * cols, seed), cols);
        let (q, s, _, _) = gpu_q8(&gpu, &values(cols, seed + 1));
        let (wd, sd, h, w1, w2) = (
            gpu.upload(&w.packed),
            gpu.upload(&w.scales),
            gpu.upload(&values(rows, seed + 2)),
            gpu.upload(&values(rows, seed + 3)),
            gpu.upload(&values(rows, seed + 4)),
        );
        let (count, pw2) = (gpu.zeros(4), if scaled { w2.ptr() } else { 0 });
        let [y, ho, qo, so] = [rows * 4, rows * 4, rows, rows / 4].map(|n| gpu.zeros(n));
        let product = [A::P(wd.ptr()), A::P(sd.ptr()), A::P(q.ptr()), A::P(s.ptr()), A::P(y.ptr()), A::U(rows as u32), A::U(cols as u32)];
        let (narrow, eps, scale) = (cols <= 256, 1e-6f32, 0.7f32);
        launch(&gpu, if narrow { GEMV_Q4_NARROW } else { GEMV_Q4 }, (rows / if narrow { 16 } else { 4 }) as u32, &product);
        let norm = [h.ptr(), ho.ptr(), y.ptr(), w1.ptr(), pw2, qo.ptr(), so.ptr()].map(A::P).into_iter().chain([A::U(rows as u32), A::F(eps), A::F(scale)]);
        launch(&gpu, NORM_Q8, rows as u32 / NORM_THREADS, &norm.collect::<Vec<_>>());
        let want = (gpu.floats(&ho), gpu.download(&qo), gpu.floats(&so));
        for _ in 0..2 {
            let tail = [count.ptr(), h.ptr(), ho.ptr(), w1.ptr(), pw2, qo.ptr(), so.ptr()].map(A::P).into_iter().chain([A::F(eps), A::F(scale)]);
            let fused: Vec<A> = product.into_iter().chain(tail).collect();
            launch(&gpu, if narrow { GEMV_Q4_NARROW_NORM } else { GEMV_Q4_NORM }, (rows / if narrow { 16 } else { 4 }) as u32, &fused);
            let (got, gq, gs) = (gpu.floats(&ho), gpu.download(&qo), gpu.floats(&so));
            for (r, (&g, &e)) in got.iter().zip(&want.0).enumerate() {
                assert!((g - e).abs() <= 1e-5 * (1.0 + e.abs()), "{rows}×{cols}: h row {r}: {g} against {e}");
            }
            for r in 0..rows {
                let value = |q: &[u8], s: &[f32]| f32::from(q[r] as i8) * s[2 * (r / 32)];
                let (g, e) = (value(&gq, &gs), value(&want.1, &want.2));
                assert!((g - e).abs() <= 1.01 * want.2[2 * (r / 32)], "{rows}×{cols}: Q8 row {r}: {g} against {e}");
            }
            assert_eq!(gpu.words(&count), vec![0], "the counter is left at zero");
        }
    }
}

/// The 256-row product followed by the gated GELU ([`GEMV_Q4_GEGLU`]) writes exactly the Q8 of [`GEMV_Q4`] then
/// [`GEGLU_Q8`].
#[test]
fn the_product_then_the_gate_matches_the_two_kernels() {
    let gpu = Gpu::new();
    let (rows, cols) = (256usize, 2560usize);
    let w = q4(&values(rows * cols, 70), cols);
    let (q, s, _, _) = gpu_q8(&gpu, &values(cols, 71));
    let (wd, sd, b, count) = (gpu.upload(&w.packed), gpu.upload(&w.scales), gpu.upload(&values(rows, 72)), gpu.zeros(4));
    let [y, qo, so, fq, fs] = [rows * 4, rows, rows / 4, rows, rows / 4].map(|n| gpu.zeros(n));
    let product = [A::P(wd.ptr()), A::P(sd.ptr()), A::P(q.ptr()), A::P(s.ptr()), A::P(y.ptr()), A::U(rows as u32), A::U(cols as u32)];
    launch(&gpu, GEMV_Q4, rows as u32 / 4, &product);
    launch(&gpu, GEGLU_Q8, 2, &[A::P(y.ptr()), A::P(b.ptr()), A::P(qo.ptr()), A::P(so.ptr()), A::U(rows as u32)]);
    let fused: Vec<A> = product.into_iter().chain([count.ptr(), b.ptr(), fq.ptr(), fs.ptr()].map(A::P)).collect();
    launch(&gpu, GEMV_Q4_GEGLU, rows as u32 / 4, &fused);
    assert_eq!(gpu.download(&fq), gpu.download(&qo), "Q8 values");
    assert_eq!(gpu.floats(&fs), gpu.floats(&so), "Q8 scales");
    assert_eq!(gpu.words(&count), vec![0], "the counter is left at zero");
}
