//! The decode kernels on a real GPU, against the CPU: `cargo test --features gpu`.
#![cfg(feature = "gpu")]
// The names follow the math of the kernel (q, s, d, n, k), as in its documentation.
#![allow(clippy::many_single_char_names)]

use tachyon::cuda::{Context, DevBuf, Module, Stream, arg};
use tachyon::ptx::{GEMV_Q4, GEMV_ROWS, QUANT_Q8, module};
use tachyon::quant::{Q4, f16_value, q4, q8};

/// Every Gemma 4 E4B decode projection (fused where the engine fuses), with rows cut where only `cols` matters,
/// and edge shapes: one group, rows not a multiple of the block, and fewer chunks than lanes.
const SHAPES: [(usize, usize); 9] = [(3072, 2560), (6144, 2560), (2560, 2048), (2560, 4096), (2048, 10240), (4096, 2560), (37, 2560), (5, 64), (3, 192)];

/// Uniform values in [-1, 1) from a fixed seed.
fn values(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

fn bytes<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: plain numbers, viewed as their bytes for the length of the borrow.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), size_of_val(v)) }
}

fn upload(ctx: &Context, stream: &Stream, data: &[u8]) -> DevBuf {
    let buf = ctx.alloc(data.len()).unwrap();
    // SAFETY: `data` outlives the copy, which `sync` completes.
    unsafe { stream.upload(&buf, 0, data) }.unwrap();
    stream.sync().unwrap();
    buf
}

fn download(stream: &Stream, buf: &DevBuf) -> Vec<u8> {
    let mut out = vec![0; buf.len()];
    // SAFETY: `out` outlives the copy, which `sync` completes.
    unsafe { stream.download(&mut out, buf, 0) }.unwrap();
    stream.sync().unwrap();
    out
}

fn floats(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(4).map(|c| f32::from_ne_bytes(c.try_into().unwrap())).collect()
}

/// Quantizes `x` on the GPU; returns the device buffers and their contents.
fn gpu_q8(ctx: &Context, stream: &Stream, module: &Module, x: &[f32]) -> (DevBuf, DevBuf, Vec<u8>, Vec<f32>) {
    let (xd, q, s) = (upload(ctx, stream, bytes(x)), ctx.alloc(x.len()).unwrap(), ctx.alloc(x.len() / 4).unwrap());
    let (px, pq, ps, len) = (xd.ptr(), q.ptr(), s.ptr(), x.len() as u32);
    // SAFETY: four arguments of the kernel's types; `q` and `s` hold `len` bytes and `len / 32` pairs.
    unsafe { stream.launch(&module.function(QUANT_Q8).unwrap(), [len.div_ceil(256), 1, 1], [256, 1, 1], 0, &[arg(&px), arg(&pq), arg(&ps), arg(&len)]) }
        .unwrap();
    let (qv, sv) = (download(stream, &q), floats(&download(stream, &s)));
    (q, s, qv, sv)
}

/// `y = W·x` on the GPU.
fn gpu_gemv(ctx: &Context, stream: &Stream, module: &Module, w: &Q4, x: &[f32], rows: usize) -> Vec<f32> {
    let (q, s, _, _) = gpu_q8(ctx, stream, module, x);
    let (wd, sd, y) = (upload(ctx, stream, &w.packed), upload(ctx, stream, bytes(&w.scales)), ctx.alloc(rows * 4).unwrap());
    let (pw, pws, pq, ps, py, n, k) = (wd.ptr(), sd.ptr(), q.ptr(), s.ptr(), y.ptr(), rows as u32, x.len() as u32);
    let args = [arg(&pw), arg(&pws), arg(&pq), arg(&ps), arg(&py), arg(&n), arg(&k)];
    // SAFETY: seven arguments of the kernel's types; every buffer has the size the layout gives for `n × k`.
    unsafe { stream.launch(&module.function(GEMV_Q4).unwrap(), [n.div_ceil(GEMV_ROWS), 1, 1], [32 * GEMV_ROWS, 1, 1], 0, &args) }.unwrap();
    floats(&download(stream, &y))
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
    let ctx = Context::new(0).unwrap();
    let (stream, module) = (ctx.stream().unwrap(), ctx.load(&module()).unwrap());
    let mut x = values(4096, 7);
    x[64..96].fill(0.0);
    x[100] = 1e-30;
    let (_, _, q, s) = gpu_q8(&ctx, &stream, &module, &x);
    let (cq, cs) = q8(&x);
    assert!(q.iter().zip(&cq).all(|(&g, &c)| g as i8 == c), "Q8 values differ");
    assert_eq!(s, cs.concat(), "Q8 scales differ");
}

#[test]
fn gemv_matches_the_reference_on_every_shape() {
    let ctx = Context::new(0).unwrap();
    let (stream, module) = (ctx.stream().unwrap(), ctx.load(&module()).unwrap());
    for (i, &(rows, cols)) in SHAPES.iter().enumerate() {
        let (wf, x) = (values(rows * cols, 11 + i as u64), values(cols, 99 + i as u64));
        let w = q4(&wf, cols);
        let y = gpu_gemv(&ctx, &stream, &module, &w, &x, rows);
        for (r, (&got, (want, bound))) in y.iter().zip(reference(&w, &x, rows)).enumerate() {
            assert!((f64::from(got) - want).abs() <= 1e-5 * bound + 1e-6, "{rows}×{cols} row {r}: {got} vs {want}");
        }
    }
}

#[test]
fn gemv_stays_close_to_the_unquantized_product() {
    let ctx = Context::new(0).unwrap();
    let (stream, module) = (ctx.stream().unwrap(), ctx.load(&module()).unwrap());
    let (rows, cols) = (512, 2560);
    let (wf, x) = (values(rows * cols, 3), values(cols, 4));
    let y = gpu_gemv(&ctx, &stream, &module, &q4(&wf, cols), &x, rows);
    let exact: Vec<f64> = wf.chunks_exact(cols).map(|row| row.iter().zip(&x).map(|(a, b)| f64::from(a * b)).sum()).collect();
    let err: f64 = y.iter().zip(&exact).map(|(&g, e)| (f64::from(g) - e).powi(2)).sum::<f64>().sqrt();
    let norm: f64 = exact.iter().map(|e| e * e).sum::<f64>().sqrt();
    assert!(err / norm < 0.15, "relative error {:.3}: the layout is wrong, not just coarse", err / norm);
}
