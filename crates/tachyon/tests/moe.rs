//! The mixture of experts' kernels on a real GPU, against the CPU: routing, the experts' GEMV, their GELU over
//! slots, and prefill's accumulation. `cargo test --features gpu`.
#![cfg(feature = "gpu")]
// The names follow the math of the kernels (q, s, w, k, e).
#![allow(clippy::many_single_char_names, clippy::similar_names)]

mod common;

use common::{Gpu, assert_q8, product, values};
use tachyon::cuda::{DevBuf, arg};
use tachyon::ptx::{GEGLU_Q8S, GEMV_Q4_MOE, GEMV_ROWS, MOE_ADD, QUANT_Q8, ROUTE_TOPK};
use tachyon::quant::{Q4, q4};

/// `x` quantized to Q8 on the GPU.
fn gpu_q8(gpu: &Gpu, x: &[f32]) -> (DevBuf, DevBuf) {
    let (xd, q, s) = (gpu.upload(x), gpu.zeros(x.len()), gpu.zeros(x.len() / 4));
    let (px, pq, ps, len) = (xd.ptr(), q.ptr(), s.ptr(), x.len() as u32);
    // SAFETY: four arguments of the kernel's types; `q` and `s` hold `len` bytes and `len / 32` pairs.
    unsafe {
        gpu.stream.launch(&gpu.module.function(QUANT_Q8).unwrap(), [len.div_ceil(256), 1, 1], [256, 1, 1], 0, &[arg(&px), arg(&pq), arg(&ps), arg(&len)])
    }
    .unwrap();
    (q, s)
}

/// Expert `e`'s rows of a matrix stacking experts of `rows × cols`.
fn expert(w: &Q4, e: usize, rows: usize, cols: usize) -> Q4 {
    let n = rows * cols;
    Q4 { packed: w.packed[e * n / 2..(e + 1) * n / 2].to_vec(), scales: w.scales[e * n / 64..(e + 1) * n / 64].to_vec() }
}

/// [`GEMV_Q4_MOE`] over experts `idx` of five stacked ones: with one input for all (the gate and up), then with an
/// input per slot and each slot's weight (the downs), each slot's rows as [`tachyon::ptx::GEMV_Q4`] gives them.
#[test]
fn gemv_runs_the_chosen_experts() {
    let gpu = Gpu::new();
    let (experts, rows, cols, idx) = (5, 96, 768, [3u32, 0, 4]);
    let k = idx.len();
    let w = q4(&values(experts * rows * cols, 21), cols);
    let (wd, sd, idd) = (gpu.upload(&w.packed), gpu.upload(&w.scales), gpu.upload(&idx));
    let weights = [0.5f32, 1.25, -0.75];
    for (slot, per_slot) in [(0u32, false), (cols as u32, true)] {
        let x = values(if per_slot { k * cols } else { cols }, 22);
        let ((q, s), y, wts) = (gpu_q8(&gpu, &x), gpu.zeros(k * rows * 4), gpu.upload(&weights));
        let (pw, pws, pq, ps, py, n, c) = (wd.ptr(), sd.ptr(), q.ptr(), s.ptr(), y.ptr(), rows as u32, cols as u32);
        let (pidx, pwts) = (idd.ptr(), if per_slot { wts.ptr() } else { 0 });
        let args = [arg(&pw), arg(&pws), arg(&pq), arg(&ps), arg(&py), arg(&n), arg(&c), arg(&pidx), arg(&pwts), arg(&slot)];
        let grid = [n.div_ceil(GEMV_ROWS), k as u32, 1];
        // SAFETY: ten arguments of the kernel's types; the matrix stacks five experts, the input holds `k` slots when
        // `slot` is set, the output `k × rows`.
        unsafe { gpu.stream.launch(&gpu.module.function(GEMV_Q4_MOE).unwrap(), grid, [32 * GEMV_ROWS, 1, 1], 0, &args) }.unwrap();
        let got = gpu.floats(&y);
        for (j, &e) in idx.iter().enumerate() {
            let input = if per_slot { &x[j * cols..(j + 1) * cols] } else { &x[..] };
            let scale = if per_slot { f64::from(weights[j]) } else { 1.0 };
            for (r, (want, bound)) in product(&expert(&w, e as usize, rows, cols), input, rows).into_iter().enumerate() {
                let g = f64::from(got[j * rows + r]);
                assert!((g - scale * want).abs() <= 1e-5 * bound * scale.abs() + 1e-6, "slot {j} (expert {e}) row {r}: {g} vs {}", scale * want);
            }
        }
    }
}

/// [`ROUTE_TOPK`] against the softmax, top-k and renormalization of Hugging Face's router, on 64 experts and on 100
/// (a block with idle threads), and with a tie (the lower number first).
#[test]
fn routing_picks_and_weighs_the_experts() {
    let gpu = Gpu::new();
    for (experts, k) in [(64usize, 4usize), (100, 8), (128, 8)] {
        let tokens = 3;
        let mut logits: Vec<f32> = values(tokens * experts, experts as u64).iter().map(|x| 4.0 * x).collect();
        logits[experts + 7] = 9.0;
        logits[experts + 3] = 9.0;
        let scale: Vec<f32> = values(experts, 5).iter().map(|x| 1.0 + 0.5 * x).collect();
        let (ld, sd, idx, w) = (gpu.upload(&logits), gpu.upload(&scale), gpu.zeros(tokens * k * 4), gpu.zeros(tokens * k * 4));
        let (pl, ps, pi, pw, e, kk) = (ld.ptr(), sd.ptr(), idx.ptr(), w.ptr(), experts as u32, k as u32);
        let args = [arg(&pl), arg(&ps), arg(&pi), arg(&pw), arg(&e), arg(&kk)];
        // SAFETY: six arguments of the kernel's types; the logits hold `tokens × experts`, the outputs `tokens × k`.
        unsafe { gpu.stream.launch(&gpu.module.function(ROUTE_TOPK).unwrap(), [tokens as u32, 1, 1], [128, 1, 1], 0, &args) }.unwrap();
        let (idx, w) = (gpu.words(&idx), gpu.floats(&w));
        for t in 0..tokens {
            let row = &logits[t * experts..(t + 1) * experts];
            let top = row.iter().fold(f32::MIN, |a, &b| a.max(b));
            let p: Vec<f64> = row.iter().map(|&l| f64::from(l - top).exp()).collect();
            let mut order: Vec<usize> = (0..experts).collect();
            order.sort_by(|&a, &b| p[b].total_cmp(&p[a]).then(a.cmp(&b)));
            let total: f64 = order[..k].iter().map(|&j| p[j]).sum();
            for (j, &e) in order[..k].iter().enumerate() {
                assert_eq!(idx[t * k + j] as usize, e, "{experts} experts, token {t}: pick {j}");
                let want = p[e] / total * f64::from(scale[e]);
                assert!((f64::from(w[t * k + j]) - want).abs() <= 1e-5 * want.abs(), "token {t}, expert {e}: {} vs {want}", w[t * k + j]);
            }
        }
        assert_eq!(&idx[k..k + 2], &[3, 7], "a tie goes to the lower number first");
    }
}

/// [`GEGLU_Q8S`] over three experts' gate and up (rows of 2·192), stored as slots of 256 (the down's padded width),
/// the padding left alone.
#[test]
fn geglu_gates_each_slot() {
    let gpu = Gpu::new();
    let (k, width, so) = (3, 192, 256);
    let gate_up: Vec<f32> = values(k * 2 * width, 31).iter().map(|x| 6.0 * x).collect();
    let (a, q, s) = (gpu.upload(&gate_up), gpu.zeros(k * so), gpu.zeros(k * so / 4));
    let (pa, pb, pq, ps) = (a.ptr(), a.ptr() + 4 * width as u64, q.ptr(), s.ptr());
    let (len, w, sa, sb, pso) = ((k * width) as u32, width as u32, 2 * width as u32, 2 * width as u32, so as u32);
    let args = [arg(&pa), arg(&pb), arg(&pq), arg(&ps), arg(&len), arg(&w), arg(&sa), arg(&sb), arg(&pso)];
    // SAFETY: nine arguments of the kernel's types; the input holds `k` rows of `2·width`, the outputs `k` slots.
    unsafe { gpu.stream.launch(&gpu.module.function(GEGLU_Q8S).unwrap(), [len.div_ceil(256), 1, 1], [256, 1, 1], 0, &args) }.unwrap();
    let (q, s) = (gpu.download(&q), gpu.floats(&s));
    for j in 0..k {
        let gelu = |x: f64| 0.5 * x * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044_715 * x * x * x)).tanh());
        let want: Vec<f64> = (0..width).map(|i| gelu(f64::from(gate_up[j * 2 * width + i])) * f64::from(gate_up[j * 2 * width + width + i])).collect();
        assert_q8(&q[j * so..j * so + width], &s[j * so / 16..(j * so + width) / 16], &want, 1e-5);
        assert!(q[j * so + width..(j + 1) * so].iter().all(|&b| b == 0), "slot {j}: the padding is not written");
    }
}

/// [`MOE_ADD`]: the first expert starts every token's sum (zero where the token did not choose it); a later one adds
/// to the tokens that chose it only, with their weights.
#[test]
fn prefill_keeps_each_tokens_experts() {
    let gpu = Gpu::new();
    let (tokens, len, k) = (3, 512, 2);
    let (idx, w) = ([1u32, 5, 5, 2, 0, 3], [0.25f32, 0.75, 0.5, 0.5, 0.9, 0.1]);
    let (xa, xb) = (values(tokens * len, 41), values(tokens * len, 42));
    let (id, wd, y) = (gpu.upload(&idx), gpu.upload(&w), gpu.upload(&values(tokens * len, 43)));
    for (e, x, fresh) in [(5u32, &xa, 1u32), (2, &xb, 0)] {
        let xd = gpu.upload(x);
        let (py, px, pi, pw, kk, n) = (y.ptr(), xd.ptr(), id.ptr(), wd.ptr(), k as u32, len as u32);
        let args = [arg(&py), arg(&px), arg(&pi), arg(&pw), arg(&e), arg(&kk), arg(&n), arg(&fresh)];
        // SAFETY: eight arguments of the kernel's types; `y` and `x` hold `tokens × len`, the choices `tokens × k`.
        unsafe { gpu.stream.launch(&gpu.module.function(MOE_ADD).unwrap(), [n.div_ceil(256), tokens as u32, 1], [256, 1, 1], 0, &args) }.unwrap();
    }
    let got = gpu.floats(&y);
    for t in 0..tokens {
        for i in (0..len).step_by(37) {
            let at = t * len + i;
            let want = [0.75 * xa[at], 0.5 * xa[at] + 0.5 * xb[at], 0.0][t];
            assert!((got[at] - want).abs() <= 1e-6, "token {t} value {i}: {} vs {want}", got[at]);
        }
    }
}
