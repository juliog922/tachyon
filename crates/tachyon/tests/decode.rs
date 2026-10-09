//! The decode kernels around the GEMV on a real GPU, against references on the CPU: `cargo test --features gpu`.
#![cfg(feature = "gpu")]
// The names follow the math (q, k, v, d, n, h), and buffers sit beside their pointers (pos, ppos).
#![allow(clippy::many_single_char_names, clippy::similar_names)]

mod common;

use common::{Gpu, NONE, assert_q8, values};
use tachyon::cuda::{DevBuf, arg};
use tachyon::ptx::{ATTEND_256, ATTEND_512, ATTEND_WARPS, EMBED_Q4, GEGLU_Q8, NORM_Q8, NORM_THREADS, QUANT_Q8, rope};
use tachyon::quant::{f16_bits, f16_value, q4, q8};

/// A Q8 vector as read back: values and `[s, s·Σq]` per block.
type Q8 = (Vec<u8>, Vec<f32>);

/// Q8 output buffers for `len` values.
fn q8_out(gpu: &Gpu, len: usize) -> (DevBuf, DevBuf) {
    (gpu.zeros(len), gpu.zeros(len / 4))
}

/// The contents of Q8 output buffers.
fn q8_read(gpu: &Gpu, (q, s): &(DevBuf, DevBuf)) -> Q8 {
    (gpu.download(q), gpu.floats(s))
}

fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044_715 * x * x * x)).tanh())
}

#[test]
fn geglu_gates_then_quantizes() {
    let gpu = Gpu::new();
    let n = 10_240;
    let (a, b): (Vec<f32>, Vec<f32>) = (values(n, 1).iter().map(|x| 6.0 * x).collect(), values(n, 2));
    let (ad, bd, out) = (gpu.upload(&a), gpu.upload(&b), q8_out(&gpu, n));
    let (pa, pb, pq, ps, len) = (ad.ptr(), bd.ptr(), out.0.ptr(), out.1.ptr(), n as u32);
    // SAFETY: seven arguments of the kernel's types (no prefetch); the outputs hold `n` values.
    unsafe {
        gpu.stream.launch(
            &gpu.module.function(GEGLU_Q8).unwrap(),
            [len.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
            &[arg(&pa), arg(&pb), arg(&pq), arg(&ps), arg(&len), arg(&NONE), arg(&NONE)],
        )
    }
    .unwrap();
    let want: Vec<f64> = a.iter().zip(&b).map(|(&x, &y)| gelu(f64::from(x)) * f64::from(y)).collect();
    let (q, s) = q8_read(&gpu, &out);
    assert_q8(&q, &s, &want, 1e-5);
}

/// One [`NORM_Q8`] launch: `chunks` blocks of `len` values.
struct Norm<'a> {
    h: Vec<f32>,
    y: Option<Vec<f32>>,
    w1: &'a [f32],
    w2: Option<&'a [f32]>,
    quantize: bool,
    eps: f32,
    scale: f32,
    len: usize,
}

impl Norm<'_> {
    /// The new `h` and, when quantizing, the Q8 output, from the GPU.
    fn gpu(&self, gpu: &Gpu) -> (Vec<f32>, Option<Q8>) {
        let (h, ho) = (gpu.upload(&self.h), gpu.upload(&self.h));
        let (y, w1, w2) = (self.y.as_deref().map(|y| gpu.upload(y)), gpu.upload(self.w1), self.w2.map(|w| gpu.upload(w)));
        let out = self.quantize.then(|| q8_out(gpu, self.h.len()));
        let ptr = |b: Option<&DevBuf>| b.map_or(0, DevBuf::ptr);
        let (ph, pho, py, pw1, pw2) = (h.ptr(), ho.ptr(), ptr(y.as_ref()), w1.ptr(), ptr(w2.as_ref()));
        let (pq, ps) = (ptr(out.as_ref().map(|o| &o.0)), ptr(out.as_ref().map(|o| &o.1)));
        let (len, chunks) = (self.len as u32, (self.h.len() / self.len) as u32);
        let grid = [len / NORM_THREADS, chunks, 1];
        let args =
            [arg(&ph), arg(&pho), arg(&py), arg(&pw1), arg(&pw2), arg(&pq), arg(&ps), arg(&len), arg(&self.eps), arg(&self.scale), arg(&NONE), arg(&NONE)];
        // SAFETY: twelve arguments of the kernel's types (no prefetch); every buffer holds `chunks × len` values or one chunk of weights.
        unsafe { gpu.stream.launch(&gpu.module.function(NORM_Q8).unwrap(), grid, [NORM_THREADS, 1, 1], 0, &args) }.unwrap();
        assert_eq!(gpu.floats(&h), self.h, "the input is left alone");
        (gpu.floats(&ho), out.map(|o| q8_read(gpu, &o)))
    }

    /// The same in `f64`: the new `h`, and the values quantized.
    fn cpu(&self) -> (Vec<f64>, Vec<f64>) {
        let mut h: Vec<f64> = self.h.iter().map(|&v| f64::from(v)).collect();
        let mut x = Vec::new();
        for (c, chunk) in h.chunks_exact_mut(self.len).enumerate() {
            if let Some(y) = &self.y {
                let y = &y[c * self.len..][..self.len];
                let r = rms(y.iter().map(|&v| f64::from(v)), f64::from(self.eps));
                chunk.iter_mut().zip(y).zip(self.w1).for_each(|((v, &y), &w)| *v = (*v + f64::from(y) * r * f64::from(w)) * f64::from(self.scale));
            }
            let r = self.w2.map_or(1.0, |_| rms(chunk.iter().copied(), f64::from(self.eps)));
            x.extend(chunk.iter().enumerate().map(|(i, v)| v * r * self.w2.map_or(1.0, |w| f64::from(w[i]))));
        }
        (h, x)
    }
}

/// `1 / √(mean(x²) + eps)`.
fn rms(x: impl ExactSizeIterator<Item = f64>, eps: f64) -> f64 {
    let n = x.len() as f64;
    1.0 / (x.map(|v| v * v).sum::<f64>() / n + eps).sqrt()
}

fn check_norm(gpu: &Gpu, case: &Norm<'_>) {
    let ((h, out), (want_h, want_x)) = (case.gpu(gpu), case.cpu());
    for (i, (&got, want)) in h.iter().zip(&want_h).enumerate() {
        assert!((f64::from(got) - want).abs() <= 1e-5 * want.abs().max(1.0), "h[{i}]: {got} vs {want}");
    }
    if let Some((q, s)) = out {
        assert_q8(&q, &s, &want_x, 1e-5 * want_x.iter().fold(0f64, |m, v| m.max(v.abs())));
    }
}

#[test]
fn norm_adds_the_residual_and_quantizes_the_next_input() {
    let gpu = Gpu::new();
    let n = 2560;
    let (w1, w2): (Vec<f32>, Vec<f32>) = (values(n, 3).iter().map(|x| 1.0 + x).collect(), values(n, 4).iter().map(|x| 0.5 + x).collect());
    let h: Vec<f32> = values(n, 5).iter().map(|x| 20.0 * x).collect();
    let y = Some(values(n, 6).iter().map(|x| 300.0 * x).collect());
    check_norm(&gpu, &Norm { h: h.clone(), y, w1: &w1, w2: Some(&w2), quantize: true, eps: 1e-6, scale: 0.75, len: n });
    check_norm(&gpu, &Norm { h, y: None, w1: &w1, w2: Some(&w2), quantize: true, eps: 1e-6, scale: 1.0, len: n });
}

#[test]
fn norm_works_per_chunk_without_quantizing() {
    let gpu = Gpu::new();
    let (len, chunks) = (256, 42);
    let w1: Vec<f32> = values(len, 7).iter().map(|x| 1.0 + x).collect();
    let h = values(len * chunks, 8);
    let y = Some(values(len * chunks, 9).iter().map(|x| 1e-3 * x).collect());
    check_norm(&gpu, &Norm { h, y, w1: &w1, w2: None, quantize: false, eps: 1e-6 * 2560.0, scale: std::f32::consts::FRAC_1_SQRT_2, len });
}

#[test]
fn norm_without_inputs_quantizes_exactly_as_quant_q8() {
    let gpu = Gpu::new();
    let n = 2560;
    let h: Vec<f32> = values(n, 10).iter().map(|x| 7.0 * x).collect();
    let (_, out) = Norm { h: h.clone(), y: None, w1: &[0.0; 2560], w2: None, quantize: true, eps: 1e-6, scale: 1.0, len: n }.gpu(&gpu);
    let (q, s) = out.unwrap();
    let (cq, cs) = q8(&h);
    assert!(q.iter().zip(&cq).all(|(&g, &c)| g as i8 == c), "Q8 values differ");
    assert_eq!(s, cs.concat());
    let x = gpu.upload(&h);
    let out = q8_out(&gpu, n);
    let (px, pq, ps, len) = (x.ptr(), out.0.ptr(), out.1.ptr(), n as u32);
    // SAFETY: six arguments of the kernel's types (no prefetch); the outputs hold `n` values.
    unsafe {
        gpu.stream.launch(
            &gpu.module.function(QUANT_Q8).unwrap(),
            [len.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
            &[arg(&px), arg(&pq), arg(&ps), arg(&len), arg(&NONE), arg(&NONE)],
        )
    }
    .unwrap();
    assert_eq!(q8_read(&gpu, &out), (q, s));
}

#[test]
fn embed_decodes_a_row_from_device_or_host_memory() {
    let gpu = Gpu::new();
    let (rows, cols, token) = (7, 2560, 5u32);
    let w = q4(&values(rows * cols, 11), cols);
    let want: Vec<f32> = (0..cols)
        .map(|i| {
            let (g, k) = ((token as usize * cols + i) / 64, i % 64);
            let n = i32::from(w.packed[g * 32 + k / 8 * 4 + k % 4] >> (k % 8 / 4 * 4) & 15) - 8;
            n as f32 * f16_value(w.scales[g]) * 50.5
        })
        .collect();
    let (tok, out) = (gpu.upload(&[token]), gpu.zeros(cols * 4));
    let mut host = gpu.ctx.alloc_host(w.packed.len() + 2 * w.scales.len()).unwrap();
    host[..w.packed.len()].copy_from_slice(&w.packed);
    host[w.packed.len()..].copy_from_slice(common::bytes(&w.scales));
    let hp = host.device_ptr().unwrap();
    let (dp, ds) = (gpu.upload(&w.packed), gpu.upload(&w.scales));
    for (pw, ps) in [(dp.ptr(), ds.ptr()), (hp, hp + w.packed.len() as u64)] {
        let (pt, py, n, scale) = (tok.ptr(), out.ptr(), cols as u32, 50.5f32);
        let args = [arg(&pw), arg(&ps), arg(&pt), arg(&py), arg(&n), arg(&scale)];
        // SAFETY: six arguments of the kernel's types; the table holds `rows × cols` weights, `out` holds `cols`.
        unsafe { gpu.stream.launch(&gpu.module.function(EMBED_Q4).unwrap(), [n.div_ceil(256), 1, 1], [256, 1, 1], 0, &args) }.unwrap();
        assert_eq!(gpu.floats(&out), want);
    }
}

/// Key-value heads in the attention tests; each serves four query heads.
const KV: usize = 2;

/// One attention launch: head size, cache slots, position of the new token, blocks per key-value head, and whether
/// the layer writes its own key and value.
#[derive(Debug, Clone, Copy)]
struct Attend {
    d: usize,
    cap: u32,
    pos: u32,
    splits: u32,
    fresh: bool,
}

/// What an attention launch reads.
struct Inputs {
    qkv: Vec<f32>,
    qn: Vec<f32>,
    kn: Vec<f32>,
    freq: Vec<f32>,
    k: Vec<u16>,
    v: Vec<u16>,
}

impl Attend {
    fn inputs(&self, seed: u64) -> Inputs {
        let rows = 4 * KV + if self.fresh { 2 * KV } else { 0 };
        let weights = |s| values(self.d, s).iter().map(|x| 0.2 + 0.1 * x).collect();
        let freq = if self.d == 256 { rope(10_000.0, 256, 1.0) } else { rope(1e6, 512, 0.25) };
        let cache = |s| values(KV * self.cap as usize * self.d, s).iter().map(|&x| f16_bits(0.3 * x)).collect();
        Inputs { qkv: values(rows * self.d, seed), qn: weights(seed + 1), kn: weights(seed + 2), freq, k: cache(seed + 3), v: cache(seed + 4) }
    }

    /// Q8 output, and the key and value caches after the launch, from the GPU; launched `times` times.
    fn gpu(&self, gpu: &Gpu, x: &Inputs, times: usize) -> (Q8, Vec<u16>, Vec<u16>) {
        let (d, splits) = (self.d, self.splits);
        let bufs = [gpu.upload(&x.qkv), gpu.upload(&x.qn), gpu.upload(&x.kn), gpu.upload(&x.freq), gpu.upload(&x.k), gpu.upload(&x.v)];
        let [pqkv, pqn, pkn, pfreq, pk, pv] = bufs.each_ref().map(DevBuf::ptr);
        let (pos, part, count, out) = (gpu.upload(&[self.pos]), gpu.zeros(KV * splits as usize * (8 + 4 * d) * 4), gpu.zeros(4 * KV), q8_out(gpu, 4 * KV * d));
        let (ppos, ppart, pcount, pq, ps) = (pos.ptr(), part.ptr(), count.ptr(), out.0.ptr(), out.1.ptr());
        let (cap, fresh) = (self.cap, u32::from(self.fresh));
        let args = [
            arg(&pqkv),
            arg(&pqn),
            arg(&pkn),
            arg(&pfreq),
            arg(&pk),
            arg(&pv),
            arg(&ppos),
            arg(&cap),
            arg(&fresh),
            arg(&ppart),
            arg(&pcount),
            arg(&pq),
            arg(&ps),
            arg(&NONE),
            arg(&NONE),
        ];
        let f = gpu.module.function(if d == 256 { ATTEND_256 } else { ATTEND_512 }).unwrap();
        for _ in 0..times {
            // SAFETY: fifteen arguments of the kernel's types (no prefetch), every buffer sized as the kernel's documentation gives.
            unsafe { gpu.stream.launch(&f, [KV as u32, splits, 1], [32 * ATTEND_WARPS, 1, 1], 0, &args) }.unwrap();
        }
        assert_eq!(gpu.words(&count), [0; KV], "the counters are left at zero");
        (q8_read(gpu, &out), gpu.halves(&bufs[4]), gpu.halves(&bufs[5]))
    }

    /// The output in `f64`, and the new token's key and value for each key-value head.
    fn cpu(&self, x: &Inputs) -> (Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let (d, n, cur) = (self.d, (self.pos + 1).min(self.cap) as usize, (self.pos % self.cap) as usize);
        let head = |i: usize| x.qkv[i * d..][..d].iter().map(|&v| f64::from(v)).collect::<Vec<f64>>();
        let (mut out, mut keys, mut vals) = (Vec::new(), Vec::new(), Vec::new());
        for g in 0..KV {
            let (mut k, mut v) = if self.fresh { (head(4 * KV + g), head(5 * KV + g)) } else { (vec![], vec![]) };
            if self.fresh {
                normalize(&mut k, Some(&x.kn));
                rotate(&mut k, &x.freq, self.pos);
                normalize(&mut v, None);
            }
            let cached = |c: &[u16], t: usize| -> Vec<f64> { c[(g * self.cap as usize + t) * d..][..d].iter().map(|&h| f64::from(f16_value(h))).collect() };
            let kv: Vec<(Vec<f64>, Vec<f64>)> =
                (0..n).map(|t| if self.fresh && t == cur { (k.clone(), v.clone()) } else { (cached(&x.k, t), cached(&x.v, t)) }).collect();
            for h in 4 * g..4 * g + 4 {
                let mut q = head(h);
                normalize(&mut q, Some(&x.qn));
                rotate(&mut q, &x.freq, self.pos);
                out.extend(attention(&q, &kv));
            }
            keys.push(k);
            vals.push(v);
        }
        (out, keys, vals)
    }
}

/// `x / √(mean(x²) + 1e-6)`, times `w` when given.
fn normalize(x: &mut [f64], w: Option<&[f32]>) {
    let r = rms(x.iter().copied(), 1e-6);
    x.iter_mut().enumerate().for_each(|(i, v)| *v *= r * w.map_or(1.0, |w| f64::from(w[i])));
}

/// RoPE at `pos`: pairs `(i, i + d/2)` turn by `pos · freq[i]`, the angle computed in `f32` as the model does.
fn rotate(x: &mut [f64], freq: &[f32], pos: u32) {
    let half = x.len() / 2;
    for (i, &f) in freq.iter().enumerate() {
        let (s, c) = f64::from(pos as f32 * f).sin_cos();
        let (a, b) = (x[i], x[i + half]);
        (x[i], x[i + half]) = (a * c - b * s, b * c + a * s);
    }
}

/// `softmax(q·k) · v` over the positions.
fn attention(q: &[f64], kv: &[(Vec<f64>, Vec<f64>)]) -> Vec<f64> {
    let scores: Vec<f64> = kv.iter().map(|(k, _)| q.iter().zip(k).map(|(a, b)| a * b).sum()).collect();
    let m = scores.iter().fold(f64::NEG_INFINITY, |m, &s| m.max(s));
    let p: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
    let z: f64 = p.iter().sum();
    (0..q.len()).map(|i| kv.iter().zip(&p).map(|((_, v), p)| p * v[i]).sum::<f64>() / z).collect()
}

fn check_attend(gpu: &Gpu, case: Attend) {
    let x = case.inputs(u64::from(case.pos) + case.d as u64);
    let ((q, s), k, v) = case.gpu(gpu, &x, 2);
    let (want, keys, vals) = case.cpu(&x);
    assert_q8(&q, &s, &want, 1e-3 * want.iter().fold(0f64, |m, v| m.max(v.abs())));
    let slot = (case.pos % case.cap) as usize;
    for g in 0..KV {
        let at = (g * case.cap as usize + slot) * case.d;
        let (got_k, got_v) = (&k[at..at + case.d], &v[at..at + case.d]);
        let (old_k, old_v) = (&x.k[at..at + case.d], &x.v[at..at + case.d]);
        if !case.fresh {
            assert_eq!((got_k, got_v), (old_k, old_v), "{case:?}: a layer sharing the cache leaves it alone");
            continue;
        }
        for (got, want) in got_k.iter().zip(&keys[g]).chain(got_v.iter().zip(&vals[g])) {
            assert!((f64::from(f16_value(*got)) - want).abs() <= 1e-3 * want.abs() + 1e-4, "{case:?}: cached {} vs {want}", f16_value(*got));
        }
    }
}

#[test]
fn attention_over_a_short_context() {
    let gpu = Gpu::new();
    check_attend(&gpu, Attend { d: 256, cap: 512, pos: 0, splits: 16, fresh: true });
    check_attend(&gpu, Attend { d: 256, cap: 512, pos: 100, splits: 16, fresh: true });
}

#[test]
fn attention_over_a_full_sliding_window() {
    let gpu = Gpu::new();
    check_attend(&gpu, Attend { d: 256, cap: 512, pos: 700, splits: 16, fresh: true });
}

#[test]
fn global_attention_with_its_own_or_a_shared_cache() {
    let gpu = Gpu::new();
    check_attend(&gpu, Attend { d: 512, cap: 2048, pos: 1500, splits: 32, fresh: true });
    check_attend(&gpu, Attend { d: 512, cap: 2048, pos: 1500, splits: 256, fresh: false });
}