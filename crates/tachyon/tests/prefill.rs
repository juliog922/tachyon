//! The prefill kernels on a real GPU: the small steps run over many tokens at once, writing Q8P for
//! `GEMM_Q4`. `cargo test --features gpu --test prefill`
#![cfg(feature = "gpu")]
// The names follow the math of the kernels (h, y, w, q, s), as in their documentation.
#![allow(clippy::many_single_char_names, clippy::float_cmp, clippy::needless_range_loop)]

mod common;

use common::{Gpu, attention, normalize, rotate, values};
use tachyon::cuda::{DevBuf, arg};
use tachyon::ptx::{EMBED_Q4, FLASH_512, GEGLU_Q8P, KV_256, KV_512, NORM_Q8, NORM_Q8P, NORM_THREADS, QUERY_256, QUERY_512, rope};
use tachyon::quant::{MAGIC, f16_bits, f16_value, q4};

/// Q8P buffers for `len` values, and their contents.
fn out(gpu: &Gpu, len: usize) -> (DevBuf, DevBuf) {
    (gpu.zeros(len), gpu.zeros(len / 8))
}

/// Checks a Q8P vector against `want`: every value within half a quantum of its block plus `slack`, and each block's
/// second word `z` as the layout defines it.
fn assert_q8p(q: &[u8], s: &[f32], want: &[f64], slack: f64) {
    for (i, w) in want.iter().enumerate() {
        let scale = f64::from(s[i / 64 * 2]);
        let got = f64::from(q[i] as i8) * scale;
        assert!((got - w).abs() <= 0.5 * scale * 1.001 + slack, "value {i}: {got} vs {w} (scale {scale})");
    }
    for (b, pair) in s.chunks_exact(2).enumerate() {
        let sum: i32 = q[b * 64..b * 64 + 64].iter().map(|&v| i32::from(v as i8)).sum();
        assert_eq!(pair[1], pair[0].mul_add(-MAGIC, sum as f32 * pair[0] * -8.0), "block {b}: z");
    }
}

/// One norm launch over rows of `len`: `kernel` with Q8 or Q8P out, `blocks` values per scale pair. Returns the new
/// `h`, and the values and scales.
fn norm(gpu: &Gpu, kernel: &str, inputs: [&[f32]; 4], len: usize, blocks: usize) -> (Vec<f32>, Vec<u8>, Vec<f32>) {
    let [h, y, w1, w2] = inputs;
    let (hd, ho, yd, w1d, w2d) = (gpu.upload(h), gpu.upload(h), gpu.upload(y), gpu.upload(w1), gpu.upload(w2));
    let (q, s) = (gpu.zeros(h.len()), gpu.zeros(h.len() / blocks * 8));
    let ptrs = [hd.ptr(), ho.ptr(), yd.ptr(), w1d.ptr(), w2d.ptr(), q.ptr(), s.ptr()];
    let (n, eps, scale) = (len as u32, 1e-6f32, 0.75f32);
    let mut args: Vec<_> = ptrs.iter().map(arg).collect();
    args.extend([arg(&n), arg(&eps), arg(&scale)]);
    let grid = [n / NORM_THREADS, (h.len() / len) as u32, 1];
    // SAFETY: ten arguments of the kernel's types; every buffer holds `tokens × len` values or one row of weights.
    unsafe { gpu.stream.launch(&gpu.module.function(kernel).unwrap(), grid, [NORM_THREADS, 1, 1], 0, &args) }.unwrap();
    (gpu.floats(&ho), gpu.download(&q), gpu.floats(&s))
}

#[test]
fn norm_q8p_matches_norm_q8_on_many_tokens() {
    let gpu = Gpu::new();
    let (len, tokens) = (2560, 5);
    let h: Vec<f32> = values(len * tokens, 5).iter().map(|x| 20.0 * x).collect();
    let y: Vec<f32> = values(len * tokens, 6).iter().map(|x| 300.0 * x).collect();
    let (w1, w2): (Vec<f32>, Vec<f32>) = (values(len, 3).iter().map(|x| 1.0 + x).collect(), values(len, 4).iter().map(|x| 0.5 + x).collect());
    let (h8, q8, s8) = norm(&gpu, NORM_Q8, [&h, &y, &w1, &w2], len, 32);
    let (hp, qp, sp) = norm(&gpu, NORM_Q8P, [&h, &y, &w1, &w2], len, 64);
    assert_eq!(hp, h8, "the residual is the same");
    let want: Vec<f64> = q8.iter().enumerate().map(|(i, &v)| f64::from(v as i8) * f64::from(s8[i / 32 * 2])).collect();
    let slack: Vec<f64> = (0..want.len()).map(|i| 0.5 * f64::from(s8[i / 32 * 2])).collect();
    assert_q8p(&qp, &sp, &want, slack.iter().fold(0f64, |m, &v| m.max(v)));
}

fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044_715 * x * x * x)).tanh())
}

/// [`GEGLU_Q8P`] over rows read in place, its output rows packed, or padded (by 64 values, as a matrix of 704
/// columns pads to 768) with the padding left alone.
#[test]
fn geglu_q8p_gates_strided_rows() {
    let gpu = Gpu::new();
    // Three tokens of gate and up stacked in one row of 2·width, and a per-layer table of rows 3·width wide.
    let (width, tokens) = (512, 3);
    let gate_up: Vec<f32> = values(2 * width * tokens, 1).iter().map(|x| 6.0 * x).collect();
    let table = values(3 * width * tokens, 2);
    let want: Vec<f64> = (0..width * tokens)
        .map(|i| {
            let (t, j) = (i / width, i % width);
            gelu(f64::from(gate_up[t * 2 * width + j])) * f64::from(table[t * 3 * width + width + j])
        })
        .collect();
    for so in [width, width + 64] {
        let (a, b, (q, s)) = (gpu.upload(&gate_up), gpu.upload(&table), out(&gpu, so * tokens));
        let (pa, pb, pq, ps) = (a.ptr(), b.ptr() + 4 * width as u64, q.ptr(), s.ptr());
        let (len, w, sa, sb, pso) = ((width * tokens) as u32, width as u32, 2 * width as u32, 3 * width as u32, so as u32);
        let args = [arg(&pa), arg(&pb), arg(&pq), arg(&ps), arg(&len), arg(&w), arg(&sa), arg(&sb), arg(&pso)];
        // SAFETY: nine arguments of the kernel's types; every row read lies inside its buffer, the outputs hold
        // `tokens` rows of `so`.
        unsafe { gpu.stream.launch(&gpu.module.function(GEGLU_Q8P).unwrap(), [len.div_ceil(256), 1, 1], [256, 1, 1], 0, &args) }.unwrap();
        let (q, s) = (gpu.download(&q), gpu.floats(&s));
        let rows = |v: &[u8], per: usize| (0..tokens).flat_map(|t| v[t * so / per..][..width / per].to_vec()).collect::<Vec<_>>();
        let scales: Vec<f32> = (0..tokens).flat_map(|t| s[t * so / 32..][..width / 32].to_vec()).collect();
        assert_q8p(&rows(&q, 1), &scales, &want, 1e-5);
        if so > width {
            assert!(q[width..so].iter().all(|&b| b == 0), "the padding is not written");
        }
    }
}

#[test]
fn embed_decodes_a_row_per_token() {
    let gpu = Gpu::new();
    let (rows, cols) = (9, 2560);
    let w = q4(&values(rows * cols, 11), cols);
    let tokens = [5u32, 0, 8, 5];
    let (tok, y) = (gpu.upload(&tokens), gpu.zeros(tokens.len() * cols * 4));
    let (wd, sd) = (gpu.upload(&w.packed), gpu.upload(&w.scales));
    let (pw, ps, pt, py, n, scale) = (wd.ptr(), sd.ptr(), tok.ptr(), y.ptr(), cols as u32, 2.0f32);
    let args = [arg(&pw), arg(&ps), arg(&pt), arg(&py), arg(&n), arg(&scale)];
    // SAFETY: six arguments of the kernel's types; the table holds `rows × cols` weights, `y` a row per token.
    unsafe { gpu.stream.launch(&gpu.module.function(EMBED_Q4).unwrap(), [n.div_ceil(256), tokens.len() as u32, 1], [256, 1, 1], 0, &args) }.unwrap();
    let y = gpu.floats(&y);
    for (t, &token) in tokens.iter().enumerate() {
        for i in (0..cols).step_by(97) {
            let (g, k) = ((token as usize * cols + i) / 64, i % 64);
            let n = i32::from(w.packed[g * 32 + k / 8 * 4 + k % 4] >> (k % 8 / 4 * 4) & 15) - 8;
            assert_eq!(y[t * cols + i], n as f32 * tachyon::quant::f16_value(w.scales[g]) * 2.0, "token {t} value {i}");
        }
    }
}

/// Key-value heads in the attention tests (each serves four query heads, or two), and new tokens.
const KV: usize = 2;
const TOKENS: usize = 40;

/// What prefill attention reads: the head size, ring, window and first new position; the tokens' projections, the
/// norm weights, the RoPE frequencies, the caches as they were.
struct Chunk {
    d: usize,
    ratio: usize,
    cap: u32,
    win: u32,
    p0: u32,
    qkv: Vec<f32>,
    qn: Vec<f32>,
    kn: Vec<f32>,
    freq: Vec<f32>,
    k: Vec<u16>,
    v: Vec<u16>,
}

impl Chunk {
    /// A chunk at `p0` with heads of `d`, `ratio` query heads per key-value head, a ring of `cap` and a window of
    /// `win`: sliding layers' RoPE for 256, global layers' (a quarter of the pairs turning) for 512.
    fn new(d: usize, ratio: usize, cap: u32, win: u32, p0: u32) -> Chunk {
        let weights = |seed| values(d, seed).iter().map(|x| 0.2 + 0.1 * x).collect();
        let cache = |seed| values(KV * cap as usize * d, seed).iter().map(|&x| f16_bits(0.3 * x)).collect();
        let freq = if d == 256 { rope(10_000.0, d, 1.0) } else { rope(1_000_000.0, d, 0.25) };
        let qkv = values(TOKENS * (ratio + 2) * KV * d, 3);
        Chunk { d, ratio, cap, win, p0, qkv, qn: weights(4), kn: weights(5), freq, k: cache(6), v: cache(7) }
    }

    /// Head `i` of token `t`.
    fn head(&self, t: usize, i: usize) -> Vec<f64> {
        self.qkv[(t * (self.ratio + 2) * KV + i) * self.d..][..self.d].iter().map(|&v| f64::from(v)).collect()
    }

    /// Key and value of key-value head `g` at age `age` for token `t`: a new token's, normalized and rotated, or the
    /// cache's.
    fn kv(&self, t: usize, age: usize, g: usize) -> (Vec<f64>, Vec<f64>) {
        let (at, cap, d) = (self.p0 as usize + t - age, self.cap as usize, self.d);
        if age > t {
            let old = |c: &[u16]| c[(g * cap + at % cap) * d..][..d].iter().map(|&h| f64::from(f16_value(h))).collect();
            return (old(&self.k), old(&self.v));
        }
        let h = self.ratio * KV;
        let (mut k, mut v) = (self.head(t - age, h + g), self.head(t - age, h + KV + g));
        normalize(&mut k, Some(&self.kn));
        rotate(&mut k, &self.freq, at as u32);
        normalize(&mut v, None);
        (k, v)
    }

    /// Every token's heads over its window, in `f64`.
    fn cpu(&self) -> Vec<f64> {
        let mut want = Vec::new();
        for t in 0..TOKENS {
            let at = self.p0 as usize + t;
            for g in 0..KV {
                let kv: Vec<_> = (0..(at + 1).min(self.win as usize)).map(|age| self.kv(t, age, g)).collect();
                for h in self.ratio * g..self.ratio * (g + 1) {
                    let mut q = self.head(t, h);
                    normalize(&mut q, Some(&self.qn));
                    rotate(&mut q, &self.freq, at as u32);
                    want.extend(attention(&q, &kv));
                }
            }
        }
        want
    }
}

/// Prefill attention on tensor cores: [`KV_256`] stores the keys and values, [`QUERY_256`] normalizes and rotates
/// the queries to `f16`, and [`FLASH_256`] gives each token's heads over its window in `f32`, within a thousandth of
/// the largest output (`f16` queries and probabilities); on the chunk above, a sliding window of a ring, and on
/// heads of 512 with global attention ([`FLASH_512`], eight tokens a block), every earlier position seen; and with two
/// query heads per key-value head ([`FLASH_256_PAIRS`], 32 tokens a block), as the larger models' sliding layers.
#[test]
fn flash_attention_matches_the_reference() {
    let gpu = Gpu::new();
    for chunk in [Chunk::new(256, 4, 1024, 512, 600), Chunk::new(512, 4, 700, 700, 600), Chunk::new(256, 2, 1024, 512, 600)] {
        let (want, got, d, h) = (chunk.cpu(), chunk.flash(&gpu), chunk.d, chunk.ratio * KV);
        let top = want.iter().fold(0f64, |m, v| m.max(v.abs()));
        for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
            assert!((f64::from(g) - w).abs() <= 1e-3 * top, "{d}/{h}: token {}, head {}, value {}: {g} against {w}", i / (h * d), i / d % h, i % d);
        }
    }
}

impl Chunk {
    /// The key-value store, the queries and the attention on tensor cores on the GPU: every token's heads, `f32`.
    fn flash(&self, gpu: &Gpu) -> Vec<f32> {
        let (d, h, wide) = (self.d, self.ratio * KV, self.d == 512);
        let shape = tachyon::ptx::flash::Shape::of(d, self.ratio as u32);
        let tiles = TOKENS.div_ceil(shape.tokens() as usize);
        let bufs = [gpu.upload(&self.qkv), gpu.upload(&self.qn), gpu.upload(&self.kn), gpu.upload(&self.freq), gpu.upload(&self.k), gpu.upload(&self.v)];
        let [pqkv, pqn, pkn, pfreq, pk, pv] = bufs.each_ref().map(DevBuf::ptr);
        let (start, qh, y) = (gpu.upload(&[self.p0]), gpu.zeros(tiles * shape.tokens() as usize * h * d * 2), gpu.zeros(TOKENS * h * d * 4));
        let (ppos, pqh, py) = (start.ptr(), qh.ptr(), y.ptr());
        let (cap, win, group, ratio, row, n) = (self.cap, self.win, 1u32, self.ratio as u32, ((self.ratio + 2) * KV * d) as u32, TOKENS as u32);
        let store = [arg(&pqkv), arg(&pkn), arg(&pfreq), arg(&pk), arg(&pv), arg(&ppos), arg(&cap), arg(&ratio)];
        let query = [arg(&pqkv), arg(&pqn), arg(&pfreq), arg(&pqh), arg(&ppos), arg(&row)];
        let flash = [arg(&pqh), arg(&pk), arg(&pv), arg(&ppos), arg(&cap), arg(&win), arg(&group), arg(&n), arg(&py)];
        let [kv, q, attend] = if wide { [KV_512, QUERY_512, FLASH_512] } else { [KV_256, QUERY_256, shape.name()] };
        let f = |name| gpu.module.function(name).unwrap();
        // SAFETY: each launch passes its kernel's arguments in order and with their types; the buffers hold whole
        // tiles of tokens.
        unsafe {
            gpu.stream.launch(&f(kv), [KV as u32, n, 1], [32, 1, 1], 0, &store).unwrap();
            gpu.stream.launch(&f(q), [h as u32, n, 1], [32, 1, 1], 0, &query).unwrap();
            let grid = [tiles as u32, KV as u32, 1];
            gpu.stream.launch(&f(attend), grid, [shape.threads(), 1, 1], tachyon::ptx::shared(attend), &flash).unwrap();
        }
        gpu.floats(&y)
    }
}
