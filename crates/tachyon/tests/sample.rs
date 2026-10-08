//! The sampler on a real GPU: greedy picks against the CPU, and draws against the filtered distribution with a
//! χ² test. `cargo test --features gpu`; `TACHYON_TEST_DRAWS` sets the draws per distribution (250 000).
#![cfg(feature = "gpu")]

mod common;

use common::{Gpu, values};
use tachyon::cuda::{DevBuf, Stream, arg};
use tachyon::ptx::{SAMPLE, SAMPLE_THREADS, Sampling};

const PROBE: &str = include_str!("kernels/probe.ptx");
/// Most sampler blocks: fewer than the vocabulary needs, so each block walks several shares of it.
const BLOCKS: u32 = 7;

/// Device buffers of one sampler: logits and their transformed copy, settings, state, seen and allowed bitmaps.
struct Sampler {
    vocab: u32,
    logits: DevBuf,
    work: DevBuf,
    cfg: DevBuf,
    state: DevBuf,
    seen: DevBuf,
    seen0: DevBuf,
    _allowed: DevBuf,
    partial: DevBuf,
    count: DevBuf,
}

impl Sampler {
    fn new(gpu: &Gpu, logits: &[f32], seen: &[u32], allowed: &[u32], cfg: Sampling) -> Sampler {
        let vocab = logits.len() as u32;
        let allowed = gpu.upload(allowed);
        let cfg = Sampling { allowed: if cfg.allowed == 0 { 0 } else { allowed.ptr() }, ..cfg };
        let blocks = BLOCKS as usize;
        Sampler {
            vocab,
            logits: gpu.upload(logits),
            work: gpu.zeros(logits.len() * 4),
            cfg: gpu.upload(&cfg.bytes()),
            state: gpu.zeros(8),
            seen: gpu.upload(seen),
            seen0: gpu.upload(seen),
            _allowed: allowed,
            partial: gpu.zeros(32 * blocks),
            count: gpu.zeros(4),
        }
    }

    /// Queues one draw with the initial seen bitmap; `out` also receives the token when not null.
    fn draw(&self, gpu: &Gpu, s: &Stream, out: u64) -> tachyon::Result<()> {
        s.copy(&self.seen, &self.seen0)?;
        let [pl, pw, pc, pst, pse, ppa, pco] = [&self.logits, &self.work, &self.cfg, &self.state, &self.seen, &self.partial, &self.count].map(DevBuf::ptr);
        let args = [arg(&pl), arg(&pw), arg(&self.vocab), arg(&pc), arg(&pst), arg(&pse), arg(&out), arg(&ppa), arg(&pco)];
        let blocks = self.vocab.div_ceil(4 * SAMPLE_THREADS).min(BLOCKS);
        // SAFETY: nine arguments of the kernel's types; every buffer is sized for `vocab`.
        unsafe { s.launch(&gpu.module.function(SAMPLE).unwrap(), [blocks, 1, 1], [SAMPLE_THREADS, 1, 1], 0, &args) }
    }
}

/// What the sampler does to logit `i`, in `f64`.
fn transform(x: f32, i: usize, cfg: &Sampling, seen: &[u32], allowed: &[u32]) -> f64 {
    let bit = |m: &[u32]| m[i / 32] >> (i % 32) & 1 == 1;
    let mut v = f64::from(x);
    if cfg.softcap > 0.0 {
        v = f64::from(cfg.softcap) * (v / f64::from(cfg.softcap)).tanh();
    }
    if cfg.allowed != 0 && !bit(allowed) {
        return f64::NEG_INFINITY;
    }
    if bit(seen) {
        v = if v > 0.0 { v / f64::from(cfg.penalty) } else { v * f64::from(cfg.penalty) };
    }
    if cfg.temperature > 0.0 { v / f64::from(cfg.temperature) } else { v }
}

/// The probability of each token under `cfg`: softmax over the tokens every filter keeps.
fn expected(x: &[f64], cfg: &Sampling) -> Vec<f64> {
    let m = x.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v));
    let p: Vec<f64> = x.iter().map(|v| (v - m).exp()).collect();
    let z: f64 = p.iter().sum();
    let keep = |i: usize| {
        let above = x.iter().zip(&p).filter(|(v, _)| **v > x[i]);
        let (count, mass) = above.fold((0, 0.0), |(c, s), (_, p)| (c + 1, s + p));
        (cfg.top_k == 0 || count < cfg.top_k) && (cfg.top_p >= 1.0 || mass < f64::from(cfg.top_p) * z || mass == 0.0) && x[i] - m >= f64::from(cfg.min_p).ln()
    };
    let kept: Vec<f64> = (0..x.len()).map(|i| if keep(i) { p[i] } else { 0.0 }).collect();
    let sum: f64 = kept.iter().sum();
    kept.iter().map(|k| k / sum).collect()
}

#[test]
fn greedy_picks_the_largest_transformed_logit() {
    let gpu = Gpu::new();
    let vocab = 262_144;
    let logits: Vec<f32> = values(vocab, 1).iter().map(|x| 40.0 * x).collect();
    let seen: Vec<u32> = (0..vocab / 32).map(|i| if i % 3 == 0 { 0xF0F0_F0F0 } else { 0 }).collect();
    let allowed: Vec<u32> = (0..vocab / 32).map(|i| if i % 5 == 0 { 0 } else { u32::MAX }).collect();
    let cfg = Sampling { softcap: 30.0, penalty: 1.3, allowed: 1, ..Sampling::default() };
    let x: Vec<f64> = logits.iter().enumerate().map(|(i, &v)| transform(v, i, &cfg, &seen, &allowed)).collect();
    let want = (0..vocab).fold(0, |b, i| if x[i] > x[b] { i } else { b }) as u32;
    let sampler = Sampler::new(&gpu, &logits, &seen, &allowed, cfg);
    let mut out = gpu.ctx.alloc_host(4).unwrap();
    for step in 1..=2 {
        sampler.draw(&gpu, &gpu.stream, out.device_ptr().unwrap()).unwrap();
        assert_eq!(gpu.words(&sampler.state), [step, want], "step counter and token");
        assert_eq!(u32::from_ne_bytes(out[..4].try_into().unwrap()), want, "token in host memory");
        assert_eq!(gpu.words(&sampler.seen)[want as usize / 32] >> (want % 32) & 1, 1, "token marked as seen");
        out.fill(0);
    }
}

/// Draws `n` tokens and returns how often each came up.
fn histogram(gpu: &Gpu, sampler: &Sampler, n: usize) -> Vec<u32> {
    let probe = gpu.ctx.load(PROBE).unwrap();
    let tally = probe.function("tally").unwrap();
    let hist = gpu.zeros(4 * sampler.vocab as usize);
    let (token, ph) = (sampler.state.ptr() + 4, hist.ptr());
    let batch = 100;
    let graph = gpu
        .stream
        .capture(|s| {
            for _ in 0..batch {
                sampler.draw(gpu, s, 0)?;
                // SAFETY: two arguments of the kernel's types; `hist` holds a counter per token.
                unsafe { s.launch(&tally, [1, 1, 1], [1, 1, 1], 0, &[arg(&token), arg(&ph)]) }?;
            }
            Ok(())
        })
        .unwrap();
    for _ in 0..n / batch {
        // SAFETY: the buffers and modules the graph uses are alive.
        unsafe { gpu.stream.replay(&graph) }.unwrap();
    }
    gpu.words(&hist)
}

/// Draws under `cfg` and checks the counts against [`expected`] with a χ² test.
fn check_draws(gpu: &Gpu, vocab: usize, cfg: Sampling) {
    let n: usize = std::env::var("TACHYON_TEST_DRAWS").ok().and_then(|v| v.parse().ok()).unwrap_or(250_000);
    let logits: Vec<f32> = values(vocab, u64::from(cfg.top_k) + 7).iter().map(|x| 3.0 * x).collect();
    let seen: Vec<u32> = (0..vocab / 32).map(|i| 0x0101_0101 << (i % 8)).collect();
    let allowed: Vec<u32> = (0..vocab / 32).map(|i| if i % 4 == 1 { 0x00FF_FF00 } else { u32::MAX }).collect();
    let x: Vec<f64> = logits.iter().enumerate().map(|(i, &v)| transform(v, i, &cfg, &seen, &allowed)).collect();
    let p = expected(&x, &cfg);
    let hist = histogram(gpu, &Sampler::new(gpu, &logits, &seen, &allowed, cfg), n);
    let mut chi2 = 0.0;
    let mut df = 0;
    for (i, (&got, &want)) in hist.iter().zip(&p).enumerate() {
        assert!(want > 0.0 || got == 0, "{cfg:?}: token {i} is filtered out but was drawn {got} times");
        if want > 0.0 {
            let e = want * n as f64;
            chi2 += (f64::from(got) - e).powi(2) / e;
            df += 1;
        }
    }
    let df = f64::from(df - 1);
    assert!(chi2 < df + 6.0 * (2.0 * df).sqrt() + 10.0, "{cfg:?}: χ² {chi2:.1} with {df} degrees of freedom");
}

#[test]
fn draws_follow_the_temperature_distribution() {
    check_draws(&Gpu::new(), 256, Sampling { temperature: 1.0, seed: 1, ..Sampling::default() });
}

#[test]
fn draws_follow_top_k_across_blocks() {
    check_draws(&Gpu::new(), 8192, Sampling { temperature: 0.7, top_k: 10, seed: 2, ..Sampling::default() });
}

#[test]
fn draws_follow_top_p() {
    check_draws(&Gpu::new(), 256, Sampling { temperature: 1.3, top_p: 0.8, seed: 3, ..Sampling::default() });
}

#[test]
fn draws_follow_min_p_softcap_penalty_and_mask() {
    check_draws(&Gpu::new(), 256, Sampling { temperature: 1.0, min_p: 0.05, softcap: 5.0, penalty: 1.5, allowed: 1, seed: 4, ..Sampling::default() });
}