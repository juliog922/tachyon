//! Gemma 4's audio tower on the GPU: audio resampled to 16 kHz, its log-mel features ([`crate::media::mel`]), two
//! convolutions that subsample them four times in time (a soft token per 40 ms), then a Conformer: per layer a
//! feed-forward block, chunked local attention with relative positions, a light convolution and a second
//! feed-forward block, each with its RMS norms; then the output projection and the projection into the decoder's
//! embeddings. In `f16` with `f32` sums and an `f32` residual stream (spec, step 11 decisions), on the clipped
//! products ([`crate::ptx::encoder`]), the vision tower's RMS norm and the tower's own kernels ([`crate::ptx::audio`]).
//! The weights are `audio.wpk` in the decoder's model directory (see [`convert`]).

mod convert;

pub use convert::convert;

use crate::cuda::{Context, DevBuf, Module, Stream, arg};
use crate::encoder::race::{CLIP, Products, launch};
use crate::json::Json;
use crate::media::{self, MAX_SAMPLES, MEL_BINS, SAMPLE_RATE};
use crate::ptx::{ATTEND_AUDIO, LCONV, RMS_F16, SUBSAMPLE};
use crate::wpk::{Loader, Wpk};
use crate::{Error, Result};
use std::path::Path;

/// Dimensions of a head: [`ATTEND_AUDIO`]'s.
const HEAD: usize = 128;
/// Rows of each `f16` buffer past the most soft tokens, so a product's last block of 128 stays inside it.
const SLACK: usize = 128;
/// Feature frames of 30 s, and the soft tokens they become.
const FRAMES: usize = (MAX_SAMPLES - 1) / 160;
const TOKENS: usize = FRAMES.div_ceil(4);

/// A kernel argument.
enum A {
    P(u64),
    U(u32),
    F(f32),
}

/// The tower's shape, from its weights and `config.json`'s `audio_config`.
#[derive(Clone, Copy)]
struct Shape {
    layers: usize,
    hidden: usize,
    /// Channels of the two subsampling convolutions.
    channels: [usize; 2],
    /// Taps of the light convolution.
    taps: usize,
    out: usize,
    width: usize,
    left: usize,
    cap: f32,
    eps: f32,
}

/// The activations: the features and the first convolution's output (`f32`), the residual stream `x`, the layers'
/// outputs `t` and queries, keys and values `q` (`f32`), what products read `a` and the wide rows `b` (`f16`).
#[derive(Clone, Copy)]
struct Buffers {
    mel: u64,
    conv: u64,
    x: u64,
    t: u64,
    q: u64,
    a: u64,
    b: u64,
}

/// Gemma 4's audio tower on a GPU, with room for 30 s of audio.
pub struct Tower {
    shape: Shape,
    wpk: Wpk,
    weights: DevBuf,
    scratch: DevBuf,
    bufs: Buffers,
    module: Module,
    stream: Stream,
    /// The product kernel each shape takes.
    products: Products,
}

impl Tower {
    /// Loads the audio tower of the model directory `dir` onto `ctx`'s GPU.
    pub fn open(ctx: &Context, dir: &Path) -> Result<Tower> {
        let wpk = Wpk::open(dir.join("audio.wpk"))?;
        let shape = describe(&wpk, &std::fs::read(dir.join("config.json"))?)?;
        let weights = ctx.alloc_pooled(wpk.data_len())?;
        Loader::new(ctx)?.to_device(&wpk, &weights)?;
        let (scratch, bufs) = scratch(ctx, shape)?;
        let (module, stream) = (ctx.load(&crate::ptx::module())?, ctx.stream()?);
        Ok(Tower { shape, wpk, weights, scratch, bufs, module, stream, products: Products::default() })
    }

    /// The width of a soft token: the decoder's hidden size.
    pub fn width(&self) -> usize {
        self.shape.width
    }

    /// The VRAM it holds, in bytes: weights and activations.
    pub fn bytes(&self) -> usize {
        self.weights.len() + self.scratch.len()
    }

    /// The soft tokens Gemma 4 makes of `audio` (its first 30 s): one per 40 ms, as [`soft_tokens`] counts them.
    pub fn soft_tokens(&mut self, audio: &media::Audio) -> Result<Vec<f32>> {
        let samples = audio.resampled(SAMPLE_RATE)?.samples;
        let n = soft_tokens(samples.len());
        if n == 0 {
            return Err(Error::Media(format!("{} samples of audio: too short for a soft token", samples.len())));
        }
        let mel: Vec<u8> = media::mel(&samples).iter().flat_map(|v| v.to_le_bytes()).collect();
        let at = (self.bufs.mel - self.scratch.ptr()) as usize;
        // SAFETY: `mel` lives until the stream is synchronized below, also when a launch fails.
        let pass = || unsafe { self.stream.upload(&self.scratch, at, &mel) }.and_then(|()| self.run(valid(samples.len()), n));
        // A pass that raced products is done again, with the kernels picked.
        let run = pass().and_then(|()| if self.products.raced() { pass() } else { Ok(()) });
        self.stream.sync()?;
        run?;
        self.download(self.bufs.t, n * self.shape.width)
    }

    /// Address of tensor `name`, 0 if there is none.
    fn at(&self, name: &str) -> u64 {
        self.wpk.tensor(name).map_or(0, |t| self.weights.ptr() + t.offset)
    }

    fn launch(&self, name: &str, grid: [u32; 3], threads: u32, args: &[A]) -> Result<()> {
        let ptrs: Vec<_> = args
            .iter()
            .map(|a| match a {
                A::P(v) => arg(v),
                A::U(v) => arg(v),
                A::F(v) => arg(v),
            })
            .collect();
        // SAFETY: every call passes the kernel's arguments in order and with their types, as `crate::ptx` documents
        // them, on buffers laid out for 30 s of audio.
        unsafe { self.stream.launch(&self.module.function(name)?, grid, [threads, 1, 1], 0, &ptrs) }
    }

    /// `y = clip(clip(a) · wᵀ) (+ bias)` (`io` is `[a, y]`, `shape` is `(tokens, mode)`), as
    /// [`crate::ptx::GEMM_CLIP`] stores it; the bias is `{w}.bias` when the file has one.
    fn gemm(&self, w: &str, io: [u64; 2], shape: (usize, u32)) -> Result<()> {
        let ([a, y], (tokens, mode)) = (io, shape);
        let matrix = self.wpk.tensor(w).ok_or_else(|| Error::Format(format!("audio.wpk: no tensor {w}")))?;
        let (rows, cols, n) = (matrix.shape[0] as u32, matrix.shape[1] as u32, tokens as u32);
        let (bias, clip) = (self.at(&format!("{w}.bias")), self.at(&format!("{w}.clip")));
        let args = [A::P(a), A::P(self.at(w)), A::P(bias), A::P(y), A::U(rows), A::U(cols), A::U(n), A::U(mode), A::P(clip)];
        let run = |k: usize| {
            let (kernel, grid, threads) = launch(&CLIP, k, n, rows);
            self.launch(kernel, [grid, 1, 1], threads, &args)
        };
        run(self.products.pick(&self.stream, &self.module, [rows, cols, n], &run)?)
    }

    /// [`RMS_F16`] over `rows` rows of `h`, `io` being `[x, y, w1, w2, a, xo]`.
    fn rms(&self, io: [u64; 6], (rows, h): (usize, usize)) -> Result<()> {
        let mut args: Vec<A> = io.into_iter().map(A::P).collect();
        args.extend([A::U(h as u32), A::F(self.shape.eps)]);
        self.launch(RMS_F16, [rows as u32, 1, 1], h as u32 / 4, &args)
    }

    /// The tower over the features in `mel`, `frames` of them valid, to `n` soft tokens in `t`.
    fn run(&self, frames: usize, n: usize) -> Result<()> {
        let (s, Buffers { mel, conv, x, a, .. }) = (self.shape, self.bufs);
        let rows = frames.div_ceil(2);
        let sub = |input: u64, (k, fin, cin): (usize, usize, usize), out: u64, grid: [u32; 3]| {
            let half = u32::from(k == 1);
            let args = [
                A::P(input),
                A::P(self.at(&format!("sub{k}"))),
                A::P(self.at(&format!("sub{k}.norm"))),
                A::P(out),
                A::U(grid[2]),
                A::U(fin as u32),
                A::U(cin as u32),
                A::U(half),
                A::F(s.eps),
            ];
            self.launch(SUBSAMPLE, [grid[0], grid[1], 1], s.channels[k] as u32, &args)
        };
        sub(mel, (0, MEL_BINS, 1), conv, [MEL_BINS as u32 / 2, rows as u32, frames as u32])?;
        sub(conv, (1, MEL_BINS / 2, s.channels[0]), a, [MEL_BINS as u32 / 4, n as u32, rows as u32])?;
        self.gemm("in", [a, x], (n, 0))?;
        (0..s.layers).try_for_each(|i| self.layer(i, n))?;
        self.gemm("out", [a, self.bufs.t], (n, 0))?;
        self.rms([self.bufs.t, 0, 0, 0, a, 0], (n, s.out))?;
        self.gemm("proj", [a, self.bufs.t], (n, 0))
    }

    /// Layer `i` over `n` tokens: `x` in and out; for the last layer, its output also as `f16` in `a`.
    fn layer(&self, i: usize, n: usize) -> Result<()> {
        let (s, Buffers { x, t, a, .. }) = (self.shape, self.bufs);
        let w = |name: &str| self.at(&format!("{i}.{name}"));
        self.ffn(i, "ffn1", n)?;
        self.rms([x, t, w("ffn1.post"), w("attn_pre"), a, 0], (n, s.hidden))?;
        self.attention(i, n)?;
        self.rms([x, t, w("attn_post"), w("conv_pre"), a, 0], (n, s.hidden))?;
        self.light(i, n)?;
        self.ffn(i, "ffn2", n)?;
        let last = i + 1 == s.layers;
        self.rms([x, t, w("ffn2.post"), w("out_norm"), if last { a } else { 0 }, if last { 0 } else { x }], (n, s.hidden))
    }

    /// Layer `i`'s light convolution over `n` tokens, from `a`, added to `x`.
    fn light(&self, i: usize, n: usize) -> Result<()> {
        let (s, Buffers { x, a, b, .. }) = (self.shape, self.bufs);
        let w = |name: &str| self.at(&format!("{i}.{name}"));
        self.gemm(&format!("{i}.start"), [a, b], (n, 5))?;
        let args = [A::P(b), A::P(w("taps")), A::P(w("conv_norm")), A::P(a), A::U(s.hidden as u32), A::U(s.taps as u32), A::F(s.eps)];
        self.launch(LCONV, [n as u32, 1, 1], s.hidden as u32 / 4, &args)?;
        self.gemm(&format!("{i}.end"), [a, x], (n, 1))
    }

    /// Layer `i`'s feed-forward block `f` over `n` tokens, from `x`, normalized into `a`, to `t`.
    fn ffn(&self, i: usize, f: &str, n: usize) -> Result<()> {
        let Buffers { x, t, a, b, .. } = self.bufs;
        self.rms([x, 0, 0, self.at(&format!("{i}.{f}.pre")), a, 0], (n, self.shape.hidden))?;
        self.gemm(&format!("{i}.{f}.up"), [a, b], (n, 3))?;
        self.gemm(&format!("{i}.{f}.down"), [b, t], (n, 0))
    }

    /// Layer `i`'s attention over `n` tokens, from `a` to `t`.
    fn attention(&self, i: usize, n: usize) -> Result<()> {
        let (s, Buffers { t, q, a, .. }) = (self.shape, self.bufs);
        self.gemm(&format!("{i}.qkv"), [a, q], (n, 0))?;
        let kscale = (std::f64::consts::E.ln_1p() / std::f64::consts::LN_2) as f32;
        let (rel, scale) = (self.at(&format!("{i}.rel")), self.at(&format!("{i}.scale")));
        let args = [A::P(q), A::P(rel), A::P(scale), A::P(a), A::U(s.hidden as u32), A::U(s.left as u32), A::F(kscale), A::F(s.cap)];
        self.launch(ATTEND_AUDIO, [n as u32, (s.hidden / HEAD) as u32, 1], 32, &args)?;
        self.gemm(&format!("{i}.o"), [a, t], (n, 0))
    }

    /// `len` values of `f32` from device address `src`, in the scratch buffer.
    fn download(&self, src: u64, len: usize) -> Result<Vec<f32>> {
        let mut raw = vec![0u8; 4 * len];
        // SAFETY: `src` is in the scratch buffer with `len` values; the stream is idle, and `sync` completes the
        // copy before `raw` is read.
        unsafe { self.stream.download(&mut raw, &self.scratch, (src - self.scratch.ptr()) as usize) }?;
        self.stream.sync()?;
        Ok(raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
    }
}

/// The feature frames of `len` samples (16 kHz) that hold no padding: those the tower reads.
fn valid(len: usize) -> usize {
    len.min(MAX_SAMPLES).saturating_sub(160).div_ceil(160)
}

/// The soft tokens Gemma 4 makes of `audio`, once at 16 kHz.
pub fn count(audio: &media::Audio) -> usize {
    soft_tokens((audio.samples.len() * SAMPLE_RATE as usize).div_ceil(audio.rate.max(1) as usize))
}

/// The soft tokens Gemma 4 makes of `len` samples at 16 kHz (at most 30 s of them): one per four valid feature
/// frames, rounding up twice, as its two convolutions of stride 2 do.
pub fn soft_tokens(len: usize) -> usize {
    valid(len).div_ceil(2).div_ceil(2)
}

/// The tower's shape: from the weights, and attention's reach, its logits' cap and the norms' `eps` from `config`
/// (Gemma 4's defaults when it has none). An error unless every tensor is there in a shape the kernels take.
fn describe(wpk: &Wpk, config: &[u8]) -> Result<Shape> {
    let dims = |name: &&str| wpk.tensor(name).map(|t| t.shape.clone()).ok_or_else(|| Error::Format(format!("audio.wpk: no tensor {name}")));
    let d = ["in", "out", "proj", "0.taps", "sub0.norm", "sub1.norm"].iter().map(dims).collect::<Result<Vec<_>>>()?;
    let (inp, out, proj, taps, channels) = (&d[0], &d[1], &d[2], &d[3], [d[4][0] as usize, d[5][0] as usize]);
    let ([left, right], cap, eps) = settings(config)?;
    let hidden = inp[0] as usize;
    let layers = (0..wpk.tensors().len()).take_while(|i| wpk.tensor(&format!("{i}.qkv")).is_some()).count();
    let shape = Shape { layers, hidden, channels, taps: taps[0] as usize / hidden.max(1), out: out[0] as usize, width: proj[0] as usize, left, cap, eps };
    if !fits(&shape, inp[1] as usize) || right != 0 {
        return Err(Error::Unsupported(format!("an audio tower of {hidden} with {channels:?} channels")));
    }
    Ok(shape)
}

/// Whether the kernels take a tower of `shape` whose input projection reads `cols` values.
fn fits(s: &Shape, cols: usize) -> bool {
    s.layers > 0
        && s.hidden % HEAD == 0
        && s.hidden <= 4096
        && s.out % 128 == 0
        && s.channels.iter().all(|c| c % 32 == 0 && *c <= 1024)
        && cols == MEL_BINS / 4 * s.channels[1]
}

/// `config.json`'s `audio_config`: the past and future positions attention sees, the logits' cap and the norms'
/// `eps`, Gemma 4's defaults where it has none.
fn settings(config: &[u8]) -> Result<([usize; 2], f32, f32)> {
    let doc = Json::parse(config)?;
    let num = |k: &str, or: f64| doc.get("audio_config").and_then(|a| a.get(k)).and_then(Json::num).unwrap_or(or);
    // A query sees itself and the `attention_context_left − 2` positions before it (Hugging Face's mask, `0 ≤ t − s <
    // attention_context_left − 1`).
    let reach = [num("attention_context_left", 13.0) as usize - 2, num("attention_context_right", 0.0) as usize];
    Ok((reach, num("attention_logit_cap", 50.0) as f32, num("rms_norm_eps", 1e-6) as f32))
}

/// The activation buffers for 30 s of audio, in one allocation.
fn scratch(ctx: &Context, s: Shape) -> Result<(DevBuf, Buffers)> {
    let rows = TOKENS + SLACK;
    let sizes = [
        4 * FRAMES * MEL_BINS,
        4 * FRAMES.div_ceil(2) * MEL_BINS / 2 * s.channels[0],
        4 * TOKENS * s.hidden,
        4 * TOKENS * s.hidden.max(s.out).max(s.width),
        4 * TOKENS * 3 * s.hidden,
        2 * rows * s.hidden.max(s.out),
        2 * rows * 4 * s.hidden,
    ];
    let buf = ctx.alloc(sizes.iter().sum())?;
    let at = |i: usize| buf.ptr() + sizes[..i].iter().sum::<usize>() as u64;
    let bufs = Buffers { mel: at(0), conv: at(1), x: at(2), t: at(3), q: at(4), a: at(5), b: at(6) };
    Ok((buf, bufs))
}
