//! Gemma 4's vision tower on the GPU: an image, fitted to its soft-token budget ([`Image::fitted`]), cut into
//! patches of [`PATCH`] pixels; the patches through a vision transformer with Gemma 4's sandwich norms, per-head norms, 2-D RoPE
//! and clipped linear layers; then pooled [`POOL`] × [`POOL`] into soft tokens and projected into the decoder's
//! embeddings. In `f16` with `f32` sums and an `f32` residual stream (spec, step 11 decisions), on the encoder's
//! tensor-core products and attention ([`crate::ptx::encoder`]) and the tower's own kernels
//! ([`crate::ptx::vision`]). The weights are `vision.wpk` in the decoder's model directory (see [`convert`]).

pub(crate) mod convert;

pub use convert::convert;

use crate::cuda::{Context, DevBuf, HostBuf, Module, Stream, arg};
use crate::encoder::race::{CLIP, Products, launch};
use crate::json::Json;
use crate::media::{BUDGETS, Image, PATCH, POOL};
use crate::ptx::{ADD_ROWS_F16, ATTEND_F16, HEADS_F16, POOL_F16, RMS_F16};
use crate::quant::f16_bits;
use crate::wpk::{Loader, Wpk};
use crate::{Error, Result};
use std::path::Path;

/// Dimensions of a head: [`HEADS_F16`]'s.
const HEAD: usize = 64;
/// Rows of each `f16` buffer past the patches a budget holds, so a product's last block of 128 stays inside it.
const SLACK: usize = 128;

/// A kernel argument.
enum A {
    P(u64),
    U(u32),
    F(f32),
}

/// The activations: the residual stream `x` and the layers' outputs `t` (`f32`), what products read `a` and the
/// wide rows `b` (`f16`).
#[derive(Clone, Copy)]
struct Buffers {
    x: u64,
    t: u64,
    a: u64,
    b: u64,
}

/// The tower's shape, from its weights and `config.json`'s `vision_config`.
#[derive(Clone, Copy)]
struct Shape {
    layers: usize,
    hidden: usize,
    inter: usize,
    out: usize,
    positions: usize,
    eps: f32,
    theta: f32,
}

/// Gemma 4's vision tower on a GPU, with room for images of up to a budget of soft tokens.
pub struct Vision {
    shape: Shape,
    wpk: Wpk,
    weights: DevBuf,
    rope: DevBuf,
    scratch: DevBuf,
    bufs: Buffers,
    /// Pinned: each patch's `x` and `y` positions, and the attention's tiles.
    inputs: HostBuf,
    module: Module,
    stream: Stream,
    /// The most patches an image has.
    cap: usize,
    /// The product kernel each shape takes.
    products: Products,
}

impl Vision {
    /// Loads the vision tower of the model directory `dir` onto `ctx`'s GPU, for images of up to `budget` soft
    /// tokens (one of [`BUDGETS`]).
    pub fn open(ctx: &Context, dir: &Path, budget: usize) -> Result<Vision> {
        if !BUDGETS.contains(&budget) {
            return Err(Error::Media(format!("{budget} soft tokens per image; Gemma 4 takes {BUDGETS:?}")));
        }
        let wpk = Wpk::open(dir.join("vision.wpk"))?;
        let shape = describe(&wpk, &std::fs::read(dir.join("config.json"))?)?;
        let cap = budget * POOL * POOL;
        let ((weights, rope), (scratch, bufs)) = (weights(ctx, &wpk, shape)?, scratch(ctx, shape, cap)?);
        let inputs = ctx.alloc_host(4 * (2 * cap + 3 * (cap / 64 + 1)))?;
        let (module, stream) = (ctx.load(&crate::ptx::module())?, ctx.stream()?);
        Ok(Vision { shape, wpk, weights, rope, scratch, bufs, inputs, module, stream, cap, products: Products::default() })
    }

    /// The width of a soft token: the decoder's hidden size.
    pub fn width(&self) -> usize {
        self.shape.out
    }

    /// The VRAM it holds, in bytes: weights, RoPE table, activations.
    pub fn bytes(&self) -> usize {
        self.weights.len() + self.rope.len() + self.scratch.len()
    }

    /// The soft tokens of `image` at `budget` soft tokens: `f32 [n, width]`, the image's squares of [`POOL`] ×
    /// [`POOL`] patches row by row.
    pub fn soft_tokens(&mut self, image: &Image, budget: usize) -> Result<Vec<f32>> {
        let image = image.fitted(budget)?;
        let (gw, gh) = (image.width / PATCH, image.height / PATCH);
        let n = gw * gh;
        if n > self.cap {
            return Err(Error::Media(format!("{budget} soft tokens: this tower holds {} at most", self.cap / (POOL * POOL))));
        }
        let patches = patches(&image);
        self.fill(gw, n);
        let at = (self.bufs.a - self.scratch.ptr()) as usize;
        // SAFETY: `patches` lives until the stream is synchronized below, also when a launch fails.
        let pass = || unsafe { self.stream.upload(&self.scratch, at, &patches) }.and_then(|()| self.run(n, gw));
        // A pass that raced products is done again, with the kernels picked.
        let run = pass().and_then(|()| if self.products.raced() { pass() } else { Ok(()) });
        self.stream.sync()?;
        run?;
        self.download(self.bufs.t, n / (POOL * POOL) * self.shape.out)
    }

    /// Writes the positions of a grid `gw` patches across, `n` in all, and the tiles of 64 queries.
    fn fill(&mut self, gw: usize, n: usize) {
        let cap = self.cap;
        let words: &mut [u8] = &mut self.inputs;
        let mut put = |i: usize, v: usize| words[4 * i..][..4].copy_from_slice(&(v as u32).to_le_bytes());
        for i in 0..n {
            put(i, i % gw);
            put(cap + i, i / gw);
        }
        for (k, q) in (0..n).step_by(64).enumerate() {
            (0..3).for_each(|j| put(2 * cap + 3 * k + j, [0, n, q][j]));
        }
    }

    /// Device address of the inputs' part `i`: `x` positions, `y` positions, tiles.
    fn input(&self, i: usize) -> Result<u64> {
        Ok(self.inputs.device_ptr()? + (4 * self.cap * i) as u64)
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
        // them, on buffers laid out for the patches this tower was opened for.
        unsafe { self.stream.launch(&self.module.function(name)?, grid, [threads, 1, 1], 0, &ptrs) }
    }

    /// `y = clip(clip(a) · wᵀ)` (`io` is `[a, y]`, `shape` is `(tokens, mode)`) for `tokens` rows of `a`, stored as `mode` says (see [`crate::ptx::GEMM_CLIP`]), by
    /// the kernel [`Products`] picks.
    fn gemm(&self, w: &str, io: [u64; 2], shape: (usize, u32)) -> Result<()> {
        let ([a, y], (tokens, mode)) = (io, shape);
        let matrix = self.wpk.tensor(w).ok_or_else(|| Error::Format(format!("vision.wpk: no tensor {w}")))?;
        let (rows, cols, n) = (matrix.shape[0] as u32, matrix.shape[1] as u32, tokens as u32);
        let args = [A::P(a), A::P(self.at(w)), A::P(0), A::P(y), A::U(rows), A::U(cols), A::U(n), A::U(mode), A::P(self.at(&format!("{w}.clip")))];
        let run = |k: usize| {
            let (kernel, grid, threads) = launch(&CLIP, k, n, rows);
            self.launch(kernel, [grid, 1, 1], threads, &args)
        };
        run(self.products.pick(&self.stream, &self.module, [rows, cols, n], &run)?)
    }

    /// [`RMS_F16`] over `rows` rows, `io` being `[x, y, w1, w2, a]`: `x += w1 ⊙ rms(y)` unless `y` is 0, then `a = w2 ⊙ rms(x)` unless `a` is 0.
    fn rms(&self, io: [u64; 5], rows: usize, eps: f32) -> Result<()> {
        let [x, y, w1, w2, a] = io;
        let h = self.shape.hidden as u32;
        self.launch(RMS_F16, [rows as u32, 1, 1], h / 4, &[A::P(x), A::P(y), A::P(w1), A::P(w2), A::P(a), A::P(0), A::U(h), A::F(eps)])
    }

    /// The tower over `n` patches in `a`, a grid `gw` across, to the soft tokens in `t`.
    fn run(&self, n: usize, gw: usize) -> Result<()> {
        let (s, Buffers { x, t, a, .. }) = (self.shape, self.bufs);
        let h = s.hidden as u32;
        self.gemm("patch", [a, x], (n, 0))?;
        let pos = self.at("pos");
        for (i, table) in [pos, pos + (s.positions * s.hidden * 2) as u64].into_iter().enumerate() {
            self.launch(ADD_ROWS_F16, [h.div_ceil(256), n as u32, 1], 256, &[A::P(table), A::P(self.input(i)?), A::P(x), A::U(h)])?;
        }
        self.rms([x, 0, 0, self.at("0.input_norm"), a], n, s.eps)?;
        (0..s.layers).try_for_each(|i| self.layer(i, n))?;
        let m = n / (POOL * POOL);
        self.launch(POOL_F16, [h.div_ceil(256), m as u32, 1], 256, &[A::P(x), A::P(t), A::U(gw as u32), A::U(h)])?;
        // The pooled tokens times √hidden, normalized: the factor moves into `eps`.
        self.rms([t, 0, 0, 0, a], m, s.eps / s.hidden as f32)?;
        self.gemm("proj", [a, t], (m, 0))
    }

    /// Layer `i` over `n` patches: `x` in, normalized into `a`; the next layer's input norm of `x` out in `a`.
    fn layer(&self, i: usize, n: usize) -> Result<()> {
        let (s, Buffers { x, t, a, b }) = (self.shape, self.bufs);
        let w = |name: &str| self.at(&format!("{i}.{name}"));
        self.attention(i, n)?;
        self.gemm(&format!("{i}.o"), [a, t], (n, 0))?;
        self.rms([x, t, w("post_attn_norm"), w("pre_ffn_norm"), a], n, s.eps)?;
        self.gemm(&format!("{i}.gate_up"), [a, b], (n, 4))?;
        self.gemm(&format!("{i}.down"), [b, t], (n, 0))?;
        let next = if i + 1 == s.layers { 0 } else { a };
        self.rms([x, t, w("post_ffn_norm"), self.at(&format!("{}.input_norm", i + 1)), next], n, s.eps)
    }

    /// Layer `i`'s attention over `n` patches, from `a` into `a`: queries, keys and values, their norms and RoPE.
    fn attention(&self, i: usize, n: usize) -> Result<()> {
        let (s, Buffers { a, b, .. }) = (self.shape, self.bufs);
        let (width, heads) = (s.hidden as u32, (s.hidden / HEAD) as u32);
        self.gemm(&format!("{i}.qkv"), [a, b], (n, 2))?;
        let (qn, kn) = (self.at(&format!("{i}.q_norm")), self.at(&format!("{i}.k_norm")));
        let args = [A::P(b), A::P(self.input(0)?), A::P(self.input(1)?), A::P(self.rope.ptr()), A::P(qn), A::P(kn), A::U(width), A::F(s.eps)];
        self.launch(HEADS_F16, [3, n as u32, 1], 32 * heads, &args)?;
        let attend = [A::P(b), A::P(a), A::P(self.input(2)?), A::U(width), A::U(0), A::F(std::f32::consts::LOG2_E)];
        self.launch(ATTEND_F16, [n.div_ceil(64) as u32, heads, 1], 128, &attend)
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

/// The tower's shape: from the weights, and `eps` and `θ` from `config` (Gemma 4's defaults when it has none). An
/// error unless every tensor is there in a shape the kernels take.
fn describe(wpk: &Wpk, config: &[u8]) -> Result<Shape> {
    let dims = |name: &str| wpk.tensor(name).map(|t| t.shape[0] as usize).ok_or_else(|| Error::Format(format!("vision.wpk: no tensor {name}")));
    let (hidden, positions, out, inter) = (dims("patch")?, dims("pos")? / 2, dims("proj")?, dims("0.gate_up")? / 2);
    let (eps, theta, head) = settings(config)?;
    let layers = (0..wpk.tensors().len()).take_while(|i| wpk.tensor(&format!("{i}.qkv")).is_some()).count();
    let pixels = wpk.tensor("patch").map_or(0, |t| t.shape[1] as usize);
    if !fits([pixels, hidden, head, out]) {
        return Err(Error::Unsupported(format!("a vision tower of {hidden} × {inter} with heads of {head}")));
    }
    check(wpk, layers).map(|()| Shape { layers, hidden, inter, out, positions, eps, theta })
}

/// Whether the kernels take a tower of `[pixels per patch, hidden, head dimensions, text hidden]`: [`RMS_F16`]
/// takes whole warps of rows, [`HEADS_F16`] heads of 64 and a warp per head.
fn fits([pixels, hidden, head, out]: [usize; 4]) -> bool {
    pixels == 3 * PATCH * PATCH && hidden % 128 == 0 && hidden <= 2048 && head == HEAD && out % 64 == 0
}

/// `config.json`'s `vision_config`: its norms' `eps`, RoPE's `θ` and its heads' dimensions, Gemma 4's defaults
/// where it has none.
fn settings(config: &[u8]) -> Result<(f32, f32, usize)> {
    let doc = Json::parse(config)?;
    let v = doc.get("vision_config");
    let num = |k: &str, or: f64| v.and_then(|v| v.get(k)).and_then(Json::num).unwrap_or(or);
    let theta = v.and_then(|v| v.get("rope_parameters")?.get("rope_theta")?.num()).unwrap_or(100.0);
    Ok((num("rms_norm_eps", 1e-6) as f32, theta as f32, num("head_dim", HEAD as f64) as usize))
}

/// An error unless every layer's tensors are in the file.
fn check(wpk: &Wpk, layers: usize) -> Result<()> {
    let parts = ["qkv", "o", "gate_up", "down"].into_iter().flat_map(|m| [m.to_string(), format!("{m}.clip")]);
    let norms = ["input_norm", "post_attn_norm", "pre_ffn_norm", "post_ffn_norm", "q_norm", "k_norm"].map(String::from);
    let names = (0..layers).flat_map(|i| parts.clone().chain(norms.clone()).map(move |n| format!("{i}.{n}")));
    match ["patch.clip", "proj.clip"].map(String::from).into_iter().chain(names).find(|n| wpk.tensor(n).is_none()) {
        Some(n) => Err(Error::Format(format!("vision.wpk: no tensor {n}"))),
        None if layers == 0 => Err(Error::Format("vision.wpk: no layers".into())),
        None => Ok(()),
    }
}

/// RoPE's cosines and sines for each position and each of a half head's 16 frequencies, `θ^(−j/16)` (both
/// computed in `f32`, as the model does): `f32 [positions, 16, 2]`.
fn rope_table(s: Shape) -> Vec<u8> {
    let freq: Vec<f32> = (0..16).map(|j| 1.0 / s.theta.powf(j as f32 / 16.0)).collect();
    (0..s.positions)
        .flat_map(|p| freq.iter().map(move |&f| f64::from(p as f32 * f)))
        .flat_map(|angle| [angle.cos() as f32, angle.sin() as f32])
        .flat_map(f32::to_le_bytes)
        .collect()
}

/// The weights of `wpk` and the RoPE table, in VRAM.
fn weights(ctx: &Context, wpk: &Wpk, shape: Shape) -> Result<(DevBuf, DevBuf)> {
    let (weights, rope) = (ctx.alloc_pooled(wpk.data_len())?, ctx.alloc(shape.positions * 16 * 8)?);
    let mut loader = Loader::new(ctx)?;
    loader.to_device(wpk, &weights)?;
    loader.upload(&rope_table(shape), &rope)?;
    Ok((weights, rope))
}

/// The activation buffers for `cap` patches, in one allocation.
fn scratch(ctx: &Context, s: Shape, cap: usize) -> Result<(DevBuf, Buffers)> {
    let rows = cap + SLACK;
    let sizes = [4 * cap * s.hidden, 4 * (cap * s.hidden).max((cap / (POOL * POOL)) * s.out), 2 * rows * s.hidden, 2 * rows * (3 * s.hidden).max(s.inter)];
    let buf = ctx.alloc(sizes.iter().sum())?;
    let at = |i: usize| buf.ptr() + sizes[..i].iter().sum::<usize>() as u64;
    let bufs = Buffers { x: at(0), t: at(1), a: at(2), b: at(3) };
    Ok((buf, bufs))
}

/// The image's patches as the tower reads them, `f16 [patches, 3·PATCH²]`: patches row by row, each its pixels row
/// by row, red, green and blue, each value `2·v/255 − 1`.
fn patches(image: &Image) -> Vec<u8> {
    let (gw, gh) = (image.width / PATCH, image.height / PATCH);
    let mut out = Vec::with_capacity(gw * gh * 3 * PATCH * PATCH * 2);
    for (py, px, dy) in (0..gh).flat_map(|py| (0..gw).flat_map(move |px| (0..PATCH).map(move |dy| (py, px, dy)))) {
        let row = ((py * PATCH + dy) * image.width + px * PATCH) * 3;
        out.extend(image.rgb[row..row + 3 * PATCH].iter().flat_map(|&v| f16_bits(2.0 * (f32::from(v) / 255.0 - 0.5)).to_le_bytes()));
    }
    out
}
