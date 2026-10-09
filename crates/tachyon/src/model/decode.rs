//! One decode step of Gemma 4 on the GPU, recorded once as a CUDA Graph and replayed per token.
//!
//! A step reads the input token from pinned host memory, embeds it, runs every layer and samples the next token,
//! which the sampler writes back to that same host word: the next replay continues from it with no copy, and the
//! CPU only waits for the step to end. To feed a prompt, the CPU writes each prompt token there before a replay.
//! The sampler's step counter is the position, so attention reads it from the same place.
//!
//! Per layer, each kernel writes what the next one reads, activations as Q8 ([`crate::ptx`] lists them):
//!
//! ```text
//! norm → qkv → attend → o → norm(+residual) → gate_up → geglu → down → norm(+residual)
//!      → per_layer_gate → geglu(per-layer input) → per_layer_proj → norm(+residual, ·layer_scalar, next layer's norm)
//! ```
//!
//! Before the layers, the token's per-layer inputs are its row of the host table plus a projection of its embedding;
//! after them, the LM head (the embedding, tied) and the sampler. Every buffer lives in one arena; the settings that
//! a new sequence resets (step counter, sampling settings, RoPE tables, seen tokens) in one small control buffer,
//! rewritten by one copy.

// Buffers and dimensions keep their one-letter names from the math: x, q, y, d, h, l.
#![allow(clippy::many_single_char_names)]

use super::Config;
use crate::Result;
use crate::cuda::{Context, DevBuf, Event, Graph, HostBuf, Stream, arg};
use crate::ptx::{ATTEND_256, ATTEND_512, ATTEND_WARPS, EMBED_Q4, GEGLU_Q8, GEMV_Q4, GEMV_ROWS, NORM_Q8, NORM_THREADS, SAMPLE, SAMPLE_THREADS, Sampling, rope};
use crate::wpk::{ALIGN, Loader, Wpk};
use std::os::unix::fs::FileExt;
use std::path::Path;

/// A kernel argument.
#[derive(Clone, Copy)]
enum A {
    P(u64),
    U(u32),
    F(f32),
}

/// What [`load`] returns: configuration, `weights.wpk` and `host.wpk`, their memory, the layer scalars.
type Loaded = (Config, [Wpk; 2], DevBuf, HostBuf, Vec<f32>);

/// What [`scratch`] returns: the token word, the control buffer, the arena, and the buffers laid out in them.
type Scratch = (HostBuf, DevBuf, DevBuf, Bufs);

/// A kernel launch: name, grid, threads per block, arguments.
type Op = (&'static str, [u32; 2], u32, Vec<A>);

/// A Q4 matrix: packed weights and scales, rows, columns.
#[derive(Clone, Copy)]
struct Mat {
    w: [u64; 2],
    rows: u32,
    cols: u32,
}

/// How a [`Decoder`] is set up.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Settings {
    /// Positions the global layers' caches hold.
    pub context: usize,
    /// Bytes of the next matrix each small kernel prefetches into L2; `None` is half the GPU's L2.
    pub prefetch: Option<usize>,
}

impl Default for Settings {
    fn default() -> Settings {
        Settings { context: 4096, prefetch: None }
    }
}

/// A model loaded on the GPU, with its decode step recorded.
pub struct Decoder {
    graph: Graph,
    cfg: Config,
    stream: Stream,
    done: Event,
    token: HostBuf,
    control: DevBuf,
    arena: DevBuf,
    logits: u64,
    loader: Loader,
    _keep: (crate::cuda::Module, DevBuf, HostBuf),
}

/// Device addresses of the step's buffers.
struct Bufs {
    x: [u64; 2],
    q: [u64; 2],
    qkv: u64,
    y: u64,
    ple: u64,
    pli: u64,
    proj: u64,
    logits: u64,
    work: u64,
    partial: [u64; 2],
    count: [u64; 2],
    kv: Vec<[u64; 2]>,
    context: u32,
    /// In the control buffer: step counter and token, sampling settings, the two RoPE tables, seen tokens.
    control: [u64; 5],
    token: u64,
}

/// Lays the step's buffers out from `base`, and returns them with the bytes they take.
fn layout(cfg: &Config, context: usize, base: u64) -> (Bufs, usize) {
    let mut end = 0;
    let mut take = |len: usize| {
        let at = end;
        end += len.next_multiple_of(256);
        base + at as u64
    };
    let (d, kv, l, pl, f) = (cfg.head_dim[1], cfg.kv_heads, cfg.layers(), cfg.layers() * cfg.per_layer, 4);
    let widest = cfg.hidden.max(cfg.ffn).max(cfg.heads * d).max(cfg.per_layer);
    let (x, q) = ([take(f * cfg.hidden), take(f * cfg.hidden)], [take(widest), take(widest / 4)]);
    let (qkv, y) = (take(f * (cfg.heads + 2 * kv) * d), take(f * (2 * cfg.ffn).max(cfg.hidden).max(pl)));
    let (ple, pli, proj, logits, work) = (take(f * pl), take(f * pl), take(f * pl), take(f * cfg.vocab), take(f * cfg.vocab));
    let (partial, count) = ([take(f * kv * 256 * (8 + 4 * d)), take(f * 8 * 1024)], [take(f * kv), take(f)]);
    let caches: Vec<[u64; 2]> = (0..l)
        .map(|i| {
            let len = 2 * kv * if cfg.global[i] { context } else { cfg.window } * cfg.head_dim[usize::from(cfg.global[i])];
            if cfg.source(i) == i { [take(len), take(len)] } else { [0, 0] }
        })
        .collect();
    let kv = (0..l).map(|i| caches[cfg.source(i)]).collect();
    let bufs = Bufs { x, q, qkv, y, ple, pli, proj, logits, work, partial, count, kv, context: context as u32, control: [0; 5], token: 0 };
    (bufs, end)
}

/// The control buffer's contents for a new sequence: step counter and token, `sampling`, the RoPE tables, and no
/// token seen; and where each starts.
fn control(cfg: &Config, sampling: &Sampling) -> (Vec<u8>, [usize; 5]) {
    let mut out = vec![0; 64];
    let mut at = [0; 5];
    out.extend(sampling.bytes());
    at[1] = 64;
    for (i, theta) in cfg.theta.iter().enumerate() {
        out.resize(out.len().next_multiple_of(64), 0);
        at[2 + i] = out.len();
        out.extend(rope(*theta, cfg.head_dim[i], if i == 0 { 1.0 } else { cfg.fraction }).iter().flat_map(|v| v.to_ne_bytes()));
    }
    out.resize(out.len().next_multiple_of(64), 0);
    at[4] = out.len();
    out.resize(at[4] + cfg.vocab / 8, 0);
    (out, at)
}

/// The tensors of a `.wpk` loaded at device address `base`.
struct Weights<'a> {
    wpk: &'a Wpk,
    base: u64,
}

impl Weights<'_> {
    /// Address of tensor `name`, or 0 when the model has none.
    fn at(&self, name: &str) -> u64 {
        self.wpk.tensor(name).map_or(0, |t| self.base + t.offset)
    }

    /// Q4 matrix `name`.
    fn mat(&self, name: &str) -> Mat {
        let shape = self.wpk.tensor(&format!("{name}.w")).map_or(&[0, 0][..], |t| &t.shape);
        Mat { w: [self.at(&format!("{name}.w")), self.at(&format!("{name}.s"))], rows: shape[0] as u32, cols: 2 * shape[1] as u32 }
    }
}

impl Decoder {
    /// Loads the model directory `dir` (see [`super::convert`]) onto `ctx`'s GPU, with room for `context` positions.
    pub fn open(ctx: &Context, dir: &Path, settings: &Settings) -> Result<Decoder> {
        let context = settings.context;
        let (stream, module, mut loader) = (ctx.stream()?, ctx.load(&crate::ptx::module())?, Loader::new(ctx)?);
        let (cfg, [wpk, hpk], weights, host, scalars) = load(ctx, dir, &mut loader)?;
        let (token, ctl, arena, b) = scratch(ctx, &stream, &cfg, context)?;
        let (w, hw) = (Weights { wpk: &wpk, base: weights.ptr() }, Weights { wpk: &hpk, base: host.device_ptr()? });
        let mut ops = step(&cfg, [&w, &hw], &b, &scalars, ctx.info().sms);
        prefetch(&mut ops, settings.prefetch.unwrap_or(ctx.info().l2 as usize / 2) as u64);
        let graph = stream.capture(|s| ops.iter().try_for_each(|op| launch(s, &module, op)))?;
        let done = ctx.event(false)?;
        let mut dec = Decoder { graph, cfg, stream, done, token, control: ctl, arena, logits: b.logits, loader, _keep: (module, weights, host) };
        dec.reset(&Sampling::default())?;
        Ok(dec)
    }

    /// The model's configuration.
    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Starts a new sequence at position 0, with `sampling` (its soft cap replaced by the model's) and nothing seen.
    pub fn reset(&mut self, sampling: &Sampling) -> Result<()> {
        self.loader.upload(&control(&self.cfg, &Sampling { softcap: self.cfg.softcap, ..*sampling }).0, &self.control)
    }

    /// Runs one step on `token` at the next position and returns the token sampled after it.
    pub fn step(&mut self, token: u32) -> Result<u32> {
        self.token.copy_from_slice(&token.to_ne_bytes());
        // SAFETY: the graph reads and writes only buffers this decoder owns, laid out for it when it was recorded.
        unsafe { self.stream.replay(&self.graph) }?;
        self.stream.record(&self.done)?;
        self.done.sync()?;
        Ok(u32::from_ne_bytes([self.token[0], self.token[1], self.token[2], self.token[3]]))
    }

    /// The last step's logits, before soft-capping.
    pub fn logits(&self) -> Result<Vec<f32>> {
        let mut out = vec![0; 4 * self.cfg.vocab];
        // SAFETY: `out` outlives the copy, which `sync` completes; the logits lie inside the arena.
        unsafe { self.stream.download(&mut out, &self.arena, (self.logits - self.arena.ptr()) as usize) }?;
        self.stream.sync()?;
        Ok(out.chunks_exact(4).map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]])).collect())
    }
}

/// The model's configuration and files, its weights in VRAM and its host table in pinned memory, and each layer's
/// `layer_scalar`.
fn load(ctx: &Context, dir: &Path, loader: &mut Loader) -> Result<Loaded> {
    let cfg = Config::from_json(&std::fs::read(dir.join("config.json"))?)?;
    let (wpk, hpk) = (Wpk::open(dir.join("weights.wpk"))?, Wpk::open(dir.join("host.wpk"))?);
    let (weights, mut host) = (ctx.alloc(wpk.data_len())?, ctx.alloc_host(hpk.data_len())?);
    loader.to_device(&wpk, &weights)?;
    loader.to_host(&hpk, &mut host)?;
    let scalars = scalars(&dir.join("weights.wpk"), &wpk, cfg.layers())?;
    Ok((cfg, [wpk, hpk], weights, host, scalars))
}

/// The token word, the control buffer and the arena, and the step's buffers in them.
fn scratch(ctx: &Context, stream: &Stream, cfg: &Config, context: usize) -> Result<Scratch> {
    let (ctl, at) = control(cfg, &Sampling::default());
    let (token, ctl, arena) = (ctx.alloc_host(4)?, ctx.alloc(ctl.len())?, ctx.alloc(layout(cfg, context, 0).1)?);
    stream.fill(&arena, 0)?;
    let (mut b, _) = layout(cfg, context, arena.ptr());
    (b.control, b.token) = (at.map(|at| ctl.ptr() + at as u64), token.device_ptr()?);
    Ok((token, ctl, arena, b))
}

/// Gives every kernel that can prefetch (see [`crate::ptx`]) the first `budget` bytes of the next matrix's weights to
/// fetch into L2 while it runs, so VRAM keeps streaming across the kernel boundaries of a step.
fn prefetch(ops: &mut [Op], budget: u64) {
    let mut next = [0, 0];
    for (name, _, _, args) in ops.iter_mut().rev() {
        match (*name, args.first(), args.get(5..7)) {
            (GEMV_Q4, Some(&A::P(w)), Some(&[A::U(rows), A::U(cols)])) => next = [w, (u64::from(rows) * u64::from(cols) / 2).min(budget)],
            (NORM_Q8 | GEGLU_Q8 | ATTEND_256 | ATTEND_512, ..) => args.extend(next.map(A::P)),
            _ => {}
        }
    }
}

/// Each layer's `layer_scalar`, read from the file.
fn scalars(path: &Path, wpk: &Wpk, layers: usize) -> Result<Vec<f32>> {
    let file = std::fs::File::open(path)?;
    (0..layers)
        .map(|i| {
            let mut v = [0; 4];
            let at = wpk.tensor(&format!("{i}.layer_scalar")).map_or(0, |t| t.offset);
            file.read_exact_at(&mut v, ALIGN + at)?;
            Ok(f32::from_le_bytes(v))
        })
        .collect()
}

/// Records `op` on the capturing stream `s`.
fn launch(s: &Stream, m: &crate::cuda::Module, (name, grid, threads, args): &Op) -> Result<()> {
    let ptrs: Vec<_> = args
        .iter()
        .map(|a| match a {
            A::P(v) => arg(v),
            A::U(v) => arg(v),
            A::F(v) => arg(v),
        })
        .collect();
    // SAFETY: every op passes its kernel's arguments in order and with their types, as `crate::ptx` documents them,
    // and every buffer was laid out for the model by `layout`.
    unsafe { s.launch(&m.function(name)?, [grid[0], grid[1], 1], [*threads, 1, 1], 0, &ptrs) }
}

/// `y = M·x`, reading the Q8 input `q`.
fn gemv(m: Mat, q: [u64; 2], y: u64) -> Op {
    let args = vec![A::P(m.w[0]), A::P(m.w[1]), A::P(q[0]), A::P(q[1]), A::P(y), A::U(m.rows), A::U(m.cols)];
    (GEMV_Q4, [m.rows.div_ceil(GEMV_ROWS), 1], 32 * GEMV_ROWS, args)
}

/// [`NORM_Q8`] on `p = [h, h_out, y, w1, w2]` over `shape = [len, chunks]` with `k = [eps, scale]`:
/// `h_out = (h + w1 ⊙ rmsnorm(y)) · scale`, then Q8 of `w2 ⊙ rmsnorm(h_out)`.
fn norm(p: [u64; 5], q: [u64; 2], shape: [usize; 2], k: [f32; 2]) -> Op {
    let args = p.into_iter().chain(q).map(A::P).chain([A::U(shape[0] as u32), A::F(k[0]), A::F(k[1])]).collect();
    (NORM_Q8, [shape[0] as u32 / NORM_THREADS, shape[1] as u32], NORM_THREADS, args)
}

/// Q8 of `gelu(a) · b` over `len` values.
fn geglu(a: u64, b: u64, q: [u64; 2], len: usize) -> Op {
    (GEGLU_Q8, [(len as u32).div_ceil(256), 1], 256, vec![A::P(a), A::P(b), A::P(q[0]), A::P(q[1]), A::U(len as u32)])
}

/// Row `*token` of Q4 table `m`, times `scale`, into `out`.
fn embed(m: Mat, token: u64, out: u64, scale: f32) -> Op {
    (EMBED_Q4, [m.cols.div_ceil(256), 1], 256, vec![A::P(m.w[0]), A::P(m.w[1]), A::P(token), A::P(out), A::U(m.cols), A::F(scale)])
}

/// The whole step.
fn step(cfg: &Config, [w, hw]: [&Weights; 2], b: &Bufs, scalars: &[f32], sms: u32) -> Vec<Op> {
    let (h, pl, l, eps) = (cfg.hidden, cfg.per_layer, cfg.layers(), cfg.eps);
    let mut ops = vec![
        embed(w.mat("embed"), b.token, b.x[0], bf16((h as f32).sqrt())),
        embed(hw.mat("embed_per_layer"), b.token, b.ple, (pl as f32).sqrt()),
        norm([b.x[0], 0, 0, 0, 0], b.q, [h, 1], [eps, 1.0]),
        gemv(w.mat("per_layer_proj"), b.q, b.proj),
        // rmsnorm(c·y) is rmsnorm(y) with eps / c²: the projection's 1/√hidden scale folds into eps.
        norm([b.ple, b.pli, b.proj, w.at("per_layer_norm"), 0], [0, 0], [pl, l], [eps * h as f32, std::f32::consts::FRAC_1_SQRT_2]),
        norm([b.x[0], 0, 0, 0, w.at("0.input_layernorm")], b.q, [h, 1], [eps, 1.0]),
    ];
    for (i, &scalar) in scalars.iter().enumerate() {
        ops.extend(attention(cfg, w, b, i, sms));
        ops.extend(mlp(cfg, w, b, i, scalar));
    }
    ops.push(gemv(w.mat("embed"), b.q, b.logits));
    let blocks = sms.min((cfg.vocab as u32).div_ceil(4 * SAMPLE_THREADS));
    let [state, sampling, .., seen] = b.control;
    let args = [b.logits, b.work].map(A::P).into_iter().chain([A::U(cfg.vocab as u32)]);
    ops.push((SAMPLE, [blocks, 1], SAMPLE_THREADS, args.chain([sampling, state, seen, b.token, b.partial[1], b.count[1]].map(A::P)).collect()));
    ops
}

/// Layer `i`'s attention, from its normalized input in `q` to the residual stream in `x[(i + 1) % 2]` and the
/// MLP's normalized input in `q`.
fn attention(cfg: &Config, w: &Weights, b: &Bufs, i: usize, sms: u32) -> Vec<Op> {
    let (g, n) = (usize::from(cfg.global[i]), |name: &str| w.at(&format!("{i}.{name}")));
    let (d, kv, fresh) = (cfg.head_dim[g], cfg.kv_heads as u32, cfg.source(i) == i);
    let (cap, splits) = if g == 1 { (b.context, sms / kv) } else { (cfg.window as u32, 2 * sms / kv) };
    let ptrs = [b.qkv, n("q_norm"), if fresh { n("k_norm") } else { 0 }, b.control[2 + g], b.kv[i][0], b.kv[i][1], b.control[0]];
    let args = ptrs.map(A::P).into_iter().chain([A::U(cap), A::U(u32::from(fresh))]).chain([b.partial[0], b.count[0], b.q[0], b.q[1]].map(A::P));
    let (x0, x1) = (b.x[i % 2], b.x[(i + 1) % 2]);
    vec![
        gemv(w.mat(&format!("{i}.qkv")), b.q, b.qkv),
        (if d == 256 { ATTEND_256 } else { ATTEND_512 }, [kv, splits.clamp(1, 256)], 32 * ATTEND_WARPS, args.collect()),
        gemv(w.mat(&format!("{i}.o")), b.q, b.y),
        norm([x0, x1, b.y, n("post_attention_layernorm"), n("pre_feedforward_layernorm")], b.q, [cfg.hidden, 1], [cfg.eps, 1.0]),
    ]
}

/// Layer `i`'s MLP and per-layer input, from the residual stream in `x[(i + 1) % 2]` back to it, times `scalar`,
/// with the next layer's normalized input (or the final norm's) in `q`.
fn mlp(cfg: &Config, w: &Weights, b: &Bufs, i: usize, scalar: f32) -> Vec<Op> {
    let (h, eps, n) = (cfg.hidden, cfg.eps, |name: &str| w.at(&format!("{i}.{name}")));
    let (x0, x1) = (b.x[i % 2], b.x[(i + 1) % 2]);
    let next = if i + 1 < cfg.layers() { w.at(&format!("{}.input_layernorm", i + 1)) } else { w.at("norm") };
    vec![
        gemv(w.mat(&format!("{i}.gate_up")), b.q, b.y),
        geglu(b.y, b.y + 4 * cfg.ffn as u64, b.q, cfg.ffn),
        gemv(w.mat(&format!("{i}.down")), b.q, b.y),
        norm([x1, x0, b.y, n("post_feedforward_layernorm"), 0], b.q, [h, 1], [eps, 1.0]),
        gemv(w.mat(&format!("{i}.per_layer_gate")), b.q, b.y),
        geglu(b.y, b.pli + 4 * (i * cfg.per_layer) as u64, b.q, cfg.per_layer),
        gemv(w.mat(&format!("{i}.per_layer_proj")), b.q, b.y),
        norm([x0, x1, b.y, n("post_per_layer_input_norm"), next], b.q, [h, 1], [eps, scalar]),
    ]
}

/// `v` rounded to bfloat16, as Gemma's embedding scale is applied.
fn bf16(v: f32) -> f32 {
    let b = v.to_bits();
    f32::from_bits((b + 0x7fff + (b >> 16 & 1)) & 0xffff_0000)
}