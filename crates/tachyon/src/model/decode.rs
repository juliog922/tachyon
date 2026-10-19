//! One decode step of Gemma 4 on the GPU, recorded once as a CUDA Graph and replayed per token.
//!
//! A step reads the input token from a word of pinned host memory, embeds it, runs every layer and samples the next
//! token, which the sampler writes to a second word. The step is recorded twice with the words swapped, and steps
//! alternate between the recordings: each continues from the token the last one wrote, with no copy, so the next
//! step is queued while one runs and the GPU never waits for the CPU, which reads each token from its word. To feed
//! a prompt, the CPU writes each prompt token before a step. The sampler's step counter is the position, so
//! attention reads it from the same place.
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
use super::prefill::Chunk;
use crate::Result;
use crate::cuda::{Context, DevBuf, Event, Graph, HostBuf, Stream, arg};
use crate::ptx::{
    ATTEND_256, ATTEND_256_PAIRS, ATTEND_512, ATTEND_WARPS, EMBED_Q4, GEGLU_Q8, GEGLU_Q8S, GEMV_Q4, GEMV_Q4_MOE, GEMV_Q4_NARROW, GEMV_ROWS, NORM_Q8,
    NORM_THREADS, ROUTE_TOPK, SAMPLE, SAMPLE_THREADS, SPIN, SUM_SLICES, Sampling, rope,
};
use crate::wpk::{ALIGN, Loader, Wpk};
use std::os::unix::fs::FileExt;
use std::path::Path;

/// A kernel argument.
#[derive(Clone, Copy)]
pub(super) enum A {
    P(u64),
    U(u32),
    F(f32),
}

/// A loaded model: configuration, `weights.wpk` and `host.wpk`, their memory, the layer scalars.
type Loaded<W> = (Config, [Wpk; 2], W, HostBuf, Vec<f32>);

/// What [`scratch`] returns: the token words and prefill's IDs, the control buffer, the arena, and the buffers laid
/// out in them.
type Scratch = ([HostBuf; 2], DevBuf, DevBuf, Bufs, Chunk);

/// A kernel launch: name, grid, threads per block, arguments.
pub(super) type Op = (&'static str, [u32; 3], u32, Vec<A>);

/// A Q4 matrix: packed weights and scales, rows, columns.
#[derive(Clone, Copy)]
pub(super) struct Mat {
    pub(super) w: [u64; 2],
    pub(super) rows: u32,
    pub(super) cols: u32,
}

/// How a [`Decoder`] is set up.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Settings {
    /// Positions the global layers' caches hold.
    pub context: usize,
    /// Prompt tokens [`Decoder::prefill`] runs at once; sliding layers' caches hold this many past the window.
    pub chunk: usize,
}

impl Default for Settings {
    fn default() -> Settings {
        Settings { context: 4096, chunk: 512 }
    }
}

/// A model loaded on the GPU, with its decode step recorded.
pub struct Decoder {
    graphs: Vec<Graph>,
    cfg: Config,
    stream: Stream,
    done: Vec<Event>,
    /// The recording [`Decoder::next_token`] waits for.
    at: usize,
    tokens: HostBuf,
    /// Position of the next step, as the step counter on the GPU holds it.
    pos: u32,
    control: DevBuf,
    arena: DevBuf,
    logits: u64,
    loader: Loader,
    ops: Vec<Op>,
    module: crate::cuda::Module,
    pre: Pre,
    mem: (DevBuf, HostBuf),
    /// The weights' copy in pinned memory a resumed model keeps, so parking it again copies nothing.
    shadow: Option<HostBuf>,
}

/// A model in the host-RAM tier ([`Decoder::park`]): its weights in pinned memory, ready to return to VRAM at full
/// PCIe speed with [`Parked::resume`], its kernels still compiled.
pub struct Parked {
    model: Loaded<HostBuf>,
    module: crate::cuda::Module,
    loader: Loader,
}

/// What prefill builds its launches from: the weight files and where they were loaded, each layer's scalar, the
/// step's buffers, prefill's buffers, its token IDs in pinned memory, and its chunk size.
struct Pre {
    files: [Wpk; 2],
    bases: [u64; 2],
    scalars: Vec<f32>,
    bufs: Bufs,
    chunk: Chunk,
    ids: HostBuf,
    size: usize,
    /// Pinned: the chunk's soft tokens, `f32 [chunk, hidden]` at their positions in it, and the row numbers
    /// `0..chunk` they are gathered by.
    soft: [HostBuf; 2],
}

/// Device addresses of the step's buffers.
pub(super) struct Bufs {
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
    pub(super) kv: Vec<[u64; 2]>,
    /// The mixture of experts' buffers: zeros, two sums, the router's logits, the chosen experts and their weights,
    /// their gate and up, and their weighted outputs (a slice each).
    moe: [u64; 8],
    pub(super) context: u32,
    /// Slots of a sliding layer's cache: the window and a prefill chunk.
    pub(super) ring: u32,
    /// In the control buffer: step counter and token, sampling settings, the two RoPE tables, seen tokens.
    pub(super) control: [u64; 5],
    /// The token a step reads, and the one it writes: two words of pinned host memory.
    token: [u64; 2],
}

/// Lays the step's buffers out from `base`, and returns them with the bytes they take.
fn layout(cfg: &Config, settings: &Settings, base: u64) -> (Bufs, Chunk, usize) {
    let (context, ring) = (settings.context, cfg.window + settings.chunk);
    let mut end = 0;
    let mut take = |len: usize| {
        let at = end;
        end += len.next_multiple_of(256);
        base + at as u64
    };
    let (d, l, pl, f) = (cfg.head_dim[1], cfg.layers(), cfg.layers() * cfg.per_layer, 4);
    let widest = cfg.hidden.max(cfg.mlp(l - 1)).max(cfg.heads * d).max(cfg.per_layer);
    let (x, q) = ([take(f * cfg.hidden), take(f * cfg.hidden)], [take(widest), take(widest / 4)]);
    let (qkv, y) = (take(f * qkv_width(cfg)), take(f * (2 * cfg.mlp(l - 1)).max(cfg.hidden).max(pl)));
    let (ple, pli, proj, logits, work) = (take(f * pl), take(f * pl), take(f * pl), take(f * cfg.vocab), take(f * cfg.vocab));
    // Attention's scratch: per block (at most heads / 2) and split (at most 256), the block's maxima, sums and outputs.
    let partial = [take(f * cfg.heads / 2 * 256 * (8 + 4 * d)), take(f * 8 * 1024)];
    let count = [take(f * cfg.heads / 2), take(f)];
    let [e, k, w] = cfg.experts;
    let (hm, ke) = (if cfg.moe() { cfg.hidden } else { 0 }, k * 2 * w);
    let moe = [take(f * hm), take(f * hm), take(f * hm), take(f * e), take(f * k), take(f * k), take(f * ke), take(f * k * hm)];
    let caches: Vec<[u64; 2]> = (0..l)
        .map(|i| {
            let len = 2 * cfg.kv(i) * if cfg.global[i] { context } else { ring } * cfg.head_dim[usize::from(cfg.global[i])];
            if cfg.source(i) == i { [take(len), take(len)] } else { [0, 0] }
        })
        .collect();
    let kv = (0..l).map(|i| caches[cfg.source(i)]).collect();
    let chunk = super::prefill::layout(cfg, settings.chunk, &mut take);
    let (context, ring) = (context as u32, ring as u32);
    let bufs = Bufs { x, q, qkv, y, ple, pli, proj, logits, work, partial, count, kv, moe, context, ring, control: [0; 5], token: [0; 2] };
    (bufs, chunk, end)
}

/// Floats of one token's query, key and value projections, in the widest kind of layer.
pub(super) fn qkv_width(cfg: &Config) -> usize {
    (0..2).map(|g| (cfg.heads + 2 * cfg.kv_heads[g]) * cfg.head_dim[g]).max().unwrap_or(0)
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
pub(super) struct Weights<'a> {
    wpk: &'a Wpk,
    base: u64,
}

impl Weights<'_> {
    /// Address of tensor `name`, or 0 when the model has none.
    pub(super) fn at(&self, name: &str) -> u64 {
        self.wpk.tensor(name).map_or(0, |t| self.base + t.offset)
    }

    /// Q4 matrix `name`.
    pub(super) fn mat(&self, name: &str) -> Mat {
        let shape = self.wpk.tensor(&format!("{name}.w")).map_or(&[0, 0][..], |t| &t.shape);
        Mat { w: [self.at(&format!("{name}.w")), self.at(&format!("{name}.s"))], rows: shape[0] as u32, cols: 2 * shape[1] as u32 }
    }
}

impl Decoder {
    /// Loads the model directory `dir` (see [`super::convert`]) onto `ctx`'s GPU, with room for `context` positions.
    pub fn open(ctx: &Context, dir: &Path, settings: &Settings) -> Result<Decoder> {
        let mut loader = Loader::new(ctx)?;
        let model = load(ctx, dir, &mut loader)?;
        Decoder::build(ctx.load(&crate::ptx::module())?, loader, model, settings)
    }

    /// Moves the model to the host-RAM tier: its weights into pinned memory (unless it kept them there, see
    /// [`Decoder::shadow_bytes`]), its VRAM freed. A generation still running stops first.
    pub fn park(mut self) -> Result<Parked> {
        self.stop()?;
        let weights = if let Some(kept) = self.shadow.take() {
            kept
        } else {
            let mut weights = self.stream.context().alloc_host(self.mem.0.len())?;
            // SAFETY: `weights` outlives the copy, which `sync` completes.
            unsafe { self.stream.download(&mut weights, &self.mem.0, 0) }?;
            self.stream.sync()?;
            weights
        };
        Ok(Parked { model: (self.cfg, self.pre.files, weights, self.mem.1, self.pre.scalars), module: self.module, loader: self.loader })
    }

    /// The decoder of `model` on `module`'s GPU: its buffers laid out and its step recorded.
    fn build(module: crate::cuda::Module, loader: Loader, model: Loaded<DevBuf>, settings: &Settings) -> Result<Decoder> {
        let (cfg, [wpk, hpk], weights, host, scalars) = model;
        let (ctx, mut steps) = (module.context().clone(), Vec::new());
        let stream = ctx.stream()?;
        let ([tokens, ids], ctl, arena, mut b, chunk) = scratch(&ctx, &stream, &cfg, settings)?;
        let bases = [weights.ptr(), host.device_ptr()?];
        let (w, hw) = (Weights { wpk: &wpk, base: bases[0] }, Weights { wpk: &hpk, base: bases[1] });
        // Two recordings with the token words traded: steps alternate between them, so one can be queued while the
        // other runs and the CPU reads each token from its own word.
        for _ in 0..2 {
            steps.push(step(&cfg, [&w, &hw], &b, &scalars, ctx.info().sms));
            b.token.reverse();
        }
        let (graphs, done) = record(&stream, &module, &steps)?;
        let (ops, logits) = (steps.swap_remove(0), b.logits);
        let mut soft = [ctx.alloc_host(4 * settings.chunk * cfg.hidden)?, ctx.alloc_host(4 * settings.chunk)?];
        (0..settings.chunk as u32).for_each(|i| soft[1][4 * i as usize..][..4].copy_from_slice(&i.to_ne_bytes()));
        let pre = Pre { files: [wpk, hpk], bases, scalars, bufs: b, chunk, ids, size: settings.chunk, soft };
        let mem = (weights, host);
        let mut dec = Decoder { graphs, cfg, stream, done, at: 0, tokens, pos: 0, control: ctl, arena, logits, loader, ops, module, pre, mem, shadow: None };
        dec.reset(&Sampling::default())?;
        Ok(dec)
    }

    /// Pinned host memory [`Decoder::park`] leaves it in: its weights and its host table.
    pub fn parked_bytes(&self) -> usize {
        self.mem.0.len() + self.mem.1.len()
    }

    /// Pinned memory held by the copy of its weights a model resumed from the host tier keeps: parking it again
    /// copies nothing. [`Decoder::drop_shadow`] frees it.
    pub fn shadow_bytes(&self) -> usize {
        self.shadow.as_ref().map_or(0, |s| s.len())
    }

    /// Frees the copy [`Decoder::shadow_bytes`] counts.
    pub fn drop_shadow(&mut self) {
        self.shadow = None;
    }

    /// The model's configuration.
    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Starts a new sequence at position 0, with `sampling` (its soft cap replaced by the model's) and nothing seen;
    /// a generation still running stops first.
    pub fn reset(&mut self, sampling: &Sampling) -> Result<()> {
        self.stop()?;
        self.pos = 0;
        self.loader.upload(&control(&self.cfg, &Sampling { softcap: self.cfg.softcap, ..*sampling }).0, &self.control)
    }

    /// Runs one step on `token` at the next position and returns the token sampled after it. Not while a generation
    /// runs: [`Decoder::stop`] it first.
    pub fn step(&mut self, token: u32) -> Result<u32> {
        self.tokens[4..8].copy_from_slice(&token.to_ne_bytes());
        self.queue(0)?;
        self.done[0].sync()?;
        Ok(self.token(0))
    }

    /// Starts generating from `token` at the next position: queues its step, which [`Decoder::next_token`] waits for.
    pub fn start(&mut self, token: u32) -> Result<()> {
        self.tokens[4..8].copy_from_slice(&token.to_ne_bytes());
        self.at = 0;
        self.queue(0)
    }

    /// The token the running step samples. Queues the step on it first, so the GPU never waits for the CPU: one
    /// step is always queued behind the running one.
    pub fn next_token(&mut self) -> Result<u32> {
        let g = self.at;
        self.queue(1 - g)?;
        self.done[g].sync()?;
        self.at = 1 - g;
        Ok(self.token(g))
    }

    /// Ends a generation: waits for the step still queued, which feeds the model the last token [`Decoder::next_token`]
    /// returned (its own sample is dropped), so the sequence continues after that token.
    pub fn stop(&mut self) -> Result<()> {
        self.stream.sync()
    }

    /// Feeds `tokens` at the next positions through prefill, [`Settings::chunk`] at a time: each kernel runs over
    /// every token of a chunk, the matrix products on tensor cores, and only the layers that write a key-value cache
    /// run (a later step needs nothing else from these tokens). Nothing is sampled: a prompt's last token goes
    /// through [`Decoder::step`]. Not while a generation runs.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<()> {
        self.prefill_with(tokens, &[])
    }

    /// [`Decoder::prefill`] with soft tokens: each of `soft` is a position in `tokens` and rows (`f32 [n, hidden]`,
    /// an image's soft tokens) that take the place of the `n` token embeddings from there. Those positions read as
    /// the pad token for the per-layer inputs, as Gemma 4 has it; the per-layer projection sees the soft tokens.
    pub fn prefill_with(&mut self, tokens: &[u32], soft: &[(usize, &[f32])]) -> Result<()> {
        for (k, part) in tokens.chunks(self.pre.size).enumerate() {
            let runs = self.stage(k * self.pre.size, part, soft);
            let p = &self.pre;
            let (w, hw) = (Weights { wpk: &p.files[0], base: p.bases[0] }, Weights { wpk: &p.files[1], base: p.bases[1] });
            let mut ops = super::prefill::ops(&self.cfg, [&w, &hw], &p.bufs, (&p.chunk, &p.scalars), part.len() as u32);
            let soft = [p.soft[0].device_ptr()?, p.soft[1].device_ptr()?];
            ops.splice(1..1, runs.iter().map(|&run| super::prefill::scatter(&self.cfg, &p.chunk, soft, run)));
            for op in &ops {
                launch(&self.stream, &self.module, op)?;
            }
            self.pos += part.len() as u32;
            self.tokens[8..12].copy_from_slice(&self.pos.to_ne_bytes());
            // SAFETY: the word is pinned memory this decoder owns, and `sync` completes the copy before it changes.
            unsafe { self.stream.upload(&self.control, 0, &self.tokens[8..12]) }?;
            self.stream.sync()?;
        }
        Ok(())
    }

    /// Writes the chunk `part` of the prompt, which starts at prompt position `start`, into the pinned IDs, and the
    /// soft tokens it holds into the pinned rows, their IDs the pad token's; returns their runs (first row, rows).
    fn stage(&mut self, start: usize, part: &[u32], soft: &[(usize, &[f32])]) -> Vec<(u32, u32)> {
        let (h, ids, rows) = (self.cfg.hidden, &mut self.pre.ids, &mut self.pre.soft[0]);
        part.iter().enumerate().for_each(|(i, t)| ids[4 * i..4 * i + 4].copy_from_slice(&t.to_ne_bytes()));
        let mut runs = Vec::new();
        for &(at, src) in soft {
            let (a, b) = (at.max(start), (at + src.len() / h).min(start + part.len()));
            if a < b {
                ids[4 * (a - start)..4 * (b - start)].fill(0);
                let bytes = src[(a - at) * h..(b - at) * h].iter().flat_map(|v| v.to_ne_bytes());
                rows[4 * h * (a - start)..4 * h * (b - start)].iter_mut().zip(bytes).for_each(|(d, s)| *d = s);
                runs.push(((a - start) as u32, (b - a) as u32));
            }
        }
        runs
    }

    /// Queues recording `g`, which writes token word `g`, and marks its end.
    fn queue(&mut self, g: usize) -> Result<()> {
        self.pos += 1;
        // SAFETY: the graphs read and write only buffers this decoder owns, laid out for them when they were recorded.
        unsafe { self.stream.replay(&self.graphs[g]) }?;
        self.stream.record(&self.done[g])
    }

    /// Token word `g`.
    fn token(&self, g: usize) -> u32 {
        u32::from_ne_bytes([self.tokens[4 * g], self.tokens[4 * g + 1], self.tokens[4 * g + 2], self.tokens[4 * g + 3]])
    }

    /// Every kind of kernel in the step (its name; for a matrix product with its shape), how many run, and the time
    /// of a graph of only those kernels, as the step records them (the mean of 20 replays): what each kind costs
    /// alone, with no other kernel and no CPU in the way.
    pub fn alone(&self) -> Result<Vec<(String, usize, std::time::Duration)>> {
        let mut kinds: Vec<(String, usize, std::time::Duration)> = Vec::new();
        for op in &self.ops {
            match kinds.iter_mut().find(|k| k.0 == kind(op)) {
                Some(k) => k.1 += 1,
                None => kinds.push((kind(op), 1, std::time::Duration::ZERO)),
            }
        }
        let [start, end] = [self.stream.context().event(true)?, self.stream.context().event(true)?];
        for (name, _, t) in &mut kinds {
            let ops: Vec<Op> = self.ops.iter().filter(|op| kind(op) == *name).cloned().collect();
            let graph = self.stream.capture(|s| ops.iter().try_for_each(|op| launch(s, &self.module, op)))?;
            // SAFETY: as in `step`: the graph uses only this decoder's buffers.
            let replay = |n| (0..n).try_for_each(|_| unsafe { self.stream.replay(&graph) });
            replay(3).and_then(|()| self.stream.record(&start)).and_then(|()| replay(20)).and_then(|()| self.stream.record(&end))?;
            end.sync()?;
            *t = std::time::Duration::from_secs_f64(f64::from(end.since(&start)?) / 2e4);
        }
        Ok(kinds)
    }

    /// Runs one step on `token` kernel by kernel, each timed on the GPU, and returns every kernel's kind (its name;
    /// for a matrix product with its shape) and time, in order. A [`SPIN`] kernel first holds the GPU while the CPU
    /// queues the whole step, so the kernels run back to back, as in the recorded step, each with its launch gap.
    pub fn timeline(&mut self, token: u32) -> Result<Vec<(String, std::time::Duration)>> {
        self.tokens[4..8].copy_from_slice(&token.to_ne_bytes());
        let marks = (0..=self.ops.len()).map(|_| self.stream.context().event(true)).collect::<Result<Vec<_>>>()?;
        launch(&self.stream, &self.module, &(SPIN, [1, 1, 1], 32, vec![A::P(50_000_000)]))?;
        self.stream.record(&marks[0])?;
        for (op, mark) in self.ops.iter().zip(&marks[1..]) {
            launch(&self.stream, &self.module, op)?;
            self.stream.record(mark)?;
        }
        marks[self.ops.len()].sync()?;
        let time = |m: &[Event]| m[1].since(&m[0]).map(|ms| std::time::Duration::from_secs_f64(f64::from(ms) / 1e3));
        self.ops.iter().zip(marks.windows(2)).map(|(op, m)| Ok((kind(op), time(m)?))).collect()
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

impl Parked {
    /// Brings the model back to VRAM with `settings`: one copy of its weights at full PCIe speed, while its buffers
    /// are laid out and its step recorded, as [`Decoder::open`] does. The pinned copy stays with the decoder (see
    /// [`Decoder::shadow_bytes`]): freeing pinned memory costs about as much as the copy.
    pub fn resume(self, settings: &Settings) -> Result<Decoder> {
        let (cfg, files, parked, host, scalars) = self.model;
        let ctx = self.module.context().clone();
        let (weights, copy) = (ctx.alloc_pooled(parked.len())?, ctx.stream()?);
        // SAFETY: `parked` stays alive and unwritten until `sync` below completes the copy, whatever `build` returns.
        unsafe { copy.upload(&weights, 0, &parked) }?;
        let built = Decoder::build(self.module, self.loader, (cfg, files, weights, host, scalars), settings);
        copy.sync()?;
        let mut dec = built?;
        dec.shadow = Some(parked);
        Ok(dec)
    }

    /// Pinned host memory it holds: its weights and its host table.
    pub fn bytes(&self) -> usize {
        self.model.2.len() + self.model.3.len()
    }
}

/// VRAM the model in `dir` takes on a GPU with `settings`: its weights and the buffers of its step, prefill and caches.
pub fn footprint(dir: &Path, settings: &Settings) -> Result<usize> {
    let cfg = Config::from_json(&std::fs::read(dir.join("config.json"))?)?;
    Ok(Wpk::open(dir.join("weights.wpk"))?.data_len() + layout(&cfg, settings, 0).2 + control(&cfg, &Sampling::default()).0.len())
}

/// The model's configuration and files, its weights in VRAM and its host table in pinned memory, and each layer's
/// `layer_scalar`.
fn load(ctx: &Context, dir: &Path, loader: &mut Loader) -> Result<Loaded<DevBuf>> {
    let cfg = Config::from_json(&std::fs::read(dir.join("config.json"))?)?;
    let (wpk, hpk) = (Wpk::open(dir.join("weights.wpk"))?, Wpk::open(dir.join("host.wpk"))?);
    // A model without per-layer inputs has an empty host table.
    let (weights, mut host) = (ctx.alloc_pooled(wpk.data_len())?, ctx.alloc_host(hpk.data_len().max(64))?);
    loader.to_device(&wpk, &weights)?;
    loader.to_host(&hpk, &mut host)?;
    let scalars = scalars(&dir.join("weights.wpk"), &wpk, cfg.layers())?;
    Ok((cfg, [wpk, hpk], weights, host, scalars))
}

/// The token words (in, out, and the position for prefill to set), the control buffer, the arena, the buffers
/// laid out in them, and prefill's buffers with its token IDs in pinned memory.
fn scratch(ctx: &Context, stream: &Stream, cfg: &Config, settings: &Settings) -> Result<Scratch> {
    let (ctl, at) = control(cfg, &Sampling::default());
    let (tokens, ids) = (ctx.alloc_host(12)?, ctx.alloc_host(4 * settings.chunk)?);
    let (ctl, arena) = (ctx.alloc(ctl.len())?, ctx.alloc(layout(cfg, settings, 0).2)?);
    stream.fill(&arena, 0)?;
    let (mut b, mut chunk, _) = layout(cfg, settings, arena.ptr());
    let word = tokens.device_ptr()?;
    (b.control, b.token, chunk.ids, chunk.sms) = (at.map(|at| ctl.ptr() + at as u64), [word + 4, word], ids.device_ptr()?, ctx.info().sms);
    Ok(([tokens, ids], ctl, arena, b, chunk))
}

/// The kind of `op`: its kernel, or for a matrix product its shape.
fn kind(op: &Op) -> String {
    match (op.0, op.3.get(5..7)) {
        (name, Some(&[A::U(rows), A::U(cols)])) if name.starts_with("gemv") => format!("{name} {rows}×{cols}"),
        (name, _) => name.into(),
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

/// Each of `steps` recorded as a graph, and an event per graph to mark its end.
fn record(stream: &Stream, module: &crate::cuda::Module, steps: &[Vec<Op>]) -> Result<(Vec<Graph>, Vec<Event>)> {
    let graphs = steps.iter().map(|ops| stream.capture(|s| ops.iter().try_for_each(|op| launch(s, module, op)))).collect::<Result<_>>()?;
    Ok((graphs, steps.iter().map(|_| stream.context().event(false)).collect::<Result<_>>()?))
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
    unsafe { s.launch(&m.function(name)?, *grid, [*threads, 1, 1], crate::ptx::shared(name), &ptrs) }
}

/// `y = M·x`, reading the Q8 input `q`.
fn gemv(m: Mat, q: [u64; 2], y: u64) -> Op {
    let args = vec![A::P(m.w[0]), A::P(m.w[1]), A::P(q[0]), A::P(q[1]), A::P(y), A::U(m.rows), A::U(m.cols)];
    if m.cols <= 256 { (GEMV_Q4_NARROW, [m.rows.div_ceil(16), 1, 1], 128, args) } else { (GEMV_Q4, [m.rows.div_ceil(GEMV_ROWS), 1, 1], 32 * GEMV_ROWS, args) }
}

/// [`NORM_Q8`] on `p = [h, h_out, y, w1, w2]` over `shape = [len, chunks]` with `k = [eps, scale]`:
/// `h_out = (h + w1 ⊙ rmsnorm(y)) · scale`, then Q8 of `w2 ⊙ rmsnorm(h_out)`.
fn norm(p: [u64; 5], q: [u64; 2], shape: [usize; 2], k: [f32; 2]) -> Op {
    let args = p.into_iter().chain(q).map(A::P).chain([A::U(shape[0] as u32), A::F(k[0]), A::F(k[1])]).collect();
    (NORM_Q8, [shape[0] as u32 / NORM_THREADS, shape[1] as u32, 1], NORM_THREADS, args)
}

/// Q8 of `gelu(a) · b` over `len` values.
fn geglu(a: u64, b: u64, q: [u64; 2], len: usize) -> Op {
    (GEGLU_Q8, [(len as u32).div_ceil(256), 1, 1], 256, vec![A::P(a), A::P(b), A::P(q[0]), A::P(q[1]), A::U(len as u32)])
}

/// Row `*token` of Q4 table `m`, times `scale`, into `out`.
fn embed(m: Mat, token: u64, out: u64, scale: f32) -> Op {
    (EMBED_Q4, [m.cols.div_ceil(256), 1, 1], 256, vec![A::P(m.w[0]), A::P(m.w[1]), A::P(token), A::P(out), A::U(m.cols), A::F(scale)])
}

/// The whole step.
fn step(cfg: &Config, [w, hw]: [&Weights; 2], b: &Bufs, scalars: &[f32], sms: u32) -> Vec<Op> {
    let (h, pl, l, eps) = (cfg.hidden, cfg.per_layer, cfg.layers(), cfg.eps);
    let mut ops = vec![embed(w.mat("embed"), b.token[0], b.x[0], bf16((h as f32).sqrt()))];
    if pl > 0 {
        ops.extend([
            embed(hw.mat("embed_per_layer"), b.token[0], b.ple, (pl as f32).sqrt()),
            norm([b.x[0], 0, 0, 0, 0], b.q, [h, 1], [eps, 1.0]),
            gemv(w.mat("per_layer_proj"), b.q, b.proj),
            // rmsnorm(c·y) is rmsnorm(y) with eps / c²: the projection's 1/√hidden scale folds into eps.
            norm([b.ple, b.pli, b.proj, w.at("per_layer_norm"), 0], [0, 0], [pl, l], [eps * h as f32, std::f32::consts::FRAC_1_SQRT_2]),
        ]);
    }
    ops.push(norm([b.x[0], 0, 0, 0, w.at("0.input_layernorm")], b.q, [h, 1], [eps, 1.0]));
    for (i, &scalar) in scalars.iter().enumerate() {
        ops.extend(attention(cfg, w, b, i, sms));
        ops.extend(mlp(cfg, w, b, i, scalar));
    }
    ops.push(gemv(w.mat("embed"), b.q, b.logits));
    let blocks = sms.min((cfg.vocab as u32).div_ceil(4 * SAMPLE_THREADS));
    let [state, sampling, .., seen] = b.control;
    let args = [b.logits, b.work].map(A::P).into_iter().chain([A::U(cfg.vocab as u32)]);
    ops.push((SAMPLE, [blocks, 1, 1], SAMPLE_THREADS, args.chain([sampling, state, seen, b.token[1], b.partial[1], b.count[1]].map(A::P)).collect()));
    ops
}

/// The residual stream buffers layer `i` reads from and writes its attention to: with per-layer inputs a layer
/// writes the stream three times and they alternate per layer; without, twice, and every layer starts in `x[0]`.
pub(super) fn residual(cfg: &Config, x: [u64; 2], i: usize) -> (u64, u64) {
    let c = if cfg.per_layer > 0 { i % 2 } else { 0 };
    (x[c], x[1 - c])
}

/// Layer `i`'s attention, from its normalized input in `q` to the residual stream's other buffer (see [`residual`])
/// and the MLP's normalized input in `q`.
fn attention(cfg: &Config, w: &Weights, b: &Bufs, i: usize, sms: u32) -> Vec<Op> {
    let (g, n) = (usize::from(cfg.global[i]), |name: &str| w.at(&format!("{i}.{name}")));
    let (d, per, fresh) = (cfg.head_dim[g], cfg.per_block(g), cfg.source(i) == i);
    let blocks = cfg.heads as u32 / per;
    let (cap, win, splits) = if g == 1 { (b.context, b.context, sms / blocks) } else { (b.ring, cfg.window as u32, 2 * sms / blocks) };
    let ptrs = [b.qkv, n("q_norm"), if fresh { n("k_norm") } else { 0 }, b.control[2 + g], b.kv[i][0], b.kv[i][1], b.control[0]];
    let args =
        ptrs.map(A::P).into_iter().chain([cap, win, u32::from(fresh), cfg.group(g)].map(A::U)).chain([b.partial[0], b.count[0], b.q[0], b.q[1]].map(A::P));
    let (x0, x1) = residual(cfg, b.x, i);
    vec![
        gemv(w.mat(&format!("{i}.qkv")), b.q, b.qkv),
        (attend_kernel(d, per), [blocks, splits.clamp(1, 256), 1], 32 * ATTEND_WARPS, args.collect()),
        gemv(w.mat(&format!("{i}.o")), b.q, b.y),
        norm([x0, x1, b.y, n("post_attention_layernorm"), n("pre_feedforward_layernorm")], b.q, [cfg.hidden, 1], [cfg.eps, 1.0]),
    ]
}

/// The decode attention kernel for heads of `d`, `per` query heads a block.
fn attend_kernel(d: usize, per: u32) -> &'static str {
    match (d, per) {
        (256, 2) => ATTEND_256_PAIRS,
        (256, _) => ATTEND_256,
        _ => ATTEND_512,
    }
}

/// Layer `i`'s MLP and per-layer input, from the residual stream's other buffer (see [`residual`]) to the next
/// layer's, times `scalar`, with the next layer's normalized input (or the final norm's) in `q`.
fn mlp(cfg: &Config, w: &Weights, b: &Bufs, i: usize, scalar: f32) -> Vec<Op> {
    let (h, eps, n) = (cfg.hidden, cfg.eps, |name: &str| w.at(&format!("{i}.{name}")));
    let (x0, x1) = residual(cfg, b.x, i);
    let next = if i + 1 < cfg.layers() { w.at(&format!("{}.input_layernorm", i + 1)) } else { w.at("norm") };
    let mut ops = vec![
        gemv(w.mat(&format!("{i}.gate_up")), b.q, b.y),
        geglu(b.y, b.y + 4 * cfg.mlp(i) as u64, b.q, cfg.mlp(i)),
        gemv(w.mat(&format!("{i}.down")), b.q, b.y),
    ];
    if cfg.moe() {
        ops.extend(experts(cfg, w, b, i));
        let [zero, t, sum, .., yd] = b.moe;
        ops.extend([
            norm([zero, t, b.y, n("post_feedforward_layernorm_1"), 0], [0, 0], [h, 1], [eps, 1.0]),
            norm([t, sum, yd, n("post_feedforward_layernorm_2"), 0], [0, 0], [h, 1], [eps, 1.0]),
            norm([x1, x0, sum, n("post_feedforward_layernorm"), next], b.q, [h, 1], [eps, scalar]),
        ]);
        return ops;
    }
    if cfg.per_layer == 0 {
        ops.push(norm([x1, x0, b.y, n("post_feedforward_layernorm"), next], b.q, [h, 1], [eps, scalar]));
        return ops;
    }
    ops.extend([
        norm([x1, x0, b.y, n("post_feedforward_layernorm"), 0], b.q, [h, 1], [eps, 1.0]),
        gemv(w.mat(&format!("{i}.per_layer_gate")), b.q, b.y),
        geglu(b.y, b.pli + 4 * (i * cfg.per_layer) as u64, b.q, cfg.per_layer),
        gemv(w.mat(&format!("{i}.per_layer_proj")), b.q, b.y),
        norm([x0, x1, b.y, n("post_per_layer_input_norm"), next], b.q, [h, 1], [eps, scalar]),
    ]);
    ops
}

/// Layer `i`'s experts on the residual stream after attention: the router (an RMS norm with its scale, the Q4
/// projection, [`ROUTE_TOPK`]), then the chosen experts' gate and up, GELU, and their downs times their weights,
/// added up in order, into the last of the experts' buffers.
fn experts(cfg: &Config, w: &Weights, b: &Bufs, i: usize) -> Vec<Op> {
    let ([e, k, width], h, eps) = (cfg.experts.map(|v| v as u32), cfg.hidden, cfg.eps);
    let (n, x1) = (|name: &str| w.at(&format!("{i}.{name}")), residual(cfg, b.x, i).1);
    let [.., logits, idx, wts, ye, yd] = b.moe;
    let (up, down) = (w.mat(&format!("{i}.experts.gate_up")), w.mat(&format!("{i}.experts.down")));
    let moe = |m: Mat, y: u64, (rows, wts, slot): (u32, u64, u32)| {
        let args = [m.w[0], m.w[1], b.q[0], b.q[1], y].map(A::P).into_iter().chain([A::U(rows), A::U(m.cols), A::P(idx), A::P(wts), A::U(slot)]);
        (GEMV_Q4_MOE, [rows.div_ceil(GEMV_ROWS), k, 1], 32 * GEMV_ROWS, args.collect())
    };
    let gate = [ye, ye + 4 * u64::from(width), b.q[0], b.q[1]].map(A::P).into_iter().chain([k * width, width, 2 * width, 2 * width, down.cols].map(A::U));
    vec![
        norm([x1, 0, 0, 0, n("router_scale")], b.q, [h, 1], [eps, 1.0]),
        gemv(w.mat(&format!("{i}.router")), b.q, logits),
        (ROUTE_TOPK, [1, 1, 1], 128, [logits, n("expert_scale"), idx, wts].map(A::P).into_iter().chain([A::U(e), A::U(k)]).collect()),
        norm([x1, 0, 0, 0, n("pre_feedforward_layernorm_2")], b.q, [h, 1], [eps, 1.0]),
        moe(up, ye, (up.rows / e, 0, 0)),
        (GEGLU_Q8S, [(k * width).div_ceil(256), 1, 1], 256, gate.collect()),
        moe(down, yd, (down.rows / e, wts, down.cols)),
        (SUM_SLICES, [(h as u32).div_ceil(1024), 1, 1], 256, vec![A::P(yd), A::U(h as u32), A::U(k)]),
    ]
}

/// `v` rounded to bfloat16, as Gemma's embedding scale is applied.
pub(super) fn bf16(v: f32) -> f32 {
    let b = v.to_bits();
    f32::from_bits((b + 0x7fff + (b >> 16 & 1)) & 0xffff_0000)
}
