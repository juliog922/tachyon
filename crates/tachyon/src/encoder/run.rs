//! The encoder on the GPU: sequences packed back to back, without padding, through the kernels of
//! [`crate::ptx::encoder`]; the hidden states, Granite's embeddings or Laya's option logits out.

use super::laya::{Head, Kind};
use super::{Config, Settings};
use crate::cuda::{Context, DevBuf, HostBuf, Module, Stream, arg};
use crate::ptx::{ADD_ROWS, ATTEND_F16, ATTEND_F16_32, EMBED_F16, GATHER_ROWS, GELU_F16, GEMM_F16, GEMM_F16_TILE, NORM_F16, ROPE_F32};
use crate::token::Tokenizer;
use crate::wpk::{Loader, Wpk};
use crate::{Error, Result};
use std::path::Path;

/// Rows of the last scorer layer, padded for the matrix product.
const SCORES: u32 = 64;

/// A kernel argument.
enum A {
    P(u64),
    U(u32),
    F(f32),
}

/// Where the activations live: the residual stream `x`, the wide `f32` rows `big` (queries, keys and values, the
/// MLP's two halves, picked rows), and two buffers of `f16` rows a matrix product reads.
#[derive(Clone, Copy)]
struct Buffers {
    x: u64,
    big: u64,
    a: u64,
    b: u64,
}

/// A matrix product: `f16` rows `a` times weight `w`'s transpose, plus bias `bias` (empty: none), into `y`.
struct Product<'s> {
    a: u64,
    w: &'s str,
    bias: &'s str,
    y: u64,
}

/// A layer norm: rows of `x` normalized with weight `weight` and bias `bias` (empty: none), into `f16` rows `a` and,
/// unless 0, `f32` rows `xo`.
struct Norm<'s> {
    x: u64,
    a: u64,
    xo: u64,
    weight: &'s str,
    bias: &'s str,
}

/// What a batch reads from pinned host memory: token IDs, positions, each token's question kind, attention tiles,
/// rows to pick out. Device addresses of each part.
struct Inputs {
    buf: HostBuf,
    at: [usize; 5],
}

/// A ModernBERT encoder on the GPU, with Laya's head when the model has one.
pub struct Encoder {
    cfg: Config,
    head: Option<Head>,
    tok: Option<Tokenizer>,
    wpk: Wpk,
    weights: DevBuf,
    table: HostBuf,
    rope: DevBuf,
    scratch: DevBuf,
    bufs: Buffers,
    /// The batch in the buffers: its tokens and attention tiles.
    batch: (usize, u32),
    inputs: Inputs,
    module: Module,
    stream: Stream,
    cap: usize,
}

impl Encoder {
    /// Loads the model directory `dir` (see [`super::convert`]) onto `ctx`'s GPU, with room for
    /// [`Settings::tokens`] tokens at once.
    pub fn open(ctx: &Context, dir: &Path, settings: &Settings) -> Result<Encoder> {
        let (cfg, head, tok) = describe(dir)?;
        let (wpk, weights, table) = weights(ctx, dir)?;
        let cap = capacity(*settings)?;
        let (rope, (scratch, bufs), inputs) = (rope(ctx, &cfg, cap)?, scratch(ctx, &cfg, head.is_some(), cap)?, inputs(ctx, cap)?);
        let (module, stream) = (ctx.load(&crate::ptx::module())?, ctx.stream()?);
        let enc = Encoder { cfg, head, tok, wpk, weights, table, rope, scratch, bufs, batch: (0, 0), inputs, module, stream, cap };
        enc.check()?;
        Ok(enc)
    }

    /// The encoder's shape.
    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Laya's head settings, if the model is a Laya checkpoint.
    pub fn head(&self) -> Option<&Head> {
        self.head.as_ref()
    }

    /// The model's tokenizer, if its directory has one.
    pub fn tokenizer(&self) -> Option<&Tokenizer> {
        self.tok.as_ref()
    }

    /// Every token's last hidden state, `f32 [tokens, hidden]`, sequence after sequence.
    pub fn hidden(&mut self, seqs: &[&[u32]]) -> Result<Vec<f32>> {
        let h = self.cfg.hidden;
        let mut out = Vec::new();
        for batch in self.batches(seqs)? {
            let n = self.encode(batch, None)?;
            out.extend(self.download(self.bufs.x, n * h)?);
        }
        Ok(out)
    }

    /// Each sequence's embedding: its first token's last hidden state, to unit length.
    pub fn embeddings(&mut self, seqs: &[&[u32]]) -> Result<Vec<Vec<f32>>> {
        let h = self.cfg.hidden;
        let mut out = Vec::with_capacity(seqs.len());
        for batch in self.batches(seqs)? {
            self.encode(batch, None)?;
            let firsts: Vec<u32> = batch.iter().scan(0, |at, s| Some(std::mem::replace(at, *at + s.len()) as u32)).collect();
            self.pick(&firsts, self.bufs.x)?;
            let rows = self.download(self.bufs.big, firsts.len() * h)?;
            out.extend(rows.chunks(h).map(|r| {
                let len = r.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
                r.iter().map(|v| v / len).collect()
            }));
        }
        Ok(out)
    }

    /// Laya's option logits of each question: its sequence, the positions of its options' markers and its kind.
    pub fn logits(&mut self, questions: &[(&[u32], &[usize], Kind)]) -> Result<Vec<Vec<f32>>> {
        if self.head.is_none() {
            return Err(Error::Unsupported("decisions with a model that has no Laya head".into()));
        }
        let seqs: Vec<&[u32]> = questions.iter().map(|q| q.0).collect();
        let mut out = Vec::with_capacity(questions.len());
        let mut done = 0;
        for batch in self.batches(&seqs)? {
            let part = &questions[done..done + batch.len()];
            let kinds: Vec<u32> = part.iter().map(|q| q.2 as u32).collect();
            self.encode(batch, Some(&kinds))?;
            self.decide()?;
            let mut at = 0;
            let marks: Vec<u32> = part
                .iter()
                .flat_map(|q| {
                    let start = at;
                    at += q.0.len();
                    q.1.iter().map(move |&m| (start + m) as u32)
                })
                .collect();
            let scores = self.score(&marks)?;
            let mut it = scores.into_iter();
            out.extend(part.iter().map(|q| it.by_ref().take(q.1.len()).collect()));
            done += batch.len();
        }
        Ok(out)
    }

    /// `seqs` in batches of at most the capacity's tokens; an error if one is longer than it, or empty.
    fn batches<'s>(&self, seqs: &'s [&'s [u32]]) -> Result<Vec<&'s [&'s [u32]]>> {
        if let Some(s) = seqs.iter().find(|s| s.is_empty() || s.len() > self.cap) {
            return Err(Error::TooLong { tokens: s.len(), context: self.cap });
        }
        let (mut out, mut from, mut used) = (Vec::new(), 0, 0);
        for (i, s) in seqs.iter().enumerate() {
            if used + s.len() > self.cap {
                out.push(&seqs[from..i]);
                (from, used) = (i, 0);
            }
            used += s.len();
        }
        out.push(&seqs[from..]);
        Ok(out)
    }

    /// Address of tensor `name`.
    fn at(&self, name: &str) -> u64 {
        self.wpk.tensor(name).map_or(0, |t| self.weights.ptr() + t.offset)
    }

    /// An error unless every tensor the encoder and head read is in the file.
    fn check(&self) -> Result<()> {
        let mut names = vec!["embed_norm".to_string(), "final_norm".into()];
        for i in 0..self.cfg.layers() {
            names.extend(["qkv", "o", "mlp_norm", "up", "down"].iter().map(|n| format!("{i}.{n}")));
            names.extend((i > 0).then(|| format!("{i}.attn_norm")));
        }
        if let Some(head) = &self.head {
            names.extend(["type_emb", "score.norm.w", "score.norm.b", "score.1.w", "score.1.b", "score.2.w", "score.2.b"].map(String::from));
            for l in 0..head.layers {
                names.extend(["norm1", "qkv", "o", "norm2", "up", "down"].iter().flat_map(|n| [format!("h{l}.{n}.w"), format!("h{l}.{n}.b")]));
            }
        }
        match names.iter().find(|n| self.wpk.tensor(n).is_none()) {
            Some(n) => Err(Error::Format(format!("weights.wpk: no tensor {n}"))),
            None => Ok(()),
        }
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
        // SAFETY: every call passes the kernel's arguments in order and with their types, as `crate::ptx::encoder`
        // documents them, on buffers laid out for the capacity this encoder was opened with.
        unsafe { self.stream.launch(&self.module.function(name)?, grid, [threads, 1, 1], 0, &ptrs) }
    }

    /// `p.y = p.a · wᵀ + bias` for `tokens` rows, stored as `mode` says (see [`crate::ptx::GEMM_F16`]).
    fn gemm(&self, p: &Product, tokens: usize, mode: u32) -> Result<()> {
        let t = self.wpk.tensor(p.w).ok_or_else(|| Error::Format(format!("weights.wpk: no tensor {}", p.w)))?;
        let (rows, cols) = (t.shape[0] as u32, t.shape[1] as u32);
        if rows % GEMM_F16_TILE != 0 || cols % 64 != 0 {
            return Err(Error::Unsupported(format!("a {rows} × {cols} matrix: products take multiples of 64")));
        }
        let bias = if p.bias.is_empty() { 0 } else { self.at(p.bias) };
        let args = [A::P(p.a), A::P(self.at(p.w)), A::P(bias), A::P(p.y), A::U(rows), A::U(cols), A::U(tokens as u32), A::U(mode)];
        self.launch(GEMM_F16, [(tokens as u32).div_ceil(GEMM_F16_TILE), rows / GEMM_F16_TILE, 1], 128, &args)
    }

    /// Layer norm of `rows` rows (see [`Norm`]).
    fn norm(&self, n: &Norm, rows: usize, eps: f32) -> Result<()> {
        let bias = if n.bias.is_empty() { 0 } else { self.at(n.bias) };
        let args = [A::P(n.x), A::P(self.at(n.weight)), A::P(bias), A::P(n.a), A::P(n.xo), A::U(self.cfg.hidden as u32), A::U(rows as u32), A::F(eps)];
        self.launch(NORM_F16, [(rows as u32).div_ceil(4), 1, 1], 128, &args)
    }

    /// Attention over `qkv` (`big`) into `a`, heads of `d`, within `window` (0: global), on the batch's tiles.
    fn attend(&self, d: usize, window: usize, tiles: u32) -> Result<()> {
        let kernel = if d == 32 { ATTEND_F16_32 } else { ATTEND_F16 };
        let scale = std::f32::consts::LOG2_E / (d as f32).sqrt();
        let args = [A::P(self.bufs.big), A::P(self.bufs.a), A::P(self.input(3)?), A::U(self.cfg.hidden as u32), A::U(window as u32), A::F(scale)];
        self.launch(kernel, [tiles, (self.cfg.hidden / d) as u32, 1], 64, &args)
    }

    /// Device address of input part `i`.
    fn input(&self, i: usize) -> Result<u64> {
        Ok(self.inputs.buf.device_ptr()? + self.inputs.at[i] as u64)
    }

    /// Writes the batch's inputs: IDs, positions, kinds (each token its question's), and tiles; their count.
    fn fill(&mut self, seqs: &[&[u32]], kinds: Option<&[u32]>) -> u32 {
        let at = self.inputs.at;
        let words: &mut [u8] = &mut self.inputs.buf;
        let mut put = |part: usize, i: usize, v: u32| words[at[part] + 4 * i..][..4].copy_from_slice(&v.to_le_bytes());
        let (mut t, mut tiles) = (0, 0);
        for (s, seq) in seqs.iter().enumerate() {
            for (p, &id) in seq.iter().enumerate() {
                put(0, t + p, id);
                put(1, t + p, p as u32);
                put(2, t + p, kinds.map_or(0, |k| k[s]));
            }
            for q in (0..seq.len()).step_by(64) {
                for (k, v) in [t, seq.len(), q].into_iter().enumerate() {
                    put(3, 3 * tiles + k, v as u32);
                }
                tiles += 1;
            }
            t += seq.len();
        }
        tiles as u32
    }

    /// Runs the encoder over `seqs`, with each sequence's question kind for Laya's head; leaves the last hidden
    /// states in `x` and returns the number of tokens.
    fn encode(&mut self, seqs: &[&[u32]], kinds: Option<&[u32]>) -> Result<usize> {
        let tiles = self.fill(seqs, kinds);
        let (n, width, Buffers { x, a, .. }) = (seqs.iter().map(|s| s.len()).sum::<usize>(), self.cfg.hidden as u32, self.bufs);
        self.launch(EMBED_F16, [width.div_ceil(256), n as u32, 1], 256, &[A::P(self.table.device_ptr()?), A::P(self.input(0)?), A::P(x), A::U(width)])?;
        self.norm(&Norm { x, a, xo: x, weight: "embed_norm", bias: "" }, n, self.cfg.eps)?;
        for i in 0..self.cfg.layers() {
            self.attention(i, n, tiles)?;
            self.mlp(i, n)?;
        }
        self.norm(&Norm { x, a, xo: x, weight: "final_norm", bias: "" }, n, self.cfg.eps)?;
        self.batch = (n, tiles);
        Ok(n)
    }

    /// Encoder layer `i`'s attention over the batch's `n` tokens, added to `x`.
    fn attention(&self, i: usize, n: usize, tiles: u32) -> Result<()> {
        let (cfg, Buffers { x, big, a, .. }) = (&self.cfg, self.bufs);
        if i > 0 {
            self.norm(&Norm { x, a, xo: 0, weight: &format!("{i}.attn_norm"), bias: "" }, n, cfg.eps)?;
        }
        self.gemm(&Product { a, w: &format!("{i}.qkv"), bias: "", y: big }, n, 0)?;
        let (global, d) = (cfg.global[i], cfg.head_dim());
        let table = self.rope.ptr() + if global { 0 } else { (self.cap * d * 4) as u64 };
        let args = [A::P(big), A::P(table), A::P(self.input(1)?), A::U(cfg.hidden as u32), A::U(d as u32), A::U(n as u32)];
        self.launch(ROPE_F32, [((n * cfg.hidden / 2) as u32).div_ceil(256), 1, 1], 256, &args)?;
        self.attend(d, if global { 0 } else { cfg.window }, tiles)?;
        self.gemm(&Product { a, w: &format!("{i}.o"), bias: "", y: x }, n, 1)
    }

    /// Encoder layer `i`'s gated MLP over the batch's `n` tokens, added to `x`.
    fn mlp(&self, i: usize, n: usize) -> Result<()> {
        let (cfg, Buffers { x, big, a, .. }) = (&self.cfg, self.bufs);
        self.norm(&Norm { x, a, xo: 0, weight: &format!("{i}.mlp_norm"), bias: "" }, n, cfg.eps)?;
        self.gemm(&Product { a, w: &format!("{i}.up"), bias: "", y: big }, n, 0)?;
        let len = (n * cfg.inter) as u32;
        self.launch(GELU_F16, [len.div_ceil(256), 1, 1], 256, &[A::P(big), A::P(a), A::U(cfg.inter as u32), A::U(len), A::U(1)])?;
        self.gemm(&Product { a, w: &format!("{i}.down"), bias: "", y: x }, n, 1)
    }

    /// Laya's head over the batch in `x`: the type embedding, then its layers.
    fn decide(&self) -> Result<()> {
        let (width, (n, tiles)) = (self.cfg.hidden as u32, self.batch);
        self.launch(ADD_ROWS, [width.div_ceil(256), n as u32, 1], 256, &[A::P(self.at("type_emb")), A::P(self.input(2)?), A::P(self.bufs.x), A::U(width)])?;
        (0..self.head.as_ref().map_or(0, |hd| hd.layers)).try_for_each(|l| self.head_layer(l, n, tiles))
    }

    /// Head layer `l`: pre-norm attention (heads of 64, no RoPE) and a ReLU MLP, each with biases.
    fn head_layer(&self, l: usize, n: usize, tiles: u32) -> Result<()> {
        let (Buffers { x, big, a, b }, w) = (self.bufs, |s: &str| format!("h{l}.{s}"));
        self.norm(&Norm { x, a, xo: 0, weight: &w("norm1.w"), bias: &w("norm1.b") }, n, 1e-5)?;
        self.gemm(&Product { a, w: &w("qkv.w"), bias: &w("qkv.b"), y: big }, n, 0)?;
        self.attend(64, 0, tiles)?;
        self.gemm(&Product { a, w: &w("o.w"), bias: &w("o.b"), y: x }, n, 1)?;
        self.norm(&Norm { x, a, xo: 0, weight: &w("norm2.w"), bias: &w("norm2.b") }, n, 1e-5)?;
        self.gemm(&Product { a, w: &w("up.w"), bias: &w("up.b"), y: b }, n, 3)?;
        self.gemm(&Product { a: b, w: &w("down.w"), bias: &w("down.b"), y: x }, n, 1)
    }

    /// The scorer at rows `marks` of `x`: one logit each.
    fn score(&mut self, marks: &[u32]) -> Result<Vec<f32>> {
        let (Buffers { x, big, a, .. }, width, rows) = (self.bufs, self.cfg.hidden as u32, marks.len());
        self.pick(marks, x)?;
        self.norm(&Norm { x: big, a, xo: 0, weight: "score.norm.w", bias: "score.norm.b" }, rows, 1e-5)?;
        self.gemm(&Product { a, w: "score.1.w", bias: "score.1.b", y: x }, rows, 0)?;
        let len = rows as u32 * width;
        self.launch(GELU_F16, [len.div_ceil(256), 1, 1], 256, &[A::P(x), A::P(a), A::U(width), A::U(len), A::U(0)])?;
        self.gemm(&Product { a, w: "score.2.w", bias: "score.2.b", y: big }, rows, 0)?;
        let rows = self.download(big, rows * SCORES as usize)?;
        Ok(rows.chunks(SCORES as usize).map(|r| r[0]).collect())
    }

    /// Copies rows `rows` of `src` (`f32 [·, hidden]`) into `big`.
    fn pick(&mut self, rows: &[u32], src: u64) -> Result<()> {
        let at = self.inputs.at[4];
        let words: &mut [u8] = &mut self.inputs.buf;
        for (i, r) in rows.iter().enumerate() {
            words[at + 4 * i..][..4].copy_from_slice(&r.to_le_bytes());
        }
        let h = self.cfg.hidden as u32;
        self.launch(GATHER_ROWS, [h.div_ceil(256), rows.len() as u32, 1], 256, &[A::P(src), A::P(self.input(4)?), A::P(self.bufs.big), A::U(h)])
    }

    /// `len` values of `f32` from device address `src`.
    fn download(&self, src: u64, len: usize) -> Result<Vec<f32>> {
        self.stream.sync()?;
        let mut raw = vec![0u8; 4 * len];
        let at = usize::try_from(src - self.scratch.ptr()).unwrap_or(usize::MAX);
        // SAFETY: `src` is in the scratch buffer, which holds at least `len` values there; `sync` completed the
        // work that wrote them, and the copy completes before `download` returns.
        unsafe { self.stream.download(&mut raw, &self.scratch, at) }?;
        self.stream.sync()?;
        Ok(raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
    }
}

/// RoPE's cosines and sines for `positions` positions: the global layers' table, then the sliding layers'.
fn rope_tables(cfg: &Config, positions: usize) -> Vec<f32> {
    let half = cfg.head_dim() / 2;
    let mut out = Vec::with_capacity(2 * positions * half * 2);
    for theta in cfg.theta {
        for p in 0..positions {
            for j in 0..half {
                let angle = p as f64 * f64::from(theta).powf(-2.0 * j as f64 / (2 * half) as f64);
                out.extend([angle.cos() as f32, angle.sin() as f32]);
            }
        }
    }
    out
}

/// The activation buffers for `cap` tokens, in one allocation.
fn scratch(ctx: &Context, cfg: &Config, laya: bool, cap: usize) -> Result<(DevBuf, Buffers)> {
    let h = cfg.hidden;
    let wide = (3 * h).max(2 * cfg.inter).max(SCORES as usize);
    let half = h.max(cfg.inter).max(if laya { 4 * h } else { 0 });
    let sizes = [4 * cap * h, 4 * cap * wide, 2 * cap * half, 2 * cap * half];
    let buf = ctx.alloc(sizes.iter().sum())?;
    let mut at = [buf.ptr(); 4];
    for i in 1..4 {
        at[i] = at[i - 1] + sizes[i - 1] as u64;
    }
    let bufs = Buffers { x: at[0], big: at[1], a: at[2], b: at[3] };
    Ok((buf, bufs))
}

/// The pinned inputs for `cap` tokens: IDs, positions, kinds, tiles (at most two per token), picked rows.
fn inputs(ctx: &Context, cap: usize) -> Result<Inputs> {
    let sizes = [4 * cap, 4 * cap, 4 * cap, 12 * 2 * cap, 4 * cap];
    let buf = ctx.alloc_host(sizes.iter().sum())?;
    let mut at = [0; 5];
    for i in 1..5 {
        at[i] = at[i - 1] + sizes[i - 1];
    }
    Ok(Inputs { buf, at })
}

/// A model directory's encoder shape, Laya head settings and tokenizer, those it has.
fn describe(dir: &Path) -> Result<(Config, Option<Head>, Option<Tokenizer>)> {
    let cfg = Config::from_json(&std::fs::read(dir.join("config.json"))?)?;
    let (head, tok) = (optional(dir, "head.json", Head::from_json)?, optional(dir, "tokenizer.tok", Tokenizer::from_bytes)?);
    if !matches!(cfg.head_dim(), 32 | 64) || head.is_some() && cfg.hidden % 64 != 0 {
        return Err(Error::Unsupported(format!("an encoder with heads of {} dimensions", cfg.head_dim())));
    }
    Ok((cfg, head, tok))
}

/// The weights of `dir`, read into VRAM, and its embedding table, into pinned host memory.
fn weights(ctx: &Context, dir: &Path) -> Result<(Wpk, DevBuf, HostBuf)> {
    let (wpk, hpk) = (Wpk::open(dir.join("weights.wpk"))?, Wpk::open(dir.join("host.wpk"))?);
    let mut loader = Loader::new(ctx)?;
    let (weights, mut table) = (ctx.alloc_pooled(wpk.data_len())?, ctx.alloc_host(hpk.data_len())?);
    loader.to_device(&wpk, &weights)?;
    loader.to_host(&hpk, &mut table)?;
    Ok((wpk, weights, table))
}

/// The file `name` of `dir` read by `parse`, if it is there.
fn optional<T>(dir: &Path, name: &str, parse: impl Fn(&[u8]) -> Result<T>) -> Result<Option<T>> {
    let path = dir.join(name);
    if path.exists() { parse(&std::fs::read(path)?).map(Some) } else { Ok(None) }
}

/// RoPE's tables for `cap` positions, in VRAM.
fn rope(ctx: &Context, cfg: &Config, cap: usize) -> Result<DevBuf> {
    let tables: Vec<u8> = rope_tables(cfg, cap).iter().flat_map(|v| v.to_le_bytes()).collect();
    let rope = ctx.alloc(tables.len())?;
    Loader::new(ctx)?.upload(&tables, &rope)?;
    Ok(rope)
}

/// The tokens a batch holds: [`Settings::tokens`] in whole tiles, at most 65535 (a grid's second dimension).
fn capacity(settings: Settings) -> Result<usize> {
    let cap = settings.tokens.div_ceil(GEMM_F16_TILE as usize) * GEMM_F16_TILE as usize;
    if cap == 0 || cap > usize::from(u16::MAX) {
        return Err(Error::Unsupported(format!("{} tokens at once: from 1 to 65535", settings.tokens)));
    }
    Ok(cap)
}
