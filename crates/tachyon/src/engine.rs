//! The engine: models on one GPU behind a blocking chat API, for one caller at a time.
//!
//! [`Engine::load`] brings a model directory into VRAM under a name; [`Engine::chat`] puts a conversation through
//! that model's chat template, feeds the prompt and returns a [`TokenStream`], which hands out the answer's text as
//! it is generated, on the caller's thread: v1 serves one waiting caller, so there is no request thread, queue or
//! async runtime. Each call to [`TokenStream::next_text`] waits for one step while the next one runs, and allocates
//! nothing. Dropping the stream cancels the answer at the cost of that one queued step.
//!
//! Models stay in VRAM while they fit [`Limits::vram`] (by default 70% of the GPU's, so caches and workspace never
//! force an eviction) and their keep-alive lasts. A model that must make room, the least recently used first, or
//! whose keep-alive has run out is parked in the host-RAM tier (its weights in pinned memory, back in VRAM at full
//! PCIe speed) while that fits [`Limits::host`]. A model back from the tier keeps that copy, so it parks again
//! without copying or allocating; to make room, the least recently used copies are freed, sending parked models
//! to disk. A pinned model
//! never leaves VRAM, and neither does the one answering: its stream borrows the engine. With no thread, keep-alives
//! are checked at every call; an application that leaves the engine idle calls [`Engine::expire`] when it likes.
//!
//! ```no_run
//! use tachyon::engine::{Engine, Limits, Load, Request};
//! use tachyon::token::{Message, Role};
//!
//! let mut engine = Engine::new(0, Limits::default())?;
//! engine.load("e4b", "gemma-4-e4b".as_ref(), &Load::default())?;
//! let messages = [Message { role: Role::User, text: "Why is the sky blue?" }];
//! let mut answer = engine.chat("e4b", &Request::new(&messages))?;
//! while let Some(text) = answer.next_text()? {
//!     print!("{text}");
//! }
//! # Ok::<(), tachyon::Error>(())
//! ```

use crate::cuda::Context;
use crate::model::{Decoder, Parked, Settings, footprint};
use crate::ptx::Sampling;
use crate::token::{Chat, Detokenizer, Message, Tokenizer};
use crate::{Error, Result};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Models on one GPU, ready to chat.
pub struct Engine {
    ctx: Context,
    models: Vec<Model>,
    /// The VRAM and host-tier limits, in bytes.
    limits: [usize; 2],
    prompt: Vec<u32>,
}

/// How much memory an [`Engine`]'s models may take.
#[derive(Debug, Clone, Copy, Default)]
pub struct Limits {
    /// VRAM for the resident models (weights, caches, buffers), in bytes; by default 70% of the GPU's.
    pub vram: Option<usize>,
    /// Pinned host memory for the host tier, in bytes; by default half the machine's RAM (E4B parked takes 4 GB: its
    /// weights and its per-layer table).
    pub host: Option<usize>,
}

/// How [`Engine::load`] keeps a model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Load {
    /// Context, and prefill's chunk.
    pub settings: Settings,
    /// How long the model stays in VRAM after its last use, unless a request says otherwise: zero moves it out at
    /// once, [`Duration::MAX`] keeps it. Five minutes by default.
    pub keep_alive: Duration,
    /// Never moves out of VRAM, by keep-alive or to make room.
    pub pin: bool,
}

impl Default for Load {
    fn default() -> Load {
        Load { settings: Settings::default(), keep_alive: Duration::from_secs(300), pin: false }
    }
}

/// Where a loaded model is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// In VRAM, ready to answer.
    Vram,
    /// In the host-RAM tier: its weights in pinned memory.
    Host,
    /// Only on disk.
    Disk,
}

/// A loaded model: its name, directory and options, its tokenizer and chat template with the tokens that end an
/// answer, the VRAM it takes, when it was last used and when its keep-alive ends (never: `None`), and where it is.
struct Model {
    name: String,
    dir: PathBuf,
    load: Load,
    tok: Tokenizer,
    chat: Chat,
    ends: [u32; 2],
    need: usize,
    used: Instant,
    until: Option<Instant>,
    at: Place,
}

// One per model, moved only between tiers: its size does not matter.
#[allow(clippy::large_enum_variant)]
enum Place {
    Vram(Decoder),
    Host(Parked),
    Disk,
}

impl Model {
    /// Marks it used now, kept for `keep`.
    fn touch(&mut self, keep: Duration) {
        self.used = Instant::now();
        self.until = self.used.checked_add(keep);
    }

    /// Its decoder: a model stays in VRAM while it answers.
    fn dec(&mut self) -> &mut Decoder {
        match &mut self.at {
            Place::Vram(dec) => dec,
            _ => unreachable!("a model answering is in VRAM"),
        }
    }
}

/// What [`Engine::chat`] answers, and how.
#[derive(Debug, Clone, Copy)]
pub struct Request<'a> {
    /// The conversation so far; the answer is the model's next message.
    pub messages: &'a [Message<'a>],
    /// Lets the model think before it answers; the thinking is part of the text.
    pub think: bool,
    /// How tokens are picked (the soft cap is the model's).
    pub sampling: Sampling,
    /// Most tokens the answer may take; the context also limits it.
    pub max_tokens: usize,
    /// How long the model stays in VRAM after this answer, instead of its [`Load::keep_alive`].
    pub keep_alive: Option<Duration>,
}

impl<'a> Request<'a> {
    /// A request to answer `messages` greedily, without thinking, until the model ends its turn.
    pub fn new(messages: &'a [Message<'a>]) -> Request<'a> {
        Request { messages, think: false, sampling: Sampling::default(), max_tokens: usize::MAX, keep_alive: None }
    }
}

impl Engine {
    /// An engine on GPU `gpu`, with no model loaded.
    pub fn new(gpu: u32, limits: Limits) -> Result<Engine> {
        let ctx = Context::new(gpu)?;
        let vram = limits.vram.unwrap_or(ctx.memory()?.1 / 10 * 7);
        let host = limits.host.unwrap_or_else(|| ram() / 2);
        Ok(Engine { ctx, models: Vec::new(), limits: [vram, host], prompt: Vec::new() })
    }

    /// Loads the model directory `dir` (see [`crate::model::convert`]) into VRAM as `name`, kept as `load` says,
    /// making room as needed; a model of that name is unloaded first.
    pub fn load(&mut self, name: &str, dir: &Path, load: &Load) -> Result<()> {
        self.unload(name);
        self.expire()?;
        let tok = Tokenizer::from_bytes(&std::fs::read(dir.join("tokenizer.tok"))?)?;
        let chat = Chat::gemma4(&tok)?;
        let ends = [chat.end_of_turn(), tok.special("<eos>").unwrap_or(chat.end_of_turn())];
        let need = footprint(dir, &load.settings)?;
        let (used, at) = (Instant::now(), Place::Disk);
        let mut model = Model { name: name.into(), dir: dir.into(), load: load.clone(), tok, chat, ends, need, used, until: None, at };
        model.touch(load.keep_alive);
        self.models.push(model);
        let loaded = self.resident(self.models.len() - 1);
        if loaded.is_err() {
            self.models.pop();
        }
        loaded
    }

    /// Unloads model `name`, freeing its memory wherever it is; whether it was loaded.
    pub fn unload(&mut self, name: &str) -> bool {
        let before = self.models.len();
        self.models.retain(|m| m.name != name);
        // The VRAM goes back to the system, not only to this engine's pool.
        let _ = self.ctx.trim();
        self.models.len() < before
    }

    /// Where model `name` is, if it is loaded.
    pub fn tier(&self, name: &str) -> Option<Tier> {
        let m = self.models.iter().find(|m| m.name == name)?;
        Some(match m.at {
            Place::Vram(_) => Tier::Vram,
            Place::Host(_) => Tier::Host,
            Place::Disk => Tier::Disk,
        })
    }

    /// The tokenizer of model `name`, if it is loaded.
    pub fn tokenizer(&self, name: &str) -> Option<&Tokenizer> {
        self.models.iter().find(|m| m.name == name).map(|m| &m.tok)
    }

    /// Moves out of VRAM every model whose keep-alive has run out, but pinned ones, and gives their VRAM back to the
    /// system. Every call does it first. A model moved out to make room keeps its VRAM in the pool for the next.
    pub fn expire(&mut self) -> Result<()> {
        let now = Instant::now();
        let mut freed = false;
        for i in 0..self.models.len() {
            let m = &self.models[i];
            if matches!(m.at, Place::Vram(_)) && !m.load.pin && m.until.is_some_and(|t| t <= now) {
                self.evict(i)?;
                freed = true;
            }
        }
        if freed { self.ctx.trim() } else { Ok(()) }
    }

    /// Feeds the conversation of `req` to model `name` and returns its answer, as a stream: every prompt token but
    /// the last through prefill, the last through a decode step, which samples the answer's first token. The model
    /// comes into VRAM first if it is out.
    pub fn chat(&mut self, name: &str, req: &Request<'_>) -> Result<TokenStream<'_>> {
        self.expire()?;
        let i = self.models.iter().position(|m| m.name == name).ok_or_else(|| Error::NoModel(name.into()))?;
        self.resident(i)?;
        let (m, prompt) = (&mut self.models[i], &mut self.prompt);
        prompt.clear();
        m.chat.encode(&m.tok, req.messages, req.think, prompt);
        let (tokens, context) = (prompt.len(), m.load.settings.context);
        if tokens >= context {
            return Err(Error::TooLong { tokens, context });
        }
        let dec = m.dec();
        dec.reset(&req.sampling)?;
        let (&last, rest) = prompt.split_last().unwrap_or((&0, &[]));
        dec.prefill(rest)?;
        let first = dec.step(last)?;
        dec.start(first)?;
        let (left, keep) = (req.max_tokens.min(context - tokens), req.keep_alive.unwrap_or(m.load.keep_alive));
        Ok(TokenStream { engine: self, model: i, keep, detok: Detokenizer::default(), text: String::with_capacity(256), next: Some(first), left, tokens: 0 })
    }

    /// Brings model `i` into VRAM, moving out the least recently used models that may go until it fits.
    fn resident(&mut self, i: usize) -> Result<()> {
        let need = self.models[i].need;
        while !matches!(self.models[i].at, Place::Vram(_)) && self.used() + need > self.limits[0] {
            let may_go = |(j, m): &(usize, &Model)| *j != i && !m.load.pin && matches!(m.at, Place::Vram(_));
            match self.models.iter().enumerate().filter(may_go).min_by_key(|(_, m)| m.used) {
                Some((j, _)) => self.evict(j)?,
                None => return Err(Error::OutOfMemory { need, free: self.limits[0].saturating_sub(self.used()) }),
            }
        }
        self.bring(i)
    }

    /// Brings model `i` into VRAM from wherever it is.
    fn bring(&mut self, i: usize) -> Result<()> {
        let m = &mut self.models[i];
        m.at = match std::mem::replace(&mut m.at, Place::Disk) {
            Place::Vram(dec) => Place::Vram(dec),
            Place::Host(parked) => Place::Vram(parked.resume(&m.load.settings)?),
            Place::Disk => Place::Vram(Decoder::open(&self.ctx, &m.dir, &m.load.settings)?),
        };
        Ok(())
    }

    /// Moves model `i` out of VRAM: to the host tier if it fits there, making room by freeing the least recently
    /// used pinned memory (a resident model's kept copy, or a parked model, which goes to disk), or else to disk.
    fn evict(&mut self, i: usize) -> Result<()> {
        let Place::Vram(dec) = std::mem::replace(&mut self.models[i].at, Place::Disk) else { return Ok(()) };
        let bytes = dec.parked_bytes();
        if bytes > self.limits[1] {
            return Ok(());
        }
        while self.hosted() + bytes > self.limits[1] {
            // Something else holds pinned memory: `bytes` alone fits.
            let holds = |m: &&mut Model| matches!(&m.at, Place::Host(_)) || matches!(&m.at, Place::Vram(d) if d.shadow_bytes() > 0);
            match self.models.iter_mut().filter(holds).min_by_key(|m| m.used).map(|m| &mut m.at) {
                Some(Place::Vram(d)) => d.drop_shadow(),
                Some(at) => *at = Place::Disk,
                None => break,
            }
        }
        self.models[i].at = Place::Host(dec.park()?);
        Ok(())
    }

    /// VRAM the resident models take.
    fn used(&self) -> usize {
        self.models.iter().filter(|m| matches!(m.at, Place::Vram(_))).map(|m| m.need).sum()
    }

    /// Pinned host memory of the host tier: parked models, and the copies resumed models keep.
    fn hosted(&self) -> usize {
        let held = |m: &Model| match &m.at {
            Place::Host(p) => p.bytes(),
            Place::Vram(d) => d.shadow_bytes(),
            Place::Disk => 0,
        };
        self.models.iter().map(held).sum()
    }
}

/// The machine's RAM, in bytes.
fn ram() -> usize {
    let info = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    info.split_whitespace().nth(1).and_then(|kb| kb.parse::<usize>().ok()).unwrap_or(0) << 10
}

/// An answer being generated; see [`Engine::chat`].
pub struct TokenStream<'e> {
    engine: &'e mut Engine,
    model: usize,
    /// How long the model stays in VRAM after the answer.
    keep: Duration,
    detok: Detokenizer,
    text: String,
    /// The token to show next, when it is already known.
    next: Option<u32>,
    left: usize,
    tokens: usize,
}

impl TokenStream<'_> {
    /// The answer's next piece of text, whole characters only, or `None` once it has ended.
    pub fn next_text(&mut self) -> Result<Option<&str>> {
        self.text.clear();
        while self.text.is_empty() && self.left > 0 {
            let m = &mut self.engine.models[self.model];
            let token = match self.next.take() {
                Some(t) => t,
                None => m.dec().next_token()?,
            };
            if m.ends.contains(&token) {
                self.left = 0;
                break;
            }
            (self.left, self.tokens) = (self.left - 1, self.tokens + 1);
            self.detok.push(&m.tok, token, &mut self.text);
        }
        if self.left == 0 {
            self.detok.finish(&mut self.text);
        }
        Ok((!self.text.is_empty()).then_some(self.text.as_str()))
    }

    /// Tokens generated so far, the end of the turn not counted.
    pub fn tokens(&self) -> usize {
        self.tokens
    }

    /// Tokens of the prompt.
    pub fn prompt(&self) -> usize {
        self.engine.prompt.len()
    }
}

impl Drop for TokenStream<'_> {
    /// Ends or cancels the answer: waits for the step still queued, then starts the model's keep-alive.
    fn drop(&mut self) {
        let m = &mut self.engine.models[self.model];
        let _ = m.dec().stop();
        m.touch(self.keep);
        let _ = self.engine.expire();
    }
}
