//! The engine: models on one GPU behind a blocking chat API, for one caller at a time.
//!
//! [`Engine::load`] brings a model directory into VRAM under a name; [`Engine::chat`] puts a conversation through
//! that model's chat template, feeds the prompt and returns a [`TokenStream`], which hands out the answer's text as
//! it is generated, on the caller's thread: v1 serves one waiting caller, so there is no request thread, queue or
//! async runtime. Each call to [`TokenStream::next_text`] waits for one step while the next one runs, and allocates
//! nothing. Dropping the stream cancels the answer at the cost of that one queued step.
//!
//! Models stay in VRAM while they fit [`Limits::vram`] (by default 70% of the GPU's, so caches and workspace never
//! force an eviction) and their keep-alive lasts. A model that must make room, the least recently used first, is
//! parked in the host-RAM tier (its weights in pinned memory, back in VRAM at full PCIe speed) while that fits
//! [`Limits::host`]. A model whose keep-alive has run out, in VRAM or parked, goes to disk: it takes no memory, keeps
//! its name and settings, and the next call reads it again. A model back from the tier keeps that copy, so it parks again
//! without copying or allocating; to make room, the least recently used copies are freed, sending parked models
//! to disk. A pinned model
//! never leaves VRAM, and neither does the one answering: its stream borrows the engine. With no thread, keep-alives
//! are checked at every call; an application that leaves the engine idle calls [`Engine::expire`] when it likes.
//!
//! Messages may show images ([`Message::images`]) and play audio ([`Message::audio`]) to a model converted with its
//! vision and audio towers: a tower comes into VRAM with the first image or clip the model sees, and leaves with the
//! model.
//!
//! ```no_run
//! use tachyon::engine::{Engine, Limits, Load, Request};
//! use tachyon::token::{Message, Role};
//!
//! let mut engine = Engine::new(0, Limits::default())?;
//! engine.load("e4b", "gemma-4-e4b".as_ref(), &Load::default())?;
//! let messages = [Message::new(Role::User, "Why is the sky blue?")];
//! let mut answer = engine.chat("e4b", &Request::new(&messages))?;
//! while let Some(text) = answer.next_text()? {
//!     print!("{text}");
//! }
//! # Ok::<(), tachyon::Error>(())
//! ```

use crate::audio::Tower;
use crate::cuda::Context;
use crate::encoder::laya::Question;
use crate::encoder::{self, Encoder};
use crate::media::DEFAULT_BUDGET;
use crate::model::{Decoder, Parked, Settings, footprint};
use crate::ptx::Sampling;
use crate::token::{Chat, Detokenizer, Message, Role, Tokenizer};
use crate::vision::Vision;
use crate::{Error, Result};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Models on one GPU, ready to chat.
pub struct Engine {
    ctx: Context,
    models: Vec<Model>,
    /// Encoders, by name: Laya's decisions and Granite's embeddings, always in VRAM.
    encoders: Vec<Held>,
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
    /// The soft tokens an image becomes (one of [`crate::media::BUDGETS`]), 280 by default: more see finer detail
    /// and take longer.
    pub budget: usize,
}

impl Default for Load {
    fn default() -> Load {
        Load { settings: Settings::default(), keep_alive: Duration::from_secs(300), pin: false, budget: DEFAULT_BUDGET }
    }
}

/// A loaded model, as [`Engine::models`] lists it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Loaded {
    /// The name it was loaded as.
    pub name: String,
    /// Where it is.
    pub tier: Tier,
    /// The VRAM it takes, in bytes, its towers included: 0 out of VRAM.
    pub vram: usize,
    /// How long it has left in memory, VRAM or the host tier; `None`: no end.
    pub left: Option<Duration>,
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
    /// Its vision tower, once it has seen an image while in VRAM; its audio tower, once it has heard a clip.
    vision: Option<Vision>,
    audio: Option<Tower>,
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

/// A loaded encoder: its name, the encoder, how long it stays after a use (`Duration::MAX`: until unloaded), whether
/// it is pinned, and when its keep-alive ends.
struct Held {
    name: String,
    enc: Encoder,
    keep: Duration,
    pin: bool,
    until: Option<Instant>,
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
        Ok(Engine { ctx, models: Vec::new(), encoders: Vec::new(), limits: [vram, host], prompt: Vec::new() })
    }

    /// Loads the model directory `dir` (see [`crate::model::convert`]) into VRAM as `name`, kept as `load` says,
    /// making room as needed; a model of that name is unloaded first. An encoder's directory (see
    /// [`crate::encoder::convert`]) loads for [`Engine::decide`] or [`Engine::embed`], kept as `load`'s keep-alive and
    /// pin say: once its keep-alive runs out it is unloaded (it has no host tier; it loads again in a moment).
    pub fn load(&mut self, name: &str, dir: &Path, load: &Load) -> Result<()> {
        self.unload(name);
        self.expire()?;
        if encoder::Config::from_json(&std::fs::read(dir.join("config.json"))?).is_ok() {
            return self.load_encoder(name, dir, load);
        }
        let tok = Tokenizer::from_bytes(&std::fs::read(dir.join("tokenizer.tok"))?)?;
        let chat = Chat::open(&tok, dir)?.with_budget(load.budget);
        let ends = [chat.end_of_turn(), tok.special("<eos>").unwrap_or(chat.end_of_turn())];
        let need = footprint(dir, &load.settings)?;
        let (used, at) = (Instant::now(), Place::Disk);
        let mut model =
            Model { name: name.into(), dir: dir.into(), load: load.clone(), tok, chat, ends, need, used, until: None, at, vision: None, audio: None };
        model.touch(load.keep_alive);
        self.models.push(model);
        let loaded = self.resident(self.models.len() - 1);
        if loaded.is_err() {
            self.models.pop();
        }
        loaded
    }

    /// Loads the encoder in `dir` as `name`, kept as `load` says, moving decoders out to make room.
    fn load_encoder(&mut self, name: &str, dir: &Path, load: &Load) -> Result<()> {
        self.room(encoder::footprint(dir, encoder::Settings::default())?, None)?;
        let enc = Encoder::open(&self.ctx, dir, &encoder::Settings::default())?;
        let until = Instant::now().checked_add(load.keep_alive);
        self.encoders.push(Held { name: name.into(), enc, keep: load.keep_alive, pin: load.pin, until });
        Ok(())
    }

    /// Unloads model `name`, freeing its memory wherever it is; whether it was loaded.
    pub fn unload(&mut self, name: &str) -> bool {
        let before = self.models.len() + self.encoders.len();
        self.models.retain(|m| m.name != name);
        self.encoders.retain(|e| e.name != name);
        // The VRAM goes back to the system, not only to this engine's pool.
        let _ = self.ctx.trim();
        self.models.len() + self.encoders.len() < before
    }

    /// Encoder `name`, its keep-alive started again.
    fn encoder(&mut self, name: &str) -> Result<&mut Encoder> {
        let held = self.encoders.iter_mut().find(|e| e.name == name).ok_or_else(|| Error::NoModel(name.into()))?;
        held.until = Instant::now().checked_add(held.keep);
        Ok(&mut held.enc)
    }

    /// Laya model `name`'s answers to `questions` about `state`: each option's probability (see
    /// [`Encoder::decide`]).
    pub fn decide(&mut self, name: &str, state: &str, conversation: bool, questions: &[Question]) -> Result<Vec<Vec<f32>>> {
        self.expire()?;
        self.encoder(name)?.decide(state, conversation, questions)
    }

    /// Embedding model `name`'s vectors of `texts`, cut to `dims` values if asked (see [`Encoder::embed`]).
    pub fn embed(&mut self, name: &str, texts: &[&str], dims: Option<usize>) -> Result<Vec<Vec<f32>>> {
        self.expire()?;
        self.encoder(name)?.embed(texts, dims)
    }

    /// Where model `name` is, if it is loaded.
    pub fn tier(&self, name: &str) -> Option<Tier> {
        if self.encoders.iter().any(|e| e.name == name) {
            return Some(Tier::Vram);
        }
        let m = self.models.iter().find(|m| m.name == name)?;
        Some(match m.at {
            Place::Vram(_) => Tier::Vram,
            Place::Host(_) => Tier::Host,
            Place::Disk => Tier::Disk,
        })
    }

    /// The loaded models and encoders: where each is, the VRAM it takes there, and how long it has left there
    /// (`None`: no end, pinned or kept for good).
    pub fn models(&self) -> Vec<Loaded> {
        let now = Instant::now();
        let left = |m: &Model| m.until.filter(|_| !m.load.pin).map(|t| t.saturating_duration_since(now));
        let vram = |m: &Model| match m.at {
            Place::Vram(_) => m.need + m.vision.as_ref().map_or(0, Vision::bytes) + m.audio.as_ref().map_or(0, Tower::bytes),
            _ => 0,
        };
        let decoders =
            self.models.iter().map(|m| Loaded { name: m.name.clone(), tier: self.tier(&m.name).unwrap_or(Tier::Disk), vram: vram(m), left: left(m) });
        let encoders = self.encoders.iter().map(|e| Loaded {
            name: e.name.clone(),
            tier: Tier::Vram,
            vram: e.enc.bytes(),
            left: e.until.filter(|_| !e.pin).map(|t| t.saturating_duration_since(now)),
        });
        decoders.chain(encoders).collect()
    }

    /// The tokenizer of model `name`, if it is loaded.
    pub fn tokenizer(&self, name: &str) -> Option<&Tokenizer> {
        self.models.iter().find(|m| m.name == name).map(|m| &m.tok)
    }

    /// Sends to disk every model whose keep-alive has run out, in VRAM or parked, but pinned ones, drops such
    /// encoders, and gives their VRAM back to the system. Every call does it first. A model moved out to make room
    /// keeps its VRAM in the pool for the next.
    pub fn expire(&mut self) -> Result<()> {
        let now = Instant::now();
        let mut freed = false;
        for m in &mut self.models {
            if !matches!(m.at, Place::Disk) && !m.load.pin && m.until.is_some_and(|t| t <= now) {
                if let Place::Vram(dec) = &mut m.at {
                    dec.stop()?;
                }
                (m.vision, m.audio, m.at) = (None, None, Place::Disk);
                freed = true;
            }
        }
        let before = self.encoders.len();
        self.encoders.retain(|e| e.pin || e.until.is_none_or(|t| t > now));
        freed |= self.encoders.len() < before;
        if freed { self.ctx.trim() } else { Ok(()) }
    }

    /// Feeds the conversation of `req` to model `name` and returns its answer, as a stream: every prompt token but
    /// the last through prefill, the last through a decode step, which samples the answer's first token. The model
    /// comes into VRAM first if it is out.
    pub fn chat(&mut self, name: &str, req: &Request<'_>) -> Result<TokenStream<'_>> {
        self.expire()?;
        let i = self.models.iter().position(|m| m.name == name).ok_or_else(|| Error::NoModel(name.into()))?;
        self.resident(i)?;
        let soft = self.media(i, req)?;
        let (m, prompt) = (&mut self.models[i], &mut self.prompt);
        let slots = encode(m, req, prompt, &soft)?;
        let (tokens, context) = (prompt.len(), m.load.settings.context);
        let first = feed(m.dec(), req, prompt, &slots)?;
        let (left, keep) = (req.max_tokens.min(context - tokens), req.keep_alive.unwrap_or(m.load.keep_alive));
        Ok(TokenStream { engine: self, model: i, keep, detok: Detokenizer::default(), text: String::with_capacity(256), next: Some(first), left, tokens: 0 })
    }

    /// The soft tokens of the images and audio of `req`'s messages, in the prompt's order, by model `i`'s towers (in
    /// VRAM by now: each brought there with its first image or clip).
    fn media(&mut self, i: usize, req: &Request<'_>) -> Result<Vec<Vec<f32>>> {
        // The chat template shows a user's images and audio, and a system message's after the first.
        let shown = |(k, m): &(usize, &Message<'_>)| m.role == Role::User || m.role == Role::System && *k > 0;
        let budget = self.models[i].load.budget;
        let mut soft = Vec::new();
        for (_, m) in req.messages.iter().enumerate().filter(shown) {
            for image in m.images {
                soft.push(self.vision(i)?.soft_tokens(image, budget)?);
            }
            for clip in m.audio {
                soft.push(self.hearing(i)?.soft_tokens(clip)?);
            }
        }
        Ok(soft)
    }

    /// Model `i`'s vision tower, brought into VRAM if it is not there, under the limit like a model.
    fn vision(&mut self, i: usize) -> Result<&mut Vision> {
        let m = &self.models[i];
        if m.vision.is_none() {
            if !m.dir.join("vision.wpk").exists() || m.chat.image_token().is_none() {
                return Err(Error::Unsupported(format!("no vision tower in {}: images", m.dir.display())));
            }
            let tower = Vision::open(&self.ctx, &m.dir, m.load.budget)?;
            self.room(tower.bytes(), Some(i))?;
            self.models[i].vision = Some(tower);
        }
        Ok(self.models[i].vision.as_mut().unwrap_or_else(|| unreachable!("opened above")))
    }

    /// Model `i`'s audio tower, brought into VRAM if it is not there, under the limit like a model.
    fn hearing(&mut self, i: usize) -> Result<&mut Tower> {
        let m = &self.models[i];
        if m.audio.is_none() {
            if !m.dir.join("audio.wpk").exists() || m.chat.audio_token().is_none() {
                return Err(Error::Unsupported(format!("no audio tower in {}: audio", m.dir.display())));
            }
            let tower = Tower::open(&self.ctx, &m.dir)?;
            self.room(tower.bytes(), Some(i))?;
            self.models[i].audio = Some(tower);
        }
        Ok(self.models[i].audio.as_mut().unwrap_or_else(|| unreachable!("opened above")))
    }

    /// Brings model `i` into VRAM, moving out the least recently used models that may go until it fits.
    fn resident(&mut self, i: usize) -> Result<()> {
        if !matches!(self.models[i].at, Place::Vram(_)) {
            self.room(self.models[i].need, Some(i))?;
        }
        self.bring(i)
    }

    /// Makes room for `need` bytes of VRAM, moving out the least recently used models that may go, never `keep`.
    fn room(&mut self, need: usize, keep: Option<usize>) -> Result<()> {
        while self.used() + need > self.limits[0] {
            let may_go = |(j, m): &(usize, &Model)| Some(*j) != keep && !m.load.pin && matches!(m.at, Place::Vram(_));
            match self.models.iter().enumerate().filter(may_go).min_by_key(|(_, m)| m.used) {
                Some((j, _)) => self.evict(j)?,
                None => return Err(Error::OutOfMemory { need, free: self.limits[0].saturating_sub(self.used()) }),
            }
        }
        Ok(())
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
        (self.models[i].vision, self.models[i].audio) = (None, None);
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

    /// VRAM the resident models take, encoders included.
    fn used(&self) -> usize {
        let encoders: usize = self.encoders.iter().map(|e| e.enc.bytes()).sum();
        let towers: usize = self.models.iter().map(|m| m.vision.as_ref().map_or(0, Vision::bytes) + m.audio.as_ref().map_or(0, Tower::bytes)).sum();
        encoders + towers + self.models.iter().filter(|m| matches!(m.at, Place::Vram(_))).map(|m| m.need).sum::<usize>()
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

/// Feeds `prompt`, with soft tokens at `slots`, to `dec` from the start, sampling as `req` says: every token but
/// the last through prefill, the last through a step; starts the answer and returns its first token.
fn feed(dec: &mut Decoder, req: &Request<'_>, prompt: &[u32], slots: &[(usize, &[f32])]) -> Result<u32> {
    dec.reset(&req.sampling)?;
    let (&last, rest) = prompt.split_last().unwrap_or((&0, &[]));
    dec.prefill_with(rest, slots)?;
    let first = dec.step(last)?;
    dec.start(first)?;
    Ok(first)
}

/// `req`'s conversation as model `m`'s prompt, into `prompt`, and where each image's soft tokens `soft` go in it;
/// an error if it does not fit the context.
fn encode<'s>(m: &Model, req: &Request<'_>, prompt: &mut Vec<u32>, soft: &'s [Vec<f32>]) -> Result<Vec<(usize, &'s [f32])>> {
    prompt.clear();
    m.chat.encode(&m.tok, req.messages, req.think, prompt);
    let (tokens, context) = (prompt.len(), m.load.settings.context);
    if tokens >= context {
        return Err(Error::TooLong { tokens, context });
    }
    let width = m.vision.as_ref().map(Vision::width).or(m.audio.as_ref().map(Tower::width)).unwrap_or(0);
    slots(prompt, [m.chat.image_token(), m.chat.audio_token()], soft, width)
}

/// Where in `prompt` each image's or clip's soft tokens go: its first slot (a run of one of the `marks`), with its
/// rows of `width`. An error unless the runs and the soft tokens match one for one.
fn slots<'s>(prompt: &[u32], marks: [Option<u32>; 2], soft: &'s [Vec<f32>], width: usize) -> Result<Vec<(usize, &'s [f32])>> {
    let mut runs = Vec::new();
    for (p, &t) in prompt.iter().enumerate() {
        if marks.contains(&Some(t)) && (p == 0 || prompt[p - 1] != t) {
            runs.push((p, prompt[p..].iter().take_while(|&&u| u == t).count()));
        }
    }
    if runs.len() != soft.len() || runs.iter().zip(soft).any(|(r, s)| r.1 * width != s.len()) {
        return Err(Error::Media(format!("{} images and clips in the prompt, but {} sets of soft tokens", runs.len(), soft.len())));
    }
    Ok(runs.iter().zip(soft).map(|(r, s)| (r.0, s.as_slice())).collect())
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
