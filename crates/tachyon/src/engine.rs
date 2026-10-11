//! The engine: a model on the GPU behind a blocking chat API, for one caller at a time.
//!
//! [`Engine::chat`] puts the conversation through the chat template, feeds the prompt and returns a [`TokenStream`],
//! which hands out the answer's text as it is generated, on the caller's thread: v1 serves one waiting caller, so
//! there is no request thread, queue or async runtime. Each call to [`TokenStream::next_text`] waits for one step
//! while the next one runs, and allocates nothing. Dropping the stream cancels the answer at the cost of that one
//! queued step.
//!
//! ```no_run
//! use tachyon::engine::{Engine, Request};
//! use tachyon::token::{Message, Role};
//!
//! let mut engine = Engine::open("gemma-4-e4b".as_ref(), &Default::default())?;
//! let messages = [Message { role: Role::User, text: "Why is the sky blue?" }];
//! let mut answer = engine.chat(&Request::new(&messages))?;
//! while let Some(text) = answer.next_text()? {
//!     print!("{text}");
//! }
//! # Ok::<(), tachyon::Error>(())
//! ```

use crate::Result;
use crate::cuda::Context;
use crate::model::{Decoder, Settings};
use crate::ptx::Sampling;
use crate::token::{Chat, Detokenizer, Message, Tokenizer};
use std::path::Path;

/// A model loaded on GPU 0, ready to chat.
pub struct Engine {
    dec: Decoder,
    tok: Tokenizer,
    chat: Chat,
    /// The tokens that end an answer: the end of the turn, and the end of the text.
    ends: [u32; 2],
    context: usize,
    prompt: Vec<u32>,
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
}

impl<'a> Request<'a> {
    /// A request to answer `messages` greedily, without thinking, until the model ends its turn.
    pub fn new(messages: &'a [Message<'a>]) -> Request<'a> {
        Request { messages, think: false, sampling: Sampling::default(), max_tokens: usize::MAX }
    }
}

impl Engine {
    /// Loads the model directory `dir` (see [`crate::model::convert`]) onto GPU 0.
    pub fn open(dir: &Path, settings: &Settings) -> Result<Engine> {
        let tok = Tokenizer::from_bytes(&std::fs::read(dir.join("tokenizer.tok"))?)?;
        let chat = Chat::gemma4(&tok)?;
        let ends = [chat.end_of_turn(), tok.special("<eos>").unwrap_or(chat.end_of_turn())];
        let dec = Decoder::open(&Context::new(0)?, dir, settings)?;
        Ok(Engine { dec, tok, chat, ends, context: settings.context, prompt: Vec::new() })
    }

    /// The tokenizer.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tok
    }

    /// Feeds the prompt of `req` to the model and returns its answer, as a stream; the prompt is read one token per
    /// step, so this takes the time of as many steps.
    pub fn chat(&mut self, req: &Request<'_>) -> Result<TokenStream<'_>> {
        self.prompt.clear();
        self.chat.encode(&self.tok, req.messages, req.think, &mut self.prompt);
        let (tokens, context) = (self.prompt.len(), self.context);
        if tokens >= context {
            return Err(crate::Error::TooLong { tokens, context });
        }
        self.dec.reset(&req.sampling)?;
        let mut first = 0;
        for &t in &self.prompt {
            first = self.dec.step(t)?;
        }
        self.dec.start(first)?;
        let left = req.max_tokens.min(context - tokens);
        Ok(TokenStream { engine: self, detok: Detokenizer::default(), text: String::with_capacity(256), next: Some(first), left, tokens: 0 })
    }
}

/// An answer being generated; see [`Engine::chat`].
pub struct TokenStream<'e> {
    engine: &'e mut Engine,
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
            let e = &mut *self.engine;
            let token = match self.next.take() {
                Some(t) => t,
                None => e.dec.next_token()?,
            };
            if e.ends.contains(&token) {
                self.left = 0;
                break;
            }
            (self.left, self.tokens) = (self.left - 1, self.tokens + 1);
            self.detok.push(&e.tok, token, &mut self.text);
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
    /// Cancels the answer: waits for the step still queued.
    fn drop(&mut self) {
        let _ = self.engine.dec.stop();
    }
}