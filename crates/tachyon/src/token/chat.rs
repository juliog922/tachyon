//! Gemma 4's chat template, coded by hand: the conversation becomes tokens directly, turn markers by ID and the
//! messages as text, so text that spells a marker stays text.
//!
//! ```text
//! <bos><|turn>system\n<|think|>\n{system}<turn|>\n<|turn>user\n{text}<turn|>\n<|turn>model\n{text}<turn|>\n<|turn>model\n
//! ```
//!
//! The system turn appears when there is a system message or thinking is on; `<|think|>` switches thinking on.
//! Some models' templates (Gemma 4 12B's) open the model's turn with an empty thinking block when thinking is off,
//! `<|channel>thought\n<channel|>`: [`Chat::open`] reads the model's `chat_template.jinja` to tell.
//! Messages are trimmed, consecutive model messages share one turn, and the conversation ends by opening the
//! model's turn. A user or system message's images come before its text, each as `<|image>`, then `<|image|>` once
//! per soft token its budget gives it ([`Chat::with_budget`]), then `<image|>`: the slots the vision tower's soft
//! tokens fill. Its audio comes after its text, each clip as `<|audio>`, a `<|audio|>` per soft token (one per 40 ms,
//! [`crate::audio::count`]), then `<audio|>`. Tools are not covered.

use super::Tokenizer;
use crate::media::{Audio, DEFAULT_BUDGET, Image};
use crate::{Error, Result};

/// Who wrote a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// Instructions: as the first message, the system turn; later, a turn of its own.
    System,
    /// The person.
    User,
    /// The model.
    Model,
}

/// One message of a conversation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Message<'a> {
    /// Who wrote it.
    pub role: Role,
    /// What it says. A model message holds its answer only, without its thinking.
    pub text: &'a str,
    /// The images it shows, before its text: a user message's, or a system message's after the first (the first is
    /// the system turn, which shows none, as Gemma 4's template has it).
    pub images: &'a [Image],
    /// The audio it plays, after its text, as its images.
    pub audio: &'a [Audio],
}

impl<'a> Message<'a> {
    /// A message of text only.
    pub const fn new(role: Role, text: &'a str) -> Message<'a> {
        Message { role, text, images: &[], audio: &[] }
    }
}

/// The special tokens of Gemma 4's chat format, and the soft-token budget images get.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Chat {
    bos: u32,
    turn: u32,
    end: u32,
    think: u32,
    /// `<|image>`, `<|image|>`, `<image|>`, and the audio's `<|audio>`, `<|audio|>`, `<audio|>`, if the tokenizer has
    /// them.
    image: Option<[u32; 3]>,
    audio: Option<[u32; 3]>,
    budget: usize,
    /// `<|channel>` and `<channel|>`, when the model's turn opens with an empty thinking block if thinking is off.
    closed: Option<[u32; 2]>,
}

impl Chat {
    /// The chat format of a Gemma 4 tokenizer.
    pub fn gemma4(tok: &Tokenizer) -> Result<Chat> {
        let get = |s: &str| tok.special(s).ok_or_else(|| Error::Format(format!("tokenizer: no {s} token, not Gemma 4's")));
        let marks = |names: [&str; 3]| {
            let ids = names.map(|s| tok.special(s));
            ids.iter().all(Option::is_some).then(|| ids.map(Option::unwrap_or_default))
        };
        let (image, audio) = (marks(["<|image>", "<|image|>", "<image|>"]), marks(["<|audio>", "<|audio|>", "<audio|>"]));
        let (bos, turn, end, think) = (get("<bos>")?, get("<|turn>")?, get("<turn|>")?, get("<|think|>")?);
        Ok(Chat { bos, turn, end, think, image, audio, budget: DEFAULT_BUDGET, closed: None })
    }

    /// The chat format of the model in `dir`: [`Chat::gemma4`], as its `chat_template.jinja` opens the model's turn
    /// (see [`Chat::with_template`]); a model converted without one opens it as E4B does.
    pub fn open(tok: &Tokenizer, dir: &std::path::Path) -> Result<Chat> {
        let chat = Chat::gemma4(tok)?;
        Ok(match std::fs::read_to_string(dir.join("chat_template.jinja")) {
            Ok(template) => chat.with_template(tok, &template),
            Err(_) => chat,
        })
    }

    /// This format as the Jinja `template` opens the model's turn: with an empty thinking block when thinking is off,
    /// if its generation prompt writes one.
    #[must_use]
    pub fn with_template(self, tok: &Tokenizer, template: &str) -> Chat {
        let prompt = template.rfind("add_generation_prompt").map_or("", |at| &template[at..]);
        let marks = [tok.special("<|channel>"), tok.special("<channel|>")];
        let closed = match marks {
            [Some(open), Some(close)] if prompt.contains(r"'<|channel>thought\n<channel|>'") => Some([open, close]),
            _ => None,
        };
        Chat { closed, ..self }
    }

    /// This format with images at `budget` soft tokens (one of [`crate::media::BUDGETS`]; 280 by default).
    #[must_use]
    pub fn with_budget(self, budget: usize) -> Chat {
        Chat { budget, ..self }
    }

    /// The token of an image's soft-token slots, if the tokenizer has one.
    pub fn image_token(&self) -> Option<u32> {
        self.image.map(|i| i[1])
    }

    /// The token of an audio clip's soft-token slots, if the tokenizer has one.
    pub fn audio_token(&self) -> Option<u32> {
        self.audio.map(|i| i[1])
    }

    /// The token that ends the model's turn, where generation stops.
    pub fn end_of_turn(&self) -> u32 {
        self.end
    }

    /// Appends the tokens of `messages`, then the opening of the model's turn, to `out`; `think` lets the model
    /// think before it answers.
    pub fn encode(&self, tok: &Tokenizer, messages: &[Message<'_>], think: bool, out: &mut Vec<u32>) {
        out.push(self.bos);
        let (system, rest) = match messages.split_first() {
            Some((m, rest)) if m.role == Role::System => (Some(m.text), rest),
            _ => (None, messages),
        };
        if system.is_some() || think {
            out.push(self.turn);
            let mut text = String::from("system\n");
            if think {
                tok.encode(&text, out);
                out.push(self.think);
                text = String::from("\n");
            }
            text.push_str(system.unwrap_or_default().trim());
            self.close(tok, &text, out);
        }
        let mut model: Option<String> = None;
        for m in rest {
            if m.role == Role::Model {
                model.get_or_insert_default().push_str(m.text.trim());
                continue;
            }
            self.model_turn(tok, model.take(), out);
            self.turn(tok, m, out);
        }
        self.model_turn(tok, model, out);
        out.push(self.turn);
        tok.encode("model\n", out);
        if let Some([open, close]) = self.closed.filter(|_| !think) {
            out.push(open);
            tok.encode("thought\n", out);
            out.push(close);
        }
    }

    /// A user or system message's turn: its images' slots, its text, its audio's slots.
    fn turn(&self, tok: &Tokenizer, m: &Message<'_>, out: &mut Vec<u32>) {
        out.push(self.turn);
        let mut text = String::from(if m.role == Role::System { "system\n" } else { "user\n" });
        if let Some(marks) = self.image.filter(|_| !m.images.is_empty()) {
            tok.encode(&std::mem::take(&mut text), out);
            m.images.iter().for_each(|image| slots(marks, image.soft_tokens(self.budget).unwrap_or(0), out));
        }
        text.push_str(m.text.trim());
        if let Some(marks) = self.audio.filter(|_| !m.audio.is_empty()) {
            tok.encode(&std::mem::take(&mut text), out);
            m.audio.iter().for_each(|clip| slots(marks, crate::audio::count(clip), out));
        }
        self.close(tok, &text, out);
    }

    /// The model's messages since the last other one, if any, as one turn.
    fn model_turn(&self, tok: &Tokenizer, text: Option<String>, out: &mut Vec<u32>) {
        if let Some(text) = text {
            out.push(self.turn);
            self.close(tok, &format!("model\n{text}"), out);
        }
    }

    /// `text`, then the end of the turn.
    fn close(&self, tok: &Tokenizer, text: &str, out: &mut Vec<u32>) {
        tok.encode(text, out);
        out.push(self.end);
        tok.encode("\n", out);
    }
}

/// `n` slots between the opening and closing marks `[open, slot, close]`.
fn slots([open, slot, close]: [u32; 3], n: usize, out: &mut Vec<u32>) {
    out.push(open);
    out.extend(std::iter::repeat_n(slot, n));
    out.push(close);
}
