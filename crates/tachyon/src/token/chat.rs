//! Gemma 4's chat template, coded by hand: the conversation becomes tokens directly, turn markers by ID and the
//! messages as text, so text that spells a marker stays text.
//!
//! ```text
//! <bos><|turn>system\n<|think|>\n{system}<turn|>\n<|turn>user\n{text}<turn|>\n<|turn>model\n{text}<turn|>\n<|turn>model\n
//! ```
//!
//! The system turn appears when there is a system message or thinking is on; `<|think|>` switches thinking on.
//! Messages are trimmed, consecutive model messages share one turn, and the conversation ends by opening the
//! model's turn. A user or system message's images come before its text, each as `<|image>`, then `<|image|>` once
//! per soft token its budget gives it ([`Chat::with_budget`]), then `<image|>`: the slots the vision tower's soft
//! tokens fill. Tools are not covered.

use super::Tokenizer;
use crate::media::{DEFAULT_BUDGET, Image};
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Message<'a> {
    /// Who wrote it.
    pub role: Role,
    /// What it says. A model message holds its answer only, without its thinking.
    pub text: &'a str,
    /// The images it shows, before its text: a user message's, or a system message's after the first (the first is
    /// the system turn, which shows none, as Gemma 4's template has it).
    pub images: &'a [Image],
}

impl<'a> Message<'a> {
    /// A message of text only.
    pub const fn new(role: Role, text: &'a str) -> Message<'a> {
        Message { role, text, images: &[] }
    }
}

/// The special tokens of Gemma 4's chat format, and the soft-token budget images get.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Chat {
    bos: u32,
    turn: u32,
    end: u32,
    think: u32,
    /// `<|image>`, `<|image|>`, `<image|>`, if the tokenizer has them.
    image: Option<[u32; 3]>,
    budget: usize,
}

impl Chat {
    /// The chat format of a Gemma 4 tokenizer.
    pub fn gemma4(tok: &Tokenizer) -> Result<Chat> {
        let get = |s: &str| tok.special(s).ok_or_else(|| Error::Format(format!("tokenizer: no {s} token, not Gemma 4's")));
        let image = ["<|image>", "<|image|>", "<image|>"].map(|s| tok.special(s));
        let image = image.iter().all(Option::is_some).then(|| image.map(Option::unwrap_or_default));
        Ok(Chat { bos: get("<bos>")?, turn: get("<|turn>")?, end: get("<turn|>")?, think: get("<|think|>")?, image, budget: DEFAULT_BUDGET })
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
    }

    /// A user or system message's turn: its images' slots, then its text.
    fn turn(&self, tok: &Tokenizer, m: &Message<'_>, out: &mut Vec<u32>) {
        out.push(self.turn);
        let role = if m.role == Role::System { "system" } else { "user" };
        let Some([boi, slot, eoi]) = self.image.filter(|_| !m.images.is_empty()) else {
            return self.close(tok, &format!("{role}\n{}", m.text.trim()), out);
        };
        tok.encode(&format!("{role}\n"), out);
        for image in m.images {
            out.push(boi);
            out.extend(std::iter::repeat_n(slot, image.soft_tokens(self.budget).unwrap_or(0)));
            out.push(eoi);
        }
        self.close(tok, m.text.trim(), out);
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
