//! Gemma 4's chat template, coded by hand: the conversation becomes tokens directly, turn markers by ID and the
//! messages as text, so text that spells a marker stays text.
//!
//! ```text
//! <bos><|turn>system\n<|think|>\n{system}<turn|>\n<|turn>user\n{text}<turn|>\n<|turn>model\n{text}<turn|>\n<|turn>model\n
//! ```
//!
//! The system turn appears when there is a system message or thinking is on; `<|think|>` switches thinking on.
//! Messages are trimmed, consecutive model messages share one turn, and the conversation ends by opening the
//! model's turn. Tools and images are not covered.

use super::Tokenizer;
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
}

/// The special tokens of Gemma 4's chat format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Chat {
    bos: u32,
    turn: u32,
    end: u32,
    think: u32,
}

impl Chat {
    /// The chat format of a Gemma 4 tokenizer.
    pub fn gemma4(tok: &Tokenizer) -> Result<Chat> {
        let get = |s: &str| tok.special(s).ok_or_else(|| Error::Format(format!("tokenizer: no {s} token, not Gemma 4's")));
        Ok(Chat { bos: get("<bos>")?, turn: get("<|turn>")?, end: get("<turn|>")?, think: get("<|think|>")? })
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
            out.push(self.turn);
            let role = if m.role == Role::System { "system" } else { "user" };
            self.close(tok, &format!("{role}\n{}", m.text.trim()), out);
        }
        self.model_turn(tok, model, out);
        out.push(self.turn);
        tok.encode("model\n", out);
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