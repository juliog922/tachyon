//! Laya's questions as the encoder reads them, the way the `laya` package (0.4.5) builds them: one sequence per
//! question, `[CLS] <kind> question: <instructions> [SEP] [MASK] option [MASK] option … [SEP] state [SEP]`, and the
//! probabilities of its options from their logits.
//!
//! The question and its options get at most [`Budget::head`] tokens, each option at most [`OPTION_TOKENS`] after its
//! `[MASK]`; options that overrun it are cut to an equal share, and the instructions to what is left (at least 8
//! tokens). The state fills the rest of [`Budget::total`]: its start is kept, or for a conversation its end.

use crate::json::Json;
use crate::token::Tokenizer;
use crate::{Error, Result};

/// The most tokens an option's text keeps.
pub const OPTION_TOKENS: usize = 48;

/// What a question asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// One option of several.
    Choice,
    /// A level on an ordered scale.
    Score,
    /// Whether a statement holds: two options, false and true.
    Noul,
}

impl Kind {
    /// The name Laya writes before the instructions, and the type embedding's index.
    fn name(self) -> &'static str {
        ["choice", "score", "noul"][self as usize]
    }
}

/// A question about a state. `options` are `(label, description)` pairs: for [`Kind::Choice`] the labels to choose
/// from; for [`Kind::Score`] the levels, lowest first, with empty labels; for [`Kind::Noul`] none, or
/// `[("false", …), ("true", …)]` to describe the two answers. An empty description is left out (a choice) or replaced
/// by Laya's default (a noul).
#[derive(Debug, Clone, Copy)]
pub struct Question<'a> {
    /// What it asks for.
    pub kind: Kind,
    /// The question itself.
    pub instructions: &'a str,
    /// Its options.
    pub options: &'a [(&'a str, &'a str)],
}

impl Question<'_> {
    /// The options' texts, in order: `label: description`, `level i: description`, or `false: …` and `true: …`.
    pub fn rendered(&self) -> Vec<String> {
        let described = |label: &str, text: &str| if text.is_empty() { label.to_string() } else { format!("{label}: {text}") };
        match self.kind {
            Kind::Choice => self.options.iter().map(|&(l, d)| described(l, d)).collect(),
            Kind::Score => self.options.iter().enumerate().map(|(i, &(_, d))| format!("level {i}: {d}")).collect(),
            Kind::Noul => {
                let text = |label, default| {
                    let d = self.options.iter().find(|o| o.0 == label).map_or("", |o| o.1);
                    format!("{label}: {}", if d.is_empty() { default } else { d })
                };
                vec![text("false", "no, the statement does not hold"), text("true", "yes, the statement holds")]
            }
        }
    }
}

/// Token budgets of a sequence: in all, and for the question with its options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Budget {
    /// The most tokens a sequence holds (`max_len`).
    pub total: usize,
    /// The most the question and its options take (`head_max_len`).
    pub head: usize,
}

/// The tokens that frame a sequence: `[CLS]`, `[SEP]` and `[MASK]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Frame {
    /// Opens the sequence.
    pub cls: u32,
    /// Closes each part.
    pub sep: u32,
    /// Marks each option.
    pub mask: u32,
}

/// Builds Laya's sequences with one checkpoint's tokenizer, frame and budget.
#[derive(Debug, Clone, Copy)]
pub struct Builder<'t> {
    /// The checkpoint's tokenizer.
    pub tok: &'t Tokenizer,
    /// Its framing tokens.
    pub frame: Frame,
    /// Its budgets.
    pub budget: Budget,
}

impl Builder<'_> {
    /// The tokens of `text` with any `[MASK]` text in it replaced by a space, as Laya does.
    pub fn tokens(&self, text: &str) -> Vec<u32> {
        let mask = String::from_utf8_lossy(self.tok.bytes(self.frame.mask)).into_owned();
        let mut ids = Vec::new();
        self.tok.encode(&text.replace(&mask, " "), &mut ids);
        ids
    }

    /// The sequence of question `q` about the state whose tokens are `state` (see [`Builder::tokens`]), into `ids`,
    /// and the position of each option's `[MASK]`. A conversation (`keep_end`) loses its start when it does not fit,
    /// other states their end. An error if some option does not fit in [`Budget::total`].
    pub fn sequence(&self, q: &Question, state: &[u32], keep_end: bool, ids: &mut Vec<u32>) -> Result<Vec<usize>> {
        let (head, options) = self.head(q);
        ids.clear();
        ids.push(self.frame.cls);
        ids.extend(head);
        ids.push(self.frame.sep);
        let mut markers = Vec::with_capacity(options.len());
        for o in &options {
            markers.push(ids.len());
            ids.extend(o);
        }
        ids.push(self.frame.sep);
        let room = self.budget.total.saturating_sub(ids.len() + 1).min(state.len());
        ids.extend(if keep_end { &state[state.len() - room..] } else { &state[..room] });
        ids.push(self.frame.sep);
        ids.truncate(self.budget.total);
        if markers.last().is_some_and(|&m| m >= self.budget.total) {
            return Err(Error::TooLong { tokens: markers[markers.len() - 1] + 1, context: self.budget.total });
        }
        Ok(markers)
    }

    /// The instructions' tokens and each option's, `[MASK]` first, cut to [`Budget::head`].
    fn head(&self, q: &Question) -> (Vec<u32>, Vec<Vec<u32>>) {
        let mut options: Vec<Vec<u32>> = q
            .rendered()
            .iter()
            .map(|o| {
                let mut t = self.tokens(&format!(" {o}"));
                t.truncate(OPTION_TOKENS);
                t.insert(0, self.frame.mask);
                t
            })
            .collect();
        let used = |o: &[Vec<u32>]| o.iter().map(Vec::len).sum::<usize>();
        if self.budget.head < used(&options) + 16 {
            let each = 4.max(self.budget.head.saturating_sub(16) / options.len().max(1));
            for o in &mut options {
                o.truncate(each);
            }
        }
        let mut head = self.tokens(&format!("{} question: {}", q.kind.name(), q.instructions));
        head.truncate(8.max(self.budget.head.saturating_sub(used(&options))));
        (head, options)
    }
}

/// A Laya checkpoint's decision head and how it reads questions, from its `rl_agent_config.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct Head {
    /// Transformer layers between the encoder and the scorer.
    pub layers: usize,
    /// The sequences' budgets.
    pub budget: Budget,
    /// Temperature of each [`Kind`].
    temperature: [f32; 3],
    /// Temperatures by kind and number of options, such as `choice:3-5`.
    by_options: Vec<(String, f32)>,
}

impl Head {
    /// Reads `rl_agent_config.json`.
    pub fn from_json(json: &[u8]) -> Result<Head> {
        let c = Json::parse(json)?;
        let num = |k: &str, default: f64| c.get(k).and_then(Json::num).unwrap_or(default) as usize;
        let mut temperature = [1.0; 3];
        for (t, v) in temperature.iter_mut().zip(c.get("temperature").and_then(Json::arr).unwrap_or_default()) {
            *t = v.num().map_or(1.0, |v| v as f32);
        }
        let by_options = c.get("temperature_by_options").and_then(Json::obj).unwrap_or_default();
        Ok(Head {
            layers: num("head_layers", 2.0),
            budget: Budget { total: num("max_len", 512.0), head: num("head_max_len", 192.0) },
            temperature,
            by_options: by_options.iter().map(|(k, v)| (k.clone(), v.num().map_or(1.0, |v| v as f32))).collect(),
        })
    }

    /// The temperature of a `kind` question with `options` options: the fitted one for its size if there is one,
    /// else its kind's, kept within 0.5 to 5 as the `laya` package does (a lower one would sharpen, not soften).
    pub fn temperature(&self, kind: Kind, options: usize) -> f32 {
        let size = match options {
            ..=2 => "2",
            3..=5 => "3-5",
            6..=10 => "6-10",
            _ => "11+",
        };
        let key = format!("{}:{size}", kind.name());
        let t = self.by_options.iter().find(|(k, _)| *k == key).map_or(self.temperature[kind as usize], |(_, t)| *t);
        if t.is_finite() { t.clamp(0.5, 5.0) } else { 1.0 }
    }
}

/// The probabilities of a question's options from their logits, at temperature `t`.
pub fn probabilities(logits: &[f32], t: f32) -> Vec<f32> {
    let top = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = logits.iter().map(|&l| ((l - top) / t).exp()).collect();
    let sum: f32 = e.iter().sum();
    e.iter().map(|v| v / sum).collect()
}
