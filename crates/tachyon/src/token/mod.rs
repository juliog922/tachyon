//! Text to token IDs and back, for Gemma's SentencePiece-style BPE: spaces become `▁`, the text is split into
//! characters, characters the vocabulary lacks fall back to their UTF-8 bytes (`<0xHH>` tokens), and adjacent
//! tokens merge, lowest-ranked pair first, until no merge applies. IDs match Hugging Face `tokenizers` on the same
//! `tokenizer.json`.
//!
//! [`Tokenizer::from_json`] reads a `tokenizer.json` once, at conversion; [`Tokenizer::to_bytes`] and
//! [`Tokenizer::from_bytes`] store and load the result, whose lookup tables are ready to use as read: loading is a
//! copy, not a rebuild. [`Detokenizer`] turns IDs back into text as they are generated, whole characters only.
//!
//! Text is encoded literally: a special token's text in a prompt stays text. Special tokens enter only by ID,
//! from the chat template, so user text cannot forge a turn.

mod chat;

pub use chat::{Chat, Message, Role};

use crate::json::Json;
use crate::{Error, Result};
use std::collections::{BinaryHeap, HashMap};

/// No entry in a lookup table, and no token.
const NONE: u32 = u32::MAX;
const MAGIC: &[u8; 8] = b"TACHYTOK";
/// Flag: the text's spaces are encoded as `▁`.
const META: u32 = 1;
/// Flag: characters outside the vocabulary become `<0xHH>` byte tokens.
const BYTES: u32 = 2;
/// Characters with a direct entry in [`Tokenizer::chars`]: the first two Unicode planes.
const DIRECT: usize = 0x20000;

/// A BPE vocabulary with its merges.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tokenizer {
    /// [`META`], [`BYTES`].
    flags: u32,
    /// The unknown token, or [`NONE`].
    unk: u32,
    /// Where each token's text starts in `text`; one more entry than tokens.
    offsets: Vec<u32>,
    /// Every token's text as decoding produces it: `▁` already a space, byte tokens already their byte.
    text: Vec<u8>,
    /// Special tokens' IDs, sorted.
    special: Vec<u32>,
    /// The token of each byte.
    bytes: Vec<u32>,
    /// The token of each character below [`DIRECT`], or [`NONE`].
    chars: Vec<u32>,
    /// `(char, id)` pairs of the single-character tokens above it.
    rare: Vec<u32>,
    /// Characters some token joins to a following `▁`, such as `>` in `>▁</`: no cut after them.
    joins: Vec<u32>,
    /// Open addressing table of `(left, right, rank, merged)`.
    merges: Vec<u32>,
}

/// Slot of `key` in a table of `mask + 1` entries.
fn slot(key: u64, mask: usize) -> usize {
    (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) as usize & mask
}

/// The entry of the merge table for `(a, b)`, or the free one where it belongs, and its index.
fn find(table: &[u32], a: u32, b: u32) -> (usize, &[u32]) {
    let mask = table.len() / 4 - 1;
    let mut i = slot(u64::from(a) << 32 | u64::from(b), mask);
    loop {
        let e = &table[4 * i..][..4];
        if e[0] == NONE || (e[0], e[1]) == (a, b) {
            return (4 * i, e);
        }
        i = (i + 1) & mask;
    }
}

impl Tokenizer {
    /// Reads a Hugging Face `tokenizer.json` of the kind Gemma ships: BPE, optionally with byte fallback, no
    /// pre-tokenizer, and either no normalizer or one that replaces spaces with `▁`.
    pub fn from_json(json: &[u8]) -> Result<Tokenizer> {
        let doc = Json::parse(json)?;
        let model = doc.get("model").filter(|m| m.get("type").and_then(Json::str) == Some("BPE")).ok_or_else(|| unsupported("a model other than BPE"))?;
        let meta = spaces_to_meta(doc.get("normalizer"))?;
        // Gemma splits on spaces after its normalizer has turned them all into `▁`: it never splits.
        let split = doc.get("pre_tokenizer").and_then(|p| p.get("pattern")).and_then(|p| p.get("String")).and_then(Json::str) == Some(" ");
        if doc.get("pre_tokenizer").is_some_and(|p| *p != Json::Null) && !(meta && split) {
            return Err(unsupported("this pre-tokenizer"));
        }
        let fallback = model.get("byte_fallback").and_then(Json::bool) == Some(true);
        let mut vocab: Vec<(&str, u32)> =
            model.get("vocab").and_then(Json::obj).unwrap_or_default().iter().map(|(k, v)| (k.as_str(), v.num().unwrap_or(-1.0) as u32)).collect();
        let special = added(doc.get("added_tokens"), &mut vocab);
        let merges: Vec<(&str, &str)> = model.get("merges").and_then(Json::arr).unwrap_or_default().iter().filter_map(pair).collect();
        let unk = model.get("unk_token").and_then(Json::str);
        Tokenizer::build(&vocab, &merges, special, unk, (u32::from(meta) * META) | (u32::from(fallback) * BYTES))
    }

    fn build(vocab: &[(&str, u32)], merges: &[(&str, &str)], mut special: Vec<u32>, unk: Option<&str>, flags: u32) -> Result<Tokenizer> {
        let n = vocab.iter().map(|&(_, id)| id as usize + 1).max().unwrap_or(0);
        let mut names = vec![""; n];
        for &(s, id) in vocab {
            names[id as usize] = s;
        }
        let ids: HashMap<&str, u32> = vocab.iter().copied().collect();
        special.sort_unstable();
        let mut t = Tokenizer {
            flags,
            unk: unk.and_then(|u| ids.get(u)).copied().unwrap_or(NONE),
            offsets: vec![0],
            text: Vec::new(),
            special,
            bytes: vec![NONE; 256],
            chars: vec![NONE; DIRECT],
            rare: Vec::new(),
            joins: Vec::new(),
            merges: vec![NONE; 4 * (2 * merges.len()).next_power_of_two().max(2)],
        };
        for (id, name) in names.iter().enumerate() {
            t.add_token(id as u32, name);
        }
        let plain = names.iter().enumerate().filter(|&(id, _)| !t.is_special(id as u32));
        let pairs = plain.flat_map(|(_, name)| name.chars().zip(name.chars().skip(1)));
        t.joins = pairs.filter(|&(c, d)| d == '▁' && c != '▁').map(|(c, _)| c as u32).collect();
        t.joins.sort_unstable();
        t.joins.dedup();
        for (rank, &(a, b)) in merges.iter().enumerate() {
            let get = |s: &str| ids.get(s).copied().ok_or_else(|| Error::Format(format!("tokenizer: merge \"{a}\" + \"{b}\" names an unknown token")));
            let key = [get(a)?, get(b)?, rank as u32, get(&format!("{a}{b}"))?];
            let (at, e) = find(&t.merges, key[0], key[1]);
            if e[0] == NONE {
                t.merges[at..at + 4].copy_from_slice(&key);
            }
        }
        Ok(t)
    }

    /// Records token `id`: its decoded text, and its place in the byte and character tables.
    fn add_token(&mut self, id: u32, name: &str) {
        let byte = name.strip_prefix("<0x").and_then(|h| h.strip_suffix('>')).filter(|h| h.len() == 2).and_then(|h| u8::from_str_radix(h, 16).ok());
        let special = self.special.binary_search(&id).is_ok();
        match byte {
            Some(b) if self.flags & BYTES != 0 && !special => {
                self.bytes[usize::from(b)] = id;
                self.text.push(b);
            }
            _ if self.flags & META != 0 && !special => self.text.extend_from_slice(name.replace('▁', " ").as_bytes()),
            _ => self.text.extend_from_slice(name.as_bytes()),
        }
        self.offsets.push(self.text.len() as u32);
        let mut chars = name.chars();
        match (chars.next(), chars.next(), special) {
            (Some(c), None, false) if (c as usize) < DIRECT => self.chars[c as usize] = id,
            (Some(c), None, false) => self.rare.extend([c as u32, id]),
            _ => {}
        }
    }

    /// The number of tokens.
    pub fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Whether the vocabulary is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The text token `id` decodes to, as bytes: a byte token is one byte, which may be part of a character.
    pub fn bytes(&self, id: u32) -> &[u8] {
        let i = id as usize;
        self.offsets.get(i..i + 2).map_or(&[], |o| &self.text[o[0] as usize..o[1] as usize])
    }

    /// Whether `id` is a special token, such as a turn marker.
    pub fn is_special(&self, id: u32) -> bool {
        self.special.binary_search(&id).is_ok()
    }

    /// The special token written `text`, such as `<bos>`.
    pub fn special(&self, text: &str) -> Option<u32> {
        self.special.iter().copied().find(|&id| self.bytes(id) == text.as_bytes())
    }

    fn char_id(&self, c: char) -> u32 {
        self.chars.get(c as usize).copied().unwrap_or_else(|| self.rare.chunks_exact(2).find(|p| p[0] == c as u32).map_or(NONE, |p| p[1]))
    }

    /// The rank and result of merging `a` and `b`.
    fn merge(&self, a: u32, b: u32) -> Option<(u32, u32)> {
        let e = find(&self.merges, a, b).1;
        (e[0] != NONE).then(|| (e[2], e[3]))
    }

    /// Appends the tokens of `text` to `out`. The text is cut before each space that follows a character no token
    /// joins to a following `▁`, so no merge can cross a cut, and each piece is merged on its own.
    pub fn encode(&self, text: &str, out: &mut Vec<u32>) {
        let (mut work, mut start, mut last) = (Work::default(), 0, '▁');
        for (i, c) in text.char_indices() {
            let space = c == ' ' || c == '▁';
            if space && self.flags & META != 0 && last != ' ' && last != '▁' && !self.joins.contains(&(last as u32)) {
                self.merge_all(&text[start..i], &mut work, out);
                start = i;
            }
            last = c;
        }
        self.merge_all(&text[start..], &mut work, out);
    }

    /// Merges the symbols of `text`, lowest-ranked pair first, and appends the result to `out`.
    fn merge_all(&self, text: &str, w: &mut Work, out: &mut Vec<u32>) {
        self.symbols(text, &mut w.ids);
        let n = w.ids.len() as u32;
        w.next.clear();
        w.next.extend(1..=n);
        w.prev.clear();
        w.prev.extend((0..n).map(|i| i.wrapping_sub(1)));
        w.heap.clear();
        (0..n.saturating_sub(1)).for_each(|i| self.queue(w, i));
        while let Some(std::cmp::Reverse((top, a, b, merged))) = w.heap.pop() {
            let i = top as u32 as usize;
            let j = w.next[i] as usize;
            if w.ids[i] != a || w.ids.get(j) != Some(&b) {
                continue;
            }
            (w.ids[i], w.ids[j], w.next[i]) = (merged, NONE, w.next[j]);
            if let Some(after) = w.prev.get_mut(w.next[i] as usize) {
                *after = i as u32;
            }
            self.queue(w, w.prev[i]);
            self.queue(w, i as u32);
        }
        out.extend(w.ids.iter().copied().filter(|&id| id != NONE));
    }

    /// Queues the merge of symbol `i` with the next one, if there is one.
    fn queue(&self, w: &mut Work, i: u32) {
        let j = w.next.get(i as usize).copied().unwrap_or(NONE);
        if let (Some(&a), Some(&b)) = (w.ids.get(i as usize), w.ids.get(j as usize))
            && let Some((rank, merged)) = self.merge(a, b)
        {
            w.heap.push(std::cmp::Reverse((u64::from(rank) << 32 | u64::from(i), a, b, merged)));
        }
    }

    /// The tokens before merging, into `ids`: one per character, or per byte of a character the vocabulary lacks.
    fn symbols(&self, text: &str, ids: &mut Vec<u32>) {
        ids.clear();
        for c in text.chars() {
            let c = if c == ' ' && self.flags & META != 0 { '▁' } else { c };
            let id = self.char_id(c);
            if id != NONE {
                ids.push(id);
                continue;
            }
            let mut buf = [0; 4];
            let bytes: Option<Vec<u32>> =
                c.encode_utf8(&mut buf).bytes().map(|b| Some(self.bytes[usize::from(b)]).filter(|&id| id != NONE && self.flags & BYTES != 0)).collect();
            match bytes {
                Some(b) => ids.extend(b),
                None if ids.last() != Some(&self.unk) && self.unk != NONE => ids.push(self.unk),
                None => {}
            }
        }
    }

    /// The tokenizer as bytes, for [`Tokenizer::from_bytes`].
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = MAGIC.to_vec();
        let text: Vec<u32> = self.text.chunks(4).map(|c| c.iter().rev().fold(0, |w, &b| w << 8 | u32::from(b))).collect();
        let head = [self.flags, self.unk, self.text.len() as u32];
        for section in [&head[..], &self.offsets, &text, &self.special, &self.bytes, &self.chars, &self.rare, &self.joins, &self.merges] {
            out.extend((section.len() as u32).to_le_bytes());
            for w in section {
                out.extend(w.to_le_bytes());
            }
        }
        out
    }

    /// A tokenizer stored by [`Tokenizer::to_bytes`].
    pub fn from_bytes(data: &[u8]) -> Result<Tokenizer> {
        let mut words = data.strip_prefix(MAGIC).ok_or_else(damaged)?.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]));
        let mut s: Vec<Vec<u32>> = (0..9).map(|_| section(&mut words)).collect::<Result<_>>()?;
        let [head, offsets, text, special, bytes, chars, rare, joins, merges] = std::array::from_fn(|i| std::mem::take(&mut s[i]));
        let mut text: Vec<u8> = text.iter().flat_map(|w| w.to_le_bytes()).collect();
        text.truncate(*head.get(2).ok_or_else(damaged)? as usize);
        let t = Tokenizer { flags: head[0], unk: head[1], offsets, text, special, bytes, chars, rare, joins, merges };
        let sound = t.offsets.last() == Some(&(t.text.len() as u32)) && t.bytes.len() == 256 && t.chars.len() == DIRECT;
        if sound && t.merges.len().is_power_of_two() { Ok(t) } else { Err(damaged()) }
    }
}

fn damaged() -> Error {
    Error::Format("tokenizer: damaged".into())
}

/// A length-prefixed run of words.
fn section(words: &mut impl Iterator<Item = u32>) -> Result<Vec<u32>> {
    let n = words.next().ok_or_else(damaged)? as usize;
    let s: Vec<u32> = words.take(n).collect();
    if s.len() == n { Ok(s) } else { Err(damaged()) }
}

/// Whether the normalizer turns spaces into `▁`; none is fine, any other is not supported.
fn spaces_to_meta(normalizer: Option<&Json>) -> Result<bool> {
    match normalizer {
        None | Some(Json::Null) => Ok(false),
        Some(n) if n.get("type").and_then(Json::str) == Some("Replace") && n.get("content").and_then(Json::str) == Some("▁") => Ok(true),
        Some(_) => Err(unsupported("this normalizer")),
    }
}

/// Adds the added tokens to `vocab`; returns the special ones' IDs.
fn added<'j>(tokens: Option<&'j Json>, vocab: &mut Vec<(&'j str, u32)>) -> Vec<u32> {
    let mut special = Vec::new();
    for t in tokens.and_then(Json::arr).unwrap_or_default() {
        let (content, id) = (t.get("content").and_then(Json::str).unwrap_or_default(), t.get("id").and_then(Json::num).unwrap_or(-1.0) as u32);
        vocab.push((content, id));
        if t.get("special").and_then(Json::bool) == Some(true) {
            special.push(id);
        }
    }
    special
}

/// The buffers [`Tokenizer::encode`] reuses from piece to piece.
#[derive(Default)]
struct Work {
    ids: Vec<u32>,
    next: Vec<u32>,
    prev: Vec<u32>,
    /// Merges by rank, then position: `(rank << 32 | position, left, right, merged)`.
    heap: BinaryHeap<std::cmp::Reverse<(u64, u32, u32, u32)>>,
}

/// A merge as `tokenizer.json` writes it: `["a", "b"]`, or `"a b"` in older files.
fn pair(m: &Json) -> Option<(&str, &str)> {
    match m {
        Json::Arr(p) => Some((p.first()?.str()?, p.get(1)?.str()?)),
        Json::Str(s) => s.split_once(' '),
        _ => None,
    }
}

fn unsupported(what: &str) -> Error {
    Error::Format(format!("tokenizer: {what} is not supported"))
}

/// Turns generated tokens back into text as they come, whole characters only: a character split across byte
/// tokens appears once its last byte arrives. Special tokens add nothing.
#[derive(Debug, Default)]
pub struct Detokenizer {
    pending: Vec<u8>,
}

impl Detokenizer {
    /// Appends to `out` the text that token `id` completes.
    pub fn push(&mut self, tok: &Tokenizer, id: u32, out: &mut String) {
        if !tok.is_special(id) {
            self.pending.extend_from_slice(tok.bytes(id));
            self.flush(out, false);
        }
    }

    /// Appends what is left, an unfinished character as U+FFFD.
    pub fn finish(&mut self, out: &mut String) {
        self.flush(out, true);
    }

    fn flush(&mut self, out: &mut String, end: bool) {
        let mut rest = &self.pending[..];
        loop {
            match std::str::from_utf8(rest) {
                Ok(s) => {
                    out.push_str(s);
                    rest = &[];
                    break;
                }
                Err(e) => {
                    let (good, bad) = rest.split_at(e.valid_up_to());
                    out.push_str(std::str::from_utf8(good).unwrap_or_default());
                    let Some(n) = e.error_len().or(end.then_some(bad.len())) else {
                        rest = bad;
                        break;
                    };
                    out.extend(std::iter::repeat_n('\u{FFFD}', n));
                    rest = &bad[n..];
                }
            }
        }
        self.pending = rest.to_vec();
    }
}