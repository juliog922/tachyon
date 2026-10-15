//! Text to token IDs and back, for Gemma's SentencePiece-style BPE: spaces become `▁`, the text is split into
//! characters, characters the vocabulary lacks fall back to their UTF-8 bytes (`<0xHH>` tokens), and adjacent
//! tokens merge, lowest-ranked pair first, until no merge applies; and for byte-level BPE (see `token/bytes.rs`), whose
//! symbols are the text's bytes, after NFC normalization if the tokenizer asks for it. IDs match Hugging Face
//! `tokenizers` on the same `tokenizer.json`.
//!
//! [`Tokenizer::from_json`] reads a `tokenizer.json` once, at conversion; [`Tokenizer::to_bytes`] and
//! [`Tokenizer::from_bytes`] store and load the result, whose lookup tables are ready to use as read: loading is a
//! copy, not a rebuild. [`Detokenizer`] turns IDs back into text as they are generated, whole characters only.
//!
//! Text is encoded literally: a special token's text in a prompt stays text. Special tokens enter only by ID,
//! from the chat template, so user text cannot forge a turn.

mod bytes;
mod chat;
mod unicode;

pub use chat::{Chat, Message, Role};

use crate::json::Json;
use crate::{Error, Result};
use std::collections::HashMap;

/// No entry in a lookup table, and no token.
const NONE: u32 = u32::MAX;
const MAGIC: &[u8; 8] = b"TACHYTOK";
/// Flag: the text's spaces are encoded as `▁`.
const META: u32 = 1;
/// Flag: characters outside the vocabulary become `<0xHH>` byte tokens.
const BYTES: u32 = 2;
/// Flag: the text is cut before every space, so no merge crosses one (a Metaspace pre-tokenizer that splits).
const SPLIT: u32 = 4;
/// Flag: a text that does not start with a space gets one (Metaspace's `prepend_scheme`).
const PREPEND: u32 = 8;
/// Flag: byte-level BPE, the text cut into pieces by GPT-2's pattern, or with [`CL100K`] or [`O200K`] by that one.
const BYTE_LEVEL: u32 = 16;
/// Flag: the cl100k pattern (see [`bytes::Pattern`]).
const CL100K: u32 = 32;
/// Flag: the text is put in Unicode Normalization Form C first.
const NFC: u32 = 64;
/// Flag: the o200k pattern (see [`bytes::Pattern`]).
const O200K: u32 = 128;
/// Characters with a direct entry in [`Tokenizer::chars`]: the first two Unicode planes.
const DIRECT: usize = 0x20000;

/// A BPE vocabulary with its merges.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tokenizer {
    /// [`META`], [`BYTES`], [`SPLIT`], [`PREPEND`], [`BYTE_LEVEL`], [`CL100K`], [`NFC`], [`O200K`].
    flags: u32,
    /// The unknown token, or [`NONE`].
    unk: u32,
    /// Where each token's text starts in `text`; one more entry than tokens.
    offsets: Vec<u32>,
    /// Every token's text as decoding produces it: `▁` already a space, byte tokens already their byte.
    text: Vec<u8>,
    /// Special tokens' IDs, sorted.
    special: Vec<u32>,
    /// The token of each byte: its `<0xHH>` token, or in byte-level BPE the one of its character.
    bytes: Vec<u32>,
    /// The token of each character below [`DIRECT`], or [`NONE`].
    chars: Vec<u32>,
    /// `(char, id)` pairs of the single-character tokens above it.
    rare: Vec<u32>,
    /// Characters some token joins to a following `▁`, such as `>` in `>▁</`: no cut after them.
    joins: Vec<u32>,
    /// Added tokens that are not special, such as Granite's `<body>`, by first byte then longest first: found in the
    /// text before anything merges, as Hugging Face does.
    added: Vec<u32>,
    /// Open addressing table of `(left, right, rank, merged)`.
    merges: Vec<u32>,
    /// Open addressing table of the vocabulary's tokens by their text, when the model ignores merges for a piece
    /// that is a token (`ignore_merges`); empty otherwise.
    words: Vec<u32>,
    /// The special tokens the post-processor puts before and after a text, or [`NONE`].
    frame: [u32; 2],
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
    /// pre-tokenizer, and either no normalizer or one that replaces spaces with `▁`; of mmBERT's kind, whose
    /// Metaspace pre-tokenizer splits the text at every space and prepends one; or byte-level BPE, as GPT-2 and `OLMo`
    /// ship it, with an NFC normalizer or none, and one of three pre-tokenizer patterns (GPT-2's, cl100k, o200k).
    pub fn from_json(json: &[u8]) -> Result<Tokenizer> {
        let doc = Json::parse(json)?;
        let model = doc.get("model").filter(|m| m.get("type").and_then(Json::str) == Some("BPE")).ok_or_else(|| unsupported("a model other than BPE"))?;
        let (meta, nfc) = normalizer(doc.get("normalizer"))?;
        let pre = pre_tokenizer(doc.get("pre_tokenizer"), meta)? | (u32::from(nfc) * NFC);
        let fallback = model.get("byte_fallback").and_then(Json::bool) == Some(true);
        let mut vocab: Vec<(&str, u32)> =
            model.get("vocab").and_then(Json::obj).unwrap_or_default().iter().map(|(k, v)| (k.as_str(), v.num().unwrap_or(-1.0) as u32)).collect();
        let (special, plain) = added(doc.get("added_tokens"), &mut vocab);
        let merges: Vec<(&str, &str)> = model.get("merges").and_then(Json::arr).unwrap_or_default().iter().filter_map(pair).collect();
        let unk = model.get("unk_token").and_then(Json::str);
        let skip: Vec<u32> = special.iter().chain(&plain).copied().collect();
        let mut t = Tokenizer::build(&vocab, &merges, [special, plain], unk, pre | (u32::from(meta) * META) | (u32::from(fallback) * BYTES))?;
        t.frame = frame(doc.get("post_processor"));
        if model.get("ignore_merges").and_then(Json::bool) == Some(true) {
            t.index_words(&skip);
        }
        Ok(t)
    }

    /// The tokenizer of `vocab` and `merges`, whose added tokens are `[special, plain]`.
    fn build(vocab: &[(&str, u32)], merges: &[(&str, &str)], [mut special, added]: [Vec<u32>; 2], unk: Option<&str>, flags: u32) -> Result<Tokenizer> {
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
            added: Vec::new(),
            merges: vec![NONE; 4 * (2 * merges.len()).next_power_of_two().max(2)],
            words: Vec::new(),
            frame: [NONE; 2],
        };
        for (id, name) in names.iter().enumerate() {
            t.add(id as u32, name);
        }
        let plain = names.iter().enumerate().filter(|&(id, _)| !t.is_special(id as u32));
        let pairs = plain.flat_map(|(_, name)| name.chars().zip(name.chars().skip(1)));
        t.joins = pairs.filter(|&(c, d)| d == '▁' && c != '▁').map(|(c, _)| c as u32).collect();
        t.joins.sort_unstable();
        t.joins.dedup();
        t.added = matched(&names, added);
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

    /// Records token `id`, named `name` in the vocabulary.
    fn add(&mut self, id: u32, name: &str) {
        if self.flags & BYTE_LEVEL != 0 && !self.is_special(id) { self.add_byte_token(id, name) } else { self.add_token(id, name) }
    }

    /// Records token `id` of a byte-level vocabulary: a name of byte characters is those bytes; another (an added
    /// token's) is its own text.
    fn add_byte_token(&mut self, id: u32, name: &str) {
        let decoded: Option<Vec<u8>> = name.chars().map(bytes::byte_of).collect();
        if let Some(&[b]) = decoded.as_deref() {
            self.bytes[usize::from(b)] = id;
        }
        self.text.extend(decoded.unwrap_or_else(|| name.as_bytes().to_vec()));
        self.offsets.push(self.text.len() as u32);
    }

    /// Fills [`Tokenizer::words`] with every token but those of `skip`, the added ones.
    fn index_words(&mut self, skip: &[u32]) {
        let mut table = vec![NONE; (2 * self.len()).next_power_of_two().max(2)];
        let mask = table.len() - 1;
        for id in (0..self.len() as u32).filter(|id| !skip.contains(id)) {
            let mut i = slot(fnv(self.bytes(id)), mask);
            while table[i] != NONE {
                i = (i + 1) & mask;
            }
            table[i] = id;
        }
        self.words = table;
    }

    /// The token whose text is `text`, if the model ignores merges for such a piece.
    fn word(&self, text: &[u8]) -> Option<u32> {
        let mask = self.words.len().checked_sub(1)?;
        let mut i = slot(fnv(text), mask);
        loop {
            match self.words[i] {
                NONE => return None,
                id if self.bytes(id) == text => return Some(id),
                _ => i = (i + 1) & mask,
            }
        }
    }

    /// The special tokens Hugging Face's post-processor puts before and after a single text, if any.
    pub fn frame(&self) -> [Option<u32>; 2] {
        self.frame.map(|id| (id != NONE).then_some(id))
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

    /// Appends the tokens of `text` to `out`. Non-special added tokens are found first, the longest at each place;
    /// the text between them is cut before each space that follows a character no token joins to a following `▁`
    /// (with a splitting Metaspace pre-tokenizer, before every space), so no merge can cross a cut, and each piece is
    /// merged on its own.
    pub fn encode(&self, text: &str, out: &mut Vec<u32>) {
        let normalized = if self.flags & NFC != 0 { unicode::nfc(text) } else { text.into() };
        let text = normalized.as_ref();
        let (mut work, mut from, mut i) = (Work::default(), 0, 0);
        while let Some(c) = text[i..].chars().next() {
            match self.added_at(&text.as_bytes()[i..]) {
                Some((id, len)) => {
                    self.segment(&text[from..i], &mut work, out);
                    out.push(id);
                    (i, from) = (i + len, i + len);
                }
                None => i += c.len_utf8(),
            }
        }
        self.segment(&text[from..], &mut work, out);
    }

    /// The longest non-special added token `rest` starts with, and its length in bytes.
    fn added_at(&self, rest: &[u8]) -> Option<(u32, usize)> {
        let first = *rest.first()?;
        let at = self.added.partition_point(|&id| self.bytes(id)[0] < first);
        let same = self.added[at..].iter().take_while(|&&id| self.bytes(id)[0] == first);
        same.map(|&id| (id, self.bytes(id))).find(|(_, t)| rest.starts_with(t)).map(|(id, t)| (id, t.len()))
    }

    /// Appends the tokens of `text`, which holds no added token: `▁` first if the pre-tokenizer prepends one, then
    /// each piece between cuts merged on its own.
    fn segment(&self, text: &str, work: &mut Work, out: &mut Vec<u32>) {
        let prefixed;
        let text = if self.flags & PREPEND != 0 && !text.is_empty() && !text.starts_with([' ', '▁']) {
            prefixed = format!("{}{text}", if self.flags & BYTE_LEVEL != 0 { ' ' } else { '▁' });
            &prefixed
        } else {
            text
        };
        if self.flags & BYTE_LEVEL != 0 {
            return self.byte_segment(text, work, out);
        }
        let (mut start, mut last) = (0, '▁');
        for (i, c) in text.char_indices() {
            if i > start && self.cuts(last, c) {
                self.merge_all(&text[start..i], work, out);
                start = i;
            }
            last = c;
        }
        if start < text.len() {
            self.merge_all(&text[start..], work, out);
        }
    }

    /// Appends the tokens of `text` in byte-level BPE: each piece its pattern cuts is a token of its own if the model
    /// ignores merges for pieces that are tokens, else merged on its own.
    fn byte_segment(&self, text: &str, work: &mut Work, out: &mut Vec<u32>) {
        let pattern = [bytes::Pattern::Gpt2, bytes::Pattern::Cl100k, bytes::Pattern::O200k]
            [usize::from(self.flags & CL100K != 0) + 2 * usize::from(self.flags & O200K != 0)];
        bytes::pieces(text, pattern, |piece| match self.word(piece.as_bytes()) {
            Some(id) => out.push(id),
            None => self.merge_all(piece, work, out),
        });
    }

    /// Whether the text is cut between `last` and `c`: before a space, unless a token joins `last` to it.
    fn cuts(&self, last: char, c: char) -> bool {
        let space = c == ' ' || c == '▁';
        let joined = last == ' ' || last == '▁' || self.joins.contains(&(last as u32));
        space && self.flags & META != 0 && (self.flags & SPLIT != 0 || !joined)
    }

    /// Merges the symbols of `text`, lowest-ranked pair first, and appends the result to `out`.
    fn merge_all(&self, text: &str, w: &mut Work, out: &mut Vec<u32>) {
        if self.flags & BYTE_LEVEL == 0 {
            self.symbols(text, &mut w.ids);
        } else {
            w.ids.clear();
            w.ids.extend(text.bytes().map(|b| self.bytes[usize::from(b)]));
        }
        let n = w.ids.len() as u32;
        w.next.clear();
        w.next.extend(1..=n);
        w.prev.clear();
        w.prev.extend((0..n).map(|i| i.wrapping_sub(1)));
        // A tournament tree of the merges' keys: leaf `half + i` is symbol `i`'s, each node the least of its two.
        // The first lookups are independent: in one plain loop, their cache misses overlap.
        let half = w.ids.len().next_power_of_two();
        w.tree.clear();
        w.tree.resize(2 * half, u64::MAX);
        w.merged.clear();
        for (i, p) in w.ids.windows(2).enumerate() {
            let (rank, merged) = self.merge(p[0], p[1]).unwrap_or((NONE, NONE));
            w.tree[half + i] = u64::from(rank) << 32 | i as u64;
            w.merged.push(merged);
        }
        w.merged.push(NONE);
        for k in (1..half).rev() {
            w.tree[k] = w.tree[2 * k].min(w.tree[2 * k + 1]);
        }
        while let Some(top) = Some(w.tree[1]).filter(|&k| k >> 32 != u64::from(NONE)) {
            let i = top as u32 as usize;
            let j = w.next[i] as usize;
            (w.ids[i], w.ids[j], w.next[i]) = (w.merged[i], NONE, w.next[j]);
            if let Some(after) = w.prev.get_mut(w.next[i] as usize) {
                *after = i as u32;
            }
            w.set(j, (u64::MAX, NONE));
            for k in [w.prev[i], i as u32] {
                if k < n {
                    w.set(k as usize, self.after(w, k));
                }
            }
        }
        out.extend(w.ids.iter().copied().filter(|&id| id != NONE));
    }

    /// The merge of symbol `i` with the next one, as `(rank << 32 | i, merged)`.
    fn after(&self, w: &Work, i: u32) -> (u64, u32) {
        let (a, b) = (w.ids.get(i as usize), w.next.get(i as usize).and_then(|&j| w.ids.get(j as usize)));
        let (rank, merged) = a.zip(b).and_then(|(&a, &b)| self.merge(a, b)).unwrap_or((NONE, NONE));
        (u64::from(rank) << 32 | u64::from(i), merged)
    }

    /// The tokens before merging, into `ids`: one per character, or per byte of a character the vocabulary lacks (in
    /// byte-level BPE, [`Tokenizer::merge_all`] takes one per byte).
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
        let head = [self.flags, self.unk, self.text.len() as u32, self.frame[0], self.frame[1]];
        let sections = [&head[..], &self.offsets, &text, &self.special, &self.bytes, &self.chars, &self.rare, &self.joins, &self.added, &self.merges];
        for section in sections.into_iter().chain((!self.words.is_empty()).then_some(&self.words[..])) {
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
        let mut s: Vec<Vec<u32>> = (0..10).map(|_| section(&mut words)).collect::<Result<_>>()?;
        let [head, offsets, text, special, bytes, chars, rare, joins, added, merges] = std::array::from_fn(|i| std::mem::take(&mut s[i]));
        let mut text: Vec<u8> = text.iter().flat_map(|w| w.to_le_bytes()).collect();
        text.truncate(*head.get(2).ok_or_else(damaged)? as usize);
        // Older files end after the merges, and their head holds no frame.
        let words = if words.len() > 0 { section(&mut words)? } else { Vec::new() };
        let frame = [3, 4].map(|i| head.get(i).copied().unwrap_or(NONE));
        let t = Tokenizer { flags: head[0], unk: head[1], offsets, text, special, bytes, chars, rare, joins, added, merges, words, frame };
        let sound = t.offsets.last() == Some(&(t.text.len() as u32)) && t.bytes.len() == 256 && t.chars.len() == DIRECT;
        if sound && t.merges.len().is_power_of_two() { Ok(t) } else { Err(damaged()) }
    }
}

fn damaged() -> Error {
    Error::Format("tokenizer.tok: damaged, or written by an older Tachyon; convert the model again".into())
}

/// A length-prefixed run of words.
fn section(words: &mut impl Iterator<Item = u32>) -> Result<Vec<u32>> {
    let n = words.next().ok_or_else(damaged)? as usize;
    let s: Vec<u32> = words.take(n).collect();
    if s.len() == n { Ok(s) } else { Err(damaged()) }
}

/// Whether the normalizer turns spaces into `▁`, and whether it is NFC; none is fine, any other is not supported.
fn normalizer(normalizer: Option<&Json>) -> Result<(bool, bool)> {
    match normalizer {
        None | Some(Json::Null) => Ok((false, false)),
        Some(n) if n.get("type").and_then(Json::str) == Some("Replace") && n.get("content").and_then(Json::str) == Some("▁") => Ok((true, false)),
        Some(n) if n.get("type").and_then(Json::str) == Some("NFC") => Ok((false, true)),
        Some(_) => Err(unsupported("this normalizer")),
    }
}

/// The added tokens of `added` that can match a text, by first byte then longest first. One with a `▁` never does:
/// added tokens are found in the raw text, where spaces are still spaces.
fn matched(names: &[&str], mut added: Vec<u32>) -> Vec<u32> {
    added.retain(|&id| names.get(id as usize).is_some_and(|n| !n.is_empty() && !n.contains('▁')));
    added.sort_unstable_by_key(|&id| (names[id as usize].as_bytes()[0], std::cmp::Reverse(names[id as usize].len()), id));
    added
}

/// The flags of the pre-tokenizer: none; Gemma's split on spaces, which never splits once the normalizer `meta`
/// has turned them into `▁`; a Metaspace with `▁`, which may split and prepend; a byte-level one with GPT-2's
/// pattern; or a split by another pattern of [`bytes::PATTERNS`] (its matches kept) then a byte-level one.
fn pre_tokenizer(pre: Option<&Json>, meta: bool) -> Result<u32> {
    let Some(pre) = pre.filter(|p| **p != Json::Null) else { return Ok(0) };
    if let Some(flags) = byte_level(pre) {
        return Ok(flags);
    }
    let field = |k| pre.get(k).and_then(Json::str);
    match field("type") {
        Some("Split") if meta && pre.get("pattern").and_then(|p| p.get("String")).and_then(Json::str) == Some(" ") => Ok(0),
        Some("Metaspace") if field("replacement") == Some("▁") => {
            let split = u32::from(pre.get("split").and_then(Json::bool) != Some(false)) * SPLIT;
            Ok(META | split | (u32::from(matches!(field("prepend_scheme"), Some("always" | "first"))) * PREPEND))
        }
        _ => Err(unsupported("this pre-tokenizer")),
    }
}

/// The flags of a byte-level pre-tokenizer: one with GPT-2's pattern, or a split by a pattern of [`bytes::PATTERNS`]
/// (its matches kept, alone or with what lies between them) then one with none; `None` for any other.
fn byte_level(pre: &Json) -> Option<u32> {
    let flags = |p: &Json, regex: bool| {
        let prepend = u32::from(p.get("add_prefix_space").and_then(Json::bool) == Some(true)) * PREPEND;
        (p.get("type").and_then(Json::str) == Some("ByteLevel") && p.get("use_regex").and_then(Json::bool) == Some(regex)).then_some(BYTE_LEVEL | prepend)
    };
    if let Some(f) = flags(pre, true) {
        return Some(f);
    }
    let [split, last] = pre.get("pretokenizers").and_then(Json::arr).unwrap_or_default() else { return None };
    let regex = split.get("pattern").and_then(|p| p.get("Regex")).and_then(Json::str);
    // Matches kept, or matches and what lies between them: the same with patterns that match everything.
    let kept = matches!(
        (split.get("behavior").and_then(Json::str), split.get("invert").and_then(Json::bool)),
        (Some("Removed"), Some(true)) | (Some("Isolated"), Some(false))
    );
    let &(_, pattern) = bytes::PATTERNS.iter().find(|p| Some(p.0) == regex).filter(|_| kept)?;
    let bit = match pattern {
        bytes::Pattern::Gpt2 => 0,
        bytes::Pattern::Cl100k => CL100K,
        bytes::Pattern::O200k => O200K,
    };
    Some(flags(last, false)? | bit)
}

/// FNV-1a 64 of `bytes`.
fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xCBF2_9CE4_8422_2325, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01B3))
}

/// The special tokens a `TemplateProcessing` post-processor (alone or in a sequence) puts before and after a single
/// text; [`NONE`] where it puts none, or more than one.
fn frame(post: Option<&Json>) -> [u32; 2] {
    let template = post.and_then(|p| {
        let all = std::iter::once(p).chain(p.get("processors").and_then(Json::arr).unwrap_or_default());
        all.into_iter().find(|q| q.get("type").and_then(Json::str) == Some("TemplateProcessing"))
    });
    let Some(t) = template else { return [NONE; 2] };
    let id = |item: &Json| {
        let name = item.get("SpecialToken")?.get("id")?.str()?;
        let ids = t.get("special_tokens")?.get(name)?.get("ids")?.arr()?;
        ids.first()?.num().map(|v| v as u32).filter(|_| ids.len() == 1)
    };
    let single = t.get("single").and_then(Json::arr).unwrap_or_default();
    let at = single.iter().position(|i| i.get("Sequence").is_some()).unwrap_or(single.len());
    let side = |items: &[Json]| match items {
        [one] => id(one).unwrap_or(NONE),
        _ => NONE,
    };
    [side(&single[..at]), side(single.get(at + 1..).unwrap_or_default())]
}

/// Adds the added tokens to `vocab`; returns the special ones' IDs and the others'.
fn added<'j>(tokens: Option<&'j Json>, vocab: &mut Vec<(&'j str, u32)>) -> (Vec<u32>, Vec<u32>) {
    let (mut special, mut plain) = (Vec::new(), Vec::new());
    for t in tokens.and_then(Json::arr).unwrap_or_default() {
        let (content, id) = (t.get("content").and_then(Json::str).unwrap_or_default(), t.get("id").and_then(Json::num).unwrap_or(-1.0) as u32);
        vocab.push((content, id));
        if t.get("special").and_then(Json::bool) == Some(true) { special.push(id) } else { plain.push(id) }
    }
    (special, plain)
}

/// The buffers [`Tokenizer::encode`] reuses from piece to piece.
#[derive(Default)]
struct Work {
    ids: Vec<u32>,
    next: Vec<u32>,
    prev: Vec<u32>,
    /// Each symbol's merge with the next as `rank << 32 | position`, so the least is the one to merge first, in a
    /// tournament tree.
    tree: Vec<u64>,
    /// What each of those merges makes.
    merged: Vec<u32>,
}

impl Work {
    /// Sets the merge of symbol `i` with the next one, and the tree above its key.
    fn set(&mut self, i: usize, (key, merged): (u64, u32)) {
        self.merged[i] = merged;
        let mut k = self.tree.len() / 2 + i;
        self.tree[k] = key;
        while k > 1 {
            let least = self.tree[k].min(self.tree[k ^ 1]);
            k /= 2;
            if std::mem::replace(&mut self.tree[k], least) == least {
                break;
            }
        }
    }
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
        let used = self.pending.len() - rest.len();
        self.pending.drain(..used);
    }
}
