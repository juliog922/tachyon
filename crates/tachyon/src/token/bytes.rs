//! Byte-level BPE, GPT-2's scheme, which the tokenizers of `OLMo` and `ModernBERT` keep: every byte of the text is a
//! symbol, written in the vocabulary as a printable character, and the text is first cut into pieces that no merge
//! crosses, by one of three regular expressions, matched here by hand:
//!
//! - [`Pattern::Gpt2`]: `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`;
//! - [`Pattern::Cl100k`] (Llama 3's, Dolma 2's): `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|
//!   ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`;
//! - [`Pattern::O200k`] (GPT-4o's, Granite Embedding 97M's): words split at case changes, `[^\r\n\p{L}\p{N}]?U*W+c?|
//!   [^\r\n\p{L}\p{N}]?U+W*c?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+`, where `U` is
//!   `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]`, `W` is `[\p{Ll}\p{Lm}\p{Lo}\p{M}]` and `c` the contractions, in any case.
//!
//! As the regular expressions do, the first alternative that matches at a place wins, each as long as it can be.

use super::unicode::{is_letter, kind};

/// The regular expression that cuts a text into pieces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Pattern {
    Gpt2,
    Cl100k,
    O200k,
}

/// The patterns as `tokenizer.json` writes them.
pub(super) const PATTERNS: [(&str, Pattern); 3] = [
    (r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+", Pattern::Gpt2),
    (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+", Pattern::Cl100k),
    (
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        Pattern::O200k,
    ),
];

/// Whether byte `b` stands for itself in a byte-level vocabulary: the printable ASCII and Latin-1 characters.
fn printable(b: u8) -> bool {
    matches!(b, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF)
}

/// The byte a character of a byte-level vocabulary stands for, if it stands for one: itself if printable, else the
/// `n`th of the others, written `U+0100 + n`.
pub(super) fn byte_of(c: char) -> Option<u8> {
    match u32::from(c) {
        c @ 0..=255 => u8::try_from(c).ok().filter(|&b| printable(b)),
        c @ 256..=323 => (0..=255u8).filter(|&b| !printable(b)).nth(c as usize - 256),
        _ => None,
    }
}

/// What a character is to the patterns.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Class {
    Letter,
    Number,
    Space,
    Other,
}

fn class(c: char) -> Class {
    if is_letter(c) {
        Class::Letter
    } else if c.is_numeric() {
        Class::Number
    } else if c.is_whitespace() {
        Class::Space
    } else {
        Class::Other
    }
}

/// Calls `f` on each piece `pattern` cuts `text` into, in order.
pub(super) fn pieces(text: &str, pattern: Pattern, mut f: impl FnMut(&str)) {
    let mut at = 0;
    while at < text.len() {
        let s = &text[at..];
        let n = match pattern {
            Pattern::Gpt2 => gpt2(s),
            Pattern::Cl100k => cl100k(s),
            Pattern::O200k => o200k(s),
        };
        f(&s[..n]);
        at += n;
    }
}

/// Bytes of the longest run of characters at the start of `s` that `keep` accepts, at most `most` of them.
fn run(s: &str, most: usize, keep: impl Fn(char) -> bool) -> usize {
    s.char_indices().take(most).find(|&(_, c)| !keep(c)).map_or_else(|| s.chars().take(most).map(char::len_utf8).sum(), |(i, _)| i)
}

/// A contraction at the start of `s` (`'s`, `'t`, `'re`, `'ve`, `'m`, `'ll`, `'d`), in any case if `fold`: its bytes.
fn contraction(s: &str, fold: bool) -> Option<usize> {
    let rest = s.strip_prefix('\'')?.as_bytes();
    let lower = |i: usize| rest.get(i).map(|&b| if fold { b.to_ascii_lowercase() } else { b });
    match (lower(0)?, lower(1)) {
        (b's' | b't' | b'm' | b'd', _) => Some(2),
        (b'r' | b'v', Some(b'e')) | (b'l', Some(b'l')) => Some(3),
        _ => None,
    }
}

/// `\s+(?!\S)|\s+` at the start of `s`, which begins with white space: the run, less its last character when more
/// text follows and the run is longer than one.
fn spaces(s: &str) -> usize {
    let n = run(s, usize::MAX, char::is_whitespace);
    match s[..n].char_indices().last() {
        Some((last, _)) if n < s.len() && last > 0 => last,
        _ => n,
    }
}

/// The length of the GPT-2 pattern's match at the start of `s`.
fn gpt2(s: &str) -> usize {
    if let Some(n) = contraction(s, false) {
        return n;
    }
    let skip = usize::from(s.starts_with(' '));
    match s[skip..].chars().next().map(class) {
        Some(c @ (Class::Letter | Class::Number | Class::Other)) => skip + run(&s[skip..], usize::MAX, |d| class(d) == c),
        _ => spaces(s),
    }
}

/// The length of the cl100k pattern's match at the start of `s`.
fn cl100k(s: &str) -> usize {
    if let Some(n) = contraction(s, true) {
        return n;
    }
    let mut chars = s.chars();
    let (c0, c1) = (chars.next().unwrap_or(' '), chars.next());
    let letters = |from: usize| from + run(&s[from..], usize::MAX, is_letter);
    match class(c0) {
        Class::Letter => letters(0),
        Class::Number => run(s, 3, char::is_numeric),
        _ if !matches!(c0, '\r' | '\n') && c1.is_some_and(is_letter) => letters(c0.len_utf8()),
        c => rest(s, c == Class::Other || (c0 == ' ' && c1.is_some_and(|c| class(c) == Class::Other)), false),
    }
}

/// The length of the o200k pattern's match at the start of `s`.
fn o200k(s: &str) -> usize {
    let c0 = s.chars().next().unwrap_or(' ');
    let prefix = (!matches!(c0, '\r' | '\n') && !is_letter(c0) && !c0.is_numeric()).then(|| c0.len_utf8());
    // Each word alternative with its optional first character, then without it, as the regular expression tries them.
    for upper_first in [false, true] {
        for start in prefix.into_iter().chain([0]) {
            if let Some(n) = word(&s[start..], upper_first) {
                let end = start + n;
                return end + contraction(&s[end..], true).unwrap_or(0);
            }
        }
    }
    if c0.is_numeric() {
        return run(s, 3, char::is_numeric);
    }
    let c1 = s.chars().nth(1);
    rest(s, class(c0) == Class::Other || (c0 == ' ' && c1.is_some_and(|c| class(c) == Class::Other)), true)
}

/// The bytes of `U*W+` (or with `upper_first`, `U+W*`) at the start of `s`, if it matches there: the longest run of
/// `U` that `W` can still follow (for `U*W+`), then the longest run of `W`.
fn word(s: &str, upper_first: bool) -> Option<usize> {
    // Each character's end and kind, while they are letters or marks.
    let run_of: Vec<(usize, u32)> = s.char_indices().map(|(i, c)| (i + c.len_utf8(), kind(c))).take_while(|&(_, k)| k != 0).collect();
    let (upper, lower) = (|k: u32| k != 2, |k: u32| k >= 2);
    let most = run_of.iter().take_while(|&&(_, k)| upper(k)).count();
    let end = |a: usize| a + run_of[a..].iter().take_while(|&&(_, k)| lower(k)).count();
    let n = if upper_first { (most > 0).then(|| end(most))? } else { (0..=most).rev().map(end).zip((0..=most).rev()).find(|&(e, a)| e > a)?.0 };
    Some(n.checked_sub(1).map_or(0, |last| run_of[last].0))
}

/// The cl100k and o200k patterns' last alternatives at the start of `s`: ` ?[^\s\p{L}\p{N}]+[\r\n]*` (with `slash`,
/// `[\r\n/]*`) if `symbols` (`s` starts with a symbol, or a space and a symbol), else `\s*[\r\n]+|\s+(?!\S)|\s+`.
fn rest(s: &str, symbols: bool, slash: bool) -> usize {
    if symbols {
        let skip = usize::from(s.starts_with(' '));
        let n = skip + run(&s[skip..], usize::MAX, |c| class(c) == Class::Other);
        return n + run(&s[n..], usize::MAX, |c| matches!(c, '\r' | '\n') || (slash && c == '/'));
    }
    let n = run(s, usize::MAX, char::is_whitespace);
    match s[..n].rfind(['\r', '\n']) {
        Some(last) => last + 1,
        None => spaces(s),
    }
}
