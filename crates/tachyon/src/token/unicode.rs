//! The Unicode the byte-level tokenizers need: which characters are letters (`\p{L}`) and of which kind, and NFC normalization. The
//! tables are `unicode.bin`, written by `scripts/unicode.py` from Python's `unicodedata` (Unicode 15.1), and read once,
//! on first use. Numbers (`\p{N}`) and white space (`\s`) are `char::is_numeric` and `char::is_whitespace`, which
//! follow the same Unicode categories.

use std::borrow::Cow;
use std::sync::OnceLock;

/// The first Hangul syllable, leading consonant, vowel and trailing consonant (less one: 0 is none).
const S0: u32 = 0xAC00;
const L0: u32 = 0x1100;
const V0: u32 = 0x1161;
const T0: u32 = 0x11A7;
/// How many leading consonants, vowels and trailing consonants (with none) there are.
const LN: u32 = 19;
const VN: u32 = 21;
const TN: u32 = 28;

/// The tables of `unicode.bin`, each of triples, sorted: ranges of letters and marks `[first, last, kind]` (see
/// [`kind`]), ranges of non-zero
/// combining classes `[first, last, class]`, one level of canonical decomposition `[character, first, second or 0]`,
/// and canonical compositions `[first, second, composite]`.
struct Tables {
    letters: Vec<[u32; 3]>,
    classes: Vec<[u32; 3]>,
    decomposed: Vec<[u32; 3]>,
    composed: Vec<[u32; 3]>,
}

/// The tables, read once.
fn tables() -> &'static Tables {
    static TABLES: OnceLock<Tables> = OnceLock::new();
    TABLES.get_or_init(|| {
        let mut data = include_bytes!("unicode.bin").iter();
        let mut next = || {
            let mut v = 0;
            for (shift, &b) in (0..).step_by(7).zip(data.by_ref()) {
                v |= u32::from(b & 0x7F) << shift;
                if b < 0x80 {
                    break;
                }
            }
            v
        };
        // The first two sections' ranges, and the third's characters, are written as gaps from the last.
        let mut section = |gaps: bool| {
            let mut last = 0;
            let n = next();
            (0..n)
                .map(|_| [next(), next(), next()])
                .map(|[a, b, c]| {
                    if gaps {
                        last += a;
                        let e = [last, last + b, c];
                        last += b + 1;
                        e
                    } else {
                        [a, b, c]
                    }
                })
                .collect::<Vec<_>>()
        };
        let (letters, classes) = (section(true), section(true));
        let mut last = 0;
        let decomposed = section(false)
            .into_iter()
            .map(|[gap, a, b]| {
                last += gap;
                let e = [last, a, b];
                last += 1;
                e
            })
            .collect();
        Tables { letters, classes, decomposed, composed: section(false) }
    })
}

/// The value of the range of `ranges` that holds `c`, 0 if none does.
fn lookup(ranges: &[[u32; 3]], c: u32) -> u32 {
    let i = ranges.partition_point(|e| e[1] < c);
    ranges.get(i).filter(|e| e[0] <= c).map_or(0, |e| e[2])
}

/// Whether `c` is a letter: general category L (`Lu`, `Ll`, `Lt`, `Lm`, `Lo`).
pub(super) fn is_letter(c: char) -> bool {
    (1..=3).contains(&kind(c))
}

/// What a letter or mark is: 1 upper or title case (`Lu`, `Lt`), 2 lower case (`Ll`), 3 neither (`Lm`, `Lo`), 4 a
/// mark (`M`); 0 anything else.
pub(super) fn kind(c: char) -> u32 {
    match c {
        'A'..='Z' => 1,
        'a'..='z' => 2,
        _ if c.is_ascii() => 0,
        _ => lookup(&tables().letters, c as u32),
    }
}

/// Appends the full canonical decomposition of `c` to `out`.
fn decompose(c: u32, out: &mut Vec<u32>) {
    if (S0..S0 + LN * VN * TN).contains(&c) {
        let s = c - S0;
        out.extend([L0 + s / (VN * TN), V0 + s % (VN * TN) / TN]);
        out.extend((s % TN != 0).then_some(T0 + s % TN));
        return;
    }
    let d = &tables().decomposed;
    match d.binary_search_by_key(&c, |e| e[0]) {
        Ok(i) => {
            decompose(d[i][1], out);
            if d[i][2] != 0 {
                decompose(d[i][2], out);
            }
        }
        Err(_) => out.push(c),
    }
}

/// The canonical composition of `a` and `b`, if they have one.
fn compose(a: u32, b: u32) -> Option<u32> {
    if (L0..L0 + LN).contains(&a) && (V0..V0 + VN).contains(&b) {
        return Some(S0 + ((a - L0) * VN + b - V0) * TN);
    }
    if (S0..S0 + LN * VN * TN).contains(&a) && (a - S0) % TN == 0 && (T0 + 1..T0 + TN).contains(&b) {
        return Some(a + b - T0);
    }
    let c = &tables().composed;
    c.binary_search_by_key(&(a, b), |e| (e[0], e[1])).ok().map(|i| c[i][2])
}

/// `text` in Normalization Form C: decomposed, marks put in canonical order, composed again. Text below U+0300 is
/// already in it and comes back as it is.
pub(super) fn nfc(text: &str) -> Cow<'_, str> {
    if text.chars().all(|c| (c as u32) < 0x300) {
        return Cow::Borrowed(text);
    }
    let mut d = Vec::with_capacity(text.len());
    text.chars().for_each(|c| decompose(c as u32, &mut d));
    let classes: Vec<u32> = d.iter().map(|&c| lookup(&tables().classes, c)).collect();
    let mut order: Vec<usize> = (0..d.len()).collect();
    // Canonical order: a stable sort of each run of marks by class.
    let mut start = 0;
    while start < d.len() {
        let end = (start..d.len()).find(|&i| classes[i] == 0).map_or(d.len(), |i| if i == start { i + 1 } else { i });
        order[start..end].sort_by_key(|&i| classes[i]);
        start = end;
    }
    let mut out: Vec<u32> = Vec::with_capacity(d.len());
    let (mut starter, mut last) = (None, 0);
    for i in order {
        let (c, cc) = (d[i], classes[i]);
        let open = starter.is_some_and(|s: usize| out.len() == s + 1 || (last != 0 && last < cc));
        if let Some(composite) = starter.filter(|_| open).and_then(|s| compose(out[s], c)) {
            out[starter.unwrap_or_default()] = composite;
            continue;
        }
        if cc == 0 {
            starter = Some(out.len());
        }
        last = cc;
        out.push(c);
    }
    Cow::Owned(out.into_iter().filter_map(char::from_u32).collect())
}

#[cfg(test)]
mod tests {
    use super::{is_letter, nfc};

    /// Python's `unicodedata.normalize("NFC", …)` on cases that exercise each step: Hangul composed from its jamo
    /// and kept apart from a mark, marks reordered by class before composing, a mark blocked by another of its
    /// class, singletons and excluded compositions that never come back, and two-level decompositions.
    #[test]
    fn nfc_matches_python() {
        let cases = [
            ("e\u{301}", "\u{e9}"),
            ("\u{1100}\u{1161}\u{11a8}", "\u{ac01}"),
            ("\u{ac00}\u{11a8}x", "\u{ac01}x"),
            ("a\u{323}\u{302}", "\u{1ead}"),
            ("a\u{302}\u{323}", "\u{1ead}"),
            ("a\u{301}\u{301}", "\u{e1}\u{301}"),
            ("\u{212b}", "\u{c5}"),
            ("\u{2126}", "\u{3a9}"),
            ("\u{958}", "\u{915}\u{93c}"),
            ("\u{1e9b}\u{323}", "\u{1e9b}\u{323}"),
            ("\u{1f80}", "\u{1f80}"),
            ("\u{3b1}\u{313}\u{300}\u{345}", "\u{1f82}"),
            ("plain ascii", "plain ascii"),
        ];
        for (text, want) in cases {
            assert_eq!(nfc(text), want, "{text:?}");
        }
    }

    #[test]
    fn letters_are_category_l() {
        assert!(['a', 'Z', 'é', 'ß', 'Ω', 'ж', 'א', 'ع', 'क', '中', 'ア', 'ㄱ', 'ʰ'].iter().all(|&c| is_letter(c)));
        assert!(!['1', '٣', 'Ⅻ', '\u{301}', '\u{93f}', ' ', '.', '😀', '_'].iter().any(|&c| is_letter(c)), "numbers, marks, spaces, symbols");
    }
}
