#!/usr/bin/env python3
"""Writes crates/tachyon/src/token/unicode.bin: the Unicode tables the byte-level tokenizers need, from Python's
`unicodedata` (Unicode 15.1 with Python 3.13).

Four sections, each a count then its entries of three numbers, every number an unsigned LEB128 varint:

1. Letters and marks: ranges as (gap from the previous range's end + 1, length − 1, kind), the kind 1 for Lu and Lt,
   2 for Ll, 3 for Lm and Lo, 4 for marks (M), which the o200k pattern groups with letters.
2. Canonical combining classes other than 0: ranges as (gap, length − 1, class).
3. Canonical decompositions, one level: (gap from the previous character + 1, first, second or 0).
4. Canonical compositions of pairs (Hangul excepted, which is algorithmic): (first, second, composite), sorted.

    python3 scripts/unicode.py
"""
import pathlib
import unicodedata

OUT = pathlib.Path(__file__).resolve().parent.parent / "crates/tachyon/src/token/unicode.bin"
HANGUL = range(0xAC00, 0xAC00 + 11172)


def varint(n, out):
    while True:
        b, n = n & 0x7F, n >> 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return


def runs(value):
    """Ranges of consecutive code points with the same non-None value: (start, end, value)."""
    out, start, current = [], None, None
    for cp in range(0x110001):
        v = value(cp) if cp < 0x110000 else None
        if v != current:
            if current is not None:
                out.append((start, cp - 1, current))
            start, current = cp, v
    return out


def decomposition(cp):
    d = unicodedata.decomposition(chr(cp))
    if not d or d.startswith("<") or cp in HANGUL:
        return None
    return [int(x, 16) for x in d.split()]


def main():
    out = bytearray()
    kinds = {"Lu": 1, "Lt": 1, "Ll": 2, "Lm": 3, "Lo": 3, "Mn": 4, "Mc": 4, "Me": 4}
    letters = runs(lambda cp: kinds.get(unicodedata.category(chr(cp))))
    classes = runs(lambda cp: unicodedata.combining(chr(cp)) or None)
    decomposed = [(cp, d) for cp in range(0x110000) if (d := decomposition(cp))]
    composed = sorted((d[0], d[1], cp) for cp, d in decomposed if len(d) == 2 and unicodedata.normalize("NFC", chr(d[0]) + chr(d[1])) == chr(cp))
    for section in (letters, classes):
        varint(len(section), out)
        last = 0
        for start, end, value in section:
            varint(start - last, out)
            varint(end - start, out)
            varint(value, out)
            last = end + 1
    varint(len(decomposed), out)
    last = 0
    for cp, d in decomposed:
        varint(cp - last, out)
        for x in (d + [0])[:2]:
            varint(x, out)
        last = cp + 1
    varint(len(composed), out)
    for a, b, c in composed:
        for x in (a, b, c):
            varint(x, out)
    OUT.write_bytes(bytes(out))
    print(f"{OUT}: {len(out)} bytes, Unicode {unicodedata.unidata_version}: {len(letters)} letter ranges, "
          f"{len(classes)} class ranges, {len(decomposed)} decompositions, {len(composed)} compositions")


if __name__ == "__main__":
    main()
