//! zlib streams (RFC 1950) and their deflate blocks (RFC 1951): stored, fixed and dynamic Huffman. Codes of up to
//! [`FAST`] bits decode through one table lookup, longer ones bit by bit.

use super::bad;
use crate::Result;

/// Bits a Huffman table resolves at once.
const FAST: u32 = 10;
/// Base lengths of length codes 257 to 285.
const BASES: [u16; 29] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
/// Extra bits of length codes 257 to 285.
const EXTRA: [u8; 29] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
/// The order in which a dynamic block lists its code-length code lengths.
const ORDER: [usize; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

/// The bits of a stream, least significant first.
struct Bits<'a> {
    data: &'a [u8],
    at: usize,
    buf: u64,
    n: u32,
}

impl Bits<'_> {
    /// The next `k` bits without consuming them, zeros past the end.
    fn peek(&mut self, k: u32) -> u32 {
        while self.n <= 56 && self.at < self.data.len() {
            self.buf |= u64::from(self.data[self.at]) << self.n;
            (self.at, self.n) = (self.at + 1, self.n + 8);
        }
        (self.buf & ((1 << k) - 1)) as u32
    }

    /// Consumes `k` bits; an error past the end.
    fn skip(&mut self, k: u32) -> Result<()> {
        if k > self.n {
            return Err(bad("a deflate stream ends early"));
        }
        (self.buf, self.n) = (self.buf >> k, self.n - k);
        Ok(())
    }

    fn take(&mut self, k: u32) -> Result<u32> {
        let v = self.peek(k);
        self.skip(k).map(|()| v)
    }
}

/// A canonical Huffman code: a table of codes up to [`FAST`] bits (symbol `<< 4 |` length, 0 for longer), and
/// the counts and symbols for the rest.
struct Huffman {
    fast: Vec<u16>,
    counts: [u16; 16],
    symbols: Vec<u16>,
}

impl Huffman {
    fn new(lengths: &[u8]) -> Huffman {
        let mut counts = [0u16; 16];
        for &l in lengths {
            counts[usize::from(l)] += 1;
        }
        counts[0] = 0;
        let mut offs = [0u16; 16];
        (1..16).for_each(|l| offs[l] = offs[l - 1] + counts[l - 1]);
        let mut symbols = vec![0u16; lengths.len()];
        let mut next = [0u32; 16];
        let (mut code, mut fast) = (0u32, vec![0u16; 1 << FAST]);
        for l in 1..16 {
            code = (code + u32::from(counts[l - 1])) << 1;
            next[l] = code;
        }
        for (s, &l) in lengths.iter().enumerate().filter(|(_, l)| **l > 0) {
            let l = usize::from(l);
            symbols[usize::from(offs[l])] = s as u16;
            offs[l] += 1;
            let (c, len) = (next[l], l as u32);
            next[l] += 1;
            if len <= FAST {
                let r = c.reverse_bits() >> (32 - len);
                (0..1 << (FAST - len)).for_each(|k| fast[(r | k << len) as usize] = (s as u16) << 4 | len as u16);
            }
        }
        Huffman { fast, counts, symbols }
    }

    fn decode(&self, bits: &mut Bits) -> Result<u16> {
        let e = self.fast[bits.peek(FAST) as usize];
        if e != 0 {
            bits.skip(u32::from(e & 15))?;
            return Ok(e >> 4);
        }
        let (mut code, mut first, mut index) = (0i32, 0i32, 0i32);
        for l in 1..16 {
            code |= bits.take(1)? as i32;
            let count = i32::from(self.counts[l]);
            if code - count < first {
                return Ok(self.symbols[(index + code - first) as usize]);
            }
            (index, first, code) = (index + count, (first + count) << 1, code << 1);
        }
        Err(bad("a deflate stream has an invalid code"))
    }
}

/// The data of zlib stream `z`, checked against its Adler-32.
pub(super) fn zlib(z: &[u8]) -> Result<Vec<u8>> {
    if z.len() < 6 || z[0] & 15 != 8 || (u16::from(z[0]) << 8 | u16::from(z[1])) % 31 != 0 || z[1] & 32 != 0 {
        return Err(bad("not a zlib stream"));
    }
    let mut bits = Bits { data: &z[2..], at: 0, buf: 0, n: 0 };
    let mut out = Vec::new();
    while !block(&mut bits, &mut out)? {}
    let end = 2 + bits.at - (bits.n / 8) as usize;
    let want = z.get(end..end + 4).map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]));
    let (mut a, mut b) = (1u32, 0u32);
    for chunk in out.chunks(5552) {
        for &v in chunk {
            (a, b) = (a + u32::from(v), b + a + u32::from(v));
        }
        (a, b) = (a % 65521, b % 65521);
    }
    if want != Some(b << 16 | a) {
        return Err(bad("a zlib stream fails its checksum"));
    }
    Ok(out)
}

/// One block into `out`; whether it was the last.
fn block(bits: &mut Bits, out: &mut Vec<u8>) -> Result<bool> {
    let head = bits.take(3)?;
    match head >> 1 {
        0 => stored(bits, out)?,
        1 => {
            let mut l = [8u8; 288];
            l[144..256].fill(9);
            l[256..280].fill(7);
            codes(bits, out, &Huffman::new(&l), &Huffman::new(&[5; 30]))?;
        }
        2 => {
            let (lit, dist) = dynamic(bits)?;
            codes(bits, out, &lit, &dist)?;
        }
        _ => return Err(bad("a deflate block of type 3")),
    }
    Ok(head & 1 == 1)
}

/// A stored block: its length, the complement, the bytes.
fn stored(bits: &mut Bits, out: &mut Vec<u8>) -> Result<()> {
    bits.skip(bits.n % 8)?;
    let (len, nlen) = (bits.take(16)?, bits.take(16)?);
    if len != !nlen & 0xffff {
        return Err(bad("a stored deflate block has a bad length"));
    }
    (0..len).try_for_each(|_| bits.take(8).map(|b| out.push(b as u8)))
}

/// A dynamic block's literal/length and distance codes.
fn dynamic(bits: &mut Bits) -> Result<(Huffman, Huffman)> {
    let (nlit, ndist, ncode) = (bits.take(5)? as usize + 257, bits.take(5)? as usize + 1, bits.take(4)? as usize + 4);
    let mut cl = [0u8; 19];
    for &i in &ORDER[..ncode] {
        cl[i] = bits.take(3)? as u8;
    }
    let code = Huffman::new(&cl);
    let mut lengths = Vec::with_capacity(nlit + ndist);
    while lengths.len() < nlit + ndist {
        let s = code.decode(bits)?;
        let (repeat, value) = length(bits, s, lengths.last())?;
        lengths.extend(std::iter::repeat_n(value, repeat as usize));
    }
    if lengths.len() > nlit + ndist {
        return Err(bad("deflate code lengths overrun"));
    }
    Ok((Huffman::new(&lengths[..nlit]), Huffman::new(&lengths[nlit..])))
}

/// Code-length symbol `s` as a run: how many, of which length (`last` the length before).
fn length(bits: &mut Bits, s: u16, last: Option<&u8>) -> Result<(u32, u8)> {
    Ok(match s {
        0..=15 => (1, s as u8),
        16 => (3 + bits.take(2)?, *last.ok_or_else(|| bad("a deflate length repeats nothing"))?),
        17 => (3 + bits.take(3)?, 0),
        _ => (11 + bits.take(7)?, 0),
    })
}

/// A Huffman-coded block's literals and back-references, to its end code.
fn codes(bits: &mut Bits, out: &mut Vec<u8>, lit: &Huffman, dist: &Huffman) -> Result<()> {
    loop {
        let s = usize::from(lit.decode(bits)?);
        match s {
            0..256 => out.push(s as u8),
            256 => return Ok(()),
            257..=285 => {
                let len = usize::from(BASES[s - 257]) + bits.take(u32::from(EXTRA[s - 257]))? as usize;
                let back = distance(bits, dist)?;
                let from = out.len().checked_sub(back).ok_or_else(|| bad("a deflate distance reaches before the start"))?;
                (0..len).for_each(|i| out.push(out[from + i]));
            }
            _ => return Err(bad("a deflate length code past 285")),
        }
    }
}

/// A back-reference's distance: its code, then the code's extra bits.
fn distance(bits: &mut Bits, dist: &Huffman) -> Result<usize> {
    let d = usize::from(dist.decode(bits)?);
    if d >= 30 {
        return Err(bad("a deflate distance code past 29"));
    }
    let extra = if d < 2 { 0 } else { d as u32 / 2 - 1 };
    let base = if d < 4 { d + 1 } else { ((2 + (d & 1)) << extra) + 1 };
    Ok(base + bits.take(extra)? as usize)
}
