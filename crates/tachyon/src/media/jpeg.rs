//! Baseline and extended sequential JPEG (8-bit, Huffman-coded) to RGB, bit for bit as libjpeg-turbo gives it to
//! PIL: the integer ("islow") IDCT, fancy upsampling of chroma, the integer YCbCr conversion, and zeros for data
//! cut short. Progressive, lossless, arithmetic-coded, 12-bit and four-component files are refused.

// Names keep JPEG's letters: h, v sampling factors, x, y block positions, c a component, q its quantization.
#![allow(clippy::many_single_char_names)]

use super::{Image, bad, refuse};
use crate::Result;

/// Where each coefficient of the zigzag order lands in the block; past 63, at 63, as libjpeg guards it.
const NATURAL: [u8; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29,
    22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// A Huffman table: codes of up to 9 bits through `fast` (value `<< 8 |` length, 0 for longer), the rest by the
/// largest code of each length.
#[derive(Clone)]
struct Table {
    fast: Vec<u16>,
    maxcode: [i32; 17],
    offset: [i32; 17],
    values: Vec<u8>,
}

impl Table {
    fn new(counts: &[u8], values: &[u8]) -> Table {
        let (mut fast, mut maxcode, mut offset) = (vec![0u16; 512], [-1i32; 17], [0i32; 17]);
        let (mut code, mut k) = (0i32, 0usize);
        for l in 1..=16 {
            offset[l] = k as i32 - code;
            for _ in 0..counts[l - 1] {
                if l <= 9 && k < values.len() {
                    let first = (code << (9 - l)) as usize;
                    fast[first..first + (1 << (9 - l))].fill(u16::from(values[k]) << 8 | l as u16);
                }
                (code, k) = (code + 1, k + 1);
            }
            if counts[l - 1] > 0 {
                maxcode[l] = code - 1;
            }
            code <<= 1;
        }
        Table { fast, maxcode, offset, values: values.to_vec() }
    }
}

/// Entropy-coded bits, most significant first: `0xFF00` is a `0xFF` byte; from a marker (`stop`) or the end on,
/// zeros.
struct Bits<'a> {
    data: &'a [u8],
    at: usize,
    buf: u32,
    n: u32,
    stop: Option<usize>,
}

impl Bits<'_> {
    fn peek(&mut self, k: u32) -> u32 {
        while self.n <= 24 {
            let mut byte = 0;
            if self.stop.is_none() && self.at < self.data.len() {
                byte = self.data[self.at];
                if byte != 0xFF {
                    self.at += 1;
                } else if self.data.get(self.at + 1) == Some(&0) {
                    self.at += 2;
                } else {
                    (byte, self.stop) = (0, Some(self.at));
                }
            }
            self.buf |= u32::from(byte) << (24 - self.n);
            self.n += 8;
        }
        self.buf >> (32 - k)
    }

    /// At a restart: the bits left dropped, the `RSTn` marker passed if it is there.
    fn restart(&mut self) {
        (self.buf, self.n) = (0, 0);
        let at = self.stop.unwrap_or(self.at);
        if self.data.get(at) == Some(&0xFF) && self.data.get(at + 1).is_some_and(|m| (0xD0..=0xD7).contains(m)) {
            (self.at, self.stop) = (at + 2, None);
        }
    }

    /// Where the data after the scan starts: its marker.
    fn end(&self) -> usize {
        let mut at = self.stop.unwrap_or(self.at);
        let marker = |at: usize| self.data[at] == 0xFF && !matches!(self.data[at + 1], 0 | 0xD0..=0xD7 | 0xFF);
        while at + 1 < self.data.len() && !marker(at) {
            at += 1;
        }
        at
    }

    fn skip(&mut self, k: u32) {
        (self.buf, self.n) = (self.buf << k, self.n - k);
    }

    fn take(&mut self, k: u32) -> i32 {
        if k == 0 {
            return 0;
        }
        let v = self.peek(k);
        self.skip(k);
        v as i32
    }

    /// `k` bits as a signed value (JPEG's `EXTEND`).
    fn signed(&mut self, k: u32) -> i32 {
        let v = self.take(k);
        if k > 0 && v < 1 << (k - 1) { v - (1 << k) + 1 } else { v }
    }

    fn decode(&mut self, t: &Table) -> u8 {
        let e = t.fast[self.peek(9) as usize];
        if e != 0 {
            self.skip(u32::from(e & 0xFF));
            return (e >> 8) as u8;
        }
        let code = self.peek(16) as i32;
        for l in 10..=16 {
            let c = code >> (16 - l);
            if c <= t.maxcode[l] {
                self.skip(l as u32);
                return t.values.get((c + t.offset[l]) as usize).copied().unwrap_or(0);
            }
        }
        self.skip(16);
        0
    }
}

/// A frame component: its ID, sampling factors, tables, DC predictor, and its samples, a block-padded plane.
struct Component {
    id: u8,
    h: usize,
    v: usize,
    tq: usize,
    dc: usize,
    ac: usize,
    pred: i32,
    width: usize,
    plane: Vec<u8>,
}

/// What the markers before a scan have set.
struct Decoder {
    width: usize,
    height: usize,
    comps: Vec<Component>,
    quant: [[u16; 64]; 4],
    dc: Vec<Option<Table>>,
    ac: Vec<Option<Table>>,
    restart: usize,
    jfif: bool,
    adobe: Option<u8>,
    orientation: u16,
}

/// A frame component from its 3 header bytes: ID, sampling factors, quantization table.
fn component(c: &[u8]) -> Result<Component> {
    let (h, v) = (usize::from(c[1] >> 4), usize::from(c[1] & 15));
    if !(1..=4).contains(&h) || !(1..=4).contains(&v) || c[2] > 3 {
        return Err(bad("a JPEG component with invalid sampling or table"));
    }
    Ok(Component { id: c[0], h, v, tq: usize::from(c[2]), dc: 0, ac: 0, pred: 0, width: 0, plane: Vec::new() })
}

/// Decodes JPEG file `bytes`; its EXIF orientation, which the caller applies.
pub(super) fn decode(bytes: &[u8]) -> Result<(Image, u16)> {
    let mut d = Decoder {
        width: 0,
        height: 0,
        comps: Vec::new(),
        quant: [[0; 64]; 4],
        dc: vec![None; 4],
        ac: vec![None; 4],
        restart: 0,
        jfif: false,
        adobe: None,
        orientation: 1,
    };
    let mut at = 2;
    while let Some((marker, seg, next)) = segment(bytes, at, !d.comps.is_empty())? {
        at = d.segment(marker, seg, bytes, next)?;
    }
    let rgb = d.rgb()?;
    Ok((Image { width: d.width, height: d.height, rgb }, d.orientation))
}

/// The segment at `at`: its marker, its data and where the next starts; `None` at the end of the image (a file
/// cut short after its `framed` frame ends there too).
fn segment(bytes: &[u8], mut at: usize, framed: bool) -> Result<Option<(u8, &[u8], usize)>> {
    while bytes.get(at) == Some(&0xFF) && bytes.get(at + 1) == Some(&0xFF) {
        at += 1;
    }
    let (marker, len) = match bytes.get(at..at + 4) {
        Some(&[0xFF, 0xD9, ..]) => return Ok(None),
        Some(&[0xFF, m, a, b]) => (m, usize::from(a) << 8 | usize::from(b)),
        None if framed => return Ok(None),
        _ => return Err(bad("a JPEG marker is missing")),
    };
    let seg = bytes.get(at + 4..at + 2 + len).ok_or_else(|| bad("a JPEG segment ends early"))?;
    Ok(Some((marker, seg, at + 2 + len)))
}

impl Decoder {
    /// Segment `seg` of `marker`; where the data after it starts (`next`, or after a scan's data).
    fn segment(&mut self, marker: u8, seg: &[u8], bytes: &[u8], next: usize) -> Result<usize> {
        let done = match marker {
            0xC0 | 0xC1 => self.frame(seg),
            0xC4 => self.huffman(seg),
            0xDB => self.quantization(seg),
            0xDA => return self.scan(seg, bytes, next),
            0xDD => seg.get(..2).map(|r| self.restart = usize::from(r[0]) << 8 | usize::from(r[1])).ok_or_else(|| bad("a short DRI")),
            0xC2..=0xCF => Err(refuse("a progressive, lossless or arithmetic-coded JPEG")),
            _ => {
                self.app(marker, seg);
                Ok(())
            }
        };
        done.map(|()| next)
    }

    /// An application segment: JFIF, EXIF's orientation, Adobe's color transform.
    fn app(&mut self, marker: u8, seg: &[u8]) {
        match marker {
            0xE0 => self.jfif |= seg.starts_with(b"JFIF\0"),
            0xE1 if seg.starts_with(b"Exif\0\0") => self.orientation = orientation(&seg[6..]).unwrap_or(1),
            0xEE if seg.starts_with(b"Adobe") && seg.len() >= 12 => self.adobe = Some(seg[11]),
            _ => {}
        }
    }

    fn frame(&mut self, s: &[u8]) -> Result<()> {
        if s.len() < 6 || s[0] != 8 {
            return Err(refuse("a JPEG of other than 8 bits per sample"));
        }
        let (height, width, n) = (usize::from(s[1]) << 8 | usize::from(s[2]), usize::from(s[3]) << 8 | usize::from(s[4]), usize::from(s[5]));
        if !matches!(n, 1 | 3) || s.len() < 6 + 3 * n {
            return Err(refuse("a JPEG of other than 1 or 3 components"));
        }
        super::check_size(width, height)?;
        (self.width, self.height) = (width, height);
        self.comps = s[6..6 + 3 * n].chunks(3).map(component).collect::<Result<_>>()?;
        let (hmax, vmax) = self.max();
        let (mx, my) = (width.div_ceil(8 * hmax), height.div_ceil(8 * vmax));
        for c in &mut self.comps {
            c.width = mx * c.h * 8;
            c.plane = vec![0; c.width * my * c.v * 8];
        }
        Ok(())
    }

    fn max(&self) -> (usize, usize) {
        (self.comps.iter().map(|c| c.h).max().unwrap_or(1), self.comps.iter().map(|c| c.v).max().unwrap_or(1))
    }

    fn huffman(&mut self, mut s: &[u8]) -> Result<()> {
        while s.len() >= 17 {
            let (class, id, counts) = (s[0] >> 4, usize::from(s[0] & 3), &s[1..17]);
            let n = counts.iter().map(|&c| usize::from(c)).sum::<usize>();
            let values = s.get(17..17 + n).ok_or_else(|| bad("a short DHT"))?;
            let t = Some(Table::new(counts, values));
            if class == 0 {
                self.dc[id] = t;
            } else {
                self.ac[id] = t;
            }
            s = &s[17 + n..];
        }
        Ok(())
    }

    fn quantization(&mut self, mut s: &[u8]) -> Result<()> {
        while !s.is_empty() {
            let (wide, id) = (s[0] >> 4 != 0, usize::from(s[0] & 3));
            let len = if wide { 128 } else { 64 };
            let q = s.get(1..1 + len).ok_or_else(|| bad("a short DQT"))?;
            for k in 0..64 {
                let v = if wide { u16::from(q[2 * k]) << 8 | u16::from(q[2 * k + 1]) } else { u16::from(q[k]) };
                self.quant[id][usize::from(NATURAL[k])] = v;
            }
            s = &s[1 + len..];
        }
        Ok(())
    }

    /// A scan whose header is `s`, its data from `at` of `bytes`; where the data after it starts.
    fn scan(&mut self, s: &[u8], bytes: &[u8], at: usize) -> Result<usize> {
        let members = self.members(s)?;
        let mut bits = Bits { data: bytes, at, buf: 0, n: 0, stop: None };
        let (units, wide) = self.units(&members);
        for u in 0..units {
            if self.restart > 0 && u > 0 && u % self.restart == 0 {
                bits.restart();
                for &i in &members {
                    self.comps[i].pred = 0;
                }
            }
            self.unit(&mut bits, &members, (u % wide, u / wide))?;
        }
        Ok(bits.end())
    }

    /// The components scan header `s` names, their tables set and predictors reset.
    fn members(&mut self, s: &[u8]) -> Result<Vec<usize>> {
        let n = usize::from(*s.first().ok_or_else(|| bad("a short SOS"))?);
        let mut members = Vec::new();
        for c in s.get(1..1 + 2 * n).ok_or_else(|| bad("a short SOS"))?.chunks(2) {
            let i = self.comps.iter().position(|k| k.id == c[0]).ok_or_else(|| bad("a scan names an unknown component"))?;
            (self.comps[i].dc, self.comps[i].ac, self.comps[i].pred) = (usize::from(c[1] >> 4 & 3), usize::from(c[1] & 3), 0);
            members.push(i);
        }
        Ok(members)
    }

    /// A scan's MCUs and how many make a row: of one component its blocks, of several the frame's MCUs.
    fn units(&self, members: &[usize]) -> (usize, usize) {
        let (hmax, vmax) = self.max();
        if let [i] = members[..] {
            let c = &self.comps[i];
            let w = (self.width * c.h).div_ceil(hmax).div_ceil(8);
            (w * (self.height * c.v).div_ceil(vmax).div_ceil(8), w)
        } else {
            let w = self.width.div_ceil(8 * hmax);
            (w * self.height.div_ceil(8 * vmax), w)
        }
    }

    /// One MCU at `(x, y)`: each member's blocks (just one for a scan of one component).
    fn unit(&mut self, bits: &mut Bits, members: &[usize], (x, y): (usize, usize)) -> Result<()> {
        let one = members.len() == 1;
        for &i in members {
            let (h, v) = if one { (1, 1) } else { (self.comps[i].h, self.comps[i].v) };
            for b in 0..h * v {
                let (bx, by) = (x * h + b % h, y * v + b / h);
                let mut coef = [0i32; 64];
                self.block(bits, i, &mut coef)?;
                let c = &mut self.comps[i];
                idct(&coef, &mut c.plane[by * 8 * c.width + bx * 8..], c.width);
            }
        }
        Ok(())
    }

    /// One block of component `i`, dequantized, in natural order.
    fn block(&mut self, bits: &mut Bits, i: usize, coef: &mut [i32; 64]) -> Result<()> {
        let c = &mut self.comps[i];
        let (dc, ac) = (self.dc[c.dc].as_ref(), self.ac[c.ac].as_ref());
        let (dc, ac) = (dc.ok_or_else(|| bad("a scan uses a missing DC table"))?, ac.ok_or_else(|| bad("a scan uses a missing AC table"))?);
        let q = &self.quant[c.tq];
        let t = u32::from(bits.decode(dc));
        c.pred += bits.signed(t);
        coef[0] = c.pred * i32::from(q[0]);
        let mut k = 1;
        while k < 64 {
            let rs = bits.decode(ac);
            let (run, size) = (usize::from(rs >> 4), u32::from(rs & 15));
            if size == 0 {
                if run != 15 {
                    break;
                }
                k += 16;
                continue;
            }
            k += run;
            let z = usize::from(NATURAL[k.min(63)]);
            coef[z] = bits.signed(size) * i32::from(q[z]);
            k += 1;
        }
        Ok(())
    }
}

/// libjpeg's `FIX(x)` for the IDCT: `x · 2¹³`, rounded.
const C: [i64; 12] = [2446, 3196, 4433, 6270, 7373, 9633, 12_299, 15_137, 16_069, 16_819, 20_995, 25_172];

/// One 1-D pass of libjpeg's islow IDCT over `v` (8 values): the 8 results before descaling.
fn idct8(v: [i64; 8]) -> [i64; 8] {
    let z1 = (v[2] + v[6]) * C[2];
    let (t2, t3) = (z1 - v[6] * C[7], z1 + v[2] * C[3]);
    let (t0, t1) = ((v[0] + v[4]) << 13, (v[0] - v[4]) << 13);
    let (e10, e13, e11, e12) = (t0 + t3, t0 - t3, t1 + t2, t1 - t2);
    let (o0, o1, o2, o3) = (v[7], v[5], v[3], v[1]);
    let (z1, z2, z3, z4) = (o0 + o3, o1 + o2, o0 + o2, o1 + o3);
    let z5 = (z3 + z4) * C[5];
    let (z1, z2, z3, z4) = (-z1 * C[4], -z2 * C[10], -z3 * C[8] + z5, -z4 * C[1] + z5);
    let (o0, o1, o2, o3) = (o0 * C[0] + z1 + z3, o1 * C[9] + z2 + z4, o2 * C[11] + z2 + z3, o3 * C[6] + z1 + z4);
    [e10 + o3, e11 + o2, e12 + o1, e13 + o0, e13 - o0, e12 - o1, e11 - o2, e10 - o3]
}

/// libjpeg's islow IDCT of dequantized `coef` into 8 rows of `out` `stride` apart: columns, then rows, each
/// descaled with rounding; samples clamped as its range-limit table does.
fn idct(coef: &[i32; 64], out: &mut [u8], stride: usize) {
    let mut ws = [0i64; 64];
    for x in 0..8 {
        let r = idct8(std::array::from_fn(|k| i64::from(coef[8 * k + x])));
        (0..8).for_each(|k| ws[8 * k + x] = (r[k] + (1 << 10)) >> 11);
    }
    for y in 0..8 {
        let r = idct8(std::array::from_fn(|k| ws[8 * y + k]));
        for (x, v) in r.iter().enumerate() {
            let s = ((v + (1 << 17)) >> 18) & 1023;
            out[y * stride + x] = (if s >= 512 { s - 1024 } else { s } + 128).clamp(0, 255) as u8;
        }
    }
}

impl Decoder {
    /// The image's RGB: each component upsampled to full size, then YCbCr converted (or RGB taken as it is, or
    /// gray repeated).
    fn rgb(&self) -> Result<Vec<u8>> {
        let (w, h) = (self.width, self.height);
        let planes = self.comps.iter().map(|c| self.upsample(c)).collect::<Result<Vec<_>>>()?;
        let ids: Vec<u8> = self.comps.iter().map(|c| c.id).collect();
        let rgb_coded = !self.jfif && (self.adobe == Some(0) || self.adobe.is_none() && ids == *b"RGB");
        let mut out = Vec::with_capacity(w * h * 3);
        for i in 0..w * h {
            match &planes[..] {
                [g] => out.extend([g[i]; 3]),
                [r, g, b] if rgb_coded => out.extend([r[i], g[i], b[i]]),
                [y, cb, cr] => out.extend(ycc(y[i], cb[i], cr[i])),
                _ => return Err(bad("a JPEG without one or three components")),
            }
        }
        Ok(out)
    }

    /// Component `c` at full size, `width × height`, as libjpeg-turbo upsamples it: fancy (triangular) by 2 across,
    /// down or both, the rows past its last copies of that last row; other ratios by repetition.
    fn upsample(&self, c: &Component) -> Result<Vec<u8>> {
        let (hmax, vmax) = self.max();
        if hmax % c.h != 0 || vmax % c.v != 0 {
            return Err(refuse("a JPEG with fractional chroma sampling"));
        }
        let (dw, dh) = ((self.width * c.h).div_ceil(hmax), (self.height * c.v).div_ceil(vmax));
        let g = &Grid { c, fx: hmax / c.h, fy: vmax / c.v, dw, dh };
        Ok((0..self.height).flat_map(|y| (0..self.width).map(move |x| g.sample(x, y))).collect())
    }
}

/// A component's samples (`dw × dh` of them real) and its upsampling ratios.
struct Grid<'a> {
    c: &'a Component,
    fx: usize,
    fy: usize,
    dw: usize,
    dh: usize,
}

impl Grid<'_> {
    /// The sample at `(x, y)`, rows past the last real one reading it instead.
    fn at(&self, x: usize, y: usize) -> i32 {
        i32::from(self.c.plane[y.min(self.dh - 1) * self.c.width + x])
    }

    /// Output sample `(x, y)`: libjpeg-turbo's `h2v2_fancy_upsample`, `h2v1_fancy_upsample`, `h1v2_fancy_upsample`
    /// or a repeat, each blending the nearer input 3:1 with the next nearest, its rounding biased by position.
    fn sample(&self, x: usize, y: usize) -> u8 {
        let (sx, sy, odd_x, odd_y) = (x / self.fx, y / self.fy, (x % 2) as i32, y % 2);
        let near = if odd_y == 0 { sy.saturating_sub(1) } else { sy + 1 };
        let side = if odd_x == 0 { sx.saturating_sub(1) } else { (sx + 1).min(self.dw - 1) };
        let v = match (self.fx, self.fy) {
            (2, 2) if self.dw > 2 => {
                let col = |cx: usize| 3 * self.at(cx, sy) + self.at(cx, near);
                (3 * col(sx) + col(side) + 8 - odd_x) >> 4
            }
            (2, 1) if self.dw > 2 => (3 * self.at(sx, sy) + self.at(side, sy) + 1 + odd_x) >> 2,
            (1, 2) => (3 * self.at(sx, sy) + self.at(sx, near) + 1 + odd_y as i32) >> 2,
            _ => self.at(sx, sy),
        };
        v as u8
    }
}

/// libjpeg's integer YCbCr to RGB: tables of `FIX(x) = x · 2¹⁶` rounded, sums shifted down 16 bits.
fn ycc(y: u8, cb: u8, cr: u8) -> [u8; 3] {
    const HALF: i64 = 1 << 15;
    let (y, cb, cr) = (i64::from(y), i64::from(cb) - 128, i64::from(cr) - 128);
    let (r, b) = ((91_881 * cr + HALF) >> 16, (116_130 * cb + HALF) >> 16);
    let g = (-22_554 * cb + HALF - 46_802 * cr) >> 16;
    [(y + r).clamp(0, 255) as u8, (y + g).clamp(0, 255) as u8, (y + b).clamp(0, 255) as u8]
}

/// The orientation tag (0x0112) of the first IFD of TIFF data `t` (an EXIF segment past its `Exif\0\0`).
fn orientation(t: &[u8]) -> Option<u16> {
    let big = t.get(..2)? == b"MM";
    let u16_at = |i: usize| t.get(i..i + 2).map(|b| if big { u16::from_be_bytes([b[0], b[1]]) } else { u16::from_le_bytes([b[0], b[1]]) });
    let u32_at =
        |i: usize| t.get(i..i + 4).map(|b| if big { u32::from_be_bytes([b[0], b[1], b[2], b[3]]) } else { u32::from_le_bytes([b[0], b[1], b[2], b[3]]) });
    let ifd = u32_at(4)? as usize;
    (0..usize::from(u16_at(ifd)?)).map(|e| ifd + 2 + 12 * e).find(|&e| u16_at(e) == Some(0x0112)).and_then(|e| u16_at(e + 8)).filter(|o| (1..=8).contains(o))
}
