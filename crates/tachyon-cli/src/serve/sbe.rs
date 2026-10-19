//! The messages of `interface/tachyon.xml`, read and written as SBE lays them out: a header, the fixed block, the
//! groups, then the variable-length data, little-endian. Reading is bounds-checked at every step and copies what it
//! keeps, so a malformed or changing request is refused, never trusted.

/// The schema's id and version, in every header.
const SCHEMA: u16 = 7;
const VERSION: u16 = 1;

/// Bytes in the data area: offset and length.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Blob {
    pub offset: u64,
    pub length: u64,
}

/// A reader over a message's bytes.
pub struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

/// A malformed message.
#[derive(Debug)]
pub struct Bad;

impl<'a> Reader<'a> {
    /// The message at the start of `bytes`: its template id and its fixed block, the reader past it.
    pub fn open(bytes: &'a [u8]) -> Result<(u16, Block<'a>, Reader<'a>), Bad> {
        let mut r = Reader { bytes, at: 0 };
        let head = r.take(8)?;
        let (len, template, schema) = (u16_at(head, 0), u16_at(head, 2), u16_at(head, 4));
        if schema != SCHEMA {
            return Err(Bad);
        }
        let block = Block(r.take(usize::from(len))?);
        Ok((template, block, r))
    }

    /// The next `n` bytes.
    fn take(&mut self, n: usize) -> Result<&'a [u8], Bad> {
        let out = self.bytes.get(self.at..self.at.checked_add(n).ok_or(Bad)?).ok_or(Bad)?;
        self.at += n;
        Ok(out)
    }

    /// A group's header: its entries' block length and their count.
    pub fn group(&mut self) -> Result<(usize, usize), Bad> {
        let h = self.take(4)?;
        Ok((usize::from(u16_at(h, 0)), usize::from(u16_at(h, 2))))
    }

    /// The next entry's fixed block of a group whose entries take `len` bytes.
    pub fn entry(&mut self, len: usize) -> Result<Block<'a>, Bad> {
        self.take(len).map(Block)
    }

    /// The next variable-length data, as text.
    pub fn text(&mut self) -> Result<String, Bad> {
        let n = u32_at(self.take(4)?, 0) as usize;
        String::from_utf8(self.take(n)?.to_vec()).map_err(|_| Bad)
    }
}

/// A fixed block: fields at their offsets, a missing field (an older, shorter block) refused.
#[derive(Clone, Copy)]
pub struct Block<'a>(&'a [u8]);

impl Block<'_> {
    pub fn u8(self, at: usize) -> Result<u8, Bad> {
        self.0.get(at).copied().ok_or(Bad)
    }

    pub fn u32(self, at: usize) -> Result<u32, Bad> {
        self.0.get(at..at + 4).map(|b| u32_at(b, 0)).ok_or(Bad)
    }

    pub fn u64(self, at: usize) -> Result<u64, Bad> {
        self.0.get(at..at + 8).map(|b| u64::from_le_bytes(b.try_into().unwrap_or_default())).ok_or(Bad)
    }

    pub fn f32(self, at: usize) -> Result<f32, Bad> {
        self.u32(at).map(f32::from_bits)
    }

    pub fn blob(self, at: usize) -> Result<Blob, Bad> {
        Ok(Blob { offset: self.u64(at)?, length: self.u64(at + 8)? })
    }
}

fn u16_at(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

/// A message being written: header and block first, then groups and data in the schema's order.
pub struct Writer(pub Vec<u8>);

impl Writer {
    /// A message of `template` whose block is `block`.
    pub fn new(template: u16, block: &[u8]) -> Writer {
        let mut out = Vec::with_capacity(64);
        for v in [block.len() as u16, template, SCHEMA, VERSION] {
            out.extend(v.to_le_bytes());
        }
        out.extend(block);
        Writer(out)
    }

    /// A group of `count` entries of `len` bytes each, to be written next.
    pub fn group(&mut self, len: usize, count: usize) -> &mut Writer {
        self.0.extend((len as u16).to_le_bytes());
        self.0.extend((count as u16).to_le_bytes());
        self
    }

    /// Bytes of an entry's block.
    pub fn bytes(&mut self, b: &[u8]) -> &mut Writer {
        self.0.extend(b);
        self
    }

    /// Variable-length data.
    pub fn text(&mut self, t: &str) -> &mut Writer {
        self.0.extend((t.len() as u32).to_le_bytes());
        self.0.extend(t.as_bytes());
        self
    }
}

/// A fixed block built field by field, in the schema's order.
#[derive(Default)]
pub struct Fields(pub Vec<u8>);

impl Fields {
    pub fn u64(mut self, v: u64) -> Fields {
        self.0.extend(v.to_le_bytes());
        self
    }

    pub fn u32(mut self, v: u32) -> Fields {
        self.0.extend(v.to_le_bytes());
        self
    }

    pub fn u16(mut self, v: u16) -> Fields {
        self.0.extend(v.to_le_bytes());
        self
    }

    pub fn u8(mut self, v: u8) -> Fields {
        self.0.push(v);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::{Fields, Reader, Writer};

    #[test]
    fn a_message_reads_back_as_written() {
        let mut w = Writer::new(2, &Fields::default().u64(9).u64(16).u64(64).u32(3).0);
        w.group(0, 2).text("one").text("two").text("granite");
        let (template, block, mut r) = Reader::open(&w.0).unwrap();
        assert_eq!((template, block.u64(0).unwrap(), block.blob(8).unwrap().length, block.u32(24).unwrap()), (2, 9, 64, 3));
        let (len, n) = r.group().unwrap();
        let texts: Vec<String> = (0..n).map(|_| r.entry(len).and_then(|_| r.text()).unwrap()).collect();
        assert_eq!((texts, r.text().unwrap()), (vec!["one".to_string(), "two".into()], "granite".into()));
        assert!(Reader::open(&w.0[..20]).is_err() || Reader::open(&w.0[..20]).unwrap().2.group().is_err(), "a cut message is refused");
    }
}
