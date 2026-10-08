//! `.wpk`, Tachyon's packed weight file, and its [`Loader`].
//!
//! A `.wpk` is written once, when a model is pulled, so that loading it is a
//! straight copy with no transformation:
//!
//! ```text
//! 0          header, 4 KiB: magic, version, sizes, CRC32C of the table and of itself
//! 4096       data: every tensor, each starting on a 4 KiB boundary, in the layout kernels read
//! 4096 + D   table: per tensor its name, type, shape, offset in the data, length and CRC32C
//! ```
//!
//! The data region is read in one sequential pass and lands in VRAM as one
//! arena whose tensors are offsets. The 4 KiB alignment lets the loader read
//! with `O_DIRECT`, so the disk writes straight into pinned memory, and keeps
//! every tensor 256-byte aligned in VRAM. Opening a file checks the header and
//! the table; [`Wpk::verify`] checks the tensors' bytes, which loading never
//! touches with the CPU. Integers are little-endian.
//!
//! [`Reads::Auto`] picks the path per load: a file whose data the page cache
//! already holds, such as one a previous process loaded, is read through the
//! cache (memory speed); any other is read from the disk with `O_DIRECT`.

mod load;

pub use load::{CHUNK, DEPTH, Loader};

use crate::sys;

use crate::{Error, Result};
use std::fs::{File, OpenOptions};
use std::os::fd::AsRawFd;
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::path::Path;

/// Alignment of the data region and of every tensor in it, in bytes.
pub const ALIGN: u64 = 4096;
const MAGIC: &[u8; 8] = b"TACHYWPK";
const VERSION: u64 = 1;
const HEADER: usize = 48;
const MAX_RANK: usize = 8;
const O_DIRECT: i32 = 0o40000;

/// Element type of a tensor. A quantized weight is stored as several tensors, such as packed values and their scales.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Dtype {
    /// 32-bit float.
    F32,
    /// 16-bit float.
    F16,
    /// bfloat16.
    BF16,
    /// 8-bit float, 4 exponent and 3 mantissa bits.
    F8E4M3,
    /// Signed byte.
    I8,
    /// Unsigned byte, also packed 4-bit values two per byte.
    U8,
}

impl Dtype {
    const ALL: [Dtype; 6] = [Dtype::F32, Dtype::F16, Dtype::BF16, Dtype::F8E4M3, Dtype::I8, Dtype::U8];

    /// Bytes per element.
    pub fn size(self) -> u64 {
        match self {
            Dtype::F32 => 4,
            Dtype::F16 | Dtype::BF16 => 2,
            Dtype::F8E4M3 | Dtype::I8 | Dtype::U8 => 1,
        }
    }
}

/// One tensor of a `.wpk`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tensor {
    /// Name, as the model's checkpoint names it.
    pub name: String,
    /// Element type.
    pub dtype: Dtype,
    /// Dimensions, outermost first.
    pub shape: Vec<u64>,
    /// Byte offset in the data region, a multiple of [`ALIGN`].
    pub offset: u64,
    /// Length in bytes: the product of `shape` and the element size.
    pub len: u64,
    /// CRC32C of the bytes.
    pub crc: u32,
}

fn bytes_of(dtype: Dtype, shape: &[u64]) -> Option<u64> {
    shape.iter().try_fold(dtype.size(), |n, &d| n.checked_mul(d))
}

fn bad<T>(why: impl Into<String>) -> Result<T> {
    Err(Error::Format(why.into()))
}

/// The message of the first check that fails.
fn first_failure<'a>(checks: &[(bool, &'a str)]) -> Option<&'a str> {
    checks.iter().find(|(ok, _)| !ok).map(|(_, why)| *why)
}

/// CRC32C (Castagnoli) of `bytes`, continuing from `crc` (0 to start).
fn crc32c(crc: u32, bytes: &[u8]) -> u32 {
    if is_x86_feature_detected!("sse4.2") {
        // SAFETY: the CPU has SSE4.2.
        return unsafe { crc32c_sse42(crc, bytes) };
    }
    crc32c_soft(crc, bytes)
}

#[target_feature(enable = "sse4.2")]
fn crc32c_sse42(crc: u32, bytes: &[u8]) -> u32 {
    use core::arch::x86_64::{_mm_crc32_u8, _mm_crc32_u64};
    let mut words = bytes.chunks_exact(8);
    let wide = (&mut words).fold(u64::from(!crc), |c, w| _mm_crc32_u64(c, u64::from_le_bytes(w.try_into().unwrap())));
    !words.remainder().iter().fold(wide as u32, |c, &b| _mm_crc32_u8(c, b))
}

fn crc32c_soft(crc: u32, bytes: &[u8]) -> u32 {
    !bytes.iter().fold(!crc, |c, &b| (0..8).fold(c ^ u32::from(b), |c, _| (c >> 1) ^ (0x82F6_3B78 & (c & 1).wrapping_neg())))
}

/// Reads little-endian fields; past the end it yields zeros and remembers it.
struct Cursor<'a> {
    rest: &'a [u8],
    short: bool,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> &'a [u8] {
        let Some((head, rest)) = self.rest.split_at_checked(n) else {
            self.short = true;
            return &[];
        };
        self.rest = rest;
        head
    }

    fn int<const N: usize>(&mut self) -> u64 {
        self.take(N).iter().rev().fold(0, |v, &b| v << 8 | u64::from(b))
    }
}

fn encode(tensors: &[Tensor]) -> Vec<u8> {
    let mut out = Vec::new();
    for t in tensors {
        out.extend_from_slice(&(t.name.len() as u16).to_le_bytes());
        out.extend_from_slice(t.name.as_bytes());
        out.extend_from_slice(&[t.dtype as u8, t.shape.len() as u8]);
        t.shape.iter().chain([&t.offset, &t.len]).for_each(|v| out.extend_from_slice(&v.to_le_bytes()));
        out.extend_from_slice(&t.crc.to_le_bytes());
    }
    out
}

fn decode(c: &mut Cursor, data_len: u64) -> Result<Tensor> {
    let name_len = c.int::<2>() as usize;
    let name = String::from_utf8_lossy(c.take(name_len)).into_owned();
    let (dtype, rank) = (Dtype::ALL.get(c.int::<1>() as usize).copied(), c.int::<1>() as usize);
    let shape: Vec<u64> = (0..rank.min(MAX_RANK)).map(|_| c.int::<8>()).collect();
    let (offset, len, crc) = (c.int::<8>(), c.int::<8>(), c.int::<4>() as u32);
    let failure = first_failure(&[
        (!c.short, "the tensor table is truncated"),
        (dtype.is_some(), "unknown element type"),
        (rank <= MAX_RANK, "more than 8 dimensions"),
        (offset % ALIGN == 0, "not 4 KiB aligned"),
        (dtype.and_then(|d| bytes_of(d, &shape)) == Some(len), "its length does not match its shape"),
        (offset.checked_add(len).is_some_and(|end| end <= data_len), "it lies outside the data region"),
    ]);
    match (failure, dtype) {
        (None, Some(dtype)) => Ok(Tensor { name, dtype, shape, offset, len, crc }),
        (why, _) => bad(format!("tensor {name:?}: {}", why.unwrap_or("invalid"))),
    }
}

fn header(count: usize, data_len: u64, table: &[u8]) -> Vec<u8> {
    let mut head = MAGIC.to_vec();
    for v in [VERSION | (count as u64) << 32, data_len, ALIGN + data_len, table.len() as u64] {
        head.extend_from_slice(&v.to_le_bytes());
    }
    head.extend_from_slice(&crc32c(0, table).to_le_bytes());
    head.extend_from_slice(&crc32c(0, &head).to_le_bytes());
    head.resize(ALIGN as usize, 0);
    head
}

/// The tensors and data length of an open file, after checking the header and the table.
fn index(file: &File) -> Result<(Vec<Tensor>, u64)> {
    let mut head = [0; HEADER];
    file.read_exact_at(&mut head, 0)?;
    let mut c = Cursor { rest: &head, short: false };
    let (magic, version, data_len, table_at, table_len) = (c.take(8), c.int::<8>(), c.int::<8>(), c.int::<8>(), c.int::<8>());
    let (table_crc, head_crc, size) = (c.int::<4>() as u32, c.int::<4>() as u32, file.metadata()?.len());
    if let Some(why) = first_failure(&[
        (magic == MAGIC, "not a .wpk file"),
        (crc32c(0, &head[..HEADER - 4]) == head_crc, "the header is corrupt"),
        (version as u32 == VERSION as u32, "unsupported version"),
        (data_len % ALIGN == 0 && table_at == ALIGN + data_len, "the data region is malformed"),
        (table_at.checked_add(table_len).is_some_and(|end| end <= size), "the file is truncated"),
    ]) {
        return bad(why);
    }
    let mut table = vec![0; table_len as usize];
    file.read_exact_at(&mut table, table_at)?;
    if crc32c(0, &table) != table_crc {
        return bad("the tensor table is corrupt");
    }
    let mut c = Cursor { rest: &table, short: false };
    let tensors = (0..version >> 32).map(|_| decode(&mut c, data_len)).collect::<Result<Vec<_>>>()?;
    if c.rest.is_empty() { Ok((tensors, data_len)) } else { bad("the tensor table has trailing bytes") }
}

/// How loads read a `.wpk`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reads {
    /// Through the page cache when it holds at least 90% of the data, with `O_DIRECT` otherwise.
    Auto,
    /// From the disk with `O_DIRECT`, bypassing the page cache.
    Direct,
    /// Through the page cache. Where the filesystem refuses `O_DIRECT`, every policy becomes this one.
    Cached,
}

/// An open `.wpk`: its checked index, and the file to load from.
pub struct Wpk {
    index: File,
    data: File,
    reads: Reads,
    tensors: Vec<Tensor>,
    data_len: u64,
}

impl Wpk {
    /// Opens and checks `path`, with [`Reads::Auto`].
    pub fn open(path: impl AsRef<Path>) -> Result<Wpk> {
        Wpk::open_with(path, Reads::Auto)
    }

    /// Opens and checks `path`, with the read policy `reads`.
    pub fn open_with(path: impl AsRef<Path>, reads: Reads) -> Result<Wpk> {
        let index = File::open(&path)?;
        let (tensors, data_len) = self::index(&index)?;
        let uncached = (reads != Reads::Cached).then(|| OpenOptions::new().read(true).custom_flags(O_DIRECT).open(&path).ok()).flatten();
        let (data, reads) = match uncached {
            Some(file) => (file, reads),
            None => (index.try_clone()?, Reads::Cached),
        };
        Ok(Wpk { index, data, reads, tensors, data_len })
    }

    /// Every tensor, in file order.
    pub fn tensors(&self) -> &[Tensor] {
        &self.tensors
    }

    /// The tensor called `name`.
    pub fn tensor(&self, name: &str) -> Option<&Tensor> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Bytes of the data region, padded to [`ALIGN`]: the size of the arena it loads into.
    pub fn data_len(&self) -> usize {
        self.data_len as usize
    }

    /// The read policy in effect.
    pub fn reads(&self) -> Reads {
        self.reads
    }

    /// The share of the data region the page cache holds now, from 0 to 1.
    pub fn cached(&self) -> f64 {
        sys::resident(self.index.as_raw_fd(), ALIGN, self.data_len as usize)
    }

    /// The file the next load reads, by the read policy and what the page cache holds now.
    fn source(&self) -> &File {
        match self.reads {
            Reads::Direct => &self.data,
            Reads::Auto if self.cached() < 0.9 => &self.data,
            Reads::Auto | Reads::Cached => &self.index,
        }
    }

    /// Checks every tensor's bytes against its CRC32C, reading through the page cache 16 MiB at a time.
    pub fn verify(&self) -> Result<()> {
        let mut buf = vec![0; CHUNK];
        for t in &self.tensors {
            let mut crc = 0;
            for at in (0..t.len).step_by(CHUNK) {
                let part = &mut buf[..CHUNK.min((t.len - at) as usize)];
                self.index.read_exact_at(part, ALIGN + t.offset + at)?;
                crc = crc32c(crc, part);
            }
            if crc != t.crc {
                return bad(format!("tensor {:?}: its bytes do not match their checksum", t.name));
            }
        }
        Ok(())
    }
}

/// Writes a `.wpk`: tensors one by one, then the table and header on [`Writer::finish`].
pub struct Writer {
    file: File,
    tensors: Vec<Tensor>,
    data_len: u64,
}

impl Writer {
    /// Creates (or truncates) `path`.
    pub fn create(path: impl AsRef<Path>) -> Result<Writer> {
        Ok(Writer { file: File::create(path)?, tensors: Vec::new(), data_len: 0 })
    }

    /// Appends a tensor; `bytes` must hold exactly `shape` elements of `dtype`.
    pub fn add(&mut self, name: &str, dtype: Dtype, shape: &[u64], bytes: &[u8]) -> Result<()> {
        if bytes_of(dtype, shape) != Some(bytes.len() as u64) || shape.len() > MAX_RANK || name.len() > usize::from(u16::MAX) {
            return bad(format!("tensor {name:?}: its bytes, type and shape do not agree"));
        }
        self.file.write_all_at(bytes, ALIGN + self.data_len)?;
        let (offset, len) = (self.data_len, bytes.len() as u64);
        self.tensors.push(Tensor { name: name.into(), dtype, shape: shape.to_vec(), offset, len, crc: crc32c(0, bytes) });
        self.data_len = (offset + len).next_multiple_of(ALIGN);
        Ok(())
    }

    /// Writes the table and the header, and flushes the file to disk.
    pub fn finish(self) -> Result<()> {
        let table = encode(&self.tensors);
        self.file.write_all_at(&table, ALIGN + self.data_len)?;
        self.file.write_all_at(&header(self.tensors.len(), self.data_len, &table), 0)?;
        Ok(self.file.sync_all()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Temp(std::path::PathBuf);

    impl Temp {
        fn new(name: &str) -> Temp {
            Temp(std::env::temp_dir().join(format!("tachyon-{}-{name}.wpk", std::process::id())))
        }
    }

    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn sample(name: &str) -> (Temp, Vec<u8>) {
        let tmp = Temp::new(name);
        let bytes: Vec<u8> = (0..10_000u32).map(|i| (i * 7 % 251) as u8).collect();
        let mut w = Writer::create(&tmp.0).unwrap();
        w.add("embed", Dtype::BF16, &[50, 100], &bytes).unwrap();
        w.add("norm", Dtype::F32, &[3], &bytes[..12]).unwrap();
        w.add("packed", Dtype::U8, &[10, 2, 5], &bytes[..100]).unwrap();
        w.finish().unwrap();
        (tmp, bytes)
    }

    fn rejection(path: &Path) -> String {
        match Wpk::open(path) {
            Err(Error::Format(why)) => why,
            other => panic!("expected a format error, got {:?}", other.err()),
        }
    }

    fn patch(path: &Path, at: u64, bytes: &[u8]) {
        OpenOptions::new().write(true).open(path).unwrap().write_all_at(bytes, at).unwrap();
    }

    #[test]
    fn crc32c_matches_the_standard_check_value() {
        assert_eq!(crc32c(0, b"123456789"), 0xE306_9283);
        assert_eq!(crc32c_soft(0, b"123456789"), 0xE306_9283);
        let long: Vec<u8> = (0..1000u32).map(|i| i as u8).collect();
        assert_eq!(crc32c(crc32c(0, &long[..333]), &long[333..]), crc32c_soft(0, &long));
    }

    #[test]
    fn written_tensors_read_back_aligned() {
        let (tmp, bytes) = sample("roundtrip");
        let wpk = Wpk::open(&tmp.0).unwrap();
        let names: Vec<&str> = wpk.tensors().iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["embed", "norm", "packed"]);
        assert!(wpk.tensors().iter().all(|t| t.offset % ALIGN == 0));
        assert_eq!(wpk.tensor("norm").unwrap().shape, [3]);
        assert_eq!(wpk.data_len(), 12288 + 2 * ALIGN as usize);
        wpk.verify().unwrap();
        let t = wpk.tensor("embed").unwrap();
        let mut back = vec![0; t.len as usize];
        File::open(&tmp.0).unwrap().read_exact_at(&mut back, ALIGN + t.offset).unwrap();
        assert_eq!(back, bytes);
    }

    #[test]
    fn mismatched_tensors_are_refused() {
        let tmp = Temp::new("mismatch");
        let mut w = Writer::create(&tmp.0).unwrap();
        assert!(w.add("x", Dtype::F16, &[3], &[0; 5]).is_err());
        assert!(w.add("x", Dtype::U8, &[1; 9], &[0; 1]).is_err());
    }

    #[test]
    fn damaged_files_are_rejected() {
        let (tmp, _) = sample("damaged");
        let size = std::fs::metadata(&tmp.0).unwrap().len();
        patch(&tmp.0, size - 1, &[0xff]);
        assert_eq!(rejection(&tmp.0), "the tensor table is corrupt");
        patch(&tmp.0, 9, &[0xff]);
        assert_eq!(rejection(&tmp.0), "the header is corrupt");
        OpenOptions::new().write(true).open(&tmp.0).unwrap().set_len(size - 1).unwrap();
        assert!(rejection(&tmp.0).contains("truncated") || rejection(&tmp.0).contains("corrupt"));
        patch(&tmp.0, 0, b"NOTAWPK!");
        assert_eq!(rejection(&tmp.0), "not a .wpk file");
        std::fs::write(&tmp.0, b"tiny").unwrap();
        assert_eq!(rejection(&tmp.0), "the file ends early");
    }

    #[test]
    fn corrupted_data_fails_verification() {
        let (tmp, _) = sample("bitrot");
        patch(&tmp.0, ALIGN + 5000, &[0xff]);
        let wpk = Wpk::open(&tmp.0).unwrap();
        assert!(matches!(wpk.verify(), Err(Error::Format(why)) if why.contains("\"embed\"")));
    }

    #[test]
    fn auto_reads_through_the_cache_only_when_it_holds_the_data() {
        let (tmp, _) = sample("auto");
        let wpk = Wpk::open(&tmp.0).unwrap();
        crate::sys::testing::evict(wpk.index.as_raw_fd());
        if wpk.reads() != Reads::Auto || wpk.cached() > 0.5 {
            return; // no O_DIRECT, or a filesystem whose pages cannot be evicted (tmpfs)
        }
        assert_eq!(wpk.source().as_raw_fd(), wpk.data.as_raw_fd(), "evicted: read from the disk");
        wpk.verify().unwrap();
        assert!(wpk.cached() > 0.9);
        assert_eq!(wpk.source().as_raw_fd(), wpk.index.as_raw_fd(), "cached: read through the cache");
    }

    #[test]
    fn misaligned_tensors_are_rejected() {
        let tmp = Temp::new("misaligned");
        let t = Tensor { name: "w".into(), dtype: Dtype::U8, shape: vec![8], offset: 100, len: 8, crc: 0 };
        let table = encode(&[t]);
        let file = File::create(&tmp.0).unwrap();
        file.write_all_at(&table, 2 * ALIGN).unwrap();
        file.write_all_at(&header(1, ALIGN, &table), 0).unwrap();
        assert_eq!(rejection(&tmp.0), "tensor \"w\": not 4 KiB aligned");
    }
}
