//! Streaming a `.wpk` data region into VRAM or pinned host memory.
//!
//! Reads go through `io_uring`, [`DEPTH`] chunks of [`CHUNK`] bytes in flight.
//! With `O_DIRECT` the disk writes straight into pinned memory and the CPU
//! never touches the bytes. On the way to VRAM each chunk lands in one of
//! [`DEPTH`] pinned staging slots and is uploaded on a copy stream while the
//! other reads proceed; a slot is refilled once its upload's event completes.
//! Into host memory, reads land straight in the destination, which then serves
//! as the host-RAM tier: moving it to VRAM is one copy at full PCIe speed.

use super::{ALIGN, Wpk, bad};
use crate::cuda::{Context, DevBuf, Event, HostBuf, Stream};
use crate::sys::{OP_READ, Ring, Sqe};
use crate::{Error, Result};
use std::os::fd::AsRawFd;
use std::ptr::null_mut;

/// Bytes per read, and per staging slot.
pub const CHUNK: usize = 16 << 20;
/// Reads in flight, and staging slots.
pub const DEPTH: usize = 4;

/// Loads `.wpk` files: an `io_uring`, [`DEPTH`] pinned staging slots and a copy stream.
pub struct Loader {
    ring: Ring,
    staging: HostBuf,
    stream: Stream,
    uploaded: Vec<Event>,
}

impl Loader {
    /// A loader for the GPU of `ctx`, holding `DEPTH × CHUNK` bytes of pinned staging memory.
    pub fn new(ctx: &Context) -> Result<Loader> {
        let ring = Ring::new(2 * DEPTH as u32).map_err(|e| Error::Io(-e as i32))?;
        let uploaded = (0..DEPTH).map(|_| ctx.event(false)).collect::<Result<_>>()?;
        Ok(Loader { ring, staging: ctx.alloc_host(DEPTH * CHUNK)?, stream: ctx.stream()?, uploaded })
    }

    /// Reads the data region of `wpk` into the start of `dst`, returning once it is in VRAM.
    pub fn to_device(&mut self, wpk: &Wpk, dst: &DevBuf) -> Result<()> {
        let len = wpk.data_len();
        if dst.len() < len {
            return Err(Error::OutOfRange);
        }
        let base = self.staging.as_mut_ptr();
        let (stream, uploaded) = (&self.stream, &self.uploaded);
        let free_slot = |_, slot: usize| uploaded[slot].sync().map(|()| base.wrapping_add(slot * CHUNK));
        let upload = |chunk: usize, slot: usize| {
            // SAFETY: the slot holds the chunk just read, and is not refilled before `uploaded[slot]` completes.
            let bytes = unsafe { std::slice::from_raw_parts(base.add(slot * CHUNK), CHUNK.min(len - chunk * CHUNK)) };
            // SAFETY: the staging memory outlives the copy, which `stream.sync` below completes.
            unsafe { stream.upload(dst, chunk * CHUNK, bytes) }.and_then(|()| stream.record(&uploaded[slot]))
        };
        let read = read(&mut self.ring, wpk, len, free_slot, upload);
        read.and(self.stream.sync())
    }

    /// Reads the data region of `wpk` straight into `dst`, which then holds the model in the host-RAM tier.
    pub fn to_host(&mut self, wpk: &Wpk, dst: &mut HostBuf) -> Result<()> {
        let len = wpk.data_len();
        if dst.len() < len {
            return Err(Error::OutOfRange);
        }
        let base = dst.as_mut_ptr();
        read(&mut self.ring, wpk, len, |chunk, _| Ok(base.wrapping_add(chunk * CHUNK)), |_, _| Ok(()))
    }

    /// Copies `src` to the start of `dst`, returning once it is in VRAM. From pinned memory it runs at full PCIe speed.
    pub fn upload(&self, src: &[u8], dst: &DevBuf) -> Result<()> {
        // SAFETY: `src` stays borrowed until the copy has completed, below.
        unsafe { self.stream.upload(dst, 0, src) }?;
        self.stream.sync()
    }
}

/// A chunk being read: where it goes, how much of it is wanted, and how much has arrived.
#[derive(Clone, Copy)]
struct Slot {
    chunk: usize,
    ptr: *mut u8,
    want: usize,
    done: usize,
}

struct Reads<'r> {
    ring: &'r mut Ring,
    fd: i32,
    len: usize,
    slots: [Slot; DEPTH],
    next: usize,
    busy: usize,
}

impl Reads<'_> {
    /// Asks for the rest of slot `s`'s chunk.
    fn submit(&mut self, s: usize) {
        let Slot { chunk, ptr, want, done } = self.slots[s];
        let off = ALIGN + (chunk * CHUNK + done) as u64;
        self.ring.push(Sqe {
            opcode: OP_READ,
            fd: self.fd,
            off,
            addr: ptr as u64 + done as u64,
            len: (want - done) as u32,
            user_data: s as u64,
            ..Sqe::default()
        });
        self.busy += 1;
    }

    /// Starts the next chunk, if any is left, in slot `s`, at the address `target` gives.
    fn refill(&mut self, s: usize, target: &mut impl FnMut(usize, usize) -> Result<*mut u8>) -> Result<()> {
        let chunk = self.next;
        if chunk * CHUNK >= self.len {
            return Ok(());
        }
        self.next += 1;
        self.slots[s] = Slot { chunk, ptr: target(chunk, s)?, want: CHUNK.min(self.len - chunk * CHUNK), done: 0 };
        self.submit(s);
        Ok(())
    }

    /// Waits for one read; returns the slot whose chunk it completed, if it did.
    fn complete(&mut self) -> Result<Option<usize>> {
        let cqe = loop {
            if let Some(cqe) = self.ring.completion() {
                break cqe;
            }
            self.ring.enter(1);
        };
        self.busy -= 1;
        let s = cqe.user_data as usize;
        match cqe.res {
            r if r < 0 => Err(Error::Io(-r)),
            0 => bad("the file ends early"),
            r => {
                self.slots[s].done += r as usize;
                if self.slots[s].done == self.slots[s].want {
                    return Ok(Some(s));
                }
                self.submit(s);
                Ok(None)
            }
        }
    }
}

/// Reads the first `len` bytes of `wpk`'s data region with [`DEPTH`] reads in flight. `target(chunk, slot)` says where
/// a chunk goes; `landed(chunk, slot)` runs once it is whole. After a failure it still waits for every read in
/// flight, so no buffer is written once this returns.
fn read(
    ring: &mut Ring,
    wpk: &Wpk,
    len: usize,
    mut target: impl FnMut(usize, usize) -> Result<*mut u8>,
    mut landed: impl FnMut(usize, usize) -> Result<()>,
) -> Result<()> {
    let idle = Slot { chunk: 0, ptr: null_mut(), want: 0, done: 0 };
    let mut reads = Reads { ring, fd: wpk.data.as_raw_fd(), len, slots: [idle; DEPTH], next: 0, busy: 0 };
    let mut result = (0..DEPTH).try_for_each(|s| reads.refill(s, &mut target));
    while reads.busy > 0 {
        match reads.complete() {
            Ok(Some(s)) if result.is_ok() => result = landed(reads.slots[s].chunk, s).and_then(|()| reads.refill(s, &mut target)),
            Err(e) if result.is_ok() => result = Err(e),
            _ => {}
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wpk::{Dtype, Writer};
    use std::alloc::{Layout, alloc_zeroed, dealloc};

    /// Host memory aligned for `O_DIRECT`, standing in for pinned memory.
    struct Aligned(*mut u8, Layout);

    impl Aligned {
        fn new(len: usize) -> Aligned {
            let layout = Layout::from_size_align(len, ALIGN as usize).unwrap();
            // SAFETY: `layout` has a non-zero size.
            Aligned(unsafe { alloc_zeroed(layout) }, layout)
        }

        fn bytes(&self) -> &[u8] {
            // SAFETY: the allocation holds `layout.size()` initialized bytes.
            unsafe { std::slice::from_raw_parts(self.0, self.1.size()) }
        }
    }

    impl Drop for Aligned {
        fn drop(&mut self) {
            // SAFETY: allocated in `new` with this layout.
            unsafe { dealloc(self.0, self.1) };
        }
    }

    /// A one-tensor `.wpk` of `len` bytes spanning several chunks.
    fn sample(name: &str, len: usize) -> (std::path::PathBuf, Vec<u8>) {
        let path = std::env::temp_dir().join(format!("tachyon-{}-{name}.wpk", std::process::id()));
        let bytes: Vec<u8> = (0..len).map(|i| (i % 251) as u8 ^ (i >> 20) as u8).collect();
        let mut w = Writer::create(&path).unwrap();
        w.add("w", Dtype::U8, &[len as u64], &bytes).unwrap();
        w.finish().unwrap();
        (path, bytes)
    }

    fn load(wpk: &Wpk, landed: impl FnMut(usize, usize) -> Result<()>) -> Result<Aligned> {
        let mut ring = Ring::new(8).unwrap();
        let buf = Aligned::new(wpk.data_len());
        read(&mut ring, wpk, wpk.data_len(), |chunk, _| Ok(buf.0.wrapping_add(chunk * CHUNK)), landed).map(|()| buf)
    }

    #[test]
    fn every_chunk_arrives_direct_or_buffered() {
        let (path, bytes) = sample("chunks", 5 * CHUNK + 12_345);
        for direct in [true, false] {
            let wpk = Wpk::open_with(&path, direct).unwrap();
            let mut seen = Vec::new();
            let buf = load(&wpk, |chunk, _| {
                seen.push(chunk);
                Ok(())
            })
            .unwrap();
            seen.sort_unstable();
            assert_eq!(seen, [0, 1, 2, 3, 4, 5], "direct: {}", wpk.is_direct());
            assert!(buf.bytes()[..bytes.len()] == bytes[..], "direct: {}", wpk.is_direct());
        }
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn failures_stop_reading_and_drain() {
        let (path, _) = sample("failures", 6 * CHUNK);
        let wpk = Wpk::open_with(&path, false).unwrap();
        let mut landed = 0;
        let refused = load(&wpk, |_, _| {
            landed += 1;
            bad("stop")
        });
        assert_eq!(refused.err(), Some(Error::Format("stop".into())));
        assert_eq!(landed, 1, "no chunk lands after a failure");
        std::fs::OpenOptions::new().write(true).open(&path).unwrap().set_len(ALIGN + 2 * CHUNK as u64).unwrap();
        assert_eq!(load(&wpk, |_, _| Ok(())).err(), Some(Error::Format("the file ends early".into())));
        std::fs::remove_file(path).unwrap();
    }
}