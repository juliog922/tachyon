//! A client's shared memory (`interface/tachyon.h`): created as a `memfd`, mapped here, handed over the socket with
//! `SCM_RIGHTS`; its control block, request area, response ring and data area; and the few system calls around it.

use std::ffi::{c_int, c_void};
use std::io::{Error, Result};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// `"TCHY"`, the layout's version, and the doorbells' bytes and the ring's wrap mark, as `tachyon.h` defines them.
pub const MAGIC: u32 = 0x5948_4354;
pub const VERSION: u32 = 1;
pub const DOORBELL_REQUEST: u8 = 1;
pub const DOORBELL_CANCEL: u8 = 2;
const WRAP: u32 = u32::MAX;

/// Where each part starts: the control block's page, then the request area and the ring, then the data area.
const REQUEST: usize = 4096;
const REQUEST_BYTES: usize = 1 << 20;
const RING_BYTES: usize = 1 << 20;
/// The control block's fields, as offsets: the layout's first block, then `ring_head`, `ring_seq`, `ring_tail`.
const HEAD: usize = 64;
const SEQ: usize = 128;
const TAIL: usize = 192;

#[repr(C)]
pub(super) struct Iovec {
    pub(super) base: *mut c_void,
    pub(super) len: usize,
}

#[repr(C)]
pub(super) struct Msghdr {
    pub(super) name: *mut c_void,
    pub(super) namelen: u32,
    pub(super) iov: *mut Iovec,
    pub(super) iovlen: usize,
    pub(super) control: *mut c_void,
    pub(super) controllen: usize,
    pub(super) flags: c_int,
}

#[repr(C)]
#[derive(Default)]
struct Ucred {
    pid: i32,
    uid: u32,
    gid: u32,
}

#[repr(C)]
pub struct Pollfd {
    pub fd: c_int,
    pub events: i16,
    pub revents: i16,
}

unsafe extern "C" {
    fn memfd_create(name: *const u8, flags: u32) -> c_int;
    fn ftruncate(fd: c_int, len: i64) -> c_int;
    pub(super) fn mmap(addr: *mut c_void, len: usize, prot: c_int, flags: c_int, fd: c_int, off: i64) -> *mut c_void;
    pub(super) fn munmap(addr: *mut c_void, len: usize) -> c_int;
    pub(super) fn recvmsg(fd: c_int, msg: *mut Msghdr, flags: c_int) -> isize;
    pub(super) fn close(fd: c_int) -> c_int;
    fn sendmsg(fd: c_int, msg: *const Msghdr, flags: c_int) -> isize;
    fn getsockopt(fd: c_int, level: c_int, name: c_int, value: *mut c_void, len: *mut u32) -> c_int;
    fn getuid() -> u32;
    fn poll(fds: *mut Pollfd, n: u64, timeout: c_int) -> c_int;
    pub(super) fn syscall(n: i64, ...) -> i64;
}

/// Readable, and the other end gone, for [`Pollfd::events`] and `revents`.
pub const POLLIN: i16 = 1;
pub const POLLHUP: i16 = 0x10;

/// Waits up to `ms` (−1: no end) for one of `fds` to be readable or closed.
pub fn wait(fds: &mut [Pollfd], ms: i32) -> Result<usize> {
    // SAFETY: `fds` is a live array of `fds.len()` entries.
    let n = unsafe { poll(fds.as_mut_ptr(), fds.len() as u64, ms) };
    if n < 0 { Err(Error::last_os_error()) } else { Ok(n as usize) }
}

/// Whether the process at the other end of `s` runs as this process's user.
pub fn same_user(s: &UnixStream) -> bool {
    const SOL_SOCKET: c_int = 1;
    const SO_PEERCRED: c_int = 17;
    let (mut cred, mut len) = (Ucred::default(), size_of::<Ucred>() as u32);
    // SAFETY: `cred` has the size `len` says, and `s` is an open socket.
    let ok = unsafe { getsockopt(s.as_raw_fd(), SOL_SOCKET, SO_PEERCRED, (&raw mut cred).cast(), &raw mut len) } == 0;
    // SAFETY: getuid cannot fail.
    ok && cred.uid == unsafe { getuid() }
}

/// A client's shared memory, mapped.
pub struct Shared {
    base: *mut u8,
    size: usize,
    fd: OwnedFd,
}

impl Shared {
    /// A new shared memory with a data area of `data` bytes, its control block written.
    pub fn new(data: usize) -> Result<Shared> {
        const PROT: c_int = 3; // PROT_READ | PROT_WRITE
        const MAP_SHARED: c_int = 1;
        let size = REQUEST + REQUEST_BYTES + RING_BYTES + data;
        // SAFETY: a NUL-terminated name; the flags are MFD_CLOEXEC.
        let raw = unsafe { memfd_create(c"tachyon".as_ptr().cast(), 1) };
        if raw < 0 {
            return Err(Error::last_os_error());
        }
        // SAFETY: memfd_create returned a new descriptor that nothing else owns.
        let fd = unsafe { OwnedFd::from_raw_fd(raw) };
        // SAFETY: `fd` is open; the length is positive.
        if unsafe { ftruncate(fd.as_raw_fd(), size as i64) } != 0 {
            return Err(Error::last_os_error());
        }
        // SAFETY: mapping `size` bytes of a file of that size.
        let base = unsafe { mmap(std::ptr::null_mut(), size, PROT, MAP_SHARED, fd.as_raw_fd(), 0) };
        if base as isize == -1 {
            return Err(Error::last_os_error());
        }
        let shm = Shared { base: base.cast(), size, fd };
        let layout = [REQUEST, REQUEST_BYTES, REQUEST + REQUEST_BYTES, RING_BYTES, REQUEST + REQUEST_BYTES + RING_BYTES, data].map(|v| v as u64);
        shm.put(0, MAGIC);
        shm.put(4, VERSION);
        shm.put(8, size as u64);
        layout.iter().enumerate().for_each(|(i, &v)| shm.put(16 + 8 * i, v));
        Ok(shm)
    }

    /// Sends `welcome` on `s` with this memory's descriptor attached.
    pub fn send(&self, s: &UnixStream, welcome: &[u8]) -> Result<()> {
        const SOL_SOCKET: c_int = 1;
        const SCM_RIGHTS: c_int = 1;
        // A cmsghdr (length, level, type) and one descriptor, padded to 8.
        let mut control = [0u64; 3];
        control[0] = 20;
        control[1] = u64::from(SOL_SOCKET as u32) | u64::from(SCM_RIGHTS as u32) << 32;
        control[2] = u64::from(self.fd.as_raw_fd() as u32);
        let mut iov = Iovec { base: welcome.as_ptr().cast_mut().cast(), len: welcome.len() };
        let msg =
            Msghdr { name: std::ptr::null_mut(), namelen: 0, iov: &raw mut iov, iovlen: 1, control: control.as_mut_ptr().cast(), controllen: 24, flags: 0 };
        // SAFETY: every pointer in `msg` is live for the call, with the lengths it gives.
        let n = unsafe { sendmsg(s.as_raw_fd(), &raw const msg, 0) };
        if n == welcome.len() as isize { Ok(()) } else { Err(Error::last_os_error()) }
    }

    /// The whole memory's size.
    pub fn size(&self) -> usize {
        self.size
    }

    /// The request area. The client writes it only between requests (`tachyon.h`), and every read of it is
    /// bounds-checked, so a client breaking that rule spoils its own request only.
    pub fn request(&self) -> &[u8] {
        // SAFETY: the area is inside the mapping, which lives as long as `self`.
        unsafe { std::slice::from_raw_parts(self.base.add(REQUEST), REQUEST_BYTES) }
    }

    /// `len` bytes at `offset` in the data area, copied out; `None` past its end.
    pub fn read(&self, offset: u64, len: u64) -> Option<Vec<u8>> {
        let at = self.data(offset, len)?;
        let mut out = vec![0; len as usize];
        // SAFETY: `data` checked the range against the mapping.
        unsafe { std::ptr::copy_nonoverlapping(at, out.as_mut_ptr(), out.len()) };
        Some(out)
    }

    /// Writes `bytes` at `offset` in the data area, if they fit in it and in `room` bytes.
    pub fn write(&self, offset: u64, room: u64, bytes: &[u8]) -> bool {
        match self.data(offset, room) {
            // SAFETY: `data` checked the range against the mapping, and `bytes` is no longer than it.
            Some(at) if bytes.len() as u64 <= room => unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), at, bytes.len()) },
            _ => return false,
        }
        true
    }

    /// The address of `len` bytes at `offset` in the data area, if they are in it.
    fn data(&self, offset: u64, len: u64) -> Option<*mut u8> {
        let start = (REQUEST + REQUEST_BYTES + RING_BYTES) as u64;
        let end = offset.checked_add(len)?;
        // SAFETY: an offset inside the mapping.
        (end <= self.size as u64 - start).then(|| unsafe { self.base.add((start + offset) as usize) })
    }

    /// Writes `v` at byte `at` of the control block or the ring, where only this process writes.
    fn put<T: Copy>(&self, at: usize, v: T) {
        debug_assert!(at % align_of::<T>() == 0 && at + size_of::<T>() <= self.size);
        // SAFETY: inside the mapping, which is page-aligned, at an offset aligned for `T` (asserted above).
        unsafe { self.base.add(at).cast::<T>().write(v) };
    }

    /// The word at byte `at`: of the control block, or (in tests) of the ring.
    fn word<T>(&self, at: usize) -> &T {
        // SAFETY: the word is aligned and inside the first page, which lives as long as `self`.
        unsafe { &*self.base.add(at).cast::<T>() }
    }

    /// Appends `msg` to the ring as a frame and wakes the client; while the ring has no room, waits, giving up
    /// with `false` when `gone` says the client left.
    pub fn append(&self, msg: &[u8], gone: &mut dyn FnMut() -> bool) -> bool {
        let need = (4 + msg.len()).next_multiple_of(8) as u64;
        let (head, tail, ring) = (self.word::<AtomicU64>(HEAD), self.word::<AtomicU64>(TAIL), RING_BYTES as u64);
        let mut at = head.load(Ordering::Relaxed);
        let skip = if at % ring + need > ring { ring - at % ring } else { 0 };
        while at + skip + need - tail.load(Ordering::Acquire) > ring {
            if gone() {
                return false;
            }
            std::thread::sleep(std::time::Duration::from_micros(50));
        }
        let ring_at = |pos: u64| (REQUEST + REQUEST_BYTES) as u64 + pos % ring;
        if skip > 0 {
            self.put(ring_at(at) as usize, WRAP);
            at += skip;
        }
        let frame = ring_at(at) as usize;
        self.put(frame, msg.len() as u32);
        // SAFETY: frames start at multiples of 8 inside the ring, and `need` bytes fit before its end after a skip.
        unsafe { std::ptr::copy_nonoverlapping(msg.as_ptr(), self.base.add(frame + 4), msg.len()) };
        head.store(at + need, Ordering::Release);
        self.word::<AtomicU32>(SEQ).fetch_add(1, Ordering::Release);
        // SAFETY: FUTEX_WAKE (1, shared) on a word of the mapping, waking every waiter.
        unsafe { syscall(202, self.base.add(SEQ), 1, i32::MAX) };
        true
    }
}

impl Drop for Shared {
    fn drop(&mut self) {
        // SAFETY: the mapping made in `new`, unmapped once.
        unsafe { munmap(self.base.cast(), self.size) };
    }
}

#[cfg(test)]
mod tests {
    use super::{HEAD, MAGIC, REQUEST, REQUEST_BYTES, RING_BYTES, Shared, TAIL, WRAP};
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Reads the frames between the tail and the head as a client does, raising the tail past them.
    fn frames(shm: &Shared) -> Vec<Vec<u8>> {
        let (head, tail) = (shm.word::<AtomicU64>(HEAD).load(Ordering::Acquire), shm.word::<AtomicU64>(TAIL));
        let mut out = Vec::new();
        let mut at = tail.load(Ordering::Relaxed);
        while at < head {
            let frame = REQUEST + REQUEST_BYTES + (at % RING_BYTES as u64) as usize;
            let len = *shm.word::<u32>(frame);
            if len == WRAP {
                at += RING_BYTES as u64 - at % RING_BYTES as u64;
                continue;
            }
            // SAFETY: the frame's bytes, which never wrap.
            out.push(unsafe { std::slice::from_raw_parts(shm.base.add(frame + 4), len as usize) }.to_vec());
            at += (4 + u64::from(len)).next_multiple_of(8);
        }
        tail.store(at, Ordering::Release);
        out
    }

    #[test]
    fn frames_wrap_and_wait_for_room() {
        let shm = Shared::new(4096).unwrap();
        assert_eq!(shm.word::<u32>(0), &MAGIC);
        // Frames of 300 KB: three fit, the fourth wraps and waits until the client reads.
        let msgs: Vec<Vec<u8>> = (0..4u8).map(|i| vec![i; 300_000 + usize::from(i)]).collect();
        for m in &msgs[..3] {
            assert!(shm.append(m, &mut || panic!("room for three")));
        }
        assert!(!shm.append(&msgs[3], &mut || true), "a full ring gives up when the client is gone");
        assert_eq!(frames(&shm), msgs[..3]);
        assert!(shm.append(&msgs[3], &mut || panic!("room once read")));
        assert_eq!(frames(&shm), msgs[3..], "after the wrap");
    }

    #[test]
    fn the_data_area_is_bounds_checked() {
        let shm = Shared::new(4096).unwrap();
        assert!(shm.write(4000, 96, &[7; 96]) && shm.read(4000, 96) == Some(vec![7; 96]));
        assert!(!shm.write(4000, 97, &[7; 97]) && shm.read(4001, 96).is_none() && shm.read(u64::MAX, 2).is_none());
        assert!(!shm.write(0, 8, &[0; 9]), "more than the blob's room");
    }
}
