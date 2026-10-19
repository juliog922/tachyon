//! The command line's side of the protocol, as `interface/tachyon.h` describes it for any program: connect, take
//! the shared memory, write a request and ring, read the answer from the ring, waking on its futex. It serves the
//! `tachyon` commands that work through a running daemon; it is not a library for other programs, which need only
//! the interface's two files.

use super::sbe::{Bad, Blob, Fields, Reader, Writer};
use super::shm::{self, DOORBELL_REQUEST, Iovec, MAGIC, Msghdr, POLLHUP, POLLIN, Pollfd};
use std::io::{self, Write as _};
use std::os::fd::AsRawFd;
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// A connection to the daemon: its socket, the shared memory mapped, and the layout its control block gives.
pub struct Daemon {
    socket: UnixStream,
    base: *mut u8,
    size: usize,
    /// Request area, ring and data area: offsets and bytes.
    layout: [usize; 6],
    next: u64,
}

/// An answer's last message: its template and bytes.
pub struct Last {
    /// Its template id.
    pub template: u16,
    /// The message, header included.
    pub bytes: Vec<u8>,
}

impl Daemon {
    /// Connects to the daemon on `path`: takes the Welcome and its descriptor, maps the shared memory, checks its
    /// header.
    pub fn connect(path: &Path) -> io::Result<Daemon> {
        let socket = UnixStream::connect(path)?;
        let (mut welcome, mut control) = ([0u8; 64], [0u64; 4]);
        let mut iov = Iovec { base: welcome.as_mut_ptr().cast(), len: welcome.len() };
        let ctl = control.as_mut_ptr().cast();
        let mut msg = Msghdr { name: std::ptr::null_mut(), namelen: 0, iov: &raw mut iov, iovlen: 1, control: ctl, controllen: 32, flags: 0 };
        // SAFETY: every pointer in `msg` is live for the call, with the lengths it gives.
        let n = unsafe { shm::recvmsg(socket.as_raw_fd(), &raw mut msg, 0) };
        let (template, block, _) = Reader::open(&welcome[..n.max(0) as usize]).map_err(|Bad| io::Error::other("not a Welcome"))?;
        let size = block.u64(0).map_err(|Bad| io::Error::other("a Welcome without its size"))? as usize;
        if template != 200 || control[1] != 1 | 1 << 32 {
            return Err(io::Error::other("the daemon sent no shared memory"));
        }
        let fd = control[2] as i32;
        // SAFETY: the descriptor the daemon sent, `size` bytes long; it is closed once mapped, the mapping stays.
        let base = unsafe { shm::mmap(std::ptr::null_mut(), size, 3, 1, fd, 0) }.cast::<u8>();
        // SAFETY: the descriptor received above, no longer needed.
        unsafe { shm::close(fd) };
        if base as isize == -1 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: the control block's first fields, written before the Welcome.
        let head = unsafe { std::slice::from_raw_parts(base, 64) };
        let word = |at: usize| u64::from_le_bytes(head[at..at + 8].try_into().unwrap_or_default()) as usize;
        let layout = [word(16), word(24), word(32), word(40), word(48), word(56)];
        let daemon = Daemon { socket, base, size, layout, next: 1 };
        if word(0) as u32 != MAGIC || word(8) != size {
            return Err(io::Error::other("not tachyon's shared memory"));
        }
        Ok(daemon)
    }

    /// The next request number.
    pub fn id(&mut self) -> u64 {
        self.next += 1;
        self.next
    }

    /// The control block's word at `at`.
    fn atomic<T>(&self, at: usize) -> &T {
        // SAFETY: an aligned word of the control block, mapped as long as `self`.
        unsafe { &*self.base.add(at).cast::<T>() }
    }

    /// Bytes of the data area.
    pub fn data_bytes(&self) -> usize {
        self.layout[5]
    }

    /// Writes `bytes` at `offset` in the data area; the blob naming them.
    pub fn put(&mut self, offset: usize, bytes: &[u8]) -> io::Result<Blob> {
        if offset + bytes.len() > self.layout[5] {
            return Err(io::Error::other(format!("{} bytes do not fit the data area (--data-mib)", offset + bytes.len())));
        }
        // SAFETY: inside the data area, checked above; the daemon reads it only while a request names it.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.base.add(self.layout[4] + offset), bytes.len()) };
        Ok(Blob { offset: offset as u64, length: bytes.len() as u64 })
    }

    /// `len` bytes at `offset` in the data area.
    pub fn get(&self, offset: usize, len: usize) -> Vec<u8> {
        let len = len.min(self.layout[5].saturating_sub(offset));
        // SAFETY: inside the data area, clamped above.
        unsafe { std::slice::from_raw_parts(self.base.add(self.layout[4] + offset), len) }.to_vec()
    }

    /// Sends `request` and reads its answer: each Text's text goes to `text`; returns the last message, or the
    /// Error's message as an error.
    pub fn call(&mut self, request: &Writer, text: &mut dyn FnMut(&str)) -> io::Result<Last> {
        if request.0.len() > self.layout[1] {
            return Err(io::Error::other("the request does not fit the request area"));
        }
        // SAFETY: the request area, which the daemon reads only after the doorbell.
        unsafe { std::ptr::copy_nonoverlapping(request.0.as_ptr(), self.base.add(self.layout[0]), request.0.len()) };
        self.socket.write_all(&[DOORBELL_REQUEST])?;
        loop {
            let msg = self.next()?;
            let (template, _, mut r) = Reader::open(&msg).map_err(|Bad| io::Error::other("a message of another schema"))?;
            match template {
                101 => text(&r.text().map_err(|Bad| io::Error::other("a Text without text"))?),
                103 => {
                    let code = msg.get(16..18).map_or(0, |c| u16::from_le_bytes([c[0], c[1]]));
                    return Err(io::Error::other(format!("{} (code {code})", r.text().unwrap_or_default())));
                }
                _ => return Ok(Last { template, bytes: msg }),
            }
        }
    }

    /// The next message in the ring, waiting on the futex for it; an error once the daemon has gone.
    fn next(&mut self) -> io::Result<Vec<u8>> {
        let (ring, bytes) = (self.layout[2], self.layout[3] as u64);
        let (head, tail, seq) = (self.atomic::<AtomicU64>(64), self.atomic::<AtomicU64>(192), self.atomic::<AtomicU32>(128));
        loop {
            let at = tail.load(Ordering::Relaxed);
            if at < head.load(Ordering::Acquire) {
                let frame = ring + (at % bytes) as usize;
                let len = self.atomic::<AtomicU32>(frame).load(Ordering::Relaxed);
                if len == u32::MAX {
                    tail.store(at + bytes - at % bytes, Ordering::Release);
                    continue;
                }
                // SAFETY: the frame's bytes, which never wrap, below the head.
                let msg = unsafe { std::slice::from_raw_parts(self.base.add(frame + 4), len as usize) }.to_vec();
                tail.store(at + (4 + u64::from(len)).next_multiple_of(8), Ordering::Release);
                return Ok(msg);
            }
            let s = seq.load(Ordering::Acquire);
            if tail.load(Ordering::Relaxed) == head.load(Ordering::Acquire) {
                let wait = [1i64, 0];
                // SAFETY: FUTEX_WAIT (0, shared) on the ring_seq word of the mapping, a second at most.
                unsafe { shm::syscall(202, std::ptr::from_ref(seq), 0, s, wait.as_ptr()) };
                let mut p = [Pollfd { fd: self.socket.as_raw_fd(), events: POLLIN, revents: 0 }];
                if shm::wait(&mut p, 0)? > 0 && p[0].revents & POLLHUP != 0 {
                    return Err(io::Error::other("the daemon closed the connection"));
                }
            }
        }
    }
}

impl Drop for Daemon {
    fn drop(&mut self) {
        // SAFETY: the mapping made in `connect`, unmapped once.
        unsafe { shm::munmap(self.base.cast(), self.size) };
    }
}

/// A request of `template` numbered `id`, its block's other fields `rest`.
pub fn request(template: u16, id: u64, rest: Fields) -> Writer {
    Writer::new(template, &Fields::default().u64(id).0.into_iter().chain(rest.0).collect::<Vec<_>>())
}
