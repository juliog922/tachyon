//! Linux `x86_64` kernel interface: raw system calls and an `io_uring` instance.
//!
//! Ported from caudal's `sys`, keeping only what file loading needs. The ring
//! asks for `SINGLE_ISSUER | DEFER_TASKRUN` (Linux 6.1+) and falls back to a
//! plain ring on older kernels, such as Ubuntu 22.04's 5.15.

use core::arch::asm;
use core::sync::atomic::{AtomicU32, Ordering};

const CLOSE: usize = 3;
const MMAP: usize = 9;
const MUNMAP: usize = 11;
const MINCORE: usize = 27;
const IO_URING_SETUP: usize = 425;
const IO_URING_ENTER: usize = 426;
const EINVAL: isize = -22;

/// `IORING_OP_READ` (Linux 5.6+): `pread` at `off` into `addr`.
pub const OP_READ: u8 = 22;

/// Raw system call; returns the kernel result (`-errno` on failure).
///
/// # Safety
/// Pointer arguments must be valid for the call's contract.
#[inline]
unsafe fn call(nr: usize, [a0, a1, a2, a3, a4, a5]: [usize; 6]) -> isize {
    let r: isize;
    // SAFETY: the caller upholds the call's contract; `syscall` clobbers only rcx and r11.
    unsafe {
        asm!("syscall", inlateout("rax") nr as isize => r, in("rdi") a0, in("rsi") a1, in("rdx") a2,
             in("r10") a3, in("r8") a4, in("r9") a5, lateout("rcx") _, lateout("r11") _, options(nostack));
    }
    r
}

/// `io_sqring_offsets` and `io_cqring_offsets`. They share head, tail and mask; of the rest, Tachyon reads the
/// submission array (`rest[SQ_ARRAY]`) and the completion entries (`rest[CQ_ENTRIES]`).
#[repr(C)]
#[derive(Default)]
struct Offsets {
    head: u32,
    tail: u32,
    ring_mask: u32,
    rest: [u32; 7],
}

const SQ_ARRAY: usize = 3;
const CQ_ENTRIES: usize = 2;

/// `io_uring_params`, with the fields Tachyon does not set or read folded into `rest`.
#[repr(C)]
#[derive(Default)]
struct Params {
    sq_entries: u32,
    cq_entries: u32,
    flags: u32,
    rest: [u32; 7],
    sq_off: Offsets,
    cq_off: Offsets,
}

/// The share of `len` bytes of file `fd`, from page-aligned `off`, held in the page cache; 0 when unknown.
pub fn resident(fd: i32, off: u64, len: usize) -> f64 {
    // SAFETY: maps the range read-only and shared; nothing reads through the mapping.
    let addr = unsafe { call(MMAP, [0, len, 1, 1, fd as usize, off as usize]) };
    if len == 0 || (-4095..0).contains(&addr) {
        return 0.0;
    }
    let mut pages = vec![0u8; len.div_ceil(4096)];
    // SAFETY: `pages` holds one byte per page of the mapping.
    let probed = unsafe { call(MINCORE, [addr as usize, len, pages.as_mut_ptr() as usize, 0, 0, 0]) } == 0;
    // SAFETY: unmaps the mapping made above.
    unsafe { call(MUNMAP, [addr as usize, len, 0, 0, 0, 0]) };
    let held = pages.iter().filter(|&&p| p & 1 == 1).count();
    if probed { held as f64 / pages.len() as f64 } else { 0.0 }
}

/// Submission queue entry (`struct io_uring_sqe`); `rest` holds the fields file reads leave at zero.
#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct Sqe {
    pub opcode: u8,
    pub flags: u8,
    pub ioprio: u16,
    pub fd: i32,
    pub off: u64,
    pub addr: u64,
    pub len: u32,
    pub op_flags: u32,
    pub user_data: u64,
    pub rest: [u64; 3],
}

/// Completion queue entry (`struct io_uring_cqe`).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Cqe {
    pub user_data: u64,
    pub res: i32,
    pub flags: u32,
}

const _: () = assert!(size_of::<Sqe>() == 64 && size_of::<Params>() == 120 && size_of::<Cqe>() == 16);

/// One `io_uring` instance, driven by the thread that created it.
pub struct Ring {
    fd: i32,
    maps: [(usize, usize); 2],
    sq_head: *const AtomicU32,
    sq_tail: *const AtomicU32,
    sq_mask: u32,
    sq_entries: u32,
    sqes: *mut Sqe,
    cq_head: *const AtomicU32,
    cq_tail: *const AtomicU32,
    cq_mask: u32,
    cqes: *const Cqe,
    tail: u32,
    submitted: u32,
    head: u32,
    seen_tail: u32,
}

fn setup(entries: u32, p: &mut Params) -> isize {
    // SAFETY: `p` is a live `io_uring_params`.
    unsafe { call(IO_URING_SETUP, [entries as usize, std::ptr::from_mut(p) as usize, 0, 0, 0, 0]) }
}

impl Ring {
    /// A ring of `entries` submission slots; `Err(-errno)` when the kernel refuses one.
    // The mappings are page-aligned and the kernel's offsets keep every field aligned.
    #[allow(clippy::cast_ptr_alignment)]
    pub fn new(entries: u32) -> Result<Ring, isize> {
        let mut p = Params { flags: 1 << 12 | 1 << 13, ..Default::default() };
        let mut fd = setup(entries, &mut p);
        if fd == EINVAL {
            p = Params::default();
            fd = setup(entries, &mut p);
        }
        if fd < 0 {
            return Err(fd);
        }
        let map = |len: usize, off: usize| -> Result<*mut u8, isize> {
            // SAFETY: maps `len` bytes of the ring fd: PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE.
            let r = unsafe { call(MMAP, [0, len, 3, 0x8001, fd as usize, off]) };
            if (-4095..0).contains(&r) { Err(r) } else { Ok(r as *mut u8) }
        };
        let ring_len = (p.sq_off.rest[SQ_ARRAY] as usize + p.sq_entries as usize * 4).max(p.cq_off.rest[CQ_ENTRIES] as usize + p.cq_entries as usize * 16);
        let sqes_len = p.sq_entries as usize * size_of::<Sqe>();
        let ring = map(ring_len, 0)?;
        let sqes = map(sqes_len, 0x1000_0000)?.cast::<Sqe>();
        // SAFETY: the offsets come from the kernel and lie inside the mapped ring.
        unsafe {
            let array = ring.add(p.sq_off.rest[SQ_ARRAY] as usize).cast::<u32>();
            (0..p.sq_entries).for_each(|i| *array.add(i as usize) = i);
            let at = |off: u32| ring.add(off as usize).cast::<AtomicU32>().cast_const();
            Ok(Ring {
                fd: fd as i32,
                maps: [(ring as usize, ring_len), (sqes as usize, sqes_len)],
                sq_head: at(p.sq_off.head),
                sq_tail: at(p.sq_off.tail),
                sq_mask: *ring.add(p.sq_off.ring_mask as usize).cast::<u32>(),
                sq_entries: p.sq_entries,
                sqes,
                cq_head: at(p.cq_off.head),
                cq_tail: at(p.cq_off.tail),
                cq_mask: *ring.add(p.cq_off.ring_mask as usize).cast::<u32>(),
                cqes: ring.add(p.cq_off.rest[CQ_ENTRIES] as usize).cast::<Cqe>(),
                tail: 0,
                submitted: 0,
                head: 0,
                seen_tail: 0,
            })
        }
    }

    /// Queues one operation; flushes the submission queue first when it is full.
    pub fn push(&mut self, sqe: Sqe) {
        // SAFETY: `sq_head` points into the live ring mapping.
        while self.tail.wrapping_sub(unsafe { (*self.sq_head).load(Ordering::Acquire) }) >= self.sq_entries {
            self.enter(0);
        }
        // SAFETY: the slot index is masked into the submission array.
        unsafe { *self.sqes.add((self.tail & self.sq_mask) as usize) = sqe };
        self.tail = self.tail.wrapping_add(1);
    }

    /// Publishes queued operations and consumed completions, then waits for `wait` completions.
    pub fn enter(&mut self, wait: u32) {
        // SAFETY: both pointers point into the live ring mapping.
        unsafe {
            (*self.sq_tail).store(self.tail, Ordering::Release);
            (*self.cq_head).store(self.head, Ordering::Release);
        }
        let pending = self.tail.wrapping_sub(self.submitted);
        // SAFETY: `io_uring_enter(fd, to_submit, min_complete, IORING_ENTER_GETEVENTS when waiting)`.
        let r = unsafe { call(IO_URING_ENTER, [self.fd as usize, pending as usize, wait as usize, usize::from(wait > 0), 0, 0]) };
        if r > 0 {
            self.submitted = self.submitted.wrapping_add(r as u32);
        }
    }

    /// Next completion, if any.
    pub fn completion(&mut self) -> Option<Cqe> {
        if self.head == self.seen_tail {
            // SAFETY: `cq_tail` points into the live ring mapping.
            self.seen_tail = unsafe { (*self.cq_tail).load(Ordering::Acquire) };
            if self.head == self.seen_tail {
                return None;
            }
        }
        // SAFETY: the slot index is masked into the completion array.
        let cqe = unsafe { *self.cqes.add((self.head & self.cq_mask) as usize) };
        self.head = self.head.wrapping_add(1);
        Some(cqe)
    }
}

impl Drop for Ring {
    fn drop(&mut self) {
        // SAFETY: unmaps the two regions `new` mapped and closes the ring's descriptor.
        unsafe {
            for (addr, len) in self.maps {
                call(MUNMAP, [addr, len, 0, 0, 0, 0]);
            }
            call(CLOSE, [self.fd as usize, 0, 0, 0, 0, 0]);
        }
    }
}

#[cfg(test)]
pub mod testing {
    /// Drops file `fd`'s clean pages from the page cache (`posix_fadvise(DONTNEED)`).
    pub fn evict(fd: i32) {
        const FADVISE64: usize = 221;
        // SAFETY: an advisory call on a descriptor the caller owns; no pointers.
        unsafe { super::call(FADVISE64, [fd as usize, 0, 0, 4, 0, 0]) };
    }
}
