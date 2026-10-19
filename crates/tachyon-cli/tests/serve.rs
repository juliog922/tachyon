//! `tachyon serve` end to end, as a client in another language would speak to it: only `interface/tachyon.h` and
//! `tachyon.xml`, written out by hand here (the socket, the descriptor, the mapping, the futex, the SBE bytes). The
//! tiny Gemma 4 model loads, lists, answers twice the same, refuses what it cannot read, and cancels. Runs where an
//! NVIDIA device is visible.

#[allow(dead_code, clippy::many_single_char_names, clippy::float_cmp)]
#[path = "../../tachyon/tests/common/tiny.rs"]
mod tiny;

use std::ffi::{c_int, c_void};
use std::io::Write as _;
use std::os::fd::AsRawFd;
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

unsafe extern "C" {
    fn recvmsg(fd: c_int, msg: *mut Msghdr, flags: c_int) -> isize;
    fn mmap(addr: *mut c_void, len: usize, prot: c_int, flags: c_int, fd: c_int, off: i64) -> *mut c_void;
    fn syscall(n: i64, ...) -> i64;
}

#[repr(C)]
struct Iovec {
    base: *mut c_void,
    len: usize,
}

#[repr(C)]
struct Msghdr {
    name: *mut c_void,
    namelen: u32,
    iov: *mut Iovec,
    iovlen: usize,
    control: *mut c_void,
    controllen: usize,
    flags: c_int,
}

/// A client, after the handshake: its socket, the mapping, and the layout of `struct tachyon_control`.
struct Client {
    socket: UnixStream,
    base: *mut u8,
    request: usize,
    ring: usize,
    ring_bytes: u64,
}

impl Client {
    /// Connects to `path`, takes the Welcome and its descriptor, maps the shared memory and checks its header.
    fn connect(path: &Path) -> Client {
        let socket = UnixStream::connect(path).unwrap();
        let (mut welcome, mut control) = ([0u8; 64], [0u64; 4]);
        let mut iov = Iovec { base: welcome.as_mut_ptr().cast(), len: welcome.len() };
        let mut msg =
            Msghdr { name: std::ptr::null_mut(), namelen: 0, iov: &raw mut iov, iovlen: 1, control: control.as_mut_ptr().cast(), controllen: 32, flags: 0 };
        // SAFETY: every pointer in `msg` is live for the call, with the lengths it gives.
        let n = unsafe { recvmsg(socket.as_raw_fd(), &raw mut msg, 0) };
        assert_eq!(n, 16, "a Welcome: header and size");
        assert_eq!(&welcome[..8], &[8, 0, 200, 0, 7, 0, 1, 0], "Welcome's header");
        let size = u64::from_le_bytes(welcome[8..16].try_into().unwrap()) as usize;
        assert_eq!(control[1], 1 | 1 << 32, "SOL_SOCKET, SCM_RIGHTS");
        // SAFETY: the descriptor the daemon sent, `size` bytes long.
        let base = unsafe { mmap(std::ptr::null_mut(), size, 3, 1, control[2] as c_int, 0) }.cast::<u8>();
        assert_ne!(base as isize, -1);
        // SAFETY: the control block's first fields, written before the Welcome.
        let head = unsafe { std::slice::from_raw_parts(base, 48) };
        let word = |at: usize| u64::from_le_bytes(head[at..at + 8].try_into().unwrap());
        assert_eq!((word(0) as u32, (word(0) >> 32) as u32, word(8) as usize), (0x5948_4354, 1, size), "magic, version, size");
        Client { socket, base, request: word(16) as usize, ring: word(32) as usize, ring_bytes: word(40) }
    }

    fn atomic<T>(&self, at: usize) -> &T {
        // SAFETY: an aligned word of the control block, mapped as long as the process runs.
        unsafe { &*self.base.add(at).cast::<T>() }
    }

    /// Writes `msg` to the request area and rings `bells` (the request's, maybe a cancel after it).
    fn ask(&mut self, msg: &[u8], bells: &[u8]) {
        // SAFETY: the request area, which the daemon reads only after the doorbell.
        unsafe { std::ptr::copy_nonoverlapping(msg.as_ptr(), self.base.add(self.request), msg.len()) };
        self.socket.write_all(bells).unwrap();
    }

    /// The next message in the ring, waiting on the futex for it.
    fn next(&mut self) -> Vec<u8> {
        let (head, tail, seq) = (self.atomic::<AtomicU64>(64), self.atomic::<AtomicU64>(192), self.atomic::<AtomicU32>(128));
        let start = Instant::now();
        loop {
            let at = tail.load(Ordering::Relaxed);
            if at < head.load(Ordering::Acquire) {
                let frame = self.ring + (at % self.ring_bytes) as usize;
                let len = self.atomic::<AtomicU32>(frame).load(Ordering::Relaxed);
                if len == u32::MAX {
                    tail.store(at + self.ring_bytes - at % self.ring_bytes, Ordering::Release);
                    continue;
                }
                // SAFETY: the frame's bytes, which never wrap.
                let msg = unsafe { std::slice::from_raw_parts(self.base.add(frame + 4), len as usize) }.to_vec();
                tail.store(at + (4 + u64::from(len)).next_multiple_of(8), Ordering::Release);
                return msg;
            }
            let s = seq.load(Ordering::Acquire);
            if tail.load(Ordering::Relaxed) == head.load(Ordering::Acquire) {
                assert!(start.elapsed() < Duration::from_secs(120), "no answer");
                let wait = [1i64, 0]; // a second, as a timespec
                // SAFETY: FUTEX_WAIT (0, shared) on the ring_seq word, with a timeout.
                unsafe { syscall(202, (&raw const *seq), 0, s, wait.as_ptr()) };
            }
        }
    }

    /// Asks `msg` and gathers the answer: every message to the last (anything but Text).
    fn answer(&mut self, msg: &[u8]) -> Vec<Vec<u8>> {
        self.ask(msg, &[1]);
        self.rest()
    }

    /// The messages up to the end of the answer being given.
    fn rest(&mut self) -> Vec<Vec<u8>> {
        let mut out = vec![self.next()];
        while template(out.last().unwrap()) == 101 {
            out.push(self.next());
        }
        out
    }
}

/// A message's template id.
fn template(msg: &[u8]) -> u16 {
    u16::from_le_bytes([msg[2], msg[3]])
}

/// A message: its header (schema 7, version 1), `block`, then `rest`.
fn message(template: u16, block: &[u8], rest: &[u8]) -> Vec<u8> {
    let head = [block.len() as u16, template, 7, 1].map(u16::to_le_bytes).concat();
    [&head, block, rest].concat()
}

/// Variable-length data.
fn text(t: &str) -> Vec<u8> {
    [&(t.len() as u32).to_le_bytes()[..], t.as_bytes()].concat()
}

/// A group header.
fn group(len: u16, count: u16) -> Vec<u8> {
    [len.to_le_bytes(), count.to_le_bytes()].concat()
}

/// A greedy Chat numbered `id` with `model`: a system message, then `prompt`; `max` tokens at most (0: no limit).
fn chat(id: u64, model: &str, prompt: &str, max: u32) -> Vec<u8> {
    let mut block = [id.to_le_bytes(), 0u64.to_le_bytes(), (-1i64).to_le_bytes()].concat();
    for v in [max, 0, 0f32.to_bits(), 1f32.to_bits(), 0f32.to_bits(), 1f32.to_bits()] {
        block.extend(v.to_le_bytes());
    }
    block.push(0);
    let said = |role: u8, t: &str| [&[role][..], &group(17, 0), &text(t)].concat();
    message(1, &block, &[group(1, 2), said(0, "Be brief."), said(1, prompt), text(model)].concat())
}

/// The text of an answer's Text messages, and its Done's output tokens.
fn streamed(answer: &[Vec<u8>]) -> (String, u32) {
    let (last, texts) = answer.split_last().unwrap();
    assert_eq!(template(last), 102, "ends with Done: {last:?}");
    let text = texts.iter().map(|m| String::from_utf8(m[20..].to_vec()).unwrap()).collect();
    (text, u32::from_le_bytes(last[8 + 28..8 + 32].try_into().unwrap()))
}

/// An Error's code, for request `id`.
fn error(answer: &[Vec<u8>], id: u64) -> u16 {
    let last = answer.last().unwrap();
    assert_eq!((template(last), u64::from_le_bytes(last[8..16].try_into().unwrap())), (103, id), "an Error for {id}");
    u16::from_le_bytes([last[16], last[17]])
}

/// The tiny model converted, and a daemon serving on a socket beside it: the model's directory and the socket.
fn daemon(name: &str) -> (String, std::path::PathBuf) {
    let src = tiny::checkpoint(name, 3, tiny::E4B);
    tachyon::model::convert(&src, &src.join("out")).unwrap();
    let socket = src.join("tachyon.sock");
    let opts = tachyon_cli::serve::Options { socket: socket.clone(), keep_alive: Duration::from_secs(60), data: 1 << 20, vram: None, host: None };
    std::thread::spawn(move || tachyon_cli::serve::serve(&opts).unwrap());
    while !socket.exists() {
        std::thread::sleep(Duration::from_millis(10));
    }
    (src.join("out").to_str().unwrap().to_string(), socket)
}

/// A cancel rung with the request stops it before it starts; one rung during the answer stops it there (unless the
/// tiny model is done first); either way the next chat answers as `first`.
fn cancels(c: &mut Client, model: &str, first: &(String, u32)) {
    c.ask(&chat(6, model, "hello there, how are you", 0), &[1, 2]);
    assert_eq!(error(&c.rest(), 6), 4, "Cancelled");
    c.ask(&chat(7, model, "hello there, how are you", 0), &[1]);
    c.next();
    c.socket.write_all(&[2]).unwrap();
    let last = c.rest().pop().unwrap();
    assert!(template(&last) == 102 || error(&[last], 7) == 4);
    assert_eq!(&streamed(&c.answer(&chat(8, model, "hello there, how are you", 30))), first, "after a cancel");
}

/// Times List answered 10,000 times: the protocol's own cost, a request to its last answer.
fn round_trips(c: &mut Client) {
    let list = message(6, &11u64.to_le_bytes(), &[]);
    let mut times: Vec<Duration> = (0..10_000)
        .map(|_| {
            let start = Instant::now();
            c.answer(&list);
            start.elapsed()
        })
        .collect();
    times.sort();
    let us = |d: Duration| d.as_secs_f64() * 1e6;
    eprintln!("List round trip: median {:.1} µs, p99 {:.1} µs", us(times[5_000]), us(times[9_900]));
}

#[test]
fn a_client_loads_lists_chats_and_cancels() {
    if !Path::new("/dev/nvidiactl").exists() && !Path::new("/dev/dxg").exists() {
        eprintln!("no NVIDIA device: skipped");
        return;
    }
    let (model, socket) = daemon("serve");
    let mut c = Client::connect(&socket);

    // Load with a short context, then List: one model, in VRAM.
    let load = message(4, &[&1u64.to_le_bytes()[..], &(-1i64).to_le_bytes(), &64u32.to_le_bytes(), &[0]].concat(), &text(&model));
    assert_eq!(template(&c.answer(&load)[0]), 102);
    let models = c.answer(&message(6, &2u64.to_le_bytes(), &[]));
    let m = &models[0];
    assert_eq!((template(m), &m[16..20]), (106, &group(17, 1)[..]), "Models, one");
    assert_eq!((m[36], String::from_utf8(m[41..].to_vec()).unwrap()), (0, model.clone()), "in VRAM, by its name");

    // Two chats answer the same.
    let first = streamed(&c.answer(&chat(3, &model, "hello there, how are you", 30)));
    assert!(first.1 > 0 && first.1 <= 30 && !first.0.is_empty(), "{first:?}");
    assert_eq!(streamed(&c.answer(&chat(4, &model, "hello there, how are you", 30))), first);

    // What cannot be answered: bytes of another schema, a model nowhere.
    assert_eq!(error(&c.answer(&[8, 0, 1, 0, 9, 0, 1, 0]), 0), 1, "BadRequest");
    assert_eq!(error(&c.answer(&chat(5, "no-such-model", "hi", 4)), 5), 2, "NoModel");
    cancels(&mut c, &model, &first);

    // A client that leaves mid-answer leaves the daemon serving the others.
    let mut gone = Client::connect(&socket);
    gone.ask(&chat(9, &model, "hello there, how are you", 0), &[1]);
    drop(gone);
    assert_eq!(streamed(&c.answer(&chat(10, &model, "hello there, how are you", 30))), first, "after a client left");
    round_trips(&mut c);
    std::fs::remove_dir_all(socket.parent().unwrap()).unwrap();
}

/// The command line's own commands against the daemon: load, ps, a chat, unload, and down, which ends the daemon
/// and removes its socket.
#[test]
fn the_command_line_drives_the_daemon() {
    use tachyon_cli::serve::{client::Daemon, commands};
    if !Path::new("/dev/nvidiactl").exists() && !Path::new("/dev/dxg").exists() {
        eprintln!("no NVIDIA device: skipped");
        return;
    }
    let (model, socket) = daemon("serve-commands");
    let mut d = Daemon::connect(&socket).unwrap();
    commands::load(&mut d, &model, -1, (64, false)).unwrap();
    let listed = commands::ps(&mut d).unwrap();
    assert!(listed.contains(&model) && listed.contains("vram"), "{listed}");
    commands::run(&mut d, &model, "hello there, how are you", &[]).unwrap();
    commands::unload(&mut d, &model).unwrap();
    assert_eq!(commands::ps(&mut d).unwrap().lines().count(), 1, "only the header once unloaded");
    assert!(commands::run(&mut d, "no-such-model", "hi", &[]).unwrap_err().to_string().contains("code 2"), "NoModel");
    drop(d);
    assert_eq!(commands::down(&socket).unwrap(), "tachyon serve: down");
    assert!(!socket.exists() && Daemon::connect(&socket).is_err(), "the socket is gone");
    std::fs::remove_dir_all(socket.parent().unwrap()).unwrap();
}

#[test]
fn settings_come_from_flags_and_the_environment() {
    let mut o = tachyon_cli::serve::Options::default();
    assert!(o.set("vram-mib", "4300") && o.vram == Some(4300 << 20));
    assert!(o.set("keep-alive", "60") && o.keep_alive == Duration::from_secs(60));
    assert!(!o.set("keep-alive", "soon") && !o.set("vram", "1"), "unreadable or unknown");
    let up = std::process::Command::new(env!("CARGO_BIN_EXE_tachyon")).arg("up").env("TACHYON_VRAM_MIB", "lots").output().unwrap();
    assert_eq!(up.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&up.stderr).contains("TACHYON_VRAM_MIB=lots: not a number"), "{up:?}");
}
