//! Hugging Face Hub downloads: a repository's files, and each file fetched in [`CHUNK`]-byte ranges over several
//! kept-alive connections at once, written in place, hashed with SHA-256 in order while the rest arrives, and
//! resumable: a killed download continues from the chunks it had.
//!
//! A file downloads to `<name>.part`, with `<name>.part.chunks` holding a byte per chunk (1 once it is written).
//! When every chunk is in and the hash matches the Hub's (Git LFS files carry one), the part file takes the
//! file's name. Files outside LFS (configuration, tokenizer) are small: one request, checked by size.

use super::crypto::Sha256;
use super::http::{Connection, Url, fetch, get};
use super::tls::ClientConfig;
use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};
use tachyon::json::Json;

/// Bytes per chunk: what is marked written, and resumed, at a time. A download ends one chunk's hashing and flushing
/// after its last byte arrives.
pub const CHUNK: u64 = 8 << 20;
/// Connections at once, unless `TACHYON_CONNECTIONS` says otherwise.
const CONNECTIONS: usize = 8;
/// Attempts per chunk before the download gives up.
const ATTEMPTS: usize = 5;
/// Chunks a download keeps requested at once, shared among its connections: one connection asks for this many in
/// one request, so the CDN's wait before the first byte is paid once per 256 MiB, not per chunk; several
/// connections ask for fewer each, which keeps the hashing, which goes in order, close behind.
const IN_FLIGHT: usize = 32;
/// The Hub: where it is, and what its TLS connections trust.
pub struct Hub {
    base: String,
    config: Arc<ClientConfig>,
    /// `Authorization: Bearer …` from `HF_TOKEN`, sent to the Hub only (not to the CDN its downloads redirect to).
    auth: Option<String>,
    /// Connections a download opens at once (1 to 64).
    pub connections: usize,
}

/// A file of a repository: its path, size, and SHA-256 when it is stored in Git LFS.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    /// Path in the repository.
    pub path: String,
    /// Bytes.
    pub size: u64,
    /// The content's SHA-256, for LFS files.
    pub sha256: Option<[u8; 32]>,
}

impl Default for Hub {
    fn default() -> Hub {
        Hub::new()
    }
}

impl Hub {
    /// Hugging Face, or the mirror `TACHYON_HUB` names, trusting the system's certificates; with the access token in
    /// `HF_TOKEN`, if set, for gated repositories and higher rate limits.
    pub fn new() -> Hub {
        let hub = Hub::at(&std::env::var("TACHYON_HUB").unwrap_or_else(|_| "https://huggingface.co".into()), ClientConfig::system());
        hub.with_token(std::env::var("HF_TOKEN").ok().filter(|t| !t.is_empty()).as_deref())
    }

    /// This Hub, sending `token` to it.
    #[must_use]
    pub fn with_token(self, token: Option<&str>) -> Hub {
        Hub { auth: token.map(|t| format!("Bearer {}", t.trim())), ..self }
    }

    /// The headers every request to the Hub carries.
    fn headers(&self) -> Vec<(&str, &str)> {
        self.auth.as_deref().map(|a| ("Authorization", a)).into_iter().collect()
    }

    /// The Hub at `base` (a URL without a trailing `/`), trusting what `config` does, downloading over
    /// `TACHYON_CONNECTIONS` connections (8 by default).
    pub fn at(base: &str, config: ClientConfig) -> Hub {
        let connections = std::env::var("TACHYON_CONNECTIONS").ok().and_then(|n| n.parse().ok()).unwrap_or(CONNECTIONS);
        Hub { base: base.trim_end_matches('/').into(), config: Arc::new(config), auth: None, connections }
    }

    /// The files of `repo`'s `main` branch, in its folders too.
    pub fn files(&self, repo: &str) -> io::Result<Vec<Entry>> {
        let body = get(&format!("{}/api/models/{repo}/tree/main?recursive=true", self.base), &self.headers(), &self.config)?;
        let json = Json::parse(&body).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("the Hub's file list: {e}")))?;
        Ok(json.arr().unwrap_or_default().iter().filter(|f| f.get("type").and_then(Json::str) == Some("file")).filter_map(entry).collect())
    }

    /// Downloads `file` of `repo` into `dir`, unless it is already there, reporting progress about once a second.
    pub fn download(&self, repo: &str, file: &Entry, dir: &Path, progress: Progress) -> io::Result<()> {
        let target = dir.join(&file.path);
        if std::fs::metadata(&target).is_ok_and(|m| m.len() == file.size) {
            return Ok(());
        }
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let url = Url::parse(&format!("{}/{repo}/resolve/main/{}", self.base, file.path)).ok_or_else(|| io::Error::other("the Hub's address is not a URL"))?;
        match file.sha256 {
            None => small(&url, file, &target, (&self.config, self.auth.as_deref())),
            Some(sha) => Chunked::start(&url, file.size, &target, (&self.config, self.auth.as_deref()))?.run(sha, self.connections, progress),
        }
    }
}

/// One file of a tree listing.
fn entry(f: &Json) -> Option<Entry> {
    let path = f.get("path")?.str()?.to_string();
    let size = f.get("size")?.num()? as u64;
    let sha256 = f.get("lfs").and_then(|l| l.get("oid")?.str()).and_then(unhex);
    Some(Entry { path, size, sha256 })
}

/// 32 bytes from 64 hex digits.
fn unhex(s: &str) -> Option<[u8; 32]> {
    let mut out = [0u8; 32];
    for (o, pair) in out.iter_mut().zip(s.as_bytes().chunks(2)) {
        *o = u8::from_str_radix(std::str::from_utf8(pair).ok()?, 16).ok()?;
    }
    (s.len() == 64).then_some(out)
}

/// What a download reports as it goes: bytes in so far, of the total.
pub type Progress<'a> = &'a (dyn Fn(u64, u64) + Sync);

/// A file outside LFS: one request, checked by size, written whole.
fn small(url: &Url, file: &Entry, target: &Path, (config, auth): (&Arc<ClientConfig>, Option<&str>)) -> io::Result<()> {
    let mut body = Vec::with_capacity(usize::try_from(file.size).unwrap_or(0));
    let headers: Vec<_> = auth.map(|a| ("Authorization", a)).into_iter().collect();
    let (response, _) = fetch(&mut None, url, &headers, config, &mut |b| {
        body.extend_from_slice(b);
        Ok(())
    })?;
    if response.status != 200 || body.len() as u64 != file.size {
        return Err(io::Error::other(format!("{}: status {}, {} of {} bytes", file.path, response.status, body.len(), file.size)));
    }
    std::fs::write(target, body)
}

/// A download in chunks: the part file, the chunk states, and the URL the Hub's redirects led to once one is known.
struct Chunked {
    url: Mutex<Option<Url>>,
    first: Url,
    size: u64,
    part: File,
    state: File,
    done: Vec<AtomicBool>,
    next: AtomicUsize,
    received: AtomicU64,
    failed: Mutex<Option<io::Error>>,
    signal: (Mutex<()>, Condvar),
    target: std::path::PathBuf,
    config: Arc<ClientConfig>,
    auth: Option<String>,
}

impl Chunked {
    /// Opens (or reopens) the part and state files of `target`, to be fetched from `url`.
    fn start(url: &Url, size: u64, target: &Path, (config, auth): (&Arc<ClientConfig>, Option<&str>)) -> io::Result<Chunked> {
        let open = |ext| OpenOptions::new().read(true).write(true).create(true).truncate(false).open(sibling(target, ext));
        let (part, state) = (open("part")?, open("part.chunks")?);
        let chunks = size.div_ceil(CHUNK) as usize;
        part.set_len(size)?;
        let mut flags = vec![0u8; chunks];
        let have = state.read_at(&mut flags, 0)?;
        flags[have..].fill(0);
        let done: Vec<AtomicBool> = flags.iter().map(|&f| AtomicBool::new(f == 1)).collect();
        let received = done.iter().enumerate().filter(|d| d.1.load(Ordering::Relaxed)).map(|(i, _)| span(i, size).1).sum();
        let (failed, signal) = (Mutex::new(None), (Mutex::new(()), Condvar::new()));
        let (next, received) = (AtomicUsize::new(0), AtomicU64::new(received));
        Ok(Chunked {
            url: Mutex::new(None),
            first: url.clone(),
            size,
            part,
            state,
            done,
            next,
            received,
            failed,
            signal,
            target: target.into(),
            config: config.clone(),
            auth: auth.map(String::from),
        })
    }

    /// Fetches the missing chunks over several connections while this thread hashes them in order; checks the
    /// hash against `sha` and gives the file its name.
    fn run(self, sha: [u8; 32], connections: usize, progress: Progress) -> io::Result<()> {
        let (hash, flushed) = std::thread::scope(|s| {
            let (this, connections) = (&self, connections.clamp(1, 64));
            let run = (IN_FLIGHT / connections).max(1);
            (0..connections).for_each(|_| _ = s.spawn(move || this.worker(run)));
            let flusher = s.spawn(|| self.flush());
            (self.hash(progress), flusher.join().expect("the flusher does not panic"))
        });
        let hash = hash?;
        flushed?;
        if let Some(e) = self.failed.lock().expect("no thread panics holding it").take() {
            return Err(e);
        }
        if hash != sha {
            // Nothing of it can be trusted: the next attempt starts over.
            std::fs::remove_file(sibling(&self.target, "part.chunks"))?;
            return Err(io::Error::new(io::ErrorKind::InvalidData, format!("{}: the SHA-256 does not match the Hub's", self.target.display())));
        }
        self.part.sync_all()?;
        std::fs::rename(sibling(&self.target, "part"), &self.target)?;
        std::fs::remove_file(sibling(&self.target, "part.chunks"))
    }

    /// Takes `run` chunks at a time, in order, and fetches those still missing, until none is left or one fails for
    /// good.
    fn worker(&self, run: usize) {
        let mut conn = None;
        'claims: loop {
            let first = self.next.fetch_add(run, Ordering::Relaxed);
            if first >= self.done.len() || self.failed.lock().is_ok_and(|f| f.is_some()) {
                break;
            }
            let end = (first + run).min(self.done.len());
            let mut at = first;
            while let Some((from, to)) = self.missing(at, end) {
                if self.fetch_run(&mut conn, from, end).is_err() {
                    break 'claims;
                }
                at = to;
            }
        }
        self.signal.1.notify_all();
    }

    /// The first run of consecutive missing chunks in `from..end`, as a range of chunk indices.
    fn missing(&self, from: usize, end: usize) -> Option<(usize, usize)> {
        let first = (from..end).find(|&i| !self.done[i].load(Ordering::Acquire))?;
        let to = (first..end).find(|&i| self.done[i].load(Ordering::Acquire)).unwrap_or(end);
        Some((first, to))
    }

    /// The missing chunks of `from..end` into the part file, over `conn`, in up to [`ATTEMPTS`] attempts, each
    /// resuming after the chunks the last one finished; the last error is kept for the download to report.
    fn fetch_run(&self, conn: &mut Option<Connection>, from: usize, end: usize) -> Result<(), ()> {
        for attempt in 1..=ATTEMPTS {
            let Some(chunks) = self.missing(from, end) else { return Ok(()) };
            match self.try_run(conn, chunks) {
                Ok(()) => return Ok(()),
                Err(e) if attempt == ATTEMPTS => _ = self.failed.lock().expect("no thread panics holding it").get_or_insert(e),
                Err(_) => {
                    *conn = None;
                    std::thread::sleep(Duration::from_millis(500));
                }
            }
        }
        Err(())
    }

    /// One attempt at chunks `from..to`: one range request whose body is written in place as it arrives, each chunk
    /// marked as soon as it is whole. The first request of a connection goes to the Hub, whose redirect leads to the
    /// CDN the connection then stays on; later ones go to the CDN straight away.
    fn try_run(&self, conn: &mut Option<Connection>, (from, to): (usize, usize)) -> io::Result<()> {
        let (start, end) = (span(from, self.size).0, span(to - 1, self.size).0 + span(to - 1, self.size).1);
        let range = format!("bytes={start}-{}", end - 1);
        let url = self.url.lock().expect("no thread panics holding it").clone().unwrap_or_else(|| self.first.clone());
        let (mut at, mut chunk) = (start, from);
        let mut headers = vec![("Range", range.as_str())];
        // The token goes to the Hub, never to the CDN a connection stays on after the first redirect.
        headers.extend(self.auth.as_deref().filter(|_| url.host == self.first.host).map(|a| ("Authorization", a)));
        let (response, last) = fetch(conn, &url, &headers, &self.config, &mut |b| {
            self.part.write_all_at(b, at)?;
            at += b.len() as u64;
            while chunk < to && at >= span(chunk, self.size).0 + span(chunk, self.size).1 {
                self.mark(chunk);
                chunk += 1;
            }
            Ok(())
        })?;
        // A signed CDN URL that expired: the next attempt asks the Hub again.
        let expired = matches!(response.status, 401 | 403 | 410);
        *self.url.lock().expect("no thread panics holding it") = (!expired).then_some(last);
        if response.status != 206 || at != end {
            return Err(io::Error::other(format!("range {range}: status {}, {} of {} bytes", response.status, at - start, end - start)));
        }
        Ok(())
    }

    /// Records chunk `i` as written, on disk and for the hasher.
    fn mark(&self, i: usize) {
        if self.state.write_all_at(&[1], i as u64).is_ok() {
            self.received.fetch_add(span(i, self.size).1, Ordering::Relaxed);
            self.done[i].store(true, Ordering::Release);
            self.signal.1.notify_all();
        }
    }

    /// Waits for chunk `i`, calling `tick` at least every 200 ms; false when the download fails first.
    fn arrived(&self, i: usize, tick: &mut dyn FnMut()) -> bool {
        while !self.done[i].load(Ordering::Acquire) {
            if self.failed.lock().is_ok_and(|f| f.is_some()) {
                return false;
            }
            let guard = self.signal.0.lock().expect("no thread panics holding it");
            drop(self.signal.1.wait_timeout(guard, Duration::from_millis(200)));
            tick();
        }
        true
    }

    /// Hashes the chunks in order as they arrive, reporting progress; stops early when the download fails.
    fn hash(&self, progress: Progress) -> io::Result<[u8; 32]> {
        let (mut sha, mut buf, mut shown) = (Sha256::default(), vec![0u8; CHUNK as usize], Instant::now());
        let mut tick = || {
            if shown.elapsed() >= Duration::from_secs(1) {
                progress(self.received.load(Ordering::Relaxed), self.size);
                shown = Instant::now();
            }
        };
        for i in 0..self.done.len() {
            if !self.arrived(i, &mut tick) {
                return Ok([0; 32]);
            }
            let (start, len) = span(i, self.size);
            self.part.read_exact_at(&mut buf[..len as usize], start)?;
            sha.update(&buf[..len as usize]);
        }
        progress(self.size, self.size);
        Ok(sha.finish())
    }

    /// Flushes the part file to disk as chunks arrive, beside the hashing, so the flush before its rename is
    /// about one chunk instead of the whole file.
    fn flush(&self) -> io::Result<()> {
        for i in 0..self.done.len() {
            if !self.arrived(i, &mut || {}) {
                return Ok(());
            }
            self.part.sync_data()?;
        }
        Ok(())
    }
}

/// `target` with `.ext` added to its name.
fn sibling(target: &Path, ext: &str) -> std::path::PathBuf {
    target.with_file_name(format!("{}.{ext}", target.file_name().unwrap_or_default().to_string_lossy()))
}

/// Where chunk `i` of a file of `size` bytes starts, and its length.
fn span(i: usize, size: u64) -> (u64, u64) {
    let start = i as u64 * CHUNK;
    (start, CHUNK.min(size - start))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    /// A stand-in for the Hub and its CDN on a free port: repository `r` with `config.json` and `model.bin` (LFS,
    /// `data`), the model's resolve URL redirecting to `/cdn`, which serves ranges of `served` and records them.
    struct Fake {
        port: u16,
        ranges: Arc<Mutex<Vec<String>>>,
    }

    fn fake(data: &[u8], served: Vec<u8>) -> Fake {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let sha: String = Sha256::digest(data).iter().fold(String::new(), |s, b| s + &format!("{b:02x}"));
        let tree = format!(
            r#"[{{"type":"file","path":"config.json","size":2}},{{"type":"directory","path":"d"}},{{"type":"file","path":"model.bin","size":{},"lfs":{{"oid":"{sha}","size":{}}}}}]"#,
            data.len(),
            data.len()
        );
        let (ranges, served) = (Arc::new(Mutex::new(Vec::new())), Arc::new(served));
        let seen = ranges.clone();
        std::thread::spawn(move || {
            for sock in listener.incoming() {
                let (tree, served, seen) = (tree.clone(), served.clone(), seen.clone());
                std::thread::spawn(move || answer(sock.unwrap(), port, &tree, &served, &seen));
            }
        });
        Fake { port, ranges }
    }

    /// Answers the requests of one connection.
    fn answer(mut sock: std::net::TcpStream, port: u16, tree: &str, served: &[u8], seen: &Mutex<Vec<String>>) {
        let mut buf = Vec::new();
        loop {
            while !buf.windows(4).any(|w| w == b"\r\n\r\n") {
                let mut b = [0; 4096];
                match sock.read(&mut b) {
                    Ok(0) | Err(_) => return,
                    Ok(n) => buf.extend_from_slice(&b[..n]),
                }
            }
            let end = buf.windows(4).position(|w| w == b"\r\n\r\n").unwrap();
            let head = String::from_utf8_lossy(&buf[..end]).into_owned();
            buf.drain(..end + 4);
            let path = head.split(' ').nth(1).unwrap().to_string();
            let range = head
                .lines()
                .find_map(|l| l.strip_prefix("Range: bytes="))
                .map(|r| r.split_once('-').map(|(a, b)| (a.parse::<usize>().unwrap(), b.parse::<usize>().unwrap() + 1)).unwrap());
            let reply = |status: &str, extra: &str, body: &[u8]| {
                [format!("HTTP/1.1 {status}\r\nContent-Length: {}\r\n{extra}\r\n", body.len()).into_bytes(), body.to_vec()].concat()
            };
            let out = match (path.as_str(), range) {
                ("/api/models/r/tree/main?recursive=true", _) => reply("200 OK", "", tree.as_bytes()),
                ("/r/resolve/main/config.json", _) => reply("200 OK", "", b"{}"),
                ("/r/resolve/main/model.bin", _) => reply("302 Found", &format!("Location: http://127.0.0.1:{port}/cdn\r\n"), b""),
                ("/cdn", Some((a, b))) => {
                    seen.lock().unwrap().push(format!("{a}-{b}"));
                    reply("206 Partial Content", "", &served[a..b.min(served.len())])
                }
                _ => reply("404 Not Found", "", b""),
            };
            if sock.write_all(&out).is_err() {
                return;
            }
        }
    }

    fn hub(port: u16) -> Hub {
        Hub::at(&format!("http://127.0.0.1:{port}"), ClientConfig { anchors: Vec::new(), random: crate::net::crypto::random, now: || 0 })
    }

    /// `HF_TOKEN`'s bearer token goes with a request to the Hub.
    #[test]
    fn the_token_goes_to_the_hub() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            let mut head = Vec::new();
            while !head.windows(4).any(|w| w == b"\r\n\r\n") {
                let mut b = [0; 1024];
                let n = sock.read(&mut b).unwrap();
                head.extend_from_slice(&b[..n]);
            }
            sock.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n[]").unwrap();
            String::from_utf8_lossy(&head).into_owned()
        });
        assert!(hub(port).with_token(Some("hf_secret")).files("r").unwrap().is_empty());
        assert!(server.join().unwrap().lines().any(|l| l == "Authorization: Bearer hf_secret"));
    }

    /// A file of two and a half chunks downloads in ranges and checks, over a connection per chunk and over one; a
    /// download that stopped after its second chunk fetches only the other two; a corrupted byte is caught by the
    /// hash and the next attempt starts over. No request is made but for the file's bytes.
    #[test]
    fn downloads_resume_and_catch_corruption() {
        let data: Vec<u8> = (0..CHUNK as usize * 5 / 2).map(|i| (i % 251) as u8 ^ (i >> 20) as u8).collect();
        let dir = std::env::temp_dir().join(format!("tachyon-hub-{}", std::process::id()));
        let good = fake(&data, data.clone());
        let mut hub = hub(good.port);
        hub.connections = IN_FLIGHT;
        let files = hub.files("r").unwrap();
        assert_eq!(files.iter().map(|f| (f.path.as_str(), f.sha256.is_some())).collect::<Vec<_>>(), [("config.json", false), ("model.bin", true)]);
        for f in &files {
            hub.download("r", f, &dir, &|_, _| {}).unwrap();
        }
        assert!(std::fs::read(dir.join("model.bin")).unwrap() == data && std::fs::read(dir.join("config.json")).unwrap() == b"{}");
        assert_eq!(good.ranges.lock().unwrap().len(), 3, "a request per chunk");
        // Killed after chunk 1 of 3: its part file and state are there; only chunks 0 and 2 are fetched again.
        std::fs::remove_file(dir.join("model.bin")).unwrap();
        let (c, n) = (CHUNK as usize, data.len());
        std::fs::write(dir.join("model.bin.part"), [&vec![0; c][..], &data[c..2 * c], &vec![0; n - 2 * c]].concat()).unwrap();
        std::fs::write(dir.join("model.bin.part.chunks"), [0, 1, 0]).unwrap();
        good.ranges.lock().unwrap().clear();
        hub.download("r", &files[1], &dir, &|_, _| {}).unwrap();
        assert_eq!(std::fs::read(dir.join("model.bin")).unwrap(), data);
        let mut ranges = good.ranges.lock().unwrap().clone();
        ranges.sort();
        assert_eq!(ranges, [format!("0-{c}"), format!("{}-{n}", 2 * c)], "chunks 0 and 2, in any order");
        // Over one connection the file is one request, and resumed, one request per run of missing chunks.
        let mut one = self::hub(good.port);
        one.connections = 1;
        std::fs::remove_file(dir.join("model.bin")).unwrap();
        good.ranges.lock().unwrap().clear();
        one.download("r", &files[1], &dir, &|_, _| {}).unwrap();
        assert_eq!(std::fs::read(dir.join("model.bin")).unwrap(), data);
        assert_eq!(*good.ranges.lock().unwrap(), [format!("0-{n}")], "the whole file");
        std::fs::remove_file(dir.join("model.bin")).unwrap();
        std::fs::write(dir.join("model.bin.part"), [&vec![0; c][..], &data[c..2 * c], &vec![0; n - 2 * c]].concat()).unwrap();
        std::fs::write(dir.join("model.bin.part.chunks"), [0, 1, 0]).unwrap();
        good.ranges.lock().unwrap().clear();
        one.download("r", &files[1], &dir, &|_, _| {}).unwrap();
        assert_eq!(std::fs::read(dir.join("model.bin")).unwrap(), data);
        assert_eq!(*good.ranges.lock().unwrap(), [format!("0-{c}"), format!("{}-{n}", 2 * c)], "chunks 0 and 2");
        // A server that flips one byte.
        let mut bad = data.clone();
        bad[c + 7] ^= 1;
        let corrupt = fake(&data, bad);
        let err = self::hub(corrupt.port).download("r", &files[1], &dir.join("bad"), &|_, _| {}).unwrap_err();
        assert!(err.to_string().contains("SHA-256"), "{err}");
        assert!(!dir.join("bad/model.bin").exists() && !dir.join("bad/model.bin.part.chunks").exists());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
