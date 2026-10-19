//! `tachyon serve`: one engine for every process of this user, its models kept in VRAM between requests as their
//! keep-alive says. The protocol is `interface/tachyon.h` (the socket, the shared memory, the ring) and
//! `interface/tachyon.xml` (the messages, SBE): a client needs no library of ours, only those two files.
//!
//! One thread serves everything: it polls the listening socket and every client's, takes each request in the order
//! its doorbell rang, one at a time, and streams the answer into that client's ring. Between tokens it looks for that
//! client's cancel, and every second it lets the engine move out models whose keep-alive ran out.

pub mod client;
pub mod commands;
mod sbe;
mod shm;

use sbe::{Bad, Blob, Fields, Reader, Writer};
use shm::{DOORBELL_CANCEL, DOORBELL_REQUEST, POLLHUP, POLLIN, Pollfd, Shared};
use std::collections::VecDeque;
use std::io::Read as _;
use std::os::fd::AsRawFd;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tachyon::encoder::laya::{Kind, Question};
use tachyon::engine::{Engine, Limits, Load, Request, Tier};
use tachyon::media::{Audio, Image};
use tachyon::ptx::Sampling;
use tachyon::token::{Message, Role};

/// How the daemon runs.
#[derive(Debug, Clone)]
pub struct Options {
    /// The socket to listen on.
    pub socket: PathBuf,
    /// How long a model stays in VRAM after its last request, unless a request says otherwise.
    pub keep_alive: Duration,
    /// Bytes of each client's data area.
    pub data: usize,
    /// VRAM for the models, in bytes; `None`: 70% of the GPU's.
    pub vram: Option<usize>,
    /// Pinned host memory for the models moved out of VRAM to make room, in bytes; `None`: half the RAM.
    pub host: Option<usize>,
}

impl Default for Options {
    fn default() -> Options {
        Options { socket: socket(), keep_alive: Duration::from_secs(300), data: 64 << 20, vram: None, host: None }
    }
}

/// The settings [`Options::set`] takes from the environment, as `TACHYON_KEEP_ALIVE` and so on.
const FROM_ENV: [&str; 4] = ["keep-alive", "data-mib", "vram-mib", "host-mib"];

impl Options {
    /// The defaults, changed by `$TACHYON_KEEP_ALIVE` (seconds), `$TACHYON_DATA_MIB`, `$TACHYON_VRAM_MIB` and
    /// `$TACHYON_HOST_MIB` (`env.example` lists every setting); an unreadable value is an error naming it.
    pub fn from_env() -> Result<Options, String> {
        let mut opts = Options::default();
        for key in FROM_ENV {
            let var = format!("TACHYON_{}", key.to_uppercase().replace('-', "_"));
            if let Ok(value) = std::env::var(&var)
                && !opts.set(key, &value)
            {
                return Err(format!("{var}={value}: not a number"));
            }
        }
        Ok(opts)
    }

    /// Sets option `key` (`socket`, `keep-alive` in seconds, `data-mib`, `vram-mib`, `host-mib`) to `value`;
    /// false if there is no such option or `value` does not read.
    pub fn set(&mut self, key: &str, value: &str) -> bool {
        let mib = |v: &str| v.parse::<usize>().ok().and_then(|n| n.checked_mul(1 << 20));
        match key {
            "socket" => {
                self.socket = value.into();
                true
            }
            "keep-alive" => value.parse().map(|s| self.keep_alive = Duration::from_secs(s)).is_ok(),
            "data-mib" => mib(value).map(|n| self.data = n).is_some(),
            "vram-mib" => mib(value).map(|n| self.vram = Some(n)).is_some(),
            "host-mib" => mib(value).map(|n| self.host = Some(n)).is_some(),
            _ => false,
        }
    }
}

/// The socket clients find by default: `$TACHYON_SOCKET`, `$XDG_RUNTIME_DIR/tachyon.sock`, `/tmp/tachyon-<uid>.sock`.
pub fn socket() -> PathBuf {
    if let Some(p) = std::env::var_os("TACHYON_SOCKET") {
        return p.into();
    }
    match std::env::var_os("XDG_RUNTIME_DIR") {
        Some(dir) => Path::new(&dir).join("tachyon.sock"),
        None => format!("/tmp/tachyon-{}.sock", uid()).into(),
    }
}

/// This process's user id, from its home in `/proc`.
fn uid() -> u32 {
    use std::os::unix::fs::MetadataExt as _;
    std::fs::metadata("/proc/self").map_or(0, |m| m.uid())
}

/// A connected client: its socket, its shared memory, whether its request waits in the queue, and whether it rang
/// the cancel doorbell since its last request.
struct Client {
    stream: UnixStream,
    shm: Shared,
    queued: bool,
    cancel: bool,
}

/// Set by a Shutdown request: the loop ends once its answer is out.
static STOP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Serves on `opts.socket`, with the engine on GPU 0, until a Shutdown request; then removes the socket and returns,
/// the engine's models freed.
pub fn serve(opts: &Options) -> tachyon::Result<()> {
    let listener = listen(&opts.socket)?;
    let mut engine = Engine::new(0, Limits { vram: opts.vram, host: opts.host })?;
    let (mut clients, mut queue): (Vec<Option<Client>>, VecDeque<usize>) = (Vec::new(), VecDeque::new());
    eprintln!("tachyon serve: listening on {}", opts.socket.display());
    loop {
        let mut fds: Vec<Pollfd> = std::iter::once(listener.as_raw_fd())
            .chain(clients.iter().map(|c| c.as_ref().map_or(-1, |c| c.stream.as_raw_fd())))
            .map(|fd| Pollfd { fd, events: POLLIN, revents: 0 })
            .collect();
        shm::wait(&mut fds, 1000)?;
        if fds[0].revents & POLLIN != 0 {
            accept(&listener, opts, &mut clients);
        }
        for i in (0..fds.len() - 1).filter(|&i| fds[i + 1].revents != 0) {
            ring(&mut clients[i], i, &mut queue);
        }
        if drain(&mut engine, (&mut clients, &mut queue), opts) {
            eprintln!("tachyon serve: shut down");
            return Ok(std::fs::remove_file(&opts.socket)?);
        }
        engine.expire()?;
    }
}

/// Answers the queued requests, in order; whether one was a Shutdown.
fn drain(engine: &mut Engine, (clients, queue): (&mut [Option<Client>], &mut VecDeque<usize>), opts: &Options) -> bool {
    while let Some(i) = queue.pop_front() {
        if let Some(c) = clients[i].as_mut().filter(|c| c.queued) {
            c.queued = false;
            answer(engine, c, opts);
        }
    }
    STOP.swap(false, std::sync::atomic::Ordering::Relaxed)
}

/// Binds `path`, replacing a socket no daemon answers on, readable and writable by this user only.
fn listen(path: &Path) -> tachyon::Result<UnixListener> {
    use std::os::unix::fs::PermissionsExt as _;
    if path.exists() && UnixStream::connect(path).is_err() {
        std::fs::remove_file(path)?;
    }
    let listener = UnixListener::bind(path)?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    Ok(listener)
}

/// Lets in a client of this user: its shared memory, sent with the Welcome.
fn accept(listener: &UnixListener, opts: &Options, clients: &mut Vec<Option<Client>>) {
    let Ok((stream, _)) = listener.accept() else { return };
    if !shm::same_user(&stream) {
        return;
    }
    let shared = Shared::new(opts.data).and_then(|shm| {
        let welcome = Writer::new(200, &Fields::default().u64(shm.size() as u64).0);
        shm.send(&stream, &welcome.0)?;
        stream.set_nonblocking(true)?;
        Ok(shm)
    });
    if let Ok(shm) = shared {
        let client = Some(Client { stream, shm, queued: false, cancel: false });
        match clients.iter().position(Option::is_none) {
            Some(free) => clients[free] = client,
            None => clients.push(client),
        }
    }
}

/// Reads a client's doorbells: a request joins the queue; a closed socket drops the client.
fn ring(slot: &mut Option<Client>, i: usize, queue: &mut VecDeque<usize>) {
    let Some(c) = slot.as_mut() else { return };
    match bells(c) {
        None => *slot = None,
        Some(true) if !c.queued => {
            c.queued = true;
            queue.push_back(i);
        }
        Some(_) => {}
    }
}

/// Reads the doorbells `c` rang: whether one asked for a request (a cancel after it is kept in `c.cancel`), or
/// `None` once the client has left.
fn bells(c: &mut Client) -> Option<bool> {
    let (mut bytes, mut asked) = ([0u8; 64], false);
    loop {
        match c.stream.read(&mut bytes) {
            Ok(0) => return None,
            Ok(n) => {
                for &b in &bytes[..n] {
                    asked |= b == DOORBELL_REQUEST;
                    c.cancel = b == DOORBELL_CANCEL || c.cancel && b != DOORBELL_REQUEST;
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => return Some(asked),
            Err(_) => return None,
        }
    }
}

/// A request, read out of the request area.
enum Ask {
    Chat(Chat),
    Embed(Embed),
    Decide(Decide),
    Load { model: String, keep: Option<Duration>, context: usize, pin: bool },
    Unload(String),
    List,
    Shutdown,
}

/// An Embed request: the model, the texts, the values kept of each embedding (0: all) and where they go.
struct Embed {
    model: String,
    texts: Vec<String>,
    dims: usize,
    out: Blob,
}

/// A Laya question: its kind, its instructions, its options' labels and descriptions.
struct Asked(Kind, String, Vec<(String, String)>);

/// A Decide request: the model, the state and whether it is a conversation, the questions, and where the
/// probabilities go.
struct Decide {
    model: String,
    state: String,
    conversation: bool,
    questions: Vec<Asked>,
    out: Blob,
}

/// A message of a Chat: its role, its text, and its attachments (media, bytes).
struct Said(Role, String, Vec<(u8, Blob)>);

/// A Chat request: the conversation (role, text, attachments as blobs, images first), how to sample, and the model.
struct Chat {
    model: String,
    messages: Vec<Said>,
    sampling: Sampling,
    max_tokens: usize,
    think: bool,
    keep: Option<Duration>,
}

/// A request keep-alive in milliseconds: −1 the default, −2 for good, else as given.
fn keep(ms: u64) -> Option<Duration> {
    match ms as i64 {
        -1 => None,
        ms if ms < 0 => Some(Duration::MAX),
        ms => Some(Duration::from_millis(ms as u64)),
    }
}

/// Reads the request in `bytes`: its number, and what it asks (or why it cannot be read).
fn read(bytes: &[u8]) -> (u64, Result<Ask, Bad>) {
    let Ok((template, block, mut r)) = Reader::open(bytes) else { return (0, Err(Bad)) };
    let id = block.u64(0).unwrap_or(0);
    let ask = match template {
        1 => read_chat(block, &mut r).map(Ask::Chat),
        2 => read_embed(block, &mut r),
        3 => read_decide(block, &mut r),
        4 => read_load(block, &mut r),
        5 => r.text().map(Ask::Unload),
        6 => Ok(Ask::List),
        7 => Ok(Ask::Shutdown),
        _ => Err(Bad),
    };
    (id, ask)
}

/// A Load's block and model.
fn read_load(b: sbe::Block<'_>, r: &mut Reader<'_>) -> Result<Ask, Bad> {
    Ok(Ask::Load { model: r.text()?, keep: keep(b.u64(8)?), context: b.u32(16)? as usize, pin: b.u8(20)? != 0 })
}

/// A Chat's block and groups.
fn read_chat(b: sbe::Block<'_>, r: &mut Reader<'_>) -> Result<Chat, Bad> {
    let (len, n) = r.group()?;
    let messages = (0..n).map(|_| read_said(r, len)).collect::<Result<_, _>>()?;
    let max_tokens = match b.u32(24)? {
        0 => usize::MAX,
        n => n as usize,
    };
    Ok(Chat { model: r.text()?, messages, sampling: read_sampling(b)?, max_tokens, think: b.u8(48)? != 0, keep: keep(b.u64(16)?) })
}

/// A Chat's sampling fields.
fn read_sampling(b: sbe::Block<'_>) -> Result<Sampling, Bad> {
    let (seed, top_k, temperature) = (b.u64(8)?, b.u32(28)?, b.f32(32)?);
    let (top_p, min_p, penalty) = (b.f32(36)?, b.f32(40)?, b.f32(44)?);
    Ok(Sampling { seed, top_k, temperature, top_p, min_p, penalty, ..Sampling::default() })
}

/// One message of a Chat, whose entries take `len` bytes.
fn read_said(r: &mut Reader<'_>, len: usize) -> Result<Said, Bad> {
    let role = [Role::System, Role::User, Role::Model].get(usize::from(r.entry(len)?.u8(0)?)).copied().ok_or(Bad)?;
    let (alen, an) = r.group()?;
    let attached = (0..an).map(|_| r.entry(alen).and_then(|e| Ok((e.u8(16)?, e.blob(0)?)))).collect::<Result<_, _>>()?;
    Ok(Said(role, r.text()?, attached))
}

/// An Embed's block and groups.
fn read_embed(b: sbe::Block<'_>, r: &mut Reader<'_>) -> Result<Ask, Bad> {
    let (len, n) = r.group()?;
    let texts = (0..n).map(|_| r.entry(len).and_then(|_| r.text())).collect::<Result<_, _>>()?;
    Ok(Ask::Embed(Embed { texts, model: r.text()?, dims: b.u32(24)? as usize, out: b.blob(8)? }))
}

/// A Decide's block and groups.
fn read_decide(b: sbe::Block<'_>, r: &mut Reader<'_>) -> Result<Ask, Bad> {
    let (len, n) = r.group()?;
    let questions = (0..n).map(|_| read_asked(r, len)).collect::<Result<_, _>>()?;
    Ok(Ask::Decide(Decide { model: r.text()?, state: r.text()?, conversation: b.u8(24)? != 0, questions, out: b.blob(8)? }))
}

/// One question of a Decide, whose entries take `len` bytes.
fn read_asked(r: &mut Reader<'_>, len: usize) -> Result<Asked, Bad> {
    let kind = [Kind::Choice, Kind::Score, Kind::Noul].get(usize::from(r.entry(len)?.u8(0)?)).copied().ok_or(Bad)?;
    let (olen, on) = r.group()?;
    let options = (0..on).map(|_| r.entry(olen).and_then(|_| Ok((r.text()?, r.text()?)))).collect::<Result<_, _>>()?;
    Ok(Asked(kind, r.text()?, options))
}

/// Why a request failed, as `tachyon.xml`'s `Code` numbers it.
#[derive(Clone, Copy)]
enum Code {
    BadRequest = 1,
    NoModel = 2,
    Engine = 3,
    Cancelled = 4,
    TooLarge = 5,
}

/// A request's failure: its code and message.
struct Failed(Code, String);

impl From<tachyon::Error> for Failed {
    fn from(e: tachyon::Error) -> Failed {
        Failed(Code::Engine, e.to_string())
    }
}

/// Answers the request waiting in `c`'s request area, ending with Done (or the answer's own last message) or Error.
fn answer(engine: &mut Engine, c: &mut Client, opts: &Options) {
    let (id, ask) = read(c.shm.request());
    let result = match ask {
        _ if std::mem::take(&mut c.cancel) => Err(Failed(Code::Cancelled, "cancelled".into())),
        Err(Bad) => Err(Failed(Code::BadRequest, "not a request of tachyon.xml, or out of the shared memory".into())),
        Ok(ask) => run(engine, c, opts, (id, ask)),
    };
    let msg = match result {
        Ok(msg) => msg,
        Err(Failed(code, text)) => {
            let mut w = Writer::new(103, &Fields::default().u64(id).u16(code as u16).0);
            w.text(&text);
            w
        }
    };
    send(c, &msg.0);
}

/// Appends `msg` to `c`'s ring; `false` if the client left meanwhile.
fn send(c: &Client, msg: &[u8]) -> bool {
    let fd = c.stream.as_raw_fd();
    c.shm.append(msg, &mut || {
        let mut p = [Pollfd { fd, events: POLLIN, revents: 0 }];
        shm::wait(&mut p, 0).is_err() || p[0].revents & POLLHUP != 0
    })
}

/// Runs request `ask` numbered `id`: its last answer, or why it failed.
fn run(engine: &mut Engine, c: &mut Client, opts: &Options, (id, ask): (u64, Ask)) -> Result<Writer, Failed> {
    let done = |block: Fields| Writer::new(102, &Fields::default().u64(id).0.into_iter().chain(block.0).collect::<Vec<_>>());
    match ask {
        Ask::Chat(chat) => chat_answer(engine, c, opts, (id, chat)),
        Ask::Embed(ask) => embed(engine, c, opts, (id, ask)),
        Ask::Decide(ask) => decide(engine, c, opts, (id, ask)),
        Ask::Load { model, keep, context, pin } => {
            let mut load = Load { keep_alive: keep.unwrap_or(opts.keep_alive), pin, ..Load::default() };
            if context > 0 {
                load.settings.context = context;
                load.settings.chunk = load.settings.chunk.min(context);
            }
            engine.load(&model, &dir(&model)?, &load).map_err(|e| Failed(Code::Engine, format!("{model}: {e}")))?;
            Ok(done(Fields::default().u64(0).u64(0).u32(0).u32(0)))
        }
        Ask::Unload(model) => {
            engine.unload(&model);
            Ok(done(Fields::default().u64(0).u64(0).u32(0).u32(0)))
        }
        Ask::List => Ok(list(engine, id)),
        Ask::Shutdown => {
            STOP.store(true, std::sync::atomic::Ordering::Relaxed);
            Ok(done(Fields::default().u64(0).u64(0).u32(0).u32(0)))
        }
    }
}

/// An Embed's embeddings, written to its `out`; returns its Embedded.
fn embed(engine: &mut Engine, c: &Client, opts: &Options, (id, ask): (u64, Embed)) -> Result<Writer, Failed> {
    let Embed { model, texts, dims, out } = ask;
    ready(engine, &model, opts, None)?;
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let rows = engine.embed(&model, &refs, (dims > 0).then_some(dims))?;
    let bytes: Vec<u8> = rows.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
    if !c.shm.write(out.offset, out.length, &bytes) {
        return Err(Failed(Code::TooLarge, format!("{} bytes of embeddings do not fit the blob", bytes.len())));
    }
    let dims = rows.first().map_or(0, Vec::len) as u32;
    Ok(Writer::new(104, &Fields::default().u64(id).u32(rows.len() as u32).u32(dims).0))
}

/// The directory of model `name`: pulled, or a directory holding a converted model.
fn dir(name: &str) -> Result<PathBuf, Failed> {
    let pulled = crate::models::home().join(name);
    [pulled, PathBuf::from(name)].into_iter().find(|d| d.join("weights.wpk").exists()).ok_or_else(|| Failed(Code::NoModel, format!("no model {name}")))
}

/// Loads model `name` unless it is loaded, kept for `keep` or the daemon's default.
fn ready(engine: &mut Engine, name: &str, opts: &Options, keep: Option<Duration>) -> Result<(), Failed> {
    if engine.tier(name).is_none() {
        let load = Load { keep_alive: keep.unwrap_or(opts.keep_alive), ..Load::default() };
        engine.load(name, &dir(name)?, &load).map_err(|e| Failed(Code::Engine, format!("{name}: {e}")))?;
    }
    Ok(())
}

/// Streams a Chat's answer as Text messages; returns its Done.
fn chat_answer(engine: &mut Engine, c: &mut Client, opts: &Options, (id, chat): (u64, Chat)) -> Result<Writer, Failed> {
    ready(engine, &chat.model, opts, chat.keep)?;
    let media = attachments(&c.shm, &chat.messages)?;
    let messages: Vec<Message> =
        chat.messages.iter().zip(&media).map(|(Said(role, text, _), (images, audio))| Message { images, audio, ..Message::new(*role, text) }).collect();
    let req = Request { messages: &messages, think: chat.think, sampling: chat.sampling, max_tokens: chat.max_tokens, keep_alive: chat.keep };
    let start = Instant::now();
    let mut stream = engine.chat(&chat.model, &req)?;
    let (prefill, mid) = (start.elapsed(), Instant::now());
    while let Some(text) = stream.next_text()? {
        let mut w = Writer::new(101, &Fields::default().u64(id).0);
        w.text(text);
        if !send(c, &w.0) || bells(c).is_none() || c.cancel {
            c.cancel = false;
            return Err(Failed(Code::Cancelled, "cancelled".into()));
        }
    }
    let nanos = |d: Duration| d.as_nanos() as u64;
    let block = Fields::default().u64(id).u64(nanos(prefill)).u64(nanos(mid.elapsed())).u32(stream.prompt() as u32).u32(stream.tokens() as u32);
    Ok(Writer::new(102, &block.0))
}

/// A message's images and audio.
type Media = (Vec<Image>, Vec<Audio>);

/// The images and audio attached to each of `messages`, decoded from the data area.
fn attachments(shm: &Shared, messages: &[Said]) -> Result<Vec<Media>, Failed> {
    let bad = |what: &str| Failed(Code::BadRequest, format!("an attachment is not {what}, or is out of the data area"));
    let bytes = |b: Blob| shm.read(b.offset, b.length);
    let mut media = Vec::with_capacity(messages.len());
    for Said(_, _, attached) in messages {
        let images = attached.iter().filter(|a| a.0 == 0).map(|a| bytes(a.1).and_then(|b| tachyon::media::decode(&b).ok()).ok_or_else(|| bad("an image")));
        let audio = attached.iter().filter(|a| a.0 == 1).map(|a| bytes(a.1).and_then(|b| tachyon::media::decode_wav(&b).ok()).ok_or_else(|| bad("a WAV file")));
        media.push((images.collect::<Result<_, _>>()?, audio.collect::<Result<_, _>>()?));
    }
    Ok(media)
}

/// Laya's probabilities for a Decide, written to its `out`; returns its Decided. `laya` picks the checkpoint by the
/// state's language.
fn decide(engine: &mut Engine, c: &Client, opts: &Options, (id, ask): (u64, Decide)) -> Result<Writer, Failed> {
    let Decide { mut model, state, conversation, questions, out } = ask;
    if model == "laya" {
        let english = tachyon::encoder::route::english(&state) && dir("laya-english").is_ok();
        model = if english { "laya-english" } else { "laya-multilingual" }.into();
    }
    ready(engine, &model, opts, None)?;
    let pairs: Vec<Vec<(&str, &str)>> = questions.iter().map(|q| q.2.iter().map(|(l, d)| (l.as_str(), d.as_str())).collect()).collect();
    let asked: Vec<Question> = questions.iter().zip(&pairs).map(|(q, p)| Question { kind: q.0, instructions: &q.1, options: p }).collect();
    let probs = engine.decide(&model, &state, conversation, &asked)?;
    let bytes: Vec<u8> = probs.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
    if !c.shm.write(out.offset, out.length, &bytes) {
        return Err(Failed(Code::TooLarge, format!("{} bytes of probabilities do not fit the blob", bytes.len())));
    }
    let mut w = Writer::new(105, &Fields::default().u64(id).0);
    w.group(4, probs.len());
    for p in &probs {
        w.bytes(&(p.len() as u32).to_le_bytes());
    }
    Ok(w)
}

/// The loaded models, as a Models message.
fn list(engine: &Engine, id: u64) -> Writer {
    let models = engine.models();
    let mut w = Writer::new(106, &Fields::default().u64(id).0);
    w.group(17, models.len());
    for m in &models {
        let left = m.left.map_or(-1, |d| d.as_millis().min(i64::MAX as u128) as i64);
        let tier = match m.tier {
            Tier::Vram => 0,
            Tier::Host => 1,
            Tier::Disk => 2,
        };
        w.bytes(&Fields::default().u64(m.vram as u64).u64(left as u64).u8(tier).0).text(&m.name);
    }
    w
}
