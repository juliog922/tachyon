//! The commands that drive the daemon: `up` starts it in the background, `down` stops it, `ps` lists its models,
//! `load` and `unload` place them; `run`, `transcribe`, `embed` and `decide` go through it when it is up, so a
//! model stays in VRAM between commands as its keep-alive says.

use super::client::{Daemon, Last, request};
use super::sbe::{Bad, Fields, Reader, Writer};
use super::{Options, socket};
use crate::encoders::{Asked, questions, render, rows};
use std::io::{self, Write as _};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tachyon::encoder::laya::Kind;
use tachyon::json::Json;

/// The daemon on the default socket ([`socket`]), if one answers.
pub fn running() -> Option<Daemon> {
    Daemon::connect(&socket()).ok()
}

/// Where `up` writes the daemon's log: `$XDG_STATE_HOME/tachyon/serve.log` (`~/.local/state/tachyon`).
fn log() -> PathBuf {
    let state =
        std::env::var_os("XDG_STATE_HOME").map_or_else(|| PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".local/state"), PathBuf::from);
    state.join("tachyon/serve.log")
}

/// Starts `tachyon serve` with `opts` in the background, its output in `$XDG_STATE_HOME/tachyon/serve.log`
/// (`~/.local/state/tachyon`), and waits until it answers.
pub fn up(opts: &Options) -> io::Result<String> {
    if Daemon::connect(&opts.socket).is_ok() {
        return Ok(format!("tachyon serve: already up on {}", opts.socket.display()));
    }
    let path = log();
    let mut child = spawn(opts, &path)?;
    let start = Instant::now();
    while Daemon::connect(&opts.socket).is_err() {
        if child.try_wait()?.is_some() || start.elapsed() > Duration::from_secs(120) {
            return Err(io::Error::other(format!("tachyon serve did not start; see {}", path.display())));
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    Ok(format!("tachyon serve: up (pid {}) on {}, log {}", child.id(), opts.socket.display(), path.display()))
}

/// `tachyon serve` with `opts`, in its own process group (a terminal's Ctrl-C does not reach it), its output
/// appended to `log`.
fn spawn(opts: &Options, log: &Path) -> io::Result<std::process::Child> {
    use std::os::unix::process::CommandExt as _;
    std::fs::create_dir_all(log.parent().unwrap_or(Path::new(".")))?;
    let out = std::fs::OpenOptions::new().create(true).append(true).open(log)?;
    let mut args =
        vec![format!("--socket={}", opts.socket.display()), format!("--keep-alive={}", opts.keep_alive.as_secs()), format!("--data-mib={}", opts.data >> 20)];
    args.extend(opts.vram.map(|b| format!("--vram-mib={}", b >> 20)));
    args.extend(opts.host.map(|b| format!("--host-mib={}", b >> 20)));
    let mut serve = std::process::Command::new(std::env::current_exe()?);
    serve.arg("serve").args(args).stdin(std::process::Stdio::null()).stdout(out.try_clone()?).stderr(out).process_group(0).spawn()
}

/// Stops the daemon on `path`: its models freed, its socket removed.
pub fn down(path: &Path) -> io::Result<String> {
    let mut d = Daemon::connect(path).map_err(|_| io::Error::other(format!("no tachyon serve on {}", path.display())))?;
    let id = d.id();
    d.call(&request(7, id, Fields::default()), &mut |_| {})?;
    let start = Instant::now();
    while path.exists() && start.elapsed() < Duration::from_secs(10) {
        std::thread::sleep(Duration::from_millis(50));
    }
    Ok("tachyon serve: down".into())
}

/// The daemon's models: name, where, VRAM, time left in memory.
pub fn ps(d: &mut Daemon) -> io::Result<String> {
    let id = d.id();
    let last = d.call(&request(6, id, Fields::default()), &mut |_| {})?;
    let (_, _, mut r) = Reader::open(&last.bytes).map_err(bad)?;
    let (len, n) = r.group().map_err(bad)?;
    let mut out = format!("{:<32} {:<5} {:>9}  {}\n", "MODEL", "TIER", "VRAM", "KEEP-ALIVE");
    for _ in 0..n {
        out += &row(&mut r, len).map_err(bad)?;
    }
    Ok(out.trim_end().into())
}

/// A Models entry, as a line of `ps`.
fn row(r: &mut Reader<'_>, len: usize) -> Result<String, Bad> {
    let e = r.entry(len)?;
    let (vram, left, tier) = (e.u64(0)?, e.u64(8)? as i64, e.u8(16)?);
    let on_disk = tier == 2;
    let tier = ["vram", "host", "disk"].get(usize::from(tier)).unwrap_or(&"?");
    let left = match left {
        _ if on_disk => "-".into(),
        ..0 => "pinned".into(),
        ms => format!("{:.0} s", ms as f64 / 1e3),
    };
    let vram = if vram == 0 { "-".into() } else { format!("{:.2} GB", vram as f64 / 1e9) };
    Ok(format!("{:<32} {tier:<5} {vram:>9}  {left}\n", r.text()?))
}

/// A malformed answer.
fn bad(_: Bad) -> io::Error {
    io::Error::other("an answer the command line cannot read")
}

/// The name the daemon knows `model` by: a pulled model's name, or a directory's full path.
pub fn name(model: &str) -> String {
    let pulled = crate::models::home().join(model).join("weights.wpk").exists();
    match std::fs::canonicalize(model) {
        Ok(path) if !pulled => path.display().to_string(),
        _ => model.into(),
    }
}

/// Loads `model`, kept `keep` milliseconds (−1: the daemon's default, −2: never moved out), with `context`
/// positions (0: the default), pinned if asked.
pub fn load(d: &mut Daemon, model: &str, keep: i64, (context, pin): (u32, bool)) -> io::Result<String> {
    let id = d.id();
    let mut w = request(4, id, Fields::default().u64(keep as u64).u32(context).u8(u8::from(pin)));
    w.text(&name(model));
    d.call(&w, &mut |_| {})?;
    Ok(format!("{model}: loaded"))
}

/// Drops `model` from every tier.
pub fn unload(d: &mut Daemon, model: &str) -> io::Result<String> {
    let id = d.id();
    let mut w = request(5, id, Fields::default());
    w.text(&name(model));
    d.call(&w, &mut |_| {})?;
    Ok(format!("{model}: unloaded"))
}

/// Answers `prompt` with `model`, greedily, its `files` (`--image=`, `--audio=`) attached; prints the answer as it
/// comes and the speed, as `tachyon run` does in-process.
pub fn run(d: &mut Daemon, model: &str, prompt: &str, files: &[&str]) -> io::Result<()> {
    let mut attached = Vec::new();
    let mut at = 0;
    for f in files {
        let (media, path) = f.strip_prefix("--image=").map(|p| (0u8, p)).or_else(|| f.strip_prefix("--audio=").map(|p| (1, p))).unwrap_or((0, f));
        let blob = d.put(at, &std::fs::read(path)?)?;
        at += blob.length as usize;
        attached.push((media, blob));
    }
    let id = d.id();
    let sampling = Fields::default().u64(0).u64(u64::MAX).u32(0).u32(0).u32(0).u32(1f32.to_bits()).u32(0).u32(1f32.to_bits()).u8(0);
    let mut w = request(1, id, sampling);
    w.group(1, 1).bytes(&[1]).group(17, attached.len());
    for (media, b) in &attached {
        w.bytes(&Fields::default().u64(b.offset).u64(b.length).u8(*media).0);
    }
    w.text(prompt).text(&name(model));
    let last = d.call(&w, &mut |t| {
        print!("{t}");
        let _ = io::stdout().flush();
    })?;
    println!();
    stats(&last)
}

/// The speed line of a Chat's Done.
fn stats(last: &Last) -> io::Result<()> {
    let (_, b, _) = Reader::open(&last.bytes).map_err(bad)?;
    let (prefill, decode) = (Duration::from_nanos(b.u64(8).map_err(bad)?), Duration::from_nanos(b.u64(16).map_err(bad)?));
    let (n, count) = (b.u32(24).map_err(bad)?, b.u32(28).map_err(bad)?);
    let rate = |n: u32, t: Duration| f64::from(n) / t.as_secs_f64().max(1e-9);
    eprintln!(
        "{n} prompt tokens in {prefill:.2?} ({:.1} tok/s); {count} tokens in {decode:.2?} ({:.1} tok/s), by tachyon serve",
        rate(n, prefill),
        rate(count, decode)
    );
    Ok(())
}

/// The embeddings of `texts` by `model`, cut to `dims` values if asked: one JSON array per line.
pub fn embed(d: &mut Daemon, model: &str, texts: &[&str], dims: Option<usize>) -> io::Result<String> {
    let (id, room) = (d.id(), d.data_bytes() as u64);
    let mut w = request(2, id, Fields::default().u64(0).u64(room).u32(dims.unwrap_or(0) as u32));
    w.group(0, texts.len());
    for t in texts {
        w.text(t);
    }
    w.text(&name(model));
    let last = d.call(&w, &mut |_| {})?;
    let (_, b, _) = Reader::open(&last.bytes).map_err(bad)?;
    let (count, dims) = (b.u32(8).map_err(bad)? as usize, b.u32(12).map_err(bad)? as usize);
    let floats: Vec<f32> = d.get(0, 4 * count * dims).chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    Ok(rows(&floats.chunks(dims.max(1)).map(<[f32]>::to_vec).collect::<Vec<_>>()))
}

/// Answers the Laya request in file `request` with `model` (`laya` picks by the state's language), as one JSON
/// object, as `tachyon decide` does in-process.
pub fn decide(d: &mut Daemon, model: &str, request_file: &Path) -> io::Result<String> {
    let bytes = std::fs::read(request_file).map_err(|e| io::Error::other(format!("{}: {e}", request_file.display())))?;
    let r = Json::parse(&bytes).map_err(|e| io::Error::other(format!("{}: {e}", request_file.display())))?;
    let state = r.get("state").ok_or_else(|| io::Error::other("request: no \"state\""))?;
    let asked = questions(&r).map_err(|e| io::Error::other(e.to_string()))?;
    let (id, room) = (d.id(), d.data_bytes() as u64);
    let mut w = request(3, id, Fields::default().u64(0).u64(room).u8(u8::from(matches!(state, Json::Arr(_)))));
    asking(&mut w, &asked);
    w.text(&if model == "laya" { model.into() } else { name(model) }).text(&state.str().map_or_else(|| state.dumps(), str::to_string));
    let last = d.call(&w, &mut |_| {})?;
    Ok(render(&asked, &probabilities(d, &last).map_err(bad)?))
}

/// A Decide's questions group.
fn asking(w: &mut Writer, asked: &[Asked]) {
    w.group(1, asked.len());
    for a in asked {
        let kind = match a.kind {
            Kind::Choice => 0,
            Kind::Score => 1,
            Kind::Noul => 2,
        };
        w.bytes(&[kind]).group(0, a.options.len());
        for (label, about) in &a.options {
            w.text(label).text(about);
        }
        w.text(&a.instructions);
    }
}

/// A Decided's probabilities, question after question, from the data area.
fn probabilities(d: &Daemon, last: &Last) -> Result<Vec<Vec<f32>>, Bad> {
    let (_, _, mut r) = Reader::open(&last.bytes)?;
    let (len, n) = r.group()?;
    let counts: Vec<usize> = (0..n).map(|_| r.entry(len).and_then(|e| e.u32(0)).map(|c| c as usize)).collect::<Result<_, _>>()?;
    let floats: Vec<f32> = d.get(0, 4 * counts.iter().sum::<usize>()).chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let mut at = 0;
    Ok(counts
        .into_iter()
        .map(|c| {
            at += c;
            floats.get(at - c..at).unwrap_or_default().to_vec()
        })
        .collect())
}
