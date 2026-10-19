//! `tachyon`, the command line of the Tachyon inference engine.
//!
//! Exit codes: 0 success, 1 failure, 2 bad usage, 3 no driver or no supported GPU.

use std::fmt::Write as _;
use std::process::ExitCode;
use tachyon::cuda::{self, DeviceInfo};
use tachyon::engine::{Engine, Limits, Load, Request};
use tachyon::token::{Message, Role};
use tachyon_cli::{encoders, models};

const HELP: &str = "tachyon: embedded inference on NVIDIA GPUs

Usage: tachyon <command> [--json]

Commands:
  catalog                      List the models Tachyon knows, and which are pulled
  recommend                    Name the model to pull for GPU 0
  pull <name> [--keep]         Download a model from Hugging Face and convert it (--keep: keep the checkpoint)
  list                         List the pulled models
  rm <name>                    Remove a pulled model
  run <model> <prompt>         Answer one prompt, greedily, and report the speed (a name or a directory;
                               --image=FILE, --audio=FILE, repeatable: PNG or JPEG files the prompt shows
                               first, WAV files it plays after)
  transcribe <model> <file>    Transcribe a WAV file's speech, up to 30 s
  decide <model> <request>     Answer a Laya request (a JSON file: a state and named questions); the model
                               `laya` picks laya-english for English text, else laya-multilingual
  embed <model> <text>...      Print each text's embedding, one JSON array per line (--dims=N: Matryoshka)
  serve                        Serve every process of this user from one engine, models kept in VRAM between
                               requests (interface/tachyon.h and tachyon.xml; --socket=PATH, --keep-alive=SECONDS,
                               --data-mib=N: each client's data area, 64 by default; --vram-mib=N: VRAM for
                               models, 70% of the GPU's by default; --host-mib=N: RAM for models moved out of
                               VRAM, half the RAM by default; env.example lists the environment variables)
  up [serve's options]         Start `tachyon serve` in the background (its log in ~/.local/state/tachyon)
  down                         Stop it: its models freed
  ps                           List its models: where they are, their VRAM, their time left in memory
  load <model>                 Load a model into it ahead of use (--keep-alive=SECONDS or never, --pin,
                               --context=N); while it is up, run, transcribe, embed and decide go through it
  unload <model>               Drop a model from it
  gpu                          List the GPUs the driver sees
  doctor                       Check the driver, GPUs, CPU, memory, certificates and the connection to Hugging Face
  convert <checkpoint> <model> Convert a Hugging Face checkpoint directory (Gemma 4, Laya, Granite) to a model
  version                      Print the version
  help                         Print this help
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (json, keep) = (args.iter().any(|a| a == "--json"), args.iter().any(|a| a == "--keep"));
    let words: Vec<&str> = args.iter().map(String::as_str).filter(|a| !["--json", "--keep"].contains(a)).collect();
    daemon_command(&words).or_else(|| models_command(&words, keep)).unwrap_or_else(|| command(&words, json))
}

/// Runs the command of `words` that drives the daemon, or goes through it while it is up, if it is one.
fn daemon_command(words: &[&str]) -> Option<ExitCode> {
    use tachyon_cli::serve::commands as daemon;
    let socket = tachyon_cli::serve::socket();
    Some(match words {
        ["up", rest @ ..] => options(rest).map_or(ExitCode::from(2), |opts| said(daemon::up(&opts))),
        ["down"] => said(daemon::down(&socket)),
        ["ps"] => said(up().and_then(|mut d| daemon::ps(&mut d))),
        ["load", model, rest @ ..] => match load_options(rest) {
            Some(how) => said(up().and_then(|mut d| daemon::load(&mut d, model, how.0, how.1))),
            None => ExitCode::from(2),
        },
        ["unload", model] => said(up().and_then(|mut d| daemon::unload(&mut d, model))),
        ["decide", _, request] if !std::path::Path::new(request).is_file() => said(Err(std::io::Error::other(format!("{request}: no such file")))),
        _ => return through(words, daemon::running()?),
    })
}

/// The commands that go through a running daemon `d`, if `words` is one.
fn through(words: &[&str], mut d: tachyon_cli::serve::client::Daemon) -> Option<ExitCode> {
    use tachyon_cli::serve::commands as daemon;
    Some(match words {
        ["run", model, rest @ ..] => {
            let (files, prompt): (Vec<&str>, Vec<&str>) = rest.iter().partition(|w| w.starts_with("--image=") || w.starts_with("--audio="));
            let [prompt] = prompt[..] else { return None };
            said(daemon::run(&mut d, model, prompt, &files).map(|()| String::new()))
        }
        ["transcribe", model, file] => said(daemon::run(&mut d, model, TRANSCRIBE, &[&format!("--audio={file}")]).map(|()| String::new())),
        ["decide", model, request] => said(daemon::decide(&mut d, model, request.as_ref())),
        ["embed", model, rest @ ..] if !rest.is_empty() => {
            let dims = rest.iter().find_map(|w| w.strip_prefix("--dims=")).map(str::parse).transpose().ok()?;
            let texts: Vec<&str> = rest.iter().copied().filter(|w| !w.starts_with("--dims=")).collect();
            said(daemon::embed(&mut d, model, &texts, dims))
        }
        _ => return None,
    })
}

/// The daemon on the default socket, or why there is none.
fn up() -> std::io::Result<tachyon_cli::serve::client::Daemon> {
    tachyon_cli::serve::commands::running().ok_or_else(|| std::io::Error::other("no tachyon serve is up; `tachyon up` starts one"))
}

/// What a daemon command printed, or its error.
fn said(r: std::io::Result<String>) -> ExitCode {
    match r {
        Ok(out) => {
            if !out.is_empty() {
                println!("{out}");
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// `load`'s options: the keep-alive in milliseconds (−1 the daemon's default, −2 never), the context and the pin.
fn load_options(rest: &[&str]) -> Option<(i64, (u32, bool))> {
    let (mut keep, mut context, mut pin) = (-1, 0, false);
    for w in rest {
        match w.split_once('=') {
            Some(("--keep-alive", "never")) => keep = -2,
            Some(("--keep-alive", s)) => keep = s.parse::<i64>().ok()? * 1000,
            Some(("--context", n)) => context = n.parse().ok()?,
            None if *w == "--pin" => pin = true,
            _ => {
                eprint!("error: unknown load option '{w}'\n\n{HELP}");
                return None;
            }
        }
    }
    Some((keep, (context, pin)))
}

/// Runs the model-management command of `words`, or `serve`, if it is one.
fn models_command(words: &[&str], keep: bool) -> Option<ExitCode> {
    Some(match words {
        ["catalog"] => {
            models::catalog();
            ExitCode::SUCCESS
        }
        ["recommend"] => recommend(),
        ["pull", name] => finished(models::pull(name, keep)),
        ["list"] => {
            models::list();
            ExitCode::SUCCESS
        }
        ["rm", name] => finished(models::rm(name)),
        ["doctor"] => ExitCode::from(u8::from(!models::doctor())),
        ["serve", rest @ ..] => serve(rest),
        _ => return None,
    })
}

/// Runs any other command of `words`.
fn command(words: &[&str], json: bool) -> ExitCode {
    match words {
        ["gpu"] => gpu(json),
        ["convert", from, to] => done(encoders::convert(from.as_ref(), to.as_ref())),
        ["decide", model, request] => printed(encoders::routed(model, request.as_ref()).and_then(|m| encoders::decide(&model_dir(&m), request.as_ref()))),
        ["embed", model, rest @ ..] if !rest.is_empty() => embed(model, rest),
        ["run", model, rest @ ..] => run_args(model, rest),
        ["transcribe", model, file] => done(run(&model_dir(model), TRANSCRIBE, &[&format!("--audio={file}")])),
        ["version" | "--version" | "-V"] => {
            println!("tachyon {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        [] | ["help" | "--help" | "-h"] => {
            print!("{HELP}");
            ExitCode::SUCCESS
        }
        [other, ..] => {
            eprint!("error: unknown command or arguments '{other}'\n\n{HELP}");
            ExitCode::from(2)
        }
    }
}

/// Runs the daemon with the options of `rest`.
fn serve(rest: &[&str]) -> ExitCode {
    options(rest).map_or(ExitCode::from(2), |opts| done(tachyon_cli::serve::serve(&opts)))
}

/// The daemon's options in `rest`, or `None` (reported) for one it does not take.
fn options(rest: &[&str]) -> Option<tachyon_cli::serve::Options> {
    let mut opts = match tachyon_cli::serve::Options::from_env() {
        Ok(opts) => opts,
        Err(e) => {
            eprintln!("error: {e}");
            return None;
        }
    };
    for w in rest {
        if !w.strip_prefix("--").and_then(|w| w.split_once('=')).is_some_and(|(k, v)| opts.set(k, v)) {
            eprint!("error: unknown serve option '{w}'\n\n{HELP}");
            return None;
        }
    }
    Some(opts)
}

/// Embeds the texts of `rest` with `model`; a `--dims=N` among them cuts the vectors to `N` values.
fn embed(model: &str, rest: &[&str]) -> ExitCode {
    let dims = rest.iter().find_map(|w| w.strip_prefix("--dims=")).map(str::parse::<usize>);
    let texts: Vec<&str> = rest.iter().copied().filter(|w| !w.starts_with("--dims=")).collect();
    let Ok(dims) = dims.transpose() else {
        eprintln!("error: --dims takes a number of values");
        return ExitCode::from(2);
    };
    printed(encoders::embed(&model_dir(model), &texts, dims))
}

/// The output printed and success, or the error printed and failure.
fn printed(r: tachyon::Result<String>) -> ExitCode {
    match r {
        Ok(out) => {
            println!("{out}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// A pulled model's directory when `name` is one, else `name` as a path.
fn model_dir(name: &str) -> std::path::PathBuf {
    let pulled = models::home().join(name);
    if !name.contains('/') && pulled.join("weights.wpk").exists() { pulled } else { name.into() }
}

/// The model to pull for GPU 0.
fn recommend() -> ExitCode {
    match cuda::device(0) {
        Ok(d) => {
            let pick = models::recommend(d.vram).map_or("none of the catalog fits", |m| m.name);
            println!("{pick}: for the {} with {:.1} GiB", d.name, d.vram as f64 / f64::from(1 << 30));
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(3)
        }
    }
}

/// Success, or the I/O error printed and failure.
fn finished(r: std::io::Result<()>) -> ExitCode {
    r.map_or_else(
        |e| {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        },
        |()| ExitCode::SUCCESS,
    )
}

/// Success, or the error printed and failure.
fn done(r: tachyon::Result<()>) -> ExitCode {
    r.map_or_else(
        |e| {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        },
        |()| ExitCode::SUCCESS,
    )
}

/// What `transcribe` asks.
const TRANSCRIBE: &str = "Transcribe the following speech segment in its language. Only output the transcription, with no newlines.";

/// `run` with `rest`: one prompt, and the media among `--image=FILE` and `--audio=FILE` words.
fn run_args(model: &str, rest: &[&str]) -> ExitCode {
    let (files, words): (Vec<&str>, Vec<&str>) = rest.iter().partition(|w| w.starts_with("--image=") || w.starts_with("--audio="));
    let [prompt] = words[..] else {
        eprint!("error: run takes one prompt\n\n{HELP}");
        return ExitCode::from(2);
    };
    done(run(&model_dir(model), prompt, &files))
}

/// Answers `prompt`, showing the images and playing the audio in `files` (`--image=FILE`, `--audio=FILE`), with the
/// model in `dir`, greedily, printing the text as it comes, then the speed.
fn run(dir: &std::path::Path, prompt: &str, files: &[&str]) -> tachyon::Result<()> {
    use std::io::Write as _;
    let read = |kind: &'static str| files.iter().filter_map(move |f| f.strip_prefix(kind)).map(std::fs::read);
    let images = read("--image=").map(|b| tachyon::media::decode(&b?)).collect::<tachyon::Result<Vec<_>>>()?;
    let audio = read("--audio=").map(|b| tachyon::media::decode_wav(&b?)).collect::<tachyon::Result<Vec<_>>>()?;
    let mut engine = Engine::new(0, Limits::default())?;
    engine.load("model", dir, &Load::default())?;
    let messages = [Message { images: &images, audio: &audio, ..Message::new(Role::User, prompt) }];
    let start = std::time::Instant::now();
    let mut answer = engine.chat("model", &Request::new(&messages))?;
    let (prefill, mid) = (start.elapsed(), std::time::Instant::now());
    while let Some(text) = answer.next_text()? {
        print!("{text}");
        let _ = std::io::stdout().flush();
    }
    let (decode, n, count) = (mid.elapsed(), answer.prompt(), answer.tokens());
    let rate = |n: usize, t: std::time::Duration| n as f64 / t.as_secs_f64();
    println!();
    eprintln!("{n} prompt tokens in {prefill:.2?} ({:.1} tok/s); {count} tokens in {decode:.2?} ({:.1} tok/s)", rate(n, prefill), rate(count, decode));
    Ok(())
}

fn gpu(json: bool) -> ExitCode {
    let found = cuda::driver_version().and_then(|v| Ok((v, cuda::devices()?)));
    let (version, devices) = match found {
        Ok(found) => found,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(3);
        }
    };
    print!("{}", if json { gpu_json(version, &devices) } else { gpu_text(version, &devices) });
    if devices.iter().any(supported) { ExitCode::SUCCESS } else { ExitCode::from(3) }
}

fn supported(d: &DeviceInfo) -> bool {
    d.compute >= (8, 0)
}

fn gpu_text(version: i32, devices: &[DeviceInfo]) -> String {
    let mut out = format!("driver supports CUDA {}.{}\n", version / 1000, version % 1000 / 10);
    for d in devices {
        let link = d.pcie().map_or("PCIe link not visible".into(), |l| format!("PCIe {} GT/s ×{} (now {} GT/s ×{})", l.max_speed, l.width, l.speed, l.width));
        let _ = writeln!(
            out,
            "{}  {}  sm_{}{}  {} SMs  {:.1} GiB  {:.0} GB/s datasheet  {link}  {}{}",
            d.ordinal,
            d.name,
            d.compute.0,
            d.compute.1,
            d.sms,
            d.vram as f64 / f64::from(1u32 << 30),
            d.nominal_bandwidth() / 1e9,
            d.pci,
            if supported(d) { "" } else { "  (unsupported: needs compute capability 8.0+)" }
        );
    }
    out
}

fn gpu_json(version: i32, devices: &[DeviceInfo]) -> String {
    let rows: Vec<String> = devices
        .iter()
        .map(|d| {
            let link = d.pcie().map_or("null".into(), |l| format!("{{\"gts\":{},\"width\":{},\"max_gts\":{},\"max_width\":{}}}", l.speed, l.width, l.max_speed, l.max_width));
            format!(
                "{{\"ordinal\":{},\"name\":\"{}\",\"compute\":\"{}.{}\",\"supported\":{},\"sms\":{},\"vram\":{},\"l2\":{},\"bandwidth\":{:.0},\"pci\":\"{}\",\"pcie\":{link}}}",
                d.ordinal,
                d.name.replace(['"', '\\'], ""),
                d.compute.0,
                d.compute.1,
                supported(d),
                d.sms,
                d.vram,
                d.l2,
                d.nominal_bandwidth(),
                d.pci
            )
        })
        .collect();
    format!("{{\"driver\":{version},\"devices\":[{}]}}\n", rows.join(","))
}
