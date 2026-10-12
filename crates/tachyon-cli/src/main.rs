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
  run <model> <prompt>         Answer one prompt, greedily, and report the speed (a name or a directory)
  decide <model> <request>     Answer a Laya request (a JSON file: a state and named questions)
  embed <model> <text>...      Print each text's embedding, one JSON array per line (--dims=N: Matryoshka)
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
    models_command(&words, keep).unwrap_or_else(|| command(&words, json))
}

/// Runs the model-management command of `words`, if it is one.
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
        _ => return None,
    })
}

/// Runs any other command of `words`.
fn command(words: &[&str], json: bool) -> ExitCode {
    match words {
        ["gpu"] => gpu(json),
        ["convert", from, to] => done(encoders::convert(from.as_ref(), to.as_ref())),
        ["decide", model, request] => printed(encoders::decide(&model_dir(model), request.as_ref())),
        ["embed", model, rest @ ..] if !rest.is_empty() => embed(model, rest),
        ["run", model, prompt] => done(run(&model_dir(model), prompt)),
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

/// Answers `prompt` with the model in `dir`, greedily, printing the text as it comes, then the speed.
fn run(dir: &std::path::Path, prompt: &str) -> tachyon::Result<()> {
    use std::io::Write as _;
    let mut engine = Engine::new(0, Limits::default())?;
    engine.load("model", dir, &Load::default())?;
    let messages = [Message { role: Role::User, text: prompt }];
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
