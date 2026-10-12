//! `tachyon`, the command line of the Tachyon inference engine.
//!
//! Exit codes: 0 success, 1 failure, 2 bad usage, 3 no driver or no supported GPU.

use std::fmt::Write as _;
use std::process::ExitCode;
use tachyon::cuda::{self, DeviceInfo};
use tachyon::engine::{Engine, Limits, Load, Request};
use tachyon::token::{Message, Role};

const HELP: &str = "tachyon: embedded inference on NVIDIA GPUs

Usage: tachyon <command> [--json]

Commands:
  gpu                          List the GPUs the driver sees
  convert <checkpoint> <model> Convert a Hugging Face Gemma 4 checkpoint directory to a model directory
  run <model> <prompt>         Answer one prompt, greedily, and report the speed
  version                      Print the version
  help                         Print this help
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let json = args.iter().any(|a| a == "--json");
    let words: Vec<&str> = args.iter().map(String::as_str).filter(|a| *a != "--json").collect();
    match words.first().copied() {
        Some("gpu") => gpu(json),
        Some("convert") if words.len() == 3 => done(tachyon::model::convert(words[1].as_ref(), words[2].as_ref())),
        Some("run") if words.len() == 3 => done(run(words[1].as_ref(), words[2])),
        Some("version" | "--version" | "-V") => {
            println!("tachyon {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        None | Some("help" | "--help" | "-h") => {
            print!("{HELP}");
            ExitCode::SUCCESS
        }
        Some(other) => {
            eprint!("error: unknown command or arguments '{other}'\n\n{HELP}");
            ExitCode::from(2)
        }
    }
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
