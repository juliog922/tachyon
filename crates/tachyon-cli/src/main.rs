//! `tachyon`, the command line of the Tachyon inference engine.
//!
//! Exit codes: 0 success, 1 failure, 2 bad usage, 3 no driver or no supported GPU.

use std::fmt::Write as _;
use std::process::ExitCode;
use tachyon::cuda::{self, DeviceInfo};
use tachyon::model::Decoder;
use tachyon::token::{Chat, Detokenizer, Message, Role, Tokenizer};

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

/// Answers `prompt` with the model in `dir`: the prompt through the chat template, then greedy tokens to the end of
/// the turn (at most 512), printed as they come.
fn run(dir: &std::path::Path, prompt: &str) -> tachyon::Result<()> {
    let tok = Tokenizer::from_bytes(&std::fs::read(dir.join("tokenizer.tok"))?)?;
    let chat = Chat::gemma4(&tok)?;
    let mut ids = Vec::new();
    chat.encode(&tok, &[Message { role: Role::User, text: prompt }], false, &mut ids);
    let ctx = cuda::Context::new(0)?;
    let mut dec = Decoder::open(&ctx, dir, 4096)?;
    let start = std::time::Instant::now();
    let first = ids.iter().map(|&t| dec.step(t)).last().unwrap_or(Ok(0))?;
    let (prefill, mid) = (start.elapsed(), std::time::Instant::now());
    let count = generate(&mut dec, &tok, first, chat.end_of_turn())?;
    let (decode, rate) = (mid.elapsed(), |n: f64, t: std::time::Duration| n / t.as_secs_f64());
    let n = ids.len();
    eprintln!(
        "\n{n} prompt tokens in {prefill:.2?} ({:.1} tok/s); {count} tokens in {decode:.2?} ({:.1} tok/s)",
        rate(n as f64, prefill),
        rate(count as f64, decode)
    );
    Ok(())
}

/// Prints the text of `first` and the tokens after it until `end` or 512 tokens; returns how many it printed.
fn generate(dec: &mut Decoder, tok: &Tokenizer, first: u32, end: u32) -> tachyon::Result<usize> {
    use std::io::Write as _;
    let (mut next, mut text, mut detok, mut count) = (first, String::new(), Detokenizer::default(), 0);
    while next != end && next != 1 && count < 512 {
        detok.push(tok, next, &mut text);
        print!("{text}");
        std::io::stdout().flush()?;
        text.clear();
        (next, count) = (dec.step(next)?, count + 1);
    }
    detok.finish(&mut text);
    println!("{text}");
    Ok(count)
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