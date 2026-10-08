//! `tachyon`, the command line of the Tachyon inference engine.
//!
//! Exit codes: 0 success, 2 bad usage, 3 no driver or no supported GPU.

use std::fmt::Write as _;
use std::process::ExitCode;
use tachyon::cuda::{self, DeviceInfo};

const HELP: &str = "tachyon: embedded inference on NVIDIA GPUs

Usage: tachyon <command> [--json]

Commands:
  gpu       List the GPUs the driver sees
  version   Print the version
  help      Print this help
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let json = args.iter().any(|a| a == "--json");
    match args.iter().map(String::as_str).find(|a| *a != "--json") {
        Some("gpu") => gpu(json),
        Some("version" | "--version" | "-V") => {
            println!("tachyon {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        None | Some("help" | "--help" | "-h") => {
            print!("{HELP}");
            ExitCode::SUCCESS
        }
        Some(other) => {
            eprint!("error: unknown command '{other}'\n\n{HELP}");
            ExitCode::from(2)
        }
    }
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
