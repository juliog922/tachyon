//! Model loading: disk → VRAM, disk → host-RAM tier, host tier → VRAM, and the page-cache path, on a synthetic
//! `.wpk` sized to the machine. Checks the step-2 exit gate: a staged disk → VRAM load reaches 90% of the slower of
//! the disk (read by the same loader into pinned memory) and PCIe (the host tier's upload).
//!
//! `cargo bench -p tachyon --features gpu --bench load`
//!
//! The file is written once to `$TACHYON_BENCH_DIR` (default `target/bench`), which should sit on the disk models
//! load from. The `perf` cycle counts show the CPU's share: with `O_DIRECT` it never touches the bytes.

mod harness;

use harness::{Suite, cpu};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use tachyon::cuda::Context;
use tachyon::wpk::{CHUNK, Dtype, Loader, Wpk, Writer};

/// Up to 4 GiB, at most half the free VRAM and a quarter of the available RAM, in whole chunks.
fn size(ctx: &Context) -> usize {
    let meminfo = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let available = meminfo.lines().find_map(|l| l.strip_prefix("MemAvailable:")?.trim().strip_suffix(" kB")?.parse::<usize>().ok()).unwrap_or(0) << 10;
    (4 << 30).min(ctx.memory().unwrap().0 / 2).min(available / 4) / CHUNK * CHUNK
}

/// The test file of `len` bytes of data, reused when a previous run left it.
fn file(len: usize) -> PathBuf {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(2).expect("the workspace root");
    let dir = std::env::var_os("TACHYON_BENCH_DIR").map_or_else(|| root.join("target/bench"), PathBuf::from);
    let path = dir.join(format!("load-{}MiB.wpk", len >> 20));
    if Wpk::open(&path).is_ok_and(|w| w.data_len() == len) {
        return path;
    }
    std::fs::create_dir_all(&dir).unwrap();
    let mut bytes: Vec<u8> = (0..CHUNK).map(|i| (i * 31 % 251) as u8).collect();
    let mut w = Writer::create(&path).unwrap();
    for i in 0..len / CHUNK {
        bytes[..8].copy_from_slice(&(i as u64).to_le_bytes());
        w.add(&format!("layer.{i}"), Dtype::U8, &[CHUNK as u64], &bytes).unwrap();
    }
    w.finish().unwrap();
    path
}

/// Rates in GB/s of: disk → host tier, staged disk → VRAM, host tier → VRAM.
fn measure(ctx: &Context, path: &Path, len: usize, suite: &mut Suite) -> [f64; 3] {
    let (direct, cached) = (Wpk::open(path).unwrap(), Wpk::open_with(path, false).unwrap());
    let uncached = if direct.is_direct() { "yes" } else { "refused by the filesystem" };
    println!("{} · {} MiB file at {} · O_DIRECT: {uncached}", ctx.info().name, len >> 20, path.display());
    let (mut loader, mut host, arena) = (Loader::new(ctx).unwrap(), ctx.alloc_host(len).unwrap(), ctx.alloc(len).unwrap());
    let rate = |suite: &mut Suite, t: std::time::Duration| {
        let gbs = len as f64 / t.as_secs_f64() / 1e9;
        suite.rate(gbs, "GB/s");
        gbs
    };
    let t = suite.run("disk → host tier (O_DIRECT)", 5, || cpu(|| loader.to_host(&direct, &mut host).unwrap()));
    let disk = rate(suite, t);
    let t = suite.run("disk → VRAM, staged (O_DIRECT)", 5, || cpu(|| loader.to_device(&direct, &arena).unwrap()));
    let staged = rate(suite, t);
    let t = suite.run("host tier → VRAM", 5, || cpu(|| loader.upload(&host, &arena).unwrap()));
    let pcie = rate(suite, t);
    let t = suite.run("disk → VRAM, page cache", 5, || cpu(|| loader.to_device(&cached, &arena).unwrap()));
    rate(suite, t);
    [disk, staged, pcie]
}

fn main() -> ExitCode {
    let ctx = Context::new(0).expect("GPU 0 with compute capability 8.0 or newer");
    let len = size(&ctx);
    let path = file(len);
    let mut suite = Suite::new("load");
    let [disk, staged, pcie] = measure(&ctx, &path, len, &mut suite);
    let ceiling = disk.min(pcie);
    let passed = staged >= 0.9 * ceiling;
    let verdict = if passed { "PASS" } else { "FAIL" };
    println!("gate: staged disk → VRAM reaches {:.0}% of min(disk {disk:.2}, PCIe {pcie:.2}) GB/s (≥ 90%): {verdict}", 100.0 * staged / ceiling);
    println!("cross-check the disk: fio --name=disk --filename={} --rw=read --bs=16M --iodepth=4 --ioengine=io_uring --direct=1", path.display());
    let code = suite.finish();
    if passed { code } else { ExitCode::FAILURE }
}