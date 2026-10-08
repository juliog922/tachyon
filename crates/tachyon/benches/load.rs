//! Model loading: disk → VRAM, disk → host-RAM tier, host tier → VRAM, the page-cache path and the automatic choice,
//! on a synthetic `.wpk` sized to the machine. Checks the step-2 exit gate: a staged disk → VRAM load reaches 90% of
//! the slower of the disk (read by the same loader into pinned memory) and PCIe (the host tier's upload), and an
//! automatic load of a cached file reaches 90% of the page-cache path.
//!
//! `cargo bench -p tachyon --features gpu --bench load`
//!
//! The file is written once to `$TACHYON_BENCH_DIR` (default `target/bench`), which should sit on the disk models
//! load from. The `perf` cycle counts show the CPU's share: with `O_DIRECT` it never touches the bytes.

mod harness;

use harness::{Suite, cpu};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;
use tachyon::cuda::Context;
use tachyon::wpk::{CHUNK, Dtype, Loader, Reads, Wpk, Writer};

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

/// Shows `len` bytes per `t` as GB/s beside the last row, and returns it.
fn rate(suite: &mut Suite, len: usize, t: Duration) -> f64 {
    let gbs = len as f64 / t.as_secs_f64() / 1e9;
    suite.rate(gbs, "GB/s");
    gbs
}

/// Rates in GB/s of: disk → host tier, staged disk → VRAM, host tier → VRAM, all with `O_DIRECT`.
fn measure_disk(ctx: &Context, loader: &mut Loader, path: &Path, suite: &mut Suite) -> [f64; 3] {
    let direct = Wpk::open_with(path, Reads::Direct).unwrap();
    let len = direct.data_len();
    let uncached = if direct.reads() == Reads::Direct { "yes" } else { "refused by the filesystem" };
    println!("{} · {} MiB file at {} · O_DIRECT: {uncached}", ctx.info().name, len >> 20, path.display());
    let (mut host, arena) = (ctx.alloc_host(len).unwrap(), ctx.alloc(len).unwrap());
    let t = suite.run("disk → host tier (O_DIRECT)", 5, || cpu(|| loader.to_host(&direct, &mut host).unwrap()));
    let disk = rate(suite, len, t);
    let t = suite.run("disk → VRAM, staged (O_DIRECT)", 5, || cpu(|| loader.to_device(&direct, &arena).unwrap()));
    let staged = rate(suite, len, t);
    let t = suite.run("host tier → VRAM", 5, || cpu(|| loader.upload(&host, &arena).unwrap()));
    [disk, staged, rate(suite, len, t)]
}

/// Rates in GB/s of: page cache → VRAM, and the automatic choice → VRAM once the cache holds the file.
fn measure_cache(ctx: &Context, loader: &mut Loader, path: &Path, suite: &mut Suite) -> [f64; 2] {
    let (cached, auto) = (Wpk::open_with(path, Reads::Cached).unwrap(), Wpk::open(path).unwrap());
    let arena = ctx.alloc(cached.data_len()).unwrap();
    let t = suite.run("page cache → VRAM", 5, || cpu(|| loader.to_device(&cached, &arena).unwrap()));
    let warm = rate(suite, arena.len(), t);
    println!("page cache holds {:.0}% of the file before the automatic loads", 100.0 * auto.cached());
    let t = suite.run("automatic → VRAM", 5, || cpu(|| loader.to_device(&auto, &arena).unwrap()));
    [warm, rate(suite, arena.len(), t)]
}

/// The step-2 exit gate: staged loads keep up with the slower of disk and PCIe, and the automatic policy takes the
/// page cache when it holds the file.
fn gate([disk, staged, pcie, warm, auto]: [f64; 5]) -> bool {
    let verdict = |ok: bool| if ok { "PASS" } else { "FAIL" };
    let ceiling = disk.min(pcie);
    let (staged_ok, auto_ok) = (staged >= 0.9 * ceiling, auto >= 0.9 * warm);
    println!("gate: staged disk → VRAM reaches {:.0}% of min(disk {disk:.2}, PCIe {pcie:.2}) GB/s (≥ 90%): {}", 100.0 * staged / ceiling, verdict(staged_ok));
    println!("gate: automatic loads of a cached file reach {:.0}% of the page-cache path, {warm:.2} GB/s (≥ 90%): {}", 100.0 * auto / warm, verdict(auto_ok));
    staged_ok && auto_ok
}

fn main() -> ExitCode {
    let ctx = Context::new(0).expect("GPU 0 with compute capability 8.0 or newer");
    let len = size(&ctx);
    let path = file(len);
    let mut suite = Suite::new("load");
    let mut loader = Loader::new(&ctx).unwrap();
    let [disk, staged, pcie] = measure_disk(&ctx, &mut loader, &path, &mut suite);
    let [warm, auto] = measure_cache(&ctx, &mut loader, &path, &mut suite);
    let passed = gate([disk, staged, pcie, warm, auto]);
    println!("cross-check the disk: fio --name=disk --filename={} --rw=read --bs=16M --iodepth=4 --ioengine=io_uring --direct=1", path.display());
    let code = suite.finish();
    if passed { code } else { ExitCode::FAILURE }
}