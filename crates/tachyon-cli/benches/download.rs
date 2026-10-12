//! Downloads: SHA-256 and AES-128-GCM rates on this CPU; a range of a Hugging Face checkpoint fetched by `curl`,
//! by Tachyon over one connection, and by Tachyon over its default connections; and conversion time per model.
//! Checks the step-9 gate: over one connection, as `curl` uses, Tachyon downloads within 5% of `curl`'s rate; and
//! SHA-256 runs faster than the fastest download, so hashing never holds one back.
//!
//! `cargo bench -p tachyon-cli --bench download` fetches the first `$TACHYON_BENCH_BYTES` (256 MiB by default) of
//! Gemma 4 E2B's `model.safetensors`, or of `$TACHYON_BENCH_REPO`'s, into `target/bench/download`. Tachyon's copy
//! is checked against the SHA-256 of `curl`'s. Conversion is timed for each model whose checkpoint `tachyon pull
//! --keep` left in the cache. Without the network or `curl` the downloads are skipped and the gate is not checked.

#[path = "../../tachyon/benches/harness/mod.rs"]
mod harness;

use harness::{Suite, cpu};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use tachyon_cli::models::{CATALOG, cache};
use tachyon_cli::net::aes_gcm::{Gcm, TAG};
use tachyon_cli::net::crypto::{Sha256, sha_ni};
use tachyon_cli::net::hub::{Entry, Hub};

const RUNS: usize = 3;
/// The largest TLS record: what AES-GCM opens at a time during a download.
const RECORD: usize = 16 << 10;

/// Where the bench writes, under the workspace's `target`.
fn scratch(what: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(2).expect("the workspace root").join("target/bench").join(what)
}

/// SHA-256 over 256 MiB, in GB/s.
fn sha256(suite: &mut Suite) -> f64 {
    let data = vec![0x5a_u8; 256 << 20];
    let t = suite.run(if sha_ni() { "SHA-256, SHA-NI, 256 MiB" } else { "SHA-256, scalar, 256 MiB" }, 9, || cpu(|| _ = Sha256::digest(&data)));
    let rate = data.len() as f64 / t.as_secs_f64() / 1e9;
    suite.rate(rate, "GB/s");
    rate
}

/// AES-128-GCM opening 64 MiB of full TLS records, in GB/s.
fn aes_gcm(suite: &mut Suite) {
    let (gcm, nonce) = (Gcm::new(&[7; 16]), [9; 12]);
    let mut sealed = Vec::with_capacity(RECORD + TAG);
    gcm.seal(&nonce, b"aad", &[&[0x33; RECORD]], &mut sealed);
    let mut out = Vec::with_capacity(RECORD);
    let records = (64 << 20) / RECORD;
    let t = suite.run("AES-128-GCM open, 16 KiB records", 9, || {
        cpu(|| {
            for _ in 0..records {
                out.clear();
                assert!(gcm.open(&nonce, b"aad", &sealed, &mut out));
            }
        })
    });
    suite.rate((records * RECORD) as f64 / t.as_secs_f64() / 1e9, "GB/s");
}

/// What to download: the first `$TACHYON_BENCH_BYTES` of the repository's largest file, and its URL.
fn target(hub: &Hub, repo: &str) -> std::io::Result<(Entry, String)> {
    let bytes = std::env::var("TACHYON_BENCH_BYTES").ok().and_then(|b| b.parse().ok()).unwrap_or(256 << 20);
    let files = hub.files(repo)?;
    let mut file = files.into_iter().max_by_key(|f| f.size).ok_or_else(|| std::io::Error::other("an empty repository"))?;
    file.size = file.size.min(bytes);
    let base = std::env::var("TACHYON_HUB").unwrap_or_else(|_| "https://huggingface.co".into());
    let url = format!("{}/{repo}/resolve/main/{}", base.trim_end_matches('/'), file.path);
    Ok((file, url))
}

/// Runs download `f` of `bytes`, printing its time and rate as it ends: a slow link shows at once.
fn shown(label: &str, bytes: u64, f: impl FnOnce()) -> std::time::Duration {
    let t = cpu(f);
    println!("  {label}: {:.1} s, {:.0} MB/s", t.as_secs_f64(), bytes as f64 / t.as_secs_f64() / 1e6);
    t
}

/// `curl` fetching `file`'s range of `url` into `dir`, in MB/s; the SHA-256 of what it fetched.
fn curl(suite: &mut Suite, file: &Entry, url: &str, dir: &Path) -> Option<(f64, [u8; 32])> {
    let out = dir.join("curl.bin");
    let range = format!("0-{}", file.size - 1);
    let mut ok = true;
    let t = suite.run("curl, one connection", RUNS, || {
        shown("curl", file.size, || ok &= Command::new("curl").args(["-sSfL", "-r", &range, "-o"]).arg(&out).arg(url).status().is_ok_and(|s| s.success()))
    });
    let data = std::fs::read(&out).unwrap_or_default();
    if !ok || data.len() as u64 != file.size {
        println!("curl: failed or missing, the download comparison is skipped");
        return None;
    }
    suite.rate(file.size as f64 / t.as_secs_f64() / 1e6, "MB/s");
    Some((file.size as f64 / t.as_secs_f64() / 1e6, Sha256::digest(&data)))
}

/// Tachyon fetching `file` (its SHA-256 the one `curl` got) over `hub.connections`, in MB/s.
fn tachyon(suite: &mut Suite, hub: &Hub, repo: &str, file: &Entry) -> f64 {
    let dir = scratch("download/tachyon");
    let name = format!("tachyon, {} connection(s)", hub.connections);
    let t = suite.run(&name, RUNS, || {
        _ = std::fs::remove_file(dir.join(&file.path));
        shown(&name, file.size, || hub.download(repo, file, &dir, &|_, _| {}).expect("the range downloads and matches curl's"))
    });
    let rate = file.size as f64 / t.as_secs_f64() / 1e6;
    suite.rate(rate, "MB/s");
    _ = std::fs::remove_dir_all(dir);
    rate
}

/// The three downloads: `curl`'s rate, Tachyon's over one connection and over its default; `None` without them.
fn downloads(suite: &mut Suite) -> Option<[f64; 3]> {
    let repo = std::env::var("TACHYON_BENCH_REPO").unwrap_or_else(|_| "google/gemma-4-E2B-it".into());
    let mut hub = Hub::new();
    let default = hub.connections;
    let (file, url) = target(&hub, &repo).map_err(|e| println!("the Hub: {e}; downloads skipped")).ok()?;
    let dir = scratch("download");
    std::fs::create_dir_all(&dir).expect("a writable target directory");
    println!("{repo}/{}: the first {:.2} GB", file.path, file.size as f64 / 1e9);
    let (reference, sha) = curl(suite, &file, &url, &dir)?;
    let file = Entry { sha256: Some(sha), ..file };
    hub.connections = 1;
    let one = tachyon(suite, &hub, &repo, &file);
    hub.connections = default;
    let many = tachyon(suite, &hub, &repo, &file);
    Some([reference, one, many])
}

/// Bytes of the files in `dir`.
fn size(dir: &Path) -> u64 {
    std::fs::read_dir(dir).map_or(0, |d| d.filter_map(|e| e.ok()?.metadata().ok()).map(|m| m.len()).sum())
}

/// Conversion time of each model whose checkpoint is in the cache, and its rate in checkpoint bytes.
fn conversions(suite: &mut Suite) {
    for model in CATALOG.iter().filter(|m| m.later.is_none()) {
        let src = cache().join(model.repo);
        if !src.join("config.json").exists() {
            println!("{}: no checkpoint in {}; `tachyon pull {} --keep` leaves one", model.name, src.display(), model.name);
            continue;
        }
        let dst = scratch("convert").join(model.name);
        // Once to warm the page cache, once measured: a conversion takes minutes.
        let name = format!("convert {}", model.name);
        let t = suite.run(&name, 1, || {
            _ = std::fs::remove_dir_all(&dst);
            let t = cpu(|| tachyon::model::convert(&src, &dst).expect("a checkpoint that converts"));
            println!("  {name}: {:.1} s", t.as_secs_f64());
            t
        });
        suite.rate(size(&src) as f64 / t.as_secs_f64() / 1e6, "MB/s");
        _ = std::fs::remove_dir_all(&dst);
    }
}

fn main() -> ExitCode {
    let mut suite = Suite::new("download");
    let sha = sha256(&mut suite);
    aes_gcm(&mut suite);
    let measured = downloads(&mut suite);
    conversions(&mut suite);
    let passed = measured.is_none_or(|[reference, one, many]| one >= 0.95 * reference && sha * 1e3 > reference.max(one).max(many));
    match measured {
        Some([reference, one, many]) => println!(
            "curl {reference:.0} MB/s · tachyon {one:.0} MB/s over one connection ({:.0}%, ≥ 95%), {many:.0} MB/s over its default · SHA-256 {sha:.2} GB/s, above every download: {}",
            100.0 * one / reference,
            if passed { "PASS" } else { "FAIL" }
        ),
        None => println!("SHA-256 {sha:.2} GB/s · downloads not measured, the gate not checked"),
    }
    let code = suite.finish();
    if passed { code } else { ExitCode::FAILURE }
}
