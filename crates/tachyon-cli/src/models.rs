//! The model catalog and the commands that manage models: `catalog`, `recommend`, `pull`, `list`, `rm`, and
//! `doctor`, which checks what the others need.
//!
//! Models live in `$TACHYON_HOME`, by default `$XDG_DATA_HOME/tachyon` (`~/.local/share/tachyon`), a directory each.
//! `pull` downloads a checkpoint into `$XDG_CACHE_HOME/tachyon` (`~/.cache/tachyon`), where an interrupted pull
//! resumes, converts it, and removes it unless asked to keep it.

use crate::net::hub::{Entry, Hub};
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// A model of the catalog: its name, Hugging Face repository, what it is for and is, the VRAM it takes with the
/// default settings (bytes), and the development step that brings it when this build cannot run it yet.
pub struct Model {
    /// The name `pull` and `run` take.
    pub name: &'static str,
    /// The Hugging Face repository of its checkpoint.
    pub repo: &'static str,
    /// The command that uses it: `run`, `decide` or `embed`.
    pub task: &'static str,
    /// One line on what it is.
    pub about: &'static str,
    /// VRAM it takes with the default settings, in bytes.
    pub vram: u64,
    /// The development step that brings what it needs, when this build cannot run it.
    pub later: Option<u8>,
}

/// Every model the catalog knows: the encoders, then the chat models, smallest first.
pub const CATALOG: [Model; 10] = [
    Model {
        name: "granite-embedding-97m",
        repo: "ibm-granite/granite-embedding-97m-multilingual-r2",
        task: "embed",
        about: "Granite Embedding 97M R2: embeddings in 200+ languages, 384 dims, fastest",
        vram: 150 << 20,
        later: None,
    },
    Model {
        name: "granite-embedding-311m",
        repo: "ibm-granite/granite-embedding-311m-multilingual-r2",
        task: "embed",
        about: "Granite Embedding 311M R2: embeddings in 200+ languages, 768 dims",
        vram: 300 << 20,
        later: None,
    },
    Model {
        name: "laya-multilingual",
        repo: "convaiinnovations/laya-multilingual",
        task: "decide",
        about: "Laya multilingual: typed decisions in one pass, 100+ languages",
        vram: 350 << 20,
        later: None,
    },
    Model {
        name: "laya-english",
        repo: "convaiinnovations/laya",
        task: "decide",
        about: "Laya English: typed decisions in one pass, the most accurate in English",
        vram: 900 << 20,
        later: None,
    },
    Model {
        name: "laya-typed-decisions",
        repo: "convaiinnovations/laya-typed-decisions",
        task: "decide",
        about: "Laya typed-decisions: tuned for support, invoices, security and agent triage",
        vram: 950 << 20,
        later: None,
    },
    Model {
        name: "gemma-4-e2b",
        repo: "google/gemma-4-E2B-it",
        task: "run",
        about: "Gemma 4 E2B: chat, fastest on small GPUs",
        vram: 1_700 << 20,
        later: None,
    },
    Model {
        name: "gemma-4-e4b",
        repo: "google/gemma-4-E4B-it",
        task: "run",
        about: "Gemma 4 E4B: chat, the default on 6–8 GB",
        vram: 2_900 << 20,
        later: None,
    },
    Model {
        name: "gemma-4-12b",
        repo: "google/gemma-4-12B-it",
        task: "run",
        about: "Gemma 4 12B: chat, text only for now, the default on 12–16 GB",
        vram: 8 << 30,
        later: None,
    },
    Model {
        name: "gemma-4-26b-a4b",
        repo: "google/gemma-4-26B-A4B-it",
        task: "run",
        about: "Gemma 4 26B A4B: mixture of experts, text only for now, the default on 24 GB",
        vram: 15 << 30,
        later: None,
    },
    Model {
        name: "gemma-4-31b",
        repo: "google/gemma-4-31B-it",
        task: "run",
        about: "Gemma 4 31B: the highest quality, text only for now, 32 GB and up",
        vram: 19 << 30,
        later: None,
    },
];

/// `name` in the catalog.
fn find(name: &str) -> io::Result<&'static Model> {
    CATALOG.iter().find(|m| m.name == name).ok_or_else(|| io::Error::other(format!("{name} is not in the catalog; `tachyon catalog` lists it")))
}

/// Where models live.
pub fn home() -> PathBuf {
    let env = |k| std::env::var_os(k).map(PathBuf::from);
    env("TACHYON_HOME").unwrap_or_else(|| env("XDG_DATA_HOME").unwrap_or_else(|| env("HOME").unwrap_or_default().join(".local/share")).join("tachyon"))
}

/// Where downloads wait to be converted: `$XDG_CACHE_HOME/tachyon`, a directory per repository.
pub fn cache() -> PathBuf {
    let env = |k| std::env::var_os(k).map(PathBuf::from);
    env("XDG_CACHE_HOME").unwrap_or_else(|| env("HOME").unwrap_or_default().join(".cache")).join("tachyon")
}

/// Prints the catalog, marking what is pulled.
pub fn catalog() {
    for m in &CATALOG {
        let state = match (m.later, home().join(m.name).join("weights.wpk").exists()) {
            (Some(step), _) => format!("arrives in step {step}"),
            (None, true) => "pulled".into(),
            (None, false) => String::new(),
        };
        println!("{:<23} {:<6} {:>5.1} GB  {:<66} {state}", m.name, m.task, m.vram as f64 / 1e9, m.about);
    }
}

/// The model to pull for a GPU of `vram` bytes: the largest this build runs that fits in 70% of it.
pub fn recommend(vram: u64) -> Option<&'static Model> {
    CATALOG.iter().rev().find(|m| m.task == "run" && m.later.is_none() && m.vram <= vram / 10 * 7)
}

/// The checkpoint files conversion reads: at the top, the configurations, the tokenizer and the weights; a Laya
/// repository's encoder configuration and tokenizer in their folders.
fn wanted(f: &Entry) -> bool {
    let p = f.path.as_str();
    let known = ["config.json", "tokenizer.json", "rl_agent_config.json", "encoder/config.json", "tokenizer/tokenizer.json"];
    known.contains(&p) || !p.contains('/') && (p.ends_with(".safetensors") || p.ends_with(".safetensors.index.json"))
}

/// Downloads model `name`'s checkpoint, converts it into the models directory, and removes the download unless
/// `keep`.
pub fn pull(name: &str, keep: bool) -> io::Result<()> {
    let model = find(name)?;
    if let Some(step) = model.later {
        return Err(io::Error::other(format!("{name} needs what step {step} of Tachyon brings; it cannot run yet")));
    }
    let (hub, dir, start) = (Hub::new(), cache().join(model.repo), Instant::now());
    let files: Vec<Entry> = hub.files(model.repo)?.into_iter().filter(wanted).collect();
    let total: u64 = files.iter().map(|f| f.size).sum();
    eprintln!("{name}: {} files, {:.2} GB from {}", files.len(), total as f64 / 1e9, model.repo);
    for f in &files {
        hub.download(model.repo, f, &dir, &|done, size| eprint!("\r  {:<40} {:>6.1}%", f.path, 100.0 * done as f64 / size.max(1) as f64))?;
        eprintln!();
    }
    let seconds = start.elapsed().as_secs_f64();
    eprintln!("downloaded in {seconds:.1} s ({:.0} MB/s); converting", total as f64 / seconds / 1e6);
    crate::encoders::convert(&dir, &home().join(name)).map_err(|e| io::Error::other(e.to_string()))?;
    if !keep {
        std::fs::remove_dir_all(&dir)?;
    }
    eprintln!("{name} is ready in {}", home().join(name).display());
    Ok(())
}

/// Bytes the files directly in `dir` take.
fn size(dir: &Path) -> u64 {
    std::fs::read_dir(dir).map_or(0, |d| d.filter_map(|e| e.ok()?.metadata().ok()).map(|m| m.len()).sum())
}

/// Prints the models in the models directory and their size.
pub fn list() {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(home()).map_or_else(|_| Vec::new(), |d| d.filter_map(|e| Some(e.ok()?.path())).collect());
    dirs.retain(|d| d.join("weights.wpk").exists());
    dirs.sort();
    for d in dirs {
        println!("{:<20} {:>6.2} GB  {}", d.file_name().unwrap_or_default().to_string_lossy(), size(&d) as f64 / 1e9, d.display());
    }
}

/// Removes model `name` from the models directory.
pub fn rm(name: &str) -> io::Result<()> {
    let dir = home().join(name);
    if name.contains('/') || !dir.join("weights.wpk").exists() {
        return Err(io::Error::other(format!("no model named {name} in {}", home().display())));
    }
    std::fs::remove_dir_all(dir)
}

/// Checks what Tachyon needs and prints each finding: the driver and GPUs, the CPU instructions downloads use,
/// memory, the models directory, the trust anchors, and a TLS connection to the Hub. False if something blocks.
pub fn doctor() -> bool {
    let mut ok = true;
    match tachyon::cuda::driver_version().and_then(|v| Ok((v, tachyon::cuda::devices()?))) {
        Ok((v, devices)) => {
            println!("driver      CUDA {}.{}", v / 1000, v % 1000 / 10);
            for d in devices {
                let link =
                    d.pcie().map_or("link not visible".into(), |l| format!("PCIe {} GT/s ×{} (max {} GT/s ×{})", l.speed, l.width, l.max_speed, l.max_width));
                let pick = recommend(d.vram).map_or("none fits", |m| m.name);
                println!(
                    "gpu {}       {} sm_{}{}, {:.1} GiB, {link}; recommended: {pick}",
                    d.ordinal,
                    d.name,
                    d.compute.0,
                    d.compute.1,
                    d.vram as f64 / f64::from(1 << 30)
                );
            }
        }
        Err(e) => {
            println!("driver      {e}");
            ok = false;
        }
    }
    let (aes, sha) = (crate::net::aes_gcm::supported(), crate::net::crypto::sha_ni());
    println!("cpu         AES-NI and PCLMULQDQ: {}; SHA-NI: {}", yes(aes), yes(sha));
    ok &= aes;
    let info = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let kb = |key: &str| info.lines().find_map(|l| l.strip_prefix(key)?.trim().strip_suffix(" kB")?.parse::<f64>().ok()).unwrap_or(0.0);
    println!("memory      {:.1} GiB, {:.1} GiB available", kb("MemTotal:") / 1_048_576.0, kb("MemAvailable:") / 1_048_576.0);
    println!("models      {}", home().display());
    ok &= network();
    ok
}

/// The trust anchors, and a TLS connection to the Hub; false if either is missing.
fn network() -> bool {
    let anchors = crate::net::x509::system_anchors().len();
    println!("trust       {anchors} certificates from the system bundle");
    let start = Instant::now();
    let url = crate::net::http::Url::parse("https://huggingface.co/").expect("a URL");
    let config = std::sync::Arc::new(crate::net::tls::ClientConfig::system());
    match crate::net::http::Connection::open(&url, &config) {
        Ok(_) => println!("network     TLS 1.3 to huggingface.co in {:.0} ms", start.elapsed().as_secs_f64() * 1e3),
        Err(e) => println!("network     huggingface.co: {e}"),
    }
    anchors > 0
}

fn yes(b: bool) -> &'static str {
    if b { "yes" } else { "no" }
}
