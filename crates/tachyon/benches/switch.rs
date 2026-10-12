//! Models on one GPU: the time to first token after a chat with another resident model, against after one with
//! the same; a model's return from the host-RAM tier, against pinned host → VRAM copies; what memory costs to take
//! and give back; whole switches between Gemma 4 E4B and E2B through the host tier and from disk; and unloading.
//! Checks the step-8 gate: switching inside the resident bundle adds ≤ 1 ms, and on native Linux a model returns
//! from the host tier at ≥ 90% of the rate of a copy into VRAM already in use (its whole return: VRAM taken, the
//! copy, its buffers laid out and its step recorded). Under WSL2 the return is printed against both copies but not
//! checked: WSL2 commits fresh VRAM on first write, which costs the RTX 3060 Laptop about a fifth of the copy.
//!
//! `cargo bench -p tachyon --features gpu --bench switch` uses E4B in `$TACHYON_MODEL` (default
//! `~/.local/share/tachyon/gemma-4-e4b`) and E2B in `$TACHYON_MODEL_SMALL` (default `…/gemma-4-e2b`), both made by
//! `tachyon convert`.

mod harness;

use harness::{Suite, cpu};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};
use tachyon::cuda::Context;
use tachyon::engine::{Engine, Limits, Load, Request, Tier};
use tachyon::model::{Decoder, Settings, footprint};
use tachyon::token::{Message, Role};
use tachyon::wpk::Wpk;

const RUNS: usize = 9;
const MESSAGES: [Message; 1] = [Message::new(Role::User, "Name three colors of the rainbow.")];

/// The model directory in `$var`, else `name` under `~/.local/share/tachyon`.
fn dir(var: &str, name: &str) -> PathBuf {
    std::env::var(var).map_or_else(|_| PathBuf::from(std::env::var("HOME").unwrap()).join(".local/share/tachyon").join(name), Into::into)
}

/// The time `engine.chat` takes to return model `name`'s first token of a short answer.
fn first(engine: &mut Engine, name: &str) -> Duration {
    let start = Instant::now();
    let answer = engine.chat(name, &Request { max_tokens: 1, ..Request::new(&MESSAGES) }).unwrap();
    let t = start.elapsed();
    drop(answer);
    t
}

/// An engine under `limits` with E4B and E2B loaded, kept forever, the last loaded E4B.
fn engine(dirs: &[PathBuf; 2], limits: Limits) -> Engine {
    let mut engine = Engine::new(0, limits).unwrap();
    let load = Load { keep_alive: Duration::MAX, ..Load::default() };
    engine.load("e2b", &dirs[1], &load).unwrap();
    engine.load("e4b", &dirs[0], &load).unwrap();
    engine
}

/// Milliseconds a switch inside the resident bundle adds to E4B's time to first token.
fn resident(suite: &mut Suite, dirs: &[PathBuf; 2]) -> f64 {
    let mut engine = engine(dirs, Limits::default());
    assert_eq!([engine.tier("e4b"), engine.tier("e2b")], [Some(Tier::Vram); 2], "both models fit in 70% of the VRAM");
    let same = suite.run("time to first token, E4B after E4B", RUNS, || first(&mut engine, "e4b"));
    let other = suite.run("time to first token, E4B after E2B", RUNS, || {
        first(&mut engine, "e2b");
        first(&mut engine, "e4b")
    });
    (other.as_secs_f64() - same.as_secs_f64()) * 1e3
}

/// E4B's return from the host tier, in GB/s of its weights: the copy, then its buffers laid out and its step recorded.
fn host_tier(ctx: &Context, suite: &mut Suite, dir: &Path) -> f64 {
    let (bytes, settings) = (Wpk::open(dir.join("weights.wpk")).unwrap().data_len() as f64, Settings::default());
    let mut parked = Some(Decoder::open(ctx, dir, &settings).unwrap().park().unwrap());
    let back = suite.run("E4B back from the host tier", RUNS, || {
        let mut dec = None;
        let t = cpu(|| dec = Some(parked.take().unwrap().resume(&settings).unwrap()));
        parked = dec.map(|d| d.park().unwrap());
        t
    });
    suite.rate(bytes / back.as_secs_f64() / 1e9, "GB/s");
    bytes / back.as_secs_f64() / 1e9
}

/// What a model's moves cost besides copies: pinned memory taken and given back, 1 GiB at a time.
fn memory(ctx: &Context, suite: &mut Suite) {
    let len = 1 << 30;
    suite.run("pinned alloc, 1 GiB", 3, || {
        let mut buf = None;
        let t = cpu(|| buf = Some(ctx.alloc_host(len).unwrap()));
        drop(buf);
        t
    });
    suite.run("pinned free, 1 GiB", 3, || {
        let buf = ctx.alloc_host(len).unwrap();
        cpu(|| drop(buf))
    });
}

/// What VRAM costs to take and give back, 1 GiB at a time: allocated plainly, as buffers are, and from the memory
/// pool, as weights are; and the pool given back to the system.
fn vram(ctx: &Context, suite: &mut Suite) {
    let len = 1 << 30;
    for (name, pooled) in [("VRAM", false), ("VRAM from the pool", true)] {
        let alloc = || if pooled { ctx.alloc_pooled(len) } else { ctx.alloc(len) }.unwrap();
        suite.run(&format!("{name}, alloc 1 GiB"), RUNS, || {
            let mut buf = None;
            let t = cpu(|| buf = Some(alloc()));
            drop(buf);
            t
        });
        suite.run(&format!("{name}, free 1 GiB"), RUNS, || {
            let buf = alloc();
            cpu(|| drop(buf))
        });
    }
    suite.run("VRAM given back from the pool, 1 GiB", RUNS, || {
        drop(ctx.alloc_pooled(len).unwrap());
        cpu(|| ctx.trim().unwrap())
    });
}

/// 2.5 GiB copied from pinned memory into VRAM allocated once (the ceiling), and into fresh VRAM, the memory pool
/// given back to the system and the buffer allocated just before (untimed); in GB/s. Fresh VRAM is slower: the
/// driver clears it first (the L40S copies into it at 70%), and WSL2 commits it on first write.
fn copies(ctx: &Context, suite: &mut Suite) -> [f64; 2] {
    const LEN: usize = 5 << 29;
    let (host, stream, mut dev) = (ctx.alloc_host(LEN).unwrap(), ctx.stream().unwrap(), None);
    [("pinned host → VRAM, 2.5 GiB, VRAM reused", true), ("pinned host → VRAM, 2.5 GiB, VRAM fresh", false)].map(|(name, reuse)| {
        let t = suite.run(name, RUNS, || {
            if !reuse || dev.is_none() {
                dev = None;
                ctx.trim().unwrap();
                dev = Some(ctx.alloc_pooled(LEN).unwrap());
            }
            let to = dev.as_ref().unwrap();
            // SAFETY: `host` outlives the copy, which `sync` completes.
            cpu(|| unsafe { stream.upload(to, 0, &host) }.and_then(|()| stream.sync()).unwrap())
        });
        suite.rate(LEN as f64 / 1e9 / t.as_secs_f64(), "GB/s");
        LEN as f64 / 1e9 / t.as_secs_f64()
    })
}

/// Whole switches when only one model fits: each chat moves the other model out (to the host tier, or with none to
/// disk) and brings its own in; and unloading E4B.
fn swaps(suite: &mut Suite, dirs: &[PathBuf; 2]) {
    let one = dirs.iter().map(|d| footprint(d, &Settings::default()).unwrap()).max();
    for (name, host) in [("switch E4B ↔ E2B through the host tier", None), ("switch E4B ↔ E2B from disk", Some(0))] {
        let mut engine = engine(dirs, Limits { vram: one, host });
        let mut turn = 0;
        suite.run(name, RUNS, || {
            turn += 1;
            first(&mut engine, ["e4b", "e2b"][turn % 2])
        });
    }
    let mut engine = engine(dirs, Limits::default());
    suite.run("unload E4B", RUNS, || {
        let start = Instant::now();
        engine.unload("e4b");
        let t = start.elapsed();
        engine.load("e4b", &dirs[0], &Load::default()).unwrap();
        t
    });
}

fn main() -> ExitCode {
    let dirs = [dir("TACHYON_MODEL", "gemma-4-e4b"), dir("TACHYON_MODEL_SMALL", "gemma-4-e2b")];
    let ctx = Context::new(0).unwrap();
    let mut suite = Suite::new("switch");
    let added = resident(&mut suite, &dirs);
    let back = host_tier(&ctx, &mut suite, &dirs[0]);
    memory(&ctx, &mut suite);
    vram(&ctx, &mut suite);
    let [ceiling, fresh] = copies(&ctx, &mut suite);
    swaps(&mut suite, &dirs);
    // Under WSL2 a returning model pays for committing fresh VRAM, which native Linux does at allocation: the gate is
    // checked on native Linux only (decision of 2026-10-12).
    let wsl = Path::new("/dev/dxg").exists();
    let passed = added <= 1.0 && (wsl || back >= 0.9 * ceiling);
    println!(
        "{} · switching adds {added:.2} ms (≤ 1 ms) · back from the host tier at {back:.1} GB/s, {:.0}% of a copy into reused VRAM ({ceiling:.1}), {:.0}% of one into fresh VRAM ({fresh:.1}); ≥ 90% of the first on native Linux: {}",
        ctx.info().name,
        100.0 * back / ceiling,
        100.0 * back / fresh,
        match (passed, wsl) {
            (false, _) => "FAIL",
            (true, true) => "PASS (the return not checked under WSL2)",
            (true, false) => "PASS",
        }
    );
    let code = suite.finish();
    if passed { code } else { ExitCode::FAILURE }
}
