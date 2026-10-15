//! Speech with Gemma 4 E4B: the audio tower alone on each fixture clip, then the engine's whole transcription of it
//! (the tower, the prompt's prefill with the soft tokens, every answer token), as time and as real-time factor: the
//! time over the clip's length. Checks the step-11 gate: a real-time factor of at most 0.05 on every clip.
//!
//! `cargo bench -p tachyon --features gpu --bench audio` uses E4B in `$TACHYON_MODEL` (default
//! `~/.local/share/tachyon/gemma-4-e4b`), converted with its audio tower.

mod harness;

use harness::{Suite, cpu};
use std::path::PathBuf;
use std::process::ExitCode;
use tachyon::engine::{Engine, Limits, Load, Request};
use tachyon::media::Audio;
use tachyon::token::{Message, Role};

const RUNS: usize = 9;
const CLIPS: [&str; 2] = ["clip-0.wav", "clip-1.wav"];
const GATE: f64 = 0.05;

/// A fixture clip.
fn clip(name: &str) -> Audio {
    tachyon::media::decode_wav(&std::fs::read(format!("{}/tests/fixtures/media/{name}", env!("CARGO_MANIFEST_DIR"))).unwrap()).unwrap()
}

fn main() -> ExitCode {
    let dir =
        std::env::var("TACHYON_MODEL").map_or_else(|_| PathBuf::from(std::env::var("HOME").unwrap()).join(".local/share/tachyon/gemma-4-e4b"), Into::into);
    let mut suite = Suite::new("audio");
    {
        let ctx = tachyon::cuda::Context::new(0).unwrap();
        let mut tower = tachyon::audio::Tower::open(&ctx, &dir).unwrap();
        for name in CLIPS {
            let audio = clip(name);
            suite.run(&format!("tower, {name}"), RUNS, || cpu(|| drop(tower.soft_tokens(&audio).unwrap())));
        }
    }
    let mut engine = Engine::new(0, Limits::default()).unwrap();
    engine.load("e4b", &dir, &Load::default()).unwrap();
    let mut worst: f64 = 0.0;
    for name in CLIPS {
        let clips = [clip(name)];
        let seconds = clips[0].samples.len() as f64 / f64::from(clips[0].rate);
        let messages = [Message { audio: &clips, ..Message::new(Role::User, "Transcribe this audio.") }];
        let took = suite.run(&format!("transcription, {name} ({seconds:.1} s)"), RUNS, || {
            cpu(|| {
                let mut answer = engine.chat("e4b", &Request::new(&messages)).unwrap();
                while answer.next_text().unwrap().is_some() {}
            })
        });
        let rtf = took.as_secs_f64() / seconds;
        suite.rate(rtf, "× real time");
        worst = worst.max(rtf);
    }
    println!("worst real-time factor {worst:.4} (≤ {GATE}): {}", if worst <= GATE { "PASS" } else { "FAIL" });
    let code = suite.finish();
    if worst <= GATE { code } else { ExitCode::FAILURE }
}
