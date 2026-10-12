//! Images with Gemma 4 E4B: the vision tower alone on the fixture photo at each budget, and the engine's time to
//! the first token of an answer about it (the tower, the prompt's prefill with the soft tokens, the first step),
//! against the same question without the image. No gate yet: step 11b records the figures.
//!
//! `cargo bench -p tachyon --features gpu --bench vision` uses E4B in `$TACHYON_MODEL` (default
//! `~/.local/share/tachyon/gemma-4-e4b`), converted with its vision tower.

mod harness;

use harness::{Suite, cpu};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;
use tachyon::engine::{Engine, Limits, Load, Request};
use tachyon::media::{BUDGETS, Image};
use tachyon::token::{Message, Role};

const RUNS: usize = 9;
const QUESTION: &str = "Describe this image in one sentence.";

/// The fixture photo, 1200 × 900.
fn photo() -> Image {
    tachyon::media::decode(&std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/media/photo.jpg")).unwrap()).unwrap()
}

/// The tower alone, from the decoded photo to soft tokens in host memory (resize included), at each budget.
fn tower(suite: &mut Suite, dir: &std::path::Path, image: &Image) {
    let ctx = tachyon::cuda::Context::new(0).unwrap();
    let mut tower = tachyon::vision::Vision::open(&ctx, dir, BUDGETS[BUDGETS.len() - 1]).unwrap();
    for budget in BUDGETS {
        suite.run(&format!("tower, {} soft tokens", image.soft_tokens(budget).unwrap()), RUNS, || cpu(|| drop(tower.soft_tokens(image, budget).unwrap())));
    }
}

/// The engine's time to the first token of an answer about the photo at 70 and 280 soft tokens, and without it.
fn first_token(suite: &mut Suite, dir: &std::path::Path, image: &Image) {
    let mut engine = Engine::new(0, Limits::default()).unwrap();
    let images = [image.clone()];
    for (budget, shown) in [(70, &images[..]), (280, &images[..]), (280, &[][..])] {
        engine.load("e4b", dir, &Load { budget, ..Load::default() }).unwrap();
        let messages = [Message { images: shown, ..Message::new(Role::User, QUESTION) }];
        let name = if shown.is_empty() { "first token, no image".to_string() } else { format!("first token, image at {budget}") };
        suite.run(&name, RUNS, || {
            let start = Instant::now();
            let answer = engine.chat("e4b", &Request { max_tokens: 1, ..Request::new(&messages) }).unwrap();
            let t = start.elapsed();
            drop(answer);
            t
        });
    }
}

fn main() -> ExitCode {
    let dir =
        std::env::var("TACHYON_MODEL").map_or_else(|_| PathBuf::from(std::env::var("HOME").unwrap()).join(".local/share/tachyon/gemma-4-e4b"), Into::into);
    let (mut suite, image) = (Suite::new("vision"), photo());
    tower(&mut suite, &dir, &image);
    first_token(&mut suite, &dir, &image);
    suite.finish()
}
