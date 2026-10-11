//! The chat API on the GPU: answers stream and repeat, cancel cleanly, and a step allocates nothing; with the `models`
//! feature, Gemma 4 E4B's 4-bit answers against Hugging Face's in bfloat16.
#![cfg(feature = "gpu")]

mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use tachyon::engine::{Engine, Request};
use tachyon::model::Settings;
use tachyon::token::{Message, Role};

thread_local! {
    /// Heap allocations made on this thread.
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

/// The system allocator, counting allocations per thread.
struct Counting;

// SAFETY: every call is passed to the system allocator unchanged.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|n| n.set(n.get() + 1));
        // SAFETY: as the caller's contract.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: as the caller's contract.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|n| n.set(n.get() + 1));
        // SAFETY: as the caller's contract.
        unsafe { System.realloc(ptr, layout, size) }
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

/// The answer to `messages`, at most `max` tokens of it: its text and token count.
fn answer(engine: &mut Engine, messages: &[Message], max: usize) -> (String, usize) {
    let mut stream = engine.chat(&Request { max_tokens: max, ..Request::new(messages) }).unwrap();
    let mut text = String::new();
    while let Some(piece) = stream.next_text().unwrap() {
        text.push_str(piece);
    }
    (text, stream.tokens())
}

#[test]
fn tiny_model_streams_cancels_and_does_not_allocate() {
    let src = common::tiny::checkpoint("engine", 3);
    tachyon::model::convert(&src, &src.join("out")).unwrap();
    let mut engine = Engine::open(&src.join("out"), &Settings { context: 64 }).unwrap();
    let messages = [Message { role: Role::System, text: "Be brief." }, Message { role: Role::User, text: "hello there, how are you" }];
    let (full, tokens) = answer(&mut engine, &messages, 30);
    assert!(tokens > 0 && tokens <= 30, "{tokens} tokens");
    assert_eq!(answer(&mut engine, &messages, 30), (full.clone(), tokens), "a second chat repeats the first");
    // Cancelled after a few pieces, by dropping the stream, the engine answers the next chat as before.
    let mut stream = engine.chat(&Request::new(&messages)).unwrap();
    for _ in 0..3 {
        stream.next_text().unwrap();
    }
    drop(stream);
    assert_eq!(answer(&mut engine, &messages, 30).0, full, "the chat after a cancelled one");
    // After two pieces, every step to the end of the context allocates nothing.
    let mut stream = engine.chat(&Request::new(&messages)).unwrap();
    let prompt = stream.prompt();
    stream.next_text().unwrap();
    stream.next_text().unwrap();
    let before = ALLOCATIONS.with(Cell::get);
    while stream.next_text().unwrap().is_some() {}
    assert_eq!(ALLOCATIONS.with(Cell::get) - before, 0, "allocations while generating");
    assert_eq!(stream.tokens(), 64 - prompt, "the answer fills the context");
    drop(stream);
    let long = "hello ".repeat(80);
    let err = engine.chat(&Request::new(&[Message { role: Role::User, text: &long }])).err();
    assert!(matches!(err, Some(tachyon::Error::TooLong { context: 64, .. })), "{err:?}");
    std::fs::remove_dir_all(src).unwrap();
}

/// Gemma 4 E4B (in `$TACHYON_MODEL`, by default `~/.local/share/tachyon/gemma-4-e4b`) against the greedy answers of
/// Hugging Face's model in bfloat16 (`scripts/reference.py`): fed each prompt and each reference answer token by
/// token, its pick after every token agrees with the reference's on ≥ 99% of the positions.
#[cfg(feature = "models")]
#[test]
fn gemma_agrees_with_hugging_face() {
    use tachyon::json::Json;
    use tachyon::token::Chat;
    let dir = std::env::var("TACHYON_MODEL")
        .map_or_else(|_| std::path::PathBuf::from(std::env::var("HOME").unwrap()).join(".local/share/tachyon/gemma-4-e4b"), Into::into);
    let tok = tachyon::token::Tokenizer::from_bytes(&std::fs::read(dir.join("tokenizer.tok")).unwrap()).unwrap();
    let chat = Chat::gemma4(&tok).unwrap();
    let ctx = tachyon::cuda::Context::new(0).unwrap();
    let mut dec = tachyon::model::Decoder::open(&ctx, &dir, &Settings::default()).unwrap();
    let fixture = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/gemma-4-e4b.ref")).unwrap();
    let nums = |j: &Json, k: &str| j.get(k).and_then(Json::arr).unwrap().iter().map(|v| v.num().unwrap()).collect::<Vec<f64>>();
    let ids = |j: &Json, k: &str| nums(j, k).into_iter().map(|v| v as u32).collect::<Vec<u32>>();
    let (mut agree, mut total) = (0, 0);
    for line in fixture.lines() {
        let j = Json::parse(line.as_bytes()).unwrap();
        let text = j.get("text").and_then(Json::str).unwrap();
        let (prompt, answer, gaps) = (ids(&j, "prompt"), ids(&j, "answer"), nums(&j, "gap"));
        let mut mine = Vec::new();
        chat.encode(&tok, &[Message { role: Role::User, text }], false, &mut mine);
        assert_eq!(mine, prompt, "the chat template's tokens");
        dec.reset(&tachyon::ptx::Sampling::default()).unwrap();
        let mut pick = mine.iter().map(|&t| dec.step(t).unwrap()).last().unwrap();
        for (k, (&want, gap)) in answer.iter().zip(&gaps).enumerate() {
            if pick == want {
                agree += 1;
            } else {
                println!("{text:?}, answer token {k}: {pick} against {want} (reference gap {gap:.2})");
            }
            total += 1;
            pick = dec.step(want).unwrap();
        }
    }
    println!("top-1 agreement {agree}/{total} ({:.2}%)", 100.0 * f64::from(agree) / f64::from(total));
    assert!(f64::from(agree) >= 0.99 * f64::from(total));
}