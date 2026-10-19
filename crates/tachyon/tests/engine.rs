//! The chat API on the GPU: answers stream and repeat, cancel cleanly, and a step allocates nothing; models move
//! between VRAM, the host tier and disk as the limits, pins and keep-alives say, and answer the same wherever they
//! come back from; with the `models` feature, Gemma 4 E4B's 4-bit answers against Hugging Face's in bfloat16.
#![cfg(feature = "gpu")]

mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::time::Duration;
use tachyon::engine::{Engine, Limits, Load, Request, Tier};
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

/// Model `name`'s answer to `messages`, at most `max` tokens of it, kept `keep` after: its text and token count.
fn answer(engine: &mut Engine, (name, keep): (&str, Option<Duration>), messages: &[Message], max: usize) -> (String, usize) {
    let mut stream = engine.chat(name, &Request { max_tokens: max, keep_alive: keep, ..Request::new(messages) }).unwrap();
    let mut text = String::new();
    while let Some(piece) = stream.next_text().unwrap() {
        text.push_str(piece);
    }
    (text, stream.tokens())
}

#[test]
fn tiny_model_streams_cancels_and_does_not_allocate() {
    let src = common::tiny::checkpoint("engine", 3, common::tiny::E4B);
    tachyon::model::convert(&src, &src.join("out")).unwrap();
    let mut engine = Engine::new(0, Limits::default()).unwrap();
    engine.load("tiny", &src.join("out"), &Load { settings: Settings { context: 64, chunk: 8 }, ..Load::default() }).unwrap();
    let tiny = ("tiny", None);
    let messages = [Message::new(Role::System, "Be brief."), Message::new(Role::User, "hello there, how are you")];
    let (full, tokens) = answer(&mut engine, tiny, &messages, 30);
    assert!(tokens > 0 && tokens <= 30, "{tokens} tokens");
    assert_eq!(answer(&mut engine, tiny, &messages, 30), (full.clone(), tokens), "a second chat repeats the first");
    // Cancelled after a few pieces, by dropping the stream, the engine answers the next chat as before.
    let mut stream = engine.chat("tiny", &Request::new(&messages)).unwrap();
    for _ in 0..3 {
        stream.next_text().unwrap();
    }
    drop(stream);
    assert_eq!(answer(&mut engine, tiny, &messages, 30).0, full, "the chat after a cancelled one");
    // After two pieces, every step to the end of the context allocates nothing.
    let mut stream = engine.chat("tiny", &Request::new(&messages)).unwrap();
    let prompt = stream.prompt();
    stream.next_text().unwrap();
    stream.next_text().unwrap();
    let before = ALLOCATIONS.with(Cell::get);
    while stream.next_text().unwrap().is_some() {}
    assert_eq!(ALLOCATIONS.with(Cell::get) - before, 0, "allocations while generating");
    assert_eq!(stream.tokens(), 64 - prompt, "the answer fills the context");
    drop(stream);
    let long = "hello ".repeat(80);
    let err = engine.chat("tiny", &Request::new(&[Message::new(Role::User, &long)])).err();
    assert!(matches!(err, Some(tachyon::Error::TooLong { context: 64, .. })), "{err:?}");
    std::fs::remove_dir_all(src).unwrap();
}

/// Where models `names` are.
fn tiers<const N: usize>(engine: &Engine, names: [&str; N]) -> [Option<Tier>; N] {
    names.map(|n| engine.tier(n))
}

#[test]
fn models_move_between_tiers_and_answer_the_same() {
    let src = common::tiny::checkpoint("tiers", 3, common::tiny::E4B);
    let dir = src.join("out");
    tachyon::model::convert(&src, &dir).unwrap();
    let settings = Settings { context: 64, chunk: 8 };
    let need = tachyon::model::footprint(&dir, &settings).unwrap();
    let (keep, pin) = (
        Load { settings, keep_alive: Duration::MAX, ..Load::default() },
        Load { settings: Settings { context: 64, chunk: 8 }, keep_alive: Duration::MAX, pin: true, ..Load::default() },
    );
    let messages = [Message::new(Role::User, "hello there")];
    let (vram, host, disk) = (Some(Tier::Vram), Some(Tier::Host), Some(Tier::Disk));
    // Room for two: a third moves the least recently used to the host tier, from where it answers the same.
    let mut engine = Engine::new(0, Limits { vram: Some(2 * need + need / 2), host: None }).unwrap();
    engine.load("a", &dir, &keep).unwrap();
    engine.load("b", &dir, &keep).unwrap();
    let want = answer(&mut engine, ("a", None), &messages, 8);
    engine.load("c", &dir, &keep).unwrap();
    assert_eq!(tiers(&engine, ["a", "b", "c"]), [vram, host, vram], "b was the least recently used");
    assert_eq!(answer(&mut engine, ("b", None), &messages, 8), want, "b back from the host tier");
    assert_eq!(tiers(&engine, ["a", "b", "c"]), [host, vram, vram], "then a was");
    // A pinned model stays, though the least recently used.
    engine.load("a", &dir, &pin).unwrap();
    answer(&mut engine, ("b", None), &messages, 1);
    answer(&mut engine, ("c", None), &messages, 1);
    assert_eq!(tiers(&engine, ["a", "b", "c"]), [vram, host, vram], "the pinned a stayed");
    engine.load("c", &dir, &pin).unwrap();
    let full = engine.load("d", &dir, &keep).err();
    assert_eq!(full, Some(tachyon::Error::OutOfMemory { need, free: need / 2 }), "two pinned models leave no room");
    assert_eq!(engine.tier("d"), None, "a model that does not fit is not loaded");
    // A model whose keep-alive runs out goes to disk, though the host tier has room: at once after an answer kept
    // zero, or once its keep-alive runs out; and comes back answering the same.
    let mut engine = Engine::new(0, Limits { vram: None, host: None }).unwrap();
    engine.load("a", &dir, &keep).unwrap();
    assert_eq!(answer(&mut engine, ("a", Some(Duration::ZERO)), &messages, 8), want);
    assert_eq!(engine.tier("a"), disk, "kept zero");
    assert_eq!(answer(&mut engine, ("a", Some(Duration::from_millis(50))), &messages, 8), want, "a back from disk");
    engine.load("m", &dir, &keep).unwrap();
    std::thread::sleep(Duration::from_millis(60));
    engine.expire().unwrap();
    assert_eq!(tiers(&engine, ["a", "m"]), [disk, vram], "a's keep-alive ran out; m's never does");
    assert!(engine.unload("m") && !engine.unload("m") && engine.tier("m").is_none());
    std::fs::remove_dir_all(src).unwrap();
}

#[test]
fn a_parked_model_is_freed_when_its_keep_alive_runs_out() {
    let src = common::tiny::checkpoint("parked", 3, common::tiny::E4B);
    let dir = src.join("out");
    tachyon::model::convert(&src, &dir).unwrap();
    let settings = Settings { context: 64, chunk: 8 };
    let need = tachyon::model::footprint(&dir, &settings).unwrap();
    let keep = Load { settings, keep_alive: Duration::MAX, ..Load::default() };
    let messages = [Message::new(Role::User, "hello there")];
    let (vram, host, disk) = (Some(Tier::Vram), Some(Tier::Host), Some(Tier::Disk));
    // Room for one: p moves to the host tier for q, then its keep-alive runs out there, freeing its pinned memory.
    let mut engine = Engine::new(0, Limits { vram: Some(need + need / 2), host: None }).unwrap();
    engine.load("p", &dir, &keep).unwrap();
    answer(&mut engine, ("p", Some(Duration::from_millis(50))), &messages, 1);
    engine.load("q", &dir, &keep).unwrap();
    assert_eq!(tiers(&engine, ["p", "q"]), [host, vram], "p made room for q");
    std::thread::sleep(Duration::from_millis(60));
    engine.expire().unwrap();
    assert_eq!(tiers(&engine, ["p", "q"]), [disk, vram], "p's keep-alive ran out on the host tier");
    let listed = engine.models();
    assert_eq!(listed.iter().map(|l| l.vram).collect::<Vec<_>>(), [0, need], "only q takes VRAM");
    std::fs::remove_dir_all(src).unwrap();
}

/// Gemma 4 E4B (in `$TACHYON_MODEL`, by default `~/.local/share/tachyon/gemma-4-e4b`; another model is checked against
/// the fixture named after its directory, `tests/fixtures/<name>.ref`) against the greedy answers of
/// Hugging Face's model in bfloat16 (`scripts/reference.py`): fed each prompt and each reference answer token by
/// token, its pick after every token agrees with the reference's on ≥ 86% of the positions. That bound is what 4-bit
/// weights allow, measured by `scripts/agreement.py`: Hugging Face's model with its weights rounded as Tachyon rounds
/// them agrees on 87.9% (and on 83.1% with the scale rounding to the largest weight, where Tachyon measured 83.25%:
/// the engine computes what its weights hold); in bfloat16 its own one-pass reading agrees on 99.4%. Disagreements
/// where the reference was sure (a gap of 4 or more) are printed.
#[cfg(feature = "models")]
#[test]
fn gemma_agrees_with_hugging_face() {
    use tachyon::json::Json;
    use tachyon::token::Chat;
    let dir = std::env::var("TACHYON_MODEL")
        .map_or_else(|_| std::path::PathBuf::from(std::env::var("HOME").unwrap()).join(".local/share/tachyon/gemma-4-e4b"), Into::into);
    let tok = tachyon::token::Tokenizer::from_bytes(&std::fs::read(dir.join("tokenizer.tok")).unwrap()).unwrap();
    let chat = Chat::open(&tok, &dir).unwrap();
    let ctx = tachyon::cuda::Context::new(0).unwrap();
    let mut dec = tachyon::model::Decoder::open(&ctx, &dir, &Settings::default()).unwrap();
    let name = dir.file_name().unwrap().to_string_lossy();
    let path = format!("{}/tests/fixtures/{name}.ref", env!("CARGO_MANIFEST_DIR"));
    let fixture = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}; scripts/reference.py makes it"));
    let nums = |j: &Json, k: &str| j.get(k).and_then(Json::arr).unwrap().iter().map(|v| v.num().unwrap()).collect::<Vec<f64>>();
    let ids = |j: &Json, k: &str| nums(j, k).into_iter().map(|v| v as u32).collect::<Vec<u32>>();
    let (mut agree, mut total) = (0, 0);
    for line in fixture.lines() {
        let j = Json::parse(line.as_bytes()).unwrap();
        let text = j.get("text").and_then(Json::str).unwrap();
        let (prompt, answer, gaps) = (ids(&j, "prompt"), ids(&j, "answer"), nums(&j, "gap"));
        let mut mine = Vec::new();
        chat.encode(&tok, &[Message::new(Role::User, text)], false, &mut mine);
        assert_eq!(mine, prompt, "the chat template's tokens");
        dec.reset(&tachyon::ptx::Sampling::default()).unwrap();
        let mut pick = mine.iter().map(|&t| dec.step(t).unwrap()).last().unwrap();
        for (k, (&want, gap)) in answer.iter().zip(&gaps).enumerate() {
            if pick == want {
                agree += 1;
            } else if *gap >= 4.0 {
                println!("{text:?}, answer token {k}: {pick} against {want} (reference gap {gap:.2})");
            }
            total += 1;
            pick = dec.step(want).unwrap();
        }
    }
    println!("top-1 agreement {agree}/{total} ({:.2}%)", 100.0 * f64::from(agree) / f64::from(total));
    assert!(f64::from(agree) >= 0.86 * f64::from(total));
}
