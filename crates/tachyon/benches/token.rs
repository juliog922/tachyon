//! The tokenizer: a 2,000-token prompt and the whole fixture encoded, and the time to load a stored tokenizer.
//! Checks the step-5 exit gate: the prompt encodes in ≤ 0.5 ms.
//!
//! `cargo bench -p tachyon --bench token` uses Gemma 4's tokenizer from
//! `$TACHYON_MODELS/gemma-4-E4B-it/tokenizer.json` when set, and the fixture's small one otherwise.

mod harness;

use harness::{Suite, cpu};
use std::hint::black_box;
use std::process::ExitCode;
use tachyon::json::Json;
use tachyon::token::Tokenizer;

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

/// The first fixture strings, joined until they make at least 2,000 tokens, and their tokens.
fn prompt(tok: &Tokenizer, strings: &[String]) -> (String, Vec<u32>) {
    let (mut prompt, mut ids) = (String::new(), Vec::new());
    for s in strings {
        prompt.push_str(s);
        ids.clear();
        tok.encode(&prompt, &mut ids);
        if ids.len() >= 2000 {
            break;
        }
    }
    (prompt, ids)
}

/// Times the 2,000-token prompt, returned with its length, and the whole fixture.
fn encode(suite: &mut Suite, tok: &Tokenizer, strings: &[String]) -> (std::time::Duration, usize) {
    let (prompt, mut ids) = prompt(tok, strings);
    let t = suite.run(&format!("prompt, {} tokens", ids.len()), 200, || {
        cpu(|| {
            ids.clear();
            tok.encode(black_box(&prompt), &mut ids);
        })
    });
    suite.rate(prompt.len() as f64 / t.as_secs_f64() / 1e6, "MB/s");
    let total: usize = strings.iter().map(String::len).sum();
    let all = suite.run("fixture, 10,000 strings", 5, || cpu(|| strings.iter().for_each(|s| tok.encode(black_box(s), &mut Vec::new()))));
    suite.rate(total as f64 / all.as_secs_f64() / 1e6, "MB/s");
    (t, ids.len())
}

fn main() -> ExitCode {
    let path = std::env::var("TACHYON_MODELS").map_or(format!("{FIXTURES}/toy.json"), |d| format!("{d}/gemma-4-E4B-it/tokenizer.json"));
    let json = std::fs::read(&path).unwrap();
    let mut suite = Suite::new("token");
    let mut tok = None;
    suite.run("read tokenizer.json", 3, || cpu(|| tok = Some(Tokenizer::from_json(&json).unwrap())));
    let (tok, text) = (tok.unwrap(), std::fs::read_to_string(format!("{FIXTURES}/strings.jsonl")).unwrap());
    let strings: Vec<String> = text.lines().map(|l| Json::parse(l.as_bytes()).unwrap().str().unwrap().to_string()).collect();
    let stored = tok.to_bytes();
    suite.run("load stored tokenizer", 20, || cpu(|| drop(black_box(Tokenizer::from_bytes(&stored).unwrap()))));
    let (t, n) = encode(&mut suite, &tok, &strings);
    println!("{path}: {} tokens, stored in {} MB", tok.len(), stored.len() >> 20);
    let passed = t.as_secs_f64() <= 0.5e-3;
    println!("gate: a {n}-token prompt encodes in {t:.1?} (≤ 0.5 ms): {}", if passed { "PASS" } else { "FAIL" });
    let code = suite.finish();
    if passed { code } else { ExitCode::FAILURE }
}