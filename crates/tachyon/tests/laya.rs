//! Laya's sequences against the `laya` package's: the 500 cases of `scripts/encoders.py` (fixture
//! `laya-multilingual.ref`), built with mmBERT's tokenizer from `$TACHYON_MODELS/laya-multilingual/tokenizer/`.
#![cfg(feature = "models")]

use std::fmt::Write as _;
use tachyon::encoder::laya::{Budget, Builder, Frame, Kind, Question, probabilities};
use tachyon::json::Json;
use tachyon::token::Tokenizer;

const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/laya-multilingual.ref");

/// `s` as a JSON string with `ensure_ascii=False`: quotes, backslashes and control characters escaped.
fn quoted(s: &str) -> String {
    let mut out = String::from('"');
    for c in s.chars() {
        match c {
            '"' | '\\' => {
                out.push('\\');
                out.push(c);
            }
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => _ = write!(out, "\\u{:04x}", c as u32),
            c => out.push(c),
        }
    }
    out + "\""
}

/// `value` as Python's `json.dumps(value, ensure_ascii=False)` writes it, which is how Laya turns a state into text.
fn dumps(value: &Json) -> String {
    match value {
        Json::Str(s) => quoted(s),
        Json::Num(n) => format!("{n}"),
        Json::Bool(b) => b.to_string(),
        Json::Null => "null".into(),
        Json::Arr(a) => format!("[{}]", a.iter().map(dumps).collect::<Vec<_>>().join(", ")),
        Json::Obj(o) => format!("{{{}}}", o.iter().map(|(k, v)| format!("{}: {}", quoted(k), dumps(v))).collect::<Vec<_>>().join(", ")),
    }
}

fn numbers(v: &Json) -> Vec<f64> {
    v.arr().unwrap().iter().map(|x| x.num().unwrap()).collect()
}

/// A case's question as the fixture writes it: `criteria` a map of labels to descriptions, a list of labels or of
/// levels, or absent.
fn options(q: &Json) -> Vec<(String, String)> {
    match q.get("criteria") {
        Some(Json::Obj(o)) => o.iter().map(|(k, v)| (k.clone(), v.str().unwrap_or("").to_string())).collect(),
        Some(Json::Arr(a)) if q.get("type").and_then(Json::str) == Some("score") => a.iter().map(|v| (String::new(), v.str().unwrap().into())).collect(),
        Some(Json::Arr(a)) => a.iter().map(|v| (v.str().unwrap().into(), String::new())).collect(),
        _ => Vec::new(),
    }
}

#[test]
fn sequences_match_the_laya_package() {
    let dir = std::env::var("TACHYON_MODELS").expect("TACHYON_MODELS: the directory holding laya-multilingual/");
    let tok = Tokenizer::from_json(&std::fs::read(format!("{dir}/laya-multilingual/tokenizer/tokenizer.json")).unwrap()).unwrap();
    let frame = Frame { cls: tok.special("<bos>").unwrap(), sep: tok.special("<eos>").unwrap(), mask: tok.special("<mask>").unwrap() };
    let builder = Builder { tok: &tok, frame, budget: Budget { total: 1024, head: 256 } };
    let mut ids = Vec::new();
    let cases = std::fs::read_to_string(FIXTURE).unwrap();
    for (i, line) in cases.lines().enumerate() {
        let case = Json::parse(line.as_bytes()).unwrap();
        let (state, q) = (case.get("state").unwrap(), case.get("question").unwrap());
        let text = state.str().map_or_else(|| dumps(state), str::to_string);
        let kind = [("choice", Kind::Choice), ("score", Kind::Score), ("noul", Kind::Noul)]
            .into_iter()
            .find(|k| Some(k.0) == q.get("type").and_then(Json::str))
            .unwrap()
            .1;
        let owned = options(q);
        let pairs: Vec<(&str, &str)> = owned.iter().map(|(l, d)| (l.as_str(), d.as_str())).collect();
        let question = Question { kind, instructions: q.get("instructions").and_then(Json::str).unwrap(), options: &pairs };
        let markers = builder.sequence(&question, &builder.tokens(&text), matches!(state, Json::Arr(_)), &mut ids).unwrap();
        let want: Vec<u32> = numbers(case.get("ids").unwrap()).iter().map(|&v| v as u32).collect();
        assert_eq!(ids, want, "case {i}: {text:.80}");
        let want: Vec<usize> = numbers(case.get("markers").unwrap()).iter().map(|&v| v as usize).collect();
        assert_eq!(markers, want, "case {i}");
        let logits: Vec<f32> = numbers(case.get("logits").unwrap()).iter().map(|&v| v as f32).collect();
        for (p, want) in probabilities(&logits, 1.0).iter().zip(numbers(case.get("probs").unwrap())) {
            assert!((f64::from(*p) - want).abs() < 1e-6, "case {i}: probability {p} against {want}");
        }
    }
}
