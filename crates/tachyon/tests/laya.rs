//! Laya's sequences against the `laya` package's: the 500 cases of `scripts/encoders.py` (fixture
//! `laya-multilingual.ref`), built with mmBERT's tokenizer from `$TACHYON_MODELS/laya-multilingual/tokenizer/`.
#![cfg(feature = "models")]

use tachyon::encoder::laya::{Budget, Builder, Frame, Kind, Question, probabilities};
use tachyon::json::Json;
use tachyon::token::Tokenizer;

const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/laya-multilingual.ref");

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
        let text = state.str().map_or_else(|| state.dumps(), str::to_string);
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

/// The checkpoint `$TACHYON_MODELS/name`, converted once into the test's temporary directory.
#[cfg(feature = "gpu")]
fn converted(name: &str) -> std::path::PathBuf {
    let dst = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    if !dst.join("host.wpk").exists() {
        let models = std::env::var("TACHYON_MODELS").expect("TACHYON_MODELS: the models' repositories");
        tachyon::encoder::convert(&std::path::Path::new(&models).join(name), &dst).unwrap();
    }
    dst
}

/// Text in, answers out: `Engine::decide` on the states and questions of the first 50 cases, as the `laya` package
/// takes them, against its float32 probabilities (within 1e-2, the GPU's `f16`).
#[cfg(feature = "gpu")]
#[test]
fn the_engine_decides_from_text() {
    use tachyon::engine::{Engine, Limits, Load};
    let mut engine = Engine::new(0, Limits::default()).unwrap();
    engine.load("laya", &converted("laya-multilingual"), &Load::default()).unwrap();
    let cases = std::fs::read_to_string(FIXTURE).unwrap();
    for (i, line) in cases.lines().take(50).enumerate() {
        let case = Json::parse(line.as_bytes()).unwrap();
        let (state, q) = (case.get("state").unwrap(), case.get("question").unwrap());
        let text = state.str().map_or_else(|| state.dumps(), str::to_string);
        let kind = [("choice", Kind::Choice), ("score", Kind::Score), ("noul", Kind::Noul)]
            .into_iter()
            .find(|k| Some(k.0) == q.get("type").and_then(Json::str))
            .unwrap()
            .1;
        let owned = options(q);
        let pairs: Vec<(&str, &str)> = owned.iter().map(|(l, d)| (l.as_str(), d.as_str())).collect();
        let question = Question { kind, instructions: q.get("instructions").and_then(Json::str).unwrap(), options: &pairs };
        let p = engine.decide("laya", &text, matches!(state, Json::Arr(_)), &[question]).unwrap().remove(0);
        for (got, want) in p.iter().zip(numbers(case.get("probs").unwrap())) {
            assert!((f64::from(*got) - want).abs() < 1e-2, "case {i}: {p:?} against {want}");
        }
    }
}
