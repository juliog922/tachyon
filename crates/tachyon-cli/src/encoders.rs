//! `decide` and `embed`: Laya's decisions and Granite's embeddings from the command line.
//!
//! A decision request is a JSON file in the `laya` package's shape: `{"state": …, "questions": {"name": {"type":
//! "choice" | "score" | "noul", "instructions": "…", "criteria": …}}}`. The state is text, or any JSON (written as
//! Laya writes it; a list is a conversation, newest last). A choice's criteria map labels to descriptions or list
//! labels; a score's list its levels, lowest first; a noul's, if any, describe `"false"` and `"true"`. The answer
//! is one JSON object: per question its probabilities, and the pick (a choice's label, a score's expected level, a
//! noul's probability of true).

use std::fmt::Write as _;
use std::path::Path;
use tachyon::encoder::laya::{Kind, Question};
use tachyon::engine::{Engine, Limits, Load};
use tachyon::json::Json;

/// A question as the request writes it, its options owned.
pub(crate) struct Asked {
    name: String,
    pub(crate) kind: Kind,
    pub(crate) instructions: String,
    pub(crate) options: Vec<(String, String)>,
}

/// The questions of request `r`.
pub(crate) fn questions(r: &Json) -> tachyon::Result<Vec<Asked>> {
    let bad = |why: &str| tachyon::Error::Format(format!("request: {why}"));
    let all = r.get("questions").and_then(Json::obj).ok_or_else(|| bad("no \"questions\" object"))?;
    all.iter()
        .map(|(name, q)| {
            let kind = match q.get("type").and_then(Json::str) {
                Some("choice") => Kind::Choice,
                Some("score") => Kind::Score,
                Some("noul") => Kind::Noul,
                _ => return Err(bad(&format!("{name}: \"type\" is choice, score or noul"))),
            };
            let instructions = q.get("instructions").and_then(Json::str).ok_or_else(|| bad(&format!("{name}: no \"instructions\"")))?.to_string();
            Ok(Asked { name: name.clone(), kind, instructions, options: options(q, kind) })
        })
        .collect()
}

/// A question's options as `(label, description)`: a map, labels, or levels.
fn options(q: &Json, kind: Kind) -> Vec<(String, String)> {
    let text = |v: &Json| v.str().map_or_else(|| if *v == Json::Null { String::new() } else { v.dumps() }, str::to_string);
    match q.get("criteria") {
        Some(Json::Obj(o)) => o.iter().map(|(k, v)| (k.clone(), text(v))).collect(),
        Some(Json::Arr(a)) if kind == Kind::Score => a.iter().map(|v| (String::new(), text(v))).collect(),
        Some(Json::Arr(a)) => a.iter().map(|v| (text(v), String::new())).collect(),
        _ => Vec::new(),
    }
}

/// The Laya checkpoint for the request in file `request` when it names `model` `laya`: `laya-english` for English text
/// if it is pulled, else `laya-multilingual` (see [`tachyon::encoder::route`]). Any other name stays as it is.
pub fn routed(model: &str, request: &Path) -> tachyon::Result<String> {
    if model != "laya" {
        return Ok(model.into());
    }
    let r = Json::parse(&std::fs::read(request)?)?;
    let state = r.get("state").ok_or_else(|| tachyon::Error::Format("request: no \"state\"".into()))?;
    let english = tachyon::encoder::route::english(&state.str().map_or_else(|| state.dumps(), str::to_string));
    let pulled = crate::models::home().join("laya-english").join("weights.wpk").exists();
    let name = if english && pulled { "laya-english" } else { "laya-multilingual" };
    eprintln!("laya: {name}");
    Ok(name.into())
}

/// Answers the request in file `request` with the Laya model in `dir`, as one JSON object.
pub fn decide(dir: &Path, request: &Path) -> tachyon::Result<String> {
    let r = Json::parse(&std::fs::read(request)?)?;
    let state = r.get("state").ok_or_else(|| tachyon::Error::Format("request: no \"state\"".into()))?;
    let asked = questions(&r)?;
    let pairs: Vec<Vec<(&str, &str)>> = asked.iter().map(|a| a.options.iter().map(|(l, d)| (l.as_str(), d.as_str())).collect()).collect();
    let qs: Vec<Question> = asked.iter().zip(&pairs).map(|(a, p)| Question { kind: a.kind, instructions: &a.instructions, options: p }).collect();
    let mut engine = Engine::new(0, Limits::default())?;
    engine.load("laya", dir, &Load::default())?;
    let text = state.str().map_or_else(|| state.dumps(), str::to_string);
    let answers = engine.decide("laya", &text, matches!(state, Json::Arr(_)), &qs)?;
    Ok(render(&asked, &answers))
}

/// The answers to `asked`, as one JSON object.
pub(crate) fn render(asked: &[Asked], answers: &[Vec<f32>]) -> String {
    let mut out = String::from("{");
    for (i, (a, p)) in asked.iter().zip(answers).enumerate() {
        let _ = write!(out, "{}{}: {}", if i > 0 { ", " } else { "" }, Json::Str(a.name.clone()).dumps(), answer(a, p));
    }
    out + "}"
}

/// One question's answer as JSON.
fn answer(a: &Asked, p: &[f32]) -> String {
    let probs =
        |labels: &mut dyn Iterator<Item = String>| labels.zip(p).map(|(l, v)| format!("{}: {v:.4}", Json::Str(l).dumps())).collect::<Vec<_>>().join(", ");
    match a.kind {
        Kind::Choice => {
            let best = p.iter().enumerate().fold(0, |b, (i, v)| if *v > p[b] { i } else { b });
            let label = Json::Str(a.options.get(best).map_or(String::new(), |o| o.0.clone())).dumps();
            format!("{{\"choice\": {label}, \"probabilities\": {{{}}}}}", probs(&mut a.options.iter().map(|o| o.0.clone())))
        }
        Kind::Score => {
            let expected: f32 = p.iter().enumerate().map(|(i, v)| i as f32 * v).sum();
            format!("{{\"score\": {expected:.4}, \"probabilities\": {{{}}}}}", probs(&mut (0..p.len()).map(|i| i.to_string())))
        }
        Kind::Noul => format!("{{\"noul\": {:.4}}}", p.get(1).copied().unwrap_or(0.0)),
    }
}

/// The embeddings of `texts` with the model in `dir`, cut to `dims` values if asked: one JSON array per line.
pub fn embed(dir: &Path, texts: &[&str], dims: Option<usize>) -> tachyon::Result<String> {
    let mut engine = Engine::new(0, Limits::default())?;
    engine.load("embed", dir, &Load::default())?;
    Ok(rows(&engine.embed("embed", texts, dims)?))
}

/// Embeddings as one JSON array per line.
pub(crate) fn rows(vectors: &[Vec<f32>]) -> String {
    vectors.iter().map(|v| format!("[{}]", v.iter().map(|x| format!("{x:.6}")).collect::<Vec<_>>().join(", "))).collect::<Vec<_>>().join("\n")
}

/// Runs `convert` while a line on standard error reports, every two seconds, how much of the checkpoint in `src`
/// it has read, how fast, and how long it has left.
fn with_progress(src: &Path, convert: impl FnOnce() -> tachyon::Result<()>) -> tachyon::Result<()> {
    use std::sync::atomic::{AtomicBool, Ordering};
    let entries = std::fs::read_dir(src)?.filter_map(Result::ok);
    let total: u64 = entries.filter(|e| e.path().extension().is_some_and(|x| x == "safetensors")).filter_map(|e| e.metadata().ok()).map(|m| m.len()).sum();
    let (done, start, from) = (AtomicBool::new(false), std::time::Instant::now(), tachyon::quant::READ.load(Ordering::Relaxed));
    std::thread::scope(|s| {
        s.spawn(|| {
            while !done.load(Ordering::Relaxed) {
                std::thread::park_timeout(std::time::Duration::from_secs(2));
                let (read, secs) = ((tachyon::quant::READ.load(Ordering::Relaxed) - from) as f64, start.elapsed().as_secs_f64());
                let left = (total as f64 - read).max(0.0) / (read / secs).max(1.0);
                eprint!("\rconverting: {:.1} of {:.1} GB, {:.0} MB/s, {left:.0} s left   ", read / 1e9, total as f64 / 1e9, read / secs / 1e6);
            }
            eprintln!("\rconverted {:.1} GB in {:.0} s{:30}", total as f64 / 1e9, start.elapsed().as_secs_f64(), "");
        });
        let result = convert();
        done.store(true, Ordering::Relaxed);
        result
    })
}

/// Converts the checkpoint in `src` to a model directory `dst`: an encoder (Laya, Granite Embedding) or Gemma 4. The
/// model is written beside `dst` as `<dst>.part` and renamed once whole, replacing an older `dst`: a conversion
/// stopped halfway leaves no model that looks whole.
pub fn convert(src: &Path, dst: &Path) -> tachyon::Result<()> {
    let config = std::fs::read(src.join("config.json")).unwrap_or_default();
    let mut part = dst.as_os_str().to_owned();
    part.push(".part");
    let part = Path::new(&part);
    // A part left by a conversion that stopped, or an older model, may or may not be there.
    _ = std::fs::remove_dir_all(part);
    let encoder = src.join("rl_agent_config.json").exists() || tachyon::encoder::Config::from_json(&config).is_ok();
    with_progress(src, || if encoder { tachyon::encoder::convert(src, part) } else { tachyon::model::convert(src, part) })?;
    _ = std::fs::remove_dir_all(dst);
    Ok(std::fs::rename(part, dst)?)
}
