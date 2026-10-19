//! The encoders on GPU 0: Laya multilingual's decisions about a support email of about 200 tokens, 1, 5 and 10
//! questions per call, and Granite Embedding 311M's embeddings of short texts, one and 64 per call. Checks the
//! step-10 gate: one Laya question answers in ≤ 6 ms, the whole call (tokenizing, encoder, head, probabilities),
//! and ten questions in one call take ≤ 4 ms each.
//! Then where the GPU time of one, ten questions and 64 embeddings goes, kernel by kernel, with the matrix
//! products' rates against the tensor cores' ceilings (`f16`, or 8-bit for Granite's Q8 layers), and the GPU's share
//! of each call's wall time.
//!
//! `cargo bench -p tachyon --features gpu --bench encoder` uses the model directories in `$TACHYON_LAYA` (default
//! `~/.local/share/tachyon/laya-multilingual`) and `$TACHYON_GRANITE` (default `…/granite-embedding-311m`), made by
//! `tachyon pull` or `tachyon convert`.

mod harness;

use harness::gpu::Gpu;
use harness::{Suite, cpu};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;
use tachyon::cuda::{Context, arg};
use tachyon::encoder::laya::{Kind, Question};
use tachyon::encoder::{Encoder, Settings};
use tachyon::engine::{Engine, Limits, Load};
use tachyon::token::Tokenizer;

const RUNS: usize = 30;
/// Calls timed kernel by kernel, for the median of each kernel.
const PROFILED: usize = 11;
const STATE: &str = "Hello, I am writing about my account. Last month I upgraded to the business plan, and since then I \
have been charged twice: once on the 3rd for 49 euros and again on the 5th for the same amount. Your website says the \
upgrade is prorated, so I expected a single charge of about 30 euros. I contacted support by chat on Monday and the \
agent promised a refund within 48 hours, but nothing has arrived and the ticket was closed without an answer. I also \
cannot download the invoices from the billing page, it shows an error when I click on the PDF link. I need both \
invoices for my accountant before Friday. If this is not solved this week I will cancel the subscription and move the \
team to another provider. My customer number is 88213 and the card ends in 4421. Thanks, Marta";

/// The model directory in `$var`, else `name` under `~/.local/share/tachyon`.
fn dir(var: &str, name: &str) -> PathBuf {
    std::env::var(var).map_or_else(|_| PathBuf::from(std::env::var("HOME").unwrap()).join(".local/share/tachyon").join(name), Into::into)
}

/// Ten questions of every kind, two to twelve options.
fn questions() -> Vec<Question<'static>> {
    const INTENT: [(&str, &str); 5] =
        [("billing", "charges, invoices, refunds"), ("technical", "bugs, errors"), ("account", "login, data"), ("shipping", "orders"), ("other", "")];
    const SENTIMENT: [(&str, &str); 3] = [("positive", ""), ("neutral", ""), ("negative", "")];
    const URGENCY: [(&str, &str); 4] = [("", "low"), ("", "medium"), ("", "high"), ("", "critical")];
    const DEPARTMENT: [(&str, &str); 6] = [("sales", ""), ("support", ""), ("legal", ""), ("hr", ""), ("finance", ""), ("logistics", "")];
    const SATISFACTION: [(&str, &str); 5] = [("", "very unsatisfied"), ("", "unsatisfied"), ("", "neutral"), ("", "satisfied"), ("", "very satisfied")];
    const SPAM: [(&str, &str); 2] = [("spam", "unsolicited advertising"), ("not_spam", "a genuine customer message")];
    const TOPIC: [(&str, &str); 12] = [
        ("payments", ""),
        ("refunds", ""),
        ("invoices", ""),
        ("login", ""),
        ("passwords", ""),
        ("privacy", ""),
        ("crashes", ""),
        ("performance", ""),
        ("delivery", ""),
        ("returns", ""),
        ("damaged goods", ""),
        ("praise", ""),
    ];
    let q = |kind, instructions, options: &'static [(&'static str, &'static str)]| Question { kind, instructions, options };
    vec![
        q(Kind::Choice, "What is the customer's main issue?", &INTENT),
        q(Kind::Noul, "The customer asks for a refund.", &[]),
        q(Kind::Score, "How urgent is this request?", &URGENCY),
        q(Kind::Choice, "What is the sentiment of the message?", &SENTIMENT),
        q(Kind::Noul, "The message contains personal data.", &[]),
        q(Kind::Choice, "Which department should handle this?", &DEPARTMENT),
        q(Kind::Score, "How satisfied is the customer?", &SATISFACTION),
        q(Kind::Noul, "The customer threatens to cancel.", &[]),
        q(Kind::Choice, "Is this message spam?", &SPAM),
        q(Kind::Choice, "Which topic fits best?", &TOPIC),
    ]
}

/// Laya with 1, 5 and 10 questions per call; the medians of one question alone and of each of ten, in milliseconds.
fn decisions(suite: &mut Suite, engine: &mut Engine) -> [f64; 2] {
    let qs = questions();
    // A second of work first: a laptop GPU raises its clocks only under load.
    let warm = std::time::Instant::now();
    while warm.elapsed() < Duration::from_secs(1) {
        engine.decide("laya", STATE, false, &qs).unwrap();
    }
    let mut each = [0.0; 3];
    for (i, k) in [1, 5, 10].into_iter().enumerate() {
        let t = suite.run(&format!("decide, {k} question(s)"), RUNS, || cpu(|| _ = engine.decide("laya", STATE, false, &qs[..k]).unwrap()));
        suite.rate(k as f64 / t.as_secs_f64(), "questions/s");
        each[i] = t.as_secs_f64() * 1e3 / k as f64;
    }
    [each[0], each[2]]
}

/// Short text `i` to embed.
fn text(i: usize) -> String {
    format!("Order {i}: the customer reports that item {} arrived damaged and asks for a replacement or a refund.", 1000 + 37 * i)
}

/// Granite on 1 and 64 short texts per call.
fn embeddings(suite: &mut Suite, engine: &mut Engine) {
    let texts: Vec<String> = (0..64).map(text).collect();
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    for n in [1, 64] {
        let t = suite.run(&format!("embed, {n} text(s)"), RUNS, || cpu(|| _ = engine.embed("granite", &refs[..n], None).unwrap()));
        suite.rate(n as f64 / t.as_secs_f64(), "embeddings/s");
    }
}

/// The median time of each kernel over `runs` of the same call, in order.
fn medians(runs: &[Vec<(String, Duration)>]) -> Vec<(String, Duration)> {
    let median = |i: usize| {
        let mut t: Vec<Duration> = runs.iter().map(|r| r[i].1).collect();
        t.sort();
        t[t.len() / 2]
    };
    (0..runs[0].len()).map(|i| (runs[0][i].0.clone(), median(i))).collect()
}

/// Operations of a product of kind `gemm tokens×cols→rows /tile`.
fn flops(kind: &str) -> Option<f64> {
    let (shape, rows) = kind.strip_prefix("gemm ")?.split_once('→')?;
    let rows = rows.split(' ').next()?;
    let (tokens, cols) = shape.split_once('×')?;
    Some(2.0 * tokens.parse::<f64>().ok()? * cols.parse::<f64>().ok()? * rows.parse::<f64>().ok()?)
}

/// Prints where the GPU time of `call` goes: each kind of kernel with its count, total, share and, for products,
/// rate against `peaks` (`f16`, 8-bit) in TOPS; then the GPU's total against the call's wall time.
fn profile(name: &str, enc: &mut Encoder, peaks: [f64; 2], call: impl Fn(&mut Encoder)) {
    let runs: Vec<_> = (0..PROFILED)
        .map(|_| {
            enc.timeline(|e| {
                call(e);
                Ok(())
            })
            .unwrap()
            .1
        })
        .collect();
    let ops = medians(&runs);
    let total: Duration = ops.iter().map(|o| o.1).sum();
    let mut kinds: Vec<(String, usize, Duration)> = Vec::new();
    for (kind, t) in &ops {
        match kinds.iter_mut().find(|k| &k.0 == kind) {
            Some(k) => (k.1, k.2) = (k.1 + 1, k.2 + *t),
            None => kinds.push((kind.clone(), 1, *t)),
        }
    }
    kinds.sort_by_key(|k| std::cmp::Reverse(k.2));
    println!("{name}: {total:.2?} of GPU time in {} kernels (median of {PROFILED} calls)", ops.len());
    println!("  {:<31} {:>5} {:>10} {:>9} {:>6} {:>7} {:>6}", "kernel", "count", "total", "each", "share", "TOPS", "%peak");
    for (kind, n, t) in &kinds {
        let (rate, peak) = (flops(kind).map(|f| f * *n as f64 / t.as_secs_f64() / 1e12), peaks[usize::from(kind.ends_with("q8"))]);
        let (tf, pct) = rate.map_or((String::new(), String::new()), |r| (format!("{r:.1}"), format!("{:.0}%", 100.0 * r / peak)));
        let share = 100.0 * t.as_secs_f64() / total.as_secs_f64();
        println!("  {kind:<31} {n:>5} {t:>10.2?} {:>9.2?} {share:>5.1}% {tf:>7} {pct:>6}", *t / *n as u32);
    }
    let mut wall: Vec<Duration> = (0..PROFILED).map(|_| cpu(|| call(enc))).collect();
    wall.sort();
    let wall = wall[PROFILED / 2];
    println!("  the call: {wall:.2?} of wall time, {:.0}% of it kernels back to back", 100.0 * total.as_secs_f64() / wall.as_secs_f64());
}

/// Each product kernel alone on the encoder's shapes, for one question's tokens and ten questions': its time and rate
/// against `peaks` (`f16`, 8-bit): the `f16` kernels, then the Q8.
fn products(suite: &mut Suite, gpu: &Gpu, peaks: [f64; 2]) {
    use tachyon::ptx::{GEMM_F16, GEMM_F16_128, GEMM_F16_TALL, GEMM_Q8, GEMM_Q8_128, GEMM_Q8_TALL};
    const KERNELS: [(&str, u32, u32, u32); 6] = [
        (GEMM_F16, 64, 64, 128),
        (GEMM_F16_128, 128, 64, 256),
        (GEMM_F16_TALL, 128, 128, 512),
        (GEMM_Q8, 64, 64, 128),
        (GEMM_Q8_128, 128, 64, 256),
        (GEMM_Q8_TALL, 128, 128, 512),
    ];
    println!("each product kernel alone (median of 20):");
    for (tokens, (cols, rows)) in [223u32, 2140].into_iter().flat_map(|t| [(768u32, 2304u32), (768, 768), (1152, 768)].map(|s| (t, s))) {
        let act = gpu.ctx.alloc(2 * (tokens as usize + 128) * cols as usize).unwrap();
        let (weights, out) = (gpu.ctx.alloc(2 * (rows * cols) as usize).unwrap(), gpu.ctx.alloc(4 * (tokens * rows) as usize).unwrap());
        let scales = gpu.ctx.alloc(4 * (tokens as usize + 128).max(rows as usize) * cols as usize / 64).unwrap();
        let line = KERNELS.map(|(name, tile, height, threads)| {
            let (kernel, q8) = (gpu.module.function(name).unwrap(), name.contains("q8"));
            let (ptrs, sizes) = ([act.ptr(), weights.ptr(), 0, out.ptr()], [rows, cols, tokens, 0]);
            let extra = [scales.ptr(), scales.ptr()];
            let args: Vec<_> = ptrs.iter().map(arg).chain(sizes.iter().map(arg)).chain(extra.iter().filter(|_| q8).map(arg)).collect();
            let blocks = tokens.div_ceil(tile) * rows / height;
            let launch = |s: &tachyon::cuda::Stream| {
                // SAFETY: the kernel's arguments in order; `act` holds whole blocks of tokens, `out` every token's row.
                unsafe { s.launch(&kernel, [blocks, 1, 1], [threads, 1, 1], 0, &args) }.unwrap();
            };
            let tall = if height == 64 { String::new() } else { format!("×{height}") };
            let label = format!("/{tile}{tall}{}", if q8 { " q8" } else { "" });
            let t = suite.run(&format!("gemm {tokens}×{cols}→{rows} {label}"), 20, || gpu.time(launch));
            let tf = 2.0 * f64::from(tokens * cols) * f64::from(rows) / t.as_secs_f64() / 1e12;
            suite.rate(tf, "TOPS");
            format!("{label} {t:>8.2?} {tf:>5.1} TOPS {:>3.0}%", 100.0 * tf / peaks[usize::from(q8)])
        });
        println!("  gemm {tokens}×{cols}→{rows}: {}", line.join(" · "));
    }
}

/// The kernel-by-kernel profiles of Laya (one and ten questions) and Granite (64 texts), against the `f16` ceilings.
fn profiles(suite: &mut Suite, laya: &Path, granite: &Path) {
    let gpu = Gpu::new();
    let peak = gpu.peak(suite, tachyon::ptx::MMA_PEAK_F16, "f16 mma peak, f32 sums");
    gpu.peak(suite, tachyon::ptx::MMA_PEAK_F16H, "f16 mma peak, f16 sums");
    let peaks = [peak, gpu.peak(suite, tachyon::ptx::MMA_PEAK, "int8 mma peak")];
    products(suite, &gpu, peaks);
    let ctx = Context::new(0).unwrap();
    let mut enc = Encoder::open(&ctx, laya, &Settings::default()).unwrap();
    let qs = questions();
    for k in [1, 10] {
        profile(&format!("decide, {k} question(s)"), &mut enc, peaks, |e| _ = e.decide(STATE, false, &qs[..k]).unwrap());
    }
    drop(enc);
    let mut enc = Encoder::open(&ctx, granite, &Settings::default()).unwrap();
    let texts: Vec<String> = (0..64).map(text).collect();
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    profile("embed, 64 texts", &mut enc, peaks, |e| _ = e.embed(&refs, None).unwrap());
}

fn main() -> ExitCode {
    let (laya, granite) = (dir("TACHYON_LAYA", "laya-multilingual"), dir("TACHYON_GRANITE", "granite-embedding-311m"));
    let tok = Tokenizer::from_bytes(&std::fs::read(laya.join("tokenizer.tok")).unwrap()).unwrap();
    let mut ids = Vec::new();
    tok.encode(STATE, &mut ids);
    let mut engine = Engine::new(0, Limits::default()).unwrap();
    engine.load("laya", &laya, &Load::default()).unwrap();
    engine.load("granite", &granite, &Load::default()).unwrap();
    let mut suite = Suite::new("encoder");
    let [one, ten] = decisions(&mut suite, &mut engine);
    embeddings(&mut suite, &mut engine);
    drop(engine);
    profiles(&mut suite, &laya, &granite);
    let passed = one <= 6.0 && ten <= 4.0;
    println!(
        "{} · one Laya question about a {}-token state in {one:.2} ms (≤ 6 ms), {ten:.2} ms each of ten in a call (≤ 4 ms): {}",
        tachyon::cuda::device(0).unwrap().name,
        ids.len(),
        if passed { "PASS" } else { "FAIL" }
    );
    let code = suite.finish();
    if passed { code } else { ExitCode::FAILURE }
}
