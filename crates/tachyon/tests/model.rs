//! Gemma 4's decoder: conversion of a checkpoint, on the CPU.

mod common;

use common::tiny;
use std::os::unix::fs::FileExt;
use tachyon::model::{Config, convert};
use tachyon::quant::f16_value;
use tachyon::wpk::{ALIGN, Wpk};

/// The bytes of tensor `name` of the `.wpk` at `path`.
fn tensor(path: &std::path::Path, name: &str) -> Vec<u8> {
    let t = Wpk::open(path).unwrap().tensor(name).unwrap_or_else(|| panic!("{name}")).clone();
    let mut out = vec![0; t.len as usize];
    std::fs::File::open(path).unwrap().read_exact_at(&mut out, ALIGN + t.offset).unwrap();
    out
}

/// A Q4 matrix of a model directory, dequantized.
fn dequant(dir: &std::path::Path, file: &str, name: &str) -> Vec<f32> {
    let (w, s) = (tensor(&dir.join(file), &format!("{name}.w")), tensor(&dir.join(file), &format!("{name}.s")));
    let scales: Vec<f32> = s.chunks_exact(2).map(|b| f16_value(u16::from_le_bytes([b[0], b[1]]))).collect();
    (0..w.len() * 2)
        .map(|j| f32::from(w[j / 64 * 32 + j % 64 / 8 * 4 + j % 4] >> (j % 8 / 4 * 4) & 15) - 8.0)
        .enumerate()
        .map(|(j, n)| n * scales[j / 64])
        .collect()
}

#[test]
fn tiny_checkpoint_converts_without_loss() {
    let src = common::tiny::checkpoint("convert", 1, tiny::E2B);
    let dst = src.join("out");
    convert(&src, &dst).unwrap();
    let cfg = Config::from_json(&std::fs::read(dst.join("config.json")).unwrap()).unwrap();
    assert_eq!((cfg.hidden, cfg.layers(), cfg.head_dim, cfg.window, cfg.softcap), (256, 6, [256, 512], 8, 30.0));
    assert_eq!((0..6).map(|i| cfg.source(i)).collect::<Vec<_>>(), [0, 1, 2, 3, 3, 2]);
    assert_eq!(((0..6).map(|i| cfg.mlp(i)).collect::<Vec<_>>(), cfg.group()), (vec![512, 512, 512, 512, 1024, 1024], 2), "E2B's wide shared layers");
    let source: std::collections::HashMap<String, Vec<f32>> = tiny::tensors(1, tiny::E2B).into_iter().map(|(n, _, v)| (n, v)).collect();
    let p = "model.language_model.";
    assert_eq!(dequant(&dst, "weights.wpk", "embed"), source[&format!("{p}embed_tokens.weight")]);
    assert_eq!(dequant(&dst, "host.wpk", "embed_per_layer"), source[&format!("{p}embed_tokens_per_layer.weight")]);
    let qkv = ["q", "k", "v"].map(|k| source[&format!("{p}layers.2.self_attn.{k}_proj.weight")].as_slice()).concat();
    assert_eq!(dequant(&dst, "weights.wpk", "2.qkv"), qkv, "query, key and value stacked");
    assert_eq!(dequant(&dst, "weights.wpk", "4.qkv").len(), 8 * 256 * 256, "a shared layer projects queries only");
    assert_eq!(dequant(&dst, "weights.wpk", "5.gate_up").len(), 2 * 1024 * 256, "a wide MLP");
    let floats = |b: Vec<u8>| b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect::<Vec<_>>();
    assert_eq!(floats(tensor(&dst.join("weights.wpk"), "5.q_norm")), source[&format!("{p}layers.5.self_attn.q_norm.weight")]);
    assert_eq!(floats(tensor(&dst.join("weights.wpk"), "3.layer_scalar")), source[&format!("{p}layers.3.layer_scalar")]);
    tachyon::token::Tokenizer::from_bytes(&std::fs::read(dst.join("tokenizer.tok")).unwrap()).unwrap();
    std::fs::remove_dir_all(src).unwrap();
}

#[test]
fn larger_models_features_are_refused_by_name() {
    let base = tiny::config(tiny::E4B);
    let refused = |from: &str, to: &str| match Config::from_json(base.replace(from, to).as_bytes()) {
        Err(tachyon::Error::Unsupported(what)) => what,
        other => panic!("{to}: {other:?}"),
    };
    assert!(refused(r#""enable_moe_block":false"#, r#""enable_moe_block":true"#).contains("mixture-of-experts"));
    assert!(refused(r#""attention_k_eq_v":false"#, r#""attention_k_eq_v":true"#).contains("values equal to keys"));
    assert!(refused(r#""hidden_size_per_layer_input":256"#, r#""hidden_size_per_layer_input":0"#).contains("per-layer inputs"));
    assert!(refused(r#""num_key_value_heads":2"#, r#""num_key_value_heads":2,"num_global_key_value_heads":1"#).contains("global layers"));
    assert!(refused(r#""num_key_value_heads":2"#, r#""num_key_value_heads":4"#).contains("four query heads"));
}

/// The tiny model on the GPU against the CPU reference in `f32`: after each of 20 tokens (past the 8-position sliding
/// window), the logits agree within the error of Q8 activations and an `f16` cache (about 2% RMS), and the greedy
/// pick is the reference's best or within that error of it.
#[cfg(feature = "gpu")]
#[test]
fn tiny_model_decodes_as_the_reference() {
    let src = common::tiny::checkpoint("decode", 2, tiny::E4B);
    let dst = src.join("out");
    convert(&src, &dst).unwrap();
    let ctx = tachyon::cuda::Context::new(0).unwrap();
    let mut dec = tachyon::model::Decoder::open(&ctx, &dst, &tachyon::model::Settings { context: 64, ..Default::default() }).unwrap();
    dec.reset(&tachyon::ptx::Sampling::default()).unwrap();
    let tokens: Vec<u32> = (0..20).map(|i| (i * 797 + 11) % tiny::VOCAB as u32).collect();
    let rms = |v: &mut dyn Iterator<Item = f32>| v.map(|x| x * x).sum::<f32>().sqrt() / (tiny::VOCAB as f32).sqrt();
    for (pos, (&tok, want)) in tokens.iter().zip(tiny::reference(2, tiny::E4B, &tokens)).enumerate() {
        let picked = dec.step(tok).unwrap() as usize;
        let got = dec.logits().unwrap();
        let (scale, err) = (rms(&mut want.iter().copied()), rms(&mut got.iter().zip(&want).map(|(a, b)| a - b)));
        let best = (0..want.len()).fold(0, |b, i| if want[i] > want[b] { i } else { b });
        assert!(err <= 0.04 * scale, "position {pos}: logits differ by {err} RMS (logits {scale} RMS)");
        assert!(want[picked] >= want[best] - 3.0 * err, "position {pos}: picked {picked} ({}), reference {best} ({})", want[picked], want[best]);
    }
    // Generating with a step always queued gives the same tokens as stepping one at a time.
    let mut runs = [Vec::new(), Vec::new()];
    for (i, run) in runs.iter_mut().enumerate() {
        dec.reset(&tachyon::ptx::Sampling::default()).unwrap();
        let mut token = dec.step(tokens[0]).unwrap();
        run.push(token);
        if i == 0 {
            dec.start(token).unwrap();
            while run.len() < 5 {
                run.push(dec.next_token().unwrap());
            }
            dec.stop().unwrap();
        }
        while run.len() < 5 {
            token = dec.step(token).unwrap();
            run.push(token);
        }
    }
    assert_eq!(runs[0], runs[1], "queued generation and single steps agree");
    std::fs::remove_dir_all(src).unwrap();
}

/// Prefill of the tiny model, in E4B's shape and in E2B's: 19 tokens through prefill and the 20th through a decode
/// step give the logits of the CPU reference (within the error of Q8P activations and an `f16` cache), and the same
/// logits, bit for bit, whether prefill runs them in chunks of 8 (so sliding caches wrap their ring) or all at once.
#[cfg(feature = "gpu")]
#[test]
fn tiny_model_prefills_as_the_reference_in_any_chunks() {
    for (name, shape) in [("prefill-e4b", tiny::E4B), ("prefill-e2b", tiny::E2B)] {
        let src = common::tiny::checkpoint(name, 4, shape);
        let dst = src.join("out");
        convert(&src, &dst).unwrap();
        let ctx = tachyon::cuda::Context::new(0).unwrap();
        let tokens: Vec<u32> = (0..20).map(|i| (i * 1013 + 7) % tiny::VOCAB as u32).collect();
        let logits = |chunk| {
            let mut dec = tachyon::model::Decoder::open(&ctx, &dst, &tachyon::model::Settings { context: 64, chunk }).unwrap();
            dec.reset(&tachyon::ptx::Sampling::default()).unwrap();
            dec.prefill(&tokens[..19]).unwrap();
            dec.step(tokens[19]).unwrap();
            dec.logits().unwrap()
        };
        let (small, whole) = (logits(8), logits(64));
        assert_eq!(small, whole, "{name}: chunks of 8 and one chunk agree exactly");
        let want = tiny::reference(4, shape, &tokens).pop().unwrap();
        let rms = |v: &mut dyn Iterator<Item = f32>| v.map(|x| x * x).sum::<f32>().sqrt() / (tiny::VOCAB as f32).sqrt();
        let (scale, err) = (rms(&mut want.iter().copied()), rms(&mut whole.iter().zip(&want).map(|(a, b)| a - b)));
        assert!(err <= 0.04 * scale, "{name}: logits differ by {err} RMS (logits {scale} RMS)");
        std::fs::remove_dir_all(src).unwrap();
    }
}
