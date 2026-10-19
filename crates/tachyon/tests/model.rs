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
    assert_eq!(((0..6).map(|i| cfg.mlp(i)).collect::<Vec<_>>(), cfg.group(0)), (vec![512, 512, 512, 512, 1024, 1024], 2), "E2B's wide shared layers");
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

/// The larger models' layout: no per-layer inputs (an empty host table), no shared caches, global layers whose
/// values are their keys (the key projection stacked twice) with their own count of key-value heads.
#[test]
fn dense_checkpoint_converts_without_loss() {
    let src = common::tiny::checkpoint("convert-dense", 1, tiny::DENSE);
    let dst = src.join("out");
    convert(&src, &dst).unwrap();
    let cfg = Config::from_json(&std::fs::read(dst.join("config.json")).unwrap()).unwrap();
    assert_eq!((cfg.kv_heads, cfg.per_layer, cfg.shared), ([4, 1], 0, 0));
    assert_eq!((cfg.per_block(0), cfg.group(0), cfg.per_block(1), cfg.group(1)), (2, 1, 4, 2), "pairs in sliding layers, two blocks of four in global ones");
    assert_eq!((0..6).map(|i| cfg.source(i)).collect::<Vec<_>>(), [0, 1, 2, 3, 4, 5]);
    let source: std::collections::HashMap<String, Vec<f32>> = tiny::tensors(1, tiny::DENSE).into_iter().map(|(n, _, v)| (n, v)).collect();
    let p = "model.language_model.";
    let qkv = ["q", "k", "k"].map(|k| source[&format!("{p}layers.2.self_attn.{k}_proj.weight")].as_slice()).concat();
    assert_eq!(dequant(&dst, "weights.wpk", "2.qkv"), qkv, "a global layer's values are its keys");
    assert_eq!(dequant(&dst, "weights.wpk", "1.qkv").len(), (8 + 2 * 4) * 256 * 256, "a sliding layer's own values");
    assert!(Wpk::open(dst.join("weights.wpk")).unwrap().tensor("0.per_layer_gate.w").is_none());
    assert_eq!(Wpk::open(dst.join("host.wpk")).unwrap().data_len(), 0, "no per-layer table");
    std::fs::remove_dir_all(src).unwrap();
}

/// The 26B A4B's layout: the router (Q4, its scale over √hidden), every expert's gate and up stacked and its down,
/// and widths that are no multiple of 128 padded with zero weights (the MLP's 320 to 384, the experts' 192 to 256).
#[test]
fn experts_convert_without_loss() {
    let src = common::tiny::checkpoint("convert-moe", 1, tiny::MOE);
    let dst = src.join("out");
    convert(&src, &dst).unwrap();
    let cfg = Config::from_json(&std::fs::read(dst.join("config.json")).unwrap()).unwrap();
    assert_eq!((cfg.experts, cfg.ffn, cfg.moe()), ([64, 4, 192], 320, true));
    let source: std::collections::HashMap<String, Vec<f32>> = tiny::tensors(1, tiny::MOE).into_iter().map(|(n, _, v)| (n, v)).collect();
    let l = "model.language_model.layers.3.";
    let padded =
        |m: &[f32], cols: usize, to: usize| m.chunks(cols).flat_map(|r| r.iter().copied().chain(std::iter::repeat_n(0.0, to - cols))).collect::<Vec<f32>>();
    let weights = dst.join("weights.wpk");
    assert_eq!(dequant(&dst, "weights.wpk", "3.down"), padded(&source[&format!("{l}mlp.down_proj.weight")], 320, 384), "the MLP's down, padded");
    assert_eq!(dequant(&dst, "weights.wpk", "3.experts.down"), padded(&source[&format!("{l}experts.down_proj")], 192, 256), "the experts' downs");
    assert_eq!(dequant(&dst, "weights.wpk", "3.experts.gate_up"), source[&format!("{l}experts.gate_up_proj")]);
    assert_eq!(dequant(&dst, "weights.wpk", "3.router"), source[&format!("{l}router.proj.weight")]);
    let floats = |b: Vec<u8>| b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect::<Vec<_>>();
    let scale: Vec<f32> = source[&format!("{l}router.scale")].iter().map(|v| v / 16.0).collect();
    assert_eq!(floats(tensor(&weights, "3.router_scale")), scale, "the router's scale over √256");
    assert_eq!(floats(tensor(&weights, "3.expert_scale")), source[&format!("{l}router.per_expert_scale")]);
    assert_eq!(floats(tensor(&weights, "3.post_feedforward_layernorm_2")), source[&format!("{l}post_feedforward_layernorm_2.weight")]);
    std::fs::remove_dir_all(src).unwrap();
}

#[test]
fn unsupported_features_are_refused_by_name() {
    let base = tiny::config(tiny::E4B);
    let refused = |from: &str, to: &str| match Config::from_json(base.replace(from, to).as_bytes()) {
        Err(tachyon::Error::Unsupported(what)) => what,
        other => panic!("{to}: {other:?}"),
    };
    let moe = tiny::config(tiny::MOE);
    let refused_moe = |from: &str, to: &str| matches!(Config::from_json(moe.replace(from, to).as_bytes()), Err(tachyon::Error::Unsupported(_)));
    assert!(refused_moe(r#""num_experts":64"#, r#""num_experts":256"#) && refused_moe(r#""top_k_experts":4"#, r#""top_k_experts":32"#));
    assert!(refused(r#""num_key_value_heads":2"#, r#""num_key_value_heads":8"#).contains("fewer than two"));
}

/// The tiny model on the GPU against the CPU reference in `f32`, in E4B's shape and the larger models' (with experts too): after each of 20 tokens (past the 8-position sliding
/// window), the logits agree within the error of Q8 activations and an `f16` cache (2–4.2% RMS measured over the
/// shapes and positions, bound at 5%), and the greedy
/// pick is the reference's best or within that error of it.
#[cfg(feature = "gpu")]
#[test]
fn tiny_model_decodes_as_the_reference() {
    for shape in [("decode", tiny::E4B), ("decode-dense", tiny::DENSE), ("decode-moe", tiny::MOE)] {
        decodes_as_the_reference(shape);
    }
}

/// [`tiny_model_decodes_as_the_reference`] for one shape.
#[cfg(feature = "gpu")]
fn decodes_as_the_reference((name, shape): (&str, tiny::Shape)) {
    let src = common::tiny::checkpoint(name, 2, shape);
    let dst = src.join("out");
    convert(&src, &dst).unwrap();
    let ctx = tachyon::cuda::Context::new(0).unwrap();
    let mut dec = tachyon::model::Decoder::open(&ctx, &dst, &tachyon::model::Settings { context: 64, ..Default::default() }).unwrap();
    dec.reset(&tachyon::ptx::Sampling::default()).unwrap();
    let tokens: Vec<u32> = (0..20).map(|i| (i * 797 + 11) % tiny::VOCAB as u32).collect();
    let rms = |v: &mut dyn Iterator<Item = f32>| v.map(|x| x * x).sum::<f32>().sqrt() / (tiny::VOCAB as f32).sqrt();
    for (pos, (&tok, want)) in tokens.iter().zip(tiny::reference(2, shape, &tokens)).enumerate() {
        let picked = dec.step(tok).unwrap() as usize;
        let got = dec.logits().unwrap();
        let (scale, err) = (rms(&mut want.iter().copied()), rms(&mut got.iter().zip(&want).map(|(a, b)| a - b)));
        let best = (0..want.len()).fold(0, |b, i| if want[i] > want[b] { i } else { b });
        assert!(err <= 0.05 * scale, "{name}, position {pos}: logits differ by {err} RMS (logits {scale} RMS)");
        assert!(want[picked] >= want[best] - 3.0 * err, "{name}, position {pos}: picked {picked} ({}), reference {best} ({})", want[picked], want[best]);
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

/// Prefill of the tiny model, in E4B's shape, E2B's and the larger models' ([`tiny::DENSE`], [`tiny::MOE`]): 19 tokens through prefill and the 20th through a decode
/// step give the logits of the CPU reference (within the error of Q8P activations and an `f16` cache: 2–4% RMS
/// measured, bound at 5%), and the same
/// logits, bit for bit, whether prefill runs them in chunks of 8 (so sliding caches wrap their ring) or all at once.
#[cfg(feature = "gpu")]
#[test]
fn tiny_model_prefills_as_the_reference_in_any_chunks() {
    for (name, shape) in [("prefill-e4b", tiny::E4B), ("prefill-e2b", tiny::E2B), ("prefill-dense", tiny::DENSE), ("prefill-moe", tiny::MOE)] {
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
        assert!(err <= 0.05 * scale, "{name}: logits differ by {err} RMS (logits {scale} RMS)");
        std::fs::remove_dir_all(src).unwrap();
    }
}
