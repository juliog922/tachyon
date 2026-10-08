//! Loading `.wpk` files onto a real GPU: `cargo test --features gpu`.
#![cfg(feature = "gpu")]

use std::path::PathBuf;
use tachyon::Error;
use tachyon::cuda::{Context, DevBuf};
use tachyon::wpk::{CHUNK, Dtype, Loader, Reads, Wpk, Writer};

/// A two-tensor file whose first tensor spans `len` bytes, enough chunks to reuse every staging slot.
fn sample(name: &str, len: usize) -> (PathBuf, Vec<u8>) {
    let path = std::env::temp_dir().join(format!("tachyon-gpu-{}-{name}.wpk", std::process::id()));
    let bytes: Vec<u8> = (0..len).map(|i| (i % 253) as u8 ^ (i >> 16) as u8).collect();
    let mut w = Writer::create(&path).unwrap();
    w.add("big", Dtype::U8, &[len as u64], &bytes).unwrap();
    w.add("small", Dtype::F32, &[250], &bytes[..1000]).unwrap();
    w.finish().unwrap();
    (path, bytes)
}

fn device_bytes(ctx: &Context, buf: &DevBuf) -> Vec<u8> {
    let (stream, mut out) = (ctx.stream().unwrap(), vec![0; buf.len()]);
    // SAFETY: `out` outlives the copy, which `sync` completes.
    unsafe { stream.download(&mut out, buf, 0) }.unwrap();
    stream.sync().unwrap();
    out
}

/// Checks that every tensor of `wpk` sits at its offset in `arena`.
fn check(wpk: &Wpk, arena: &[u8], bytes: &[u8]) {
    for t in wpk.tensors() {
        let at = t.offset as usize;
        assert!(arena[at..at + t.len as usize] == bytes[..t.len as usize], "tensor {} ({:?})", t.name, wpk.reads());
    }
}

#[test]
fn disk_to_device_direct_and_buffered() {
    let ctx = Context::new(0).unwrap();
    let mut loader = Loader::new(&ctx).unwrap();
    let (path, bytes) = sample("device", 6 * CHUNK + 777);
    for reads in [Reads::Direct, Reads::Cached, Reads::Auto] {
        let wpk = Wpk::open_with(&path, reads).unwrap();
        let arena = ctx.alloc(wpk.data_len()).unwrap();
        loader.to_device(&wpk, &arena).unwrap();
        check(&wpk, &device_bytes(&ctx, &arena), &bytes);
    }
    std::fs::remove_file(path).unwrap();
}

#[test]
fn disk_to_host_tier_then_to_device() {
    let ctx = Context::new(0).unwrap();
    let mut loader = Loader::new(&ctx).unwrap();
    let (path, bytes) = sample("host", 3 * CHUNK + 4096);
    let wpk = Wpk::open(&path).unwrap();
    let mut host = ctx.alloc_host(wpk.data_len()).unwrap();
    loader.to_host(&wpk, &mut host).unwrap();
    check(&wpk, &host, &bytes);
    let arena = ctx.alloc(wpk.data_len()).unwrap();
    loader.upload(&host, &arena).unwrap();
    check(&wpk, &device_bytes(&ctx, &arena), &bytes);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn arenas_too_small_are_refused() {
    let ctx = Context::new(0).unwrap();
    let mut loader = Loader::new(&ctx).unwrap();
    let (path, _) = sample("small", CHUNK);
    let wpk = Wpk::open(&path).unwrap();
    assert_eq!(loader.to_device(&wpk, &ctx.alloc(wpk.data_len() - 4096).unwrap()), Err(Error::OutOfRange));
    assert_eq!(loader.to_host(&wpk, &mut ctx.alloc_host(4096).unwrap()), Err(Error::OutOfRange));
    std::fs::remove_file(path).unwrap();
}