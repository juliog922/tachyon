//! Helpers the GPU tests share: a context with the kernels loaded, copies, and seeded values.
// Each test file uses part of this module; the Q8 check compares a stored product exactly, as the kernel computes it.
#![allow(dead_code, clippy::float_cmp, clippy::many_single_char_names)]

use tachyon::cuda::{Context, DevBuf, Module, Stream};

/// GPU 0 with every kernel of [`tachyon::ptx::module`] loaded.
pub struct Gpu {
    pub ctx: Context,
    pub stream: Stream,
    pub module: Module,
}

impl Gpu {
    pub fn new() -> Gpu {
        let ctx = Context::new(0).expect("GPU 0 with compute capability 8.0 or newer");
        let (stream, module) = (ctx.stream().unwrap(), ctx.load(&tachyon::ptx::module()).unwrap());
        Gpu { ctx, stream, module }
    }

    /// A buffer holding `data`.
    pub fn upload<T: Copy>(&self, data: &[T]) -> DevBuf {
        let buf = self.ctx.alloc(size_of_val(data)).unwrap();
        // SAFETY: `data` outlives the copy, which `sync` completes.
        unsafe { self.stream.upload(&buf, 0, bytes(data)) }.unwrap();
        self.stream.sync().unwrap();
        buf
    }

    /// `len` zeroed bytes.
    pub fn zeros(&self, len: usize) -> DevBuf {
        let buf = self.ctx.alloc(len).unwrap();
        self.stream.fill(&buf, 0).unwrap();
        buf
    }

    /// The contents of `buf`, once the queued work is done.
    pub fn download(&self, buf: &DevBuf) -> Vec<u8> {
        let mut out = vec![0; buf.len()];
        // SAFETY: `out` outlives the copy, which `sync` completes.
        unsafe { self.stream.download(&mut out, buf, 0) }.unwrap();
        self.stream.sync().unwrap();
        out
    }

    pub fn floats(&self, buf: &DevBuf) -> Vec<f32> {
        self.download(buf).chunks_exact(4).map(|c| f32::from_ne_bytes(c.try_into().unwrap())).collect()
    }

    pub fn words(&self, buf: &DevBuf) -> Vec<u32> {
        self.download(buf).chunks_exact(4).map(|c| u32::from_ne_bytes(c.try_into().unwrap())).collect()
    }

    pub fn halves(&self, buf: &DevBuf) -> Vec<u16> {
        self.download(buf).chunks_exact(2).map(|c| u16::from_ne_bytes(c.try_into().unwrap())).collect()
    }
}

pub mod tiny;

/// Plain numbers viewed as their bytes.
pub fn bytes<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: plain numbers, viewed as their bytes for the length of the borrow.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), size_of_val(v)) }
}

/// Uniform values in [-1, 1) from a fixed seed.
pub fn values(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

/// Checks a Q8 vector against `want`: every value within half a quantum of its block, plus `slack`.
pub fn assert_q8(q: &[u8], s: &[f32], want: &[f64], slack: f64) {
    for (i, w) in want.iter().enumerate() {
        let scale = f64::from(s[i / 32 * 2]);
        let got = f64::from(q[i] as i8) * scale;
        assert!((got - w).abs() <= 0.5 * scale * 1.001 + slack, "value {i}: {got} vs {w} (scale {scale})");
    }
    for (b, pair) in s.chunks_exact(2).enumerate() {
        let sum: i32 = q[b * 32..b * 32 + 32].iter().map(|&v| i32::from(v as i8)).sum();
        assert_eq!(pair[1], sum as f32 * pair[0], "block {b}: the stored sum");
    }
}