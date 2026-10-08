//! The driver layer on a real GPU: `cargo test --features gpu`.
#![cfg(feature = "gpu")]

use tachyon::Error;
use tachyon::cuda::{Context, DevBuf, Stream, arg};

const PTX: &str = include_str!("kernels/probe.ptx");
const OUT_OF_MEMORY: i32 = 2;
const INVALID_DEVICE: i32 = 101;

fn gpu() -> (Context, Stream) {
    let ctx = Context::new(0).expect("GPU 0 with compute capability 8.0 or newer");
    let stream = ctx.stream().unwrap();
    (ctx, stream)
}

fn floats(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_ne_bytes()).collect()
}

fn read(stream: &Stream, buf: &DevBuf) -> Vec<u8> {
    let mut out = vec![0; buf.len()];
    // SAFETY: `out` outlives the copy, which `sync` completes.
    unsafe { stream.download(&mut out, buf, 0) }.unwrap();
    stream.sync().unwrap();
    out
}

/// Queues `sum = x + y` over `count` floats.
fn add(stream: &Stream, kernel: &tachyon::cuda::Function<'_>, bufs: [&DevBuf; 3], count: u32) {
    let [x, y, sum] = bufs.map(DevBuf::ptr);
    // SAFETY: four arguments of the kernel's types; the buffers outlive the stream's work.
    unsafe { stream.launch(kernel, [count.div_ceil(256), 1, 1], [256, 1, 1], 0, &[arg(&x), arg(&y), arg(&sum), arg(&count)]) }.unwrap();
}

#[test]
fn kernels_add_what_was_uploaded() {
    let (ctx, stream) = gpu();
    let module = ctx.load(PTX).unwrap();
    let n = 1000u32;
    let (a, b): (Vec<f32>, Vec<f32>) = (0..n).map(|i| (i as f32, 2.0 * i as f32)).unzip();
    let bufs = [ctx.alloc(4000).unwrap(), ctx.alloc(4000).unwrap(), ctx.alloc(4000).unwrap()];
    let (ha, hb) = (floats(&a), floats(&b));
    // SAFETY: `ha` and `hb` outlive the copies, completed by `read`.
    unsafe { stream.upload(&bufs[0], 0, &ha).and(stream.upload(&bufs[1], 0, &hb)) }.unwrap();
    add(&stream, &module.function("add_f32").unwrap(), [&bufs[0], &bufs[1], &bufs[2]], n);
    assert_eq!(read(&stream, &bufs[2]), floats(&(0..n).map(|i| 3.0 * i as f32).collect::<Vec<_>>()));
}

#[test]
fn graphs_replay_what_they_recorded() {
    let (ctx, stream) = gpu();
    let module = ctx.load(PTX).unwrap();
    let f = module.function("add_f32").unwrap();
    let (acc, one) = (ctx.alloc(64).unwrap(), ctx.alloc(64).unwrap());
    stream.fill(&acc, 0).unwrap();
    // SAFETY: the source outlives the copy, completed by the sync below.
    unsafe { stream.upload(&one, 0, &floats(&[1.0; 16])) }.unwrap();
    stream.sync().unwrap();
    let graph = stream
        .capture(|s| {
            add(s, &f, [&acc, &one, &acc], 16);
            Ok(())
        })
        .unwrap();
    assert_eq!(read(&stream, &acc), floats(&[0.0; 16]), "capturing does not run the work");
    for _ in 0..3 {
        // SAFETY: the module and buffers the graph uses are alive.
        unsafe { stream.replay(&graph) }.unwrap();
    }
    assert_eq!(read(&stream, &acc), floats(&[3.0; 16]));
}

#[test]
fn xor_read_reads_every_word_once() {
    let (ctx, stream) = gpu();
    let module = ctx.load(PTX).unwrap();
    let words: Vec<u32> = (0..100_003 * 4).map(|i: u32| i.wrapping_mul(2_654_435_761)).collect();
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_ne_bytes()).collect();
    let (src, out) = (ctx.alloc(bytes.len()).unwrap(), ctx.alloc(7 * 128 * 4).unwrap());
    // SAFETY: `bytes` outlives the copy, completed by `read`.
    unsafe { stream.upload(&src, 0, &bytes) }.unwrap();
    let (ptr, count, dst) = (src.ptr(), (bytes.len() / 16) as u64, out.ptr());
    // SAFETY: three arguments of the kernel's types; `out` holds one word per thread.
    unsafe { stream.launch(&module.function("xor_read").unwrap(), [7, 1, 1], [128, 1, 1], 0, &[arg(&ptr), arg(&count), arg(&dst)]) }.unwrap();
    let got = read(&stream, &out).chunks_exact(4).fold(0, |x, w| x ^ u32::from_ne_bytes(w.try_into().unwrap()));
    assert_eq!(got, words.iter().fold(0, |x, w| x ^ w));
}

#[test]
fn kernels_read_mapped_host_memory_in_place() {
    let (ctx, stream) = gpu();
    let module = ctx.load(PTX).unwrap();
    let mut host = ctx.alloc_host(64).unwrap();
    host.copy_from_slice(&floats(&[2.5; 16]));
    let (zero, sum) = (ctx.alloc(64).unwrap(), ctx.alloc(64).unwrap());
    stream.fill(&zero, 0).unwrap();
    let (a, b, c, n) = (host.device_ptr().unwrap(), zero.ptr(), sum.ptr(), 16u32);
    // SAFETY: four arguments of the kernel's types; every buffer outlives the work.
    unsafe { stream.launch(&module.function("add_f32").unwrap(), [1, 1, 1], [32, 1, 1], 0, &[arg(&a), arg(&b), arg(&c), arg(&n)]) }.unwrap();
    assert_eq!(read(&stream, &sum), floats(&[2.5; 16]));
}

#[test]
fn events_time_gpu_work() {
    let (ctx, stream) = gpu();
    let (buf, start, end) = (ctx.alloc(64 << 20).unwrap(), ctx.event(true).unwrap(), ctx.event(true).unwrap());
    stream.record(&start).unwrap();
    stream.fill(&buf, 7).unwrap();
    stream.record(&end).unwrap();
    end.sync().unwrap();
    assert!(end.since(&start).unwrap() > 0.0);
}

#[test]
fn bad_ptx_is_rejected_with_the_compiler_log() {
    let (ctx, _) = gpu();
    match ctx.load(".version 7.1\n.target sm_80\n.address_size 64\n.visible .entry k() { bogus; }") {
        Err(Error::Ptx(log)) => assert!(!log.is_empty()),
        other => panic!("expected a PTX error, got {:?}", other.err()),
    }
}

#[test]
fn driver_failures_carry_their_codes() {
    let (ctx, stream) = gpu();
    assert_eq!(Context::new(1 << 20).err(), Some(Error::Cuda(INVALID_DEVICE)));
    assert_eq!(ctx.alloc(1 << 50).err(), Some(Error::Cuda(OUT_OF_MEMORY)));
    assert!(Error::Cuda(OUT_OF_MEMORY).to_string().contains("CUDA_ERROR_OUT_OF_MEMORY"));
    let small = ctx.alloc(16).unwrap();
    // SAFETY: the copies are refused before anything is queued.
    unsafe {
        assert_eq!(stream.upload(&small, 8, &[0; 16]), Err(Error::OutOfRange));
        assert_eq!(stream.download(&mut [0; 4], &small, usize::MAX), Err(Error::OutOfRange));
    }
    assert_eq!(stream.copy(&small, &ctx.alloc(32).unwrap()), Err(Error::OutOfRange));
}

#[test]
fn devices_describe_the_gpu() {
    let (ctx, _) = gpu();
    let info = ctx.info();
    assert!(info.compute >= (8, 0) && info.sms > 0 && info.vram > 0 && !info.name.is_empty());
    let (free, total) = ctx.memory().unwrap();
    assert!(free <= total && total as u64 <= info.vram);
    assert!(tachyon::cuda::driver_version().unwrap() >= 12_000);
}
