# tachyon

An embedded inference engine for NVIDIA GPUs on Linux x86_64, with zero dependencies: it needs only `libc` and the NVIDIA driver's `libcuda.so.1`, loaded at run time.

**Early development.** This release contains `tachyon::cuda`, a safe owning layer over the CUDA Driver API (contexts, memory, streams, events, PTX modules, CUDA Graphs) that works with any driver supporting CUDA 12.0 or newer. Model inference arrives in later releases.

```rust,no_run
use tachyon::cuda::Context;

let ctx = Context::new(0)?;
println!("{} with {} SMs", ctx.info().name, ctx.info().sms);
# Ok::<(), tachyon::Error>(())
```

See the [repository README](../../README.md) for requirements, the roadmap and the development rules.
