# Tachyon

An embedded inference engine for NVIDIA GPUs, written in Rust with zero dependencies.

Tachyon runs a fixed catalog of open-weight models (Gemma 4, Laya, Granite embeddings) inside your process, at the speed the GPU's memory bandwidth allows, and switches between them in tenths of a second. It needs only `libc` and the NVIDIA driver's `libcuda.so.1`, loaded at run time: no CUDA Toolkit, no crates, no Python.

> **Status: step 3 of 14.** The CUDA driver layer and the weight file loader are done; the 4-bit decode kernel is built and awaits GPU validation. Model inference arrives in later steps; see the roadmap below.

## Requirements

| | |
| --- | --- |
| OS | Linux on x86_64, native or WSL2 |
| GPU | NVIDIA, compute capability 8.0 or newer (RTX 30-series, A100 and later) |
| Driver | Supports CUDA 12.0 or newer (R525+). The CUDA Toolkit is not needed |
| Rust | 1.85 or newer, to build |

On WSL2 the driver lives on Windows and is found at `/usr/lib/wsl/lib/libcuda.so.1`. There, `nvidia-smi` and the driver report different version numbers; that is normal. To use another driver library, set `TACHYON_LIBCUDA` to its path.

## Try it

```sh
cargo run --release -p tachyon-cli -- gpu          # the GPUs the driver sees
cargo run --release -p tachyon-cli -- gpu --json
```

## Layout

```
crates/tachyon/          the library
  src/cuda/driver.rs     libcuda.so.1 loaded with dlopen; every function fetched at the CUDA 12.0 ABI
  src/cuda/mod.rs        owning, non-Send wrappers: Context, DevBuf, HostBuf, Stream, Event, Module, Graph
  src/wpk/mod.rs         the .wpk weight file: writer, checked index, CRC32C verification
  src/wpk/load.rs        io_uring + O_DIRECT loader: disk → VRAM, disk → host-RAM tier, host tier → VRAM
  src/sys.rs             raw syscalls and io_uring, ported from caudal
  src/quant.rs           Q4 weight and Q8 activation layouts, on the CPU
  src/ptx/mod.rs         the GPU kernels, generated as PTX text: the Q8 quantizer and the Q4 × Q8 GEMV
  tests/                 GPU tests (feature `gpu`) and their PTX kernels
  benches/               machine ceilings and the bench harness
crates/tachyon-cli/      the `tachyon` command
bench/results/<machine>/ benchmark baselines, one directory per machine
budget.txt               code-line budget of each module
ci.sh                    every quality gate
```

## Development rules

The best code is code not written. Every module has a line budget (`budget.txt`), every function a complexity cap, and a step is done only when its tests, benches and gates pass. `./ci.sh` runs every gate:

1. `cargo fmt --check`
2. `cargo clippy` with `pedantic` and `undocumented_unsafe_blocks`, warnings as errors
3. `cargo doc` with warnings as errors; `missing_docs` is denied
4. `lizard`: cyclomatic complexity ≤ 10, functions ≤ 50 lines, ≤ 6 parameters
5. `scripts/budget.sh`: code lines within `budget.txt`
6. `cargo test`, plus the GPU tests when a GPU is visible
7. `ptxas` on every PTX file for sm_80, sm_86, sm_89 and sm_90, when installed

Tools: `pip install lizard`, and for `ptxas` without the CUDA Toolkit, `pip install nvidia-cuda-nvcc-cu12`.

## Tests and benchmarks on a GPU

```sh
cargo test --workspace --features tachyon/gpu
cargo bench -p tachyon --features gpu --bench machine
cargo bench -p tachyon --features gpu --bench load
cargo bench -p tachyon --features gpu --bench gemv
```

The `machine` bench measures the ceilings every later target is a fraction of: VRAM read bandwidth (the decode roofline), PCIe in both directions, and the CPU cost of a kernel launch and of a graph replay. It prints a table, writes `target/bench/machine.json`, and checks the step-1 exit gate. The `load` bench writes a test `.wpk` sized to the machine (to `$TACHYON_BENCH_DIR`, default `target/bench`; put it on the disk models load from) and checks the step-2 gate: a staged disk → VRAM load reaches 90% of the slower of the disk and PCIe. The `gemv` bench runs the decode kernel on every Gemma 4 E4B projection and checks the step-3 gate: each streams its weights at 90% of the measured VRAM read bandwidth.

Baselines live in `bench/results/<machine>/`. Save one with `TACHYON_BENCH_SAVE=1`; later runs fail when a median is more than 3% slower. The machine name is `$TACHYON_BENCH_MACHINE`, else the host name. For stable numbers, lock the GPU clocks (`nvidia-smi -lgc`) and keep laptops on AC power.

## Roadmap

| Step | What | State |
| --- | --- | --- |
| 0 | Workspace, quality gates, bench harness | done |
| 1 | CUDA driver layer | done |
| 2 | Weight file format and `io_uring` loader | done |
| 3 | 4-bit GEMV kernel at ≥ 90% of VRAM bandwidth | built; GPU validation pending |
| 4 | Remaining decode kernels | |
| 5 | Tokenizer and chat template | |
| 6 | Gemma 4 E4B text generation | |
| 7 | Prefill on tensor cores | |
| 8 | Residency, keep-alive, model switching | |
| 9 | `pull`, catalog, conversion | |
| 10 | ModernBERT encoder: Laya decisions, Granite embeddings | |
| 11 | Vision and audio inputs | |
| 12 | Larger tiers: 12B, 31B, FP8, MoE | |
| 13 | Speculative decoding | |

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT), at your option.