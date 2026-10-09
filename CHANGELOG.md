# Changelog

All notable changes to this project are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- Attention gives each block at least 24 positions and loads a chunk of positions' keys and values at once, instead of relying on prefetch hints, which the RTX 3060 ignores: 11 µs → 8 µs per sliding layer at short contexts.

### Added

- `Decoder::generate`: decoding with the next step always queued behind the running one (two recordings of the step trade the token words), so the GPU never waits for the CPU. `Decoder::timeline` and `Decoder::alone` profile a step kernel by kernel and each kind of kernel alone; the `generate` benchmark reports both and gates decode at ≥ 80% of the roofline: 107–109 tok/s on the RTX 3060 Laptop (81–83%), against 61–77 tok/s for Ollama (llama.cpp) on the same machine.
- `ptx::GEMV_Q4_NARROW`, the decode GEMV for rows of 256 weights: 8 lanes per row. `ptx::SPIN`, a kernel that holds the GPU, for profiling.
- `tachyon::model::Decoder`: Gemma 4 on the GPU. One decode step (embedding, per-layer inputs, 42 layers of sliding or global attention with shared caches, LM head, sampler) is recorded once as a CUDA Graph; the token goes in and comes out through one pinned host word, so the CPU only waits. Tested against a CPU reference that matches Hugging Face transformers to 1e-5, on a generated tiny Gemma 4. `tachyon run <model> <prompt>` answers a prompt greedily and reports the speed.
- `tachyon::model`: Gemma 4's decoder configuration and `convert`, which turns a Hugging Face checkpoint into a model directory: Q4 weights for VRAM with query/key/value and gate/up fused, the per-layer embedding table for pinned host memory, the stored tokenizer. `tachyon convert <checkpoint> <model>` runs it. `tachyon::quant::Safetensors` reads checkpoints a slice of rows at a time, so a checkpoint larger than memory converts.
- `tachyon::token`: the tokenizer. `Tokenizer::from_json` reads a Hugging Face `tokenizer.json` (BPE with byte fallback, as Gemma uses), `to_bytes`/`from_bytes` store and load it ready to use; `encode` matches `tokenizers` exactly on 10,000 strings in 34 languages, emoji and code, and cuts the text where no merge can cross. `Detokenizer` emits whole characters as tokens arrive. `Chat` is Gemma 4's chat template coded by hand, with the system turn and thinking switch; turn markers enter only by ID, so text cannot forge them.
- `tachyon::json`: a JSON reader for model files.
- `token` benchmark, with the step-5 exit gate; `scripts/fixture.py`, which makes the tokenizer fixtures.
- `tachyon::quant`: Q4 weights (groups of 64, f16 scale, zero point 8, packed for `dp4a`) and Q8 activations (blocks of 32 with their sums), and exact f16 conversion.
- `tachyon::ptx`: kernels generated as PTX text for `sm_80`+: `quant_q8`, and `gemv_q4`, the decode matrix-vector product, Q4 × Q8 with `dp4a`, one warp per row and every load of a short row in flight at once.
- `tachyon::ptx`: the rest of the decode kernels. `norm_q8`, the residual add with RMS normalization and the next input's Q8; `geglu_q8`, the GELU-gated activation into Q8; `embed_q4`, a Q4 embedding row from VRAM or pinned host memory; `attend_256` and `attend_512`, decode attention with query and key norms, RoPE (including Gemma 4's proportional RoPE through `ptx::rope`), an `f16` key-value cache as a sliding window or global, shared caches, and positions split across blocks; `sample`, greedy or exact sampling with temperature, top-k, top-p, min-p, repetition penalty, an allowed-token mask and logit soft-capping, set through `ptx::Sampling` in device memory.
- `decode` benchmark: every non-GEMV kernel per launch, and one E4B token's worth against a decode step at the VRAM ceiling.
- `gemv` benchmark: every Gemma 4 E4B projection, including the per-layer input projections, against the measured VRAM read bandwidth, with the step-3 exit gate: one token's projections, weighted by bytes, at ≥ 90%.
- `tachyon::wpk`: the `.wpk` weight file (4 KiB-aligned tensors, checked header and table, CRC32C per tensor) with `Writer`, `Wpk::open`, `Wpk::verify`.
- `wpk::Loader`: `io_uring` reads with `O_DIRECT` straight into pinned memory; disk → VRAM through staging slots overlapped with uploads, disk → host-RAM tier, and host tier → VRAM. `Reads::Auto` reads through the page cache when it already holds the file (a model a previous process loaded) and with `O_DIRECT` otherwise.
- `load` benchmark: every load path on a file sized to the machine, with the step-2 exit gate.
- `tachyon::cuda`: the CUDA Driver API loaded at run time from `libcuda.so.1` (or `/usr/lib/wsl/lib`, or `$TACHYON_LIBCUDA`), with every function fetched at the CUDA 12.0 ABI so newer drivers keep working.
- Owning, non-`Send` wrappers: `Context` (primary context), `DevBuf`, `HostBuf` (pinned, mappable), `Stream`, `Event`, `Module` (PTX with the JIT error log), `Graph` (stream capture and replay).
- `tachyon gpu [--json]`: the GPUs the driver sees, with compute capability, memory, datasheet bandwidth and PCIe link.
- `machine` benchmark: VRAM read bandwidth, PCIe in both directions, launch and graph-replay cost, with the step-1 exit gate.
- Bench harness with `perf` counters, JSON results and a 3% regression check against per-machine baselines.
- `ci.sh`: format, clippy pedantic, docs, complexity, line budgets, tests, PTX checks.