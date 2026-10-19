# Tachyon

An embedded inference engine for NVIDIA GPUs, written in Rust with zero dependencies.

Tachyon runs a fixed catalog of open-weight models (Gemma 4, Laya, Granite embeddings) inside your process, at the speed the GPU's memory bandwidth allows, and switches between them in tenths of a second. It needs only `libc` and the NVIDIA driver's `libcuda.so.1`, loaded at run time: no CUDA Toolkit, no crates, no Python.

> **Status: step 9 of 14.** Gemma 4 E4B and E2B generate text, prefill on tensor cores and share a GPU, switching in tenths of a second: 108 tok/s on the RTX 3060 Laptop (1.45× llama.cpp), with time to first token 1.1–1.7× faster than llama.cpp's on that laptop and on an L40S. Step 9, under validation, downloads models itself: `tachyon pull` speaks TLS 1.3 and HTTP/1.1 with nothing but `std`. See the roadmap below.

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
cargo run --release -p tachyon-cli -- doctor       # driver, GPUs, CPU, memory, trust anchors, the Hub
cargo run --release -p tachyon-cli -- catalog      # the models Tachyon knows
cargo run --release -p tachyon-cli -- pull         # downloads and converts the model recommended for GPU 0
cargo run --release -p tachyon-cli -- pull gemma-4-e4b
cargo run --release -p tachyon-cli -- list
cargo run --release -p tachyon-cli -- run gemma-4-e4b "Why is the sky blue?"
```

Laya answers typed questions about a text in one pass, and Granite embeds texts:

```sh
cargo run --release -p tachyon-cli -- pull laya-multilingual
cat > request.json <<'JSON'
{"state": "I was charged twice this month, please refund the second payment.",
 "questions": {"intent": {"type": "choice", "instructions": "What is the customer's main issue?",
                          "criteria": {"billing": "charges, invoices, refunds", "technical": "bugs", "other": ""}},
               "refund": {"type": "noul", "instructions": "The customer asks for a refund."}}}
JSON
cargo run --release -p tachyon-cli -- decide laya-multilingual request.json
cargo run --release -p tachyon-cli -- pull granite-embedding-311m
cargo run --release -p tachyon-cli -- embed granite-embedding-311m --dims=256 "how do I get a refund"
```

Laya comes in three checkpoints, each pulled by name: `laya-multilingual` (100+ languages, the default choice), `laya-english` (ModernBERT-large, the most accurate in English) and `laya-typed-decisions` (fine-tuned on four triage workflows). Name one, or name `laya` and Tachyon picks: `laya-english` for English text (if pulled), `laya-multilingual` for anything else, by a small rule on the text's script, accents and function words (`tachyon::encoder::route`). Granite comes in two sizes, `granite-embedding-311m` and `granite-embedding-97m` (384 dims, the fastest); the 311M's layers run in 8-bit (Q8), the 97M's in f16, as Q8 costs it too much precision.

`pull` downloads from Hugging Face over 8 connections (`TACHYON_CONNECTIONS`) in 8 MiB chunks, checks each file's SHA-256, and resumes where an interrupted pull stopped. Checkpoints wait in `$XDG_CACHE_HOME/tachyon` and are removed after conversion unless `--keep`; models live in `$TACHYON_HOME`, by default `~/.local/share/tachyon`. `TACHYON_HUB` points it at a mirror; `HF_TOKEN`, if set, is sent to the Hub (not to the CDN its downloads come from) for gated repositories and higher rate limits. A conversion writes `<model>.part` and renames it when whole, so an interrupted one leaves no broken model. A checkpoint downloaded some other way still converts with `tachyon convert <checkpoint> <model>`.

In your program, `tachyon::engine` is the API (see its documentation):

```rust
let mut engine = Engine::new(0, Limits::default())?;
engine.load("e4b", "gemma-4-e4b".as_ref(), &Load::default())?;
let messages = [Message { role: Role::User, text: "Why is the sky blue?" }];
let mut answer = engine.chat("e4b", &Request::new(&messages))?;
while let Some(text) = answer.next_text()? {
    print!("{text}");
}
```

### From other processes

`tachyon serve` keeps one engine for every process of your user, its models in VRAM between requests: each stays
as long as its keep-alive says (5 minutes by default, `--keep-alive=SECONDS`; a request can set its own, or pin a
model), so the next request to it starts at once; then it is freed, VRAM and RAM, and read from disk when asked
for again.

Models share a VRAM limit (`--vram-mib`, by default 70% of the GPU's), image and audio towers and encoders
included; a tower loads with a model's first image or clip and stays while the model does, so further requests
take no more. When a model does not fit beside the others, the least recently used ones that are not pinned move
to the host tier, pinned RAM (`--host-mib`, by default half the RAM), and come back at PCIe speed; past that, to
disk. Requests run one at a time, so models over the limit together take turns, each switch a copy. A model that
does not fit even alone fails with an error naming what it needs: no layer runs off the GPU. `env.example` lists
every setting as an environment variable.

There is no client library. A program in any language speaks to the daemon with the two files of `interface/`:
`tachyon.xml`, the messages as a Simple Binary Encoding schema (feed it to any SBE generator, or read the fixed
offsets by hand), and `tachyon.h`, the shared memory's layout and its rules. No TCP and no copies through the
kernel: the socket ($TACHYON_SOCKET, else `$XDG_RUNTIME_DIR/tachyon.sock`) carries the handshake and one-byte
doorbells only. Each client gets its own shared memory (`memfd`, passed over the socket): it writes a request there
and rings; the answer streams back into a ring in the same memory, a futex waking the reader. Images, audio,
embeddings and probabilities go through its data area, named by offset and length. Requests: Chat (streamed Text,
then Done with timings), Embed, Decide (Laya), Load, Unload, List; one at a time per daemon, in the order they rang,
a cancel stopping one between tokens. Only processes of the daemon's user are let in.

The command line drives it too: `tachyon up` starts it in the background (its log in
`~/.local/state/tachyon/serve.log`), `tachyon ps` lists its models (where, VRAM, time left in memory), `tachyon load` and
`tachyon unload` place them, and `tachyon down` stops it. While it is up, `run`, `transcribe`, `embed` and `decide`
go through it, so a model loads once and answers the next command from VRAM.

```sh
tachyon up --keep-alive=600
tachyon run gemma-4-e4b "Why is the sky blue?"     # loads it
tachyon run gemma-4-e4b "And the sunset?"          # answers from VRAM
tachyon ps
tachyon down
```

## Layout

```
crates/tachyon/          the library
  src/cuda/driver.rs     libcuda.so.1 loaded with dlopen; every function fetched at the CUDA 12.0 ABI
  src/cuda/mod.rs        owning, non-Send wrappers: Context, DevBuf, HostBuf, Stream, Event, Module, Graph
  src/wpk/mod.rs         the .wpk weight file: writer, checked index, CRC32C verification
  src/wpk/load.rs        io_uring + O_DIRECT loader: disk → VRAM, disk → host-RAM tier, host tier → VRAM
  src/sys.rs             raw syscalls and io_uring, ported from caudal
  src/quant.rs           Q4 weight and Q8 activation layouts on the CPU; the safetensors reader
  src/model/mod.rs       Gemma 4's decoder: configuration, checkpoint conversion
  src/model/decode.rs    the decode step as one CUDA Graph: every layer's kernels, KV caches, sampling
  src/json.rs            a JSON reader for the files models ship with
  src/token/mod.rs       the tokenizer: Gemma's BPE with byte fallback, stored ready to use, streaming detokenizer
  src/token/bytes.rs     byte-level BPE: GPT-2's byte characters, its pre-tokenizer patterns matched by hand
  src/token/unicode.rs   Unicode letters and NFC, from tables scripts/unicode.py writes (unicode.bin)
  src/token/chat.rs      Gemma 4's chat template, coded by hand
  src/ptx/mod.rs         the GPU kernels, generated as PTX text: quantizers, residual norm, embedding lookup
  src/ptx/gemv.rs        the Q4 × Q8 decode GEMV
  src/ptx/attend.rs      decode attention: QK norms, RoPE, KV cache, split over positions, merged by the last block
  src/ptx/sample.rs      sampling on the GPU: greedy, temperature, top-k, top-p, min-p, penalty, allowed tokens
  src/ptx/moe.rs         the mixture of experts: routing (softmax, top-k), prefill's sums of each token's experts
  tests/                 tests; GPU ones behind feature `gpu`, model-file ones behind `models`
  tests/fixtures/        tokenizer fixtures: 10,000 strings and the reference token IDs
  benches/               machine ceilings and the bench harness
  src/encoder/mod.rs     ModernBERT encoders (Laya multilingual, Granite Embedding 311M): configuration
  src/encoder/laya.rs    Laya's question sequences, head settings and probabilities
  src/encoder/convert.rs encoder checkpoints to model directories: f16, Granite 311M's layers in Q8
  src/encoder/run.rs     the encoder on the GPU: packed sequences, embeddings, Laya's logits
  src/encoder/race.rs    which product kernel each shape takes, raced on the GPU
  src/ptx/encoder.rs     the encoder's kernels: f16 and Q8 tensor-core products, layer norm, attention
crates/tachyon-cli/      the `tachyon` command
  src/models.rs          the catalog and pull, list, rm, doctor
  src/net/               downloads with std only: SHA-256 (SHA-NI), HMAC/HKDF, AES-128-GCM (AES-NI), X25519,
                         P-256 and RSA verification, X.509 chains, a TLS 1.3 client, HTTP/1.1, the Hub
  src/serve/             `tachyon serve`: the poll loop and the requests, the shared memory and ring, SBE
  tests/serve.rs         a client written from interface/ alone, end to end on the GPU
  benches/download.rs    SHA-256, AES-GCM, downloads against curl, conversion
interface/               the daemon's protocol: tachyon.xml (SBE messages) and tachyon.h (shared memory)
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

## Tests with model files

Tests and benches behind the `models` feature read model files from `$TACHYON_MODELS`, a directory holding, for now, `gemma-4-E4B-it/tokenizer.json`. The reference token IDs in `tests/fixtures` come from Hugging Face `tokenizers` through `scripts/fixture.py`, run once and committed.

```sh
hf download google/gemma-4-E4B-it tokenizer.json chat_template.jinja --local-dir ~/models/gemma-4-E4B-it
TACHYON_MODELS=~/models cargo test -p tachyon --features models --test token
TACHYON_MODELS=~/models cargo bench -p tachyon --bench token
```

The engine's agreement test compares the converted Gemma 4 E4B (`$TACHYON_MODEL`, by default `~/.local/share/tachyon/gemma-4-e4b`) with Hugging Face's answers in bfloat16, made once on a 20 GB GPU by `scripts/reference.py` and committed:

```sh
python3 scripts/reference.py ~/models/gemma-4-E4B-it > crates/tachyon/tests/fixtures/gemma-4-e4b.ref
cargo test --release -p tachyon --features gpu,models --test engine -- --nocapture
```

## Tests and benchmarks on a GPU

```sh
cargo test --workspace --features tachyon/gpu
cargo bench -p tachyon --features gpu --bench machine
cargo bench -p tachyon --features gpu --bench load
cargo bench -p tachyon --features gpu --bench gemv
cargo bench -p tachyon --features gpu --bench decode
cargo bench -p tachyon-cli --bench download
```

The `machine` bench measures the ceilings every later target is a fraction of: VRAM read bandwidth (the decode roofline), PCIe in both directions, and the CPU cost of a kernel launch and of a graph replay. It prints a table, writes `target/bench/machine.json`, and checks the step-1 exit gate. The `load` bench writes a test `.wpk` sized to the machine (to `$TACHYON_BENCH_DIR`, default `target/bench`; put it on the disk models load from) and checks the step-2 gate: a staged disk → VRAM load reaches 90% of the slower of the disk and PCIe. The `gemv` bench runs the decode kernel on every Gemma 4 E4B projection and checks the step-3 gate: one token's projections, weighted by the bytes each reads, stream at 90% of the measured VRAM read bandwidth. The `decode` bench times every other decode kernel per launch and reports one token's worth of them against a decode step whose GEMVs run at that bandwidth. The `download` bench fetches the first 256 MiB of Gemma 4 E2B's checkpoint with `curl` and with Tachyon over one connection and over eight, and checks the step-9 gate: over one connection Tachyon is within 5% of `curl`, and SHA-256 outruns every download.

Baselines live in `bench/results/<machine>/`. Save one with `TACHYON_BENCH_SAVE=1`; later runs fail when a median is more than 3% slower. The machine name is `$TACHYON_BENCH_MACHINE`, else the host name. For stable numbers, lock the GPU clocks (`nvidia-smi -lgc`) and keep laptops on AC power.

## Roadmap

| Step | What | State |
| --- | --- | --- |
| 0 | Workspace, quality gates, bench harness | done |
| 1 | CUDA driver layer | done |
| 2 | Weight file format and `io_uring` loader | done |
| 3 | 4-bit GEMV kernel at ≥ 90% of VRAM bandwidth | done |
| 4 | Remaining decode kernels | done |
| 5 | Tokenizer and chat template | done |
| 6 | Gemma 4 E4B text generation | done: 108 tok/s on the RTX 3060 Laptop, 1.45× llama.cpp |
| 7 | Prefill on tensor cores | done: time to first token 1.1–1.7× faster than llama.cpp |
| 8 | Residency, keep-alive, model switching | done: Gemma 4 E2B added; back from the host tier at 96% of a copy (L40S) |
| 9 | `pull`, catalog, conversion | under validation |
| 10 | ModernBERT encoder: Laya decisions, Granite embeddings | |
| 11 | Vision and audio inputs | |
| 12 | Larger tiers: 12B, 31B, FP8, MoE | 12a done: 12B 101.8 tok/s, 31B 41.7 tok/s (L40S); 12b (26B A4B MoE) under validation |
| 13 | Speculative decoding | |
| 14 | `tachyon serve`: other processes over shared memory, `interface/` | under validation |

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT), at your option.
