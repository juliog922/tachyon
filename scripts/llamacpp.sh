#!/usr/bin/env bash
# The step-6 and step-7 comparisons: llama.cpp with CUDA, built at a pinned commit, running Gemma 4 E4B in Q4_0 (4.5
# bits per weight; Tachyon's Q4 is 4.25), measured by llama-bench as Tachyon's `generate` bench measures it: decoding
# 256 tokens at batch 1 (tg256), and prompts of 64 to 8192 tokens (pp64…pp8192) for the time to first token, which
# Tachyon's figure exceeds by one decode step.
# Needs git, cmake, a C++ compiler, the CUDA Toolkit (nvcc) and Python with torch and transformers.
#
#   scripts/llamacpp.sh ~/models/gemma-4-E4B-it
#
# Builds and converts under $TACHYON_BENCH_DIR (default target/bench); later runs reuse both, and a Q4_0 file copied
# there from another machine skips the conversion. llama.cpp's Q4_0 keeps both embedding tables in Q6_K (5.5 bits per
# weight overall), so it reads more bytes per token than Tachyon.
set -euo pipefail
commit=23b0202a189c44a54625aadcb37a946dd1d6278d # tag b11552, 2026-10-10
src=${1:?usage: scripts/llamacpp.sh <Hugging Face checkpoint directory>}
dir=${TACHYON_BENCH_DIR:-target/bench}/llama.cpp
command -v nvcc >/dev/null || { echo "llama.cpp's CUDA build needs nvcc, from the CUDA Toolkit" >&2; exit 1; }
[[ -d $dir ]] || git clone -q https://github.com/ggml-org/llama.cpp "$dir"
git -C "$dir" fetch -q origin "$commit" && git -C "$dir" checkout -q "$commit"
cmake -S "$dir" -B "$dir/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF >/dev/null
cmake --build "$dir/build" -j "$(nproc)" --target llama-quantize llama-bench >/dev/null
bf16=$dir/gemma-4-e4b-bf16.gguf q4=$dir/gemma-4-e4b-q4_0.gguf
if [[ ! -f $q4 ]]; then
    [[ -f $bf16 ]] || python3 "$dir/convert_hf_to_gguf.py" "$src" --outtype bf16 --outfile "$bf16"
    "$dir/build/bin/llama-quantize" "$bf16" "$q4" Q4_0 >/dev/null
fi
"$dir/build/bin/llama-bench" -m "$q4" -ngl 99 -fa 1 -p 64,256,512,1024,2048,4096,8192 -n 256 -r 5