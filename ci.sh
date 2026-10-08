#!/usr/bin/env bash
# The quality gates of the spec (§2.2). Every commit passes them.
# GPU tests run when an NVIDIA device is visible; PTX is checked when ptxas is installed.
set -euo pipefail
cd "$(dirname "$0")"

step() { printf '\n== %s\n' "$*"; }

step format
cargo fmt --all --check

step lint
cargo clippy --workspace --all-targets --all-features -- -D warnings

step docs
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --quiet

step complexity
lizard --CCN 10 --length 50 --arguments 6 --warnings_only --languages rust crates/

step budget
scripts/budget.sh

step tests
cargo test --workspace --quiet

if [[ -e /dev/nvidiactl || -e /dev/dxg ]]; then
    step "GPU tests"
    cargo test --workspace --features tachyon/gpu --quiet
fi

if command -v ptxas > /dev/null; then
    step PTX
    for ptx in $(find crates target/ptx -name '*.ptx'); do
        for arch in sm_80 sm_86 sm_89 sm_90; do ptxas -arch="$arch" "$ptx" -o /dev/null; done
    done
fi

step "all gates passed"