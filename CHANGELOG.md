# Changelog

All notable changes to this project are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `tachyon::wpk`: the `.wpk` weight file (4 KiB-aligned tensors, checked header and table, CRC32C per tensor) with `Writer`, `Wpk::open`, `Wpk::verify`.
- `wpk::Loader`: `io_uring` reads with `O_DIRECT` straight into pinned memory; disk → VRAM through staging slots overlapped with uploads, disk → host-RAM tier, and host tier → VRAM. Falls back to the page cache where `O_DIRECT` is refused.
- `load` benchmark: every load path on a file sized to the machine, with the step-2 exit gate.
- `tachyon::cuda`: the CUDA Driver API loaded at run time from `libcuda.so.1` (or `/usr/lib/wsl/lib`, or `$TACHYON_LIBCUDA`), with every function fetched at the CUDA 12.0 ABI so newer drivers keep working.
- Owning, non-`Send` wrappers: `Context` (primary context), `DevBuf`, `HostBuf` (pinned, mappable), `Stream`, `Event`, `Module` (PTX with the JIT error log), `Graph` (stream capture and replay).
- `tachyon gpu [--json]`: the GPUs the driver sees, with compute capability, memory, datasheet bandwidth and PCIe link.
- `machine` benchmark: VRAM read bandwidth, PCIe in both directions, launch and graph-replay cost, with the step-1 exit gate.
- Bench harness with `perf` counters, JSON results and a 3% regression check against per-machine baselines.
- `ci.sh`: format, clippy pedantic, docs, complexity, line budgets, tests, PTX checks.