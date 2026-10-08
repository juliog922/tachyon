//! Tachyon: an embedded inference engine for NVIDIA GPUs.
//!
//! Tachyon runs a fixed catalog of open-weight models (Gemma 4, Laya, Granite
//! embeddings) inside the calling process, at the speed the GPU's memory
//! bandwidth allows, with zero dependencies: it needs only `libc` and the
//! NVIDIA driver's `libcuda.so.1`, which it loads at run time.
//!
//! # Platform
//!
//! Linux on `x86_64` with an NVIDIA GPU of compute capability 8.0 or newer
//! (Ampere and later) and a driver supporting CUDA 12.0 or newer (R525+).
//! WSL2 works; the driver is found in `/usr/lib/wsl/lib`.
//!
//! # Status
//!
//! Development step 5 of the spec: [`cuda`], the driver layer every later
//! module builds on; [`wpk`], the weight file format and its loader; [`ptx`],
//! the decode kernels, with [`quant`] their weight layouts; and [`token`], the
//! tokenizer, with [`json`] to read model files. The inference API comes in later steps.

#[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
compile_error!("tachyon targets x86_64 Linux only");

pub mod cuda;
mod error;
pub mod json;
pub mod ptx;
pub mod quant;
mod sys;
pub mod token;
pub mod wpk;

pub use error::{Error, Result};