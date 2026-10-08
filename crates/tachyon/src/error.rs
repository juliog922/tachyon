//! The crate's one error type.

use std::fmt;

/// Everything that can go wrong in Tachyon.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// The NVIDIA driver (`libcuda.so.1`) was not found.
    NoDriver,
    /// The driver supports a CUDA version older than 12.0; carries that version (`11080` for 11.8).
    DriverTooOld(i32),
    /// The GPU's compute capability is below 8.0.
    UnsupportedGpu {
        /// Major compute capability.
        major: u32,
        /// Minor compute capability.
        minor: u32,
    },
    /// A copy reaches past the end of a buffer.
    OutOfRange,
    /// The driver rejected a PTX module; carries the JIT compiler's log.
    Ptx(String),
    /// A driver call failed with this `CUresult` code.
    Cuda(i32),
}

/// `Result` with [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NoDriver => f.write_str("the NVIDIA driver (libcuda.so.1) was not found; install it or set TACHYON_LIBCUDA to its path"),
            Error::DriverTooOld(v) => write!(f, "the NVIDIA driver supports CUDA {}.{}; tachyon needs 12.0 or newer (driver R525+)", v / 1000, v % 1000 / 10),
            Error::UnsupportedGpu { major, minor } => write!(f, "compute capability {major}.{minor} is not supported; tachyon needs 8.0 or newer"),
            Error::OutOfRange => f.write_str("a copy reaches past the end of a buffer"),
            Error::Ptx(log) => write!(f, "the driver rejected the PTX: {log}"),
            Error::Cuda(code) => write!(f, "CUDA error {code} ({})", crate::cuda::error_name(*code)),
        }
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::Error;

    #[test]
    fn messages_name_the_versions() {
        assert!(Error::DriverTooOld(11080).to_string().contains("CUDA 11.8"));
        assert!(Error::UnsupportedGpu { major: 7, minor: 5 }.to_string().contains("7.5"));
    }
}
