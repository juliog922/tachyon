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
    /// A file operation failed with this `errno`.
    Io(i32),
    /// A file is not a valid `.wpk`, or its contents do not match their checksums.
    Format(String),
    /// A prompt does not fit the context the model was loaded with.
    TooLong {
        /// The prompt's tokens.
        tokens: usize,
        /// Positions the context holds.
        context: usize,
    },
    /// A model does not fit the engine's VRAM limit beside the models it may not evict (pinned, or in use).
    OutOfMemory {
        /// Bytes the model needs.
        need: usize,
        /// Bytes free under the limit once every model that may go has gone.
        free: usize,
    },
    /// No model of this name is loaded.
    NoModel(String),
    /// A model needs a feature Tachyon does not run yet; carries the feature.
    Unsupported(String),
    /// An image or a recording is damaged, or in a format Tachyon does not decode; carries what.
    Media(String),
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
            Error::Io(errno) => write!(f, "{}", std::io::Error::from_raw_os_error(*errno)),
            Error::Format(why) => write!(f, "invalid weight file: {why}"),
            Error::TooLong { tokens, context } => write!(f, "the prompt has {tokens} tokens; the context holds {context} positions"),
            Error::OutOfMemory { .. } | Error::NoModel(_) | Error::Unsupported(_) | Error::Media(_) => self.engine(f),
        }
    }
}

impl Error {
    /// The message of an engine error.
    fn engine(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::OutOfMemory { need, free } => write!(f, "the model needs {} MiB of VRAM; {} MiB can be freed under the limit", need >> 20, free >> 20),
            Error::NoModel(name) => write!(f, "no model named {name:?} is loaded"),
            Error::Unsupported(what) => write!(f, "the model has {what}, which tachyon does not run yet"),
            Error::Media(what) => write!(f, "media: {what}"),
            _ => Ok(()),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Error {
        const EIO: i32 = 5;
        match e.kind() {
            std::io::ErrorKind::UnexpectedEof => Error::Format("the file ends early".into()),
            _ => Error::Io(e.raw_os_error().unwrap_or(EIO)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Error;

    #[test]
    fn messages_name_the_versions() {
        assert!(Error::DriverTooOld(11080).to_string().contains("CUDA 11.8"));
        assert!(Error::UnsupportedGpu { major: 7, minor: 5 }.to_string().contains("7.5"));
    }
}
