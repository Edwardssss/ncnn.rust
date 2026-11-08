use std::{ffi::NulError, fmt};

#[derive(Debug)]
/// Failure from the native layer or from a validated Rust check.
pub enum Error {
    /// A native call failed without a named operation.
    Native(i32),
    /// A native call failed during a named operation.
    NativeAt(&'static str, i32),
    /// A caller argument or state failed validation.
    Invalid(&'static str),
    /// The platform lacks the requested capability.
    Unsupported(&'static str),
    /// A string could not be converted to a C string.
    Nul(NulError),
    /// A file or buffer operation failed.
    Io(std::io::Error),
}
/// Result of a fallible wrapper operation.
pub type Result<T> = std::result::Result<T, Error>;
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Native(code) => write!(f, "ncnn returned {code}"),
            Self::NativeAt(operation, code) => write!(f, "ncnn {operation} returned {code}"),
            Self::Invalid(message) => f.write_str(message),
            Self::Unsupported(message) => write!(f, "Unsupported capability: {message}"),
            Self::Nul(error) => error.fmt(f),
            Self::Io(error) => error.fmt(f),
        }
    }
}
impl Error {
    /// Returns the upstream ncnn status code carried by a native failure.
    pub fn native_code(&self) -> Option<i32> {
        match self {
            Self::Native(code) | Self::NativeAt(_, code) => Some(*code),
            _ => None,
        }
    }
}
impl std::error::Error for Error {}
impl From<NulError> for Error {
    fn from(e: NulError) -> Self {
        Self::Nul(e)
    }
}
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
pub(crate) fn status_at(operation: &'static str, code: i32) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(Error::NativeAt(operation, code))
    }
}
pub(crate) fn status(code: i32) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(Error::Native(code))
    }
}
