use crate::{sys, Error, Result};
use std::{io::Read, marker::PhantomData, rc::Rc};

/// An owned, bounded binary reader for advanced native loading.
pub struct DataReader {
    raw: sys::ncnn_datareader_t,
    _thread: PhantomData<Rc<()>>,
}
impl DataReader {
    /// Copies the buffer into a native reader that outlives it and supports both
    /// formatted scanning and raw reads, so parameter text and weight streams load
    /// through the same handle.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let raw = unsafe { sys::ncnn_rs_reader_memory(data.as_ptr(), data.len()) };
        if raw.is_null() {
            return Err(Error::Native(-100));
        }
        Ok(Self {
            raw,
            _thread: PhantomData,
        })
    }
    /// Reads the whole stream into an owned buffer.
    pub fn from_reader(mut reader: impl Read) -> Result<Self> {
        let mut data = Vec::new();
        reader.read_to_end(&mut data)?;
        Self::from_bytes(&data)
    }
    /// Opens a file as a native stdio reader.
    ///
    /// The reader owns the file handle and closes it on drop, so a stream that does not fit a Rust
    /// buffer loads without an intermediate copy. The path must be valid UTF-8.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = path
            .to_str()
            .ok_or_else(|| Error::invalid("The native reader needs a UTF-8 path"))?;
        let path = std::ffi::CString::new(text)?;
        let raw = unsafe { sys::ncnn_rs_reader_file(path.as_ptr()) };
        if raw.is_null() {
            return Err(Error::Native(-100));
        }
        Ok(Self {
            raw,
            _thread: PhantomData,
        })
    }
    /// Borrows the native reader handle.
    pub fn as_raw(&self) -> sys::ncnn_datareader_t {
        self.raw
    }
}
impl Drop for DataReader {
    fn drop(&mut self) {
        unsafe {
            sys::ncnn_datareader_destroy(self.raw);
        }
    }
}
