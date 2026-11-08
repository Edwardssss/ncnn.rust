use crate::{sys, Error, Result};
use std::{io::Read, marker::PhantomData, rc::Rc};

/// An owned, bounded binary reader for advanced native loading.
pub struct DataReader {
    raw: sys::ncnn_datareader_t,
    _thread: PhantomData<Rc<()>>,
}
impl DataReader {
    /// Copies the buffer into a native reader that outlives it.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let raw = unsafe { sys::ncnn_rs_reader_create(data.as_ptr(), data.len()) };
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
