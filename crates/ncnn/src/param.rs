use crate::{sys, Error, Mat, Result};
use std::{marker::PhantomData, rc::Rc};

/// Owns a native pool allocator and releases it after its allocations.
pub struct PoolAllocator {
    raw: sys::ncnn_allocator_t,
    _thread: PhantomData<Rc<()>>,
}
impl PoolAllocator {
    /// Creates a pool allocator with native defaults.
    pub fn new() -> Result<Self> {
        let raw = unsafe { sys::ncnn_allocator_create_pool_allocator() };
        if raw.is_null() {
            return Err(Error::Native(-100));
        }
        Ok(Self {
            raw,
            _thread: PhantomData,
        })
    }
    /// Sets the native reuse ratio and size threshold.
    pub fn configure(&mut self, ratio: f32, threshold: usize) -> Result<()> {
        unsafe { crate::error::status(sys::ncnn_rs_pool_configure(self.raw, 0, ratio, threshold)) }
    }
    /// Releases every cached block that has no outstanding allocation.
    pub fn clear(&mut self) {
        unsafe {
            sys::ncnn_rs_pool_clear(self.raw, 0);
        }
    }
    /// Borrows the allocator for advanced FFI allocation paths.
    pub fn as_raw(&self) -> sys::ncnn_allocator_t {
        self.raw
    }
}
impl Drop for PoolAllocator {
    fn drop(&mut self) {
        unsafe {
            sys::ncnn_allocator_destroy(self.raw);
        }
    }
}

/// Storage form of one parameter dictionary entry.
///
/// The variant follows the native type code, which decides the getter a layer uses. A binary
/// parameter stream carries an untyped scalar that both scalar getters read, while a text entry
/// stores an integer, a float, a tensor or a string.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ParamKind {
    /// Untyped binary scalar, readable through either scalar getter.
    Scalar,
    /// Integer value.
    Int,
    /// Single precision value.
    Float,
    /// Tensor or binary array value.
    Array,
    /// Text value.
    Text,
}

/// Owns a native parameter dictionary for standalone layers.
pub struct ParamDict {
    raw: sys::ncnn_paramdict_t,
    _thread: PhantomData<Rc<()>>,
}
impl ParamDict {
    /// Creates an empty dictionary.
    pub fn new() -> Result<Self> {
        let raw = unsafe { sys::ncnn_paramdict_create() };
        if raw.is_null() {
            return Err(Error::Native(-100));
        }
        Ok(Self {
            raw,
            _thread: PhantomData,
        })
    }
    fn check(id: i32) -> Result<()> {
        if !(0..32).contains(&id) {
            return Err(Error::invalid("Parameter id must be 0..32"));
        }
        Ok(())
    }
    /// Stores an integer parameter at the given id.
    pub fn set_int(&mut self, id: i32, value: i32) -> Result<()> {
        Self::check(id)?;
        unsafe {
            sys::ncnn_paramdict_set_int(self.raw, id, value);
        }
        Ok(())
    }
    /// Stores a float parameter at the given id.
    pub fn set_float(&mut self, id: i32, value: f32) -> Result<()> {
        Self::check(id)?;
        unsafe {
            sys::ncnn_paramdict_set_float(self.raw, id, value);
        }
        Ok(())
    }
    /// Reports how a parameter is stored, or `None` when the identifier is unset.
    ///
    /// A custom layer reads parameters by identifier, so this decides which getter applies.
    pub fn kind(&self, id: i32) -> Result<Option<ParamKind>> {
        Self::check(id)?;
        let code = unsafe { sys::ncnn_paramdict_get_type(self.raw, id) };
        Ok(match code {
            1 => Some(ParamKind::Scalar),
            2 => Some(ParamKind::Int),
            3 => Some(ParamKind::Float),
            4..=6 => Some(ParamKind::Array),
            7 => Some(ParamKind::Text),
            _ => None,
        })
    }
    /// Reads an integer parameter, falling back to the default.
    pub fn get_int(&self, id: i32, default: i32) -> Result<i32> {
        Self::check(id)?;
        unsafe { Ok(sys::ncnn_paramdict_get_int(self.raw, id, default)) }
    }
    /// Reads a float parameter, falling back to the default.
    pub fn get_float(&self, id: i32, default: f32) -> Result<f32> {
        Self::check(id)?;
        unsafe { Ok(sys::ncnn_paramdict_get_float(self.raw, id, default)) }
    }
    /// Stores an owned copy of a tensor parameter.
    pub fn set_array(&mut self, id: i32, value: &Mat) -> Result<()> {
        Self::check(id)?;
        let copy = value.deep_clone()?;
        unsafe {
            sys::ncnn_paramdict_set_array(self.raw, id, copy.raw);
        }
        Ok(())
    }
    /// Stores a byte string parameter.
    pub fn set_bytes(&mut self, id: i32, value: &[u8]) -> Result<()> {
        Self::check(id)?;
        unsafe {
            crate::error::status(sys::ncnn_rs_param_set_string(
                self.raw,
                id,
                value.as_ptr(),
                value.len(),
            ))
        }
    }
    /// Copies a byte string parameter.
    pub fn get_bytes(&self, id: i32) -> Result<Vec<u8>> {
        Self::check(id)?;
        unsafe {
            let size = sys::ncnn_rs_param_get_string(self.raw, id, std::ptr::null_mut(), 0);
            if size == usize::MAX {
                return Err(Error::Native(-100));
            }
            let mut bytes = vec![0; size];
            if sys::ncnn_rs_param_get_string(self.raw, id, bytes.as_mut_ptr(), size) != size {
                return Err(Error::Native(-100));
            }
            Ok(bytes)
        }
    }
    /// Borrows the native dictionary handle.
    pub fn as_raw(&self) -> sys::ncnn_paramdict_t {
        self.raw
    }
}
impl Drop for ParamDict {
    fn drop(&mut self) {
        unsafe {
            sys::ncnn_paramdict_destroy(self.raw);
        }
    }
}
