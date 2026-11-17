//! CPU-side allocators.

use crate::{sys, Error, Result};
use std::{marker::PhantomData, rc::Rc};

/// Owns a native pool allocator and releases it after its allocations.
///
/// A pool serves the intermediate blobs, workspace or attention KV cache that a network or an
/// extractor allocates. [`Options`](crate::Options) and the per-extractor setters store the native
/// handle without taking ownership, so the pool must outlive every network and extractor configured
/// from it and must not be shared with one that runs concurrently. The type is thread-affine, like
/// the objects that use it.
///
/// ```
/// let mut pool = ncnn_rust::alloc::Pool::new()?;
/// pool.configure(0.5, 1024 * 1024)?;
/// let mut options = ncnn_rust::Options::new()?;
/// // SAFETY: `pool` outlives these options and every network configured from them.
/// unsafe { options.set_blob_allocator(&pool)? };
/// let net = ncnn_rust::NetBuilder::with_options(options).load_bytes(
///     "7767517\n2 2\nInput input 0 1 input\nReLU relu 1 1 input output\n",
///     &[],
/// )?;
/// assert_eq!(
///     net.infer([("input", ncnn_rust::Mat::from_f32(&[-1.0, 2.0], &[2])?)], &["output"])?[0].values,
///     [0.0, 2.0]
/// );
/// # Ok::<(), ncnn_rust::Error>(())
/// ```
pub struct Pool {
    raw: sys::ncnn_allocator_t,
    _thread: PhantomData<Rc<()>>,
}

impl Pool {
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

impl Drop for Pool {
    fn drop(&mut self) {
        unsafe {
            sys::ncnn_allocator_destroy(self.raw);
        }
    }
}
