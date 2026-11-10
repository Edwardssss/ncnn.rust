//! CPU inference and optional synchronous Vulkan inference with explicit ownership.
//!
//! [`Net`] owns a loaded graph, [`Extractor`] runs one request, and [`Mat`] holds tensor
//! storage. Owned input removes the aliasing question at the API boundary: [`Net::infer`]
//! returns independent vectors and [`Extractor::extract`] borrows a scored CPU output.
//!
//! The forms that return independent storage allocate and free an output-sized buffer per request,
//! which is the dominant cost of a small request on a host whose heap returns freed pages to the
//! kernel. [`heap::retain_pages`] keeps those pages mapped on glibc Linux,
//! [`Net::infer_borrowed_into`] removes the allocation by reusing a caller buffer, and
//! [`NetBuilder::reuse_blob_memory`] gives the network's intermediate tensors a pool it owns.
//!
//! ```compile_fail
//! let net = ncnn_rust::Net::new().unwrap();
//! std::thread::spawn(move || drop(net));
//! ```
//!
//! ```compile_fail
//! let ex;
//! {
//!     let net = ncnn_rust::Net::new().unwrap();
//!     ex = net.extractor().unwrap();
//! }
//! drop(ex);
//! ```
#![deny(missing_docs)]

pub mod cpu;
#[cfg(feature = "vulkan")]
pub mod gpu;
pub mod heap;
pub mod image;
pub use ncnn_sys as sys;

mod error;
mod layer;
mod mat;
mod net;
mod options;
mod reader;
pub use reader::DataReader;
mod param;

pub use error::{Error, Result};
pub use layer::{layer_type_index, Layer, ModelBin};
pub use mat::{
    BorderMode, Borders, ElementKind, Mat, MatView, PixelFormat, ReshapeView, TensorInfo,
};
pub use net::{
    BlobId, BlobInfo, Endpoint, Extractor, FinalOutput, LayerInfo, Net, NetBuilder, OutputGuard,
    OutputRange, ShapeInfo,
};
pub use options::{CpuOption, OptionFlag, Options};
pub use param::{ParamDict, ParamKind, PoolAllocator};

/// Returns the pinned native version.
pub fn version() -> &'static str {
    unsafe {
        std::ffi::CStr::from_ptr(sys::ncnn_version())
            .to_str()
            .unwrap_or("unknown")
    }
}

/// Returns the pinned native version as the integer ncnn encodes, which is the release date.
///
/// Compare against a known value to decide whether a newer declaration is available; the string
/// form is in [`version`].
pub fn version_number() -> i32 {
    unsafe { sys::ncnn_version_number() }
}

/// Returns the native logical CPU count.
pub fn cpu_count() -> usize {
    unsafe { sys::ncnn_rs_cpu_count().max(1) as usize }
}
