//! CPU inference and optional synchronous Vulkan inference with explicit ownership.
//!
//! [`Net`] owns a loaded graph, [`Extractor`] runs one request, and [`Mat`] holds tensor
//! storage. Owned input removes the aliasing question at the API boundary: [`Net::infer`]
//! returns independent vectors and [`Extractor::extract`] borrows a scored CPU output.
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
pub use layer::{Layer, ModelBin};
pub use mat::{Mat, PixelFormat, ReshapeView, TensorInfo};
pub use net::{
    BlobId, Endpoint, Extractor, FinalOutput, Net, NetBuilder, OutputGuard, OutputRange,
};
pub use options::{CpuOption, OptionFlag, Options};
pub use param::{ParamDict, PoolAllocator};

/// Returns the pinned native version.
pub fn version() -> &'static str {
    unsafe {
        std::ffi::CStr::from_ptr(sys::ncnn_version())
            .to_str()
            .unwrap_or("unknown")
    }
}

/// Returns the native logical CPU count.
pub fn cpu_count() -> usize {
    unsafe { sys::ncnn_rs_cpu_count().max(1) as usize }
}
