//! CPU inference and optional synchronous Vulkan inference with explicit ownership.
//!
//! [`Net`] owns a loaded graph, [`Extractor`] runs one request, and [`Mat`] holds tensor
//! storage. Owned input removes the aliasing question at the API boundary: [`Net::infer`]
//! returns independent vectors and [`Extractor::extract`] borrows a scored CPU output.
//!
//! A request has one storage decision per side. [`Extractor::extract`] borrows a read-only guard and
//! [`TensorWidth`] selects FP32 conversion or the stored width, [`Extractor::extract_mat`] keeps an
//! owned tensor, [`Extractor::extract_into`] appends to a caller vector and
//! [`Extractor::extract_owned`] returns values. [`Net::infer`] takes an [`Input`], which carries the
//! same choice on the input side: owned storage, borrowed FP32 values or a tensor sub view.
//!
//! One extractor serves repeated requests: [`Extractor::input`] restarts the native handle once a
//! request extracted, so a second request recomputes instead of returning the cached answer, and
//! [`Extractor::continue_session`] keeps that cache when a graph carries state across inputs. A
//! tensor exposes read-only sub views ([`Mat::channel`], [`Mat::row_range`] and their siblings),
//! which alias its storage: the slice readers borrow it, and the copying readers and the submit path
//! copy out of it.
//!
//! The forms that return independent storage allocate and free an output-sized buffer per request,
//! which is the dominant cost of a small request on a host whose heap returns freed pages to the
//! kernel. [`heap::retain_pages`] keeps those pages mapped on glibc Linux,
//! [`Net::infer_into`] removes the allocation by reusing a caller buffer, and
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
//!
//! ```compile_fail
//! let layer;
//! {
//!     let net = ncnn_rust::Net::new().unwrap();
//!     layer = net.layer_handle(0).unwrap();
//! }
//! drop(layer);
//! ```
#![deny(missing_docs)]

pub mod alloc;
pub mod cpu;
#[cfg(feature = "vulkan")]
pub mod gpu;
pub mod heap;
pub mod image;
/// Raw ncnn bindings and the project's C adapters.
///
/// These items follow ncnn's C ABI rather than the safe crate's contracts, and they carry no
/// independent semver guarantee: they move with this crate's version. The invariants in the safety
/// contracts apply when they are called directly.
///
/// The module is behind the `raw` feature, so the default build publishes no generated bindings.
#[cfg(feature = "raw")]
pub use ncnn_sys as sys;

// The crate's own code reaches the generated bindings under the same name either way: a `pub`
// re-export is visible inside the crate too, so the two arms never overlap.
#[cfg(not(feature = "raw"))]
pub(crate) use ncnn_sys as sys;

mod error;
mod layer;
mod mat;
mod net;
mod options;
mod reader;
pub use reader::DataReader;
mod param;

pub use error::{Error, Result};
pub use layer::{layer_type_index, GraphLayer, Layer, ModelBin};
pub use mat::{
    BorderMode, Borders, ElementKind, Mat, MatView, PixelFormat, ReshapeView, TensorInfo,
};
pub use net::{
    BlobId, BlobInfo, Endpoint, Extractor, FinalOutput, Input, LayerInfo, Net, NetBuilder,
    OutputGuard, OutputRange, ParamSource, ShapeInfo, TensorWidth, WeightSource,
};
pub use options::{CpuOption, OptionFlag, Options};
pub use param::{ParamDict, ParamKind};

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
