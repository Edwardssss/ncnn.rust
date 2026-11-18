//! The crate documentation is the package README, so the page crates.io renders and the page
//! rustdoc builds cannot drift apart: one file answers both.
//!
#![doc = include_str!("../README.md")]
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
