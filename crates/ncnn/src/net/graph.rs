//! Read-only description of a loaded network: its layers and its blobs.

use super::Net;
use crate::{sys, Error, Result};
use std::ffi::CStr;

/// One shape a layer recorded for one of its blobs.
///
/// A layer records the extents it resolved for its inputs and outputs when the graph carries shape
/// information, which is how a caller sees a layer contract before running it. `dims` stays at zero
/// when ncnn recorded no shape.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShapeInfo {
    /// Shape rank.
    pub dims: i32,
    /// Width extent.
    pub w: i32,
    /// Height extent.
    pub h: i32,
    /// Depth extent.
    pub d: i32,
    /// Channel extent.
    pub c: i32,
}
impl ShapeInfo {
    /// Reports whether ncnn recorded a shape.
    pub fn has_shape(&self) -> bool {
        self.dims > 0
    }
}

/// One layer of a loaded network.
///
/// The support flags are ncnn's declarations for that layer type, which is where the runtime
/// fallback decision starts: a layer without `support_vulkan` runs on the CPU even when the network
/// uses a device. They describe the layer type, not a promise about the selected kernel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerInfo {
    /// Layer name from the parameter text.
    pub name: String,
    /// Registered layer type, such as `Convolution`.
    pub kind: String,
    /// Registered type index.
    pub typeindex: i32,
    /// Shapes this layer recorded for its inputs, in bottom order.
    pub bottom_shapes: Vec<ShapeInfo>,
    /// Shapes this layer recorded for its outputs, in top order.
    pub top_shapes: Vec<ShapeInfo>,
    /// Number of bottom blobs the layer reads.
    pub bottoms: usize,
    /// Number of top blobs the layer writes.
    pub tops: usize,
    /// Declares a Vulkan implementation.
    pub support_vulkan: bool,
    /// Declares Vulkan support for packed layouts.
    pub support_vulkan_packing: bool,
    /// Declares support for packed CPU layouts.
    pub support_packing: bool,
    /// Declares support for FP16 storage.
    pub support_fp16_storage: bool,
    /// Declares support for BF16 storage.
    pub support_bf16_storage: bool,
    /// Reads exactly one bottom blob.
    pub one_blob_only: bool,
    /// Computes in place on its input blob.
    pub support_inplace: bool,
    /// Accepts a bottom blob whose layout has any element packing.
    pub support_any_packing: bool,
    /// Accepts any element packing on a device.
    pub support_vulkan_any_packing: bool,
    /// Accepts INT8 storage.
    pub support_int8_storage: bool,
    /// Accepts the shader tensor storage layout.
    pub support_tensor_storage: bool,
    /// Accepts a batch count above one.
    pub support_batch: bool,
}

/// One blob of a loaded network.
///
/// A blob that ncnn could not infer keeps `dims` at zero and zero extents, which happens when the
/// parameter text carries no shape hint and no layer produced a shape yet.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlobInfo {
    /// Blob name from the parameter text.
    pub name: String,
    /// Shape hint rank.
    pub dims: i32,
    /// Width extent.
    pub w: i32,
    /// Height extent.
    pub h: i32,
    /// Depth extent.
    pub d: i32,
    /// Channel extent.
    pub c: i32,
    /// Index of the layer that produces this blob, or `-1` for a graph input.
    pub producer: i32,
    /// Index of the first layer that consumes this blob.
    pub consumer: i32,
}
impl BlobInfo {
    /// Reports whether ncnn recorded a shape for this blob.
    pub fn has_shape(&self) -> bool {
        self.dims > 0
    }
}

impl Net {
    /// Describes every layer of the loaded network in graph order.
    ///
    /// Use the support flags to see which layer types can run on a device before selecting one;
    /// the answer is a property of the layer type, and native per-layer fallback still applies.
    /// Names are copied, so the result outlives later loads.
    pub fn layers(&self) -> Result<Vec<LayerInfo>> {
        if !self.ready {
            return Err(Error::invalid(
                "Load parameters and weights before reading the graph",
            ));
        }
        let count = unsafe { sys::ncnn_rs_net_layer_count(self.raw) };
        if count < 0 {
            return Err(Error::NativeAt("read the layer count", count));
        }
        let mut layers = Vec::with_capacity(count as usize);
        for index in 0..count {
            layers.push(self.layer_info(index)?);
        }
        Ok(layers)
    }
    fn layer_info(&self, index: i32) -> Result<LayerInfo> {
        let mut name = std::ptr::null();
        let mut kind = std::ptr::null();
        let mut typeindex = 0;
        let mut bottoms = 0;
        let mut tops = 0;
        let mut flags = 0;
        let code = unsafe {
            sys::ncnn_rs_net_layer_info(
                self.raw,
                index,
                &mut name,
                &mut kind,
                &mut typeindex,
                &mut bottoms,
                &mut tops,
                &mut flags,
            )
        };
        if code != 0 || name.is_null() || kind.is_null() {
            return Err(Error::NativeAt("read a layer", code));
        }
        let text = |pointer: *const std::os::raw::c_char| unsafe {
            CStr::from_ptr(pointer).to_string_lossy().into_owned()
        };
        let shape = |top: bool, blob: usize| -> Result<ShapeInfo> {
            let mut shape = ShapeInfo::default();
            let code = unsafe {
                sys::ncnn_rs_net_layer_shape(
                    self.raw,
                    index,
                    i32::from(top),
                    blob as i32,
                    &mut shape.dims,
                    &mut shape.w,
                    &mut shape.h,
                    &mut shape.d,
                    &mut shape.c,
                )
            };
            if code != 0 {
                return Err(Error::NativeAt("read a layer shape", code));
            }
            Ok(shape)
        };
        let bottom_shapes = (0..bottoms.max(0) as usize)
            .map(|blob| shape(false, blob))
            .collect::<Result<Vec<_>>>()?;
        let top_shapes = (0..tops.max(0) as usize)
            .map(|blob| shape(true, blob))
            .collect::<Result<Vec<_>>>()?;
        Ok(LayerInfo {
            name: text(name),
            kind: text(kind),
            typeindex,
            bottom_shapes,
            top_shapes,
            bottoms: bottoms.max(0) as usize,
            tops: tops.max(0) as usize,
            support_vulkan: flags & (1 << 0) != 0,
            support_vulkan_packing: flags & (1 << 1) != 0,
            support_packing: flags & (1 << 2) != 0,
            support_fp16_storage: flags & (1 << 3) != 0,
            support_bf16_storage: flags & (1 << 4) != 0,
            one_blob_only: flags & (1 << 5) != 0,
            support_inplace: flags & (1 << 6) != 0,
            support_any_packing: flags & (1 << 7) != 0,
            support_vulkan_any_packing: flags & (1 << 8) != 0,
            support_int8_storage: flags & (1 << 9) != 0,
            support_tensor_storage: flags & (1 << 10) != 0,
            support_batch: flags & (1 << 11) != 0,
        })
    }
    /// Describes every blob of the loaded network in graph order.
    ///
    /// `producer` and `consumer` are layer indices, so the pair of this call and
    /// [`layers`](Self::layers) describes the whole graph without running it.
    pub fn blobs(&self) -> Result<Vec<BlobInfo>> {
        if !self.ready {
            return Err(Error::invalid(
                "Load parameters and weights before reading the graph",
            ));
        }
        let count = unsafe { sys::ncnn_rs_net_blob_count(self.raw) };
        if count < 0 {
            return Err(Error::NativeAt("read the blob count", count));
        }
        let mut blobs = Vec::with_capacity(count as usize);
        for index in 0..count {
            let mut name = std::ptr::null();
            let mut dims = 0;
            let mut w = 0;
            let mut h = 0;
            let mut d = 0;
            let mut c = 0;
            let mut producer = 0;
            let mut consumer = 0;
            let code = unsafe {
                sys::ncnn_rs_net_blob_info(
                    self.raw,
                    index,
                    &mut name,
                    &mut dims,
                    &mut w,
                    &mut h,
                    &mut d,
                    &mut c,
                    &mut producer,
                    &mut consumer,
                )
            };
            if code != 0 || name.is_null() {
                return Err(Error::NativeAt("read a blob", code));
            }
            blobs.push(BlobInfo {
                name: unsafe { CStr::from_ptr(name) }
                    .to_string_lossy()
                    .into_owned(),
                dims,
                w,
                h,
                d,
                c,
                producer,
                consumer,
            });
        }
        Ok(blobs)
    }
}
