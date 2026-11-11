use crate::{error::status, sys, Error, Mat, Options, ParamDict, Result};
use std::{ffi::CString, marker::PhantomData, rc::Rc};

/// Returns the registered type index of a layer type.
///
/// The index identifies the type in a model and is what
/// [`Net::register_custom_layer_by_typeindex`](crate::Net::register_custom_layer_by_typeindex)
/// replaces. An unknown name returns [`Error::Invalid`].
pub fn layer_type_index(name: &str) -> Result<i32> {
    let name = CString::new(name)?;
    let index = unsafe { sys::ncnn_layer_type_to_index(name.as_ptr()) };
    if index < 0 {
        return Err(Error::invalid(format!(
            "Unknown layer type {}",
            name.to_string_lossy()
        )));
    }
    Ok(index)
}

/// A single CPU layer with automatic pipeline cleanup.
pub struct Layer {
    raw: sys::ncnn_layer_t,
    options: Options,
    pipeline: bool,
    _thread: PhantomData<Rc<()>>,
}
impl Layer {
    /// Creates a built-in CPU layer.
    pub fn new(name: &str, options: Options) -> Result<Self> {
        let name = CString::new(name)?;
        let raw = unsafe { sys::ncnn_layer_create_by_type(name.as_ptr()) };
        if raw.is_null() {
            return Err(Error::invalid("Unknown layer type"));
        }
        Ok(Self {
            raw,
            options,
            pipeline: false,
            _thread: PhantomData,
        })
    }
    /// Loads layer-specific parameters before pipeline creation.
    ///
    /// # Safety
    /// Parameters must satisfy the selected layer's shape, type, and range contract.
    pub unsafe fn load_param(&mut self, parameters: &ParamDict) -> Result<()> {
        if self.pipeline {
            return Err(Error::invalid("Load parameters before pipeline creation"));
        }
        status((*self.raw)
            .load_param
            .ok_or(Error::invalid("Missing load_param callback"))?(
            self.raw,
            parameters.as_raw(),
        ))
    }
    /// Loads weights through an upstream ModelBin.
    ///
    /// # Safety
    /// The model and its reader must be valid for this call. Borrowed weight storage
    /// and allocator dependencies must remain valid until the layer is destroyed.
    pub unsafe fn load_model(&mut self, model: sys::ncnn_modelbin_t) -> Result<()> {
        if self.pipeline {
            return Err(Error::invalid("Load weights before pipeline creation"));
        }
        status(sys::ncnn_rs_layer_load_model(self.raw, model))
    }
    /// Creates a pipeline after parameter and weight loading.
    ///
    /// # Safety
    /// Required parameters and weights must be loaded according to the layer contract.
    pub unsafe fn create_pipeline(&mut self) -> Result<()> {
        if self.pipeline {
            return Err(Error::invalid("Pipeline already created"));
        }
        // Mark before calling so partial pipeline construction is also cleaned up.
        self.pipeline = true;
        status((*self.raw)
            .create_pipeline
            .ok_or(Error::invalid("Missing create_pipeline callback"))?(
            self.raw,
            self.options.raw,
        ))
    }
    /// Executes a one-input, one-output CPU layer.
    ///
    /// # Safety
    /// The layer must support this forward form and the input's dimensions, scalar
    /// format, and packing. The native callback must initialize the output elements.
    pub unsafe fn forward(&mut self, input: &Mat) -> Result<Mat> {
        if !self.pipeline {
            return Err(Error::invalid("Create the pipeline before forward"));
        }
        let input = input.deep_clone()?;
        let mut raw = std::ptr::null_mut();
        let code = (*self.raw)
            .forward_1
            .ok_or(Error::invalid("Missing forward callback"))?(
            self.raw,
            input.raw,
            &mut raw,
            self.options.raw,
        );
        if code != 0 {
            if !raw.is_null() {
                sys::ncnn_mat_destroy(raw);
            }
            return Err(Error::Native(code));
        }
        Mat::owned(raw, false)?.deep_clone()
    }
    /// Borrows the native layer handle.
    pub fn as_raw(&self) -> sys::ncnn_layer_t {
        self.raw
    }
}
impl Drop for Layer {
    fn drop(&mut self) {
        unsafe {
            if self.pipeline {
                if let Some(destroy) = (*self.raw).destroy_pipeline {
                    destroy(self.raw, self.options.raw);
                }
            }
            sys::ncnn_layer_destroy(self.raw);
        }
    }
}

/// A layer a loaded graph owns, borrowed for native calls.
///
/// [`Net::layer_handle`](crate::Net::layer_handle) builds one for an index the graph loaded. The
/// native layer belongs to the network and stays valid until the graph is cleared or reloaded,
/// while this value owns only the C handle around it, which releases on drop and never destroys the
/// layer. The borrow of the network keeps the layer alive for as long as the handle exists. A
/// registered custom layer keeps its userdata, so a native call that rewrites userdata breaks the
/// registration.
pub struct GraphLayer<'net> {
    raw: sys::ncnn_layer_t,
    _net: std::marker::PhantomData<&'net crate::Net>,
}
impl<'net> GraphLayer<'net> {
    pub(crate) fn borrowed(raw: sys::ncnn_layer_t) -> Self {
        Self {
            raw,
            _net: std::marker::PhantomData,
        }
    }
    /// Borrows the native layer handle.
    ///
    /// The handle is the one the C API expects. Native calls that write through it, or that read a
    /// tensor, follow the same rules as the rest of the raw module: the graph stays loaded and no
    /// request runs on the network meanwhile.
    pub fn as_raw(&self) -> sys::ncnn_layer_t {
        self.raw
    }
}
impl Drop for GraphLayer<'_> {
    fn drop(&mut self) {
        // Only the C handle is released; the layer stays with the network.
        unsafe { sys::ncnn_rs_borrowed_layer_release(self.raw) };
    }
}

/// A ModelBin backed by owned, deep-copied tensors.
pub struct ModelBin {
    raw: sys::ncnn_modelbin_t,
    _weights: Vec<Mat>,
    _handles: Vec<sys::ncnn_mat_t>,
}
impl ModelBin {
    /// Builds a reader over owned copies of the supplied tensors.
    pub fn from_mats(weights: &[&Mat]) -> Result<Self> {
        let count = i32::try_from(weights.len()).map_err(|_| Error::invalid("Too many weights"))?;
        let owned: Vec<Mat> = weights
            .iter()
            .map(|m| m.deep_clone())
            .collect::<Result<_>>()?;
        let handles: Vec<_> = owned.iter().map(Mat::as_raw).collect();
        let raw = unsafe { sys::ncnn_rs_modelbin_from_mats(handles.as_ptr(), count) };
        if raw.is_null() {
            return Err(Error::Native(-100));
        }
        Ok(Self {
            raw,
            _weights: owned,
            _handles: handles,
        })
    }
    /// Borrows the native reader handle.
    pub fn as_raw(&self) -> sys::ncnn_modelbin_t {
        self.raw
    }
}
impl Drop for ModelBin {
    fn drop(&mut self) {
        unsafe {
            sys::ncnn_modelbin_destroy(self.raw);
        }
    }
}
