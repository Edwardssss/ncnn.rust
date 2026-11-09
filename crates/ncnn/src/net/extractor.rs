//! Extractor operations, scoped outputs and guard lifetime rules.

use crate::{error::status_at, sys, Error, Mat, Result};

use super::{BlobId, Endpoint, FinalOutput, Net};
use crate::net::blob::BlobKind;
use std::{cell::Cell, ffi::CString, marker::PhantomData};

/// Runs one request against a borrowed network.
pub struct Extractor<'net> {
    pub(crate) raw: sys::ncnn_extractor_t,
    pub(crate) _net: &'net Net,
    pub(crate) low_copy: Cell<bool>,
}

enum Target {
    Named(CString),
    Indexed(i32),
}

impl Extractor<'_> {
    fn resolve(&self, endpoint: Endpoint<'_>, output: bool) -> Result<Target> {
        match endpoint {
            Endpoint::Name(name) => Ok(Target::Named(CString::new(name)?)),
            Endpoint::Index(index) => {
                self.ensure_exposed_index(index, !output)?;
                Ok(Target::Indexed(index))
            }
            Endpoint::Blob(blob) => {
                let kind = if output {
                    BlobKind::Output
                } else {
                    BlobKind::Input
                };
                Ok(Target::Indexed(self.validate_token(blob, kind)?))
            }
        }
    }
    fn validate_token(&self, blob: &BlobId, kind: BlobKind) -> Result<i32> {
        if !blob.belongs_to(&self._net.graph) {
            return Err(Error::Invalid(
                "Blob identifier belongs to another graph or an earlier load",
            ));
        }
        if blob.kind() != kind {
            return Err(Error::Invalid(
                "Blob identifier was resolved for the other endpoint kind",
            ));
        }
        Ok(blob.index())
    }
    fn ensure_exposed_index(&self, index: i32, input: bool) -> Result<()> {
        if !self._net.ready {
            return Err(Error::Invalid(
                "Load the graph and weights before using blob indices",
            ));
        }
        let kind_inputs = unsafe { sys::ncnn_net_get_input_count(self._net.raw) };
        for slot in 0..kind_inputs {
            if index >= 0 && unsafe { sys::ncnn_net_get_input_index(self._net.raw, slot) } == index
            {
                return Ok(());
            }
        }
        if input {
            return Err(Error::Invalid("Index is not an exposed input"));
        }
        let outputs = unsafe { sys::ncnn_net_get_output_count(self._net.raw) };
        for slot in 0..outputs {
            if index >= 0 && unsafe { sys::ncnn_net_get_output_index(self._net.raw, slot) } == index
            {
                return Ok(());
            }
        }
        Err(Error::Invalid("Index is not an exposed output"))
    }
    fn ensure_not_reserved(&self) -> Result<()> {
        if self._net.guard.output_borrowed.get() {
            return Err(Error::Invalid("A scoped output reserves this network"));
        }
        Ok(())
    }
    fn ensure_borrowable(&self) -> Result<()> {
        if self._net.guard.extractors.get() != 1 || self._net.guard.output_borrowed.get() {
            return Err(Error::Invalid(
                "Scoped output requires the network's only live extractor",
            ));
        }
        if !self.low_copy.get() || !self._net.guard.low_copy.get() {
            return Err(Error::Invalid(
                "Low-copy outputs require safe configuration and built-in layers without raw-handle export",
            ));
        }
        Ok(())
    }
    fn extract_native(&self, target: &Target, stored: bool) -> Result<Mat> {
        self.ensure_not_reserved()?;
        let mut raw = std::ptr::null_mut();
        let type_mask = i32::from(stored);
        let code = unsafe {
            match target {
                Target::Named(name) => {
                    if stored {
                        sys::ncnn_rs_extractor_extract_stored(
                            self.raw,
                            name.as_ptr(),
                            type_mask,
                            &mut raw,
                        )
                    } else {
                        sys::ncnn_extractor_extract(self.raw, name.as_ptr(), &mut raw)
                    }
                }
                Target::Indexed(index) => {
                    if stored {
                        sys::ncnn_rs_extractor_extract_stored_index(
                            self.raw, *index, type_mask, &mut raw,
                        )
                    } else {
                        sys::ncnn_extractor_extract_index(self.raw, *index, &mut raw)
                    }
                }
            }
        };
        if code != 0 {
            if !raw.is_null() {
                unsafe {
                    sys::ncnn_mat_destroy(raw);
                }
            }
            return Err(Error::NativeAt("extract output", code));
        }
        unsafe { Mat::owned(raw, false) }
    }

    /// Consumes an owned input, copying raw-adopted storage.
    pub fn input<'a>(&mut self, endpoint: impl Into<Endpoint<'a>>, mat: Mat) -> Result<()> {
        let target = self.resolve(endpoint.into(), false)?;
        self.ensure_not_reserved()?;
        let mat = if mat.is_transferable() {
            mat
        } else {
            mat.deep_clone()?
        };
        unsafe {
            status_at(
                "submit input",
                match &target {
                    Target::Named(name) => {
                        sys::ncnn_extractor_input(self.raw, name.as_ptr(), mat.raw)
                    }
                    Target::Indexed(index) => {
                        sys::ncnn_extractor_input_index(self.raw, *index, mat.raw)
                    }
                },
            )
        }
    }
    /// Submits inputs in order; earlier inputs stay cached on failure.
    pub fn inputs<'a>(
        &mut self,
        inputs: impl IntoIterator<Item = (impl Into<Endpoint<'a>>, Mat)>,
    ) -> Result<()> {
        for (endpoint, mat) in inputs {
            self.input(endpoint.into(), mat)?;
        }
        Ok(())
    }
    /// Borrows eligible CPU output through a scoped guard.
    pub fn extract<'a>(&mut self, endpoint: impl Into<Endpoint<'a>>) -> Result<OutputGuard<'_>> {
        self.ensure_borrowable()?;
        let target = self.resolve(endpoint.into(), true)?;
        let mat = self.extract_native(&target, false)?;
        self._net.guard.output_borrowed.set(true);
        Ok(OutputGuard {
            mat,
            network_borrowed: &self._net.guard.output_borrowed,
            _extractor: PhantomData,
        })
    }
    /// Borrows an output in the layout the producing layer wrote.
    ///
    /// Native extraction converts packed, FP16, BF16 and INT8 storage to FP32 unless the
    /// caller asks for stored storage, so this form keeps the element width in
    /// [`TensorInfo`](crate::TensorInfo) and the caller names the scalar format through
    /// [`OutputGuard::to_f32_typed`] or [`OutputGuard::to_i8`].
    pub fn extract_stored<'a>(
        &mut self,
        endpoint: impl Into<Endpoint<'a>>,
    ) -> Result<OutputGuard<'_>> {
        self.ensure_borrowable()?;
        let target = self.resolve(endpoint.into(), true)?;
        let mat = self.extract_native(&target, true)?;
        self._net.guard.output_borrowed.set(true);
        Ok(OutputGuard {
            mat,
            network_borrowed: &self._net.guard.output_borrowed,
            _extractor: PhantomData,
        })
    }
    /// Returns validated values with independent owned storage.
    pub fn extract_owned<'a>(&mut self, endpoint: impl Into<Endpoint<'a>>) -> Result<FinalOutput> {
        let target = self.resolve(endpoint.into(), true)?;
        let mat = self.extract_native(&target, false)?;
        Ok(FinalOutput {
            info: mat.info(),
            values: mat.to_f32()?,
        })
    }
    /// Replaces a caller vector with the output and returns its shape metadata.
    pub fn extract_into<'a>(
        &mut self,
        endpoint: impl Into<Endpoint<'a>>,
        buffer: &mut Vec<f32>,
    ) -> Result<crate::TensorInfo> {
        let target = self.resolve(endpoint.into(), true)?;
        let mat = self.extract_native(&target, false)?;
        let info = mat.info();
        mat.to_f32_into(buffer)?;
        Ok(info)
    }
    /// Appends the output to a caller vector and returns its shape metadata.
    pub fn extract_append<'a>(
        &mut self,
        endpoint: impl Into<Endpoint<'a>>,
        buffer: &mut Vec<f32>,
    ) -> Result<crate::TensorInfo> {
        let target = self.resolve(endpoint.into(), true)?;
        let mat = self.extract_native(&target, false)?;
        let info = mat.info();
        mat.append_f32_to(buffer)?;
        Ok(info)
    }
    /// Consumes outputs one at a time through scoped guards.
    ///
    /// ```compile_fail
    /// fn escape(ex: &mut ncnn_rust::Extractor<'_>) {
    ///     let mut escaped = None;
    ///     ex.outputs_with(&["output"], |_, output| {
    ///         escaped = Some(output.as_f32_slice()?);
    ///         Ok(())
    ///     }).unwrap();
    ///     println!("{:?}", escaped);
    /// }
    /// ```
    pub fn outputs_with(
        &mut self,
        names: &[&str],
        mut consume: impl FnMut(&str, &OutputGuard<'_>) -> Result<()>,
    ) -> Result<()> {
        for name in names {
            let output = self.extract(*name)?;
            consume(name, &output)?;
        }
        Ok(())
    }
    /// Exposes advanced access and permanently disables this extractor's guards.
    ///
    /// # Safety
    /// The caller upholds native input/output rules and must not use the handle
    /// after this extractor is dropped.
    pub unsafe fn as_raw(&self) -> sys::ncnn_extractor_t {
        self.low_copy.set(false);
        self.raw
    }
}

/// A read-only output that borrows its extractor and reserves the network.
///
/// ```compile_fail
/// fn reuse(ex: &mut ncnn_rust::Extractor<'_>) {
///     let output = ex.extract("output").unwrap();
///     ex.extract("output").unwrap();
///     println!("{:?}", output.as_f32_slice());
/// }
/// ```
/// ```compile_fail
/// fn escape(net: &ncnn_rust::Net) -> ncnn_rust::OutputGuard<'_> {
///     let mut ex = net.extractor().unwrap();
///     ex.extract("output").unwrap()
/// }
/// ```
/// ```compile_fail
/// fn escape_slice(ex: &mut ncnn_rust::Extractor<'_>) {
///     let output = ex.extract("output").unwrap();
///     let values = output.as_f32_slice().unwrap();
///     drop(output);
///     println!("{}", values[0]);
/// }
/// ```
/// ```compile_fail
/// fn escape_handle(output: ncnn_rust::OutputGuard<'_>) -> ncnn_rust::Mat {
///     output.into_raw()
/// }
/// ```
/// ```compile_fail
/// fn destroy_extractor(net: &ncnn_rust::Net) {
///     let mut ex = net.extractor().unwrap();
///     let _output = ex.extract("output").unwrap();
///     drop(ex); // The unused guard must still release its handle first.
/// }
/// ```
pub struct OutputGuard<'extractor> {
    mat: Mat,
    network_borrowed: &'extractor Cell<bool>,
    _extractor: PhantomData<&'extractor mut ()>,
}

impl OutputGuard<'_> {
    /// Returns the borrowed tensor metadata.
    pub fn info(&self) -> crate::TensorInfo {
        self.mat.info()
    }
    /// Borrows contiguous unpacked storage with an FP32 interpretation.
    pub fn as_f32_slice(&self) -> Result<&[f32]> {
        self.mat.as_f32_slice()
    }
    /// Borrows one channel, skipping inter-channel padding.
    pub fn channel_f32(&self, channel: usize) -> Result<&[f32]> {
        self.mat.channel_f32(channel)
    }
    /// Copies into storage that outlives the guard, extractor and network.
    pub fn detach(&self) -> Result<Mat> {
        self.mat.deep_clone()
    }
    /// Copies logical FP32 elements into a new vector.
    pub fn to_f32(&self) -> Result<Vec<f32>> {
        self.mat.to_f32()
    }
    /// Copies any scalar format into FP32 values, including quantized outputs.
    pub fn to_f32_typed(&self, kind: crate::ElementKind) -> Result<Vec<f32>> {
        self.mat.to_f32_typed(kind)
    }
    /// Copies signed 8-bit quantized values.
    pub fn to_i8(&self) -> Result<Vec<i8>> {
        self.mat.to_i8()
    }
}

impl Drop for OutputGuard<'_> {
    fn drop(&mut self) {
        self.network_borrowed.set(false);
    }
}

impl Drop for Extractor<'_> {
    fn drop(&mut self) {
        unsafe {
            sys::ncnn_extractor_destroy(self.raw);
        }
        let live = self._net.guard.extractors.get();
        self._net.guard.extractors.set(live - 1);
    }
}
