//! Extractor operations, scoped outputs and guard lifetime rules.

use crate::{error::status_at, sys, Error, Mat, Result};

use super::{BlobId, Endpoint, FinalOutput, Net};
use crate::net::blob::BlobKind;
use std::{cell::Cell, ffi::CString, marker::PhantomData};

/// Runs one request against a borrowed network.
///
/// # Thread affinity
///
/// An extractor borrows its network, so it stays on the thread that created both. A worker thread
/// builds its own network and extractor instead of sharing them.
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
            return Err(Error::invalid(
                "Blob identifier belongs to another graph or an earlier load",
            ));
        }
        if blob.kind() != kind {
            return Err(Error::invalid(
                "Blob identifier was resolved for the other endpoint kind",
            ));
        }
        Ok(blob.index())
    }
    fn ensure_exposed_index(&self, index: i32, input: bool) -> Result<()> {
        if !self._net.ready {
            return Err(Error::invalid(
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
            return Err(Error::invalid("Index is not an exposed input"));
        }
        let outputs = unsafe { sys::ncnn_net_get_output_count(self._net.raw) };
        for slot in 0..outputs {
            if index >= 0 && unsafe { sys::ncnn_net_get_output_index(self._net.raw, slot) } == index
            {
                return Ok(());
            }
        }
        Err(Error::invalid("Index is not an exposed output"))
    }
    fn ensure_not_reserved(&self) -> Result<()> {
        if self._net.guard.output_borrowed.get() {
            return Err(Error::invalid("A scoped output reserves this network"));
        }
        Ok(())
    }
    fn ensure_borrowable(&self, stored: bool) -> Result<()> {
        if self._net.guard.extractors.get() != 1 || self._net.guard.output_borrowed.get() {
            return Err(Error::invalid(
                "Scoped output requires the network's only live extractor",
            ));
        }
        if !self.low_copy.get() || !self._net.guard.low_copy.get() {
            // A device network downloads into its own CPU blob storage and detaches that copy
            // from the pool allocator, so a stored borrow is as safe there as on the host while
            // the FP32-converting guard path stays CPU-only.
            let device_stored = stored && self.low_copy.get() && self._net.has_device();
            if !device_stored {
                return Err(Error::invalid(
                    "Guarded outputs need the low-copy path: configure safe options before loading, avoid layer types that export raw handles, and keep one extractor per network; use extract_owned, infer or extract_stored_owned for independent storage instead",
                ));
            }
        }
        Ok(())
    }
    fn extract_native(&self, target: &Target, stored: bool) -> Result<Mat> {
        self.ensure_not_reserved()?;
        let mut raw = std::ptr::null_mut();
        let extraction_type = i32::from(stored);
        let code = unsafe {
            match target {
                Target::Named(name) => {
                    if stored {
                        sys::ncnn_rs_extractor_extract_stored(
                            self.raw,
                            name.as_ptr(),
                            extraction_type,
                            &mut raw,
                        )
                    } else {
                        sys::ncnn_extractor_extract(self.raw, name.as_ptr(), &mut raw)
                    }
                }
                Target::Indexed(index) => {
                    if stored {
                        sys::ncnn_rs_extractor_extract_stored_index(
                            self.raw,
                            *index,
                            extraction_type,
                            &mut raw,
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
    ///
    /// The tensor's storage is transferred when the crate created or independently cloned it, and
    /// copied when it came from a raw handle, because the caller keeps managing that storage. The
    /// endpoint resolves to a graph input; a reserved network rejects the call until its scoped
    /// output drops.
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
    ///
    /// An error stops the sequence, and the inputs already submitted remain applied, so a caller
    /// that retries continues from the failing entry rather than from the start.
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
    ///
    /// The guard reserves the network for its lifetime, so no other extractor starts and no input
    /// replacement happens until it drops. Eligibility needs the low-copy path: safe options before
    /// loading, no raw handle exports, and one live extractor. A device network uses
    /// [`extract_stored`](Self::extract_stored) instead, and networks that miss the requirements use
    /// [`extract_owned`](Self::extract_owned).
    pub fn extract<'a>(&mut self, endpoint: impl Into<Endpoint<'a>>) -> Result<OutputGuard<'_>> {
        self.ensure_borrowable(false)?;
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
    ///
    /// ```
    /// let net = ncnn_rust::Net::builder()?.threads(1)?.load_bytes(
    ///     "7767517\n2 2\nInput input 0 1 input\nQuantize quantize 1 1 input stored 0=1\n",
    ///     &1.0f32.to_le_bytes(),
    /// )?;
    /// let mut extractor = net.extractor()?;
    /// extractor.input("input", ncnn_rust::Mat::from_f32(&[0.4, -0.6], &[2])?)?;
    /// let bytes = {
    ///     let guard = extractor.extract_stored("stored")?;
    ///     assert_eq!(guard.info().elemsize, 1);
    ///     guard.to_i8()?
    /// };
    /// assert_eq!(bytes, [0, -1]);
    /// # Ok::<(), ncnn_rust::Error>(())
    /// ```
    pub fn extract_stored<'a>(
        &mut self,
        endpoint: impl Into<Endpoint<'a>>,
    ) -> Result<OutputGuard<'_>> {
        self.ensure_borrowable(true)?;
        let target = self.resolve(endpoint.into(), true)?;
        let mat = self.extract_native(&target, true)?;
        self._net.guard.output_borrowed.set(true);
        Ok(OutputGuard {
            mat,
            network_borrowed: &self._net.guard.output_borrowed,
            _extractor: PhantomData,
        })
    }
    /// Returns stored storage with independent ownership.
    ///
    /// The result keeps the element width the producing layer wrote, so
    /// [`Mat::to_f32_typed`](crate::Mat::to_f32_typed) or [`Mat::to_i8`](crate::Mat::to_i8)
    /// applies. Use this form when the network may run more requests before the tensor is read.
    pub fn extract_stored_owned<'a>(&mut self, endpoint: impl Into<Endpoint<'a>>) -> Result<Mat> {
        let target = self.resolve(endpoint.into(), true)?;
        let mat = self.extract_native(&target, true)?;
        mat.deep_clone()
    }
    /// Returns validated values with independent owned storage.
    ///
    /// This is the form that works for every network, including device networks, networks with
    /// exported raw handles and networks with more than one live extractor. It converts the output
    /// to FP32 and copies it, so a quantized model needs
    /// [`extract_stored_owned`](Self::extract_stored_owned).
    pub fn extract_owned<'a>(&mut self, endpoint: impl Into<Endpoint<'a>>) -> Result<FinalOutput> {
        let target = self.resolve(endpoint.into(), true)?;
        let mat = self.extract_native(&target, false)?;
        Ok(FinalOutput {
            info: mat.info(),
            values: mat.to_f32()?,
        })
    }
    /// Replaces a caller vector with the output and returns its shape metadata.
    ///
    /// The vector keeps its capacity across calls, which is what makes the steady-state path
    /// allocation-free. A failure leaves the vector empty because the replacement starts before the
    /// native copy.
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
    ///
    /// Existing values stay in place, so several outputs concatenate in call order. A failure
    /// leaves the vector as it was before the call, which makes this the building block behind
    /// [`Net::infer_into`](crate::Net::infer_into).
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
    ///
    /// The metadata describes the layout the native extractor produced: extents, `elemsize`,
    /// `elempack` and the channel stride. `elemsize` decides which reader applies.
    pub fn info(&self) -> crate::TensorInfo {
        self.mat.info()
    }
    /// Borrows contiguous unpacked storage with an FP32 interpretation.
    ///
    /// The borrow needs aligned, unpacked, single-batch storage whose channels are adjacent, so a
    /// 16-bit or packed output returns [`Error::Invalid`](crate::Error::Invalid) with the stored
    /// width in the message. A multi-channel output with padded planes needs
    /// [`channel_f32`](Self::channel_f32) or [`to_f32`](Self::to_f32).
    pub fn as_f32_slice(&self) -> Result<&[f32]> {
        self.mat.as_f32_slice()
    }
    /// Borrows one channel, skipping inter-channel padding.
    ///
    /// ncnn pads each channel plane, so this returns the logical elements of `channel` without the
    /// padding a whole-tensor slice would include. An index outside the channel count returns
    /// [`Error::Invalid`](crate::Error::Invalid).
    pub fn channel_f32(&self, channel: usize) -> Result<&[f32]> {
        self.mat.channel_f32(channel)
    }
    /// Copies into storage that outlives the guard, extractor and network.
    ///
    /// The copy keeps the element width, so a quantized output stays readable through
    /// [`Mat::to_i8`](crate::Mat::to_i8) after the next request has run.
    pub fn detach(&self) -> Result<Mat> {
        self.mat.deep_clone()
    }
    /// Copies logical FP32 elements into a new vector.
    ///
    /// Native padding is excluded, so the length is the product of the reported extents. Quantized
    /// and half-precision outputs convert here; use [`to_f32_typed`](Self::to_f32_typed) to name a
    /// 16-bit format.
    pub fn to_f32(&self) -> Result<Vec<f32>> {
        self.mat.to_f32()
    }
    /// Copies any scalar format into FP32 values, including quantized outputs.
    ///
    /// The tensor records an element width without a format tag, so a 16-bit output needs the
    /// caller to say whether it holds IEEE half or bfloat16. A width that disagrees with `kind`
    /// returns [`Error::Invalid`](crate::Error::Invalid).
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
