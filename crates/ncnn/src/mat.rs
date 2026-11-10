use crate::{error::status, sys, Error, Result};
use std::{marker::PhantomData, rc::Rc};

/// Scalar format of a native tensor.
///
/// ncnn records the element width but not whether a 16-bit element is IEEE half
/// or bfloat16, so the caller supplies the format for those tensors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ElementKind {
    /// IEEE 754 single precision.
    F32,
    /// IEEE 754 half precision.
    F16,
    /// Brain floating point, 16 bits.
    Bf16,
    /// Signed 8-bit integer, the storage of a quantized tensor.
    I8,
    /// Unsigned 8-bit integer.
    U8,
    /// Signed 16-bit integer.
    I16,
    /// Signed 32-bit integer.
    I32,
}

impl ElementKind {
    fn code(self) -> i32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Bf16 => 2,
            Self::I8 => 3,
            Self::U8 => 4,
            Self::I16 => 5,
            Self::I32 => 6,
        }
    }

    /// Returns the byte width of one element in this format.
    pub fn element_bytes(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::Bf16 | Self::I16 => 2,
            Self::I8 | Self::U8 => 1,
        }
    }

    /// Returns the format name used in error messages.
    pub fn label(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::I8 => "int8",
            Self::U8 => "uint8",
            Self::I16 => "int16",
            Self::I32 => "int32",
        }
    }
}

#[derive(Debug, Clone, Copy)]
/// Native tensor metadata for one extracted output.
pub struct TensorInfo {
    /// Number of populated axes, 1 to 4.
    pub dims: i32,
    /// Logical width in elements.
    pub w: i32,
    /// Logical height in elements.
    pub h: i32,
    /// Logical depth in elements.
    pub d: i32,
    /// Channel count.
    pub c: i32,
    /// Batch count.
    pub n: i32,
    /// Size of one packed element in bytes.
    pub elemsize: usize,
    /// Elements stored per packed element.
    pub elempack: i32,
    /// Native channel stride, which can exceed the logical plane size.
    pub cstep: usize,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
/// Edge treatment of [`Mat::copy_make_border`].
///
/// The variant follows the native padding mode, so a `Reflect` border mirrors the image and a
/// `Replicate` border repeats the edge row.
#[non_exhaustive]
pub enum BorderMode {
    /// Fills the added rows and columns with the supplied value.
    Constant = 0,
    /// Repeats the edge row and column into the border.
    Replicate = 1,
    /// Mirrors the image into the border without repeating the edge row and column.
    Reflect = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
/// Source and destination pixel layout for image conversion.
#[non_exhaustive]
pub enum PixelFormat {
    /// Three interleaved channels in red, green, blue order.
    Rgb = 1,
    /// Three interleaved channels in blue, green, red order.
    Bgr = 2,
    /// One luminance channel.
    Gray = 3,
    /// Four interleaved channels with alpha last.
    Rgba = 4,
    /// Four interleaved channels in blue, green, red, alpha order.
    Bgra = 5,
}
impl PixelFormat {
    /// Returns the channel count of the layout.
    pub fn channels(self) -> usize {
        match self {
            Self::Gray => 1,
            Self::Rgb | Self::Bgr => 3,
            _ => 4,
        }
    }
}

/// An immutable native tensor. Transformations return independently owned storage.
/// Native storage provenance. Only owned storage may be transferred to a network.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Provenance {
    Owned,
    Adopted,
    /// Borrows caller storage; the tensor never frees it and cannot transfer it.
    External,
}

/// An owned tensor handle released through the native allocator.
///
/// # Thread affinity
///
/// A tensor belongs to the thread that created it: the handle is not `Send` or `Sync` because
/// native allocation, packing and the allocator a tensor came from are thread-affine. Move the
/// input bytes instead and build a tensor on the thread that runs inference.
pub struct Mat {
    pub(crate) raw: sys::ncnn_mat_t,
    pub(crate) f32_known: bool,
    provenance: Provenance,
    _thread: PhantomData<Rc<()>>,
}
pub(crate) fn shape(shape: &[usize]) -> Result<([i32; 4], usize)> {
    if !(1..=4).contains(&shape.len()) {
        return Err(Error::invalid("Shape must have 1..=4 dimensions"));
    }
    let mut count = 1usize;
    let mut native = [0; 4];
    for (dimension, &v) in shape.iter().enumerate() {
        if v == 0 || v > i32::MAX as usize {
            return Err(Error::invalid("Invalid dimension"));
        }
        count = count
            .checked_mul(v)
            .ok_or(Error::invalid("Shape overflow"))?;
        native[dimension] = v as i32;
    }
    if count > (isize::MAX as usize / 4).saturating_sub(64) {
        return Err(Error::invalid("Tensor exceeds addressable length"));
    }
    Ok((native, count))
}
fn native_extent(value: usize, operation: &str) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| Error::invalid(format!("{operation} exceeds the native extent range")))
}
fn padded(extent: i32, before: usize, after: usize, operation: &str) -> Result<i32> {
    (extent as usize)
        .checked_add(before)
        .and_then(|total| total.checked_add(after))
        .filter(|total| *total <= i32::MAX as usize)
        .map(|total| total as i32)
        .ok_or_else(|| Error::invalid(format!("{operation} exceeds the native extent range")))
}
impl Mat {
    /// Copies FP32 values into a new tensor of the given rank and extents.
    ///
    /// ```
    /// let tensor = ncnn_rust::Mat::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    /// assert_eq!(tensor.info().w, 2);
    /// assert_eq!(tensor.to_f32()?, [1.0, 2.0, 3.0, 4.0]);
    /// # Ok::<(), ncnn_rust::Error>(())
    /// ```
    pub fn from_f32(data: &[f32], dimensions: &[usize]) -> Result<Self> {
        let (s, count) = shape(dimensions)?;
        if data.len() != count {
            return Err(Error::invalid(format!(
                "Data length {} does not match a shape of {count} elements",
                data.len()
            )));
        }
        let mut raw = std::ptr::null_mut();
        unsafe {
            status(sys::ncnn_rs_mat_copy_f32(
                data.as_ptr(),
                dimensions.len() as i32,
                s.as_ptr(),
                &mut raw,
            ))?;
            Self::owned(raw, true)
        }
    }
    /// Copies typed bytes into a new tensor of the given rank and extents.
    ///
    /// `data` holds elements of `kind.element_bytes()` bytes in channel-plane order, the
    /// layout the native layers read and the typed readers expect. This is how a quantized
    /// or half-precision tensor enters a network.
    pub fn from_typed(kind: ElementKind, data: &[u8], dimensions: &[usize]) -> Result<Self> {
        let (s, count) = shape(dimensions)?;
        let width = kind.element_bytes();
        let expected = count
            .checked_mul(width)
            .ok_or_else(|| Error::invalid("Tensor byte length overflows"))?;
        if data.len() != expected {
            return Err(Error::invalid(format!(
                "Data length {} does not match {count} {} elements of {width} bytes",
                data.len(),
                kind.label()
            )));
        }
        let mut raw = std::ptr::null_mut();
        unsafe {
            status(sys::ncnn_rs_mat_copy_typed(
                kind.code(),
                data.as_ptr(),
                data.len(),
                dimensions.len() as i32,
                s.as_ptr(),
                &mut raw,
            ))?;
            Self::owned(raw, kind == ElementKind::F32)
        }
    }
    /// Copies signed 8-bit values into a new quantized tensor.
    ///
    /// This is the signature `Quantize` produces; `Mat::from_typed` covers the unsigned
    /// format and every other element width.
    pub fn from_i8(data: &[i8], dimensions: &[usize]) -> Result<Self> {
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len()) };
        Self::from_typed(ElementKind::I8, bytes, dimensions)
    }
    pub(crate) unsafe fn owned(raw: sys::ncnn_mat_t, f32_known: bool) -> Result<Self> {
        if raw.is_null() {
            return Err(Error::Native(-100));
        }
        let result = Self {
            raw,
            f32_known,
            provenance: Provenance::Owned,
            _thread: PhantomData,
        };
        if sys::ncnn_mat_get_data(raw).is_null() {
            return Err(Error::Native(-100));
        }
        Ok(result)
    }
    pub(crate) fn is_transferable(&self) -> bool {
        self.provenance == Provenance::Owned
    }
    /// Returns the native shape and stride metadata.
    pub fn info(&self) -> TensorInfo {
        unsafe {
            TensorInfo {
                dims: sys::ncnn_mat_get_dims(self.raw),
                w: sys::ncnn_mat_get_w(self.raw),
                h: sys::ncnn_mat_get_h(self.raw),
                d: sys::ncnn_mat_get_d(self.raw),
                c: sys::ncnn_mat_get_c(self.raw),
                n: sys::ncnn_mat_get_n(self.raw),
                elemsize: sys::ncnn_mat_get_elemsize(self.raw),
                elempack: sys::ncnn_mat_get_elempack(self.raw),
                cstep: sys::ncnn_mat_get_cstep(self.raw),
            }
        }
    }
    /// Copies 32-bit unpacked storage as FP32 values. The caller selects the
    /// interpretation for tensors whose scalar format is supplied by the model.
    pub fn to_f32(&self) -> Result<Vec<f32>> {
        let count = self.f32_copy_count()?;
        let mut values: Vec<f32> = Vec::with_capacity(count);
        status(unsafe { sys::ncnn_rs_mat_read_f32(self.raw, values.as_mut_ptr(), count) })?;
        // The native copy initializes every element before the length is set.
        unsafe { values.set_len(count) };
        Ok(values)
    }
    /// Copies any scalar format into FP32 values.
    ///
    /// Native tensors carry an element width but no format tag, so a 16-bit
    /// tensor needs the caller to say whether it stores IEEE half or bfloat16.
    /// Quantized outputs of `Quantize` and `Requantize` are signed 8-bit.
    pub fn to_f32_typed(&self, kind: ElementKind) -> Result<Vec<f32>> {
        let info = self.info();
        let width = kind.element_bytes();
        let stored = width.saturating_mul(info.elempack.max(1) as usize);
        if info.elemsize != stored {
            return Err(Error::invalid(format!(
                "Tensor stores {} bytes per element, so it is not {} storage; read it with the matching kind, or extract it with FP32 conversion",
                info.elemsize,
                kind.label()
            )));
        }
        let count = self.logical_count(&info)?;
        let mut values = vec![0f32; count];
        if count == 0 {
            return Ok(values);
        }
        status(unsafe {
            sys::ncnn_rs_mat_read_typed(self.raw, kind.code(), values.as_mut_ptr(), values.len())
        })?;
        Ok(values)
    }
    /// Copies signed 8-bit quantized values without going through FP32 storage.
    ///
    /// The tensor records one byte per element but not whether the byte is signed, so this
    /// reads the storage as `I8`: a stored `200` returns `-56`. Read unsigned one-byte
    /// storage with [`to_f32_typed(ElementKind::U8)`](Self::to_f32_typed) instead.
    pub fn to_i8(&self) -> Result<Vec<i8>> {
        Ok(self
            .to_f32_typed(ElementKind::I8)?
            .into_iter()
            .map(|value| value as i8)
            .collect())
    }
    fn logical_count(&self, info: &TensorInfo) -> Result<usize> {
        if info.n <= 0 || info.w <= 0 || info.h <= 0 || info.d <= 0 || info.c <= 0 {
            return Err(Error::invalid("Tensor has an empty extent"));
        }
        let pixels = (info.w as usize)
            .checked_mul(info.h as usize)
            .and_then(|value| value.checked_mul(info.d as usize))
            .ok_or(Error::invalid("Tensor extent overflows"))?;
        let pack = info.elempack.max(1) as usize;
        pixels
            .checked_mul(pack)
            .and_then(|value| value.checked_mul(info.c as usize))
            .and_then(|value| value.checked_mul(info.n as usize))
            .ok_or(Error::invalid("Tensor extent overflows"))
    }
    /// Copies logical FP32 elements into an exact-sized initialized slice.
    /// Channel padding is omitted. A size/layout error leaves the slice unchanged.
    pub fn copy_to_f32(&self, out: &mut [f32]) -> Result<()> {
        let count = self.f32_copy_count()?;
        if out.len() != count {
            return Err(Error::invalid("Output buffer size mismatch"));
        }
        unsafe {
            status(sys::ncnn_rs_mat_read_f32(
                self.raw,
                out.as_mut_ptr(),
                out.len(),
            ))
        }
    }
    /// Reuses vector capacity and replaces its logical FP32 values.
    /// A layout error leaves the vector unchanged; a failed native copy empties it.
    pub fn to_f32_into(&self, out: &mut Vec<f32>) -> Result<()> {
        let count = self.f32_copy_count()?;
        out.clear();
        out.reserve(count);
        unsafe {
            status(sys::ncnn_rs_mat_read_f32(self.raw, out.as_mut_ptr(), count))?;
            // The native copy initializes every element before the length grows.
            out.set_len(count);
        }
        Ok(())
    }
    /// Appends logical FP32 values without an intermediate tensor/vector copy.
    pub fn append_f32_to(&self, out: &mut Vec<f32>) -> Result<()> {
        let count = self.f32_copy_count()?;
        let start = out.len();
        let end = start
            .checked_add(count)
            .ok_or(Error::invalid("Output length overflow"))?;
        out.reserve(count);
        let result = unsafe {
            status(sys::ncnn_rs_mat_read_f32(
                self.raw,
                out.as_mut_ptr().add(start),
                count,
            ))
        };
        if result.is_ok() {
            // The native copy initializes the appended range before publication.
            unsafe { out.set_len(end) };
        }
        result
    }
    /// Borrows contiguous, unpacked 32-bit storage with an FP32 interpretation.
    ///
    /// The borrow needs single-batch storage whose channels are adjacent, which is the layout a
    /// single-channel tensor or a network output without channel padding has. A packed, 16-bit or
    /// batched tensor returns [`Error::Invalid`](crate::Error::Invalid), and a multi-channel tensor
    /// with padded planes uses [`channel_f32`](Self::channel_f32) or [`to_f32`](Self::to_f32).
    ///
    /// ```compile_fail
    /// let mat = ncnn_rust::Mat::from_f32(&[1.], &[1]).unwrap();
    /// let values = mat.as_f32_slice().unwrap();
    /// drop(mat);
    /// println!("{}", values[0]);
    /// ```
    pub fn as_f32_slice(&self) -> Result<&[f32]> {
        let (info, plane) = self.f32_layout()?;
        if info.c > 1 && info.cstep != plane {
            return Err(Error::invalid(
                "Tensor channels contain padding; use channel_f32 or to_f32",
            ));
        }
        let (_, count) = shape(&[
            info.w as usize,
            info.h as usize,
            info.d as usize,
            info.c as usize,
        ])?;
        unsafe {
            Ok(std::slice::from_raw_parts(
                sys::ncnn_mat_get_data(self.raw).cast::<f32>(),
                count,
            ))
        }
    }
    /// Borrows one logical channel, excluding native alignment padding.
    pub fn channel_f32(&self, channel: usize) -> Result<&[f32]> {
        let (info, plane) = self.f32_layout()?;
        if channel >= info.c as usize {
            return Err(Error::invalid("Channel index out of bounds"));
        }
        let offset = channel
            .checked_mul(info.cstep)
            .ok_or(Error::invalid("Channel offset overflow"))?;
        if offset
            .checked_add(plane)
            .filter(|count| *count <= isize::MAX as usize / 4)
            .is_none()
        {
            return Err(Error::invalid("Channel exceeds addressable storage"));
        }
        unsafe {
            Ok(std::slice::from_raw_parts(
                sys::ncnn_mat_get_data(self.raw).cast::<f32>().add(offset),
                plane,
            ))
        }
    }
    fn f32_layout(&self) -> Result<(TensorInfo, usize)> {
        let info = self.info();
        let (_, plane) = shape(&[info.w as usize, info.h as usize, info.d as usize])?;
        if info.elemsize != 4
            || info.elempack != 1
            || info.n != 1
            || info.c <= 0
            || info.cstep < plane
        {
            return Err(Error::invalid(
                "FP32 views require valid unpacked single-batch storage",
            ));
        }
        let pointer = unsafe { sys::ncnn_mat_get_data(self.raw) };
        if pointer.is_null() || !pointer.cast::<f32>().is_aligned() {
            return Err(Error::invalid("FP32 storage is not aligned"));
        }
        Ok((info, plane))
    }
    fn f32_copy_count(&self) -> Result<usize> {
        let m = self.info();
        if m.elemsize != 4 || m.elempack != 1 || m.n <= 0 {
            return Err(Error::invalid(format!(
                "FP32 copies need unpacked 32-bit storage but the tensor stores {} bytes per element with pack {}; extract it with stored storage and read it with to_f32_typed or to_i8",
                m.elemsize, m.elempack
            )));
        }
        let (_, count) = shape(&[m.w as usize, m.h as usize, m.d as usize, m.c as usize])?;
        count
            .checked_mul(m.n as usize)
            .filter(|n| *n <= (isize::MAX as usize / 4).saturating_sub(64))
            .ok_or(Error::invalid("Tensor exceeds addressable length"))
    }
    /// Copies logical elements, excluding channel and batch padding.
    ///
    /// # Safety
    /// Every logical element must contain initialized FP32 data with pack 1.
    pub unsafe fn read_f32_unchecked(&self) -> Result<Vec<f32>> {
        let count = self.f32_copy_count()?;
        let mut out: Vec<f32> = Vec::with_capacity(count);
        status(sys::ncnn_rs_mat_read_f32(self.raw, out.as_mut_ptr(), count))?;
        // The native copy initializes every element before the length is set.
        out.set_len(count);
        Ok(out)
    }
    /// Copies the tensor into independent native storage.
    pub fn deep_clone(&self) -> Result<Self> {
        let mut raw = std::ptr::null_mut();
        unsafe {
            status(sys::ncnn_rs_mat_clone(self.raw, &mut raw))?;
            Self::owned(raw, self.f32_known)
        }
    }
    /// Returns independent storage, detaching only when native reshape aliases.
    pub fn reshape(&self, dimensions: &[usize]) -> Result<Self> {
        self.reshape_impl(dimensions, true)
    }
    /// Returns an explicitly read-only reshape that borrows this tensor.
    /// Native padding changes may allocate. No raw handle or mutable Mat escapes;
    /// the source borrow retains external allocator and adopted dependencies.
    pub fn reshape_view(&self, dimensions: &[usize]) -> Result<ReshapeView<'_>> {
        Ok(ReshapeView {
            mat: self.reshape_impl(dimensions, false)?,
            _source: self,
        })
    }
    fn reshape_impl(&self, dimensions: &[usize], independent: bool) -> Result<Self> {
        let (s, count) = shape(dimensions)?;
        let m = self.info();
        let (_, old_count) = shape(&[m.w as usize, m.h as usize, m.d as usize, m.c as usize])?;
        if count != old_count || m.n != 1 || m.elempack != 1 {
            return Err(Error::invalid(
                "Reshape requires matching unpacked dimensions",
            ));
        }
        let mut raw = std::ptr::null_mut();
        unsafe {
            status(sys::ncnn_rs_mat_reshape(
                self.raw,
                dimensions.len() as i32,
                s.as_ptr(),
                independent as i32,
                &mut raw,
            ))?;
            Self::owned(raw, self.f32_known)
        }
    }
    /// Returns a copy with mean subtraction and scaling applied per channel.
    pub fn normalized(&self, mean: &[f32], norm: &[f32]) -> Result<Self> {
        let m = self.info();
        if !self.f32_known || m.elemsize != 4 || m.elempack != 1 || m.n != 1 {
            return Err(Error::invalid("Normalization requires unpacked FP32"));
        }
        if (!mean.is_empty() && mean.len() != m.c as usize)
            || (!norm.is_empty() && norm.len() != m.c as usize)
        {
            return Err(Error::invalid(
                "Normalization requires one value per channel",
            ));
        }
        self.f32_layout()?;
        let mut raw = std::ptr::null_mut();
        unsafe {
            status(sys::ncnn_rs_mat_normalized(
                self.raw,
                if mean.is_empty() {
                    std::ptr::null()
                } else {
                    mean.as_ptr()
                },
                if norm.is_empty() {
                    std::ptr::null()
                } else {
                    norm.as_ptr()
                },
                &mut raw,
            ))?;
            Self::owned(raw, true)
        }
    }
    /// Applies mean subtraction and scaling in place, without a second allocation.
    ///
    /// The tensor must own its storage: a tensor adopted from a raw handle, or built over an
    /// external buffer, returns [`Error::Invalid`] because the write would reach memory the caller
    /// still manages. Lengths follow [`normalized`](Self::normalized).
    pub fn normalize_in_place(&mut self, mean: &[f32], norm: &[f32]) -> Result<()> {
        self.ensure_writable("In-place normalization")?;
        let m = self.info();
        if !self.f32_known || m.elemsize != 4 || m.elempack != 1 || m.n != 1 {
            return Err(Error::invalid("Normalization requires unpacked FP32"));
        }
        if (!mean.is_empty() && mean.len() != m.c as usize)
            || (!norm.is_empty() && norm.len() != m.c as usize)
        {
            return Err(Error::invalid(
                "Normalization requires one value per channel",
            ));
        }
        self.f32_layout()?;
        unsafe {
            sys::ncnn_mat_substract_mean_normalize(
                self.raw,
                if mean.is_empty() {
                    std::ptr::null()
                } else {
                    mean.as_ptr()
                },
                if norm.is_empty() {
                    std::ptr::null()
                } else {
                    norm.as_ptr()
                },
            );
        }
        Ok(())
    }
    /// Fills owned unpacked FP32 storage with one value.
    ///
    /// This is the allocation-free way to clear a reusable input buffer. The storage rules match
    /// [`normalize_in_place`](Self::normalize_in_place).
    pub fn fill(&mut self, value: f32) -> Result<()> {
        self.ensure_writable("In-place fill")?;
        let m = self.info();
        if !self.f32_known || m.elemsize != 4 || m.elempack != 1 {
            return Err(Error::invalid("Fill requires unpacked FP32 storage"));
        }
        unsafe { sys::ncnn_mat_fill_float(self.raw, value) };
        Ok(())
    }
    fn ensure_writable(&self, operation: &'static str) -> Result<()> {
        if self.provenance != Provenance::Owned {
            return Err(Error::invalid(format!(
                "{operation} writes through the tensor, so it needs storage this tensor owns; copy it with deep_clone first"
            )));
        }
        Ok(())
    }
    /// Builds a tensor that borrows caller storage instead of copying it.
    ///
    /// The native tensor keeps the pointer, so this is the zero-copy input path: a caller that
    /// already holds frame data avoids one copy per request. Reads through the returned tensor
    /// behave like any other tensor, and the tensor never frees the buffer.
    ///
    /// # Safety
    ///
    /// `data` must stay alive, keep its length and stay free of mutable aliases until this tensor
    /// and every native object that received it are dropped. An extractor that submitted it holds
    /// the pointer until it is dropped, so the borrow has to cover the extraction. Use
    /// [`Net::infer_borrowed`](crate::Net::infer_borrowed) when the borrow only has to cover one
    /// request.
    pub unsafe fn from_external(data: &[f32], dimensions: &[usize]) -> Result<Self> {
        let (s, count) = shape(dimensions)?;
        if data.len() != count {
            return Err(Error::invalid(format!(
                "Data length {} does not match a shape of {count} elements",
                data.len()
            )));
        }
        let pointer = data.as_ptr().cast_mut().cast::<std::ffi::c_void>();
        let raw = match dimensions.len() {
            1 => sys::ncnn_mat_create_external_1d(s[0], pointer, std::ptr::null_mut()),
            2 => sys::ncnn_mat_create_external_2d(s[0], s[1], pointer, std::ptr::null_mut()),
            3 => sys::ncnn_mat_create_external_3d(s[0], s[1], s[2], pointer, std::ptr::null_mut()),
            _ => sys::ncnn_mat_create_external_4d(
                s[0],
                s[1],
                s[2],
                s[3],
                pointer,
                std::ptr::null_mut(),
            ),
        };
        if raw.is_null() {
            return Err(Error::Native(-100));
        }
        Ok(Self {
            raw,
            f32_known: true,
            provenance: Provenance::External,
            _thread: PhantomData,
        })
    }
    /// Converts unpacked FP32 planes into interleaved pixels.
    ///
    /// `stride` is the destination distance between rows in bytes and must cover
    /// `width * format.channels()`. `out` must hold `stride * height` bytes; the remaining bytes of
    /// a wider stride stay untouched.
    pub fn to_pixels(&self, format: PixelFormat, stride: usize, out: &mut [u8]) -> Result<()> {
        let (width, height) = self.pixel_layout(format)?;
        let row = (width as usize)
            .checked_mul(format.channels())
            .ok_or(Error::invalid("Pixel row overflows"))?;
        if stride < row {
            return Err(Error::invalid("Pixel stride is shorter than one row"));
        }
        let needed = stride
            .checked_mul(height as usize)
            .ok_or(Error::invalid("Pixel buffer overflows"))?;
        if out.len() < needed {
            return Err(Error::invalid("Pixel buffer is too small for the stride"));
        }
        unsafe {
            sys::ncnn_mat_to_pixels(
                self.raw,
                out.as_mut_ptr(),
                format as i32,
                i32::try_from(stride)
                    .map_err(|_| Error::invalid("Pixel stride exceeds native range"))?,
            );
        }
        Ok(())
    }
    /// Converts unpacked FP32 planes into interleaved pixels of another size.
    ///
    /// The resize uses the native bilinear path. Buffer requirements match
    /// [`to_pixels`](Self::to_pixels) with the target width and height.
    pub fn to_pixels_resize(
        &self,
        format: PixelFormat,
        target: (usize, usize),
        stride: usize,
        out: &mut [u8],
    ) -> Result<()> {
        self.pixel_layout(format)?;
        if target.0 == 0
            || target.1 == 0
            || target.0 > i32::MAX as usize
            || target.1 > i32::MAX as usize
        {
            return Err(Error::invalid("Invalid target size"));
        }
        let row = target
            .0
            .checked_mul(format.channels())
            .ok_or(Error::invalid("Pixel row overflows"))?;
        if stride < row {
            return Err(Error::invalid("Pixel stride is shorter than one row"));
        }
        let needed = stride
            .checked_mul(target.1)
            .ok_or(Error::invalid("Pixel buffer overflows"))?;
        if out.len() < needed {
            return Err(Error::invalid("Pixel buffer is too small for the stride"));
        }
        unsafe {
            sys::ncnn_mat_to_pixels_resize(
                self.raw,
                out.as_mut_ptr(),
                format as i32,
                i32::try_from(target.0)
                    .map_err(|_| Error::invalid("Target width exceeds native range"))?,
                i32::try_from(target.1)
                    .map_err(|_| Error::invalid("Target height exceeds native range"))?,
                i32::try_from(stride)
                    .map_err(|_| Error::invalid("Pixel stride exceeds native range"))?,
            );
        }
        Ok(())
    }
    fn pixel_layout(&self, format: PixelFormat) -> Result<(i32, i32)> {
        let (info, _) = self.f32_layout()?;
        if info.c as usize != format.channels() {
            return Err(Error::invalid(format!(
                "Pixel conversion of a {} channel tensor into a {} channel format",
                info.c,
                format.channels()
            )));
        }
        if info.d != 1 {
            return Err(Error::invalid(
                "Pixel conversion needs a single-plane tensor",
            ));
        }
        Ok((info.w, info.h))
    }
    /// Adds a border around every plane with the native padding kernel.
    ///
    /// This is the letterbox step: the native kernel writes the border instead of a Rust-side loop.
    /// The tensor must be a 2-D or 3-D unpacked single-batch tensor, and the result keeps the
    /// element width and the rank of `self`.
    pub fn copy_make_border(
        &self,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
        mode: BorderMode,
        value: f32,
    ) -> Result<Self> {
        let info = self.plane_layout("Border padding")?;
        padded(info.w, left, right, "Border padding")?;
        padded(info.h, top, bottom, "Border padding")?;
        let raw = unsafe { sys::ncnn_mat_create() };
        if raw.is_null() {
            return Err(Error::Native(-100));
        }
        unsafe {
            sys::ncnn_copy_make_border(
                self.raw,
                raw,
                native_extent(top, "Border padding")?,
                native_extent(bottom, "Border padding")?,
                native_extent(left, "Border padding")?,
                native_extent(right, "Border padding")?,
                mode as i32,
                value,
                std::ptr::null_mut(),
            );
            Self::owned(raw, self.f32_known)
        }
    }
    /// Removes rows and columns from every plane.
    ///
    /// The crop runs on the native kernel and keeps the remaining storage layout, which is the
    /// detection-side counterpart of [`copy_make_border`](Self::copy_make_border). Requirements
    /// match that method, and the crop keeps at least one row and one column.
    pub fn copy_cut_border(
        &self,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
    ) -> Result<Self> {
        let info = self.plane_layout("Border crop")?;
        if left.saturating_add(right) >= info.w as usize
            || top.saturating_add(bottom) >= info.h as usize
        {
            return Err(Error::invalid(
                "Border crop removes every row or column of the tensor",
            ));
        }
        let raw = unsafe { sys::ncnn_mat_create() };
        if raw.is_null() {
            return Err(Error::Native(-100));
        }
        unsafe {
            sys::ncnn_copy_cut_border(
                self.raw,
                raw,
                native_extent(top, "Border crop")?,
                native_extent(bottom, "Border crop")?,
                native_extent(left, "Border crop")?,
                native_extent(right, "Border crop")?,
                std::ptr::null_mut(),
            );
            Self::owned(raw, self.f32_known)
        }
    }
    fn plane_layout(&self, operation: &str) -> Result<TensorInfo> {
        let info = self.info();
        if !(2..=3).contains(&info.dims) || info.elempack != 1 || info.n != 1 || info.c <= 0 {
            return Err(Error::invalid(format!(
                "{operation} needs a 2-D or 3-D unpacked single-batch tensor"
            )));
        }
        Ok(info)
    }
    /// Copies interleaved pixels into a new tensor.
    pub fn from_pixels(
        data: &[u8],
        format: PixelFormat,
        width: usize,
        height: usize,
    ) -> Result<Self> {
        let (_, count) = shape(&[width, height, format.channels()])?;
        let stride = width
            .checked_mul(format.channels())
            .and_then(|v| i32::try_from(v).ok())
            .ok_or(Error::invalid("Pixel stride overflow"))?;
        if count != data.len() {
            return Err(Error::invalid("Pixel buffer size mismatch"));
        }
        unsafe {
            Self::owned(
                sys::ncnn_mat_from_pixels(
                    data.as_ptr(),
                    format as i32,
                    width as i32,
                    height as i32,
                    stride,
                    std::ptr::null_mut(),
                ),
                true,
            )
        }
    }
    /// Borrows the native handle without transferring ownership.
    ///
    /// Unsafe native writers must preserve Rust tensor borrows, including output
    /// guard borrows if this storage is later transferred with `input_owned`.
    /// Returns the native handle. The caller keeps it alive exactly as long as this Mat.
    pub fn as_raw(&self) -> sys::ncnn_mat_t {
        self.raw
    }
    /// Releases the Rust handle and transfers release responsibility to the caller.
    pub fn into_raw(self) -> sys::ncnn_mat_t {
        let raw = self.raw;
        std::mem::forget(self);
        raw
    }
    /// Takes ownership of a native tensor with independent storage dependencies.
    ///
    /// # Safety
    /// The handle must be uniquely owned, valid and backed by initialized logical
    /// elements. Its allocator must outlive this object. Native writers respect Rust
    /// tensor-view borrows. `f32_known` asserts FP32 semantics. Consuming input methods
    /// copy adopted storage so borrowed native dependencies keep their original lifetime.
    pub unsafe fn from_raw_owned(raw: sys::ncnn_mat_t, f32_known: bool) -> Result<Self> {
        let mut mat = Self::owned(raw, f32_known)?;
        mat.provenance = Provenance::Adopted;
        Ok(mat)
    }
}
/// A read-only reshape retaining its source and allocator provenance.
/// It cannot be transferred into an extractor or exported as a native handle.
pub struct ReshapeView<'a> {
    mat: Mat,
    _source: &'a Mat,
}
impl ReshapeView<'_> {
    /// Returns the borrowed tensor metadata.
    pub fn info(&self) -> TensorInfo {
        self.mat.info()
    }
    /// Copies logical FP32 elements into a new vector.
    pub fn to_f32(&self) -> Result<Vec<f32>> {
        self.mat.to_f32()
    }
    /// Borrows contiguous FP32 storage without padding.
    pub fn as_f32_slice(&self) -> Result<&[f32]> {
        self.mat.as_f32_slice()
    }
    /// Borrows one channel, skipping inter-channel padding.
    pub fn channel_f32(&self, channel: usize) -> Result<&[f32]> {
        self.mat.channel_f32(channel)
    }
    /// Replaces the vector contents with logical FP32 elements.
    pub fn to_f32_into(&self, buffer: &mut Vec<f32>) -> Result<()> {
        self.mat.to_f32_into(buffer)
    }
}
impl Drop for Mat {
    fn drop(&mut self) {
        unsafe {
            sys::ncnn_mat_destroy(self.raw);
        }
    }
}
