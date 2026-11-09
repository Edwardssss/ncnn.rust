use crate::{error::status, sys, Error, Result};
use std::{marker::PhantomData, rc::Rc};

/// Scalar format of a native tensor.
///
/// ncnn records the element width but not whether a 16-bit element is IEEE half
/// or bfloat16, so the caller supplies the format for those tensors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
/// Source and destination pixel layout for image conversion.
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
}

/// An owned tensor handle released through the native allocator.
pub struct Mat {
    pub(crate) raw: sys::ncnn_mat_t,
    pub(crate) f32_known: bool,
    provenance: Provenance,
    _thread: PhantomData<Rc<()>>,
}
pub(crate) fn shape(shape: &[usize]) -> Result<([i32; 4], usize)> {
    if !(1..=4).contains(&shape.len()) {
        return Err(Error::Invalid("Shape must have 1..=4 dimensions"));
    }
    let mut count = 1usize;
    let mut native = [0; 4];
    for (dimension, &v) in shape.iter().enumerate() {
        if v == 0 || v > i32::MAX as usize {
            return Err(Error::Invalid("Invalid dimension"));
        }
        count = count
            .checked_mul(v)
            .ok_or(Error::Invalid("Shape overflow"))?;
        native[dimension] = v as i32;
    }
    if count > (isize::MAX as usize / 4).saturating_sub(64) {
        return Err(Error::Invalid("Tensor exceeds addressable length"));
    }
    Ok((native, count))
}
impl Mat {
    /// Copies FP32 values into a new tensor of the given rank and extents.
    pub fn from_f32(data: &[f32], dimensions: &[usize]) -> Result<Self> {
        let (s, count) = shape(dimensions)?;
        if data.len() != count {
            return Err(Error::Invalid("Data length does not match shape"));
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
        unsafe { self.read_f32_unchecked() }
    }
    /// Copies any scalar format into FP32 values.
    ///
    /// Native tensors carry an element width but no format tag, so a 16-bit
    /// tensor needs the caller to say whether it stores IEEE half or bfloat16.
    /// Quantized outputs of `Quantize` and `Requantize` are signed 8-bit.
    pub fn to_f32_typed(&self, kind: ElementKind) -> Result<Vec<f32>> {
        let info = self.info();
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
    pub fn to_i8(&self) -> Result<Vec<i8>> {
        Ok(self
            .to_f32_typed(ElementKind::I8)?
            .into_iter()
            .map(|value| value as i8)
            .collect())
    }
    fn logical_count(&self, info: &TensorInfo) -> Result<usize> {
        if info.n <= 0 || info.w <= 0 || info.h <= 0 || info.d <= 0 || info.c <= 0 {
            return Err(Error::Invalid("Tensor has an empty extent"));
        }
        let pixels = (info.w as usize)
            .checked_mul(info.h as usize)
            .and_then(|value| value.checked_mul(info.d as usize))
            .ok_or(Error::Invalid("Tensor extent overflows"))?;
        let pack = info.elempack.max(1) as usize;
        pixels
            .checked_mul(pack)
            .and_then(|value| value.checked_mul(info.c as usize))
            .and_then(|value| value.checked_mul(info.n as usize))
            .ok_or(Error::Invalid("Tensor extent overflows"))
    }
    /// Copies logical FP32 elements into an exact-sized initialized slice.
    /// Channel padding is omitted. A size/layout error leaves the slice unchanged.
    pub fn copy_to_f32(&self, out: &mut [f32]) -> Result<()> {
        let count = self.f32_copy_count()?;
        if out.len() != count {
            return Err(Error::Invalid("Output buffer size mismatch"));
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
            .ok_or(Error::Invalid("Output length overflow"))?;
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
    /// ```compile_fail
    /// let mat = ncnn_rust::Mat::from_f32(&[1.], &[1]).unwrap();
    /// let values = mat.as_f32_slice().unwrap();
    /// drop(mat);
    /// println!("{}", values[0]);
    /// ```
    pub fn as_f32_slice(&self) -> Result<&[f32]> {
        let (info, plane) = self.f32_layout()?;
        if info.c > 1 && info.cstep != plane {
            return Err(Error::Invalid(
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
            return Err(Error::Invalid("Channel index out of bounds"));
        }
        let offset = channel
            .checked_mul(info.cstep)
            .ok_or(Error::Invalid("Channel offset overflow"))?;
        if offset
            .checked_add(plane)
            .filter(|count| *count <= isize::MAX as usize / 4)
            .is_none()
        {
            return Err(Error::Invalid("Channel exceeds addressable storage"));
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
            return Err(Error::Invalid(
                "FP32 views require valid unpacked single-batch storage",
            ));
        }
        let pointer = unsafe { sys::ncnn_mat_get_data(self.raw) };
        if pointer.is_null() || !pointer.cast::<f32>().is_aligned() {
            return Err(Error::Invalid("FP32 storage is not aligned"));
        }
        Ok((info, plane))
    }
    fn f32_copy_count(&self) -> Result<usize> {
        let m = self.info();
        if m.elemsize != 4 || m.elempack != 1 || m.n <= 0 {
            return Err(Error::Invalid(
                "FP32 copies require unpacked 32-bit storage",
            ));
        }
        let (_, count) = shape(&[m.w as usize, m.h as usize, m.d as usize, m.c as usize])?;
        count
            .checked_mul(m.n as usize)
            .filter(|n| *n <= (isize::MAX as usize / 4).saturating_sub(64))
            .ok_or(Error::Invalid("Tensor exceeds addressable length"))
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
            return Err(Error::Invalid(
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
            return Err(Error::Invalid("Normalization requires unpacked FP32"));
        }
        if (!mean.is_empty() && mean.len() != m.c as usize)
            || (!norm.is_empty() && norm.len() != m.c as usize)
        {
            return Err(Error::Invalid(
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
            .ok_or(Error::Invalid("Pixel stride overflow"))?;
        if count != data.len() {
            return Err(Error::Invalid("Pixel buffer size mismatch"));
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
