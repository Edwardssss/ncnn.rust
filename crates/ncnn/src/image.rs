//! Checked CPU image views, color conversion, resizing, orientation, and affine warping.
use crate::{sys, Error, Mat, PixelFormat, Result};

/// A borrowed pixel view with an explicit source format and row stride.
#[derive(Clone, Copy)]
pub struct PixelImage<'a> {
    data: &'a [u8],
    size: (usize, usize),
    stride: usize,
    format: PixelFormat,
}
impl<'a> PixelImage<'a> {
    /// Borrows packed pixels and checks that the buffer matches the declared size.
    pub fn new(data: &'a [u8], size: (usize, usize), format: PixelFormat) -> Result<Self> {
        let stride = size
            .0
            .checked_mul(format.channels())
            .ok_or(Error::invalid("Pixel stride overflow"))?;
        if stride.checked_mul(size.1) != Some(data.len()) {
            return Err(Error::invalid("Pixel buffer size mismatch"));
        }
        Self::with_stride(data, size, stride, format)
    }
    /// Accepts initialized logical rows, including optional padding between rows.
    pub fn with_stride(
        data: &'a [u8],
        size: (usize, usize),
        stride: usize,
        format: PixelFormat,
    ) -> Result<Self> {
        let row = size
            .0
            .checked_mul(format.channels())
            .ok_or(Error::invalid("Pixel row overflow"))?;
        if size.0 == 0
            || size.1 == 0
            || stride < row
            || stride > i32::MAX as usize
            || size.1 > i32::MAX as usize
        {
            return Err(Error::invalid("Invalid image dimensions or stride"));
        }
        let count = (size.1 - 1)
            .checked_mul(stride)
            .and_then(|n| n.checked_add(row))
            .ok_or(Error::invalid("Image size overflow"))?;
        if count > data.len() || count > i32::MAX as usize {
            return Err(Error::invalid(
                "Image rows exceed the buffer or native range",
            ));
        }
        Ok(Self {
            data,
            size,
            stride,
            format,
        })
    }
    /// Narrows the view to a checked sub-rectangle.
    pub fn roi(self, origin: (usize, usize), size: (usize, usize)) -> Result<Self> {
        if origin
            .0
            .checked_add(size.0)
            .filter(|end| *end <= self.size.0)
            .is_none()
            || origin
                .1
                .checked_add(size.1)
                .filter(|end| *end <= self.size.1)
                .is_none()
        {
            return Err(Error::invalid("Image region is out of bounds"));
        }
        let offset = origin
            .1
            .checked_mul(self.stride)
            .and_then(|n| {
                origin
                    .0
                    .checked_mul(self.format.channels())
                    .and_then(|x| n.checked_add(x))
            })
            .ok_or(Error::invalid("Image offset overflow"))?;
        let data = self
            .data
            .get(offset..)
            .ok_or(Error::invalid("Image region exceeds its buffer"))?;
        Self::with_stride(data, size, self.stride, self.format)
    }
    fn conversion(self, destination: PixelFormat, target: (usize, usize)) -> Result<i32> {
        let channels = self.format.channels().max(destination.channels());
        let count = target
            .0
            .checked_mul(target.1)
            .and_then(|n| n.checked_mul(channels));
        if target.0 == 0 || target.1 == 0 || count.filter(|n| *n <= i32::MAX as usize).is_none() {
            return Err(Error::invalid("Target image exceeds native dimensions"));
        }
        Ok(if self.format == destination {
            self.format as i32
        } else {
            self.format as i32 | ((destination as i32) << 16)
        })
    }
    /// Converts the view into an owned tensor without resizing.
    pub fn to_mat(self, destination: PixelFormat) -> Result<Mat> {
        let kind = self.conversion(destination, self.size)?;
        unsafe {
            Mat::owned(
                sys::ncnn_mat_from_pixels(
                    self.data.as_ptr(),
                    kind,
                    self.size.0 as i32,
                    self.size.1 as i32,
                    self.stride as i32,
                    std::ptr::null_mut(),
                ),
                true,
            )
        }
    }
    /// Converts the view and resizes it to the requested target size.
    pub fn resize_to_mat(self, destination: PixelFormat, target: (usize, usize)) -> Result<Mat> {
        let kind = self.conversion(destination, target)?;
        if self.size != target && (self.size.0 < 2 || self.size.1 < 2) {
            return Err(Error::invalid(
                "Bilinear source dimensions must be at least two",
            ));
        }
        unsafe {
            Mat::owned(
                sys::ncnn_mat_from_pixels_resize(
                    self.data.as_ptr(),
                    kind,
                    self.size.0 as i32,
                    self.size.1 as i32,
                    self.stride as i32,
                    target.0 as i32,
                    target.1 as i32,
                    std::ptr::null_mut(),
                ),
                true,
            )
        }
    }
}

fn dimensions(w: usize, h: usize, channels: usize, len: usize) -> Result<(i32, i32)> {
    if !(1..=4).contains(&channels)
        || w == 0
        || h == 0
        || w > i32::MAX as usize / channels
        || h > i32::MAX as usize
    {
        return Err(Error::invalid("Invalid image dimensions"));
    }
    if w.checked_mul(h).and_then(|n| n.checked_mul(channels)) != Some(len)
        || len > isize::MAX as usize
    {
        return Err(Error::invalid("Image byte count mismatch"));
    }
    Ok((w as i32, h as i32))
}

/// Resizes a tightly packed one- to four-channel image with bilinear interpolation.
pub fn resize(
    data: &[u8],
    size: (usize, usize),
    channels: usize,
    target: (usize, usize),
) -> Result<Vec<u8>> {
    let (w, h) = dimensions(size.0, size.1, channels, data.len())?;
    if w < 2 || h < 2 {
        return Err(Error::invalid(
            "Bilinear source dimensions must be at least two",
        ));
    }
    let count = target
        .0
        .checked_mul(target.1)
        .and_then(|n| n.checked_mul(channels))
        .ok_or(Error::invalid("Image size overflow"))?;
    let (tw, th) = dimensions(target.0, target.1, channels, count)?;
    let mut out = vec![0; count];
    unsafe {
        let call = match channels {
            1 => sys::ncnn_rs_image_resize_bilinear_c1,
            2 => sys::ncnn_rs_image_resize_bilinear_c2,
            3 => sys::ncnn_rs_image_resize_bilinear_c3,
            _ => sys::ncnn_rs_image_resize_bilinear_c4,
        };
        call(data.as_ptr(), w, h, out.as_mut_ptr(), tw, th);
    }
    Ok(out)
}

/// Border treatment for [`warp_affine`].
///
/// The variants match the native border codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum WarpBorder {
    /// Fills the uncovered pixels with the supplied value.
    Constant = 0,
    /// Repeats the nearest edge pixel.
    Replicate = 1,
    /// Leaves the uncovered pixels untouched.
    Transparent = 2,
}

/// Computes the affine matrix that maps one point set onto another.
///
/// `from` and `to` hold corresponding points as `[x, y]` pairs; three points describe a full affine
/// transform and one point describes a translation. The result is the six coefficients in the order
/// the native warp expects.
pub fn affine_transform(from: &[[f32; 2]], to: &[[f32; 2]]) -> Result<[f32; 6]> {
    if from.len() != to.len() || from.is_empty() || from.len() > 3 {
        return Err(Error::invalid(
            "An affine transform needs one to three matching point pairs",
        ));
    }
    let flatten = |points: &[[f32; 2]]| -> Vec<f32> { points.iter().flatten().copied().collect() };
    let points_from = flatten(from);
    let points_to = flatten(to);
    let mut matrix = [0.0f32; 6];
    unsafe {
        sys::ncnn_rs_image_get_affine_transform(
            points_from.as_ptr(),
            points_to.as_ptr(),
            from.len() as i32,
            matrix.as_mut_ptr(),
        );
    }
    Ok(matrix)
}

/// Inverts an affine matrix, so a destination-to-source map can be used as a source-to-destination.
pub fn invert_affine(matrix: &[f32; 6]) -> [f32; 6] {
    let mut inverted = [0.0f32; 6];
    unsafe {
        sys::ncnn_rs_image_invert_affine_transform(matrix.as_ptr(), inverted.as_mut_ptr());
    }
    inverted
}

/// Warps a tightly packed one- to four-channel image with bilinear interpolation.
///
/// `matrix` maps destination coordinates to source coordinates, which is the form
/// [`invert_affine`] produces from an [`affine_transform`]. `border` value fills the pixels that
/// map outside the source.
pub fn warp_affine(
    data: &[u8],
    size: (usize, usize),
    channels: usize,
    target: (usize, usize),
    matrix: &[f32; 6],
    border: WarpBorder,
    value: u32,
) -> Result<Vec<u8>> {
    let (w, h) = dimensions(size.0, size.1, channels, data.len())?;
    let count = target
        .0
        .checked_mul(target.1)
        .and_then(|n| n.checked_mul(channels))
        .ok_or(Error::invalid("Image size overflow"))?;
    let (tw, th) = dimensions(target.0, target.1, channels, count)?;
    let mut out = vec![0; count];
    unsafe {
        let call = match channels {
            1 => sys::ncnn_rs_image_warpaffine_bilinear_c1,
            2 => sys::ncnn_rs_image_warpaffine_bilinear_c2,
            3 => sys::ncnn_rs_image_warpaffine_bilinear_c3,
            _ => sys::ncnn_rs_image_warpaffine_bilinear_c4,
        };
        call(
            data.as_ptr(),
            w,
            h,
            out.as_mut_ptr(),
            tw,
            th,
            matrix.as_ptr(),
            border as i32,
            value,
        );
    }
    Ok(out)
}

/// Applies an EXIF orientation (1鈥?), returning bytes and output dimensions.
pub fn orient(
    data: &[u8],
    size: (usize, usize),
    channels: usize,
    orientation: u8,
) -> Result<(Vec<u8>, (usize, usize))> {
    let (w, h) = dimensions(size.0, size.1, channels, data.len())?;
    if !(1..=8).contains(&orientation) {
        return Err(Error::invalid("Orientation must be 1..=8"));
    }
    let target = if orientation >= 5 {
        (size.1, size.0)
    } else {
        size
    };
    let mut out = vec![0; data.len()];
    unsafe {
        let call = match channels {
            1 => sys::ncnn_rs_image_kanna_rotate_c1,
            2 => sys::ncnn_rs_image_kanna_rotate_c2,
            3 => sys::ncnn_rs_image_kanna_rotate_c3,
            _ => sys::ncnn_rs_image_kanna_rotate_c4,
        };
        call(
            data.as_ptr(),
            w,
            h,
            out.as_mut_ptr(),
            target.0 as i32,
            target.1 as i32,
            orientation as i32,
        );
    }
    Ok((out, target))
}
