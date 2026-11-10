//! Feeds arbitrary bytes to the pixel view: buffer checks, region arithmetic, color conversion and
//! resizing.
//!
//! Every entry point validates sizes and strides before it reaches native code, so a panic or an
//! abort here is a defect while a returned error is the expected outcome.

#![no_main]

use libfuzzer_sys::fuzz_target;
use ncnn_rust::image::PixelImage;
use ncnn_rust::{Mat, PixelFormat};

const FORMATS: [PixelFormat; 5] = [
    PixelFormat::Rgb,
    PixelFormat::Bgr,
    PixelFormat::Gray,
    PixelFormat::Rgba,
    PixelFormat::Bgra,
];

fn dimension(data: &[u8], slot: usize) -> usize {
    data.get(slot).map_or(0, |value| usize::from(*value % 16))
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }
    let width = dimension(data, 0);
    let height = dimension(data, 1);
    let stride = width.saturating_mul(1 + usize::from(data[2] % 4));
    let source = FORMATS[usize::from(data[3]) % FORMATS.len()];
    let destination = FORMATS[usize::from(data[4]) % FORMATS.len()];
    let pixels = &data[8..];

    if let Ok(image) = PixelImage::new(pixels, (width, height), source) {
        let _ = image.to_mat(destination);
    }
    if let Ok(image) = PixelImage::with_stride(pixels, (width, height), stride, source) {
        let _ = image.to_mat(destination);
        let _ = image.resize_to_mat(destination, (dimension(data, 5), dimension(data, 6)));
        let _ = image.roi((dimension(data, 7), 0), (width, height));
    }
    let _ = Mat::from_pixels(pixels, source, width, height);
});
