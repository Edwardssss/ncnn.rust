//! Feeds arbitrary bytes to the typed tensor paths: element decoding, shape validation, reshaping,
//! normalization and sub views.
//!
//! Quantized and low-precision tensors arrive from model files, so a panic or an abort here is a
//! defect while a returned error is the expected outcome.

#![no_main]

use libfuzzer_sys::fuzz_target;
use ncnn_rust::{ElementKind, Mat};

const KINDS: [ElementKind; 7] = [
    ElementKind::F32,
    ElementKind::F16,
    ElementKind::Bf16,
    ElementKind::I8,
    ElementKind::U8,
    ElementKind::I16,
    ElementKind::I32,
];

fn extent(data: &[u8], slot: usize) -> usize {
    data.get(slot)
        .map_or(1, |value| 1 + usize::from(*value % 4))
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }
    let kind = KINDS[usize::from(data[0]) % KINDS.len()];
    let shape = [extent(data, 1), extent(data, 2), extent(data, 3)];
    let payload = &data[4..];

    let Ok(tensor) = Mat::from_typed(kind, payload, &shape) else {
        return;
    };
    let _ = tensor.to_f32();
    let _ = tensor.to_f32_typed(ElementKind::F32);
    let _ = tensor.to_i8();
    let _ = tensor.normalized(&[0.5, 0.5, 0.5, 0.5], &[1., 1., 1., 1.]);
    for dimensions in [
        vec![shape[0], shape[1] * shape[2]],
        vec![shape[0] * shape[1] * shape[2]],
        vec![shape[0], 0],
    ] {
        let _ = tensor.reshape(&dimensions);
        if let Ok(view) = tensor.reshape_view(&dimensions) {
            let _ = view.to_f32();
        }
    }
    if let Ok(view) = tensor.channel(0) {
        let _ = view.to_f32();
        let _ = view.is_empty();
    }
    if let Ok(view) = tensor.row_range(0, 1) {
        let _ = view.to_f32();
    }
    let _ = tensor.with_packing((shape[0] % 4) as i32);
});
