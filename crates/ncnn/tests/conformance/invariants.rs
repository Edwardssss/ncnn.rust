//! Pairs of layers and graphs that must agree with themselves. A round trip proves
//! layout and scale agreement without needing a second implementation as a reference.

use crate::support::{self, assert_close_with};
use ncnn_rust::{ElementKind, Mat};

#[test]
fn unfold_and_fold_round_trip() {
    let params = "1=2 11=2 3=2 13=2 4=0";
    let net = support::chain(
        &[
            ("Unfold", params),
            ("Fold", "1=2 11=2 3=2 13=2 4=0 20=4 21=4"),
        ],
        &[],
    );
    let source = support::ramp(32);
    let restored = support::run_labeled(&net, &source, &[4, 4, 2], "unfold round trip");
    assert_close_with(&restored, &source, 1e-5);
}

#[test]
fn splitting_then_summing_doubles_the_input() {
    let net = support::graph(
        3,
        4,
        "Input input 0 1 input\n\
         Split split 1 2 input left right\n\
         Eltwise sum 2 1 left right output 0=1\n",
        &[],
    );
    let source = support::ramp(16);
    let doubled = support::run_labeled(&net, &source, &[4, 4], "split sum");
    let expected: Vec<f32> = source.iter().map(|value| value * 2.0).collect();
    assert_close_with(&doubled, &expected, 1e-6);
}

#[test]
fn narrow_elements_convert_to_fp32() {
    // ncnn tensors record an element width but no format tag, so the reader takes the
    // format. The buffers stay alive for the lifetime of the views.
    let int8: [i8; 4] = [-3, 0, 5, 127];
    let half: [u16; 4] = [0x3C00, 0xC000, 0x3800, 0x0000];
    let shape = [4i32];

    for (data, elemsize, kind, expected) in [
        (
            int8.as_ptr() as *mut std::ffi::c_void,
            1usize,
            ElementKind::I8,
            [-3.0, 0.0, 5.0, 127.0],
        ),
        (
            half.as_ptr() as *mut std::ffi::c_void,
            2usize,
            ElementKind::F16,
            [1.0, -2.0, 0.5, 0.0],
        ),
    ] {
        let mut raw: ncnn_rust::sys::ncnn_mat_t = std::ptr::null_mut();
        let status = unsafe {
            ncnn_rust::sys::ncnn_rs_mat_external_batch(
                data,
                1,
                shape.as_ptr(),
                1,
                elemsize,
                1,
                &mut raw,
            )
        };
        assert_eq!(status, 0, "external view");
        let mat = unsafe { Mat::from_raw_owned(raw, false) }.expect("view");
        let values = mat
            .to_f32_typed(kind)
            .expect("typed read of a narrow tensor");
        assert_eq!(values, expected);
        std::mem::forget(mat);
    }
}
