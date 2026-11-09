//! Pairs of layers and graphs that must agree with themselves. A round trip proves
//! layout and scale agreement without needing a second implementation as a reference.

use crate::support::{self, assert_close_with, tagged};
use ncnn_rust::{ElementKind, Endpoint, Mat, Net};

#[test]
fn key_value_cache_keeps_what_it_already_saw() {
    // A three-input attention step reads `q`, `past_k` and `past_v` and returns the grown
    // caches, so leaving the past empty runs the first step and feeding the caches back
    // extends them by one step.
    let weights = support::blocks(&[
        &tagged(&[0.25; 16]),
        &support::raw(&[0.0; 4]),
        &tagged(&[0.25; 16]),
        &support::raw(&[0.0; 4]),
        &tagged(&[0.25; 16]),
        &support::raw(&[0.0; 4]),
        &tagged(&[0.25; 16]),
        &support::raw(&[0.0; 4]),
    ]);
    let net = support::graph(
        4,
        7,
        "Input q 0 1 q\nInput k 0 1 k\nInput v 0 1 v\n\
         MultiHeadAttention attn 3 3 q k v out cache_k cache_v 0=4 1=2 2=16 7=1\n",
        &weights,
    );
    let step = |rows: usize| {
        Mat::from_f32(
            &(0..rows * 4)
                .map(|index| index as f32 * 0.125 + 0.5)
                .collect::<Vec<_>>(),
            &[4, rows],
        )
        .expect("step")
    };

    let (first, cache_k, cache_v) = run_cache_step(&net, step(2), None);
    assert_eq!(first, 2, "first step sequence length");
    let (second, grown_k, grown_v) = run_cache_step(&net, step(1), Some((cache_k, cache_v)));
    assert_eq!(second, 3, "second step sequence length");
    assert_eq!(grown_k.info().h, 3, "the cache grows by the submitted step");
    assert_eq!(
        grown_v.info().h,
        3,
        "the value cache grows with the key cache"
    );
}

/// Runs one attention step and returns the cached sequence length plus both caches.
fn run_cache_step(net: &Net, query: Mat, past: Option<(Mat, Mat)>) -> (i32, Mat, Mat) {
    let mut extractor = net.extractor().expect("extractor");
    extractor
        .input(Endpoint::Name("q"), query)
        .expect("query input");
    if let Some((past_k, past_v)) = past {
        extractor
            .input(Endpoint::Name("k"), past_k)
            .expect("past key input");
        extractor
            .input(Endpoint::Name("v"), past_v)
            .expect("past value input");
    }
    let key = extractor
        .extract_stored(Endpoint::Name("cache_k"))
        .expect("key cache")
        .detach()
        .expect("key cache copy");
    let value = extractor
        .extract_stored(Endpoint::Name("cache_v"))
        .expect("value cache")
        .detach()
        .expect("value cache copy");
    let length = key.info().h;
    (length, key, value)
}

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

#[test]
fn quantized_pipeline_keeps_its_one_byte_storage() {
    // A quantized convolution loads a float weight block followed by one scale per
    // output channel and one input scale, and `Quantize` stores its result one byte
    // wide. The scales make the stored bytes readable with `to_i8`, which needs the CPU
    // copy path, so the fixtures stay on the host even when the suite targets a device.
    let quantized = support::cpu_graph(
        3,
        3,
        "Input input 0 1 input\n\
         Convolution conv 1 1 input scaled 0=1 1=2 5=0 6=4 8=1\n\
         Quantize quantize 1 1 scaled stored 0=1\n",
        &support::blocks(&[
            &tagged(&[0.25; 4]),
            &support::raw(&[16.0]),
            &support::raw(&[1.0]),
            &support::raw(&[1.0]),
        ]),
    );
    let float = support::cpu_graph(
        2,
        2,
        "Input input 0 1 input\nConvolution conv 1 1 input output 0=1 1=2 5=0 6=4\n",
        &support::blocks(&[&tagged(&[0.25; 4])]),
    );

    let values = support::ramp(16);
    let reference = support::run_labeled(&float, &values, &[4, 4, 1], "float convolution");
    let mut extractor = quantized.extractor().expect("extractor");
    extractor
        .input(
            Endpoint::Name("input"),
            Mat::from_f32(&values, &[4, 4, 1]).expect("input"),
        )
        .expect("input");
    let scaled = extractor
        .extract(Endpoint::Name("scaled"))
        .expect("scaled")
        .to_f32()
        .expect("scaled values");
    assert_close_with(&scaled, &reference, 0.5);
    let stored = {
        let guard = extractor
            .extract_stored(Endpoint::Name("stored"))
            .expect("stored");
        assert_eq!(guard.info().elemsize, 1, "quantized storage byte width");
        guard.to_i8().expect("stored values")
    };

    assert_eq!(stored.len(), scaled.len());
    assert!(
        stored.iter().any(|value| *value != 0),
        "quantized storage is not all zeros"
    );
    for (index, (value, source)) in stored.iter().zip(&scaled).enumerate() {
        let expected = source.round();
        assert!(
            (f32::from(*value) - expected).abs() <= 1.0,
            "element {index}: stored {value}, expected {expected}"
        );
    }
}
