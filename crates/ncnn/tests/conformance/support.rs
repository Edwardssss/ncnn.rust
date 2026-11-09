//! Fixtures and helpers shared by the conformance suite.

use ncnn_rust::{Mat, Net, NetBuilder, OptionFlag, Result};

/// A deterministic ramp that covers negative, zero and positive values.
pub fn ramp(count: usize) -> Vec<f32> {
    (0..count).map(|i| (i as f32 * 0.25) - 3.0).collect()
}

/// A single-channel or multi-channel image in ncnn plane order.
pub fn image(width: usize, height: usize, channels: usize) -> (Vec<f32>, [usize; 3]) {
    (ramp(width * height * channels), [width, height, channels])
}

/// A `type = 0` weight block, which ncnn prefixes with a zero tag.
pub fn tagged(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity((values.len() + 1) * 4);
    bytes.extend_from_slice(&0f32.to_le_bytes());
    bytes.extend(raw(values));
    bytes
}

/// A `type = 1` weight block, stored without a tag.
pub fn raw(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// Joins weight blocks in the order the layer loads them.
pub fn blocks(parts: &[&[u8]]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for part in parts {
        bytes.extend_from_slice(part);
    }
    bytes
}

/// Builds a network from an explicit layer and blob count.
pub fn graph(layers: usize, blobs: usize, body: &str, weights: &[u8]) -> Net {
    graph_with(layers, blobs, body, weights, Ok)
}

/// Builds a network and reports load failures instead of panicking.
pub fn try_graph(layers: usize, blobs: usize, body: &str, weights: &[u8]) -> Result<Net> {
    let text = format!("7767517\n{layers} {blobs}\n{body}");
    fp32_pipeline().load_bytes(&text, weights)
}

/// Builds a network after the caller adjusts the builder.
pub fn graph_with(
    layers: usize,
    blobs: usize,
    body: &str,
    weights: &[u8],
    configure: impl FnOnce(NetBuilder) -> Result<NetBuilder>,
) -> Net {
    let text = format!("7767517\n{layers} {blobs}\n{body}");
    configure(fp32_pipeline())
        .expect("configure")
        .load_bytes(&text, weights)
        .expect("load")
}

/// Pins a single-threaded FP32 pipeline.
///
/// Native ncnn enables FP16 storage and arithmetic by default on ARM64 and Apple
/// Silicon, which would round every result to FP16 before the reference comparison.
/// The conformance suite states one numeric contract, so the tests disable those
/// presets explicitly and `packing_and_fp16_options_preserve_values` covers the
/// non-default pipelines separately.
fn fp32_pipeline() -> NetBuilder {
    let mut builder = Net::builder()
        .expect("builder")
        .threads(1)
        .expect("threads");
    for flag in [
        OptionFlag::Fp16Packed,
        OptionFlag::Fp16Storage,
        OptionFlag::Fp16Arithmetic,
        OptionFlag::Bf16Packed,
        OptionFlag::Bf16Storage,
        OptionFlag::Packing,
    ] {
        builder = builder.flag(flag, false).expect("flag");
    }
    builder
}

/// Builds one operator between `input` and `output`.
pub fn layer(operation: &str, parameters: &str, weights: &[u8]) -> Net {
    let body =
        format!("Input input 0 1 input\n{operation} operation 1 1 input output {parameters}\n");
    graph(2, 2, &body, weights)
}

/// Builds one operator that consumes two inputs named `a` and `b`.
pub fn two_input(operation: &str, parameters: &str, weights: &[u8]) -> Net {
    let body = format!(
        "Input a 0 1 a\nInput b 0 1 b\n{operation} operation 2 1 a b output {parameters}\n"
    );
    graph(3, 3, &body, weights)
}

/// Builds a linear chain of operators that ends at `output`.
pub fn chain(steps: &[(&str, &str)], weights: &[u8]) -> Net {
    let mut body = String::from("Input input 0 1 input\n");
    let mut previous = String::from("input");
    for (index, (operation, parameters)) in steps.iter().enumerate() {
        let blob = if index + 1 == steps.len() {
            "output".to_string()
        } else {
            format!("stage{index}")
        };
        body.push_str(&format!(
            "{operation} {operation}{index} 1 1 {previous} {blob} {parameters}\n"
        ));
        previous = blob;
    }
    graph(steps.len() + 1, steps.len() + 1, &body, weights)
}

fn tensor(values: &[f32], shape: &[usize]) -> Mat {
    Mat::from_f32(values, shape).expect("tensor")
}

/// Runs one input through `output`.
pub fn run(net: &Net, values: &[f32], shape: &[usize]) -> Vec<f32> {
    run_labeled(net, values, shape, "infer")
}

/// Runs one input through `output`, naming the caller in failure messages.
pub fn run_labeled(net: &Net, values: &[f32], shape: &[usize], label: &str) -> Vec<f32> {
    match net.infer([("input", tensor(values, shape))], &["output"]) {
        Ok(mut outputs) => outputs.remove(0).values,
        Err(error) => panic!("{label}: {error:?}"),
    }
}

/// Runs `a` and `b` through `output`.
pub fn run_two(net: &Net, a: (&[f32], &[usize]), b: (&[f32], &[usize])) -> Vec<f32> {
    net.infer(
        [("a", tensor(a.0, a.1)), ("b", tensor(b.0, b.1))],
        &["output"],
    )
    .expect("infer")
    .remove(0)
    .values
}

pub fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_close_labeled(actual, expected, "", 1e-4);
}

pub fn assert_close_with(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_close_labeled(actual, expected, "", tolerance);
}

pub fn assert_close_labeled(actual: &[f32], expected: &[f32], label: &str, tolerance: f32) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
        if a == e || (a.is_nan() && e.is_nan()) {
            continue;
        }
        assert!(
            (a - e).abs() <= tolerance * e.abs().max(1.0),
            "{label} index {index}: {a} != {e}"
        );
    }
}

/// Maps a logical `(w, h, c)` position onto the plane-major output buffer.
pub fn plane_index(width: usize, height: usize, x: usize, y: usize, channel: usize) -> usize {
    (channel * height + y) * width + x
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Abramowitz and Stegun 7.1.26, absolute error below 1.5e-7.
pub fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_4 * t - 1.453_152_1) * t + 1.421_413_8) * t - 0.284_496_72) * t
            + 0.254_829_6)
            * t)
            * (-x * x).exp();
    sign * y
}

/// Half-pixel bilinear sampling with edge clamping, matching ncnn `align_corner = 0`.
pub fn bilinear(
    source: &[f32],
    width: usize,
    height: usize,
    out_width: usize,
    out_height: usize,
) -> Vec<f32> {
    let scale_x = width as f32 / out_width as f32;
    let scale_y = height as f32 / out_height as f32;
    let sample = |x: usize, y: usize| source[y * width + x];
    let mut out = Vec::with_capacity(out_width * out_height);
    for y in 0..out_height {
        let fy = ((y as f32 + 0.5) * scale_y - 0.5).clamp(0.0, (height - 1) as f32);
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(height - 1);
        let wy = fy - y0 as f32;
        for x in 0..out_width {
            let fx = ((x as f32 + 0.5) * scale_x - 0.5).clamp(0.0, (width - 1) as f32);
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(width - 1);
            let wx = fx - x0 as f32;
            let top = sample(x0, y0) * (1.0 - wx) + sample(x1, y0) * wx;
            let bottom = sample(x0, y1) * (1.0 - wx) + sample(x1, y1) * wx;
            out.push(top * (1.0 - wy) + bottom * wy);
        }
    }
    out
}
