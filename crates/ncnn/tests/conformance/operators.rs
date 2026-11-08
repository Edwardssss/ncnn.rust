//! Single-operator conformance against Rust reference loops.

use super::support::{
    assert_close, assert_close_labeled, assert_close_with, bilinear, blocks, chain, erf, image,
    layer, plane_index, raw, run, run_labeled, run_two, sigmoid, tagged, two_input,
};
use ncnn_rust::Mat;

const WIDTH: usize = 10;
const HEIGHT: usize = 6;
const CHANNELS: usize = 1;

fn bounded() -> Vec<f32> {
    (0..WIDTH * HEIGHT).map(|i| i as f32 * 0.1 - 3.0).collect()
}

#[test]
fn unary_activations_match_reference() {
    let values = bounded();
    let shape = [WIDTH, HEIGHT, CHANNELS];
    let positive: Vec<f32> = values.iter().map(|v| v.abs() + 0.5).collect();

    type UnaryCase = (&'static str, &'static str, fn(f32) -> f32);
    let cases: &[UnaryCase] = &[
        ("ReLU", "", |x| x.max(0.0)),
        ("ReLU", "0=0.1", |x| if x > 0.0 { x } else { 0.1 * x }),
        ("AbsVal", "", f32::abs),
        ("Sigmoid", "", sigmoid),
        ("TanH", "", f32::tanh),
        ("Exp", "", f32::exp),
        ("Power", "0=2", |x| x * x),
        ("Clip", "0=-1 1=1", |x| x.clamp(-1.0, 1.0)),
        ("Mish", "", |x| {
            if x >= 20.0 {
                x
            } else {
                x * (1.0 + x.exp()).ln().tanh()
            }
        }),
        ("Swish", "", |x| x * sigmoid(x)),
        ("HardSwish", "", |x| x * (0.2 * x + 0.5).clamp(0.0, 1.0)),
        ("HardSigmoid", "", |x| (0.2 * x + 0.5).clamp(0.0, 1.0)),
        ("ELU", "0=0.5", |x| {
            if x > 0.0 {
                x
            } else {
                0.5 * (x.exp() - 1.0)
            }
        }),
        ("SELU", "", |x| {
            let alpha = 1.673_263_2f32;
            let lambda = 1.050_701f32;
            if x > 0.0 {
                lambda * x
            } else {
                lambda * alpha * (x.exp() - 1.0)
            }
        }),
        ("CELU", "", |x| if x > 0.0 { x } else { x.exp() - 1.0 }),
        ("Softplus", "", |x| (1.0 + x.exp()).ln()),
        ("BNLL", "", |x| (1.0 + x.exp()).ln()),
        ("Erf", "", erf),
        ("Threshold", "0=0.5", |x| if x > 0.5 { 1.0 } else { 0.0 }),
        ("Shrink", "0=0.25", |x| {
            if x > 0.5 {
                x - 0.25
            } else if x < -0.5 {
                x + 0.25
            } else {
                0.0
            }
        }),
        ("Dropout", "", |x| x),
        ("Log", "", f32::ln),
    ];

    for (operation, parameters, reference) in cases {
        let source = if *operation == "Log" {
            &positive
        } else {
            &values
        };
        let net = layer(operation, parameters, &[]);
        let actual = run(&net, source, &shape);
        let expected: Vec<f32> = source.iter().map(|v| reference(*v)).collect();
        assert_close_labeled(&actual, &expected, operation, 1e-4);
    }
}

#[test]
fn weighted_layers_match_reference() {
    let (values, shape) = image(WIDTH, HEIGHT, 3);
    let plane = WIDTH * HEIGHT;

    let prelu = layer("PReLU", "0=1", &raw(&[0.25]));
    let expected: Vec<f32> = values
        .iter()
        .map(|v| if *v > 0.0 { *v } else { 0.25 * v })
        .collect();
    assert_close(&run(&prelu, &values, &shape), &expected);

    let scale = layer(
        "Scale",
        "0=3 1=1",
        &blocks(&[&raw(&[2.0, -1.0, 0.5]), &raw(&[1.0, 2.0, 3.0])]),
    );
    let bias = layer("Bias", "0=3", &raw(&[0.5, -0.5, 1.5]));
    let expected_scale: Vec<f32> = values
        .iter()
        .enumerate()
        .map(|(index, v)| v * [2.0, -1.0, 0.5][index / plane] + [1.0, 2.0, 3.0][index / plane])
        .collect();
    assert_close(&run(&scale, &values, &shape), &expected_scale);

    let expected_bias: Vec<f32> = values
        .iter()
        .enumerate()
        .map(|(index, v)| v + [0.5, -0.5, 1.5][index / plane])
        .collect();
    assert_close(&run(&bias, &values, &shape), &expected_bias);

    let slope = [1.5f32, 0.5, 2.0];
    let mean = [0.25f32, -0.5, 1.0];
    let variance = [4.0f32, 1.0, 0.25];
    let shift = [0.1f32, -0.2, 0.3];
    let batch_norm = layer(
        "BatchNorm",
        "0=3 1=0",
        &blocks(&[&raw(&slope), &raw(&mean), &raw(&variance), &raw(&shift)]),
    );
    let expected_norm: Vec<f32> = values
        .iter()
        .enumerate()
        .map(|(index, v)| {
            let channel = index / plane;
            slope[channel] * (v - mean[channel]) / variance[channel].sqrt() + shift[channel]
        })
        .collect();
    assert_close(&run(&batch_norm, &values, &shape), &expected_norm);
}

#[test]
fn binary_operators_match_reference() {
    let values = bounded();
    let divisors: Vec<f32> = values.iter().map(|v| v.abs() + 0.5).collect();
    let shape = [WIDTH, HEIGHT, CHANNELS];

    type BinaryCase = (i32, fn(f32, f32) -> f32);
    let cases: &[BinaryCase] = &[
        (0, |a, b| a + b),
        (1, |a, b| a - b),
        (2, |a, b| a * b),
        (3, |a, b| a / b),
        (4, |a, b| a.max(b)),
        (5, |a, b| a.min(b)),
        (6, |a, b| a.powf(b)),
        (7, |a, b| b - a),
        (8, |a, b| b / a),
    ];

    let positive: Vec<f32> = values.iter().map(|v| v.abs() + 0.5).collect();

    for (code, reference) in cases {
        // ncnn evaluates powf directly, whose negative-base domain is unspecified and
        // differs between the scalar and vector paths, so pow uses a positive base.
        let base = if *code == 6 { &positive } else { &values };
        let net = two_input("BinaryOp", &format!("0={code}"), &[]);
        let actual = run_two(&net, (base, &shape), (&divisors, &shape));
        let expected: Vec<f32> = base
            .iter()
            .zip(&divisors)
            .map(|(a, b)| reference(*a, *b))
            .collect();
        assert_close_labeled(&actual, &expected, &format!("code {code}"), 1e-4);
    }
}

#[test]
fn eltwise_operators_match_reference() {
    let values = bounded();
    let other: Vec<f32> = values.iter().map(|v| v * 0.5 + 1.0).collect();
    let third: Vec<f32> = values.iter().map(|v| v * -0.25).collect();
    let body =
        "Input a 0 1 a\nInput b 0 1 b\nInput c 0 1 c\nEltwise operation 3 1 a b c output 0=1\n";
    let net = super::support::graph(4, 4, body, &[]);
    let mat = |source: &[f32]| Mat::from_f32(source, &[WIDTH, HEIGHT, CHANNELS]).expect("mat");
    let outputs = net
        .infer(
            [("a", mat(&values)), ("b", mat(&other)), ("c", mat(&third))],
            &["output"],
        )
        .expect("infer");
    let expected: Vec<f32> = values
        .iter()
        .zip(&other)
        .zip(&third)
        .map(|((a, b), c)| a + b + c)
        .collect();
    assert_close(&outputs[0].values, &expected);
}

#[test]
fn softmax_axes_match_reference() {
    let (width, height, channels) = (5, 4, 3);
    let (values, shape) = image(width, height, channels);

    for axis in 0..3 {
        let net = layer("Softmax", &format!("0={axis} 1=1"), &[]);
        let actual = run(&net, &values, &shape);
        let expected = softmax_reference(&values, width, height, channels, axis);
        assert_close(&actual, &expected);
    }
}

fn softmax_reference(
    values: &[f32],
    width: usize,
    height: usize,
    channels: usize,
    axis: i32,
) -> Vec<f32> {
    let mut output = values.to_vec();
    match axis {
        0 => {
            for y in 0..height {
                for x in 0..width {
                    softmax_group(
                        (0..channels)
                            .map(|c| plane_index(width, height, x, y, c))
                            .collect::<Vec<_>>()
                            .as_slice(),
                        &mut output,
                    );
                }
            }
        }
        1 => {
            for c in 0..channels {
                for x in 0..width {
                    softmax_group(
                        (0..height)
                            .map(|y| plane_index(width, height, x, y, c))
                            .collect::<Vec<_>>()
                            .as_slice(),
                        &mut output,
                    );
                }
            }
        }
        _ => {
            for c in 0..channels {
                for y in 0..height {
                    softmax_group(
                        (0..width)
                            .map(|x| plane_index(width, height, x, y, c))
                            .collect::<Vec<_>>()
                            .as_slice(),
                        &mut output,
                    );
                }
            }
        }
    }
    output
}

fn softmax_group(indices: &[usize], buffer: &mut [f32]) {
    let maximum = indices
        .iter()
        .map(|index| buffer[*index])
        .fold(f32::MIN, f32::max);
    let mut total = 0.0;
    for index in indices {
        let value = (buffer[*index] - maximum).exp();
        buffer[*index] = value;
        total += value;
    }
    for index in indices {
        buffer[*index] /= total;
    }
}

#[test]
fn structural_layers_match_reference() {
    let (values, shape) = image(8, 6, 4);
    let plane = 8 * 6;

    for operation in ["Flatten", "DeepCopy", "Noop", "Dropout"] {
        let net = layer(operation, "", &[]);
        assert_close_labeled(
            &run_labeled(&net, &values, &shape, operation),
            &values,
            operation,
            1e-4,
        );
    }

    let cast = layer("Cast", "0=0 1=0", &[]);
    assert_close_labeled(
        &run_labeled(&cast, &values, &shape, "Cast"),
        &values,
        "Cast",
        1e-4,
    );

    let reshape = layer("Reshape", "0=6 1=8 2=4", &[]);
    assert_close_labeled(
        &run_labeled(&reshape, &values, &shape, "Reshape"),
        &values,
        "Reshape",
        1e-4,
    );

    let permute = layer("Permute", "0=1", &[]);
    let mut expected = Vec::with_capacity(values.len());
    for chapter in 0..4 {
        for y in 0..8 {
            for x in 0..6 {
                expected.push(values[plane_index(8, 6, y, x, chapter)]);
            }
        }
    }
    assert_close_labeled(
        &run_labeled(&permute, &values, &shape, "Permute"),
        &expected,
        "Permute",
        1e-4,
    );

    let crop = layer("Crop", "0=1 1=1 3=6 4=4 5=4", &[]);
    let mut expected = Vec::new();
    for chapter in 0..4 {
        for y in 1..5 {
            for x in 1..7 {
                expected.push(values[plane_index(8, 6, x, y, chapter)]);
            }
        }
    }
    assert_close_labeled(
        &run_labeled(&crop, &values, &shape, "Crop"),
        &expected,
        "Crop",
        1e-4,
    );

    let padding = layer("Padding", "0=1 1=1 2=1 3=1 4=0 5=0.5", &[]);
    let mut expected = vec![0.5f32; 10 * 8 * 4];
    for chapter in 0..4 {
        for y in 0..6 {
            for x in 0..8 {
                expected[plane_index(10, 8, x + 1, y + 1, chapter)] =
                    values[plane_index(8, 6, x, y, chapter)];
            }
        }
    }
    assert_close_labeled(
        &run_labeled(&padding, &values, &shape, "Padding"),
        &expected,
        "Padding",
        1e-4,
    );

    let shuffle = layer("ShuffleChannel", "0=2", &[]);
    let mut expected = Vec::new();
    for source in [0usize, 2, 1, 3] {
        expected.extend_from_slice(&values[source * plane..(source + 1) * plane]);
    }
    assert_close_labeled(
        &run_labeled(&shuffle, &values, &shape, "ShuffleChannel"),
        &expected,
        "ShuffleChannel",
        1e-4,
    );

    let tile = layer("Tile", "0=0 1=2", &[]);
    let mut expected = values.clone();
    expected.extend_from_slice(&values);
    assert_close_labeled(
        &run_labeled(&tile, &values, &shape, "Tile"),
        &expected,
        "Tile",
        1e-4,
    );
}

#[test]
fn convolution_families_match_reference() {
    let (width, height) = (8, 6);
    let (values, shape) = image(width, height, 1);
    let kernel = [0.05f32, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45];

    for (parameters, stride, dilation, activation) in [
        ("0=1 1=3 4=1 6=9", 1usize, 1usize, false),
        ("0=1 1=3 3=2 4=1 6=9", 2, 1, false),
        ("0=1 1=3 2=2 4=1 6=9", 1, 2, false),
        ("0=1 1=3 4=1 6=9 9=1", 1, 1, true),
    ] {
        let net = layer("Convolution", parameters, &tagged(&kernel));
        let actual = run(&net, &values, &shape);
        let mut expected = convolution_reference(
            &values, width, height, &kernel, 3, 3, stride, stride, dilation, dilation, 1,
        );
        if activation {
            expected.iter_mut().for_each(|v| *v = v.max(0.0));
        }
        assert_close_labeled(&actual, &expected, parameters, 1e-4);
    }

    let depthwise = layer(
        "ConvolutionDepthWise",
        "0=3 1=1 7=3 6=3",
        &tagged(&[2.0, -1.0, 0.5]),
    );
    let plane = width * height;
    let (multi, multi_shape) = image(width, height, 3);
    let expected: Vec<f32> = multi
        .iter()
        .enumerate()
        .map(|(index, v)| v * [2.0, -1.0, 0.5][index / plane])
        .collect();
    assert_close(&run(&depthwise, &multi, &multi_shape), &expected);

    let mut delta = vec![0f32; 9];
    delta[4] = 1.0;
    let deconvolution = layer("Deconvolution", "0=1 1=3 3=1 4=1 6=9", &tagged(&delta));
    assert_close(&run(&deconvolution, &values, &shape), &values);
}

#[allow(clippy::too_many_arguments)]
fn convolution_reference(
    values: &[f32],
    width: usize,
    height: usize,
    kernel: &[f32],
    kernel_width: usize,
    kernel_height: usize,
    stride_x: usize,
    stride_y: usize,
    dilation_x: usize,
    dilation_y: usize,
    pad: usize,
) -> Vec<f32> {
    let (pad_x, pad_y) = (pad, pad);
    let out_width = (width + 2 * pad_x - (kernel_width - 1) * dilation_x - 1) / stride_x + 1;
    let out_height = (height + 2 * pad_y - (kernel_height - 1) * dilation_y - 1) / stride_y + 1;
    let mut output = Vec::with_capacity(out_width * out_height);
    for oy in 0..out_height {
        for ox in 0..out_width {
            let mut sum = 0f32;
            for ky in 0..kernel_height {
                for kx in 0..kernel_width {
                    let x = (ox * stride_x + kx * dilation_x) as isize - pad_x as isize;
                    let y = (oy * stride_y + ky * dilation_y) as isize - pad_y as isize;
                    if x < 0 || y < 0 || x >= width as isize || y >= height as isize {
                        continue;
                    }
                    sum += values[y as usize * width + x as usize] * kernel[ky * kernel_width + kx];
                }
            }
            output.push(sum);
        }
    }
    output
}

#[test]
fn pooling_and_interp_variants_match_reference() {
    let (width, height) = (8, 6);
    let (values, shape) = image(width, height, 1);

    let maximum = layer("Pooling", "0=0 1=2 2=2", &[]);
    let mut expected = Vec::new();
    for y in (0..height).step_by(2) {
        for x in (0..width).step_by(2) {
            let mut best = f32::MIN;
            for dy in 0..2 {
                for dx in 0..2 {
                    best = best.max(values[(y + dy) * width + x + dx]);
                }
            }
            expected.push(best);
        }
    }
    assert_close(&run(&maximum, &values, &shape), &expected);

    let average = layer("Pooling", "0=1 1=3 2=1", &[]);
    let mut expected = Vec::new();
    for y in 0..height - 2 {
        for x in 0..width - 2 {
            let mut sum = 0f32;
            for dy in 0..3 {
                for dx in 0..3 {
                    sum += values[(y + dy) * width + x + dx];
                }
            }
            expected.push(sum / 9.0);
        }
    }
    assert_close(&run(&average, &values, &shape), &expected);

    let global = layer("Pooling", "0=1 4=1", &[]);
    assert_close(
        &run(&global, &values, &shape),
        &[values.iter().sum::<f32>() / values.len() as f32],
    );

    let nearest = layer("Interp", "0=1 1=2 2=2", &[]);
    let mut expected = Vec::with_capacity(width * 4 * height * 2);
    for y in 0..height * 2 {
        for x in 0..width * 2 {
            expected.push(values[(y / 2) * width + (x / 2)]);
        }
    }
    assert_close(&run(&nearest, &values, &shape), &expected);

    let bilinear_net = layer("Interp", "0=2 1=2 2=2", &[]);
    assert_close(
        &run(&bilinear_net, &values, &shape),
        &bilinear(&values, width, height, width * 2, height * 2),
    );
}

#[test]
fn matrix_layers_match_reference() {
    let (gemm_values, gemm_shape) = image(8, 6, 1);
    let mut identity = vec![0f32; 64];
    for index in 0..8 {
        identity[index * 8 + index] = 1.0;
    }
    let gemm = chain(
        &[("Reshape", "0=8 1=6"), ("Gemm", "1=1 3=1 5=1 8=8 9=8")],
        &tagged(&identity),
    );
    assert_close_labeled(
        &run(&gemm, &gemm_values, &gemm_shape),
        &gemm_values,
        "Gemm",
        1e-4,
    );

    let (values, shape) = image(4, 3, 1);

    let mut weights = vec![0f32; 24];
    weights[0] = 1.0;
    weights[13] = 1.0;
    let bias = [0.5f32, -0.5];
    let inner_weights = blocks(&[&tagged(&weights), &raw(&bias)]);
    let inner = super::support::graph(
        2,
        2,
        "Input input 0 1 input\nInnerProduct operation 1 1 input output 0=2 1=1 2=24\n",
        &inner_weights,
    );
    let expected = [
        dot(&weights[0..12], &values) + 0.5,
        dot(&weights[12..24], &values) - 0.5,
    ];
    assert_close_labeled(
        &run(&inner, &values, &shape),
        &expected,
        "InnerProduct",
        1e-4,
    );
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

#[test]
fn packing_and_fp16_options_preserve_values() {
    let (values, shape) = image(16, 12, 4);
    let body = "Input input 0 1 input\nReLU operation 1 1 input output\n";
    let plain = super::support::graph(2, 2, body, &[]);
    let packed = super::support::graph_with(2, 2, body, &[], |builder| {
        builder.flag(ncnn_rust::OptionFlag::Packing, true)
    });
    let fp16 = super::support::graph_with(2, 2, body, &[], |builder| {
        builder.flag(ncnn_rust::OptionFlag::Fp16Storage, true)
    });

    let expected: Vec<f32> = values.iter().map(|v| v.max(0.0)).collect();
    assert_close(&run(&plain, &values, &shape), &expected);
    assert_close(&run(&packed, &values, &shape), &expected);
    assert_close_with(&run(&fp16, &values, &shape), &expected, 1e-2);
}

#[test]
fn rearrangement_layers_match_reference() {
    let (values, shape) = image(4, 3, 4);
    let factor = 2;
    let pixel_shuffle = layer("PixelShuffle", "0=2", &[]);
    let mut expected = vec![0f32; values.len()];
    for sh in 0..factor {
        for sw in 0..factor {
            let source = sh * factor + sw;
            for y in 0..3 {
                for x in 0..4 {
                    let target = plane_index(8, 6, x * factor + sw, y * factor + sh, 0);
                    expected[target] = values[plane_index(4, 3, x, y, source)];
                }
            }
        }
    }
    assert_close_labeled(
        &run_labeled(&pixel_shuffle, &values, &shape, "PixelShuffle"),
        &expected,
        "PixelShuffle",
        1e-4,
    );

    let (single, single_shape) = image(8, 6, 1);
    let squeeze = layer("Squeeze", "2=1", &[]);
    assert_close_labeled(
        &run_labeled(&squeeze, &single, &single_shape, "Squeeze"),
        &single,
        "Squeeze",
        1e-4,
    );
}

#[test]
fn one_dimensional_layers_match_reference() {
    let width = 12;
    let values: Vec<f32> = (0..width).map(|i| i as f32 * 0.5 - 2.0).collect();
    let shape = [width];

    let kernel = [0.5f32, -0.25, 0.75];
    let convolution = layer("Convolution1D", "0=1 1=3 4=1 6=3", &tagged(&kernel));
    let mut expected = Vec::with_capacity(width);
    for x in 0..width as isize {
        let mut sum = 0f32;
        for (k, weight) in kernel.iter().enumerate() {
            let source = x + k as isize - 1;
            if source < 0 || source >= width as isize {
                continue;
            }
            sum += values[source as usize] * weight;
        }
        expected.push(sum);
    }
    assert_close_labeled(
        &run_labeled(&convolution, &values, &shape, "Convolution1D"),
        &expected,
        "Convolution1D",
        1e-4,
    );

    let pooling = layer("Pooling1D", "0=0 1=2 2=2", &[]);
    let expected: Vec<f32> = values
        .chunks_exact(2)
        .map(|pair| pair[0].max(pair[1]))
        .collect();
    assert_close_labeled(
        &run_labeled(&pooling, &values, &shape, "Pooling1D"),
        &expected,
        "Pooling1D",
        1e-4,
    );
}

#[test]
fn deconvolution_upsampling_matches_reference() {
    let (width, height) = (4, 3);
    let (values, shape) = image(width, height, 1);
    let kernel = [0.5f32, 1.0, 1.5, 2.0];
    let deconvolution = layer("Deconvolution", "0=1 1=2 3=2 6=4", &tagged(&kernel));

    let out_width = width * 2;
    let out_height = height * 2;
    let mut expected = vec![0f32; out_width * out_height];
    for y in 0..height {
        for x in 0..width {
            let value = values[y * width + x];
            for ky in 0..2 {
                for kx in 0..2 {
                    expected[(y * 2 + ky) * out_width + x * 2 + kx] = value * kernel[ky * 2 + kx];
                }
            }
        }
    }
    assert_close_labeled(
        &run_labeled(&deconvolution, &values, &shape, "Deconvolution"),
        &expected,
        "Deconvolution",
        1e-4,
    );
}

#[test]
fn spatial_normalisation_matches_reference() {
    let (width, height, channels) = (6, 5, 3);
    let (values, shape) = image(width, height, channels);
    let plane = width * height;
    let eps = 1e-4f32;

    let means: Vec<f32> = (0..channels)
        .map(|c| values[c * plane..(c + 1) * plane].iter().sum::<f32>() / plane as f32)
        .collect();
    let variances: Vec<f32> = (0..channels)
        .map(|c| {
            values[c * plane..(c + 1) * plane]
                .iter()
                .map(|v| (v - means[c]) * (v - means[c]))
                .sum::<f32>()
                / plane as f32
        })
        .collect();
    let normalised: Vec<f32> = values
        .iter()
        .enumerate()
        .map(|(index, v)| (v - means[index / plane]) / (variances[index / plane] + eps).sqrt())
        .collect();

    // Each layer numbers its parameters independently; affine stays off so the
    // reference needs no learned weights.
    for (operation, parameters) in [
        ("MVN", format!("0=1 1=0 2={eps}")),
        ("InstanceNorm", format!("0={channels} 1={eps} 2=0")),
        (
            "GroupNorm",
            format!("0={channels} 1={channels} 2={eps} 3=0"),
        ),
    ] {
        let net = layer(operation, &parameters, &[]);
        assert_close_labeled(
            &run_labeled(&net, &values, &shape, operation),
            &normalised,
            operation,
            1e-3,
        );
    }
}

#[test]
fn reduction_matches_reference() {
    let (values, shape) = image(6, 5, 1);
    let cases: &[(i32, f32)] = &[
        (0, values.iter().sum()),
        (3, values.iter().sum::<f32>() / values.len() as f32),
        (4, values.iter().fold(f32::MIN, |a, b| a.max(*b))),
        (5, values.iter().fold(f32::MAX, |a, b| a.min(*b))),
    ];
    for (operation, expected) in cases {
        let net = layer("Reduction", &format!("0={operation} 1=1"), &[]);
        assert_close_labeled(
            &run_labeled(&net, &values, &shape, "Reduction"),
            &[*expected],
            &format!("Reduction {operation}"),
            1e-4,
        );
    }
}
