//! Multi-input, multi-branch and multi-output graph conformance.

use super::support::{assert_close, bilinear, graph, image, plane_index, run, run_two, tagged};
use ncnn_rust::{Mat, Net};

#[test]
fn split_and_concat_match_reference() {
    let (width, height) = (8, 6);
    let (values, shape) = image(width, height, 1);
    let body = "Input input 0 1 input\n\
Split split 1 2 input branch_a branch_b\n\
ReLU relu 1 1 branch_a rectified\n\
AbsVal abs 1 1 branch_b magnitude\n\
Concat concat 2 1 rectified magnitude output 0=0\n";
    let net = graph(5, 6, body, &[]);
    let mut expected: Vec<f32> = values.iter().map(|v| v.max(0.0)).collect();
    expected.extend(values.iter().map(|v| v.abs()));
    assert_close(&run(&net, &values, &shape), &expected);
}

#[test]
fn residual_convolution_matches_reference() {
    let (width, height) = (8, 6);
    let (values, shape) = image(width, height, 1);
    let kernel = [0.05f32, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45];
    let body = "Input input 0 1 input\n\
Convolution conv 1 1 input convolved 0=1 1=3 4=1 6=9\n\
Eltwise add 2 1 convolved input output 0=1\n";
    let net = graph(3, 4, body, &tagged(&kernel));

    let mut convolved = vec![0f32; width * height];
    for y in 0..height as i32 {
        for x in 0..width as i32 {
            let mut sum = 0f32;
            for ky in -1..=1 {
                for kx in -1..=1 {
                    let (sx, sy) = (x + kx, y + ky);
                    if sx < 0 || sy < 0 || sx >= width as i32 || sy >= height as i32 {
                        continue;
                    }
                    sum += values[(sy * width as i32 + sx) as usize]
                        * kernel[((ky + 1) * 3 + (kx + 1)) as usize];
                }
            }
            convolved[y as usize * width + x as usize] = sum;
        }
    }
    let expected: Vec<f32> = values.iter().zip(&convolved).map(|(v, c)| v + c).collect();
    assert_close(&run(&net, &values, &shape), &expected);
}

#[test]
fn pool_split_interp_concat_matches_reference() {
    let (width, height) = (8, 6);
    let (values, shape) = image(width, height, 1);
    let body = "Input input 0 1 input\n\
Pooling pool 1 1 input pooled 0=0 1=2 2=2\n\
Split split 1 2 pooled near far\n\
Interp nearest 1 1 near up_nearest 0=1 1=2 2=2\n\
Interp linear 1 1 far up_linear 0=2 1=2 2=2\n\
Concat concat 2 1 up_nearest up_linear output 0=0\n";
    let net = graph(6, 7, body, &[]);

    let pooled_width = width / 2;
    let pooled_height = height / 2;
    let mut pooled = Vec::new();
    for y in (0..height).step_by(2) {
        for x in (0..width).step_by(2) {
            let mut best = f32::MIN;
            for dy in 0..2 {
                for dx in 0..2 {
                    best = best.max(values[(y + dy) * width + x + dx]);
                }
            }
            pooled.push(best);
        }
    }

    let mut expected = Vec::new();
    for y in 0..pooled_height * 2 {
        for x in 0..pooled_width * 2 {
            expected.push(pooled[(y / 2) * pooled_width + x / 2]);
        }
    }
    expected.extend(bilinear(&pooled, pooled_width, pooled_height, 8, 6));
    assert_close(&run(&net, &values, &shape), &expected);
}

#[test]
fn dual_input_chain_matches_reference() {
    let (width, height) = (6, 5);
    let (a, shape) = image(width, height, 1);
    let b: Vec<f32> = a.iter().map(|v| v * 0.5 + 1.0).collect();
    let body = "Input a 0 1 a\n\
Input b 0 1 b\n\
BinaryOp multiply 2 1 a b product 0=2\n\
ReLU relu 1 1 product output\n";
    let net = graph(4, 4, body, &[]);
    let expected: Vec<f32> = a
        .iter()
        .zip(&b)
        .map(|(left, right)| (left * right).max(0.0))
        .collect();
    let actual = net
        .infer(
            [
                ("a", Mat::from_f32(&a, &shape).expect("a")),
                ("b", Mat::from_f32(&b, &shape).expect("b")),
            ],
            &["output"],
        )
        .expect("infer");
    assert_close(&actual[0].values, &expected);
}

#[test]
fn multi_output_extraction_matches_reference() {
    let (values, shape) = image(8, 6, 1);
    let mut identity = vec![0f32; 64];
    for index in 0..8 {
        identity[index * 8 + index] = 1.0;
    }
    let body = "Input input 0 1 input\n\
Reshape reshape 1 1 input flattened 0=8 1=6\n\
Gemm gemm 1 1 flattened logits 1=1 3=1 5=1 8=8 9=8\n\
Softmax softmax 1 1 logits output 0=1 1=1\n";
    let net = graph(4, 5, body, &tagged(&identity));

    let outputs = net
        .infer(
            [("input", Mat::from_f32(&values, &shape).expect("input"))],
            &["flattened", "logits", "output"],
        )
        .expect("infer");
    assert_eq!(outputs.len(), 3);
    assert_close(&outputs[0].values, &values);
    assert_close(&outputs[1].values, &values);

    // Gemm returns a two-dimensional blob, where axis 1 normalises each row.
    let probabilities = &outputs[2].values;
    assert_eq!(outputs[2].info.dims, 2);
    assert_eq!(probabilities.len(), values.len());
    for y in 0..6 {
        let row: f32 = (0..8)
            .map(|x| probabilities[plane_index(8, 6, x, y, 0)])
            .sum();
        assert!((row - 1.0).abs() < 1e-5, "row {y} sums to {row}");
    }
    assert!(probabilities.iter().all(|value| *value >= 0.0));
}

#[test]
fn repeated_runs_reproduce_results() {
    let (values, shape) = image(8, 6, 1);
    let net: Net = graph(
        3,
        3,
        "Input input 0 1 input\nBinaryOp sub 1 1 input sum 0=7 1=1 2=0.25\nReLU relu 1 1 sum output\n",
        &[],
    );
    let first = run(&net, &values, &shape);
    let second = run(&net, &values, &shape);
    assert_eq!(first, second);

    let other: Vec<f32> = values.iter().map(|v| v - 2.0).collect();
    let third = run(&net, &other, &shape);
    assert_ne!(first, third);
    let expected: Vec<f32> = other.iter().map(|v| (0.25 - v).max(0.0)).collect();
    assert_close(&third, &expected);
}

#[test]
fn dual_input_elementwise_graph_matches_reference() {
    let (width, height) = (5, 4);
    let (a, shape) = image(width, height, 1);
    let b: Vec<f32> = a.iter().map(|v| v * -0.75 + 2.0).collect();
    let net = graph(
        3,
        3,
        "Input a 0 1 a\nInput b 0 1 b\nBinaryOp maximum 2 1 a b output 0=4\n",
        &[],
    );
    let expected: Vec<f32> = a.iter().zip(&b).map(|(l, r)| l.max(*r)).collect();
    assert_close(&run_two(&net, (&a, &shape), (&b, &shape)), &expected);
}
