//! One case per layer type that the operator families do not compare numerically.
//! Each case loads the layer with the smallest valid parameter set and runs one
//! inference, which proves that the type is registered, that its parameters parse and
//! that the result holds finite values. Detection heads, recurrent stacks and attention
//! variants therefore carry the weights, operand values or producer layer their forward
//! pass needs instead of stopping at the parameter boundary.
//!
//! Array parameters use the `-<23300 + id>=count,values` form that ncnn writes, so an
//! array for key 4 starts with `-23304`.

use crate::support::{self, tagged};

#[derive(Clone, Copy)]
enum Inputs {
    Memory,
    One(&'static [usize]),
    Two(&'static [usize], &'static [usize]),
    Three(&'static [usize], &'static [usize], &'static [usize]),
}

struct Case {
    operation: &'static str,
    parameters: &'static str,
    weights: Vec<u8>,
    inputs: Inputs,
    /// Explicit operand values, for layers whose inputs carry coordinates, scores or
    /// image metadata that a plain ramp cannot express.
    values: Vec<(usize, &'static [f32])>,
    /// Replacement graph with its layer and blob counts, for layers that need a producer.
    graph: Option<(&'static str, usize, usize)>,
}

impl Case {
    fn new(operation: &'static str, parameters: &'static str) -> Self {
        Self {
            operation,
            parameters,
            weights: Vec::new(),
            inputs: Inputs::One(&[4, 4, 1]),
            values: Vec::new(),
            graph: None,
        }
    }

    fn weights(mut self, weights: Vec<u8>) -> Self {
        self.weights = weights;
        self
    }

    fn inputs(mut self, inputs: Inputs) -> Self {
        self.inputs = inputs;
        self
    }

    fn operand(mut self, index: usize, values: &'static [f32]) -> Self {
        self.values.push((index, values));
        self
    }

    fn graph(mut self, body: &'static str, layers: usize, blobs: usize) -> Self {
        self.graph = Some((body, layers, blobs));
        self
    }
}

fn depthwise(kernel: usize, num_output: usize, input_channels: usize) -> Vec<u8> {
    support::blocks(&[&tagged(&vec![
        0.25;
        num_output * input_channels * kernel * kernel
    ])])
}

fn depthwise3d(kernel: usize, num_output: usize, input_channels: usize) -> Vec<u8> {
    support::blocks(&[&tagged(&vec![
        0.25;
        num_output
            * input_channels
            * kernel
            * kernel
            * kernel
    ])])
}

fn cases() -> Vec<Case> {
    vec![
        Case::new("LRN", "0=0 1=3 2=0.0001 3=0.75 4=1"),
        Case::new("UnaryOp", ""),
        Case::new("ExpandDims", "-23300=1,0"),
        Case::new("Reorg", "0=2 1=0"),
        Case::new("StatisticsPooling", ""),
        Case::new("GELU", ""),
        Case::new("GLU", "0=0").inputs(Inputs::One(&[4, 4, 2])),
        Case::new("CumulativeSum", "0=0"),
        Case::new("Diag", "0=0").inputs(Inputs::One(&[4])),
        Case::new("Flip", "-23300=1,0"),
        Case::new("Pooling3D", "0=0 1=1 2=1 3=1"),
        Case::new("Pooling1D", "0=0 1=1").inputs(Inputs::One(&[4, 1, 1])),
        Case::new("Convolution1D", "0=1 1=2 3=1 5=0 6=2")
            .weights(support::blocks(&[&tagged(&[0.5, 0.5])]))
            .inputs(Inputs::One(&[4, 1, 1])),
        Case::new("Deconvolution1D", "0=1 1=2 3=1 5=0 6=2")
            .weights(support::blocks(&[&tagged(&[0.5, 0.5])]))
            .inputs(Inputs::One(&[2, 1, 1])),
        Case::new("Convolution3D", "0=1 1=2 11=2 21=2 3=1 13=1 23=1 5=0 6=8")
            .weights(support::blocks(&[&tagged(&[0.25; 8])]))
            .inputs(Inputs::One(&[2, 2, 2, 1])),
        Case::new("Deconvolution3D", "0=1 1=2 11=2 21=2 3=1 13=1 23=1 5=0 6=8")
            .weights(support::blocks(&[&tagged(&[0.25; 8])]))
            .inputs(Inputs::One(&[2, 2, 2, 1])),
        Case::new("Embed", "0=3 1=4 2=0 3=12")
            .weights(support::blocks(&[&tagged(&[
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
            ])]))
            .inputs(Inputs::One(&[4, 1, 1])),
        Case::new("LayerNorm", "0=4 1=4")
            .weights(support::blocks(&[&tagged(&[1.0; 4]), &tagged(&[0.0; 4])]))
            .inputs(Inputs::One(&[4, 1, 1])),
        Case::new("RMSNorm", "0=4")
            .weights(support::blocks(&[&tagged(&[1.0; 4])]))
            .inputs(Inputs::One(&[4, 1, 1])),
        Case::new("Normalize", "3=1").weights(support::blocks(&[&tagged(&[1.0])])),
        Case::new("MatMul", "").inputs(Inputs::Two(&[2, 2, 1], &[2, 2, 1])),
        // Depthwise families carry the group count in key 7.
        Case::new("ConvolutionDepthWise", "0=2 1=2 11=2 5=0 6=8 7=2")
            .weights(depthwise(2, 2, 1))
            .inputs(Inputs::One(&[4, 4, 2])),
        Case::new("DeconvolutionDepthWise", "0=2 1=2 11=2 5=0 6=8 7=2")
            .weights(depthwise(2, 2, 1))
            .inputs(Inputs::One(&[2, 2, 2])),
        Case::new("ConvolutionDepthWise1D", "0=1 1=2 5=0 6=2 7=1")
            .weights(depthwise(2, 1, 1))
            .inputs(Inputs::One(&[4, 1, 1])),
        Case::new("DeconvolutionDepthWise1D", "0=1 1=2 5=0 6=2 7=1")
            .weights(depthwise(2, 1, 1))
            .inputs(Inputs::One(&[2, 1, 1])),
        Case::new(
            "ConvolutionDepthWise3D",
            "0=1 1=2 11=2 21=2 3=1 13=1 23=1 5=0 6=8 7=1",
        )
        .weights(depthwise3d(2, 1, 1))
        .inputs(Inputs::One(&[2, 2, 2, 1])),
        Case::new(
            "DeconvolutionDepthWise3D",
            "0=1 1=2 11=2 21=2 3=1 13=1 23=1 5=0 6=8 7=1",
        )
        .weights(depthwise3d(2, 1, 1))
        .inputs(Inputs::One(&[2, 2, 2, 1])),
        // Quantisation trio: one scale per tensor, optional bias on requantize.
        Case::new("Quantize", "0=1").weights(support::blocks(&[&tagged(&[1.0])])),
        Case::new("Dequantize", "0=1").weights(support::blocks(&[&tagged(&[1.0])])),
        Case::new("Requantize", "0=1 1=1")
            .weights(support::blocks(&[&tagged(&[1.0]), &tagged(&[1.0])])),
        // Shape and sampling layers.
        Case::new("Slice", "-23300=2,1,1"),
        Case::new("Unfold", "1=2 11=2").inputs(Inputs::One(&[4, 4, 2])),
        // Sampling modes are covered by the whole matrix in this case; padding mode 0 is
        // not a valid value, and the ARM64 run reports any kernel divergence by name.
        Case::new("GridSample", "0=1 1=1 2=1").inputs(Inputs::Two(&[4, 4, 2], &[1, 1, 2])),
        Case::new("Spectrogram", "0=8 1=1 2=2 3=8 5=0").inputs(Inputs::One(&[16, 1, 1])),
        Case::new("MemoryData", "0=4 1=1 2=1 21=1")
            .weights(support::blocks(&[&support::raw(&[0.5; 4])]))
            .inputs(Inputs::Memory),
        // Layers whose forward pass decodes regions or detections, so their operands carry
        // coordinates, scores or image metadata instead of a plain ramp.
        // Proposal propagates 9 generated anchors over a two by two score map and keeps
        // what survives the size filter, so the metadata carries a real image size.
        Case::new("Proposal", "0=16 1=16 2=100 3=50 4=0.7 5=16")
            .inputs(Inputs::Three(&[2, 2, 18], &[2, 2, 36], &[3]))
            .operand(2, &[128., 128., 1.]),
        Case::new("ROIPooling", "0=2 1=2").inputs(Inputs::Two(&[4, 4, 2], &[1, 1, 5])),
        // The region pooling reads one region per row of four coordinates, and the
        // channel count must match the pooled grid times the output dimension.
        Case::new("PSROIPooling", "0=1 1=1 2=1 3=4")
            .inputs(Inputs::Two(&[8, 8, 4], &[4]))
            .operand(1, &[0., 0., 5., 5.]),
        // Region alignment samples inside the feature map, so the case passes a region
        // whose corners sit within the four by four operand.
        Case::new("ROIAlign", "0=2 1=2")
            .inputs(Inputs::Two(&[4, 4, 2], &[1, 1, 5]))
            .operand(1, &[0., 0., 0., 2., 2.]),
        Case::new("PriorBox", "-23300=1,16 9=8 10=8").inputs(Inputs::Two(&[4, 4, 2], &[8, 8, 3])),
        // Detection heads decode locations against priors, so the case supplies a prior
        // pair of box and variance rows plus confidence scores that clear the threshold.
        Case::new("DetectionOutput", "0=2 1=0.05 2=300 3=100 4=0.5")
            .inputs(Inputs::Three(&[16], &[8], &[16, 2]))
            .operand(0, &[0.1; 16])
            .operand(1, &[0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9])
            .operand(2, &[
                10., 10., 50., 50., 10., 10., 50., 50., 10., 10., 50., 50., 10., 10., 50., 50.,
                0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.2,
            ]),
        Case::new("YoloDetectionOutput", "0=2 1=1 2=0.01 3=0.45 -23304=2,1,2")
            .inputs(Inputs::One(&[4, 4, 7])),
        Case::new(
            "Yolov3DetectionOutput",
            "0=2 1=1 2=0.01 3=0.45 -23304=2,1,2 -23305=1,0 -23306=1,8",
        )
        .inputs(Inputs::One(&[4, 4, 7])),
        // Recurrent stacks load three tagged blocks in order: input weights, bias and
        // recurrent weights. The input carries four steps of four features.
        Case::new("RNN", "0=2 1=8")
            .weights(support::blocks(&[
                &tagged(&[0.1; 8]),
                &tagged(&[0.0; 2]),
                &tagged(&[0.1; 4]),
            ]))
            .inputs(Inputs::One(&[4, 4])),
        Case::new("LSTM", "0=2 1=32 3=2")
            .weights(support::blocks(&[
                &tagged(&[0.1; 32]),
                &tagged(&[0.0; 8]),
                &tagged(&[0.1; 8]),
            ]))
            .inputs(Inputs::One(&[4, 4])),
        Case::new("GRU", "0=2 1=24")
            .weights(support::blocks(&[
                &tagged(&[0.1; 24]),
                &tagged(&[0.0; 8]),
                &tagged(&[0.1; 12]),
            ]))
            .inputs(Inputs::One(&[4, 4])),
        // Attention loads eight blocks: four tagged projection weights and four raw
        // biases. `2` is the product of the embedding width and the projected width.
        Case::new("MultiHeadAttention", "0=4 1=2 2=16")
            .weights(support::blocks(&[
                &tagged(&[0.25; 16]),
                &support::raw(&[0.0; 4]),
                &tagged(&[0.25; 16]),
                &support::raw(&[0.0; 4]),
                &tagged(&[0.25; 16]),
                &support::raw(&[0.0; 4]),
                &tagged(&[0.25; 16]),
                &support::raw(&[0.0; 4]),
            ]))
            .inputs(Inputs::Three(&[4, 4, 1], &[4, 4, 1], &[4, 4, 1])),
        Case::new("SDPA", "").inputs(Inputs::Three(&[4, 4, 1], &[4, 4, 1], &[4, 4, 1])),
        Case::new("RotaryEmbed", "").inputs(Inputs::Three(&[4, 4, 1], &[4, 4, 1], &[4, 4, 1])),
        Case::new("DeformableConv2D", "0=1 1=2 11=2 5=0 6=8")
            .weights(support::blocks(&[&tagged(&[0.25; 8])]))
            .inputs(Inputs::Two(&[4, 4, 2], &[4, 4, 8])),
        Case::new("Einsum", "-23300=2,105,105").inputs(Inputs::One(&[4, 4])),
        // Fold consumes the layout Unfold produces, so the case chains the pair.
        Case::new("Fold", "")
            .inputs(Inputs::One(&[4, 4, 2]))
            .graph(
                "Input input 0 1 input\nUnfold unfold 1 1 input mid 1=2 11=2 3=2 13=2 4=0\nFold fold 1 1 mid output 1=2 11=2 3=2 13=2 4=0 20=4 21=4\n",
                3,
                3,
            ),
        // Copy overwrites the leading corner of the first operand with the smaller one.
        Case::new("CopyTo", "")
            .inputs(Inputs::Two(&[4, 4, 1], &[2, 2, 1])),
        // The inverse transform reads one channel per frequency bin, so the operand carries
        // a real and imaginary pair per frame and `n_fft / 2 + 1` bins.
        Case::new("InverseSpectrogram", "0=8 1=1 2=2 3=8 5=0").inputs(Inputs::One(&[2, 4, 5])),
    ]
}

fn tensor(shape: &[usize]) -> ncnn_rust::Mat {
    let count = shape.iter().product::<usize>();
    let values: Vec<f32> = (0..count).map(|index| index as f32 * 0.125 + 0.5).collect();
    ncnn_rust::Mat::from_f32(&values, shape).expect("tensor")
}

/// Assembled graph text, its declared layer and blob counts, the input tensors and the
/// shapes those tensors were built from.
type Fixture = (
    String,
    usize,
    usize,
    Vec<(&'static str, ncnn_rust::Mat)>,
    Vec<&'static [usize]>,
);

fn run(case: &Case) -> Result<usize, String> {
    let (body, mut layers, mut blobs, mut inputs, shapes): Fixture = match case.inputs {
        Inputs::Memory => (
            format!(
                "{} operation 0 1 output {}\n",
                case.operation, case.parameters
            ),
            1,
            1,
            Vec::new(),
            Vec::new(),
        ),
        Inputs::One(shape) => (
            format!(
                "Input input 0 1 input\n{} operation 1 1 input output {}\n",
                case.operation, case.parameters
            ),
            2,
            2,
            vec![("input", tensor(shape))],
            vec![shape],
        ),
        Inputs::Two(a, b) => (
            format!(
                "Input a 0 1 a\nInput b 0 1 b\n{} operation 2 1 a b output {}\n",
                case.operation, case.parameters
            ),
            3,
            3,
            vec![("a", tensor(a)), ("b", tensor(b))],
            vec![a, b],
        ),
        Inputs::Three(a, b, c) => (
            format!(
                "Input a 0 1 a\nInput b 0 1 b\nInput c 0 1 c\n{} operation 3 1 a b c output {}\n",
                case.operation, case.parameters
            ),
            4,
            4,
            vec![("a", tensor(a)), ("b", tensor(b)), ("c", tensor(c))],
            vec![a, b, c],
        ),
    };
    for &(index, values) in &case.values {
        inputs[index].1 = ncnn_rust::Mat::from_f32(values, shapes[index])
            .map_err(|error| format!("operand {index} failed with {error:?}"))?;
    }
    let body = match case.graph {
        Some((graph, graph_layers, graph_blobs)) => {
            layers = graph_layers;
            blobs = graph_blobs;
            graph.to_string()
        }
        None => body,
    };
    let net = support::try_graph(layers, blobs, &body, &case.weights)
        .map_err(|error| format!("load failed with {error:?}"))?;
    let mut outputs = net
        .infer(inputs, &["output"])
        .map_err(|error| format!("inference failed with {error:?}"))?;
    let values = outputs.remove(0).values;
    if values.is_empty() {
        return Err("empty output".into());
    }
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(format!("non-finite value at index {index}"));
    }
    Ok(values.len())
}

#[test]
fn uncovered_layers_load_and_forward() {
    // A layer that aborts in native code takes the whole process with it, so each
    // case runs in a child process of this binary. The child reports a case by name
    // and the parent keeps the failure, which also survives a heap corruption.
    if let Ok(name) = std::env::var("NCNN_LAYER_CASE") {
        let case = cases()
            .into_iter()
            .find(|case| case.operation == name)
            .expect("known case");
        if let Err(reason) = run(&case) {
            panic!("{name}: {reason}");
        }
        return;
    }

    // NCNN_LAYER_INLINE runs the whole selection in one process, which is how the
    // sequential-state investigation reproduces the abort.
    let start: usize = std::env::var("NCNN_LAYER_START")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    let limit: usize = std::env::var("NCNN_LAYER_LIMIT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(usize::MAX);
    if std::env::var("NCNN_LAYER_INLINE").is_ok() {
        let mut failures = Vec::new();
        for (index, case) in cases().into_iter().enumerate() {
            if index < start || index >= start + limit {
                continue;
            }
            eprintln!("inline case {}", case.operation);
            if let Err(reason) = run(&case) {
                failures.push(format!("{}: {reason}", case.operation));
            }
        }
        assert!(
            failures.is_empty(),
            "inline failures:\n{}",
            failures.join("\n")
        );
        return;
    }

    let binary = std::env::current_exe().expect("test binary");
    let mut failures = Vec::new();
    let mut passed = 0;
    for (index, case) in cases().into_iter().enumerate() {
        if index < start || index >= start + limit {
            continue;
        }
        let output = std::process::Command::new(&binary)
            .args([
                "uncovered_layers_load_and_forward",
                "--exact",
                "--test-threads=1",
                "--nocapture",
            ])
            .env("NCNN_LAYER_CASE", case.operation)
            .output()
            .expect("child process");
        if output.status.success() {
            passed += 1;
            continue;
        }
        let stderr = String::from_utf8_lossy(&output.stderr);
        let detail = stderr
            .lines()
            .find(|line| line.contains(&format!("{}:", case.operation)))
            .or_else(|| stderr.lines().find(|line| line.contains("panicked")))
            .unwrap_or("child process aborted without a message");
        failures.push(format!(
            "{}: {} (status {:?})",
            case.operation,
            detail.trim(),
            output.status.code()
        ));
    }
    assert!(
        failures.is_empty(),
        "layer case failures ({passed} of {} cases passed):\n{}",
        cases().len(),
        failures.join("\n")
    );
}
