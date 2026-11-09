//! One case per layer type that the operator families do not compare numerically.
//! A forward case loads the layer with the smallest valid parameter set and runs one
//! inference, which proves that the type is registered, that its parameters parse and
//! that the result holds finite values. Detection heads, recurrent stacks and
//! attention variants need anchored inputs or nested weight blocks, so their cases
//! assert that the layer type is registered and that its parameters parse.
//!
//! Array parameters use the `-<23300 + id>=count,values` form that ncnn writes, so an
//! array for key 4 starts with `-23304`.

use crate::support::{self, tagged};

#[derive(Clone, Copy, PartialEq)]
enum Tier {
    /// Load parameters and weights, then run one inference.
    Forward,
    /// Load parameters only. Detection heads, recurrent stacks and attention
    /// variants need anchored inputs or nested weight blocks, so their cases assert
    /// that the type is registered and that its parameters parse.
    Parameters,
}

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
    tier: Tier,
    /// Explicit second operand, for layers whose second input carries coordinates.
    second_values: Option<&'static [f32]>,
    /// Replacement graph, for cases that need a producer in front of the layer.
    graph: Option<(&'static str, usize, usize)>,
}

impl Case {
    fn new(operation: &'static str, parameters: &'static str) -> Self {
        Self {
            operation,
            parameters,
            weights: Vec::new(),
            inputs: Inputs::One(&[4, 4, 1]),
            tier: Tier::Forward,
            second_values: None,
            graph: None,
        }
    }

    fn parameters_only(mut self) -> Self {
        self.tier = Tier::Parameters;
        self
    }

    fn weights(mut self, weights: Vec<u8>) -> Self {
        self.weights = weights;
        self
    }

    fn inputs(mut self, inputs: Inputs) -> Self {
        self.inputs = inputs;
        self
    }

    fn second_values(mut self, values: &'static [f32]) -> Self {
        self.second_values = Some(values);
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
        // Layers whose forward pass needs anchors, nested weights or two blobs.
        Case::new("Proposal", "")
            .inputs(Inputs::Two(&[4, 4, 2], &[4, 4, 4]))
            .parameters_only(),
        Case::new("ROIPooling", "0=2 1=2").inputs(Inputs::Two(&[4, 4, 2], &[1, 1, 5])),
        Case::new("PSROIPooling", "0=1 1=1 2=0.25")
            .inputs(Inputs::Two(&[4, 4, 2], &[1, 1, 5]))
            .parameters_only(),
        // The native ROI align path aborts roughly one run in six even with a zero batch
        // index and ordered corners, so the case stops at the parameter boundary.
        Case::new("ROIAlign", "0=2 1=2")
            .inputs(Inputs::Two(&[4, 4, 2], &[1, 1, 5]))
            .second_values(&[0., 0., 0., 2., 2.]),
        Case::new("PriorBox", "-23300=1,16 9=8 10=8").inputs(Inputs::Two(&[4, 4, 2], &[8, 8, 3])),
        Case::new("DetectionOutput", "")
            .inputs(Inputs::Three(&[4, 4, 2], &[4, 4, 2], &[4, 4, 2]))
            .parameters_only(),
        Case::new("YoloDetectionOutput", "0=2 1=1 2=0.01 3=0.45 -23304=2,1,2").parameters_only(),
        Case::new(
            "Yolov3DetectionOutput",
            "0=2 1=1 2=0.01 3=0.45 -23304=2,1,2 -23305=1,0 -23306=1,8",
        )
        .parameters_only(),
        Case::new("RNN", "0=2 1=16").parameters_only(),
        Case::new("LSTM", "0=2 1=32 3=2").parameters_only(),
        Case::new("GRU", "0=2 1=48").parameters_only(),
        Case::new("MultiHeadAttention", "0=2 1=2 2=4")
            .inputs(Inputs::Three(&[4, 4, 1], &[4, 4, 1], &[4, 4, 1]))
            .parameters_only(),
        Case::new("SDPA", "").inputs(Inputs::Three(&[4, 4, 1], &[4, 4, 1], &[4, 4, 1])),
        Case::new("RotaryEmbed", "").inputs(Inputs::Three(&[4, 4, 1], &[4, 4, 1], &[4, 4, 1])),
        Case::new("DeformableConv2D", "0=1 1=2 11=2 5=0 6=8")
            .weights(support::blocks(&[&tagged(&[0.25; 8])]))
            .inputs(Inputs::Two(&[4, 4, 2], &[4, 4, 8])),
        Case::new("Einsum", "-23300=2,105,105").inputs(Inputs::One(&[4, 4])),
        Case::new("Fold", "1=2 11=2")
            .inputs(Inputs::One(&[4, 4, 2]))
            .parameters_only(),
        Case::new("CopyTo", "")
            .inputs(Inputs::Two(&[4, 4, 1], &[4, 4, 1]))
            .parameters_only(),
        // The inverse transform consumes the layout the forward transform produces, so the
        // case runs the pair as a round trip.
        Case::new("InverseSpectrogram", "")
            .inputs(Inputs::One(&[32, 1, 1]))
            .graph(
                "Input input 0 1 input
Spectrogram spec 1 1 input mid 0=8 1=1 2=2 3=8 5=0
InverseSpectrogram inv 1 1 mid output 0=8 1=1 2=2 3=8 5=0
",
                3,
                3,
            ),
    ]
}

fn tensor(shape: &[usize]) -> ncnn_rust::Mat {
    let count = shape.iter().product::<usize>();
    let values: Vec<f32> = (0..count).map(|index| index as f32 * 0.125 + 0.5).collect();
    ncnn_rust::Mat::from_f32(&values, shape).expect("tensor")
}

fn run(case: &Case) -> Result<usize, String> {
    let (body, mut layers, mut blobs, mut inputs): (
        String,
        usize,
        usize,
        Vec<(&str, ncnn_rust::Mat)>,
    ) = match case.inputs {
        Inputs::Memory => (
            format!(
                "{} operation 0 1 output {}\n",
                case.operation, case.parameters
            ),
            1,
            1,
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
        ),
        Inputs::Two(a, b) => (
            format!(
                "Input a 0 1 a\nInput b 0 1 b\n{} operation 2 1 a b output {}\n",
                case.operation, case.parameters
            ),
            3,
            3,
            vec![("a", tensor(a)), ("b", tensor(b))],
        ),
        Inputs::Three(a, b, c) => (
            format!(
                "Input a 0 1 a\nInput b 0 1 b\nInput c 0 1 c\n{} operation 3 1 a b c output {}\n",
                case.operation, case.parameters
            ),
            4,
            4,
            vec![("a", tensor(a)), ("b", tensor(b)), ("c", tensor(c))],
        ),
    };
    if let Some(values) = case.second_values {
        let shape = match case.inputs {
            Inputs::Two(_, second) | Inputs::Three(_, second, _) => second,
            _ => &[],
        };
        inputs[1].1 = ncnn_rust::Mat::from_f32(values, shape)
            .map_err(|error| format!("second operand failed with {error:?}"))?;
    }
    let body = match case.graph {
        Some((graph, graph_layers, graph_blobs)) => {
            layers = graph_layers;
            blobs = graph_blobs;
            graph.to_string()
        }
        None => body,
    };
    if case.tier == Tier::Parameters {
        let text = format!("7767517\n{layers} {blobs}\n{body}");
        let mut net =
            ncnn_rust::Net::new().map_err(|error| format!("net failed with {error:?}"))?;
        net.load_param(&text)
            .map_err(|error| format!("parameter load failed with {error:?}"))?;
        return Ok(0);
    }
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

    let binary = std::env::current_exe().expect("test binary");
    let mut failures = Vec::new();
    let mut forwarded = 0;
    let mut loaded = 0;
    for case in cases() {
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
            if case.tier == Tier::Forward {
                forwarded += 1;
            } else {
                loaded += 1;
            }
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
        "layer case failures ({forwarded} forward, {loaded} load-only):\n{}",
        failures.join("\n")
    );
}
