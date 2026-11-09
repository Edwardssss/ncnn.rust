//! One smoke case per layer type that the operator families do not compare
//! numerically. Each case loads the layer with the smallest valid parameter set and
//! runs one forward pass, which proves that the type is registered, that its
//! parameters parse and that the result holds finite values.

use crate::support::{self, tagged};

struct Case {
    operation: &'static str,
    parameters: &'static str,
    weights: Vec<u8>,
    shape: &'static [usize],
    /// Second operand for layers that take two bottoms.
    second: Option<&'static [usize]>,
}

impl Case {
    fn new(operation: &'static str, parameters: &'static str) -> Self {
        Self {
            operation,
            parameters,
            weights: Vec::new(),
            shape: &[4, 4, 1],
            second: None,
        }
    }

    fn weights(mut self, weights: Vec<u8>) -> Self {
        self.weights = weights;
        self
    }

    fn shape(mut self, shape: &'static [usize]) -> Self {
        self.shape = shape;
        self
    }

    fn two_inputs(mut self, second: &'static [usize]) -> Self {
        self.second = Some(second);
        self
    }
}

fn cases() -> Vec<Case> {
    vec![
        Case::new("LRN", "0=0 1=3 2=0.0001 3=0.75 4=1"),
        Case::new("UnaryOp", ""),
        Case::new("ExpandDims", "-23300=1,0"),
        Case::new("Reorg", "0=2 1=0"),
        Case::new("StatisticsPooling", ""),
        Case::new("GELU", ""),
        Case::new("GLU", "0=0").shape(&[4, 4, 2]),
        Case::new("CumulativeSum", "0=0"),
        Case::new("Diag", "0=0").shape(&[4]),
        Case::new("Flip", "-23300=1,0"),
        Case::new("Pooling3D", "0=0 1=1 2=1 3=1"),
        Case::new("Pooling1D", "0=0 1=1").shape(&[4, 1, 1]),
        Case::new("Convolution1D", "0=1 1=2 3=1 5=0 6=2")
            .weights(support::blocks(&[&tagged(&[0.5, 0.5])]))
            .shape(&[4, 1, 1]),
        Case::new("Deconvolution1D", "0=1 1=2 3=1 5=0 6=2")
            .weights(support::blocks(&[&tagged(&[0.5, 0.5])]))
            .shape(&[2, 1, 1]),
        Case::new("Convolution3D", "0=1 1=2 11=2 21=2 3=1 13=1 23=1 5=0 6=8")
            .weights(support::blocks(&[&tagged(&[0.25; 8])]))
            .shape(&[2, 2, 2, 1]),
        Case::new("Deconvolution3D", "0=1 1=2 11=2 21=2 3=1 13=1 23=1 5=0 6=8")
            .weights(support::blocks(&[&tagged(&[0.25; 8])]))
            .shape(&[2, 2, 2, 1]),
        Case::new("Embed", "0=3 1=4 2=0 3=12")
            .weights(support::blocks(&[&tagged(&[
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
            ])]))
            .shape(&[4, 1, 1]),
        Case::new("LayerNorm", "0=4 1=4")
            .weights(support::blocks(&[&tagged(&[1.0; 4]), &tagged(&[0.0; 4])]))
            .shape(&[4, 1, 1]),
        Case::new("RMSNorm", "0=4")
            .weights(support::blocks(&[&tagged(&[1.0; 4])]))
            .shape(&[4, 1, 1]),
        Case::new("Normalize", "3=1").weights(support::blocks(&[&tagged(&[1.0])])),
        Case::new("MatMul", "")
            .two_inputs(&[2, 2, 1])
            .shape(&[2, 2, 1]),
    ]
}

fn tensor(shape: &[usize]) -> ncnn_rust::Mat {
    let count = shape.iter().product::<usize>();
    let values: Vec<f32> = (0..count).map(|index| index as f32 * 0.125 + 0.5).collect();
    ncnn_rust::Mat::from_f32(&values, shape).expect("tensor")
}

fn run(case: &Case) -> Result<Vec<f32>, String> {
    let body = match case.second {
        None => format!(
            "Input input 0 1 input\n{} operation 1 1 input output {}\n",
            case.operation, case.parameters
        ),
        Some(_) => format!(
            "Input a 0 1 a\nInput b 0 1 b\n{} operation 2 1 a b output {}\n",
            case.operation, case.parameters
        ),
    };
    let (layers, blobs) = if case.second.is_some() {
        (3, 3)
    } else {
        (2, 2)
    };
    let net = support::try_graph(layers, blobs, &body, &case.weights)
        .map_err(|error| format!("load failed with {error:?}"))?;
    let inputs: Vec<(&str, ncnn_rust::Mat)> = match case.second {
        None => vec![("input", tensor(case.shape))],
        Some(second) => vec![("a", tensor(case.shape)), ("b", tensor(second))],
    };
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
    Ok(values)
}

#[test]
fn uncovered_layers_load_and_forward() {
    let mut failures = Vec::new();
    for case in cases() {
        if let Err(reason) = run(&case) {
            failures.push(format!("{}: {reason}", case.operation));
        }
    }
    assert!(
        failures.is_empty(),
        "layer smoke failures:\n{}",
        failures.join("\n")
    );
}
