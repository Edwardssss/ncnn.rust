//! Steady-state allocation gate for the reusable-buffer path.

use super::support::{image, layer};
use super::MEASURING;
use ncnn_rust::{BlobId, Mat, Net};

#[test]
fn resolved_buffer_requests_allocate_nothing() {
    let (values, shape) = image(8, 6, 1);
    let net = layer("ReLU", "", &[]);
    let input_id = net.resolve_blob("input").expect("input token");
    let output_id = net.resolve_blob("output").expect("output token");
    let mut buffer = Vec::new();

    // The first request may initialise allocator bookkeeping, so steady state is
    // measured over the following requests.
    request(&net, &input_id, &output_id, &values, &shape, &mut buffer);
    let mut measured = Vec::new();
    MEASURING.with(|flag| flag.set(true));
    for _ in 0..3 {
        let before = super::allocations();
        request(&net, &input_id, &output_id, &values, &shape, &mut buffer);
        measured.push(super::allocations() - before);
    }
    MEASURING.with(|flag| flag.set(false));

    assert!(
        measured[1..].iter().all(|count| *count == 0),
        "steady-state allocations: {measured:?}"
    );
    let expected: Vec<f32> = values.iter().map(|value| value.max(0.0)).collect();
    assert_eq!(buffer, expected);
}

fn request(
    net: &Net,
    input_id: &BlobId,
    output_id: &BlobId,
    values: &[f32],
    shape: &[usize],
    buffer: &mut Vec<f32>,
) {
    let mut extractor = net.extractor().expect("extractor");
    extractor
        .input(input_id, Mat::from_f32(values, shape).expect("input"))
        .expect("submit");
    extractor.extract_into(output_id, buffer).expect("extract");
}
