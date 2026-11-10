//! Measures the owned-output request before and after heap retention.
//!
//! `ncnn_rust::heap::retain_pages` keeps the pages of a freed output buffer mapped, which removes
//! the per-request page faults a glibc heap otherwise pays on the forms that return independent
//! storage. The call applies to glibc Linux and reports `false` elsewhere, so on other targets the
//! two figures differ only by host noise. The reused-buffer form allocates no data buffer, so
//! retention must not move it.
//!
//! Run it with `cargo run --release -p ncnn-rust --example heap_retention`.

use ncnn_rust::{heap, Mat, Net};
use std::hint::black_box;
use std::time::Instant;

const GRAPH: &str = "7767517\n2 2\nInput input 0 1 input\nReLU relu 1 1 input output\n";
const ELEMENTS: usize = 65536;
const ITERATIONS: usize = 200;

fn owned(net: &Net, values: &[f32]) -> f32 {
    let output = net
        .infer(
            [("input", Mat::from_f32(values, &[values.len()]).unwrap())],
            &["output"],
        )
        .unwrap();
    black_box(output[0].values[0])
}

fn fused(net: &Net, values: &[f32], buffer: &mut Vec<f32>) -> f32 {
    let ranges = net
        .infer_borrowed_into(&[("input", values, &[values.len()])], &["output"], buffer)
        .unwrap();
    black_box(buffer[ranges[0].values.clone()][0])
}

fn timed(mut body: impl FnMut() -> f32) -> f64 {
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        body();
    }
    start.elapsed().as_nanos() as f64 / ITERATIONS as f64
}

/// The host keeps moving, so each metric is the minimum of three rounds.
fn best(mut body: impl FnMut() -> f32) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..3 {
        best = best.min(timed(&mut body));
    }
    best
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let values = vec![1.0f32; ELEMENTS];
    let net = Net::builder()?.threads(1)?.load_bytes(GRAPH, &[])?;
    let mut buffer = Vec::new();
    for _ in 0..20 {
        owned(&net, &values);
        fused(&net, &values, &mut buffer);
    }

    let owned_before = best(|| owned(&net, &values));
    let fused_before = best(|| fused(&net, &values, &mut buffer));
    let retained = heap::retain_pages();
    let owned_after = best(|| owned(&net, &values));
    let fused_after = best(|| fused(&net, &values, &mut buffer));

    println!(
        "heap_retention={retained}, elements={ELEMENTS}, owned_before_ns={owned_before:.0}, \
         owned_after_ns={owned_after:.0}, fused_before_ns={fused_before:.0}, fused_after_ns={fused_after:.0}"
    );
    Ok(())
}
