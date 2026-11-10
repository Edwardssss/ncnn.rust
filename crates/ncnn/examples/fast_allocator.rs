//! Runs the request path under a drop-in allocator.
//!
//! The wrapper never sets an allocator for its caller, so a library cannot choose one: the
//! application declares it, exactly as below. On Windows the default heap is much slower than a
//! thread-caching allocator for the repeated allocate-and-free of an output-sized buffer, which is
//! what an owned-output request performs. Measured on one host, the owned-output request moved from
//! about 360 us to about 40 us per request, while the borrowed-input and reused-buffer form stayed
//! where it was because it allocates no data buffer at all.
//!
//! Run it with `cargo run --release -p ncnn-rust --example fast_allocator`.

use ncnn_rust::{Mat, Net};
use std::hint::black_box;
use std::time::Instant;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let values = vec![1.0f32; ELEMENTS];
    let net = Net::builder()?.threads(1)?.load_bytes(GRAPH, &[])?;
    let mut buffer = Vec::new();
    for _ in 0..20 {
        owned(&net, &values);
        fused(&net, &values, &mut buffer);
    }
    // The host keeps moving, so each metric is the minimum of three rounds.
    let mut owned_ns = f64::MAX;
    let mut fused_ns = f64::MAX;
    for _ in 0..3 {
        owned_ns = owned_ns.min(timed(|| owned(&net, &values)));
        fused_ns = fused_ns.min(timed(|| fused(&net, &values, &mut buffer)));
    }
    println!(
        "allocator=mimalloc, elements={ELEMENTS}, owned_ns_per_request={owned_ns:.0}, fused_ns_per_request={fused_ns:.0}"
    );
    Ok(())
}
