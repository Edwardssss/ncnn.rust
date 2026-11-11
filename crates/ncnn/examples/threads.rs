//! Runs one network per worker thread and reports aggregate throughput.
//!
//! A network and its extractor are single-threaded by design: both borrow their owning thread, so a
//! server scales by giving each worker its own pair rather than by sharing one. This example builds
//! that pair inside every worker, sets the native thread count to one so the workers do not
//! oversubscribe the cores, and reports what the run achieved next to a single network that uses
//! every core.
//!
//! Run it with `cargo run --release -p ncnn-rust --example threads`.

use ncnn_rust::{Mat, Net};
use std::hint::black_box;
use std::thread;
use std::time::{Duration, Instant};

const GRAPH: &str = "7767517\n2 2\nInput input 0 1 input\nReLU relu 1 1 input output\n";
const ELEMENTS: usize = 65536;
const REQUESTS: usize = 1000;
const ROUNDS: usize = 3;

fn build(native_threads: usize) -> Net {
    Net::builder()
        .unwrap()
        .threads(native_threads)
        .unwrap()
        .load_bytes(GRAPH, &[])
        .unwrap()
}

fn serve(native_threads: usize, requests: usize) -> Duration {
    let net = build(native_threads);
    let values = vec![1.0f32; ELEMENTS];
    let mut extractor = net.extractor().unwrap();
    let mut buffer = Vec::new();
    let start = Instant::now();
    for _ in 0..requests {
        // SAFETY: `values` outlives the extractor, and the request finishes before it is read again.
        let tensor = unsafe { Mat::from_external(&values, &[ELEMENTS]) }.unwrap();
        unsafe { extractor.input_borrowed("input", &tensor) }.unwrap();
        buffer.clear();
        extractor.extract_append("output", &mut buffer).unwrap();
        black_box(buffer[0]);
    }
    start.elapsed()
}

fn parallel(workers: usize, requests: usize) -> Duration {
    let start = Instant::now();
    thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(move || serve(1, requests));
        }
    });
    start.elapsed()
}

/// Runs one configuration for [`ROUNDS`] rounds and returns the median microseconds per request.
fn median_of(mut run: impl FnMut() -> Duration) -> f64 {
    let mut samples: Vec<f64> = (0..ROUNDS)
        .map(|_| run().as_secs_f64() * 1e6 / REQUESTS as f64)
        .collect();
    samples.sort_by(|a, b| a.partial_cmp(b).expect("finite durations"));
    samples[samples.len() / 2]
}

fn report(label: &str, workers: usize, us_per_request: f64) {
    println!(
        "{label:<24} {workers:>7} {:>9} {:>12.0} {us_per_request:>12.1}",
        workers * REQUESTS,
        workers as f64 * 1e6 / us_per_request
    );
}

fn main() {
    let cores = thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    serve(1, 1);
    println!(
        "{:<24} {:>7} {:>9} {:>12} {:>12}",
        "mode", "workers", "requests", "requests/s", "us/request"
    );
    report(
        "one network, every core",
        1,
        median_of(|| serve(cores, REQUESTS)),
    );
    for workers in [1, 2, cores.min(4)] {
        report(
            "one network per worker",
            workers,
            median_of(|| parallel(workers, REQUESTS)),
        );
    }
    println!("\n{cores} logical cores available; native thread count is one per worker.");
}
