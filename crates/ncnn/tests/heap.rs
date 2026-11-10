//! Checks that heap retention removes the page faults of a freed output buffer.
//!
//! The measurement needs a glibc heap, because the settings the crate exposes are the glibc ones.
//! Both checks run in one test, because the settings they change are process-wide.

#![cfg(all(target_os = "linux", target_env = "gnu"))]

use ncnn_rust::heap;

const BUFFER_BYTES: usize = 256 * 1024;
const ROUNDS: u64 = 200;

fn minor_faults() -> u64 {
    let stat = std::fs::read_to_string("/proc/self/stat").expect("process statistics");
    let after_command = stat.rsplit_once(')').expect("command field").1;
    after_command
        .split_whitespace()
        .nth(7)
        .expect("minor fault counter")
        .parse()
        .expect("minor fault count")
}

/// Holds two distinct buffers alive at once, so the second one cannot reuse the first free.
fn two_buffer_rounds(rounds: u64) -> u64 {
    let before = minor_faults();
    for _ in 0..rounds {
        let first = vec![1.0f32; BUFFER_BYTES / size_of::<f32>()];
        let second = vec![2.0f32; BUFFER_BYTES / size_of::<f32>()];
        std::hint::black_box(first[0] + second[0]);
    }
    minor_faults() - before
}

#[test]
fn retaining_pages_removes_the_faults_of_a_freed_buffer() {
    let untuned = two_buffer_rounds(ROUNDS);
    assert!(heap::retain_pages(), "the glibc heap accepted the settings");
    let tuned = two_buffer_rounds(ROUNDS);
    assert!(
        tuned * 4 < untuned,
        "faults per round fell from {} to {}",
        untuned / ROUNDS,
        tuned / ROUNDS
    );
    assert!(heap::set_mapping_limit(heap::DEFAULT_RETAINED_LIMIT));
    assert!(heap::disable_trimming());
    assert!(heap::retain_pages_up_to(heap::DEFAULT_RETAINED_LIMIT));
}
