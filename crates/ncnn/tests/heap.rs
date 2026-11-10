//! Checks the heap behaviour that the retention settings target on each host.
//!
//! The buffer pattern is the same everywhere: two distinct output-sized buffers held alive at once,
//! which is what an owned-output request and its input copy perform. A glibc heap serves them from a
//! fresh mapping, so every round faults the pages again, and `heap::retain_pages` removes those
//! faults. The Windows heap serves the same pattern from committed pages and faults almost nothing,
//! which is why the crate reports the settings as unavailable there.

#![cfg(any(all(target_os = "linux", target_env = "gnu"), windows))]

const BUFFER_BYTES: usize = 256 * 1024;
const ROUNDS: u64 = 200;

/// Holds two distinct buffers alive at once, so the second one cannot reuse the first free.
fn two_buffer_rounds(rounds: u64, count: impl Fn() -> u64) -> u64 {
    let before = count();
    for _ in 0..rounds {
        let first = vec![1.0f32; BUFFER_BYTES / size_of::<f32>()];
        let second = vec![2.0f32; BUFFER_BYTES / size_of::<f32>()];
        std::hint::black_box(first[0] + second[0]);
    }
    count() - before
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
mod glibc {
    use ncnn_rust::heap;

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

    #[test]
    fn retaining_pages_removes_the_faults_of_a_freed_buffer() {
        let untuned = super::two_buffer_rounds(super::ROUNDS, minor_faults);
        assert!(heap::retain_pages(), "the glibc heap accepted the settings");
        let tuned = super::two_buffer_rounds(super::ROUNDS, minor_faults);
        assert!(
            tuned * 4 < untuned,
            "faults per round fell from {} to {}",
            untuned / super::ROUNDS,
            tuned / super::ROUNDS
        );
        assert!(heap::set_mapping_limit(heap::DEFAULT_RETAINED_LIMIT));
        assert!(heap::disable_trimming());
        assert!(heap::retain_pages_up_to(heap::DEFAULT_RETAINED_LIMIT));
    }
}

#[cfg(windows)]
mod windows {
    use ncnn_rust::heap;
    use std::ffi::c_void;

    #[repr(C)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }

    #[link(name = "psapi")]
    extern "system" {
        fn GetProcessMemoryInfo(
            process: *mut c_void,
            counters: *mut ProcessMemoryCounters,
            size: u32,
        ) -> i32;
    }

    extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
    }

    fn page_faults() -> u64 {
        let mut counters = ProcessMemoryCounters {
            cb: std::mem::size_of::<ProcessMemoryCounters>() as u32,
            page_fault_count: 0,
            peak_working_set_size: 0,
            working_set_size: 0,
            quota_peak_paged_pool_usage: 0,
            quota_paged_pool_usage: 0,
            quota_peak_non_paged_pool_usage: 0,
            quota_non_paged_pool_usage: 0,
            pagefile_usage: 0,
            peak_pagefile_usage: 0,
        };
        // SAFETY: both handles and the record size come from this process, and the record lives
        // for the call.
        let status = unsafe {
            GetProcessMemoryInfo(
                GetCurrentProcess(),
                std::ptr::addr_of_mut!(counters),
                counters.cb,
            )
        };
        assert_ne!(status, 0, "process memory information");
        u64::from(counters.page_fault_count)
    }

    #[test]
    fn the_windows_heap_keeps_the_pages_of_a_freed_buffer() {
        let faults = super::two_buffer_rounds(super::ROUNDS, page_faults);
        let per_round = faults / super::ROUNDS;
        // The same pattern faults about 96 pages per round on a glibc host.
        assert!(per_round < 8, "faults per round: {per_round}");
    }

    #[test]
    fn the_retention_settings_are_unavailable() {
        assert!(!heap::retain_pages());
        assert!(!heap::retain_pages_up_to(heap::DEFAULT_RETAINED_LIMIT));
        assert!(!heap::set_mapping_limit(heap::DEFAULT_RETAINED_LIMIT));
        assert!(!heap::disable_trimming());
    }
}
