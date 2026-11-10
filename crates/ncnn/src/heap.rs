//! Process heap tuning for the request forms that allocate output storage.
//!
//! [`Net::infer`](crate::Net::infer), [`Extractor::extract_owned`](crate::Extractor::extract_owned)
//! and their stored variants hand back independent storage, so every request allocates an
//! output-sized buffer and frees it again. The cost of that cycle belongs to the process heap rather
//! than to the wrapper: a C library that serves an output-sized request from a fresh mapping and
//! returns the pages to the kernel on free makes the next request fault the same pages again.
//!
//! [`retain_pages`] keeps those pages mapped for the rest of the process. It is the in-library
//! counterpart of the `MALLOC_MMAP_THRESHOLD_` and `MALLOC_TRIM_THRESHOLD_` environment variables
//! that `docs/performance.md` records, and it exists because a service cannot always set the
//! environment of its own process.
//!
//! The thresholds are glibc settings. The Windows heap serves the same pattern from committed pages
//! and faults almost nothing, and its public API exposes no equivalent threshold, so every function
//! here reports `false` on that target and leaves the heap alone. `crates/ncnn/tests/heap.rs`
//! measures both hosts: it proves the faults and their removal on glibc, and asserts the absence of
//! the faults and the `false` result on Windows.
//!
//! The setting is process-wide and holds for the rest of the run, like a global allocator, so an
//! application calls it once at start-up. Reusing a buffer through
//! [`Net::infer_into`](crate::Net::infer_into) or
//! [`Net::infer_borrowed_into`](crate::Net::infer_borrowed_into) removes the allocation instead and
//! needs no tuning at all.

/// The allocation size, in bytes, that `retain_pages` keeps on the heap.
///
/// Large enough for any pinned trained-model output, the biggest of which is about 2.9 MB, so an
/// output buffer stays on the heap while a genuinely large allocation still takes its own mapping.
pub const DEFAULT_RETAINED_LIMIT: usize = 64 * 1024 * 1024;

const M_TRIM_THRESHOLD: i32 = -1;
const M_MMAP_THRESHOLD: i32 = -3;

#[cfg(all(target_os = "linux", target_env = "gnu"))]
mod imp {
    use core::ffi::c_int;

    extern "C" {
        fn mallopt(param: c_int, value: c_int) -> c_int;
    }

    pub(super) fn set(param: i32, value: i32) -> bool {
        // SAFETY: `mallopt` only reads its two integer arguments and updates the process heap
        // parameters; it neither keeps pointers nor depends on any Rust state.
        unsafe { mallopt(param, value) != 0 }
    }
}

#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
mod imp {
    pub(super) fn set(_param: i32, _value: i32) -> bool {
        false
    }
}

/// Keeps an output-sized buffer on the heap up to `mapping_limit` bytes.
///
/// The C library serves an allocation above this limit from a fresh mapping and returns it to the
/// kernel on free, which is what makes a later request fault its pages again. Returns `false` when
/// the platform heap has no such setting, which is every target outside glibc Linux.
pub fn set_mapping_limit(mapping_limit: usize) -> bool {
    let value = i32::try_from(mapping_limit).unwrap_or(i32::MAX);
    imp::set(M_MMAP_THRESHOLD, value)
}

/// Stops the heap from returning freed pages to the kernel.
///
/// Returns `false` when the platform heap has no such setting, which is every target outside glibc
/// Linux.
pub fn disable_trimming() -> bool {
    imp::set(M_TRIM_THRESHOLD, -1)
}

/// Keeps the pages of freed buffers mapped so the next request does not fault them again.
///
/// Sets the mapping limit to [`DEFAULT_RETAINED_LIMIT`] and disables trimming. The effect lasts for
/// the rest of the process and grows its resident set by the buffers an application allocates and
/// frees, because those pages stay charged to the process.
///
/// Returns `true` when the heap accepted both settings and `false` when the platform heap has no
/// such setting, which is every target outside glibc Linux, so portable code can call it
/// unconditionally.
///
/// ```
/// let applied = ncnn_rust::heap::retain_pages();
/// println!("retained pages: {applied}");
/// ```
pub fn retain_pages() -> bool {
    retain_pages_up_to(DEFAULT_RETAINED_LIMIT)
}

/// [`retain_pages`] with an explicit mapping limit.
///
/// Returns `true` when the heap accepted both settings.
pub fn retain_pages_up_to(mapping_limit: usize) -> bool {
    let mapped = set_mapping_limit(mapping_limit);
    let trimmed = disable_trimming();
    mapped && trimmed
}
