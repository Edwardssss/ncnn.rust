//! In-repository operator conformance and allocation checks.
//!
//! Fixtures are written here, so the suite runs without the external comparison
//! directory. Expected values come from Rust reference loops instead of a second
//! backend.

mod allocation;
mod api;
mod graphs;
mod layers;
mod operators;
mod support;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

struct CountingAllocator;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static MEASURING: Cell<bool> = const { Cell::new(false) };
}

fn measuring() -> bool {
    MEASURING.try_with(Cell::get).unwrap_or(false)
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if measuring() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        System.dealloc(pointer, layout)
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn allocations() -> usize {
    ALLOCATIONS.load(Ordering::Relaxed)
}
