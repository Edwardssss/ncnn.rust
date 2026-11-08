//! Raw CPU bindings to ncnn 20261020.
//!
//! Calls preserve upstream ownership and validation requirements. Handles, callback
//! contexts, allocators, and borrowed buffers must remain valid for every native use.
//! Callbacks must contain unwinding, and shared storage requires external alias control.
//!
//! The generated declarations carry the upstream C comments rather than Rust doc
//! comments, so `missing_docs` stays off here; the safe wrappers in `ncnn-rust` deny it.
#![allow(non_upper_case_globals, non_camel_case_types, non_snake_case)]
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
