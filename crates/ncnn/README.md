# ncnn-rust

Rust CPU and optional Vulkan inference interfaces for an externally installed ncnn 20261020 SDK.
The companion `ncnn-rust-sys` package provides raw bindings and C ABI adapters; this package
re-exports them as `ncnn_rust::sys`.

## Install

```sh
cargo add ncnn-rust
```

The crate ships Rust code only. Building it compiles the C ABI adapters against an ncnn 20261020
SDK that the build environment supplies.

## Build environment

Set `NCNN_INSTALL_DIR` to a matching SDK with `include/ncnn` and `lib` or `lib64`.
`NCNN_LINK_KIND` selects `static` (default) or `dylib`. A shared SDK also needs the directory that
holds the shared library on the runtime loader path: `lib` on Unix, and the `bin` directory that
CMake installs `ncnn.dll` into on Windows, where a static and an import library share the
`ncnn.lib` name. `NCNN_EXTRA_LIBS` supplies comma-separated
native runtime dependencies in link order, optionally using `static=` or `dylib=` prefixes, such as
`gomp` for a GCC OpenMP static SDK. `NCNN_EXTRA_LIB_DIRS` supplies existing dependency library
directories separated by the platform path separator. Cargo compiles the adapters and generates
bindings with a target-compatible C++ compiler and host-loadable libclang.

The SDK enables the C API, stdio, strings, pixel helpers, batch, INT8, BF16 and validation. Default
builds use Vulkan disabled; `--features vulkan` requires a Vulkan-enabled SDK. Static Vulkan SDKs
additionally require an explicit nonempty `NCNN_EXTRA_LIBS` dependency closure matching their
exported CMake targets. Headers and libraries come from the same build and match the Rust target,
C++ ABI and CRT. The build accepts Linux GNU, Windows MSVC and macOS on x86-64 or ARM64.

## Inference

```rust,no_run
use ncnn_rust::{Mat, Net};

fn main() -> ncnn_rust::Result<()> {
    let net = Net::builder()?.threads(2)?.load_files("model.param", "model.bin")?;
    let input = Mat::from_f32(&vec![0.; 224 * 224 * 3], &[224, 224, 3])?;
    let outputs = net.infer([("input", input)], &["output"])?;
    println!("{:?} {}", outputs[0].info, outputs[0].values.len());
    Ok(())
}
```

`PixelImage` checks color formats, stride and ROI before conversion or resizing. Read-only tensor
views borrow their tensor and expose logical initialized data. `input` transfers crate-owned
storage; raw-adopted inputs are copied so external storage dependencies stay with the caller.

Batched storage is built with `Mat::from_f32_batch`, and `TensorInfo::n` reports the batch count.
`Mat::into_normalized` and `Mat::into_reshape` consume a tensor the caller no longer needs, which lets
the native call run on that storage instead of on a duplicate; the `&self` forms keep returning
independent storage. `Net::layers` and `Net::blobs` describe a loaded graph without running it.

A fresh input can transfer storage and borrow an output while its extractor remains exclusively
borrowed:

```rust,no_run
# fn run(net: &ncnn_rust::Net) -> ncnn_rust::Result<()> {
let mut ex = net.extractor()?;
ex.input("input", ncnn_rust::Mat::from_f32(&[1., 2., 3.], &[3])?)?;
let output = ex.extract("output")?;
println!("{:?}", output.channel_f32(0)?);
# Ok(())
# }
```

Multiple inputs transfer in order with `inputs`. Multiple outputs can be consumed sequentially with
`outputs_with`; each callback borrows one guard and releases it before the next extraction:

```rust,no_run
# fn run(net: &ncnn_rust::Net, a: ncnn_rust::Mat, b: ncnn_rust::Mat) -> ncnn_rust::Result<()> {
let mut ex = net.extractor()?;
ex.inputs([("a", a), ("b", b)])?;
ex.outputs_with(&["sum", "product"], |name, output| {
    println!("{name}: {:?}", output.channel_f32(0)?);
    Ok(())
})?;
# Ok(())
# }
```

Both methods stop at the first error; earlier inputs or callbacks remain applied. `Mat::from_f32`
copies the source slice into owned storage, and `inputs` adds no wrapper tensor copy for that
storage. `outputs_with` adds no detached tensor copy, and `to_f32` explicitly copies logical values.

`OutputGuard` exposes checked immutable views and explicit FP32 vector copying; it exposes no raw or
owned tensor handle. Drop the guard, or end its scope, before reusing its extractor, including when
the guard is otherwise unused. `Endpoint::Index` selects a blob by index. Native unpacking,
precision conversion and local-pool detachment still apply, and the caller selects FP32
interpretation according to the model. Raw-handle export, options whose raw handle was exported
before configuration, native path model loading and custom-layer registration disable guard
extraction. Detached extraction remains available for advanced allocator and callback workflows.

For repeated inference, resolve endpoints once and fill one reusable vector:

```rust,no_run
# fn run(net: &ncnn_rust::Net, values: Vec<f32>) -> ncnn_rust::Result<()> {
let input_id = net.resolve_blob("input")?;
let output_id = net.resolve_blob("output")?;
let mut ex = net.extractor()?;
ex.input(&input_id, ncnn_rust::Mat::from_f32(&values, &[3])?)?;
let mut buffer = Vec::new();
ex.extract_into(&output_id, &mut buffer)?;
# Ok(())
# }
```

Forms that return independent storage, such as `Net::infer` and `Extractor::extract_owned`, allocate
and free an output-sized vector per request. A glibc heap returns those pages to the kernel, so the
next request faults them again; `heap::retain_pages` keeps them mapped for the rest of the process,
and the reuse form above needs no such call because it allocates no data buffer.

Exact affinity is an OS capability. Linux and Windows expose processor IDs and thread masks; macOS
reports `Error::Unsupported` for exact mask operations while core inference and physical-core
queries remain separate.

Quantized and half-precision outputs keep the width the producing layer wrote. `Net::infer_stored`
returns `TensorInfo` and `Mat`, `Extractor::extract_stored` borrows one for the extractor scope and
`Extractor::extract_stored_owned` returns independent storage, all readable through
`Mat::to_f32_typed(kind)` or `Mat::to_i8()`. `Mat::from_typed` and `Mat::from_i8` build the same
storage for an input side, and `ElementKind` names each format. `infer` converts to FP32 and stays
the form for models that end in FP32.

MSRV: Rust 1.85.1. License: BSD-3-Clause. Native ncnn retains its upstream license.

## Optional Vulkan

The `vulkan` feature needs a matching Vulkan-enabled SDK, either shared or an explicitly configured
static SDK with nonempty `NCNN_EXTRA_LIBS` matching its exported dependency closure. Device
selection is explicit through `Net::builder()?.vulkan(device)`, and CPU stays the default.
`cargo run -p ncnn-rust --features vulkan --example add -- --vulkan` demonstrates synchronous
CPU-Mat inference on the selected device. Native per-layer CPU fallback remains possible.

Resident `gpu::Tensor`, `Allocator`, blocking-completion `Command`, completed `TensorGuard`, cache
persistence and unsafe same-device buffer imports are available. FP32 extractor guards reject device
networks, while stored extraction serves them because the download copy is detached from the pool
allocator. Nonblocking submission and arbitrary external device adoption are unavailable in the
pinned native API.

## Performance

The benchmark compares against the native C API from the same SDK in one process, so it isolates the
binding from the engine. Median milliseconds per request, one thread:

| Model | Device | native | ncnn-rust | Delta |
|---|---|---:|---:|---:|
| NanoDet-m, 6 outputs | CPU | 23.011 | 23.379 | +1.6% |
| PP-OCRv5 recognition | CPU | 57.566 | 58.842 | +2.2% |
| NanoDet-m, 6 outputs | Vulkan | 4.489 | 4.702 | +4.7% |
| PP-OCRv5 recognition | Vulkan | 10.863 | 10.210 | within spread |

The `ncnn-rust` column resolves endpoints once and reuses a caller vector, which allocates nothing
after warm-up. Single-layer graphs stay at native timing: ReLU and AbsVal both measure 0.002 ms on
CPU. Returning an owned vector per request instead costs one output copy, which is where the
NanoDet Vulkan gap of 18% comes from.
