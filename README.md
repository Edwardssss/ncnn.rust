**English** | [简体中文](README_zh.md)

# ncnn.rust

Rust bindings for [ncnn](https://github.com/Tencent/ncnn) with resource-safe ownership.

The repository contains two packages:

| Package | Role |
|---|---|
| `ncnn-rust` | Safe API: networks, tensors, images, options, GPU resources |
| `ncnn-rust-sys` | Generated C bindings and the project C++ adapters |

## Features

- Owned-input inference with final-vector or reusable-buffer output.
- Typed tensor access: `ElementKind` readers and writers cover FP16, BF16 and the integer widths,
  and `infer_stored` keeps the element width a layer wrote.
- Batched tensors, layout rewriting with `flatten` and `with_packing`, and consuming helpers
  (`into_normalized`/`into_reshape`) that skip the copy when the caller owns the storage.
- Zero-copy requests: `Mat::from_external`, `Extractor::input_borrowed` and `Net::infer_borrowed`
  submit caller memory without a copy.
- Graph introspection: `Net::layers` and `Net::blobs` report the wiring, the shapes and the support
  flags ncnn declares for each layer type, and `layer_type_index` resolves a type name to its index.
- Scoped read-only CPU output guards that reserve the network without copying.
- Graph-bound endpoint tokens that reject foreign or reloaded graphs.
- Tensors, image preprocessing, custom layers, readers and allocators, including in-place
  normalization, pixel output, letterbox and depth borders, and affine warping.
- Effective option readback on a loaded network, and caller-owned pools for blobs, workspace and the
  attention KV cache.
- Optional Vulkan backend: device selection, resident tensors, allocator, recording
  commands, completed guards, buffer import and persistent pipeline caches.
- Explicit native configuration mapping without Rust-side hardware gating.

## Install

```sh
cargo add ncnn-rust
```

The crates ship Rust code only. Building them compiles the C++ adapters against an ncnn SDK that
the build environment supplies, so the next section sets that SDK up.

## Build

Cargo consumes an externally installed ncnn SDK. `NCNN_INSTALL_DIR` is required and must
contain `include/ncnn` plus `lib`. Rust 1.85.1 is the tested MSRV.

```sh
export NCNN_INSTALL_DIR=/absolute/path/to/ncnn-install
export NCNN_LINK_KIND=dylib
cargo test --workspace
cargo run -p ncnn-rust --example add
```

Required native features: `NCNN_C_API`, `NCNN_STDIO`, `NCNN_STRING`, `NCNN_PIXEL`,
`NCNN_PIXEL_ROTATE`, `NCNN_PIXEL_AFFINE`, `NCNN_PIXEL_DRAWING`, `NCNN_BATCH`, `NCNN_INT8`,
`NCNN_BF16`, `NCNN_VALIDATION`. The default build requires `NCNN_VULKAN=OFF`; the `vulkan`
Cargo feature requires `NCNN_VULKAN=ON` and a shared SDK with its dependency closure.

`NCNN_LINK_KIND` selects `static` (default) or `dylib`. A shared SDK also needs the directory
that holds the shared library on the runtime loader path: `lib` through `LD_LIBRARY_PATH` or
`DYLD_LIBRARY_PATH`, and the `bin` directory that CMake installs `ncnn.dll` into through `PATH`
on Windows, where a static and an import library share the `ncnn.lib` name. `NCNN_EXTRA_LIBS`
lists additional
libraries in link order, optionally prefixed with `static=` or `dylib=`. `NCNN_EXTRA_LIB_DIRS` adds
search directories. The SDK version is pinned to 20261020; `NCNN_ALLOW_UNTESTED_VERSION=1` builds
against another release for compatibility checks.

`tools/fetch-ncnn.py` verifies the pinned source archive, and `tools/build-ncnn-sdk.py` applies the
patches in `tools/patches/` and builds an SDK with the required feature set.

## Usage

```rust
use ncnn_rust::{Mat, Net};

let net = Net::builder()?.threads(2)?.load_files("model.param", "model.bin")?;
let input = Mat::from_f32(&values, &[224, 224, 3])?;
let outputs = net.infer([("input", input)], &["output"])?;
for output in outputs {
    println!("{:?} {}", output.info, output.values.len());
}
```

For repeated inference resolve endpoints once and fill one reusable vector:

```rust
let input_id = net.resolve_blob("input")?;
let output_id = net.resolve_blob("output")?;
let mut ex = net.extractor()?;
ex.input(&input_id, Mat::from_f32(&values, &[224, 224, 3])?)?;
let mut buffer = Vec::new();
ex.extract_into(&output_id, &mut buffer)?;
```

Run on a Vulkan device by attaching it before the model is read. Native per-layer CPU fallback stays
available, so selection is not a guarantee that every layer runs on the GPU:

```rust
let device = ncnn_rust::gpu::Device::default_device()?;
let net = Net::builder()?.vulkan(device).load_files("model.param", "model.bin")?;
```

Quantized and half-precision outputs keep their stored width. `infer` converts to FP32, so read them
through the stored forms and name the format:

```rust
let outputs = net.infer_stored([("input", input)], &["output"])?;
let (info, stored) = &outputs[0];
println!("{} bytes per element", info.elemsize);
let bytes: Vec<i8> = stored.to_i8()?;
```

`Mat::from_typed` and `Mat::from_i8` build the same storage for an input side, and
`Extractor::extract_stored` borrows it while one extractor runs. The
[typed tensor example](crates/ncnn/examples/typed_tensors.rs) and the
[resident Vulkan example](crates/ncnn/examples/vulkan_resident.rs) run end to end.

An extractor is safe to reuse: submitting an input restarts the request, so a second inference
recomputes instead of returning the output the first one cached. Per-extractor settings cover what
the network options do not, and they survive that restart:

```rust
let mut ex = net.extractor()?;
if net.shared_blobs()?.is_empty() {
    ex.set_light_mode(true)?;
}
ex.set_kvcache_max_seqlen_hint(2048)?;
```

A graph that keeps state across inputs, such as an attention cache for token-by-token decoding,
calls `ex.continue_session()` before the next input so that input extends the cached state instead
of restarting the request. Light mode releases each intermediate after its first consumer, so
`set_light_mode(true)` is refused while `net.shared_blobs()` lists a blob that feeds two layers;
insert an explicit `Split` layer or keep light mode off.

Part of a tensor is readable without copying it. A view aliases the source storage: the slice
readers borrow it, `to_f32` copies out of it, and `Extractor::input` copies it because an alias keeps
no storage of its own. `Net::infer_view_into` reads the region where it lives, which is the
zero-copy form for a model that takes a crop, a channel or one batch item:

```rust
let mut buffer = Vec::new();
let ranges = net.infer_view_into(&[("input", &image.channel(0)?)], &["output"], &mut buffer)?;
let range = &ranges[0].values;
let values = &buffer[range.start..range.end];
```

A network and its extractor stay on the thread that created them, so a service gives each worker its
own pair; the [thread example](crates/ncnn/examples/threads.rs) measures that against one network
that uses every core.

## Performance

Wrapper cost is measured against the native C API from the same SDK in one process, so the tables
isolate the binding rather than the engine. Every number below comes from one machine and one
campaign; the medians move with host load, so compare rows rather than absolute values.

> **Hardware:** AMD Ryzen 7 5800H (8 cores, 16 threads), 16 GB DDR4-3200, NVIDIA GeForce RTX 3060
> Laptop GPU (6 GB, driver 596.49), Windows 10 22H2 build 19045, MSVC 19.50, Rust 1.96.1, ncnn
> 20261020 built from source with the repository's build script. Vulkan rows run on the same GPU;
> the Linux figures quoted in [Performance](docs/performance.md) come from a Debian WSL2 guest on
> this machine with 8 logical cores.

Trained models, median milliseconds per request, one thread, release build:

| Model | Device | native | ncnn-rust | Delta |
|---|---|---:|---:|---:|
| NanoDet-m, 6 outputs | CPU | 23.011 | 23.379 | +1.6% |
| PP-OCRv5 recognition | CPU | 57.566 | 58.842 | +2.2% |
| NanoDet-m, 6 outputs | Vulkan | 4.489 | 4.702 | +4.7% |
| PP-OCRv5 recognition | Vulkan | 10.863 | 10.210 | within spread |

The `ncnn-rust` column resolves endpoints once and fills a caller vector, which allocates nothing
after warm-up. A vector owned per request instead adds one output copy, which is why the reuse form
is the one quoted; the CPU deltas sit inside the campaign's 10% run-to-run spread.

A single-layer graph shows the same picture at microsecond scale, where the request path rather than
the arithmetic dominates:

| Case | Device | native | ncnn-rust |
|---|---|---:|---:|
| ReLU | CPU | 0.002 ms | 0.002 ms |
| AbsVal | CPU | 0.002 ms | 0.002 ms |
| ReLU | Vulkan | 0.193 ms | 0.197 ms |
| AbsVal | Vulkan | 0.183 ms | 0.221 ms |

The request path over one 65536-element FP32 tensor, median microseconds per request, one thread.
The native reference creates one extractor, borrows its input and copies into a reusable buffer, the
same work the first row performs:

| Form | native | ncnn-rust | Delta |
|---|---:|---:|---:|
| borrowed input, reusable output buffer | 25.462 | 19.734 | -22% |
| owned input, borrowed output view | 25.462 | 13.720 | -46% |
| owned input, independent output | 25.462 | 107.419 | +322% |
| cloned input, cloned output | 25.462 | 368.352 | +1347% |

The first two rows hold every result where it already lives, so the wrapper stays below the C
reference; the last two show what one and three extra copies of a 256 KB output cost on this host.
That is why the reuse forms exist: `NetBuilder::reuse_blob_memory()` moves the engine's intermediate
tensors into pools it owns, `infer_into` and `infer_borrowed_into` remove the caller vector, and
`ncnn_rust::heap::retain_pages()` keeps a caller-allocated buffer's pages mapped on glibc. The
per-helper preprocessing medians, the pooling comparison, the Linux page-fault campaign, the worker
scaling and the reproduction commands are in [Performance](docs/performance.md).

## Documentation

API documentation is generated by `.github/workflows/docs.yml` on every push to the default
branch and published at <https://edwardssss.github.io/ncnn.rust/ncnn_rust/>. docs.rs cannot build this workspace: its sandbox has
no ncnn SDK, and the build script compiles C++ adapters against the installed headers, so a
missing `NCNN_INSTALL_DIR` fails before rustdoc starts.

- [Documentation map](docs/README.md)
- [Safety contracts](docs/contracts.md)
- [Native ABI and adapter decisions](docs/native-abi.md)
- [Layer coverage](docs/layer-coverage.md)
- [Supported platforms](docs/support-matrix.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Migration notes](docs/migration.md)
- [Evidence history](docs/evidence-history.md)
- [Performance](docs/performance.md)
- [Advanced CPU adapters](docs/advanced-cpu.md)
- [Model fixtures](docs/model-fixtures.md)
- [Changelog](CHANGELOG.md)
- [Security policy](SECURITY.md)
- [Contributing](CONTRIBUTING.md)

## License

BSD-3-Clause.
