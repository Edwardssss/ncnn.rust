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
isolate the binding rather than the engine. Median milliseconds per request, one thread, release
build:

| Model | Device | native | ncnn-rust | Delta |
|---|---|---:|---:|---:|
| NanoDet-m, 6 outputs | CPU | 23.011 | 23.379 | +1.6% |
| PP-OCRv5 recognition | CPU | 57.566 | 58.842 | +2.2% |
| NanoDet-m, 6 outputs | Vulkan | 4.489 | 4.702 | +4.7% |
| PP-OCRv5 recognition | Vulkan | 10.863 | 10.210 | within spread |

The `ncnn-rust` column resolves endpoints once and fills a caller vector, which allocates nothing
after warm-up. Returning an owned vector per request instead adds one output copy: 5.294 ms against
4.702 ms on NanoDet Vulkan (+13%), whose six outputs total 235,200 values. The CPU figures move
within the run-to-run spread, which the method section sets at 10%.

A single-layer graph shows the same picture at microsecond scale, where the request path rather than
the arithmetic dominates:

| Case | Device | native | ncnn-rust |
|---|---|---:|---:|
| ReLU | CPU | 0.002 ms | 0.002 ms |
| AbsVal | CPU | 0.002 ms | 0.002 ms |
| ReLU | Vulkan | 0.193 ms | 0.197 ms |
| AbsVal | Vulkan | 0.183 ms | 0.221 ms |

Preprocessing and tensor lifecycle, median microseconds over 65536 FP32 elements, one core, minimum
of three runs:

| Operation | ncnn-rust |
|---|---:|
| borrow (`Mat::from_external`) | 0.09 |
| `deep_clone` | 11.6 |
| `reshape` | 12.5 |
| `into_reshape` | 0.19 |
| `normalized` | 17.7 |
| `normalize_in_place` | 4.7 |
| `flatten` | 0.92 |
| `with_packing(4)` | 0.91 |
| `copy_make_border` | 34.5 |
| `to_pixels` | 36.7 |
| `warp_affine` | 250.2 |

The forms that return independent storage pay one allocation and one copy: `normalized` lands at
about 1.5 times a bare `deep_clone` and the in-place form at 4.7 microseconds, while `borrow`,
`into_reshape`, `flatten` and `with_packing` rewrite or lend a header and stay under a microsecond.
The native reference for the copying forms, including the independent-storage column, is in
[Performance](docs/performance.md).

A request also allocates inside the engine: `NetBuilder::reuse_blob_memory()` gives the network's
intermediate tensors a pool it owns, which is the measured lever on both hosts, and the reuse forms
(`infer_into`, `infer_borrowed_into`) remove the caller-side vector. Microseconds per 65536-element
ReLU request, one core:

| Form | WSL2 glibc, default | WSL2 glibc, pooled | Windows, default | Windows, pooled |
|---|---:|---:|---:|---:|
| `guard` (borrowed output) | 24.2 to 25.6 | 16.4 to 17.0 | 97.4 | 30.1 |
| `extract_owned` (owned output) | 220 to 225 | 22.8 to 22.9 | 194.8 | 84.4 |
| `infer_into` (owned input, reused buffer) | 29.7 to 31.9 | 22.7 to 23.6 | 97.4 | 37.1 |
| `infer_borrowed_into` (borrowed input and buffer) | 22.0 to 23.1 | 16.4 to 17.3 | 39.8 | 18.1 |

The guest numbers are the range of three runs, the Windows numbers the minimum of three runs on a
host that other work was using, so its absolute values move by up to twice between runs while the
ratio holds: the owned-output form gains 2 to 10 times, and the borrowed-buffer form, which allocates
nothing on the caller side, still gains a quarter on the guest and about half on the Windows host from
the pooled intermediates. Linux hosts show the same ranking through page faults, and
`ncnn_rust::heap::retain_pages()` keeps the pages of a caller-allocated buffer mapped. The
[thread example](crates/ncnn/examples/threads.rs) covers the next lever: a network and its extractor
per worker, measured below.

One network per worker thread scales a service. On an 8-core WSL2 guest, four workers reach 105k
requests per second against 38k for one worker, while a single network that uses every core lands at
42k: the native thread pool and one worker on one thread cost nearly the same on a graph this small,
and independent workers multiply the throughput. On a 16-core Windows host the same ranking holds at
45k to 62k for four workers against 13k to 17k for one network over every core, with a much wider
spread because that host was shared. The example reports the median of three rounds. The method, the
full tables, the pooled, worker and Linux measurements and the reproduction commands are in
[Performance](docs/performance.md).

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
