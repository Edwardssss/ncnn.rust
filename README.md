**English** | [简体中文](README.zh-CN.md)

# ncnn.rust

Rust bindings for [ncnn](https://github.com/Tencent/ncnn) with resource-safe ownership.

The repository contains two packages:

| Package | Role |
|---|---|
| `ncnn-rust` | Safe API: networks, tensors, images, options, GPU resources |
| `ncnn-rust-sys` | Generated C bindings and the project C++ adapters |

## Features

- Owned-input inference with final-vector or reusable-buffer output.
- Scoped read-only CPU output guards that reserve the network without copying.
- Graph-bound endpoint tokens that reject foreign or reloaded graphs.
- Tensors, image preprocessing, custom layers, readers and allocators.
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

## Performance

Wrapper cost is measured against the native C API from the same SDK in one process, so the table
isolates the binding rather than the engine. Median milliseconds per request, one thread:

| Model | Device | native | ncnn-rust | Delta |
|---|---|---:|---:|---:|
| NanoDet-m, 6 outputs | CPU | 23.011 | 23.379 | +1.6% |
| PP-OCRv5 recognition | CPU | 57.566 | 58.842 | +2.2% |
| NanoDet-m, 6 outputs | Vulkan | 4.489 | 4.702 | +4.7% |
| PP-OCRv5 recognition | Vulkan | 10.863 | 10.210 | within spread |

The `ncnn-rust` column resolves endpoints once and reuses a caller vector, which allocates nothing
after warm-up. Returning an owned vector per request instead costs one output copy: NanoDet's six
outputs of 235,200 values add 18% on Vulkan. Single-layer graphs stay at native timing (0.002 ms
for ReLU and AbsVal on CPU). Method and full tables: [Performance](docs/performance.md).

## Validation

CI builds ncnn from source on Linux x86-64, Linux ARM64, Windows, macOS x86-64 and macOS Apple
Silicon, then runs formatting, Clippy, the workspace tests, the example and a publish dry run for
both packages.
Separate jobs check the declared minimum Rust version and run the Vulkan contracts against the
Mesa lavapipe software device. A scheduled version matrix reads `ci/ncnn-versions.json`, builds the pinned
release plus the newest and previous tags, audits the enabled C API against generated bindings and
library symbols, and compares trained-model output with a native C++ reference. `cargo test`
covers the in-repo conformance suite: operator families compared against Rust reference loops,
one load-and-forward case for every other layer type the SDK ships, the public API paths that
the families do not reach, and a zero-allocation steady-state gate. No external model files are
needed.

## Documentation

API documentation is generated by `.github/workflows/docs.yml` on every push to the default
branch and published at <https://edwardssss.github.io/ncnn.rust/ncnn_rust/>. docs.rs cannot build this workspace: its sandbox has
no ncnn SDK, and the build script compiles C++ adapters against the installed headers, so a
missing `NCNN_INSTALL_DIR` fails before rustdoc starts.

- [Documentation map](docs/README.md)
- [Safety contracts and API coverage](docs/contracts.md)
- [Supported platforms](docs/support-matrix.md)
- [Evidence history](docs/evidence-history.md)
- [Performance](docs/performance.md)
- [Advanced CPU adapters](docs/advanced-cpu.md)
- [Model fixtures](docs/model-fixtures.md)
- [Contributing](CONTRIBUTING.md)

## License

BSD-3-Clause.
