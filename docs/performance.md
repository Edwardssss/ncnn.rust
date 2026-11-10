# Performance

The reference is the native ncnn C API from the same SDK, measured in the same process and the
same campaign. This isolates the wrapper cost instead of engine or toolchain differences. Every
timed request includes input preparation, extractor setup, inference, output materialisation,
Vulkan synchronisation when applicable, and cleanup. Model loading is measured separately.

## Method

- Release builds; three warmups and seven rotated rounds per mode.
- Synthetic graphs use 30 requests per round; trained models use two.
- One thread, packing off, FP16 and BF16 storage off, GPU device 0.
- Correctness is checked before and after timing against the native reference on the same device.
- Cross-repetition spread reaches 10%, so a difference below 5% is reported as `within spread`.

## Trained models

Median milliseconds per request. `native` calls the C API directly. `buffer` resolves endpoints
once and fills a caller vector, which allocates nothing after warm-up. `final` returns a fresh
vector per request.

| Mode | NanoDet CPU | OCR CPU | NanoDet Vulkan | OCR Vulkan |
|---|---:|---:|---:|---:|
| native | 23.011 | 57.566 | 4.489 | 10.863 |
| ncnn-rust buffer | 23.379 | 58.842 | 4.702 | 10.210 |
| ncnn-rust final | 21.471 | 59.421 | 5.294 | 11.186 |

Deltas against native: buffer +1.6% (NanoDet CPU), +2.2% (OCR CPU), +4.7% (NanoDet Vulkan) and
-6.0% (OCR Vulkan, within spread). The owned-vector mode adds one output copy per request; NanoDet
returns six outputs totalling 235,200 values, which is where the +18% on NanoDet Vulkan comes from.
Absolute values shift between campaigns with host state, so compare rows measured in one campaign.

## Operator overhead

Steady-state medians for a single-layer graph, including the full request path:

| Case | Device | native | ncnn-rust buffer |
|---|---|---:|---:|
| ReLU | CPU | 0.002 ms | 0.002 ms |
| AbsVal | CPU | 0.002 ms | 0.002 ms |
| ReLU | Vulkan | 0.193 ms | 0.197 ms |
| AbsVal | Vulkan | 0.183 ms | 0.221 ms |

Host-side work is unchanged at the microsecond scale. The Vulkan figures are dominated by command
submission and fence wait, which both paths perform.

## Preprocessing and lifecycle

`cargo run --release -p ncnn-rust --example api_benchmark -- preprocess 200 65536` reports one
median per helper on the host that runs it. The recorded campaign medians, in microseconds, for
65536 FP32 elements:

| Operation | native in place or shared | native independent storage | ncnn-rust |
|---|---:|---:|---:|
| create and copy | 6.022 | 6.022 | 7.043 |
| RGB conversion | 75.769 | 75.769 | 79.125 |
| clone and normalize | 12.958 | 26.618 | 28.366 |
| reshape | 7.221 | 15.607 | 16.892 |
| RGB conversion and resize | 107.121 | 107.121 | 103.260 |

Normalization and reshape in `ncnn-rust` return independent storage, so the matching native
reference is the independent column; those rows sit within the campaign spread of it. The gap
against the in-place column is the cost of one full-tensor copy, not wrapper overhead.

The in-repo benchmark separates the same helpers, taken on one Windows host and reported as the
minimum of three runs to limit host noise:

| Helper | median us | note |
|---|---:|---|
| borrow (`from_external`) | 0.083 | header only |
| `deep_clone` | 13.309 | allocation and one copy |
| `reshape` | 12.174 | detaches because the native result aliases |
| `into_reshape` | 0.174 | rewrites the header |
| `normalized` | 13.328 | one pass into fresh storage |
| `normalize_in_place` | 4.910 | the same pass, resident storage |
| `flatten` | 0.984 | |
| `with_packing(4)` | 1.115 | |
| `copy_make_border` | 23.797 | 1 MB output |
| `to_pixels` | 36.776 | |
| `warp_affine` | 251.441 | bilinear, full frame |

`normalized` and `normalize_in_place` apply the scale and bias of the native `Scale` path in one pass
over the destination, so the copying form costs what a bare `deep_clone` costs instead of a clone
followed by a native layer that walks the buffer again. Allocation and first touch dominate every row
that allocates, which is why a rotation of the cases and a minimum over runs is used.

GPU lifecycle: allocator create/drop 0.330 us, upload-download-drop 577.283 us, cache save
914.730 us, cache load 331.940 us. Seven randomized cold/warm process pairs report first-request
medians of 23.16 and 10.95 ms with steady medians of 5.53 and 5.52 ms.

## Reproducing

The tables come from a campaign harness that is maintained outside this repository. In-repo
commands cover the same code paths, so a change can be checked without it:

```sh
python3 tools/fetch-models.py                                   # pinned fixtures into target/model-fixtures
cargo run --release -p ncnn-rust --example api_benchmark         # per-request cost and allocation counts
cargo run --release -p ncnn-rust --example model_regression -- \
  target/model-fixtures target/cpu-reference/ncnn_cpu_reference  # native C++ versus Rust outputs
cargo test --workspace --features vulkan                         # contracts and the conformance suite
```

The [model fixtures](model-fixtures.md) page lists the reference build commands. The recorded
campaign behind the tables ran 314 processes, 146 CPU cases, 146 Vulkan-enabled cases, 3358 steady
groups and 23506 steady samples.

Fourteen cases extend the highest-risk operators with parameter variants taken from the upstream
tests: Convolution dilation, stride and activation; Pooling average, global and counted padding;
Interp nearest, bilinear and bicubic; three Gemm shapes; and Softmax on two axes. Each variant is
validated against the native reference on both devices.
