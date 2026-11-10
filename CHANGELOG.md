# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Each release pins one ncnn release, so a
library upgrade is a minor change of this crate.

## Unreleased

### Added

- `Net::layers` and `Net::blobs` describe the loaded graph: layer names, registered types, type
  indices, blob counts and the support flags ncnn declares per layer type, plus the wiring and
  recorded shape of every blob.
- `LayerInfo` also carries the shape each layer recorded for its inputs and outputs, and
  `layer_type_index` resolves a type name to the index the registry uses.
- `ParamDict::kind` reports how an entry is stored through `ParamKind`, which is what a custom layer
  needs before it picks a getter, and `version_number` returns the pinned release date as the integer
  ncnn encodes.
- Zero-copy input: `Mat::from_external` borrows caller storage, `Extractor::input_borrowed` submits it
  without a copy and `Net::infer_borrowed` bounds the borrow to one request.
  `Net::infer_borrowed_into` fuses that with the reusable output buffer, which is the request form
  that allocates no data buffers.
- Consuming tensor helpers that keep the storage a tensor already owns: `Mat::into_reshape` rewrites
  the header instead of duplicating aliased storage, and `Mat::into_normalized` normalizes in place
  instead of copying first.
- Batched tensors: `Mat::from_f32_batch` builds storage with a batch count above one, `TensorInfo::n`
  reports it and `Mat::to_f32` reads the batches back.
- Layout helpers `Mat::flatten` and `Mat::with_packing`, and the four-axis border helpers
  `Mat::copy_make_border_3d` and `Mat::copy_cut_border_3d` with the `Borders` argument.
- `DataReader::from_file` streams a file through the native stdio reader, and the file loaders
  `Net::load_param_file` and `Net::load_model_file` now read natively instead of copying the file
  into a Rust buffer first.
- `Device::pipeline_cache` and `Device::set_pipeline_cache` move the serialized Vulkan pipeline cache
  through memory, next to the existing path helpers.
- `image::affine_transform`, `image::invert_affine`, `image::warp_affine` and `image::WarpBorder`
  cover the affine preprocessing path.
- `Options::set_blob_allocator`, `Options::set_workspace_allocator` and
  `Options::set_kvcache_allocator` attach caller-owned pools to a network.
- `crates/ncnn/examples/fast_allocator.rs` measures the request path with a drop-in allocator, and the
  performance page records what it changes: on Windows the owned-output form is dominated by the
  process heap, while the forms that reuse a buffer do not move.
- `Net::option` reads the option value a loaded network actually uses.
- Tensor helpers that run on the native kernels: `Mat::fill`, `Mat::normalize_in_place`,
  `Mat::to_pixels`, `Mat::to_pixels_resize`, `Mat::copy_make_border`, `Mat::copy_cut_border` and the
  `BorderMode` argument.
- `Net::load_param_reader` and `Net::load_model_reader` load from a `DataReader`, and
  `register_custom_layer_by_typeindex` registers a factory for a type index.
- CI compares the public API with the published release, resolves every documentation link and
  anchor offline, and a `v*` tag publishes both packages in dependency order before creating the
  GitHub release from the changelog section.

### Changed

- `Mat::normalize_in_place` applies the same per-channel view as `Mat::normalized`, so both forms
  agree on tensors below three dimensions, which the native call otherwise treats as per-row
  coefficients.
- Normalization writes one pass over the destination instead of copying and then running a native
  `Scale` layer, so `Mat::normalized` costs about as much as a bare clone. The scale and bias match
  the native layer; the last bit can differ when the compiler contracts the multiply and the add.
- `Mat::flatten` returns unpacked storage, so its result reads through `Mat::to_f32` without a typed
  conversion.
- `DataReader::from_bytes` builds a bounded memory reader: a read is clamped to the bytes that
  remain instead of failing, and the same handle scans parameter text, which the reader-based
  loaders rely on.
- A borrowed input now returns `Error::Invalid` when the network runs with light mode. The native
  forward path reads the storage refcount before an in-place layer may write, and an external tensor
  has none, so the previous combination reached a native null read; owned input is unaffected.
- The fuzz target also feeds the binary parameter loader and the reader loaders.
- The coverage job writes an lcov with per-line records, prints the summary, fails below 85 percent
  line coverage, and rejects a report without `DA` records.
- The nightly fuzz workflow runs its target on the nightly toolchain, which `rust-toolchain.toml`
  otherwise shadows, and the version matrix resolves tags with `GITHUB_TOKEN`, imports the shared SDK
  into the loader path, and reports the declaration gap for tags that cannot link the adapter. A
  passing matrix run comments on the open tracking issue and closes it.
- Both CPU examples and both device examples run in CI, and rustdoc builds with `-D warnings` on
  every push instead of only during the documentation deploy.
- The README drops its validation section and gains single-layer and preprocessing tables next to
  the trained-model table; the CI description stays in `docs/support-matrix.md`.
- The `Net`, `NetBuilder`, `Extractor`, `OutputGuard`, `Mat` view, `gpu` and `CpuOption` items state
  their preconditions, error cases and layout requirements instead of one summary line.
- A probe test prints the finiteness of three attention configurations. The `ubuntu-24.04-arm`
  runner matches x86-64 for the two-dimensional operand, while a three-dimensional operand with one
  channel returns `NaN` for one sequence row there, so the layer case uses the two-dimensional form
  and every case asserts finite output on every target again.
- `Allocator::upload`, imports and command recording name the failing operation
  (`Error::NativeAt`), so a refused allocation reads as `Vulkan upload` instead of a bare status.
- `docs/contracts.md`, `docs/troubleshooting.md` and `docs/evidence-history.md` record the measured
  device-memory exhaustion behaviour, and `Mat::to_i8` states that it reads signed storage.

## 0.2.0 - 2026-11-10

### Added

- `ElementKind`, `Mat::to_f32_typed`, `Mat::to_i8`, `Mat::from_typed` and `Mat::from_i8` read and
  build packed, 16-bit and quantized storage.
- `Extractor::extract_stored`, `Extractor::extract_stored_owned` and `Net::infer_stored` keep the
  element width a layer wrote, including on Vulkan networks, whose extraction detaches its download
  copy from the pool allocator.
- `examples/typed_tensors.rs` and `examples/vulkan_resident.rs`.
- Runnable documentation examples for `Mat::from_f32`, `Net::infer` and `Extractor::extract_stored`.
- `CHANGELOG.md`, `SECURITY.md`, `docs/troubleshooting.md`, `docs/migration.md`,
  `docs/native-abi.md` and `docs/layer-coverage.md`.
- Every push compares the generated INT8 and operator fixtures against a native C++ reference, a
  coverage job reports line coverage of the wrapper, and a nightly workflow fuzzes the text
  parameter loader and the bounded weight reader.

### Changed

- Every conformance layer case runs one inference; the parameter-only tier is gone.
- `NCNN_TEST_DEVICE=vulkan` repeats the operator families, graph pairs and layer cases on the
  default device.
- Layer cases cover the key/value cache, channels inside depth copies, negative slice starts and
  requantizing convolutions.
- `Error::Invalid` carries formatted detail, so shape, name and width mismatches name the values.
- `Mat::to_f32` validates the storage it reads, and the copy-count messages name
  `Extractor::extract_stored` as the path that avoids a conversion.
- Public enums are `#[non_exhaustive]`, and the guard requirements are stated as remedies.
- Thread-affinity requirements are stated on `Net`, `Mat` and `Device`.

### Fixed

- The layer suite selected its child test by a bare name that matched no test, so no case ran.
- `LSTM` fixtures supplied half of the recurrent weight block.
- `InverseSpectrogram` fixtures supplied one channel where the transform reads one channel per
  frequency bin.
- `ROIAlign` fixtures placed the region in a channel-padded tensor that the layer reads flat.

## 0.1.1

### Added

- Vulkan contracts covering device selection, allocator accounting, resident inference, guard
  descriptors, external buffer import, pipeline caches and cross-device rejection.
- A conformance case for every layer type the SDK registers, each isolated in a child process.
- A packaged-crate job that rebuilds from the archives, and a Chinese README.

### Fixed

- The native link kind is derived from the SDK layout, and a static request against a shared SDK is
  reported instead of failing at link time.

## 0.1.0

### Added

- Initial release: `ncnn-rust-sys` raw bindings with the C ABI adapters and `ncnn-rust` safe
  wrappers for networks, extractors, tensors, options, parameter dictionaries, allocators, images,
  custom readers and the optional Vulkan backend.
