# Migration notes

This crate pins one ncnn release. Moving to a different ncnn release, or moving between crate
versions, changes what compiles and what the wrapper promises.

## Moving to a new ncnn release

1. Point `NCNN_INSTALL_DIR` at the new SDK and set `NCNN_ALLOW_UNTESTED_VERSION=1` for the first
   build.
2. Regenerate the bindings: they come from `crates/ncnn-sys/native/adapter.h`, so a changed C API
   signature appears as a compile error rather than as a runtime surprise.
3. Run the workspace suite on the host and on a device. Layer parameters change between releases;
   the layer cases name the layer that broke.
4. Compare a trained model per `docs/model-fixtures.md` when the release touches kernels.
5. Drop `NCNN_ALLOW_UNTESTED_VERSION` and record the new pinned version in `crates/ncnn-sys/build.rs`
   and `ci/ncnn-versions.json` once the suite passes.

Behaviour that has changed between ncnn releases without a C API change:

| Area | What to check |
|---|---|
| Layer parameter keys | `-<23300 + id>` array forms and integer-array types |
| Weight block encoding | `type = 0` tagged blocks versus raw blocks |
| Vulkan capability flags | A layer can advertise more than it implements; the repository patches such cases in `tools/patches/` |
| Attention layouts | `MultiHeadAttention` gained key/value caches and a packed weight form |

## Moving between crate versions

- **0.1.x to 0.2.0** — public enums are `#[non_exhaustive]`, so `match` statements need a wildcard
  arm, and `Error::Invalid` carries `Cow<'static, str>` instead of `&'static str`. `Mat::to_f32`
  keeps its behaviour; narrow storage is read through the new typed readers.
- **0.2.x to 0.3.0** — reusing one extractor now restarts the request. The native extractor caches
  the blobs a request produced, so a second `input` on the same handle used to answer from that
  cache; `input` and `input_borrowed` now restart first, and a caller that relied on the cache calls
  `Extractor::continue_session` before the next input. Enabling light mode on a graph that reads one
  blob from two layers now returns `Error::Invalid` through `Extractor::set_light_mode` and
  `Net::extractor`, where it used to return the values of the first consumer; insert an explicit
  `Split` layer to keep such a graph on light mode. Everything else in this release adds API:
  `Mat` sub views (`channel`, `channel_range`, `depth`, `depth_range`, `column_range`, `row_range`,
  `batch`, `batch_range`) returning `MatView`, `MatView::as_mat`, `Mat::is_empty`,
  `TensorInfo::total`, `TensorInfo::shape`, `Net::shared_blobs`, `Net::infer_view_into`, the
  per-extractor settings (`set_light_mode`, the three allocators and the
  cache length hint), `Extractor::clear` and `Net::layer_handle`. No signature changed except that
  `Extractor::set_light_mode` now returns `Result`, so an existing caller compiles unchanged.
- **0.3.x to 0.4.0** - the request forms collapse onto one storage decision per side.
  `Extractor::extract(endpoint)` and `Extractor::extract_stored(endpoint)` become
  `extract(endpoint, TensorWidth::Fp32)` and `extract(endpoint, TensorWidth::Stored)`, and
  `extract_mat` and `extract_stored_owned` become `extract_mat(endpoint, TensorWidth::Fp32)` and
  `extract_mat(endpoint, TensorWidth::Stored)`. `extract_into` appends instead of replacing, and
  `extract_append` is gone: a caller that wants the output alone clears the vector first, which keeps
  its capacity and its steady state allocation-free. On `Net`, `infer` and `infer_into` take an
  `Input`, so `infer_borrowed(&[("input", values, &[4])], outputs)` becomes
  `infer([("input", Input::borrowed(values, &[4]))], outputs)` and `infer_view_into(inputs, outputs,
  buffer)` becomes `infer_into(inputs, outputs, buffer)` with `Input::view`; `infer_borrowed_into`
  becomes `infer_into` the same way. The extractor request surface drops from eight methods to four
  and the network from six to three, and no capability is lost.
  `Extractor::inputs` and `Extractor::outputs_with` are gone as well: submit each input with
  `input` and extract each output with `extract`, which keeps the error policy in the caller's loop.
  `OutputGuard::detach` is gone; `into_mat` consumes the guard and copies only when it cannot own
  the storage. The loading methods take a source: `load_param_file(path)`, `load_param_binary(data)`
  and `load_param_reader(&reader)` become `load_param(ParamSource::{File, Binary, Reader})`, and
  `load_model_file(path)` and `load_model_reader(&reader)` become
  `load_model(WeightSource::{File, Reader})`. `From` conversions keep `load_param("text")` and
  `load_model(&bytes)` compiling as before, and `Net::load(params, weights)` loads both stages at
  once. `Mat::append_f32_to` is gone: `Mat::to_f32_into` appends now, and a caller that wants one
  tensor alone clears the vector first. `ncnn_rust::sys` needs the `raw` feature; the default build
  publishes no generated bindings.
- **Patch releases** never change the pinned ncnn release, the public API or the storage contracts.

## Compatibility with the native library

`ncnn-rust-sys` records the pinned release at build time and refuses a different one unless
`NCNN_ALLOW_UNTESTED_VERSION` is set. A mismatched shared library that satisfies the version check but
differs in structure layout is outside what the build can detect; keep one SDK per target.
