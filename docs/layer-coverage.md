# Layer coverage

Every layer type a default ncnn build registers has one conformance case. A case loads the layer,
runs one inference and asserts a finite result, so a registered type that cannot parse its
parameters or cannot run fails by name. The tables below record what the cases reach and which
native differences the fixtures work around.

## Operator conformance

`crates/ncnn/tests/conformance/` builds its parameters and weights in memory and requires no
downloaded fixtures. Expected values come from Rust reference loops, so the suite detects native or
wrapper regressions without a second backend.

- `operators.rs` covers activation, arithmetic, normalisation, softmax, structural, convolution,
  pooling, interpolation, matrix, rearrangement, one-dimensional, upsampling and reduction
  families. Each family lists every layer variant it checks, including the high-risk parameter
  combinations: convolution stride, dilation and fused activation; pooling average, global and
  strided; interpolation nearest and bilinear; Gemm shape; softmax on three axes.
- `graphs.rs` covers multi-input, split/concat, residual, pool/split/interp/concat and
  multi-output topologies, plus repeated-run determinism and packing/FP16 option parity.
- `allocation.rs` asserts that the reusable-buffer path allocates nothing once warm.
- `api.rs` reaches the public paths the families do not: file loading, custom readers, parameter
  dictionaries, unchecked reads, reshape views, the three endpoint forms and concatenated outputs.
- `invariants.rs` checks pairs that must agree with themselves: `Unfold` with `Fold`, `Split` with
  `Eltwise`, and the narrow-element conversion against known bit patterns.
- `layers.rs` holds one case per layer type the SDK registers. Cases run in child processes of the
  test binary, so a layer that aborts in native code becomes a named failure instead of a lost run.
  Every case runs one inference and asserts a finite output, which requires the shapes and operand
  values each forward pass reads: a spectrum carries one channel per frequency bin, and the
  detection heads decode regions from scores and priors.

Native ncnn enables FP16 packed, storage and arithmetic by default on ARM64 and Apple Silicon, so
results there round to FP16 before any comparison. The conformance suite pins a single-threaded
FP32 pipeline and leaves the non-default pipelines to `packing_and_fp16_options_preserve_values`.

Weights use two block forms because ncnn reads `type = 0` blocks with a leading zero tag and
`type = 1` blocks without one; `support.rs` provides both constructors. Layers that require an
external blob, such as `CopyTo`, stay out of the suite because a single-input graph does not
satisfy their contract. The campaign harness kept outside this repository remains the source for
trained-model accuracy and native-reference timings; [performance](performance.md) lists the
in-repo commands that exercise the same paths.

## Coverage

| Domain | Available interface | Validation and remaining scope |
|---|---|---|
| Version and CPU count | Safe functions | Local CPU suite |
| Text parameters and binary weights | Safe string/file/byte loaders; unsafe native mapped-path loading | Tiny graphs, bounded short reads, native mapped model inference and Unicode paths |
| Binary parameters | Safe bounded byte loader; raw file/reader APIs | Successful binary Input graph and truncated input covered |
| Multi-input/output inference | Safe name/index input, detached extraction and exclusive read-only output guards | Native C++ scalar comparison; NanoDet/OCR all-output equivalence; repeated extraction, detached lifetimes, guard pointer identity, light mode/pool/packing combinations and independent extractor aliases |
| CPU options | Safe typed option keys plus raw C access | All 18 fields compared with C++; 17 trained-model profiles plus 25 generated INT8/operator/shape/reload cases cover observable precision, guarded conversion, packing, threads, algorithms, pooling, light mode and native mapped loading |
| Tensor creation | Safe FP32 dimensions 1 to 4; raw batch/external/typed creation | Shape rejection and padded 4D copying covered |
| Tensor transforms | Safe deep copy, reshape and normalization | Shared-input independence and logical element copying covered |
| FP16/BF16/FP8/INT8 storage | Raw typed tensors, scalar and tensor casts, quantization helpers | FP16/BF16 roundtrip, E4M3/E5M2 scalar encodings, BF16 model storage, INT8 inference and numeric quantize/dequantize/requantize covered |
| Images | Safe pixel import, resize and orientation; 70 raw public helper overloads | Pixel import, rotation, resize, matrix, YUV, strided guard bytes, affine and drawing cases |
| ParamDict | Safe scalar/array setters and byte strings | Scalar, byte-string roundtrip and bounds covered |
| DataReader | Copying binary reader and owned bounded native memory reader | Exact/short reads, bounded text scan, reference bounds covered |
| ModelBin | Owned array adapter and raw reader/callback loaders | Weighted layer survives ModelBin destruction; dimensions 1 to 4, cursor exhaustion and retained outputs covered |
| CPU allocator | Pool RAII, tuning and clearing; raw unlocked/external hooks | Outstanding allocation survives clear; allocation/free/clear covered |
| Standalone layer | Owned layer and pipeline, unsafe loading/forward | ReLU and weighted InnerProduct; multi-tensor/in-place forms available in sys |
| Custom layers | Checked registration bridge, INT8/batch flags, raw callbacks | Failures, partial outputs, contained panic, pipeline cleanup, failed graph loading, concurrent networks and destruction counts |
| Border and packing | Raw C API | Border dimensions, packing roundtrip and flattened output conversion covered |
| Public CPU C++ extras | Runtime controls, bounded CpuSet, OS processor groups, dynamic affinity, raw extraction, views and graph metadata | Actual pin/query/restore, native mask boundaries, physical-core counts, denormal arithmetic/native no-op behavior, views and extractor lifetime covered |
| External native installations | Required SDK prefix, static/shared selection, explicit extra libraries | Linux and Windows static/shared installations pass CPU calls and independent native model comparisons |

## Known native differences

- `PriorBox_vulkan::upload_model` uploads `min_sizes` and `aspect_ratios` without an emptiness guard while it guards `max_sizes`, so a network whose `PriorBox` omits `aspect_ratios` aborts with an integer divide by zero during weight upload. The conformance fixture supplies both arrays, and the defect is reported upstream as Tencent/ncnn#7259.
- `MultiHeadAttention` on Linux aarch64 takes ncnn's ARM path, which builds its own Gemm and Softmax layers; this synthetic case returns one non-finite output row there while the x86-64, Windows and macOS runs stay finite. The layer case therefore asserts finiteness everywhere except that target.
