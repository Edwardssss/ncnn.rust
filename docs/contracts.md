# Safety contracts

Each section states one contract area: what the safe API guarantees, which native behaviour it
relies on, and what remains unsafe. Dated counts, machine details and log paths live in the
[evidence history](evidence-history.md); the coverage table at the end lists what the test suite
exercises.

## Networks and extraction

An extractor borrows its network. The default options retain intermediate blobs (`lightmode = false`) for multi-output graphs. Applications may enable light mode for graphs with explicit Split nodes and matching consumption semantics. Configuration and loading require mutable access before extraction. Owned input methods transfer storage created or independently cloned by the crate; tensors adopted through raw handles are copied because their storage/allocator dependencies remain caller-managed. `Net::infer` returns independent final Rust vectors, and `infer_into` returns metadata and ranges into the caller's vector. Final-vector extraction copies validated logical FP32 elements while the native source is live and excludes native padding. Output handles are released on success and failure. Input setup failure preserves the old `infer_into` buffer; output failure leaves only completed output prefixes, with no ranges returned. Publication occurs after success when transactional results are required.

`extract` returns an `OutputGuard` with an exclusive extractor borrow. The guard owns and releases the returned native handle; its private tensor cannot escape as a `Mat` or raw handle. Slices borrow the guard. The extractor and network therefore remain alive, and input replacement, re-extraction and destruction remain unavailable until the guard is dropped. Forgetting a guard can leak its handle but cannot extend a slice borrow. The guard exposes metadata, checked read-only FP32 views, explicit vector copying and `detach()`, which deep-copies into an independent `Mat`. It never transfers the potentially shared native storage. Model semantics determine whether FP32 interpretation is meaningful; the checks establish storage layout rather than semantic dtype.

This path supports built-in CPU layers with safe configuration. Custom-layer registration, native path model loading (including optional mapping), or exporting a network/extractor raw handle permanently disables guard extraction for that object; exporting an options raw handle disables guard extraction in networks subsequently configured from those options. These conservative eligibility flags prevent custom allocators, callbacks, external output storage or raw extractor input from being mistaken for crate-managed dependencies. Final-vector extraction remains available. Raw-adopted inputs retain the provenance-protection copy in `input`. Native raw writers must also preserve all existing tensor and guard borrows.

The pinned native ownership audit covers these alias paths:

- `src/c_api.cpp` creates a heap `Mat` by shallow-copying the extracted native `Mat`, even on extraction failure. Success transfers that handle to the guard; errors destroy it. `src/mat.h` increments the storage refcount on copy and frees through the recorded allocator only on final release.
- `src/net.cpp` stores input by shallow assignment. Only initialized crate-created or deep-cloned storage is transferred without copying. External/raw-adopted storage is detached before submission, including caller-owned allocators.
- Light mode releases cached intermediates and checks shared refcounts before in-place forwarding. Guard creation requires the network's only live extractor. A network-wide reservation blocks safe input, extraction and new extractors until guard destruction; this also covers model-owned aliases and allocator state beyond the current cache. Existing peers cause guard creation to fail, and dropping peers permits recovery. Forgetting an extractor conservatively prevents subsequent guards; forgetting a guard leaks its handle and permanently reserves the network. Neither case permits native reuse through safe inference APIs.
- The guard's lifetime comes from `&mut Extractor`, and that extractor holds `&Net`; its private owned `Mat` releases the handle before either resource can die. The phantom borrow records this reservation without storing a second reference. An explicit guard destructor makes Rust drop checking retain that borrow through implicit handle destruction, even when the guard is otherwise unused. Safe methods expose neither the tensor nor its raw handle, and all slice lifetimes come from the guard. Advanced raw writers must preserve these borrows even if a raw input pointer was obtained before `input`.

`inputs` applies the same storage-provenance check as `input` to each input in order. `outputs_with` scopes one guard per callback and releases it before the next extraction. A callback can retain copied values while its borrowed slices remain scoped to that callback. Errors stop iteration; earlier input transfers and callback effects remain applied. Final-vector output methods retain independent Rust storage; explicit guard detachment retains independent native storage.

Low-copy describes omission of the wrapper's final deep clone. A null native allocator or different data pointer does not prove output independence. Native extraction shallow-assigns cached tensors, including input-blob aliases; consuming a Rust input does not remove those native references. Final vectors copy logical values instead of certifying output independence from allocator identity. Independent native output storage requires explicit `detach` or `deep_clone`. The installed 20261020 `Extractor::extract` still performs type-0 unpacking, supported precision conversions and a default-allocator clone when the result uses the network's local blob pool. Packing, light mode and local-pool defaults are unchanged. Guard access does not promise pointer identity for computed outputs or bypass these native operations. Input-blob extraction can retain the transferred input pointer; padded channels still require channel views.

Model files are trusted native input. The bounded binary reader limits reads to the supplied bytes and copies weights through `DataReader::read`. Text parameters use a terminated owned string. Safe file APIs use Rust paths and file reads on both platforms. The explicit unsafe `load_model_file_native` method calls the native narrow/wide path API and honors mapped loading; the caller keeps the file contents and length stable until the network is cleared or destroyed. Native allocation failure may terminate the process in upstream allocation paths.

## Tensors

Safe tensors expose immutable data. `as_f32_slice` borrows only aligned, contiguous, unpacked 32-bit storage; `channel_f32` skips inter-channel padding. The borrow ends before the tensor can be dropped or consumed. Raw/native writers preserve these Rust borrows. Reshape and normalization operate on independent copies. Logical channel contents are copied separately from channel padding. Dimensions, element counts, and integer conversions are checked before allocation. `to_f32` explicitly selects FP32 interpretation and checks 32-bit scalar storage, pack 1, and batch 1; native metadata alone carries no complete scalar type tag.

`to_f32_typed` accepts an `ElementKind` and converts packed or narrow storage: signed and unsigned
8-bit, signed 16-bit, signed 32-bit, IEEE half and bfloat16. ncnn records the element width but not
whether a 16-bit element is IEEE half or bfloat16, so the caller names the format. `to_i8` returns
the signed 8-bit values of a quantized tensor. `OutputGuard` exposes both methods, which makes a
quantized blob readable without an FP32 copy on the Rust side.

Raw adoption requires initialized logical elements and live allocator dependencies. Raw-pointer users preserve native aliasing rules. A borrowed raw handle transfers no ownership. `into_raw` transfers release responsibility to the caller. All FFI calls using these handles require unsafe code.

## Advanced resources

`Layer` owns its pipeline options and attempts pipeline destruction before destroying its handle. Parameter loading, weight loading, and execution are unsafe because each layer has its own tensor and parameter contract. ModelBin and reader handles retain their owned backing storage. A ModelBin cursor is consumed by native loading; callers provide the number and formats of weights expected by the layer.

`PoolAllocator` provides RAII ownership for advanced FFI use. Objects allocated through its raw handle are released before the allocator. Custom-layer creators, destroyers, callbacks, and userdata remain available through `sys`; contexts remain alive until all native users are destroyed. Callback implementations contain Rust panics and satisfy upstream output ownership rules.

The wrapper types remain thread-affine. Native runtime queries are separate from object sharing. CPU worker-affinity changes and raw callback synchronization remain application-managed advanced operations. OS processor identities are value types. Current-thread affinity acts on the caller only, with a group-local mask on Windows; native worker masks retain their upstream capacity and scheduling semantics.

## Adapter ABI

Project adapter names use `ncnn_rs_`. The adapter compiles against the selected installed ncnn headers and links to the matching static or shared native library. Applications supply compatible compiler ABI, CRT, and native runtime dependencies. Adapter functions that allocate C++ objects translate exceptions to status codes or null handles. Upstream C API entry points retain upstream allocation and termination behavior.

## Vulkan ownership

Optional Vulkan devices share a process-owned native runtime and cached device objects. Initialization is serialized and exit cleanup precedes driver teardown; no safe manual shutdown is exposed. Networks retain their device handles across configuration and clear/reload, and reject selection after parameter loading. Device, network and extractor handles remain single-thread-affine. Raw sys users must not destroy or replace the runtime, devices or Vulkan allocators while wrapper objects or work remain alive. Ordinary inputs use CPU Mat storage and ordinary extraction synchronizes and detaches outputs. CPU extractor guarded views reject GPU networks because their low-copy proof is CPU-only. Completed resident tensor guards use the separate contract below. Native unsupported layers may execute on CPU; Vulkan selection is not an all-GPU guarantee.


## Resident Vulkan resources

`Device::allocator` creates a thread-affine arena with blob and staging allocators. `Allocator::upload` accepts unpacked FP32 Mat storage and completes its transfer before returning a `Tensor`. Tensors retain their allocator; allocators retain the device and process runtime. `Allocator::command(&net)` borrows the configured network. `Command::input` borrows resident inputs, and `extract` records detached FP32 outputs, returning indices. Consuming `wait` submits once and blocks until native fence completion before exposing the output vector. Tensor downloads synchronize and return independent CPU Mats. `Tensor::guard` only borrows completed storage; no mapped pointer is exposed.

The native recording owner is destroyed before retained intermediate tensors. Native input or extraction recording failures poison the command: later recording and `wait` fail, and consuming or dropping the owner cancels the partial recording without submission. Input borrows are retained even when native recording fails. Host-only validation errors do not poison otherwise valid recordings. Dropping an unsubmitted command cancels recording and releases its resources. Forgetting a command intentionally leaks its native recording and inaccessible outputs; it does not submit work or expose unfinished storage. Network mutation and input destruction are prevented while the recording owner remains usable. The existing ten compile-fail tests cover CPU ownership/view contracts; the GPU command compile-fail test rejects network mutation while a recording remains usable. The consolidated GPU runtime contract covers recorded-work cancellation, allocator retention, completed guards, imports and cache persistence.

The pinned `VkCompute::submit_and_wait` includes deferred host transfers/conversions after fence completion. There is no supported nonblocking submit/poll adapter: raw queue submission alone would skip required completion work. `Command` is therefore an explicit recording/blocking-completion API, not an asynchronous executor. Native allocation and conversion failures return errors; adapter exceptions do not cross the C ABI.

`Device::raw_device_handle` borrows ncnn's VkDevice without transferring ownership. Unsafe `Allocator::import_buffer` imports one-dimensional unpacked FP32 storage allocated on exactly this device, with valid storage/transfer usage, compatible memory and queue ownership, padded capacity, aligned offset and correct access/stage state. The caller keeps storage alive through every tensor and pending command, synchronizes external queues, and prevents external mutation during Rust use. `TensorGuard::buffer_descriptor` is unsafe for the same interoperability reasons. A copied descriptor carries no ownership or lifetime enforcement. Import does not adopt or free the buffer. Arbitrary caller-owned VkDevice adoption is unavailable because the pinned VulkanDevice has no adoption constructor.

Cache persistence uses native device/driver compatibility metadata, shader-source identities, lengths and payload hashes. Truncated files, incompatible headers and payload corruption are rejected. The consolidated contract loads a valid saved cache and then rejects truncations, a header mutation and a payload mutation. Cache files are disposable acceleration data, not trusted executable artifacts or authenticated input; applications use trusted local files. Saving uses an ordinary non-atomic file write, so interrupted files can require regeneration.

## Native allocation boundary

Clone-preserving normalization checks wrapper-visible allocation failures and retains source independence. The native void normalization implementation performs internal allocations without exposing a complete recoverable failure status. These checks do not catch every native OOM. GPU download waits for completion and returns independent reference-counted host storage; removing a redundant post-download clone does not weaken default extraction independence, imported-buffer copying or cancellation synchronization. See the [Phase B evidence](performance.md).

## Raw escape hatches

`as_raw` on `Net`, `Extractor` and `Options` is unsafe. Each call permanently disables guard
extraction for that object, since a raw handle allows native writers to bypass the borrows the
guards rely on. The returned handle borrows the owning object and stays valid only while it is
alive and unmodified. `Mat::into_raw` transfers release responsibility, and
`Device::raw_device_handle`, `Allocator::import_buffer` and `TensorGuard::buffer_descriptor`
expose native Vulkan objects with the same rule: the caller keeps the underlying storage alive,
synchronized and unmodified for the whole Rust lifetime of every object derived from it.

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

## Upstream adapter decisions

- Extractor option assignment in this release is a no-op. High-level configuration applies to the network before loading.
- C extraction creates an output handle on failure. Rust releases it before returning an error.
- The C Mat-array ModelBin constructor passes a temporary vector's storage to a retaining reader. The project adapter owns its vector.
- The built-in layer C load-model callback treats a C handle as a C++ ModelBin object. The project adapter unwraps `pthis` explicitly.
- The native custom-layer bridge dereferences every output on callback failure. Project registration supplies owned empty handles for missing outputs, preserves failure codes, and rejects empty successful outputs.
- The pinned Linux topology implementation encodes sibling IDs in an integer, which truncates masks wider than 32 bits. `tools/patches/ncnn-20261020-cpu-siblings.patch` retains canonical kernel sibling strings for physical-core and cache accounting without changing the native ABI, and `tools/build-ncnn-sdk.py` applies it.
- The pinned Vulkan `Cast` layer advertises every conversion although only FP32 and FP16 shaders exist. `tools/patches/ncnn-20261020-vulkan-cast-capability.patch` limits Vulkan support to those pairs, so other conversions fall back to CPU instead of running an absent shader.

## Runtime validation and topology scope

Runtime coverage spans four link and CRT combinations with the same tests and models: Linux shared
with simpleomp, Linux static with GCC OpenMP, Windows shared with the dynamic CRT, and Windows
static with the static CRT. Each installation compares every trained-model output name, layout and
value against an independent C++ reference, and compares the 18 option fields separately.
[Model fixtures](model-fixtures.md) lists the cases; the [evidence history](evidence-history.md#runtime-and-topology-campaign)
records counts, paths and memory results.

The compact suite keeps one integration entry point plus the compile-fail ownership and view checks:
extractor reuse, escaping the extractor, escaping a guard slice, owned-handle escape, dropping an
extractor before an unused guard, and escaping a sequential callback slice. It also covers callback
failures with partial outputs, contained panics, partial pipeline destruction, concurrent
registrations, reader and model cursor state, and detached output lifetime.

OS topology interfaces preserve Linux kernel CPU IDs and Windows group/index pairs. Linux affinity
storage grows dynamically. Native `CpuSet` keeps the upstream fixed-mask and worker semantics, while
the separate OS API controls the calling thread and uses one Windows group per call. Multi-group
Windows execution, NUMA behavior, large-machine throughput and additional processor topologies
remain hardware characterization work; [advanced CPU adapters](advanced-cpu.md) records these
contracts. Application-level accuracy evaluation uses downstream datasets and preprocessing.

## Optional Vulkan coverage

The feature-gated adapter covers initialization, device enumeration and metadata, validity, explicit
pre-load network selection, and synchronous CPU-Mat inference. Safe resident tensor, allocator and
recording-command ownership, completed tensor guards, pipeline-cache persistence and unsafe
same-device buffer imports are implemented. Cache tests reject truncation, header corruption and
payload hash corruption. CPU extractor guards remain CPU-only.

Native per-layer CPU fallback stays available, so device selection is not proof of GPU placement for
every layer. True nonblocking submission and caller-owned `VkDevice` adoption are unavailable in the
pinned native API. Labeled task metrics and macOS/ARM64 Vulkan execution remain unverified. The
[evidence history](evidence-history.md#vulkan-coverage) records the executed contracts, and the
[trained-model evidence](model-fixtures.md#windows-vulkan-trained-model-evidence) records commands
and logs.
