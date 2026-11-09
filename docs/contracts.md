# Safety contracts

Each section states one contract area: what the safe API guarantees, which native behaviour it
relies on, and what remains unsafe. Dated counts, machine details and log paths live in the
[evidence history](evidence-history.md). Native adapter decisions and the ABI the wrappers rely on
are in [native ABI and adapter decisions](native-abi.md); the layer list and known native
differences are in [layer coverage](layer-coverage.md).

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

ncnn pads every channel plane to a 64-byte boundary, so a tensor with fewer than sixteen FP32
elements per plane carries padding between planes. `channel_f32` and the copying readers skip that
padding, while layers that read a whole tensor through one pointer take the values from the first
plane only. A region of interest therefore belongs in a one-dimensional tensor or in the first plane
of a two-dimensional one, and a five-channel tensor of single-element planes reaches the layer as
padding.

Extraction converts packed, FP16, BF16 and INT8 storage to FP32 unless the caller asks for stored
storage. `Extractor::extract_stored` keeps the layout the producing layer wrote, so `TensorInfo`
reports the stored element width and the typed readers apply, while `extract` and the owned forms
return converted FP32 values. The guarded borrow needs safe configuration and built-in layers; it
also serves Vulkan networks, whose extraction downloads into network-owned CPU storage and detaches
that copy from the pool allocator.

Raw adoption requires initialized logical elements and live allocator dependencies. Raw-pointer users preserve native aliasing rules. A borrowed raw handle transfers no ownership. `into_raw` transfers release responsibility to the caller. All FFI calls using these handles require unsafe code.

## Advanced resources

`Layer` owns its pipeline options and attempts pipeline destruction before destroying its handle. Parameter loading, weight loading, and execution are unsafe because each layer has its own tensor and parameter contract. ModelBin and reader handles retain their owned backing storage. A ModelBin cursor is consumed by native loading; callers provide the number and formats of weights expected by the layer.

`PoolAllocator` provides RAII ownership for advanced FFI use. Objects allocated through its raw handle are released before the allocator. Custom-layer creators, destroyers, callbacks, and userdata remain available through `sys`; contexts remain alive until all native users are destroyed. Callback implementations contain Rust panics and satisfy upstream output ownership rules.

The wrapper types remain thread-affine. Native runtime queries are separate from object sharing. CPU worker-affinity changes and raw callback synchronization remain application-managed advanced operations. OS processor identities are value types. Current-thread affinity acts on the caller only, with a group-local mask on Windows; native worker masks retain their upstream capacity and scheduling semantics.

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

## Optional Vulkan coverage

The feature-gated adapter covers initialization, device enumeration and metadata, validity, explicit
pre-load network selection, and synchronous CPU-Mat inference. Safe resident tensor, allocator and
recording-command ownership, completed tensor guards, pipeline-cache persistence and unsafe
same-device buffer imports are implemented. Cache tests reject truncation, header corruption and
payload hash corruption. FP32 extractor guards remain CPU-only; stored extraction also serves device
networks.

`NCNN_TEST_DEVICE=vulkan` attaches the default device to every conformance fixture, so the operator
families, the graph pairs and all 108 layer cases repeat on the GPU. One contract compares
`Convolution`, `ConvolutionDepthWise` and `InnerProduct` with int8 weights, whose scale pushes `0.25`
below one quantization step, against the float run on the host and against the device and
resident-recording results. It also reads the quantized bytes back through stored extraction, both
after an explicit `Quantize` layer and from requantizing convolutions, whose `8=101` scale term makes
the convolution itself write one-byte output.

### Device memory exhaustion

A device that refuses resident storage returns `Error::NativeAt("Vulkan upload", -1)`: ncnn
translates `VK_ERROR_OUT_OF_DEVICE_MEMORY` into its own failure and writes `vkAllocateMemory failed`
to stderr, so the code names the operation rather than the allocator result. The refusal leaves the
allocator, the tensors it already returned and later requests usable: the measured run refused the
23rd 256 MiB upload against a 5226 MiB heap budget, then downloaded the retained tensors, accepted a
16-byte upload and completed a resident inference. `Allocator::tensor_bytes` reports live storage
dropping to zero when the tensors are released while its second value keeps the high-water mark.
Empty tensors are rejected before any native call.

Device loss, driver reset and out-of-host-memory on a shared allocation are outside what the pinned
API reports distinctly; they surface as the same native failure, and a device that no longer answers
leaves the tensors unusable. Applications that run long resident sessions re-create the device and
its allocator after a failure they cannot attribute.

### Capability limits

Native per-layer CPU fallback stays available, so device selection is not proof of GPU placement for
every layer. True nonblocking submission and caller-owned `VkDevice` adoption are unavailable in the
pinned native API. Labeled task metrics and macOS/ARM64 Vulkan execution remain unverified. The
[evidence history](evidence-history.md#vulkan-coverage) records the executed contracts, and the
[trained-model evidence](model-fixtures.md#windows-vulkan-trained-model-evidence) records commands
and logs.

## Related pages

- [native-abi.md](native-abi.md) records the adapter ABI, the upstream decisions and the runtime validation scope.
- [layer-coverage.md](layer-coverage.md) records the conformance cases, the coverage table and the known native differences.
- [troubleshooting.md](troubleshooting.md) maps symptoms to fixes.
