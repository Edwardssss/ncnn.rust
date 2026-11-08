# Advanced CPU adapters

All functions below are exported through `ncnn_rust::sys`. Raw calls use the selected native library instance. Input handles remain live for the call; borrowed storage, allocator dependencies, readers, and callback contexts remain live for every subsequent native use. Mutable operations require exclusive access to affected native state. Image buffers contain initialized bytes, satisfy native dimensions and strides, and are disjoint unless the native operation supports aliasing. Native C++ helpers returning `void` retain their upstream failure semantics.

## Runtime

`ncnn_rs_cpu_query` keys 0–7 select logical, little, big, physical, physical little, physical big, L2, and L3 counts/sizes. Keys 22–35 expose ARM EDSP, NEON, VFPv4, FP16, CPUID, dot product, FP16 FMA, BF16, I8MM, SVE, SVE2, SVE BF16, SVE I8MM and SVE F32MM queries. Existing numeric keys remain stable. Keys 8–21 select AVX, FMA, FMA4, XOP, F16C, AVX2, AVX VNNI, AVX VNNI INT8, AVX VNNI INT16, AVX NE CONVERT, AVX512, AVX512 VNNI, AVX512 BF16, and AVX512 FP16.

`ncnn_rs_runtime_get/set` keys 0, 1, 2, 3, 4, 5 select powersave, OpenMP team size/requested thread count, OpenMP dynamic, OpenMP thread index (read-only), KMP blocktime, and denormal mode. The team-size query normally returns one outside a parallel region. KMP blocktime follows the selected native runtime. Denormal controls follow the native compile-time SSE3 branch: the suite checks mode readback and subnormal arithmetic for both an active implementation and the native no-op implementation. Serialize native runtime configuration with inference and other runtime users.

Native affinity masks are owned opaque objects released by `ncnn_rs_cpuset_destroy`. `CpuSet::capacity` reports the native bound, and `for_powersave` copies a native mask. Windows retains the upstream group-local `ULONG_PTR` mask; Linux retains `CPU_SETSIZE`. Native worker affinity follows upstream OpenMP semantics, including partial changes when a worker assignment fails.

`supports_thread_affinity` reports exact-affinity availability. macOS uses `sysctl` for physical-core counts and returns `Error::Unsupported` for exact masks/processor IDs; native raw setters use `NCNN_RS_UNSUPPORTED`. Darwin policy tags do not satisfy the exact-mask contract. The separate OS interface represents processors as `ProcessorId { group, index }`. Linux uses group zero and kernel CPU IDs, reads online IDs from sysfs, and grows the calling thread's affinity buffer dynamically. Windows enumerates `RelationGroup` active masks, preserving sparse processor numbers across groups, and counts physical cores with `RelationProcessorCore`. `current_thread_affinity` reads the caller's mask; on Windows this is its current group affinity. `set_current_thread_affinity` validates online IDs, applies one Windows group per call, verifies readback and attempts restoration on a mismatch. Applications assign their own threads to groups explicitly; this operation does not redistribute the native OpenMP worker pool.

Local execution covers a 12-logical-CPU Linux host and the Windows VM's available group. Synthetic OS responses exercise Linux CPU ID 1536; the real patched sibling parser preserves IDs 32, 64 and 1024. The Windows suite executes inference in every group exposed by the guest. The current Windows 10 Home VM still reports one group after applying Microsoft's [processor-group test settings](https://learn.microsoft.com/en-us/windows-hardware/drivers/devtest/boot-parameters-to-test-drivers-for-multiple-processor-group-support) (`groupsize=2` and `groupaware=on`) and performing a cold boot. These settings were removed after testing. Nonzero-group runtime verification requires an environment that exposes another group.

## Pools and parameters

Pool configuration and clearing require a matching pool type: `unlocked=0` is PoolAllocator; `unlocked=1` is UnlockedPoolAllocator. Ratio lies in 0–1. Clearing reclaims cached blocks while live allocations retain their ownership. The pool outlives every allocation. Parameter byte strings preserve embedded nulls. String reads first report size, then copy into a sufficient buffer; `SIZE_MAX` reports failure. Missing strings produce empty bytes.

## Extractors and layers

`ncnn_rs_extract` supports normal conversion (`type=0`) and retained native layout (`type=1`). Successful outputs own a Mat header and share native storage. Failed calls leave the output unchanged. Per-extractor allocator kinds 0–2 select blob, workspace, and KV cache. Allocators outlive escaped outputs. `clear` releases extractor state; create a new extractor for subsequent inference. Cloning shares native tensor references and the borrowed network.

Layer flag keys 0–2 select INT8 storage, batch support, and feature mask. Configure before pipeline creation. Explicit backend factory 0 selects naive, 1 selects CPU; returned handles use `ncnn_layer_destroy` after pipeline teardown. Graph layer handles borrow the network's layer and are released using **only** `ncnn_rs_borrowed_layer_release`. Layer shape queries write six integers: dimensions, width, height, depth, channels, batch. Graph metadata uses copied scalar fields. Raw userdata mutation preserves registration machinery: registered custom layers reserve userdata for their C handle.

Graph mutation adapters operate on existing vector slots, with an explicit resize entry point. Resizing may partially change state on allocation failure; callers restore consistency before inference. Shrinking requires prior cleanup of removed layers. The caller updates all edge indexes, shapes, layer ownership, and cached input/output invariants consistently before inference. Replacing a layer leaves cleanup of the displaced layer to the caller and transfers the replacement C++ object's destruction to the network; release only its C header afterward. Custom-layer destruction and pipeline teardown remain caller obligations for displaced entries. Graph mutation is an unsafe advanced operation.

Registration preserves native status codes and retains its factory record in the network. Names and user context outlive registration; Rust `Net` retains its owned name strings. The project bridge snapshots callbacks in a synchronized registry and releases the registry lock before invoking user code. Missing failure outputs become owned empty Mat handles for the upstream C dispatcher, and missing successful outputs become errors. The dispatcher consumes partial outputs exactly once. Destruction removes the registry entry before invoking user cleanup. Callbacks contain Rust panics and implement the layer's tensor contract. Native network loading normalizes a pipeline-creation failure to `-1`; standalone `Layer` execution preserves the callback's status.

## Tensor operations

`ncnn_rs_tensor_transform` operations 0–9 select nearest resize, bilinear resize, bicubic resize, FP32→FP16, FP16→FP32, INT8→FP32, FP32→BF16, BF16→FP32, INT8 quantization, and INT32 dequantization. Input scalar types match the operation. Resize dimensions are positive and representable. Scale and bias tensors follow upstream per-tensor/per-channel cardinalities; optional bias is represented by an empty native Mat handle. Requantization accepts native activation IDs and matching activation parameter tensors. Output handles are created on success and released with `ncnn_mat_destroy`. A configured allocator remains alive through output destruction.

Native aligned allocation uses paired `ncnn_rs_fast_malloc/free` calls. Sizes satisfy native allocation-overhead arithmetic. Allocator-aware clone and reshape retain the configured allocator dependency; reshape may share original storage when its layout permits. In-place clone/create convenience overloads are represented by creating a replacement owned handle and releasing the previous handle after dependent views are finished.

Shape-only Mat handles contain metadata without initialized data storage and are released through `ncnn_mat_destroy`; use them only for shape inspection or native shape contracts. `ncnn_rs_mat_info` returns emptiness and reports the native total and element bit width, which include native packing/layout semantics rather than a flat logical FP32 length.

Raw external batch constructors borrow caller memory with native alignment, channel padding, and batch stride. The caller supplies sufficient initialized storage and retains it through all dependent views. Shallow copies share storage and allocator dependencies. Scalar fill requires writable exclusive storage and a compatible 32-bit scalar layout; integer fill additionally requires a finite representable integer value. SIMD overloads are represented by equivalent scalar or explicit buffer initialization rather than architecture-specific vector ABI types.

View axes 0–4 select linear range, rows, depth, channels, and batches. View headers borrow source storage; retain the source and its allocator until all views are released. Depth views follow the upstream single-channel volume semantics. View calls validate ranges, and raw consumers preserve packing and stride semantics. ModelBin 4D loading consumes its reader cursor and returns an owned Mat header with the reader's storage dependencies.

Scalar FP16/BF16/FP8 adapters preserve upstream bit conversion behavior.

ModelBin callbacks receive dimensions 1–4 and return an owned Mat handle or null. The adapter consumes that handle and retains its storage reference. Callback context outlives the ModelBin and all invocations; callbacks contain panics and provide initialized tensors matching the requested format.

## Readers

`ncnn_rs_reader_memory` owns its byte storage and cursor, supports bounded scanning, copying reads, and references. Release it with `ncnn_datareader_destroy` after all referenced weights are released. Reference results borrow that storage; callers verify required alignment before zero-copy model loading. The existing high-level DataReader uses copying binary reads and avoids escaped storage references.

## Images

`ncnn_rs_image_*` maps each enabled public image helper to its upstream name. `_strided` selects the byte-stride overload. Non-strided functions use tightly packed rows. YUV420sp buffers use even dimensions and contain width × height × 3/2 bytes. RGB conversion destinations contain three bytes per output pixel. Rotation orientations 5–8 swap output dimensions. Affine matrices contain six floats; affine point sets contain two floats per point with at least two points. Warp destinations are initialized, including transparent-border destinations. Drawing text is null-terminated; font size and geometric dimensions follow native positive-range requirements. These raw adapters preserve native drawing and interpolation behavior.

The generator is bounded to the pinned public image helper families and emits ordinary C declarations and thin C++ calls. Regeneration uses `python3 tools/generate-image-adapters.py` after fetching the pinned source.

## Phase B portable native validation

The final matched-SDK [comparative campaign](performance.md) uses runtime CPU dispatch with AVXNECONVERT, AVX512 and AVXVNNI disabled on Ryzen 7 5800H. BF16 storage remains supported through portable implementations; absent BF16 hardware instructions are not a reason to disable that storage API. Changing native ISA flags requires a clean rebuild and a new common baseline. `tools/patches/ncnn-20261020-vulkan-cast-capability.patch` restricts Vulkan Cast to its FP32/FP16 shaders, so INT8 Cast uses native CPU fallback in GPU-enabled graphs. GPU-enabled does not establish shader execution for every operator.
