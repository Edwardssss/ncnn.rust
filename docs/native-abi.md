# Native ABI and adapter decisions

This page covers the layer between the safe wrappers and ncnn: the adapter ABI, the decisions the
project adapter makes where the upstream C API is incomplete, and the scope the runtime validation
campaigns reach. The [safety contracts](contracts.md) page covers what the safe API promises.

## Adapter ABI

Project adapter names use `ncnn_rs_`. The adapter compiles against the selected installed ncnn headers and links to the matching static or shared native library. Applications supply compatible compiler ABI, CRT, and native runtime dependencies. Adapter functions that allocate C++ objects translate exceptions to status codes or null handles. Upstream C API entry points retain upstream allocation and termination behavior.

## Upstream adapter decisions

- Extractor option assignment in this release is a no-op, so per-extractor overrides go through `ncnn_rs_extractor_lightmode`, `ncnn_rs_extractor_allocator` and `ncnn_rs_extractor_cache_hint`, which call the C++ setters directly.
- The native `Extractor::clear` empties the blob table and leaves the handle unable to accept input, and a completed request leaves its blobs cached, so the wrapper restarts a handle by destroying the native extractor and creating another from the same network. The extractor setters are the void adapters, which match the upstream ones: they neither allocate nor throw.
- `ncnn_rs_net_layer` allocates a C handle around the layer a loaded network owns, and `ncnn_rs_borrowed_layer_release` frees that handle alone, so the native layer stays with the network. `Net::layer_handle` wraps the pair in a guard that releases the handle on drop.
- The native forward path reads the storage refcount of an in-place layer's bottom blob before it copies, and an external tensor records none, so the wrapper rejects a borrowed input while light mode is on instead of reaching that read.
- Layer and blob fields reach Rust through `ncnn_rs_net_layer_info`, `ncnn_rs_net_blob_count` and `ncnn_rs_net_blob_info`. The adapter copies the fields of one layer or blob into a single call and returns the names as pointers into the network's own objects, which the wrapper copies before the network can change. `ncnn_rs_net_layer_shape` reads the shape a layer recorded for one of its blobs without the caller holding a layer handle; a layer without a recorded shape reports zero extents.
- `ncnn_rs_mat_view` implements the sub views over one axis switch: axis 0 is `Mat::range`, 1 `row_range`, 2 `depth_range`, 3 `channel_range` and 4 `batch_range`. The adapter range-checks the requested run against the extent the tensor records and requires the rank that carries the axis, and the returned header keeps no refcount, so the Rust view borrows its source.
- `layer_type_index` uses `ncnn_layer_type_to_index`, and `ParamDict::kind` uses `ncnn_paramdict_get_type`. Those codes are not one enum: a binary parameter stream carries an untyped scalar, a text stream stores int, float, tensor or string, and the array forms have separate float and integer codes, so the wrapper maps them onto one documented set.
- `ncnn_rs_net_load_file` owns the mapped-loading switch, so the safe file loaders read through stdio while the mapped path stays an explicit unsafe opt-in. It widens a UTF-8 path on Windows, because the narrow native loaders cannot open a non-ASCII path there.
- `ncnn_rs_mat_flatten` disables the packing layout for the call, because the native kernel would otherwise return a packed one-dimensional tensor that the plain FP32 readers cannot read.
- `ncnn_rs_mat_normalize_in_place` applies the same channel view as `ncnn_rs_mat_normalized`, so the copying and the in-place form normalize below three dimensions identically.
- Both normalization adapters compute the scale and bias of the native `Scale` path in one loop over
the destination. The native `Mat::substract_mean_normalize` creates a `Bias` or `Scale` layer, loads
its parameters and weights and builds a pipeline on every call, which dominates the arithmetic for
the tensor sizes a request uses.
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
value against an independent C++ reference, and compares the 18 option keys those profiles configure separately.
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
