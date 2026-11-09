# Native ABI and adapter decisions

This page covers the layer between the safe wrappers and ncnn: the adapter ABI, the decisions the
project adapter makes where the upstream C API is incomplete, and the scope the runtime validation
campaigns reach. The [safety contracts](contracts.md) page covers what the safe API promises.

## Adapter ABI

Project adapter names use `ncnn_rs_`. The adapter compiles against the selected installed ncnn headers and links to the matching static or shared native library. Applications supply compatible compiler ABI, CRT, and native runtime dependencies. Adapter functions that allocate C++ objects translate exceptions to status codes or null handles. Upstream C API entry points retain upstream allocation and termination behavior.

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
