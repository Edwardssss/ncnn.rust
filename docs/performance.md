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

The in-repo benchmark separates the same helpers. `api_benchmark preprocess 200 65536` on one
Windows host, pinned to one core, minimum of three runs:

| Helper | median us | note |
|---|---:|---|
| borrow (`from_external`) | 0.09 | header only |
| `deep_clone` | 11.6 | allocation and one copy |
| `reshape` | 12.5 | detaches because the native result aliases |
| `into_reshape` | 0.19 | rewrites the header |
| `normalized` | 17.7 | one pass into fresh storage |
| `normalize_in_place` | 4.7 | the same pass, resident storage |
| `flatten` | 0.92 | |
| `with_packing(4)` | 0.91 | |
| `copy_make_border` | 34.5 | 260 KB output |
| `to_pixels` | 36.7 | |
| `warp_affine` | 250.2 | bilinear, full frame |

`normalized` and `normalize_in_place` apply the scale and bias of the native `Scale` path in one pass
over the destination, so the copying form costs about 1.5 times a bare `deep_clone` instead of a clone
followed by a native layer that walks the buffer again. Allocation and first touch dominate every row
that allocates, and a shared host moved these medians by three times until the process was pinned to
one core, which is why the table reports a minimum over runs. CI runs the same command in a debug
build as a smoke check, so its printed medians verify the path rather than the timings.

GPU lifecycle: allocator create/drop 0.330 us, upload-download-drop 577.283 us, cache save
914.730 us, cache load 331.940 us. Seven randomized cold/warm process pairs report first-request
medians of 23.16 and 10.95 ms with steady medians of 5.53 and 5.52 ms.

## Request path against the native C API

`api_benchmark` also runs the native C API request path over the same graph, so the wrapper modes
have a reference in the same build. Three references cover the shapes a request takes: a borrowed
input read into a caller buffer (`native-request`), an owned input read into a caller buffer
(`native-buffer`), and an owned input read in place through the pointer the handle carries
(`native-owned`). The borrowed reference keeps light mode off, because a borrowed input cannot run it
([Tencent/ncnn#1543](https://github.com/Tencent/ncnn/issues/1543)); the owned references enable the
option, which is what the wrapper modes they compare against use.

Nine process runs per mode over one 65536-element FP32 tensor, one thread, pinned to one core. The
Linux column is the median and the Windows column is the lowest run.

| Mode | Windows ns | Linux ns | Rust allocations per request |
|---|---:|---:|---:|
| `native-request` (borrowed input, caller buffer) | 25586 | 23804 | none |
| `native-buffer` (owned input, caller buffer) | 18764 | 18200 | none |
| `native-owned` (owned input, read in place) | 12050 | 10422 | none |
| `owned-guard` (owned input, borrowed output) | 12209 | 10930 | 2 small |
| `guard-into` (guard consumed into an owned tensor) | 12197 | 10784 | 2 small |
| `owned-mat` (owned input, zero-copy owned output) | 12927 | 11124 | 2 small |
| `resolved-reused` (endpoint tokens, caller buffer) | 20121 | 18040 | 0 |
| `borrowed-reused` (borrowed input, caller buffer) | 26404 | 23908 | 5 x 123 B |

The two modes that allocate a data buffer per request sit in a different regime, so they are listed
separately with the guest's median: `owned-view` (owned value vector, one 262 KB vector) 226298 ns,
and `copy` (cloned input and output, two 262 KB buffers plus the native clone of the input)
497350 ns. On the Windows host the same two modes measured anywhere between 19573 and 143391 ns, and
between 386847 and 471300 ns: the heap alternates between recycled and freshly committed blocks, so a
median there describes the host's page-commit state rather than the request path. The modes in the
first table kept their ranking under the same alternation, and their spreads stayed inside a factor
of three. The allocation counts are the reproducible part: the guard, the two owned-tensor forms and
the two buffer forms allocate no data buffer, `owned-view` allocates the output vector, and `copy`
allocates two data buffers plus the native clone its input needs.

The table separates four effects.

The wrapper matches the C API when both run the same steps. `guard-into`, `owned-mat` and
`resolved-reused` land within one to seven percent of `native-owned` and `native-buffer` on both
platforms, so the binding adds the call, the endpoint resolution and nothing else.

The owned forms pay one native copy, and the wrapper adds none. `Extractor::extract` deep-clones an
output whose storage comes from the network's local blob pool (`src/net.cpp`, the branch that
detaches the result so it can outlive the network), and that pool is the default. The clone is
therefore the copy an independent tensor costs; `extract_mat`, `extract_stored_owned` and
`into_mat` keep that buffer instead of duplicating it, which is why they hold no allocation of their
own. The value-vector forms add a second buffer: `owned-view` allocates 262144 bytes per request on
top of the native clone, which costs twenty times the owned-tensor form on the guest and more as soon
as the Windows host leaves its fast heap state.

A caller that keeps its native input handle alive defeats light-mode reuse. Measured on the guest
with the same graph, submitting an owned tensor through the C API and dropping the caller's handle
before the request costs 10.2 to 11.2 us; keeping the handle alive until after extraction costs 219
to 251 us, because the input blob never reaches a refcount of zero and the layer output cannot reuse
its buffer. `Extractor::input` consumes the tensor, so the wrapper cannot hit that state.

The copies themselves are cheap and the allocator decides their cost. A standalone probe copies 256
KB in 5.6 to 6.4 us on the Windows host and 6 us on the guest, while allocating and freeing a 256 KB
buffer per request costs an order of magnitude more on a heap that returns freed pages to the
kernel. Raising the glibc thresholds isolates the effect on the guest (default against
`MALLOC_MMAP_THRESHOLD_=268435456` and `MALLOC_TRIM_THRESHOLD_=268435456`): `owned-view` drops from
222878 ns to 16187 ns, `copy` from 489899 ns to 31019 ns, while `owned-guard` and `owned-mat` hold at
about 10.4 us because they allocate no data buffer. At 1024 elements, where the same buffers come
from recycled chunks, the four modes collapse to 1128, 1134, 1265 and 1565 ns. Windows has no
equivalent setting, so the host that cannot raise a threshold picks a reuse form or an allocator that
retains pages; `heap::retain_pages` reports `false` there and the crate's own test asserts both
halves of that claim.

A handle from `Net::extractor` restarts itself once a request extracted, because the native
extractor caches the blobs it produced and would otherwise answer the next request from that cache.
The restart costs one native extractor construction, which stays inside the run-to-run spread of the
figures above; a caller that builds one handle per request never pays it, because a fresh handle has
nothing to restart.

## Reusing native memory

The network allocates its intermediate tensors per request, and on a heap that returns freed memory
to the kernel that allocation costs more than the arithmetic. `NetBuilder::reuse_blob_memory` gives
the network one pool allocator for intermediate blobs and one for layer workspace, so every
extractor takes that memory from the pool instead. The `api_benchmark pooled-request` mode times the
five request forms on a pooled network next to the same forms on a default one, rotating the cases
and reporting medians; 65536 elements, one core, three runs per host:

| Form | WSL2 glibc, default | WSL2 glibc, pooled | Windows, default | Windows, pooled |
|---|---:|---:|---:|---:|
| `guard` (borrowed output) | 24232 to 29170 ns | 17352 to 17866 ns | 46250 ns | 22474 ns |
| `extract_mat` (zero-copy owned output) | 23871 to 31283 ns | 23610 to 25118 ns | 47356 ns | 44354 ns |
| `extract_owned` (owned value vector) | 29392 to 34542 ns | 24091 to 24962 ns | 98998 ns | 47087 ns |
| `infer_into` (owned input, reused buffer) | 30697 to 31413 ns | 23589 to 23976 ns | 55282 ns | 29154 ns |
| `infer_borrowed_into` (borrowed input and buffer) | 22916 to 23433 ns | 16793 to 17920 ns | 29552 ns | 18502 ns |

The guest values are the range of three runs and the Windows values the minimum of three runs.
Rotating the cases warms the heap for every form, so the absolute figures here are lower than the
single-mode medians above and describe a mixed workload; the ratios within a row are what the mode
measures. The owned vector gains most because its buffer is the one that allocates. The borrowed
forms gain too even though they allocate nothing on the caller side, because the pooled intermediates
remove the engine's own per-request allocation. `extract_mat` gains nothing, and that is the point:
a pooled network skips the native detach clone, so the wrapper copies instead of keeping the buffer,
which makes pooling and zero-copy ownership alternatives rather than additions. The pools keep the
blocks they served for as long as the network lives, so the retained set grows by the working set of
one request and drops with the network.

`heap::retain_pages` addresses the other half: the pages of a buffer the caller allocates. A pooled
network does not remove that buffer, and a network without pools still allocates its tensors, so the
two settings are independent.

## Threads and workers

A network and its extractor are single-threaded: both borrow their owning thread, so a service
scales by giving each worker its own pair. `crates/ncnn/examples/threads.rs` builds that pair inside
every worker with the native thread count at one, and reports the aggregate throughput next to a
single network that uses every core. 65536 elements, one ReLU layer, 1000 requests per round, median
of three rounds, on the WSL2 guest with 8 logical cores:

| Configuration | requests/s | us per request |
|---|---:|---:|
| one network, every core | 41.7k | 24.0 |
| one network per worker, 1 worker | 37.6k | 26.6 |
| one network per worker, 2 workers | 77.5k | 25.8 |
| one network per worker, 4 workers | 104.6k | 38.2 |

The native thread pool and one worker on one thread reach nearly the same per-request cost on a graph
this small, where dispatch rather than arithmetic dominates; the pool neither wins nor loses.
Independent workers then multiply the aggregate: four workers reach 2.8 times the single-worker
throughput at 1.4 times the latency, because they share the cores. A graph whose layers are heavy
enough to amortize the dispatch cost moves the other way, so the choice is a measurement rather than
a rule, and the example prints both forms for the graph an application uses. The 16-core Windows host
showed the same ranking across three runs, 13k to 17k requests per second for one network over every
core and 45k to 62k for four workers, with a much wider spread because other work was using it; the
table therefore reports the guest, whose three runs agree.

## Allocator choice

The wrapper never sets an allocator for its caller, so an application chooses one. The choice matters
for the forms that return independent storage, because each request allocates and frees an
output-sized buffer. The same code with and without a thread-caching allocator, 65536 FP32 elements on
one Windows host, minimum of three rounds:

| Measurement | system heap | `mimalloc` |
|---|---:|---:|
| owned-output request (`infer`) | 321679 ns | 36711 ns |
| borrowed-input, reused-buffer request | 32330 ns | 27832 ns |
| plain 256 KB `Vec` clone | 12142 ns | 5420 ns |

The owned-output form drops by roughly eight times, and the form that allocates no data buffer does
not move, which is the confirmation that the cost is the allocate-and-free cycle rather than the
wrapper. `cargo run --release -p ncnn-rust --example fast_allocator` prints the same two figures with
the allocator declared; remove the `#[global_allocator]` line to measure the system heap.

A glibc heap needs no replacement allocator, only the two thresholds that decide when a buffer
leaves the heap. `heap::retain_pages` applies both from inside the process, which matters when the
service cannot set its own environment. The same two forms, 65536 FP32 elements on the WSL2 guest
below, minimum of three rounds each:

| Measurement | before | after `heap::retain_pages` |
|---|---:|---:|
| owned-output request (`infer`) | 255541 ns | 31754 ns |
| borrowed-input, reused-buffer request | 23603 ns | 23912 ns |

The owned-output form drops by about eight times, and the reused-buffer form does not move, which is
the same confirmation the allocator table gives. `cargo run --release -p ncnn-rust --example
heap_retention` prints both figures; the call reports `false` on targets whose heap has no such
setting, so the figures there differ only by host noise.

The thresholds are glibc settings, and the same buffer pattern shows why Windows needs none of them.
The example also prints `buffer_pair_ns`, the cost of allocating and freeing two output-sized
buffers per round without a network: 203 us on the WSL2 guest above, and 15 to 24 us on the Windows
host next to the recorded tables, where the heap serves the pattern from committed pages and faults
almost nothing. `heap::retain_pages` therefore reports `false` on Windows, and the crate's own test
asserts both halves of that claim on whichever host runs the suite.

## Linux hosts

A Linux run of the same request-path modes on a WSL2 Debian guest (8 vCPU AMD Ryzen 7 5800H, glibc
2.41, gcc 14.2, `libncnn.so` built from `vendor/ncnn-20261020`, one thread, pinned to one core,
median of nine runs) reproduces the Windows ranking and shows the mechanism:

| Mode | ns per request | Rust allocations per request |
|---|---:|---:|
| `native-request` | 23804 | none |
| `native-buffer` | 18200 | none |
| `native-owned` | 10422 | none |
| `owned-guard` | 10930 | 2 small |
| `guard-into` | 10784 | 2 small |
| `owned-mat` | 11124 | 2 small |
| `resolved-reused` | 18040 | 0 |
| `borrowed-reused` | 23908 | 5 x 123 B |
| `owned-view` | 226298 | 1 x 262 KB |
| `copy` | 497350 | 2 x 262 KB |

The modes that borrow or reuse an output match the C API within a few percent; the mode that
returns an owned value vector pays for page faults rather than for copies. A standalone probe of
the allocator shows the whole effect: allocating and freeing one 256 KB buffer per iteration
costs 4.8 us and faults nothing, because the freed chunk is reused immediately, while holding two
distinct 256 KB buffers alive per iteration costs 203 us and faults 96 times per iteration,
because glibc serves each buffer from a fresh mapping and returns it to the kernel on free. The
fault count is the cost: 28803 minor faults over 300 iterations, about 2.1 us per fault on this
guest. Running the same two-buffer probe with `MALLOC_MMAP_THRESHOLD_=1073741824` and
`MALLOC_TRIM_THRESHOLD_=-1` removes both the mapping round trip and the faults and brings it to
9.4 us with 4 faults in total. The request path follows the same rule: `owned-view` allocates the
output vector per request and lands at 226 us, `copy` allocates two data buffers plus the native
clone and lands at 497 us, and setting the same two variables in the environment drops them to
16.2 us and 31.0 us.

Two conclusions follow. Reusing storage is the portable fix, and `infer_into`, `infer_borrowed_into`
and `extract` already do that at native parity. Where an application cannot reuse, the heap decides:
`heap::retain_pages` sets both thresholds from inside the process, a host that starts the process
can set the same two variables instead, and an allocator that retains pages is the equivalent on a
host whose heap has no such setting. A thread-caching allocator did not help on this guest (226 to
235 us for `owned-view` with `MIMALLOC_PURGE_DELAY=-1` and `MIMALLOC_ARENA_EAGER_COMMIT=1`), so
those two thresholds are the reliable lever there. The per-fault cost of a WSL2 guest is several
times that of bare metal, so the absolute gap is smaller on a native Linux host, but the ranking and
the mechanism are the same.

## Reproducing

The tables come from a campaign harness that is maintained outside this repository. In-repo
commands cover the same code paths, so a change can be checked without it:

```sh
python3 tools/fetch-models.py                                   # pinned fixtures into target/model-fixtures
cargo run --release -p ncnn-rust --example api_benchmark         # per-request cost and allocation counts
cargo run --release -p ncnn-rust --example api_benchmark -- native-request 300 65536
cargo run --release -p ncnn-rust --example api_benchmark -- preprocess 200 65536
cargo run --release -p ncnn-rust --example api_benchmark -- pooled-request 200 65536
cargo run --release -p ncnn-rust --example heap_retention       # heap retention before and after
cargo run --release -p ncnn-rust --example model_regression -- \
  target/model-fixtures target/cpu-reference/ncnn_cpu_reference  # native C++ versus Rust outputs
cargo test --workspace --features vulkan                         # contracts and the conformance suite
# Linux: compare the owned-output form with and without the mapping thresholds
cargo run --release -p ncnn-rust --example api_benchmark -- owned-view 300 65536
MALLOC_MMAP_THRESHOLD_=1073741824 MALLOC_TRIM_THRESHOLD_=-1 \
  cargo run --release -p ncnn-rust --example api_benchmark -- owned-view 300 65536
```

The [model fixtures](model-fixtures.md) page lists the reference build commands. The recorded
campaign behind the tables ran 314 processes, 146 CPU cases, 146 Vulkan-enabled cases, 3358 steady
groups and 23506 steady samples.

Fourteen cases extend the highest-risk operators with parameter variants taken from the upstream
tests: Convolution dilation, stride and activation; Pooling average, global and counted padding;
Interp nearest, bilinear and bicubic; three Gemm shapes; and Softmax on two axes. Each variant is
validated against the native reference on both devices.
