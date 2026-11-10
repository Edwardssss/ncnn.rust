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
that allocates, which is why a rotation of the cases and a minimum over runs is used. CI runs the same
command in a debug build as a smoke check, so its printed medians verify the path rather than the
timings.

GPU lifecycle: allocator create/drop 0.330 us, upload-download-drop 577.283 us, cache save
914.730 us, cache load 331.940 us. Seven randomized cold/warm process pairs report first-request
medians of 23.16 and 10.95 ms with steady medians of 5.53 and 5.52 ms.

## Request path against the native C API

`api_benchmark` also has a `native-request` mode that runs the C API request path over the same graph,
so the wrapper modes have a reference in the same build. At 65536 elements on one Windows host, one
run, against the C path that borrows its input and copies the output into a reusable buffer:

| Mode | ns per request | Rust allocations per request | note |
|---|---:|---:|---|
| `native-request` | 25462 | none | C API reference |
| `owned-guard` (owned input, borrowed output) | 13720 | 2 small | no output copy |
| `resolved-reused` (endpoint tokens, reusable buffer) | 19734 | 0 | allocation-free steady state |
| `owned-view` (owned input, independent owned output) | 107419 | 1 x 262 KB | one allocation and copy of the output |
| `copy` (double-copy stress mode) | 368352 | 4 x 262 KB | input clone plus output clone |

The allocation counts are the reproducible part: timing on a loaded host drifts by several times
between runs of the same mode, while the zero-allocation and copy counts do not move. The modes that
borrow the output or reuse a buffer land at or below the C reference, and the mode that returns a
fresh independently owned vector pays one allocation and one full copy of the output. A standalone
measurement of a 256 KB `Vec` copy on the same host reported 77 to 108 us, the same order as the whole
inference, so the owned-output form is dominated by allocator and memory behaviour rather than by the
wrapper.

A handle from `Net::extractor` restarts itself once a request extracted, because the native
extractor caches the blobs it produced and would otherwise answer the next request from that cache.
The restart costs one native extractor construction, which stays inside the run-to-run spread of the
figures below; a caller that builds one handle per request never pays it, because a fresh handle has
nothing to restart.

## Reusing native memory

The network allocates its intermediate tensors per request, and on a heap that returns freed memory
to the kernel that allocation costs more than the arithmetic. `NetBuilder::reuse_blob_memory` gives
the network one pool allocator for intermediate blobs and one for layer workspace, so every
extractor takes that memory from the pool instead. The `api_benchmark pooled-request` mode times the
four request forms on a pooled network next to the same forms on a default one, rotating the cases
and reporting medians; 65536 elements, one thread, one run of the mode per host:

| Form | WSL2 glibc, default | WSL2 glibc, pooled | Windows, default | Windows, pooled |
|---|---:|---:|---:|---:|
| `guard` (borrowed output) | 23731 ns | 16724 ns | 154280 ns | 37474 ns |
| `extract_owned` (owned output) | 227135 ns | 23157 ns | 332489 ns | 147012 ns |
| `infer_into` (owned input, reused buffer) | 31041 ns | 22960 ns | 155006 ns | 48503 ns |
| `infer_borrowed_into` (borrowed input and buffer) | 22768 ns | 16911 ns | 40910 ns | 18978 ns |

The owned form gains most because it was the form that allocated: about ten times on the glibc guest
and two to four times on the Windows host, whose heap serves the same pattern more cheaply, and the
Windows figures move between runs with the load of the host. The pooled forms land within a factor of
two of each other on the glibc guest, which is what a request path bound by the engine rather than by
the allocator looks like; the Windows host stays above them because each request still copies its
output through that heap. The pools keep the blocks they served for as long as the network lives, so
the retained set grows by the working set of one request and drops with the network.

`heap::retain_pages` addresses the other half: the pages of a buffer the caller allocates. A pooled
network does not remove that buffer, and a network without pools still allocates its tensors, so the
two settings are independent.

## Threads and workers

A network and its extractor are single-threaded: both borrow their owning thread, so a service
scales by giving each worker its own pair. `crates/ncnn/examples/threads.rs` builds that pair inside
every worker and sets the native thread count to one, then times a single network that uses every
core on the same graph. 65536 elements, one ReLU layer; the Windows figures are three runs of one
configuration each, the WSL2 figures one run:

| Mode | Windows, 16 cores | WSL2 glibc, 8 cores |
|---|---:|---:|
| one network, every core, 1 worker | 192 to 221 us per request | 108 us per request |
| one network per worker, 1 worker | 147 to 188 us per request | 38 us per request |
| one network per worker, 2 workers | 95 to 101 us per request | 27 us per request |
| one network per worker, 4 workers | 44 to 57 us per request | 9 us per request |

The native thread pool returns little on a graph this small, because each request dispatches over
the pool and waits: one worker that spreads the request over every core lands within a small factor
of one worker on one thread. Independent workers then scale within the core budget, and four workers
reach two to four times the aggregate throughput of one. A graph whose layers are heavy enough to
amortize the dispatch cost moves the other way, so the choice is a measurement rather than a rule;
the example prints both forms for the graph an application uses. The host load moves these figures
between runs, which is why the table gives the spread instead of a single median.

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
minimum of three runs) reproduces the Windows ranking and shows the mechanism:

| Mode | ns per request | Rust allocations per request |
|---|---:|---:|
| `native-request` | 23371 | none |
| `resolved-reused` | 15934 | 0 |
| `borrowed-reused` | 24349 | 5 x 123 B |
| `owned-guard` | 10053 | 2 small |
| `owned-view` | 208833 | 1 x 262 KB |
| `copy` | 473106 | 4 x 262 KB |

The modes that borrow or reuse an output match or beat the C API; the modes that return
independent storage pay for page faults rather than for copies. A standalone probe of the
allocator shows the whole effect: allocating and freeing one 256 KB buffer per iteration costs
4.8 us and faults nothing, because the freed chunk is reused immediately, while holding two
distinct 256 KB buffers alive per iteration costs 203 us and faults 96 times per iteration,
because glibc serves each buffer from a fresh mapping and returns it to the kernel on free. The
fault count is the cost: 28803 minor faults over 300 iterations, about 2.1 us per fault on this
guest. Running the same two-buffer probe with `MALLOC_MMAP_THRESHOLD_=1073741824` and
`MALLOC_TRIM_THRESHOLD_=-1` removes both the mapping round trip and the faults and brings it to
9.4 us with 4 faults in total. The request path follows the same rule: `owned-view` allocates the
input tensor and the output vector per request and lands at 209 us, `copy` allocates four buffers
and lands at 473 us, and setting the same two variables in the environment drops `owned-view` to
17.6 us.

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
