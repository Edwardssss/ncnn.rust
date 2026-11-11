# Evidence history

Dated campaign records behind the conclusions in [Platform support](support-matrix.md). Each
section names the machine, the command and the outcome. Numbers describe that campaign only.

## CPU and Vulkan validation campaign

Current shared and static Windows Vulkan and CPU-only checks plus Debian WSL llvmpipe are recorded
in the [performance report](performance.md). Vulkan runs passed three integration contracts and
eleven doctests; CPU-only runs passed two contracts and ten doctests. Physical Linux GPU, native
ARM and macOS execution were not established by that campaign.

The owned-batch and sequential-guard additions passed Linux x86-64 CPU and MSRV checks and ARM64
QEMU CPU checks. The extended native runner passed all 42 cases on Linux x86-64 and its 25
generated cases under ARM64 QEMU; the compact run repeats a generated subset rather than adding
distinct cases. The Windows CPU SDK passed the workspace tests, the 42 full-runner cases and the
25-case compact runner.

Earlier CPU evidence came from Linux x86-64 with a Windows virtual machine. Vulkan evidence comes
from native Windows/NVIDIA hardware. No Apple SDK or connected Mac runner is recorded for that
period, so those results do not establish macOS execution.

## Debian WSL request-path campaign

Debian WSL2 guest on a Ryzen 7 5800H host, 8 vCPU, 25 GB RAM, glibc 2.41, gcc 14.2, Rust 1.92,
ncnn 20261020 built as a shared SDK from `vendor/ncnn-20261020`, one thread, pinned with
`taskset -c 2` and repeated at least three times per mode. The request-path modes ranked as they do
on Windows: `owned-guard` 10053 ns, `resolved-reused` 15934 ns, `native-request` 23371 ns,
`borrowed-reused` 24349 ns, `owned-view` 208833 ns and `copy` 473106 ns per request. The workspace
tests passed on the same guest: 52 conformance cases, 3 further integration cases and 13 doctests.
That run built without the Vulkan feature; the campaign below adds it.

A separate probe isolated the cause of the owned-output cost. One 256 KB `Vec` allocated and freed
per iteration costs 4.8 us with no faults; two distinct 256 KB buffers held per iteration cost
203 us with 28803 minor faults over 300 iterations; the same probe with
`MALLOC_MMAP_THRESHOLD_=1073741824` and `MALLOC_TRIM_THRESHOLD_=-1` costs 9.4 us with 4 faults in
total; and `owned-view` drops from 219 us to 17.6 us under the same two variables.
`ncnn_rust::heap::retain_pages` applies the same two settings from inside the process and moved the
owned-output request from 255541 ns to 31754 ns on this guest while the reused-buffer form stayed at
23603 ns and 23912 ns.

Vulkan ran on the same guest against Mesa lavapipe 25.0.7, the software device the Linux Vulkan CI
job uses: `vulkaninfo` reported one physical device, `llvmpipe (LLVM 19.1.7, 256 bits)`, with FP16
and INT8 packing, storage, uniform and arithmetic support, and `cargo test --workspace --features
vulkan` passed the 52 conformance cases, the four Vulkan contracts, the heap retention test and the
15 doctests. The `add -- --vulkan` and `vulkan_resident` examples ran on the device: the resident
example printed the device line, `[5.0, 7.0, 9.0]` from a resident upload, and the pool allocator as
32 bytes live against a 48-byte high-water mark.

## Upstream light-mode defect

The rejection that `Extractor::input_borrowed` applies to light mode guards a native defect. Recorded
on 2026-11-10 against ncnn master `14fb7ed7` with a standalone C++ program, no Rust involved: a
two-layer `Input -> ReLU` graph of 64x64x3 FP32, light mode at its native default, and the input built
as `ncnn::Mat(w, h, c, pixels.data(), sizeof(float))`.

| Host | Before | After the null-refcount guard |
|---|---|---|
| Windows x64, MSVC 14.50, release, static CPU library | exit code `0xC0000005` inside extract | all four input/light-mode combinations extract, `caller_kept=1` |
| Linux x86-64, GCC 14.2, release, `NCNN_VULKAN=OFF` | `SIGSEGV`, exit code 139, `NetPrivate::do_forward_layer -> NetPrivate::forward_layer -> Extractor::extract` | the same four combinations pass |

Upstream tracks the defect as [Tencent/ncnn#1543](https://github.com/Tencent/ncnn/issues/1543) and the
proposed fix is [Tencent/ncnn#7270](https://github.com/Tencent/ncnn/pull/7270), which treats a missing
refcount as shared storage so the in-place layer clones instead of writing through the caller
buffer. test_relu, test_absval, test_convolution and test_packing pass on the patched tree under
Linux x86-64.

## Extractor reuse

Recorded on 2026-11-11 with a three-layer `Input -> ReLU -> BinaryOp` graph whose output is the sum of
the ReLU and the input, one extractor, two requests with different inputs, ncnn 20261020 on Windows
x64. A fresh handle answered every request, while a reused handle answered the second request with
the values of the first:

| Input | Fresh extractor | Reused extractor |
|---|---|---|
| `-1` | `[-1, -1, -1]` | `[-1, -1, -1]` |
| `5` | `[10, 10, 10]` | `[-1, -1, -1]` |

The cause is upstream: a layer runs only while its output blob is empty, and a completed request
leaves that blob filled. `Extractor::clear` looks like the fix but calls `blob_mats.clear()`, after
which the handle rejects every input, so a restart recreates the native extractor instead. The
wrapper therefore restarts in `input` and `input_borrowed`, re-applies the per-extractor settings to
the new handle, and both cases above return their own answer.

The same campaign recorded the light-mode restriction the setter documents: with light mode on, a
graph whose input feeds two layers returns the values of the second consumer from a released blob
(`[1, 2, 3]` instead of `[-1, -1, -1]` for an input of `-1`), because `Input` cannot refill a blob
after its first consumer released it. Linear graphs are unaffected.

## Pooled request paths

`api_benchmark pooled-request 200 65536` on the WSL2 guest and on the Windows host, five rotated
rounds of 100 to 200 iterations each, medians in nanoseconds per request. Each column is one run of
the mode, the WSL2 pair from one command and the Windows pair from another:

| Form | WSL2 default | WSL2 pooled | Windows default | Windows pooled |
|---|---:|---:|---:|---:|
| `guard` | 27546 | 18308 | 104975 | 33679 |
| `extract_owned` | 251569 | 24375 | 225228 | 112721 |
| `infer_into` | 32288 | 25400 | 104164 | 40784 |
| `infer_borrowed_into` | 24718 | 17602 | 43721 | 19169 |

A separate probe attributed the default cost before the pools existed: on the Windows host the
extractor itself costs 207 ns, an owned input costs 27.9 us, and the forward plus read-out costs 200
to 290 us, while the same forward under light mode costs 16 to 44 us. The native intermediate-tensor
allocation, not the wrapper, is what the pools remove.

The pooled figures were measured again after the extractor restart landed, three runs of the same
mode per host at 65536 elements: the WSL2 guest reported `guard` 16426 to 17228 ns, `extract_owned`
23157 to 24121 ns, `infer_into` 22960 to 24310 ns and `infer_borrowed_into` 16911 to 17696 ns, and
the Windows host reported `guard` 37474 ns, `extract_owned` 147012 ns, `infer_into` 48503 ns and
`infer_borrowed_into` 18978 ns at 100 iterations. The restart costs one native extractor
construction per request, which stays inside the spread of these runs. A third round on 2026-11-11,
with both hosts pinned to one core, reproduced that ranking while the Windows host was shared:
`guard` 30081 ns, `extract_owned` 84442 ns, `infer_into` 37113 ns and `infer_borrowed_into` 18102 ns
pooled, against 97436, 194811, 97429 and 39843 ns on a default network.

## Worker scaling

`cargo run --release --example threads` on both hosts, 65536 elements and one ReLU layer, native
thread count one per worker, 1000 requests per round and the median of three rounds. The WSL2 guest
on 8 logical cores reported 41.7k requests per second (24.0 us each) for one network over every
core, 37.6k (26.6 us) for one worker, 77.5k (25.8 us) for two and 104.6k (38.2 us) for four. The
native pool therefore costs nearly what one worker on one thread costs on a graph this small, and
independent workers multiply the aggregate throughput. The 16-core Windows host reported 13k to 17k
requests per second for one network over every core and 45k to 62k for four workers in three runs, a
spread that follows the load of that host rather than the configuration.

## Preprocessing helpers

`api_benchmark preprocess 200 65536` on the Windows host, pinned to one core, minimum of three runs:
`borrow` 0.09 us, `deep_clone` 11.6, `reshape` 12.5, `into_reshape` 0.19, `normalized` 17.7,
`normalize_in_place` 4.7, `flatten` 0.92, `with_packing(4)` 0.91, `copy_make_border` 34.5,
`to_pixels` 36.7 and `warp_affine` 250.2. The same command without the affinity reported 19 to 42 us
for `deep_clone`, which is what a shared host does to a benchmark bounded by allocation and first
touch; pinning the process to one core cut that spread to under a tenth.

## Windows heap and Vulkan runs

The same heap probe on the Windows host that produced the allocator tables: two distinct 256 KB
buffers allocated and freed per round cost 15 to 24 us and fault almost no pages, against 203 us and
96 faults per round on the WSL2 guest, so the glibc thresholds address a mechanism the Windows heap
does not have. The crate reports the settings as unavailable there, and `cargo test --test heap` on
that host asserts both the absent faults and the `false` results.

Full Vulkan conformance runs on the Windows device (GeForce RTX 3060 Laptop, 6 GB, driver 596.49)
passed 52 of 52 cases at 1, 2, 8 and 16 test threads, six runs in a row, and four suites run
concurrently all passed. During a passing run `nvidia-smi` reported 1293 MiB of the device in use,
of which 1241 MiB came from other applications, so the suite held tens of megabytes. Three earlier
runs in the same session aborted with an access violation after reporting about half the cases as
failed; the same abort reproduced on the commit before the change under measurement, and no
subsequent run reproduced it. The observation is recorded with the ruled-out causes on the
[troubleshooting page](troubleshooting.md#the-vulkan-suite-aborts-on-a-physical-gpu).

## Windows Vulkan hardware

Ryzen 7 5800H, GeForce RTX 3060 Laptop GPU, NVIDIA driver 596.49, Rust 1.96.1, CMake 4.2.1, MSVC
14.50.35717, Windows SDK 10.0.26100.0, ncnn 20261020 and glslang 15.4.0. Native device metadata
reported index 0, API 4211017, vendor 4318, device 9568 and driver 2500608000. FP16
packed/storage/uniform/arithmetic and BF16 packed/storage capability reported supported.

Full workspace CPU and Vulkan contracts passed with both shared and static Vulkan SDKs, including
native VkMat execution, FP32 and FP16, lifetime and reload checks, and clean process exit. Each
linkage also passed four trained-model native C++ versus Rust comparisons, NanoDet-m and PP-OCRv5
recognition with FP32 and FP16 options. All comparisons reported maximum absolute error zero under
identical native and Rust options, and repeated Rust output was exactly equal. That result is
neither a CPU-versus-GPU nor an FP32-versus-FP16 equality claim. Submitted nonempty native VkMat
intermediates contained 32-bit and 16-bit elements respectively; downloaded Mats contained FP32
values.

Resident tensors, buffer alias imports, completed guards, corrupted-cache rejection and
synchronized benchmarks also passed on this device. Cross-profile output differences were measured
rather than labeled task accuracy. Other physical Vulkan platforms and exclusive GPU placement of
every layer remain unverified.

## Static Vulkan configurations

An early closure used `static=SPIRV,static=glslang` from the SDK's exported glslang targets with
`NCNN_SIMPLEVK=ON`, `NCNN_OPENMP=OFF`, runtime CPU dispatch on and the dynamic MSVC CRT. Both static
executables imported no ncnn DLL; CRT, system DLLs and a Vulkan driver were still required.

The final Phase B clean-first campaign used runtime CPU dispatch on, AVXNECONVERT/AVX512/AVXVNNI
off, BF16 on and the explicit closure
`static=glslang,static=SPIRV,static=MachineIndependent,static=GenericCodeGen,static=OSDependent`.
Two integration tests, eleven doctests and the static native-reference GPU model regression passed.
A campaign manifest recorded the exact flags; it is part of the local evidence tree and not of this
repository. Both closures are configuration-specific rather than an automatic dependency
resolver.

## Debian WSL software Vulkan

The same workspace Vulkan tests passed in Debian x86-64 WSL with Rust 1.92, GNU 14.2, libclang 19
and Mesa lavapipe `llvmpipe (LLVM 19.1.7, 256 bits)`. This is software Vulkan, not GPU passthrough.
The consolidated CPU and GPU contracts and all eleven doctests passed, covering FP32 and FP16
execution, native-reference agreement, resident transfers, imported-buffer readback,
failed-recording cancellation and cache corruption checks. The shared SDK was built from the
pinned source with an explicit `-DNCNN_VERSION=20261020`; omitting the override yields the source's
date-derived version and correctly fails the Rust SDK compatibility check.

That environment exposed an imported-buffer readback bug on unified-memory devices: native download
maps native allocations directly, while an external buffer descriptor has no mapped CPU address.
Imported downloads now clone into owned resident storage before native readback. The import
roundtrip covers the path on lavapipe, and Windows shared/static runs cover discrete-device
behavior.

Reproduction in WSL with an installed toolchain and SDK:

```sh
export PATH=/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin
export NCNN_INSTALL_DIR="$PWD/target/sdk-linux-vulkan" NCNN_LINK_KIND=dylib
export LIBCLANG_PATH=/lib/x86_64-linux-gnu
export LD_LIBRARY_PATH="$NCNN_INSTALL_DIR/lib"
export CARGO_TARGET_DIR="$PWD/target/rust-linux-vulkan"
cargo test -q --workspace --features vulkan
cargo clippy -q --workspace --all-targets --features vulkan -- -D warnings
```

The enumerated Vulkan device was lavapipe even though WSL exposed `/dev/dxg`, so Linux
physical-GPU execution remains open.

## Runtime and topology campaign

The four-installation matrix (Linux shared/simpleomp, Linux static/GCC OpenMP, Windows
shared/dynamic CRT, Windows static/static CRT) passed 17 trained-model cases per installation with
all output names, layouts and values compared against independent C++. All 68 cases reported
maximum absolute error zero and exact repeated Rust extraction, and the 18 option fields were
compared separately. Windows paths contained Unicode characters and spaces, including the native
mapped-loading path.

Valgrind reported zero memory errors and zero definite or indirect leaks; its process summary
contained 48 possibly-lost bytes and 648 reachable bytes. Synthetic DetectionOutput, LSTM and INT8
graphs provided expected-value cases alongside the trained models.

The OS topology API grew affinity storage dynamically and was exercised with synthetic CPU ID 1536.
The upstream sibling parser was tested with synthetic IDs 32, 64 and 1024, and the patched native
physical-core count agreed with the host OS at 12 logical CPUs and six physical cores. Real host
tests pinned, queried, executed and restored affinity. The Windows 10 Home virtual machine exposed
one processor group even with Microsoft's group-testing boot options, so nonzero-group execution
remains pending on a multi-group environment.

## Vulkan coverage

Each Windows linkage passed two integration contracts, ten compile-fail documentation tests and
four trained-model comparisons: NanoDet-m and PP-OCRv5 recognition with FP32 and FP16 options,
eight comparisons in total. Native C++ versus Rust maximum absolute error was zero under identical
options, and Rust outputs repeated exactly. Submitted nonempty native VkMat intermediates contained
32-bit and 16-bit elements; downloaded Mats were FP32. The same suites ran on Linux x86-64 WSL with
eleven documentation tests after the GPU ownership compile-fail case was added. Because native
per-layer fallback is allowed, none of this establishes GPU placement for every layer.

## Release 0.2.0 checks

The 0.2.0 tag published both packages from a Windows host with `cargo publish -p ncnn-rust-sys`
followed by `cargo publish -p ncnn-rust` once the index listed the first. A temporary crate outside
the workspace depended on `ncnn-rust = "0.2.0"`, resolved both packages from the registry, compiled
the adapters against the pinned SDK and printed the expected sum and INT8 round trip, which proves
the published archives build without the repository. crates.io reported `0.2.0` as the maximum
stable version for both packages afterwards.

`Release` then ran from a `workflow_dispatch` of the existing `v0.2.0` tag. It matched the tag with
the workspace version, skipped both packages because the versions already existed, and replaced the
GitHub release body with the changelog section for that version.

## Device memory exhaustion

Measured on the Windows host with the RTX 3060 Laptop GPU through the pinned Vulkan SDK. The device
reported a 5226 MiB heap budget. Uploading 256 MiB FP32 tensors and keeping every result alive
succeeded 22 times, 5,905,580,032 bytes in total; the 23rd upload failed with `vkAllocateMemory
failed -2` on stderr and `Error::NativeAt("Vulkan upload", -1)` in Rust. After the refusal the
retained tensors still downloaded, a 16-byte upload succeeded, and a resident `BinaryOp` graph
still returned `[5, 7, 9]`. `Allocator::tensor_bytes` reported both values equal at the high-water
mark and the live value dropping to zero after the tensors were released. Device loss and driver
reset were not exercised.

The test run that started in a new process immediately after that probe failed 23 of 37 conformance
cases and exited with status `0xC0000409`; the same command passed twice afterwards. A deliberate
exhaustion therefore leaves the driver in a state that outlives the process, which is the reason the
troubleshooting page sends an unattributable failure to device re-creation rather than to a retry.

## Attention on native ARM64

Dispatched `ci.yml` with `arm_mha_probe` on `ubuntu-24.04-arm` with the pinned SDK. The probe used
three configurations of `MultiHeadAttention` with a two-dimensional (embedding, sequence) operand:
`0=4 1=2 2=16`, `0=8 1=2 2=32` and `0=4 1=1 2=16`. All three returned finite values that match the
x86-64 run exactly (`1.72267`, `15.747786` and `1.818392` as the leading values). Running the layer
case with a three-dimensional `[4, 4, 1]` operand on the same runner returned `NaN` for the second
sequence row, index 4 to 7 of sixteen, while the x86-64 run stayed finite. The case therefore uses
the two-dimensional form, and no target carries an exemption any more.

## Owned extraction without a copy

The native extractor detaches every output from the network's local blob pool before it returns the
handle (`src/net.cpp`, branch at the end of `Extractor::extract`), so the buffer ncnn hands out
already belongs to the tensor. A probe that kept that handle across a second request and read it
afterwards still returned the first request's value, which confirms the buffer survives the
extractor, the network and later requests.

`extract_mat`, `extract_stored_owned` and `OutputGuard::into_mat` therefore take that buffer instead
of duplicating it. Two configurations cannot hand it out: a network built with `reuse_blob_memory`
installs its own blob allocator, which skips the native detach clone, and a probe that retained such
a tensor across two requests ended with ncnn's pool allocator aborting on destruction while two
blocks were still referenced; an extractor that submitted a borrowed input returns caller storage.
Both cases copy inside the wrapper.

The same probe campaign found the native counterpart of the copy cost. Submitting an owned tensor
through the C API and releasing the caller's handle before the request costs 10.2 to 11.2 us on the
WSL2 guest, while keeping the handle alive until after extraction costs 219 to 251 us with the same
graph, because the input blob never reaches a refcount of zero and the layer output cannot reuse its
buffer. `Extractor::input` consumes the tensor and therefore cannot reach that state.

Raising `MALLOC_MMAP_THRESHOLD_` and `MALLOC_TRIM_THRESHOLD_` isolates the remaining difference:
with the thresholds raised, the value-vector form drops from 222878 ns to 16187 ns and the
double-clone form from 489899 ns to 31019 ns, while the forms that allocate no data buffer stay at
10.4 us. At 1024 elements the four modes collapse to 1128, 1134, 1265 and 1565 ns.
## SDK patches in the measured builds

Every measured build applied the two patches in `tools/patches/`, which
`tools/build-ncnn-sdk.py` now applies automatically:

- `ncnn-20261020-cpu-siblings.patch` replaces the integer sibling mask with canonical kernel
  sibling strings, so physical-core and cache counts survive CPU IDs above 31.
- `ncnn-20261020-vulkan-cast-capability.patch` restricts Vulkan `Cast` to its FP32 and FP16
  shaders. Before it, INT8 and other conversions reported Vulkan support and produced mismatched
  results; with it they fall back to CPU.

The second patch explains why earlier campaigns recorded INT8 `Cast` failures under Vulkan while
CPU runs matched the reference. Local SDKs built before this change carry both patches applied by
hand, so their behaviour matches the current build script.
