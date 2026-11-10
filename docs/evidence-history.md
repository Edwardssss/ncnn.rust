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
