# Platform support

CPU is the default backend. Vulkan is optional and requires a matching SDK and driver. Every
target consumes an externally installed ncnn 20261020 SDK; the build script rejects other versions
unless `NCNN_ALLOW_UNTESTED_VERSION=1` is set. Dated campaign results, machine details and
reproduction logs live in the [evidence history](evidence-history.md).

## ncnn version matrix

`ci/ncnn-versions.json` holds the pinned release and the dynamic entries the scheduled workflow
resolves at run time. `.github/workflows/ncnn-matrix.yml` builds an SDK for each version. The pinned
release runs the workspace tests, audits enabled C API declarations against generated bindings and
defined symbols, and compares trained-model output with a native C++ reference built from the same
SDK. Other entries report the declarations the pinned header needs that the build lacks, because the
adapter is compiled against one release and cannot link another.

| Entry | Resolution | Requirement |
|---|---|---|
| `pinned` | `20261020` | Tests, C API audit and model comparison must pass |
| `latest` | Newest tag that is not already listed | Declaration gap reported each run |
| `previous` | Next distinct tag below that one | Declaration gap reported each run |

Because the pinned release is the newest tag today, `latest` and `previous` resolve to the two
releases below it, so the matrix records how far back the pinned headers reach. A failure in a
pinned row opens or updates a tracking issue, and a passing run comments on that issue and closes it,
so an open tracker means the matrix is currently failing.

## Targets

The adapter compiles for the Cargo target; bindgen runs on the host with the target triple and
target headers, and `cc` selects the target C++ runtime. CPU inference, tensors, loading and
ownership interfaces are identical on every target.

| Target | Implementation | Validated on |
|---|---|---|
| Linux x86-64 GNU | Static and shared SDKs, simpleomp or system OpenMP | Physical hardware and a WSL2 Debian guest, native CPU and Vulkan tests |
| Linux ARM64 GNU | Target compiler and sysroot, ARM feature queries, Linux topology | QEMU user emulation; native runner in CI |
| Windows x86-64 MSVC | `/MD` and `/MT`, static and shared SDKs, processor groups | Physical hardware, native CPU and Vulkan tests |
| macOS x86-64 | `.a` and `.dylib`, libc++, physical-core query, no exact affinity | `macos-15-intel` CI job |
| macOS ARM64 | ARM feature queries, `.a` and `.dylib`, libc++, no exact affinity | `macos-15` CI job |
| Windows ARM64 MSVC | Accepted with a matching SDK and toolchain | Not validated |

## OS extensions

`supports_thread_affinity` reports exact mask support. Linux exposes kernel CPU IDs and
dynamically sized masks, Windows exposes group and index identities, and macOS reports
`Error::Unsupported` for exact processor IDs while providing `sysctl` physical-core counts.
Native affinity tags and exact CPU binding have different semantics; the portable core works
independently of both.

Native precision options keep upstream behavior. The model runner compares actual intermediate
formats with C++ rather than requiring the x86 FP16 fallback on every architecture. Unsafe
standalone-layer tests select FP32 pipelines when supplying FP32 tensors directly; network
execution performs native layout and precision conversion.

## Vulkan

A Vulkan build needs an SDK built with `-DNCNN_VULKAN=ON` and a working driver. The Cargo feature
and the SDK setting must agree; both mismatch directions fail during the build. SDKs built by
`tools/build-ncnn-sdk.py` also carry the patches in `tools/patches/`, which the validated results
assume.

Static linking is the default; `NCNN_LINK_KIND=dylib` selects shared linking. Static Vulkan needs the explicit dependency closure, for example
`static=glslang,static=SPIRV` or the extended closure in the evidence history; the ncnn source tree
ships no glslang, so the SDK is built with system glslang or the SDK's exported targets. Missing,
delimiter-only or wrongly typed dependencies fail before linking.

Networks select a device before loading parameters and keep native per-layer CPU fallback, so
Vulkan selection is not an all-GPU guarantee. Evidence for Windows/NVIDIA, Debian WSL lavapipe and
the trained-model comparisons is recorded in the [evidence history](evidence-history.md) and
[model fixtures](model-fixtures.md#windows-vulkan-trained-model-evidence).

## Cross execution

A Linux ARM64 host needs `aarch64-linux-gnu-g++`, the GNU ARM64 sysroot, QEMU user emulation and
Rust's `aarch64-unknown-linux-gnu` target:

```sh
export NCNN_INSTALL_DIR=/absolute/path/to/arm64-ncnn-install
export CXX_aarch64_unknown_linux_gnu=aarch64-linux-gnu-g++
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUNNER='qemu-aarch64 -L /usr/aarch64-linux-gnu'
export BINDGEN_EXTRA_CLANG_ARGS='-idirafter /usr/aarch64-linux-gnu/include'
export QEMU_CPU=max
cargo test --workspace --target aarch64-unknown-linux-gnu
```

A relocated toolchain also supplies its compiler sysroot and host tool-library paths. QEMU
validates ABI, ownership, numeric behavior and instruction paths; its timing and `/proc` topology
reflect the emulator, so native ARM performance remains real-device work. The `ubuntu-24.04-arm`
CI job covers the native ARM64 build and test run. It also accepts a `workflow_dispatch` input
`arm_mha_probe`, which runs `invariants::attention_probe_reports_finiteness` and prints the values
for three attention configurations, so a target-specific attention difference names its
configuration instead of one failing case.

## Hosted CI coverage

GitHub-hosted runners cover these configurations without additional hardware:

| Configuration | Runner | What the job proves |
|---|---|---|
| Linux x86-64 CPU | `ubuntu-latest` | Format, Clippy, the workspace suite twice, both CPU examples, `cargo doc -D warnings`, generated model fixtures against a native reference, package verification |
| Windows x86-64 CPU | `windows-latest` | MSVC build, workspace tests, example, package verification |
| Linux ARM64 CPU | `ubuntu-24.04-arm` | Native workspace tests |
| macOS x86-64 CPU | `macos-15-intel` | libc++ build and workspace tests on Intel |
| macOS ARM64 CPU | `macos-15` | Apple Silicon build and workspace tests |
| Minimum Rust | `ubuntu-latest` | `cargo check --workspace --all-targets` at 1.85.1 |
| Semver | `ubuntu-latest` | `cargo semver-checks check-release` for both packages against the published release |
| Documentation links | `ubuntu-latest` | `lychee --offline --include-fragments` over the Markdown pages, which resolves local files and anchors without network access |
| Vulkan | `ubuntu-latest` | The whole conformance suite plus the Vulkan contracts against Mesa lavapipe with `NCNN_TEST_DEVICE=vulkan`, the Vulkan-gated Clippy pass, and both device examples |
| Coverage | `ubuntu-latest` | `cargo llvm-cov` line coverage of the wrapper, uploaded as an artifact |
| Parser fuzzing | `ubuntu-latest` nightly schedule | `cargo fuzz` on the text parameter loader and the bounded weight reader |
| API documentation | `ubuntu-latest` | `cargo doc --workspace --no-deps` with a built SDK, deployed to <https://edwardssss.github.io/ncnn.rust/> |

A `v*` tag starts `Release`, which publishes both packages in dependency order and creates the
GitHub release from the changelog section for that version. The workflow needs a
`CARGO_REGISTRY_TOKEN` secret; `CONTRIBUTING.md` records the manual steps that precede the tag.

The first hosted run of this matrix passed every job on 2026-11-09 for commit `f3bf72b`: Linux
x86-64, Linux ARM64, Windows, both macOS images, the minimum Rust version and software Vulkan.

Hosted runners expose no GPU, so the Vulkan job covers correctness and lifetime contracts on a
software device. Timing, memory-type selection and driver behavior on discrete hardware remain
local evidence. That job takes glslang and its CMake targets from `glslang-dev` plus `glslang-tools`,
which the hosted image provides.
Windows ARM64 uses the preview `windows-11-arm` label and stays outside this matrix.
