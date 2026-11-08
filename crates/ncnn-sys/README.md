# ncnn-rust-sys

Raw C bindings and small C++ adapters for an externally installed ncnn 20261020
SDK. Cargo's native link identifier remains `ncnn`.

Set `NCNN_INSTALL_DIR` to an SDK containing `include/ncnn` and `lib` or `lib64`.
Select `NCNN_LINK_KIND=static` or `dylib`; additional native libraries are listed
in `NCNN_EXTRA_LIBS` in link order. Dynamic libraries are `libncnn.so` on Linux,
`libncnn.dylib` on macOS, and `ncnn.lib` plus `ncnn.dll` on Windows.

Required enabled native features: `NCNN_C_API`, `NCNN_STDIO`, `NCNN_STRING`,
`NCNN_PIXEL`, `NCNN_PIXEL_ROTATE`, `NCNN_PIXEL_AFFINE`, `NCNN_PIXEL_DRAWING`,
`NCNN_BATCH`, `NCNN_INT8`, `NCNN_BF16`, and `NCNN_VALIDATION`.
Default builds require `NCNN_VULKAN=OFF`. The `vulkan` feature requires `NCNN_VULKAN=ON` and supports both `NCNN_LINK_KIND=dylib` and `static`. Static Vulkan requires nonempty `NCNN_EXTRA_LIBS` matching the SDK's exported dependency closure, with library search directories supplied through `NCNN_EXTRA_LIB_DIRS` where needed. The verified Windows SIMPLEVK/OpenMP-off static closure is `static=SPIRV,static=glslang`; other SDK configurations provide their own dependencies. Both feature/SDK mismatch directions fail during the build. The workspace documentation records shared/static deployment and validation.

A target C++ compiler and host-loadable libclang generate and build the bindings.
Cross builds select the target compiler and linker, plus
`BINDGEN_EXTRA_CLANG_ARGS` for the target sysroot and system headers. The build
passes the Cargo target explicitly to Clang. Applications provide matching
architecture, compiler ABI, deployment target, CRT, and runtime libraries.
MSVC static CRT uses a matching `/MT` SDK and Rust `crt-static` target feature.

All raw calls retain their pointer, initialization, ownership and native state
contracts. `ncnn_rs_os_affinity_supported` reports exact-affinity availability.
Unsupported exact-affinity setters return `NCNN_RS_UNSUPPORTED` on macOS;
processor enumeration returns its documented failure sentinel. The high-level
`ncnn-rust` package exposes a typed unsupported result.

`tools/patches/` holds the patches the tests were validated against, and
`tools/build-ncnn-sdk.py` applies them to the source tree before configuring it.
`ncnn-20261020-cpu-siblings.patch` keeps Linux sibling-ID accounting for CPU
masks wider than 32 bits. Apply the patches while preparing SDKs for
large CPU-ID topologies. Native SDK preparation remains application-owned.

MSRV: Rust 1.85.1. License: BSD-3-Clause. Native ncnn retains its upstream license.
