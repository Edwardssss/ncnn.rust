# Troubleshooting

Each section states the symptom first, then the cause and the fix.

## The loader cannot find the ncnn library

**Symptom:** a test or binary starts and fails with `ncnn.dll` or `libncnn.so` not found, or the
process exits before `main` prints anything.

**Cause:** a shared SDK was selected, but its library directory is not on the loader path. The build
script prints a warning that names the directory.

**Fix:** place that directory on the platform loader path before starting the process:

| Platform | Variable |
|---|---|
| Linux | `LD_LIBRARY_PATH=<prefix>/lib` |
| macOS | `DYLD_LIBRARY_PATH=<prefix>/lib` |
| Windows | append `<prefix>\bin` to `PATH` |

Alternatively set `NCNN_LINK_KIND=static` when the SDK ships a static library.

## The build stops with "Required ncnn version"

**Symptom:** the build script panics with `Required ncnn version: <pinned> (found <other>)`.

**Cause:** `NCNN_INSTALL_DIR` points at a different ncnn release than the pinned one. The bindings and
the adapter are generated against one release, so an untested release is refused.

**Fix:** install the pinned release, or set `NCNN_ALLOW_UNTESTED_VERSION=1` to try another release.
An untested release can change structures, layer parameters or C API signatures; the
[migration notes](migration.md) describe what to check.

## The build stops with "Static Vulkan requires explicit NCNN_EXTRA_LIBS"

**Symptom:** a static SDK with Vulkan enabled is rejected.

**Cause:** a static ncnn Vulkan build pulls glslang and the Vulkan loader into the link, and the crate
does not guess that closure.

**Fix:** pass the closure the SDK exports, for example:

```sh
NCNN_EXTRA_LIBS=glslang,SPIRV,glslang-default-resource-limits,MachineIndependent,GenericCodeGen,OSDependent
NCNN_EXTRA_LIB_DIRS=<prefix>/lib;/path/to/vulkan/lib
```

## `NCNN_LINK_KIND` contradicts the SDK

**Symptom:** the build script reports that a static library was requested but only a shared one is
installed, or the reverse.

**Fix:** omit `NCNN_LINK_KIND` and let the layout decide, or install the other layout. The inference
uses both `ncnn.lib`/`libncnn.a` and `ncnn.dll`/`libncnn.so` names.

## A layer case aborts, or the process dies without a message

**Symptom:** `cargo test` reports `layer case failures` with a case name, or a run stops silently.

**Cause:** a native abort ends the process of that case. The suite runs each case in a child process
precisely so the failure survives.

**Fix:** reproduce the case alone and read the native message:

```sh
NCNN_LAYER_CASE="CopyTo depth" cargo test -p ncnn-rust --test conformance uncovered -- --nocapture
```

A case that reports non-finite values prints the leading values. Platform-specific differences are
recorded in the [layer coverage](layer-coverage.md) page.

## Vulkan is unavailable

**Symptom:** `Device::enumerate` returns an empty list, or `Device::default_device` fails.

**Cause:** no ICD is installed or the loader cannot see it.

**Fix:** install the driver ICD, or use a software ICD such as Mesa lavapipe. `NCNN_TEST_DEVICE=vulkan`
makes the conformance suite repeat on the default device, so a device-free machine skips nothing
until that variable is set.

## A resident upload fails with `Vulkan upload returned -1`

**Symptom:** `Allocator::upload`, `import_buffer` or `Command::extract` returns
`Error::NativeAt("Vulkan upload", -1)` while ncnn writes `vkAllocateMemory failed` to stderr.

**Cause:** the device heap cannot serve the request. ncnn reports its own failure code rather than
the Vulkan result, so the message names the operation and leaves the cause on stderr.

**Fix:** release tensors that are no longer needed and retry; the allocator, the tensors that are
still alive and the network stay usable after a refusal. Long sessions that cannot attribute a
failure re-create the device and its allocator, because a lost device reports the same code.

## The Vulkan suite aborts on a physical GPU

**Symptom:** `cargo test --features vulkan` with `NCNN_TEST_DEVICE=vulkan` reports a group of cases
as failed and then aborts with an access violation (`0xC0000005`), while the same suite passes on a
software device and passes again on the same machine after a rerun.

**Cause:** the process ended before a wrapper call could return an error, so the abort is inside the
driver rather than in a status code the crate reports. The recorded campaign ruled out the two
wrapper-side causes: peak device memory over a passing run was 1.3 GB of a 6 GB device, of which the
suite itself held tens of megabytes, and four suites run concurrently on the same device passed. The
same abort reproduced on the commit before the change under measurement, so no wrapper path
introduced it.

**Fix:** rerun. A crashed run leaves the driver and its caches in an unknown state, and the next run
starts from a fresh process. A case that fails on its own is a device or driver limitation and
`--test-threads=1` separates it from an interaction. The conformance suite creates one device per
test binary instead of one per worker thread, which keeps a single native device path whatever the
thread count.

## Outputs come back as errors instead of values

**Symptom:** `infer` or `extract_owned` fails with a message about 32-bit storage.

**Cause:** the network returns quantized or half-precision storage, which those forms convert only
when the pipeline stores FP32.

**Fix:** read the stored width instead:

```rust
let outputs = net.infer_stored([("input", input)], &["output"])?;
let bytes = outputs[0].1.to_i8()?;
```

## Owned-output requests are far slower than the C API

**Symptom:** `infer`, `extract_owned` or `infer_stored` cost several times more than the same request
through the C API, while `infer_into`, `infer_borrowed_into` and `extract` are at native parity or
faster.

**Cause:** those forms return independent storage, so every request allocates an output-sized buffer
and frees it again. On Windows the default process heap is much slower than a thread-caching
allocator for that allocate-and-free cycle; on Linux glibc serves each output-sized buffer from a
fresh mapping and returns it to the kernel on free, so every request faults its pages again. The
copy itself is not the cost.

**Fix:** either reuse a buffer, which removes the allocation entirely, or keep the freed pages
mapped. On glibc Linux the crate applies the two settings itself:

```rust
ncnn_rust::heap::retain_pages();
```

The call is process-wide and lasts for the rest of the run, like a global allocator. A service that
cannot change its own process environment can use it, and a host that starts the process itself can
raise the same thresholds instead:

```sh
MALLOC_MMAP_THRESHOLD_=67108864 MALLOC_TRIM_THRESHOLD_=-1 ./my-inference-server
```

Windows needs neither: its heap serves the same two-buffer pattern from committed pages and faults
almost nothing, and it exposes no equivalent threshold, so `heap::retain_pages` reports `false`
there. A Windows application that still measures allocation-bound owned requests declares a drop-in
allocator:

```rust
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
```

The [performance page](performance.md#allocator-choice) records both figures, and the
[Linux section](performance.md#linux-hosts) records the fault counts behind the Linux behaviour. The
example in `crates/ncnn/examples/fast_allocator.rs` reproduces the Windows figures, and
`crates/ncnn/examples/heap_retention.rs` prints the Linux figures before and after the call. A
library cannot set the global allocator for its caller, so the allocator stays an application
decision; the heap settings are the one part the crate can apply itself.
