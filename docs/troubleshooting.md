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

**Cause:** the request allocates in two places. The value-vector forms allocate the output vector and
free it again on top of the native clone that `Extractor::extract` performs when the result comes
from the network's blob pool, and the native network allocates its intermediate tensors per request.
On Windows the default process heap is much slower than a thread-caching allocator for that cycle; on
Linux glibc serves output-sized requests from a fresh mapping and returns it to the kernel on free,
so every request faults the pages again. The copy itself is not the cost.

**Fix:** ask for an owned tensor instead of an owned vector. `extract_mat`, `extract_stored_owned`
and `OutputGuard::into_mat` keep the buffer the extractor already detached, so they allocate no data
buffer at all.

```rust
let mut extractor = net.extractor()?;
extractor.input("input", input)?;
let output = extractor.extract_mat("output")?;
```

Where a request needs a value vector, reuse storage and give the network a pool.

```rust
let net = Net::builder()?.reuse_blob_memory()?.load_files("model.param", "model.bin")?;
let mut outputs = Vec::new();
net.infer_into(inputs, &["out"], &mut outputs)?;
```

`reuse_blob_memory` attaches the pool allocators that serve the network's intermediate tensors,
which is the half no process allocator setting reaches; `infer_into` and `infer_borrowed_into`
remove the output vector allocation. The `pooled-request` mode of `api_benchmark` times both halves
on one host. A caller that keeps the value-vector form still pays one allocation per request, and the
settings below address it. On glibc Linux the crate applies those settings itself:

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

## A reused extractor answers with the previous request

**Symptom:** the second `extract` on one handle returns the values the first request produced, even
though a different input was submitted.

**Cause:** the native extractor caches the blob it produced and recomputes a layer only while that
blob is empty, so a handle that already extracted answers from the cache. Native ncnn offers
`Extractor::clear` for this, but that call empties the blob table and leaves the handle unable to
accept input.

**Fix:** nothing, when the handle comes from this crate: `input` and `input_borrowed` restart the
handle first, which recreates the native extractor and re-applies the per-extractor settings. A
handle built directly from `ncnn_rust_sys` recreates its extractor the same way instead of calling
`clear`. The restart costs one native extractor construction per request, which `api_benchmark`
shows inside the request-path figures.

A graph that carries state across inputs, such as an attention layer with a key/value cache, asks for
the opposite: `Extractor::continue_session` keeps the cached state for the next input, and the
following input starts a new request again.

## A graph whose input feeds two layers returns wrong values in light mode

**Symptom:** enabling light mode on a graph with a residual or skip connection fails with a message
about a blob that feeds more than one layer.

**Cause:** light mode releases a bottom blob after its first consumer ran and refills it from its
producer, and the `Input` layer cannot refill a blob once its data was released. A blob that feeds
two layers, or an input consumed twice, therefore reaches its second consumer empty, which produced
wrong values before this was checked.

**Fix:** keep light mode off for such graphs, which is the default, or insert an explicit `Split`
layer so each consumer reads its own blob:

```rust
for name in net.shared_blobs()? {
    println!("{name} feeds more than one layer");
}
let mut extractor = net.extractor()?;
extractor.set_light_mode(true)?;
```

`Extractor::set_light_mode` and `Net::extractor` return `Error::Invalid` while
`Net::shared_blobs` lists a blob, so the request never reaches the native release. The
per-extractor call leaves the network option unchanged, so a second extractor still starts from the
option value the network was loaded with. `Extractor::light_mode` reports the value a handle uses,
and `Extractor::input_borrowed` rejects a borrowed input while that value is on, because the native
forward path reads the storage refcount an external tensor does not carry.