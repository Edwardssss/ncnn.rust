# Model regression fixtures

The [fresh trained-model report](performance.md) contains matched NanoDet/OCR CPU/Vulkan medians, two-thread precision profiles, native parity, resident GPU diagnostics and separately instrumented shader traces. FP16 preferences are not guaranteed storage width or trained-model accuracy.

Model checks reuse the compact suite's external installation environment. Set `NCNN_INSTALL_DIR`, `NCNN_LINK_KIND`, and the runtime loader path as described in the [build instructions](../README.md#build). The installed SDK includes ncnn's CMake package exports. Build a standalone C++ reference against that same installation:

```sh
python3 tools/fetch-models.py
cmake -S crates/ncnn/tests/native -B target/cpu-reference -DCMAKE_BUILD_TYPE=Release
cmake --build target/cpu-reference --config Release --parallel 2
cargo run -p ncnn-rust --example model_regression -- target/model-fixtures ./target/cpu-reference/ncnn_cpu_reference
```

For the generated INT8/operator fixtures alone, omit the model download and use:

```sh
cargo run -p ncnn-rust --example model_regression -- target/cpu-sdk-fixtures ./target/cpu-reference/ncnn_cpu_reference --compact
```

The runner creates this fixture directory and generates 25 compact cases locally.
The default invocation also runs the 17 pinned trained-model cases, for 42 cases
total. On the guard-enabled cases, the second Rust extraction checks the same
native outputs through guards; it is a second wrapper path, not an additional
independent native comparison.

MSVC places the reference at `target/cpu-reference/Release/ncnn_cpu_reference.exe`; pass that path as the final argument. The reference links directly to the installed ncnn C++ library and reads model files independently of the Rust adapters.

The script pins source revisions and SHA-256 hashes. Assets stay in ignored `target/model-fixtures`; the repository distributes the fetch manifest and original test code. Total downloaded assets are approximately 10 MB.

| Fixture | Conversion source | Model project |
|---|---|---|
| NanoDet-m | nihui/ncnn-android-nanodet, commit `321276bf9fd465896dd685d64ab0bb08dd68179b` | [NanoDet](https://github.com/RangiLyu/nanodet), Apache-2.0 project |
| PP-OCRv5 mobile recognition | nihui/ncnn-android-ppocrv5, commit `458269b48a434a4301070f83f7ce90e8fd11180f` | [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), Apache-2.0 project |

Conversion repositories supply the actual asset files. Their repository license endpoints did not provide a separate asset license during this session; downstream redistributors review model terms separately. This workspace keeps those files outside its published source.

Rust and standalone C++ executions generate the same nonzero FP32 input, with logical element `i` equal to `(((i * 17 + 3) % 251) - 125) / 64`. The C++ process reports all 18 native option fields for comparison with Rust's typed getters. Each profile also probes a convolution intermediate in native layout, making actual 16-bit versus 32-bit storage observable.

| Profile | Configuration |
|---|---|
| 0 | FP32, one thread without packing and two threads with packing |
| 1 | BF16 storage; the intermediate contains 16-bit elements |
| 2 | FP16 packing/storage/arithmetic requested; observed storage follows native architecture/kernel support (32-bit on tested x86, 16-bit under tested ARM64 emulation) |
| 3 | Direct algorithm settings, local pooling off, mapped path loading, blocktime/denormal mode zero, and KV-cache hint 32 |
| 4 | Light mode on the OCR graph with explicit Split nodes |
| 5 | Individual Winograd 23/43/63 settings with SGEMM off and denormal modes 1/2/3 |

There are eight NanoDet cases and nine OCR cases per installation. Hint fields are compared independently even when the selected graph does not consume a hint. Profile 3 uses `Net::load_model_file_native` and C++ native path loading with stable model files. The standalone C++ binary uses `/MT` when checking a static-CRT SDK.

Each comparison uses identical native options on both sides; cross-precision accuracy is a separate model-quality measurement. Every output name, dimension, batch count, element size, packing, channel stride, and logical FP32 value is compared. Values use absolute tolerance `1e-5` plus relative tolerance `1e-4`. The first Rust extractor uses detached outputs; the second transfers owned input and sequentially consumes guarded outputs, checking exact equality with the detached run. Profile 3 checks guard rejection and retains detached extraction because native path loading disables guards.

The runner also generates a two-convolution INT8 fixture in the fixture directory. Its signed quantized weights and nonzero input exercise requantization to observable 8-bit intermediate storage followed by dequantization to FP32 output. Five configurations reuse the same profile loop: FP32 packed/unpacked, BF16 requested, FP16 requested, and light mode. All final guarded layouts and values are compared with standalone C++ and detached extraction. BF16/FP16 requests preserve the native INT8 route in this graph; trained-model profiles separately observe 16-bit conversion or architecture-dependent FP16 fallback.

A generated six-layer graph splits a reshaped input into three independent outputs: Gemm with a constant matrix and broadcast bias, affine LayerNorm, and two-head self-attention with four nonconstant projection matrices and biases. Five shared configurations cover unpacked/packed FP32, BF16, FP16 request, and light mode. Each network processes shapes `8x2x3`, `8x3x3`, and `8x1x3` through fresh extractors without reloading its model. A fourth run reloads parameters and changed weights into the same network, compares all outputs with C++, and verifies a retained detached output still contains its original values. This adds 20 graph/profile/shape cases; the three operators are observed in each case, rather than counted as independent benchmark runs.

Native Gemm intermediate storage is compared separately from final type-0 output conversion. Final guarded channel slices, contiguous access or padding rejection, copied values, and layouts agree with detached extraction. Each generated output contains finite, nonconstant, nonzero data. These checks measure inference behavior and make no timing claims.

Dynamic shapes follow model constraints: this fixture keeps embedding width eight while varying sequence length. A fresh extractor provides a fresh native inference cache; replacing an input in an already-executed extractor preserves downstream cached blobs. `load_model` accepts weights once per parameter load. To replace weights, drop extractors, load parameters again, then load weights; parameter loading clears the previous graph. `clear` releases graph/model state and enables configuration before another load. Detached outputs survive reload and destruction; guards exclusively borrow the network's sole live extractor and reserve the whole network against safe mutation or new extractors. Forgotten guards retain the reservation, and forgotten extractors conservatively prevent future guards.

The historical native x86 runtime matrix contains Linux shared/simpleomp, Linux static/GCC OpenMP, Windows shared/dynamic CRT, and Windows static/static CRT; its 68 model/profile/runtime comparisons produced maximum absolute error zero. Linux ARM64 additionally passes all 17 profiles through QEMU with an independently cross-built C++ reference, also with zero observed error. Emulated timing and host-backed topology are recorded separately from real ARM hardware evidence. Windows loads the pinned fixtures from a checkout path containing Unicode and spaces. NanoDet produces six outputs totaling 235,200 finite values per run. PP-OCRv5 recognition produces one output totaling 735,400 finite values per run. The OCR asset emits upstream deprecation notices for legacy 1D Convolution compatibility; execution succeeds on the pinned release.

These checks exercise trained model loading and repeatable inference. Synthetic DetectionOutput and LSTM fixtures in the compact suite provide small expected-value regression cases. Application-level preprocessing, box decoding, dictionary selection, CTC decoding, and quality metrics remain downstream concerns.

## Windows Vulkan trained-model evidence

`--vulkan` reuses the same native reference, pinned assets, deterministic input, and comparison code. It selects two precision profiles for each trained model, not the CPU generated-operator matrix. `--compact` and `--vulkan` select different suites. Ordinary GPU extraction returns detached CPU outputs. The resident command path also returns completed GPU tensors with read-only guards; CPU extractor views remain rejected.

The following commands completed in an x64 MSVC developer PowerShell in the repository root. `LIBCLANG_PATH` points to `D:\LLVM\bin`. The separately installed shared/static SDKs use ncnn 20261020, glslang 15.4.0, SIMPLEVK, runtime CPU dispatch, and OpenMP off. SDK configuration includes `-DNCNN_VERSION=20261020 -DNCNN_VULKAN=ON -DNCNN_SIMPLEVK=ON -DNCNN_OPENMP=OFF -DNCNN_RUNTIME_CPU=ON`; static uses `-DNCNN_SHARED_LIB=OFF` and shared uses `ON`. Each installed SDK includes CMake exports and its shader compiler dependencies. Reconfiguring an existing SDK also rebuilds its native objects; a stale version object in the local static build was rebuilt and reinstalled before the final passing run.

```powershell
$env:NCNN_INSTALL_DIR = (Resolve-Path target\sdk-vulkan-static).Path
$env:NCNN_LINK_KIND = 'static'
$env:NCNN_EXTRA_LIBS = 'static=SPIRV,static=glslang'
$env:NCNN_EXTRA_LIB_DIRS = "$env:NCNN_INSTALL_DIR\lib"
cmake -S crates\ncnn\tests\native -B target\reference-vulkan-static -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build target\reference-vulkan-static
cargo test --workspace --features vulkan
cargo run -p ncnn-rust --features vulkan --example model_regression -- target\model-fixtures target\reference-vulkan-static\ncnn_cpu_reference.exe --vulkan
cargo clippy --workspace --all-targets --features vulkan -- -D warnings
cargo fmt --all -- --check
dumpbin /dependents target\debug\examples\model_regression.exe target\reference-vulkan-static\ncnn_cpu_reference.exe

$env:NCNN_INSTALL_DIR = (Resolve-Path target\sdk-vulkan).Path
$env:NCNN_LINK_KIND = 'dylib'
Remove-Item Env:NCNN_EXTRA_LIBS,Env:NCNN_EXTRA_LIB_DIRS
$env:PATH = "$env:NCNN_INSTALL_DIR\bin;$env:PATH"
cmake -S crates\ncnn\tests\native -B target\reference-vulkan -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build target\reference-vulkan
cargo test --workspace --features vulkan
cargo run -p ncnn-rust --features vulkan --example model_regression -- target\model-fixtures target\reference-vulkan\ncnn_cpu_reference.exe --vulkan
```

Both linkages pass workspace tests (two integration contracts and ten compile-fail documentation tests), four trained comparisons, and normal process exit. Static clippy and formatting checks pass. Both static executables import no `ncnn.dll`; `/MD` CRT/system dependencies remain. The four comparisons per linkage are:

| Model | Profile / threads / packing | Submitted native VkMat bits | Outputs / values | Maximum absolute error |
|---|---|---|---|---|
| NanoDet-m | FP32 / 1 / off | 32 | 6 / 235,200 | 0 |
| NanoDet-m | FP16 / 2 / on | 16 | 6 / 235,200 | 0 |
| PP-OCRv5 recognition | FP32 / 1 / off | 32 | 1 / 735,400 | 0 |
| PP-OCRv5 recognition | FP16 / 2 / on | 16 | 1 / 735,400 | 0 |

All final outputs are downloaded FP32 Mats and repeat exactly across fresh Rust extractors. Zero maximum absolute error compares native C++ with Rust under identical options; it does not compare CPU with GPU or FP32 with FP16. Each native reference submits and waits for a nonempty GPU intermediate; this establishes actual GPU work, not GPU placement of every layer. OCR's upstream deprecated Convolution1d notices do not prevent successful inference. CPU workspace tests (one integration contract and ten compile-fail documentation tests) and the existing full 42-case and compact 25-case suites also pass with the separate CPU SDK after enabling runtime CPU dispatch. The compact suite repeats the 25 generated cases included in the full suite, rather than adding distinct comparisons. Their logs are `target\cpu-completion-tests.log`, `target\cpu-completion-regression.log`, and `target\cpu-completion-compact.log`; CPU clippy and formatting evidence is in `target\cpu-clippy.log` and `target\completion-fmt.log`.

Hardware and software versions are recorded in the [support matrix](support-matrix.md#vulkan) and the [evidence history](evidence-history.md). Local ignored evidence resides in `target\trained-vulkan.log`, `target\trained-vulkan-static.log`, `target\shared-vulkan-tests.log`, `target\static-tests.log`, `target\static-dependents.log`, and `target\vulkan-clippy.log`. Build rejection logs cover both feature/SDK mismatch directions, absent or delimiter-only static extras, invalid dependency kinds, invalid library names, and nonexistent dependency directories, each exiting 101 for the intended configuration error. The last two checks use `cargo check -p ncnn-rust-sys --features vulkan` with `NCNN_EXTRA_LIBS=static=bad/name` or `NCNN_EXTRA_LIB_DIRS` pointing to a nonexistent SDK subdirectory, respectively; their logs are `target\reject-static-name.log` and `target\reject-static-directory.log`. Additional resident, numerical and timing results appear below and in the performance document. Other physical GPU platforms and labeled model task metrics remain unverified.

## Resident and cross-profile evaluation

The existing Vulkan model harness also exercises upload, recorded resident extraction, completion and download for every model/profile. `target\resident-model-evaluation.log` records the Windows shared execution. The command is:

```powershell
cargo run -q -p ncnn-rust --features vulkan --example model_regression -- target\model-fixtures target\reference-vulkan\ncnn_cpu_reference.exe --vulkan
```

| Model | FP16 versus FP32 max absolute | Max relative (denominator floor 1e-6) | RMSE |
| --- | ---: | ---: | ---: |
| NanoDet-m | 0.043503761291503906 | 31.109179528838343 | 0.001197550283310651 |
| PP-OCRv5 recognition | 0.0405268669128418 | 0.19694348913781787 | 0.00015239125209531423 |

Same-profile native C++ versus Rust and resident versus ordinary output differences are zero, and repeated Rust inference is exact in this run. Cross-profile differences are nonzero. Profiles also differ in packing/thread options, so this is not a precision-only experiment. Inputs are deterministic synthetic tensors applied to trained models. No labeled detection or text-recognition dataset is present: mAP, CER, WER and application acceptance accuracy are not established. Synchronized end-to-end/resident latency and scoped logical storage measurements appear in the performance document.
