[English](README.md) | **简体中文**

# ncnn.rust

为 [ncnn](https://github.com/Tencent/ncnn) 提供的 Rust 绑定，按资源安全的方式管理所有权。

仓库包含两个包：

| 包 | 职责 |
|---|---|
| `ncnn-rust` | 安全 API：网络、张量、图像、选项、GPU 资源 |
| `ncnn-rust-sys` | 生成的 C 绑定与项目自带的 C++ 适配层 |

## 特性

- 所有权转移式输入推理，输出可选最终向量或可复用缓冲区。
- 类型化张量访问：`ElementKind` 读写覆盖 FP16、BF16 与各整数位宽，`infer_stored` 保留层写出的元素宽度。
- 批量张量、`flatten` 与 `with_packing` 布局改写，以及调用方持有存储时免拷贝的消费式助手（`into_normalized`/`into_reshape`）。
- 零拷贝请求：`Mat::from_external`、`Extractor::input_borrowed` 与 `Net::infer_borrowed` 直接提交调用方内存。
- 计算图自省：`Net::layers` 与 `Net::blobs` 报告连线、形状以及 ncnn 为每种层类型声明的支持标志，`layer_type_index` 把层类型名解析为索引。
- 作用域内的只读 CPU 输出守卫，在不复制的前提下占用网络。
- 绑定到计算图的端点令牌，拒绝来自其他图或重新加载后的图的令牌。
- 张量、图像预处理、自定义层、读取器与分配器，包括就地归一化、像素输出、letterbox 与深度边距、仿射变换。
- 已加载网络上的生效选项回读，以及面向中间张量、工作区与注意力 KV 缓存的调用方自有池。
- 可选 Vulkan 后端：设备选择、常驻张量、分配器、命令录制、完成守卫、外部缓冲区导入与持久化流水线缓存。
- 显式的原生配置映射，不在 Rust 侧按硬件做门控。

## 安装

```sh
cargo add ncnn-rust
```

两个包只包含 Rust 代码。构建时会针对构建环境提供的 ncnn SDK 编译 C++ 适配层，因此下一节说明如何准备该 SDK。

## 构建

Cargo 使用外部安装的 ncnn SDK。`NCNN_INSTALL_DIR` 必填，且必须包含 `include/ncnn` 与 `lib`。测试用的最低 Rust 版本是 1.85.1。

```sh
export NCNN_INSTALL_DIR=/absolute/path/to/ncnn-install
export NCNN_LINK_KIND=dylib
cargo test --workspace
cargo run -p ncnn-rust --example add
```

必需的原生特性：`NCNN_C_API`、`NCNN_STDIO`、`NCNN_STRING`、`NCNN_PIXEL`、`NCNN_PIXEL_ROTATE`、`NCNN_PIXEL_AFFINE`、`NCNN_PIXEL_DRAWING`、`NCNN_BATCH`、`NCNN_INT8`、`NCNN_BF16`、`NCNN_VALIDATION`。默认构建要求 `NCNN_VULKAN=OFF`；Cargo 特性 `vulkan` 要求 `NCNN_VULKAN=ON` 的共享 SDK，并需要其依赖闭包。

`NCNN_LINK_KIND` 选择 `static`（默认）或 `dylib`。共享 SDK 还需要把动态库所在目录加入运行时查找路径：Unix 上通过 `LD_LIBRARY_PATH` 或 `DYLD_LIBRARY_PATH` 指向 `lib`，Windows 上通过 `PATH` 指向 CMake 安装 `ncnn.dll` 的 `bin` 目录——在 Windows 上静态库与导入库同名 `ncnn.lib`。`NCNN_EXTRA_LIBS` 按链接顺序列出额外库，可带 `static=` 或 `dylib=` 前缀。`NCNN_EXTRA_LIB_DIRS` 追加库搜索目录。SDK 版本固定为 20261020；`NCNN_ALLOW_UNTESTED_VERSION=1` 可针对其他版本做兼容性验证。

`tools/fetch-ncnn.py` 校验固定版本的源码归档，`tools/build-ncnn-sdk.py` 应用 `tools/patches/` 中的补丁并构建具备上述特性的 SDK。

## 用法

```rust
use ncnn_rust::{Mat, Net};

let net = Net::builder()?.threads(2)?.load_files("model.param", "model.bin")?;
let input = Mat::from_f32(&values, &[224, 224, 3])?;
let outputs = net.infer([("input", input)], &["output"])?;
for output in outputs {
    println!("{:?} {}", output.info, output.values.len());
}
```

重复推理时先解析端点，再复用同一个缓冲区：

```rust
let input_id = net.resolve_blob("input")?;
let output_id = net.resolve_blob("output")?;
let mut ex = net.extractor()?;
ex.input(&input_id, Mat::from_f32(&values, &[224, 224, 3])?)?;
let mut buffer = Vec::new();
ex.extract_into(&output_id, &mut buffer)?;
```

读取模型前挂载 Vulkan 设备即可在 GPU 上运行。ncnn 保留逐层回退到 CPU 的能力，因此选中设备并不保证每一层都在 GPU 上执行：

```rust
let device = ncnn_rust::gpu::Device::default_device()?;
let net = Net::builder()?.vulkan(device).load_files("model.param", "model.bin")?;
```

量化与半精度输出保持存储宽度。`infer` 会把结果转换为 FP32，因此读取它们要用存储形式并显式声明格式：

```rust
let outputs = net.infer_stored([("input", input)], &["output"])?;
let (info, stored) = &outputs[0];
println!("每个元素 {} 字节", info.elemsize);
let bytes: Vec<i8> = stored.to_i8()?;
```

`Mat::from_typed` 与 `Mat::from_i8` 为输入侧构建同样的存储，`Extractor::extract_stored` 在单个 extractor 生命周期内借用它。[类型化张量示例](crates/ncnn/examples/typed_tensors.rs)与[常驻 Vulkan 示例](crates/ncnn/examples/vulkan_resident.rs)可端到端运行。

extractor 可以安全复用：提交输入会重启本次请求，因此第二次推理会重新计算，而不是返回上一次缓存的结果。逐 extractor 的设置覆盖网络选项之外的部分，并在该重启后保留：

```rust
let mut ex = net.extractor()?;
ex.set_light_mode(false);
ex.set_kvcache_max_seqlen_hint(2048)?;
```

跨输入保留状态的图（例如逐 token 解码时的注意力缓存）在下一个输入前调用 `ex.continue_session()`，该输入便延续缓存状态而不重启请求。

张量的一部分可以直接读取而不复制。视图别名源存储：切片读取直接借用，`to_f32` 从中拷贝，而把视图作为输入提交时会复制，因为别名不持有自己的存储：

```rust
let plane = output.channel(0)?;
let values = plane.to_f32()?;
```

网络与 extractor 留在创建它们的线程上，因此服务端为每个工作线程准备一对；[多线程示例](crates/ncnn/examples/threads.rs)对比了它与"单个网络用满所有核心"两种形式。

## 性能

封装开销在同一进程内针对同一 SDK 的原生 C API 测量，因此表中反映的是绑定层而非推理引擎。每请求毫秒中位数，单线程，release 构建：

| 模型 | 设备 | native | ncnn-rust | 差异 |
|---|---|---:|---:|---:|
| NanoDet-m，6 个输出 | CPU | 23.011 | 23.379 | +1.6% |
| PP-OCRv5 识别 | CPU | 57.566 | 58.842 | +2.2% |
| NanoDet-m，6 个输出 | Vulkan | 4.489 | 4.702 | +4.7% |
| PP-OCRv5 识别 | Vulkan | 10.863 | 10.210 | 在波动范围内 |

`ncnn-rust` 一列解析端点一次并填充调用方缓冲区，预热后不再分配。若改为每请求返回独立向量，则多一次输出拷贝：NanoDet Vulkan 上由 4.702 ms 变为 5.294 ms（+13%），其六个输出共 235,200 个值；CPU 上的差异落在测量方法的 10% 波动范围内。

单层计算图在微秒尺度上呈现同样的情况，此时主导开销是请求路径而非算术本身：

| 用例 | 设备 | native | ncnn-rust |
|---|---|---:|---:|
| ReLU | CPU | 0.002 ms | 0.002 ms |
| AbsVal | CPU | 0.002 ms | 0.002 ms |
| ReLU | Vulkan | 0.193 ms | 0.197 ms |
| AbsVal | Vulkan | 0.183 ms | 0.221 ms |

预处理与张量生命周期，以 65536 个 FP32 元素计的微秒中位数。拷贝形式返回独立存储，因此 native 一侧同时给出就地与独立两种参照：

| 操作 | native 就地 | native 独立存储 | ncnn-rust |
|---|---:|---:|---:|
| 创建并拷贝 | 6.022 | 6.022 | 7.043 |
| RGB 转换 | 75.769 | 75.769 | 79.125 |
| 克隆并归一化 | 12.958 | 26.618 | 28.366 |
| reshape | 7.221 | 15.607 | 16.892 |

对会分配的这些行，中间一列才是参照：与就地形式的差距是一次完整张量拷贝，而非封装开销。仓库内基准在同样口径上给出 `normalize_in_place` 4.8–5.4 微秒、拷贝形式 25–39 微秒，以及 `into_reshape` 0.2–0.4 微秒、`reshape` 27 微秒。请求在引擎内部也会分配：`NetBuilder::reuse_blob_memory()` 为中间张量提供网络自有的池，这是两个平台上实测最有效的杠杆；复用形式（`infer_into`、`infer_borrowed_into`）则去掉调用方的向量分配。对一个 65536 元素的 ReLU 请求，池化网络把独立输出形式从 Windows 上的 332 微秒降到 147、从 WSL2 glibc 客户机的 227 微秒降到 23，借用缓冲区的形式则分别从 41 降到 19、从 23 降到 17。Linux 上的成本来自缺页，`ncnn_rust::heap::retain_pages()` 会保留调用方缓冲区的页面。

服务端靠每线程一套网络扩展。在 8 核 WSL2 客户机上，四个工作线程达到每秒 102k–114k 次请求，单个工作线程为 38k–41k，而用单个网络撑满所有核心是 40k–42k：在这个规模的图上，原生线程池与单线程单工作线程代价相同，独立工作线程则成倍提升吞吐。主机负载会让这些数字变动数倍，因此示例报告三轮的中位数。测量方法、完整表格、池分配、线程与 Linux 测量结果以及复现命令见[性能文档](docs/performance.md)。

## 文档

API 文档由 `.github/workflows/docs.yml` 在每次推送到默认分支时生成，发布在 <https://edwardssss.github.io/ncnn.rust/ncnn_rust/>。docs.rs 无法构建本工作区：其沙箱没有 ncnn SDK，而构建脚本需要针对已安装的头文件编译 C++ 适配层，缺少 `NCNN_INSTALL_DIR` 会在 rustdoc 启动前失败。

- [文档索引](docs/README.md)
- [安全契约](docs/contracts.md)
- [原生 ABI 与适配层决策](docs/native-abi.md)
- [层覆盖](docs/layer-coverage.md)
- [支持的平台](docs/support-matrix.md)
- [故障排查](docs/troubleshooting.md)
- [迁移说明](docs/migration.md)
- [证据历史](docs/evidence-history.md)
- [性能](docs/performance.md)
- [高级 CPU 适配层](docs/advanced-cpu.md)
- [模型夹具](docs/model-fixtures.md)
- [变更日志](CHANGELOG.md)
- [安全策略](SECURITY.md)
- [参与贡献](CONTRIBUTING.md)

## 许可

BSD-3-Clause。