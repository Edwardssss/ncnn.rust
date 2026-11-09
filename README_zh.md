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
- 作用域内的只读 CPU 输出守卫，在不复制的前提下占用网络。
- 绑定到计算图的端点令牌，拒绝来自其他图或重新加载后的图的令牌。
- 张量、图像预处理、自定义层、读取器与分配器。
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

## 性能

封装开销在同一进程内针对同一 SDK 的原生 C API 测量，因此表中反映的是绑定层而非推理引擎。每请求毫秒中位数，单线程：

| 模型 | 设备 | native | ncnn-rust | 差异 |
|---|---|---:|---:|---:|
| NanoDet-m，6 个输出 | CPU | 23.011 | 23.379 | +1.6% |
| PP-OCRv5 识别 | CPU | 57.566 | 58.842 | +2.2% |
| NanoDet-m，6 个输出 | Vulkan | 4.489 | 4.702 | +4.7% |
| PP-OCRv5 识别 | Vulkan | 10.863 | 10.210 | 在波动范围内 |

`ncnn-rust` 一列解析端点一次并复用调用方缓冲区，预热后不再分配。若每次请求返回独立向量，代价是一次输出拷贝：NanoDet 的六个输出共 235,200 个值，在 Vulkan 上增加 18%。单层计算图的耗时与原生一致（CPU 上 ReLU 与 AbsVal 为 0.002 ms）。方法与完整表格见[性能文档](docs/performance.md)。

## 验证

CI 在 Linux x86-64、Linux ARM64、Windows、macOS x86-64 与 macOS Apple Silicon 上从源码构建 ncnn，然后运行格式化检查、Clippy、工作区测试、示例、两个包的发布试运行，以及一个依赖打包归档的独立 crate 的构建。另有独立作业检查声明的最低 Rust 版本，并在 Mesa lavapipe 软件设备上运行 Vulkan 契约；契约覆盖设备选择、分配器记账、常驻推理、守卫描述符、外部缓冲区导入、流水线缓存与跨设备拒绝。每月运行的版本矩阵读取 `ci/ncnn-versions.json`，构建固定版本、最新版本与上一个版本，对照生成的绑定与库符号审计启用的 C API，并用原生 C++ 参考实现比对训练模型的输出。`cargo test` 覆盖仓库内的符合性测试集：算子家族与 Rust 参考循环逐元素比对；SDK 提供的 108 种 layer 类型各有用例，其中 45 种执行到前向；算子家族覆盖不到的公开 API 路径；以及零分配的稳态检查。全部无需外部模型文件。

## 文档

API 文档由 `.github/workflows/docs.yml` 在每次推送到默认分支时生成，发布在 <https://edwardssss.github.io/ncnn.rust/ncnn_rust/>。docs.rs 无法构建本工作区：其沙箱没有 ncnn SDK，而构建脚本需要针对已安装的头文件编译 C++ 适配层，缺少 `NCNN_INSTALL_DIR` 会在 rustdoc 启动前失败。

- [文档索引](docs/README.md)
- [安全契约与 API 覆盖](docs/contracts.md)
- [支持的平台](docs/support-matrix.md)
- [证据历史](docs/evidence-history.md)
- [性能](docs/performance.md)
- [高级 CPU 适配层](docs/advanced-cpu.md)
- [模型夹具](docs/model-fixtures.md)
- [参与贡献](CONTRIBUTING.md)

## 许可

BSD-3-Clause。