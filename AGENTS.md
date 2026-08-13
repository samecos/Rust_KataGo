# Rust_KataGo

KataGo 围棋引擎的 Rust 移植（基座：KataGo-Lite/katago-rs，约 10 万行，含与 C++ 对拍的 oracle 测试）
+ 针对本机 RTX 5070 Ti（Blackwell SM120）的 CUDA 推理后端。

## 构建与运行

- 构建（默认特性，dummy 后端）：`cargo build --workspace`
- TRT 后端：`cargo build -p katago --features trt`（需本机 CUDA + TensorRT 头文件）
- CUDA 后端（手写 kernel，nvcc 编译）：`cargo build -p katago --features cuda`；冒烟 `cargo test -p kata_nn --test test_cuda --features cuda`
- 测试：`cargo test --workspace`
- GTP 冒烟：`scripts/gtp_smoke.sh`；真实推理：`configs/gtp_trt.cfg` + `--model D:/code/b11fix.onnx`（首次构建 TRT 引擎较慢，plan 缓存在模型旁）
- 数值对拍：`cargo test -p kata_nn --test dump_nn_io --features trt` 生成转储 → `.venv/Scripts/python.exe scripts/compare_nn_output.py <dump目录>`（ORT FP32 黄金参考）

## 后端选择

- 配置键 `nnBackend`（每模型可用 `nnBackend{i}` 覆盖）：`dummybackend`（默认）/ `trtbackend` / `cudabackend` / `eigenbackend`（后两者未实现，选择即报错）
- `trtbackend` 需要 `katago` 的 `trt` feature（透传 `kata_nn/trt`），构建时探测本机 CUDA+TensorRT；未启用 feature 时选择 trtbackend 会得到明确的报错
- 命令行示例：`--override-config nnBackend=trtbackend`

## 目录

- `crates/kata_*`：引擎各层（core/game/data/search/nn/program/book/distributed）；`crates/katago` 为 CLI（bin `katago-rs`）
- `crates/kata_nn`：NN 抽象层与后端（`backends/`：dummy、trt；cuda/cpu 计划中）；`cuda-kernels/` 计划为手写 kernel 源码
- `cpp-shim/`：TensorRT C++ FFI shim
- `configs/`：`sm-targets.json`（SM 目标配置，待建）、`gtp_smoke.cfg`
- `scripts/`：冒烟与对拍脚本
- `docs/`：`需求汇总.md`（v1.0 已对齐）、实施计划、`upstream-notes/`（katago-rs 规划文档存档）

## 约定

- 仅支持 19 路标准棋盘
- 数值纪律：激活/归约一律 FP32 计算、half 存储边界与官方 KataGo 逐位对齐（正确性门前提，见 `docs/需求汇总.md` §7）
- 后端路线：手写 CUDA C++/PTX kernel 为主线（nvcc 编译 + cudarc 加载），TensorRT 兜底，plan JSON fail-closed
- 依赖真实模型的测试用 `KATAGO_TEST_MODEL_DIR` 环境变量定位模型，缺失时自动跳过
- 新后端实现需实现 `kata_nn::backend::Backend` trait 并接入 `kata_program::setup` 的 backend 选择
