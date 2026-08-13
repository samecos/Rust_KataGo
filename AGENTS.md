# Rust_KataGo

KataGo 围棋引擎的 Rust 移植（基座：KataGo-Lite/katago-rs，约 10 万行，含与 C++ 对拍的 oracle 测试）
+ 针对本机 RTX 5070 Ti（Blackwell SM120）的 CUDA 推理后端。

## 构建与运行

- 构建（默认特性，dummy 后端）：`cargo build --workspace`
- 全特性（含 `trt`，需本机 CUDA + TensorRT 头文件）：`cargo build --workspace --all-features`
- 测试：`cargo test --workspace`
- GTP 冒烟：`scripts/gtp_smoke.sh`（或手动 `cargo run --bin katago-rs -- gtp --config configs/gtp_smoke.cfg --model D:/code/b11fix.onnx`）

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
