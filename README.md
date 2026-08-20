# Rust_KataGo

KataGo 围棋引擎的 Rust 移植，基于 KataGo-Lite/`katago-rs`，并包含面向 NVIDIA CUDA 的手写推理后端。项目当前的主要使用方式是运行 `katago-rs`，通过 GTP 接入 Sabaki、Lizzie、KaTrain 等 GUI，或通过 JSON-lines 分析协议供脚本和服务调用。

项目的实现边界很明确：棋盘、规则、搜索、GTP、分析协议和模型推理链路已经在 workspace 中组织完成；CUDA/TensorRT 是可选后端，默认构建不依赖 GPU，方便先跑协议和回归测试。

> 当前推理执行器针对 KataGo 导出的 19 路 b11 Transformer 模型开发。CUDA 后端要求 19x19 输入和 NCHW 布局；模型文件不随仓库提交。

## 能做什么

- 19 路围棋棋盘、提子、打劫、superko、计分、贴目、让子和 encore 规则。
- 标准 GTP 指令，以及 KataGo/KGS/Lizzie 分析扩展（包括 `kata-analyze`、`lz-analyze`、`kata-raw-nn` 等）。
- `analysis` JSON-lines 引擎，支持 `query_version`、`query_models`、`analyze`、`clear_cache`、`terminate`、`terminate_all` 等动作。
- 多线程 MCTS、NN 结果缓存、批处理和搜索时间控制。
- `dummybackend`、TensorRT 和手写 CUDA 后端；后两者按 Cargo feature 编译。
- benchmark、`nnbench`、CUDA 设备指纹、SGF 评估、self-play/match 等 CLI 入口。

## 快速开始：先跑通 GTP

### 依赖

- Rust stable，`rustc >= 1.85`（edition 2024）。
- 默认构建只需要 Rust 工具链。
- 真实推理需要一个兼容的 KataGo ONNX 模型；CUDA 还需要 CUDA Toolkit/nvcc，TensorRT 还需要 TensorRT 头文件和库。

### 构建与测试

```bash
cargo build --workspace
cargo test --workspace
```

默认后端是 `dummybackend`。它用于验证棋盘、搜索、GTP 和测试链路，不产生真实神经网络评估。

### dummy GTP 冒烟

Windows PowerShell：

```powershell
# /dev/null 是 katago-rs 识别的 dummy 模型 sentinel，在 Windows 也照写。
"protocol_version`nname`nboardsize 19`nclear_board`nkomi 7.5`ngenmove B`nquit`n" |
  .\target\debug\katago-rs.exe gtp `
    --config .\configs\gtp_smoke.cfg `
    --model /dev/null
```

Linux/WSL：

```bash
cargo build --workspace
KATAGO_MODEL=/dev/null scripts/gtp_smoke.sh
```

GTP 是标准输入/输出协议：GUI 的引擎命令直接指向 `katago-rs`，参数使用 `gtp --config ... --model ...`。

## 真实模型推理

仓库不携带模型。请准备一个 KataGo 导出的 ONNX 模型，并把路径替换为本机路径。当前 CUDA 执行器的验证模型是 modelVersion 15 的 19 路 b11 Transformer：c768 trunk、12 attention heads、384 bottleneck；输入为 `input_spatial [B,22,19,19]` 和 `input_global [B,19]`，输出为 policy/value/misc/ownership 五组张量。`b11fix.onnx` 是开发时使用的实例；任意 ONNX 或其它 KataGo 网络结构并不保证兼容。

### CUDA 后端（主线）

CUDA feature 会在构建时使用 nvcc 编译 `crates/kata_nn/cuda-kernels/*.cu`，目标架构来自 [`configs/sm-targets.json`](configs/sm-targets.json)。

```powershell
cargo build -p katago --features cuda --release

.\target\release\katago-rs.exe gtp `
  --config .\configs\gtp_smoke.cfg `
  --model D:\models\b11fix.onnx `
  --override-config nnBackend=cudabackend,nnMaxBatchSize=16
```

Linux/WSL 使用相同参数，将路径写成 `./target/release/katago-rs` 和实际的 Unix 模型路径。

配置至少需要：

```ini
nnBackend = cudabackend
nnMaxBatchSize = 16
```

本机调优后可额外指定认证 plan：

```ini
cudaTacticPlan = /path/to/plans/best-tactic-plan.json
```

plan 会校验 GPU 指纹、模型 SHA-256 和 tactic 组合；不匹配时故意启动失败，而不是静默使用错误配置。更换 GPU 或模型后重新运行：

```bash
python scripts/autotune.py --threads both
```

仓库中的 [`configs/gtp_cuda.cfg`](configs/gtp_cuda.cfg) 和 `plans/best-tactic-plan.json` 是当前 Windows RTX 5070 Ti/b11fix 的 schema-2 认证配置（q64 attention + DualFFN），前者包含该 checkout 的绝对 plan 路径。WSL 使用对应的 `plans/best-tactic-plan-sm120-q64-wsl.json`；Windows/WSL 的 CUDA build fingerprint 不同，不能交叉复用。直接复用前先修改或删除 `cudaTacticPlan`。`scripts/autotune.py` 目前同样在文件顶部指定模型路径，运行前需要改为目标模型。

不指定 `cudaTacticPlan` 时不会自动运行 autotune，而是使用代码内置的默认 tactic（当前 attention 默认仍为 q128）；生产 q64 必须使用与本机 build fingerprint 匹配的认证 plan。

CUDA GTP 初始化默认会对 B1 和 `nnMaxBatchSize` 各执行一次真实 graph warmup，以消除
首请求的 lazy graph 编译延迟。若需测量冷启动，可临时加入 `cudaDisableWarmup = true`；
warmup 失败会在 server ready 前显式报错。

### TensorRT 后端（兜底）

```powershell
cargo build -p katago --features trt --release

.\target\release\katago-rs.exe gtp `
  --config .\configs\gtp_trt.cfg `
  --model D:\models\b11fix.onnx
```

首次启动会构建 TensorRT engine，并在模型旁缓存 TensorRT engine plan（不要与 CUDA tactic plan 混淆）。构建失败通常表示 CUDA/TensorRT 头文件、库、运行时驱动或 MSVC host compiler 不在工具链搜索路径中。TensorRT 路径也以当前 19x19 KataGo b11 模型为验证范围；并非所有 ONNX 模型或棋盘尺寸都兼容。

### 后端选择

配置键是 `nnBackend`，也可以用命令行覆盖：

| 值 | 需要的 feature | 用途 |
| --- | --- | --- |
| `dummybackend` | 无 | 协议、棋盘和测试 smoke；不做真实 NN 推理 |
| `cudabackend` | `--features cuda` | 手写 CUDA kernel，当前仅 19x19/NCHW |
| `trtbackend` | `--features trt` | TensorRT ONNX 推理兜底 |
| `eigenbackend` | 当前未实现 | 选择后会报错 |

例如：

```bash
katago-rs gtp --config configs/gtp_smoke.cfg --model /path/to/model.onnx \
  --override-config nnBackend=cudabackend
```

`--override-config` 接受逗号分隔的 `key=value` 列表；命令中的 `katago-rs` 假设二进制已加入 `PATH`，否则使用 `target/debug/katago-rs[.exe]` 或 `target/release/katago-rs[.exe]`。

## JSON-lines 分析协议

`analysis` 从 stdin 读取一行一个 JSON 请求，并向 stdout 写回 JSON 行：

```powershell
@(
  '{"id":"v","action":"query_version"}'
  '{"id":"p1","boardXSize":19,"boardYSize":19,"rules":"chinese","moves":[["b","d4"]],"maxVisits":5}'
  '{"id":"q","action":"terminate_all"}'
) | .\target\debug\katago-rs.exe analysis `
      --config .\configs\gtp_smoke.cfg `
      --model /dev/null `
      --analysis-threads 2
```

分析引擎保持 stdin/stdout 打开；上例发送版本查询和分析请求，再用 `action=terminate_all` 结束。Linux/WSL 可用 `printf '%s\n' ... | ./target/debug/katago-rs analysis ...` 提交相同的逐行 JSON。

最小请求示例：

```json
{"id":"p1","boardXSize":19,"boardYSize":19,"rules":"chinese","moves":[["b","d4"],["w","q16"]],"maxVisits":20}
```

真实分析时把 `/dev/null` 换成模型，并在配置中选择 `cudabackend` 或 `trtbackend`。请求字段和输出字段以 `crates/katago/src/cmd/analysis.rs` 为准；常用输出包括 `moveInfos`、胜率/分数、PV、ownership 和 policy。

## 性能与正确性工具

固定线程做搜索 benchmark：

```bash
./target/release/katago-rs benchmark \
  --config configs/gtp_benchmark.cfg \
  --model /path/to/b11fix.onnx \
  -v 800 -n 1 -t 1,4,8
```

不要省略 `-t`：省略时 benchmark 会自动调优，并可能为每个线程档位重新加载模型。

CUDA NN 吞吐 benchmark：

```bash
./target/release/katago-rs nnbench \
  --model /path/to/b11fix.onnx \
  --override-config nnBackend=cudabackend \
  --mode eval --batch 1,2,4,8,12,16,24,32 --iterations 400
```

查看 CUDA 设备和模型指纹：

```bash
./target/release/katago-rs cuda-fingerprint --model /path/to/b11fix.onnx
```

CUDA smoke 和整图对拍：

```powershell
py -m venv .venv
.venv\Scripts\python.exe -m pip install numpy onnxruntime
$env:KATAGO_ONNX_MODEL = "D:\models\b11fix.onnx"
cargo test -p kata_nn --test test_cuda --features cuda
cargo test -p kata_nn --test dump_nn_io_cuda --features cuda --release
.venv\Scripts\python.exe scripts\compare_nn_output.py `
  crates\kata_nn\target\nn_io_dump_cuda
```

对拍前需要 Python 3、`numpy` 和 `onnxruntime`（例如在仓库的 `.venv` 中安装）；测试会生成 `crates/kata_nn/target/nn_io_dump_cuda` 下的 `meta.json` 和每个位置的输入/输出二进制。脚本以 ONNX Runtime FP32 为参考，门限为 policy `5e-2`、value `2.5e-2`、misc `1e-2`、ownership `5e-3`，并要求 policy top-1 100% 一致，最后输出 `RESULT: PASS`。优化 CUDA kernel 或 tactic 后，应先通过对拍，再比较 benchmark。

## 仓库结构

```text
crates/
  kata_core          配置、线程、数学、日志和通用基础设施
  kata_game          棋盘、规则、历史、SGF 相关游戏逻辑
  kata_data          SGF、训练数据和模型文件辅助
  kata_nn            NN 输入/输出、ONNX 解析、Backend trait 和后端实现
  kata_search        MCTS、分析输出、时间控制和 NN 批处理调用
  kata_program       配置装载、对局/self-play 初始化和运行时组装
  kata_book          开局书
  kata_distributed   分布式训练服务客户端
  katago             katago-rs CLI
configs/             GTP、benchmark 和 SM 目标配置
plans/               autotune 生成的 tactic plan（本机相关）
scripts/             冒烟、对拍、对局和 autotune 脚本
docs/                设计、GUI 接入、CUDA 优化和上游模块映射
cpp-shim/            TensorRT C++ FFI shim
```

推荐阅读顺序：

1. [`docs/使用与GUI接入.md`](docs/使用与GUI接入.md)：GUI/GTP 接入和 CUDA 配置。
2. [`docs/需求汇总.md`](docs/需求汇总.md)：功能边界、里程碑和已知缺口。
3. [`docs/cuda-fork-parity-plan.md`](docs/cuda-fork-parity-plan.md)：CUDA 优化主规划和正确性/性能纪律。
4. [`docs/cuda-optimization-plan.md`](docs/cuda-optimization-plan.md)：SM120、kernel、autotune 和实测记录。

## 已知限制

- 当前产品化推理路径（CUDA 和 TensorRT）只保证 19x19 标准棋盘；CUDA 后端强制 NCHW，TensorRT 也以该 b11 模型和布局为验证范围。
- 模型文件和大型数据文件被 `.gitignore` 排除，需要用户自行准备。
- `eigenbackend`、部分历史 KataGo 数据/开局书命令仍是未实现或部分实现状态；以 CLI help 和对应源码为准。
- CUDA 构建依赖 nvcc 和可用的 MSVC host compiler。CUTLASS DualGemm tactic 在找不到 CUTLASS 时会跳过并回退到现有路径；CUDA host 源码编译失败默认会让构建失败，除非显式设置 `KATAGO_ALLOW_BROKEN_CUDA_HOST=1`。
- 性能数字高度依赖 GPU、模型、线程数、batch 和工作负载。不要把仓库中的某次 RTX 5070 Ti 测量当作通用基线；请用 `benchmark`/`nnbench` 在目标机器复测。

## 开发约定

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets
cargo test --workspace
```

数值相关修改遵守“FP32 激活/归约、half 只作为存储边界、禁止 fast-math”的纪律。性能修改需要 ABBA 基准，推理修改需要整图对拍；tactic 环境变量必须通过 `kata_nn::tactic_plan::tactic_var()` 读取，以确保认证 plan 的优先级不被绕过。

## 许可与上游

workspace 的 Cargo 元数据声明 MIT 许可；代码基于 [KataGo](https://github.com/lightvector/KataGo) 及 KataGo-Lite/`katago-rs` 的 Rust 移植工作。请在分发或提交改动时保留上游许可证和对应版权声明。
