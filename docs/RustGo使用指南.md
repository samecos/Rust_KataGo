# RustGo 使用指南

2026-09-20 新入口：新 CUDA 电脑生成 PLAN 使用 [FULL AUTOTUNE](RustGo-FULL-AUTOTUNE.md)（`scripts/full_autotune.ps1`）；已有计划选配使用[一键自动配置优化](RustGo自动配置优化.md)（`scripts/tune_rustgo.ps1`）。另见[分布式 Worker / 单机运行说明](RustGo运行模式说明.md)，涵盖跨机器连接、GTP 实时分析及当前 JSON analysis 的 EOF 批量输出限制。下文保留现成认证配置的快速启动方法。

剪枝 TF3 模型请先看[剪枝模型支持](RustGo剪枝模型支持.md)：需要支持可变 FFN 宽度的新二进制、模型自己的参考包与重新生成的 PLAN。

适用日期：2026-09-16。本机 Windows、RTX 5070 Ti、当前已认证 CUDA release 版本。本轮性能优化已结项，可以直接使用现有引擎；首次启动无需再跑 benchmark、对拍或 autotune。

## 1. 先选使用方式

| 需求 | 启动方式 | 谁负责搜索 |
|---|---|---|
| 给 Go Server 提供 GPU 评估 | `katago-rs.exe nnworker` | Go Server；Worker 只做神经网络评估 |
| 在围棋 GUI 中对局、分析 | `katago-rs.exe gtp` | RustGo 自己的搜索引擎 |
| 脚本批量分析棋局 | `katago-rs.exe analysis` | RustGo；通过标准输入/输出交换 JSON 行 |

本轮最新吞吐优化主要面向第一种方式。Worker 的 RPC/s、纯前向 NN rows/s 和 GTP 的 visits/s 是不同指标，不能互换。

## 2. 本机直接启动高吞吐 Worker

前提：Go Server 已在 `127.0.0.1:50051` 提供 gRPC，且绑定下列 TF3 模型哈希。服务器在其他机器时，只改 `-Server` 地址。

在 PowerShell 执行：

```powershell
Set-Location D:/code/Rust_KataGo

.\scripts\run_go_worker.ps1 `
  -Server 127.0.0.1:50051 `
  -WorkerId rustgo-5070ti `
  -Model D:/Go/Server/models/kata1-tf3-b11c768-s11001M-d5973M.bin.gz `
  -ModelSha256 1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6 `
  -Config configs/worker_tf3_sm120_throughput.cfg `
  -Capacity 64
```

脚本调用 `target/release/katago-rs.exe`，并自动将进程工作目录设为仓库根目录，使配置中的 `plans/...` 相对路径正确解析。原始 `.bin.gz` 直接加载，无需解压、转换 ONNX 或重新训练。

这套配置是 **B14 / S2 / C64**：物理批量上限 14、两个 NN 服务线程、最多 64 个在途请求。实际批量取决于供数，`Capacity 64` 不会自动产生 64 个请求，也不意味着 batch 为 64。它适合持续高并发；单局低负载请使用下一节的日常配置。

当前采用 strict Attention、只读 QKV/RoPE、QKV N128、既有 half 全零 FFN 通道压缩、输出投影 N128，并关闭 CUDA Graph。最新 FFN down NN 布局实验未被采纳；不要使用 `target/fork-parity-20260908/` 中的实验二进制或候选计划来部署。

## 3. 日常和 C32 配置

| 使用场景 | `-Config` | `-Capacity` |
|---|---|---:|
| 低并发、日常对局、负载不确定 | `configs/worker_tf3_sm120.cfg` | 32 |
| 长期维持约 32 个在途请求 | `configs/worker_tf3_sm120_c32.cfg` | 32 |
| 长期高并发，使用本轮最新吞吐组合 | `configs/worker_tf3_sm120_throughput.cfg` | 64 |
| 新环境暂无匹配认证计划 | `configs/worker_cuda.cfg` | 32 |

例如，日常配置：

```powershell
Set-Location D:/code/Rust_KataGo
.\scripts\run_go_worker.ps1 -Server 127.0.0.1:50051 `
  -WorkerId rustgo-5070ti -Config configs/worker_tf3_sm120.cfg -Capacity 32
```

省略 `-Model` 时，脚本默认就是第 2 节的本机 TF3 文件；省略 `-Config` 时默认是日常配置，**不会自动选择最新吞吐配置**。切换配置时先正常停止旧 Worker，再运行新命令。同一 GPU 通常只运行选定的一套；多个同时连接的 Worker 必须使用不同 ID。

默认和 C32 配置使用 B16、单 NN 服务线程及 CUDA Graph，保留各自的认证策略。没有一套配置在所有负载下都最快。

## 4. 尚未启动 Go Server 时

已有同模型 Server 时跳过本节。新建本机 Server 可在另一个 PowerShell 窗口运行已有可执行文件：

```powershell
Set-Location D:/Go/Server
.\target\release\go-server.exe `
  --http 127.0.0.1:8090 `
  --grpc 127.0.0.1:50051 `
  --model-sha256 1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6 `
  --max-in-flight 128
```

随后启动 Rust Worker。Server 只需要模型哈希，Worker 才加载模型文件。前端仍连接 Server 的 `/ws`；不要将前端地址改成 Worker 的 gRPC 端口。单会话在途窗口需足以供给 Worker，128 是上限而非始终发满的保证。

在 `http://127.0.0.1:8090/api/workers` 查看 Worker 注册、完成请求数及 NN rows/batches。没有请求时计数不增加是正常现象。完整 Server 构建和协议说明见 [TF3 部署文档](TF3权重适配与Worker部署.md) 和 [Worker 接入文档](Go-Server-Worker.md)。

## 5. 通过 GTP 在 GUI 中使用

TF3 模型使用它自己的计划。以下命令选择日常单线程 NN 配置，搜索上限为每手 800 visits；这是使用示例，不代表本轮重新测过 GTP 棋力或搜索速度。

```powershell
Set-Location D:/code/Rust_KataGo
.\target\release\katago-rs.exe gtp `
  --model D:/Go/Server/models/kata1-tf3-b11c768-s11001M-d5973M.bin.gz `
  --config configs/gtp_smoke.cfg `
  --override-config "nnBackend=cudabackend,cudaTacticPlan=plans/worker-tf3-sm120.json,nnMaxBatchSize=16,numSearchThreads=12,maxVisits=800"
```

GUI 中填写：

- 可执行文件：`D:/code/Rust_KataGo/target/release/katago-rs.exe`。
- 工作目录：`D:/code/Rust_KataGo`。
- 参数：上面从 `gtp` 开始的所有参数，写在一行；不要复制 PowerShell 的续行反引号。

`numSearchThreads` 控制 MCTS 搜索线程，`numNNServerThreadsPerModel` 控制 NN 服务线程，两者不同。Worker 不运行 MCTS，所以调 `numSearchThreads` 不能提高 Worker 吞吐。

GTP 通过标准输入/输出与 GUI 通信，单独启动后等待命令是正常行为。带 interval 的 `kata-analyze` / `lz-analyze` 持续输出，`stop` 停止分析，`quit` 退出。仅支持 19 路。

已有 `b11fix.onnx` 和旧 GTP 配置的用法见 [GUI 接入说明](使用与GUI接入.md)。`plans/best-tactic-plan.json` 绑定 ONNX，不能给原生 TF3 使用。

## 6. 精度、身份与启动问题

引擎使用混合精度：权重和部分中间张量以 FP16 存储，GEMM、QK/PV 累加及关键激活/归约使用 FP32，主干残差和最终输出为 FP32。它不是全程 FP32，也没有为追赶 Fork 而改用 Fork 的部分 FP16 累加路径。

| 现象 | 处理 |
|---|---|
| `connection refused` 或反复重连 | 核对 Server 是否启动、gRPC 地址和端口是否可达 |
| 模型 SHA 不匹配 | Server、实际 `.bin.gz` 文件和 plan 必须对应；换模型需要相应身份，不能仅改声明哈希 |
| `backend build ... mismatch`、设备或 artifact 不匹配 | 当前可执行文件、驱动/运行库、GPU 或计划不匹配；使用匹配的一整套，不能手改 plan 身份强行通过 |
| 环境变量没有改变效果 | 已绑定 tactic 的优先级是 plan > 环境变量 > 默认值；正常使用不需要实验环境变量 |
| 吞吐未达到报告数字 | 先区分配置、供数、实际 batch 和统计口径；1574.82 RPC/s 是已记录的本机持续 C64 uncached 测量，不是所有局面和负载的保证 |
| 二进制无 CUDA 后端 | 使用已认证 release 二进制；默认 `cargo build` 的 dummy 后端不是真实棋力推理 |

认证计划在不匹配时会明确拒绝启动，不会悄悄假装使用优化路径。不带 plan 的基础配置可以用于兼容环境，但内置默认 tactic 与认证组合不同，不能继承认证性能结论。

## 7. 停止、重启和重新构建

- Worker 按 `Ctrl+C` 停止接单并等待已提交计算收尾；Server 发 Drain 时完成已接纳任务后退出。正常断线默认重连，`-Once` / `--once` 则只尝试一次连接。
- 当前机器已有匹配 release 版本时直接启动即可，不必为了使用而重新构建。
- 确实需要源码构建时，在仓库根目录执行 `cargo build -p katago --features cuda --release`；本机 CUTLASS 位于 `D:/code/cutlass`，可通过 `KATAGO_CUTLASS_ROOT` 指定。构建说明见 [编译指南](编译指南.md)。
- 重编译、移动源码后重编译、升级 CUDA/CUTLASS、换模型或换机器，都可能改变认证身份。不要用实验目录的构建覆盖已认证版本；备份可执行文件、`configs/`、`plans/` 和模型身份后再做升级。

本轮已完成结果、适用边界和证据入口见 [优化结项记录](RustGo性能优化结项记录.md)。进一步调优属于以后单独发起的工作，本轮没有继续运行或等待的测试。
