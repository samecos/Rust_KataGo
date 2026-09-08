# RustGo 接入 Go Server

RustGo 提供原生 `katago-rs nnworker` 命令，实现 Go Server 的
`goeval.v1.WorkerService/Connect` 双向 gRPC 协议。Server 维护规则状态和搜索图；
RustGo 重放局面、调用 `NnEvaluator` 并返回后处理后的 policy/value/ownership。
Worker 不运行自己的搜索，不经 GTP 子进程转发，也不调用 `analysis` 搜索命令。

## 构建和启动

在 Rust_KataGo 目录执行：

```powershell
cargo build -p katago --features cuda --release
.\scripts\run_go_worker.ps1 -Server 127.0.0.1:50051 -WorkerId rustgo-5070ti
```

等价命令：

```powershell
.\target\release\katago-rs.exe nnworker `
  --server 127.0.0.1:50051 --worker-id rustgo-5070ti `
  --model D:/Go/Server/models/kata1-tf3-b11c768-s11001M-d5973M.bin.gz `
  --config configs/worker_tf3_sm120.cfg --capacity 32
```

`--server` 可为 `HOST:PORT` 或 `http://HOST:PORT`。每个同时运行的 Worker 使用不同 ID。
默认断线重连；`--once` 仅尝试一次连接。Ctrl+C 停止接单并等待已经提交的计算收尾。
Server 的 Drain 会让已接纳任务完成并退出，不重新加入。

`capacity` 是在途评估请求上限，包括排队及取消后尚未算完的任务，独立于
`nnMaxBatchSize`。示例 capacity 32、batch 上限 16，仍由单个 NN 服务消费者凑批。
GPU、精度、batch 和 tactic 使用常规 NN 配置。启动脚本默认使用
`worker_tf3_sm120.cfg`，加载本机 TF3 专属的 DualFFN + q64-serial 认证 plan。
脚本先解析用户传入的模型/配置路径，再切换到仓库根目录运行，结束后恢复原目录。
旧 `best-tactic-plan.json` 绑定 b11fix ONNX，不能用于 TF3；模型/GPU/构建指纹
不匹配会明确拒绝启动。`worker_cuda.cfg` 仍提供不使用认证 plan 的基础配置。

持续维持 32 个在途请求时，可显式选择独立 C32 配置：

```powershell
.\scripts\run_go_worker.ps1 -Server 127.0.0.1:50051 `
  -Config configs/worker_tf3_sm120_c32.cfg -Capacity 32
```

该配置仍用一个 NN 服务线程、物理 batch 上限 16 和 CUDA Graph，独立 plan
`worker-tf3-sm120-c32.json` 只在实际 B16 启用 K384 NN 权重布局。
本机 TF3 持续 C32 的 uncached Worker ABBA 为 1073.48→1093.68 RPC/s
（+1.88%），已通过最终配置的 TF3 全字段对拍和 top-1 128/128。
这不是所有负载的默认升级：低并发仍使用原 `worker_tf3_sm120.cfg`。

持续高并发时可显式选择原有双 NN 服务线程配置，capacity 至少为 64：

```powershell
.\scripts\run_go_worker.ps1 -Server 127.0.0.1:50051 `
  -Config configs/worker_tf3_sm120_throughput.cfg -Capacity 64
```

双线程配置关闭 CUDA Graph，按其独立 plan 运行；低并发时可能因 batch 不足而更慢，
默认配置保持单线程。新布局在双线程 C64 仅 +0.59%，未过 1% 采纳门，因此原
吞吐配置不变。性能对比、数值门及复现工具见
[TF3 Worker 性能审计](tf3-worker-performance-audit.md)。
当前 Worker 的真实推理接入限定为 CUDA；TensorRT 的模型与引擎缓存身份尚未按
Worker 契约验证，选择该后端会明确报错。

## 模型身份必须一致

Worker 计算实际模型文件 SHA-256；`--model-sha256 HEX` 可额外固定期望值。
Server 会固定它的模型哈希，所有 Worker 必须匹配。

RustGo 与 C++ Worker 统一使用：

- 文件：`D:/Go/Server/models/kata1-tf3-b11c768-s11001M-d5973M.bin.gz`
- SHA-256：`1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6`
- 名称：`b11c768h12nbt3tflrs-fson-silu`，modelVersion 17，22/19 输入特征。

CUDA 加载器对读取到的原始文件字节验 SHA、解压和解析，再把 ModelDesc 转成 CUDA
层图。直接使用原生权重、RoPE、BN/RMS、两个策略头和六个 score 输出及模型内的
后处理系数，不依赖外部 ONNX 转换产物。当前仅接受经过形状检查的该 b11 Transformer
结构，其他结构明确报错。原有兼容 ONNX 路径仍可使用，但不同文件哈希不能混池。

如需手动启动 Server（已有 TF3 Server 时直接启动 Rust Worker 即可）：

```powershell
$modelPath = 'D:/Go/Server/models/kata1-tf3-b11c768-s11001M-d5973M.bin.gz'
$modelHash = (Get-FileHash -LiteralPath $modelPath -Algorithm SHA256).Hash.ToLowerInvariant()
& D:/Go/Server/target/release/go-server.exe `
  --http 127.0.0.1:8090 --grpc 127.0.0.1:50051 --model-sha256 $modelHash
```

随后启动 Worker，在 `http://127.0.0.1:8090/api/workers` 查看注册、完成数和真实 NN
rows/batches。前端仍连接 Server 的 `/ws`，不直接连接 Worker。

## 协议边界

- 19 路、`chinese`、半目贴目、严格交替的完整历史；校验非法落点、重复摆子、终局和下一执子方。
- 支持搜索专用的普通第二次 friendly pass 及其后续历史许可，保持实际棋局终局规则。
- 当前实现 Server 的 history modes `(false,false)`；请求其他模式会返回
  `INVALID_CONTEXT`，不会忽略语义参数。
- policy 为原始棋盘方向的 361 点加 pass，非法点保留负值；value、目差和 ownership
  均为白方视角。短期误差能力按模型版本声明；不重复 softmax 或翻转视角。
- 精确回显 session/generation/task/input hash，活跃重复任务只计算一次。
  取消和租约过期不强行中断 GPU kernel，完成后丢弃输出；旧流任务不会进入新连接。
- 后端失败返回 `EVALUATOR_ERROR`，不把默认输出当作真实评估；输入错误、输出错误和
  内部错误分别分类。心跳发送实际 NN 计数，阶段耗时不冒充 kernel 时间。

协议源文件为 `crates/kata_worker/proto/worker.proto`，从
`D:/Go/Server/proto/worker.proto` 原样同步。服务器协议变更时需同步并重跑验证。
`engine_commit` 声明实际 Rust_KataGo 源码版本及 dirty 标记，不冒充 C++ 基座提交。

## 验证

```powershell
cargo test -p kata_worker
.venv/Scripts/python.exe scripts/verify_go_server_worker.py `
  --server D:/Go/Server/target/release/go-server.exe
```

第二条默认使用显式 `--allow-dummy` 合成夹具，需要先 `cargo build -p katago`。
测试使用独立的随机本机端口和 Server 进程；只停止自己创建的进程。
合成模式有独立模型身份和醒目的能力标记，不能视为真实 NN 验收。

真实 CUDA 验证：

```powershell
.venv/Scripts/python.exe scripts/verify_go_server_worker.py `
  --server D:/Go/Server/target/release/go-server.exe `
  --worker target/release/katago-rs.exe `
  --model D:/Go/Server/models/kata1-tf3-b11c768-s11001M-d5973M.bin.gz `
  --config configs/worker_cuda.cfg --output target/worker-smoke-cuda
```

脚本检查注册、模型身份、真实 Server `genmove`、悔棋后再搜索、双 pass 终局及任务
回收，生成 `report.json` 和双方日志。取消、租约、重连、排空等竞态由真实 tonic 流
集成测试覆盖。

TF3 的数值金标为 C++ CUDA FP32，测试从协议输入开始，包含完整历史重放、特征构造、
所有对称方向、策略温度/乐观系数、后处理和白方视角转换。FP32 参考另外用 Eigen CPU
抽查。固定容差先于测量定义，脚本失败返回非零状态：

```powershell
D:/Go/Server/worker/.venv-windows/Scripts/python.exe scripts/compare_worker_outputs.py `
  --model D:/Go/Server/models/kata1-tf3-b11c768-s11001M-d5973M.bin.gz
```

默认 16 局面 × 8 对称，每个请求完成后才发下一个；`--request-window 32` 验证并发
凑批。参考 Worker 与 Rust 顺序运行，结果保存在 `target/worker-parity-tf3`。
`--policy-optimism 1 --policy-temperature 1.3` 可单独验证乐观策略头及非默认温度，
建议配 `--output target/worker-parity-tf3-optimistic` 保存独立结果。
该 Python 环境需要 `grpcio` 和 `grpcio-tools`。原有 ONNX 路径仍用 CUDA dump + ORT
执行整图回归。

真实 C++ FP16 + Rust 混合池：

```powershell
D:/Go/Server/worker/.venv-windows/Scripts/python.exe scripts/verify_go_server_worker.py `
  --server D:/Go/Server/target/release/go-server.exe `
  --worker target/release/katago-rs.exe `
  --model D:/Go/Server/models/kata1-tf3-b11c768-s11001M-d5973M.bin.gz `
  --config configs/worker_cuda.cfg `
  --cpp-worker D:/Go/Server/worker/build-windows-CUDA/Release/katago.exe `
  --cpp-config D:/Go/Server/worker/cuda-5070ti.cfg `
  --output target/worker-smoke-tf3-mixed
```

混合测试使用真实 Go Server 调度，断言两 Worker 都完成任务且无失败，执行两次
`genmove`、悔棋和双 pass；结束后通过透明协议中继向 Worker 发送 Drain，确认它们
正常退出且 Server 池清空。仅测试使用此中继，日常启动直接连接 Server。

## 本机验收记录（2026-09-08）

- 原生 TF3 真实文件解析成功：70,442,025 参数，33 attention + 33 FFN，
  模型 SHA/所有权重排列/各头通道及后处理倍率通过结构回归。
- CUDA release 构建成功；TF3 对 C++ FP32 的 128 个请求全字段过门、策略首选
  128/128。最大绝对误差：胜率 0.001312、policy 0.001426、ownership 0.003383。
- 32 请求并发、物理 batch 上限 16：128 请求分 12 批执行，零失败；
  全字段过原门，策略首选 128/128。
- 乐观系数 1、温度 1.3、并发 32 的另一组 128 请求同样全字段过门，首选 128/128。
- 已用 Eigen CPU 核对空盘八方向及四个误差较大的样本，Rust 均在原容差内。
- 真实 Server 混合池通过：C++ 完成 415、Rust 完成 81，失败均为 0；
  同一 TF3 哈希、两次搜索/悔棋/终局通过，Drain 后池为空。
- NN/Worker/CLI 相关回归 273 项通过，4 个旧测试按现有规则忽略；原 ONNX 的
  16 局面 CUDA/ORT 整图回归通过，策略首选 16/16。
- 最终 CUDA 特性下 NN 单测 125 项通过，含策略通道 stride、passing hack 下溢边界。

C++ FP16 与 Rust CUDA 采用不同的中间舍入路径，输出不要求逐位相同。实测 C++ FP16
自身相对 FP32 的最大胜率误差 0.004450、ownership 0.006274；Rust 的对应误差更小。
验收坚持使用 FP32 金标和原定容差。

本机报告位于 `target/worker-smoke-tf3-mixed/report.json`、
`target/worker-parity-tf3-cuda-fp32/precision-calibration.json`；这些是可重新生成的产物。
