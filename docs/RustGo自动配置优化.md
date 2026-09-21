# RustGo 自动配置优化

需要在新 CUDA 机器上从零搜索并生成 **新 PLAN JSON**，请使用 [FULL AUTOTUNE](RustGo-FULL-AUTOTUNE.md) 的 `scripts/full_autotune.ps1`。本页的 `tune_rustgo.ps1` 继续用于已有计划选配。

更新：2026-09-20。日常一键入口为 `scripts/tune_rustgo.ps1`，核心实现为 `scripts/tune_runtime.py`。已有 CUDA release 二进制即可使用，不必重新构建。它自动选择当前模型和机器适用的推理配置，并生成启动脚本。

本次实现的是**已有认证推理组合的自动选配，以及单机搜索线程优化**。它不修改权重、CUDA kernel、精度边界或认证计划内容，也不承诺找到所有可能配置的全局最优。旧 Fork 性能优化仍保持结项；这次是新的配置工具交付。

## 1. 直接运行

先退出正在占用同一 GPU 的 RustGo、其它 Worker 或推理程序，使各次测量可比较。脚本不会擅自停止已有进程。以下均在 PowerShell 执行。

单机 GUI / 批量分析：

```powershell
Set-Location D:/code/Rust_KataGo
.\scripts\tune_rustgo.ps1 -Mode local
```

分布式 Worker，日常约 32 个在途请求：

```powershell
Set-Location D:/code/Rust_KataGo
.\scripts\tune_rustgo.ps1 -Mode worker -Capacity 32
```

持续高并发 Worker：

```powershell
.\scripts\tune_rustgo.ps1 -Mode worker -Capacity 64
```

默认模型为本机 `D:/Go/Server/models/kata1-tf3-b11c768-s11001M-d5973M.bin.gz`。换模型时显式指定：

```powershell
.\scripts\tune_rustgo.ps1 -Mode local -Model D:/code/b11fix.onnx
```

默认二进制为仓库的 `target/release/katago-rs.exe`；可用 `-Binary` 指定另一份。不测速预检用 `-DryRun`：它读取 CUDA 指纹、列出匹配候选，仍会创建诊断目录、报告、候选 cfg 和计划副本；不执行推理测速，也不产生最终配置。

```powershell
.\scripts\tune_rustgo.ps1 -Mode worker -Capacity 64 -DryRun
```

运行中每个测量进程都会打印候选名称与结果。ABBA 需要反复加载模型，不是几秒内结束；耗时取决于 GPU、模型和候选数。默认 local 最多 12 个组合，Worker 最多 3 个组合，全部顺序运行。

## 2. 运行条件与产物

需要 Python 3.11+、CUDA release 二进制、兼容模型及可用的 GPU 运行环境。仅支持引擎已适配的 19 路 b11 Transformer；脚本不能让尚未支持的模型结构变得兼容。

- local 核心使用 Python 标准库。
- Worker 复用仓库真实 gRPC 测量工具，需要 `grpcio`、`grpcio-tools`、`protobuf`。PowerShell 入口优先寻找已具备依赖的 Python；本机也检查 `D:/Go/Server/worker/.venv-windows`。找不到合适环境时，自动在仓库 `.venv-runtime-tune` 建立隔离环境并安装 `scripts/requirements-runtime-tune.txt`，首次安装需要联网。
- Worker 测试创建临时本机 gRPC 服务，运行真实 `nnworker` 子进程；不需要 Go Server/C++ Worker，不连接生产 Server。结束时发送 Drain，异常时仅清理自己创建的 Worker。
- 当前认证二进制可以直接使用。缺少二进制时，先按[编译指南](编译指南.md)构建 `cargo build -p katago --features cuda --release`。脚本不会自行覆盖现用引擎。

默认每次写入独立的 `autotune-output/<时间>-<模式>/`。终端打印完整目录。可以用 `-Output` 指定一个**尚不存在**的目录，重跑时用新目录；已有目录会报错，避免旧报告和新结果混在一起。

| 文件 | 用途 |
|---|---|
| `report.json` | 模型/GPU/构建身份、二进制 SHA、跳过的计划、每次原始指标、ABBA 裁决、最终选择与功能验证 |
| `rustgo.cfg` | 完成选配并验证后的最终配置；通过绝对路径引用本次复制的 plan |
| `run-gtp.ps1`、`run-analysis.ps1` | local 模式生成的启动入口 |
| `run-nnworker.ps1` | Worker 启动入口，已包含本次 capacity 和模型 SHA；Server 地址及 Worker ID 可在启动时追加 |
| `*.json` 计划副本、`candidate-*.cfg` | 保留完整原计划内容与候选配置，方便追溯 |
| `*.log`、子目录 `report.json` / `timings.json` | 引擎日志与真实 Worker 计时记录 |

只有 `report.json.status = PASS` 才表示完整流程通过。`selection_status = IMPROVED` 表示最终候选相对初始基线也通过复核；`BASELINE_RETAINED` 表示保留初始配置，可能是收益不足、波动过大或没有其它匹配候选，不能据此宣称提速。配置功能通过也不等于重新认证所有数值。

`DRY_RUN` / `SMOKE_ONLY` 是诊断结果，不输出 `rustgo.cfg` 和启动脚本。子进程失败、输出缺失、超时或中断会退出非零并保留失败报告，不替换现用配置。

生成后启动示例（把目录换成实际输出）：

```powershell
# 本例假设调优时指定了 -Output autotune-output/local-ready
.\autotune-output\local-ready\run-gtp.ps1

# 本例假设 Worker 调优时指定了 -Output autotune-output/worker-ready
.\autotune-output\worker-ready\run-nnworker.ps1 --server 127.0.0.1:50051 --worker-id rustgo-5070ti
```

启动脚本使用绝对模型/配置路径，并临时清理 `KATAGO_*` 环境变量，保持与测量相同的环境；子进程退出后恢复调用者环境。移动输出目录、模型或仓库后需要重新生成相应路径。保留整份输出目录，不要只复制 `rustgo.cfg`。

## 3. 自动选择的具体过程

1. 检查文件，运行 `cuda-fingerprint --model ...`，核对原始模型 SHA、GPU 名称/计算能力/SM/L2 和完整 backend build 指纹。
2. 从明确列出的现有认证组合中匹配，不扫描历史实验目录。TF3 当前为 daily B16/S1、C32 B16/S1、throughput B14/S2；ONNX 使用相应平台的 q64 B16/S1。B 是 batch 上限，S 是 NN 服务线程数，均随完整 plan 固定。未纳入标为 preliminary 的 5060 Ti 配置。
3. 没有匹配认证计划时，明确报告并使用无 plan 的 CUDA 基础 B16/S1。local 仍可扫描搜索线程；Worker 此时只验证基础配置，不虚构跨机/跨模型内核优化结果。模型不受引擎支持则初始化失败。
4. local 遍历匹配 profile × `numSearchThreads` 网格，默认 `8,4,12,16`，首档 8 为初始基线。每次实际执行 `benchmark -v 800 -n 3 -t <单档> --fixed-batch-size <B>`，以完整结果的 **visits/s** 评分，同时记录 NN rows/s 和实际平均 batch。显式固定 batch 防止 benchmark 自行按线程数改变测试条件。
5. Worker 固定本次 `capacity`，对各完整 profile 执行真实 uncached gRPC 请求：默认至少 128 次预热、1024 次计时请求，覆盖已有 16 局面 × 8 对称夹具。评分为 **RPC requests/s**；检查身份、完成数、失败数和实际 NN rows，并记录 RTT、平均 batch。启动、预热、最终心跳及 Drain 不纳入测速。
6. 每个挑战者均按 A→B→B→A 顺序测量；两边各自 `max/min - 1 ≤ 5%`，且候选几何平均速度至少提升 1%，才替换当前选择。不稳定或慢于基线就保留基线，不自动反复重测直到出现好数字。
7. 若选择发生变化，再对原始初始配置做一次完整 ABBA。未通过就回到初始配置，避免逐步选择掩盖总体回退。
8. local 对最后配置验证 GTP `genmove` 和 EOF 批量 JSON analysis；Worker 再验证真实 RPC/NN 计数及 Drain。核对过程中模型、二进制及计划副本未变更后，生成最终配置与启动脚本。

认证计划按原字节复制，不重写 `target` / `backend_build` 来规避拒绝启动，也不拼接不同计划里的 tactic。现有数值认证只在原身份和原组合范围内沿用。本次不新启用休眠 kernel，也不修改算术。

## 4. 如何选调优目标

| 目标 | 使用方式 | 结果解释 |
|---|---|---|
| 一盘棋的本地实时分析 | `-Mode local` | 优化单个 MCTS 搜索的速度；通过 GTP 实时输出 |
| JSON 批量分析，每次一局面 | `-Mode local`，运行时 `--analysis-threads 1` | 沿用同一搜索配置；脚本不调多局面并发 |
| Worker 日常 C32 负载 | `-Mode worker -Capacity 32` | 在固定 32 在途窗口下比较 profile |
| Worker 长期 C64 负载 | `-Mode worker -Capacity 64` | 在固定 64 在途窗口下比较 profile |
| Worker 请求包含领地输出 | 再加 `-Ownership` | 使测试包含相应输出传输成本 |

capacity 是给定的业务负载条件，不作为“越大越好”的搜索变量。脚本不会改 Server 的窗口或 CPU 搜索参数。loopback 结果包含本机协议开销，但不等于跨机器网络、真实 Server 调度和缓存条件下的吞吐保证。

单机更高 visits/s 也不直接证明棋力提高；本次没有对局测 Elo。改变 `--analysis-threads`、同时运行多局搜索、改变 GPU 负载或使用不同 Server 窗口，都可能改变最合适的配置。

## 5. Python 入口与维护边界

Linux/WSL 或需要细调参数时直接用 Python；WSL 必须使用该环境自己的 CUDA 二进制和匹配计划：

```bash
python scripts/tune_runtime.py --mode local --model /path/model.bin.gz \
  --binary ./target/release/katago-rs --threads 8,4,12,16

python -m pip install -r scripts/requirements-runtime-tune.txt
python scripts/tune_runtime.py --mode worker --model /path/model.bin.gz --capacity 64
```

生成的 `.ps1` 启动脚本面向 PowerShell。Linux/WSL 没有 PowerShell 时，直接用生成配置启动；以下 Bash 子进程先清理实验环境变量，退出后不影响原 shell。替换模型和输出路径：

```bash
(
  for key in ${!KATAGO_@}; do unset "$key"; done
  ./target/release/katago-rs gtp --model /path/model.bin.gz \
    --config /absolute/path/to/autotune-output/run/rustgo.cfg
)
# Worker 将 gtp 换成 nnworker，并加：
# --server HOST:50051 --worker-id gpu-1 --capacity 64
# capacity 使用本次调优值；模型 SHA 可从 report.json 读取并传 --model-sha256。
```

Python 入口还支持 `--visits`、`--positions`、`--requests`、`--max-visits`、`--timeout`，完整参数见 `--help`。正常选择要求 visits 至少 400、Worker 计时请求至少 1024；收益门不能低于 1%，波动门不能高于 5%。可以收紧门槛。`--smoke-only` 仅用于开发验证，不发布优化结果。

旧 `scripts/autotune.py` 保留为历史内核候选实验工具。它没有整合当前 TF3 全组合数值门、真实 Worker 目标及稳定性验收，不能作为本次一键交付入口，也不能仅换模型名就覆盖正式 plan。

后续补齐的 [FULL AUTOTUNE](RustGo-FULL-AUTOTUNE.md) 已接入原生 TF3 对 C++ FP32 的完整 Worker 数值门、实际路径标记、硬件能力筛选及 ABBA，并生成新计划。ONNX 对 ORT 的 FULL 自动认证仍未接入。现有 plan 失配时不能自动“重新签名”。可执行文件内置 `autoconfig` 子命令仍是可选后续封装；当前入口使用脚本。

运行方式见 [Worker 与单机分析说明](RustGo运行模式说明.md)。

## 6. 本次交付验证（2026-09-20）

本机现有 RTX 5070 Ti、正式 CUDA release、上述原生 TF3 模型，未修改引擎或正式计划。

| 检查 | 结果 |
|---|---|
| `python -m unittest discover -s scripts -p test_tune_runtime.py -v` | 14 项 CPU 回归通过；覆盖模型/构建失配、波动拒绝、总体回退、完整结果解析、子进程失败、冒烟失败不发布、路径引用及 GTP 回包格式 |
| PowerShell 预检 | 自动识别 3 个匹配 TF3 profile，跳过 ONNX；线程参数支持 `8,12` 与 `'8,12'` |
| Worker 完整流程，C32 | 3 个 profile，1024 计时请求/轮，ABBA + 最终复核 + 真实 RPC/Drain 通过；选择 C32 profile |
| local 完整流程 | 验证网格为 3 个 profile × 8/12 搜索线程，每轮 800 visits × 3 局面；最终复核、GTP genmove 与真实 JSON analysis 通过 |
| 启动脚本 | PowerShell 语法、追加 CLI 参数通过；用 dummy `query_version` 独立验证管道 stdin 转发，未把 dummy 结果计入真实性能 |
| 文档读者检查 | 核对自动入口、模式边界、失败行为、预检产物和 Linux 启动方式 |

原始本机报告分别在 `target/runtime-tune-check/worker-full/report.json` 和 `target/runtime-tune-check/local-verified/report.json`，属于忽略提交的可再生产物。前一次 local 运行因检查器未兼容 `= 1` 格式而失败的报告保留在 `target/runtime-tune-check/local-full/`；当时未发布最终配置，修正后在新目录完成验证。没有为追求收益重复重测失败的性能候选。

这些是本次工具验证，不能替代新 kernel 的整图数值认证，也未将所选配置覆盖到现用 `configs/`。默认 local 还包含 4/16 线程候选；本次没有额外扩大到默认完整网格或 C64 测量。
