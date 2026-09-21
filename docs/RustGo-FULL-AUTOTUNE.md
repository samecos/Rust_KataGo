# RustGo FULL AUTOTUNE：在另一台 CUDA 电脑生成 PLAN

更新：2026-09-20。入口为 `scripts/full_autotune.ps1`（Windows）或 `scripts/full_autotune.py`（跨平台）。成功后生成新的 **`result/plan.json`**，同时生成配套 `rustgo.cfg`、启动脚本和数值/性能报告。

本工具从目标机器的指纹和基础 tactic 开始搜索，不要求目标机器已有认证 PLAN。原有 `tune_rustgo.ps1` 仍负责已有计划选配；两者用途不同。

## 1. 适用范围与要带走的文件

本工具面向本项目 CUDA 后端已适配的 **原生 TF3 v17 / 19 路 Transformer**。旧正式版支持 b11；新增剪枝支持版本可读取每层独立的 FFN 宽度，针对 `b15-ffn-pruned-a8.bin.gz` 的使用和验证记录见[剪枝模型支持](RustGo剪枝模型支持.md)。必须选用支持该模型的二进制；仅更新调优脚本不能扩展旧引擎的模型结构。ONNX 会明确拒绝：不能把原生 TF3 的参考输出拿去给 ONNX 认证。也不支持任意神经网络或仅因为装有 CUDA 就支持所有 GPU 架构。

目标机器需要：

1. 能运行本项目 CUDA 后端的 NVIDIA GPU、驱动和运行库，以及**兼容该 GPU 的 CUDA release 二进制**。默认寻找 `target/release/katago-rs.exe`，也可通过 `-Binary` / `--binary` 指定。使用已有兼容二进制不要求安装 Rust、nvcc、CUTLASS 或 C++ 编译器；源码构建则按[编译指南](编译指南.md)准备。
2. Python 3.11+。Windows 入口会找到合适 Python，缺少 gRPC 依赖时自动创建 `.venv-runtime-tune` 并安装固定依赖；首次安装需要联网。
3. 实际模型文件。模型不包含在工具包内，不能用 `/dev/null`。
4. 工具代码、协议、局面夹具、FP32 参考包。本仓库当前模型的参考包已内置，目标机器**不需要 C++ Worker，也不需要 Go Server**。

可以复制当前完整工作区；这些新工具若尚未提交/推送，不会因为在另一台机器拉取旧版本而自动出现。也可使用 `package_full_autotune.py` 生成独立工具包：

```powershell
# 在本机仓库根目录打包；输出路径必须尚不存在。
python scripts/package_full_autotune.py --output D:/RustGo-Full-Autotune-tools.zip

# 可选：把一份已构建的 Windows CUDA 二进制一起放入包中。
python scripts/package_full_autotune.py `
  --binary target/release/katago-rs.exe `
  --output D:/RustGo-Full-Autotune-Windows.zip
```

后一种不会安装驱动、打包模型或安装 CUDA DLL。包内二进制支持范围仍取决于它编译时包含的 SM 目标；换更老的 GPU 可能需要重新构建。工具包中的 `MANIFEST.json` 记录每个文件的 SHA256。

工具包只包含运行调优所需的脚本和资料，不含完整 Rust 源码；需要重新构建引擎时使用完整项目仓库。

## 2. Windows：完整运行命令

假设工具包解压到 `D:/RustGo-Autotune`，模型放在 `D:/models/`，引擎放在 `D:/RustGo/`。以下路径按实际位置替换。

先做预检，不运行 CUDA 推理测速：

```powershell
Set-Location D:/RustGo-Autotune
.\scripts\full_autotune.ps1 `
  -Model D:/models/kata1-tf3-b11c768-s11001M-d5973M.bin.gz `
  -Binary D:/RustGo/katago-rs.exe `
  -Mode worker -Capacity 32 -DryRun
```

预检会检查模型/GPU/build 指纹、参考包身份及 128 个参考结果的完整性，并写诊断报告；不会生成可部署 PLAN。若使用尚未导出的自定义模型参考、仅提供 `-CppWorker`，预检只登记待生成参考，正式运行时才启动 C++。

完整优化 Worker 推理吞吐：

```powershell
.\scripts\full_autotune.ps1 `
  -Model D:/models/kata1-tf3-b11c768-s11001M-d5973M.bin.gz `
  -Binary D:/RustGo/katago-rs.exe `
  -Mode worker -Capacity 32
```

持续高并发业务用 `-Capacity 64`。capacity 固定本轮测量的请求窗口；它不是物理 batch，不会自动修改生产 Server 的窗口，也不会凭空制造生产流量。

完整优化单机搜索/分析：

```powershell
.\scripts\full_autotune.ps1 `
  -Model D:/models/kata1-tf3-b11c768-s11001M-d5973M.bin.gz `
  -Binary D:/RustGo/katago-rs.exe `
  -Mode local
```

Worker 模式以真实 uncached gRPC requests/s 评分；local 模式以完整搜索的 visits/s 评分，并额外扫描搜索线程。两者都先通过真实 Worker 通路对照 FP32 数值参考，验证局面重放、特征、整网推理和后处理。

包内有二进制时可省略 `-Binary`。本机仓库里的最短调用为：

```powershell
.\scripts\full_autotune.ps1 `
  -Model D:/Go/Server/models/kata1-tf3-b11c768-s11001M-d5973M.bin.gz
```

不要给正式完整运行加 `-Groups`；这是开发诊断的缩小范围参数，报告会标为 `CUSTOM_GROUPS`。

## 3. Linux / WSL：等价命令

使用该环境自身可执行的 CUDA 二进制，不要把 Windows `.exe` 的认证身份当作 Linux 构建身份。进入工具包/仓库根目录：

若解压器没有保留 Linux 二进制的执行权限，先执行 `chmod +x /opt/rustgo/katago-rs`（替换为实际路径）。

```bash
python3 -m venv .venv-runtime-tune
.venv-runtime-tune/bin/python -m pip install -r scripts/requirements-runtime-tune.txt

# 预检
.venv-runtime-tune/bin/python scripts/full_autotune.py \
  --model /data/models/kata1-tf3-b11c768-s11001M-d5973M.bin.gz \
  --binary /opt/rustgo/katago-rs --mode worker --capacity 32 --dry-run

# 完整搜索，生成本机的新计划
.venv-runtime-tune/bin/python scripts/full_autotune.py \
  --model /data/models/kata1-tf3-b11c768-s11001M-d5973M.bin.gz \
  --binary /opt/rustgo/katago-rs --mode worker --capacity 32
```

单机分析将 `--mode worker` 换成 `--mode local`。Windows 如果不使用 PowerShell 入口，也可按上述步骤创建 `venv`，把解释器路径换成 `.venv-runtime-tune/Scripts/python.exe`，使用同一个 Python 命令。

## 4. 运行过程与判定

调优时使同一 GPU 空闲，暂停其它推理、游戏和 GPU 基准。脚本不会杀掉已有应用。它对所有候选顺序执行，每次子进程启动、预热、测量、排空后退出，不同时跑两个 GPU 测试。

默认有序搜索组：

| 组 | 搜索内容 |
|---|---|
| `batch` | 物理 batch 上限，默认 8、4、12、14、16；初始基线 B8/S1 |
| `lanes_graph` | 1/2 个 NN 服务线程 × CUDA Graph 开/关 |
| `gemm` | cuBLASLt、手写 GEMM 及两个小 tile 选项 |
| `dual_ffn` | 编译能力允许时比较 DualFFN 开/关 |
| `attention` | 根据编译能力比较 q128、q64、q64-serial、v3 |
| `layout` / `rank` | cuBLASLt 权重布局与启发式/计时重排 |
| `fusion` / `rms` / `splitk` | 融合、RMSNorm、Split-K 选项 |
| `batching` | padding、异步流水线开/关、凑批等待窗口 |
| `specialized` | 仅在目标和构建完全匹配时，把现有专用组合作为候选重新验证；其它 GPU 自动跳过 |
| `reschedule` | 在选定内核组合上再次比较 batch、NN 服务线程与 Graph；遵守专用路径依赖 |
| `threads` | 仅 local：搜索线程默认 8、4、12、16 |

这是有限候选目录上的完整有序搜索，后续组基于当前胜出配置生成候选，**不是所有开关的笛卡尔积穷举，也不保证数学上的全局最优**。专用 SM120 AOT/模型压缩等路径保留原硬件、模型和 artifact 限制，不会把 5070 Ti 参数强行塞给其它 GPU。

每个实际测速候选均经过：

1. 写入独立的候选 JSON，绑定本机 GPU、原始模型 SHA、当前 CUDA backend build/artifact 身份。候选文件标记为 `UNVALIDATED_CANDIDATE`，不可当作最终产物使用。
2. 与 C++ FP32 参考对拍：16 局面 × 8 对称 = 128 请求，检查全部 11 个数值字段和策略 top-1 100%。分别测试 W1、本次 capacity，以及足以观察目标 batch/lane 路径的并发窗口；相同窗口去重。容差沿用 `compare_worker_outputs.py`，没有放宽门槛。
3. 核对已有的 `[cuda-tactic]` 实际路径标记；只设置了开关但未观察到生效的候选会被拒绝。部分调度参数没有独立内核标记，依据实际已加载计划、相同二进制与端到端数值/性能记录验证，不冒充内核路径证明。
4. 通过后才执行 A→B→B→A。两侧各自 `max/min - 1 ≤ 5%`，且候选几何平均速度至少提升 1% 才采纳；失败、无收益或不稳定均保留当前胜者。
5. 最后相对**最初基线**再做一次 ABBA。未过门则回到最初基线；随后再次进行全部数值验证，重新核对模型/二进制/运行库指纹。
6. 使用最终 JSON/CFG 启动真实引擎验证。Worker 检查 RPC、NN 计数和 Drain；local 验证 GTP genmove 与 JSON analysis。成功后才写 `result/READY`。

FP32 参考覆盖的是完整 Worker 输入到后处理输出，不是各中间张量逐位证明，也不是对所有围棋局面的形式化正确性证明。默认 corpus 使用其固定策略参数；需要额外策略参数/模型结构认证时应另行扩展金标及检查器。

完整运行会反复加载模型和创建 CUDA Graph，可能耗时几十分钟至数小时；终端持续显示当前组、数值窗口及测速结果。Worker 默认每轮至少 128 次预热、1024 次计时请求；local 默认每轮 800 visits × 3 局面。不要用只跑几次推理的成绩替代完整流程。

## 5. 找到 PLAN 并接入

默认输出到 `autotune-output/full-<时间>-<模式>/`，末尾打印 `READY`、`PLAN`、`CFG` 的绝对路径。也可用 `-Output D:/tuning/my-run` 指定一个尚不存在的目录；不会覆盖旧目录。

```text
full-.../
  report.json                 # 总报告、环境、各候选结果、原始指标
  reference.json.gz           # 本次采用的 FP32 参考
  candidates/                 # 未认证的候选配置，仅供检查
  ...numeric.../              # 原始 NN 输出、对拍结果与路径日志
  ...                         # ABBA 测量日志
  result/
    READY                     # 成功发布标记
    plan.json                 # 你需要的 CUDA PLAN JSON
    rustgo.cfg                # 与 PLAN 成套的运行参数
    run-nnworker.ps1          # Worker 模式
    run-gtp.ps1               # local 模式
    run-analysis.ps1          # local 模式
```

同时满足 **`report.json.status = PASS` 和 `result/READY` 存在**，才使用结果。

- `search_scope = FULL_CATALOG`：运行了全部搜索组；具体 batch/线程范围仍以报告参数为准。
- `search_scope = CUSTOM_GROUPS`：人为缩小了搜索组，仅对该范围负责。
- `selection_status = IMPROVED`：最终候选相对初始基线通过收益和稳定性门。
- `selection_status = BASELINE_RETAINED`：没有接受提速配置，仍可生成通过数值/功能验证的基础 PLAN；不能宣称提速。

Windows 可直接执行输出脚本：

```powershell
& 'D:/tuning/my-run/result/run-nnworker.ps1' --server 192.168.1.10:50051 --worker-id gpu-2
```

或在干净的环境中直接调用引擎：

```powershell
& D:/RustGo/katago-rs.exe nnworker `
  --model D:/models/kata1-tf3-b11c768-s11001M-d5973M.bin.gz `
  --config D:/tuning/my-run/result/rustgo.cfg `
  --server 192.168.1.10:50051 --worker-id gpu-2 --capacity 32
```

Worker capacity 使用本次调优值。local 使用 `gtp --model ... --config ...` 或 `analysis --analysis-threads 1 --model ... --config ...`。当前 JSON analysis 等 EOF 后集中输出，实时分析用 GTP，详见[运行模式说明](RustGo运行模式说明.md)。

Linux 没有 PowerShell 时可在 Bash 子进程中清除实验环境变量，再用同一 cfg：

```bash
(
  for key in ${!KATAGO_@}; do unset "$key"; done
  /opt/rustgo/katago-rs nnworker --model /data/models/model.bin.gz \
    --config /data/tuning/my-run/result/rustgo.cfg \
    --server 192.168.1.10:50051 --worker-id gpu-2 --capacity 32
)
```

`plan.json` 保存 tactic 与身份；batch、NN 服务线程、搜索线程等实际运行值在配套 CFG 中。因此不要只复制 PLAN 却继续用另一组运行参数。CFG 中 `cudaTacticPlan` 是绝对路径；移动目录后需更新该路径及启动脚本路径，不要修改 PLAN 的模型/GPU/build 身份。换模型、GPU 或不兼容构建时应重新调优。

## 6. 换成另一个兼容 TF3 模型

内置参考仅匹配此原始文件 SHA256：

```text
1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6
```

不同权重文件即使网络大小相同，也不能共用它。可在任意一台装有 C++ CUDA FP32 **且支持本项目 `nnworker` 协议**的机器上生成新参考；普通上游 KataGo 若没有 `nnworker` 命令则不能用作此导出器。

```powershell
# 使用具备 grpcio/grpcio-tools/protobuf 的 Python。
python scripts/autotune_reference.py `
  --model D:/models/another-compatible-tf3.bin.gz `
  --cpp-worker D:/cpp-worker/katago.exe `
  --cpp-config configs/worker_cpp_fp32.cfg `
  --output D:/references/another-tf3-fp32.json.gz
```

`cudaUseFP16 = false` 必须在参考配置中明确设置。导出器验证完整 128 个请求、所有结果身份、数值合法性及来源信息；不会接受 Rust/dummy 输出当作 C++ FP32 金标。已有成功的 `compare_worker_outputs.py` 参考输出，也可用 `--source <cpp/outputs.json>` 代替 `--cpp-worker` 导出。

把模型和参考包带到目标机器后：

```powershell
.\scripts\full_autotune.ps1 `
  -Model D:/models/another-compatible-tf3.bin.gz `
  -Binary D:/RustGo/katago-rs.exe `
  -Reference D:/references/another-tf3-fp32.json.gz `
  -Mode worker -Capacity 32
```

也可在目标机直接传 `-CppWorker` / `-CppConfig`，缺少匹配参考时先自动生成。参考可以跨 GPU 复用，**优化 PLAN 不因此跨 GPU 通用**。金标仍会严格核对模型、协议、局面夹具和完整请求字节摘要；任一不匹配就停止。

剪枝模型的新二进制会通过 `cuda-fingerprint --model` 返回实际层数、通道数和逐层 FFN 宽度。FULL 自动跳过不适用的固定尺寸 DualFFN、fusion、split-K、K384 layout 候选，剩余候选仍逐一过相同数值门和 ABBA。剪枝不会降低认证要求。

## 7. 失败、中断与可调参数

| 现象 | 行为或处理 |
|---|---|
| 数值不合格 / top-1 未全过 | 候选淘汰，不测速，不放宽门槛；基础配置就失败则整体停止 |
| 路径标记缺失或静默 fallback | 候选淘汰，不把空转开关计作优化 |
| 候选 OOM、启动失败、超时 | 记录拒绝/失败并保留基线；基线不可运行则终止 |
| 波动超过 5% | 保留基线，不反复重测直到出现好数字 |
| Ctrl+C | 停止运行，保存中断报告，子进程清理；没有 READY 就不部署 |
| 最后配置验证失败 | 失败的最终文件移到 `staging/`，不会保留成功发布标记 |
| 锁文件已存在 | 同一个工具目录不允许并发 FULL 调优。若前次进程被强制结束，先确认其 PID 已退出，再删除 `autotune-output/full-autotune.lock` |
| 显存较小，B8 基线 OOM | 用 `-Batches 1,2,4,8` 从 B1 开始；首个 batch 是基线，其余为候选 |
| 旧卡没有匹配 PTX 目标 | 准备兼容该架构的二进制；自动调优不能补出编译时没有的 kernel |

当前不支持断点续跑；中断报告保留，下一次使用新输出目录。检查失败记录后调整明确的环境/参数问题，不将反复重跑当成性能收益证据。

Python 高级参数见 `python scripts/full_autotune.py --help`。可调整 `--batches`、`--threads`、`--capacity`、`--requests`、`--visits`、`--positions`、`--timeout`。收益门最低 1%、波动门最高 5%；可以收紧。`--ownership` 让 Worker 性能负载包含领地输出，数值验证始终包含领地。

## 8. 验证边界

交付测试包括 CPU 的能力筛选/数值前置/路径匹配/错误处理/发布原子性检查、内置金标的完整身份与数值合法性检查，以及本机实际 CUDA 的端到端运行。最初开发验证使用 `--groups` 明确缩小范围；之后额外完成了下面的本机 local 完整运行。这些结果不代表所有 CUDA GPU 都经过实机验证。

2026-09-20，RTX 5070 Ti / 当前 TF3 模型实测：Worker 的 `dual_ffn` 子集与 local 的 `attention,threads` 子集均完成数值验证、ABBA、最终复验及发布。报告分别位于工作区 `target/full-autotune-check/worker-integration/report.json`、`target/full-autotune-check/local-integration/report.json`。这些开发报告不包含在便携工具包中，也不作为其它机器的性能承诺。

同日按用户要求完成 **local 默认全部 14 组**，报告为 `autotune-output/full-local-20260920-104706/report.json`，状态 `PASS / FULL_CATALOG / IMPROVED`。最终 B16 / 单 NN 服务线程 / 16 搜索线程、Graph 开启、q64-serial + DualFFN；相对本轮初始 B8 / 8 搜索线程基础配置，最终配对 ABBA 为 **837.33 → 1045.49 visits/s（+24.86%）**，两侧波动 0.24% / 2.22%。这是本轮基线比较，不是相对 Worker 配置的收益。

完整运行发现并修复了路径检查器的两处契约错误：T32/T64N32 的实际日志名称没有 `_kernel` 后缀；compact FFN 使用独立执行标记，不打印普通 DualFFN 的执行标记。原始报告及脚本快照保留，三个误判候选在 `path-correction/report.json` 中按原决策点重新过数值门和 ABBA，均未胜出，因此不改变完整运行的选择。新增两项回归测试，FULL 调优测试共 16 项通过。

本机独立交付位于该运行目录下的 `local-files/`：`plan-local.json`、`rustgo-local.cfg`、`run-local-gtp.ps1`、`run-local-analysis.ps1`。另命名并更新计划路径后，再次通过 GTP/analysis；`deployment-local.json` 记录来源和 SHA256，`READY-LOCAL` 表示交付验证成功。现有 11 份 Worker 计划/配置 SHA256 均保持原样。

每个目标机器仍需自行运行上述不带 `--groups` 的完整命令，以该次报告决定结果。本工具不覆盖现用引擎、`plans/` 或 `configs/`，也不改变数值精度边界。
