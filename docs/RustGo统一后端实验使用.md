# RustGo 统一后端实验使用

2026-09-24。统一入口为 `nnBackend=cudaquantbackend`：省略 `cudaQuantPlan` 使用全 FP16，提供配方则按投影执行 FP16 / INT8 / 实验 MXFP8。兼容当前支持的原生 B11、剪枝 B15 与现有 ONNX 层图，限 19 路；不支持的结构明确报错。本说明使用独立 r9 二进制，不替换正式二进制或认证计划。

本机验证环境为 Windows、RTX 5070 Ti、CUDA 13.3、cuBLASLt 13.6、驱动 610.88。以下配置用于接口诊断，`maxVisits=1`、单搜索线程和关闭缓存不是日常对弈建议；实际分析需要按负载设置搜索参数。

## 1. 在专用 PowerShell 中准备

以下路径对应本机工作区；所有输出放入新目录。清理环境只影响当前 PowerShell 及其子进程，避免继承历史 tactic。

```powershell
Set-Location D:/code/Rust_KataGo
$exe = (Resolve-Path target/unified-quant-g3/frozen-r9/katago-rs.exe).Path
$sha = '40f1c226ff6f0e7b0a17892ee0de111fb1092afe04d5842f7e5862518e908dd0'
if ((Get-FileHash $exe -Algorithm SHA256).Hash -ne $sha) { throw 'r9 binary SHA mismatch' }
Get-ChildItem Env:KATAGO_* | Remove-Item
$run = Join-Path (Get-Location) ('target/quant-user-' + (Get-Date -Format yyyyMMdd-HHmmss))
New-Item -ItemType Directory -Path $run -ErrorAction Stop | Out-Null
$model = (Resolve-Path models/b11c768h12nbt3tflrs-fson-silu.bin.gz).Path
$cfg = Join-Path $run 'quant.cfg'
@'
rules=chinese
komi=7.5
nnBackend=cudaquantbackend
nnMaxBatchSize=8
numNNServerThreadsPerModel=1
numSearchThreads=1
maxVisits=1
ponderingEnabled=false
nnCacheSizePowerOfTwo=-1
nnMutexPoolSizePowerOfTwo=8
'@ | Set-Content $cfg -Encoding ascii
```

换模型时修改 `$model`，并重新生成配方。已有配方只适用于完全相同的模型文件 SHA，文件名相同也不能替代核验。

| 模型 | 本机模型路径 | 已有全 FFN INT8 配方，相对仓库根目录 |
|---|---|---|
| B11 | `models/b11c768h12nbt3tflrs-fson-silu.bin.gz` | `target/unified-quant-g1/worker-b11-trace/recipes/new-ffn.json` |
| 剪枝 B15 | `models/b15-ffn-pruned-a8.bin.gz` | `target/unified-quant-g1/worker-b15-trace/recipes/new-ffn.json` |
| 现有 ONNX | `D:/code/b11fix.onnx` | 从该文件导出模板；不要套用原生 B11 配方 |

## 2. 先使用 FP16，再显式选择 INT8

`quant-inspect` 只用 CPU，输出完整投影 ID、形状与 FP16 模板；输出目录必须不存在。下面的 GTP 是交互入口，输入 `quit` 退出；随后 `nnbench` 是短冒烟测量，不能作为 ABBA 性能结论。

```powershell
& $exe quant-inspect --model $model --output "$run/inspect"
& $exe gtp --model $model --config $cfg
& $exe nnbench --model $model --config $cfg --mode eval --batch 8 --workers 8 --warmup 10 --iterations 50
```

采用表中与模型匹配的全 FFN INT8 配方，attention 投影仍为 FP16。先通过 CPU 核验，再启动；下面示例对应 B11，B15 请替换配方路径。

```powershell
$recipe = (Resolve-Path target/unified-quant-g1/worker-b11-trace/recipes/new-ffn.json).Path
& $exe quant-inspect --model $model --recipe $recipe --output "$run/checked-int8"
if ($LASTEXITCODE -ne 0) { throw 'Recipe validation failed' }
& $exe gtp --model $model --config $cfg --override-config "cudaQuantPlan=$recipe"
```

自选投影时编辑 `$run/inspect/fp16-recipe.json` 的副本：保留 schema/version、模型与图 SHA、`id`、`expected_n/k`，只将目标 `precision` 改为 `int8`。例如将第一个 FFN 双上投影设为 INT8：

```powershell
$plan = Get-Content "$run/inspect/fp16-recipe.json" -Raw | ConvertFrom-Json
$projection = $plan.projections | Where-Object { $_.id -like '*.ffn.dual' } | Select-Object -First 1
if ($null -eq $projection) { throw 'No FFN dual projection in this model' }
$projection.precision = 'int8'
$recipe = Join-Path $run 'custom-recipe.json'
$plan | ConvertTo-Json -Depth 20 | Set-Content $recipe -Encoding ascii
& $exe quant-inspect --model $model --recipe $recipe --output "$run/checked-custom"
if ($LASTEXITCODE -ne 0) { throw 'Recipe validation failed' }
```

随后使用上一条 GTP 命令消费 `$recipe`。未列出的投影默认 FP16；未知/重复字段、错误形状或 SHA 均拒绝。`--recipe` 导出额外的 `recipe-identity.json` 与原字节 `source-recipe.json`，是加载身份核验，不是精度认证。不要混入 `cudaTacticPlan`、`cudaInt8Scope` 或 `cudaInt8MinFfnWidth`；统一后端会拒绝它们。`direct/kernel` 不能核验 `cudaQuantExpectedProfile`，需要身份核验时使用 `eval`/GTP。

## 3. 隔离本机 Worker 诊断

下面采集器启动临时本机 gRPC 服务和 Worker，先通过 GTP 发现实际 profile，再写入 `cudaQuantExpectedProfile` 并核对 Worker；无需改造或连接生产 Server。使用本机已具备 gRPC 依赖的 Worker Python 环境，并显式选择连续补充请求模式。当前请求文件仅包含一盘棋的 32 个 calibration 局面。`capacity` 是在途请求上限，`batch` 是 NN 容量，窗口不得超过 capacity。

```powershell
$requests = 'target/ptq-corpus-two-games-r2/calibration.requests.jsonl'
$python = 'D:/Go/Server/worker/.venv-windows/Scripts/python.exe'
& $python scripts/collect_quantization_corpus.py --binary $exe --model $model --requests $requests --output "$run/worker-fp16" --backend cudaquantbackend --batch 8 --capacity 32 --window 1 --window-mode continuous
& $python scripts/collect_quantization_corpus.py --binary $exe --model $model --requests $requests --output "$run/worker-candidate" --backend cudaquantbackend --recipe $recipe --batch 8 --capacity 32 --window 1 --window-mode continuous
& $python scripts/compare_quantization_corpus.py --reference "$run/worker-fp16" --candidate "$run/worker-candidate" --output "$run/worker-comparison"
```

`rustgo-quant-v1:<sha>` 绑定模型/配方、二进制、设备/库/驱动、实际 tactic 与 batch 配置；变更这些条件须重新获取身份。Worker 显式 ExpectedProfile 不能代替生产 Server 的任务、缓存与精度身份隔离，本版没有实现该协议协商。

## 4. 当前边界与证据

- MXFP8 已接通，但 r8 全 FFN MXFP8 在本机 B11/B15 的 C1、C32 诊断均慢于同版本 INT8，不推荐默认启用。见 [r8 汇总](../target/unified-quant-g3/mxfp8-vs-int8-summary-r8.json)。实验 v2 配方见 [B11](../target/unified-quant-g3/cpu-recipe-exports-r7/b11-ffn-mxfp8/source-recipe.json)、[B15](../target/unified-quant-g3/cpu-recipe-exports-r7/b15-ffn-mxfp8/source-recipe.json)；必须由当前模型重新 CPU 核验，不能仅把 v1 的精度字符串改成 `mxfp8`。
- `KATAGO_CUDA_MXFP8_SCALE_CLEAR_FUSION=0|1` 默认 **0**；r9 算子/整网逐位门已过，仅 B11 C1 确认约 1.5% 主指标速度比收益，B15 C1 与两模型 C32 均未过收益门，因此不改变默认值。该开关仅用于 MXFP8；collector 可用 `--mxfp8-scale-clear-fusion 0` 或 `1` 显式记录并核验实际路径。
- 本机实卡为 RTX 5070 Ti；5060 Ti、5080 尚未实卡验证。本页短命令仍使用一盘 32 点 calibration 的诊断语料，其 [manifest](../target/ptq-corpus-two-games-r2/manifest.json) 为 **NOT_READY**。另有完整语料的正式 selection C1 流程：B11 的 2004 局面/32 盘、4 份采集与 5 组 ABBA 已独立审计通过，统一全 FFN INT8 最大胜率差 5.30 个百分点，但没有统一候选通过完整性能矩阵；B15 执行中。两类旧控制已接入自动流程，使用方法见[离线选型](RustGo统一后端离线选型.md)；最终留出和发布尚未完成。NVFP4、Elo 不是使用本版入口的前置条件。
- 完整规格、误差门与阶段证据见 [统一推理优化量化后端](RustGo统一推理优化量化后端.md)；正式使用入口仍为 [使用指南](RustGo使用指南.md) 与 [性能优化结项记录](RustGo性能优化结项记录.md)。本页 `target/` 产物属于本机实验归档，复制到别处须一并保留二进制、模型和配方并复核 SHA。
