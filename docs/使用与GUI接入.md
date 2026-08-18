# Rust_KataGo 使用与 GUI 接入说明

> 2026-08-16。适用二进制：`target/release/katago-rs.exe`（`--features cuda`，release）。
> 模型：`D:/code/b11fix.onnx`。硬件：RTX 5070 Ti（SM120，70 SM）。

## 1. 优化配置（推荐起点）

生产配置：`configs/gtp_cuda.cfg`，核心三行：

```
nnBackend = cudabackend
cudaTacticPlan = D:/code/Rust_KataGo/plans/best-tactic-plan.json
nnMaxBatchSize = 16
```

- `cudaTacticPlan` 加载 2026-08-16 autotune 认证的 tactic plan（8 决策组
  ABBA 全量裁决）。fail-closed：换 GPU/换模型/换机器时启动即报错，此时
  重跑 `python scripts/autotune.py --threads both` 重新认证，或临时删掉
  该行退回代码默认值（两者当前等价）。
- `nnMaxBatchSize=16`：实测 B16 之后每行成本转劣，线程再多也别放大。
- 不带 `cudaTacticPlan` 也完全可用：所有 tactic 默认值就是认证结果。

线程数按用途选（实测 v=1500，认证 plan）：

| 用途 | numSearchThreads | 实测 |
|---|---|---|
| GUI 对局/分析（人机对战） | 12–16 | ~1160 visits/s，响应最快 |
| 批量自对弈/纯吞吐 | 48 | ~800 nnEvals/s 峰值 |
| 后台挂着（低占用） | 4 | 549 nnEvals/s，占用低 |

**高并发饱和服务（2026-08-18 新增）**：分析引擎多查询/批量评估等供数
持续 ≥4×batch 的场景，双 NN server + 无图直发再 +9.5%：

```
--override-config numNNServerThreadsPerModel=2   # + 环境变量 KATAGO_CUDA_NOGRAPH=1
```

（eval B16 W64：1112 vs 单流 1016。注意仅限饱和供数——GTP 对局等搜索
语境**不要**用：双流在搜索是负收益甚至灾难性回退。）

**WSL 部署**：同硬件 Linux 口径比 Windows 再 +2~5%（搜索 +4.6%），测量
也稳定得多；环境一键脚本 `scripts/wsl_setup.sh`，基准 `scripts/wsl_bench.sh`。

## 2. 启动命令

```
D:/code/Rust_KataGo/target/release/katago-rs.exe gtp ^
  -config D:/code/Rust_KataGo/configs/gtp_cuda.cfg ^
  -model D:/code/b11fix.onnx
```

**分析命令为流式输出**（2026-08-16 起，对齐官方 KataGo 语义）：
`lz-analyze`/`kata-analyze` 带 interval（厘秒）时，info 行按 interval 持续
刷新（winrate/score/pv/ownership 逐帧更新），直到 `stop` 或下一条命令；
GUI 里的胜率曲线与领地热图实时变化。不带 interval 则保留一次性输出语义
（跑满 maxVisits/maxTime 输出一帧，供脚本批量用）。`lz-genmove_analyze`/
`kata-genmove_analyze` 同样在搜索期间流式输出、落子后返回。

## 3. 接入围棋 GUI（标准 GTP 引擎方式）

任何支持 GTP 引擎的 GUI 都可接。以常见三款为例：

### Sabaki（分析 + 对局，推荐）

1. 引擎设置 → 添加引擎：
   - 路径：`D:/code/Rust_KataGo/target/release/katago-rs.exe`
   - 参数：`gtp --config D:/code/Rust_KataGo/configs/gtp_cuda.cfg --model D:/code/b11fix.onnx`
2. 附加命令（可选）：`boardsize 19`、`komi 7.5`。
3. 对局直接开；分析模式用引擎的 analyze（Sabaki 发 `kata-analyze`，
   本实现支持，一次性输出一帧）。

### KaTrain（陪练/教学）

- 设置 → 引擎 → 添加，路径同上，参数 `gtp`（KaTrain 传自己的 --config）。
- KaTrain 需要它自带配置模板时，把上面三行核心配置粘进它的引擎配置，
  或直接引用 `configs/gtp_cuda.cfg` 后再补 KaTrain 特有键。

### Lizzie / LizzieYzy（Leela Zero 系界面）

- 引擎配置选 GTP，命令行同 Sabaki；`lz-analyze` 支持（info 含
  winrate/scoreLead/pv，与 LZ 格式兼容）。

### 通用核对清单（任何 GUI）

| GUI 发的命令 | 支持情况 |
|---|---|
| boardsize/komi/play/undo/genmove | ✅ 标准 |
| time_settings/kata-time_settings/kgs-time_settings | ✅ |
| kata-get/set-rules、kgs-rules | ✅ |
| kata-analyze / lz-analyze | ✅ 流式（带 interval 持续刷新帧 + ownership 热图） |
| kata-genmove_analyze / lz-genmove_analyze | ✅ |
| kata-raw-nn | ✅（调试用） |
| set_position / clear_cache / final_score | ✅ |

## 4. 常见问题

- **启动报 `cudaTacticPlan ... mismatch`**：GPU/模型与认证时不符。重跑
  autotune 或删掉该配置行。
- **引擎无输出就退出**：确认 `gtp` 子命令在前、`--model` 指到
  b11fix.onnx；CUDA 版二进制需在本机跑（无 CUDA 时改 `nnBackend =
  dummybackend` 只能冒烟，不能真实推理）。
- **GUI 分析无胜率**：确认命令带 interval（如 `lz-analyze B 20` = 每
  200ms 一帧）；无 interval 是一次性输出语义（需 maxVisits/maxTime）。
- **分析帧里无领地热图**：kata-analyze 加 `ownership true` 参数。
- **想看引擎确认加载了 plan**：GTP 启动横幅有
  `cudaTacticPlan '...' installed (plan id: ...)`。
- **性能复核**：`benchmark --config configs/gtp_cuda.cfg --model
  D:/code/b11fix.onnx -v 1500 -n 1 -t 12,48`（务必带 -t）。
