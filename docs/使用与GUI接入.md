# Rust_KataGo 使用与 GUI 接入说明

> 更新说明（2026-09-16）：本文保留 `b11fix.onnx` 的 GTP 用法和历史性能数据。原生 TF3 与 Go Server Worker 请看[使用指南](RustGo使用指南.md)；本轮优化已[结项](RustGo性能优化结项记录.md)，无需为正常使用重跑性能测试。

> 2026-08-19。适用二进制：`target/release/katago-rs.exe`（`--features cuda`，release）。
> 模型：`D:/code/b11fix.onnx`。硬件：RTX 5070 Ti（SM120，70 SM）。

## 1. 优化配置（推荐起点）

生产配置：`configs/gtp_cuda.cfg`，核心三行：

```
nnBackend = cudabackend
cudaTacticPlan = D:/code/Rust_KataGo/plans/best-tactic-plan.json
nnMaxBatchSize = 16
```

- `cudaTacticPlan` 加载 2026-08-19 schema-2 认证 plan（q64 attention + DualFFN，
  Windows/WSL 各有独立 build fingerprint）。fail-closed：换 GPU/模型/机器、
  重编 CUDA/CUTLASS 或跨平台复用时启动即报错；此时重跑
  `python scripts/autotune.py --threads both` 重新认证，或临时删掉该行退回
  代码默认 tactic（q128，性能不同）。
- `nnMaxBatchSize=16`：实测 B16 之后每行成本转劣，线程再多也别放大。
- 不带 `cudaTacticPlan` 会使用内置默认 tactic，并不等于认证组合，也不能继承认证性能。TF3 不能使用本页的 ONNX plan。

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

**WSL 部署**：旧ONNX版本曾记录与Windows不同的性能；不能据此推断当前TF3组合在WSL更快。Windows/WSL需要各自匹配的计划与构建身份。`scripts/wsl_setup.sh` 等脚本保留为部署参考，本轮不继续测试WSL。

## 2. 启动命令

```powershell
Set-Location D:/code/Rust_KataGo
.\target\release\katago-rs.exe gtp `
  --config configs/gtp_cuda.cfg `
  --model D:/code/b11fix.onnx `
  --override-config virtualLossUtilityBlend=1.0
```

现有 `gtp_cuda.cfg` 文件末尾保留 `virtualLossUtilityBlend=0.0` 实验设置；上例显式覆盖为1.0，使用本文下方说明的标准搜索行为，不改变认证NN计划。

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
   - 参数：`gtp --config D:/code/Rust_KataGo/configs/gtp_cuda.cfg --model D:/code/b11fix.onnx --override-config virtualLossUtilityBlend=1.0`
2. 附加命令（可选）：`boardsize 19`、`komi 7.5`。
3. 对局直接开；分析模式用引擎的 analyze（Sabaki 发 `kata-analyze`，
   本实现支持；带interval时持续输出直到stop或下一条命令）。

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

### 后台思考(pondering)与选点分散

- **对局场景**:`--override-config ponderingEnabled=true` 可在等待对手落子期间后台预搜,
  落子后即刻响应并大幅缓解"走几步后新局面首帧候选集中"(树已在后台展开)。
  未配置任何 pondering 限额(`maxVisitsPondering`/`maxPlayoutsPondering`/`maxTimePondering`)
  时自动套 60 秒/手的安全上限;如需更长可显式设 `maxTimePondering`(秒)。
- **分析场景(Lizzie/Sabaki 等)**:保持默认 `ponderingEnabled=false`(GUI 频繁 play/undo
  会让后台搜索持续占 GPU)。分析时想看到更多候选点,用
  `--override-config analysisWideRootNoise=0.25` 左右(官方语义"强制分析更多样的着法",
  0.04 为默认微扰,1.0 会摊到全盘,推荐 0.2-0.3)。
- **跨手洞见(EvalCache)**:`gtp_cuda.cfg` 已默认 `useEvalCache=true`——交互分析往深处走、
  解决某个盲点后回退到早先局面,引擎会记住该分支的解,早先局面的搜索更容易顺着已解
  分支继续深挖(官方 v1.16.4 机制;`clear_cache` 命令可整体清空)。
- **实验开关 `puctVarExploration`(PUCT-V 子级方差探索,默认 0 关闭)**:探索项按子节点
  效用方差加权。实测在调优过的 KataGo 权重下会**加剧**选点集中(方差项被优势子自增强),
  且与既有父级方差缩放重复计数——目前不推荐开启;留作低 visit 场景后续实验,
  详见 `docs/搜索算法调研与改进计划.md` P2-6。
- **实验开关 `virtualLossUtilityBlend`(WU-UCT 模式虚拟损失,默认 1.0 = 官方)**:多线程
  搜索的虚拟损失由两半组成——子权重膨胀(通缩探索项)+ 效用向败势混合。该参数缩放后者:
  1.0 逐位同官方;0.0 = 纯 WU-UCT(Liu et al. 2018)——保留权重膨胀、不扭曲效用,
  允许多线程同时深挖明显最优分支。单线程完全无感。实测 12 线程:固定 visit 口径棋力
  中性(ABBA 50.0%),但吞吐约 -30%(冗余 NN eval)——**不推荐开启**,时间预算对局
  口径下为净亏,详见 `docs/搜索算法调研与改进计划.md` P2-11。

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
