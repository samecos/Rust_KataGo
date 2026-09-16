# CUDA 对齐/超越 fork 路线图(cuda-fork-parity-plan)

> **本轮已结项（2026-09-16，用户确认）**：保留当前认证版本，停止追加测试、实验与自动优化；[结项记录](RustGo性能优化结项记录.md)是当前交付状态，[使用指南](RustGo使用指南.md)是运行入口。本文其余部分保留研究过程；历史“active/下一步”、开放问题及未测方向不再是本轮待办，也不表示其已通过验证。

> 目标:将本仓库 CUDA 后端推理能力提升并超越 KataGomo_fork 的 SM120 认证水平。
> 本文档是跨对话对齐用的**主规划**;战术史与 ABBA 留痕仍在
> `docs/cuda-optimization-plan.md`,fork 技术细节在 `docs/fork-sm120-kernel-notes.md`。
> 每个新会话开始工作前:先读本文件 §1(目标)、§2(现状快照)、§8(进度看板)。

## 1. 目标与验收口径

- **当前对标（2026-09-08 核查）**：本机 RTX 5070 Ti（70 SM、48 MiB L2），
  原生 TF3 模型 `kata1-tf3-b11c768-s11001M-d5973M.bin.gz`，SHA-256
  `1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6`。
  fork 已有同 GPU、同 SHA 的固定 B14/S2 认证方案，记录值 **2418.947898**。
  证据：fork `final-migration/plans/sm120/rtx5070ti-b14-s2/README.md:3`、
  `best-tactic-plan.json:1240`。这是历史认证参考；本轮同卡、同计时边界的稳定
  空盘基线已完成，见 §8 G0，精度与输出布局差异仍须单列。
- **2418.9 的计时边界**：fork 对各并发 lane 的 CUDA event 前向耗时取中位数，
  再计算 `sum(batch / lane_median_seconds)`；不是 gRPC Worker 吞吐，也不是
  timed rows 除以统一主机墙钟。H2D 在计时循环外，event 只包围 `apply`。
  证据：fork `python/sm120_benchmark_metrics.py:17`、
  `cpp/neuralnet/nneval.cpp:1231`、`cpp/neuralnet/cudabackend.cpp:5481`、`:5519`。
  `actualWallSeconds` 另含 warmup 等阶段（`nneval.cpp:1224`），不能直接当作
  剔除预热的生产吞吐。先统一边界再比较，严禁用 2418.9 除 Rust Worker rows/s。
- **验收分层**：GPU 前向、含拷贝的 direct、evaluator、gRPC Worker、搜索分别报告。
  每层固定模型 SHA、物理 batch、lane 数、graph、请求/缓存条件和环境；单 GPU
  顺序 ABBA。G0 先建立同卡、同边界且明确精度差异的参考基线；同精度性能归因
  另做受控验证，最终以真实 Worker 的 uncached
  NN rows/s 与延迟验收。旧 RTX 5080 B16/S2 的 2836、`2836×70/84≈2360`
  和 `2836/2≈1418` 只保留为历史推算，**均不再是验收线或实测单流速度**。
- **精度边界**：Rust CUDA 已是混合精度：FP16 权重和多处中间存储，GEMM/QK/PV
  FP32 累加，激活/归约 FP32，主干残差及最终输出 FP32。fork B14 的 `qk16`
  是 QK FP16 累加、PV FP32 累加，且选中的 linear2 CUTLASS 残差 GEMM 也用
  FP16 累加；2026-09-15 实际加载 SASS 又确认 FFN 两个第一投影、attention
  输出投影和外层投影均为 FP16 累加，外层权重还含通道缩放变换。同原始模型
  SHA 不代表内部权重位模式一致（详见 §8 对应审计；QK/linear2 源码见 fork
  `fa4_aot/build_aot.py:150`、
  `sm120_aot/linear2_residual_cutlass.cu:16`，
  两者相对 `cpp/neuralnet/`）。其 8192 行认证 top-1 为 99.8046875%，门槛 99.5%
  （plan `:1179`、`:1230`），不同于本仓库 100% 门；本轮借鉴布局/组织，保留
  Rust 现有累加与存储边界，不据 fork 认证直接接受降精度。
- **本仓库测量工具**：`katago-rs nnbench` 固定物理 batch；`--mode eval` =
  evaluator 链路（不含 gRPC），`--mode direct` = 含拷贝的 `CudaModel::apply`
  直连，`--mode kernel` = 纯前向(无拷贝无 graph)。kernel 新增
  `--timing cuda-event --handles 1,2 --input empty`，逐 lane event 中位数已与
  fork 对齐；独立保留墙钟统计，不混算比例。注意:eval 的 `--workers` 默认 2×**当前** batch
  ——serve_pipelined 的 target 是发射阈值非上限,W/3>target 时 avgBatch 被
  抬到 ~W/3(2026-08-17 实测修复)。
- **direct/kernel 参数边界（2026-09-09 核查）**：当前 `nnbench` 的这两条
  路径直接创建 CudaModel，不加载 `--config`/`--override-config` 中的 plan。
  对拍脚本须显式传入冻结 plan 对应的环境 tactic，并核实际 JSON/路径；
  不能将“传了配置文件”写成“已应用计划”。`--handles` 是 ID 列表，
  `--handles 1,2` 才是两 lane，单写 `2` 仅创建 ID 2 的一 lane。
  G0/G2 既有前向脚本及原始日志已复核，均正确传入环境开关和两 lane。
- **正确性门**(任何性能变更的前置):
  `cargo test -p kata_nn --test dump_nn_io_cuda --features cuda --release` →
  `.venv/Scripts/python.exe scripts/compare_nn_output.py <dump目录>`,
  RESULT: PASS(policy 5e-2 / value 2.5e-2 / misc 1e-2 / ownership 5e-3 /
  top-1 100%);启用休眠代码路径前同样先过对拍。
- **性能裁决**:ABBA/BAAB,min_improvement 1%,慢于基线即回退;新 kernel
  一律注册为 tactic 候选经 `scripts/autotune.py` 裁决并入
  `plans/best-tactic-plan.json`(fail-closed)。

## 2. 现状快照(2026-09-08)

本轮最新安装状态（2026-09-16）：Attention输出投影classic TN N128已在现用
QKV N128+compact FFN之上纳入Windows B14/S2/C64吞吐计划。完整前向共同墙钟
**+1.456147%**，真实Worker **1557.022297→1574.822297 RPC/s
（+1.143208%）**，均通过1%收益及5%波动门。整网612,960值逐位相同、
真实Worker768/768首选及11个C++FP32字段、旧ONNX/graph/默认与C32兼容、
六计划迁移及正式路径复核通过。仅throughput显式启用OUTPROJ_CLASSIC_N128_R1；
FP16存储/FP32累加、激活、归约与残差输出保持，其余五计划显式关闭新键。
新一轮同边界ABBA：Rust **1595.982821**、Fork测试副本 **1563.247273 NN rows/s**，比值 **102.094074%**；两臂spread **0.176416% / 1.125760%**。这是Windows Rust与WSL Fork测试副本的统一墙钟参考；平台和累加精度仍有差异，不代表同精度优势或Fork Worker吞吐。
本轮已按用户确认完成并结束；以下组合热点和阶段记录保留作历史依据：FFN up19.273%、QKV17.348%、
Attention核心15.480%、FFN down14.287%，均为累计内核时间占比。
M64/N128三级流水输出投影虽过组件、整网及真实Worker数值门，唯一完整
前向共同墙钟稳定仅+0.025736%，未达1%门，已关闭，未测Worker性能或采纳。
两级寄存器流水虽通过整网612,960值逐位与Worker768/768原数值门，
唯一完整前向共同墙钟稳定−0.518775%，已关闭，未测Worker性能或采纳。
显式异步两级流水通过整网612,960值逐位及真实Worker768/768原门，
唯一完整前向共同墙钟−0.240451%，两臂波动通过5%门但未达1%收益，已关闭。
三项M64/N128均未采纳。compact DualFFN寄存器half8重排已过整网612,960值
逐位、真实Worker768/768与全部路径/身份门，但唯一完整前向共同墙钟
稳定+0.480137%，未达1%收益门，已关闭，未测Worker性能或采纳。
原DualMma直接half2写出也通过整网612,960值逐位、真实Worker768/768
与全部路径/身份门，但唯一完整前向稳定+0.742377%，未达1%收益，已关闭。
现用版本保持。重复融合配置/日志的CPU审计已完成：179层358→1次分配，
但未建立整网收益证据，未登记引擎候选。热点覆盖核对已完成；新的Attention
丢弃尾块裁剪虽过整网612,960值逐位与Worker768/768原门，唯一完整前向
稳定−0.262664%，已关闭，未测Worker性能或采纳。FUSION之外的主机控制CPU
审计也已完成：按现用路径重放305次tactic查询及180次直接环境查询，计划模式
489次临时分配、每lane61.642–63.466微秒；这不是实际前向主机耗时或整网收益。
未登记引擎候选。现用Worker原始32768条计时及一次普通权限时间线已复核：
82批/1024行与心跳精确对应，连续B14段内核活动覆盖98.3591%。164次输入
拷贝均为锁页内存；15次长API驻留大多与内核执行重叠，不能当作可消除时间。
独立上传流候选已完成：实际Backend540行、Worker768/768与11字段门、
B1/B14/B16原始全头306,480值逐位通过。检查器连接问题已解决，开关两臂
完整报告均无内存错误诊断且泄漏0；非零API报告逐条确认是函数查找失败。
唯一包含传输的Backend ABBA稳定-0.990064%，未达1%收益门，候选关闭；
未测Worker性能、未采纳，正式56项文件不变。随后显式两级compact DualFFN
保持原分块/精度，约2.28亿half组件值逐位及memcheck/racecheck通过；唯一
33层上投影ABBA event/共同主机稳定-8.217018%/-8.328185%，
已关闭，未做整网或Worker接入。随后三级FFN延后等待的机器码仍有53/64条
MMA在等待后，未实现预设顺序，GPU执行前关闭。随后compact down TN N128
通过23种K参数、34组数值逐位及内存/竞态/路径门；唯一算子ABBA主机
稳定+8.523185%，但event名义−11.974102%且基线波动6.708339%超门，
按原双指标门关闭，未做整网/Worker接入。随后compact down NN布局组件
主机+4.619976%/event−4.627401%，原失败保留；独立隔离整网612,960值
逐位、Worker768/768及路径/兼容门通过，唯一完整前向主机+1.264881%、
event+0.430978%，均稳定但event未达1%，按原双指标门关闭。未测Worker
性能或采纳，现用状态保持，详见§8。

上一轮安装状态（2026-09-16）：classic N128/K32 QKV已纳入Windows
B14/S2/C64吞吐计划。保留FP16存储/FP32计算和原标量RoPE，整网612,960值
逐位一致、真实Worker768/768及十一FP32字段通过；唯一完整前向共同墙钟
**+1.784916%**，真实Worker **1545.265623→1564.201170 RPC/s（+1.225391%）**，
两层均通过5%稳定门。旧ONNX/graph/默认与C32兼容检查、6计划迁移与正式路径
复核全部通过。只在吞吐计划启用QKV_CLASSIC_N128_R1，其余五计划显式关闭。
50个现用文件已有验收指纹和原件备份。随后完成新的同边界ABBA：Rust
**1572.689169**、Fork测试副本 **1569.773023 NN rows/s**，比值**100.186%**，
墙钟spread0.509%/1.235%通过5%门。名义差小于波动，只能称基本持平；Fork
event spread13.858%不稳定，未发布event速度差。Windows/WSL与精度差异保持
单列，不是原Fork认证ELF或Worker对照。Linux N128隔离版本已完成ABI、45项CPU、
组件/sanitizer、六组整网及768请求真实Worker数值验证，但唯一前向ABBA候选
spread10.003%超过5%门，已关闭性能接入，没有Worker测速或生产迁移。
随后默认N192因输入/输出映射不完整在CPU阶段关闭；N128的warp32×64与
warp64×32两种八warp布局虽过完整数值门，唯一前向共同墙钟−0.178%/+0.367%
未达1%门，已关闭。classic N256也过完整映射、组件/整网及Worker768/768
原门，但唯一前向ABBA稳定−4.877%，关闭、未测Worker性能。现用
N128+compact FFN热点已刷新：内核累计FFN up19.666%、QKV17.524%、
Attention核心15.458%、FFN down14.232%。M256×N128 QKV也通过组件、整网
612,960值逐位及Worker768/768原门，但唯一前向ABBA稳定−3.344%，已关闭。
K循环审计确认K16直接替换违反管线约束；源码延后等待候选的实际wait位置
仍相同，已在机器码机制检查后关闭、未测速。固定K384三stage分组实际减少
循环指令且通过全部数值门，但唯一前向+0.190%未达1%，已关闭。随后RMS
十六lane双元素方案也已过组件、整网612,960值逐位与Worker768/768原门，
但唯一前向共同墙钟−0.428%，未达1%门，已关闭。随后Attention输出
投影classic TN N128通过组件139,732,992个FP32值逐位及sanitizer检查，
保留FP32累加/残差输出；尚未做整网、Worker或性能验证，下一步隔离接入，详见§8。

上一轮FFN安装记录（2026-09-15）：Windows B14/S2/C64 throughput 已在
strict Attention、stem三维映射、C768四通道门和只读QKV/RoPE之上，启用
**既有half全零FFN通道压缩**。同新CLI唯一Worker ABBA为
**1270.22→1546.71 RPC/s（+21.766825%）**，完整前向event **+23.940638%**、
统一主机 **+23.008092%**，均稳定通过。原FP16存储/FP32累加及精确激活
纪律不变；只删除上传后已是half零的通道，整网输出差异通过原门。
33层打包/整网/真实Worker数值、旧ONNX/graph/默认和C32兼容门均通过。
新CLI与六Windows计划已安装，最终六计划加载及吞吐128请求/Drain通过。
新ffn_compact_artifact绑定代码和权重映射，仅throughput计划启用；
其他五份显式关闭，WSL保持原值。历史QKV认证1274.991675及其收益保留，
不与本轮局部增幅相加。详见§8最新记录与TF3性能审计。
压缩后轻量时间线显示Attention三段累计约40.270%、FFN约34.810%；
此为内核累计诊断口径。新compact down Lt21候选数值逐位通过，但算子
event ABBA稳定下降6.935%，已关闭，未变更认证版本。

上一版Windows对Fork统一前向参考（2026-09-15，历史）：同模型/同卡/B14双流、共同墙钟下，
Rust **1555.98**、Fork测试副本 **1588.30 NN rows/s**，约 **98%**；两臂
波动0.106%/4.157%，通过预设5%门。Fork副本加入与Rust一致的完成后收尾
屏障，因此这是统一测试流程参考，不替换原认证ELF的历史成绩。名义2%
差距小于Fork自身波动，不作精确性能归因。输入逐字节对齐，平台/累加
精度差异仍单列；认证Worker1546.71 RPC/s保持，见§8最新记录。

上一版同Linux参考（2026-09-15，未含N128）：完整压缩栈已在隔离WSL副本通过
B1/B14/B16原生及W1/W32/W64真实Worker数值门。统一墙钟ABBA为Rust
**1383.35**、Fork测试副本 **1552.98 NN rows/s**，约 **89.08%**；两臂
波动4.666%/2.141%，通过5%门。累加精度及运行库差异仍存在；该结果
不是Windows→Linux收益，未更换生产版本，详见§8最新记录。

最新 Linux 主机端实验（2026-09-15）：初始化期只读函数表已将计时前向
函数查询从 67000 次降至 0，整网与真实 Worker 数值门全部通过。唯一
共同墙钟前向 ABBA **+23.70%** 且稳定；真实 Worker ABBA 基线波动
**7.26%** 超出 5% 门，因此不认证、不上线、不原样重测。随后独立验证
Windows：修复 plan 安装时机后数值与实际路径全过，前向稳定 **+0.77%**，
未达 1% 门，关闭候选、未测 Worker 性能。现用认证版本与上述 Fork
参考保持；后续回到 GPU GEMM 热点，详见 §8。
2026-09-16 压缩 FFN down tile112 候选也已关闭：34组数值逐位相同，
但主机算子 ABBA 稳定仅 +0.47%，event 基线不稳定；未进入整网或
Worker 性能、未上线。随后 compact DualGemm up 稠密网格候选也已
关闭：移除440个空CTA后，算子event/主机稳定仅 +0.074% / +0.056%。
原核及约2.28亿half输出逐位相同，未进入整网/Worker。随后strict
Attention逐行half打包虽将寄存器255→222、栈40→0，算子ABBA仍
稳定慢约2.5%/2.6%，数值及sanitizer通过后按性能门关闭。现用版本
保持。独立8-warp方案也已完成：数值及sanitizer通过，但算子event/
主机稳定下降20.09%/19.23%，关闭。随后独立N32已通过完整native及
真实Worker对C++FP32数值门，但完整前向ABBA共同墙钟稳定−0.611%，
event稳定−0.882%，关闭，未测Worker性能、未上线。随后独立KV64虽通过
六组原生整网门，但真实Worker W32首选126/128，按原数值门关闭，未做
任何性能ABBA。RMS→QKV审计确认输入已在寄存器复用，完整M128归一化
共享块超限；全模型QKV零对压缩仅省594块中的2块，未登记压缩候选。随后
QKV双片段epilogue将同步16→8，整网逐位及Worker768/768全过，但完整
前向共同墙钟稳定仅+0.181%（event+0.774%），未达1%门，关闭、未测
Worker性能。随后8片段方案同步16→2、共享分配不增，整网逐位及
Worker768/768仍全过，但完整前向墙钟−0.053%/event−0.012%，均稳定
且未达收益门，关闭、未上线。随后classic N128/K32通过独立资源/ABI/数值验证，完整前向+1.785%、
Worker+1.225%均稳定，已安装。新的Fork近似持平参考与Linux移植进展见 §8。

### 当前 TF3 Worker：工作区既有成果

下列数字来自本轮 fork 借鉴动工前已有的实现与验证记录，不是 G1–G4 新收益。
比较对象是本机官方 C++ v1.18.2 Worker；该对象与 fork B14/S2 必须分开标注。

| 口径 | 工作区既有实测 | 证据 |
|---|---|---|
| B16、uncached、C32、单线程 Worker ABBA | Rust **882.4 → 1078.2** NN rows/s；最终同轮 C++ **1332.2** | `tf3-worker-performance-audit.md`「比较对象与测量边界」 |
| B16、uncached、C64、双线程吞吐配置 | Rust **1163.8**，同轮 C++ **1338.1** NN rows/s；单线程默认保留 | 同上，`worker-throughput-final/report.json` |
| TF3 数值与 attention | DualFFN、q64-serial 已验证；q64-serial 恢复 q128 softmax 顺序，15 组 half bitwise 一致，TF3 top-1 **128/128** | 同文「已确认的原因」及 §8 TF3 追加 |
| K384 布局受控探针 | 相同 FP32 compute/FP32 out 下 NN 比现有 TN 的部分形状快约 **12–15%**；尚非整网或 Worker 收益 | 同文「C++ 的部分 GEMM 速度来自不同计算/存储路径」 |

本轮 G0–G4 状态见 §8。模型与开关依赖独立 TF3 plan，旧 ONNX plan 不得直接套用。
SM120 不支持 tcgen05/TMEM 的硬件结论保留；这不意味着 FA4 必须依赖 tcgen05，
也不能把过去两个本地 FA4 变体失败外推成 attention/FA4 路线永久无收益。
fork 的 SM120 专用 AOT generator 使用 `FlashAttentionForwardSm120`
（`cpp/neuralnet/fa4_aot/build_aot.py:71`、`:182`）；具体 FP32 版本仍须本机验证。

### 历史 ONNX 已落地(2026-08-19；详见 cuda-optimization-plan.md 决策组表 G1-G10 全绿)

| 层 | 状态 |
|---|---|
| GEMM | cuBLASLt 接管 hgemm/hgemm_residual/hgemm_f16(默认,`KATAGO_CUDA_CUBLASLT=0` 回退手写 t64/t128) |
| attention | FA2 全 tensor core(QK mma + online softmax + PV mma),RoPE 融合进加载;对照官方 `cudarocmopt` 后生产 tile 已由 q128 晋级 q64 |
| 融合 | FUSION=none 为默认(autotune 裁决:cuBLASLt 时代手写融合净负);gatesilu 等保留为 CUBLASLT=0 组合候选 |
| 调度 | 事件门控流水线(submit/finish 分离 + 双槽 + 完成即投递)+ per-size CUDA graph 缓存;serve=1 单消费者 |
| batch | 精确尺寸 + nnMaxBatchSize=16 上限(padding 与 B16 凑满已分别证伪,见 §4 关键警示) |
| autotune | scripts/autotune.py 8 决策组 ABBA + schema 2 plan(JSON 绑定 device/model/CUDA/CUTLASS/kernel build/capabilities,fail-closed) |
| 精度路线 | FP16/FP32 混合精度；当时 FP8 精度实验未过门，INT8 无厂商路径 |

### 历史 ONNX 性能数字（保留原始记录，跨 GPU 推算不作当前验收）

| 口径 | 值 | 出处 |
|---|---|---|
| search benchmark t=1 | 296→**433** visits/s(cuBLASLt→pipeline) | cuda-optimization-plan.md |
| search benchmark t=8 / t=16 | **1034 / 1274** visits/s | 同上 |
| cap16 nnEvals/s(t=32,avgBatch≈15.9) | 历史 ~765(坏构建期)/ **919.7**(2026-08-17 修复后 r1 plan) | 优化文档「M3 追加」 |
| **nnbench eval B16(固定批,W=32)** | 871 →(M1/M2 记录 915.4/1112.4,提交断裂作废)→ **991.5**(修复后 r1,当日环境 -10%) | 优化文档「M3 追加」 |
| **nnbench kernel B16(纯前向)** | 18.48ms 基线;B2 后 15.08ms(记录值);kernel 模式在 WDDM 深队列下有提交路径污染,以 direct 配对为准 | 同上 |
| B16 单流上限(直测) | 871~894 行/s(M2 前) | 同上 |
| **WSL(fork 基线一致环境,2026-08-17)** | **eval B16 单流 1015.7 / 搜索 t=32 962.3**(plan r1);对齐线达成 43%,**per-SM 86%(真实差距 ~14%)** | 优化文档「M3 追加 2」 |
| **cudarocmopt q64(2026-08-19)** | Windows eval B16 **+2.95%**;WSL B16 **+3.44%**,search t8/t16 **+3.0/+2.4%**,attention core **+14.7%**;schema 2 生产 plan 已启用 | `cudarocmopt-validation-plan.md` |
| **双流复活(2026-08-18,WSL+Windows)** | **serve=2+NOGRAPH+W64 = 1112(+9.5%)**;搜索语境维持 serve=1 | 优化文档「M4 前哨」 |
| fork 每流(推算) | 2836/2 ≈ **1418 行/s/流** | fork plan |

> M0 全量数据与 H1-H5 裁决见 cuda-optimization-plan.md「M0 Phase 0 测量」。
> 关键反转:B16 kernel 已计算饱和(util 98%,B8→B32 线性),生产栈在固定 B
> 口径下已达 kernel 上限 97%;**剩余杠杆在 kernel(GEMM 代际/融合),不在
> 调度拓扑**——双流实测 852.8 vs 单流 871(-2%),证伪。
> **2026-08-17 再修订:C2 tcgen05 硬件证伪(ptxas 三源互证,见优化文档
> 「M3-pre」)——SM120 无 tcgen05/TMEM,FP16 GEMM 无代际杠杆;剩余杠杆
> 重排为 E1(FP8 精度门)/B4/A-lite/WSL(E4)。**

> **2026-09-08 适用范围修订**：上表及下方 §3–§7 保留当时实验与决策史。
> 其中 5080 每流/per-SM 的“真实差距”、fork 2.0×流间扩展及路线收益上限
> 都含未验证的跨硬件或计时假设，现已撤回作为当前结论；当前目标和精度门以 §1 为准，
> 当前执行顺序以 §8 G0–G4 为准。原始本机实测值及已验证收益仍保留。

## 3. 差距分解(2026-08-17 M0 后修订)

M0 直测后的三组数字:

1. **单流 kernel 效率差 fork ~37%**(894 vs 1418 行/s/流),而非旧估 11%——
   旧 T(B16)=12.5ms 系误读。按 SM 数(70/84)归一后仍有 ~20-26% per-SM 差,
   与「cuBLASLt 全走 mma.sync(H3 证伪 tcgen05)+ fork 用 CUTLASS AOT」一致。
2. **固定 B 口径下调度开销仅 ~3%**:eval B16 批周期 18.37ms vs kernel 18.48ms
   (graph 净省 ~0.7ms)。旧「每批 8ms」是搜索语境假象(凑批/CPU 编码竞争,
   搜索 t=32 口径 765 vs nnbench 871,差 12%——这部分是 A-lite 的唯一剩余空间)。
3. **fork 的 2836 = kernel 质量 × 双流,两者我们都缺**:双流在我们饱和的
   kernel 栈上无收益(实测证伪);kernel 代际差(H3)才是主差距。

**结论(修订):最大剩余杠杆是 kernel——C(tcgen05 GEMM 通路,占 75-85%
前向时间、全在 mma.sync)> B3(FA4,attn core 13-22%)> B2(dual-FFN 融合,
FFN 62%);方案 A 降级为 A-lite(仅搜索语境凑批)。**

> **2026-08-17 再修订:C 主线(tcgen05)❌ 硬件证伪**——SM120 物理上无
> tcgen05/TMEM(ptxas + CUTLASS 4.7 + PTX ISA 9.3 三源互证,优化文档 M3-pre)。
> B3 的两个本地变体亦已于 M2 测得负收益（不代表 FA4 全路线证伪）。当时评估的 kernel 剩余空间 = FP8 mma.sync(E1,精度门先行)与
> 融合微优化(B4);~~fork per-SM 差(20~26%)重归因于时钟/L2/带宽/融合度,
> 非指令代际~~(2026-08-17 再修订:DUALFFN 断裂修复 + WSL 复测后,
> **真实 per-SM 差 ~14%**——旧估 20-26% 中有 ~6-12pt 是 DUALFFN 空转
> 的损失,见优化文档「M3 追加 2」。),

## 4. 关键警示(不要重蹈的坑)

- **孤立组件的证伪结论不能外推到组合**:padding、双流、L2 persisting 的
  本地证伪都是在当前 kernel 速度(T(B) 曲线形状)和单流语境下做的。kernel
  变快或拓扑变化后,`KATAGO_CUDA_PADBATCH=1` 等开关保留着,结论需重审。
- **cuBLASLt 时代手写融合净负**(FUSION=none 裁决):新融合 kernel 必须先
  证明其 GEMM 主循环效率 ≥ cuBLASLt,融合才有净收益。
- **休眠路径启用前先对拍**(act384f16 断裂 bug 先例):autotune 只测速
  不测正确性。
- **数值纪律**:FP32 计算/half 存储边界、SiLU 精确形式、禁 fast_math;
  tactic 开关一律经 `tactic_plan::tactic_var()`(plan > env > 默认)。
- 搜索 benchmark 的 t=64 数字受同根树 NN cache 污染,只作相对比较。

## 5. Phase 0:测量先行(✅ 2026-08-17 完成,数据见 cuda-optimization-plan.md「M0 Phase 0 测量」)

**产出与决策表**(裁决结果;全量数据在 cuda-optimization-plan.md):

| 待验证假设 | 裁决 | 分支后果 |
|---|---|---|
| H1: T(B16) kernel ≈12.5ms | ❌ 证伪:实测 18.48ms(graph 批周期 18.37),单流上限 871~894 行/s | 调度空间远小于旧估;kernel 差距比旧估大 |
| H2: 每批 ~8ms 非 kernel 开销 | ❌ 证伪(固定 B 口径):eval = kernel 上限 97% | 方案 A 主体失去依据,降级 A-lite(搜索语境 12%) |
| H3: cuBLASLt 已用 tcgen05 | ❌ 未走:全池 algoId=21 遗留 tile/stages(mma.sync),3 形状×8 候选无 nvjet | **方案 C 升级为主 GEMM 通路替换,收益上限大开** |
| H6: sm_120f 可解锁 tcgen05(C2 通路存在) | ❌ **硬件证伪**(2026-08-17):ptxas 对 tcgen05.alloc/mma 在 sm_120a/f 均报 not supported(sm_100f 对照通过);CUTLASS 4.7.0 无 SM120 tcgen05(dense builder 仅 F8F6F4 mma.sync);PTX ISA 9.3 Target ISA Notes 确认 sm_120 族排除 | **方案 C 主线永久关闭**;FP16 GEMM 停留 mma.sync 屋顶;杠杆重排 E1/B4/A-lite/WSL |
| H4: attention ~14% | ⬆ 上修:FA2 core 13~22%(全段 47% 含 GEMM) | B3(FA4)排期价值上升 |
| H5: 初始卷积 >5% | ❌ 1.6% | 方案 D 搁置 |

附加发现:① cuBLASLt qkv 形状启发式首选非最优(候选 #1 -26%),top-N
计时重排≈前向 4%(方案 C1);② 双流 × 固定 B16 × W64 实测 852.8 vs 单流
871(-2%),WDDM 双线程 graph capture 互斥(开放问题 4 得证)——fork 拓扑
在当前 kernel 栈上无收益。

## 6. 方案清单

### 方案 A:双流拓扑(**2026-08-18 部分复活**:饱和供数语境 +9.5%;搜索语境维持证伪)

**复活条件(§4 重审条款兑现——原证伪系 DUALFFN 断裂的慢 kernel 上所测)**:
`numNNServerThreadsPerModel=2 + KATAGO_CUDA_NOGRAPH=1 + 供数 W≥4×batch` →
nnbench eval B16 W64 = **1112**(WSL ±0.5%)/1104.9(Windows),vs 单流默认
1015.7 = **+9.5%**。机制 = 填 host 间隙(单流批周期 31.4ms vs kernel
15.5ms),非 GPU 并行(kernel 已 SM 饱和,流间扩展仅 1.095× vs fork 2.0×)。
失败模式(全部实测):供数不足(W32=746)、PADBATCH 组合(259,永久证伪)、
graph 组合(GLOBAL 互斥=1014/RELAXED=1027,均劣于 NOGRAPH)、搜索语境
(t=32 858/-11%,t=48 **123/灾难**——批碎片+CPU 超订阅)。
代码附带改动:capture 模式 GLOBAL→RELAXED(serve=1 无变化,对拍 PASS)。
**使用边界:高并发饱和服务(分析引擎多查询)用双流+NOGRAPH;GTP 对局
保持 serve=1 默认。**数据见 cuda-optimization-plan.md「M4 前哨」。

<details><summary>原方案 A 存档(M0 降级版 + 初版)</summary>

**降级后的剩余范围(A-lite)**:仅搜索语境的凑批/窗口策略(修复后搜索
919.7-966 vs nnbench 991-1019,剩余 ~5-7%);~~双流 × 固定 B16 × padding
组合已实测无收益(852.8 vs 871;WDDM graph capture 互斥),不再追求完整
复刻~~(2026-08-18 上方复活节推翻此结论的适用前提——彼时 kernel 慢 25%)。

**原方案 A 内容(初版)**:固定物理 B16 + 尾批复制 padding + 双槽双流 +
batch-aware dispatch(攒满 target 才发射、GPU 空闲也发射,fork
`maybeLaunchFillingBatch` 语义已在 serve_pipelined 中部分存在)。

- 组件全部现成:`KATAGO_CUDA_PADBATCH=1`、serve 拓扑参数、事件流水线双槽;
  缺的是**按 fork 语义组装成组合并整体 ABBA**(组合从未测过,见 §3-3)。
- 低并发 padding 纯亏是已证事实(t=4 PAD 253 vs NOPAD 745)→ 做**负载自
  适应**:闲时(队列深度 < 阈值)精确尺寸单流,忙时切 B16 双流;切换逻辑
  进 tactic plan 候选。
- 验收:nnbench eval B16 口径,目标区间 **1500~2000+ nnEval/s**(fork
  per-SM 对齐线 2360 的第一阶段)。
- 风险:双槽双流在 WDDM 下的事件语义;自适应切换引入批尺寸抖动——
  per-size graph 缓存已解决重捕获,但需复测确认。

</details>

### 方案 B:kernel 融合对齐 fork CUTLASS AOT 三件套

直接复用 `D:/code/KataGomo_fork/cpp/neuralnet/sm120_aot/*.cu` 的 CUTLASS
包装(header-only,build.rs nvcc 编译,与现有 cuda-kernels 同链路),
注册为 autotune tactic 候选:

| # | 项 | fork 战术 | 收益机制 |
|---|---|---|---|
| B1 | wide QKV packed GEMM | `wide_qkv-m128-n128-k64-s2-cute-atom4x2-packed`,M=B×361、N=1152,packed 行 `[Q384\|K384\|V384]` | 3 GEMM→1;packed 布局是 FA4 前提 |
| B2 | dual FFN + SwiGLU epilogue | `dual_ffn-m128-n64-k32-s2-mb3-tanh-half2`(LeftSiLUAndMul) | 省 2304 宽缓冲整趟往返 + 独立 SwiGLU kernel |
| B3 | FA2→FA4 tile | M128×N64 s1、128 线程、noncausal 无掩码 | 当时估 attention ~14% 前向、收益上限 ~7%；本地两个变体测得负收益。保留 FP32 累加；不同累加类型是否同速须按 GPU/形状实测，不能由架构名称推断 |
| B4 | linear2/outproj beta=1 原位残差 | CUTLASS `GemmShape<128,128,32>` warp<64,64,32> 3 stages,C==D | 已部分有(hgemm_residual);~~CUTLASS 版作候选对照~~ **❌ 门槛证伪(2026-08-17)**:微基准(scripts/b4_residual_probe)全残差形状 CUTLASS 128x128/128x64/128x256 均不赢 cuBLASLt top-8(outproj 0.0251 vs 0.0250 持平;linup +0.9% 噪声;ffn_down -2.6%;B1/B2 慢 19-72%),且现行 hgemm_residual 本就是 cuBLASLt beta=1 原位——fork 战术在我们栈上无对应收益空间,关闭 |

- **前置门槛**:每个融合 kernel 的 GEMM 主循环微基准 ≥ cuBLASLt 同形状,
  否则免谈(FUSION=none 教训);逐项 ABBA + 对拍。
- B16 下 trunk 缓冲 8.9MB、mid 4.4MB,每省一趟往返 ≈10~20µs × 12 block,
  累计预期数个百分点到 10%+。

### 方案 C:tcgen05(sm_120f)——❌ **硬件证伪,永久关闭**(2026-08-17)

**证伪结论(H6)**:SM120(消费级 Blackwell)物理上无 tcgen05/tensor-memory,
"GEMM 代际差"杠杆不存在。三源互证:① ptxas(CUDA 13.2)对 tcgen05 指令在
sm_120a/f 均报 not supported(sm_100f 对照通过);② CUTLASS 4.7.0 的 SM120
支持全部为 TMA + f8f6f4 mma.sync,config.hpp 不为 SM120 定义任何 TCGEN05 宏;
③ PTX ISA 9.3 tcgen05.mma Target ISA Notes 仅列 sm_100/101/103/110 族。
完整证据链与工具链留档见 cuda-optimization-plan.md「M3-pre」节,
探针源码在 scripts/c2_tcgen05_probe/。

**衍生事实(E1 可用)**:sm_120f 解锁窄精度 mma.sync(kind::f8f6f4);CUTLASS
4.7.0 已克隆至 D:/code/cutlass4;SM120 FP8 GEMM 模板 sm_120f device 编译通过
(25.8s/TU);Windows/MSVC host 发射受 C2719 阻塞(SM90+ TMA 通病,路线:
WSL / clang-cl / driver-API launcher)。FP8 精度门未过前不动工。

<details><summary>原方案 C 内容(存档)</summary>

- **M0 裁决(H3 证伪)**:cuBLASLt 13.3 在 sm120 对我们全部 FP16 形状只给
  algoId=21 遗留 tile/stages(mma.sync s16816 类),3 形状×8 候选池无任何
  nvjet/tcgen05 痕迹;GEMM 占前向 75~85% 且已在 mma.sync 屋顶(78~92 TF)。
  → C 按"主 GEMM 通路替换"分支动工,上限 = 代际差。
- **C0(零工程试探)**:经典 cuBLAS(cublasGemmEx/hgemm)在 CUDA 12.8+ 的
  sm120 上对部分 FP16 形状会派发 nvjet(tcgen05)kernel——M1 已实测证伪:
  经典 cublas 同走 mma.sync,与 Lt 逐位同速。
- **C1(cuBLASLt top-N 计时重排,M0 彩蛋)**:已落地(M1,+2.55%,入 plan JSON)。
- ~~**C2 主线**:CUTLASS 4.x sm120 tcgen05 GEMM 模板替换主通路~~——硬件证伪。

</details>

### 方案 D:初始 3×3 卷积 cuDNN frontend(fork eng45-tile0-stages2)

引入 cuDNN ≥9.24 依赖换 fork 同款 graph engine;本机现状 im2col K=198→208
已实现。**仅当 H5 显示初始卷积占比 >5% 才启动**,预期很小。

### 方案 E:超越 fork 的储备项(大工程,暂不动工,按序评估)

1. **FP8 官方 PTQ scope 重模拟**:✅ 完成(2026-08-17,M3)——**精度门不过,
   FP8 kernel 工程封存**。官方精确语义(per-out-ch 权重+静态 per-tensor
   激活,scope 含 linear2)在 b11fix 硬失败(KL 4.3e-2、value 2.5);FFN-only
   scope 最佳(KL 2.4e-3)仍超门(top-1 近平局翻转+score 2.2×)。全部损害
   集中在注意力投影,FFN 三类几乎免费(隔离实验)。数据与重开条件见
   cuda-optimization-plan.md「E1」节。
2. **每批 ~8ms host 开销消减**(依赖 H2 分解):特征编码 GPU 化、pinned
   half 预转换(fork `enablePinnedHalfInputs` 语义)、解码路径裁剪。
3. **持久化 megakernel**:12 trunk block 融合为少数常驻 kernel,activation
   驻留 48MB L2(B16 双缓冲够放);研究型,最后考虑。
4. **WSL 部署**:✅ 已验证(2026-08-17,scripts/wsl_bench.sh 一键复现)——
   搜索 t=32 +4.6% / eval B16 +2.4%(vs 同日 Windows WDDM);Linux 口径
   配对方差 ±0.05%,为 fork 基线一致的首选测量环境;cuda feature 的
   WSL 首建 bug 已修(cudart_static 命名 + stdc++ 链接)。

## 7. 执行顺序与里程碑(M0 后重排:kernel 优先,拓扑降级)

| 里程碑 | 内容 | 验收 | 依赖 |
|---|---|---|---|
| M0 | Phase 0 测量 + 决策表落档 | ✅ H1-H5 全部裁决(2026-08-17),数据入 cuda-optimization-plan.md | nnbench ✅ 已提交 |
| M1 | C1 cuBLASLt top-N 计时重排 + C0 经典 cublas nvjet 试探 | C0 证伪(2026-08-17);C1 当时 ABBA +2.55% 采纳,**2026-08-17 晚复审下架**(B2 后边际归零 + 搜索口径净负 -4.3%/-4.4%,见优化文档「M3 追加」) | M0 |
| M2 | B3 FA4(❌ 两个本地变体测得负收益；不能外推整个 FA4 路线)+ B2 dual-FFN(当日 ABBA +29.8%,**但提交源码编译断裂从未生效——2026-08-17 晚修复后重验证 +26.5% eval / +28.0% 搜索,对拍 PASS**) | 修复完成(见优化文档「M3 追加」) | M0 |
| M3 | ~~C2 tcgen05 主 GEMM 通路~~ ❌ 硬件证伪关闭(2026-08-17);B4 ❌ 门槛证伪(CUTLASS 残差 GEMM 全形状不赢 cuBLASLt top-8);E1 ❌ 精度门不过(官方语义硬失败,FFN-only 仍超门);**M3 收官:三项全证伪**。剩余:A-lite(搜索凑批)、WSL(E4) | ABBA + 对拍(未及——全部在门槛/证伪阶段关闭) | M1 |
| M4 | 冲击 per-SM 对齐线 2360;评估 E1/E2、A-lite 与官方 `cudarocmopt` 增量 | 🔶 进行中(2026-08-19):E1 ❌ 精度门;双流复活 +9.5%(饱和口径);A-lite/host staging 证伪;布尔 tactic 与 graph UB 已修;**q64 attention 双平台通过并晋级 schema 2**(eval +3%左右,attention core +14.7%)。对齐线仍不可达,但已兑现本轮最后一个可复现 kernel 增量 | M1-M3 |

每个里程碑完成标准:对拍 PASS + ABBA 留痕 + plan JSON 更新 + 本文档 §8
看板更新。**任何一步慢于基线即回退并在 cuda-optimization-plan.md 记证伪。**

## 8. 进度看板(跨对话对齐用,每次会话结束更新)

### 历史执行顺序：同 SHA TF3 / RTX 5070 Ti（2026-09-08，本轮已于2026-09-16结项）

下列 G0–G4 是本轮阶段编号，与历史优化文档的决策组 G1–G10 不同。
G0 的稳定参考基线已完成；G1/G4 的持续 C32 配置已有旧编码验收记录。
主机 FP16 次正规数转换缺陷已修复，七份配置已重新认证并迁移；此前数值与性能记录
只对应其保存的二进制。Windows 双线程吞吐配置新增通过验收的 Lt 残差候选。
G2 的共享内存 RoPE 融合变体已因负收益拒绝。
下表区分本轮新证据与此前已落地成果，不声明尚未跑完的新方向收益。

| 阶段 | 工作与验收 | 当前状态 |
|---|---|---|
| G0 公平基线 | 同卡同SHA TF3、B14/S2、80 warmup/1000 iterations、无graph、空盘特征逐位相同；统一端点与完成后收尾 | ✅ 当前输出投影N128共同墙钟参考：正式Rust Windows1595.982821、Fork统一测试副本WSL1563.247273 NN rows/s，比值102.094074%；两臂波动0.176416%/1.125760%通过5%门。副本已对齐收尾屏障，平台/累加精度仍不同，不能称同精度或Fork Worker优势；详见outproj-current-fork-reference-r1记录 |
| G1 K384 cuBLASLt NN 布局 | 权重在加载期组织为候选布局，FP16 输入/权重、FP32 累加及既有输出/残差边界不变；按 shape 选 tactic，先数值再整图/Worker ABBA | ✅ `nn_k384_b16` 仅采纳于独立持续 C32 单线程配置，1073.48→1093.68 RPC/s（+1.88%）；C16 −0.72%、双线程 C64 +0.59% 均不采纳，旧默认/吞吐配置保留；B8 阈值未采纳 |
| G2 Attention / RoPE FP32 | 评估独立、跨 batch 共享 RoPE 与 SM120 attention tile/数据组织；QK/PV、激活/归约继续 FP32，维持精确函数与 half 边界 | ✅ Windows B14 strict+Fixed 通过完整数值与稳定 ABBA；同 B14 Worker +1.77%，最终对原最佳 B16 Fixed +2.09%；throughput plan/config 和 CLI 已安装、7 计划最终路径加载及 Worker 128/Drain 通过。2026-09-14 新增已认证 immutable QKV/RoPE：完整前向 +2.986361%、Worker +3.333133%，新 CLI/六 Windows 计划已安装并最终复验；当时剖析 FFN 45.11% 为主要热点，DualFFN epilogue vector4 候选稳定回退已关闭。WSL 新路径未认证 |
| G3 B12–B16 联合调优 | 在已过数值门的 G1/G2 候选上联合扫描 B12/13/14/15/16、lane 数、graph 与供数；记录真实 avgBatch、吞吐、延迟；不照抄 fork 的 DualFFN/L2 开关 | 🔶 WSL生产Worker仍有top-1 126/128，阻止该矩阵性能验收；2026-09-14两处GEMM修复已通过完整模型及真实Worker W1/W32的128/128与11字段门，但单独前向ABBA为-0.298%，已关闭、未上线。strict/QKV新组合已用Linux原生表通过整网及Worker128/128，前向event+4.345%/wall+2.383%；Worker ABBA基线spread6.362%超门，完整组合关闭、未迁移。B12/B13精确算子与实际Linux原生整网/Worker W1/W32/W64数值均通过（候选128/128）；新B13前向event两臂spread10.670%/10.479%超门，候选关闭，未测Worker性能/未迁移。baseline共同时轴诊断已完成：87批独流尾段中位12.757ms/912批重叠段23.652ms；主机apply占据大部分运行时间，主机API归因已完成：模块函数查找占线程累计前向56.979%，描述符合计0.582%；每次91个成功Lt之前的fallback查找未使用；2026-09-15共享精确residual replay的延后查询候选已过整网/Worker数值门，但完整前向event−0.390%/wall+0.262%未达1%，已关闭，未上线，不重试旧共享缓存。已有负收益/不稳定项保持关闭。✅ Windows stem 三维映射已采纳，前向 +2.35%、Worker +1.159%。2026-09-14 新增 C768 四通道门，整网逐位相同，前向 +2.233%、Worker +1.511%；新 CLI/六 Win plans 已安装，最终六计划加载及 Worker 128/Drain 通过。旧五计划保持 STEM0/GATE0，WSL 不变。各项收益来自各自配对 ABBA，不能直接相加；详见本节逐项证据 |
| G4 数值 + ABBA + plan | TF3 对 C++ FP32、旧 ONNX 对 ORT FP32 整图门，路径标记/失败关闭负例；ABBA 达 ≥1% 才认证，回退慢项，绑定模型/设备/构建与能力指纹 | ✅ G1 独立 C32 profile 与 G2 Windows B14 throughput 已按各自数值/ABBA 证据认证；G2 每平台 35 CPU + 14 实际 plan 正反例、固定批次 raw、Windows 组合 128/128 黄金门及最终 7 计划加载通过。WSL 两处GEMM已证明可修复 G3 top-1，但单独性能未过门；strict/QKV 新组合数值与前向通过、Worker稳定性未过门，不扩大认证范围 |

### 本轮证据与裁决（2026-09-08，阶段快照）

2026-09-09 补充：`stem-global-preproject-r1` 的全 batch FP32 global
预投影候选因 Worker W1 top-1 126/128 被拒绝，未测速/安装；其他三组
Worker 和六组 native 通过不能抵销此失败。后续 B14-only r2 六组
native 与四组 Worker 全通过，B1/B16 回退逐位一致，但固定 B14/S2
前向 ABBA 仅 +0.106%（稳定、未达1%），wall −0.166%，因此同样
拒绝采纳，不跑 Worker ABBA、不重试；实验源码已归档并精确恢复。
已安装最佳版本继续为上表的 stem 三维映射，原100% top-1门保持。

同日 `dualffn-current-map3d-r2` 用已安装 binary 重测 DUALFFN1/0：
六组 native 实际全头逐位一致，四组 Worker 均128/128；固定 B14/S2
前向 ABBA 中关闭融合 **−18.647%**（两臂稳定），wall **−18.358%**。
保留 DUALFFN1，不跑 Worker ABBA、不重试、不改生产文件。这仅验证
当前 Rust 开关收益；Fork B14 关闭融合的选择不直接迁移到 Rust。
实际 Fork 数学/dispatch 审计见 TF3 性能审计末尾。随后 L2 双流 inner-only
独立实验 `l2-current-b14-s2-r1` 通过56行整网逐位一致门，唯一 B14/S2
前向 ABBA **+1.083%**（两臂波动0.323%/0.152%，wall +0.832%），
8000事件重算通过。随后 `l2-worker-integration-r1` 完成真实后端接入，
W32/W1 四组 Worker 均128/128、11字段门与缓存清理通过。唯一 C64
Worker ABBA 为 **1203.382690→1208.342278 RPC/s（+0.412137%）**，
两臂波动0.469080%/0.358965%，稳定但未达1%采纳门，故拒绝该接入，
不重试。32768请求计时已独立重算；两份生产源码精确恢复，两个新增
实验文件归档后移除，已安装21个保护文件未变。第一阶段前向收益不
代替 Worker 收益，正式 Fork/Rust 比例不变。全部阶段记录保留。

随后有界算子实验将 DualFFN 的两份 B 各自改为 NN RowMajor 布局，
保留 N64/stages3/swizzle2、FP32 累加与精确 SwiGLU。现有 G1 NN 权重
是整块[K,2304]，不能切半充当两个[K,1152]；Dual1此前仍读原TN权重。
只借鉴 Fork 的布局源码，不把该未选中的融合路径写成其 B14 实测选择。
先复用冻结 QKV-prefix RMS 张量作算子 fixture，不能称为实际首FFN输入。
`dualffn-nn-layout-r1` 的 Windows 探针全部5,822,208个half与旧TN逐位
一致；三轮ABBA保留12000事件，123.338569→123.442400微秒，吞吐
**−0.084112%**，两臂spread0.363545%/0.583959%。稳定但无收益，
拒绝整网接入，不扩大候选池、不重试，生产源码与21个保护文件未变。

后续 `dualffn-exact-silu-lut-r1` 将已有 half gate 对应的 FP32 SiLU
精确结果缓存在256KiB GPU表中，保留FP32累加及所有half边界。完整
65,536表项、1,310,720边界组合、两流读取与5,822,208算子输出逐位
一致；一次三轮ABBA算子事件吞吐 **+10.845109%**，两臂spread
0.155845%/0.720460%，独立复核通过。辅助墙钟数据不稳定，不用于准入。
临时接入 `dualffn-exact-silu-lut-integration-r1` 已完成唯一隔离构建、
34个计划单测、B1/B14/B16两流六组整网全头逐位门及四组Worker
W32/W1的128/128黄金门、11字段与Drain检查。表初始化、实际M16探针
和正常释放均通过；W1请求128行/128批，不能把启动B14预热标记当作
请求期B14。新候选默认关闭，仅注册TF3/5070Ti的B14使用表；其他批
走原DualFFN。2026-09-14 完成整网裁决：唯一 B14/S2、W80/N1000 的
ABBA 为 **1197.926051→1172.962169 event rows/s（−2.083925%）**，
两臂spread0.087717%/0.541806%，稳定负收益；独立墙钟同样下降2.465370%。
四组Worker的512份原始协议响应与JSON核对、8000原始事件重算均通过，
不能用算子收益抵销整网退化。因此拒绝该接入，不跑Worker性能、不重试；
六份修改源码已从精确归档恢复，build.rs确认未变，21个正式保护文件
保持原值。三项实际CLI配置越界均在初始化被拒绝；非法值由更早的通用
plan校验拦截，首版检查器的错误文案预期已留档并以CPU复核更正，未重跑。
完整裁决在该实验目录的`decision-stage2.json`，归档的已测源码与隔离
binary保留。尚未证明整网退化的具体机制，正式Fork/Rust比例不变。

同日 `fa4-strict-stages2-r1` 只把现有 strict Attention 的 K/V 预取
从1级改为2级，M128/N96、FP32 QK/PV、精确softmax及half边界不变。
唯一离线导出通过196处完整exp数据流、192处FP32 MMA和新ABI检查；
动态共享内存20→32KiB，实际local memory 40→72字节/线程。4组固定
合成输入每组1,940,736个half与原内核逐位一致，第二流复跑和输入/输出
guard通过。启动脚本在首个子进程前遇到ROOT变量遗漏，保留零样本失败
并修正启动路径；唯一实际ABBA保存4000事件，四段中位均50.336微秒，
测得收益0%，未过1%门。拒绝整网接入、不重试；这只说明该单流算子
协议未测到收益，不代表所有负载等速。21个正式保护文件未改，见实验
`decision.json`。当前仍保留原stage1 strict Attention。

同日 FFN down 的 `ffn-down-lt-swizzle-r1` 查询确认当前 Fixed index1
的公开 CTA_SWIZZLING 能力为0（不支持）；未强设参数或执行测速。
独立候选 `ffn-down-dense-grid-r1` 使用 CUTLASS 64×64×32、6级预取、
identity swizzle1，实际79×6网格、128线程，FP32累加/残差/输出保持。
固定合成输入下1,940,736个FP32结果与现用Lt逐位一致，第二流复跑、
完整guard及256个FP64采样通过；实际内核80寄存器、无local memory、
48KiB动态共享内存，16个静态MMA站点均为FP32。唯一单流B14算子ABBA
（W80/N1000）保存4000事件，四段中位均54.432微秒，注册收益0%，
未达1%门，因此拒绝接入且不重试。每个测速进程前后完整输出均已CPU
复核，16份数组逐位通过。该协议复用固定A/B，beta=1在计时段内累计，
reset和拷贝均在事件外；不等同整网或Worker。更少网格块不能单独证明
加速，相同中位数也不能证明所有负载等速。21个正式保护文件保持原值，
完整裁决见实验`decision.json`，正式Fork/Rust参考比例不变。

随后 `mid-gate-dual-store-r1` 针对块尾SiLU→FP16转换，冻结原始两核
作为基线（均block1024），候选在同一FP32 SiLU寄存器上双写FP32/
FP16，并使用block256；融合与线程块大小是两项已声明的改动，不能
把结果只归因于其中一项。8组固定合成输入（rows1/361/5054/5776，
普通/边界两类）共8,595,456元素的两种输出与原路径逐位一致，第二流
复跑及guard通过；CPU检查48份数组、完整expf范围规约/修正/除法
数据流和同寄存器RN half转换通过。唯一B14单流ABBA的基线中位为
30.944/30.784微秒，候选12.736/40.864微秒；候选spread220.854%，
超过5%门，名义+35.289%的几何平均不具备采纳资格。4000事件及计时
前后32份完整输出复核通过，按`REJECT_UNSTABLE_OPERATOR_ABBA`关闭，
不接入、不跑Worker性能、不重试，21个正式保护文件未变。波动原因
尚未确定，不能据此宣称该融合在整网提速或退化。

同日 `dualffn-m64n32-r1` 将DualFFN从128×64改为64×32，保持3级预取、
FP32累加和原精确SiLU/half边界。N32需要把epilogue向量宽度8改为4；
首版Count8只在编译阶段失败，未跑GPU。修正后5,822,208个half结果与
原核、冻结参考及第二流全部逐位一致；实际寄存器236→96、共享内存
48→24KiB，occupancy API允许的每SM线程块2→4（不是实测占用率）。
唯一三轮ABBA合并123.514526→113.527963微秒，算子事件吞吐
**+8.796567%**，两臂spread0.363161%/1.530177%；12000事件及15份
完整输出复核通过。该夹具不是实际第一层FFN输入，算子结果不替代整网。
临时接入 `dualffn-small-tile-integration-r1` 默认关闭，仅注册TF3/5070Ti
的B14启用；独立构建、35项计划单测、B1/B14/B16双流六组整网全头
逐位门、四组Worker W32/W1的128/128黄金门及三项实际CLI越界拒绝均
通过。唯一整网B14/S2 ABBA为1206.220897→1183.759330 event rows/s，
**−1.862144%**；两臂spread0.990373%/0.159787%，稳定退化，墙钟
同样下降1.806302%。512份原始Worker响应、六组整网数据和8000事件
经CPU复核，因此拒绝接入、不跑Worker性能、不重试。六份源码已按
精确归档恢复，build.rs确认未变，21个正式保护文件保持原值。完整
裁决见`dualffn-small-tile-integration-r1/decision-stage2.json`；具体
退化机制尚未确定，不能以单算子+8.80%抵销整网负收益，Fork/Rust
参考比例不变。连续两次DualFFN候选呈现单算子收益未传递到整网，
后续新候选需重视双流和相邻层的实际负载，不继续盲扫该分块族。

随后 `value-fc-kn-layout-r1` 转向 Value FC 的权重读取：把原 NK half
权重转为 KN，使相邻输出线程读取相邻权重；两段576/192项的FP32
累加顺序、完整精确SiLU和全部输出格式不变。实际native权重配12组
合成输入（B1/B14/B16×4），2604个FP32输出在原核、候选和第二流均
逐位一致；实际PTX的完整FP32运算数据流一致。唯一双流算子ABBA
保存8000事件，492680.174864→938056.325428 event rows/s，
**+90.398635%**，两臂spread0%/2.420229%。这是Value FC算子证据，
不是整网或Fork比例。`value-fc-kn-integration-r1` 临时接入默认0的
`KATAGO_CUDA_VALUE_FC_KN`，仅注册native SHA、5070Ti、物理B14生效；
独立构建35项计划测试、六组整网逐位门、四组Worker 128/128及全部
误差/Drain门、三项实际CLI越界拒绝均通过。唯一整网ABBA为
1210.139138→1214.649600 event rows/s，**+0.372723%**，两臂spread
1.030793%/0.387064%；墙钟+0.803080%。收益低于预注册1%门，按
`REJECT_FULL_FORWARD_BELOW_MINIMUM_GAIN_SOURCE_RESTORED`关闭；这是
未达门槛，不是测得退化。512份原始Worker响应、六组整网和8000事件
CPU重算通过，六份临时源码恢复，build.rs未改，不跑Worker性能、
不重试。正式二进制、计划和21个保护文件保持原值；Fork比例未更新。

后续源代码核对见 `outer-up-layout-audit-r1/audit.json`（纯CPU审计，
未编译或执行新候选）：当前Fork B14 plan的外层上/下投影专用AOT
均为disabled，历史5080的warp64x32不能作为当前优势来源。Rust B14
上投影为M5054/N768/K384、FP32原位残差，每网11次；旧剖析占GPU
累计时间4.402%，上下投影合计7.52%，均不是可削减墙钟比例。随后
将假设收窄为B14上投影TN/NN独立布局，不改变QKV/outproj/DualFFN等
其他选择；既有泛K384布局探针仅覆盖B1/B4/B8/B16。

`outer-up-nn-b14-r1` 的前缀提取在首个FFN down遭既有Fixed算法身份
检查拒绝，未执行布局候选或测速；同一正式CLI的完整B14前向仍通过，
具体差异原因未确定，不掩码opaque身份。`outer-up-nn-b14-r2` 改由
未修改的正式CLI完整前向在边界15/16转储，只读取已经写入的上投影
输入、原位残差和结果；两次空盘输入哈希、共同act384及设备half转换
一致。独立TN重放与完整前向的3,881,472个FP32结果逐位一致，NN和
第二流同样全位一致；全部元素的FP64参考、条件误差界、尾guard及
只读操作数检查通过。唯一双流ABBA为TN 423135.12/515159.77、NN
430407.50/534071.91 event rows/s，名义+2.690265%，但两臂spread
**21.748291%/24.085178%**，超过5%门，裁决
`REJECT_UNSTABLE_OPERATOR_ABBA`。8000事件及16份计时完整输出已CPU
复核，不整网接入、不跑Worker性能、不重试；正式21个保护文件和
源码不变，Fork比例不变。每次GEMM前在事件外D2D恢复残差，可能与
另一流重叠；算子协议不能替代完整前向。失败前缀、正式转储、模型
权重与独立重放证据全部保留，详见r2的`closure.json`。

下一项当前Fork差异见`model-per-lane-audit-r1/audit.json`：B14计划为
`cudaShareModelWeights=false`，各ComputeHandle构建自己的Model，
MatMulLayer仅在该开关为true时使用共享权重缓存。Rust的backend与
nnbench均给两流克隆同一个Arc<CudaModel>。`model-per-lane-full-r1`
在独立测试程序固定B14/S2、同CudaRuntime、原有tactic与精度边界，
比较共享模型与每流完整上传一份模型；同时拆分模型持有的DualFFN
句柄管理状态，不将结果解释成单一缓存因素。预注册完整输出逐位门、
唯一完整前向ABBA及显存/加载观测；历史G7改变服务器拓扑和平均batch，
不能代替这项固定batch比较。此处记录审计与实验协议，尚未采纳候选。

该实验隔离构建成功，但共享完整模型控制在首次FFN down被原有Fixed
身份门拒绝，两流均返回`data[4]=0x0001fe2818300001`，其余opaque词
与认证值相同；当前build/device/Lt DLL均匹配。179层模型已完整加载，
因此不能再把此类拒绝简单归因于前缀执行。未运行replica或测速，按
`CLOSED_SHARED_FULL_MODEL_CONTROL_IDENTITY_REJECTION`关闭。单次控制
模型加载同步238.0492ms、显存free快照减少160MiB，只是该次控制观测，
不是双模型额外成本。21个保护文件、生产源码和Fork比例保持不变。

`residual-algo-identity-audit-r3`在主机读取已有描述符，27次有类型的
公开属性查询全部成功，三份描述符的ID21/tile15/splitK1/stages12及
其他五项配置一致。r1/r2属性读取因大小错误无效，记录保留且撤回其
推断；有效证据仅为r3。生产heuristic结果原本已清零，不能归因于
Rust结果数组未初始化，也不据公开属性相等掩码opaque词。下一检查
按同库版本的序列化约定复用原始认证64字节描述符，并用AlgoCheck、
工作区/对齐及完整数值门验证，详见r3的`next-protocol.json`；尚未
实现或运行该路径。该API用途和边界见[NVIDIA文档](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmulalgocheck)，
与本机固定CUDA13.3头文件核对一致。

后续`residual-serialized-check-r1/r2`已实际验证认证描述符复用，仍未
修改生产源码。r1在首个AlgoCheck后因测试假设`result.algo`保持哨兵
而中止；实际库将输入描述符写入返回字段，传入描述符本身未改变。
该次API状态未及保存，不能解释为算法不支持。r2仅将返回algo字段
改为观测、不用它选算法；三个原认证形状的检查均status/state=0、
工作区0字节、A/B/C/D最低对齐16字节。FP64输入负例status15拒绝。
随后使用原生模型真实FFN-down/outproj权重和确定性合成输入/残差，
在两流各重放两次未改动的64字节算法，12份输出的25,506,816个活跃
FP32元素全部通过全元素FP64条件误差界，最大误差6.98783e-7，流间
及重复逐位相等，尾guard及只读操作数检查通过。r2的
`stage1-receipt.json`绑定全部证据；这是算子验证，完整网络、Worker
与性能仍待验证，正式21个保护文件及Fork比例保持不变。

后续整网接入见`residual-replay-integration-r1/r2/r3`：五个Rust源码文件
加入默认关闭的`KATAGO_CUDA_RESIDUAL_REPLAY_R1`，只复用原认证64字节，
原Fixed选择器在开关0时保持原路径。开关1要求原生TF3原文件SHA、
既有Fixed preset、显式plan绑定、原GPU/Lt版本及三个已测形状，加载时
校验模型来源，执行前检查AlgoCheck/workspace与实际指针对齐。r1整网
通过后独立提升host_tactic_revision到2；CUDA kernel build仍为01764…。
r2的一个负例硬编码Some(2)，升级后已不再是错误版本，32/33通过后
中止并保留记录；r3只将该测试改为当前版本+1，运行代码未再改变。

r3隔离CLI/测试构建和37项边界测试通过；B1/B14/B16候选及B14旧路径
共449,504个完整输出元素与既有参考全位一致，包含两流、全部head和
补齐行。旧/新路径各W32/W1四组Worker共512请求，通过原C++ FP32的
11项门及policy top1 128/128，全部Drain且零失败；CPU从所有原始
请求/result protobuf重建输出并重算门限。七个实际CLI负例拒绝旧
host版本、旧binary新key、未绑定env、非法值、缺失Fixed及ONNX模型/
通用加载入口。证据入口为r3的`stage-receipt.json`。这仍是隔离源码
实验：正式binary、六份Win计划及21个保护文件未替换；live源码host2，
新构建会拒绝旧host1计划，尚无部署或性能认证，Fork比例不变。

`model-per-lane-replay-r1`在上述数值/绑定门通过后重新冻结独立实验，
两臂都使用同一host2序列化选择器和既有tactic，唯一比较因素为共享
完整模型与每流独立模型（包括模型持有的DualFFN状态）。沿用固定
B14/S2、完整head逐位门、唯一ABBA和显存/加载观测；前次共享控制的
opaque身份拒绝保持关闭，不掩码、不修改旧测试结果。共享/独立两臂
各完整网络12份输出共858,144元素逐位门通过。原CPU reviewer将Rust
带bytes及混合分隔符的收据与Python字典直接比较而失败；保留旧脚本，
新版仅改为规范路径/SHA/bytes精确校验，复核原始输出，没有GPU重跑。
唯一ABBA的shared event吞吐1222.2900/1210.8523，replica为
1213.0459/1211.0395，几何均值1216.5577→1212.0423，**−0.371159%**；
两臂spread0.944602%/0.165675%，wall **−0.454888%**同样稳定。
8000个事件及计时前后完整输出均已CPU重建验证，裁决
`REJECT_FULL_FORWARD_BELOW_MINIMUM_GAIN`，不跑Worker性能、不重试。
数值阶段模型加载free显存差为共享160MiB、独立320MiB，是快照观测，
不作为精确权重字节统计。独立模型不能带来当前Rust路径的性能收益。

`model-per-lane-replay-r1/closure.json`随后记录精确恢复五个实验源码
至原host1状态；`residual-replay-integration-r3/closure.json`保留隔离
selector构建及其数值通过证据，但未做selector独立性能/旧ONNX/graph/
六计划迁移，因此未部署。正式binary、六Win计划及21保护文件全不变，
新编译源码重新与既有host1计划匹配；Fork比例不变。

下一项CPU审计见`trunk-gate-rowpair-audit-r1/audit.json`：当前Fork选择
`cudaAffineSiluTacticSm120=half2`，C768一行一CTA、384线程、每线程两
通道，但其affine为half2 FMA。Rust的异位C768门读取FP32残差/scale/
bias并用FP32仿射与精确SiLU，flat1024线程；只借鉴逐行成对读写，
保留全部FP32边界，不能直接复制Fork的half运算。既有Nsight中trunk
和final共11次这类门占kernel sum约2.892%，作为下一有界算子候选；
尚未实现、GPU验证或测速。此前C384原位gate+cast融合的失败方案不重开。
Fork的G1+V1宽头布局亦已核查，Rust两段对应投影合计仅约0.297%，
暂缓；占比用于排优先级，不作为墙钟收益的严格上界。

`trunk-gate-rowpair-r1`已完成独立算子测试。原标量1024线程与逐行
384线程float2候选，四种行数×常规/边界输入共保存103,145,472个half
结果，全部逐位一致；两条独立流、重复执行、只读和前后guard通过。
唯一双流ABBA的event吞吐+224.088%且稳定，但候选wall两轮spread
10.568%超5%门，按预注册双指标纪律裁决`REJECT_UNSTABLE_OPERATOR_ABBA`。
保持关闭，不重试或整网接入；`closure.json`保存全部8000事件及输出。

随后仅新增一个不同的四通道候选`trunk-gate-rowquad-r1`：每行192线程，
float4读取及64位half输出打包，四次FP32仿射/精确SiLU和RN半精度转换
不变。相同103,145,472个half结果全位门通过；唯一ABBA event吞吐
160546377→647420316矩阵行/s，**+303.261%**，两臂spread
0.050839%/1.067623%；wall **+94.0069%**，spread1.002262%/2.664414%。
全部8000事件及计时前后结果已CPU复核，算子门通过，不能据此宣称
NN/Worker收益。两通道旧实验保持关闭，未放宽门限。

**2026-09-14 C768 四通道门控已验收并安装。**
`trunk-gate-rowquad-integration-r1`接入默认关闭的
`KATAGO_CUDA_GATE_ROWQUAD_R1`，只在物理B14/S361/C768异位门启用
192线程/行、float4读取及64位half写入；其他批次走原标量kernel。
FP32仿射、精确expf/SiLU、RN half边界均保留。已有plan开启须显式
绑定该key及schema2精确构建；无plan时允许环境变量诊断，不扩大认证。

- 两臂B1/B14/B16、两lane的完整native输出共**612,960个FP32元素**
  与已验收参考逐位相同。两臂Worker W1/W32共**512请求/结果**，
  原C++ FP32全部11项门、每组top-1 **128/128**、Drain通过，原始
  protobuf重新解码、重建请求与比较结果均通过。34项plan单测、
  1项形状边界单测、6个实际CLI失败关闭负例通过。
- 唯一固定B14/S2前向ABBA（W80/N1000/lane）：event NN rows/s
  **1207.5010→1234.4680，+2.2333%**；两臂spread **0.6407%/0.0237%**。
  独立wall **+2.0514%**，spread **0.7261%/0.1200%**；8000原始事件复核。
- 唯一Windows B14 cap/S2/C64 uncached Worker ABBA（W256/N8192，
  ownership=false）：RPC/s **1216.6413→1235.0275，+1.5112%**，
  spread **0.4745%/0.0921%**，两臂平均物理batch均**13.93197**。
  全部32,768原始延迟行及计时边界、计数、身份、Drain复核。
- 迁移门：旧ONNX q64/q128 × B1/B16整图ORT通过，两个graph首批及
  再捕获后发射回归通过；旧default/c32 × W1/W32各128局黄金门通过。
  编译PTX的18模块原字节相同，2个executor模块只增加新入口，原函数
  参数/指令不变（局部label编号可变）；CLI/test PTX一致。
  六份staged及六份最终路径实际eval加载、新key与STEM反向环境优先级
  均通过；最终throughput再跑128局原始protobuf/11门/top-1/Drain通过。
- 正式Windows binary SHA **43ff8eb87cfe20ef0b9db7ef3d784b727f9960d1795c34bf9915c1995df20bda**；
  kernel ID **sha256:f126639f1afead4116f51559ffd64208c321f12aa83cda85cf16dc76b0c05de9**，host revision仍1。
  新key仅throughput plan=1，其他五份=0；只替换binary、六Win计划及
  throughput配置注释，原八文件已逐字节备份。其余13保护文件（含WSL）不变。
  最终收据：`target/fork-parity-20260908/trunk-gate-rowquad-integration-r1/promotion-r1/acceptance.json`。

这些Windows同binary A/B收益不能更新G0的WSL/WSL Fork比例**47.889522%**，
也不等同四通道重复算子的+303%收益。此前两通道不稳定及模型副本负收益
继续关闭；总目标仍在推进，后续从新基线审计剩余热点再注册新候选。

**2026-09-14 新基线剖析与 Attention 输出投影布局裁决。**
`rowquad-current-kernel-profile-r1`对当前正式rowquad binary采集一次
Nsight轨迹。两lane各380次完整前向逐项验证335个kernel位置/形状/
资源，按预注册每lane末100次统计。FFN占kernel sum **41.498%**，
Attention **37.733%**；其中DualFFN **27.938%**、FFN down **13.560%**、
QKV **13.345%**、Attention core **13.334%**、输出投影 **7.334%**。
只描述当前有剖析开销的kernel时间构成，不把不同日期的轨迹差值当收益。
新rowquad实际替换10个块间C768门，final BN仍是原`bn_silu_f16_kernel`；
原先11次C768门的热点统计包含该未替换final BN。
实际stdout构建/16项plan tactics、两流路径及335位置语法全通过。
提取器r1误把ProcessStreams的`stdout_UUID.log`当成裸`stdout`，仅CPU
r2修正文件名匹配，无GPU重跑；原失败保留。

随后`attention-out-nn-b14-r1`只测试输出投影M5054/N384/K384的NN
权重布局。Fork当前该层采用独立TileLang AOT及[K,N]权重，但其
FP16累加/残差/输出不迁移；Rust仍FP32 compute、beta1、FP32 C=D。
这与已关闭的N768外层up实验不同，旧全K384布局探针也未覆盖B14。
通过当前正式完整模型DEBUG边界3/4捕获第一层Attention的真实输入/
残差/输出，CPU解析同SHA模型导出权重；无截断前缀推理。两条同时
存活的流中TN、NN、NN重复共六份完整输出，各**1,940,736**个FP32
元素均与正式前向结果逐位相同，全部FP64条件数界/只读/guard通过。

唯一双流ABBA（W80/N1000，C的D2D恢复在event外）得到event算子
矩阵行/s均值**390,545,215→419,646,668（+7.4515%）**，但TN/NN
spread **3.5061%/5.6423%**；独立lane host计时均值+18.9765%，
spread **10.8257%/1.1313%**。NN event和TN host均超过固定5%门，
因此**拒绝、不重试、不接入整网或Worker**。8000原始event及16份
计时前后完整输出均CPU复核。正式源码/binary/计划不变，收据为
`target/fork-parity-20260908/attention-out-nn-b14-r1/closure.json`。
下一项仍从剩余GEMM/Attention热点选择不同实现；本项算子均值不可
作为已采纳收益，也不更新Fork比例。

**2026-09-14 DualFFN 同CTA、16 warp候选关闭。**
`dualffn-warp16-r1`针对当前27.938% kernel sum的DualFFN，保持
M128/N64及80×9 CTA网格，把每CTA由4 warp改为16 warp，分摊
FP32累加器。原K32版因512线程的B加载映射无法覆盖最小向量而编译
失败；未启动GPU，保留失败并在计时前登记K64修订（warp32×16×64，
epilogue vector4，stage3），不涉及split-K或降精度。
当前完整原生模型DEBUG边界6捕获真实第一FFN的normed及SwiGLU
结果，权重与同SHA压缩模型解包字节核对。两条同时存活的流中六份
完整输出共**34,933,248 half元素**逐位一致；PTX确认FP32 MMA，
精确SiLU及两次RN half边界沿用生产实现。CPU检查器对裸`.entry`
的首版误匹配保留，r2修正，无GPU重跑。

候选寄存器**236→96/线程**，共享内存**48→96 KiB/CTA**，线程
**128→512**；初始化前occupancy API的0值发生在CUTLASS大共享内存
opt-in之前，不能作为实际occupancy结论。唯一双流W80/N1000 ABBA
event矩阵行/s几何均值**52,117,067→37,487,020（−28.0715%）**，
两臂spread **0.0784%/0.0297%**；lane host **−27.8580%**，spread
**0.1276%/0.2667%**。8000事件及16份计时前后完整输出复核通过，
稳定负收益，**拒绝、不重试、不接入整网/Worker**。不将寄存器减少
等同占用率或吞吐提升；没有硬件计数器证据，不指定慢速机制。
正式源码/binary/六份Windows计划及WSL保持，收据：
`target/fork-parity-20260908/dualffn-warp16-r1/closure.json`。

**2026-09-14 Fork 的双独立FFN投影组织验证。**
`ffn-two-nn-projections-r1`核对当前Fork B14 winner为
`dual_ffn-fallback-cublas-swiglu`，禁用fused及single-wide，实际FFN
组织为两个独立N1152投影再SwiGLU。Rust的`DUALFFN=0`是一个
N2304投影，不能用其既有负收益代替本项验证。本项只借鉴双投影和
NN布局，使用Rust的FP32 compute/scale、half输入及输出边界，精确
SiLU复用原functor；不复制Fork的半精度计算设置。
当前真实第一FFN完整夹具不变，六份最终输出共34,933,248 half逐位
一致，另保存八份完整投影中间值验证重复/跨lane一致。实际Lt130600
DLL绑定，REQ8首候选ID21/tile15/splitK1/reduction0/swizzle0，两lane
一致，描述符读回FP32，无算法计时筛选。两次投影和独立SiLU均包在
同一event内，32 MiB workspace按lane分配、Lt handle共享。

唯一双流W80/N1000 ABBA，event矩阵行/s几何均值
**51,636,766→34,113,160（−33.9363%）**，两臂spread
**0.0245%/0.1162%**；lane host **−34.1112%**，spread
**0.5759%/0.4762%**。全部8000事件及16份前后完整输出复核；
**稳定负收益，关闭、不重试、不接入整网/Worker**。当前融合
DualFFN保留，正式源码和安装项不变，Fork比例不更新。收据：
`target/fork-parity-20260908/ffn-two-nn-projections-r1/closure.json`。
该结果只裁决本候选，不能外推为Fork在其精度设置下也慢。

**2026-09-14 RMS 四通道向量化候选关闭。**
`rms-eight-lane-vec4-r1`以当前4.403% kernel sum的RMS为对象，
借鉴Fork向量访存组织，保留Rust FP32输入/scale与求和顺序。每行
32线程改为8线程，每线程同时计算原来的4个相邻逻辑lane；原XOR
16/8/4映射为线程间4/2/1，原2/1在寄存器内合并，二叉求和树仅
交换可交换的子项、不重关联。128线程CTA由4行变16行，float4
读入和uint2打包half写出，仍runtime ncols除法、rsqrtf及RN half。
第一RMS夹具复用当前正式完整模型边界3，scale与原压缩模型norm1
字节重新核对；没有使用旧WSL前缀输入。真实B14及20组尾行/数值
范围用例，共126份、**63,548,928 half元素逐位一致**。PTX复核
12→48条FP32平方FMA链、对应shuffle顺序、向量读写和half舍入；
寄存器36→72，local/shared均0。

唯一W80/N1000双流ABBA的event矩阵行/s几何均值为
**659,126,292→729,999,643（+10.7526%）**，但两臂spread
**9.4880%/39.7104%**；lane host均值+10.1090%，spread
**0.6398%/13.5551%**。超过固定5%门，**关闭、不重试、不接入
整网/Worker**；8000事件和16份前后完整输出留存并复核。不以均值
报告已获收益，不推断波动原因。正式RMS、已验收rowquad版本和
WSL保持，收据：`target/fork-parity-20260908/rms-eight-lane-vec4-r1/closure.json`。

**2026-09-14 M64 strict Attention：算子门通过，整网波动超门，已关闭。**
`fa4-strict-m64-r1`只把当前M128/N96/stage1的query tile缩为M64，
保留N96、128线程、严格FP32 QK/PV和softmax源文件。离线隐藏GPU
编译成功；新ABI从实际PTX/host IR提取，grid3×12×14→6×12×14、
动态共享20→16 KiB，参数仍[8,8,8,8,4,12]。每CTA query行减半，
完整精确指数数据流196→98处、FP32 MMA96处，未改变指数或归约门。
Driver加载属性255→168寄存器；local bytes40→0是函数属性读数，
与此前profiler的local-memory字段0不是同一观测，不作流量/因果推断。

四组固定合成输入及当前完整模型第一Attention真实Q/K/V，五组×
基线/候选两流共**29,111,040 half输出逐位一致**。真实fixture
只拼接边界4中仍有效的旋转Q/K和原V，参考为normed；未比较未写
scratch。唯一真实B14/S2 W80/N1000 ABBA得到核心算子event
NN行/s **344,861.57→372,558.43（+8.0313%）**，两臂spread
**0.0197%/0.0319%**；lane host **+7.5712%**，spread
**0.2062%/0.0505%**。8000事件、16份完整计时前后输出独立复核。
CPU计时脚本首版导入元组漏项在admit/register前失败，r2修正并保留
原文件；没有GPU重跑。算子门通过，**尚非整网或Worker收益**。
算子阶段要求以默认关闭、单独资产指纹绑定的B14选项接入，经过完整
原生/Worker数值门、真实路径、整网和Worker各唯一ABBA后再裁决。
该阶段生产未改，算子阶段收据：
`target/fork-parity-20260908/fa4-strict-m64-r1/ready-for-integration.json`。

后续 `fa4-strict-m64-integration-r1` 完成默认关闭的独立开关和资产
指纹接入；旧 strict 指纹与 core build ID 不变，20 个既有 PTX 模块
在新旧 CLI/对拍构建中逐字节相同。36 项计划测试、3 项资产/作用域
测试、12 项真实 CLI 拒绝/兼容/优先级检查通过。两臂 B1/B14/B16、
各两流的 **612,960 个整网 FP32 输出逐位一致**；Worker W1/W32
四组共 **512 对原始请求/响应**重建复核，11 项 FP32 黄金门、
top-1 **512/512** 和 Drain 均通过。M64 只在物理 B14 启动，
其他批次保持 q64-serial。

唯一完整前向 B14/S2、W80/N1000 ABBA 的 event NN 行/s 为
基线 **765.6988/650.7573**、M64 **670.4171/669.7399**；两臂
spread **17.6627%/0.1011%**，超过固定 5% 门。event 几何均值
705.8924→670.0784（−5.0736%），主机统一墙钟 789.6103→
788.9361（−0.0854%，spread 3.7001%/1.0761%）；这些均值不能
认定为稳定收益或回退。8000 个事件与计时/路径信息已重建复核。
**不重试，不进入 Worker 性能阶段，不安装。**四份运行时/测试源码
已按冻结原件恢复，M64 包移至实验档案；正式 rowquad binary、六份
Windows 计划、配置与 WSL 保持。算子 +8.0313% 不转写为整网收益，
Fork 历史同口径比例不更新。最终收据：
`target/fork-parity-20260908/fa4-strict-m64-integration-r1/closure.json`。

**2026-09-14 DualFFN horizontal CTA：稳定无收益，已关闭。**
`dualffn-horizontal-r1` 保留 M128/N64/K32、warp64×32、stage3、
Count8、FP32 累加及原精确 SwiGLU，只把 swizzle2 的 grid80×9
改为 horizontal grid18×40。CPU 枚举确认同一组 720 个逻辑 tile
各覆盖一次；补齐 CUTLASS DualGemm 要求的 `get_tile_offset(int)`。
这不是 Fork 当前选中的战术：Fork 源码保留相同原瓦片族，认证 B14
明确关闭 fused/wide-single FFN；Rust 的拆分路线已有独立负结果。

当前正式模型第一 FFN 的真实输入/输出、原模型权重字节重新核验，
六份完整输出共 **34,933,248 half** 逐位一致。新旧核均236寄存器、
48KiB共享、无local；完整浮点opcode/常数多重集一致，不将这一
静态检查代替数值门。唯一双流 W80/N1000 ABBA，event矩阵行/s
几何均值 **51,749,201.29→51,636,767.55（−0.217267%）**，
spread **0.411333%/0.057227%**；lane host **+0.053779%**，
spread **0.113840%/0.100699%**。8000事件和16份前后完整输出复核。
两流计时结束后才提取事件和下载输出，避免下载与另一流计时重叠。
稳定但未达1%门，**不接入整网/Worker、不重试、不改正式代码**。
Fork比例保持；收据：
`target/fork-parity-20260908/dualffn-horizontal-r1/closure.json`。

**2026-09-14 strict Attention Graph：数值通过，Worker 稳定回退，已撤回。**
`strict-attention-graph-r1` 临时加入默认关闭的 Graph 开关及独立
`strict_graph_revision=1`，要求计划同时绑定 strict family、q64-serial
和 NOGRAPH=0；旧计划保留 NOGRAPH=1，捕获失败直接报错。
实验 nnbench 通过显式 `--graph-replay` 测量真实重放；两臂都串行
准备各 lane，捕获/预热不计时，计时阶段仍由屏障同时开始。

20 份 PTX 和原 strict AOT 身份完全不变。B1/B14/B16、双流双 slot
的首次启动、后续捕获后重放、变输入全部通过，612,960 个整网输出
与正式版本逐位一致；Worker W32/W1 两臂共512个原始协议请求/响应
通过11项门和top-1 512/512。36项计划测试、2项资产测试、2项阶段
测试、17项实际计划加载边界和3项测量参数拒绝检查通过。

唯一完整前向 B14/S2 W80/N1000 ABBA：event 行/s 几何均值
**1223.045792→1240.978964（+1.466272%）**，spread
1.464865%/0.005108%；统一墙钟 **1219.242398→1231.646198
（+1.017337%）**，spread1.323092%/0.265597%，获准进入Worker。
随后唯一 B14/S2/C64、uncached、W256/N8192 Worker ABBA：
**1231.483873→1124.477401 RPC/s（−8.689230%）**，spread
**0.411025%/0.270187%**。每阶段恰有8192 NN行，平均批次约
13.88/13.92，Graph组确认实际重放且无失败或直接回退。
8000事件与32768条Worker计时全部重算。固定前向的小收益未延续
到服务链路，不能将其视为Worker收益，也不据本结果归因某一驱动机制。

不重试、不安装；五份实验源码已按冻结原件恢复，候选源码和独立
构建留档。正式 rowquad binary、六份Windows计划、配置和WSL不变；
当前源码仍强制 strict NOGRAPH=1，普通 nnbench kernel 仍无Graph。
Fork同口径历史比例不更新。收据：
`target/fork-parity-20260908/strict-attention-graph-r1/closure.json`。

**2026-09-14 FFN down 的 Fork NN CUTLASS 分块：数值通过，计时不稳定，关闭。**
`ffn-down-nn-cutlass-r1` 对照当前Fork B14的
`linear2-m128-n128-k32-s3-cutlass`，借用NN权重布局及
M128/N128/K32、warp64×64、stage3、swizzle1。累加、epilogue、
残差C=D与输出保留FP32；区别于旧TN三分块和TN dense-grid候选。
当前正式binary完整模型边界4/6重新捕获残差、FFN中间输入及输出，
同SHA TF3从头解析至EOF，确认首层 `ffn_linear2` 原始权重及转置。
两条同时存活的流上，基线/候选/重复共 **11,644,416个FP32输出**
全部逐位一致，全量FP64条件数界、只读及guard检查通过。实际候选
grid40×3、128线程、234寄存器、48KiB共享、0 local；编译MMA保持FP32。

唯一S2 W80/N1000 ABBA：event矩阵行/s几何均值
**162,967,608.43→123,797,001.99（−24.035823%）**，A/B spread
**6.351429%/2.479591%**；lane host **+9.716498%**，spread
**0.755327%/19.600659%**。8000事件和16份计时前后完整输出复核。
稳定门失败，均值仅作诊断，不能认定稳定性能回退或收益；
**不重试、不接入整网/Worker、不改正式源码和安装项**，Fork比例不更新。
CPU矩阵名断言与Windows `FIXED` 符号冲突两处修正均版本化留档，
修正发生在算子GPU执行前，不改候选算法或数值/性能门。收据：
`target/fork-parity-20260908/ffn-down-nn-cutlass-r1/closure.json`。

**2026-09-14 QKV producer 与 RoPE 同CTA融合：数值通过，稳定无收益，关闭。**
`qkv-producer-rope-r1` 使用TN CUTLASS M128/N64/K32、warp64×32、
stage3、swizzle1，在GEMM half epilogue结束并同步后，由同CTA读取其
自身相邻half对，按原FP32指令序列旋转Q/K并原地RN写half；V保持原值。
保留两次half舍入，减少一次启动，仍有全局内存写入与读回。
Fork当前计划的producer-RoPE AOT为disabled；本项是新候选，区别于
旧TN M128N128K64 CuTe QKV和Attention消费侧RoPE实验。
基线使用当前Lt heuristic首选及正式RoPE PTX的grid948/1024线程配置。

当前完整模型捕获的边界3 RMS输入、边界4原QKV与旋转Q/K，结合当前
Rust原生模型解析器导出的权重和cos/sin作为参考。两条流、六类输出
共 **69,866,496个half值逐位一致**，全量FP64条件数界、重复、
跨流、只读和guard通过；CPU枚举确认1,940,736个Q/K相邻对各由
唯一生产CTA持有。实际融合kernel为grid40×18、128线程、142寄存器、
36KiB共享、0 local，MMA保持FP32。

唯一S2 W80/N1000 ABBA：event矩阵行/s
**85,504,503.77→85,498,716.03（−0.006769%）**，A/B spread
**0.040614%/0.027071%**；lane host **+0.742876%**，spread
**0.990229%/0.358071%**。8000事件与16份计时前后完整输出复核。
两项收益均未达到1%门，**稳定关闭，不重试、不接入整网或Worker**；
不据组合结果单独归因GEMM或RoPE，也不否定其他producer融合实现。
正式源码、binary、计划和配置保持，Fork同口径历史比例不更新。收据：
`target/fork-parity-20260908/qkv-producer-rope-r1/closure.json`。

**2026-09-14 QKV half epilogue 内融合RoPE：算子通过，整网稳定回退，已撤回。**
`qkv-epilogue-rope-r1` 保持前一候选的TN M128/N64/K32、warp64×32、
stage3，改在CUTLASS目的迭代器写回前旋转half寄存器片段。先RN成half，
再原样FP32旋转、再次RN成half；Q/K省掉中间全局写回和读回，V不变。
实际编译ThreadMap每片段8个half、8轮epilogue，CPU覆盖及完整模型
首QKV参考共 **69,866,496个half值** 逐位一致。实际156寄存器、
36KiB共享、400字节stack/local，编译器报告0 spill。
唯一算子ABBA稳定通过：event **+2.402227%**，lane host **+3.211299%**。

`qkv-epilogue-rope-integration-r1` 随后直接嵌入计时可执行文件提取的
相同CUBIN；C++导出的400字节、8字节对齐参数模板与PTX声明一致，
四个指针槽在64/112/208/304。临时默认关闭的B14接入新增独立
`qkv_epilogue_artifact` 指纹，绑定CUBIN、参数ABI和主机调用源码；
B1/B16保留原路径，计划必须显式绑定新开关和TN布局。
六组native两流全部输出 **612,960个FP32值逐位一致**，四组Worker
**512/512 top-1、11项数值门与Drain通过**；36项计划测试、2项资产/ABI
测试及13项实际CLI绑定检查通过，旧/新CLI与测试构建的20份核心PTX相同。

唯一B14/S2 W80/N1000完整前向ABBA：event
**1228.867158→1209.689310（−1.560612%）**，A/B spread
**1.128787%/0.804839%**；wall **−1.492148%**，spread
**0.444160%/0.088439%**。全部8000事件复核，属于稳定整网回退。
**不重试、不运行Worker性能、不安装或迁移计划**；五份主机/测试源码
按原SHA恢复，新模块和AOT资产移入实验归档，正式rowquad版本不变。
算子收益不转写为整网或Fork差距收益；这些计时没有确定回退的具体原因。
原C++类型查找编译失败及CPU marker筛选修正均版本化保存，没有重跑GPU
数值或计时。收据：
`target/fork-parity-20260908/qkv-epilogue-rope-integration-r1/closure.json`。

**2026-09-14 QKV/RoPE 只读参数：整网与Worker通过，已安装。**
`qkv-immutable-params-rope-r1` 是独立的新参数ABI/数据流：主机预先填写
cos/sin，GPU只接收一个const grid-constant Params，不再修改按值参数。
GEMM分块、half寄存器片段迭代器和精确RoPE指令保持前次实现原样；
编译及实际加载均确认 **400→0字节stack/local、156→146寄存器**，
36KiB共享不变，0 spill。两流六类输出 **69,866,496个half值逐位一致**，
完整ThreadMap覆盖、FP64条件数界、重复/跨流/只读/guard通过。
唯一算子ABBA：event **+26.393169%**，lane host **+25.929807%**，
两臂event spread **0.108409%/0.025666%**，host **0.050160%/0.227934%**。
这是算子证据；没有重跑已关闭的可变参数候选，也不据此归因其整网回退。

`qkv-immutable-params-integration-r1` 嵌入从本轮计时exe提取的相同CUBIN。
C++导出单个400字节、8字节对齐参数及六个地址槽，输入64/权重112/
源208/cos288/sin296/输出304，Rust加载器与PTX核对。新开关
`KATAGO_CUDA_QKV_IMMUTABLE_R1` 默认0；schema2显式绑定独立artifact、
strict Attention与TN权重，B1/B16保持原路径。
36项计划测试、2项资产/ABI测试、13项真实CLI检查通过；
六组native两流 **612,960个FP32输出逐位一致**，四组Worker
**512/512 top-1、11字段与Drain通过**；20份核心PTX保持原字节。

唯一B14/S2 W80/N1000完整前向ABBA：event
**1241.986476→1279.076671（+2.986361%）**，spread
**0.495805%/0.537163%**；wall **+3.617951%**，spread
**0.487810%/0.167020%**。8000事件重算通过。
随后唯一uncached B14cap/S2/C64 W256/N8192 Worker ABBA：
**1233.865301→1274.991675 RPC/s（+3.333133%）**，spread
**0.471766%/0.301123%**；平均batch两臂均 **13.920146**。
32768条原始计时、请求身份、零失败与Drain检查通过。

四组旧ONNX/ORT、两组Graph回归、四组原默认/C32 Worker黄金门、
六份暂存计划加载均通过后，已安装新CLI、六份Windows计划及吞吐配置注释；
最终路径六次加载及吞吐128局/Drain再次通过。旧五计划显式新开关0，
吞吐新开关1，原rowquad等开关保持。WSL及原数值门不变，不累计收益，
不更新跨口径Fork比例。AOT目录保存精确源码、ABI及独立重建脚本。
验收收据：
`target/fork-parity-20260908/qkv-immutable-params-integration-r1/promotion-r1/acceptance.json`。

**2026-09-14 已安装immutable-QKV后的新剖析与DualFFN vector4排除。**
`immutable-current-kernel-profile-r1` 对已安装新CLI、当前17项吞吐tactic
做一次Windows B14/S2空盘Nsight诊断。全部760个完整前向符合新的
**302-kernel**源码序列（融合省掉33次独立RoPE）；每流固定取最后100个，
合计200前向/60,400核。真实路径和QKV资源 **146寄存器/36KiB/0 local**
均通过核对。kernel时间总和中，FFN **45.105548%**、Attention
**34.809079%**；DualFFN **29.221469%（123.912257微秒/次）**、
FFN down **15.884078%（67.355682微秒）**、QKV/RoPE
**14.387483%（61.009440微秒）**、Attention core **13.130922%**。
总kernel时长2798.698737毫秒，跨流并集2220.540085毫秒；这些有剖析开销、
跨流重叠和不同起止窗口，**不作为Worker吞吐、可节省墙钟比例或新Fork比值**。
该诊断用于确定下一项热点，未从旧/新trace直接推算收益。

随后独立 `dualffn-epilogue4-r1` 保持M128/N64/K32、warp64×32、stage3、
swizzle2/grid80×9，仅将投影与精确SwiGLU的epilogue片段从8half改为4half。
FP32累加和既有half边界保持；实际寄存器 **236→234**、共享48KiB、
0 local及潜在2CTA/SM不变。用当前已安装CLI在完整模型边界6重新捕获
首FFN输入/输出，权重直接核对同SHA解压模型原字节，未套用旧prefix输入。
两条同时存活的流、6份输出共 **34,933,248个half值逐位一致**，
全部guard、只读、FP32 MMA/精确SiLU模板与编译浮点操作/常量集合通过。

唯一S2 W80/N1000算子ABBA：event矩阵行/s几何均值
**52,039,888.13→50,080,263.96（−3.765619%）**，两臂spread
**0.392035%/0.011894%**；lane host **51,597,974.98→49,825,154.36
（−3.435834%）**，spread **0.041686%/0.140066%**。
8000事件与16份完整计时前后输出复核，属于稳定回退；
**关闭，不重试、不接入整网、不跑Worker性能**。Fork当前B14仍关闭
DualFFN，不能把此候选称为其已选战术。正式immutable-QKV安装项、源码、
计划及配置均保持，已有Worker +3.333133%验收和历史Fork比值不变。
收据：`target/fork-parity-20260908/dualffn-epilogue4-r1/closure.json`。

**2026-09-14 DualFFN仅权重L1缓存候选已排除。**
`dualffn-weight-l1-r1` 针对当前剖析中占kernel时间总和29.221469%的DualFFN，
保持M128/N64/K32、warp64×32、stage3、swizzle2/grid80×9和vector8，
仅把两份权重B0/B1的异步加载从Global改为Always，输入A仍为Global。
CUTLASS设备头使用隔离副本；除类/构造函数改名、相同依赖的include路径及
一个CacheOpB模板参数外，源码一致。迭代器、MMA策略、epilogue类型静态断言
相同。编译前登记、GPU前核对：PTX从24个cg静态位置变为12cg+12ca，
FP32操作/常量多重集相同；事后CPU反汇编也确认实际SASS从24个
LDGSTS BYPASS变为12个BYPASS+12个普通LDGSTS，均保留LTC128B。
两侧均 **236寄存器/48KiB/0 local、0 spill/潜在2CTA每SM**。

复用并重新CPU核对当前正式CLI完整模型边界6的输入/参考、17项tactic、
模型SHA和解压权重原字节；未复用上一候选的输出或性能数据。
两条存活流、6份输出 **34,933,248个half值全部逐位一致**，guard和只读检查通过。
唯一S2 W80/N1000算子ABBA：event矩阵行/s几何均值
**51,634,655.70→51,224,481.49（−0.794378%）**，A/B spread
**0.016352%/0.438816%**；lane host **51,313,381.76→50,914,397.32
（−0.777545%）**，spread **0.160570%/0.149889%**。
8000事件和16份计时前后完整输出复核。**稳定回退，关闭，不重试、
不接入整网、不跑Worker性能、不迁移计划**。该结果不能推断实际cache hit率。
Fork当前B14关闭DualFFN；本候选不是Fork已选战术。现有immutable-QKV
生产安装、Worker +3.333133%验收和历史同口径Fork比值保持；下一步转向
独立的FFN加载流水或FFN down实现，不能复活这个已关闭缓存候选。
收据：`target/fork-parity-20260908/dualffn-weight-l1-r1/closure.json`。

**2026-09-14 交错权重单GEMM、8→4压缩SwiGLU写回候选已排除。**
`dualffn-interleaved-single-r3` 以相邻输出列放置gate/up权重
（TN `[2*j,:]=gate[j,:]`、`[2*j+1,:]=up[j,:]`），单次FP32累加
M128/N128/K32、warp64×64、stage3完成两份投影。投影先RN half，
在原CUTLASS寄存器片段内调用生产版精确SwiGLU，再以8字节写出4个half，
不产生2N宽全局中间张量。最终有效CTA仍128×64、grid80×9/swizzle2，
原来的两个half边界、精确expf、FP32计算保持。
Fork源码的wide-single思路使用拼接NN权重、cublasHgemm与独立SwiGLU，
且B14计划禁用；这里只借鉴单投影组织，不宣称是Fork已选战术。

首版r1在**GPU前编译门**被拦截：CUTLASS half数组代理之间直接赋值，
使局部写回payload未初始化，候选PTX缺少SwiGLU计算；r2显式整数打包又
因对代理直接调用`.raw()`编译失败。两份源码、合同和失败构建均保留，
未做GPU数值或测速。r3先将代理物化为half_t，再明确组合uint2原始位；
没有修改数学、布局或性能协议。这里是代码修复，未通过重复测速挽救结果。

实际编译ThreadMap的CPU导出检查覆盖720个CTA，**5,822,208个有效输出
均恰好写一次**；全部片段槽位和尾行谓词通过。当前完整模型首FFN
输入/参考、正式CLI/17项tactic和模型原始权重重新核对；GPU两条流6份
输出 **34,933,248个half全部逐位一致**，guard与全部只读权重检查通过。
实际资源 **236→226寄存器、48KiB、0 local/0 spill、潜在2CTA每SM**。

唯一S2 W80/N1000 ABBA的event矩阵行/s几何均值
**51,643,097.05→50,529,896.74（−2.155565%）**，A/B spread
**0.049056%/0.056004%**；lane host **51,390,322.72→50,072,292.25
（−2.564744%）**，spread **0.062503%/0.149517%**。
8000原始事件及16份计时前后完整输出独立复核。静态PTX的MMA、cp.async、
barrier、div与exp位置数量相同；写回从8个16字节变为16个8字节位置。
这只是后续定位依据，不能替代计数器或认定退化原因。
**关闭这个vector8→compact4配置，不重试、不接入整网、不跑Worker性能。**
下一项可独立检查更宽的投影片段能否恢复16字节压缩写回，或转向FFN down；
必须重新登记新实现、映射/数值门与ABBA。正式生产源码/安装/配置不变，
已认证immutable-QKV收益和历史Fork比例保持。
收据：`target/fork-parity-20260908/dualffn-interleaved-single-r3/closure.json`。

**2026-09-14 单FFN的16→8压缩写回候选已排除。**
`singleffn-compact8-r1` 是独立登记的新线程映射：在相同交错TN单GEMM
M128/N128/K32、warp64×64、stage3/swizzle2上，将投影片段改为16half，
精确SwiGLU后用uint4写回8half。CUTLASS从half8专用mixed epilogue
迭代器转为通用vector16迭代器；并非只替换一个存储指令，旧compact4
候选保持关闭。FP32累加、精确expf及原half边界保持。

CPU导出实际ThreadMap：每CTA 1024个16half投影向量，720个CTA的
**5,822,208个有效输出恰好覆盖一次**，全部片段和尾行通过。
当前正式模型/CLI/17项tactic/原始权重与中间输入复核；两流6份输出
**34,933,248个half全部逐位一致**，所有guard和只读检查通过。
实际 **236→226寄存器、48KiB、0 local/0 spill、潜在2CTA每SM**。
静态PTX已恢复8个16字节写回位置，与基线相同；MMA64、cp.async24、
barrier19、div64、exp64的位置计数也相同，不能据此断言动态开销相同。

唯一S2 W80/N1000算子ABBA，event矩阵行/s几何均值
**51,955,521.06→50,568,318.82（−2.669980%）**，A/B spread
**0.602065%/0.016007%**；lane host **51,510,467.33→50,281,635.92
（−2.385596%）**，spread **0.037845%/0.094360%**。
8000原始事件和16份计时前后完整输出独立复核。两臂稳定，
**关闭本配置，不重试、不接入整网、不跑Worker性能**；不能把独立两轮
compact4/compact8记录当成两者的直接配对性能比较。
两种已验证压缩写回配置都没有达到相对生产DualFFN的采纳门，
下一步转向FFN down加载/计算组织。正式immutable-QKV生产项、
已认证Worker收益和历史同口径Fork比例保持。
收据：`target/fork-parity-20260908/singleffn-compact8-r1/closure.json`。

**2026-09-14 FFN down的N32双warp候选已排除。**
`ffn-down-n32-r1` 保持TN half输入/权重、FP32累加、FP32原位残差/输出和
alpha=beta=1，采用独立M64/N32/K32、warp32×32、stage3、swizzle1。
相对当前Fixed的64×64×32/s6、632块/128线程、88寄存器/48KiB，
候选实际 **79×12=948块、64线程、96寄存器、18KiB、0 local**，
潜在5CTA/SM（10warp）。这是新的N32组织，之前N64/s6稠密网格和
Fork的N128/NN配置继续关闭；驻留数量不能单独证明更快。

用当前正式immutable-QKV CLI及17项tactic重新捕获完整模型边界4/6，
输入同SHA空盘B14/S1；严格核对QKV融合、strict Attention、DualFFN和
Fixed身份标记。完整TF3解析至EOF，首FFN down原始1152×384权重及
TN/NN转置逐字节验证。CPU编译门的初版名称筛选误匹配辅助kernel的
warp形状，r2仅改为精确主分块类型筛选，保留原检查器/修正记录；
二进制、候选源码和后续数值/性能协议未变，GPU运行前FP32检查通过。

两流6份输出 **11,644,416个FP32全部逐位一致**，完整FP64 A@W+C与
误差条件界通过，最大绝对误差 **2.1841656235e-6**、最大条件界占比
**0.0051114454**。全部输入、两布局权重、残差源只读和guard通过。
唯一B14/S2 W80/N1000 ABBA：event矩阵行/s几何均值
**168,781,733.62→127,553,663.44（−24.426856%）**，A/B spread
**0%/1.177950%**；lane host **88,467,805.22→71,145,594.09
（−19.580243%）**，spread **0.072083%/0.283003%**。
每次launch之前同流D2D恢复残差，reset在本流event外而计入host；
可能与另一流事件重叠。此边界与整网、Worker及DualFFN算子口径不同。
8000事件和16份计时前后完整输出独立重算，两个口径均稳定回退。
**关闭N32候选，不重试、不接入整网、不跑Worker性能。**
42项正式保护文件及生产计划保持；本次不能更新历史Fork/Rust比例。
新的完整FFN down fixture可供独立加载/流水方案定位，不能复活本配置。
收据：`target/fork-parity-20260908/ffn-down-n32-r1/closure.json`。

**2026-09-14 FFN down的N64三级预取未通过主性能门。**
`ffn-down-n64-s3-r1` 保持TN M64/N64/K32、warp32×32、FP32累加/
残差/输出和alpha=beta=1；采用stage3/swizzle1、79×6网格。
实际 **128线程、80寄存器、24KiB共享、0 local/stack/spill、潜在4CTA/SM**。
相比生产Fixed的64×64/s6、48KiB，测试减少预取级数是否受益；
此前N64/s6 dense-grid和N32/s3均保持关闭，本次没有扫参或重复测速。
Fork所选s3仍是NN N128/FP16累加，本候选只借鉴流水组织，未降低精度。

CPU重新完整解析同SHA模型至EOF，复核当前正式CLI/17项tactic、完整模型
边界4/6捕获和首FFN down权重，再逐字节复制fixture；未声称新做GPU捕获。
两条存活流6份输出 **11,644,416个FP32全部逐位一致**，完整FP64 A@W+C
及条件误差界通过，最大绝对误差 **2.1841656235e-6**，最大界占比
**0.0051114454**；FP32 MMA代码生成、所有guard与只读检查通过。

唯一B14/S2 W80/N1000 ABBA：event矩阵行/s几何均值
**167,364,171.33→155,002,663.62（−7.385994%）**，A/B spread
**1.430213%/1.046036%**；lane host
**88,882,634.46→101,003,758.38（+13.637224%）**，spread
**0.115655%/0.629638%**。8000事件、16份计时前后完整输出独立复核。
两个口径方向不同：host额外包含同流残差D2D恢复、发射和同步，
跨流重叠也不同；仅凭这些记录不能把差异归因于CPU发射开销。
**按预注册event主指标关闭，不改用host指标采纳、不重试、不接入整网。**
正向host结果完整保留，它不代表整网、Worker或Fork对照收益。
42项正式保护文件、生产计划和已认证immutable-QKV收益保持。
后续不继续按驻留数扫描此分块/预取族，转向尚未完成的同精度差距归因、
WSL新路径数值问题或有源码依据的不同热点组织。
收据：`target/fork-parity-20260908/ffn-down-n64-s3-r1/closure.json`。

**2026-09-14 Fork linear2累加精度受控归因：GPU计时不稳定，保留资源/数值证据。**
`fork-linear2-accumulator-attribution-r1` 从Fork所选linear2源码精确提取
CUTLASS类型作为FP16臂；另一臂只把GEMM和epilogue输入累加类型升为FP32。
两臂仍是NN M128/N128/K32、warp64×64、stage3/swizzle1、40×3网格，
half输入/权重/残差/输出、half epilogue计算及alpha=beta=1完全相同。
FP32臂先把累加结果RN half，再执行同一个half残差加法；这**不是Rust的
FP32残差/输出模式**，也不是拟降低生产精度的候选。

这是Windows CUDA13.3/CUTLASS3.9.2本地重编译的算子归因。输入使用重新
核对的当前Rust完整模型首FFN张量；原FP32残差统一RN half，原模型NN权重
完整解析至EOF并逐字节复核。并非Fork原生中间张量或已安装WSL Fork二进制。
实际SASS分别为 **64个HMMA.16816.F16/F32静态位置**；寄存器
**168→234**，均128线程、48KiB共享、0 local/stack/spill、潜在2CTA/SM。
这些证明累加精度和资源差异，不能将硬件计算成本与编译资源影响再拆开归因。

两个可精确求值的输入、真实模型输入，两流合计18份/34,933,248个half
完成检查：12份解析控制全部精确，6份真实输出各臂跨流/重复逐位一致，
全部guard和只读检查通过。两种精度之间有 **1,287,874/1,940,736** 个
真实输出位模式不同，maxAbs **0.00390625**。相对完整FP64点积加half残差，
FP16/FP32累加臂的RMSE分别 **3.1734719435e-4/9.9100111548e-5**。
这里的精度相关误差界仅为诊断，尤其FP16理论界较宽；未运行/替代Rust
整网黄金门，不能据此宣称任一臂满足Worker top-1。

唯一B14/S2 W80/N1000 ABBA保存8000事件和16份计时前后完整输出。
event矩阵行/s两臂几何均值 **261,797,926.48/122,349,184.43**，
名义变化 **−53.265793%**；FP16臂spread **10.187269%** 超过5%，
FP32臂 **0.058114%**。因此状态 **UNSTABLE_ACCUMULATOR_ONLY_DIAGNOSTIC**，
**不把约一半吞吐当成稳定GPU精度成本，更不能套到全网Fork/Rust差距**。
独立lane-host几何均值 **170,508,886.65/109,430,845.11（−35.821031%）**，
spread **1.669147%/0.123523%**，只在含half残差恢复/发射/同步的该口径稳定；
它不是统一墙钟吞吐，也不抵销GPU主指标波动。全部原始算术独立复核。
本归因配置关闭、不重复测速、不接入生产。42项生产保护文件和历史Fork
参考比例保持。下一步从已有WSL整网干预证据继续定位首层之后的数值分歧，
而非复活已被证伪的“仅替换第一次down即可修复”假说。
收据：`target/fork-parity-20260908/fork-linear2-accumulator-attribution-r1/closure.json`。

**2026-09-14 WSL首层之后的分歧定位：指向Attention输出投影。**
`g3-post-down-trace-r1` 沿用原G3 CUDA后端rlib及匹配依赖，直接rustc链接
独立诊断程序；未重编生产后端或更新已安装CLI。冻结691份依赖工件，实际
WSL完整backend指纹、输入及设备与原故障记录相同。完整图、分段图、
旧首层注入各跑B1/B14，分段为0–2、2–3、3–4、4–5、5–6及剩余层。
六组全部五头 **114,930个FP32输出**逐位复现原G3记录；完整/分段控制
通过，B1注入的全部中间张量也逐位自控。新增 **52份活跃张量**已独立复核。
这次重复旧注入仅用于观察后续边界，没有重新把已失败的首层修复当候选。

将B14首次down结果替换为重复B1后，首RMS输入和half输出均逐位一致；
首Attention的packed QKV与Attention核心half输出也全部逐位一致。
首次后续分歧出现在其 **FP32输出投影加残差**：每盘138624项中
**115,018项**位模式不同，maxAbs **7.1525573730e-7**、RMSE
**4.3619974055e-8**，14盘内部完全一致。下一RMS每盘 **241个half**
不同；首FFN的SwiGLU每盘 **7,422个half**不同，残差结果maxAbs
**2.0444393158e-4**。此前普通路径首RMS的46个half分歧也被完整复现。
该控制将新的局部差异定位到输出投影计算，尚未读取其实际Lt算法配置，
也不能证明只修这里即可恢复整个网络的两例top-1。

为后续重放准备了同输入/同残差的B1、B14输出，及完整模型解析得到的
首out_proj权重（offset **3,630,305**、source IC×OC=384×384），
同时保存NN/TN的RNE half布局。全部138624项的源权重FP64参考中，
B1/B14最大误差 **5.3059557104e-7/1.0869243852e-6**，RMSE
**4.0969642859e-8/7.8581998242e-8**；128个math.fsum样本差异为0。
权重尚未做这次GPU回读，FP64结果明确为**源权重参考**；下一步须先
用Lt重放逐位复现两种输出并核对上传权重，再交叉算法和batch形状。

完整输出仍保持原有B1 margin **+0.001103163**、B14 **−0.000593185**，
首层注入后的B14 **−0.000309229**；问题尚未修复。未跑新128例黄金门、
性能ABBA或计划迁移，42项生产保护文件和历史Fork参考比例不变。
收据：`target/fork-parity-20260908/g3-post-down-trace-r1/closure.json`；
重放输入：同目录`outproj-operands/manifest.json`。

**2026-09-14 WSL两处GEMM算法归因与联合干预：故障输入的排序恢复。**
`g3-outproj-lt-replay-r2` 在相同half输入、权重和FP32残差下逐位复现
首Attention输出投影B1/B14结果；交换原始Lt算法对象后，两个方向的
输出都逐位跟随算法。实际FP16 A/B、FP32累加/scale/C/D、alpha=beta=1，
6,238,080项输出及上传权重/输入/残差、257项尾保护已独立复核。
随后 `g3-down-outproj-injection-r1` 同时注入B1首次down与首次out_proj
结果，B14原始policy ch0的60−300分差 **−0.000593185→+0.002077579**，
14盘raw argmax均回到B1。完整/分段/旧单点注入和B1联合自控通过，
共153,240项五头输出、84份边界张量；后续首FFN down仍有FP32差异。
这是**一份唯一输入（映射到原两例）的张量干预**，不是完整128例黄金门
通过或生产修复。下一步改为真实Lt计算并验证独立输入及128例；此前
不测性能、不迁移WSL计划。42项生产文件与Fork参考比例保持不变。
收据：`target/fork-parity-20260908/g3-down-outproj-injection-r1/closure.json`。

**2026-09-14 WSL实际两处GEMM计算：独立完整128例门通过。**
`g3-two-gemm-execution-r1` 已将冻结输出替换改为真实模型权重和当前输入
驱动的Lt计算：首次down用beta0，首Attention out_proj用beta1及提前
保存的原FP32残差。114,930项五头输出、58份中间张量逐位复现前一轮
联合干预；算法模块不访问冻结输出。随后 `g3-two-gemm-golden128-r3`
使用全部128份原请求（113个唯一post-symmetry输入），保留生产上下文
重建、解码和后处理代码及完整C++FP32门槛。基线/分段B14复现原两例
**126/128**，真实候选B1/B14均 **128/128**，全部11字段零违例。
B1的93,696个数值逐位匹配存档WorkerW1 protobuf；共276次真实算法
调用、1,726,504项原始输出核验，FP16输入/权重与FP32累加边界不变。
这仍是**单lane固定批次的完整模型数值门**，没有宣称实际Worker并发
验收；诊断还重复原投影，不能测速。下一步在隔离后端中替换原调用，
再跑真实Worker128/W1/W32及Drain，之后才进行ABBA。生产42项及Fork
比例保持不变。收据：`target/fork-parity-20260908/g3-two-gemm-golden128-r3/closure.json`。

**2026-09-14 实际后端与Worker门通过；单独性能候选关闭。**
`g3-two-gemm-backend-r2` 在隔离源码中直接替换首次down和首次out_proj，
模型/权重/层位置/形状/FP32精度/stream/CheckForStream及独立host artifact
均有约束，无重算和额外拷贝。完整/分段/旧输出控制的1,726,504项五头
数值逐位复现前轮。实际两流Worker128例：W1/W32均 **128/128、全部
11字段PASS、Drain PASS**；基线W32仍126/128。候选W32业务128rows/
29batches，并记录连接后B2/3/4/5/8/11/13实际路径；预热标记不冒充业务。
唯一B14/S2/empty完整前向ABBA稳定，但event **-0.297832%**、wall
**-0.103291%**，未达+1%门槛；候选关闭，不重试、不跑Worker性能
ABBA、不迁移WSL计划。42项生产文件和691份旧依赖未变，Fork参考比例
未更新。下一步新假设是审计Windows已验收strict attention/immutable
QKV在WSL的可移植性；新组合需重新完整数值/Worker/ABBA验收，不能
把此修复的数值通过等同于性能达标。收据：
`target/fork-parity-20260908/g3-two-gemm-backend-r2/closure.json`。

**2026-09-14 strict/QKV 的 WSL 固定输入可移植性通过。**
隔离 Rust 封装加载同一份已验收 Windows CUBIN/参数模板，Linux ABI
经编译器证明并规范化两段对象填充后完整 400 字节相同。两流共
42,696,192 项输出 half 逐位相同；4 项 CPU 测试、只读/尾部保护及
6 次非法调用检查通过。Linux 原生 cos/sin 各有 801/834 项 1 ULP
差异，已完整导出；本次 GPU 使用冻结同输入，所以尚不代表 Linux
原生整网/Worker通过。下一阶段用实际 Linux 表验证 strict/QKV +
两处 GEMM 修复的新组合，再做独立 ABBA。无新性能结论，42+691项
保护文件不变。收据：
`target/fork-parity-20260908/qkv-strict-wsl-portability-r3/closure.json`。

**2026-09-14 WSL原生组合数值/前向通过，Worker稳定性未过门。**
`wsl-strict-qkv-two-gemm-r1` 在实际Linux原生表上完成完整128例直接与
真实Worker W1/W32：候选均128/128、11字段PASS、Drain PASS；基线
W32仍126/128。完整B14/S2前向ABBA event **+4.345099%**、wall
**+2.382690%**稳定通过。随后唯一C64 Worker ABBA的基线spread
**6.362287%**超5%门，名义+3.888471%仅诊断，不验收；完整
B14-only组合关闭，不重试/迁移。42+691保护项未变，Fork比例不更新。
下一新假设为显式B12/B13支持，先证明ABI/shape与完整输出再测性能，
不能用平均batch推算实际分布。收据：
`target/fork-parity-20260908/wsl-strict-qkv-two-gemm-r1/closure.json`。

**2026-09-14 B12/B13明确算子支持通过，整网接入待验收。**
`qkv-strict-b12-b13-scope-r1` 的编译参数仅M/grid-M不同，线程映射与
B14模板控制完全相同；实际PTX批次仅参与独立寻址。旧B14方法体未改，
14个不同输入batch、两流混合顺序重复的 **102,027,264项half全部逐位
通过**，12,288项尾部保护和90次拒绝检查通过。尚无完整模型/Worker/
性能结论，既有模型派发与42+691保护项未变。新默认关闭的完整接入位于
`wsl-batch12-13-integration-r1`，先过实际Linux native门再测新B13/Worker
ABBA；原B14-only失败实验保持关闭。收据：
`target/fork-parity-20260908/qkv-strict-b12-b13-scope-r1/closure.json`。

**2026-09-14 B12/B13整网及Worker数值通过，B13性能稳定性未过门。**
`wsl-batch12-13-integration-r1` 新增默认关闭的形状开关及独立指纹；
实际Linux表下B1/B12/B13/B14候选均128/128、11字段PASS，完整头
2,707,240项；实际Worker W1/W32/W64候选均128/128/Drain PASS，W32/W64
业务中确认新B13。40项plan测试与实际1正例/14负例通过。唯一新B13/S2
前向ABBA event名义+16.920988%，但两臂spread
**10.669825%/10.479489% > 5%**；wall +3.630155%
稳定不能替代event门。候选关闭，未运行Worker性能、未迁移或重试。
8000原始event CPU复核通过；现有数据缺少共同时轴，波动原因尚未定位。
下一步仅独立baseline时间线诊断，不重新裁决已关闭候选。42+691保护项
未变，Fork比例不更新。收据：
`target/fork-parity-20260908/wsl-batch12-13-integration-r1/closure.json`。

**2026-09-14 baseline共同时轴诊断完成，尚无具体API归因。**
`wsl-baseline-timeline-r1` 链接原冻结CUDA库，只有baseline B13/S2一次
诊断，2000条共同GPU原点/CPU提交记录及132,808项前后输出逐位门通过。
完整前向区间交叠95.317%（不是kernel并发率）；流2最后87批独流尾段
中位12.757216ms，912批重叠段23.652048ms。host apply合计约22.94/24.01秒，
接近各流运行窗口，需进一步定位调用内等待，不能全算作纯CPU开销。
WSL限定位置未发现Nsight/CUPTI；下一步隔离主机调用分段诊断，先做完备
计账/输出控制，不预判描述符缓存收益。旧候选保持关闭，生产与Fork比值
不变。收据：`target/fork-parity-20260908/wsl-baseline-timeline-r1/closure.json`。

**2026-09-15 主机API归因完成，下一项为延后未使用的fallback函数查找。**
`wsl-host-call-attribution-r2` 的一次baseline B13/S2诊断完整验证
**8,342,000条计时记录/2000前向/每次179层**，132,808项前后输出与冻结
基线逐位一致。记录器r1失败保留无结论；r2用第80次既有预热实测4171条/
前向后分配容量，溢出与嵌套CPU门通过，20份GPU PTX不变。函数模块查找
占两线程累计前向 **56.979271%**，描述符创建/设置/销毁合计 **0.582402%**。
完整树和源码确认每次有91次GetFunc/910次模块查询发生在成功Lt之前，
所得fallback句柄未使用。下一独立默认关闭候选延后这两个FP32 helper
的查找，保留真实fallback与Lt计算；不重试旧Mutex函数缓存，观测时间份额
不能预测加速。正式状态及Fork比例不变；WSL数值资格仍须处理。收据：
`target/fork-parity-20260908/wsl-host-call-attribution-r2/closure.json`。

**2026-09-15 延后备用查找已过整网逐位门，Worker基础选择器待处理。**
`defer-fallback-r1`在隔离Windows源码实现默认关闭、独立artifact绑定的
延后查找；32组实际GPU分支控制/4组缺失符号、1,439,744项算子参考及
B1/B14/B16两组两流**612,960项整网输出逐位门**通过，38项plan测试和
20份GPU PTX一致性通过。第一组Worker基线在B14预热遭原Fixed身份门
拒绝：`data[4]=0x0001fe2803000001`，认证为`0x0001fe2800000001`；
尚未连接Server、零完成请求，没有ABBA或加速结论。失败保留不重试。
18次正确大小的CPU属性读取确认九项公开配置相同，但不放宽内部身份门。
下一步把已通过独立整网/512请求的原认证描述符恢复机制作为两组共同
基础，重新验证当前QKV/门控组合，然后只对延后查找登记新ABBA；旧模型
复制、AlgoInit及函数缓存的性能裁决保持关闭。生产/旧依赖与Fork比例
未变，目标active。收据：`target/fork-parity-20260908/defer-fallback-r1/closure.json`。


**2026-09-15 延后备用查找：数值全过，完整前向稳定未达标，关闭。**
`defer-fallback-replay-r1/source-r3`把已验证的原认证64字节残差描述符恢复
作为两组共同基础，host revision2；只改变`DEFER_FALLBACK_R1`。39项plan、
4项恢复选择器测试、32组实际分支/4组缺失符号、1,439,744项算子输出，
以及B1/B14/B16两组两流612,960项整网五头逐位检查通过，20份GPU PTX不变。
四组Worker共512个真实请求全部通过11字段门及128/128 top-1；原始protobuf
重新解析核验。13项延后查找与7项共同恢复机制CLI绑定检查通过。
两次准备脚本锚点失败及一处CPU错误消息预期修正均保留，无已完成GPU重复。

唯一Windows B14/S2空盘NOGRAPH完整前向ABBA（80预热、每流1000次，8000项
原始event）得到event吞吐1281.599943→1276.602100，**−0.389969%**；
主机墙钟吞吐1270.313093→1273.638776，**+0.261800%**。所有组内spread
均低于1%，两指标都未达到1%采纳门。**关闭该候选，不运行Worker性能ABBA、
不重试、不安装**。WSL累计查找耗时份额没有转化为这项Windows整网收益。
原描述符恢复机制此轮通过数值资格，不等同独立性能认证。42项生产保护、
691项旧依赖及586项观测依赖保持一致，历史Fork比例不更新，目标继续active。
本机CUTLASS明确限制DualGemm stages>=3，stage2不作为直接参数候选；下一项
独立检查同tile/stage3的8-warp分工及寄存器占用。Fork当前B14关闭fused FFN，
不能把其历史stage2名称当作当前选择。收据：
`target/fork-parity-20260908/defer-fallback-replay-r1/closure.json`。


**2026-09-15 DualFFN 8-warp 候选稳定回退，关闭。**
`dualffn-warp8-r1`保持CTA128×64×32、stage3、swizzle2、vector8和原FP32
累加/精确SiLU/RN-half边界，只把warp64×32分工改为warp32×32（4→8个warp）。
本机CUTLASS要求stages>=3，未绕过该限制；旧warp16/K64、tile、vector、
缓存等性能实验没有重开。Fork当前B14实际关闭fused FFN，本实验是Rust
现有热点的独立配置，不冒称Fork现行tactic。

使用当前已认证CLI实际整网第一FFN的输入/输出，重新核验原始模型权重和
捕获身份。两条同时存在的stream、六份完整输出共**34,933,248项half**
逐位一致，前后guard和只读输入未变；编译代码为FP32 MMA、没有spill。
每线程寄存器236→150、共享内存仍48KiB，可驻留CTA2→1，warp总数仍8；
这是资源查询结果，不能单凭它断定性能原因。
唯一双流80预热/1000次算子ABBA复核8000项原始event及16份前后完整输出，
event吞吐**−32.509518%**，主机墙钟吞吐**−32.556256%**，所有组内spread
低于0.31%。因此关闭，不做整网/Worker集成、不重试或安装。

另以CPU核对原生输出头兼容补零：所有头仅占既有剖析内核累计时间2.4195%；
涉及pass、末层policy、concat和value FC的整个内核合计1.1833%，其中大量
隐藏层计算仍需保留。删除零通道的MAC比例不是墙钟加速预测，暂不优先实施。
下一步在现有证据上审计DualFFN的共享内存、同步及寄存器生命周期，再提出
独立内核设计；不继续凭参数数量试探。42项生产保护、691项旧依赖和586项
观测依赖复核一致；历史Fork比例不更新，目标active。收据：
`target/fork-parity-20260908/dualffn-warp8-r1/closure.json`。


**2026-09-15 SiLU移到共享内存重排前：算子有收益，整网未过1%门，关闭。**
`dualffn-register-silu-r2`保持原FP32 DualMma、tile128×64×32、warp64×32、
stage3、vector8、swizzle2与精确SiLU。两份累加片段使用相同元素布局，
候选在共同重排前调用原projection0/1和SiLU functor，保留各自RN-half边界，
只把最终half精确扩展成FP32后送入一份原共享内存重排，再原样写回half。
没有K分区归约或原投影输出。r1将片段coarsening2误写成1的本地编译断言
失败已保留；r2仅纠正该断言，r1没有GPU执行，没有绕过vendor stage限制。

实际PTX/SASS证实epilogue共享写入指令**64→32**、读取**32→16**；
FP32 HMMA64、异步加载24、LDSM24和BAR19个静态点均不变。寄存器仍236、
共享内存仍48KiB、潜在2CTA/8warp每SM、无spill。两流六份真实首FFN输出
共**34,933,248项half逐位一致**。唯一算子ABBA的event **+3.214647%**、
主机墙钟 **+3.304415%**，所有spread低于0.30%，故进入完整模型验证。

`dualffn-register-silu-integration-r1`使用两个新隔离构建，两臂源码只差
`kRegisterSiluB14=false/true`，B1/B16均走原内核；两臂共同REPLAY1、DEFER0。
新头文件纳入core指纹，实际FFI成功后发出epilogue分支标记。200个Rust
文件和20份通用GPU PTX不变；各39项plan+4项恢复选择器测试通过。
B1/B14/B16两臂两流整网五头**612,960项FP32输出逐位一致**；四组Worker
共512个真实请求通过11字段门和128/128 top-1，全部原始protobuf复核。
8项实际CLI负例确认跨臂计划、旧host、错误模型和环境扩展均拒绝。

唯一Windows B14/S2空盘完整前向ABBA（80预热/每流1000次，8000项event）
得到event **1277.109855→1283.653350，+0.512367%**，主机墙钟
**1273.775226→1280.904103，+0.559665%**。所有spread≤1.095%，结果稳定，
但两项都未达1%门：**关闭，不进入Worker性能测试，不重试、组合或安装**。
指令减少和算子收益不能代替整网采纳证据，历史Fork比例保持原值。

下一项先核查原生产主循环的跨投影MMA指令交错：当前SASS64个HMMA点，
A操作数无reuse标记，B有48个；这只是静态线索，不是寄存器带宽瓶颈证明。
候选须保留每个累加器的K顺序及原epilogue，不复用本次关闭的SiLU改动。
42项生产保护、691项旧依赖、586项观测依赖及上一阶段新增依赖复核一致，
目标继续active。收据：
`target/fork-parity-20260908/dualffn-register-silu-integration-r1/closure.json`。


**2026-09-15 两项独立候选：MMA交错无算子收益；QKV系数向量读未过整网门，关闭。**
`dualffn-paired-mma-r1`保持原CTA/warp、stage3、布局、加载/同步、精确
SiLU和RN-half边界，仅将同m/n的gate/up FP32 MMA相邻发射；CPU逐累加器
检查K顺序不变。实际SASS相邻同A指令对14→38，A reuse仍0，B reuse48→0，
64个FP32 HMMA和数据移动指令数不变。236寄存器/48KiB/潜在2CTA与8warp
不变、无spill。六份完整输出34,933,248项half逐位一致；唯一8000事件
算子ABBA event **−0.053081%**、host **−0.308073%**，最大spread0.209%。
候选稳定无收益，未进入整网/Worker，不重试或组合；收据位于同目录closure.json。

`qkv-rope-coeff-vector4-r1`是另一个独立改动：原已安装immutable QKV/RoPE
迭代器每次持有8项half，对应4个连续FP32 cos/sin。只将各自4条标量系数
加载替换为一条16字节向量加载；表的字节和大小、FP32 MMA、RoPE指令顺序
及前后两次RN-half全保留。基线完整SASS与已安装CUBIN一致；两边编译导出
ThreadMap一致，485,184个Q/K向量均证明对齐、不越界、地址逐项对应。
实际SASS **128条标量load→32条vector4 load**，这是静态指令数减少，
不是传输字节减少。146寄存器/36KiB/潜在2CTA与8warp不变，无spill。

用当前已安装CLI重新捕获完整模型首Attention输入和旋转后的packed Q/K/V，
原FP32系数由绑定CPU导出器重导，与旧认证字节完全一致。六份完整输出
**34,933,248项half逐位通过**。唯一双流80预热/1000次算子ABBA复算8000
event及16份前后输出：event **+2.107045%**，host **+1.863065%**，
全部spread≤0.852%，通过算子门。复核器仅在保存裁决后的最终JSON打印缺少
import；保留原脚本和裁决，补做CPU重算得到完全相同JSON，没有重跑GPU。

`qkv-rope-coeff-vector4-integration-r1`两份隔离源码只差
`COEFFICIENT_VECTOR4_B14=false/true`；共同REPLAY1/DEFER0，原DualFFN保持。
候选真实C++ Params导出与生产400字节模板及6指针偏移完全一致；新CUBIN
原字节嵌入候选CLI/native/lib-test，加载器额外核验cos/sin 16字节对齐。
200个Rust文件、20份通用GPU PTX不变，每臂39plan+4replay+2asset测试通过。
首次asset筛选漏cuda模块导致0项测试，原记录保留；修正仅CPU筛选与新输出
目录，之后两项真实测试通过，未改变候选或重跑GPU。
B1/B14/B16两臂两流全部头 **612,960项FP32逐位一致**；四组Worker共
**512个请求/结果**通过11字段门、全部top-1与Drain，原始protobuf重建复核。
10项实际CLI负例确认跨臂、QKV资产、旧host、错模型和环境扩展被拒绝。

唯一Windows B14/S2空盘完整前向ABBA（80预热/每流1000次）复算8000事件：
event **1277.053089→1286.605698（+0.748020%）**，
wall **1268.206984→1274.928757（+0.530022%）**。
测量稳定，但未同时达到两项1%收益门，**关闭，不测Worker性能、不重试、组合或安装**。
Fork当前B14仍选择wide QKV，producer QKV/RoPE AOT为disabled；本项是Rust
已认证融合内核的独立实现优化，不冒称Fork现行tactic。历史WSL Fork比例不更新。

下一步只做新假设的源码/编译审计：QKV原alpha1/beta0且无split-K，考虑
OnlyAlphaScaling静态消除本来不执行的source-addend分支，保留FP32乘法。
不能以静态死代码减少推算动态访存或速度；从原scalar系数生产版本开始，
不组合本轮已关闭的向量加载。42项生产保护、691旧依赖、586观测依赖及
近期各阶段依赖复核一致，整体目标active。收据：
`target/fork-parity-20260908/qkv-rope-coeff-vector4-integration-r1/closure.json`。


**2026-09-15 QKV零beta编译期分支：静态代码缩短，算子无收益，关闭。**
`qkv-only-alpha-r2`从已安装的原scalar RoPE/Default scaling内核出发，
仅将QKV epilogue policy改为OnlyAlphaScaling。固定alpha1/beta0且无
split-K，原路径已经选择不读取source；候选在编译期做同一选择，保留
FP32 alpha乘法、原标量系数读取、精确FP32 RoPE及两次RN-half，不使用
ScaleType::Nothing，不组合已关闭的vector4或MMA交错。r1本地重命名误将
vendor kernel::Gemm写成kernel::AlphaGemm导致编译失败；r2只纠正这个
限定名，保留原记录，r1没有GPU运行。

基线完整SASS与生产CUBIN一致，候选实际静态指令**2706→1522**；FP32
HMMA32、主循环异步加载/LDSM保持，源分支中不执行的重复epilogue被移除。
这不代表实际执行的RoPE/访存减半。寄存器仍146、共享仍36KiB、潜在
2CTA/8warp、无spill。完整原标量RoPE源码除类型名外相同；实际新C++
Params导出400字节模板、全部地址槽与生产完全一致，ThreadMap与边界门通过。
六份真实首Attention输出**34,933,248项half逐位一致**；唯一双流80预热/
1000次ABBA复算8000事件和16份前后完整输出：event **−0.230422%**，
host **−0.174885%**，最大spread1.154%，稳定无收益，关闭且不进入整网/
Worker、不重试或安装。42生产保护、691旧依赖、586观测依赖和近期各阶段
依赖复核一致；历史Fork比例不变，整体目标active。

下一项源码假设为QKV M64/N64/K32、warp32×32、stage3，保持4warp，预计
共享由36降为24KiB，实际可驻留数量须查询。5054行需要79个M64 tile，
无效尾行2（M128为66），但B加载重复率79/40，更小tile未必更快。先做
真实编译、资源与完整ThreadMap/Params验证，不能从占用率预测性能。
此项与旧CuTe M128N128、Attention M64、DualFFN M64分开，不重开旧裁决。
收据：`target/fork-parity-20260908/qkv-only-alpha-r2/closure.json`。


**2026-09-15 QKV M64：潜在驻留翻倍，稳定退步，关闭；Fork outproj 实物审计完成。**
`qkv-m64-occupancy-r1`保留生产 scalar RoPE、Default scaling、FP32累加与
两次RN-half，仅改 CTA M128→64、warp M64→32，仍128线程/三阶段。
实际寄存器146→80、共享36→24KiB、潜在CTA2→4（warp8→16），无spill；
5054行尾部无效行66→2，但更小tile会增加权重加载重复。完整生产SASS复现、
新400字节Params和全矩阵ThreadMap通过，34,933,248项half逐位一致。
唯一双流ABBA的8000事件与16份前后输出复核：event **−2.153284%**、
host **−3.129363%**，最大spread0.324%，稳定退步，关闭且不进入整网/
Worker、不重试、不安装。占用率是资源事实，不能代替吞吐实测。

进一步核查 Fork 真正选中的 `outproj-m128-n128-k32-s3-t128-mb3-tilelang-49k`：
认证二进制与计划hash保持；保留对象与二进制中选中函数的完整SASS及PTX
完全一致，确认 **HMMA.16816.F16**、half2残差加法、FP16 C/D。
其NN权重配三阶段cp.async、共享内存异或布局、swizzle10及寄存器直接写回；
并非TMA/persistent。实际162寄存器、49,152动态共享、cuobjdump SHARED字段1,024、无local，
`min_blocks=3`并非实测3CTA，共享资源上限为2CTA。
旧父级汇总manifest hash已变化，已留痕且未作为历史身份依据；选中源码、
元数据、计划及二进制独立核实。反汇编初次解析误包含后续模块警告，只修正
CPU函数边界解析，未重跑编译/GPU。

下一项是保留Rust FP32累加/残差的独立 TileLang 生成数据组织与寄存器
直接写回候选：M128N128K32、三阶段、128线程，launch bounds设2以适配
FP32寄存器和实际共享上限。先编译与完整地址映射门，不把Fork FP16性能
视为可直接获得的收益，不重开已关闭的Lt NN/CUTLASS候选。
42生产保护和全部旧/近期依赖保持，Fork历史比例未更新，整体目标active。
收据：`target/fork-parity-20260908/qkv-m64-occupancy-r1/closure.json`；
源码/产物审计：`target/fork-parity-20260908/fork-selected-outproj-audit-r1/assessment.json`。


**2026-09-15 Fork输出投影组织的FP32改写：完整算子逐位一致，ABBA基线不稳，关闭。**
`outproj-tilelang-direct-fp32-r2`复用已核实的Fork生成索引、三阶段cp.async、
共享异或布局、swizzle10和寄存器直接写回，保持FP16输入/权重，改用Rust既有
FP32累加及FP32残差C/D，无新增half边界；M128N128K32、128线程、bounds2。
源主循环除累加类型/指针外相同，完整CPU地址映射覆盖1,940,736项且恰好一次，
实际HMMA全部F32，238寄存器、无spill。32位残差读写已验证。

当前TileLang头文件中的四个未使用FP8→FP16 helper与CUTLASS3.9.2不兼容，
r1编译失败保留；隔离副本只删这四个无关声明，保留FP32 dispatcher与所有
通用断言，candidate.cu字节不变。第一次运行资源守卫在任何GEMM发射前停止：
cuobjdump的SHARED字段1024不能直接当作Driver的静态分配；独立只读查询为
**静态0、动态49152、238寄存器、local0、潜在2CTA**。只修正主机守卫，
CUBIN与数值/性能门不变，所有前序失败及CPU定位记录保留。

真实模型捕获的三份outproj操作数与已有绑定存档逐字节核实；基线复现当前
cuBLASLt TN/FP32/beta1 heuristic0，B14 K384不走FFN-down的Fixed索引。
两流六份完整结果共 **11,644,416项FP32逐位一致**；全量FP64参考最大绝对
误差1.0441e-6，边界、只读、重复和两流一致性通过。
唯一W80/N1000双流ABBA的8000事件与16份前后输出复算完成，但基线event
spread **16.0839%**、host **7.5342%**，均超过5%门。候选spread分别
0.3327%/3.0169%；整组仍判不稳，名义event−13.687%/host+6.967%仅留诊断，
**不宣称可靠收益或退步**，关闭、不补跑，不进入整网/Worker或安装。
旧Fork生成源码审计中的SHARED1024应读作cuobjdump报告字段，未经加载查询
不能据此断言静态分配；这一限定不改变其FP16指令/残差证据。

下一项只做DualFFN参数包装的CPU代码生成核查：生产仍经legacy
`cutlass::Kernel(Params params)`，DualGemm主体已经接收const Params。
保持原主体/epilogue/形状，比较const grid-constant包装的实际SASS和ABI；
若没有生成代码差异就直接收口，不将QKV已采纳的只读参数收益外推到DualFFN。
Fork当前关闭DualFFN，本项是同一对标目标下的Rust实现改进假设。
42生产保护及全部旧/近期依赖一致，Fork历史比例不变，目标active。
收据：`target/fork-parity-20260908/outproj-tilelang-direct-fp32-r2/closure.json`。


**2026-09-15 DualFFN只读参数包装：生成指令完全相同，CPU阶段关闭。**
`dualffn-immutable-wrapper-r1`保持生产DualGemm/精确SiLU及全部half边界，
只把legacy `Kernel(Params params)`包装换成const grid-constant参数。
基线完整2969条SASS与当前安装的SM120函数完全一致，候选同样**逐条/编码
一致**：236寄存器、无local/spill、FP32 HMMA64处。两者使用同一Kernel::Params，
实际CPU导出720字节、8字节对齐，input/gate/up/output槽为64/112/360/640；
两组地址样本除四槽外完全一致，清零后完整模板一致。无生成代码变化，
按预登记规则关闭，**未运行GPU数值或性能测试**，不宣称测得零收益。
首次生产函数查找因NVCC把源文件名/hash编码进匿名命名空间而未命中；修正
唯一符号匹配后只允许该名字片段变化，完整SASS正文仍严格相等。另修正
CPU解析器兼容本例内部链接的plain .entry，两处原记录保留，无重编译。

下一项独立主机假设：同一个49,152字节共享属性在DualGemm::run每次调用时
重复设置，考虑移到现有每tokens State的initialize。当前每完整前向33个
DualFFN；静态调用次数不能证明底层重复配置或可得收益，旧Nsight表没有
cudaFuncSetAttribute同名记录，也不能据此断言开销为零。先验证原Kernel
SASS/Params及完整输出，再按单次ABBA裁决，不组合本次关闭的参数包装。
生产42项保护及全部旧/近期依赖不变，整体目标active。
收据：`target/fork-parity-20260908/dualffn-immutable-wrapper-r1/closure.json`。


**2026-09-15 DualFFN共享属性只在初始化设置：稳定但未达收益门槛，关闭。**
`dualffn-attribute-init-r1`仅将原49,152字节cudaFuncSetAttribute调用从
每次run移到成功initialize；原Kernel类型、Params、更新与发射逻辑保持。
两个主机类实际共用同一个设备函数，完整2969条SASS与安装版本相同，
FP32 HMMA64处、236寄存器、无local；实载仍为2 CTA/8 warp。
候选先于基线运行，六份完整输出共34,933,248个half逐位相同，双流、
输入只读与守卫检查通过。唯一B14/S2/W80/N1000算子ABBA共8000事件：
event **+0.004121%**，主机墙钟 **−0.139624%**，最大arm波动0.343%，
稳定但均未达到1%门，关闭，不进行整网/Worker性能测试，不重试、不安装。
该算子循环只初始化一次Params，不含共同的生产外层FFI/update开销，
因此数字不作完整模型或Worker结论。首次构建缺少相对头文件路径，
仅补独立include目录后完成；失败记录保留，修正前没有GPU运行。

下一项先作CPU可行性检查：完整N384输出投影与其后的FFN RMSNorm融合。
当前生产RMS采用每warp一行、12项FP32平方和与XOR归约。新候选必须
保留该顺序及FP32残差写回，只尝试省去独立RMS读回与发射；先确认
非二次幂tile映射与共享内存容量，不复用已关闭TileLang或split-K/RMS候选。
42项生产保护及历史/近期依赖已核对不变，整体目标active。
收据：`target/fork-parity-20260908/dualffn-attribute-init-r1/closure.json`。


**2026-09-15 完整N384输出投影与RMSNorm融合：数值逐位通过，性能关闭。**
`outproj-rms-whole-n384-r1`独立使用CUTLASS FP32 MmaMultistage，
CTA64×384×32、warp32×96×32、256线程、stage3、TN；整个384通道
由一个CTA负责，mainloop的84KiB共享复用为96KiB完整FP32 tile。
保留FP32残差写回、原每warp一行RMS的12项FMA与XOR16/8/4/2/1顺序，
div.rn、rsqrtf、两次FP32乘法及RN half不变。输入与normed允许原位复用，
完整行归属和cp.async drain后写回边界已检查，不组合旧TileLang或延迟add方案。

编译实载196寄存器、0 static/local/spill、98,304字节dynamic，潜在1CTA/
8warp；完整24,576格tile及1,940,736有效/768尾部元素映射通过。实际
双流、原位/异位、重复输出共**23,288,832 FP32＋23,288,832 half逐位相同**，
对应当前完整模型的第一Attention残差及第一FFN归一化输入，全部守卫通过。
最初CUTLASS读迭代器const API不匹配，仅补两个构造转换；另修正CPU map
printf类型，完整设备SASS在该CPU修正前后相同。失败/警告与修正记录均保留。

唯一B14/S2/W80/N1000成对算子ABBA保存8000事件、32份完整前后输出：
event **−35.872915%**，A/B spread **2.674864%/0.416534%**，稳定负收益。
主机墙钟名义 **−12.259032%**，但基线spread **8.007171%** 超过5%，
不作为可靠墙钟效果。整体关闭，不重测、不调整tile挽救，不接入整网/Worker。
event包含输出投影＋RMS；两边共同的输入/残差恢复在event外、主机墙钟内。
这些算子结果不改写Fork/Rust整网比例。

后续先核对Fork所选FFN第一投影/SwiGLU的实际dispatch及设备累加指令。
保留的B14计划关闭fusedFFN、wide-single-GEMM和专用SwiGLU，当前本地源码
对应两次MatMulLayer；最终projection-Lt选项及源码与WSL已保留二进制的
绑定仍需完成。API叫Hgemm不直接证明实际HMMA累加精度，也不重试已关闭的
DUALFFN-off或两GEMM候选。42项生产保护及全部既有依赖不变，目标active。
收据：`target/fork-parity-20260908/outproj-rms-whole-n384-r1/closure.json`。


**2026-09-15 Fork 所选 FFN 投影：实际加载机器码确认 FP16 累加。**
`fork-selected-ffn-audit-r1` 完成原始 WSL Fork 二进制、同 SHA TF3 模型、
B14/S2 计划、base config 与原 RUNPATH cuBLAS 13 库的绑定。B14 关闭
fusedFFN、wide-single-GEMM、专用 SwiGLU；projection-Lt 默认关闭且配置
没有覆盖。已链接 MatMulLayer 的 half 分支调用 Hgemm，W1/N1 诊断实际
捕获两流共四次 M1152/N5054/K384、NN、ld1152/384/1152 的成功调用。
每流两个调用共用输入，权重及输出不同，符合两个独立 FFN 投影。

实际函数名为 `cutlass_80_tensorop_h16816gemm_256x128_32x3_nn_align2`
的 CUTLASS Kernel2 实例；名字含80，驱动实际报告 binary/PTX version **120**，
grid160×2、256线程、**154寄存器、72KiB dynamic、0 static/local**。
从驱动加载的 ELF 中提取该符号完整34,688字节 text，反汇编2168条指令，
**64处 HMMA.16816.F16、没有 FP32 HMMA**。因此 Fork 所选 FFN 两个
第一投影的 FP16 累加已由设备机器码确认；本仓库 DualFFN 仍为 FP32。
不能把 Fork 的速度优势全部归于布局/发射，也没有量化精度差异的贡献。

观测器 getter ABI、WSL 代理递归和 RTLD_LOCAL loader 指针问题均保留失败
与修正。ELF 采集漏了尾部五个程序头，全文件 cuobjdump 失败；最终仅使用
原始捕获中边界完整的 section/symbol/text 与 SM120 raw 指令解码，没有
构造 ELF 头，也没有重跑模型补文件。原库未找到完整捕获前缀，不推断其
私有封装或 JIT 来源。函数名与加载镜像符号一致，未捕获 library→module
句柄直接映射。SwiGLU 另有本地/WSL 源码与已链接 host 函数证据：
`swiGLUHalfStrideKernel<4>`、256线程、每线程四个half2；其独立运行时
发射没有纳入这次 Hgemm scope。所有观测运行时间排除出性能比较。
证据：`target/fork-parity-20260908/fork-selected-ffn-audit-r1/assessment.json`。

**2026-09-15 DualFFN 固定 K384 与普通 GEMM 模式特化：数值通过，性能关闭。**
`dualffn-fixed-k384-r1` 保留原128×64×32/stage3/TN/四warp、迭代器、
FP32 MMA 顺序、两次 RN half 投影边界和精确 SwiGLU，仅将已知 K384、
offsetK0 与普通单 slice 模式交给编译器；主机拒绝其他形状/模式。
原核完整 SASS 与已安装 CLI 逐条一致，候选完整指令 **2992→2864**、
寄存器 **236→230**，两者均64处 FP32 HMMA。旧2969计数漏掉 uniform
predicate 指令；本次改用完整指令流比较，计数审查失败及修正均保留。

实际资源两者均0 local、49,152字节dynamic、潜在2CTA/8warp；较少寄存器
没有增加潜在占用。当前第一FFN输入，候选先运行、两次重复和基线，双流
共 **34,933,248 half逐位相同**，守卫及只读输入通过。唯一 B14/S2/W80/
N1000 算子 ABBA 保存8000事件和16份完整前后输出：event **0.000000%**，
A/B spread均 **0.016346%**；统一墙钟 **−0.157041%**，spread分别
**0.142138%/0.438652%**。两项稳定，未达1%，关闭，不接入整网/Worker，
不调整或组合挽救。本轮生产源码、CLI、计划的42项保护未变，正式 Fork/Rust
比例不更新；目标继续 active。后续先审查当前 DualFFN 异步拷贝/等待，找到
具体可删减指令或依赖后再注册独立候选，不据静态指令数推断速度。
收据：`target/fork-parity-20260908/dualffn-fixed-k384-r1/closure.json`。


**2026-09-15 DualFFN epilogue 双缓冲：数值与竞争检查通过，稳定小收益未达门槛。**
`dualffn-epilogue-pingpong-r1` 从原始生产128×64×32/stage3/TN核独立派生，
保留原异步拷贝、等待和主循环，只将 epilogue 的两份 FP32 共享缓冲改为
两组交替使用；每轮写后屏障保留，删去七次后续写前屏障。CPU 对128线程、
八轮完成229,376项先后关系检查；迭代器以float元素为单位移动4608，
相邻轮不重叠，同一组复用前经下一轮屏障保证上一轮读取完成。

共享 epilogue **18,432→36,864字节**，仍落在原49,152字节主循环 union 内。
完整 SASS **2992→2984条**、BAR.SYNC **19→12**；两侧均64处 FP32 HMMA，
原异步提交/等待数量与 PTX 算术 opcode/literal 多重集相同。两侧实际均
236寄存器、0 local、49,152字节dynamic、潜在2CTA/8warp。当前首FFN
完整输入上，候选先运行、两次重复及原核，双流 **34,933,248 half逐位相同**；
守卫和只读输入通过。Compute Sanitizer racecheck 完成且报告 **0 hazard、
0 error、0 warning**，该诊断中的六份完整输出也一致。

首次 racecheck 默认通信 attach 超时，按已核验进程身份结束目标，原失败保留。
零kernel初始化诊断证明命名管道通信可用；随后多余报告选项被本机2026.2.1
拒绝，目标未启动。仅去掉该选项后，同一候选二进制完成上述 racecheck。
命名管道使用 NVIDIA 文档所列的进程内环境配置，未改驱动、注册表或防火墙；
所有插桩时间均排除出性能。工具依据：
[Compute Sanitizer 环境变量](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#environment-variables)。

唯一 B14/S2/W80/N1000 算子 ABBA 保留8000事件与16份完整前后输出：
event **+0.387824%**，A/B spread **0.585952%/0.041311%**；统一墙钟
**+0.598734%**，spread **0.365339%/0.344768%**。两项稳定，但均未达1%，
故关闭，不重试、不与旧特化组合，不接入整网/Worker。生产42项保护未变，
正式 Fork/Rust 比例不更新。下一步仅作完整FFN融合的CPU资源/依赖可行性
审查，尚未注册新性能候选。收据：
`target/fork-parity-20260908/dualffn-epilogue-pingpong-r1/closure.json`。


**2026-09-15 完整 FFN 分段融合：全部数值/竞争门通过，稳定显著回退，关闭。**
`whole-ffn-resource-audit-r1` 先绑定当前首FFN生产者/消费者夹具：上投影
SwiGLU 的完整half输出 SHA 与下投影完整输入 SHA 相同。本机零kernel
资源查询确认 SM120 每SM 65,536个寄存器、100KiB shared，每CTA可选
99KiB。CUTLASS例13要求首投影完整N及次投影完整N落在CTA内，且原例
只有一个首投影；它不能直接代替两投影加精确SwiGLU。将现有三阶段
Dual样式放大到完整1152通道，即使 M32 也需 **438KiB shared**，不适用。
这只排除该直接套用方式，不声称所有完整融合都不可能。Fork当前B14
没有选择其可选fusedFFN；该接口仅输出SwiGLU中间张量，也未传入down
权重，不能当作 Fork 已采用完整三GEMM融合的证据。

据此独立注册 `whole-ffn-stream32-r1`：每CTA32行/128线程，隐层每64
通道一段，共18段；原序FP32 gate/up累加后各RN half，精确FP32 SiLU，
乘积RN half留在shared，立即按递增K16累加完整384列down，最后只加一次
FP32残差。没有全局中间张量或独立down发射，也未组合任何已关闭候选。
CPU逐项检查存储双射、16字节复制、PTX lane片段、half2 bank、全输出
唯一写入、最后30有效行及完整K顺序。up双缓冲20KiB、down与中间量共
28KiB；编译实测 **254寄存器、0 spill/local、28KiB dynamic、潜在2CTA**，
40处静态FP32 HMMA。算术寄存器的原128个下界不含地址/操作数/exp代码，
最终资源必须以编译和实际函数属性为准。

测试通过Driver加载已经审查的原CUBIN，不重编译候选；原Dual源码逐字
冻结，完整2992条SASS与已安装核相同，down沿用原Fixed算法身份、FP32
compute、共用Lt句柄及独立lane workspace。主机报告变量与Windows
`FIXED`类型冲突、历史SASS引用路径替换错误均在首个GPU kernel前修正；
原源码、编译失败和错误引用记录保留，候选CUBIN未变。

候选先运行、两次重复及原路径，双流 **11,644,416 FP32输出逐位一致**；
原路径另有 **11,644,416 half中间值** 完整一致。输入/权重/残差源只读与
边界守卫通过；候选未读写预置哨兵的全局中间缓冲。命名管道racecheck
完成并报告 **0 hazard/0 error/0 warning**，诊断全输出仍逐位相同，
所有插桩时间排除出性能。

唯一 B14/S2/W80/N1000 完整FFN算子 ABBA：event token-row 吞吐
**−41.742302%**，A/B spread **0.460280%/0.148361%**；统一墙钟
**−30.209494%**，spread **0.545640%/0.010434%**。独立重算全部8000
事件、16份FP32前后输出、8份基线half中间输出通过。两个原kernel都在
A事件内，完整融合核在B事件内；同一残差复位拷贝在事件外、墙钟内。
这些数字是孤立FFN的token行吞吐，不能与NN棋盘行/Worker RPC口径混算。
方案稳定退步，关闭，不扫参或与旧方案组合挽救，不跑整网/Worker性能。

每层省去23,288,832字节中间写读的同时，理论CTA级上投影权重请求从
70,778,880增至279,576,576字节；这不是实测DRAM/L2流量，也没有据此
精确归因回退比例。下一步先审查实际所选Fixed down核的完整依据，找到
独立可改的加载/依赖后再注册候选。生产42项保护未变，正式Fork/Rust
比例和已认证Worker结果均不更新，目标仍active。证据：
`target/fork-parity-20260908/whole-ffn-stream32-r1/closure.json`。


**2026-09-15 实际 Fixed down 内核审计及直接写回实验：数值通过、性能稳定回退。**
`fixed-down-selected-kernel-audit-r1` 先查重，保留 dense-grid/s6、N64/s3、
N32/s3、NN及完整FFN融合等已关闭裁决。现用cuBLASLt DLL中虽存在目标
符号字符串，但按SM120筛选的静态反汇编未返回该函数；这不能证明JIT
来源。随后注册一次受限CUPTI观测：原首FFN实际夹具、原Fixed算法、
两个存活流各一次matmul，所有诊断时间排除。回调只复制当前进程中有效
的数据，不在回调内调用CUDA API；函数属性在回调返回后查询。

成功保存实际加载的 **11,744,928字节完整ELF**（全部section/program
header范围检查通过），以及两次成功发射、360字节原参数、完整FP32
输出。两流 **3,881,472项输出逐位一致**。实际目标为
`cutlass_80_tensorop_s16816gemm_f16_64x64_32x6_tn_align8`：
632×1×1网格、128线程、88寄存器、49,152字节动态shared、0 local，
binary/PTX version均120。CUPTI moduleId与Driver CUmodule是不同类型，
未声称二者数值相等或已建立直接映射，也未断言私有库/JIT生成来源。

完整SASS共904条，16处HMMA均为FP32累加。参数及开头指令共同证明
`tile_m=CTAID.X>>3`、`tile_n=CTAID.X&7`：474块有效，158块在访存前
退出；旧去空块实验已得0%，不因此重开。mode0、split数1、alpha/beta
均1，C=D：通用split-K信号量分支实际被跳过，不能将整份二进制中的
同步都当成当前开销。当前source-needed epilogue有8次CTA barrier、
16处STS.64、8处LDS.128；其FMNMX参数为quiet NaN，未将其误认作开启
ReLU，也未修改厂商机器码。六阶段主循环的提交/等待关系保留。

由此独立注册 `ffn-down-direct-store-r1`：维持64×64×32、warp32×32、
stage6/TN、FP32累加/残差/输出与原632块padded网格，只用CUTLASS
direct-store epilogue按float2从寄存器读取残差并写回。OutputOp Count2
避免未使用片段元素，默认alpha/beta计算策略不变；编译期证明Mma类型
与普通float4 epilogue的同形状源核相同。这不是重试去空块、stage3或
已关闭的TileLang输出投影，也未组合其他关闭方案。

CPU穷举PTX片段索引与direct-store索引：128线程各32个FP32累加值，
全部4096局部输出和1,940,736有效全局输出唯一覆盖，尾块62行及float2
对齐通过。实际候选SASS **824条、80寄存器、0 spill/local、48KiB动态
shared、潜在2CTA**；保留16处FP32 HMMA、24处异步复制、12处LDSM和
3处主循环barrier，尾部无共享中转。二进制另有5处`@!PT LDS RZ,[RZ]`，
谓词恒假，不执行访存；它们仍计入总指令数。

诊断脚本在首个GPU运行前修正了候选PTX筛选范围、v2.b32指令拼写及
恒假谓词识别，原脚本/失败检查与修正记录全部保留，候选源码/二进制
未变。两流candidate-first、重复及原路径共 **11,644,416 FP32输出
逐位一致**，输入/权重/残差源只读及guard通过；racecheck命名管道运行
报告 **0 hazard/0 error/0 warning**，另6份完整输出仍逐位一致。

唯一B14/S2/W80/N1000算子ABBA：event吞吐 **−29.182089%**，A/B spread
**0%/0.441723%**；统一口径host吞吐 **−29.250703%**，spread
**0.274657%/0.185480%**。独立重算全部8000事件、16份计时前后输出，
并复核数值/竞争检查的12份完整数组。残差D2D复位在事件外、墙钟内，
各臂相同；这里是孤立FFN-down token行吞吐，不能当作NN棋盘行或Worker
RPC结果。更少寄存器/同步并未转化为收益；未用未取得的硬件counter
归因具体损失。**关闭此候选，不扫参挽救，不接入整网或Worker测速。**

生产42项保护未变，正式Fork/Rust比例及已认证Worker成绩不更新。
下一步先审计独立C384原位SiLU门的映射与精确计算，检查float4/固定通道
寻址是否有独立空间；后续half转换保持独立，不重开已关闭的gate+cast
融合。本目标仍active。收据：
`target/fork-parity-20260908/ffn-down-direct-store-r1/closure.json`。


**2026-09-15 C384 原位门四通道映射：数值全位通过，稳定性/墙钟门未过，关闭。**
`c384-gate-rowquad-r1` 从当前剖析与源码重新定位11个C384原位SiLU门：
原grid1896/block1024、16寄存器，占当前kernel耗时之和1.689302%。Fork
B14计划选择half2，一行192线程，输入/参数/输出及仿射为half；其精确
SiLU转为FP32计算。只借鉴逐行连续访问，保留Rust所有FP32仿射/激活/
原位输出边界，后面的独立half转换不动，不重开旧gate+cast双写融合。

新方案每行96线程，每线程float4读输入、scale、bias，并在原位置写回
float4；四个通道仍各自执行原FP32仿射和精确expf/除法。CPU穷举行数
1/361/5054/5776下全部输出唯一归属、行/通道边界及16字节对齐。用于
基线的完整191,435字节PTX（SHA b8d1cf48…）与正式CLI内嵌字节完全相同，
通过Driver原样加载；候选也通过相同零JIT选项接口加载，未以重新编译
的标量核冒充实际基线。CPU准备曾误读Fork tactics数据类型、尝试绑定
不匹配的普通release构建目录；检查均在注册/GPU前拒绝。改用已认证
core-preservation产物后验证完整PTX在正式exe中恰好出现一次，失败
脚本与修正记录保留，普通构建目录没有被改动。

候选从保护中的executor源文件仅追加一个入口；编译确认原31个入口的
参数与指令在局部label归一后全部不变。新入口有3次float4读取、1次
float4写入，没有动态channel remainder或half运算；4份完整expf范围
规约/修正和div.rn与标量原式一致。实际Driver函数属性：基线16寄存器、
候选24寄存器，均0 local、binary/PTX120；潜在驻留block分别1/16，
这是占用率API上限，不是实测占用率或加速证据。

正式CLI按原17项tactic、同模型SHA/构建/空盘输入，分别捕获首次C384门
前后的完整边界14/15；日志分别确认Ffn与GateSilu384，时间全部排除。
完整解析原TF3文件到EOF，从11个C384 post-BN中取首个参数，逐步FP32
计算合并scale/bias，原标量核全输出与捕获结果逐位相同。四种行数×
常规/边界输入，加真实B14夹具共9组；candidate-first、重复、基线及
两条同时存活的流，共 **63,217,152 FP32输出逐位一致**，只读/guard
通过。memcheck和racecheck各自再核6份真实全输出，均0错误，racecheck
0 hazard/0 warning；插桩时间不进入性能。

唯一真实B14/S2/W80/N1000算子ABBA：event名义吞吐 **+38.671194%**，
A/B spread **0.324224%/7.059692%**；统一host包络吞吐仅 **+0.290414%**，
spread **4.154013%/6.311590%**。候选两指标均超过5%稳定性门，host也
未达到1%收益门；名义event收益不具备采纳资格。独立重算8000个原始
事件、16份计时前后全输出及数值/插桩86,505,984项输出通过。原位输入
每次共同D2D复位在event外、host内；这里是孤立门的token行吞吐，不能
报告为NN/Worker加速。**关闭此映射方案，不重复测速或进入整网/Worker
接入，不改用旧融合方案挽救。**

下一项仅做CPU审计：确认正式PTX的函数归属，评估让get_func直接定位
所属module能否减少失败的跨module探测，同时保留原load_function调用。
它不缓存CudaFunction，也不新增共享Mutex；需先与已关闭的函数缓存/
延后fallback方案查重，不把旧WSL调用时间份额当作Windows收益。生产42
项保护、正式Fork比例与已认证Worker结果保持，目标仍active。证据：
`target/fork-parity-20260908/c384-gate-rowquad-r1/closure.json`。


**2026-09-15 module owner 直达查找：整网/Worker 数值通过，稳定收益低于门槛，关闭。**
`module-owner-audit-r1` 绑定正式CLI中全部20份PTX原始字节：SM120/SM89
各10个有序module、61个无重复入口；当前52处get_func调用涉及47个名称，
唯一动态helper参数也追踪到其实际调用者。cudarc仍每次构造CString、
调用Driver并保留Arc<CudaModule>。新设计只保存不可变的名称→module索引，
保留每次原load_function和完整失败回退；不缓存CudaFunction，不添加共享
Mutex，不移走GEMM中的get_func调用，也不混入旧描述符重放方案。

Driver实查两个目标的1220个symbol×module组合、双主机线程1232次直接/
回退控制通过；SM89是本机SM120兼容加载，不代表SM89硬件实测。1650项
状态机控制覆盖first-owner、缺失/错误索引与直接失败；成功命中后面的
重复名称不能靠fallback纠正，因此索引启用必须绑定完整PTX哈希和顺序。
普通缺失名称返回Driver500，空名称返回1，两者都保留原Rust完整扫描后
的通用notfound错误。初始空名称断言与API表检查错误均留档修正；Windows
已有Nsight导出虽含Runtime表及部分Driver调用，未记录ModuleGetFunction，
不能沿用旧WSL耗时份额估计收益。61名称的一次人工全扫423→61调用也不
是实际整网调用量或速度证明。

`module-owner-routing-r1/source` 从正式源码隔离实现默认0的
`KATAGO_CUDA_MODULE_OWNER_ROUTING_R1`。元数据CPU准备置于原module加载/
cuBLASLt初始化之后，两臂共同执行；坏元数据仅在启用时拒绝。schema2
要求显式key1和独立module_owner_artifact，指纹绑定路由源码、表、cuda.rs
及tactic_plan.rs。38项计划测试、4项相关测试通过（后者含2项计划测试的
重叠执行）。原cuda_exec及全部GEMM方法保持，CLI/native/lib-test均内嵌
原20份PTX各一次；strict/QKV、half编码及host revision保持。

构建复核发现原build.rs把12个源文件绝对路径计入core指纹，另一个配置
路径保留正斜杠。Rust实际路径拼写+完整哈希重算准确复现正式f126639f…
与隔离920d13cd…：13份输入文件内容全部相同，差异仅目录前缀。原“隔离
构建core ID也应相同”的准备假设被修正；实验计划诚实绑定新ID与独立
ec88e149…路由指纹，正式计划不迁移。函数签名解析、两份不同依赖统一
结果的rlib枚举、Thin-LTO诊断链接错误也留档；冻结候选源码和产物未改。

两份冻结Rust库的实际get_func分别验证61个基线符号，双线程重复共488
次直接句柄一致、12次缺失/空名称错误一致，三个真实分支标记通过。
B1/B14/B16、两臂双流共 **612,960个整网输出逐位一致**；Worker W32/W1
四组各128请求，11项数值门及top-1 128/128、Drain通过；重建512对原始
protobuf再算门通过。相反环境变量服从计划显式0/1；另8项真实CLI负例
拒绝旧binary、新字段缺失/过期、core/schema/model错误及未绑定开关。

唯一B14/S2 W80/N1000整网ABBA：event行/s几何均值
**1268.470391→1270.851745（+0.187734%）**，A/B spread
**0.463604%/0.696154%**；统一墙钟 **1265.492397→1269.427002
（+0.310915%）**，spread **0.464038%/0.712321%**。8000原始事件独立
重算一致；两指标稳定，但均未到1%门。**关闭，不进行Worker吞吐ABBA、
不采纳、不重试或与旧缓存/延后查找组合。** Worker本轮只做正确性认证，
不更新正式Worker吞吐或历史Fork比例；生产42项保护保持。

下一步先复用现有严格Attention的源码、SASS及资源证据，核查未覆盖的
精确softmax/主循环依赖、寄存器、同步与访存机制，再决定新候选。查重
已确认`strict-attention-graph-r1`此前Worker稳定下降8.689230%，维持关闭；
M64/stages2等历史路线也必须先查重。目标仍active。证据：
`target/fork-parity-20260908/module-owner-routing-r1/closure.json`。


**2026-09-15 strict Attention 固定 KV 循环展开：逐位/内存检查通过，稳定回退，关闭。**
`strict-attention-current-audit-r1` 把正式CLI内完整CUBIN与历史原PTX逐字节
绑定，并将原SASS全部3360个指令槽对齐53760字节ELF代码段。原先解析器
漏掉7个恒不执行的uniform-predicate指令槽，已保留记录并修正。实际首个
masked tile3之后，统一回跳精确执行tiles2、1、0三次；每轮显式线程栈读
32字节、写12字节。REG255/STACK40是真实静态资源；LOCAL0不能说明没有
栈访问，这些静态字节数也不等于DRAM流量或耗时。

`strict-attention-kv-unroll-r1` 只把这三次body改为固定完全展开，保持
M128/N96/stage1、grid3×12×14、block128、动态共享内存20480字节、六参数
ABI及KV顺序。StrictSoftmax与SM120子类源码逐字节相同。固定工具链499项
核验后离线导出：384个FP32 HMMA、396处完整普通expf数据流和RN归一化保持，
回跳消失，REG **255→217**、STACK **40→0**，无LDL/STL；代码段却由
**53760→99328字节（+84.7619%）**。满足预登记CPU门，尚不代表速度收益。

五组有限输入包含四组合成数据及实际TF3第一Attention的历史捕获数据。
真实fixture从留存的live Q/K、原V重新拼接，并确认half往返逐位准确；明确
标注其生产者是历史CLI，未冒充当前CLI的新捕获。两条独立stream、独立
输入/输出分配，每组candidate先跑两次再跑正式原CUBIN：普通数值运行
**58,222,080个half输出逐位一致**，全部有限、输入和两端guard不变；同
负载memcheck零错误、racecheck零hazard，三次共174,666,240元素再核验。
初始测试误将cuobjdump SHARED1024当作Driver静态共享属性，在kernel发射
前失败；只读复核两臂Driver属性均为0后，仅修正测试断言。候选CUBIN、
launch和精度门未改，失败记录保留。动态共享仍为20480字节。

唯一Windows B14/S2 W80/N1000算子ABBA，8000原始事件独立重算：event行/s
**344963.540595→336667.949281（−2.404773%）**，A/B spread
**0.039432%/0%**；公共主机enqueue到两流完成的吞吐
**339577.603694→332892.879891（−1.968541%）**，spread
**0.142967%/0.446714%**。16份测速前后完整输出仍逐位正确。原报告的ABI
字段共用候选geometry模板，其中sha不能代表A臂产物；独立复核按冻结
加载分支、绑定CUBIN及255/local40与217/local0实际Driver属性重新列出
两臂身份，原始记录保持。这不涉及测速重跑。

两指标稳定但均回退，**关闭，不接入整网、不跑Worker吞吐、不采纳、不
重试或调整展开度补救**。不能把下降归因于代码体积的单一因素；本轮没有
相关硬件计数器证据。正式42项保护、Worker既有1274.991675RPC/s验收和
历史同口径Fork比例保持。下一步仅审计原Attention的QK/softmax/PV数据
依赖、fragment复制及共享内存同步，先与所有关闭候选查重；找不到独立
机制则转向其他已测热点。目标仍active。证据：
`target/fork-parity-20260908/strict-attention-kv-unroll-r1/closure.json`。


**2026-09-15 strict Attention 独立 Q/V 的输出前屏障：数值安全，稳定无收益，关闭。**
`strict-attention-epilogue-barrier-r1` 先绑定当前共享内存布局：V在
[0,6144)、Q及输出在[6144,14336)、K在[14336,20480)。当前Q_in_regs=False，
三块互不重叠；最后一次Q/K矩阵读取在原主循环barrier0之前完成，剩余PV
只读V。通用epilogue的首个“等待V读完”屏障对Q/V复用配置有必要，但当前
输出只复用Q。因此候选仅在Q_in_regs=False时跳过这一屏障，保留输出
共享重排的后一个屏障、原KV循环、M128/N96/stage1及全部算术。

固定工具链499项核验及离线导出通过。完整SASS按符号指令序列对齐后，
**差异恰为删除一个barrier并增加一个末尾NOP**，其他指令顺序和寄存器
操作数完全保留；125处按PC比较的差异只是地址前移，不能写成125项独立
调度变化。REG255/STACK40、20480字节动态共享、53760字节代码段保持；
192处静态FP32 HMMA、196处完整普通expf、原三次回跳与half边界保持。

五组输入含四组合成及真实模型第一Attention的历史捕获，每组在两条独立
活跃stream上candidate先两次再原CUBIN，**58,222,080个half输出逐位一致**，
全部有限、输入/guard不变。相同负载memcheck零错误、racecheck零hazard；
三次共174,666,240元素独立复核。两臂实际加载的模块、函数及CUBIN身份
单列绑定，资源相同不能替代路径身份。

唯一B14/S2 W80/N1000算子ABBA：event几何均值两臂均为
**344827.593332行/s（0%）**，A/B spread均**0.039417%**；公共主机
enqueue至双流完成吞吐 **337581.459821→337493.210910行/s
（−0.026142%）**，spread **0.078277%/0.459682%**。8000事件和16份测速
前后完整输出独立重算一致。两项稳定但未过1%，**关闭，不接入整网、不
跑Worker吞吐、不采纳、不重试或与其他关闭候选组合**。正式42项保护、
Worker既有验收与历史同口径Fork比例保持。

下一步转向当前bottleneck上下投影的实际算法/布局审计：已有完整前向
剖析中两者各11次，合占kernel时间总和8.089511%（含重叠，不是墙钟占比）。
down K768/N384/beta0选中nvjet128×240×64 TMA核、255寄存器/95232字节
动态共享；up K384/N768/beta1为Fixed同族64×64×32/s6 TN核，88寄存器/
49152字节动态共享。先绑定真实描述符与SASS并核旧候选，尚未登记新性能
方案；不把其他形状的失败或资源数直接外推成这一层的性能结论。
目标仍active。证据：
`target/fork-parity-20260908/strict-attention-epilogue-barrier-r1/closure.json`。

**2026-09-15 bottleneck 下投影：实际 FP32/TMA 内核与真实数据已绑定。**
`bottleneck-down-selected-audit-r1` 确认当前 B14 的下投影为
M5054/N384/K768、TN、alpha1/beta0、half 输入及权重、FP32 累加和输出；
cuBLASLt 请求8项、仅按32MiB工作区过滤后取第0项。上投影虽然与 Fixed
FFN down 使用同一 CUTLASS 家族，但 M5054/N768/K384 不命中固定预设，
同样取启发式第0项。现有 NN 布局加载器只覆盖 K384，不覆盖此处 K768。

正式程序和当前17项 tactic 完整前向捕获第一处 Linear；完整 TF3 解析至
EOF，提取首层权重并核实输入可逐位往返 half。独立观察原算法在两条流
上的输出，**3,881,472个 FP32 元素与正式程序逐位一致**。实际加载的
29264字节 CUBIN（SHA256 `93bcdf91e5ace5d15922898c43ae861729cb86fd9909abeb5cf870d601c42166`）
为 nvjet SM120a 核：grid24×3、block64×4、REG255、动态共享95232字节、
STACK/LOCAL0；完整1624条 SASS 含120处 FP32 HMMA、4处 TMA load，
均为静态计数。实际 ABI 为一个576字节参数，由文档规定的 extra buffer
传递。两次诊断工具修正仅补 CUkernel 句柄解析和 packed 参数支持，失败
日志保留；含这些运行在内共9,703,680个 FP32 输出均与原始捕获一致。

Fork 认证计划关闭专用 outer down/up tactic；其源码在默认 FP16/NHWC
条件下将1×1卷积改为 NN Hgemm、half输出。此处是源码条件路径，尚未
独立观察 Fork 这两个形状的实际内核或累加精度，不借其他 FFN 的观察
代替。下一候选方向为仅 K768/N384、B14 下投影的 NN 权重布局，保留
half位模式、FP32计算/输出和全部数值门。须另行预登记实际 NN 路径与
ABBA；本次没有候选测速、没有性能收益声明，正式程序和认证计划不变。
证据：`target/fork-parity-20260908/bottleneck-down-selected-audit-r1/assessment.json`。

**2026-09-15 K768 下投影 NN 布局：算子 ABBA 通过，待隔离整网验证。**
`bottleneck-down-nn-layout-r1` 仅将已有 half 权重逐位转置为[K,N]，使用
NN 描述符；FP32 compute/scale/C/D、alpha1/beta0、32MiB工作区上限、
请求8项后按工作区过滤并取第0项均保持。实际选中64×64×32/s6 NN
CUTLASS 核，grid632×1、128线程、96寄存器、49152字节动态共享、
无本地溢出；完整960条SASS含16处静态FP32 HMMA。它与原TN nvjet
是不同路径，算法不透明字节相同也不能替代布局和形状身份。

真实首层输入加uniform/alternating/impulse/zeros共五组，各在两条活跃
stream上先运行候选两次、再运行原TN：**58,222,080个FP32输出全部
逐位一致**，完整FP64参考与条件误差界通过，重复/跨流确定性、全部
guard和只读缓冲检查通过。cuobjdump共享字段1024与Driver静态共享0
分别记录；修正的是CPU审阅器对两个工具字段相等的错误假设，没有重跑
GPU诊断或更改候选、数值门。

唯一B14/S2 W80/N1000 ABBA：event算子token行吞吐几何均值
151389887.716→165911632.903，**+9.592282%**；A/B spread分别
**0.047939%/0.013132%**。公共主机入队至两流均完成吞吐
149022131.688→162394858.248，**+8.973651%**；spread分别
**1.245357%/0.001125%**。8000原始事件与16份测速前后完整输出已复核；
beta0无需逐次D2D重置，两臂同口径。两项均过1%及5%波动门。

这只是该算子的收益，**尚未通过整网/Worker验收，不更新认证计划或
Fork整网速度比**。接入采用独立、默认关闭且绑定指纹的下投影开关，
只覆盖B14、N384/K768的11层，保留全局TN及QKV现有布局约束；不能为
此关闭已采纳的QKV路径。cuda_exec.rs参与QKV artifact哈希，隔离修改
后须如实更新该源码绑定哈希，同时验证QKV CUBIN/参数与全部20份PTX
不变；不得硬编码旧指纹。下一阶段为隔离构建、路径/负例、整网数值与
Worker golden，再按顺序进行整网和Worker ABBA。证据：
`target/fork-parity-20260908/bottleneck-down-nn-layout-r1/operator-decision.json`。

**2026-09-15 K768 NN 隔离集成：B1/B14 数值通过，B16 算法身份问题已定位至库返回。**
`bottleneck-down-nn-integration-r1` 已在独立源码/构建目录实现默认关闭的
`KATAGO_CUDA_DOWN_NN_R1`，仅 M5054/N384/K768、FP32 输出/beta0 使用
新 NN 布局；其他 batch/角色保持 TN。独立缓存键、形状/算法身份检查、
schema2/源码 artifact/模型/设备绑定与 plan 优先级已实现。CLI 构建、
38 项 plan 测试及 3 项 down 测试通过，20 份 PTX 和全部 QKV/strict AOT
资源与认证版逐字节一致；隔离路径导致的 core 与源码绑定 QKV 哈希如实更新。

开关两侧 B1/B14 的 **449,504 个整网输出元素逐位一致**，B13 双 handle
回退路径控制通过。两侧 B16 均在旧 residual preset 的
M5776/N384/K384、filtered index2 检查失败：word4 从认证值
`0x0001fe2800000001` 变为 `0x0001fe2803000001`。旧认证测试程序的
单次 B16 对照通过，81,728 个输出逐位一致，实际 Lt DLL 哈希未变。
因此不能用旧程序或其他 batch 的成功补足隔离版 B16 验收。

两份只加公共 API 读取的源码诊断版均通过，输入描述符/布局/偏好读值相同，
共163,456个输出逐位一致；它们改变了程序布局/调度，不作为修复证据。
随后对**原失败二进制**做一次硬件断点观察：6 次查询/返回、48 项成功
结果，输入结果数组均已清零；两条 lane 的 index2 在库刚返回、Rust
调用者首条指令尚未执行时已含相同失败字节，进程仍退出101。另有不同
查询返回 `0x0001fe28679f0001`；不推定这些字节是无害 padding。

9 项公开算法配置与原记录一致。通过公开 `AlgoInit` 和全部可写
`ConfigSet` 重建，零值/A5 两种预填充均得到原认证的完整64字节。这只
登记为下一步正确性调查入口：须先验证实际内核、数值和 fail-closed
绑定，不能掩码忽略字节、放宽旧检查、替换固定索引或直接采纳重建方式。
本阶段**无整网/Worker ABBA、无正式源码或计划更新**；原算子
+9.592282% event / +8.973651% host 仅保留为算子结果，Worker1274.99
RPC/s与历史Fork比例不更新。目标仍 active。证据：
`target/fork-parity-20260908/bottleneck-down-nn-integration-r1/diagnostic-review.json`
及同目录 `native-win-r1/partial-review.json`。

**2026-09-15 K768 NN 整网裁决：数值全部通过，性能未达门限，候选关闭。**
后续 `lt-public-constructor-r1` 独立验证了公共算法构造方式：四种认证
GEMM 形状、17 组输入、两条 stream 的 211,262,976 个 FP32 输出均与
原认证算法逐位相同，完整 FP64 误差界、83 项负例和102次实际内核观察
通过。保存的失败描述符可以经公开配置构造出原认证的全部64字节；
实际执行仍核对全部字节，不屏蔽未知位，也不推定其私有含义。

`bottleneck-down-nn-integration-r3` 将该构造方式绑定到独立
`lt_algorithm_artifact`，使用残差预设或新 NN 路径的计划必须为 schema2
并精确匹配。r2 的一次旧测试预期失败保留；r3 仅修正该测试对 schema1
的旧预期，运行时逻辑不变。39项计划、2项构造、3项下投影测试通过；
20份PTX及已有AOT资源不变。开关0/1×B1/B14/B16的 **612,960个整网
输出元素逐位一致**，B13回退通过；4组Worker golden的512请求全部通过
11类门、top1与原始protobuf复核。16项实际CLI负例通过；其中原schema1
控制先被QKV门拒绝的记录保留，另加关闭无关tactic的负例验证构造专属门。

唯一整网ABBA比较**正式CLI A与完整隔离候选B**，B14/S2、空盘、
W80/N1000、相同计时边界，复核8000事件。event几何均值
**1281.715578→1278.414375 NN rows/s（−0.257561%）**，A/B spread
0.293541%/0.073177%；公共host几何均值
**1276.248399→1273.108613（−0.246017%）**，spread
1.062466%/0.532937%。均稳定但未过1%收益门，状态
`REJECT_FULL_FORWARD_BELOW_MINIMUM_GAIN`。这是完整候选的比较，不能
单独归因于构造方式，也不能用算子+9.59%覆盖整网裁决。

**关闭本轮K768 NN性能候选，不重跑、不组合挽救，不运行Worker性能ABBA，
不迁入正式源码或计划。** 公共构造方式的正确性证据单独保留，尚未部署。
正式Worker仍为1274.991675 RPC/s的既有验收记录，历史Fork比例不更新。
下一步回到Fork实际外层投影内核的只读观察，须取得新的、不同于已关闭
NN方案的路径证据后才登记新候选。证据：
`target/fork-parity-20260908/bottleneck-down-nn-integration-r3/final-review.json`
及同目录 `forward-win-r1/review.json`。总目标仍 active。

**2026-09-15 Fork 外层投影实际内核审计完成；上下投影均确认FP16累加。**
`fork-outer-selected-audit-r1` 使用原绑定WSL Fork二进制
`5df63305086e…`、原RUNPATH库与同SHA模型，未重编Fork。一次最小诊断
捕获两stream各11次down和11次up，共44次真实`cublasHgemm`，逐调用
核对NN、half输入/输出、alpha1、down beta0/up beta1及原调用者PC。
调用返回PC均为`ConvLayer::apply+0x442`；结合链接分支可确认进入通用
GEMM回退，未直接采样hook返回寄存器。所有诊断耗时排除。

| Fork实际角色 | 内核命名中的tile | grid / 线程 | 寄存器 / 动态共享 | 累加 |
|---|---|---|---|---|
| down，列主序M384/N5054/K768 | h16816 128×256×32/s3 NN | 80×1×1 / 256 | 160 / 73728B | FP16 |
| up，列主序M768/N5054/K384 | h16816 256×128×32/s3 NN | 160×1×1 / 256 | 154 / 73728B | FP16 |

实际`CUfunction→CUmodule→CUlibrary`映射绑定到同一原加载ELF；完整
程序头、节表与命名text均保留。两段35968/34688字节分别解码2248/2168
条指令，各含64处`HMMA.16816.F16`，无FP32 HMMA或本地溢出。这是
这两个外层核的证据，不能用其速度代表Rust保留FP32累加的效率。

另确认Fork加载期的通道缩放：trunk-tip BN提取的768项因子中，766项
为0.5，第302/600项为0.58962971/0.59746695。它们在half转换前以FP32
乘到11层up的输出通道。重建后两流22层共12,976,128个实际half权重
全部逐位匹配；down对应原始权重。最初把up直接等同文件原始half权重
的CPU检查失败留档，按真实源码变换修正参考，没有重复GPU诊断。
因此同文件SHA还不足以证明双方内部权重和激活范围相同；不据此移植
缩放或更改Rust的half边界。

当前Rust上投影仍为FP32的64×64×32/s6 TN、88寄存器/49152B动态共享。
历史`outer-up-nn-b14-r2`因唯一ABBA波动21.75%/24.09%已关闭，新K768
down NN也已关闭，两者均不重试。下一步只审核不同的FP32 TN分块组织：
先绑定当前上投影操作数/算法，再筛查单一128×128×32/s3原型是否与
历史候选重复、资源是否可行；尚未登记性能候选、未编译该原型。
证据入口为本审计的`assessment.json`与`final-review.json`。正式配置、
Worker验收值和历史Fork速度比不变，目标继续active。


**2026-09-15 B14 外层上投影 TN128：数值逐位通过，算子稳定性门失败，关闭。**
`outer-up-tn128-b14-r1` 先筛查历史B4：相同128×128×32/s3族的上投影
只测过B16，后续TF3三分块探针仅覆盖N384。本项限定当前B14/N768/K384，
不是重跑B16 B4、旧上投影NN或新K768 down NN；旧关闭结论保留。
当前正式CLI的完整前向边界15/16重新取得真实输入/残差/输出，模型原始
权重解析至EOF；没有套用Fork加载期缩放。基线实际64×64/s6 TN核、
88寄存器与当前剖析一致，新128×128/s3为226寄存器、49152B动态共享，
grid40×6；两者完整SASS确认FP32 HMMA、零local spill。CUPTI观察30次
matmul，每次只发射一个已绑定的核。辅助源码编译了其他tile模板，但未执行。

五组输入×两流×基线/候选/重复，共 **116,444,160** 个FP32输出逐位一致，
全量FP64条件误差门、只读操作数与前后guard通过。唯一S2/W80/N1000 ABBA：
event基线 **508501.94/441688.62**、候选 **399928.28/399739.77** rows/s，
名义 **−15.632471%**，基线spread **15.126792%** 超过5%；共同主机墙钟
名义 **+14.898154%**，不能抵销event失败。该算子每次事件外恢复原位残差，
不代表整网计时；全部8000事件、16份计时前后完整输出保留复核，不丢弃样本。
按`CLOSED_TN128_OUTER_UP_UNSTABLE_OPERATOR_NO_PRODUCTION_CHANGE`关闭，
不接入整网、不测Worker、不调参补救或重试。正式42项保护文件保持原值。

下一项仅做当前四行/四warp RMS的固定C384源码及生成代码审核：Fork已有
固定384入口，Rust现有PTX仍含动态列数乘法/转换/除法。保留Rust原FP32
归约顺序与正确舍入除法，不搬入Fork的half输入和不同归约树，不重开旧八lane
vec4或RMS融合候选。是否有独立代码差异与资源收益尚待验证，未登记性能候选。
收据：`target/fork-parity-20260908/outer-up-tn128-b14-r1/closure.json`。
当前Worker已验收值 **1274.991675 RPC/s** 与历史Fork比例不变；目标继续active。


**2026-09-15 固定C384 RMS：实际代码缩短、数值逐位通过，算子波动门失败，关闭。**
`rms-fixed384-r1` 从当前认证的完整executor PTX出发，仅将w4入口的动态
`ncols`加载替换为常量384；其余字节保持，12项FP32 FMA、XOR归约顺序、
`div.rn.f32`、rsqrt和RN half边界全部不动。Fork的固定384入口提供专项化
参照，没有搬入其half输入/参数或不同归约树，也没有重跑八lane vec4候选。
离线与实际Driver默认JIT的完整代码逐字节相同：264→248条指令，36→34
寄存器，零spill；可驻留块数均12，不能将寄存器下降当成驻留提升。

当前正式CLI完整前向边界3重新取得真实RMS输入/half输出，模型解析至EOF
取得首norm1参数。26组输入×两流×基线/候选/重复，共 **64,348,416** 个half
输出逐位一致；含不同epsilon、短尾行、次正规数和大动态范围。另用精确
整数有理数及ties-to-even构造 **591,735** 个FP32除法参考，动态/常量除数
在两流的 **2,366,940** 个输出全位相同。156次RMS和4次除法实际路径已绑定，
只读/前后guard通过，6项错误列数在发射前拒绝。首个CPU runner的manifest
类型错误保留；在创建运行目录前修正，没有重跑GPU数值进程。

唯一S2/W80/N1000 ABBA：event名义 **−3.798460%**，基线spread
**11.323521%**；共同主机墙钟名义 **+17.065273%**，两臂spread
**8.458607%/17.288966%**。按5%稳定性和双指标≥1%门拒绝；全部8000事件
与16份完整输出保留，不接入整网、不测Worker、不重试或改判。
关闭收据为`target/fork-parity-20260908/rms-fixed384-r1/closure.json`。

CPU复查发现这轮RMS的1000次计时窗口实际仅 **27.42–36.20ms**，上一项
上投影为 **122.46–142.23ms**；原始数据不足以断言是GPU频率、系统调度
或流重叠导致波动。下一步独立审核轻量算子的测量可靠性：仅用当前正式
RMS基线，预登记持续预热/较长多次调用窗口及只读时钟遥测，保留所有样本；
不运行任何已关闭候选、不设置锁频或改变驱动。该诊断尚未执行，不产生收益
或重认证结论。42项正式保护文件、**1274.991675 RPC/s** 已验收Worker值
和历史Fork比例均保持，目标继续active。


**2026-09-15 RMS测量可靠性诊断完成：长窗口仍有波动，孤立算子不能代表整网。**
`rms-baseline-window-audit-r1`只运行当前正式完整PTX的w4 RMS，使用真实B14
输入、两流和相同默认Driver JIT。4个进程各预热1秒、计时1000次逐调用事件，
再保留8个每流25000次调用的长窗口。全部8000个短事件、64个长事件区间和
**31,051,776** 个half输出保留，逐位相同、guard及只读检查通过。首次构建的
Windows `SHORT`类型名冲突已保存，仅在GPU测量前重命名计数常量。

短窗口跨进程event/共同主机墙钟spread **20.124655%/63.486851%**；每进程
合并8个长窗口后为 **4.764036%/6.871017%**。单个长窗口跨全体的spread
仍为 **35.856050%/67.887329%**；不丢弃窗口，不单独取较平稳的event作结论。
长窗口204个遥测样本的SM频率 **2797–2820MHz**、Pstate均1。请求50ms
采样，实际中位间隔约62ms；4个短窗口仅有0/1/1/1个样本。没有设置锁频、
功耗或驱动策略，离散遥测不能确定历史失败原因。

独立预登记的单次`rms-baseline-host-gpu-trace-r1` Nsight诊断逐一关联
**550,738** 个实际RMS kernel与launch API，额外 **7,762,944** 个half输出
逐位相同。8个窗口无本进程RMS运行时间占 **36.1214%–37.6914%**；各流窗口
kernel中位5.632–5.760µs，launch API中位5.221–5.380µs、p99约123–167µs。
这些是剖析中的观察，不能区分OS调度、驱动内部等待或其他GPU客户端；也不能
与未剖析数据混成性能收益。全部实际发射数量匹配，没有剔除样本或重采集。

CPU重新核算已保存整网剖析，严格保留原来的每流最后100次完整前向：选中
内核并集占GPU时间包络 **99.652122%**。当另一选中流没有内核运行时，FFN、
Attention、RMS分别占包络 **38.469443%/27.109761%/0.273690%**；完整角色
重叠矩阵保存于`context-accounting.json`。独占时间不是可实现收益上限，
但支持回到FFN/Attention热点，不将孤立RMS发射间隙外推为整网损失。

本轮不新增认证性能收益，不改变整网/Worker计时口径，也不重开旧候选。
下一项仅审核FFN全局到共享内存搬运机制：先查完整历史，沿当前DualGemm及
Fork源码核对搬运路径，验证SM120 TMA可行性与精度/布局成本；尚未确认可行
或登记性能候选，不以更改旧tile、warp或stage重试代替新机制。42项正式
保护文件、**1274.991675 RPC/s** 已验收Worker值和历史Fork比例保持。
详细报告：`target/fork-parity-20260908/rms-baseline-window-audit-r1/assessment.md`。
目标继续active。


**2026-09-15 FFN TMA搬运机制验证通过：完整GEMM与性能尚未验证。**
`ffn-tma-transfer-feasibility-r1`沿当前DualGemm/CUTLASS源码确认，正式路径
使用每线程`cp.async`，三个共享内存阶段交错存放。新机制保留全局TN布局，
以64B swizzle的TMA将A/B0/B1放入各自连续阶段；保持原生warp读取器类型，
改为阶段基址和leading dimension32。24,576项共享布局映射验证通过，其中
24,192项与旧地址不同，不能只替换搬运指令。该机制不改变FP32累加或激活。

真实首FFN输入/权重加覆盖全部65,536种16位模式的数据，两流、原布局参考/
TMA/TMA重复共12次kernel发射：**47,185,920** 个逻辑共享值和
**94,371,840** 个原生warp片段逐位一致。覆盖全部40个M瓦片、18个N瓦片
（配对n=m%18，不宣称全部720组合）、12个K步、三槽四轮复用；尾瓦片
62行有效、66行补零。所有全局/共享guard、输入和tensor-map只读检查通过。
生成的TMA函数有12个静态`UTMALDG.2D`及16个`LDSM`站点，无浮点算术；
56寄存器/零local只属于搬运探针，不能外推到完整GEMM。

完整未缩减探针通过Compute Sanitizer **memcheck 0 errors** 和
**racecheck 0 hazards/0 warnings**；两次检查各24份输出与未插桩结果全位一致。
默认附加方式曾超时，连单线程单整数CUDA对照也在初始化阶段无法附加；原始
失败保留。调试接口原本已开启，采用文档支持的命名管道及application-only
跟踪后对照与完整探针通过，未改注册表/驱动/锁频。另保留LF/CRLF源码哈希
比较中断及MSVC不接受128B对齐描述符按值参数的构建失败；后者改为预先上传
的对齐、带guard、只读全局描述符。全部修正在性能测量前，当前无性能测量。

Fork仍以实际选中代码为参照：首/gate投影是独立cublasHgemm、FP16累加及
LDGSTS搬运，并非选中的TMA融合FFN。Rust此项是保持精度的新搬运机制，
没有将旧tile/warp/stage或NN布局失败候选重试改判。527份相关保留源码已作
搬运标识筛查，范围与排除项均留档。

下一步实现完整TMA DualGemm：维持128x64x32、四warp、三阶段及原swizzle2，
K循环保留运行时边界，使用原FP32 MMA顺序及原DualEpilogue、RN half边界/
精确SiLU；完整720个B14 CTA、实际首FFN与边界数据、内存/竞争检查通过后
才登记唯一算子ABBA，随后仍需整网/Worker双门。不得将本轮搬运兼容性当
成GEMM正确性或性能结果。42项正式保护文件及 **1274.991675 RPC/s** 已验收
Worker值保持，目标继续active。


**2026-09-15 完整 FP32 TMA DualGemm：数值与竞争门通过，稳定退步，关闭。**
`ffn-tma-dualgemm-r1`把已验证的64B-swizzle TMA搬运接入完整计算；维持
128×64×32、四warp、三阶段、TN权重、swizzle2和运行时K边界，沿用原生
FP32 MMA顺序与完整DualEpilogue、RN half投影边界及精确FP32 SiLU。
三槽用mbarrier累计每阶段16384字节，所有消费者完成后才覆盖；公开编码的
TensorMap由每lane对齐、带guard的只读全局缓冲持有，上传在计时外。

8组真实/零/稠密随机/有限half边界输入，M覆盖1/127/128/129/361/5054/5776，
双流、原核/TMA/重复共48次发射；每进程完整核对 **80,904,960 half**。
B14全部 **720个Cartesian CTA**、尾行和完整输出通过，真实首FFN与当前
整网捕获逐位一致。未插桩、memcheck、racecheck三个完整进程均通过：
合计242,714,880个输出half检查，所有输出有限，guard/输入/描述符未改。
memcheck **0 errors**，racecheck **0 hazards/0 errors/0 warnings**；沿用
命名管道/application-only，不改注册表/驱动，不缩减数据或强制串行。

原核2992条SASS指令与已安装CLI逐条相同，计时二进制的两核也与数值/竞争
验证二进制逐条相同。新核2848条、6处静态UTMALDG、64处FP32 HMMA；
寄存器 **236→165**、dynamic shared **49152→49664B**，双方0 local且仍
潜在2CTA/SM。少指令/少寄存器没有自动带来性能收益，不以静态计数归因。
两次CPU提取脚本中断保留（函数前缀未匹配、已写文件的freshness断言），
仅修正读取/同字节验证；未重新构建或重跑GPU以修正提取。

唯一 B14/S2/W80/N1000 ABBA，8000原始events与16份完整前后输出复核：
event token-row吞吐 **−11.093764%**，A/B spread **0.557296%/0.025623%**；
统一主机完成时间吞吐 **−10.873833%**，spread **1.092695%/0.614926%**。
统一墙钟由两lane最早开始至最晚stream完成计算，双方完成前无结果拷贝或
event读取。两口径稳定且退步，候选关闭；不调整尺寸/阶段/warp/流水重测，
不与旧候选组合，不接入整网或Worker，42项生产保护文件保持。

下一步仅审查独立的Fork FFN累加精度受控参考是否可实现：先查1357份历史
源码/报告筛查的48个命中，复现Fork实际Hgemm的布局/数据/句柄状态与原
输出，再判断能否只改FP32累加请求。旧Rust两NN实验是Lt FP32与融合TN
之比，不能代替Fork实际FP16/FP32精度贡献测量。此入口尚未授权性能测量，
不降低Rust精度、不改Fork生产，也不将不同库/算法差异藏进精度结论。
已有 **1274.991675 RPC/s** Worker验收值与历史Fork/Rust比值保持，目标active。


**2026-09-15 Fork真实FFN精度策略参考完成：库路径差异显著，不能外推整网。**
`fork-ffn-accumulator-reference-r1`完成1357份历史源码/报告筛查中48项命中的
复核。旧Windows linear2累加归因（不稳定）和Rust两NN投影候选均保持关闭；
本项仅研究实际WSL Fork首FFN first/gate，未复用其性能结果或降Rust精度。

原Fork二进制、模型与认证计划SHA保持，原RUNPATH实际cuBLAS版本为
**13.1.1（130101）**，不是本地CUDA13.3工具包所带的13.6库。原生双流
捕获4次Hgemm、12份完整张量：1,769,472个权重half、7,762,944个输入half、
23,288,832个输出half。math/pointer/atomics/SM target均0，未观察到显式
workspace设置；不推测库私有workspace大小。first/gate在同lane共享输入
指针，两lane各有句柄/流。数据来自初始化warmup，不能默认等同计时空盘。

Hgemm没有单独的累加精度参数，故先建API等价对照：在原13.1.1运行库、
原half数据和NN形状下，**GemmEx16（共同math16）与原Hgemm的完整输出
及实际加载机器码完全相同**。随后GemmEx32及重复通过全部有限性、guard/
只读检查。16次原/Ex16/Ex32/重复调用共检查93,155,328个输出half。
原/Ex16为FP16 HMMA、154寄存器、256线程、73728B shared、grid160×2；
Ex32为FP32 HMMA、96寄存器、128线程、49152B shared、grid632×3，均0 local。
这同时改变了库选择的分块/阶段数，不能称为单独HMMA精度成本。

完整FP64点积参考按361种输入行无损重建每lane全部5054行，验证预先定义的
逐元素前向误差界与half输出舍入。first投影FP16/FP32 RMSE分别
**4.057815e-4 / 1.210127e-4**，gate分别 **3.617230e-4 / 1.053779e-4**；
双lane一致。FP16理论界较宽，只是诊断，未代替Rust整网top-1门。
有关GemmEx混合精度归约控制见[NVIDIA cuBLAS说明](https://docs.nvidia.com/cuda/cublas/index.html#gemm-algorithms-numerical-behavior)。

计时使用每lane一个共享输入、独立两份权重和输出，first/gate两次GemmmEx
共同放进事件；双方采用同API、DEFAULT算法枚举、math16禁止降精度归约，
只变compute type及其要求的标量存储类型。数值探针曾为两role各建等值输入
副本，计时前已按原Fork共享指针组织，并用两个独立短进程复核完整输出及
实际选中机器码。性能进程不加载观察器，也不设置LD_LIBRARY_PATH。

唯一B14/S2/W80/N1000 ABBA，8000事件、32份完整前后输出（186,310,656
half）全部复核。**FP32相对FP16吞吐**：
- GPU event：**−59.041246%**，FP16/FP32 arm spread **4.821439%/0.034251%**；
- 两lane统一主机完成时间：**−15.112136%**，spread **0.750964%/0.107593%**。

两口径通过5%稳定门（FP16 event较接近上限），保留所有样本、不重跑。
这是两个FFN投影的token-row吞吐；不含SwiGLU/其他层，不是整网boards/s或
Worker RPC/s。两口径差距也表明不能将局部event比例直接乘到整网。
当前Rust正式版本、42项保护文件、**1274.991675 RPC/s**已验收Worker值及
历史Fork/Rust比例保持不变。

下一步先审查完整前向FFN精度参考的可复现方法：原benchmark的
actualWallSeconds包含warmup/线程收尾，getOutput结果随后删除，尚不满足
统一计时跨度及整网输出留存要求。须绑定原WSL源码/对象文件，隔离必要的
主机记录改动，证明原核/全头/双lane输出等价，再考虑唯一完整前向ABBA。
当前模型有33个FFN，目标调用数须按实际完整前向验证，不能假定11个；不得
误选同形状QKV。该入口未登记整网性能，不改变生产或Rust FP32纪律，目标active。

**2026-09-15 Fork整网FFN精度策略参考完成：event −30.86%，公共墙钟 −3.01%，均稳定**

`fork-whole-ffn-precision-reference-r1` 已完成同一原生TF3模型、RTX5070Ti、
B14/S2的完整前向对照。原Fork339个链接输入重链接后的完整可执行文件、
原cudabackend.cpp重编译的完整对象文件均逐字节一致。隔离版本只改主机
记录及FFN first/gate调用，复用全部原GPU/AOT对象与实际cuBLAS13.1.1。
原始对象上的只读getOutput包装取得完整输入及五类原始输出；隔离control
与之逐位一致。语义调用点实际验证每网**33个FFN、66个投影**，不是11个。

control/Ex16/Ex32各在2lane×3次前向检查396个完整投影输出的SHA、有限性、
全部权重及输入；各自跨lane/重复一致。Ex16与原Hgemm全部投影数据、整网
五头以及实际加载的完整机器码均一致。Ex32全部投影选中已核实的FP32 HMMA
代码；两臂都在目标调用内设置math16、立即恢复原math0，其他算子保持原路。
关闭逐层转储后又以同计时二进制、只读外部观察器核实每臂660次实际目标
调用（含启动）及全部输入/整网输出；正式计时不加载观察器。

独立数值参照采用原Fork通用FP32路径：关闭SM120专用后端及plan、
cudaUseFP16=false、NHWC=true、NVIDIA_TF32_OVERRIDE=0。完整输入字节相同，
两lane原始输出一致。Ex16/Ex32均通过预先固定的五头绝对误差门及top-1
28/28；policy最大误差分别**0.015619/0.011580**，value分别
**0.010069/0.001942**，ownership分别**0.002125/0.001406**。scoreValue
误差则从0.005332增至0.006268，不能宣称每一头都因FFN升精度而更接近参考。
这是同语义空盘重复输入的完整网络检查，不是多局面/Worker认证；该FP32
参考也是Fork内的独立运算路径，不是另一引擎或FP64形式化oracle。

唯一固定ABBA（Ex16/Ex32/Ex32/Ex16，W80/N1000）保存8000个原始event；
四进程的前/预热后/计时后输出共367,584个FP32元素均与各自数值样本逐位
一致。统一主机跨度为最早lane开始至最晚stream完成，排除加载、预热、
转储、事件读回及线程join；两lane完成屏障阻止较快lane转储干扰另一lane。

| 整网指标 | Ex16 | Ex32 | Ex32相对变化 | Ex16/Ex32 spread |
|---|---:|---:|---:|---:|
| lane event中位数合计 NN rows/s | 2222.771691 | 1536.874315 | −30.857752% | 3.385109% / 0.147155% |
| 两lane统一主机跨度 NN rows/s | 1565.857062 | 1518.799879 | −3.005203% | 2.723049% / 0.108901% |

两口径通过5%稳定门，完整结果留存且不重跑。这里比较的是公共GemmEx精度
策略，库选中的tile/内核也不同；两臂还有相同的逐调用handle检查及math
设置/恢复。故不能将event差异当成原Fork真实吞吐损失，也不能把墙钟的
−3.01%直接解释为未修改原生Hgemm路径的升精度成本。其余Fork QK/linear2
仍保留原精度，不更新历史Fork/Rust比例或Worker结论。

下一步只审查原生Hgemm与公共Ex16调用层的主机影响：复用可逐字节重建的
Fork对象，保留最小整网计时/输出记录，先证明原始GPU调用及全头等价，再
登记独立的测量控制实验。这不重试本轮ABBA，也不重启任何旧负收益候选。
完成边界核实后回到Rust现有FFN热点，按历史筛查选择新的实现改动。
本阶段7个数值/路径诊断和4个性能进程均成功；1次观察器C++重载指针编译
失败已留存，并在GPU运行前以显式ABI修正。原Fork文件、Rust42项保护文件、
正式plan/CLI及已验收**1274.991675 RPC/s**保持不变，未采纳新性能候选，
总目标继续active。


**2026-09-15 原生 Hgemm 主机控制与顺序 FFN 候选关闭，转入三流验证**

`fork-native-host-control-r1` 只在原 Fork benchmark 加公共计时与完整输出留存，
apply 热路径不插入检查/计数器。只读诊断以完整权重字节识别 33 个 FFN 的
first/gate，660 次调用的原生 Hgemm 机器码、输入和五头输出均与既有参考一致。
唯一 B14/S2、W80/N1000 的 native/Ex16/Ex16/native ABBA 已完成：event 几何均值
2087.132736/2354.818122 NN rows/s，spread 0.960519%/8.211611%；公共主机吞吐
1574.505130/1559.947090 NN rows/s，spread 1.499989%/0.764502%。Ex16 event 超过
5% 稳定门，**整项控制实验关闭，不重跑**。主机口径测得 −0.924610%，不能据此
反推纯查询/设置开销或修订既有精度、Fork/Rust、Worker 性能结论。五个 GPU
进程均成功，8000 个原始 event 及 367,584 个性能输出 FP32 元素完整留存。

独立 `ffn-sequential-projection-feasibility-r1` 尝试先完成一个 FP32 投影，在
原 RN-half 边界将 64 个元素打包为 32 个寄存器字，再计算第二投影，复用原精确
SwiGLU epilogue。固定 128×64×32、两级单投影流水的 CPU 编译验证显示 shared
scratch 49152→24576 字节、寄存器 236→168，仍保留 FP32 HMMA，但产生 104 字节
栈及各 120 字节 spill 读写，**未通过预定零溢出资源门，关闭且不进入 GPU 验证**。
一次半精度位转换辅助函数编译错误已保留，并在 GPU 运行前按实际 CUDA API 修正。
不将 cuobjdump 的 SHARED 字段当作驱动实际静态分配，也不声称已实现三 CTA 驻留。

下一项只验证新的 Rust B14 三流拓扑：nnbench 的 `--handles 1,2,3` 已支持共享
只读模型与独立 stream/workspace；Worker 配置可创建三个计算句柄。每句柄的
两个异步 slot 不是全局流数限制，无需改 slot/token 生命周期。先隔离扩展完整
16 局面转储，核实三流全部原始输出/重复与两流逐位相同，再准入唯一完整前向
S2/S3 ABBA；通过后才验证相同 C64 Worker 的 128 局面 C++ FP32、Drain 和独立
Worker ABBA。记录实际 batch 分布及 B14 路径覆盖，不能从饱和前向推断 Worker
收益。本轮未改生产、42 项保护文件及正式 plan/CLI，已验收 Worker 仍为
**1274.991675 RPC/s**，总目标 active。


**2026-09-15 Rust 三流实验：当前双流数值控制遭算法身份拒绝，未进入测速**

`rust-three-stream-r1` 已隔离扩展16局面转储，允许三流并保留两轮完整输出。
第一次复用默认 target/release/deps 的测试程序/rlib，三进程虽执行成功，
审核发现该测试仍是旧 core f718a…、无 immutable-QKV；正式 CLI 则保持当前
f1266…。这三组数据全部判为无效当前基线证据，未作数值/性能采纳。
依赖发现阶段另保留 proc-macro DLL、相同 Cargo run-build 收据两次 CPU
修正；没有因此重编推理库或改生产文件。

随后 `current-runtime` 以正式 CLI 同次验收的完整测试/rlib 重新链接，
先用不发射 GPU kernel 的指纹程序确认 core/QKV/strict/host/Lt 全部匹配。
真正的原始 B14/S2 数值控制退出101：FFN down (5054,384,1152)、过滤索引1
返回 data[4]=0x0001fe2800df0001，认证值为0x0001fe2800000001，其余七词相同。
全头转储尚未完成，三流当前版本根本未运行；因此本项按控制失败关闭，
**完整前向和 Worker 性能均为0，不评价三流快慢、不重跑、不放宽身份门**。

新源码审计确认缓存查询锁在 heuristic 前释放、结果插入另行加锁，两个
lane 可以同时初始化同一缺失键。此次日志在失败前已有同形状 fixed GEMM
成功标记，值得独立检查该初始化窗口；尚未证明两次查询实际重叠，也未解释
未知字节根因。既有 public-constructor 正确性证据保留，但不重启已关闭的
K768 NN/构造、模型副本或三流性能实验。算法内部结构和公开属性接口边界见
[NVIDIA 文档](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmulalgo-t)，
不据此掩码私有位。下一步仅为独立缓存初始化的源码/CPU并发与错误传播验证，
保持原算法索引、全部64字节身份、精度和失败传播，再决定是否登记新的诊断。

全部4个数值 GPU 进程中3个旧运行时样本无效、1个当前基线失败；不存在通过的
当前S3完整输出或性能证据。42项生产保护文件和正式CLI/计划均未改变，已验收
Worker **1274.991675 RPC/s**及历史Fork比例不更新，总目标继续active。


**2026-09-15 Lt 算法缓存并发初始化：隔离源码、CPU 和整网原始输出验证通过**

`lt-cache-initialization-r1` 实现独立的每键初始化：缺失键以 OnceLock 共享
一次已校验结果，成功或校验错误均保留；失败请求不会由另一线程自动重选。
缓存映射锁在查询/排名前释放，实际 GEMM 执行不持锁；不同键可并行初始化。
初始化 panic 仍向原调用者传播，等待方收到固定错误。普通冷图捕获保持临时
选择、已就绪键可复用，空候选仍返回原 false 回退且不永久缓存。原 heuristic
顺序/固定索引、全部64字节身份检查、排名 scratch 及设备运算代码保持原路。

9项确定性CPU测试全部通过，覆盖同键竞争、不同键并发、校验错误/panic、
执行时无锁、冷/热/等待中的图捕获及空候选。测试用明确注入的两份查询结果
重现了旧拆分锁序列中“已有合法缓存、另一竞争者仍拒绝”的时序；这证明源码
窗口可产生该现象，**不是 cuBLASLt 私有字节或库内竞态的根因证明**。

隔离 Rust 推理库复用当前验收构建的全部 CUDA 生成物与依赖。首次编译因复制
src 后缺少6份相对路径AOT文件失败；补齐原字节文件后通过。资产扫描又遇到
已存在的自引用源码，保留失败记录并改为验证原字节，未修改数值门或GPU代码。
traced/untraced 两个实际运行时的 core、strict、QKV、FP16/host版本和Lt版本
均与当前设备构建指纹一致；新增主机代码另由源码/rlib/可执行文件SHA绑定，
**旧设备指纹本身不构成新增主机行为的生产认证**。

唯一带初始化记录的B14/S2诊断中，6个键各恰好1次begin/end，固定残差算法
返回原认证完整64字节；双lane、两重复、16局面及所有补齐行的**286,048个**
原始输出FP32元素与已认证参考逐位一致。移除日志后另行登记B1/B14/B16三组
完整数值检查，双lane/两重复合计**612,960个**输出元素全部逐位一致，完整输入
字节、独立stream/workspace、B14 strict/QKV/rowquad与B1/B16回退标记均核实。
4个GPU数值进程均成功，无GPU重试；本轮没有性能样本或真实Worker验收。

下一阶段接入隔离CLI/Worker，先建立覆盖缓存模块及CUDA/plan调用者的独立
主机指纹，要求临时schema2计划精确绑定并拒绝旧/缺失身份，然后做128局面
C++FP32、全部11字段/top1、原始protobuf及Drain验证。通过必要数值/路径/
绑定门后，才可登记此独立候选的唯一完整前向及Worker ABBA；正确性通过不等于
吞吐收益或普遍启动可靠性。旧三流、K768 NN、模型副本及构造性能候选不重启。
42项保护文件、正式CLI/计划、已验收Worker **1274.991675 RPC/s**与Fork历史
比例保持不变，当前初始化候选待CLI/Worker验证，总目标继续active。

**2026-09-15 Lt 并发初始化 CLI：数值通过，完整前向未达性能门，关闭。**

`lt-cache-cli-integration-r1` 把上一轮每键初始化实现接入独立源码/目标目录的
CLI，增加覆盖缓存模块及 CUDA/plan 调用者的 `lt_initialization_artifact`。
全部实验计划强制 schema2 和精确身份，38项计划CPU测试、9项并发CPU测试及
15项实际CLI拒绝检查通过。旧计划/旧CLI、缺失或错误主机/core/QKV/strict/
FP16/Lt/model身份均拒绝，未给正式计划补签。

隔离编译触发现有 build.rs 的绝对源码路径参与 core hash，实际 core 变为
`2a1f4002…`，并非设备运算发生改变。CPU重建同时精确复现原/新core哈希；
20份运行时PTX逐字节相同。DualFFN两个SM的11,776个机器码字及全部资源行
相同，仅两个匿名函数符号头变化；不声称COFF/库完整字节相同。主机新指纹
`1cb915b8…` 与三文件完整字节哈希一致，实验计划绑定实际新core及host身份。
保留初次core预期、路径分隔符、完整SASS文本比较三项前置断言失败与修正记录。

新Cargo运行时B14/S2、16局面/双重复/全部补齐行共286,048个原始输出元素
逐位一致。真实Worker W1与W64各128例：11字段原门、top1 128/128、完整
请求/结果protobuf重建、身份及Drain均通过。W1确认为128行/128批。W64
原脚本因参数上限32在启动Worker前拒绝；仅隔离helper放到64，全部非main
函数AST和数值门保持，W1未重跑。共2个实际Worker进程，无失败GPU进程；
这些检查不证明图捕获/rank=time或普遍冷启动可靠性，也不证明库内部竞态根因。

唯一B14/S2/W80/N1000完整前向ABBA对照正式CLI与隔离候选，17项tactic相同。
全部8,000项event与共同时轴独立重算：event **1283.293323→1279.143137（-0.323401%）**；
主机 **1274.963074→1272.516960（-0.191858%）**。
所有臂spread≤1.064%，均稳定但未达两项≥1%门，因此关闭性能候选，
不测Worker ABBA、不重试/缩放/组合救回、不上线。正确性成果作为隔离证据保留。

下一步只读整理当前Fork实际选中路径、Rust最新热点及已关闭候选的对应清单，
明确还有哪些未验证的数据流差异值得实现；没有证据的方向不启动新GPU测试。
42项生产保护、正式CLI/计划、已验收Worker **1274.991675 RPC/s**及历史Fork
比例保持。总目标active；详见本阶段 `assessment.json`、`forward-win-r1/review.json`、
`worker-review.json` 与 `next-gap-coverage-intake.json`。


**2026-09-15 Fork/Rust 覆盖核对与首次成功的管理员 FFN 计数器诊断。**

`fork-rust-mechanism-coverage-r1` 对照当前已认证 B14/S2 剖析中的29种角色、
每前向302个kernel，关联52个实验阶段。目录索引188项，其中109项找到直接
裁决；未找到直接裁决不等于失败。布局、融合、精度和已关闭变体逐项归档后，
尚无证据充分的新kernel设计；这不表示算法已经穷尽或达到硬件上限。
Fork源文件、实际B14 plan及364个外部依赖再次哈希核验通过。除了QK、linear2
和FFN第一投影，Fork实际输出投影与外层投影也使用FP16累加，内部权重还有
通道缩放变换。同压缩模型SHA不能当作相同内部权重位模式或同精度性能。

用户明确授权**一次UAC管理员诊断**后，NCU2026.2.1成功采集当前正式CLI、
同TF3模型和17项认证tactic的首FFN；原ERR_NVGPUCTRPERM未再出现。
B14/S1空盘、3次完整预热、跳过99个匹配launch，仅采集第四前向的首FFN。
这是**1个逻辑kernel、23次计数器replay pass**，未改驱动/时钟/永久权限。
raw报告1004项字段、SASS源视图2992条指令，核对到grid80×9、128线程、
236寄存器、49152B动态共享与0B静态共享，FP32 HMMA及当前build/plan完全对应。
源视图省略reuse/control编码；2992条opcode/operand在限定显示差异归一后相同，
不声称重新逐机器字验证。初次只读导出拒绝`section-folder`参数，修正后只导入
现存报告；没有额外GPU运行。初次SASS显示差异记录也完整保留。

| 本次单kernel诊断 | 观测值与解释 |
|---|---|
| Tensor管线 | elapsed口径79.33%，active口径85.14%；不是整网吞吐或硬件上限 |
| DRAM / L2吞吐 | 峰值口径1.08% / 37.13%；此样本不支持显存带宽为首要瓶颈 |
| 驻留资源 | 寄存器和共享内存各自限制2 CTA/SM；理论8 warp，实际占用率15.90%；只降低一种资源不能保证多驻留 |
| 发射 | 每scheduler平均0.24个eligible warp，issue-active21.56%；多周期Tensor指令下不能把它直接当作79%闲置 |
| math-pipe停顿采样 | not-issued2762项，其中2759项（99.89%）在`HMMA.16816.F32`；采样数不是可消除时间 |
| 共享额外wavefront | 835671 / 4748226（17.60%），全部落在主循环`LDGSTS.E.BYPASS.LTC128B.128`；LDSM/LDS/STS未记录额外项，不能误归因到输出epilogue |

共享访问记录定位了异步拷贝，尚未独立证明具体bank映射根因；报告中的
16.4%/20.67%估算和118.528us采集耗时均不是实测收益。缓存和时钟未固定，
单流kernel replay也不能代表S2竞争或Worker性能。launch stack配置1024B
不等于本地spill；报告local/shared spilling requests均为0。

下一步仅据这份现存报告做CPU侧异步拷贝lane/地址映射与依赖等待定位，先与
52项历史核对，再决定是否有独立设计可进入数值和ABBA门。已关闭TMA、tile、
SiLU/LUT、Lt缓存、三流和布局项不因诊断而重开；不启用fast_math或降低精度。
本轮无候选性能ABBA、无新整网数值认证、无正式代码/计划/二进制变更。
已认证Worker **1274.991675 RPC/s**及历史Fork比值保持原测量语境；目标active。
完整证据位于本阶段`counter-review.json`、`counter-source-analysis.json`、
`mechanism-map.json`和`counter-capture-r1/first-ffn-b14.ncu-rep`。


**2026-09-15 FFN地址与依赖审计完成：形成精确零值SiLU候选，尚未GPU验证。**

`ffn-copy-address-audit-r1`复核当前42份正式产物及上一轮NCU报告。六个生产
CUTLASS global/shared迭代器通过`std::is_same`编译期证明；纯MSVC CPU程序
不链接CUDA运行时，独立Python地址式逐项核对301056次16B拷贝、2408448个
half地址。A覆盖40个tile、B覆盖18个tile、各12个K步；297888次有效拷贝及
3168次尾部无效拷贝均匹配。每个tile/step的逻辑覆盖无重叠或遗漏。
明确的32个4B bank、每8-lane/128B phase模型下最大重数为1；完整warp的
源访问占8条128B全局线，目标占4条128B共享线。它说明缓存线分组不同，
尚不能证明SM120 LDGSTS内部事务或NCU额外wavefront的具体根因。两处PC
`0x1ec0/0x1ff0`有非零predicated执行却无共享wavefront，进一步限制逐PC归因。
没有发现正式地址错误，不据此重开TMA、旧layout或tile变体。

依赖定位找到64个调用同一精确除法helper的静态位置。helper范围
`0xb370–0xb9f0`占1161/3156（36.79%）的not-issued wait采样；
其带符号零返回路径有1482873次predicated线程执行。这不是有效输出发生率：
采样包含padded无效行，也不能换算成可节省时间。该路径同时包括零分子和
无限分母，不能全部称为输入g等于0。

独立检查已有原表达式的65536种half编码GPU黄金表及规范化索引，确认
**9846种有限输入**的SiLU严格为带符号零：±0，以及half范围
`0xd58c..0xfbff`（−88.75至−65504）。前一编码−88.6875仍非零。
exp(88.75)比FP32 RN溢出阈值高约2.753%，这与既有黄金表的整段结果一致。
因此隔离候选只对这个精确零值类直接返回原符号零；其他输入仍执行原
`g/(1+expf(-g))`，保留g/u的原half规范化、FP32累加、原`u*s`顺序与
RN-half输出，包括±Inf/NaN路径。它不加载查表数据，未复用旧LUT性能结论。

新候选已在隔离目录编译成CUBIN。新基线2992条SASS的opcode/operand及
第一机器字与正式绑定参考全部一致；候选3784条，前605条（覆盖完整MMA和
drain barrier）保持，64处FP32 HMMA文本/顺序相同。两侧236寄存器、0 stack/
spill；epilogue新增条件，设置寄存器开始不同是预期变化。尚未重新核全部
control字，也未证明设备数值或性能；额外792条静态指令可能带来退步。
CPU构建/命令、缺少整数min、原harness遗漏global iterator++、以及过宽的
epilogue前缀检查均已修正并保留失败证据；正式源码未改。

下一步为这个**独立、默认关闭**的精确零值候选登记设备数值/路径门，先覆盖
全部gate half编码与up特殊值、真实完整FFN，再决定唯一算子ABBA是否可运行。
之后仍须整网/Worker数值、完整前向和真实Worker独立ABBA及构建指纹认证。
本阶段GPU kernel发射0、性能ABBA0；正式Worker1274.991675 RPC/s及既有
Fork参考不更新，目标active。证据：`address-review-r2.json`、
`dependency-review.json`、`zero-class-codegen-review.json`。

**2026-09-15 精确零值 SiLU：算子有收益；整网接入因对照组身份校验失败关闭。**

本阶段独立目录为 `target/fork-parity-20260908/ffn-exact-zero-class-r1/`
与 `ffn-exact-zero-class-integration-r1/`。原 FP32 HMMA、精确 expf 路径和
RN-half 边界保留；仅 gate half 的 ±0 与 0xd58c..0xfbff 有限负值直接
给出原 SiLU 的符号零，无查表。真实 GPU 对拍覆盖 2,097,152 组输入编码，
以及真实首 FFN/零/0.125/交替 ±0.0625/逐行脉冲五组完整算子输入。
两条独立流、不同输出填充值、重复候选与原算子、越界哨兵和只读检查均通过。
两内核仍为 236 寄存器、无 local/spill、49,152 字节动态共享内存，64 处
FP32 HMMA；实际链接的原/候选指令与冻结算子版本一致。

唯一预登记算子 B14/S2、W80/N1000 ABBA：event 吞吐 **+4.639414%**，
统一主机计时吞吐 **+5.225144%**；event 两臂 spread 0.424263%/0.008622%，
主机两臂 1.184159%/0.049890%，均在 5% 内。8000 事件和计时前后输出已重算。
这是完整首 FFN 算子的收益，不能当成整网、Worker 或 Fork/Rust 比例。

隔离 Rust 接入默认关闭、只允许 B14；其他 batch 回退原内核，开关经
`tactic_var` 读取，schema2 与新构建指纹绑定。37 项计划/选项 CPU 测试通过。
首轮 baseline/candidate × B1/B14/B16 的六组全网运行成功，612,960 个原始
FP32 输出值全部与当前认证版逐位一致，B1/B16 回退和 B14 实际路径已核验。
原生测试每进程只有一次 pass；此前 harness 错将旧扩展测试的 repeats 元数据
用于原生测试，已保留 CPU 检查错误并明确第二 pass 使用新进程、独立 context。

预登记的第二 pass 在 **baseline B14、候选关闭** 时失败：原残差 GEMM
(5054,384,1152) 的 filtered heuristic index 1 未通过完整算法身份校验。
opaque 对象 u64[4] 由预期 `0x0001fe2800000001` 变成
`0x0001fe288b9f0001`，仅 LE 字节偏移 34、35 不同。该事实不能证明这些字节
是 padding、可以忽略，或代表算法数值变化；也不能外推为已安装生产 CLI 失败。
失败发生在隔离新 CLI 对照路径。第二 pass 共完成 B1、B14 失败即停止，
没有重试、没有 Worker 黄金运行、没有整网/Worker 性能 ABBA；完整准入未通过。

本次隔离接入阶段已关闭，算子正收益仅作为其历史证据，不恢复此阶段重采样。
下一步针对保留的原残差身份错误，结合既有 public-constructor 审计做独立根因
检查，不静默屏蔽字节、不改变精度。原 42 个生产保护文件保持相同；已认证
Worker **1274.991675 RPC/s** 与 Fork 参考不更新。目标仍 active，尚未完成。
证据：算子 `numeric-review.json`/`decision.json`、接入 `build-review.json`、
`native-review.json`、`native-repeat-r1/baseline-b14.log` 和 `decision.json`。

**2026-09-15 残差算法身份故障：已控制复现栈预填字节传播，形成按需重建原型。**

新独立目录 `target/fork-parity-20260908/residual-identity-origin-r1/`。
上一阶段 SiLU 接入对照组在 opaque 字节 34、35 处失败。本轮核实生产
启发式结果数组本来已清零；此前一次同类变化的 heuristic state 也是成功。
因此不把问题归为 Rust 忘记初始化数组，也不重跑已关闭的 SiLU/NN 方案。

新主机元数据实验使用原 (5054,384,1152) TN/half A/B/FP32 compute及C/D、
32 MiB workspace、原 workspace-filtered index1。在进程默认 cache8192 和
仅该进程关闭 cache 两种条件下，分别交叉栈预填 00/5a/a5、输出缓冲区预填
00/3c，每格16次，共192次查询/1536条成功返回记录。所选对象的字节34、35
每次都随栈预填变成0000/5a5a/a5a5；输出缓冲区预填不影响它，关闭cache
也不消除它。其余62字节逐位等于认证对象，九项公开配置始终为
21/15/1/0/0/0/12/0/0。未调用 Matmul/GEMM，未运行性能测量。

实际编译对象反汇编确认：query自己的结果区为其RSP+[0xd0,0x3cf]；
salt子调用只写RSP+[-0x10010,-0x11]，两区不相交，且调用顺序是结果区
memset→salt→GetHeuristic。观测证明调用前栈预填能控制这两个 opaque
字节，具体厂商DLL内部的拷贝指令尚未定位；这不构成屏蔽私有字节或直接
执行修改过对象的依据。[NVIDIA文档](https://docs.nvidia.com/cuda/cublas/)
将算法对象定义为opaque，并提供同版本对象序列化及公开初始化/配置API。

隔离的按需重建原型保留原64字节精确匹配快速返回；只在不匹配且明确允许
修复时，检查成功状态/原三个shape-index范围/九项公开配置，调用AlgoInit
和ConfigSet生成新对象，再读回配置并要求**全部64字节精确匹配**才返回。
582组检查通过：192份观测对象×3个残差形状，加本次失败对象×3，以及
关闭修复时的3个精确对象。195组快速路径不调用任何重建API；387组恢复
完整认证对象，每次28个版本/配置/初始化API调用；82个负例拒绝错误配置、
状态/shape/index，以及重建结果任意一个字节的改动（包括34、35）。输入
只读、输出前后哨兵、失败输出不写均通过；实际加载cuBLASLt DLL路径及SHA
已记录。此原型尚未接入Rust，也没有整网/Worker或吞吐认证。

下一步独立接入**仅身份不匹配时重建**的基线可靠性修复。必须保留原索引、
TN/beta1、FP32与half边界、型号/库/构建绑定，补齐成功heuristic状态传递，
确保普通缓存/精确匹配路径不增加重建调用；新Rust完整数值/路径/负例及
预登记ABBA之后才能决定迁移。旧SiLU及NN组合阶段保持关闭，不能据本轮
元数据结果宣称其性能已认证。生产42文件未变，Worker1274.991675 RPC/s与
Fork参考不更新，目标仍active。证据：`review.json`、`host-codegen-review.json`、
`repair-review.json`与`repair-raw.json`。

**2026-09-15 按需重建已通过 Rust 整网及 Worker 数值门；同 CLI 性能对照在启动阶段失败，未上线。**

独立目录 `target/fork-parity-20260908/residual-canonical-rust-r1/`。
隔离 Rust 实现新增默认关闭的 `KATAGO_CUDA_RESIDUAL_CANONICAL_R1`：仅原三个
固定残差 shape/index、TN/beta1、成功 heuristic 状态且完整64字节不匹配时，
经九项公开配置读取/初始化/设置重建对象，再要求全部64字节精确匹配。
已匹配对象与缓存命中不增加重建 API；保持 FP32 计算和原 half 存储边界。
新源码指纹绑定实现，计划必须 schema2、显式绑定新开关及依赖，环境不能
扩展已安装计划的认证。初版遗漏该环境边界，已在完整模型运行前修正；
初版源码/二进制和日志保留。38项计划测试、真实 Rust ABI 12组检查通过。
实际链接的原 DualFFN 2992条指令与认证算子逐项相同（地址、操作数与第一
机器字；未声称校验第二控制字），未加入旧 SiLU 候选或新 device kernel。

8组原生整网：关闭/开启修复各 B1/B14/B16，另加开启修复并注入原失败
字节8b9f的 B14/B16；每组2 lane、每 lane16局、每进程一次 pass。
837,712个原始 FP32 输出值与当前认证 QKV 版本逐位一致；输入、padding、
独立lane资源、实际指纹与选择标记全部核实。测试只修改选中对象的局部
副本，Matmul只能使用新建且全64字节认证相同的对象。

实际 CLI 的15个拒绝测试通过；原17项计划的第16项（plan0/env1）在
B14 warmup触发旧身份错误，u64[4]=`0x0001fe28e4ed0001`，因此原计划
整体没有通过，也没有重跑该失败对照。随后独立登记两次候选加载：自然
路径及原先尚未执行的强制恢复路径，均在plan1/env0下成功。此处自然
加载成功不能当作自然发生故障后恢复的证据；强制路径明确触发重建。

Worker 另行预登记：关闭/开启/强制恢复 × W1/W64 共6组，每组128局。
实际6组全部通过11字段FP32黄金门、top-1 128/128及Drain；768对请求/
响应 protobuf 已逐条重建并重算数值门。计划对抗环境变量、真实窗口64、
W1的128 rows/128 batches已核实。启动B14标记不被当作W1请求批次证据。

可靠性评估单独预登记一次同CLI ABBA，B14/S2、80 warmup/1000 iterations；
event与统一主机时间均要求不回退且每臂spread≤5%，性能优化的≥1%门保持。
首个baseline进程第一lane通过原身份校验，但第二lane初始化失败：
`0x0001fe28f4df0001`。该进程无完整nnbench性能JSON；后续B/B/A未执行，
未跑Worker性能，不重试本次ABBA。故没有有效的配对收益或吞吐认证。

当前修复保留为**数值已验证、部署性能未确定**的隔离产物。下一步须独立
评估以现有已认证CLI作为部署对照的可行性，明确旧/新二进制与绑定差异，
新协议先登记，所有失败对照继续保留；不能推称新开关速度收益或静默重试
本轮同CLI测量。生产42文件未变，Worker仍为1274.991675 RPC/s，Fork参考
不更新，旧SiLU/NN关闭项不恢复。证据：`build-review.json`、`native-review.json`、
`binding-r1/failure.json`、`binding-recovery-r1/review.json`、
`worker-golden-r1/review.json`、`forward-r1/failure.json`及`decision.json`。

**2026-09-15 修复部署对照为稳定回退，关闭；发现既有 half 边界后的 FFN 全零通道。**

`residual-canonical-deployment-r1` 独立比较当前已认证CLI（9e0c6d…）与
隔离修复CLI（fd8d4a…），各用自己的精确构建/计划指纹，模型/设备/输入/
B14/S2/W80/N1000保持相同。原同CLI关闭开关的失败对照未重试。
本次四进程完整结束，8000个原始event及统一主机计时重算通过：
event几何均值1270.4533→1265.2833（**−0.406939%**），主机
1269.1032→1262.1935（**−0.544453%**）。两臂event spread为0.232%/0.067%，
主机0.145%/0.370%，均稳定。可靠性部署的“不回退”门未过，性能优化≥1%
门也未过，故关闭本次部署评估，不运行Worker性能、不安装、不重试或组合
救回旧修复/SiLU/NN方案。这是旧/新可执行文件的配对结果，不能单独归因为
重建API；新开关缓存命中本来也不调用该API。

随后只读检查现有完整模型首FFN下投影fixture，发现1152个隐藏通道中289
个权重列在half表示中全零，输入激活对应通道也全零。全模型原始FP32审计
确认：这些通道在FP32文件中并非全零，不能据此直接剪原始FP32权重。
关键是**现有上传边界已将它们精确舍入成FP16的正/负零**，本候选不新增
量化或改变存储精度。

提取未改动的生产Rust `f32_to_f16_bits` 编译成CPU核验器，对99个FFN矩阵
共43,794,432个原始权重逐值转换，全部位模式与独立NumPy half转换相同。
native lowering只转置这些FFN矩阵、上传按既有函数转half；首下投影的完整
布局字节另与已保存的实际权重fixture相等。33个FFN内，gate/up/down的
整通道half零掩码均三者完全相同：每层48至1059个，共16,529个全零通道。
按16对齐可删除16,304个，其余225个零通道保留作为对齐，所有保留索引
仍按原顺序排列。

新独立目录 `ffn-half-channel-compact-r1` 已完成33层权重打包和CPU映射证明：
所有删除项在三个矩阵里都已是half零，所有保留位模式（含有符号零）原样
复制，隐藏总宽38016→21712。对应FFN三次GEMM的逻辑乘加量可减少
**42.887205%**；这只是静态工作量，尚无新算子或整网性能结果。
还未执行紧凑算子。压缩后的下投影K分组会改变，不能预先声称FP32输出
逐位相同，必须实际验证全部精度门。

下一步从当前生产源码实现独立紧凑FFN算子：保持原DualGemm瓦片/布局/
stages/精确epilogue，动态隐藏宽度；初始化缓存必须同时区分tokens和宽度。
保留FP32累加、精确SiLU及两次RN half边界，先完整算子/宽度/guard数值，
再原生整网和Worker，之后才预登记唯一ABBA。旧修复及SiLU候选不混入。
生产42文件未变，认证Worker仍为1274.991675 RPC/s，Fork参考不更新。
证据：部署 `forward-r1/review.json`、`decision.json`、
`ffn-channels-review.json`、`half-channels-review.json`，及新目录
`packing-contract.json`、`packing-review.json`。

**2026-09-15 既有 half 全零 FFN 通道压缩：整网/Worker稳定通过，已安装。**

ffn-half-channel-integration-r1 将33层FFN总隐藏宽38016压到21712。
只删除既有FP32→RN half上传后，gate/up/down均已整通道为正/负零的项；
保留索引升序、所有保留位模式、原FP32累加/精确SiLU/half边界，未新增量化。
设备端仍为原DualGemm的2992条指令，主机参数增加动态隐藏宽，缓存同时
区分tokens和宽度。down按实际压缩K计算，保持FP32累加及残差输出；
B14下不再匹配原K1152 Fixed形状，实际使用相应K的Lt选择。

独立算子34个完整FFN用例通过：34,160,320个隐藏half值保留位模式精确相同，
14,001,408个FP32下投影值通过完整FP64点积门。算子唯一ABBA的候选
event spread35.43%、主机5.002%超门，结果原样保留、未重测，未认证其增幅。
随后整网按独立预登记协议验证。六组开/关×B1/B14/B16双流共612,960个原始
输出值通过原门；对照组逐位等于原生产，候选六个policy通道top1全相同。
候选输出并非逐位相同：最大policy差0.014199、value0.011452、misc0.001324、
moremisc0.002817、ownership0.001719。真实Worker开/关×W1/W64共512请求
通过C++ FP32的11字段及每组128/128 top1；原始protobuf与请求身份复核，
Drain全部正常。

唯一完整前向ABBA固定B14/S2/W80/N1000，8000个原始event：
event几何均值1269.1044→1572.9360（**+23.940638%**），统一主机
1264.6357→1555.6042（**+23.008092%**）。两臂两指标spread均<0.8%。
随后唯一uncached Worker ABBA固定B14/S2/C64，每轮预热256/计时8192，
共32,768条原始计时：**1270.2209→1546.7076 RPC/s（+21.766825%）**。
两臂spread0.1161%/0.6745%，实际平均batch均约13.9083，全部满足稳定且
至少1%的门。此值更新当前认证吞吐，历史1274.991675保留；不同轮局部
收益不能相加，也不能用Windows Worker数值除Fork历史event数值。

KATAGO_CUDA_FFN_COMPACT_R1默认0，经plan>env>默认读取。开启绑定模型
SHA、99个原half矩阵/33个打包映射、目标GPU/Lt与DualFFN/CUBLASLt/TN/
NOGRAPH/FUSION依赖；独立ffn_compact_artifact绑定源码和映射。37个plan
单测、2个打包测试、16个真实CLI负例及4个Worker反向环境正例通过。
旧q64/q128的B1/B16四组ORT和两组graph回归、旧默认/C32的W1/W32四组
Worker均通过。六计划隔离加载后，安装6个源码文件、原样新CLI、6份
Windows计划与吞吐配置注释。最终六次实际路径加载、吞吐W64的128请求/
11字段/top1/Drain全部通过。只有throughput启用，其他五份显式0，WSL不变。

认证CLI SHA：848e3f9ce4ae7ce5cc7571c0589b3f1479885163e857b9dc06ebe18386cfe62b。
FFN artifact：sha256:83225c2c7e33e99ece3caea9bbfcbf5356c088d00bcb2b0e49b242158ccacc03。
证据在 target/fork-parity-20260908/ffn-half-channel-integration-r1 的
native-review.json、worker-golden-r1/review.json、forward-r1/review.json、
worker-performance-r1/review.json、promotion-r1/acceptance.json。
兼容构建的invocation继承了错误cwd字段，实际在隔离source构建；原记录
保留，附 migration-r1/build-tests/cwd-correction.json，无重建或GPU重采样。

下一步重新审计压缩后的剩余热点，再选择独立候选。旧修复/SiLU/NN失败
方案未混入。新Windows路径尚无同环境Fork复测，总目标继续。

**2026-09-15 压缩后热点复核；compact down 的 Lt21 候选关闭。**

compact-forward-profile-r1 对当前已安装848e3f9c构建采集两份普通用户Nsight
Systems时间线。一份含cuBLAS/API细节与event完成跟踪；后者产生可能干扰
跨流依赖的警告，因此补充关闭event完成跟踪的CUDA轻量时间线，无该警告。
两份各48次前向、每次302个实际内核，全部符号/网格/资源序列一致；剔除
每流4次预热后，各保留40次前向、12080条内核记录。33层权重映射、23种
实际执行宽度、模型SHA/构建/tactic均匹配当前认证版本。

轻量记录按两流内核持续时间之和归一：QKV/RoPE 16.919%、
Attention核心 15.337%、输出投影 8.014%，三段合计 **40.269%**；
FFN up/SiLU 20.047%、down 14.763%，合计 **34.810%**。
重叠内核分别计入分母，所以这不是互斥墙钟占比或硬件利用率，也不是新的
吞吐认证。单流逐层同步事件的最后三帧为14.681/14.489/14.527ms，仅作
层级诊断，不能与双流CUDA activity耗时混算。没有新增管理员计数器采集。

另外用未变的Rust RN-half编码器审计132个Attention矩阵，共19,464,192
元素，与独立NumPy编码逐位相同。Q/K各有174个全零通道，但没有全零整头；
V及输出投影没有全零整头或对应全零通道。整头删除候选为0，不迁移FFN
的压缩机制到Attention，不改RoPE头序/频率、维数或精度。

23种compact K的top8 Lt只读查询共184个候选均通过公开AlgoCheck。
独立compact-down-lt21-r1只在K>=432的22层换为已查询的id21/tile15/
stages12，其他11层和所有FP32累加/残差/输出不变。34组算子包含首层真实
B14 hidden及33层代表性31行hidden重复到5054行；后者不是整网逐层输入。
所有131,970,048个输出通过完整FP64门，34组候选与基线逐位相同，双流
逐位/guard/只读检查通过。

唯一33层down序列ABBA（B14/S2、80预热/1000迭代、8000事件）结果：
event序列吞吐1046.4196→973.8469（**−6.935332%**），两臂spread
0.263881%/0.490406%；含33次残差重置的主机吞吐543.5647→558.4189
（+2.732748%），两臂也稳定。预登记要求两个指标均至少+1%，因此拒绝。
528份计时前后完整输出均与已过门数值逐位一致；不重试、不整网接入，
不把主机局部正收益写成完整前向或Worker提速。

当前44个生产保护文件保持原SHA，认证Worker仍为1546.707634 RPC/s。
Fork历史event参考未更新，目标继续；下一步回到当前QKV/Attention的数据
访问与实际执行布局，登记独立候选前先检查结构及精确边界，不复活已关闭项。
证据：compact-forward-profile-r1/profile-review.json、actual-path-review.json、
attention-half-r1/review.json、compact-lt-query-r1/review.json，以及
compact-down-lt21-r1/numeric-review.json、timing-r1/review.json、decision.json。

**2026-09-15 QKV两级流水的两个独立实现均稳定退步，关闭。**

在FFN压缩后的当前版本上，QKV/RoPE累计内核耗时占比为16.919%，
Attention三段合计40.269%；该剖析口径按两流内核时长之和计算，不是
互斥墙钟占比。已关闭的M64/vector4/OnlyAlphaScaling等候选保持关闭。

qkv-stage2-pipeline-r1先只改变Gemm模板的stage3→2，维持原CTA128×64×32、
warp64×32、40×18网格、128线程及原scalar RoPE/Default scaling。
实际编译却选择MmaPipelined：原cp.async变为寄存器中转的全局加载/共享
写入。因此它不是仅改变异步缓冲数量的实现。寄存器146→167，共享36→
24KiB，潜在驻留2→3 CTA/SM、8→12 warp，无spill；实际占用率尚未测量。

qkv-async-stage2-r1是独立的显式MmaMultistage实现，复用原global/shared
迭代器、cache操作、warp MMA与policy，只令其异步stage3→2；实际仍为
cp.async。寄存器146→150，共享36→24KiB，潜在驻留同样2→3 CTA，无spill。
它未复用上一同步候选的MmaPipelined，也未组合其他已关闭改动。

两项基线完整SASS均复现当前生产CUBIN。每项32个FP32 MMA站点、原FP32
RoPE及两次RN-half保留；整段scalar迭代器源码（除类型/流水绑定）相同。
编译导出的ThreadMap逐字节相同，全矩阵5,822,208输出元素覆盖一次，
485,184个Q/K向量及242,592个V向量、66行尾部谓词正确。两项实际C++
400字节Params、六指针偏移和模板均与生产逐字节一致。

先从当前已安装CLI848e3f9c、启用compact的完整模型重新捕获首Attention
RMS输入及旋转后的packed QKV。模型/构建/18项tactic和特征SHA匹配，
CPU重导learned RoPE表与原字节相同。首Attention位于首FFN之前，重新
捕获的两份张量也与历史冻结数据逐位相同。异步候选复用此次捕获，不声称
再次捕获。首次采集前校验误用完整dict比较路径斜杠格式，所有文件SHA
实际相同；只修正CPU路径规范化，保留原脚本与失败记录，失败时无GPU运行。

每项六份完整输出、34,933,248个half值与当前模型参考逐位相同，两流
独立输入/输出、poison、guard与只读门通过。两项各自唯一B14/S2、
80预热/1000次ABBA各复核8000事件和16份前后完整输出：

| 独立候选 | event吞吐变化 | 每流主机计时吞吐变化 | 稳定性 |
|---|---|---|---|
| 默认两级MmaPipelined | −4.186439% | −4.474297% | 最大spread0.503709% |
| 显式两级cp.async | −1.887764% | −0.710852% | 最大spread0.383772% |

两项均稳定但不满足两个指标至少+1%的门，关闭、不整网接入、不跑Worker
性能、不重试或组合。潜在驻留提高不是加速证据；不能用它抵销实际回退，
也不能据此宣称QKV已到硬件上限。

当前44个生产保护文件及认证Worker1546.707634 RPC/s保持；Fork历史参考
未更新，整体目标继续。新qkv-current-counter-r1已准备当前首QKV的单内核
NCU诊断：B14/S1、3次完整预热后跳过99个匹配launch、采第四前向首QKV，
不改时钟/驱动/永久权限。普通用户尝试新报ERR_NVGPUCTRPERM，没有报告；
此前一次管理员FFN授权已用完，QKV管理员执行单独等待新授权。

证据：两个候选目录的codegen-review.json、operands/manifest.json（异步
目录复用当前捕获）、numeric-review.json、decision-stage1.json、closure.json；
诊断目录proposal.json与ordinary-capture-r1/completion.json。

**2026-09-15 其他矩阵精确half零通道审计完成；QKV有界索引候选低于收益门，关闭。**

outer-head-half-audit-r1用当前44项保护文件、同SHA模型及未改动的Rust
RN-half编码函数，核对166个非FFN矩阵、26,567,424个half值；每一位
与独立NumPy转换相同。22个外层投影和11个原始学习头部矩阵均没有
整零输入/输出通道，不能复制此前FFN压缩收益。原生头部兼容行、对齐
padding及FP32全局投影均未冒充新的零通道。205组其他原始FP32数组仅作
只读检查，BN均值为零不等于门控输出为零。

首层卷积只有原输入通道8在既有half边界后全零：理论上可由K198/pad208
改为K189/pad192，让当前K32循环7→6。现存轻量双流剖析中，首层GEMM与
im2col的累计内核占比分别为0.848%/0.170%；按计算量比例的粗筛仅约
0.1288%累计耗时节省，
明显低于1%门槛，暂不实现。此估算不是独占墙钟比例、严格速度上界或实测
增益。Q/K原half整零通道各174，合计87个共同零旋转对；V/out无整零
通道，仍没有可删除的完整head。不改RoPE频率/配对、32维缩放或RMS分母。

qkv-rope-bounded-index-r1是另一项独立改动：依据已编译ThreadMap证明
Q/K pair起始列限定0..766，将`(col+e)%384`替换为一次条件减384。
原CTA128×64×32、stage3、scalar系数读取、Default scaling、FP32 MMA及
原RoPE算式/两次RN-half均保持。完整5,822,208个half位置覆盖一次，
1,940,736个系数地址逐项一致；CPU版及算子版基线完整SASS与当前安装版
相同，算子版候选完整SASS也与CPU审查版相同。静态指令2706→2458，
但资源仍146寄存器/36KiB、无stack/spill，实际Driver潜在驻留仍2CTA/8warp。
不能把静态指令减少9.16%当成运行时收益。

复用上一轮同当前CLI/模型/18项tactic认证的首Attention完整输入和参考，
两条独立live stream、六份输出共34,933,248个half全部逐位相同，poison/
guard/readonly通过。唯一ABBA共8000个event及16份前后完整输出再次通过；
设备计时吞吐仅+0.075529%，每lane主机计时+0.448397%，最大spread
0.630014%，稳定但均低于1%，因此关闭、不重测、不进入整网/Worker。
CPU审查脚本仅在全部门通过、报告写入并复核后，最后console print因缺少
json import报错；原文件及错误保留，独立重读报告全部引用通过，未重跑
编译/映射/GPU。该报告输出错误不作为数值或性能失败隐藏。

生产44项保持，已验收Worker仍1546.707634 RPC/s；没有新Fork实测或比例
更新。QKV硬件计数器诊断仍等待新的单次管理员授权，此前FFN授权不复用。
本阶段普通权限算子测试不涉及计数器或提权。整体目标继续active。
收据：`target/fork-parity-20260908/outer-head-half-audit-r1/assessment.json`、
`target/fork-parity-20260908/qkv-rope-bounded-index-r1/closure.json`。

**2026-09-15 压缩后小 FFN 全隐藏层融合：资源门失败，未进入测速。**

`compact-full-hidden-ffn-r1` 核对当前33层的实际grid：14480个CTA中14040个
执行有效tile，440个越界CTA在矩阵计算前返回，不能计为完整GEMM浪费。
当前N64宽度容量22464、逻辑21712，padding为3.347578%；不是原H1152
全部工作量。选定第9/10/11/27/28/29层，H分别112/128/96/256/112/144，
其FFN两段在现存双流时间线中的累计均值319.067225us，占内核累计时间
2.679305%；该比例包含流间重叠，不是独占墙钟比例或可实现收益。

新原型利用认证压缩后的H，在一个CTA中只产生一次完整hidden，再计算down；
不同于已关闭的H1152 streamed64多块循环。TM32、128线程、两级异步拷贝，
HC128/256仅补第一投影列，down仍按实际H访问；保留原half权重、两次投影
half舍入、精确SiLU和FP32累加/残差。CPU全地址映射、尾行、K顺序与原生
编译导出的6144条half2位置记录一致；这不是GPU数值等价验证。

| 实际H | 正常/诊断版寄存器 | 动态共享B | 正常/诊断版本地B | 理论CTA/SM |
|---|---:|---:|---:|---:|
|96、112|168/168|36864|0/0|2|
|128|206/208|36864|0/0|2|
|144|255/255|69632|24/56|1|
|256|255/255|69632|112/120|1|

ptxas与CUDA函数属性一致；全部10个入口的MMA为FP32累加，精确除法及
RN-half转换仍在。H144/H256未过预登记的“五个尺寸全部无spill”门，
**整项关闭，不事后只挑通过尺寸**。正常版静态GPR存活峰值分别248/244，
位于第一投影主循环LDS/HMMA附近，spill还关联预计算地址；源码中gate/up
先结束、down后创建，仍不足以消除实际生成代码的资源压力。静态liveness
不是动态停顿测量，尚未证明某一地址表达式就是唯一根因。

本轮仅CPU证明、编译/反汇编及零kernel的CUDA资源查询；没有执行候选kernel、
新模型捕获、数值对拍或ABBA。准备脚本空格语法错误和host缺少string头文件
均在GPU执行前修正并保留记录，后者只加host包装头。只读审查初次发现
nvdisasm liveness省略末尾对齐NOP，改为逐项验证确为尾部NOP后再关联；
未补造存活计数，也未重编译或重跑GPU。

正式44项代码/二进制/配置/计划哈希保持，已认证Worker仍为
**1546.707634 RPC/s**，没有新Fork比值。获准的一次FFN管理员采集已完成，
本轮只核验既有报告；独立QKV计数器请求仍待新的授权，不复用FFN授权。
后续可据源码与编译器生成地址存活范围研究独立机制，不能重测此失败版本
或直接缩小样本范围。完整目标保持active，资源失败不等于硬件上限。
证据：本阶段`contract.json`、`shape-audit.json`、`cpu-proof.json`、
`codegen-review.json`与`closure.json`。

**2026-09-15 小 FFN 仿射地址改写：数值及竞态门通过，完整算子 ABBA 明显回退。**

`compact-ffn-affine-address-r1` 在已关闭的全hidden原型上，只改第一投影的
地址表示：固定128线程的row/col范围，以及不改变XOR swizzle的恒定行偏移。
gate/up拷贝次序、无效源地址钳制、共享位置、MMA次序和精度全部保留；
六层范围仍为9/10/11/27/28/29，未事后缩小到容易通过的尺寸。
CPU核对187392条完整拷贝描述、1213440条实际CTA输入描述和114688次
MMA half2地址。源码的同址表达式确实影响了编译结果：H144/H256从255
降为226个寄存器、local/spill归零；H128从206降为168，其他正常入口168。
全部10个入口通过资源门，FP32 MMA及原half/精确激活边界保持；
HC128/256共享仍为36864/69632B，理论驻留仍为2/1 CTA。

从正式CLI、当前18项认证tactic和原TF3 SHA重新采集六层前后共12个边界。
算子基线使用当前compact DualGemm（2992条SASS指令核对）及各实际H的
完整64B原Lt heuristic0；候选直接加载过资源门的同一CUBIN。两条独立流
共检查25714752个half与93155328个FP32值：基线hidden/FP32均逐位匹配
新鲜模型捕获，候选hidden全部逐位一致，所有输出通过原half存储边界后
FP64 down+残差参考门，最大归一误差0.008812，远低于1。不同poison、
DUMP开关及两流输出也逐位一致，完整尾行/guard/readonly通过。
racecheck覆盖全部六层与两种入口，0 errors / 0 warnings；84份完整数组
均与普通数值运行逐位一致。此处是算子数值门，未声称新整网或Worker认证。

唯一预登记ABBA、B14两流、80 warmup/1000 iterations，每次event包住
原顺序的六个完整FFN；相同六次残差D2D复位放在event之前，统一主机计时
包含复位。8000个event全保留，144份前后数组复核通过：

| 六FFN序列口径 | 基线 | 候选 | 收益 | 最大臂间波动 |
|---|---:|---:|---:|---:|
|event 序列/s|4711.019634|2317.936887|−50.797554%|0.738160%|
|统一主机 序列/s|3039.429678|1848.107645|−39.195578%|0.520301%|

这是六层算子序列，不能写成NN rows/s、完整前向或Worker下降。
**候选稳定回退，关闭；不重测、不选子集、不接入正式版本。** 消除spill
没有使这个完整融合调度更快，不能将静态指令/寄存器减少当作性能收益；
本次数据也不独立证明具体某个动态停顿就是全部回退原因。

保留两项工具层修正：捕获校验初次将实际`Ffn`误写为`FFN`，复用已成功
结束的两个捕获后继续其余10个，没有重复它们；racecheck默认连接超时，
确认工具/目标均已退出后改用既有named-pipes连接方式，第二次得到有效
报告，第一次失败记录保留。没有修改内核或数值/性能门，没有管理员操作。

正式44项哈希保持，已认证Worker **1546.707634 RPC/s**。完整对标目标
保持active；本轮完成一个有明确结果的机制验证，不表示所有融合已穷尽。
下一步应优先核对并刷新“当前已安装压缩版本 versus Fork”的同卡、同B14/S2、
同event/统一主机边界前向参考，明确平台与累加精度差异，再选择剩余热点。
旧2357.54 Fork与1546.71 Worker不能相除；本轮未生成新的Fork比值。
独立QKV管理员计数器请求仍待新授权，已用的一次FFN授权不复用。
证据：本阶段`codegen-review.json`、`capture-review.json`、
`numeric-review.json`、`racecheck-review.json`、`timing-r1/review.json`、
`independent-review.json`与`closure.json`。

**2026-09-15 当前 Fork/Rust 前向参考核验：空盘输入逐位对齐，两个计时稳定性门均未通过。**

证据目录 `target/fork-parity-20260908/current-fork-rust-forward-r1`。
Windows正式Rust CLI仍为`848e3f9c…`，原版Fork WSL ELF仍为`5df63305…`，
原TF3压缩模型SHA仍为`1881600c…`；B14、两条流、19路空盘、无graph、
80预热/1000计时迭代。Rust显式传入throughput计划的18项环境开关，JSON
完整build与计划一致，strict Attention、只读QKV、stem、gate、33层compact
映射及23种实际hidden执行标记均核验。两端GPU UUID相同，为同一5070 Ti；
Windows/WSL及CUDA13.3/13.0差异保留。桌面GPU程序未关闭，进程枚举
有权限缺口，不声称完全独占GPU。

为观察Fork主机计时，制作了隔离副本，仅替换`cudabackend.cpp`一个主机
对象，原331个链接对象均核验；全部CUDA/AOT对象复用，内嵌`.nv_fatbin`
逐字节一致。副本只在benchmark边界记录每lane开始/最终同步后的时间，
原timed apply循环原样保留；输入与已有raw event在计时外保存。
两个真实GTP局面×8对称的全部显示字段与原版相同，属于打印精度检查，
不冒充完整FP32输出逐位对拍。空盘B14/S2/W1/N1诊断捕获中，Fork NHWC
输入转NCHW后，两lane的FP32特征与正式Rust输入**逐字节相同**：spatial
`6a8a3ff7…`、global `56d4b091…`。这补齐当前受控空盘的输入身份，不能
外推所有局面。副本binary为`6bdb69c8…`，不是原认证ELF；原plan未重绑。

原版/副本/副本/原版唯一ABBA预定：两种统计变化绝对值≤1%，各臂波动≤5%。
event中位数速率和原版2337.060107、副本2002.973985（−14.295145%），
副本两次波动33.522404%，**计时副本未获准用于比较**。同原版legacy墙钟
口径的变化+0.641120%、最大波动1.757618%，单项通过不改变整体失败裁决。
副本两次统一主机吞吐1555.059188/1591.465851 rows/s与event统计明显不同；
现有记录不足以把差异归因为计时器、队列调度或后台负载中的某一项。
不修改门槛，不重跑筛选，不用该副本发布Fork/Rust主机吞吐比例。

随后独立登记**两个未修改原版引擎**的event-only Fork/Rust/Rust/Fork ABBA，
同B14/S2/W80/N1000，5%稳定性门；不复用副本性能样本，也不比较墙钟：

| 原版CUDA event中位数速率和（名义NN rows/s） | 第一次 | 第二次 | 几何均值 | 臂间波动 |
|---|---:|---:|---:|---:|
| Fork WSL |2177.959205|2311.883693|2243.922541|6.149082%|
| 当前Rust Windows |1567.914015|1579.691520|1573.791750|0.751158%|

Fork超过预定5%门，**本轮没有新的稳定Fork/Rust比值**；日志中的诊断比值
不作验收或收益。Rust两次4000个原始event全保留，Fork原版只公开两lane
中位数；独立重算一致。统计是`sum(B/median(event))`，不是完整主机吞吐，
更不是Worker RPC/s。原版Fork测量参数/source未改，不以副本失败否定原版
的输入身份；原版比较因自身稳定性失败单独关闭。

保留一次校验修正：脚本最初误期望每层上传两次，实际Rust在启动lane前
加载一个`Arc<CudaModel>`共享权重，故33层各上传一次，stream/workspace
独立。依据受保护源码更正为一次；复用已成功结束的前两次测量，仅执行
未开始的后两次，未重复生成或挑选样本，计时参数和性能门均未改变。

工具副作用如实记录：GNU objcopy提取段时未指定独立输出，重写了原版Fork
ELF，**SHA/长度不变，但mtime更新**；不伪造旧时间戳。以后仅操作副本或
指定不同输出。本轮最后364项WSL源码/对象/运行时内容哈希复核通过；
Rust正式44项内容保持不变。上述“未修改原版”指可执行内容，非全部元数据。

Rust仍是FP16存储/FP32累加与关键计算；Fork QK、FFN与多个投影含FP16累加，
PV为FP32，外层权重还有缩放变换。同模型原文件SHA及同输入不等于同内部
算术。已认证Worker **1546.707634 RPC/s**保留；本轮没有新优化安装、Worker
测试或管理员采集。一次FFN管理员授权已用，独立QKV请求仍待其自身授权。
完整目标保持active。下一步先审计原版Fork可直接报告的actualWallSeconds与
Rust源代码中预热/线程/事件/收尾范围，形成真正匹配的主机计时方案；
不得把本轮失败的副本事后放行，或在相同条件下反复跑上述ABBA选好结果。
证据：`observer-functional-review.json`、`observer-neutrality-r1/review.json`、
`original-event-reference-r1/review.json`、`independent-review.json`、`closure.json`。

**2026-09-15 统一前向测试流程：共同墙钟参考通过，当前Rust约为Fork测试副本的98%。**

证据目录 `target/fork-parity-20260908/matched-forward-harness-r1`。
源码审计先确认：原版Fork的`actualWallSeconds`在compute handle与一次完整
getOutput准备好后开始，包含输入上传、显式预热、事件分配/读出/销毁、
handle和stream释放，直到线程join结束；Rust现有wall只记录每lane计时
循环开始和最终stream同步结束的端点。原版墙钟不能直接与Rust相除。

本阶段采用**共同测试流程**，没有将上一阶段失败的观察副本事后放行：
从原版Fork重新编译一个主机对象，增加两个端点记录，并在最终同步后、
读取任何event前增加完成屏障。所有lane完成后才允许读event和释放资源，
与现有Rust的done barrier一致。此屏障**有意改变早结束lane的收尾调度**，
不声称性能中立；以下是“统一测试流程下Fork”参考，不是原认证ELF原行为
的实测，也不替换原版Fork历史2418.9/2357.54。此前两个失败ABBA仍失败。

Fork新测试ELF SHA `78da7616028b6e6ee9559a10e900969a736011a608bd294ead0b38baf8853158`。
去掉header和三处插入后，整个C++文件精确恢复原文；每次event/apply循环及
全部GPU调用参数未变，331个原链接对象已核验，所有CUDA/AOT对象复用。
内嵌CUDA字节与原版完全相同。CPU覆盖正常双lane、等待中的peer异常、
重复lane、非法形状和关闭状态五个控制。两真实GTP局面×8对称的全部
显示字段与原版相同；这只是打印精度检查，未冒充原始FP32整网逐位对拍。
实际B14/S2/W1/N2检查确认两个结束端点均早于任何lane的postprocess端点。
现有受控空盘输入逐字节证明重新核验，正式Rust两轮输入SHA也与之匹配。

唯一预登记Fork测试副本/Rust/Rust/Fork测试副本ABBA，同一5070 Ti UUID、
同TF3原文件SHA `1881600c…`、B14/S2、19路空盘、无graph、80预热/1000迭代。
主指标在采样前固定为 **28000 / [max(最终同步结束) − min(计时循环开始)]**，
包含提交和GPU执行等待，排除上传、预热、事件分配/处理、线程与handle收尾。
各臂`max/min−1≤5%`，无重试或样本选择；event中位数和预先列为诊断项。

| 共同墙钟 NN rows/s | 第一次 | 第二次 | 几何均值 | 臂间波动 |
|---|---:|---:|---:|---:|
| Fork统一测试副本（WSL） |1620.978753|1556.282982|1588.301498|4.157070%|
| 当前正式Rust（Windows） |1556.804320|1555.155570|1555.979726|0.106018%|

两臂通过事前稳定性门，比例 **97.965010%（约98%）**。名义差距2.034990%
小于Fork自身4.157070%的轮间波动，不能把2%当作精确可回收开销、硬件上限
或已建立显著性。Windows/WSL与CUDA13.3/13.0差异仍在；Rust是FP16存储+
FP32累加/关键计算，Fork多个GEMM含FP16累加与外层权重缩放。内部输出
布局也不同，因此这是实际实现参考，不是同精度单kernel效率比。

同一轮event中位数速率和：Fork1745.705000/2292.885300，波动31.344374%；
Rust1569.162613/1557.856319，波动0.725760%。Fork该统计不稳定，不发布
event比例；原始诊断记录保留。它与共同墙钟的巨大差别说明，不能用
`sum(B/median(event))`替代实际完成吞吐。全部Rust4000个原event、四轮
stdout/stderr与两组Fork完整端点保存，独立计算结果一致。

原版Fork ELF的内容、mtime和plan均保持（本次objcopy指定不同输出）；
364项WSL源码/对象/运行时内容复核，Rust44项正式文件保持原SHA。
本地Fork与固定WSL源码最初字节哈希不同，核实只是换行不同后，改用
固定运行时的原始字节快照；没有拿工作区文本冒充认证源码身份。
桌面GPU程序未关闭，进程可见性不完整，不声称完全独占GPU。

本轮没有优化安装或Worker测试，当前认证Worker **1546.707634 RPC/s**保持，
不能拿它除本表得出Worker/Fork比例。完整目标继续，未证明已超越Fork。
下一步先审计将**当前完整认证组合**用于WSL的可行性，以消除平台差异：
QKV immutable和FFN compact的fingerprint均显式限Windows x64，旧WSL binary
不能靠同一组环境开关变成当前实现。WSL CUDA13.3.73已确认存在；仍需验证
400字节QKV参数ABI、Linux host编译、实际路径、整网/Worker数值及独立ABBA。
这是新的完整组合可移植性审计，不重试本轮基线、不直接启用旧失败组合，
也不改变正式计划。没有新增管理员操作；独立QKV计数器仍待自身授权。
证据：`timing-boundary-review.json`、`functional-review.json`、
`common-wall-abba-r1/review.json`、`independent-review.json`、
`preservation-review.json`、`next-step-audit.json`和`closure.json`。


2026-09-15 `wsl-current-compact-port-r1` 建立了**当前完整优化栈的同Linux参考**。
先复制现用源码，仅开放QKV/FFN压缩的Linux x64入口；原QKV 400字节ABI
证据与当前资产一致，33层打包通过。首次原生B1/B14过门，B16被残差算法
的opaque字节全等检查拒绝。只读查询确认三种固定形状的原候选序号不变，
全部公开配置及数值标志相同，但不同进程返回的内部字节不同；不把这些
字节直接假定为padding，也不屏蔽后强行通过。

隔离`public-identity-r2`只对Linux改为核验全部9项公开属性查询结果和
数值标志66050（HMMA/FP16输入/FP32累加）；保留原序号、GPU/Lt/模型/
形状限制，执行未修改的库返回对象，无搜索、降精度或回退。3项残差、
37项plan、33层打包与QKV CPU门通过；FFN的2992条指令操作数/首编码字
仍与Windows一致。修复前后B1/B14的224752个原始输出逐位一致。
新B1/B14/B16双lane共306480个输出通过全部原始误差门，六策略通道
首选720/720。实际Worker W1/W32/W64各128/128、11字段及原始协议、
计数/Drain均通过，使用真实Linux RoPE表，无旧two-GEMM修复。

唯一新ABBA为Fork测试副本/RustLinux/RustLinux/Fork测试副本，B14/S2、
80预热/1000轮，沿用统一墙钟端点与收尾屏障。主指标Fork **1552.98**、
Rust **1383.35 NN rows/s**，比值 **89.077548%**；两臂spread **2.140602%/
4.665555%**，通过预设5%门。event辅助指标两臂spread34.709%/47.893%
不稳定，不发布其比值。原模型/空盘特征逐字节对齐，但Fork的FP16累加/
权重变换及运行库差异仍单列。这是已改收尾屏障的Fork测试副本，不能
写成原认证ELF成绩，也不是Worker或Windows→Linux收益。

此次只形成可复现Linux副本和参照，未安装、未测Linux Worker性能；
44个正式保护文件不变，Windows认证Worker **1546.71 RPC/s**及此前
Windows/Fork共同墙钟约98%的独立参考保留。后续基于当前Linux完整栈
剖析共同墙钟中的主机提交/双流重叠与设备耗时，再选有界优化；不拿旧
WSL栈诊断当作当前归因。完整证据见该目录`assessment.md`/`final-review.json`。


2026-09-15 `wsl-current-compact-profile-r1` 对当前完整 Linux 压缩栈完成了
两份 CUDA 软件活动追踪（完整 API / 最少 API）。每份均核实 48 次完整
302-kernel 前向；40 次计时前向的内核累计中，Attention 三段约 41%、
FFN 约 35%。共同跨度内没有本进程所选内核的区间约 27.55% / 28.01%；
这不是全 GPU 空闲率，也不证明 CPU 饥饿。完整 API 中，40 次前向有
67000 次 cuModuleGetFunction（每次前向 1675 次）；累计 API 时间含等待，
不能直接解释为 CPU 计算成本。独立无内核控制发现：真实 Driver 返回
成功 0 和缺失符号 500，Nsight returnValue 列却都为 0，因此不按该列
拆分查询成功/失败。使用本机 Nsight 的 Linux target，无硬件计数器、
CPU 采样或新管理员操作。

基于该现象，`wsl-eager-functions-r1` 仅在隔离完整 Linux 副本中增加
初始化期准备的只读函数表，运行时 clone 现有函数句柄，无共享 Mutex。
它不同于已关闭的旧懒加载 Mutex 缓存，也不同于仍每次查询 Driver 的
模块索引路由。61 个符号及 488 次跨线程读取、两个缺失名称、句柄寿命
检查通过；20 份 PTX 逐字节一致，FFN 2992 条指令保持原值。B1/B14/B16
共 306480 个原始输出与当前 Linux 基线逐位相同，六策略通道 720/720；
真实 Worker W1/W32/W64 各 128/128、11 字段、原始协议及 Drain 全通过。
新完整 API 捕获确认相同 302-kernel 序列，40 次计时前向的查询降到 0。

唯一无追踪 B14/S2 前向 ABBA：共同墙钟 **1267.296869→1567.644216
NN rows/s（+23.699841%）**，两臂 spread **0.926421% / 0.169682%**，
通过原 5% 稳定性与 1% 增益门；event 辅助指标 +0.838130%，单列。
随后唯一真实 C64 Worker ABBA 原始结果为基线 1056.439453 / 984.958355、
候选 1409.185753 / 1371.547879 RPC/s。两臂 spread **7.257271% /
2.744190%**，基线超门，故 Worker **诊断有效、性能不认证**。几何均值
1020.072971→1390.239451 的 +36.288235% 仅作诊断；不据此前向结果上线。
四轮实际平均批大小 10.014670 / 13.363785 / 13.676127 / 9.706161；
同为 B14 上限 / S2 / C64，但并非固定物理 B14，不单独据此认定波动原因。
8000 个前向事件、32768 条请求计时、完成计数及分布已重算；全部样本保留。

本次 Linux Worker 实验关闭，不原样重测，不迁移生产。44 个正式保护文件
及 365 个外部运行库/参考内容不变，Windows 已认证 Worker 仍为
**1546.707634 RPC/s**。没有新 Fork 配对比值、没有 Windows 迁移增益。
后续在当前 Windows 生产栈建立独立的 opt-in 主机函数表候选，绑定新构建/
计划，并重新通过该平台数值、路径、兼容及前向/Worker ABBA。
证据见 `wsl-eager-functions-r1/assessment.md` 与 `final-review.json`。


2026-09-15 `windows-eager-functions-r1/r2` 独立验证了当前 Windows 完整栈的
只读函数表候选。初版为默认关闭的 EAGER_FUNCTIONS_R1，新增精确 host
artifact，原生两臂 B1/B14/B16 的 612960 个输出逐位一致。但首个真实
Worker plan0/env1 检查发现：CudaRuntime 在 plan 安装前创建，提前读取
环境值而准备了开启路径；数值仍 128/128，但路径门失败。停止剩余五个
Worker，保留初版证据，未测速，未上线。

修复版 `windows-eager-functions-r2` 不在 Runtime 构造器读取该开关；
context 安装 plan 后、CudaModel 准备权重及 strict fallback 前，使用
OnceLock 一次发布只读函数表。直连模型加载同样先准备；选择已固定后
再切换即报错。热路径无共享 Mutex，保留原模块顺序、句柄寿命与缺失
名称行为。新 artifact 同时覆盖 cuda_exec 调用时机。38 项 plan 测试、
入口解析、61 符号/488 次跨线程读取/寿命与选择冻结检查全部通过。
20 份唯一 PTX 及其所有 CLI/test 副本逐字节一致，原 FFN 2992 条指令
保持；core hash 随隔离源码路径变化，plan 绑定实际新 core/host 指纹。

修复后重新完成两臂 B1/B14/B16：612960 个原始输出逐位相同、六策略
通道首选 1440/1440。真实 Worker 两臂 W1/W32/W64 共六进程，各128/128、
11 字段、768 对原始 protobuf、计数及 Drain 全通过；全部故意设置相反
环境值，实际日志确认 plan 安装在函数准备之前。12 项真实 CLI 拒绝门
也通过，包括旧 binary、缺失/错误 host、未绑定环境、schema/core/GPU/
模型/Lt 错误。两个 Windows CUDA 软件追踪各核实48次完整302-kernel
前向；相同静态内核序列下，40次计时前向函数查询 **67000→0**。
无需硬件计数器或管理员权限，不用追踪耗时作为性能成绩。

唯一 Windows 同 binary B14/S2/W80/N1000 无追踪 ABBA，共同墙钟
**1552.344810→1564.309569 NN rows/s（+0.770754%）**；基线/候选
spread **1.570064% / 0.701469%**，稳定但未达预设 1% 增益门，故关闭。
event 辅助指标 +0.374487% 单列；8000 个原始事件及共同墙钟重算通过。
不跑 Worker 性能 ABBA、不继续迁移兼容测试、不原样重试或上线。
Linux 之前 +23.70% 的前向结果不外推为 Windows 收益，其 Worker
不稳定结论也保持。Windows 生产仍为原认证 **1546.707634 RPC/s**，
44 个正式保护文件与365个外部内容不变；没有更新 Fork 比值。

后续回到当前 Windows GPU 热点，先审计压缩 FFN down 的实际 K 形状、
Lt 选择，以及历史 rank=time / 已关闭 K>=432 Lt21 的覆盖范围，再选择
有实质差异的有界 GEMM 候选。保留原 FP32 累加与 half/精确激活边界。
完整记录见 `windows-eager-functions-r2/assessment.md` 与 `final-review.json`。


**2026-09-16 compact FFN down tile112：数值逐位通过，性能未过门，关闭。**
`compact-down-tile112-r1/r2` 只对14种 K>=432、33层中的22层，使用现有
Lt池完整64字节 index2（id67/tile403/custom28/stage34，128×112×32）；
其他形状继续 index0。实际 M5054/N384、TN、FP16输入/权重、FP32累加/
输出、beta=1及32MiB工作区保持。与已关闭的id21/tile15候选不同，未改
全局rank=time；后者beta=0临时输出计时不等于真实残差计算。

初版诊断在任何matmul前停止：把数值能力标志固定为66050过严，DLL
地址还指向exe导入跳板。只读查询确认id67为197122，即HMMA、唯一FP32
累加及FP16/BF16输入能力；描述符仍固定FP16，不代表改用BF16。修正版
按算法核精确完整标志并定位实际DLL；原失败/查询/源码完整保留。34组、
131970048个输出全过原FP64门，两臂全部逐位一致，双lane、完整guard和
只读输入均通过。136项实际选择及加载DLL身份也核实。

一次普通权限CUDA软件追踪核实136次真实GEMM及68份输出逐位一致。
较小tile确实生效，但大K共享内存是92672字节，原tile为95232；此前
从K96推测的46592不适用于大K。实际grid72→144个block，每block线程
256→128（须乘blockX与blockY），每线程寄存器均255。原追踪保留，
修正元数据解释，未重采集，不据此推断occupancy或吞吐。

唯一33层down序列、双lane、W80/N1000 ABBA，共同主机计时为
**541.389238→543.930906 序列/s（+0.469472%）**，两臂spread
**1.106565% / 0.006310%**，稳定但低于1%门。event序列指标约
**−15.591910%**，基线spread **6.533930%** 超过5%（候选0.043433%），
因此该降幅仅作诊断，不能称稳定回退。两项均须通过的预设门未满足，
关闭候选，不重测、不选择子集，不进入整网/Worker性能、不上线。
8000个原始事件、528份前后输出和共同墙钟独立重算通过。这里只测down
GEMM序列，主机边界含33次残差D2D重置，event不含；不是NN rows/s。

44个正式保护文件及365个外部参考内容不变，生产Worker仍为
**1546.707634 RPC/s**，没有更新Fork比值。下一步审计当前compact
DualGemm up的实际寄存器/地址/epilogue开销，并对照已关闭实验，找到
有实质差异的机制后再登记候选。证据见
`compact-down-tile112-r2/assessment.md`、`final-review.json`。


**2026-09-16 compact FFN up 稠密网格：原核逐位相同，稳定微增未达门。**
`compact-ffn-dense-grid-r1`先核查33层现用half权重：每层gate/up联合的
逐位行向量无重复，同位置无相同/符号反转行，也无两个投影共同全零的
K列。该结论针对原始位模式，不把有符号零规范化或新增精度变换。
原核仍236寄存器/48KiB共享、2992条指令。当前swizzle2在11层奇数N
tile数上每层多发射40个直接返回的CTA，完整前向14480个中440个为空；
原未压缩N1152为18个tile，没有这个空CTA来源。

候选只对奇数ceil(hidden/64)>1设置swizzle_log_tile=0，并改用
ceil(tokens/128)×ceil(hidden/64)网格；偶数保持原值。这也改变受影响
层的有效tile发射顺序，不声称只是删除空block且顺序不变。两臂使用
同一Params构造和原cudaFuncSetAttribute/launch路径，原M128/N64/K32、
warp64×32、stage3、FP32累加、精确SiLU和全部half边界不变。此项不同于
历史N1152的horizontal或小CTA候选，也不复用全FFN affine融合。

CPU对B1–B16全部23种宽度完成178360个block坐标覆盖检查；544组实际
C++ Params构造证明两臂只差swizzle字段。隔离编译的原2992条SASS及
两机器字/控制字完全一致。34组、228198208个half输出与现用参考全部
逐位相同，双lane、guard和只读输入均通过。除真实首FFN外，33个逐层
算子输入是31行周期重复至5054，不能称完整模型激活覆盖。
一次普通权限CUDA软件追踪核实136次实际网格和相同内核/资源，68份
输出仍逐位一致；未使用管理员权限或硬件计数器。

唯一双lane、33层up/SwiGLU序列W80/N1000 ABBA：event
**521.491171→521.878522 序列/s（+0.074278%）**，两臂spread
**0.963315% / 0.371220%**；共同主机 **520.000250→520.291067
序列/s（+0.055926%）**，spread **1.026622% / 0.447175%**。
两项稳定但均低于1%门，因此关闭，不重测、不进入整网/Worker、不上线。
8000事件及528份前后完整输出独立重算通过。此上投影序列不含down、
其他整图计算或D2D残差重置，不能与上一轮down序列或NN rows/s混算。

44个正式保护文件和365个外部参考内容不变，现用认证Worker仍为
1546.707634 RPC/s，Fork参考保持。下一步审计strict Attention核心的
KV尾块/softmax依赖，对照已关闭M64、stage2、KV展开和epilogue屏障
实验后，再决定有实质差异的候选。证据见
`compact-ffn-dense-grid-r1/assessment.md`、`final-review.json`。


**2026-09-16 strict Attention 逐行 half 打包：数值通过，稳定变慢，关闭。**
`strict-attention-row-pack-r1`只缩短概率矩阵的FP32生存期：每行完成原
FP32 exp和未舍入row sum之后，立即写入PV使用的half片段，取消调用端
整片转换。保留原M128/N96/stage1、KV顺序3/2/1/0、QK/PV FP32累加、
四线程归约、精确exp及每个概率唯一的RN-half边界，不合并旧关闭候选。

固定工具链离线导出及源/机器码检查通过：寄存器255→222，每线程栈
40→0，SASS槽3360→3336；192处FP32 HMMA、196处完整普通expf数据流、
拷贝/屏障和原KV回跳均保持。实际加载的模块/CUBIN独立绑定；驱动报告
的静态共享为0、动态共享20480字节。事后只读occupancy查询显示两臂
理论上限均2 CTA/SM、8 warp/SM；这是理论上限，不是实测占用率或
变慢原因的证明，也没有使用管理员权限或硬件计数器。

五组输入（四组合成及首Attention真实模型历史捕获）各在两条独立流
candidate先两次、再原CUBIN，**58,222,080个half输出逐位一致**。
输入/guard保持且输出全部有限。相同负载memcheck零错误、racecheck
零hazard；含工具运行共174,666,240个元素独立复核。历史捕获不是
本轮新CLI整网输出，不能据此省略后续整网或Worker验证。

唯一B14/S2、W80/N1000算子ABBA：event **344963.540595→336344.420503
算子行/s（−2.498560%）**，A/B spread **0.039432%/0.038447%**；
共同主机 **338620.691633→329796.106732算子行/s（−2.606038%）**，
spread **0.240104%/0.088967%**。8000个原始事件、16份完整前后输出
独立重算一致。两项稳定下降，关闭；未进入整网或Worker性能、未上线、
未原样重测。44个正式文件和365个外部参考内容保持，现用Worker仍为
1546.707634 RPC/s，Fork对照比例没有更新。

后续先审计独立的8-warp Attention组织：原源码将warp只分配到M维，
M128的can_implement允许256线程，softmax仍为4线程组归约。考虑让
8个warp分担同一tile，减少每线程row state；尚未登记新候选，仍需
证明拷贝布局、算术顺序、真实资源与数值。不能从行打包失败推断该方案
有效，也不组合行打包改动。证据：
`strict-attention-row-pack-r1/assessment.md`、`final-review.json`。


**2026-09-16 strict Attention 8-warp：数值通过，稳定明显变慢，关闭。**
`strict-attention-eight-warp-r1`从现用原始生成器独立出发，只把CTA由
128改为256线程。三个生成模块与现用版本逐字节相同；保留M128/N96、
stage1、Q/K/V分离共享区、原KV顺序、FP32 QK/PV和精确softmax，未带入
已关闭的逐行half打包。warp沿M维从4增到8，每线程处理的行状态减半。

499项固定工具链、离线导出及代码门通过。实际候选REG160/STACK0、
1824个SASS槽，静态96个FP32 HMMA、98个完整expf数据流，对应每线程
工作减半；不能把静态指令减少直接当作整个CTA工作减少。原核为
REG255/STACK40、3360槽、192 HMMA/196 exp。实际ABI保持六参数，
block为256/128分别绑定；动态共享均20480字节。原KV回跳、异步复制
阶段/等待保持，两个输出屏障的参与线程数按256正确变化。

五组输入在两条独立流上候选先两次再原核，普通运行58,222,080个half
输出逐位一致，有限输出、输入只读和guard全部通过。相同负载memcheck
零错误、racecheck零hazard，含工具运行共174,666,240元素复核。真实
模型输入仍是冻结的首Attention历史捕获，未冒充本轮整网覆盖。
只读驱动occupancy查询：CTA上限由2/SM变为1/SM，两者均8warp/SM。
它未增加理论驻留warp数；此为资源上限，不是硬件计数器或变慢归因。

唯一B14/S2 W80/N1000算子ABBA：event **344759.653183→275503.781005
算子行/s（−20.088160%）**，两次中位数在计时分辨率下相同，A/B spread
均0；共同主机 **336927.933817→272140.202441算子行/s（−19.228958%）**，
spread **0.155710%/0.289177%**。8000原始事件和16份前后完整输出
独立重算一致。两项稳定下降，关闭，不进入整网/Worker性能、不上线、
不原样重测。CPU收尾检查完成数值/事件复核后遇到旧目录替换错误；已用
独立只读收尾脚本补全外部校验，未重跑任何GPU工作，原记录完整保留。

44个正式文件和643个外部运行库/代码生成参考均保持。现用认证Worker
仍为1546.707634 RPC/s，Fork对照比例不变。下一步先审计现用严格FP32
实现的KV tile状态量与舍入影响，核对旧B3/N64历史，再考虑独立N32方案；
KV分块会改变online softmax的归约顺序，不能预设逐位相同或降低数值门。
已有132矩阵审计没有可删除整头，不重复零头压缩。证据：
`strict-attention-eight-warp-r1/assessment.md`、`final-review.json`。

**2026-09-16 strict Attention N32 整网数值通过，完整前向未达性能门，关闭。**

`strict-attention-n32-r1` 从现用严格生成器独立改为 M128/N32/stage1、
128线程，三个生成模块保持原字节；不叠加已关闭的逐行打包或8-warp。
QK/PV仍FP32累加，完整exp及归一化除法不变。KV循环从4块变为12块，
online softmax分组及half概率的舍入结果可能变化，因此事先要求整网
原数值门先于任何性能测试，未用算子误差自定新容差。CUBIN为
REG168/STACK0、动态共享12288字节；驱动理论上限3CTA/12warp每SM，
原核为2CTA/8warp。这不是实测占用率，也不是性能收益证明。

五组双流算子诊断完成，含memcheck/racecheck共174,666,240个half
输出复核，零内存错误/竞态，重复及跨lane输出逐位一致。与原核有数值
差异，真实首层输入最大绝对差0.00048828125。首轮误把“uniform随机
分布”当作“常数张量”的检查失败完整保留；R2仅纠正输入属性判断，
没有调宽数值门，也没有性能重试。

`strict-attention-n32-integration-r1/source` 是隔离副本。独立N32
artifact和开关覆盖两条launch路径，保留原核作为同一构建的对照。
39个plan测试、3个asset测试、12个实际CLI负例通过；20份生成PTX、
现用QKV CUBIN和原DualGemm的2992条指令保持。QKV/FFN的主机绑定
指纹按实际新源码重算。六组native B1/B14/B16双lane核对612,960个
输出值，原五个输出头阈值及六通道top-1全过；N32 B14为336/336，
B1/B16和同构建baseline全部逐位相同。真实Worker W1/W32/W64双臂
768/768对C++FP32首选一致，11字段、原始protobuf重建和Drain全过。

数值通过后，仅运行一次完整前向B14/S2、W80/N1000 ABBA：共同墙钟
**1555.872109→1546.358674 NN rows/s（−0.611454%）**，双臂spread
**1.047416%/0.341877%**；event诊断 **1574.825090→1560.929511
（−0.882357%）**，spread **1.216771%/0.156819%**。8000个原始事件
独立重算一致。两项通过5%稳定性门，主指标未达≥1%收益门，因此关闭；
没有算子性能ABBA、没有Worker性能ABBA、没有上线或原样重测。

44个正式文件及643个外部内容保持，现用认证Worker仍1546.707634
RPC/s，既有Fork参考比例不变。下一步先审计独立KV64的状态量、循环
和舍入影响并核对历史，只有新机制证据才注册新实验；旧候选保持关闭。
证据：`strict-attention-n32-integration-r1/assessment.md`、
`closure.json`、`forward-abba-r1/review.json`、`final-review.json`。

**2026-09-16 strict Attention KV64：整网原始输出通过，真实Worker首选失败，关闭。**

`strict-attention-n64-r1`独立采用当前严格生成器M128/N64/stage1、
128线程，三个生成模块保持原字节，没有复用旧B3手写P-reg/Q-hoist，
也没有组合已关闭的N32/逐行打包/8-warp。分数状态96→64，KV块4→6，
逐线程动态完整exp调用396→404；FP32 QK/PV、原half边界与精确函数保持，
分块导致的舍入差异不假设逐位等价。编译得到REG222/STACK0、2384个
SASS槽、动态共享16384字节；理论上限仍2CTA/8warp每SM，不声称占用率提升。

五组双流算子诊断、memcheck、racecheck共174,666,240个half输出复核，
有限值、guard、只读输入、重复及跨lane一致性均通过，零内存错误/竞态。
真实首层历史输入最大差0.00048828125；这些组件结果不构成整网准入。

隔离接入位于`strict-attention-n64-integration-r1/source`，独立N64
artifact/开关覆盖两条Attention调用。39个plan、3个asset测试通过；
20份PTX、QKV CUBIN及原DualGemm的2992条指令保持。主机依赖指纹按
新源码重算。六组B1/B14/B16原生双lane共612,960个输出值通过原头部
误差门和六通道首选门，N64 B14为336/336；B1/B16及baseline逐位一致。

真实Worker依次完成baseline-W1、candidate-W1、baseline-W32，均
128/128；**candidate-W32为126/128，失败**。该组正常完成128请求和
Drain，11字段误差门均通过，但`benchmark-ply-001-sym2`的首选72→288，
`benchmark-ply-001-sym5`的首选288→72。C++FP32两首选概率差分别为
0.00055453和0.00055454；near-tie仅是诊断标记，不能豁免100%首选门。
四组512对原始请求/结果protobuf已独立重建，完整复现两处差异。

**按预先约定的数值门关闭KV64**。W64两组未执行，实际CLI负例阶段未执行，
算子/完整前向/Worker性能ABBA均为0；未重试、未调宽阈值、未上线。
44个正式文件和643个外部内容保持，现用Worker仍1546.707634 RPC/s，
Fork参考比例不变。后续先审计当前RMS→QKV数据交接及历史融合边界，
保留12项FMA/XOR归约与RN-half输入边界；不继续KV尺寸扫参，也不重复
已关闭的完整N384 outproj+RMS或L2 inner持久化实验。L2历史Worker仅
+0.412137%，且当前Fork B14计划两项L2开关均false，已再次核对。
证据：`strict-attention-n64-integration-r1/assessment.md`、
`worker-golden-r1/rejection-review.json`、`closure.json`、`final-review.json`。

**2026-09-16 RMS→QKV 数据交接与零旋转对筛查：未登记新内核。**

`rms-qkv-handoff-audit-r1` 核对现用CLI嵌入的完整executor PTX和QKV
CUBIN。RMS入口与已有实际Driver JIT取证逐字节对应：每lane只有12次
输入加载和12次scale加载，输入值在归约后由寄存器复用。保留原12项
FP32 FMA、XOR16/8/4/2/1、div.rn、现有rsqrtf和两次乘法、RN-half；
源码的第二个循环并未造成第二轮输入显存加载，不能据此宣称消除重复读取。

当前每次QKV为40×18个CTA、M128/N64/K32、128线程，146寄存器、
36864字节共享。把完整RMS搬到每个消费者CTA，会将每行归约重复18次。
保留M128完整K384的归一化half块需98304字节，配一块B仍需102400字节，
超过本机新查得的每block opt-in上限101376；三阶段B合计110592字节。
这些是该具体存储设计的资源核算，不是全部融合方案不可行的证明。
局部块、寄存器缓存或先计算行统计需要另行设计，不能免费省掉这些成本。

另按原模型偏移独立重建132个Attention矩阵，19,464,192个half值的
SHA与原Rust编码审计一致。Q/K各174个零输出通道是33层合计，只有
87个共同零旋转对。按当前N64投影块成对压缩，仅
`model.blocks.1.blockstack.2`可将18块减至16块：全模型594块只少2块，
即80个CTA。依既有轻量时间线按块数比例粗筛约为累计内核耗时的
0.05559%，尚未扣除输出位置恢复等开销；不是实测收益或墙钟速度上界。
因此不把FFN的通道压缩收益套用于QKV，也不改变RoPE配对/头维度/缩放。

Fork当前B14实际plan选择one-warp-exact，RMS独立于QKV，输入/参数/
输出均half且归约树与Rust不同。历史笔记的warp4-vec8不代表此B14选择。
本地与留存WSL两份完整调用源码仅有CRLF/LF差别；初次原始SHA等同检查
失败留档，经仅换行归一后逐文件匹配，不把不同原始字节称为相同SHA。
核对了既有八lane、固定384、完整N384 outproj+RMS、M64 QKV和L2裁决，
均不重开。本轮没有GPU kernel、性能测试或管理员采集；只有设备属性
读取和CPU审计。44个正式文件与643项外部内容保持，认证Worker仍为
1546.707634 RPC/s，Fork参考比值未更新。

下一步以现存QKV源码/机器码核对异步拷贝地址和等待依赖，对照已经关闭的
stage2实验；静态假设不冒充硬件计数器结论。独立QKV计数器仍等待原先
请求的自身授权，不复用已执行的一次FFN授权。目标继续active。
证据：`rms-qkv-handoff-audit-r1/review.json`、`assessment.md`、`final-review.json`。

**2026-09-16 QKV 异步拷贝审计与双片段 epilogue：整网数值通过，前向未达收益门，关闭。**

`qkv-copy-wait-audit-r1` 以实际模板类型等同证明核对现用 QKV 的四个
global/shared iterator，复核已有 CPU 地址导出的301,056个向量、2,408,448个
half地址；完整40×18网格对应6,635,520次逻辑16B拷贝。八lane、128B分段的
解析bank映射未发现重复，但这不是SM120硬件事务或stall实测。现用三阶段
等待与两个已关闭stage2实现区别保留，不重新测速旧候选。

据此登记独立 `qkv-epilogue-batch2-r1`：把两个累加片段写入**已经分配的**
两个共享槽，再依原顺序读取和输出。选中beta0分支的CTA同步从16次降到8次；
整个CUBIN函数的静态BAR为35→19，不能把两条互斥输出分支同时计入一次执行。
保持146寄存器、零栈/局部内存、36864B动态共享、原三阶段MMA、全部half边界
及标量RoPE。32处FP32 MMA的两个机器字均相同；前段另有47处设置/reuse调度
差异，未称完整主循环机器码相同。新构建内的原QKV完整SASS与正式版一致。

新Params实际CPU导出仍为400B、align8、六指针偏移及模板逐字节相同。
CPU共享槽映射保留8192个累加值顺序；首次检查脚本的C++换行转义错误保留，
修正仅作用于CPU检查器。首次数值登记误写expected.f16le，已依据冻结manifest
改为reference.f16le；两次准备失败均未启动GPU或改变候选内核/门槛。
真正数值测试直接加载新CUBIN符号，未执行源码中遗留的旧main。普通、memcheck、
racecheck各双流/三份输出，独立复核合计18数组、104,799,744个half值全等，
guard/输入只读/重复与跨流位模式通过，内存错误与竞态均为零。

`qkv-epilogue-batch2-integration-r1` 从当前正式源码另建隔离CLI，新增默认0的
`KATAGO_CUDA_QKV_EPILOGUE_BATCH2_R1`，统一走tactic_var并要求immutableQKV及
显式plan绑定。现有qkv_immutable_artifact覆盖两份CUBIN和宿主源码，不复用旧
认证指纹。38个plan单测、2个QKV资产测试通过；原20份PTX、原QKV/Attention
CUBIN及FFN机器码保持。实际仅core/QKV/FFN三个构建身份改变，strict Attention
身份保持；原inspection字段列入了未改变的strict身份，单独identity-review已
澄清并保留原报告字节。

完整native B1/B14/B16×两臂、双lane共612,960个输出值逐位一致，六路policy
top1全等。真实Worker W1/W32/W64×两臂共768请求，C++FP32十一字段与
top1 **768/768**全过；原始请求/响应protobuf重建、身份、真实batch与Drain通过。
每组环境值故意与plan相反，实际选择仍服从plan；12个CLI负例全部fail-closed。

全部数值门之后，仅运行一次完整前向B14/S2、W80/N1000 ABBA。共同墙钟吞吐
几何均值**1539.043350→1541.826768 NN rows/s（+0.180854%）**；event
**+0.773895%**。共同墙钟两臂spread **0.378600%/0.265186%**，event
**0.294348%/0.165690%**，均稳定，但共同墙钟未达1%门。8000条原始event
和共同墙钟公式已独立重算。**关闭batch2，不原样重测，不测Worker性能、不上线。**
同步次数下降不能直接换算为整网收益。

44个正式文件和643项外部内容保持，认证Worker仍 **1546.707634 RPC/s**，
既有Fork统一前向比例未更新；本轮无管理员操作。下一步仅先核算4/8片段独立
共享槽是否可使用原MMA union，逐项证明布局、ABI和实际寄存器，再决定至多一个
新候选。18432B/36864B目前只是源码存储算术，不是新内核资源、数值或性能证据。
完整数值、竞态、真实Worker及ABBA门继续保持，整体目标active。
证据：`qkv-copy-wait-audit-r1/review.json`、`qkv-epilogue-batch2-r1/independent-component-review.json`、
`qkv-epilogue-batch2-integration-r1/{native-review.json,worker-golden-r1/review.json,forward-abba-r1/review.json,independent-timing-review.json,closure.json,final-review.json}`。

**2026-09-16 QKV 八片段 epilogue：全部数值通过，完整前向无收益，关闭。**

`qkv-epilogue-wide-audit-r1` 先用实际CUTLASS类型导出4/8片段布局：epilogue
分别18432B/36864B，kernel与MMA的union均36864B；两种Params仍400B、align8，
六指针偏移、完整模板及tag样本与正式版本逐字节一致。读写迭代器类型、行stride72、
每slot1152个FP32元素及128线程偏移一致，整数标签证明8192个累加值顺序不变。
第一次CPU导出器编译因依赖类型缺typename及局部变量重名失败；原记录保留，
r2只修正检查器，未执行GPU。只选择8片段进入设备候选，没有4片段内核或测速。

`qkv-epilogue-batch8-r1` 在原MMA共享分配中设置8个独立slot，一次写完全部
累加片段，然后按原顺序转换和RoPE输出。epilogue共享9216→36864B，但实际
kernel共享仍36864B；146寄存器、零stack/local、原三阶段MMA不变。选中输出
分支同步16→**2**，完整函数静态BAR35→7（MMA3、两条互斥输出分支各2）。
32处FP32 MMA机器字均与原核相同，但前段323处寄存器/设置/reuse调度不同，
未声称整个主循环机器码一致。新构建内原核完整SASS与正式版匹配。
实际新类型的ABI、共享及全局输出映射再次导出，与上述CPU证明及正式模板匹配。

普通、memcheck、racecheck均直接加载新CUBIN符号：双流/重复/不同NaN填充、
guard及完整只读检查通过。18份保存数组合计**104,799,744个half值**与现用
完整第一Attention捕获逐位一致，独立重新读回所有字节；内存错误/竞态为零。

隔离 `qkv-epilogue-batch8-integration-r1` 新增默认0的
`KATAGO_CUDA_QKV_EPILOGUE_BATCH8_R1`，统一tactic_var、显式plan绑定及组合
QKV指纹，未把已关闭batch2装入这个CLI。38个plan、2个资产单测通过；原20份
PTX、原QKV/Attention CUBIN与FFN机器码保持。完整native B1/B14/B16×两臂、
双lane共**612,960个输出值逐位一致**；真实Worker W1/W32/W64×两臂的
C++FP32十一字段、首选 **768/768**、原始protobuf、实际batch及Drain全过。
每个Worker的环境值与plan相反，实际仍遵循plan；12个CLI负例全部被拒绝。

全部数值门之后唯一完整前向B14/S2、W80/N1000 ABBA，共同墙钟吞吐几何均值
**1552.739310→1551.916908 NN rows/s（−0.052965%）**，event **−0.011767%**。
共同墙钟两臂spread **0.409594%/0.149514%**，event **0.545429%/0.066145%**，
均满足5%稳定门；均值差很小，不据此声称明确硬件回退。8000个event与共同
墙钟公式独立重算，未达1%收益门，**关闭8片段方案；不原样重测、不测Worker
性能、不上线**。同步大幅减少仍没有形成可验收的整网收益。

44个正式文件与643项外部内容保持，认证Worker仍**1546.707634 RPC/s**，
既有Fork对照比例未更新，本轮无管理员操作。下一步回到QKV分块的数据复用：
先审计classic M128/N128/K32、warp64×64、s3的资源/ABI/标量RoPE映射，核对
旧Rust CuTe登记M128/N128/K64、288线程、无RoPE及已关闭M64/N64方案的差异。
旧CuTe的实际stage来自导出，不按历史s2标签推定。此处没有登记新内核；49152B
仅为拟议MMA存储算术，实际寄存器、布局和数值均待证明。整体目标继续active。
证据：`qkv-epilogue-wide-audit-r1/review.json`、`qkv-epilogue-batch8-r1/independent-component-review.json`、
`qkv-epilogue-batch8-integration-r1/{native-review.json,worker-golden-r1/review.json,forward-abba-r1/review.json,independent-timing-review.json,closure.json,final-review.json}`。

**2026-09-16 classic N128/K32 QKV：完整前向 +1.785%、真实Worker +1.225%，已验收安装。**

在原有混合精度纪律内，QKV分块由CTA128×64×32/warp64×32×32改为
CTA128×128×32/warp64×64×32，保持三阶段、128线程、普通CUTLASS epilogue
和原标量RoPE。与旧CuTe N128/K64、288线程无RoPE实验不同；没有复用已关闭
batch2/batch8的输出调度。实际CPU类型导出共享49152B、epilogue17408B，
每线程累加器128，Params400B/align8与六地址偏移保持，但模板及网格独立生成。
编译后的226寄存器、零stack/local；Driver实际动态共享49152B、静态0，
与基线一样每SM驻留2个CTA。新网格40×9，原40×18，输入A的逻辑装载字节
减半；这不是DRAM流量或带宽实测。64处MMA均为FP32，未声称与原32处机器码相同。

CPU完整映射覆盖5,822,208个half输出，尾部与RoPE索引通过；普通、memcheck、
racecheck的18份输出合计104,799,744个half值逐位一致，内存/竞争错误为零。
隔离整网B1/B14/B16×两臂、双lane共612,960个输出值逐位一致。真实Worker
W1/W32/W64×两臂768/768首选、C++FP32十一字段、原始protobuf及Drain全部通过。
继承的路径复核器固定40×18，首次CPU复核因此停止；保留原脚本，改为按两臂
明确检查40×18/40×9，再复核同一批原始输出，没有重跑GPU或降低数值门。
38个plan与2个资产单测、12个CLI负例及6个plan优先于反向环境值的Worker正例通过。
原FFN设备指令、Attention/原QKV CUBIN和20份PTX保持，组合指纹绑定新旧
CUBIN、ABI、参数模板和主机代码；开关统一经tactic_var读取，默认0且须显式绑定。

唯一完整前向B14/S2、W80/N1000 ABBA：共同墙钟几何均值
1541.248170→1568.758162 NN rows/s（**+1.784916%**），两臂spread
0.759771%/0.734504%；event **+1.590239%**，spread0.127740%/0.216363%。
8000个event和墙钟公式独立重算，稳定且超过1%门后才执行真实Worker ABBA。
Worker B14/S2/C64、每臂2轮、每轮预热256/计时8192 uncached请求：
**1545.265623→1564.201170 RPC/s（+1.225391%）**，spread
**0.010705%/0.482587%**；32768条原始时延和计数器全部复核。
平均物理batch约13.8731/13.8966，NN rows/RPC=1，非NN比例为0。

迁移前4组ONNX ORT、2组graph、默认/C32 Worker的4组128请求及6计划加载通过。
随后安装4个验证过的源码修改、6个新AOT文件、相同CLI、6份Windows计划及
吞吐配置注释，共18个文件，12份原文件已备份；32个其他正式文件保持。
仅 `worker-tf3-sm120-throughput.json` 启用 `KATAGO_CUDA_QKV_CLASSIC_N128_R1=1`，
其余五份Windows计划显式0。正式路径再次通过6计划加载与128请求FP32对拍/Drain，
当前认证Worker更新为 **1564.201170 RPC/s**。50个现用文件纳入新指纹快照，
643项Fork/WSL外部内容保持；本轮无管理员操作，没有新的Fork比例。
下一步重新冻结当前安装版本与原Fork测试副本，按相同输入/双流/完成墙钟边界
做新的顺序ABBA；既有约98% Windows/Fork比例仍是旧版本历史证据，不能外推。
整体优化目标继续active。证据：
`qkv-classic-n128-audit-r1/review.json`、`qkv-classic-n128-r1/independent-component-review.json`、
`qkv-classic-n128-integration-r1/{forward-abba-r1/review.json,worker-performance-r1/review.json,promotion-r1/acceptance.json,final-review.json}`。

**2026-09-16 N128安装后重新对照Fork：共同墙钟基本持平，event仍不稳定。**

`n128-current-fork-reference-r1` 重新冻结现用N128 CLI/吞吐计划、原TF3模型、
同一RTX5070Ti及原Fork测试副本。50个现用文件与643项外部内容前后匹配；
直接读取原Fork ELF和测试副本的实际.nv_fatbin节，GPU字节相同。重新检查
完整源码插入边界：两边从首个计时event提交前开始，到每lane末次流同步后结束，
读取/销毁events要等待所有lane完成。Fork副本有这个刻意加入的完成屏障，
**不是原认证ELF性能**，也不声称屏障性能中性。既有原Fork/副本功能和输入证明
的字节保持；Rust本次输出再次匹配同一空盘输入hash，实际启用N128/40×9网格。

唯一顺序ABBA：Fork测试副本→当前Rust→当前Rust→Fork测试副本，
两边B14/S2、W80/N1000，主要口径为28000行/共同完成墙钟：
- Fork：1579.434939、1560.170211，几何均值 **1569.773023 NN rows/s**，
  spread **1.234784%**。
- Rust：1568.700894、1576.687584，几何均值 **1572.689169 NN rows/s**，
  spread **0.509128%**。
- 两臂通过5%稳定门，Rust/Fork **1.001857687（100.186%）**，名义差+0.186%。
  差值小于两臂各自波动，因此结论是本轮共同墙钟**基本持平**，不是稳定超越证明。

event几何均值Fork2116.563866、Rust1575.997921，Fork两次1983.578319/
2258.465197的spread **13.858131%**，Rust0.345639%；次要event稳定门失败，
不据其约0.745原始比值发布吞吐差距或选新默认值。Rust4000个原始event与
两边共同墙钟端点独立重算。Fork只导出每lane中位数和汇总值，没有原始逐次
event；独立复核初次对小数序列化舍入要求过严，改为传播中位数及汇总数各自
文本精度的舍入区间后通过，原始计时、主墙钟公式和5%稳定门均未改变，也未重测。

这是Windows Rust对WSL2 Fork的实用前向参考，平台因素与Fork部分FP16累加/
内部权重缩放仍单列；不能解释成同操作系统/同精度的算法速度或gRPC对照。
正式N128版本和已验收Worker **1564.201170 RPC/s** 保持。本轮没有新内核、
管理员操作或Worker测速。下一步先审计当前WSL Rust实际二进制、计划与源码
是否含已接受优化，以及Linux的ABI/指纹边界，再决定同环境对照或移植；
保留原数值门，不重跑这组未变更ABBA、不重开已关闭tactic。整体目标继续active。
证据：`n128-current-fork-reference-r1/{source-review.json,external-before/elf.json,
common-wall-abba-r1/review.json,independent-review.json,external-after/external.json,final-review.json}`。

**2026-09-16 WSL现状审计与N128隔离移植：ABI、构建和CPU检查通过，GPU数值尚待验证。**

`wsl-n128-current-audit-r1` 实读两份Linux二进制。正式路径
`target/cudarocmopt-wsl/release/katago-rs` 仍匹配旧ONNX/q64计划，
没有strict/QKV/FFN压缩指纹。此前合格的
`wsl-current-compact-port-r1/public-identity-r2` 则包含18项TF3开关及
FFN压缩、旧QKV，尚未包含N128；新读指纹与原记录完全相符。
因此此前89.08%的同Linux前向参考只能归属该旧隔离版本。

Linux编译器使用实际N128 CUTLASS类型导出参数：400字节、8字节对齐，
六个指针偏移64/112/208/304/288/296，grid40×9、128线程、共享49152字节。
完整模板和两组不同指针样本与已验收Windows N128相同；仅将实际成员地址
证明的28..31、372..375填充字节归零，未改动成员数据。独立检查两种tile
的全部5,822,208个half位置均恰好覆盖一次，RoPE系数范围0..69311，
有效727,776个向量及9,504个尾部屏蔽向量一致。标量RoPE仅类型重命名，
FP16存储/FP32计算及显式舍入顺序保持。

与当前Windows比较，326个既有源文件中有7个差异。追溯原N128安装备份后，
差异严格归属于N128接入、两处平台守卫、既有Linux cuBLASLt身份验证及配置注释。
新 `wsl-n128-port-r1` 固定332个文件（含6个N128资源），相对当前Windows只适配
cuda.rs/ffn_compact.rs/qkv_immutable.rs三文件。Linux残差仍检查原候选索引的
七个公开属性、两个不支持属性和数值flags66050，执行原返回对象；不掩码、
不换索引、不回退。保留Linux原生RoPE，不引入已关闭的eager函数表。

隔离CLI及原生测试程序编译成功；45项相关CPU检查通过：3残差、38计划、
2打包（覆盖33层）、2QKV资源/参数。实时新指纹已绑定，QKV/FFN复合hash从
实际嵌入源码和资源独立重算相符；CLI和测试ELF均含原N64与新N128的精确CUBIN。
FFN完整反汇编除编译器匿名命名空间的8位hash外，与旧合格Linux版本相同，
2,992条指令的5,984个64位编码字（含控制字）全部一致。

这一步证明ABI与构建，不替代GPU数值和性能验收。新Linux版本尚未执行推理，
没有新ABBA或生产迁移；50个现用文件及643项外部内容保持。当前Windows
Worker仍为1564.201170 RPC/s；与Fork测试副本共同墙钟仍是100.186%的近似持平参考。
下一步在该已构建副本进行组件/resource/memcheck/racecheck，再做两臂
B1/B14/B16原生及W1/W32/W64真实C++FP32 Worker和实际计划正反例；全部原门
通过后才进入唯一前向/Worker ABBA。整体目标继续active。
证据：`wsl-n128-current-audit-r1/{independent-review.json,next-intake.json,final-review.json}`、
`wsl-n128-port-r1/{source-freeze.json,build-wsl-r1/completion.json,inspection-r1/review.json,
inspection-r1/linked-ffn-independent.json}`。

**2026-09-16 Linux N128数值全部通过；唯一前向ABBA候选不稳定，关闭性能接入。**

`wsl-n128-port-r1` 在已冻结Linux二进制继续验证，无重建或精度变更。
原N64与新N128均在两路独立CUDA流实际发射；资源分别146/226寄存器，
动态共享36864/49152字节、无local/static shared，两者均可驻留2个CTA/SM。
普通、memcheck、racecheck共18份完整输出，合计104,799,744个half逐位匹配
冻结首层QKV参考，NaN覆盖、边界、只读输入和重复发射检查全过。两个核均受
sanitizer检查，零错误/零hazard。最初固定工具路径缺失，目标尚未执行即返回127；
随后按本机NVIDIA仓库元数据SHA校验并将13.3.75-1诊断包仅解压到实验目录，
保留失败记录，实际检查各执行一次，没有安装系统包或修改驱动。

六组原生整网B1/B14/B16×N1280/1、每组双lane，612,960个输出值通过与
旧合格Linux及当前Windows N128的原绝对误差和六通道100%首选门。Linux
两臂直接比较306,480个输出值逐位一致；B14实际grid40×9，B1/B16回退范围正确。
全程使用Linux原生RoPE生成；组件中的冻结系数未注入整网。
随后真实Worker W1/W32/W64×两臂，768/768首选、11个C++FP32字段、全部原始
protobuf重建、128 NN行/组及Drain通过。六组均验证plan覆盖相反环境值，
33层FFN映射与实际QKV/strict路径正确；12项CLI错误计划全部按预期拒绝。

唯一共同墙钟ABBA（baseline→candidate→candidate→baseline，B14/S2、
W80/N1000、同一Linux binary，仅N128开关变化）：
- baseline：1338.981373 / 1353.562967，几何均值 **1346.252428 NN rows/s**，
  spread **1.089007%**。
- candidate：1382.371310 / 1256.663849，几何均值 **1318.019746 NN rows/s**，
  spread **10.003269%**。
- 原始共同墙钟增幅−2.097131%，候选超过5%稳定门；**不采纳、不进入Worker性能测试**。
  event原始均值虽+9.575319%，候选spread37.994822%也失败，不能替代主口径。
8,000个原始f32事件、几何均值和稳定判定均独立重算；未剔除、补跑或重测。

保存数据的进一步分析发现两臂都存在明显的单次event时长变化，各lane总耗时
减事件区间总和只有约9.35–18.46ms。但源码是begin→主机model.apply提交→end，
区间可以包含等待主机提交时的GPU空闲，不能解释为纯GPU忙碌时间。未导出
每lane绝对起止时间，迭代编号也不是两流共同时间轴；现有数据不足以归因到
CPU、驱动、WSL调度或GPU。前后进程列表同样不能证明期间的竞争负载。

本Linux接入按性能门关闭，数值证据保留；生产50项不变，现用Windows
N128 Worker仍为 **1564.201170 RPC/s**。此前Windows/Fork共同墙钟100.186%
近似持平参考保持，没有新的Fork或Worker性能比值。643项既有外部内容及
2项新增实际运行依赖已核对。下一步回到现用Windows栈，先按已有压缩后热点
及试验历史审计新的GEMM/Attention组织；例如先检查classic N192/K32的真实
CUTLASS类型、完整映射和资源可行性，不能预设六warp/共享配置有效或会更快。
关闭项不原样重试，原精度与整网/Worker/稳定ABBA门保持，整体目标继续active。
证据：`wsl-n128-port-r1/{component-linux-r1/independent-review.json,native-linux-r1/review.json,
worker-linux-r1/review.json,binding-linux-r1/review.json,forward-abba-r1/review.json,
forward-abba-r1/independent-review.json,forward-abba-r1/saved-event-diagnostic.json,final-review.json}`。

**2026-09-16 QKV新布局可行性：默认N192关闭，N128 warp32通过组件验证，待整网。**

`qkv-classic-n192-audit-r1` 实例化CTA128×192×32、warp64×64×32的
实际CUTLASS类型：192线程、Params400字节、共享61440字节，CPU编译和
can_implement均成功。但完整ThreadMap导出发现输出每tile只覆盖12288/24576
个half，完整形状漏写2,909,952个half；A加载只覆盖3072/4096，漏25%，B完整。
默认映射中的8/3与24/16整数除法发生截断，不能仅因编译成功就运行。
现用N128对照的所有输入/输出映射完整。该精确默认N192方案在CPU阶段关闭，
未编译设备候选、未发射、未测速；这不是现用内核缺陷，也不排除未来自定义映射。

随后独立登记`qkv-n128-warp32-audit-r1`：保留现用CTA128×128×32、
grid40×9、三stage、49152字节共享，仅将warp64×64×32改为32×64×32，
每block128→256线程。完整A/B复制映射与5,822,208个输出half均各覆盖一次，
RoPE系数、Q/K/V边界和尾行掩码通过；实际400字节Params六指针重建通过。
每线程累加值128→64、epilogue迭代8→4，标量RoPE源码仅类型重命名，
FP16存储/FP32累加和原精确函数不变。

离线编译的新文件中，对照核完整5024条SASS及机器字与已安装N128一致；
候选2648条，32处HMMA全部F32，无LDL/STL、栈或spill。寄存器226→128。
驱动实际加载核验static shared均0、dynamic shared49152，两个方案均可驻留
2 CTA/SM；可驻留warp上限8→16。这是资源上限，不是实际占用率或性能收益。
cuobjdump的SHARED1024标记在原核也存在，不应当成额外静态共享分配。

冻结真实首层QKV输入的双lane组件、memcheck和racecheck各执行一次；
两核都受sanitizer检查，无kernel过滤。18份输出共104,799,744个half逐位一致，
NaN覆盖、2048字节边界、只读输入、候选重复发射均通过，零错误/零hazard。
独立复读全部原始字节通过。这里只验证首层冻结输入，不能替代原生整网和
真实Worker对C++FP32的原误差/100%首选门，尚未进行任何性能ABBA。

生产50项及六份Windows计划保持，现用N128 Worker历史验收仍为
1564.201170 RPC/s；Windows/Fork共同墙钟100.186%近似持平参考不变。
645项既有外部运行依赖已重新核对。下一步在隔离副本接入本warp32候选，
先完成B1/B14/B16双lane整网、W1/W32/W64真实Worker与计划失败关闭检查，
通过后才运行唯一同口径稳定ABBA；不开启旧Linux、N192或其他已关闭方案。
整体目标继续active，未宣称整网收益、未更改累加精度或生产认证。
证据：`qkv-classic-n192-audit-r1/review.json`及
`qkv-n128-warp32-audit-r1/{cpu-review.json,codegen-review.json,
loaded-resources-r1/report.json,independent-component-review.json,final-review.json}`。

**2026-09-16 N128 warp32整网与Worker数值全过；唯一前向ABBA未达1%，关闭。**

`qkv-n128-warp32-integration-r1`从当前Windows源代码复制332项到隔离目录，
增加独立B14候选开关`KATAGO_CUDA_QKV_N128_WARP32_R1`，强制依赖N128和
immutable QKV、默认关闭、计划显式绑定。CUBIN/Params/ABI及选择器源码均入
combined QKV指纹，正式50项文件与六计划不变。最初构建后发现N128也关闭时
新增日志误称N128路径，在首次整网运行前修正并重建；旧源码/二进制/指纹已
归档。最终39项plan与2项参数/缓冲CPU检查通过，20份PTX与原FFN2992条SASS
及全部机器字一致，strict/QKV三个旧CUBIN均保留，新CUBIN实际内嵌一次。

六组B1/B14/B16×两臂、各双lane的612,960个整网输出值与已接受N128参考
逐位一致，原五组head阈值、六通道100%策略首选全部通过。两臂始终N128=1，
仅新warp32开关0/1；实际B14 grid40×9、block128/256，B1/B16走已有回退。
33层FFN映射和23种压缩宽度真实路径通过。
真实Worker W1/W32/W64×两臂，768/768首选及11个C++FP32字段、Drain通过。
1536份原始request/result protobuf独立重建一致；六组均证明plan覆盖相反
环境值。13项错误计划实际CLI检查全部拒绝，包括旧二进制不识别新键、未绑定
环境、非法值、schema/模型/设备/库/构建/资产错误及N128依赖关闭。

验证脚本两处问题均保留原记录并修正CPU复核：Worker汇总列表被新增marker
循环同名变量覆盖，六份独立完成回执及原始结果完整，按原回执重建；错误计划
脚本的三条预期错误文本被连续替换写成错误键名，实际CLI已正确拒绝真实键。
首个拒绝结果复用原日志，其余12项各执行一次；没有重跑数值或放宽门槛。

唯一完整前向ABBA，Windows同binary/B14/S2/W80/N1000，仅warp32开关变化：
- baseline共同墙钟：1587.142945 / 1567.069985，几何均值
  **1577.074529 NN rows/s**，spread **1.280923%**。
- candidate共同墙钟：1573.143474 / 1575.392681，几何均值
  **1574.267676 NN rows/s**，spread **0.142975%**。
- 共同墙钟 **−0.177978%**；event诊断 **−0.261526%**，两种口径均稳定。
  未达1%采纳线，候选关闭，未进入Worker性能测试、未上线、未原样重测。
8,000个原始f32事件、均值/稳定性/计划安装顺序已独立重算。资源上限改善
没有形成可采纳的整网收益，不能据这个微小差值作具体性能归因。

当前Windows N128 Worker历史验收1564.201170 RPC/s与对Fork共同墙钟
100.186%近似持平参考保持；没有新Worker/Fork比值。645项外部依赖重验通过。
下一步按现有热点和实验目录审计不同的warp分工，例如同CTA128×128×32下的
warp64×32×32；先实际导出完整输入/输出映射与资源，不预设支持或加速。
所有已关闭项保持关闭，完整目标继续active，FP32计算和原数值门不变。
证据：`qkv-n128-warp32-integration-r1/{native-review.json,worker-golden-r1/review.json,
binding-r1/review.json,identity-review.json,forward-abba-r1/review.json,
independent-timing-review.json,closure.json,final-review.json}`。

**2026-09-16 N128 warp64×32整网与Worker数值全过；唯一前向ABBA仅+0.367%，关闭。**

`qkv-n128-warpn32-audit-r1`独立验证CTA128×128×32、warp64×32×32、stage3，
区别于此前关闭的warp32×64布局。完整A/B输入与输出/RoPE映射通过，全部
5,822,208个half输出各写一次；Params400字节及六指针位置已实际导出。
epilogue共享17,408字节，运行时动态共享49,152字节、静态共享0；256线程、
128寄存器、零栈/本地内存，occupancy API上限2 CTA/SM、16 warp/SM。
这只是资源上限，不能当作实测占用率。候选SASS 2656条、32处FP32 HMMA；
同包基线5024条指令和全部机器字与已安装四warp N128一致。
组件、memcheck、racecheck各一次，18份原始输出共104,799,744个half逐位一致；
覆盖两lane、重复执行、越界守卫、NaN覆盖和只读输入。组件固定系数仅用于
算子对拍；后续整网和Worker使用原生生成的RoPE，未注入黄金系数。

`qkv-n128-warpn32-integration-r1`复制332项当前源码到隔离目录，增加8项
资产，仅编辑3个源文件。独立开关`KATAGO_CUDA_QKV_N128_WARPN32_R1`默认0，
强制依赖immutable/N128，计划显式绑定，选择器与CUBIN/Params/ABI纳入指纹。
首次构建39项plan和2项参数/缓冲检查通过；20份PTX、原FFN2992条SASS及全部
机器字不变，旧strict/QKV资产保留，新CUBIN实际内嵌一次。前轮发现的日志与
验证脚本问题在本次冻结/执行前处理，本轮没有构建后修订或GPU重跑。

六组B1/B14/B16×两臂、各双lane的612,960个整网输出值与已接受N128参考
逐位一致，原head阈值和六通道100%策略首选全部通过。B14实际grid40×9、
block128/256，B1/B16走已有路径；33层FFN映射和23种压缩宽度真实执行。
真实Worker W1/W32/W64×两臂的768/768首选、11个C++FP32字段及Drain通过；
1536份原始request/result protobuf独立重建一致。六组均确认plan覆盖相反
环境值且先于选择器安装。13项错误计划在实际CLI全部正确拒绝。

唯一完整前向ABBA，Windows同binary/B14/S2/W80/N1000，仅新开关0/1：
- baseline共同墙钟1571.010864 / 1548.854111，几何均值
  **1559.893149 NN rows/s**，spread **1.430525%**。
- candidate共同墙钟1565.330488 / 1565.907413，几何均值
  **1565.618924 NN rows/s**，spread **0.036856%**。
- 共同墙钟 **+0.367062%**，event诊断 **+0.624728%**，两臂均过5%稳定门。
  未达1%采纳线，关闭候选，未运行Worker性能、未上线、未原样重测。
8000个原始f32事件与均值/稳定性独立重算通过。名义收益小于基线波动，
不能宣称稳定优势或据此作具体性能归因；event含主机发射间隙，亦非纯GPU忙时。

现用50项文件不变，Windows N128历史Worker验收1564.201170 RPC/s及Fork
共同墙钟100.186%近似持平参考保持，没有新Worker/Fork比值。645项外部依赖
重验通过，无新增管理员采集。两种N128八warp布局均保持关闭。
下一步先审计实验目录中不同的classic CTA128×256×32、warp64×64×32方案；
尚未登记或宣称可用。N1152需向上取整为5块及末尾128列掩码，必须先证明
完整输入/输出/RoPE映射、实际ABI与资源，再决定是否进入GPU验证。
完整目标继续active，FP16存储/FP32计算、原整网/Worker/唯一稳定ABBA门不变。
证据：`qkv-n128-warpn32-integration-r1/{native-review.json,worker-golden-r1/review.json,
binding-r1/review.json,identity-review.json,forward-abba-r1/review.json,
independent-timing-review.json,closure.json,final-review.json}`及对应audit目录。

**2026-09-16 classic N256完整数值通过；唯一前向ABBA稳定−4.877%，关闭。**

`qkv-classic-n256-audit-r1`检查独立CTA128×256×32、warp64×64×32、stage3。
目录模式匹配仅命中旧full-hidden FFN的H128/256记录，经实际geometry核对不是
相同QKV实验。候选完整A/B与输出/RoPE映射通过：M5054/N1152/K384，grid40×5，
全部5,822,208个half输出各写一次。末尾128列掩码实际验证；输出向量中仅M越界
9504、仅N越界80864、两者越界1056，均不进入有效输出。Q/K与V边界及系数
0..69311覆盖正确。Params400字节及六指针位置已导出，原标量RoPE/half边界不变。

实际main/kernel共享73,728字节、epilogue33,792字节；256线程、226寄存器、
零栈/本地内存，加载后静态共享0，occupancy API上限1 CTA/8 warp每SM。
当前N128为49,152字节、128线程、226寄存器、2 CTA/8 warp每SM。
新核SASS4880条、64处FP32 HMMA；同包基线5024条指令及全部机器字与已安装
N128一致。CTA从360降至200，同时列填充计算多11.111%；资源变化不是收益证明。
首次准备资源查询脚本时，生成器因替换文本匹配两处停止，保留原脚本，按精确
上下文补齐尚未写出的查询脚本后才编译，无内核变更或GPU重跑。
组件、memcheck、racecheck各一次，18份原始数组的104,799,744个half全部逐位
一致，双lane/重复输出、NaN覆盖、守卫及只读输入通过；853项实际编译输入绑定。

`qkv-classic-n256-integration-r1`复制332项正式源码到隔离目录，增加8项资产，
仅编辑3个源文件。新键`KATAGO_CUDA_QKV_CLASSIC_N256_R1`默认0，显式依赖
N128/immutable；CUBIN/Params/ABI及选择器参与组合指纹，plan覆盖相反环境值。
同一首次构建39项plan与2项参数/缓冲检查通过，20份PTX及FFN2992条指令和
全部机器字保持，旧strict/QKV三份CUBIN保留，新CUBIN实际内嵌一次。
候选实际B14 grid40×5/block256/shared73728，基线40×9/block128/shared49152。

六组B1/B14/B16×两臂、各双lane的612,960个整网输出值与已接受N128逐位一致，
原head阈值与六通道100%策略首选通过。真实Worker W1/W32/W64×两臂全部
768/768首选、11个C++FP32字段及Drain通过；1536份原始request/result protobuf
独立重建一致。33层FFN和23种宽度真实执行，12组路径记录与组合指纹独立复核。
13项错误计划实际CLI均正确拒绝，六组plan安装均先于选择器。

唯一完整前向ABBA，Windows同binary/B14/S2/W80/N1000，仅N256开关0/1：
- baseline共同墙钟1563.480552 / 1560.116409，几何均值
  **1561.797574 NN rows/s**，spread **0.215634%**。
- candidate共同墙钟1485.525822 / 1485.721132，几何均值
  **1485.623474 NN rows/s**，spread **0.013148%**。
- 共同墙钟 **−4.877335%**；event诊断 **−4.951299%**，两臂均通过5%稳定门。
  候选按性能门关闭，未运行Worker性能、未上线、未原样重测。8000个原始f32
  事件及均值/稳定性已独立重算；event含主机发射间隙，不是纯GPU忙时。
此实验说明这组更宽分块没有收益，不能单独区分填充、驻留与访存各自贡献。

现用50项文件保持，历史已接受Windows Worker1564.201170 RPC/s和Fork
共同墙钟100.186%近似持平参考保持；645项外部依赖重验，无新Worker/Fork比值。
下一步先刷新现用四warp N128+compact FFN的热点诊断：旧compact-forward-profile
在N128安装前采集。将层级同步计时、并发CUDA activity及共同墙钟分开，绑定
当前正式binary/plan/模型/实际路径，诊断不作性能验收、不申请或复用管理员采集。
按新证据选择不同机制；完整目标继续active，原精度与数值/性能门不变。
证据：`qkv-classic-n256-integration-r1/{native-review.json,worker-golden-r1/review.json,
binding-r1/review.json,identity-review.json,forward-abba-r1/review.json,
independent-timing-review.json,closure.json,final-review.json}`及对应audit目录。

**2026-09-16 正式N128热点刷新；无损连续2:4结构不足，M256 QKV组件通过待整网。**

`n128-current-profile-r1`绑定当前正式CLI c45ba0fc、同TF3模型及19项tactic，
核对50项正式产物。首次登记比较了正式与已接受构建的完整路径对象而拒绝；
两路径内容相同，改为同时验证两文件SHA及正式文件在保护集后才登记/采集，
保留原脚本和纠正记录。仅采集一次单流逐层诊断、一次普通权限双流CUDA
activity；未启用硬件计数器、CPU采样、上下文切换或事件完成跟踪。
逐层是6帧中的后3帧；双流各24次完整前向，每次302个kernel，逐个核对
48份相同符号/网格/资源序列，排除每流4次预热后保留40帧、12080个kernel。
两种诊断均核对33层压缩映射、23种实际宽度和N128 grid40×9/block128。
与旧compact时间线比较，302个位置中仅33个QKV内核签名发生变化。

双流**内核持续时间之和**占比：FFN up/SwiGLU **19.666%**，QKV/RoPE
**17.524%**，Attention核心 **15.458%**，FFN down **14.232%**，Attention
输出投影 **8.178%**；三段Attention合计41.160%，两段FFN合计33.898%。
原始SQLite独立重算：内核累计486.755641ms、时间区间并集347.653091ms、
选择窗口348.875608ms，流间重叠139.102550ms。重叠会重复计入累计值；
这些不是互斥墙钟占比、硬件利用率或吞吐验收，也不用于计算历史提速。
同步逐层均值Attention5.876ms、FFN4.910ms，含同步/提交开销，单列解释。

对绑定当前manifest的99个gate/up/down保留矩阵做CPU结构审计，共25,012,224
个half，其中零值260,861（约1.043%）。连续K方向每4项的非零数直方图为
[45909,6156,11929,34899,6154163]，完整满足每组至多2个非零的矩阵为**0/99**。
每个矩阵的反例均从原始half字节独立复核；这是当前排列的无损数据条件审计，
不把局部合格组当作完整稀疏GEMM，不删除任何非零、不改变权重或声称硬件支持。

据当前QKV热点，独立登记`qkv-classic-m256-audit-r1`：CTA256×128×32，
warp64×64×32、stage3。目录未发现同形状记录。与已关闭N256的扩N不同，
本项沿M扩大，grid20×9将CTA360→180，保持同样5120填充行/1152列，
减少重复B块加载且没有额外列填充计算；是否加速仍待整网实测。
完整A/B及输出/RoPE映射通过，5,822,208个half各写一次，仅M尾部9504个
向量被屏蔽，无N尾部；Params400字节、六指针位置与系数边界实际导出。
动态共享73,728字节、epilogue34,816字节、256线程/226寄存器、零栈/本地内存，
实际加载静态共享0，驻留上限1 CTA/8 warp每SM。5008条候选指令、64处FP32
HMMA；同包N128基线5024条和全部机器字与正式资产一致，853项编译输入绑定。
组件、memcheck、racecheck各一次，18份输出共104,799,744个half逐位一致；
双流、重复、NaN覆盖、guard及只读数据通过。原FP32计算和标量RoPE不变。

M256尚未接入整网、Worker或性能测试，不能采纳。下一步新隔离副本绑定
对应grid20×9/Params/资源，以原六组整网、真实C++FP32 Worker、错误计划和
唯一稳定ABBA门验证。N256与两种N128八warp等既有关闭项保持关闭。
现用Worker历史验收1564.201170 RPC/s及Fork近似持平参考不变；645项外部
内容重验通过。无生产修改、无新性能收益或Fork比值，完整目标继续active。
证据：`n128-current-profile-r1/{profile-review.json,actual-path-review.json,
independent-review.json,ffn-structure-r1/review.json,final-review.json}`与
`qkv-classic-m256-audit-r1/{cpu-review.json,codegen-review.json,
loaded-resources-r1/report.json,independent-component-review.json}`。

**2026-09-16 classic M256完整数值通过；唯一前向ABBA稳定−3.344%，关闭。**

`qkv-classic-m256-audit-r1`的CTA256×128×32、warp64×64×32、stage3候选
已完成上一阶段的完整映射、资源、组件与sanitizer门。M5054/N1152/K384，
grid20×9/block256，全部5,822,208个half输出各写一次；同现用N128保持
5120×1152填充算术量，CTA由360减至180，没有增加N方向填充。
实际共享73,728字节、226寄存器、零栈/本地内存，加载后的occupancy上限
1 CTA/8 warp每SM；现用N128为49,152字节、2 CTA/8 warp每SM。
候选5008条SASS、64处FP32 HMMA；同包基线5024条指令及两机器字完全一致。
组件、memcheck、racecheck各一次，18份数组104,799,744个half逐位一致。

`qkv-classic-m256-integration-r1`复制332项正式源码并增加8项候选资产，
仅修改隔离目录中3个源文件。默认关闭的新键`KATAGO_CUDA_QKV_CLASSIC_M256_R1`
经tactic_var读取并依赖immutable/N128；CUBIN/Params/ABI和选择器绑定组合指纹。
首次构建39项plan与2项参数/资产检查通过，无源码修订或GPU重跑。
20份PTX及FFN2992条指令和两机器字保持，旧strict/QKV资产保留，新核内嵌一次。
实际候选grid20×9/block256/shared73728；基线grid40×9/block128/shared49152。

六组B1/B14/B16×两臂、各双lane的612,960个整网输出值与已接受N128逐位一致，
原head阈值及六通道100%策略首选通过。真实Worker W1/W32/W64×两臂全部
768/768首选、11个C++FP32字段及Drain通过；1536份原始request/result protobuf
独立重建一致。33层FFN/23种宽度实际执行，12组路径与组合指纹复核通过。
13项错误计划实际CLI均正确拒绝；六组plan覆盖相反环境值且先于选择器安装。

唯一完整前向ABBA，Windows同binary/B14/S2/W80/N1000，仅M256开关0/1：
- baseline共同墙钟1553.241769 / 1561.744236，几何均值
  **1557.487201 NN rows/s**，spread **0.547401%**。
- candidate共同墙钟1500.301252 / 1510.537314，几何均值
  **1505.410583 NN rows/s**，spread **0.682267%**。
- 共同墙钟 **−3.343631%**；event诊断 **−3.289623%**，两臂均通过5%稳定门。
  候选按性能门关闭，未运行Worker性能、未上线、未原样重测。8000个原始f32
  事件及统计已独立重算；event含主机发射间隙，不是纯GPU忙时。
扩大M分块在此整网配置下变慢，不能据此单独判定驻留、寄存器或访存的贡献。

现用50项文件、已接受Windows Worker1564.201170 RPC/s及Fork共同墙钟
100.186%近似持平参考保持；645项外部依赖重验，本轮无新Worker/Fork比值。
下一步先对现用N128的K循环分组/调度做源码与历史实验审计，再决定新候选。
当前CUTLASS multistage在倒数第二个warp-MMA迭代推进复制和stage，不能
仅改K16就假定管线有效；先核实际类型、循环进展、完整映射及资源。旧CuTe K64
与FFN实验须按实际机制排重。本记录没有注册新的K分块候选。
精度仍是FP16存储、FP32计算与原精确激活；目标保持active。
证据：`qkv-classic-m256-integration-r1/{native-review.json,worker-golden-r1/review.json,
binding-r1/review.json,identity-review.json,forward-abba-r1/review.json,
independent-timing-review.json,closure.json,final-review.json}`及对应audit目录。

**2026-09-16 N128 K循环审计：K16直接替换不可行；源码延后等待被编译器提前，关闭。**

`qkv-n128-latewait-r1`绑定473份实验记录/顶层CUDA源码，核对旧CuTe K64、
FFN K64及两种stage2 QKV，未发现相同的延后等待机制。实际实例化N128的
warpK32/instructionK16为2次warp迭代；MmaBase同时要求迭代数>1且为偶数，
所以直接替换warpK16得到1次迭代不符合约束，未编译/启动K16内核。

独立候选保留CTA128×128×32、warp64×64×32、stage3，只替换主循环函数：
将已在寄存器中的最后一组MMA置于异步等待前，下一stage的共享读取仍在
wait/barrier后。原prologue、drain、复制/掩码、累加类型和标量RoPE保持。
CPU实际类型、400字节Params和全部输入/输出映射通过；模板与正式版本逐位相同，
5,822,208个half输出均覆盖一次。固定K384的源码依赖审计保持24组K序累加及
12次主循环commit/wait；这只是源码语义审计，不能代替实际机器码检查。

编译与加载确认候选224寄存器（基线226）、共享49,152字节、零栈/本地内存，
128线程、2 CTA/8 warp每SM上限保持。两侧均5024条SASS、64处FP32 HMMA；
同包基线及两机器字与正式N128一致，853项编译输入已绑定。
**实际两侧主循环均为0x16d0..0x1f80，wait均在第10条HMMA后**，HMMA操作数
顺序及共享读取序列相同。候选wait地址0x1b20，基线0x1b70；地址和少量调度
不同不代表延后等待机制生效。源码层移动被优化器提前，按机制检查关闭候选，
不推断它的真实性能影响。

关闭前已完成一次普通组件运行：六份原始数组共34,933,248个half与真实首层
QKV参考逐位一致，双lane/重复输出/守卫通过，CPU独立复核原始字节。
未运行memcheck/racecheck、整网、Worker或性能ABBA，不能称完整数值认证。
首次catalog脚本在访问无关WSL lib64链接时停止；改为先筛文件名后检查文件类型，
保留原脚本。修正发生在注册/源码生成/构建前，无内核修改或GPU重跑。

现用50项文件与645项外部依赖核验保持，Worker1564.201170 RPC/s及Fork
共同墙钟100.186%近似持平的历史参考保持，本轮无新增性能比值。
下一步审计固定K384的三stage循环分组能否减少地址/控制开销；保留原等待和
累加顺序，先确认历史排重及实际代码生成，不再对相同源码等待改写测速。
新分组候选尚未注册，FP16存储/FP32计算及所有数值/ABBA门保持，目标active。
证据：`qkv-n128-latewait-r1/{schedule-review.json,actual-schedule-review.json,
ordinary-numeric-review.json,closure.json,final-review.json}`。

**2026-09-16 固定K384三stage循环分组：完整数值通过，唯一前向+0.190%，未达门槛，关闭。**

`qkv-n128-k384-cycle3-r1`保留N128/K32四warp几何、原mac_loop_iter、
FP32累加顺序、wait/barrier和标量RoPE，仅将固定K384的12次stage迭代分为
4轮×3次。482份历史记录/顶层源码核查区分了旧DualFFN固定K（未改展开）和
Attention KV展开；没有复用已关闭的源码延后等待改写。
CPU实际类型、全部A/B与输出映射、400字节Params通过；ABI/模板与正式N128
逐位相同，5,822,208个half输出各覆盖一次。

这次机器码机制确实生效：候选主循环0x1da0..0x34b0，均匀分支计数器
7→4→1→−2→−5，执行4轮；每轮192处FP32 HMMA，总计768，与原12轮×64相同。
主循环按次数加权指令数1680→1480，wait仍12次。候选整核5312条SASS，
基线5024条；初始化增加，不能把循环指令下降当作吞吐收益。
实际寄存器254（基线226），共享49,152字节、零栈/本地内存，128线程，
2 CTA/8 warp每SM上限保持。基线两机器字保持，853项实际编译输入绑定。
组件、memcheck、racecheck各一次，18份数组104,799,744个half逐位通过。

`qkv-n128-k384-cycle3-integration-r1`隔离复制332项源码、增加8项资产，
只改3个源文件。新键`KATAGO_CUDA_QKV_K384_CYCLE3_R1`默认0并依赖N128，
资产与选择器绑定组合指纹；调用前保持B14及TN权重1152/384/384检查，固定参数
仅表示M5054/N1152/K384。首次构建39项plan与2项资产测试通过，20份PTX和
FFN2992条指令及两机器字保持。两臂实际grid40×9/block128/shared49152。
六组B1/B14/B16、各双lane，612,960个整网输出与正式N128逐位一致；原阈值
及六策略通道100%首选通过。真实Worker W1/W32/W64×两臂全部768/768首选，
11个C++FP32字段及Drain通过，1536份request/result protobuf独立重建一致。
13项错误计划CLI、12组路径/组合指纹及六次plan覆盖相反环境值检查通过。

唯一ABBA，Windows同binary/B14/S2/W80/N1000，仅新键0/1：
- baseline共同墙钟1589.433122 / 1565.890406，几何均值
  **1577.617849 NN rows/s**，spread **1.503471%**。
- candidate共同墙钟1582.275334 / 1578.959169，几何均值
  **1580.616382 NN rows/s**，spread **0.210022%**。
- 共同墙钟 **+0.190067%**，未达1%采纳线；event **−0.134639%**。
  两臂均通过5%稳定门；名义差小于基线波动。8000个原始f32事件独立重算。
  未运行Worker性能、未上线、未原样重测，不能单独归因于寄存器或初始化。
计划负例检查运行期间，编排提前调用的两次前向入口均在qualify阶段拒绝，
没有生成测速目录/contract/attempt或启动GPU；等待原负例进程完成后才注册
并执行上述唯一ABBA。该编排纠正保留在preflight-ordering-correction.json。

现用50项文件与645项外部依赖保持；历史Worker1564.201170 RPC/s和Fork
共同墙钟100.186%近似持平参考保持，无新Worker/Fork比值。
下一步审计现用Attention RMS的实际线程/读写映射及历史RMS候选，先判断
能否在保持原FP32归约树和half边界下减少访存/地址开销。该角色占当前内核
累计3.516%，不是墙钟收益上限或实测提速；新候选尚未注册，目标继续active。
证据：`qkv-n128-k384-cycle3-integration-r1/{native-review.json,
worker-golden-r1/review.json,forward-abba-r1/review.json,closure.json,final-review.json}`
及对应`qkv-n128-k384-cycle3-r1`组件目录。

**2026-09-16 RMS十六lane双元素映射：组件逐位与sanitizer通过，待隔离整网验证。**

`rms-sixteen-lane-vec2-r1`确认现用executor PTX完整嵌入正式CLI，RMS入口与
历史实际Driver JIT逐字节相同。现用每lane的12次输入读取已复用寄存器，
不存在源码第二轮循环造成的重复输入读；当前33个Attention RMS及33个FFN
RMS均为grid1264/block128/36寄存器，内核累计占比分别3.516%/2.058%。
历史8lane vec4、固定384、整块outproj+RMS及短窗口波动结论保持关闭。

新候选每行16线程、每线程两个相邻通道/24个输入float，每CTA8行；与已关闭
8lane方案的48float/线程、16行/CTA不同。保留两条12项FP32 FMA链，将原
XOR16/8/4/2/1依次映射为跨线程8/4/2/1及线程内一次加法，不改变加法结合。
动态ncols除法、rsqrtf、两次FP32乘法及RN-half边界保持。CPU逐项读写覆盖
与归约树检查通过；实际PTX输入/scale、累加、shuffle、除法及输出依赖独立
核对通过。适用有限实数输入，不额外声称NaN payload逐位等价。

机器码确有24条64位读取及12条32位写回；每warp两行，使每行全局读写指令
36→18，但传输数据量没有据此减少。实际寄存器36→44，零spill/stack/shared；
每SM上限12→10 CTA、48→40 warp，驻留CTA涵盖的行数48→80。
静态SASS264→328条，两臂actual Driver JIT与offline机器字一致；这些都不是
吞吐实测。307项编译输入已绑定。

26组保留真实输入/合成边界与epsilon、两lane、baseline/candidate/repeat：
普通、memcheck、racecheck合计468份数组、193,045,248个half逐位相同；
18项错误宽度拒绝通过，所有保护区/只读输入通过。普通进程156次实际调用的
函数及网格已核查，两项sanitizer零错误。最初memcheck遗漏本机既有named-pipes
设置，未能附加，在进入fixture前失败；保留记录并仅终止明确归属的诊断进程。
修正传输与30秒失败超时后完成两项检查，未改内核或参数。另保留编译输入
#line解析及PTX寄存器表示的CPU审阅器修正，均未重编译或重跑GPU数值。

本阶段未做性能测量、未运行整网/Worker候选、未上线；50项现用文件和645项
外部依赖保持。下一步仅在隔离副本接入default-off tactic，完成原生B1/B14/B16、
真实C++FP32 Worker W1/W32/W64及计划负例/路径门，才允许唯一完整前向ABBA。
仍须稳定共同墙钟至少1%后才能进入Worker性能与采纳。历史Worker1564.201170
RPC/s及Fork共同墙钟约100.186%的近似持平参考保持，整体目标继续active。
证据：`rms-sixteen-lane-vec2-r1/{cpu-review.json,offline-review.json,
ordinary-review.json,memcheck-review.json,racecheck-review.json,
independent-component-review.json,next-intake.json,final-review.json}`。

**2026-09-16 RMS十六lane完整验证与裁决：数值通过，唯一前向−0.428%，关闭。**

`rms-sixteen-lane-vec2-integration-r1`隔离复制332项现用源码，只改executor、
CUDA调用、tactic验证和原生dump测试四文件。新键`KATAGO_CUDA_RMS_E16V2_R1`
默认关闭，经tactic_var读取；启用时要求B1..16/S361/C384及完整缓冲，实际
输入/scale指针8字节和输出4字节对齐，与旧RMS=v1明确互斥。原12项FP32
累加链、归约结合及RN-half边界保持；旧路径保留，按物理batch记录实际网格。

首次构建39项plan和2项资产测试通过。SM120/SM89只在executor追加新入口，
所有旧入口逐字节相同，其他18份PTX不变；QKV/strict Attention CUBIN与FFN
2992条指令及两机器字保持。完整SM120模块中的两RMS机器码与组件阶段的
实际Driver JIT一致，实际模块查询仍36/44寄存器、12/10 CTA每SM、零本地/
共享存储。kernel及两项主机指纹重建并核对。首次检查器将旧SM89 CRLF转换
成LF而新文件保持原字节，触发比较失败；修正为两端原字节解码后通过，未
重编译或重跑指纹查询，原检查与修正均保存。

六组B1/B14/B16×两臂、各两lane：612,960个整网输出逐位相同，全部原阈值
及六策略通道100%首选通过。真实Worker W1/W32/W64×两臂768/768首选、
11个C++FP32字段及Drain通过，1536份请求/结果protobuf独立重建一致。
13项错误计划CLI与12组路径/指纹检查通过，六次实际Worker均证明plan值
覆盖相反环境值。独立时序审阅器在测速前纠正一个遗留QKV标记，没有改门槛。

唯一完整前向ABBA，同Windows二进制、B14/S2/W80/N1000，仅新键0/1：

- baseline共同墙钟1590.454104 / 1559.303280，几何均值
  **1574.801670 NN rows/s**，spread **1.997740%**。
- candidate共同墙钟1569.158292 / 1566.959898，几何均值
  **1568.058710 NN rows/s**，spread **0.140297%**。
- 共同墙钟 **-0.428178%**；event **-0.063622%**。
  两臂均通过5%波动门，但未达1%收益门；不将小幅负差单独归因于寄存器或
  主机检查开销。8000个原始f32事件独立重算。候选关闭、未原样重测。

未进入Worker性能或生产迁移；50项现用文件与645项外部依赖保持。
历史Worker1564.201170 RPC/s和Fork共同墙钟约100.186%的近似持平参考保持。
下一步审计Attention输出投影的实际cuBLASLt路径，以及classic CUTLASS TN
N128在M5054/N384/K384、FP32残差输出上的可行性，先与关闭的NN/TileLang/
整块融合候选逐机制核对；此处未登记新候选，未采用Fork的FP16累加。
证据：`rms-sixteen-lane-vec2-integration-r1/{native-review.json,
worker-golden-r1/review.json,forward-abba-r1/review.json,closure.json,final-review.json}`。

**2026-09-16 Attention输出投影classic TN N128：组件逐位与sanitizer通过，待整网验证。**

`outproj-classic-tn-n128-r1`保留现用TN half权重、FP32累加/残差/输出及
alpha=beta=1，采用classic CUTLASS CTA128×128×32、warp64×64×32、
四warp/三级预取、float4 epilogue、grid40×3；无RMS融合或新增half转换。
它与已关闭的NN布局、TileLang直接写回及整块N384+RMS方案不同。现用
完整模型日志确认对应cuBLASLt为id21/tile64×64/stages32×6；两次全模型
捕获绑定正式CLI、原始模型SHA和现用N128+compact FFN开关。独立相同
启发式查询冻结64字节算法；日志只公开id/tile/stages，未读取CLI内存中的
opaque数据，另由基线重放逐位复现实际边界输出来核对。

实际类型导出368字节Params，四个指针偏移及A/B和输出映射全覆盖，含
最后66行mask。机器码226寄存器、零stack/spill/local、48KiB动态共享，
每SM最多2CTA；PTX/SASS均为FP32累加。首版CPU报告序列化遇到NumPy
int32，修正类型后通过；首版资源检查误将离线SHARED=1024当作用户静态
共享，实际驱动函数属性为0、驱动每块保留量为1024，已保存原报告和修正。
另修正一个旧FFN固定算法标记预期，只继续未执行的第二次捕获，没有重跑
已成功捕获或改内核、ABI、数值门。708项内核和851项探针编译输入已绑定。

当前真实输入、全零输入、逐行基向量/尾块及确定性有符号随机四组，两个
独立流，baseline/candidate/repeat；普通、memcheck、racecheck三次合计
72份数组、**139,732,992个FP32输出逐位相同**。完整FP64条件误差界、
保护区/只读输入及重复运行检查全过，sanitizer零错误/零竞争。组件一致性
不能替代整网或Worker门，也不证明提速。没有性能运行、没有生产迁移。

50项现用文件与645项外部依赖保持；现用Worker1564.201170 RPC/s及Fork
共同墙钟约100.186%的近似持平参考保持，无新比值。下一步在隔离副本
注册默认关闭的OUTPROJ_CLASSIC_N128_R1，经原生B1/B14/B16、真实
C++FP32 Worker W1/W32/W64及计划/路径门后，才做唯一完整前向ABBA；
稳定共同墙钟至少1%才允许Worker性能。整体目标继续active。
证据：`outproj-classic-tn-n128-r1/{codegen-review-r2.json,
baseline-identity-review.json,independent-component-review.json,
next-intake.json,final-review.json}`。

**2026-09-16 Attention输出投影classic TN N128整网与Worker通过，安装并刷新Fork参考。**

`outproj-classic-tn-n128-integration-r1`从332项现用源码隔离接入新输出投影，
修改五个既有Rust文件，新增一个loader和五份AOT资产。默认关闭的新键
`KATAGO_CUDA_OUTPROJ_CLASSIC_N128_R1`经tactic_var读取；schema2、原始TF3
SHA、outproj_n128_artifact、实际SM120设备和六项依赖明确绑定。只在物理B14
发射grid40×3/block128；其他batch保留原路径。FP16输入/权重、FP32累加、
alpha=beta=1、FP32残差和输出不变。39项plan、2项资产测试通过；20份既有
PTX逐字节保持，FFN2992条指令及两个机器字保持，原strict/QKV AOT不变。

首版主机构建出现Arc借用错误，在任何模型运行前修正为`&stream`后编译通过，
原文件、失败日志及修正均保留。构建指纹因隔离路径改变；用Rust重建build.rs
实际路径表示及逐文件内容，精确重建新SHA，证明CUDA编译输入内容不变。
两次CPU路径字符串检查失败及修正全部留痕，没有重跑GPU数值或挑选测速样本。

六组原生B1/B14/B16×两臂、两lane：**612,960输出逐位相同**，原门和六策略
通道100%首选通过。真实Worker W1/W32/W64×两臂对C++FP32黄金参考
**768/768首选、11字段零违规、Drain通过**，1536份原始请求/结果protobuf
独立复建。18个错误计划CLI均按预期拒绝；实际路径、指纹及plan优先于相反
环境值验证通过。新路径没有新增half转换或采用Fork的FP16累加。

唯一完整前向ABBA，B14/S2/W80/N1000、同新Windows二进制、仅新键0/1：
baseline **1558.013679**、candidate **1580.700645 NN rows/s**，
共同墙钟 **+1.456147%**，两臂spread
**0.944256% / 0.027822%**；
event **+1.459239%**，也稳定。8000个原始f32事件独立重算通过。

随后唯一真实Worker ABBA，uncached、B14/S2/C64、ownership=false，
每轮256预热和8192计时请求：baseline **1557.022297**、
candidate **1574.822297 RPC/s**，**+1.143208%**；两臂spread
**0.853161% / 0.154437%**。
32768行原始时序、请求身份、计数器、延迟、模型和路径独立核对；未重测。

迁移前四组旧ONNX B1/B16×q64/q128通过ORT原门，两组graph首批发射通过；
默认/C32×W1/W32四组真实Worker共512/512和11字段、Drain通过。六份暂存
计划真实加载通过后安装19项确切测试字节：五处修改、一个loader、五AOT资产、
CLI、六Windows计划和配置注释；13个既有文件完整备份。仅吞吐计划启用新键，
另外五份显式0，WSL保持。正式路径六计划加载及吞吐128请求/Drain再次通过。
原GPU文件、ABI、host fingerprint与模型身份全部绑定；本轮没有管理员采集。

新一轮同边界ABBA：Rust **1595.982821**、Fork测试副本 **1563.247273 NN rows/s**，比值 **102.094074%**；两臂spread **0.176416% / 1.125760%**。这是Windows Rust与WSL Fork测试副本的统一墙钟参考；平台和累加精度仍有差异，不代表同精度优势或Fork Worker吞吐。

Fork原ELF与受控副本GPU fatbin保持、645项外部依赖前后核对，输入逐字节及
完成后收尾边界复核；4000个Rust原始event及Fork共同墙钟时间戳独立重算。
同模型SHA不代表内部权重位模式相同，Fork现用FP16累加仍单列；不把历史
2418.947898 event值与当前Worker RPC/s相除。整体目标继续active。
下一步刷新已安装组合的热点，再按原数值和唯一稳定ABBA门筛选新机制；
既有已关闭候选不原样重测。证据：
`outproj-classic-tn-n128-integration-r1/{native-review.json,worker-golden-r1/review.json,
forward-abba-r1/review.json,worker-performance-r1/review.json,promotion-r1/acceptance.json,final-review.json}`，
`outproj-current-fork-reference-r1/{common-wall-abba-r1/review.json,independent-review.json}`。

**2026-09-16 已安装输出投影组合热点刷新：仅诊断，无新性能认证。**

`outproj-current-profile-r1`绑定刚安装的CLI、模型、20项tactic和56项现用文件，
普通用户运行一次最小CUDA activity追踪；关闭CPU采样、上下文切换、event完成
追踪及all-API tracing，不使用硬件计数器或管理员权限。两流各4次预热+20次
前向，48份完整序列均为302个kernel，剔除8次预热后独立重建40次前向、
12,080条原始SQLite内核记录。对比上一版实际签名，仅33个Attention输出投影
替换，其余269个位置保持。新核实际grid40×3/block128/226寄存器、48KiB动态
共享、零static/local；QKV N128、strict Attention及33层compact FFN路径均匹配。

按跨两流的**内核持续时间之和**，当前热点为FFN up/SwiGLU
**19.273%**、QKV **17.348%**、
Attention核心 **15.480%**、FFN down
**14.287%**、Attention输出投影
**9.077%**。重叠内核分别计入，
这些不是独占墙钟占比或硬件利用率，也不与上一轮插桩时长相减推导收益。
另有单流同步逐层诊断，含主机/event开销，保持单列。复核原33层99份FFN
矩阵的实际half反例，仍无完整矩阵满足无损连续K4的2:4结构；没有剪枝候选。

已安装Worker **1574.822297 RPC/s**及新Fork共同墙钟比值 **102.094074%**
保持，Windows/WSL和精度差异不变。本阶段没有新的ABBA、内核编译或生产改动。
56项现用文件、645项外部依赖核对通过。下一步仅审计输出投影M64/N128/K32
几何与既有M64 QKV、N64残差及其他关闭路径的实际差异，再判断CPU/ABI/
资源可行性；尚未登记设备候选，不承诺收益。原数值和唯一稳定ABBA门保持，
整体目标active。证据：`outproj-current-profile-r1/{profile-review.json,
actual-path-review.json,independent-review.json,next-intake.json,final-review.json}`。

**2026-09-16 Attention输出投影M64/N128：组件逐位与sanitizer通过，待整网。**

`outproj-classic-tn-m64n128-r1`在现用TN M128/N128输出投影之上，独立检查
CTA64×128×32、warp32×64×32、四warp/三级流水、标准float4 epilogue。
保留原half输入/权重、FP32累加、alpha=beta=1与FP32残差/输出，不含RMS融合。
历史核查确认关闭的M64 QKV是N64/half+RoPE输出，旧残差探针CTA M128，
NN/TileLang布局及完整N384/RMS另有计算组织；本项不是原样复测它们。

708份实际编译输入绑定。CPU从实际类型导出368字节/align8 Params，四指针
64/112/192/272，完整A/B及float4输出映射通过；1,940,736有效输出各写一次，
尾部mask由25,344降至768。grid40×3变79×3，CTA120增至237。PTX与SASS
均为FP32 MMA，无新增half转换，无stack/local/spill。实际驱动加载资源：
基线226寄存器/48KiB动态共享，候选146寄存器/36KiB，用户静态共享均0。
cuobjdump离线SHARED=1024单列为驱动保留量；**两边每SM仍最多2CTA**，
没有驻留数量提升。更多CTA及更少块内复用可能抵消资源节省，不能推导提速。

从当前正式CLI和20项tactic重新捕获完整模型第3/4边界输入、残差与输出，
核对模型SHA、build及实际N128投影路径。探针直接加载现用不可变N128 CUBIN
及新M64/N128 CUBIN，各绑定自己的Params；不再用cuBLASLt启发式作为基线。
基线重放必须逐位复现当前模型边界输出。当前真实输入、全零输入、逐行基向量/
尾块、有符号随机四组，两条独立live stream，baseline/candidate/repeat：
普通、memcheck、racecheck共72数组、**139,732,992个FP32输出逐位相同**。
完整FP64条件误差界、重复/跨lane一致性、只读输入和保护区通过；sanitizer
零错误/零竞争。204份探针编译输入已绑定，独立审阅器重新读取全部输出检查。

没有新性能ABBA、候选整网/Worker运行或生产改动；56项现用文件和645项外部
依赖保持。已安装Worker1574.822297 RPC/s及Fork共同墙钟比102.094074%的
参考保持，平台/累加精度差异继续单列。下一步隔离接入默认关闭的新M64N128
tactic并绑定实际选择与资产，原生B1/B14/B16和真实C++FP32 Worker门通过后，
才允许唯一完整前向共同墙钟ABBA；稳定至少1%后才能进入Worker性能。
整体目标active。证据：`outproj-classic-tn-m64n128-r1/{history-review.json,
cpu-review.json,codegen-final-review.json,resource-query.json,
independent-component-review.json,next-intake.json,final-review.json}`。

**2026-09-16 Attention输出投影M64/N128整网裁决：稳定+0.026%，关闭。**

`outproj-classic-tn-m64n128-integration-r1`把已过组件门的TN
CTA64×128×32/warp32×64×32、四warp/三级流水投影接入隔离副本。
新键`KATAGO_CUDA_OUTPROJ_M64N128_R1`默认关闭，经tactic_var读取，
要求原OUTPROJ_CLASSIC_N128_R1=1及原模型/设备/布局依赖；组合指纹绑定
两份CUBIN、Params、ABI及选择代码。B14实际grid79×3与原40×3分别确认，
B1/B16回退原路径，FP16输入/权重和FP32累加、残差、输出保持。

隔离构建、40项plan单测、3项资产/别名/Params单测通过。20份原PTX、
原FFN的2992条双机器字指令均保持，五份嵌入CUBIN身份通过；构建路径
引起的kernel_build_id变化已从全部原输入和Rust路径独立重建。
六组B1/B14/B16双lane整网**612,960个输出逐位相同**，六个policy通道
首选全过。真实W1/W32/W64两臂共**768/768首选、11个C++FP32字段及
Drain通过**，1536份请求/结果protobuf独立重构。计划0/env1与计划1/env0
均实际执行正确；18项非法身份/模型/设备/依赖负例均拒绝。旧ONNX四局面
对ORT FP32为RESULT: PASS，B1/B14 graph及默认无CUDA构建通过。

唯一B14/S2/W80/N1000完整前向ABBA，共同墙钟
**1597.915012→1598.326247 NN rows/s（+0.025736%）**，基线/候选
spread **0.780621% / 0.557598%**，稳定但远低于1%收益门。
辅助event **1608.171519→1608.415728（+0.015186%）**，spread
0.622428%/0.187176%，也无足够收益。8000原始event和墙钟公式独立重算，
所有样本保留，没有重测、Worker性能测试或迁移。

**关闭该三级流水M64/N128候选**。56项现用文件及645项外部依赖保持；
已安装M128/N128投影的Worker1574.822297 RPC/s和最近Fork共同墙钟
比102.094074%仅为原参考，未由本候选刷新。Windows/WSL与累加精度差异
继续单列。下一步先审计两级流水是否具有未测试的实际机制与资源差异，
不得从更少共享内存直接推导提速，也不重新运行本项。整体目标active。
证据：`outproj-classic-tn-m64n128-integration-r1/{native-review.json,
worker-golden-r1/review.json,binding-r1/review.json,identity-review.json,
compatibility-r1/review.json,forward-abba-r1/review.json,
independent-timing-review.json,next-intake.json,final-review.json}`。

**2026-09-16 M64/N128两级寄存器流水：组件通过，待整网验证。**

`outproj-m64n128-stage2-r1`核查两级流水的实际实现后，使用独立候选：
CTA64×128×32、warp32×64×32、四warp、标准float4 epilogue、FP32残差输出。
CUTLASS的RowMajor stage2特化实际选择**MmaPipelined**，全局数据先加载
到寄存器再写共享；编译期完整类型断言与PTX/SASS均确认没有cp.async。
这与已关闭三级M64/N128的异步复制主循环不同。历史两级QKV为M128/N64、
N1152及half/RoPE输出，分别有负收益，不能将其当成本项结果；14份有界
历史源码及两份精确关闭记录已审计，不宣称搜索覆盖任意嵌套文件。

708份实际编译输入绑定。CPU导出完整368字节/align8 Params，指针仍为
64/112/192/272；完整A/B和FP32输出映射通过，1,940,736有效元素各写一次，
尾部mask768。链接代码保持FP32 HMMA，无新增half输出转换，无stack/local/
spill。实际加载**158寄存器、24KiB动态共享、每SM最多3CTA**；现用
M128/N128为226寄存器、48KiB、2CTA。离线SHARED1024为驱动保留量，
用户静态共享0。较高潜在驻留量不是实测提速，寄存器搬运和更短预取可能抵消。

从现用CLI及20项tactic重新捕获模型第3/4边界，身份、输入及实际投影路径
通过。直接重放现用M128/N128 CUBIN必须逐位还原模型输出；候选另加载
自己的不可变CUBIN和Params。真实输入、零输入、基向量/尾块、有符号随机，
两独立live lane下baseline/candidate/repeat，普通、memcheck、racecheck
共72数组、**139,732,992个FP32输出与现用内核逐位相同**。完整FP64
条件误差界、重复/跨lane一致性、保护区与只读输入通过，sanitizer零错误/
零竞争；204份探针编译输入绑定，独立审阅器复读全部原始数组。

最初CPU文件目录审计遇到无关WSL lib64链接的Windows stat错误，已在任何
候选注册或构建前修正为先筛扩展名，原脚本保留；未因此重跑GPU或性能测试。
没有候选整网、Worker或性能ABBA，未采纳；56项现用文件及645项外部依赖
保持。现用Worker1574.822297 RPC/s及Fork共同墙钟比102.094074%保留为
原参考，平台/精度差异继续单列。下一步隔离接入默认关闭的新S2 tactic，
通过整网、真实Worker和实际路径门后才允许性能验证。整体目标active。
证据：`outproj-m64n128-stage2-r1/{history-review.json,mechanism-review.json,
cpu-review.json,resource-query.json,independent-component-review.json,
next-intake.json,final-review.json}`。

**2026-09-16 M64/N128两级寄存器流水整网裁决：稳定−0.519%，关闭。**

`outproj-m64n128-stage2-integration-r1`将MmaPipelined两级候选接入隔离副本，
新键`KATAGO_CUDA_OUTPROJ_M64N128_S2_R1`默认关闭并经tactic_var读取。
要求现用OUTPROJ_CLASSIC_N128_R1=1及其模型/设备/布局依赖，组合资产指纹
绑定现用M128/N128与新S2的CUBIN、368字节Params、ABI和主机选择。
实际B14候选grid79×3、128线程、158寄存器、24KiB共享；基线grid40×3。
B1/B16沿用原路径，FP16输入/权重和FP32累加、残差及输出不变。

构建、40项计划单测、3项资产单测、原FFN2992条双机器字指令检查通过。
初始CPU审阅器因假定重复Cargo产物相同而停止：**CLI的20份PTX全部与现用
版本相同**，native测试的19份相同，另一个模块12个函数中的11个相同，
唯一不同项为`hgemm_v2_gatesilu_f32_kernel`。它唯一的调用受fuse_down
控制，仅FUSION=all/down开启。六次实际native均注册并打印FUSION=none，
差异函数未走；真实Worker数值与性能基准均使用20份PTX一致的CLI。
审阅器随后按每个二进制实际嵌入内容绑定；未重建、替换或重跑GPU，原失败
与诊断源码保留。不据源码哈希声称编译产物全部一致，也不声称已证明两版
差异函数语义等价。五份AOT CUBIN均在CLI/native各嵌入一次。

六组B1/B14/B16双lane整网**612,960个输出逐位一致**，六policy通道
首选全过。真实W1/W32/W64两臂共**768/768首选、11个C++FP32字段、
Drain通过**；1536份请求/结果protobuf独立复核。计划与环境相反时选路
正确，18项非法身份/模型/依赖负例拒绝，旧ONNX四局面对ORT为RESULT: PASS，
B1/B14 graph和默认构建通过。所有编译与读者差异在首次性能运行前处理。

唯一B14/S2/W80/N1000完整前向ABBA共同墙钟
**1598.723118→1590.429336 NN rows/s（−0.518775%）**，两臂spread
**1.664567% / 0.513332%**，稳定负收益。辅助event
**1609.875744→1598.041639（−0.735094%）**，spread1.018712%/0.072457%，
同样稳定。8000原始event与墙钟公式独立复算；未测Worker性能、未迁移。

**关闭两级寄存器流水候选**，潜在驻留3CTA未转化为整网收益，不能据此单独
归因于某条指令。56项现用文件和645项外部依赖保持。现用Worker1574.822297
RPC/s与Fork共同墙钟比102.094074%仍为之前的参考，平台/累加精度差异单列。
下一步先审计显式异步两级MmaMultistage是否有不同的实际机制与资源收益；
不重试本项或旧QKV两级候选，整体目标active。
证据：`outproj-m64n128-stage2-integration-r1/{inspection-ptx-variance-review.json,
native-fusion-exclusion-review.json,native-review.json,worker-golden-r1/review.json,
binding-r1/review.json,compatibility-r1/review.json,forward-abba-r1/review.json,
independent-timing-review.json,next-intake.json,final-review.json}`。

**2026-09-16 M64/N128显式异步两级流水：组件通过，待整网验证。**

`outproj-m64n128-async2-r1`显式构造MmaMultistage两级主循环，保留
CTA64×128×32、warp32×64×32、四warp、原A/B迭代器与FP32 warp算子、
标准float4残差epilogue。与已关闭的两级MmaPipelined寄存器搬运路径不同，
实际PTX含cp.async，三处wait_all对应等待深度0，SASS为三处DEPBAR0。
31份有界顶层历史源码及关闭记录已审计；旧异步两级QKV为M128/N64、
N1152和half/RoPE输出，不能代替本项结果。

708份实际编译输入绑定；368字节/align8 Params、指针64/112/192/272、
完整A/B映射及FP32输出覆盖通过，1,940,736个有效元素各写一次、mask768。
实际加载为**146寄存器、24KiB动态共享、每SM最多3CTA**，用户静态共享0，
无stack/local/spill；现用M128/N128为226寄存器、48KiB、2CTA。
所有MMA仍FP32累加，无新增half输出转换。资源改善尚不代表性能收益。

使用现用CLI和20项tactic新采集第3/4边界，模型、输入、构建身份及实际
投影路径均核验。直接重放现用M128/N128不可变CUBIN/Params，逐位还原
整网边界输出。真实输入、零输入、基向量/尾块、有符号随机，两独立live
lane下baseline/candidate/repeat，经普通、memcheck、racecheck共72数组、
**139,732,992个FP32输出与现用内核逐位相同**。完整FP64条件误差界、
重复及跨lane一致性、只读输入/保护区均通过，sanitizer零错误/零竞争；
204份探针编译输入绑定，独立审阅器复读全部原始数组。

初始CPU审阅器只接受wait_group0字面形式，实际CUTLASS零等待特化为
wait_all；按原始helper源码和SASS修正读法，保留初版。紧随其后的资源
查询因缺少审阅文件在CUDA初始化前停止；未重建内核，未因此重跑GPU。
本项尚无候选整网、真实Worker或性能ABBA，未采纳。56项现用文件和645项
外部依赖保持；现用Worker1574.822297 RPC/s及Fork共同墙钟比102.094074%
继续作为原参考，平台/精度差异仍单列。下一步以默认关闭AS2 tactic隔离
接入，先过整网/Worker/路径门，再做唯一完整前向ABBA。整体目标active。
证据：`outproj-m64n128-async2-r1/{history-review.json,mechanism-review.json,
wait-instruction-reader-correction.json,independent-component-review.json,
next-intake.json,final-review.json}`。

**2026-09-16 M64/N128显式异步两级整网裁决：−0.240%，关闭。**

`outproj-m64n128-async2-integration-r1`在当前已安装版本的隔离副本接入
默认关闭的`KATAGO_CUDA_OUTPROJ_M64N128_AS2_R1`，经tactic_var读取，
要求原OUTPROJ_CLASSIC_N128_R1=1及模型/设备/布局依赖。组合资产指纹
绑定基线/候选CUBIN、Params、ABI和主机选择。B14实际候选grid79×3、
128线程、146寄存器、24KiB共享，MmaMultistage两级异步复制；基线
grid40×3、226寄存器、48KiB。B1/B16沿用原路径，FP16存储和FP32累加、
激活、归约、残差与输出保持。

构建、40项计划测试、3项资产测试通过；FFN2992条指令的两个机器字保持。
本次CLI与native测试各自实际嵌入的**20份PTX全部逐位等于现用版本**，
没有沿用上轮差异函数的豁免。五份AOT CUBIN各嵌入一次，新候选与已过
组件验证的资产相同，全部路径与组合指纹独立重构通过。

六组B1/B14/B16双lane整网**612,960个输出逐位一致**、六policy通道
首选全过。真实W1/W32/W64两臂**768/768首选及11个C++FP32字段**、
Drain通过，1536份请求/结果protobuf逐份复核；计划与相反环境变量选路
正确，18项非法身份/依赖负例拒绝。旧ONNX四局面对ORT为RESULT: PASS，
默认构建及B1/B14 graph回归通过。

唯一B14/S2/W80/N1000完整前向ABBA共同墙钟
**1598.886887→1595.042350 NN rows/s（−0.240451%）**，两臂spread
**1.555528% / 0.466449%**，通过5%波动门但未达到1%收益门。
辅助event为1605.968939→1601.694323（−0.266171%），spread
1.140442%/0.256655%；8000原始event和墙钟公式已独立复算。
没有Worker性能测试、迁移或采纳，不能把潜在3CTA驻留量当作收益。

此候选关闭，现用56项文件和645项外部依赖保持。现用Worker1574.822297
RPC/s及Fork共同墙钟比102.094074%仍为原参考，平台/累加精度差异单列。
三级异步、两级寄存器和两级显式异步三项M64/N128均未过完整前向收益门，
不原样重测。下一步转向当前最大热点compact DualFFN up的有界源码/历史
审计，先确认33层实际宽度、已关闭机制及未覆盖的具体差异，再决定候选。
整体目标active。证据：`outproj-m64n128-async2-integration-r1/{compiled-ptx-review.json,
native-review.json,worker-golden-r1/review.json,binding-r1/review.json,
compatibility-r1/review.json,forward-abba-r1/review.json,independent-timing-review.json,
next-intake.json,final-review.json}`。

**2026-09-16 compact DualFFN half8寄存器输出重排：组件通过，整网/Worker待验证。**

`compact-ffn-shuffle-epilogue-r1`针对现用FFN up热点（原诊断内核累计
19.273%，不是新测速）保留原CTA128×64×32、warp64×32、三stage、swizzle2、
两路FP32累加和原精确SiLU/half边界，仅用四lane整数蝶形交换重排最终
half位模式并写出half8。原输出共享重排的64条STS.64、32条LDS.128及
16次CTA同步被32条SHFL.BFLY替换，8条STG.E.128保持；主循环末尾的
cp.async drain及同步保持。寄存器236→234，动态共享仍48KiB，设备查询
两者均每SM最多2CTA，静态共享/local/stack/spill均0。额外选择和地址指令
可能抵消收益，不能据此宣称更快。

有界审计覆盖665份顶层源码/脚本和77份命名历史决定，区分旧register-SiLU
共享输出、pingpong共享输出及FFN down float2直接存储；不原样重开它们。
694份编译输入绑定，当前对照原核2992条指令及两机器字与已采纳版本相同。
CPU检查tile8192元素各写一次、1024个16B写入、368种B1–B16/23宽度映射；
编译导出的69组B1/B14/B16 Params均720字节、align8、指针64/112/360/640。

现用CLI/20tactic新采集全部33个B14 FFN边界，权重、映射、输入和执行
身份通过。另用逐行重映射覆盖全部23宽度的B1/B16，并加入零输入、基向量、
随机用例，共82例。B1/B16重映射是组件样本，不是整网验证。两个独立流的
baseline/candidate/repeat，经普通、memcheck、racecheck三种运行方式，
累计1476数组、**3,517,341,408个half输出逐位一致**，33层真实边界也逐位
还原。每个输出通过FP64条件误差包络；重复、跨流、跨检查模式、只读输入
与保护区均通过，sanitizer零错误/零竞争。199份探针编译输入绑定，独立
审阅器复读全部输出。样本中最大gate/up绝对值11.805/15.945。

首版CPU导出器的packed-half代理访问/缺失string头已在设备执行前修正；
首版采集脚本对manifest不存在name字段的检查也在首次GPU启动前修正，
改为核对原有三个packed SHA，原件和修正记录保留。

本项尚无候选整网、真实Worker或性能ABBA，未采纳。56项现用文件和645项
外部依赖保持；现用Worker1574.822297 RPC/s及Fork共同墙钟比102.094074%
仍为原参考，平台/精度差异单列。下一步默认关闭FFN_SHUFFLE tactic仅隔离
接入B14，先过整网/真实Worker/身份与路径门，再做唯一完整前向ABBA。
整体目标active。证据：`compact-ffn-shuffle-epilogue-r1/{cpu-review.json,
codegen-review.json,resource-query.json,independent-component-review.json,
next-intake.json,final-review.json}`。

**2026-09-16 compact DualFFN half8寄存器重排整网裁决：+0.480%，未达门槛，关闭。**

`compact-ffn-shuffle-epilogue-integration-r1`在当前安装版本的隔离副本中接入
默认关闭的`KATAGO_CUDA_FFN_SHUFFLE_R1`，经tactic_var读取，依赖
FFN_COMPACT_R1及原模型/设备/布局条件。B14的23种实际宽度使用同一只读
CUBIN及原720字节Params模板，仅补入四个指针；B1/B16保留原路径。
扩展ffn_compact_artifact绑定候选源码、CUBIN、ABI、模板与主机代码，
两臂共用同一新CLI，仅切换该键。主循环、FP32累加及激活、half舍入边界保持。

候选保留128×64×32 tile、三级流水、48KiB共享内存；234寄存器、无spill，
每SM最多2CTA。移除共享epilogue，改为四lane蝶形重排和half8写出。
构建、40项计划测试及2项资产测试通过；原FFN2992条指令的两个机器字保持。
CLI和native测试各自20份PTX全部逐位保持，五份AOT CUBIN逐一绑定。

六组B1/B14/B16双lane整网**612,960个输出逐位一致**，六policy通道首选通过。
真实W1/W32/W64两臂**768/768首选及11个C++FP32字段**、原始protobuf和
Drain通过；相反环境变量证明计划优先，19项非法计划/身份/依赖拒绝。
实际时间线8帧共2416次kernel：候选/基线各132次FFN，逐帧其余269个
kernel位置与签名保持。默认构建、旧ONNX四局面ORT、B1/B14 graph通过。

唯一B14/S2/W80/N1000完整前向ABBA共同墙钟
**1600.113349→1607.796081 NN rows/s（+0.480137%）**，两臂spread
**0.483139% / 0.017927%**，波动通过5%门但收益未达1%门。
辅助event为1608.568888→1611.307693（+0.170263%），spread
0.112062%/0.052294%；8000原始event及墙钟公式已独立复算。
此候选关闭，无Worker性能测试、迁移或采纳，不原样重测。

现用56项文件与645项外部依赖保持；现用Worker1574.822297 RPC/s和
Fork共同墙钟比102.094074%仍是原参考，平台/累加精度差异继续单列。
下一步先做原DualMma寄存器直接half2写出的CPU/历史/机器码审计，检查
能否消除重排代价并保留合并访存；这是待验证假设，不预报性能收益。
原整体目标active。证据：`compact-ffn-shuffle-epilogue-integration-r1/`
下`native-review.json`、`worker-golden-r1/review.json`、
`all33-paths-r1/review.json`、`forward-abba-r1/review.json`、
`independent-timing-review.json`、`next-intake.json`和`final-review.json`。

**2026-09-16 compact DualFFN直接half2写出：组件通过，整网/Worker待验证。**

`compact-ffn-half2-direct-r1`保持原DualMmaMultistage、128×64×32 tile、
warp64×32、三级流水、swizzle2和两路FP32累加，仅让每个lane把自己持有的
最终half对直接写出。原Count8投影舍入/FP32精确SiLU与乘法/final half边界
保持；每组相邻4lane写16连续字节。相对已关闭half8方案，实际机器码移除
32条SHFL.BFLY和96条SEL，但8条STG.E.128变为32条STG.E，地址/边界
判断增加。主循环warp ID广播的1条SHFL.IDX保持。不能把静态指令减少当作提速。

编译与驱动查询均为234寄存器、48KiB动态共享、0 local/stack/spill，
每SM最多2CTA；原基线236寄存器、2992条指令的两个机器字与现用版本保持。
694份内核编译输入和199份探针编译输入绑定，原FP32 MMA及精确除法保持。
CPU证明8192个tile元素各写一次、4096次4字节写出、368种B1–B16/23宽度
及尾部/对齐；69组编译导出Params与前版逐位一致（720字节、align8、
四指针偏移64/112/360/640）。主循环末尾async drain和同步未经改动。

复用仍与现用CLI/plan/模型及56项文件相符的33层真实B14输入、权重与输出，
新增候选实际执行，未复用旧候选测试结果。普通运行和memcheck各82例，
覆盖33层B14、23宽度B1/B16行重映射及零/基向量/随机样本；racecheck预注册
26例覆盖23宽度B1、B14最窄/最宽和B16最宽。竞争检查范围依据未改动的完整
共享主循环及不访问共享内存的新输出段；全82例仍全部做memcheck。
两独立流×baseline/candidate/repeat共**1140数组、2,449,451,424个half值
逐位一致**，FP64误差包络、真实参考、保护区与只读输入均通过；内存/竞争
检查零错误。每模式输出复读后按原始文件哈希交叉复核。

准备阶段的原源码换行归一化被SHA检查拦截并改为精确字节复制；机器码审阅器
最初把主循环SHFL.IDX也计入“输出无shuffle”而停止，随后仅修正检查范围。
两次修正均在候选GPU执行前完成，没有修改已编译内核或重复测速。

本项没有整网、真实Worker或性能结果，尚未采纳。现用56项文件、645项
外部依赖和原Fork参考保持。下一步只在新隔离副本中默认关闭接入
FFN_HALF2_R1，完成全模型/Worker/身份与路径门后再允许唯一性能ABBA。
整体目标active。证据：`compact-ffn-half2-direct-r1/{cpu-review.json,
codegen-review.json,resource-query.json,independent-component-review.json,
next-intake.json,final-review.json}`。

**2026-09-16 compact DualFFN直接half2整网裁决：+0.742%，未达门槛，关闭。**

`compact-ffn-half2-direct-integration-r1`从当前安装源码新建隔离副本，
默认关闭的`KATAGO_CUDA_FFN_HALF2_R1`仅选择B14的23种compact宽度。
FfnCompact持有一次加载的只读CUBIN/720字节Params，组合artifact绑定
候选、原模板、ABI及主机代码，两臂共用同一CLI，仅该键0/1变化。
234寄存器、48KiB、原swizzle2、FP32累加/激活/归约和half边界保持。

40项计划与2项资产测试通过；原FFN2992条指令的两个机器字保持。
CLI和native测试的20份PTX各自全部逐位一致，五份AOT资产身份通过。
B1/B14/B16两臂双lane共**612,960输出逐位一致**，六policy通道首选通过；
真实W1/W32/W64两臂**768/768首选及11个C++FP32字段**、原始protobuf、
Drain及相反环境变量下计划优先通过。19项非法计划/身份/依赖正确拒绝。
实际8帧2416次kernel证明候选与基线各132次FFN，逐帧其余269个位置与
签名保持。默认构建、ONNX四局面ORT、B1/B14 graph兼容通过。

唯一B14/S2/W80/N1000完整前向ABBA共同墙钟
**1592.551348→1604.374077 NN rows/s（+0.742377%）**，两臂spread
**0.745434% / 0.523101%**，通过5%稳定门但未达1%收益门。
辅助event为1603.791674→1612.161560（+0.521881%），spread
0.295163%/0.153426%；8000原始event和共同墙钟公式已独立重构。
没有Worker性能测试、迁移或采纳。half8与直接half2两种FFN输出改写均关闭，
不能跨轮对比其绝对速度或叠加收益，也不原样重测。

现用56项文件与645项外部依赖保持；现用Worker1574.822297 RPC/s和
原Fork共同墙钟比102.094074%仍是原参考，平台/精度差异单列。
历史核查确认strict Graph、固定K384已关闭，QKV整头删除也已排除。
下一步有界审计每层重复的主机控制：当前循环内重复读取FUSION、格式化
一次性标记并计算融合布尔值。先确定可移出循环的范围、计划优先级与
调用间环境变化语义，再考虑默认关闭的单项实现；静态次数不代表性能瓶颈。
整体目标active。证据：`compact-ffn-half2-direct-integration-r1/`
下`native-review.json`、`worker-golden-r1/review.json`、`all33-paths-r1/review.json`、
`forward-abba-r1/review.json`、`independent-timing-review.json`、
`next-intake.json`和`final-review.json`。

**2026-09-16 每次前向融合配置/一次性日志的CPU审计完成，未登记整网候选。**

`per-apply-host-control-audit-r1`复用现用版本六帧逐层记录：每次前向179层。
源码中FUSION在每层读一次、复制String，并在OnceLock判断之前构造日志；
Fork所读MatMulLayer/initial-conv片段则使用成员选择和先判断日志标志，
不据此声称Fork所有配置都只在加载时查询。

独立Rust控制探针直接提取原`tactic_var`、日志函数和融合表达式，模拟179次
控制循环；它**没有模型、CUDA调用、GPU工作或真实Worker**。候选原型每次
控制前向读一次、只在OnceLock闭包内打印。两组分别验证计划覆盖环境、
调用之间配置变化、缺省/未知/非Unicode环境值行为及双线程读取；独立八线程
OnceLock构造测试为一次。未把这些CPU测试写成计划绑定或整网数值认证。

单独分配计数构建显示已安装计划下**358→1次分配**，6802→4字节；仅环境
路径537→2次，13604→42字节。未插桩构建按预登记双线程ABBA各运行一次：
已安装计划原控制段每lane每次**14.030–16.929微秒**，原型0.110–0.111微秒；
环境路径原控制段44.233–44.632微秒，原型0.262–0.264微秒。
计划基线共同墙钟spread **18.558%**，原样保留、未重测，不发布稳定CPU比例。
这些独立控制计时不能推算整网提升或其严格上限。

当前证据不足以支持此窄范围的完整前向/Worker至少1%收益，**不登记整网
候选，不进行GPU数值、性能或迁移**。ATTN/RMS/GEMM/门控等其他配置查询
仅做过源码定位，尚未量化整体开销，不据本结果否定所有主机端优化。
338份顶层历史叙述核查未找到相同已完成机制；本项不恢复CUDA函数表或Lt缓存。
56项现用文件和645项外部依赖保持。现用Worker1574.822297 RPC/s、原Fork
同边界102.094074%参考及精度/平台差异均保持，整体目标active。
下一步按现用FFN up/QKV/Attention/FFN down热点核对剩余机制覆盖，先给出
具体且未重复的源码/资源假设或明确暂无合格候选，再决定是否实施。
证据：`per-apply-host-control-audit-r1/{source-audit.json,history-review.json,
cpu-r1/completion.json,review.json,next-intake.json,final-review.json}`。

**2026-09-16 Attention丢弃尾块裁剪完成整网裁决：-0.262664%，关闭、未采纳。**

`current-mechanism-coverage-r2`核对148个命名阶段目录，并补齐22份最终裁决或
验收记录，形成19组机制表；小tile/精确SiLU等旧组件通过记录服从后续整网
关闭结果，N64数值失败与不稳定测试分别保留。不是穷尽所有可能优化的声明。
原FFN计数器是压缩前H1152/B14/S1，不能当作当前compact/S2瓶颈或上限。

`strict-attention-discarded-tiles-r1`只省略固定M128/N96中已被丢弃的完整
16行/列组：首KV块local80..95及末查询块local112..127。保留原四KV顺序、
有效单元FP32 QK/PV累加、half概率边界、精确exp、原归约布局及所有同步。
实际CuTe映射与384个线程/查询块控制流核查通过；静态warp MMA减少8.15972%，
寄存器仍255、栈40→0，**这些均不是时间收益**。一次rank-view离线编译失败
在生成PTX前保留并修正；一次本地inspect.py遮蔽stdlib的准备失败也保留，
未覆盖原件。12组输入、双流、候选先执行两次再对基线，普通运行/memcheck/
racecheck共**419,198,976个half输出逐位相同**，无错误或竞争警告。
历史模型激活注明为旧捕获，没有充当当前整网证明。

`strict-attention-discarded-tiles-integration-r1`从现用源码创建隔离版本，
新增默认关闭的`KATAGO_CUDA_ATTN_TAIL_R1`及独立artifact。旧strict身份、
RoPE和回退保持，候选加载与日志身份共用同一选择，调用间改开关会拒绝。
41项计划、3项资产测试通过；CLI/native20份PTX全部逐位保持，原FFN2992条
指令两个机器字相同。B1/B14/B16双臂双lane **612,960个输出逐位一致**；
真实W1/W32/W64双臂**768/768首选及11个C++FP32字段**、原始protobuf、
Drain和plan覆盖相反环境变量通过，12项非法计划/身份负例正确拒绝。
8帧2416个activity记录确认两臂各132次Attention、其余269个位置签名保持。
因两核符号和几何相同，代码选择由冻结主机分支、二进制内资产及artifact
标记共同证明，**没有只凭kernel名称判断代码**。默认构建、旧ONNX/ORT与
B1/B14 graph兼容通过。

唯一B14/S2/W80/N1000完整前向ABBA共同墙钟：
**1588.052559→1583.881312 NN rows/s
（-0.262664%）**；spread **0.094074% /
0.683101%**，通过5%稳定门但未达1%收益门。
辅助event **1606.913297→1602.740207
（-0.259696%）**；8000原始event与墙钟公式独立重构通过。
关闭此实现，未测Worker性能、未迁移或原样重测；不从减少静态指令推算收益。

现用56项文件、645项外部依赖保持。正式Worker **1574.822297 RPC/s**及
已有Fork统一墙钟 **102.094074%**参考保持，平台/精度差异仍单列。
下一步仅审计此前FUSION窄范围之外、当前确实执行的主机选择/依赖查询；
先核调用次数、分配和语义，缺乏证据就不登记新引擎候选。整体目标active。
证据：上述三个目录，尤其`coverage-final-resolution.json`、`numeric-review.json`、
`native-review.json`、`worker-golden-r1/review.json`、`forward-abba-r1/review.json`、
`independent-timing-review.json`、`next-intake.json`及`final-review.json`。

**2026-09-16 剩余主机控制CPU审计完成，未登记引擎候选。**

`remaining-host-control-audit-r1`以现用179层/302内核的冻结执行记录核对源码：
15次plain与44次residual GEMM包装调用、66次RMS查询、33次Attention选择，
10次C768门选择。33个compact FFN宽度都低于1152，故不触发B14旧残差预设；
DUALFFN查询、compact依赖校验和布局上传不在当前热循环中。stem/C768日志
已有延迟格式化，只有stem手写GEMM日志仍每次构造一次。每层另查一次未启用的
DEBUG_LAYER，PROFILE每次前向查一次；未重跑此前179次FUSION循环审计。

独立CPU探针提取29个原始函数/常量，按核实的调用表重放：305次tactic查询、
180次直接环境查询；已安装计划产生239次配置String复制、246次环境查询，
另有3次已缓存artifact字符串复制和1次日志格式化。独立分配计数构建实测
**489次/11860字节**；纯环境模式728次/21056字节。两模式各32个控制语义
案例及200次双线程调用通过，覆盖计划优先级、无效/缺失/非Unicode值及调用间
环境变化。它们不替代真实计划绑定、整网或Worker数值门。

未插桩双线程构建各做四次原控制诊断，每lane W1000/N20000：计划模式每次
**61.642–63.466微秒**，纯环境模式113.582–116.035微秒；各模式共同墙钟
spread为1.167%/1.114%。没有候选臂、模型、CUDA或Worker工作，未测驱动调用、
函数查询、Lt描述符等其他开销；artifact只重放已预热OnceLock的String复制。
**这些是独立控制重放耗时，不是实际apply耗时、整网收益或其严格上限。**
一次分配输出的JSON格式括号导致计数构建失败，发生在所有测量前；原件及错误
保留，修正后才执行上述唯一一组诊断，没有重复计时挑选结果。

只读复核Fork实际WSL源文件和固定计划，源码与本地文本一致：选中QKV/FA4使用
成员选项和按batch保存的tactic指针；ProjectionGemmLt与outer-projection AOT
回调在该计划中未开启，不能把其缓存设计当作当前执行路径或速度归因。

当前证据尚不足以支持新引擎候选达到完整前向和真实Worker至少1%收益，故不登记
候选、不做GPU测试或迁移。56项现用文件校验保持，正式Worker1574.822297 RPC/s
及此前Fork共同墙钟102.094074%参考不变；平台和精度差异仍单列。
下一步核对现用Worker原始批次、请求延迟和调度证据及G3既有裁决，先找尚未处理
的实际端到端损失；不相减不同轮次的前向/RPC速率，不恢复不变的关闭候选。
整体目标active。证据：本目录`contract.json`、`schedule.json`、`extracts.json`、
`fork-read.json`、`cpu-r2/completion.json`、`review.json`及`final-review.json`。

**2026-09-16 当前Worker调度与输入拷贝时间线核对完成，未登记优化候选。**

`current-worker-dispatch-audit-r1`重算正式输出投影N128验收的32768条请求记录，
其中现用臂16384条。两轮平均batch为13.931973/13.908319，8192行对应588/589批；
按B14容量仅缺40/54个槽位，但不能由平均值推断实际小批次数或其出现位置。
源码确认`queue_us`仅是Tokio工作线程启动等待；`evaluator_us`同时包含特征生成、
NN排队/凑批、提交/推理等待和后处理，不能等同GPU时间。当前NOPIPELINE=0，
同步serve分支的200/8000微秒等待窗口没有执行。原始记录没有NN批次ID、内部
排队时间戳或Worker CPU耗时，缺失值不当作零。旧placeholder复用与简单等待、
finish公平性、pinned staging候选已有关闭记录，本次不重开。

为回答真实Worker的小批次与GPU空档问题，使用原正式CLI和原计划做唯一一次
普通权限NSYS活动采集，C64、128预热、1024测量请求、无ownership/缓存；无
硬件计数器、CPU采样、管理员操作或生产改动。完整Drain通过。NSYS系统时钟
原点与主机计时对应，测量段**82批/1024行**与心跳增量完全一致；另分离4次
模型初始化和18次请求预热。每批302个kernel，B14的QKV/strict Attention/
输出投影各33次，非B14走原回退，正式plan与二进制身份复核通过。

这次采集中70批是B14；其余为B1×7、B4/B5/B6/B9/B13各1。10个小批次在
启动填充阶段、2个在末尾；不能把此受插桩的短窗口比例推广至原8192请求ABBA。
按固定“两lane各去首尾两批”裁剪，kernel时间区间并集覆盖96.4226%；另在
所有启动小批次结束到末尾小批次开始之间的连续B14段，覆盖**98.3591%**。
这是内核活动时间覆盖率，不是SM利用率、吞吐成绩或可优化空间的严格上限。

164次H2D记录都明确为**Pinned→Device**。其中15次空间输入API驻留超过1ms，
范围1.382–18.251ms，而实际设备拷贝仅8.608–9.921微秒；这些长API期间
92.818%–99.120%的时间仍有内核在执行，API返回通常位于前一同流前向结束后
113–636微秒。不能把API累计时间当作可消除墙钟时间，也没有据此认定驱动、
流依赖、主机调度或采集器中的某一项是原因。当前已有逐槽锁页输入；单纯换
锁页内存或重复旧staging实验没有依据。

下一步只用现存记录和源码核对cudarc的事件依赖、双槽复用与Fork输入提交链路，
判断独立上传流是否存在明确且未重复的可测试机制，再决定是否登记候选。
不重复本次采集，不推算性能收益。56项正式文件保持；Worker1574.822297 RPC/s
和此前Fork共同墙钟102.094074%参考不变，整体目标active。
证据：本目录`raw-review.json`、`trace-contract.json`、`trace-r1/completion.json`、
`trace-review.json`、`copy-dependency-review.json`、`next-intake.json`及`final-review.json`。

**2026-09-16 独立上传流候选已隔离实现并通过后端/Worker数值门，内存检查器附加失败，尚未测速或采纳。**

`upload-stream-integration-r1`从现用源码与已有Worker时间线核实：H2D、apply、
D2H现用同一流；每个batch尺寸有两个输入/输出槽、一个共享workspace。
CudaRuntime关闭了cudarc自动事件跟踪，PinnedHostSlice的自动事件不会在
SyncOnDrop中记录，不能将as_mut_slice当作跨流复用保证。现用evaluator最多
两批在途、FIFO收尾与计算流顺序共同保证原路径安全。Fork独立upload/download
链路在完整apply后记录inputConsumed，仍是单设备输入缓冲，不能解释为早释放
双缓冲。原603.214ms诊断段里136次H2D设备时长合计656.182us，其中578.993us
没有任何内核活动；此按kernel边界截取的窗口还会包含下一尾批的输入拷贝。
仅重叠拷贝本身空间很小，长API等待是否改善仍需实测，未推算收益。

候选`KATAGO_CUDA_UPLOAD_STREAM_R1`默认0，只允许NOGRAPH=1。输入仍以FP32
上传，全部计算、FP16存储边界与FP32累加保持。隔离副本显式完成分配可见性、
host槽复用、upload→compute就绪事件；计算与全部D2H仍在原计算流，共享
workspace不跨流。部分提交出错和句柄释放时显式排空两流。正式源码、CLI、
计划及配置56项保护文件均未改。

隔离release CLI和后端测试已构建，40项计划单测通过。实际后端两lane各24批，
混合B1/B14/B16并最多两批在途，共540行、395,820个对外输出值；同计划顺序
与流水线、开关两臂均逐位一致，包含8种对称与不同policy optimism。候选
另验证初次使用后两次提交不收尾直接释放句柄、再新建句柄推理。7个配置负例
按预期拒绝，开关两臂都用与plan相反的环境值验证优先级。

W1/W32/W64各开关两组实际Worker全部通过，768/768首选、11项真实C++FP32
字段门、输入/请求身份和Drain通过。逐条protobuf重建与结果重算也通过。
这些是正确性证据，不是性能样本，也不是内存泄漏或竞态检查结果。

普通权限Compute Sanitizer第一次多进程跟踪报告`No attachable process found`，
控制器和目标仍存活但无进展，核对PID和注入模块后显式结束并回收。第二次
有依据地改为application-only、30秒连接上限与kill-on-error，仍附加失败并退出。
没有获得ERROR SUMMARY或被插桩测试结果，不能宣称0错误，更不能说发现了
内存泄漏；正式版本不受本隔离实验影响。未使用管理员权限、未改驱动设置。

构建身份差异已独立重算：kernel_build_id包含绝对源码路径，compact/outproj
身份包含本次改动的Rust宿主源码；87份CUDA/AOT素材逐文件不变，原DualFFN
2992条指令与既有验收产物相同。几个读取器断言修正保留原件和记录，包括
裁剪窗口边界、构建ID范围、测试行数在执行前的算术修正和负例错误文案；
没有重跑已完成的负例或性能实验来筛选结果。

下一步先定位检查器与这个Rust测试进程的附加问题，用最小CUDA初始化/分配
诊断区分环境和加载路径，再补足原内存检查门。可以独立补齐旧路径/原始全头
兼容证据。通过后才登记唯一包含H2D/计算/D2H/解码的真实Backend流水线ABBA，
再按原>=1%收益、<=5%波动门进入C64 Worker ABBA。`nnbench kernel`绕过本次
变更，不能作为它的性能裁决。保留完整目标，不重用一次性管理员授权。
正式Worker1574.822297 RPC/s与Fork共同墙钟102.094074%参考均不更新。

**2026-09-16 独立上传流内存检查与性能裁决完成：安全检查未发现内存错误，包含传输的Backend吞吐稳定下降0.990064%，候选关闭。**

`upload-stream-sanitizer-recovery-r1`核实本机此前成功的Compute Sanitizer使用
Windows命名管道连接；只为检查进程设置
`NV_COMPUTE_SANITIZER_LOCAL_CONNECTION_OVERRIDE=named-pipes`后附加恢复。
没有使用管理员权限，没有改驱动、时钟或系统设置。首个完整运行通过540行
实际Backend与未收尾释放句柄测试，泄漏摘要0字节/0分配，但工具还报告
135,732次CUDA API错误且默认只打印100条，因此未直接判为整体通过。

后续保留全部memcheck、full leak-check和stream-ordered-races检查，仅取消
打印条数限制并省略回溯。候选135,732条、关闭优化的基线
131,811条报告全部是`cuModuleGetFunction`的
`CUDA_ERROR_NOT_FOUND (500)`；源码现有get_func逐模块尝试查找并处理失败，
与此报告一致。逐条计数等于各自ERROR SUMMARY，无未打印报告、无其他内存
诊断，两臂泄漏均0字节/0分配。不是“ERROR SUMMARY: 0 errors”，没有关闭
API错误检查或使用忽略规则。插桩输出与普通运行395,820值逐位相同。
一次读取器因并发日志拼接漏计1条，保留原失败记录后从同一原始日志修正；
没有为修正读取器重跑候选。

新构建直接apply的B1/B14/B16、两lane原始全头共306,480值与现用输出投影
N128参考逐位一致，六通道policy首选全部一致。这三组只证明未改变原始
计算路径，不覆盖被绕过的上传流；上传流由540行实际Backend及此前真实
Worker768/768、11字段门覆盖。未重复执行这些已通过的Worker数值实验。

唯一实际Backend ABBA保持B14/S2、每handle最多两批在途，每lane预热80批、
计时1000批，共112,000行。两臂使用同一个二进制及相同20项原tactic，只切换
UPLOAD_STREAM_R1，plan与环境反向值及不同stream指针均确认实际生效。
共同主机墙钟包含对称/锁页暂存、H2D、apply、D2H、解码与最终排空，排除
初始化和预热，无gRPC或缓存。原始lane时间戳已独立重算。

基线两次1605.072515/1590.767946、候选两次
1582.919986/1581.248364 NN rows/s；几何均值
**1597.904224→1582.083954（-0.990064%）**。
基线/候选spread为0.899224%/0.105715%，
均通过5%稳定门，但未达1%收益门。按原裁决关闭候选，不重试筛选，不进入
Worker性能、旧ONNX/graph迁移验证或生产安装。该结果也不能当作Fork对照。

正式56项源码/CLI/计划/配置指纹保持；现用Worker1574.822297 RPC/s及
Rust/Fork共同墙钟102.094074%参考保持。整体目标继续active。下一步回到
现用GPU热点及已关闭候选清单，优先寻找保留FP32累加的FFN/GEMM新机制；
不得原样重开独立上传流或已关闭的主机缓存/暂存方案。性能与数值原门保留。

**2026-09-16 compact DualFFN显式两级异步流水：组件数值/内存/竞态通过，算子稳定回退，关闭。**

`compact-ffn-explicit-stage2-r1`从现用FFN up19.273%累计热点出发，核对65个
FFN相关阶段、985份顶层记录。历史warp8曾在构建前排除device包装的stage2；
未找到相同独立底层组合已完成的实验。旧全hidden/顺序投影的两级实现另有
分块或融合，仍保持关闭。Fork当前TF3 FFN实际为原生Hgemm，旧DualGemm
s2配置不是该路径的速度证据，本次没有照抄其低精度计算或推导收益。

保持CUTLASS现有device::DualGemm要求stages>=3的断言及全部库源码，另组合
原Mma0/Mma1的异步迭代器、policy与DualMmaMultistage<...,2>。CTA128×64×32、
warp64×32、swizzle2、两套FP32累加、投影half舍入及完整精确SwiGLU epilogue
均保持。抽象K384环形读写顺序审查与544对编译Params逐字节相同通过；
这些CPU证据不代替GPU竞态检查。原基线2992条SASS及两个机器字完全一致。

候选2944条指令；两核均64条静态FP32 MMA，寄存器都236、无stack/spill/local。
实际动态共享49152→32768B，但**每SM仍最多2CTA**。PTX原两处wait_group1变为
wait_all，共享读取和同步结构由数值及racecheck验证；不能由更少共享或静态
LDGSTS指令直接推出吞吐提升。补充PTX读取器曾误假设`.visible .entry`和
`wait_group0`拼写，改为解析实际`.entry`/`wait_all`后复读原PTX；原读取器
留档，未因此重编译或重复GPU实验。

全部34组冻结组件样本（33层宽度与额外完整首层）均覆盖两臂/两独立stream、
guard及只读输入。普通运行、memcheck与racecheck每次
**228,198,208个half值**与既有参考逐位相同；memcheck为0 errors、0字节泄漏，
racecheck为0 errors/0 warnings。命名管道连接，普通权限，无系统设置改动。
除额外首层外，这些是逐层31行周期扩展到5054行的历史组件输入，不是新捕获
的当前整网激活；两stream组件数值流程不声称并发重叠覆盖。

唯一B14双流ABBA包含原顺序33个FFN up/SwiGLU、W80/N1000，无D2D残差重置；
H2D和前后快照在计时外，两臂使用相同Params构造及cudaFuncSetAttribute/发射
包装，只改变实际kernel类型。8000个event和528份完整前后数组独立重算：

| 算子序列指标 | 原三级 | 显式两级 | 变化 | 基线/候选spread |
|---|---:|---:|---:|---:|
| event 序列/s | 526.957424 | 483.657240 | -8.217018% | 0.383731%/0.047987% |
| 共同主机 序列/s | 525.677410 | 481.898025 | -8.328185% | 0.472247%/0.149206% |

两口径稳定回退，按原门关闭，不挑宽度子集、不重测、不进入整网、Worker
或迁移。这是33个上投影算子序列，不是完整NN rows/s或RPC/s；未证明单一
停顿机制就是回退根因。现用56项文件保持，Worker1574.822297 RPC/s与
Fork共同墙钟102.094074%参考不变，整体目标active。

下一步可只读审查当前DualMma中异步等待相对最后一组MMA/下一块预取的位置，
核对既有late-wait实验后判断是否存在不同的有效调度机制。必须先证明实际
生成指令和数据依赖发生所需变化；不能原样重开此stage2版本或直接假定移后
等待能抵销回退。无合格机制则转向其他现用GPU热点，完整原门不变。

**2026-09-16 compact DualFFN三级流水延后等待：机器码未实现预期顺序，GPU执行前关闭。**

`compact-ffn-latewait-r1`保留现用三级、CTA128×64×32/warp64×32、两套FP32
MMA、原copy分组/环形偏移/mask、half舍入及完整精确SwiGLU。隔离头文件只
替换固定K384主循环：源代码先完成当前块第二组gate/up MMA，再等待异步
拷贝和CTA同步，下一块共享读取仍在等待之后。CUTLASS库及生产文件未改。
源码依赖顺序模型保持12块、每块两组的累加次序，预取/commit/wait与原版
相同；这只是源码审查，不是GPU数值或竞态认证。

先核对已关闭的QKV N128 late-wait：其编译器曾将等待重新调回原位置。
因此本次预先规定，必须在实际FFN K循环中看到全部64条FP32 HMMA先于唯一
DEPBAR，否则不启动GPU测试。编译成功，基线2992条SASS与此前正式核的
两个机器字相同，候选也保留FP32 MMA。但实际原循环仅10条HMMA在等待前，
候选为11条，**仍有53条在等待后**。两个独立读取器按后跳边界重算一致。
预期“最后一组计算先于等待”没有实现；不是把11/64当成已实现的优化。

按预登记机制门关闭。未运行GPU kernel、资源查询、组件数值、sanitizer、
整网、Worker或性能测试，不声称性能回退或收益。测试源码与失败机制证据
保留，不改门槛或添加无依据的同步来迫使调度。正式56项文件不变，现用
Worker1574.822297 RPC/s及Fork共同墙钟102.094074%参考保持。

下一步转查compact FFN下投影：当前K随33层隐藏宽度变化，而已采纳的输出
投影classic TN N128仅绑定K384。先核查旧N32/N64/NN CUTLASS与Lt21/tile112
实验的实际范围，再审查N128通用内核能否为这些动态K导出正确的参数和尾部
映射；不能直接把K384参数模板用于FFN，不能继承输出投影的性能收益。
只有明确不同的有效路径与CPU/资源证据才登记后续候选，原数值与ABBA门保留。
整体对标目标继续active。

**2026-09-16 compact FFN down classic TN N128：数值/内存/竞态通过，主机与event指标分歧，按原门关闭。**

`compact-down-classic-tn-n128-r1`核对旧dense K1152 NN CUTLASS、TN N32/N64、
direct-store和compact Lt21/tile112范围后，复用已安装输出投影的同一份通用
TN128×128×32/warp64×64、三级CUBIN，针对当前33层的23种K独立导出Params。
K范围96–1104，11种有K16尾部；不能复用固定K384模板。所有权重half位、
FP32累加/残差/输出、alpha=beta=1及无split-K保持。没有新增半精度累加。

234份原设备编译CUTLASS头文件仍一致。23份参数模板及CPU输入映射通过，
每个有效输入元素恰好访问一次，向量不跨行；128×128输出映射覆盖完整M5054。
K384模板与现用模板逐字节相同。首次CPU检查在nvcc的设备函数主机替身
`exit(1)`处停止，未触发访问断言；隔离观察头文件只为4个原函数增加host
标注，函数体、正式库和CUBIN未改。缺少string声明及离线SHARED1024的读取
修正也保留；后者包含驱动预留，实际函数static shared仍0。

34组两臂共131,970,048个保存FP32输出逐位一致，全部对FP64点积参考通过
5e-5绝对/相对门，第二条独立live stream的输出也逐位相同。fixture为历史
33层各31行重复至5054行，加首FFN完整5054行，不是新采集的现用整网激活。
memcheck为0错误/0泄漏，racecheck为0错误/0警告，guard和只读输入保持。
普通权限软件追踪136次GEMM及68份输出通过：基线逐K匹配当前正式时间线，
候选确为classicTN N128，grid40×3、128线程、226寄存器、49152B动态共享。

唯一B14/S2、80warm/1000iter、33层down序列ABBA：

| 指标（序列/秒） | 基线 | 候选 | 变化 | 两臂spread |
|---|---:|---:|---:|---:|
| 共同主机墙钟 | 539.368830 | 585.340236 | +8.523185% | 0.713322% / 0.000269% |
| lane event中位数合计 | 1011.755129 | 890.606538 | −11.974102% | 6.708339% / 0.097762% |

主机指标稳定且有收益，但event基线超过5%波动门，且名义收益未达1%；按预登记
两项均须通过的门关闭，不改门槛、不原样重测。不能把event结果称为稳定回退，
也不能把主机+8.52%写成整网或Worker收益。8000原始事件及528份完整前后输出
独立复算通过。event只包33个GEMM，主机还含33次残差D2D重置和发射/同步间隙；
两lane各自event区间合计占共同墙钟，基线约53.48–57.57%、候选65.87–66.18%。
这不是GPU利用率，现有样本不足以归因给主机派发、拷贝或调度，未额外采集性能。

未接入整网/Worker，未迁移计划；正式56项文件保持。认证Worker1574.822297
RPC/s及最新Fork共同墙钟102.094074%参考保持。后续候选仍须完整数值/路径/
稳定整网ABBA/真实WorkerABBA后才采纳。本项关闭不表示整体目标完成。

下一步先核对当前compact下投影的NN布局覆盖：旧FFN NN CUTLASS绑定dense
K1152，现用G1 NN仅是K384且throughput选择TN。只做源码/能力范围审计，确认
23种K的独立NN布局是否已有相同关闭实验及真实候选；不得继承本次主机收益、
重跑此TN N128或事后按本次成绩筛选K子集。整体对标目标继续active。

**2026-09-16 compact FFN down NN布局：组件失败保留；独立整网数值通过，完整前向event收益未达1%，关闭。**

`compact-down-nn-layout-r1`只在当前33个compact FFN下投影的23种K上，
将已舍入half权重按位转置为NN；其他GEMM、FP32累加/残差/输出不变。
这与旧dense K1152 NN CUTLASS、G1 K384 NN及已关闭TN N128范围不同。
每K仅选Lt公开启发式首个有效项，无按计时选K子集或候选池择优。
34组两臂131,970,048个保存FP32值对FP64门通过且逐位相同；memcheck
0错误/0泄漏、racecheck0错误/0警告，136次实际内核路径匹配。fixture仍为
历史33层各31行重复及首FFN完整行，不能称新采集整网激活。
唯一组件ABBA：共同主机543.770611→568.892680序列/s（+4.619976%），
event1048.853755→1000.319090（−4.627401%）；两臂均稳定，但原双指标
门失败，结论保留，没有重测、改变门限或组件性能采纳。

组件主机区间含每轮33次残差D2D重置，真实整网由上游产生残差，且包含
完整QKV/Attention/up/outer/head及相邻依赖。稳定主机收益仅作为开展
独立整网实验的依据，不能证明其收益或解释指标分歧。登记
`compact-down-nn-integration-r1`，明确组件仍失败，完整前向仍须event和
共同主机收益各≥1%、两臂spread各≤5%，再过真实Worker性能门。

隔离实现默认关闭，仅physicalB14的33个compact down使用NN，B1/B16
保持TN；新键、模型SHA、构建artifact和依赖均显式绑定，B14禁止静默
fallback。40项计划单测及2项全23K转置/精确批量单测通过。20份PTX、
四份AOT机器码及原DualFFN2992条指令保持。首次CPU指纹重算误用了
Python路径规则，保留失败并用Rust PathBuf导出原build.rs的路径/排序，
重算完全匹配；未改源码/二进制/门限或重跑GPU测试。

六组B1/B14/B16双流整网612,960个输出值与现用版本逐位相同；真实
Worker W1/W32/W64两臂768/768首选、11个C++FP32字段、原始协议及Drain
通过，且plan值与环境相反验证优先级。20项CLI负例、旧ONNX对ORT四局、
默认无CUDA构建及B1/B14 Graph回归通过。普通权限完整路径追踪7248次
内核确认每前向仅33个down改为已验证NN签名，其余269个内核保持。

唯一完整前向B14/S2、W80/N1000 ABBA：

| 指标（NN rows/s） | 基线 | 候选 | 变化 | 两臂spread |
|---|---:|---:|---:|---:|
| 共同主机墙钟 | 1573.280145 | 1593.180273 | +1.264881% | 2.509526% / 0.311098% |
| lane event中位数合计 | 1604.971310 | 1611.888377 | +0.430978% | 0.414080% / 0.129475% |

8000原始event已独立复算。两指标均过稳定性门，但event收益未达1%，
按预登记双指标门关闭，不进入Worker性能、不原样重测、不迁移计划。
不能将共同主机+1.265%称作Worker收益或已认证优化。正式56项文件保持；
认证Worker1574.822297 RPC/s与最新Fork共同墙钟102.094074%参考保持，
平台/累加精度差异仍单列。整体目标继续active。

下一步审查现用完整组合的B14单流/双流与凑批覆盖：先核对旧实验实际绑定
的模型、二进制、kernel组合和物理batch，再决定是否存在尚未验证的组合。
不将旧未压缩版本的结论直接外推，也不重开已关闭的相同实验；本项NN布局
与前项TN N128均保持关闭，未按本次结果挑选K子集。

- **G0 稳定基线**：`target/fork-parity-20260908/fork-rust-wsl-empty-abba/report.json`
  保存 Fork/Rust/Rust/Fork 的全部样本。Fork 为 **2231.37 / 2357.02**，
  Rust 为 **1125.07 / 1128.85** event NN rows/s；各自几何平均为
  **2293.33 / 1126.96**。重复轮次 spread（`max/min−1`）分别 **5.63% / 0.34%**，
  低于固定 10% 最大波动门，状态为 `STABLE`。49.14% 是 lane event 吞吐之比，
  不是 gRPC Worker、evaluator 或端到端速度之比；也不是同精度 kernel 效率之比。
- **计时与输入**：两侧每次 forward 前后记录 CUDA event，整段入队后同步；
  先取每 lane 中位数，再求 `sum(14 / median_seconds)`。预热后两 lane 屏障
  对齐，Fork 使用 `phase-offset-us=0`。19 路空盘、黑先、TrompTaylorish、
  komi 7.5、symmetry 0、optimism 0 按语义对齐，未验证跨实现特征逐字节一致。
  Fork `actualWallSeconds` 包含预热等阶段，Rust wall 不含，故不计算 wall 比例。
- **不可省略的差异**：Fork 是 CUDA 13.0.88/cuDNN 9.24 认证构建，Rust 此轮为
  CUDA 13.3.73；Fork qk16 及部分残差 GEMM 使用 FP16 累加，Rust 保留 FP32
  与精确 expf。双方都算 ownership 与完整语义输出；Fork 原生 policy/terminal
  为 2/9 通道，Rust 兼容布局实际算 6/21，其中 4/12 个通道为零。
  `partial-c288-g1-v1` 仅合并 G1+V1，P1 仍计算，不是裁掉输出头。
- **诊断轮次保留**：`fork-rust-wsl-baseline-abba/report.json` 使用不同输入，
  且 Fork spread 为 **37.06%**。该轮仅供诊断；其中约 55.6% 的比例不得引用为
  有效基线，不挑其中快样本替换稳定空盘 ABBA。
- **G1 B8 裁决**：`worker-layout-b8-abba/report.json` 中 C1/C8/C16/C32 的
  变化分别为 **+0.34% / −0.52% / −0.84% / +1.91%**。C1/C8 实际仍执行 TN，
  C16 的平均 batch 约 7.99，C32 约 15.97；实际 B8 没有稳定收益，因此未采纳
  `nn_k384_b8`，转而验证只在 B16 启用 NN 的候选。
- **G1 最终裁决**：`worker-layout-b16-abba/report.json` 中 C16 为
  **930.55→923.82（−0.72%）**，C32 为 **1073.48→1093.68（+1.88%）** RPC/s；
  `worker-layout-b16-throughput-abba/report.json` 的双线程 C64 仅 **+0.59%**，
  未过 1% 门。只新增 `configs/worker_tf3_sm120_c32.cfg` 和
  `plans/worker-tf3-sm120-c32.json`，面向持续 C32、单线程、物理 B16、graph；
  旧默认与双线程吞吐 profile 的 tactic 不变。该收益是同 Rust binary 的 Worker
  ABBA，不是 G0 Fork 比例的更新；G0 保留当时绑定的原运行二进制 SHA。
- **G4 B8 留痕**：`layout-tactics-final.log`、`numeric-final-layout-b8.log`、
  `graph-final-layout-b8.log` 留存 7 组合 ORT 4 局/组合、TF3 全字段与 top-1
  128/128、graph 首次发射回归通过。
- **G4 最终配置**：`numeric-layout-b16`、`numeric-layout-b16-throughput` 与
  `numeric-certified-c32-plan` 的 `comparison.json` 均 PASS、top-1 128/128。
  最后一组设置相反环境布局 `tn`，日志仍确认 plan 的 `nn_k384_b16` 在 rows=5776
  生效。`layout-b16-tactics` 通过 7 case，其中 5 个各有 4 局 ORT、2 个验证缺失/
  错误 host revision 拒绝；16 个 tactic plan CPU 单测及 Python paired 6/6、
  tactics 9/9 通过。先安装/校验 plan 再上传权重的生命周期修复保留；
  `host_tactic_revision=1` 令旧 binary 实际因 unknown field 拒绝新布局 plan，
  证据为 `old-binary-rejects-layout-plan.log`。G1 的 C32 profile 验收完成，
  不向 G2 或其他未测负载推广。
- **主机函数缓存已撤回**：整网数值、128 输出一致性及 graph 回归通过，但
  `function-cache-forward-abba-r2` 的同 Rust WSL 前向变化为 −0.77%，
  `function-cache-worker-throughput-abba-r2` 的 C64 变化为 −0.76%，未达 1% 门。
  初轮大幅波动的全部样本保留为诊断，不使用其中虚高收益；脚本正常串行，
  尚未确定异常原因。G1 布局保留，后续优先验证 TF3 残差 GEMM 的 FP32
  分块与 strict attention 的 RoPE 融合，不能以静态 Driver 调用数量推断收益。
- **残差算子与剖析**：TF3 四形状/三 tile 的 FP32 残差 CUTLASS 探针数值门
  通过；128×64×32 在 B16 FFN down/outproj 独立 ABBA 为 +9.26%/+15.67%，
  B14 无对应收益，尚未生产采纳。Lt top-8 诊断有接近候选，先做固定算法
  配对复测。另修复 profile 子段覆盖层起点的问题；实际三次前向的层/子段
  关系及 128/128 整网数值门通过。插桩耗时不用于 G0 性能比例。
- **RoPE 融合裁决**：`fa4-strict-rope-smem` 保留独立 CUBIN、代码生成审计、
  首次失败与修正测试参考后的全部证据。三组完整 attention 输出通过原门，
  但共享内存旋转与额外同步使耗时达到约 72.9–74.8 微秒，高于 q64-serial
  的约 66.7 微秒，故不集成。独立 RoPE GPU helper 的 half bit 对拍不等于
  读取 AOT 内部 shared memory；不把该变体的失败推广到整个融合路线。
- **FP16 权重编码修复前置**：`fp16-conversion-audit/results.json` 确认共享 TF3
  的 70,361,856 个 half 权重源元素中，713,115 个旧转换结果错误（1.0135%）。
  根因是次正规分支多移一位且舍入不符合 ties-to-even。修复保留既有 FP32
  计算与 half 存储边界，新增独立 `fp16_encoding_revision=1`，旧 plan 必须拒绝，
  在新编码数值门及性能复验通过前，不给旧认证直接补版本号。
- **新编码复验进度**：`fp16-rne-r1/numeric-index.json` 及构建/回归日志记录了 Windows/WSL 构建、
  FP16 边界回归、25 项 plan 测试、13 项真实启动拒绝检查、ONNX 六组 B1/B16
  对拍（每组 16/16）、三组 graph 与四组真实 plan 覆盖检查，以及默认/C32/
  双线程 Worker 各 window 1/32 的六组 128/128 黄金门。Worker 的 window32
  实际为混合批次；另用原生 TF3 转储验证固定 B14/B16 与独立双流完整输出。
  新编码 G1 C32 ABBA 为 **1066.46→1088.63 RPC/s（+2.08%）**，双线程 C64
  为 **1072.78→1159.08（+8.04%）**，均稳定；这些是对应新二进制的独立复测，
  不是替换历史样本。七份活跃 plan 已完成迁移，旧字节保存在
  `plans/archive/fp16-legacy-pre-rne-20260908/`，最终路径加载与相反环境覆盖
  的七项检查全部通过；默认 ONNX alias 与 Windows q64 新配置逐字节相同。
- **Lt 候选的平台边界**：Windows 的 B14/B16 及 NN 布局 B16 候选与同形状
  基线的全部原始输出逐位相同，Worker 128 请求门也通过。整图 ABBA 只在
  Windows 双线程 C64 达到门槛：**1158.67→1170.68 RPC/s（+1.04%）**，
  已加入吞吐配置；默认 C32 +0.91%、NN C32 +0.95% 均拒绝。
  WSL B14 在相同 cuBLASLt 版本号下返回不同 opaque algorithm 数据，运行时
  正确拒绝，保存于 `fp16-rne-r1/native-wsl-lt-b14-l2.dump.log`；不掩码差异、
  不直接套用 Windows 的编号。WSL 的独立候选仍须另做数值和性能筛选。
- **新编码 Fork 对照**：`fp16-rne-r1/fork-rust-rne-empty-abba-r2/report.json`
  保留相同 G0 协议的完整稳定样本，Fork/Rust 几何平均为 **2357.54/1129.01**，
  比例 **47.89%**；之前的新编码首轮 Fork 波动 18.94% 被排除为诊断，未挑选
  其中样本替换。Windows 吞吐配置的 +1.04% 不用于改写 WSL G0 的前向比值。
- **公开初始化 Lt 独立候选（2026-09-09）**：公开 AlgoInit/属性回读/实际流
  Check 得到稳定完整算法身份，Win/WSL 三种残差 GEMM 均逐位数值通过且算子
  ABBA 正收益；两平台 B14/B16 双流的原生整网输出也与同批次基线逐位相同。
  然而 WSL 完整前向 ABBA 为 B14 **1127.95→1127.14（−0.072%）**、B16
  **1130.96→1106.29（−2.182%）**，均稳定，拒绝采纳并回退生产实现。
  `fp16-rne-r1/lt-initialized-r1/` 留存独立探针、冻结候选二进制/源码和完整
  数值及性能证据，旧 Windows Fixed 配置和七份认证 plan 保留。
- **G2 批次展开独立 RoPE**：`fa4-rope-batch-unrolled-r1/run-r1-summary.json`
  记录三个幅度的精确 Q/K/V 与同 AOT 输出逐位通过；预登记 event 中位数
  的同 AOT 总段收益 **+0.048%/+0.325%/−1.397%**，相对 q64-serial 为
  **+0.264%/+1.548%/+0.023%**。全部波动在 5% 内，未持续达到 1%，
  拒绝生产推进。仅拒绝这个线程组织变体；去掉 V 拷贝与 G3 批次/流数
  扫描按独立因素继续，不继承局部 helper 的加速作为整网结论。
- **G2 去除 V 拷贝**：`fa4-rope-no-vcopy-r1/run-r1-summary.json` 保留
  7200 个原始 event 样本。三种幅度先过全部 Q/K 位、V 原值、scratch
  canary、同 AOT 输出逐位门，再测同 AOT 总段 **+6.570%/+7.990%/+6.370%**，
  相对 q64-serial **+8.083%/+7.127%/+6.413%**；六组完整 Attention 均稳定。
  helper-only 一组波动 82.10%，只作诊断，不采纳该局部加速值。
  `fa4-strict-b14-r1` 整网候选保持原 half 边界与 FP32 QK/PV，复用 FFN
  scratch，显式 q64-serial 回退与 NOGRAPH=1，并独立绑定三份 AOT/ABI
  实际字节指纹。Win/WSL 构建、每平台 30 plan + 2 ABI/范围 + 3 half
  CPU 测试通过。`g2-strict-attention-integration-r1/native-raw-summary.json`
  汇总两平台各六 dump/十二比较：旧路径复现、B1/B16 回退全部逐位一致，
  B14 新路径过原门及所有 policy 通道 top-1 门。Windows B14/S2 两组
  Worker C++ FP32 各 128/128；这些 Worker 请求自然混批，B14 warmup
  标记不等于实测请求批次分布。两平台各 14 项实际 plan 正反例通过。
  `forward-abba-win-r1/report.json` 同一新 binary 的 B14/S2/W80/N1000
  ABBA：event **1126.11→1145.05（+1.6817%）**，完整墙钟
  **1115.69→1135.03（+1.7331%）**，最大 arm 波动 1.093%；8000 原始
  event 及全部算术独立 CPU 重算 PASS。实际同 B14/S2 Worker ABBA 为
  **1151.91→1172.25 RPC/s（+1.7656%）**，两侧波动均不超过 0.14%；
  `best-worker-comparison/` 将严格 Attention 与已有 Fixed Lt 合用，
  先过两侧各 128/128 C++ FP32，再以同新 binary 直接对原最佳 B16/S2：
  **1169.59→1194.02 RPC/s（+2.0888%）**，波动 **0.02775%/0.10477%**，
  C64、每轮 8192 请求，原始数据 CPU 重算 PASS。安装前补充的固定 B14/L2
  strict+Fixed 组合原始门也通过，全部 56 行、所有头及六 policy 通道，
  对 strict/heuristic、fa2/heuristic 同批次和旧 B1 三参考分别检验。
  已只更新 Windows throughput plan/config 与 CLI；其他六计划、WSL CLI
  字节保持。最终 6 Win + 1 WSL 计划真实加载和 throughput Worker
  128 请求/Drain 通过，收据见 `throughput-install-r1/installation-receipt.json`，
  状态 `INSTALLED_AND_FINAL_PATH_LOADS_WORKER_SMOKE_VERIFIED`。
  不将不同基线的收益相加，也不乘到 WSL Fork 比值。
- **G3 扫描及数值阻挡**：`g3-forward-scan-r1` 的 B12–B16 × S1/S2
  十组单次扫描中，B14/S2 event 指标最高，但 B13–B16/S2 前向墙钟几乎
  相同；不能据此换配置。`g3-worker-b14-vs-b16-r1/numeric-r1` 当前正确
  RNE1 WSL 二进制的 B16/S2 基线全 128 例连续输出误差都在阈值内，
  policy top-1 只有 **126/128**，因此数值门 FAIL，原矩阵停止且未启动 ABBA。
  两例是同局面两个对称变换中的近并列落点翻转；实际 128 rows/17 batches，
  B16 是上限而非每例实测批次。现有新编码 Windows 数值通过不能代替
  WSL 认证；保留失败并定位，不放宽 top-1 门。后续独立诊断中，S1/W1
  的真 B1（128 rows/128 batches）恢复 128/128；S1/W32 为 126/128
  （128/12 batches），B14/S2/W32 也为 126/128。故偏差不依赖第二条
  服务线程或 B16 上限。两坏例的 policy 在 S1/S2/W32 相同，ownership
  仍有末位差异，不宣称整输出逐位相同；既有原始 dump fixture 没有这一子
  局面，下一步冻结真实 pre-GPU 特征，定位批次变化时的首个分歧层。

详细头部工作量、二进制身份和复现命令见
[TF3 Worker 性能审计](tf3-worker-performance-audit.md)文末「本轮 Fork 对标追加」。

### 历史里程碑（原测量语境保留）

| 项 | 状态 | 备注/链接 |
|---|---|---|
| 规划文档 | ✅ 2026-09-08 更新同卡 TF3 基线与 G0–G4；保留 M0 后历史 | §1、§2、§8 |
| nnbench 工具 | ✅ 已提交已实测(含 workers=2×当前 batch 修复) | crates/katago/src/cmd/nnbench.rs |
| M0 Phase 0 测量 | ✅ 完成(2026-08-17) | cuda-optimization-plan.md「M0 Phase 0 测量」 |
| H1-H5 假设 | ✅ 全部裁决 | §5 决策表;H1/H2/H3 证伪,H4 上修,H5 否决 |
| M1(C0+C1) | C0 证伪;**C1 已落地又于 2026-08-17 晚复审下架**(B2 后边际归零+搜索口径净负);当前 schema 2 plan 为 DUALFFN+q64 | 数据见优化文档「M3 追加」「M4 追加 2」 |
| 方案 A 拓扑组合 | 🔶 **部分复活(2026-08-18)**:饱和供数双流 +9.5%(serve=2+NOGRAPH+W≥4b,WSL 1112/Windows 1105);搜索语境维持证伪(t=48 灾难 123);PADBATCH 组合永久证伪(259) | 原 -2% 证伪系 DUALFFN 断裂慢 kernel;数据见优化文档「M4 前哨」 |
| 方案 B1-B4 | B1 ✅ 等价完成;B2 ✅ DualFFN 落地(+26.5% eval/+28.0% 搜索);B3 两个本地 FA4 变体 ❌ 负收益,官方不同组织的 q64 FA2 ✅ 采纳(eval +3%左右/core +14.7%);B4 ❌ 当时门槛未过 | DualFFN+q64 已入 ONNX schema 2 plan；不宣称整个 FA4 路线已证伪；数据见优化文档 M3/M4 追加 |
| 方案 C tcgen05 | C0 证伪/C1 落地(M1);**C2 ❌ 硬件证伪(2026-08-17)** | ptxas+CUTLASS 4.7+PTX ISA 9.3 三源互证,见优化文档 M3-pre;FP8/sm_120f 工具链留档(D:/code/cutlass4,device ✅,MSVC host C2719 待解) |
| 方案 D cuDNN | ❌ 搁置 | H5:InitialConv 1.6% < 5% |
| 方案 E 储备 | E1 ✅ 已裁决(2026-08-17):精度门不过,FP8 封存;E2 host staging/直接 pinned/输出零拷贝 ❌ ABBA 无收益(2026-08-19);E3 未动;**E4 WSL ✅ 已验证**(搜索 +4.6%/eval +2.4%,scripts/wsl_bench.sh) | DUALFFN 修复后 per-SM 差距 ~14%;确定杠杆仅剩 WSL 部署，简单 A-lite 已收口 |
| 官方 v1.18.1 审计 + opt 测试制度化 | ✅ 2026-08-24:kernel 层无增量(官方三件套已等价);`[cuda-tactic]` 标记扩至全部 tactic + validate_cuda_tactics.py 24 case(含数值门)全绿;**附带修复存量 graph 丢批 bug**(pre-capture warm 无条件化,INVALID_VALUE 15→0,对拍 PASS) | 审计:`docs/upstream-notes/upstream-v1.18.1-inference-audit.md`;数据:优化文档「M4 追加 4」 |

状态图例:⬜ 未开始 / 🔶 进行中 / ✅ 已落地 / ❌ 已证伪(须附 ABBA 数据)

### TF3 Worker 追加（2026-09-08，本机 C++ v1.18.2 同卡实测）

以下保留本轮 G0–G4 前的工作区既有实现与验收记录。

- 比较对象改为 Go Server 实际 TF3/B16 Worker，同模型、同请求、同并发。
  原 C32 ABBA：Rust 882.4、C++ 1325.4 NN rows/s；与上面的旧 ONNX/fork
  跨 GPU 推算分开，不能混用基线。
- TF3 补 DualFFN 约 +17.7%；新增 q64-serial 恢复 q128 softmax 顺序，
  15 组 half bitwise 一致、TF3 128/128 top-1，独立 Worker ABBA +3.53%。
  普通 q64 在 TF3 只有 126/128，不能照搬旧 ONNX plan。
- 持续 C64 Worker：双 NN 服务线程、NOGRAPH +7.80%；保留单线程默认，
  双线程用独立吞吐配置，不向低并发或 GTP 搜索推广。
- 修复 evaluator 结果漏写请求 hash 导致的缓存失效，以及补 ownership 时
  保留缓存 policy/value 的语义。队列 placeholder 优化未过 1% 门，已回退。
- Windows 原 q128/q64 PTX 不变；三个原 ONNX plan 经过 ORT FP32 16/16
  重验后更新构建指纹。新 TF3 plan 独立绑定模型与 q64-serial 编译能力。
- 完整阶段耗时、精度/布局受控实验、最终配置与复现命令：
  [TF3 Worker 性能审计](tf3-worker-performance-audit.md)。

## 9. 开放问题

> 2026-09-08：以下保留历史问题。第 1 项的除二/per-SM 推算已撤回作为定量结论，
> 由 §8 G0 的同卡、同计时边界参考及后续同精度受控测量重新回答；不能忽略精度差异。

1. fork 每流 1418 的推算基于"2836 = 2 流均匀",未考虑其双流互相抢占 SM
   的折损——真实单流值可能更高,即我们的 kernel 差距可能比 §3-1 估计的大。
   Phase 0 的 direct T(B16) 与 fork 同口径对比后才能定案。
   (M0 追记:直测 T(B16)=18.48ms 后,kernel 差距确认为 ~37% 未归一 /
   ~20-26% per-SM,比旧 §3-1 估计的 11% 大——方向与本问题一致。)
2. ~~每批 ~8ms 非 kernel 开销里,凑批窗口占多少~~——M0 证伪:固定 B 口径
   非 kernel 开销仅 ~0.5ms;搜索语境的 12% 差(A-lite)中窗口策略占多少
   仍开放,但已属搜索侧优化。
3. 负载自适应切换的批尺寸抖动是否引入新的 graph 捕获/淘汰压力
   (per-size 缓存的 LRU 容量)。(方案 A 降级后优先级降低)
4. ~~WDDM 下双计算流的 graph 重放是否有驱动级串行化~~——M0 得证:双线程
   graph **capture** 直接互斥(CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED),
   自动回退 nograph;且双流本身在饱和 kernel 栈上无收益(-2%),问题失去
   实际意义。
