# CUDA 对齐/超越 fork 路线图(cuda-fork-parity-plan)

> 目标:将本仓库 CUDA 后端推理能力提升并超越 KataGomo_fork 的 SM120 认证水平。
> 本文档是跨对话对齐用的**主规划**;战术史与 ABBA 留痕仍在
> `docs/cuda-optimization-plan.md`,fork 技术细节在 `docs/fork-sm120-kernel-notes.md`。
> 每个新会话开始工作前:先读本文件 §1(目标)、§2(现状快照)、§8(进度看板)。

## 1. 目标与验收口径

- **fork 认证成绩**:RTX 5080(84 SM)2836 nnEval/s,拓扑 = 固定物理 B16 +
  尾批 padding + 2 流(plan `sm120-rtx5080-96c8d332dc452f3d`)。
- **本机硬件**:RTX 5070 Ti(70 SM,48 MB L2)。per-SM 归一化对齐线 ≈
  2836 × 70/84 ≈ **2360 nnEval/s**(B16 口径);**绝对超越 2836** 需 kernel
  代际优势(tcgen05 或 FP8)补 20% 的 SM 数差。
- **测量口径**:统一用 `katago-rs nnbench`(对齐 C++ `benchmarknn`,固定物理
  batch 打满,统计物理 nnEval/s)。`--mode eval` = 生产栈全链路;
  `--mode direct` = `CudaModel::apply` 直连,kernel 下界对照;`--mode kernel`
  = 纯前向(无拷贝无 graph)。注意:eval 的 `--workers` 默认 2×**当前** batch
  ——serve_pipelined 的 target 是发射阈值非上限,W/3>target 时 avgBatch 被
  抬到 ~W/3(2026-08-17 实测修复)。
- **正确性门**(任何性能变更的前置):
  `cargo test -p kata_nn --test dump_nn_io_cuda --features cuda --release` →
  `.venv/Scripts/python.exe scripts/compare_nn_output.py <dump目录>`,
  RESULT: PASS(policy 5e-2 / value 2.5e-2 / misc 1e-2 / ownership 5e-3 /
  top-1 100%);启用休眠代码路径前同样先过对拍。
- **性能裁决**:ABBA/BAAB,min_improvement 1%,慢于基线即回退;新 kernel
  一律注册为 tactic 候选经 `scripts/autotune.py` 裁决并入
  `plans/best-tactic-plan.json`(fail-closed)。

## 2. 现状快照(2026-08-17)

### 已落地(详见 cuda-optimization-plan.md 决策组表 G1-G10 全绿)

| 层 | 状态 |
|---|---|
| GEMM | cuBLASLt 接管 hgemm/hgemm_residual/hgemm_f16(默认,`KATAGO_CUDA_CUBLASLT=0` 回退手写 t64/t128) |
| attention | FA2 全 tensor core(QK mma + online softmax + PV mma),RoPE 融合进加载(10c4f1c) |
| 融合 | FUSION=none 为默认(autotune 裁决:cuBLASLt 时代手写融合净负);gatesilu 等保留为 CUBLASLT=0 组合候选 |
| 调度 | 事件门控流水线(submit/finish 分离 + 双槽 + 完成即投递)+ per-size CUDA graph 缓存;serve=1 单消费者 |
| batch | 精确尺寸 + nnMaxBatchSize=16 上限(padding 与 B16 凑满已分别证伪,见 §4 关键警示) |
| autotune | scripts/autotune.py 8 决策组 ABBA + plan JSON fail-closed 指纹校验 |
| 精度路线 | FP16 终局(FP8 精度不足/INT8 无厂商路径,三段实验闭环) |

### 最新性能数字

| 口径 | 值 | 出处 |
|---|---|---|
| search benchmark t=1 | 296→**433** visits/s(cuBLASLt→pipeline) | cuda-optimization-plan.md |
| search benchmark t=8 / t=16 | **1034 / 1274** visits/s | 同上 |
| cap16 nnEvals/s(t=32,avgBatch≈15.9) | **~765** | B16 高并发复测 |
| **nnbench eval B16(固定批,W=32)** | **871 → 915.4 nnEval/s**(C1 重排后,批周期 17.5ms) | M1 ABBA(2026-08-17) |
| **nnbench kernel B16(纯前向)** | **18.48ms**(graph 批周期 18.37;H2D/D2H +0.8ms) | 同上 |
| B16 单流上限(直测) | **871~894 行/s** | 同上(旧推算 12.5ms/1280 作废) |
| fork 每流(推算) | 2836/2 ≈ **1418 行/s/流** | fork plan |

> M0 全量数据与 H1-H5 裁决见 cuda-optimization-plan.md「M0 Phase 0 测量」。
> 关键反转:B16 kernel 已计算饱和(util 98%,B8→B32 线性),生产栈在固定 B
> 口径下已达 kernel 上限 97%;**剩余杠杆在 kernel(GEMM 代际/融合),不在
> 调度拓扑**——双流实测 852.8 vs 单流 871(-2%),证伪。

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
| H4: attention ~14% | ⬆ 上修:FA2 core 13~22%(全段 47% 含 GEMM) | B3(FA4)排期价值上升 |
| H5: 初始卷积 >5% | ❌ 1.6% | 方案 D 搁置 |

附加发现:① cuBLASLt qkv 形状启发式首选非最优(候选 #1 -26%),top-N
计时重排≈前向 4%(方案 C1);② 双流 × 固定 B16 × W64 实测 852.8 vs 单流
871(-2%),WDDM 双线程 graph capture 互斥(开放问题 4 得证)——fork 拓扑
在当前 kernel 栈上无收益。

## 6. 方案清单

### 方案 A:完整复刻 fork 认证拓扑(**M0 后降级**:固定 B 口径调度开销仅 ~3%,双流实测 -2% 证伪)

**降级后的剩余范围(A-lite)**:仅搜索语境的凑批/窗口策略(搜索 t=32 口径
765 vs nnbench 871 的 12% 差距);双流 × 固定 B16 × padding 组合已实测无收益
(852.8 vs 871;WDDM graph capture 互斥),**不再追求完整复刻**。

<details><summary>原方案 A 内容(存档)</summary>

**内容**:固定物理 B16 + 尾批复制 padding + 双槽双流 + batch-aware
dispatch(攒满 target 才发射、GPU 空闲也发射,fork `maybeLaunchFillingBatch`
语义已在 serve_pipelined 中部分存在)。

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
| B3 | FA2→FA4 tile | M128×N64 s1、128 线程、noncausal 无掩码 | attention ~14% 前向(H4 待证),上限 ~7%;**不做 both16**(已证:Hopper 起 FP32/FP16 累加同速,both16 只降精度) |
| B4 | linear2/outproj beta=1 原位残差 | CUTLASS `GemmShape<128,128,32>` warp<64,64,32> 3 stages,C==D | 已部分有(hgemm_residual);CUTLASS 版作候选对照 |

- **前置门槛**:每个融合 kernel 的 GEMM 主循环微基准 ≥ cuBLASLt 同形状,
  否则免谈(FUSION=none 教训);逐项 ABBA + 对拍。
- B16 下 trunk 缓冲 8.9MB、mid 4.4MB,每省一趟往返 ≈10~20µs × 12 block,
  累计预期数个百分点到 10%+。

### 方案 C:tcgen05(sm_120f)——"超越"的唯一确定 kernel 杠杆(**M0 后升为最高优先级**)

- **M0 裁决(H3 证伪)**:cuBLASLt 13.3 在 sm120 对我们全部 FP16 形状只给
  algoId=21 遗留 tile/stages(mma.sync s16816 类),3 形状×8 候选池无任何
  nvjet/tcgen05 痕迹;GEMM 占前向 75~85% 且已在 mma.sync 屋顶(78~92 TF)。
  → C 按"主 GEMM 通路替换"分支动工,上限 = 代际差。
- **C0(零工程试探,动工第一步)**:经典 cuBLAS(cublasGemmEx/hgemm)在
  CUDA 12.8+ 的 sm120 上对部分 FP16 形状会派发 nvjet(tcgen05)kernel——
  加一个 cublas feature 微基准即可验证,若放出 nvjet 则 C 的收益不经
  CUTLASS 工程即可先吃一部分。
- **C1(cuBLASLt top-N 计时重排,M0 彩蛋,独立于 C 主线)**:启发式首选并非
  最优(qkv #1 -26%、ffn_down #2 -3%);load 时对每形状 top-8 候选计时要选,
  预期前向 +3~5%。注册 tactic,ABBA + 对拍。
- **C2 主线**:CUTLASS 4.x sm120 tcgen05 GEMM 模板替换主通路(编译经
  `configs/sm-targets.json` 加 `sm_120f` target,失败回退 sm_120)。
- 注意 SM120 时效资料(cuda-optimization-plan.md §SM120):tcgen05 需
  **sm_120f 家族特性**编译;cta_group::2 有 255 寄存器天花板,tile 取
  cta_group::1 128×N 安全区;参考 CUTLASS 4.x sm120 GEMM 模板与
  gau-nernst 的 tcgen05 逐指令教程。

### 方案 D:初始 3×3 卷积 cuDNN frontend(fork eng45-tile0-stages2)

引入 cuDNN ≥9.24 依赖换 fork 同款 graph engine;本机现状 im2col K=198→208
已实现。**仅当 H5 显示初始卷积占比 >5% 才启动**,预期很小。

### 方案 E:超越 fork 的储备项(大工程,暂不动工,按序评估)

1. **FP8 官方 PTQ scope 重模拟**:KataGo 官方 2026-08(commit ff27077)只
   量化 attention/SwiGLU 权重投影、heads 保持 FP32——比我们已模拟的 fp8w
   scope 更窄,用 `scripts/sim_quant.py` 按官方精确 scope 重跑;微基准已有
   M=5776 fp8 down-proj -58%、qkv -38%。精度过门才谈 kernel 工程。
2. **每批 ~8ms host 开销消减**(依赖 H2 分解):特征编码 GPU 化、pinned
   half 预转换(fork `enablePinnedHalfInputs` 语义)、解码路径裁剪。
3. **持久化 megakernel**:12 trunk block 融合为少数常驻 kernel,activation
   驻留 48MB L2(B16 双缓冲够放);研究型,最后考虑。
4. **WSL 部署**:白捡 +7~12%(t=8 +12%),非 kernel 但验收数字算数;
   环境已备好(scripts/wsl_setup.sh)。

## 7. 执行顺序与里程碑(M0 后重排:kernel 优先,拓扑降级)

| 里程碑 | 内容 | 验收 | 依赖 |
|---|---|---|---|
| M0 | Phase 0 测量 + 决策表落档 | ✅ H1-H5 全部裁决(2026-08-17),数据入 cuda-optimization-plan.md | nnbench ✅ 已提交 |
| M1 | C1 cuBLASLt top-N 计时重排 + C0 经典 cublas nvjet 试探 | ✅ C1 +2.55%(ABBA)+对拍 PASS,入 plan JSON;C0 证伪(2026-08-17) | M0 |
| M2 | B3 FA4 attention tile | 逐项 ABBA + 对拍;入 plan JSON | M0 |
| M3 | C2 tcgen05 主 GEMM 通路(CUTLASS 4.x sm120f);B2 dual-FFN(门槛:主循环 ≥ cuBLASLt ffn_up 0.130ms) | ABBA + 对拍 | M1 |
| M4 | 冲击 per-SM 对齐线 2360;评估 E1/E2 与 A-lite(搜索语境) | nnbench eval B16 对照 fork 认证口径 | M1-M3 |

每个里程碑完成标准:对拍 PASS + ABBA 留痕 + plan JSON 更新 + 本文档 §8
看板更新。**任何一步慢于基线即回退并在 cuda-optimization-plan.md 记证伪。**

## 8. 进度看板(跨对话对齐用,每次会话结束更新)

| 项 | 状态 | 备注/链接 |
|---|---|---|
| 规划文档 | ✅ 本文档(2026-08-17 M0 后修订) | — |
| nnbench 工具 | ✅ 已提交已实测(含 workers=2×当前 batch 修复) | crates/katago/src/cmd/nnbench.rs |
| M0 Phase 0 测量 | ✅ 完成(2026-08-17) | cuda-optimization-plan.md「M0 Phase 0 测量」 |
| H1-H5 假设 | ✅ 全部裁决 | §5 决策表;H1/H2/H3 证伪,H4 上修,H5 否决 |
| M1(C0+C1) | ✅ 完成(2026-08-17) | C0 证伪(经典 cublas 同走 mma.sync);C1 top-N 计时重排落地:ABBA +2.55%(B16 892.6→915.4),对拍 PASS,已入 plan JSON |
| 方案 A 拓扑组合 | ❌ 证伪/降级 A-lite | 双流×固定B16×W64=852.8 vs 单流 871(-2%);WDDM graph capture 互斥 |
| 方案 B1-B4 | ⬜ 未开始 | B3 优先(H4 上修至 13~22%);B2 门槛 ffn_up 0.130ms |
| 方案 C tcgen05 | 🔶 C0 证伪/C1 已落地;C2(CUTLASS sm120f)待动工 | H3 证伪 cuBLASLt 走 tcgen05;探针 tests/probe_cublaslt_algos.rs 已入库 |
| 方案 D cuDNN | ❌ 搁置 | H5:InitialConv 1.6% < 5% |
| 方案 E 储备 | ⬜ 未开始 | E1 重模拟成本最低 |

状态图例:⬜ 未开始 / 🔶 进行中 / ✅ 已落地 / ❌ 已证伪(须附 ABBA 数据)

## 9. 开放问题

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
