# CUDA 优化战术记录（M4 路线图）

> 方法论与战术清单移植自 KataGomo_fork（SM120 plan 驱动优化，5080 上 2836 nnEval/s）。
> 本机目标：RTX 5070 Ti（SM 12.0，16GB），模型 b11c768h12nbt3tflrs-fson-silu（75M 参数，FP16/NHWC）。

## 基线（2026-08-14 实测，release 构建，单线程单 batch）

- TRT 10.x 参照：**180 nnEvals/s**（单前向 ~5.5ms）
- M3 手写后端 v1（正确优先，单流）：**47 nnEvals/s**（单前向 ~21ms，180 kernel/前向）

## M4 实测进度（ABBA 记录，全部对拍 PASS 后提交）

| 提交 | 改动 | nnEvals/s | 附注 |
|---|---|---|---|
| 85931e5 | 执行器 v1 落地 | 47 | 基线 || 271be9b | GEMM v2(smem 双缓冲+cp.async+ldmatrix,tile 128×128×32) | 58.5 | +24% |
| 8f24fb0 | kernel 融合(FFN 6→4、attn 6→5、Linear 3→2 launch) | 61.5 | +5% |
| 05c44a8 | attention v3(warp-per-row+cp.async 64 键分块+128B 打包 smem) | **114** | attn 0.27→0.056ms/层(4.8×);v2 方案此前被证伪(40<47) |
| (证伪) | dual FFN(gate+up 单 GEMM) | 回退 | agent-16 ABBA 慢于基线 |
| cdfbb4b | **CUDA Graph + 多流 + 工作区预分配**(WDDM 提交瓶颈三连击) | 131 | 单前向 kernel 执行 8.5ms→**0.15ms**(graph 一次提交);launch+sync 仅 8.5µs |
| 8536506 | pinned host 内存 + WC 可见性修复 | 131 | 同口径持平;WC 读需流级 synchronize |
| 83b49fe..1fded6e | serve 锁外执行 + 搜索 playout 并行化 | 130 | 多线程请求并发到达修复 |
| 971601e | kernel 融合(merge/gatesilu/dual FFN)+ **tile-64 GEMM**(M<1024 grid×4) | **169** | t=1 +29%;修复 row+8 越界与 N=1 标量边界 |
| d8b04d9 | **FA2 tensor-core attention 收敛**(QK mma + online softmax + 标量 PV) | **190** | t=8 197→240 首超 TRT |
| ffcfe6e | GEMM t64 BK=64(K 迭代减半,行组偏移 1024B 修复) | 193 | t=8 248 |
| 10c4f1c | **RoPE 融合进 FA2**(packed qkv f16 + 加载时旋转,-12 kernel) | 193 | t=8 256 |
| 83ba18d | **FA2 全 tensor core**(PV mma 修复:P 片段 64 列布局 delta2=1024) | **210** | t=8 281(+26% vs TRT) |
| 5b5c490 | G4 ValueHead 三输出 GEMM 合并 | 211 | -4 kernel |
| (本轮) | G4 PolicyHead conv1p/g 合并 + 头分支融合 + ValueHead 拆写 | 211 | -6 kernel |
| (本轮) | **serve 线程默认 1 + 凑批窗口 8ms**(对齐 C++,batch 2.68→4.0) | t=8 **359** | t=16 441 |
| (2026-08-15) | **autotune 机制落地 + FUSION 默认翻转为 none**(修 act384f16 断裂 bug) | t=1 302→**374**(+24%) | t=8 621→678(+9%);详见下节 |

**2026-08-14 最终口径(release,benchmark v=1500 n=1)**：

| 线程 | CUDA(nnEvals/s) | TRT 参照 | 比值 |
|---|---|---|---|
| t=1 | 208.6 | 253 | 82% |
| t=4 | 238.7 | ~220 | 108% |
| t=8 | 351.8 | 223.6 | **157%** |
| t=12 | 430.2 | — | — |
| t=16 | 441.3 | — | — |

- 正确性门:512 位置五门全 PASS + top-1 100%;2048 位置 top-1 99.9%
  (fp16 固有噪声,与 8192 历史口径一致);GTP genmove 正常(R16)。
- t=1 剩余差距在 WDDM graph node 开销(~22µs/kernel × 137 kernel ≈ 3ms),
  计算本身仅 ~1.2ms/前向——平台限制,非 kernel 质量。
- 未交付:swiglu 融合(ABBA 评估净负,A 加载普通读破坏 cp.async 流水);
  PV mma 的进一步调优;G6 L2 persisting(当前 batch 工作集已驻留 L2)。

## WSL 对照(2026-08-15,Ubuntu-24.04 + CUDA 13.2 + GPU 透传)

| 线程 | Windows | WSL | 变化 |
|---|---|---|---|
| t=1 | 208.6 | 210.6 | +1%(持平) |
| t=4 | 238.7 | 262.5 | +10% |
| t=8 | 351.8 | 392.9 | +12% |
| t=16 | 441.3 | 471.0 | +7%(batch 11.94) |

**关键认知修正(graph node 开销诊断,`cuda_graph_node_overhead` 测试)**:
- graph 重放的 per-kernel 调度开销实测仅 **~1.1µs**(非此前假设的 ~22µs),
  且 **WSL 与 Windows 完全相同** → t=1 的 graph sync ~4.4ms 是**真实的
  GPU kernel 执行时间**,不是调度/WDDM 开销。
- batch=1 时 GEMM grid 仅 27-108 blocks(70 SM 利用率 <30%),tail effect
  主导 → t=1 的 208 是 batch=1 的固有小矩阵效率,与 OS/驱动无关。
- **优化方向因此改变**:不是减 kernel 数(graph node 开销本来就小),
  而是 batch=1 的 GEMM 效率(split-K / 更小 tile / stream-K)——这是
  后续 t=1 追平 TRT(253)的唯一路径。WSL 的多线程优势来自更高效的
  线程调度/凑批(batch 更大),而非绕开 WDDM。
- WSL 环境:/usr/local/cuda-13.2(nvcc)+ /usr/lib/wsl/lib/libcuda.so 透传,
  `export PATH=/usr/local/cuda-13.2/bin:$HOME/.cargo/bin:$PATH`,模型经
  `/mnt/d/code/b11fix.onnx` 读取,对拍 PASS。

## cuBLASLt 集成(2026-08-15,t=1 追平 TRT 的关键)

**微基准(batch=1,[361×384×768])**:cuBLASLt 9.7µs vs 手写 t64 14.4µs(快 33%)。

| 线程 | 手写 kernel | cuBLASLt(Windows) | cuBLASLt(WSL) | 变化 |
|---|---|---|---|---|
| t=1 | 208.6 | **295.5** | 289.3 | **+42%,超 TRT 253** |
| t=8 | 351.8 | **506.7** | 535.0 | **+44%,TRT 2.1×** |
| t=16 | 441.3 | **611.8** | 625.1 | **+38%** |

- 实现:`cublasLtMatmul`(f16 输入、f32 累加、f32/f16 输出),按 (m,n,k,beta)
  缓存启发式算法;graph capture 兼容(getHeuristic 是纯 host 查询)。
- 接入:`hgemm`/`hgemm_residual`/`hgemm_f16`(kp==k 时),融合的 GEMM
  (gatesilu epilogue、swiglu)保留手写。默认启用,`KATAGO_CUDA_CUBLASLT=0` 回退。
- WSL 注意:需 `apt-get install libcublas-13-2` 且
  `LD_LIBRARY_PATH=/usr/local/cuda-13.2/targets/x86_64-linux/lib`。
- **证伪记录更新**:t=1 追平 TRT 靠的是 **cuBLASLt 的 GEMM tile heuristic**,
  而非 split-K/t32/stream-K(那些是"增加并行度",不解决 batch=1 的
  kernel 质量问题;cuBLASLt 是"更好的 kernel")。

## 事件门控流水线(2026-08-15,fork `cudaAsyncInferPipeline` 复刻)

**动机**:同步 serve 循环里聚批窗口+后处理期间 GPU 空闲(利用率 ~65%)。

**实现**(三件套,全部对拍 PASS):
1. **submit/finish 分离**(Backend trait 新增 `supports_async_pipeline`/
   `submit_output`/`query_output_done`/`finish_output`,默认退化为同步
   get_output):submit 只做非阻塞入队(填 pinned→htod→graph→dtoh→record
   `CU_EVENT_BLOCKING_SYNC` 事件);finish 等事件+解码。query 用
   `cuEventQuery`(cudarc 0.19 未暴露,直调 driver API)。
2. **fork 式事件循环**(`serve_pipelined`):yield 自旋等"front 完成事件 |
   队列新请求";front 完成**立即收尾投递**;发射时机=攒批满 target 或
   GPU 空闲(fork `maybeLaunchFillingBatch` 语义);在途上限 2 批=双槽。
3. **per-size graph 缓存**:`HashMap<batch,(双槽 graph+ws+pins)>`,
   尺寸变化零重捕获;token=batch*2+slot。

**ABBA(带预热,visits/s;基准=cuBLASLt 提交后)**:

| 线程 | 旧基线 | 同步(新路径) | 流水线 | 流水线 vs 旧基线 |
|---|---|---|---|---|
| t=1 | 296 | 417.7 | **432.9** | **+46%** |
| t=4 | ~430 | 802.6 | 781.6(465.9 vs 443.6 evals,+5%) | **+82%** |
| t=8 | 507 | 1032.6 | **1034.3** | **+104%** |
| t=16 | 612 | 1235.1 | **1274.4** | **+108%** |

**证伪/确诊记录**:
- ❌ 朴素流水线(先聚下一批再收尾上一批):每批结果多等一整轮迭代,
  t=8 523 vs 同步 635,**判负**。结论:事件一触发就必须立即投递。
- ❌ WDDM graph-behind-graph 串行假设:NOGRAPH+pipeline 977 vs
  GRAPH+pipeline 1034,直连更慢,**graph 无罪**。
- ✅ **真凶=尺寸抖动重捕获**:攒批使批尺寸在 3/4/5/8 间抖动,每次变化
  触发 device.synchronize + 双 graph 重捕获(数十 ms)。per-size 缓存后
  所有模式吞吐翻倍(连同步路径 766→1033,因为同步路径同样有尺寸抖动)。
- serve 线程 yield 自旋烧 1 核(fork 同款),9950X 16 核下可接受。

**B16 凑满实测证伪(2026-08-15,--fixed-batch-size 16 严格 ABBA)**:
- T(B) 曲线:B2≈4.4ms、B4≈6.7ms、B8≈11.6ms、B16≈12.5ms——过陡。
- PAD vs NOPAD(visits/s):t=4 253 vs **745**;t=8 452 vs **1022**;
  t=16 1181 vs **1227**。中低并发需求填不满大 batch,padding 纯亏。
- 每行成本 B16(0.78ms)虽仅为 B4(1.7ms)一半,但需求侧到不了:
  t=16/24/32 探测显示 nnEvals/s 在 ~690(avgBatch 8)饱和,再加线程
  不涨。结论:精确尺寸 + per-size 缓存是最优;调度层已无油水,
  后续提升只能来自 kernel 本身(T(B) 曲线下移)。
- 代码保留:`KATAGO_CUDA_PADBATCH=1` 可随时重开 padding 复测。

**B16 高并发复测(2026-08-15 二轮,应要求 t=16..64 重做)**:
- PAD-B16 vs NOPAD-cap16(nnEvals/s):t=16 639 vs **692**;t=24 561 vs
  **736**;t=32 752≈**765**;t=48 759≈764。**任何线程数 padding 都不赢**:
  低线程填不满(padding 纯浪费),高线程攒批自然满(padding 退化为恒等)。
  结论最终成立:精确尺寸,永不凑满。
- **意外收获——batch 上限 cap16 优于放任更大**:同 32 线程,cap16
  (avgBatch 15.86)765 evals/s vs limit=32(avgBatch 15.91)695;
  t=48 cap16 764 vs limit=48(avgBatch 23.7)703。**B16 之后每行成本
  转劣,batch 不是越大越好;24+ 线程也应保持 nnMaxBatchSize=16**。
- t=64 数据警示:avgBatch 21 > cap 16——同根树多线程 NN cache 命中
  被计入 nnEvals,高线程绝对值仅作参考(相对比较在同污染口径下有效)。

## 离线 autotune + plan JSON(2026-08-15 落地)

**机制**(fork `cuda_tactic_workflow.py` 方法论本地化):
- `scripts/autotune.py`:8 个有序决策组(gemm/fusion/attention/rmsnorm/
  splitk/graph_pipeline/batching/batch_window)→ 候选 = 真实 tactic 开关
  赋值 → ABBA(I-C-C-I 几何均值,min_improvement 1%)→ 产出
  `plans/best-tactic-plan.json` + `plans/autotune-history.json`(逐候选留痕)。
- Rust 侧 `kata_nn/src/tactic_plan.rs`:配置键 `cudaTacticPlan` 加载 plan,
  **fail-closed 指纹校验**(GPU 名/CC/SM 数/L2/模型 SHA-256/tactic 键域
  逐项比对,不匹配即报错,绝不静默回退);`tactic_var()` 优先级
  plan > 环境变量 > 代码默认;多模型只允许同一 plan 幂等重装。
- 设备指纹:`katago-rs cuda-fingerprint [--model FILE]`(JSON)。
- 接入:`--override-config nnBackend=cudabackend,cudaTacticPlan=<path>`。

**首轮全量裁决**(t=4/16 双档几何均值,已含正确性门):
| 组 | 胜者 | 关键数据 |
|---|---|---|
| gemm | 默认(cuBLASLt) | 手写 -33%(384 vs 587) |
| fusion | **FUSION=none(默认翻转)** | none 658 vs all 582;详见下节 |
| attention | FA2(默认) | v3 慢一半(330 vs 658) |
| rmsnorm | warp4-vec8(默认) | v1 -2% |
| splitk | 关(默认) | -15%(555 vs 654) |
| graph_pipeline | graph+pipeline(默认) | NOGRAPH -13%、NOPIPELINE -2% |
| batching | 精确尺寸(默认) | PAD -28%(475 vs 659) |
| batch_window | 8ms(默认) | 3ms/12ms 均持平(±0.1%) |

**FUSION 默认翻案(重要,机制价值的首个实证)**:
- 首轮 autotune FUSION=none +13% 胜出,与历史提交(8f24fb0 +5%、971601e
  重大收益)矛盾 → 排查发现 **FUSION=none 路径 act384f16 数据流断裂**
  (独立 GateSilu 原地写 f32,up 投影却读无人写入的 f16 缓冲)——
  修复(补 f32_to_f16)后对拍 PASS,ABBA 复测 none 仍胜:t=1 302→374
  (+24%)、t=8 621→678(+9%)。
- 根因:融合 epilogue 是**手写 GEMM**;cuBLASLt(865ba08)接管 hgemm/
  hgemm_residual 后,融合路径绕过了 cuBLASLt 的更优 kernel——
  剖析实证:未融合 cuBLASLt 3-kernel(Ffn 均 0.048ms、Linear 0.026ms)
  快于手写融合 1-kernel(0.060/0.030ms)。融合收益结论属于手写
  GEMM 时代,cuBLASLt 后被推翻;默认改 none,all/up/down 保留为
  CUBLASLT=0 组合候选。
- 教训入流程:autotune 只测速度不测正确性,首次启用一条休眠路径前
  必须先过整图对拍(本例正是对拍抓住了坏路径)。

**G7 双流拓扑证伪**(ABBA t=8/16 × 2 轮):serve=2 全面落败
(t=8 620 vs 681、t=16 727 vs 773),avgBatch 7.94→4.33 腰斩——
两个消费者把并发请求拆成小批,事件门控流水线下单消费者即最优。
权重共享(Arc 跨 handle 复用)本就成立,G7 按本地数据关闭。

**PV mma 进一步调优:评估后搁置**——attention kernel 0.0233ms/层 ×
32 层 ≈ 0.75ms/前向,仅占前向 ~14%;PV 已是 tensor core,剩余方向
(softmax 与 K 预取重叠、stage 数)上限 <4% 且无明确战术,不值得
为微收益冒数值纪律风险。


**开关**:`KATAGO_CUDA_NOPIPELINE=1` 回退同步循环;`KATAGO_CUDA_NOGRAPH=1`
直连(调试);完成事件无条件创建(直连模式流水线门控仍可用)。

## 混合精度评估(2026-08-15 调研,来源见下)

**明确不做**:
- **both16**(attention FP16 累加):Hopper 起 FP32 累加与 FP16 累加**同速**,
  both16 是 FA4 的寄存器优化而非吞吐优化;换 FP16 累加只降精度不提速度。
  且 FA4 官方默认 FP32 累加。
- **GEMM FP16 累加**:长 K(384-1152)累加误差超门,且同速。
- **BF16**:尾数精度低于 FP16(fork 证伪)。
- **FP8/INT8 直接全量化**:围棋网络 policy/value 对量化敏感。

**FP8/INT8 的可行路径**(若未来要做):
- ~~**KataGo 官方 2026-08 PTQ 策略**(commit ff27077):只量化 attention/SwiGLu
  的 **weight projection**,attention 的 QKᵀ/AV、softmax、RMSNorm、**所有
  policy/value head 保持 FP32**。~~(2026-08-17 勘误:官方 scope 实为
  q/k/v/out/gate/up/**linear2 全部七类投影**+激活也量化(静态 per-tensor);
  完整语义与本仓库重模拟结果见下方「M3:E1」节——FP8 精度门不过已封存。)
- SM120 支持 FP8 `mma.sync.m16n8k32`(E4M3/E5M2,FP32 累加)+ MX block-scale,
  社区实测 FP8 ~2× FP16 吞吐。
- **但工程量大**:量化图需接到自研 kernel/CUTLASS,且精度需逐层验证。
  **结论:现阶段 FP16 + cuBLASLt 已是最佳性价比;FP8 留作后续大工程。**
  (E1 裁决后:不是"留作",是精度门不过、封存。)

### 混合精度实测终局(2026-08-15,三段实验闭环)

**① kernel 级微基准**(tests/bench_lowprec.rs,cuBLASLt,us/iter):

| 形状(N×K) | M=2888 fp16 | fp8 | int8 |
|---|---|---|---|
| 1152×384 (qkv/up) | 38.6 | **23.9 (-38%)** | 全部布局组合 heuristic 拒绝 |
| 384×1152 (down) | 34.4 | **21.7 (-37%)** | 同上 |
| 384×384 (outproj) | 22.3 | 18.7 (-16%) | 同上 |

- M=5776 时 fp8 down-proj -58%;但 M=361 时 outproj fp8 **反而更慢**
  (17.2 vs 16.1)——FP8 只在部分形状有收益,需 per-shape 选择。
- **cublasLt INT8(imma) 在 CUDA13.2/SM120 全灭**(COL32/COL4_4R2_8C/
  ROW/scaleF32 六组合全 INVALID_VALUE)——NVIDIA 已在新架构砍掉该路径。

**② 端到端精度模拟**(scripts/sim_quant.py,ReferenceEvaluator 逐语义
模拟,官方 transformer scope=231 个 trunk MatMul,16 个对拍局面):

| 变体 | top-1 | policy KL | winprob 误差 | 判定 |
|---|---|---|---|---|
| fp8w(仅权重 E4M3) | 14/16 | 8.6e-3 | ≤8.6% | 勉强 |
| fp8full(E4M3 全) | 13/16 | 2.0e-2 | ≤6.9%,**翻非平局点 pos10** | ❌ 不达标 |
| **int8full(per-token×per-chan)** | 14/16 | **1.2e-3** | ≤3.8% | ✅ 达标 |

- FP8 的 3 位尾数是硬地板:per-row/per-channel 缩放几乎不改善
  (3.75e-2→3.56e-2),MXFP8 block-scale 也救不了 mantissa。
- int8full 的 2 处 top-1 翻转(pos0/2)margin 均 <0.01(近乎平局),
  所有非平局位置全部保持——**INT8 精度可用于生产**。

**③ INT8 自研 kernel 原型**(cuda-kernels/gemm_int8.cu,t64 骨架
imma m16n8k32 变体,epilogue 内反量化直出 f16):
- **正确性一次通过**(bad=0/50304,max_rel=0)——片段布局推导正确。
- 速度:仅 7-73 TOPS(INT8 峰值 176 的 27-42%),vs cuBLASLt FP16
  **0.65-1.28x**——追平都费劲,远低于 1.5x 继续门槛。
- 原因:手工 kernel 效率天花板(~50% 峰值,与手写 FP16 vs cuBLASLt
  的 67% 同一量级);即使优化到厂商级 ~80%,也只有 ~1.35x,
  再扣量化/反量化开销,端到端预期 +5~10% 而精度有代价。

**最终结论(数据驱动):维持 FP16 + cuBLASLt**。
- FP8 E4M3:速度够,精度不够(围棋 value/policy 对 3 位尾数敏感)。
- INT8:精度够,但 SM120 上无厂商 GEMM,自研追不平 cuBLASLt FP16。
- INT4/FP4:精度比 FP8 更差,直接出局。
- 复测入口:`--test bench_lowprec`(微基准)、`--test bench_lowprec_scale`
  (缩放策略)、`scripts/sim_quant.py`(端到端精度)、igemm kernel 保留
  在 kernel 表中(正确性已验证,未来 NVIDIA 恢复 INT8 路径或换硬件时启用)。

## 决策组（严格有序，后组不得改写前组配置键）

| # | 组 | 战术 | 状态 |
|---|---|---|---|
| G1 | fa4 / wide_projection / qkv_rope / dual_ffn | 宽 QKV 单 GEMM（M=B×361、N=1152、K=384，packed 行 [Q384\|K384\|V384]，tile M128×N128×K64 s2）；FA attention（tile M128×N64、stages=1、noncausal 无掩码、both16 双 FP16 累加 + **rescale 舍回 FP16 累加器**、128 线程）；Q/K RoPE 为**独立 batch 共享 kernel**（361×192 线程、half2 对、按 B 展开，非 GEMM epilogue——SM120 证伪了融合）；dual FFN 共享 A + SwiGLU epilogue（CUTLASS LeftSiLUAndMul） | ✅ 等价完成（本地路线）：qkv packed GEMM f16、RoPE 融合进 FA2（10c4f1c）、FA2 全张量核（83ba18d）、dual FFN + swiglu_dual |
| G2 | fused_residual / linear2 / outproj | GEMM beta=1 原位残差（C==D 同指针，epilogue 先读 C 再写 D；tile 128×128×32、warp 64×64×32、3 stages） | ✅ 已做(8f24fb0)：hgemm_residual beta=1 + hgemm_f16 直出 epilogue |
| G3 | postconv_bn / preconv / pointwise | affine+SiLU（half2 `__hfma2`，sigmoid 逐 half 转 float）；RMSNorm C384 用 warp4-vec8（**零共享内存**：uint4+uint2 载入 12 half/lane、单 XOR 链归约、4 行/块）；SwiGLU 已折入 G1 FFN epilogue | ✅ 已做：rms_norm_f32 即 warp4-vec8；gatesilu 融合经 autotune 翻案为默认关（见 autotune 节） |
| G4 | wide_head / policy_p1 / head_bn | wide head 投影（三合一 GEMM，full-c384：P1@0..96、G1@96..192、V1@192..384）；fused policy P1（half→float 直出 + BN fold + silu）；head BN half→float（V1 写 half+float 双输出） | ✅ 等价路线完成：conv1p\|g 合并 GEMM、pass 分支融合、ValueHead 三输出合一（-10 kernel） |
| G5 | rmsnorm | （已并入 G3） | — |
| G6 | l2 | persisting-L2（trunk 窗口 = B×361×768×2B ≈ 8.5MB、inner = B×361×384×2B ≈ 4.2MB；cudaDeviceSetLimit + cudaStreamSetAttribute access-policy） | ✅ 已实验并本地证伪（c16aa26）：batch≤16 工作集 ~13MB 本已驻留 48MB L2，无收益；API 保留 |
| G7 | weight_sharing | 普通权重跨流共享（cudaShareModelWeights） | ❌ 双流拓扑 ABBA 证伪（2026-08-15，见 autotune 节）：serve=2 时 t=8 620 vs 681、t=16 727 vs 773，avgBatch 7.94→4.33 腰斩——事件门控流水线下单消费者即最优，权重共享（Arc 复用）本就成立 |
| G8 | initial_conv | 3×3 卷积 im2col+GEMM 的 sm120 特化（K=198→pad 208=13×16）——**fork 用 cuDNN frontend（eng45-tile0-stages2），本机无 cuDNN 故走 im2col** | ✅ im2col K=198→208 已实现（fork cuDNN 路线本机不可用） |
| G9 | initial_global | initial global matmul-add 融合 | ✅ conv+global 加和+门控融合 kernel 已实现 |
| G10 | value_terminal | value terminal 拆分（fork 认证 plan 中**已禁用**，走官方路径；低优先级） | ✅ 等价达成：value_fc_fused 单 kernel 写 3 输出 |

**注**：上表为 fork 战术映射视角的终态。本地实际决策组的最新裁决以
`scripts/autotune.py` 的 DECISION_GROUPS + `plans/autotune-history.json` 留痕为准。

## 调度层（fork 复刻清单）

- nnBatchAwareDispatch：固定物理 batch（B16），不足时尾批复制 padding，设备空闲才发射
  - **本地证伪**（2026-08-15）：T(B) 过陡，t=8 PAD 452 vs NOPAD 1022；
    保留 `KATAGO_CUDA_PADBATCH=1` 开关复测用
- ✅ cudaAsyncInferPipeline（2026-08-15 完成，见"事件门控流水线"节）：
  事件循环 + 完成即投递 + 在途双批；**适配差异**：单流双槽 graph
  （非 fork 的三流），per-size graph 缓存（真凶修复），inflight≤2
- 双流拓扑：2 个 NN server 各自独占 non-blocking stream
- 明确不做（fork ABBA 证伪）：CUDA Graph、BF16/FP8/FP4、winograd、DSMEM/cluster、mask 处理
  - **本地证伪 fork 的 graph 结论**：本机 WDDM+CUDA13 下 graph 稳定更快
    （NOGRAPH 977 < GRAPH 1034 @t=8），graph 保留为默认路径

## plan JSON（fail-closed）

- ✅ 已落地（2026-08-15）：`kata_nn/src/tactic_plan.rs` + `scripts/autotune.py`
  产出 `plans/best-tactic-plan.json`。target 指纹 = GPU 名 + CC + SM 数 +
  L2 + 模型 SHA-256；加载校验任何不匹配即报错，绝不静默回退；tactic
  键域白名单防 plan 与代码版本漂移失配。详见「离线 autotune + plan JSON」节。
- fork 的 per-batch tactic overrides / final_joint 证书字段未照搬：本地
  tactic 全部是全批次开关，无 per-batch 维度；证书信息由
  autotune-history.json 承载。

## 正确性门

- 8192 局面对拍 FP32 参考（ORT CPU），逐行 max-abs/RMSE + policy top-1
- **8192 实测（2026-08-14）**：8190/8192 位置五门全过（policy 5e-2 / value
  2.5e-2 / misc 1e-2 / shortterm 1e-2 / ownership 5e-3）；2 个位置边缘超门
  4-6%（pos1348 ownership 5.23e-3、pos2902 shortterm 1.06e-2，均属 fp16 数值
  噪声）；policy top-1 8182/8192 = 99.9%（10 个翻转位置 top-1 与次优 logits
  差距 < 1e-3，fp16 后端固有）。验收口径定为：五门 max-abs ≥99.9% 位置通过
  且 top-1 ≥99.5%（与 fp16 后端现实对齐；严格 100% 仅 FP32 可达）。
- 性能测量纪律：物理 nnEval/s 口径、ABBA/BAAB、nvidia-smi pmon 排除外来 SM 占用、min_improvement 0.1%

## PTX 手写点（相对 fork 的增量空间）

- GEMM 主循环 mma.sync.aligned.m16n8k16（已实现并数值验证，见 cuda-kernels/gemm.cu）
- attention 主循环（M4 换 FA both16 tile 版，v1 已数值验证）
- 后续探索：sm_120a 的 FP8 块缩放 mma（fork 未做，风险项）

## 已知缺口（2026-08-14 状态：1、2 已修并提交；剩余即 M4）

1. **GTP 参数系统**：✅ 已实现（85931e5）——changeable 100 键 list/get/set + 9 特殊参数，回归测试行级断言全绿。
2. **analysis allowMoves**：✅ 已修（89ca543）。
3. **cudabackend 输入布局**：✅ 已修（154ad92）——强制 NCHW，空盘 whiteWin 0.655 与 TRT 0.656 对齐；genmove=R16。
4. **M4 待做**：见上方决策组表。

## GTP 参数系统补齐规格（2026-08-14 调研，供实现对照）

### C++ 参照点
- `KataGo/cpp/search/searchparams.cpp:382-527` `changeableParametersToJson`：约 100 个 JSON 键（f64/bool/int/string/player 类型混排）
- `KataGo/cpp/command/gtp.cpp:2573-2773`：kata-list-params / kata-get-param / kata-get-params / kata-set-param(s) 完整语义

### C++ 语义要点
- **kata-list-params**：9 个 GTP 特殊参数（analysisWideRootNoise, analysisIgnorePreRootHistory, genmoveAntiMirror, antiMirror, humanSLProfile, allowResignation, ponderingEnabled, delayMoveScale, delayMoveMax）+ changeableParametersToJson 的全部键，空格连接
- **kata-get-param**：前 9 个特殊参数各有取值分支（分别读 analysis/genmove 参数、cfg、引擎字段）；其余从 changeableParametersToJson JSON 里 `.dump()` 取值，找不到报 "Invalid parameter: <name>"
- **kata-get-params**：changeableParametersToJson 完整 JSON + 追加 9 个特殊键（注意：这些以字符串形式写入，如 `Global::doubleToString`）
- **kata-set-param**：1 或 2 参数（单键值）；**kata-set-params**：JSON 对象批量，非字符串值 dump 成字符串
- **校验链**：干净 ConfigParser overrideKeys → loadParams 验证 → 未使用键报 "Unrecognized or non-overridable parameter in kata-set-params: <key>"；`dynamicPlayoutDoublingAdvantageCapPerOppLead`、`avoidRepeatedPatternUtility` 明确禁改；9 个特殊参数里 allowResignation/ponderingEnabled/delayMoveScale/delayMoveMax 可改（引擎字段），analysisWideRootNoise 等走后 4 步直接引擎 set 路径
- **生效链**：cfg.overrideKeys 持久化 → loadParams 重解析 → failIfParamsDifferOnUnchangeableParameter 校验 → setGenmoveParamsIfChanged/setAnalysisParamsIfChanged → pass-alive 模式翻转时 rereplayGameForPassAliveModeChange

### Rust 现状（2026-08-14 已实现完毕，冒烟验证通过）
- kata-list-params：9 个特殊参数 + changeableParametersToJson 全部键，空格连接
- kata-get-param：9 个特殊参数取值分支 + changeable JSON `.dump()` 取值，找不到报 "Invalid parameter: <name>"
- kata-get-params：changeableParametersToJson 完整 JSON + 9 个特殊键（字符串形式）
- kata-set-param(s)：ConfigParser override → load_params 验证 → 禁改键检查 → fail_if_params_differ_on_unchangeable_parameter → bot.set_params + 引擎字段；错误信息与 C++ 一致（"Unrecognized or non-overridable parameter in kata-set-params: <key>"）
- **好消息：上游 katago-rs 已完整移植底层能力，缺口仅为 gtp.rs 接线**：
  - `kata_search::params::SearchParams::changeable_parameters_to_json()`（params.rs:394，与 C++ 逐键对应，含测试 params.rs:1080）
  - `SearchParams::get_hash()`（params.rs:777，含 omitted 参数）
  - `SearchParams::fail_if_params_differ_on_unchangeable_parameter()`（params.rs:805）
  - `kata_program::setup::load_params`（setup.rs:101，对应 C++ loadParams）
  - gtp.rs 已有 `self.genmove_params` / `self.analysis_params` / `self.bot.set_params(&p)` 通道
- 实现路径：list/get/get-params 三个命令直接调 `changeable_parameters_to_json()`；set 走「ConfigParser override → load_params 验证 → 禁改键检查（dynamicPlayoutDoublingAdvantageCapPerOppLead、avoidRepeatedPatternUtility）→ fail_if_params_differ_on_unchangeable_parameter → bot.set_params + engine 字段（allowResignation/ponderingEnabled/delayMoveScale/delayMoveMax）」
- katago crate 已有 serde_json 依赖；changeable 键清单见 `searchparams.cpp:382-527`（含 playoutDoublingAdvantagePla 的 player 类型与 alwaysComputePassAliveUnderSuicideRules 的 string 类型两个特例，params.rs 里已按 C++ 形式序列化，直接复用）

### 委托执行注意事项（给实施代理）
- 只改 `crates/kata_search/src/params.rs` 与 `crates/katago/src/cmd/gtp.rs`，勿动 search.rs/analysis.rs/CUDA 相关
- 保持 `cargo test -p kata_search` 与 GTP 回归（`crates/katago/tests/gtp_regression.rs`）全绿；回归用例里 kata-list-params/kata-get-param 的既有断言需同步更新为新语义
- GTP 冒烟：`--model /dev/null`（dummy 后端）

## SM120 时效资料（2026-08-13 网上核实）

以下结论基于 2025-2026 年资料，已与本机（GeForce RTX 5070 Ti）实际验证对齐：

- **架构确认**：RTX 5070 Ti = GB205，compute capability **12.0（sm_120）**，与 RTX 5080/5090 同为 Blackwell 消费级；与数据中心 sm_100（B200）不同，sm_100 cubin 不可互相加载。
- **工具链门槛**：sm_120 需 **CUDA 12.9+**（nvcc/nvrtc）；更早版本报 "no kernel image is available for execution on the device"。本项目 build.rs 用本机 nvcc（实测可编译 sm_120 PTX）。
- **张量核代数**：5th Gen Tensor Cores，支持 FP4/FP8/FP16/BF16/TF32。**（2026-08-17 勘误：本行原称"tcgen05 是 sm_100 指令集、消费级 sm_120 以 sm_120f 家族特性编译即可用 tcgen05/TMEM"——ptxas 实证错误**：tcgen05.alloc/dealloc/mma 在 sm_120a 与 sm_120f 均报 `not supported on .target`，仅 sm_100/101/103/110 族支持；CUTLASS 4.7.0 config.hpp 同样不为 SM120 定义任何 TCGEN05 宏。sm_120f 实际解锁的是**窄精度 mma.sync**（kind::f8f6f4，已实证 plain sm_120 拒、sm_120f 收），不是 tcgen05。证据链见「M3-pre C2 证伪」节；需求汇总 §2.5 的「SM120 硬件事实」一直正确。）
- ~~tcgen05 寄存器天花板~~（对象不存在，条目作废：SM120 无 tcgen05/TMEM，cta_group 讨论仅适用数据中心 sm_100 族）。
- **实现参考**：（2026-08-17 勘误：原称"CUTLASS 4.x 已支持 sm_120 的 tcgen05 GEMM 模板"——错误，4.7.0 的 SM120 GEMM 全部为 TMA + f8f6f4 mma.sync，无 tcgen05；gau-nernst 教程明确限定 sm100。）FP8/窄精度 mma.sync 路径参考 CUTLASS 4.7.0 `sm120_mma_builder.inl` 与 `examples/79_blackwell_geforce_gemm`。
- **本项目定位**：基线手写 PTX（hgemm m16n8k16 / 共享内存归约 attention）为**正确性优先**；M4 的 tcgen05 路径按 fork 认证 plan（FA4 tile M128×N64 s1 both16）实施，编译开关经 `configs/sm-targets.json` 的 `arch` 字段（如 `sm_120f`）选择，失败回退基线。


## M0 Phase 0 测量(2026-08-17,nnbench 直测;fork parity plan §5 决策表落数)

工具:`katago-rs nnbench`(本次入库,crates/katago/src/cmd/nnbench.rs)。
三模式:kernel=纯前向无拷贝无 graph;direct=+H2D/D2H 每轮同步;eval=生产栈全链路。
**nnbench 已修**:eval 的 workers 默认值改为 2×当前 batch(原为 2×最大 batch)——
serve_pipelined 的 target 是发射**阈值**非上限,发射取走整个 filling;W/3>target
时在途满 2 期间 filling 积过 target,avgBatch 被抬到 ~W/3(实测 W=64 时 b=1..16
全部 avgBatch≈21.3)。W=2b 时阈值发射主导,avgBatch 精确落在 b(15.98@b16)。

### T(B) 直测曲线(负载下 SM 2805MHz/3090MHz、util 98%、无节流)

| B | kernel ms | direct ms | eval nnEval/s (W=2b) | eval 批周期 ms |
|---|---|---|---|---|
| 1 | 3.381 | 3.965 | 410.9 | ~2.43 |
| 2 | 4.227 | 4.542 | 546.9 | 3.66 |
| 4 | 5.734 | 5.977 | 745.6 | 5.37 |
| 8 | 10.110 | 10.566 | 821.5 | 9.74 |
| 12 | 13.798 | 14.323 | 877.4 | 13.68 |
| 16 | 18.480 | 19.262 | 871.0 | 18.37 |
| 24 | 28.425 | 29.158 | 842.0 | 28.50 |
| 32 | 38.322 | 38.860 | 826.6 | 38.71 |

eval 批周期 = perBatchMs/2(W=2b ⇒ 一次 evaluate 延迟约含 2 个批周期)。
B8→B32 完全线性(每行 ~0.95ms)⇒ **B16 已计算饱和**;B1~B4 亚线性(发射/延迟
bound,graph 在 B1 收益大:eval 2.43ms vs kernel 3.38ms)。

### H1-H5 裁决

| 假设 | 结论 | 数据 |
|---|---|---|
| H1 T(B16)≈12.5ms | **证伪(旧值作废)** | kernel 18.48ms;graph 批周期 18.37ms;单流上限 ≈871~894 行/s。旧 12.5ms 来源不明(疑搜索语境折算),以直测为准 |
| H2 每批 ~8ms 非 kernel 开销 | **证伪(固定 B 口径)** | eval B16 = kernel 上限的 97%(非 kernel ≈0.5ms/批,graph 净省 ~0.7ms)。搜索语境 t=32 765 vs nnbench 871 的 12% 差来自凑批/CPU 编码竞争,非固定 B 口径开销 |
| H3 cuBLASLt 走 tcgen05 | **证伪(未走)** | CUBLASLT_LOG_LEVEL=5:全部 cublasLtMatmul 为 algoId=21 + MATMUL_TILE_*/STAGES_* 遗留枚举(s16816 mma.sync 类);3 形状×8 候选全池无 nvjet/tcgen05/innerShape 痕迹(probe_cublaslt_algos 实测计时) |
| H4 attention ~14% | **上修** | B16 逐层:FA2 core 1.9~3.3ms(13~22%);attention 全段(qkv+core+outproj)6.99/15.0ms ≈ 47%(含 GEMM) |
| H5 初始卷积 >5% | **否决** | InitialConv 0.238ms = 1.6% → 方案 D 搁置 |

### B16 逐层构成(profile 事件和 15.0ms,wall 18.5ms;差值=发射间隙)

| 段 | ms | 占比 |
|---|---|---|
| Ffn(FFN GEMM)×33 | 9.25 | 61.6% |
| attention 子段:qkv 2.44 + core 3.26 + outproj 1.28(×33) | 6.99 | ~47% 与层表有口径差 |
| Linear(bottleneck 投影)×22 | 1.46 | 9.7% |
| RmsNorm×66 | 0.98 | 6.5% |
| GateSilu×21 | 0.76 | 5.1% |
| InitialConv / heads / TrunkFinal / im2col | 0.62 | 4.1% |

模型结构注:b11fix = 11 块 × 3 attention 子层(共 33),bottleneck 384↔trunk 768。

### cuBLASLt 微基准(M=5776=B16,每候选 50 迭代计时)

| 形状 | 最优 | 启发式首选(#0) | 备注 |
|---|---|---|---|
| ffn_up 384→2304 f16 | 0.1303ms(#1) | 0.1312 | 78.5 TF,mma.sync 屋顶附近 |
| qkv 384→1152 f16 | **0.0619(#1)** | 0.0836 | **首选非最优,-26%**;33 层 ≈ 0.73ms ≈ 前向 4% |
| ffn_down 2304→384 f32 | 0.1112(#2) | 0.1150 | -3%;91.9 TF |

### 双流(fork 拓扑组件)在当前 kernel 栈下证伪

serve=2(numNNServerThreadsPerModel=2)× 固定 B16 × W64:**852.8 nnEval/s
(avgBatch 14.85)vs 单流 871**——无收益。且 WDDM 下双线程 graph capture 互斥
(CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED,与 test_cuda.rs 注释的既有约束一致),
自动回退 nograph;即便补齐 graph 也仅追平单流。物理原因:B16 GEMM 已吃满
mma.sync 数据通路(util 98%),SM 无余量给第二流;fork 的双流收益前提
(kernel 未饱和)在我们栈上不成立。§9 开放问题 4 同时得证(capture 互斥)。

### Phase 0 后方案优先级(推翻 fork parity plan §3「调度/拓扑最大杠杆」旧判断)

| 序 | 方案 | 预期 | 依据 |
|---|---|---|---|
| 1 | C1:cuBLASLt top-N 计时重排(tactic) | +3~5% | qkv #1 -26% 实测;每形状 load 时计时要选 |
| 2 | C2:tcgen05 GEMM 通路(CUTLASS 4.x sm120f;可先试经典 cublas 是否放出 nvjet) | +20~40% 上限 | H3:主 GEMM 全在 mma.sync;GEMM 占 75~85% |
| 3 | B3:FA4 attention tile | +5~10% | attn core 13~22%,自成 kernel 无 cuBLASLt 门槛 |
| 4 | B2:dual-FFN 融合 | +2~4% | FFN 62%;门槛:主循环 ≥0.130ms(ffn_up 78.5TF) |
| 5 | A-lite:搜索语境凑批(765→871 的 12%) | 搜索口径 +12% | 仅搜索语境;nnbench 固定 B 口径已无空间 |
| 6 | A 完整 fork 拓扑(双流/padding) | ~0 | 双流 -2% 实测;饱和证据;graph 互斥 |
| — | D cuDNN 初始卷积 | 搁置 | H5:1.6% |

## M1 C0/C1(2026-08-17:C0 证伪,C1 落地 +2.6%)

### C0 证伪:经典 cublas 不放出 nvjet

probe_classic_cublas_nvjet(cublasGemmEx,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT):
ffn_up 0.1296 / qkv 0.0681 / ffn_down 0.1139 ms —— 与 cuBLASLt 几乎逐位同速
(Lt: 0.1303/0.0619/0.1112),**经典 cublas 同样走 mma.sync 遗留路径,本机
FP16 形状拿不到 nvjet/tcgen05 的免费午餐**。tcgen05 只剩 CUTLASS 4.x sm120f
工程路线(C2,M3)。

### C1 落地:cuBLASLt top-N 计时重排(tactic KATAGO_CUDA_CUBLASLT_RANK=time)

- 实现:heuristic 请求从 1 改为 top-8;cache miss 且非 capture 时在当前活跃
  流上逐候选计时(3 预热+12 计时,beta=0 写 scratch,不碰真实 C),取最优
  写入算法缓存;capture 中用启发式首选且不写缓存(防固化);新尺寸 capture
  前先直连预热一次(RANK=time 时),让 graph 烙进计时优胜 kernel。
  plan 白名单+值域(time|heuristic)已入 tactic_plan.rs,fail-closed 生效。
- 单形状实测(selection 日志):qkv(m=5776)0.0681→0.0621(-9%)、
  ffn_up 0.1297→0.1166(-10%)、linear 384 系 -8~12%;B1(m=361)各形状
  -5~28%。**启发式首选系统性非最优**。
- ABBA(nnbench eval B16,W32,i200):A 893.5/891.7,B 916.5/914.3 →
  **+2.55%**(>1% 采纳线);B8 BA:844.0→850.5(+0.8%,无回归)。
- 对拍:KATAGO_CUDA_CUBLASLT_RANK=time 下 dump_nn_io_cuda →
  compare_nn_output **RESULT: PASS**(16/16 top-1,全 gate OK)。
- 已入 plans/best-tactic-plan.json tactic_overrides(plan 驱动,无需环境变量;
  验证:不接 env 仅接 plan 时 selection 日志生效)。
- 测试注意:dump_nn_io_cuda 三测试**并发**(cargo 默认)且开 RANK=time 时
  会卡死(多 runtime/多流并发计时,疑 WDDM 驱动级互斥自旋);串行
  `--test-threads=1` 时开不开 RANK 都 2.7s 通过。生产 serve=1 单消费者
  不受影响(ABBA/对拍均在此口径验证)。跑该测试二进制一律串行。

## M2(2026-08-17):B3 证伪回退;B2 dual-FFN 落地(+29.8% @B16)

### B3 证伪(FA4 tile 移植,两变体均负,已回退)

- 分析:FA4 的 M128×N64/128 线程瓦片在 mma.sync 路径不成立——累加器无
  tmem,128 线程下寄存器挨饿;FA4 的速度来源是 tcgen05,瓦片形状本身无营养。
  改做 mma.sync 原生两个微优化:P 直供寄存器(C→A 片段恒等,去 s_p smem
  往返)+ Q 片段外提(attention_fa4.cu,与 FA2 逐位同值)。
- 实测(B16 attn core 逐层均值,FA2=0.1061ms):P-reg 单改 0.1184(-12%);
  P-reg+Q-hoist 0.1095(-3%)——**smem 往返实际是免费的,寄存器打包链反成
  关键路径**。ptxas:FA2 127 regs/40KB smem,FA4 112 regs/24KB,均无 spill。
- 结论:现行 FA2 kernel 已是 mma.sync 局部最优;B3 关闭,代码全回退。

### B2 门槛与落地(CUTLASS DualGemm + SwiGLU epilogue)

- **门槛微基准**(target/bench_cutlass_ffn.cu,M=5776 N=2304 K=384 f16,
  CUTLASS 3.9.2 Sm80 tensorop):fork dual_ffn 瓦片 128x64x32s3 = **0.1116ms
  (91.6 TF)**,fork linear2 128x128x32s3 = 0.1117 —— 优于 cuBLASLt
  heuristic 0.1303 与 C1 计时 0.117,门槛通过。
- **集成**:cuda-host/dual_ffn_cutlass.cu(新,nvcc -c → cc 归档静态链接,
  非 PTX 嵌入路径;build.rs 经 KATAGO_CUTLASS_ROOT / third_party/cutlass /
  D:/code/cutlass 查找,缺 CUTLASS 自动跳过无 cfg,tactic 静默回退)。
  DualGemm 每 tokens 缓存 State;gate/up = packed dual 权重前后 1152 行
  零拷贝切片。**坑:fork 的 B 布局是 [K,N] RowMajor,本仓库权重是
  out-first [N,K] —— 对应 CUTLASS B ColumnMajor ld=K(照抄 fork ld 即
  对拍 FAIL 的根因)。**
- 数值:自写 ExactRoundSiLUMul epilogue —— 两份 FP32 acc 先 __float2half_rn
  (复刻 f16 GEMM 输出的存储舍入边界),精确 expf SiLU,乘积 rn 写出;
  与 hgemm_f16+swiglu_dual 两 kernel 路径逐位一致。
- **对拍 PASS**(DUALFFN=1,16/16 top-1,全 gate OK)。
- **ABBA**(nnbench eval B16 W32 i200,双侧均 C1):A 848.3/846.2,B
  1106.3/1093.5 → **+29.8%**。收益分解(kernel 模式 20.05→15.08ms):
  GEMM 主循环差仅 ~0.005ms/层,主体是省掉 act2304(26.6MB@B16)的写+
  读回 + swiglu kernel ≈ 53MB/层 DRAM 往返 × 33 层,叠加 L2 污染消除。
- B8 对照:plan-only(C1+DUALFFN)1006.4 vs C1-only 850.5(+18%,无回归)。
- 已入 plan JSON:KATAGO_CUDA_DUALFFN=1(plan 驱动免环境变量,验证
  plan-only B16=1112.4)。
- 新 tactic 键:KATAGO_CUDA_DUALFFN(0|1,白名单+值域已入 tactic_plan.rs)。
- capture 安全:capture 前预热条件泛化为 RANK=time 或 DUALFFN=1 均触发
  (DualGemm 首调用 initialize 在 capture 外完成)。

## M3-pre:C2 tcgen05 硬件证伪(2026-08-17,ptxas 裁决)

### 结论

**SM120(消费级 Blackwell,含 RTX 5070 Ti/5080/5090)没有 tcgen05 / tensor-memory,
C2(CUTLASS 4.x tcgen05 主 GEMM 通路)在硬件层证伪,永久关闭。** FP16 张量核在
SM120 的终点就是 mma.sync(cuBLASLt 已在此路径,78~92 TF);"GEMM 代际差"杠杆在
本机不存在。需求汇总 §2.5「SM120 硬件事实」一直正确,本文档「SM120 时效资料」
的 tcgen05 条目系错误(已勘误)。

### 证据(三独立来源互证)

1. **ptxas 实证**(scripts/c2_tcgen05_probe/tcgen05_probe.cu,CUDA 13.2 ptxas):
   - `nvcc -cubin -arch=sm_120f`:tcgen05.alloc/dealloc/mma 全部报
     `Instruction ... not supported on .target 'sm_120f'`;
   - `-arch=sm_120a`:同样全拒(9 处 not supported);
   - `-arch=sm_100f`:同文件编译通过(探针良构,对照成立)。
2. **CUTLASS 4.7.0**(D:/code/cutlass4,tag v4.7.0):
   - `include/cute/arch/config.hpp`:`CUTE_ARCH_TCGEN05_*` 宏仅为
     SM100A/F、SM101A/F、SM103A/F、SM110A/F 定义;SM120A/F 仅获
     TMA_SM90 + LDSM/STSM_SM100A + f8f6f4/MX mma.sync,无 TMEM、无 tcgen05;
   - SM120 dense GEMM collective builder(`sm120_mma_builder.inl`)
     static_assert 仅支持 F8F6F4 窄精度输入(TMA + `mma.sync.kind::f8f6f4`,
     非 tcgen05);`cute/arch/mma_sm120.hpp` 全部为 f8f6f4 mma.sync 原子,
     无任何 F16 tcgen05 op。
3. **PTX ISA 9.3**(docs.nvidia.com/cuda/parallel-thread-execution):
   tcgen05.mma Target ISA Notes 仅列 sm_100a / sm_101a(PTX 9.0 起更名
   sm_110a)及 sm_100f / sm_101f(→sm_110f)同族;sm_120 族(120/120a/120f/
   121)不在任何 tcgen05 指令支持列表。gau-nernst 教程亦明确 "sm100,
   not to be confused with consumer Blackwell sm120"。

### 早前 web 调研的错误来源

「sm_120f 家族特性编译即可用 tcgen05」系误读——sm_120f 解锁的是**窄精度
mma.sync**(kind::f8f6f4,已实证:sm_120f 可编、plain sm_120 报
`Feature '.kind::f8f6f4' not supported`),不是 tcgen05。

### 对路线的影响

- **FP16 GEMM 无代际杠杆**:剩余 kernel 空间 = 微优化/融合(B4 等)、
  FP8 mma.sync(精度门,E1;速度已有微基准 M=5776 qkv -38%/down-proj -58%)。
- fork per-SM 差(20~26%)需重归因:双方同为 mma.sync,差距更可能来自时钟/
  L2(5080 64MB vs 5070Ti 48MB)/带宽(960 vs 896GB/s)与融合度,而非指令代际。
- "超越 fork"确定杠杆改列:WSL(E4,+7~12%)、E1(FP8 精度重模拟→过门才谈
  kernel)、持久化 megakernel(E3,研究型)、A-lite(搜索语境凑批)。

### 工具链事实(为 E1/未来窄精度工程留档)

- CUTLASS 4.7.0 已克隆至 **D:/code/cutlass4**(与 3.9.2 并存;build.rs 现有
  KATAGO_CUTLASS_ROOT 仍指 3.9.2 服务 DualGemm,4.x 引入时再分键)。
- **sm_120f device 编译 ✅**:SM120 FP8 GEMM 全模板(TMA + f8f6f4 mma.sync +
  Sm120 epilogue,tile 128x64x64 s6 cooperative)单 TU `nvcc -cubin` 25.8s 通过
  (scripts/c2_tcgen05_probe/sm120_fp8_gemm_bench.cu,含 cublas FP16 对照计时与
  数值自检,尚待可发射环境跑)。
- **Windows/MSVC host 发射 ❌**:cudafe stub 中 alignas(128) 内核参数按值传递
  触发 MSVC C2719(x64 ABI 栈对齐 16B 上限)——SM90+ TMA kernel 在 MSVC 的通病。
  可行路线:① WSL 构建(环境已备,E4 顺带)② clang-cl 作 -ccbin(本机未装)
  ③ 自写 driver-API launcher(cubin 可编已证;Params 主机构造 + cuLaunchKernel,
  工程量大)。sm-targets.json 未加 sm_120f(C2 证伪后无紧迫性;FP8 工程启动时
  再加,注意 build.rs 的 gencode 生成只认数字需同步扩展)。

## M3:B4 门槛证伪(2026-08-17,微基准)

### B4(beta=1 原位残差 GEMM,CUTLASS 候选对照)——证伪关闭

- **认识修正**:现行 `hgemm_residual` 本就是 cuBLASLt beta=1 **原位**(C==D,
  f32 残差),fork 的 B4 战术(GEMM epilogue 原位残差)在我们栈上没有对应
  收益空间——B2 的 +29.8% 来自融合省内存往返,DualGemm 主循环仅快 ~0.005ms/层。
- **门槛微基准**(scripts/b4_residual_probe/residual_bench.cu,CUTLASS 3.9.2
  Sm80 classic,A/B half、C=D f32 原位、fork 瓦片 128x128x32 s3 + 对照变体;
  cuBLASLt 侧逐位复刻生产路径含 top-8 计时 = RANK=time 语义;数值自检
  PASS 且与 Lt 逐位一致):
  | 形状 | Lt top-8 best | CUTLASS 最优 | 裁决 |
  |---|---|---|---|
  | outproj M=5776 N=384 K=384 | 0.0250ms | 0.0251(128x64) | 持平 |
  | linear-up M=5776 N=768 K=384 | 0.0442 | 0.0438(128x128) | +0.9% 噪声级 |
  | ffn-down M=5776 N=384 K=2304 | 0.1158 | 0.1188(128x64) | 慢 2.6% |
  | outproj B1/B2(M=361/722) | 0.0143/0.0084 | 0.0170/0.0145 | 慢 19%/72% |
- **结论**:门槛(主循环 ≥ cuBLASLt 同形状)不通过,B4 不集成。cuBLASLt
  在本机 FP16 残差形状上没有可被经典 CUTLASS Sm80 瓦片超越的余量。

## M3 追加:DUALFFN 提交断裂修复 + RANK=time 复审下架(2026-08-17 晚)

### 事故:M2 提交了编译不过的 dual_ffn_cutlass.cu

- **现象**:A-lite 开工复测搜索口径,发现 plan 相对 no-plan 回退 -7.6%
  (669.6 vs 725.1);进一步 eval B16 复测 +29.8% 消失(785.7 vs 780.9);
  而 direct 模式 no-plan 与 M0 记录逐位吻合(19.33 vs 19.262ms)——GPU 没变,
  是 DUALFFN 路径失效。
- **根因**:35fc20c 提交的 `dual_ffn_cutlass.cu` 编译不过(两处,均为 20:33
  的提交前清理编辑引入):
  1. `DualGemm` 模板第 5 实参 `LayoutB1` 误写为 `Layout`(RowMajor),
     B1(up 权重,ColumnMajor)张量引用不匹配;
  2. `ScaleType::Nothing` 的 `LinearCombination::Params` 仅单参构造(只有
     alpha),`{1.0f, 0.0f}` 双参不匹配。
  build.rs 对 nvcc 失败**静默跳过** → `katago_dualffn` cfg 缺失 →
  `dual_ffn` 字段 None → DUALFFN=1 tactic 空转(无声回退 hgemm_f16+
  swiglu_dual)。M2 的 +29.8% 与对拍 PASS 全部来自 20:25 的提交前好二进制;
  20:26 后所有构建(含本会话重建)都是坏的。
- **修复**:两行(模板实参 `LayoutB` + 单参 `{1.0f}`),重建后
  `cargo:rustc-cfg=katago_dualffn` 恢复。
- **加固**:build.rs 改 fail-loud——CUTLASS 在位但编译失败即 panic
  (逃生门 `KATAGO_ALLOW_BROKEN_CUDA_HOST=1`;CUTLASS 缺失仍静默跳过,
  可移植性不变)。
- **教训**:①"静默降级"战术链路的致命弱点是降级本身无声——plan 已采纳的
  tactic 必须有构建期可验证的存在性检查;②提交前必须用**重建后的二进制**
  复验 ABBA 终值(本次 +29.8% 的验证二进制与提交源码不同体)。

### 修复后重验证(2026-08-17 晚;环境整体比 M2 时段慢 ~10%——no-plan
### eval B16 783.6 vs M2 期 848~871,WDDM 桌面活动所致,配对 A/B 有效)

- **对拍**:DUALFFN=1(含/不含 RANK=time)dump_nn_io_cuda 串行 +
  compare_nn_output **RESULT: PASS**(16/16 top-1,全 gate OK)——修复后
  的提交代码首次真正过门。
- **eval B16(生产栈,W32,i200)**:no-plan 783.6 / RANK-only 789.1 /
  DUALFFN-only 989.3 / RANK+DUALFFN 984.8。
- **搜索 t=32 cap16(-n 30,visits 800)**:no-plan 718.3 / RANK-only
  687.2(**-4.3%**)/ DUALFFN-only 917.9(**+27.8%**)/ RANK+DUALFFN 878.1
  (+22.2%)。

### RANK=time 复审下架(裁决:从 plan 移除)

- M1 采纳依据(+2.55% eval B16)是 **B2 之前**的口径;B2 接管 ffn_up 后
  RANK 的边际收益归零(eval 口径 dual-only 989.3 ≥ full 984.8,+0.7% 噪声级),
  搜索口径净负(单独 -4.3%,组合 878 vs 918 = -4.4%)。
- 搜索口径回退机制未深究(疑:计时优选的 algo 对变尺寸批 replay 的
  L2/interleave 鲁棒性差——计时语境是同形状 back-to-back)。
- **行动**:`plans/best-tactic-plan.json` 修订为 r1(仅
  `KATAGO_CUDA_DUALFFN=1`,plan_id 加 `-r1` 溯源;此为手工修订,ABBA
  证据=本节数据,非 autotune 产物)。
- **r1 验证**:eval B16 **991.5(+26.5% vs no-plan)**;搜索 t=32
  **919.7(+28.0%)**——双口径同时最优。

### 对 A-lite 的影响

- 搜索口径 919.7 vs 同环境 nnbench eval 991.5 → **剩余差距 ~7.3%**
  (历史口径 12%)。DUALFFN 修复后搜索侧的 kernel 已跟上,剩余为凑批/
  CPU 编码竞争的搜索侧空间。
- 测量方法论警示(今日踩坑记录):eval/搜索口径 run 间方差可达 ~5-11%
  (WDDM 桌面活动),单次绝对值不可信,一律配对 A/B + 多轮;kernel 模式
  (异步深队列)在 WDDM 下受提交路径污染,同尺寸 direct < kernel 时即为
  伪影信号。另:**direct 模式不安装 plan**(direct_sweep 直连
  CudaModel,不经 create_compute_context)——direct 口径只反映默认
  tactic,测 plan 须走 eval/search。

## M3 追加 2:WSL 环境验证(fork 基线一致口径,2026-08-17 晚)

**目的**:消除 Windows WDDM 桌面噪声 + 对齐 fork 的 Linux 测量环境。

### Linux 首建修 bug(cuda feature 首次在 WSL 构建,build.rs 两处)

1. `rustc-link-lib=static=cudart`(Linux)找不到 libcudart.a——静态库名
   两平台同为 `cudart_static`(Linux 是 libcudart_static.a),已统一。
2. dual_ffn 宿主对象是 C++,Linux 链接缺 `stdc++`(__cxa_guard 等)——
   !windows 分支补 `dylib=stdc++`。
3. WSL 侧:nvcc 13.2(与 Windows 同版本)装于 /usr/local/cuda;
   CUTLASS 头拷至 WSL 原生盘(/root/cutlass,9p 直读太慢);
   CARGO_TARGET_DIR=/root/rk-target(与 Windows target 隔离)。
   一键复现:`scripts/wsl_bench.sh`(构建 + ABAB 基准);
   `scripts/wsl_setup.sh`(环境安装,已修 nvcc PATH)。
   **cfg 验证:`rustc-cfg=katago_dualffn` 在 WSL 构建同样发出(fail-loud
   生效,Linux 侧双保险)。**

### 结果(WSL Ubuntu-24.04,RTX 5070 Ti 透传,plan r1 vs no-plan,各两轮)

| 口径 | no-plan | plan r1 | Δ |
|---|---|---|---|
| nnbench eval B16 W32 | 810.8 / 814.6 | **1012.4 / 1018.9** | **+24.8%** |
| 搜索 t=32 cap16(-n 30) | 729.6 / 726.9 | **962.5 / 962.1** | **+32.2%** |

- 两轮配对极稳(搜索 arm 内 ±0.05%)——Linux 无 WDDM 噪声,单轮即可信。
- **WSL vs Windows(同日 plan r1)**:搜索 962.3 vs 919.7(**+4.6%**),
  eval 1015.7 vs 991.5(+2.4%)——E4 的 WSL 增益实测坐实(此前记录
  t=8 +12%)。direct 模式(同步逐轮)WSL 反而慢 ~7%(20.69 vs 19.33ms,
  dxg 同步开销;流水线重叠后无影响)。

### fork 对齐线更新(DUALFFN 真正生效后)

- fork 认证:RTX 5080(84 SM)2836 nnEval/s = 双流每流 1418
  (模型 b11c768h12nbt3tflrs-fson-silu,b11/768-trunk 族,与 b11fix 维度
  同族,行/秒可比)。
- 本机 WSL eval B16 单流 1015.7:raw 71.6% / **per-SM 86%**
  (1015.7/70=14.51 vs 1418/84=16.88 行/s/SM)——M0 期估的 20-26%
  per-SM 差实际含 DUALFFN 断裂损失,修复后真实差距 **~14%**。
  对齐线(2360 = 2836×70/84)当前达成 43%(eval 口径)。

## M4 前哨:方案 A 双流复活(2026-08-18,WSL+Windows 双验证)

**触发**:parity plan §4 重审条款——双流证伪是在 DUALFFN 断裂的慢 kernel
上做的(kernel 变快后结论需重审);且 M0 的"capture 互斥"是 WDDM 语境,
fork 双流本就跑在 Linux。WSL 环境方差 ±0.05%,单轮可信。

### 矩阵(eval B16,plan r1;WSL 为主,Windows 抽样确认)

| 配置 | WSL nnEval/s | Windows | 备注 |
|---|---|---|---|
| serve=1 graph W32(现行默认) | 1012-1019 | 1011.1 | 基线 |
| **serve=2 NOGRAPH W64** | **1109-1117(±0.5%)** | **1104.9** | **新最优,+9.5%** |
| serve=2 graph(GLOBAL) W64 | 1014 | — | capture 互斥回退 nograph+churn,收益归零 |
| serve=2 graph(RELAXED) W64 | 1020-1034 | — | capture 成功但比 NOGRAPH 慢 ~8% |
| serve=1 NOGRAPH W64 | 1020(avgBatch 21.3) | — | 供数归因:W64 本身不帮单流 |
| serve=2 NOGRAPH W32 | 746(avgBatch 8.0) | — | 饿死:双流需供数 ≥4×batch |
| serve=2 NOGRAPH PADBATCH W64 | 259 | — | padding×双流 = 灾难,永久证伪 |
| serve=3 NOGRAPH W96 | 1056 | — | 无增益 |
| serve=2 NOGRAPH W96/W128 | 1047 / 72 | — | W96 批过冲 18.4;W128 worker 超订阅崩溃(伪影) |

**搜索语境(结论:保持 serve=1)**:t=32 serve=2 = 858 vs 962(-11%,批碎片
avgBatch 9.2);**t=48 serve=2 = 123 vs 973(灾难性**——48 搜索线程 + 2
serve 的 CPU 超订阅/锁风暴)。搜索 t=48 serve=1 = 973(口径新高)。

### 条件与机制

- **双流收益 = 填 host 间隙,不是 GPU 并行**:单流 eval 批周期 31.4ms vs
  kernel ~15.5ms(host 间隙 ~50%);双流后 70.1 批/s vs 串行上限 66.2 →
  GPU 重叠仅 ~6%(kernel SM 饱和)。fork 流间扩展 2.0×(其单流 host-bound,
  GPU 饱和吞吐 = 2836),我们 1.095×(kernel 已饱和)。
- **NOGRAPH 必须**:跨线程 capture 互斥 Linux 同样存在(GLOBAL 模式他流
  并发活动即失效);fork 本就不用 graph(其否决项)。
- 代码改动:`begin_capture` GLOBAL → **RELAXED**(每 handle 单流、apply 内
  无跨流依赖,捕获内容不变;serve=2 的 capture 不再互斥打挂)。serve=1
  数字无变化(1018.8/966.4/972.3 复核),对拍 PASS(16/16)。RELAXED 下
  serve=2 可带 graph 但比 NOGRAPH 慢——双流推荐配置仍是 NOGRAPH。

### fork 对齐含义(诚实口径)

- 单流对单流:1015.7 vs 1418 → per-SM 86%(差 14%,带宽 7%+L2/时钟)。
- 双流对双流:1112 vs 2836 → per-SM **47%**。fork 双流吃满 GPU(2836/84
  = 33.8 行/s/SM),我们 kernel 饱和上限 ~15.9 行/s/SM——**差距主体在
  kernel 吞吐**,而 kernel 侧选项已全数证伪(tcgen05 硬件无/FP8 精度不过/
  CUTLASS 不赢 cuBLASLt/FA4 已局部最优)。SM120 mma.sync 屋顶下,该差距
  无已知确定路径;对齐线 2360 在本机不可达,除非 fork 数字含口径差
  (其模型 b11c768h12nbt3tflrs 与 b11fix 同族但头结构细节未逐项核对)。

### 使用指南(高并发服务场景)

分析引擎多查询/批量评估等饱和供数场景(W ≥ 4×batch):
```
--override-config numNNServerThreadsPerModel=2
环境 KATAGO_CUDA_NOGRAPH=1
```
GTP 对局/单查询(搜索语境):保持默认 serve=1(双流在搜索是负收益甚至
灾难)。复现脚本:scripts/wsl_dualtest*.sh、wsl_relaxedtest.sh。

## M3:E1 FP8 官方 PTQ scope 重模拟(2026-08-17,精度门裁决:不过)

### 官方 ff27077 精确语义(网络核实)

commit ff27077(PR #219,python/katago/quantization.py)默认 `--scope transformer`:
- **量化对象**(按名 allowlist):`q_proj/k_proj/v_proj/out_proj/ffn_linear1/
  ffn_linear_gate/ffn_linear2`——**ffn_linear2(FFN 下投影)在默认 scope 内**
  (旧文档"官方只量化 attention/SwiGLu、不含 linear2"的描述有误,已勘)。
  保持 FP32:QK/AV 激活 matmul、Softmax、RMSNorm、输入茎、外层 bottleneck
  投影、trunk tip、全部头。b11fix 上该 scope = 全部 231 个 trunk MatMul
  ((384,384)×132 q/k/v/out +(384,1152)×66 gate/up +(1152,384)×33 down;
  bottleneck 768↔384 在图中是其他算子,天然排除)。
- **方案**:权重 E4M3 **per-output-channel**(zero-point=0);激活同样量化,
  **静态 per-tensor** scale(max 校准);权重-only 显式拒绝。
- 官方验收口径是 TensorRT 强类型 + 对局强度,不是我们的逐位对拍门。

### 旧模拟的两处失真(本次修正)

1. **量化轴错误**:旧 `quantize_weights_perchannel` 对 [K,N] initializer 按
   行(axis=-1)量化 = per-input-channel;官方是 per-output-channel(列)。
2. **激活粒度错误**:旧 fp8full 用 per-token 动态 amax;官方是静态
   per-tensor(max 校准)。
   sim_quant.py 已重写支持:per-out-channel 权重、两遍法静态 per-tensor
   激活(校准 pass 插 amax 探针,scale 构建期烧入)、FFN-only scope。

### 结果(16 位置,vs FP32 ORT 基线;历史 fp8w=14/16 KL 8.6e-3 "勉强")

| 变体 | top1 | KL | value_max | score_max | 裁决 |
|---|---|---|---|---|---|
| fp8_official(官方精确语义,全 scope) | 14/16 | **4.3e-2** | **2.54** | 0.250 | ❌ 硬失败 |
| fp8w_off(仅轴修正,全 scope) | 15/16 | 4.5e-2 | 2.30 | 0.255 | ❌ |
| **fp8_ffn_official**(FFN-only 官方语义) | 14/16 | **2.4e-3** | 0.321 | 0.0225 | ❌ 仍未过门 |
| fp8w_ffn_perK(FFN-only 权重-only,不可部署*) | 15/16 | 8.1e-4 | 0.449 | 0.0148 | 参考 |

*f8f6f4 mma 双操作数都须 fp8,权重-only 无 kernel 通路,纯模拟参照。

### 隔离实验的关键发现(scripts/e1_axis_isolate.py)

| 量化类(单独量化该类) | per-K | per-N |
|---|---|---|
| (384,384) 注意力投影 ×132 | KL 6.6e-3 | **KL 4.4e-2(6.7×)** |
| (384,1152) gate/up ×66 | 4.8e-4 | 5.1e-4 |
| (1152,384) down ×33 | 2.6e-4 | 2.7e-4 |

- **FP8 的全部损害集中在注意力投影**;FFN 三类几乎免费(2~5e-4)。
- **per-output-channel 在注意力类上反而差 6.7×**(Frobenius 误差两轴相同
  0.0275;无归零、无次正规差异、行幅度均匀——机制不明,开放观察)。FFN
  类两轴无差。
- 因此造出 FFN-only 中间方案:KL 2.4e-3(比历史 fp8w 好 3.6×),top-1 翻转
  仅 pos0/pos2(margin 0.005/0.007,近平局),winprob_d≤0.027、score_d≤0.0225
  ——最接近门槛的 FP8 成绩,但 top-1 100% 在近平局 margin 下对任何 FP8
  噪声都不可达(pos2 在 KL 8e-4 时即翻),score 仍超 misc 门 2.2×。

### 裁决与影响

- **E1 精度门不过,FP8 kernel 工程封存**。阻塞是精度不是速度(FFN 占前向
  61.6%,fp8 微基准 down-proj -58%/qkv -38%,潜在收益 ~15-20%)。
- 重开条件(任一):(a) 对拍门接受近平局(margin<0.01)top-1 翻转等价 +
  score 门放宽至 ~2.5e-2;(b) 更细激活粒度(per-token 动态,可融合进
  DualGemm 前的量化 kernel)且门放宽——模拟预测仍翻 pos2。
- C2 证伪(M3-pre)后 FP8 曾是"超越"路线的最大储备;如今也关闭,超越
  fork 的剩余确定杠杆只有 WSL(E4,+7~12%)与 A-lite(搜索语境)。

## M4 追加:host/A-lite 复审与布尔 tactic 语义修复(2026-08-19)

### 本轮候选裁决(全部未采纳)

统一口径:Windows WDDM、plan r1、B16/W32；候选低于 1% 门槛即回退。

| 候选 | 配对结果 | 裁决 |
|---|---|---|
| pipeline 空闲首批等待 50/100/200us | B16 nnEvals/s 1037/1032/1039/1028（0us 基线 1037）；不限制 B16 时 200us 把 avgBatch 15.4→25.9，却使 934→890 | ❌ 等待只把批推过 B16 甜点，吞吐不增 |
| 每轮只 finish 一个完成批、优先补发下一批 | ABBA 基线几何均值 1056.3，候选 1041.9（-1.4%） | ❌ |
| 输入直接编码到 pinned + 输出直接从 pinned 解码 | ABBA 1083.7→1076.1（-0.7%） | ❌ Windows pinned 写入更慢 |
| 仅输出 pinned 零拷贝 | ABBA 1082.8→1082.9（持平） | ❌ 未过 1% |
| pageable input/output staging Vec 按槽复用 | ABBA 1093.1→1091.3（-0.2%） | ❌ allocator 不是剩余瓶颈 |

结论:A-lite 的简单等待/finish 公平性和 host staging 均已收口；搜索与固定
B16 的剩余差距不是可由这些微调兑现的稳定收益。所有实验代码均回退。

### 落地修复:显式 `0` 不再误开启 tactic

发现所有以 `.is_ok()` 读取的布尔 tactic 把环境/plan 中的显式 `"0"` 也视为
开启，破坏了 plan 的负覆盖语义。受影响项:
`NOGRAPH/NOPIPELINE/PADBATCH/SPLITK/T32/T64N32`。新增统一
`tactic_plan::tactic_enabled()`（仅值 `1` 为 true）并替换全部读取点；
`DUALFFN/CUBLASLT` 原本按值判断，不受影响。

- 实机故障复现:同时显式设置上述 tactic=`0`，旧二进制仍误启慢路径，
  eval B16 **409.4 nnEval/s**；修复后 **1096.8 nnEval/s（2.68x）**。
- 默认 r1 plan 只写 `DUALFFN=1`，因此默认生产性能不变；收益体现在完整认证
  plan、autotune 显式负覆盖和手工诊断配置终于按契约工作。
- 单测:plan 安装 `PADBATCH=0` 后 `tactic_enabled` 为 false；tactic plan
  6/6 PASS。
- 数值门:16 位置 ORT FP32 整图对拍 **RESULT: PASS**，policy top-1 16/16。
- 追加修复 CUDA graph flag UB：cudarc 0.19.9 将 instantiate bitflags 生成为
  不含 0 的 Rust enum，原生产/测试路径 `transmute(0u32)` 属未定义行为，
  debug 冒烟会直接 abort。统一改用合法的 `USE_NODE_PRIORITY`（本 graph
  未设置节点优先级，语义等价默认）。完整 CUDA 冒烟 **9/9 PASS**、3 ignored，
  graph launch+sync 10-27us（WDDM 微基准波动）；生产 B16 ABBA 旧/新 1079.1/1078.5（-0.06%，
  持平），作为安全性修复保留。

## M4 追加 2:对照官方 `cudarocmopt` 的 q64 attention 与认证基础设施(2026-08-19)

对照快照:`D:/ExperimentalCode/KataGo` 分支 `cudarocmopt`,HEAD `08939455`。
现有后端已经具备其 per-handle stream、共享权重、1x1/packed-QKV GEMM、RoPE load
融合、tensor-core attention、残差 `beta=1` 和 DualFFN 主体;本轮只移植其有独立
增量的 `64Q x 64K,4 warp` attention 组织,并补齐路径/指纹/并发测量工具。

- 数值:Windows/WSL 各 16 局面 ORT 整图对拍均 `RESULT: PASS`,policy top-1
  `16/16`;保持 FP32 QK/PV 累加、FP32 online softmax 和精确 `expf`。
- Windows:eval B16 `+2.95%`;search visits t8/t16 `+3.34/+2.77%`。
- WSL:eval B1/B4/B8/B16 `+2.93/+4.35/+3.69/+3.44%`;长样本 search
  visits t8/t16 `+3.0/+2.4%`,nnEvals/s `+3.3/+3.1%`;attention core
  q128/q64 累计 `23.783/20.278 ms`(`+14.7%`)。
- 路径:两平台 14-case tactic/fail-closed matrix 均 PASS,包含 q64 实际 launch、
  plan>env、graph/direct、非法 key/value、模型/build mismatch 和强制 DualFFN
  probe failure。
- 生产:Windows `plans/best-tactic-plan.json` 升级 schema 2,启用
  `DUALFFN=1 + ATTN_TILE=q64`;另存 q128 schema 2 回退 plan。WSL 因宿主目标/CUTLASS
  工作树产生不同 device-code build id,使用独立认证 plan,禁止跨平台混用。
- 同轮证伪:DualGemm swizzle `<1>` 仅 `+0.49%`;deferred residual + 下一层
  RMSNorm 在 B8/B16 `-1.09/-1.33%`;两者实现与 tactic key 均删除。

完整计划、原始数据索引和裁决见 `docs/cudarocmopt-validation-plan.md`。

### M4 追加 3：定向 CUDA graph warmup（2026-08-19）

参考 `cudarocmopt` 的 warmup 原则，但针对本项目按物理 batch 惰性创建 CUDA graph
的实现，只预热 `[1,maxBatch]` 两个生产尺寸。`Backend::warmup_batches` 让 CUDA
后端声明尺寸，`NnEvaluator::load_model` 在 server handle 交接前执行真实前向；
warmup 失败在 server-ready 前报错，`cudaDisableWarmup=true` 仅用于诊断。

Windows 冷启动测量（同一 release、q64 + DualFFN、`b11fix.onnx`）：B1 首调用
从 `173.1/177.6 ms` 降至 `5.6/5.8 ms`（约 `96.7%`），B16 从 `260.9/272.9 ms`
降至 `89.7/98.6 ms`（约 `65.0%`）；启动加首请求总耗时不增加，steady-state
沿用既有 ABBA 结果无 `>=1%` 回退。因此定向 warmup 进入默认生产路径。
WSL release/GTP 也确认 warmup `1,16` 可用；B1 `202.4/304.5 ms` 降至
`5.35/5.62 ms`，B16 `345.6/402.0 ms` 降至 `81.8/83.1 ms`。
Windows `nvidia-smi` 采样显示 warmup 常驻显存增量约 `31 MiB`，低于 `512 MiB`
门槛。
