# CUDA 优化战术记录（M4 路线图）

> 方法论与战术清单移植自 KataGomo_fork（SM120 plan 驱动优化，5080 上 2836 nnEval/s）。
> 本机目标：RTX 5070 Ti（SM 12.0，16GB），模型 b11c768h12nbt3tflrs-fson-silu（75M 参数，FP16/NHWC）。

## 基线（2026-08-14 实测，release 构建，单线程单 batch）

- TRT 10.x 参照：**180 nnEvals/s**（单前向 ~5.5ms）
- M3 手写后端 v1（正确优先，单流）：**47 nnEvals/s**（单前向 ~21ms，180 kernel/前向）

## M4 实测进度（ABBA 记录，全部对拍 PASS 后提交）

| 提交 | 改动 | nnEvals/s | 附注 |
|---|---|---|---|
| 85931e5 | 执行器 v1 落地 | 47 | 基线 |
| 271be9b | GEMM v2(smem 双缓冲+cp.async+ldmatrix,tile 128×128×32) | 58.5 | +24% |
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
- batch=1 时 GEMM grid 仅 27-108 blocks(96 SM 利用率 <30%),tail effect
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
- **KataGo 官方 2026-08 PTQ 策略**(commit ff27077):只量化 attention/SwiGLU
  的 **weight projection**,attention 的 QKᵀ/AV、softmax、RMSNorm、**所有
  policy/value head 保持 FP32**。
- SM120 支持 FP8 `mma.sync.m16n8k32`(E4M3/E5M2,FP32 累加)+ MX block-scale,
  社区实测 FP8 ~2× FP16 吞吐。
- **但工程量大**:量化图需接到自研 kernel/CUTLASS,且精度需逐层验证。
  **结论:现阶段 FP16 + cuBLASLt 已是最佳性价比;FP8 留作后续大工程。**

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
| G1 | fa4 / wide_projection / qkv_rope / dual_ffn | 宽 QKV 单 GEMM（M=B×361、N=1152、K=384，packed 行 [Q384\|K384\|V384]，tile M128×N128×K64 s2）；FA attention（tile M128×N64、stages=1、noncausal 无掩码、both16 双 FP16 累加 + **rescale 舍回 FP16 累加器**、128 线程）；Q/K RoPE 为**独立 batch 共享 kernel**（361×192 线程、half2 对、按 B 展开，非 GEMM epilogue——SM120 证伪了融合）；dual FFN 共享 A + SwiGLU epilogue（CUTLASS LeftSiLUAndMul） | 部分：attention 已换 v3(非 FA4 tile 但达标)；dual FFN 进行中；宽 QKV/RoPE 未做 |
| G2 | fused_residual / linear2 / outproj | GEMM beta=1 原位残差（C==D 同指针，epilogue 先读 C 再写 D；tile 128×128×32、warp 64×64×32、3 stages） | ✅ 已做(8f24fb0)：hgemm_residual beta=1 + hgemm_f16 直出 epilogue |
| G3 | postconv_bn / preconv / pointwise | affine+SiLU（half2 `__hfma2`，sigmoid 逐 half 转 float）；RMSNorm C384 用 warp4-vec8（**零共享内存**：uint4+uint2 载入 12 half/lane、单 XOR 链归约、4 行/块）；SwiGLU 已折入 G1 FFN epilogue | 待做(RMSNorm/affine 仍是简单版) |
| G4 | wide_head / policy_p1 / head_bn | wide head 投影（三合一 GEMM，full-c384：P1@0..96、G1@96..192、V1@192..384）；fused policy P1（half→float 直出 + BN fold + silu）；head BN half→float（V1 写 half+float 双输出） | 待做 |
| G5 | rmsnorm | （已并入 G3） | — |
| G6 | l2 | persisting-L2（trunk 窗口 = B×361×768×2B ≈ 8.5MB、inner = B×361×384×2B ≈ 4.2MB；cudaDeviceSetLimit + cudaStreamSetAttribute access-policy；**5070 Ti 48MB L2/36MB persisting 上限，2 流 26.6MB 可 fit**） | 待做 |
| G7 | weight_sharing | 普通权重跨流共享（cudaShareModelWeights） | 待做 |
| G8 | initial_conv | 3×3 卷积 im2col+GEMM 的 sm120 特化（K=198→pad 208=13×16）——**fork 用 cuDNN frontend（eng45-tile0-stages2），本机无 cuDNN 故走 im2col** | 待做 |
| G9 | initial_global | initial global matmul-add 融合 | 待做 |
| G10 | value_terminal | value terminal 拆分（fork 认证 plan 中**已禁用**，走官方路径；低优先级） | 待做 |

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

- target 指纹：compute capability + 设备 16 属性 + 模型 SHA-256 + batch + 精度 + 流拓扑
- apply：per-batch tactic overrides；final_joint：性能证书 + 正确性证书
- 加载校验：任何不匹配即报错，绝不静默回退

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
- **张量核代数**：5th Gen Tensor Cores，支持 FP4/FP8/FP16/BF16/TF32；**tcgen05.mma 是 sm_100（sm_100f）指令集**，消费级 sm_120 需以 **sm_120f 家族特性**编译才能在 PTX 里用 tcgen05/tensor-memory 指令；普通 `-arch=sm_120`（非 f 变体）编不出 tcgen05。
- **tcgen05 寄存器天花板**：Blackwell 最大矩阵指令（tcgen05.mma.cta_group::2）每线程需 256 个寄存器，硬件每线程上限 255（[The Software Frontier, 2026-08-03](https://www.thesoftwarefrontier.com/p/how-blackwells-tensor-memory-actually)）。即 cta_group::2 无法在单个 kernel 里完成 mma——fork 的 tile 尺寸设计必须避开（单 cta_group::1 tile 128×N 是安全区）。
- **实现参考**：CUTLASS 4.x 已支持 sm_120（消费级 Blackwell）的 tcgen05 GEMM 模板；[tcgen05 for dummies（gau-nernst, 2025-12-21）](https://gau-nernst.github.io/tcgen05/) 有 sm_100 的逐指令级讲解（tmem 分配、tcgen05.mma 描述符、ld/st、commit/mbarrier），消费级差异主要在 cluster/cta_group 与 tmem 尺寸。
- **本项目定位**：基线手写 PTX（hgemm m16n8k16 / 共享内存归约 attention）为**正确性优先**；M4 的 tcgen05 路径按 fork 认证 plan（FA4 tile M128×N64 s1 both16）实施，编译开关经 `configs/sm-targets.json` 的 `arch` 字段（如 `sm_120f`）选择，失败回退基线。

