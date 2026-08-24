# 官方 KataGo v1.18.1 推理侧审计(2026-08-24)

> 对照仓库:`D:/code/KataGo`(lightvector/KataGo,v1.18.1,92ee95c0)。
> 目的:确认官方推理侧有哪些可移植技术、我们已对齐哪些、还缺什么。
> 本文档是跨对话对齐入口之一;性能结论仍以 `docs/cuda-fork-parity-plan.md`
> 与 `docs/cuda-optimization-plan.md` 为准。

## 1. 审计范围与关键提交

v1.18.1 的推理侧优化集中在 2026-06 ~ 2026-08 的几笔提交:

| 提交 | 日期 | 内容 | 对我们的意义 |
|---|---|---|---|
| `ecfbeb46` Major cuda and rocm optimizations | 2026-08-16 | 官方推理优化大提交:CUTLASS DualGemm+LeftSiLUAndMul 融合 FFN(cudafusedffn.cu)、自研 tensor-core flash attention(cudaflashmma.cuh,Q64/KV64/4 warps/mma.sync/FP32 累加)、`benchmarknn` 子命令、runcudaopttests.sh 组合测试 | 三件套我们全部已有等价物且更细(DualFFN 认证 plan、FA2 q64 晋级、nnbench 三模式) |
| `439a760a` Greatly reduce cuda transformer mem usage | 2026-08-11 | CUDA 侧 transient buffer 从精确尺寸改为 max-batch 尺寸,避免 SimpleAllocator 按尺寸池化导致 O(maxBatch) 倍显存 | 我们的 workspace 本就按 max batch 预分配,等价 |
| `41491ce3` Squash #1237 ROCm 1x1 convs as GEMM | 2026-08-18 | ROCm 侧把 1x1 conv 改走 GEMM(社区贡献) | 我们所有 conv 全 GEMM 化,更彻底 |
| `2e2d772d` Remove per-server-thread max batch size | 2026-08-05 | 移除 `nnMaxBatchSizeThread<N>`(warmup 一致性/`setNumThreads` 语义问题,review 中被砍) | 见 §3 跟踪 |
| `5ff1f3d2`/`ecb1b2a5`/`1abc4a18` benchmark multi-thread testing | 2026-08-21~23 | benchmark 开始测双线程 serving、半批推荐逻辑 | 官方正在探索的方向,见 §3 |

## 2. 技术 vs 我们的对照(结论:kernel 层无增量)

| 官方技术 | 证据 | 我们的现状 |
|---|---|---|
| flash attention:FMMA_BLOCK_Q=64/BLOCK_KV=64、4 warps、mma.sync m16n8k16、FP32 累加、online softmax、GQA、head dim 32/64 | `cudaflashmma.cuh:36-37,465` | `attention_fa2_q64.cu` 同款 tile q64(plan r1 启用,+3% eval);RoPE 融合进加载,packed QKV 单 GEMM,更彻底 |
| attention 三级回退:mma flash → cudnn graph SDPA(per-shape plan 缓存+warmup 容错降级) → 标量 online-softmax | `cudaandrocmbackend.inc:2956-3062` | plan 静态选择 FA2/v3;我们模型族固定,无运行时回退需求;不为回退引 cuDNN 依赖 |
| CUTLASS DualGemm + LeftSiLUAndMul(epilogue 融合 SwiGLU) | `cudafusedffn.cu/.h`、`ecfbeb46` | `cuda-host/dual_ffn_cutlass.cu` 同款(CUTLASS 3.9.2,128x64x32/warp 64x32x32/s3),+26.5% eval 已认证 |
| QKV 合并投影(useCombinedQKV,strided-batched GEMM) | `cudaandrocmbackend.inc:2992-2999` | 权重侧拼 1152x384,单 GEMM 直出 packed,FA2 直吃 |
| 1x1 conv → cuBLAS GEMM(cudaUse1x1Matmul) | `cudaandrocmbackend.inc:969-998` | 全 conv GEMM 化(1x1 直接,3x3 im2col K=208) |
| cuDNN implicit precomp GEMM conv(NCHW/FP16 tensor core) | `cudaandrocmbackend.inc:1020-1050` | 不用 cuDNN;手写 im2col+GEMM 替代,H5 证明 InitialConv 仅 1.6% |
| 常规 conv cuDNN(IMPLICIT_PRECOMP_GEMM + FP16 tensor core) | 同上 | 同上 |
| L2 persistence | 曾尝试,已移除(注释残留 `:5163`) | 双方一致:不做 |
| FP16 存储 + FP32 累加 + 精确 SiLU | 全后端 | 同款纪律,整图对拍门在位 |
| pinned memory / FP16 输出拷贝 | `cudaandrocmbackend.inc:5391+` | E2 已 ABBA 证伪无收益(我们 host 开销更小) |
| benchmarknn 固定物理批测量 | `cpp/command/benchmarknn.cpp`(`ecfbeb46` 新增) | `nnbench.rs` 三模式(eval/direct/kernel)对齐它做的 |

**结论:kernel/调度层面无新增移植项。** 官方与我们的 FP16 GEMM 都在
mma.sync 屋顶下(SM120 无 tcgen05,已三源证伪);官方的融合三件套
(DualGemm/flash-mma/QKV 合并)我们有等价或更彻底的实现。

## 3. 官方 multi-server-thread 跟踪线(v1.19 展望)

时间线与语义:

1. `815378d8`(ONNX PR 期间)引入 `nnMaxBatchSizeThread<N>`,让每个
   server 线程可不同批上限——**被 review 否决**,`2e2d772d`(2026-08-05)
   移除,理由:改动侵入共享 NNEvaluator 核心、warmup 一致性与
   `setNumThreads` 重置语义未解决。
2. `5ff1f3d2`(2026-08-21)"Make benchmark test half batch size and on
   cuda and rocm test two threads" + `ecb1b2a5` "Fix up benchmark logic"
   + `1abc4a18`(08-23)"Multi server thread testing"(只动测试脚本
   `rungpuerrortest.sh`/`testbackendreference.cpp`)——官方在为多
   server 线程铺测试覆盖,尚未给出单 GPU 多线程收益结论。

**对照我们**:serve=1 单消费者为默认;饱和供数语境 serve=2+NOGRAPH+W≥4×batch
已复活(+9.5%,cuda-optimization-plan「M4 前哨」);搜索语境维持证伪
(t=48 灾难 123)。官方遇到的 warmup 一致性坑与我们发现的批碎片/CPU
超订阅坑互相印证。**跟踪动作**:v1.19 发布时复查其 benchmark 推荐逻辑
是否给出多线程口径,与我们 serve=2 复活条件对照。

## 4. 唯一被采纳的增量:opt 路径组合测试体系

官方 `runcudaopttests.sh` 的方法论(2026-08-24 已移植落地):

- 每个 knob 组合跑数值阈值测试(testgpuerror 对 Eigen 参考);
- **断言后端一次性路径选择日志确认期望路径真的启用**(防 knob 静默失效);
- 负例:非法组合必须启动期报错;
- FP16/FP32 双评测器 + unbatched/random-batched 覆盖。

我们的等价物(2026-08-24):

| 官方组件 | 我们的实现 |
|---|---|
| loggedUsingMmaAttention 等 logged* 标志 | `[cuda-tactic]` 一次性标记,本轮扩展覆盖全部 tactic:dual_ffn/attention(已有)+ gemm kind×engine/tile、splitk、fusion、rms、padbatch、cublaslt_rank(新增) |
| testgpuerror 数值阈值 | `dump_nn_io_cuda` + `compare_nn_output.py`(ORT FP32 黄金参考,gates: policy 5e-2/value 2.5e-2/misc 1e-2/ownership 5e-3/top-1 100%) |
| runcudaopttests.sh | `scripts/validate_cuda_tactics.py`(快速模式 24 case 全绿;`--with-numeric` 每个正例组合跑 dump+对拍) |

运行方式:

```bash
# 快速(路径标记断言,~2min)
.venv/Scripts/python.exe scripts/validate_cuda_tactics.py
# 全量(含数值门,每个组合 dump 4 局面 + ORT 对拍,~30-60min)
.venv/Scripts/python.exe scripts/validate_cuda_tactics.py --with-numeric
```

已知边界:pad×graph 组合会打出 graph 重放 INVALID_VALUE WARNING——
ABBA 证伪组合(259),脚本只断言 pad 路径接管+进程不崩,不断言性能。

## 5. 后续跟踪清单

- [ ] v1.19 发布时:复查 multi-server-thread 是否给出单 GPU 多线程方案(§3)
- [ ] 官方若引入 FP8/新 attention kernel(tcgen05 后继),对照我们 E1 封存结论
- [ ] benchmark 线程推荐逻辑(`ecb1b2a5`)的半批启发式是否值得吸收(我们
      auto-tune 已有 ABBA 框架,暂无动作)
