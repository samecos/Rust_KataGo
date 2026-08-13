# CUDA 优化战术记录（M4 路线图）

> 方法论与战术清单移植自 KataGomo_fork（SM120 plan 驱动优化，5080 上 2836 nnEval/s）。
> 本机目标：RTX 5070 Ti（SM 12.0，16GB），模型 b11c768h12nbt3tflrs-fson-silu（75M 参数，FP16/NHWC）。

## 基线（待 M3 执行器落地后实测记录）

- 官方参照（本机无 cuDNN，用 TensorRT 10.16 兜底作基准）：nnEval/s 待测
- M3 手写后端 v1（正确优先，单流）：nnEval/s 待测

## 决策组（严格有序，后组不得改写前组配置键）

| # | 组 | 战术 | 状态 |
|---|---|---|---|
| G1 | fa4 / wide_projection / qkv_rope / dual_ffn | 宽 QKV 单 GEMM（[1152,384] 列拼）；FA attention（tile 128×64、stages=1、noncausal 无掩码、both16 双 FP16 累加 + rescale 转回累加器类型）；Q/K 融合 RoPE（half2、batch 共享、按 B 展开）；dual FFN 共享 A + SwiGLU epilogue | 待做 |
| G2 | fused_residual / linear2 / outproj | GEMM beta=1 原位残差（C==D 同指针） | 待做 |
| G3 | postconv_bn / preconv / pointwise | affine+SiLU（half2/vec8）、SwiGLU（half8）、RMSNorm（warp4-vec8：uint4+uint2 载入、单 XOR 归约） | 待做 |
| G4 | wide_head / policy_p1 / head_bn | wide head 投影（C768→384 三合一）、fused policy P1（FP32 直出）、head BN half→float | 待做 |
| G5 | rmsnorm | （已并入 G3） | — |
| G6 | l2 | persisting-L2（trunk/inner 窗口按 B×361×C×2B 计算，cudaStreamSetAttribute access-policy） | 待做 |
| G7 | weight_sharing | 普通权重跨流共享（cudaShareModelWeights） | 待做 |
| G8 | initial_conv | 3×3 卷积 im2col+GEMM 的 sm120 特化（K=198→pad 208） | 待做 |
| G9 | initial_global | initial global matmul-add 融合 | 待做 |
| G10 | value_terminal | value terminal 拆分 | 待做 |

## 调度层（fork 复刻清单）

- nnBatchAwareDispatch：固定物理 batch（B16），不足时尾批复制 padding，设备空闲才发射
- cudaAsyncInferPipeline：upload/compute/download 三流 + 事件握手单槽复用 + pinned staging
- 双流拓扑：2 个 NN server 各自独占 non-blocking stream
- 明确不做（fork ABBA 证伪）：CUDA Graph、BF16/FP8/FP4、winograd、DSMEM/cluster、mask 处理

## plan JSON（fail-closed）

- target 指纹：compute capability + 设备 16 属性 + 模型 SHA-256 + batch + 精度 + 流拓扑
- apply：per-batch tactic overrides；final_joint：性能证书 + 正确性证书
- 加载校验：任何不匹配即报错，绝不静默回退

## 正确性门

- 8192 局面对拍 FP32 参考（ORT CPU），逐行 max-abs/RMSE + policy top-1
- 性能测量纪律：物理 nnEval/s 口径、ABBA/BAAB、nvidia-smi pmon 排除外来 SM 占用、min_improvement 0.1%

## PTX 手写点（相对 fork 的增量空间）

- GEMM 主循环 mma.sync.aligned.m16n8k16（已实现并数值验证，见 cuda-kernels/gemm.cu）
- attention 主循环（M4 换 FA both16 tile 版，v1 已数值验证）
- 后续探索：sm_120a 的 FP8 块缩放 mma（fork 未做，风险项）

## 已知缺口（待修，按优先级）

1. **GTP 参数系统**：`kata_search/src/params.rs` 的 changeable 参数表仅 9 条，C++ `searchparams.cpp` 有 115 条；`kata-set-params`（JSON 批量）未实现。影响 kata-get/set-param、kata-list-params、kata-set-params 的完整性（需求 F2）。修法：对照 C++ changeableParametersToJson 补齐 + gtp.rs 的批量设置。
2. **analysis allowMoves**：每请求限 1 条（C++ 允许每玩家 1 条共 2 条）。
3. **M4 待做**：见上方决策组表。
