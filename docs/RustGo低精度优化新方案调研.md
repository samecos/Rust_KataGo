# B11 / 剪枝 B15：低精度优化新方案调研

2026-09-23。目标是 RTX 5070 Ti（SM120）上同时改善单机搜索和 `nnworker`，调研结论与已测性能分开。历史基线见 [INT8 后端](RustGoINT8后端.md)、[B15 剪枝诊断](RustGoB15剪枝INT8诊断.md)、[INT8 专项优化](RustGoINT8专项优化.md)。

## 判断与本轮入口

最先落实现有整数 GEMM 的逐形状选算法，以及共享 CUDA 路径中的注意力/浮点 GEMM 优化。下一种低精度优先试 **MXFP8 与按层混合精度**，NVFP4 放在后续校准或训练恢复阶段。不是位数越低，整网就越快。

本轮已实现 **INT8 GEMM 按实际形状计时选择 cuBLASLt 候选算法**，诊断开关默认关闭；候选组合另包含仓库已有的 **q64 attention**，并单独验证省去 GEMM 计时搜索的轻量 q64 配置。已有 q64 generic 路径覆盖剪枝 B15 的 C512/D32 注意力热点。具体数值验证、单机/Worker ABBA、启动代价与最终选择见 [本轮实测与使用入口](RustGoINT8注意力与算法调优.md)。下文的 MXFP8 等后续路线仍是研究建议，不能视作已测收益。

用户随后明确允许任意混合精度，并选定新的实验验收口径：**相对同模型 FP16，验证集中胜率输出的最大绝对偏差不超过 6 个百分点**，即 `max(abs(p_candidate - p_fp16)) ≤ 0.06`；另外报告 policy 和目差误差。这是独立实验门，原 FP32 严格门与认证身份仍保留，不能把新门通过写成原门通过。校准参数必须只在校准集拟合、在留出集验收；未经校准的结果若已满足新门，应如实标明，不能称为“校准恢复”。

这两个模型的约束决定了优先级：

- 仅 19 路，每局面 361 个空间 token；当前 CUDA 要求 `head_dim=32`、`mid=heads*32`。不能直接套长上下文 LLM 的速度比例或自回归 KV 缓存优化。
- B15 的 45 层 FFN hidden 总宽度只剩完整宽度的 **18.33%**，22 层 hidden≤128，最窄 16。它已经是通道结构化剪枝，留下的矩阵仍按稠密矩阵执行。
- 前轮 B15 追踪中，注意力核心及前列浮点 GEMM 约占 kernel 时间 **67.5%**；这是带 profiler 的局部诊断。进一步压缩 FFN 的整网收益空间有限。
- 现有 INT8 仍未通过原 FP32 严格门；前轮一盘 SGF 的输出仿射校准也没有泛化。新的量化格式和按层策略都需要独立精度身份。

## 可落地路线

| 优先级 | 路线 | 实施与验收要点 |
|---|---|---|
| P0 | 现有 INT8 按形状选 GEMM | 专项测试使用 CPU INT32 oracle；运行时查询候选池，固定输入先完整对拍生产算法的 INT32 输出，再计时，两种证据分开。按实际 M/N/K 保留进程内选择，验证 Graph 重放。本轮已完成两模式验证，结果见实测报告。 |
| P0 | q64 注意力及其余浮点 GEMM | 已测试既有 q64 单独候选及其与新 INT8 调优的组合，包括 B15 C512/D32 generic 分支。注意力策略改变归约顺序，分别保留原 FP32 门结果及用户的 FP16 最大胜率偏差门；数值检查先于 ABBA。其余浮点 GEMM 仍是后续方向。 |
| P1 | 按层敏感度与速度选择精度 | 从现有 FP16/INT8 两档开始，联合考虑留出集误差与实际形状延迟，替代只有 `hidden≥384` 的宽度规则。避免把越少量化层等同于 value 误差越小。 |
| P1 | MXFP8 原型 | 独立后端或独立精度策略，先试 B11 宽 FFN、B15 较宽层，窄层保留 FP16。权重离线打包，动态激活块量化尽量与已有 RMSNorm/SwiGLU 融合；先测完整的量化—GEMM—后处理，而非仅 GEMM。 |
| P1 | SmoothQuant 类逐层校准 | 采集多局激活的通道统计，平衡权重与激活离群值，再做独立输出对拍。它主要改善量化误差，不能假定能减少运行时间。 |
| P2 | NVFP4、QAT/QAD、进一步稀疏 | 在精度数据和训练/导出流程齐备后再做；保留敏感层较高精度。不把论文上的 LLM 或 GB200 结果当作本机围棋模型收益。 |

### 新方案：AutoQuantize 的可借鉴部分

NVIDIA 于 2026-08-24 发布的 [AutoQuantize 说明](https://nvidia.github.io/Model-Optimizer/announcements/autoquantize.html)介绍了逐层敏感度评分和成本约束下的混合精度搜索。[官方 API](https://nvidia.github.io/Model-Optimizer/reference/generated/modelopt.torch.quantization.model_quant.html)以 PyTorch 模型为入口，不能直接加载本项目的原生 TF3/Rust 层图。

适合本项目的是借鉴搜索方法：先比较 FP16/INT8，再增加 MXFP8；使用真实 policy/value/ownership 误差与本机延迟约束，得到模型专属的逐层配置。这里把成本设为实测延迟是本项目建议，不是声称官方默认位数预算能够保证端到端提速。

### SM120 的 MXFP8 / NVFP4 支持边界

[cuBLAS 缩放支持表](https://docs.nvidia.com/cuda/cublas/index.html#scaling-mode-support-overview)列出了 FP8 每 32 元素共享 UE8M0 scale，以及 FP4 每 16 元素共享 UE4M3 scale 的块缩放。SM120 可使用这些模式，计算为 FP32，并可输出 half。相对于现有 INT8→INT32 中间结果，这提供了减少中间结果存储/反量化成本的实现机会；**是否变快、是否更准确都尚未在本项目验证**。

需要注意：

- cuBLAS 的 FP8 outer-vector 逐行/逐通道缩放支持表列为 SM90；不能将当前 INT8 的逐行激活、逐输出通道权重 scale 原封不动映射到 SM120 FP8。32 元素块缩放是另一种精度策略。
- 按实际矩阵和 scale 布局满足对齐、打包要求；B15 的 8 对齐剪枝宽度不保证直接适合新内核。padding、scale 处理和窄矩阵低利用率可能抵消收益。
- [CUTLASS SM120 文档](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html#blackwell-sm120-gemms)明确它使用不同于 SM100 的 MMA 路径，TN 布局、cluster 为 1×1×1；不能直接复用 B200/GB200 的 SM100a 内核。可从 NVIDIA 的 [SM120 MXFP8 示例](https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79c_blackwell_geforce_mixed_mxfp8_mxfp6_bf16_gemm.cu)构建独立原型，不直接更换现有认证路径的 CUTLASS 依赖。

[NVIDIA NVFP4 说明](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)采用 FP4 值、16 值块的 FP8 scale 和全局 FP32 scale。细粒度缩放不能消除 4 位量化的信息损失。[QAD 技术报告](https://research.nvidia.com/labs/nemotron/files/NVFP4-QAD-Report.pdf)研究通过量化感知蒸馏恢复精度，这需要可训练模型及教师输出；不是推理末端加一个偏差修正。

### 为什么暂不直接接 SageAttention3 或 2:4 稀疏

[SageAttention3 官方实现](https://github.com/thu-ml/SageAttention/blob/main/sageattention3_blackwell/sageattn3/blackwell/api.cu)确有 SM120 支持，但本次查阅的前向 dispatch 实际针对 D64/D128，本项目是 D32；还需适配旋转位置编码、缩放和 half 边界。补维会增加工作量，作者也[未保证所有模型无损](https://github.com/thu-ml/SageAttention/blob/main/sageattention3_blackwell/README.md)。应先借鉴其融合/流水线思路，不直接替换已认证注意力。

2026-09-03 的 [Hardware-Aware FP4 FlashAttention-4](https://arxiv.org/abs/2609.04105)在 GB200 报告 Direct-P 收益，并强调 softmax 转换与片上依赖可能成为瓶颈；不能据此承诺 RTX 5070 Ti 的 S361/D32 加速。

[cuSPARSELt 已支持 SM120](https://docs.nvidia.com/cuda/cusparselt/release_notes.html)，但 FP16/INT8 等格式需要[每四值保留两个的 2:4 结构](https://docs.nvidia.com/cuda/cusparselt/types.html)。现有 B15 的删通道剪枝并不满足该条件；继续套稀疏需要重新改变权重并验证精度，不能把已经删掉的通道再算一次稀疏收益。

## 后续数据与共同验收

[SmoothQuant](https://arxiv.org/abs/2211.10438)对 GEMM 输入和权重做互相抵消的通道缩放，将激活离群值的量化难度转移一部分到权重；它与之前失败的输出校准不同。实际量化后仍有误差，必须有多局、双方、开中后盘及战斗/收官覆盖的数据。按整盘划分校准、选型和最终留出集合；同局相邻局面与八种对称不能当作独立棋局分开泄漏到验收集。

优化放在 `kata_nn` 共享 CUDA 执行层，可被单机和 Worker 共同调用。两模式的 batch 与请求节奏不同，必须分别检验：

1. 算子/Graph 正确性及同模型整网 FP32 金标；对于有损格式，如实保留原门 PASS/FAIL，不修改门槛制造通过。另以同模型 FP16 为参照检验最大胜率输出偏差 ≤6 个百分点，报告 policy KL、top-1、目差和 ownership 误差；新门只约束胜率输出，不能据此称其他输出均在 6% 以内。
2. 固定物理 batch 的算子和整网测量，再做单机真实搜索、真实无缓存 loopback Worker 的 ABBA；绑定模型、二进制、精度配置与库版本。分别记录低并发延迟和高并发吞吐。
3. 速度与误差候选稳定后再测等时对局；RPC/s、nnEvals/s 或局部 kernel 时间不能换算成 Elo。不同精度的 Worker/缓存身份继续隔离。

本调研不代表 MXFP8、NVFP4、SmoothQuant 或 SageAttention3 已在 B11/B15 通过数值门、获得棋力认证或进入默认部署。
