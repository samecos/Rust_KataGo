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
| (进行中) | dual FFN(gate+up 单 GEMM) | 待测 | agent-16 |

单 batch 下 M=361 的 GEMM grid 仅 3×N/128 块,SM 利用率低是根本限制;
fork 的 2836 nnEval/s 依赖 B16 批 + 双流(batch 聚合是后续最大杠杆)。

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

