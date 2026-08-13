# CUDA 优化战术记录（M4 路线图）

> 方法论与战术清单移植自 KataGomo_fork（SM120 plan 驱动优化，5080 上 2836 nnEval/s）。
> 本机目标：RTX 5070 Ti（SM 12.0，16GB），模型 b11c768h12nbt3tflrs-fson-silu（75M 参数，FP16/NHWC）。

## 基线（待 M3 执行器落地后实测记录）

- 官方参照（本机无 cuDNN，用 TensorRT 10.16 兜底作基准）：nnEval/s 待测
- M3 手写后端 v1（正确优先，单流）：nnEval/s 待测

## 决策组（严格有序，后组不得改写前组配置键）

| # | 组 | 战术 | 状态 |
|---|---|---|---|
| G1 | fa4 / wide_projection / qkv_rope / dual_ffn | 宽 QKV 单 GEMM（M=B×361、N=1152、K=384，packed 行 [Q384\|K384\|V384]，tile M128×N128×K64 s2）；FA attention（tile M128×N64、stages=1、noncausal 无掩码、both16 双 FP16 累加 + **rescale 舍回 FP16 累加器**、128 线程）；Q/K RoPE 为**独立 batch 共享 kernel**（361×192 线程、half2 对、按 B 展开，非 GEMM epilogue——SM120 证伪了融合）；dual FFN 共享 A + SwiGLU epilogue（CUTLASS LeftSiLUAndMul） | 待做 |
| G2 | fused_residual / linear2 / outproj | GEMM beta=1 原位残差（C==D 同指针，epilogue 先读 C 再写 D；tile 128×128×32、warp 64×64×32、3 stages） | 待做 |
| G3 | postconv_bn / preconv / pointwise | affine+SiLU（half2 `__hfma2`，sigmoid 逐 half 转 float）；RMSNorm C384 用 warp4-vec8（**零共享内存**：uint4+uint2 载入 12 half/lane、单 XOR 链归约、4 行/块）；SwiGLU 已折入 G1 FFN epilogue | 待做 |
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

## 已知缺口（待修，按优先级）

1. **GTP 参数系统**：`kata_search/src/params.rs` 的 changeable 参数表仅 9 条，C++ `searchparams.cpp` 有 115 条；`kata-set-params`（JSON 批量）未实现。影响 kata-get/set-param、kata-list-params、kata-set-params 的完整性（需求 F2）。修法：对照 C++ changeableParametersToJson 补齐 + gtp.rs 的批量设置。
2. **analysis allowMoves**：每请求限 1 条（C++ 允许每玩家 1 条共 2 条）。**已修**（2026-08-14：放宽到 2 条 + 同玩家重复报错，消息与 C++ 逐字对齐；rustfmt 语法验证通过，待编译验证后提交）。
3. **M4 待做**：见上方决策组表。

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

### Rust 现状（gtp.rs:1337-1391；2026-08-14 复查更新）
- kata-list-params：已硬编码 9 个特殊参数名（与 C++ 完全一致），缺 changeable 部分
- kata-get-param：已实现 9 个特殊参数取值分支，缺 changeable 部分
- kata-get-params：返回 `{}`（需实现完整 JSON）
- kata-set-param(s)：直接报 "not implemented"
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

