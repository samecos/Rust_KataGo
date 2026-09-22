# RustGo B11 / B15 INT8 后端

2026-09-22。新增独立的 `nnBackend=cudaint8backend`（别名 `cudaint8`），复用 CUDA 的原生模型加载、GTP、JSON analysis 和 Go Server Worker。它是有损的混合精度后端，原有 `cudabackend` 和 FP32 认证门保持原样。

初次交付的记录如下：B11 的 FFN INT8 在 C32 提高约 12.5%，剪枝 B15 则慢约 3.5%。用户随后授权了剪枝专项优化：新候选 B15 C32 为 **+3.26%**（全 FFN）或 **+5.43%**（宽层 INT8），B11 C32 为 **+17.78%**，均相对同配置默认 FP16。最新二进制、配置、复测与 `0831.sgf` 校准未通过的边界见 [剪枝 B15 诊断](RustGoB15剪枝INT8诊断.md)。本文下方表格和旧二进制 SHA 保留初次交付历史，不是新候选的复测结果。两轮均未证明棋力净收益。

## 支持范围与实现

- 原生 TF3 v17、19 路 Transformer，覆盖 B11 / B15 的主干和注意力头尺寸。
- 支持模型声明的 **FFN 通道结构化剪枝**：各层读取自己的 hidden 宽度，正数、8 对齐、不超过 `3*mid`；下投影内部补到 16 对齐，补零不参与量化范围统计。不是任意稀疏格式或 2:4 稀疏加速。
- `cudaInt8Scope=ffn` 为默认：量化 FFN 的 gate/up/down。`transformer` 额外量化 QKV 和 attention 输出投影。
- 新候选支持 `cudaInt8MinFfnWidth=384` 等显式阈值，较窄 FFN 保留 FP16；默认 0。阈值是独立精度身份的一部分。新脚本的阈值测试须使用最新构建，旧交付二进制不支持此配置。
- 量化层使用真实 **INT8 × INT8 → INT32** cuBLASLt GEMM。本机 RTX 5070 Ti 选择 IMMA 整数 Tensor Core；没有可用整数算法时明确报错，不悄悄执行 FP16 GEMM。
- 权重从原始 FP32 按输出通道量化；激活按每个 token 的实际值动态量化。对称范围 `[-127,127]`、round-to-nearest-even；不需要离线校准数据。版本标识 `w8a8-row-out-rne-v1`。
- 反量化、归约、激活、残差使用 FP32 计算，保留约定的 half 舍入边界。Attention 核心、stem、外层瓶颈投影、policy/value/ownership heads 保留原浮点路径。`transformer` 也不是所有算子都改成 INT8。
- FFN 将上投影反量化、SwiGLU、下投影输入量化融合，减少两份中间 half 张量的写回；独立对照测试保留原舍入边界并逐项检查结果，不以融合为理由改变数学语义。
- 已量化的投影不再上传一份重复 FP16 权重；这部分权重存储约减半，但 scales、INT32 中间结果和其他浮点工作区仍占显存，**总显存不保证减半**。

原始模型文件和 SHA256 不变。默认模式选择较小的量化范围，不能由此推断棋力损失为零。

## 构建与启动

本机独立二进制：`target/int8-backend/release/katago-rs.exe`。在仓库根目录执行：

```powershell
cargo build -p katago --features cuda --release --target-dir target/int8-backend

# B11 GTP；换成 B15 文件即可使用相同配置。
.\target\int8-backend\release\katago-rs.exe gtp `
  --model .\models\b11c768h12nbt3tflrs-fson-silu.bin.gz `
  --config .\configs\gtp_int8.cfg

# 剪枝 B15 Worker。这里示例连接独立测试 Server 的 50052 端口。
# Server 必须使用同一原始模型 SHA，勿与 FP16 Worker 混入同一池。
.\target\int8-backend\release\katago-rs.exe nnworker `
  --server 127.0.0.1:50052 --worker-id rustgo-int8-b15 `
  --model .\models\b15-ffn-pruned-a8.bin.gz `
  --config .\configs\worker_int8.cfg --capacity 32
```

两个配置都是物理 batch 上限 8、单 NN 服务线程；`capacity 32` 是在途请求数，不是物理 batch。JSON analysis 将 `gtp` 改为 `analysis`，其余模型和配置相同。

扩大到 Transformer 投影量化时，显式添加：

```text
--override-config cudaInt8Scope=transformer
```

不要带入旧 FP16 `cudaTacticPlan`。该后端会拒绝旧认证计划，以及与之不兼容的 DualFFN、fusion、split-K、专用投影和模型专属 compact artifact。已经写进模型结构的剪枝仍支持。当前 FULL AUTOTUNE / 日常自动调优保持 FP16 路线，不产生 INT8 认证计划。

Worker 的 `backend_info` 明确报告量化版本和 scope；模型 SHA 仍绑定原文件。Server 现有协议不会自动把同 SHA 的不同精度隔离，因此切换时使用独立的 Worker 池/会话并清空旧评估缓存，避免同一搜索树混用两种输出。不能只依据模型 SHA 将两个后端当作数值等价。

## 复现验证

```powershell
$env:KATAGO_TEST_MODEL_DIR = (Resolve-Path .\models).Path
cargo test -p kata_nn --features cuda --release --target-dir target/int8-backend `
  --test int8_kernels --test int8_pruned_model -- --nocapture

# 复用仓库的 Python 依赖入口，选择/创建含 grpcio 的环境。
. .\scripts\tuning_python.ps1
$int8Python = Resolve-RustGoTunePython
& $int8Python scripts/validate_int8_backend.py `
  --binary target/int8-backend/release/katago-rs.exe `
  --model models/b15-ffn-pruned-a8.bin.gz `
  --reference autotune-output/references/b15-ffn-pruned-a8-3f216ee8-fp32.json.gz `
  --output target/my-int8-b15-accuracy

& $int8Python scripts/smoke_int8_backend.py `
  --binary target/int8-backend/release/katago-rs.exe `
  --model models/b15-ffn-pruned-a8.bin.gz --output target/my-int8-b15-smoke

& $int8Python scripts/benchmark_int8_backend.py `
  --binary target/int8-backend/release/katago-rs.exe `
  --model models/b15-ffn-pruned-a8.bin.gz `
  --accuracy-report target/my-int8-b15-accuracy/report.json `
  --scope ffn --requests 4096 --output target/my-int8-b15-speed
```

输出目录必须是新目录。B11 的现有原始模型可省略 `--reference`，使用内置 SHA 匹配的参考包；其它模型必须先取得对应的 C++ FP32 参考。

数值脚本使用 16 个固定局面 × 8 个对称变换，各跑 W1/W32；这不是 128 局独立对局。先确认 FP16 对照通过原门，再保留 INT8 的原门 PASS/FAIL、policy KL、首选变化、胜率/目差误差。`COLLECTED` 仅表示数据收集完成，不代表数值认证或棋力验收。

性能脚本绑定模型、二进制、数值报告 SHA，使用真实无缓存 loopback Worker，按 ABBA 顺序运行，核对实际 NN 行数及 Drain；1% 收益/5% 波动只用于性能判定。默认对照为同一新二进制的 FP16 默认配置，不代表历史最优认证计划，也不把 RPC/s 换算成 Elo。

## 本机验证与测量记录

最终数据和限制见下表及同目录证据摘要。未取得未剪枝 B15 的真实权重文件，因此不能声称已完成该文件的整网金标验证；完整宽度覆盖来自把真实剪枝 B15 的零通道恢复到 hidden=1536 的等价测试。

| 验证项 | 结果 |
|---|---|
| CPU 回归 | kata_nn 156、kata_program 79、kata_worker 13 项通过，共 248 项 |
| 整数算子 | CPU INT32 独立点积参考、输出通道量化轴、RNE、零通道、非 16 对齐尾部、动态 Graph 输入通过 |
| FFN 融合 | 5 组尺寸的融合/未融合输出、量化结果、scale 逐项一致；Graph 重放通过 |
| 剪枝 B15 | 45 层补零恢复到完整 hidden=1536，五组原始输出完全一致 |
| 整网融合等价 | 两模型 × 两 scope × W1 各 128 份，共 512 份后处理输出与未融合版本完全一致 |
| Worker 整网 | 最终版本两模型 × 三精度 × W1/W32，共 1,536 请求；身份、合法性、有限数值、计数和 Drain 正常；其中 FP16 对照全部通过原门 |
| 旧 ONNX 回归 | B8、16 个局面，原 policy/value/misc/ownership 门通过，top-1 16/16 |
| GTP / analysis / 配置边界 | B11、剪枝 B15 各 7 项通过：两 scope 的 GTP genmove、带 ownership 的 JSON analysis、错误 scope、FP16 plan、错误后端 scope、专用 FP16 tactic 拒绝 |
| 共享内存竞争 | 融合 FFN 的 Compute Sanitizer racecheck：0 hazards、0 errors、0 warnings |
| 显存检查 | memcheck 未报告越界/泄漏，0 bytes leaked；另有 664 条已处理的 cuModuleGetFunction NOT_FOUND 查找诊断，因此不宣称整个检查器为“0 errors” |

memcheck 默认通信曾附加超时，切换 NVIDIA 文档支持的进程内 `NV_COMPUTE_SANITIZER_LOCAL_CONNECTION_OVERRIDE=named-pipes` 后完成检查；没有改驱动、注册表或防火墙。诊断来自现有 `CudaRuntime::get_func` 逐模块寻找符号的实现，664 条均核对为同一类查找错误；测试仍实际执行并通过。原始日志包括首次失败，均保留。接口说明见 [Compute Sanitizer 环境变量](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#environment-variables)。

量化误差如下，胜率误差单位为**百分点**，目差为网络预估分差绝对偏差；全部比较同模型 C++ FP32，不能解释为胜率或棋力下降量。

| 模型 / scope | 窗口 | 胜率平均 / 最大偏差 | 目差平均 / 最大偏差 | policy 首选一致 |
|---|---:|---:|---:|---:|
| B11 / ffn | 1 | 0.308 / 2.106 | 0.036 / 0.125 | 116/128 |
| B11 / ffn | 32 | 0.279 / 1.957 | 0.036 / 0.119 | 114/128 |
| B11 / transformer | 1 | 1.808 / 4.552 | 0.328 / 1.035 | 108/128 |
| B11 / transformer | 32 | 1.775 / 4.691 | 0.328 / 1.060 | 114/128 |
| 剪枝 B15 / ffn | 1 | 0.132 / 1.084 | 0.018 / 0.066 | 115/128 |
| 剪枝 B15 / ffn | 32 | 0.109 / 0.650 | 0.018 / 0.087 | 123/128 |
| 剪枝 B15 / transformer | 1 | 0.644 / 6.311 | 0.085 / 0.289 | 126/128 |
| 剪枝 B15 / transformer | 32 | 0.634 / 4.751 | 0.087 / 0.297 | 124/128 |

所有 INT8 组均未通过原 FP32 严格门；没有改门来制造通过。首选变化包含接近的候选，也有较大概率差的候选，不能一律解释为无害平局。`transformer` 虽有时首选一致率更高，value 偏差仍明显更大。没有开展等时对局，也没有证明额外速度能抵消棋力损失。

实际无缓存 Worker 吞吐，单位 RPC/s。每臂预热 256、测量 4,096 请求，ABBA 两次取几何均值；物理 batch 上限 8、单服务线程、Graph，ownership 关闭。不同模型各用自己的同 SHA FP16 对照。

| 模型 / scope | 在途数 | FP16 基线 | INT8 | 变化 | 两臂最大重复波动 |
|---|---:|---:|---:|---:|---:|
| B11 / ffn | 1 | 378.14 | 363.70 | −3.82% | 0.792% |
| B11 / ffn | 32 | 827.57 | 931.15 | **+12.52%** | 0.268% |
| 剪枝 B15 / ffn | 1 | 312.21 | 285.71 | −8.49% | 0.157% |
| 剪枝 B15 / ffn | 32 | 669.90 | 646.35 | −3.52% | 0.367% |
| 剪枝 B15 / transformer | 32 | 670.72 | 578.76 | −13.71% | 0.345% |

这些组全部通过 5% 波动检查，只有 B11 / ffn / C32 通过 1% 性能收益门。B11 的 transformer 范围仅验证了数值和接口，未测其吞吐。不能把上述比例套用到其它 batch、流数、GPU 或原最优 FP16 计划。

未融合初版 B11 / ffn 的 C1/C32 分别为 −13.29% / −12.81%，失败记录保留。融合版已通过组件和整网等价验证，其最终 ABBA 结果如上；没有为拿到收益调整精度。剪枝 B15 已减少 FFN 计算量，当前量化/整数中间结果处理开销抵消了收益；本轮未将其切成生产默认，也没有继续筛选多轮参数直到出现正数。

因此，本机 B11 高并发可将 FFN INT8 作为后续等时对局的候选；B11 低并发和当前剪枝 B15 保留 FP16。扩大量化范围在已测剪枝 B15 上既更慢、value 误差也更大，不推荐。模型能加载并正确执行整数算法，并不意味着速度和棋力都更好。

测试平台为 Windows、RTX 5070 Ti、驱动 610.88、CUDA 编译器 13.3.73。本轮 GPU 工作顺序执行；已有生产 Worker 保持驻留，抽查其 CPU 活动极小、运行间隙 GPU 空闲，未宣称独占 GPU 环境。

独立交付二进制 SHA256：

```text
d6a911e9cc9cf89ffcfe63910982f0dde4723e718c45581e206da561d2718acc
```

原 `target/release/katago-rs.exe` 的 SHA 仍为 `d218b98dfdf2eb75820465cbf625dfd3f690b91c3bb112f3be1a56d2cc644dcc`。没有改写旧计划、模型、运行中的 Worker，也没有提交或推送代码。

可携带的摘要在 [int8-evidence-20260922.json](int8-evidence-20260922.json)，包含模型/二进制/报告 SHA、每次测量、误差、失败尝试和验证范围。完整本机证据在 `target/int8-validation-{b11,b15}-fused/`、`target/int8-bench-*-fused/`、`target/int8-smoke-*-fused-r2/` 及摘要所列日志；这些目录被 Git 忽略，保留实验档案时需一并携带。
