# RustGo 原生 TF3 剪枝模型支持

2026-09-20。本次针对 `models/b15-ffn-pruned-a8.bin.gz` 增加 CUDA 支持，并让 FULL AUTOTUNE 按模型结构筛选候选。模型 SHA256 为 `3f216ee88226ee49ca826eaa5f7e1e9982ff48a96625914bfc4e5ad20ee095a7`。

## 实现范围

该模型有 15 个嵌套块、1024 主干通道、512 瓶颈通道、16 个 D32 注意力头。45 层 FFN 的隐藏宽度分别读取模型声明，范围 16～1160，其中部分仅按 8 对齐。

新增支持包括原生层图参数化、可变头数的 QKV/RoPE/FlashAttention、可变通道 RMSNorm、头部输入通道与工作区尺寸，以及 FFN 下投影所需的逐行 16 对齐补零。模型文件和权重保持原样；补零是 CUDA 内部存储布局。激活和归约仍使用 FP32，原有 half 存储边界不变。

仍有明确结构限制：原生 TF3 v17、19 路、22 空间/19 全局输入、无 metadata、标准 trunk-tip BN、SiLU、SwiGLU、等量 Q/K/V 头与 D32 learned 2D RoPE；policy 头 96、value 头 192，头部 BN scale 必须为 1。主干须大于瓶颈，主干按 16 对齐，瓶颈按 32 对齐；每层 FFN hidden 为正、按 8 对齐且不超过 `3*mid`。这不是任意剪枝网络或稀疏权重格式支持。

旧 C768/M384/H1152 的固定尺寸 DualFFN、fusion 和 split-K 路径不会用于新结构；引擎会明确拒绝不兼容开关，FULL 会提前跳过。新结构仍可调 batch、Graph、服务线程、GEMM、attention、RMS、批处理策略和 local 搜索线程，候选必须逐一通过原有数值门。

## 本机使用

独立二进制位于 `target/pruned-support/release/katago-rs.exe`。原 `target/release/katago-rs.exe`、旧 Worker 和 local PLAN 保留。新内核及 host revision 改变了构建指纹，因此新二进制必须重新生成 PLAN，不能重写旧 PLAN 的指纹来复用。

完整 Worker 调优命令（在仓库根目录运行，未指定输出目录时自动创建新目录）：

```powershell
.\scripts\full_autotune.ps1 `
  -Binary .\target\pruned-support\release\katago-rs.exe `
  -Model .\models\b15-ffn-pruned-a8.bin.gz `
  -Reference .\autotune-output\references\b15-ffn-pruned-a8-3f216ee8-fp32.json.gz `
  -Mode worker -Capacity 64
```

生成的 PLAN 仍是 **JSON**。只使用本次输出中有 `result/READY` 标记的 `result/plan.json` 与配套 `rustgo.cfg`。报告中的 `FULL_CATALOG` 表示执行了全部适用决策组；不适用的固定尺寸组列为 skipped，不能强行开启。

单机搜索/分析另跑同一命令，改为 `-Mode local`。每次使用独立输出目录；local 与 Worker 的运行参数及结果不能混用。

`cuda-fingerprint --model <模型路径>` 现在也会验证原生结构，并输出 `model_architecture`，包括每层 FFN 宽度和固定尺寸 tactic 的适用性。架构不支持时会在调优开始前报错。

## 另一台 CUDA 电脑

携带支持目标 GPU 的新二进制、FULL 工具文件、原始模型和上面的 FP32 参考包。参考包仅 128 个固定请求的输出，不包含模型权重，可以用于该原始模型在另一台机器上的数值检查。参数中的二进制、模型、参考包路径均改成目标机器的实际路径，再运行完整调优。不要直接复制本机 PLAN 作为另一块 GPU 的认证结果。

如果模型文件发生变化，必须重新生成与其 SHA256 匹配的 C++ FP32 参考。可以传 `-CppWorker <支持 nnworker 的 C++ 可执行文件> -CppConfig .\configs\worker_cpp_fp32.cfg`，代替 `-Reference`，让 FULL 自动生成。普通未实现该 Worker 协议的 KataGo 不可用作此导出器。更多参数见 [FULL AUTOTUNE](RustGo-FULL-AUTOTUNE.md)。

重新构建支持版本：

```powershell
cargo build -p katago --features cuda --release --target-dir target/pruned-support
```

## 验证记录

最终记录以本节列出的独立报告为准；首次寻址错误的失败报告保留用于诊断，不是可用 PLAN。

| 检查 | 本机结果与证据 |
|---|---|
| 剪枝模型原生解析/下沉 | 45 层真实 FFN 宽度、权重转置、QKV/RoPE 形状、非法宽度/不完整层对拒绝均通过；`target/pruned-cpu-test.log` |
| 新 CUDA 核函数 | 补零 SwiGLU、32/384/512/1024 通道 RMSNorm、三种 16 头 FlashAttention 均通过；注意力相对独立 CPU 参考最大绝对误差约 `3.04e-5`；`target/pruned-kernels-final.log` |
| 剪枝模型整图 FP32 对拍 | W1/W32/W64，每组 128 局面，所有字段通过且 policy top-1 都为 128/128；`autotune-output/pruned-support-numeric-v2/report.json` |
| 旧原生 b11 回归 | 普通路径、DualFFN 路径分别通过 W1/W32/W64 的全 128 局面对拍；`autotune-output/pruned-support-legacy-native/report.json` |
| 旧 b11 ONNX 回归 | 16 个局面 raw policy/value/misc/ownership 门全部通过，top-1 16/16；`target/pruned-legacy-onnx-compare.log` |
| 剪枝模型 local 接口冒烟 | GTP genmove、JSON analysis 完整返回，手动开启不兼容 DualFFN/fusion/split-K 均明确拒绝；`autotune-output/pruned-support-local-smoke/report.json`。这是接口兼容性检查，不是 local 调优结果 |
| PLAN 和调优工具 | 39 项 Rust PLAN 测试、31 项 Python 调优测试通过；旧/缺失构建身份仍按既有规则拒绝，未放宽数值门 |

测试平台为 RTX 5070 Ti；其它 GPU 和其它权重仍需用目标机器的 FULL 运行报告验证。这里的 Worker 数值认证包含特征构建与后处理，不是新模型的独立逐层 FP32 对拍，也不代表棋力评估。

本次已生成可用 Worker 结果：

- PLAN：`autotune-output/pruned-worker-plan-20260920/result/plan.json`
- 配置：同目录 `rustgo.cfg`；启动入口：`run-nnworker.ps1`
- 完成标记：同目录 `READY`；完整报告：上级 `report.json`
- 范围为 **CUSTOM_GROUPS**：固定 B8，检验 attention/RMS 候选，未宣称完成全部 14 组 FULL 搜索。
- 选择：B8、单 NN 服务线程、Graph、q64、默认 w4 RMS；W1/W32/W64 最终复验再次全部通过。
- 最终 ABBA：相对同一剪枝模型的 q128 基线，`672.32 → 694.05` uncached loopback gRPC requests/s，提升 **3.23%**，两侧波动 `0.145% / 0.054%`。这不是相对旧 b11 模型的比较。

上文不带 `-Groups` 的命令才是完整搜索。现有 Worker 结果可以直接验证接入；local 优化仍需单独跑 `-Mode local`。

本次被测新二进制 SHA256：`6ca478ab0065f66140a71fdc26a1b1e253475467ff5971d92ba57b41f9821d32`。原正式二进制 SHA256 仍为 `d218b98dfdf2eb75820465cbf625dfd3f690b91c3bb112f3be1a56d2cc644dcc`，未被覆盖。
