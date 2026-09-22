# 剪枝 B15 的 INT8 变慢原因与修正

后续专项优化和当前交付见 [INT8 专项优化](RustGoINT8专项优化.md)。本文保留上一轮结果、二进制及 SGF 校准记录。

2026-09-22，用户明确授权的新一轮工作，并允许改变 INT8 实验中的中间累加精度，以最终输出及独立校准验证为准。本次找到的是量化处理与模型尺寸不匹配的问题；采用的 kernel 优化没有改变原量化数学结果。

## 结果

剪枝 B15 的高并发 INT8 已从此前慢于 FP16，改进为快于同配置 FP16。单请求没有稳定达到 1% 收益门。精度恢复仍未通过验证：使用用户提供的 `0831.sgf` 做简单输出偏差校准，训练样本上改善，独立验证的平均胜率输出误差反而增大，因此没有接入这些校准参数，也没有替换正式 FP16 部署。

RTX 5070 Ti、驱动 610.88、CUDA 13.3.73、cuBLASLt 130600。下表为最终同一二进制中的默认 FP16 对照与 INT8 候选：真实无缓存 loopback Worker，ABBA，每臂预热 256、测量 4096 请求，物理 batch 上限 8、单 NN 服务线程、Graph、ownership 关闭。C 是在途请求数，不是物理 batch。

| 模型 / 量化范围 | 并发 | FP16 RPC/s | INT8 RPC/s | 变化 | 性能门 |
|---|---:|---:|---:|---:|---|
| B15 / 全部 45 层 FFN | 32 | 670.71 | 692.57 | +3.26% | 通过 |
| B15 / hidden ≥ 384 的 12 层 FFN | 32 | 670.54 | 706.98 | +5.43% | 通过 |
| B15 / hidden ≥ 384 的 12 层 FFN | 1 | 308.48 | 310.26 | +0.58% | 未达 1% 收益门 |
| B11 / 全部 33 层 FFN | 32 | 829.00 | 976.42 | +17.78% | 通过 |

所有最终组两次重复的波动均低于 0.9%，满足 5% 波动门。最终测速没有与编译、校准推理或其他本轮 GPU 测试重叠。已有生产 Worker 保持运行，抽查 CPU 活动极小；不宣称独占 GPU。对照是同配置默认 FP16，不是原最优认证 plan；没有测等时对局，不能将吞吐提升直接换算为 Elo。

## 为什么剪枝后 INT8 反而慢

模型 `b15-ffn-pruned-a8.bin.gz` 有 45 层 FFN、mid=512。未剪枝宽度应为每层 1536；实际 hidden 总和 12672，只剩完整 FFN 宽度总和 69120 的 **18.33%**。这是 FFN 的比例，不是整网只剩 18.33% 计算量。22 层 hidden ≤ 128，最小只有 16。

原路径对每层都执行输入量化、整数上投影、融合 SwiGLU/再量化、整数下投影、反量化残差。剪枝缩小了矩阵乘法，前后处理仍然存在。INT32 中间结果也没有随权重变成一字节。Nsight 的 B8 kernel 追踪中，45 层的量化/激活/反量化三个 kernel 合计约 2086 μs，两个整数 GEMM 合计约 956 μs。前者包含必要的激活计算，不能全部算成额外量化成本；但它已占这五个 kernel 耗时的约 68.6%。

以下是原实现逐层三次测量中位数，已去掉两次预热。单位 μs；有 profiler 开销，不能当作线上单请求延迟。

| hidden | 输入量化 | 整数上投影 | SwiGLU / 再量化 | 整数下投影 | 反量化残差 |
|---:|---:|---:|---:|---:|---:|
| 16 | 15.905 | 4.064 | 15.488 | 7.040 | 11.936 |
| 472 | 15.872 | 20.416 | 19.585 | 10.368 | 12.096 |
| 1160 | 16.000 | 52.642 | 28.609 | 19.392 | 12.512 |

这里不能靠把 INT32 累加改成更低精度来消除成本：当前 cuBLASLt INT8 路径使用 `CUBLAS_COMPUTE_32I` 和 INT32 输出。放宽浮点中间求和约束，可以扩展其他候选实现，但不是当前这些额外 kernel 的直接解法。[NVIDIA cuBLAS 数据类型与计算模式](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul)

## 已实现的改动与回退

- 新增显式 `cudaInt8MinFfnWidth`，默认 `0` 保留全部 FFN 量化。设置 `384` 时，模型实际 hidden ≥ 384 的层使用 W8A8，其余层走原 FP16。阈值必须在 0..8192 且按 8 对齐；FFN scope 排除全部层会报错，不能把全浮点执行伪装成 INT8。
- 输入量化在 token 行数 ≥ 1024 时，每个 warp 处理一行、每个 block 处理四行，用 warp shuffle 计算最大值，减少块内同步。小 batch 保留原 CTA 实现。
- 融合 SwiGLU/再量化仅在行数 ≥ 1024 且 hidden ≤ 512 时使用寄存器和 warp 归约。宽层保留原实现。RNE、padding、非有限值传播以及 half 边界不变。
- Worker 身份、nnbench JSON、加载后上下文一致性检查都包含宽度阈值；实际 kernel 分支有一次性 `[cuda-tactic]` 标记。不能复用旧 FP16 plan。性能脚本从绑定二进制/模型 SHA 的数值报告读取同一阈值。

尝试过把宽层也改为 warp。hidden=1160 时 SwiGLU/再量化由 28.609 μs 变成 44.609 μs，追踪到每线程寄存器数由 30 增到 87；该版本 B15 全 FFN 的 C1 比 FP16 慢 10.05%。因此已回退宽层分支，并保留小 batch 原实现。失败代码、二进制及日志在实验目录，未作为最终版本交付。

最初仅按宽度选层的探索曾得到 C1 +2.23%、C32 +4.46%，其中部分运行与 CPU 编译重叠。最终结果只使用上表独立复测，不采用探索中较好看的单请求数字。

## 最终数值与 SGF 校准

kernel 改写与对应旧量化策略的 W1 整网输出逐项一致：B15 全 FFN、B15 阈值 384、B11 全 FFN，各 128 份，共 384 份。它没有新增量化误差；改变量化层集合本身仍是另一种有损精度策略。

与同模型 C++ FP32 参考相比，最终未校准的 B15 结果如下。胜率误差单位为百分点，指网络输出偏差，不是实际对局胜率损失。

| 范围 | 窗口 | 胜率平均 / 最大误差 | Policy KL 均值 | Policy 首选一致 |
|---|---:|---:|---:|---:|
| 全 FFN | 1 | 0.1321 / 1.0841 | 0.0002750 | 115/128 |
| 全 FFN | 32 | 0.1091 / 0.6498 | 0.0002906 | 123/128 |
| hidden ≥ 384 | 1 | 0.1339 / 1.6772 | 0.0002157 | 123/128 |
| hidden ≥ 384 | 32 | 0.1435 / 1.5984 | 0.0002352 | 123/128 |

减少量化层数改善了某些 policy 指标，但没有单调改善 value。各 FP16 对照都通过原 FP32 门；上述 INT8 组仍未通过原严格门，没有通过改门制造认证。

校准使用 `C:/Users/Administrator/Documents/0831.sgf`，SHA256 为 `f8c422cad562d091d8fd59b7949a697d45b296095db45d9b7623a9dd5cdb805f`。复用原生 SGF 解析器和 GTP 的最长分支规则，得到 78 手。从第 8 手起每隔 8 手及相邻后一手抽样，最终 **18 个历史局面 × 8 个对称，共 144 请求**，双方轮次各半。初始仅取偶数手的 9 局面探索记录保留，最终采用双方覆盖的结果。

只使用棋谱落子，不读取注释中的引擎胜率作为标签。Teacher 为同版本 FP16，独立验收仍使用 C++ FP32 基准包。与验证集检查了八种对称下的相同历史及至少 8 手的共同完整前缀，均无重叠。18 个历史仍来自一局棋，不能当作 144 个独立样本，也未覆盖完整中后盘分布。

本轮做的是有约束的**输出偏差校准试验**：在行棋方视角拟合 value logit 和目差的仿射修正，policy 拟合一个正温度；只在 SGF 上拟合，使用恒等映射正则化与幅度限制。它不是 SmoothQuant、激活范围校准或 QAT，也不能用正温度修复已经交换的 policy 首选次序。

| 范围 | 校准集胜率平均误差：前 → 后 | 独立 W32 胜率平均误差：前 → 后 | 独立 W32 Policy KL：前 → 后 |
|---|---:|---:|---:|
| 全 FFN | 0.1637 → 0.1239 | 0.1091 → 0.1305 | 0.0002906 → 0.0003005 |
| hidden ≥ 384 | 0.1178 → 0.1066 | 0.1435 → 0.1544 | 0.0002352 → 0.0002441 |

两种范围在独立 W1/W32 上均未恢复原精度门，参数**未接入推理后端**。这说明此次简单输出校准不能泛化，不说明所有 PTQ 或 QAT 都无效。若继续追求更低中间精度，应补充多局、双方、中后盘的校准/留出数据，研究逐层权重与激活校准，再验证搜索效果；不能承诺任意 INT8 信息损失都能在输出端补回。[SmoothQuant 论文](https://arxiv.org/abs/2211.10438)

## 使用与复现

本轮验证包括：整数/FFN 三项测试（八组 FFN 形状，含 1025 行、部分 warp、512/520 边界和 Graph 重放）；三组配置的 W1/W32 整网采集共 1536 请求，FP16 对照通过原门；B11/B15 各九项 GTP、analysis 与错误配置测试通过。校准数据另外运行三个配置各 144 请求。

Compute Sanitizer 限定 `kns=int8_` 检查本次相关 kernel 时，融合/Graph 测试完整通过，0 hazards、0 errors、0 warnings。包含全部第三方 kernel 的首次 racecheck 则以“target application returned an error”结束，没有完整跑完；该日志保留，不能以其显示 0 hazards 宣称全范围检查通过。未改驱动、系统超时或生产进程。

独立候选二进制：`target/int8-pruned-study/build/release/katago-rs.exe`，SHA256：

```text
8a084a514a88b1d37fe8e640e3ad184add52228f6f4cb66cba19e61119f6e87d
```

阈值 384 的 B15 候选配置，在现有 INT8 Worker/GTP 配置上追加：

```text
nnBackend=cudaint8backend
cudaInt8Scope=ffn
cudaInt8MinFfnWidth=384
```

`0` 是全 FFN。当前数据支持把这两种 B15 配置作为高并发速度候选，不支持单请求切换或宣称棋力恢复。相同原模型 SHA 的不同量化配置仍需独立 Worker 池/会话，不能混合 NN 缓存。

```powershell
cargo build -p katago --features cuda --release --target-dir target/int8-pruned-study/build
cargo test -p kata_nn --features cuda --release --target-dir target/int8-pruned-study/build `
  --test int8_kernels -- --nocapture --test-threads=1

# 沿用原 INT8 数值/性能脚本；数值命令新增 --min-ffn-width 384。
# benchmark_int8_backend.py 会从该数值报告读取并绑定相同阈值。

cargo run -p kata_data --features kata_nn/cuda --release `
  --target-dir target/int8-pruned-study/build --example sgf_worker_fixture -- `
  C:/Users/Administrator/Documents/0831.sgf target/my-calibration-fixture.json

. ./scripts/tuning_python.ps1
$int8Python = Resolve-RustGoTunePython
& $int8Python scripts/calibrate_int8_outputs.py `
  --binary target/int8-pruned-study/build/release/katago-rs.exe `
  --model models/b15-ffn-pruned-a8.bin.gz --fixture target/my-calibration-fixture.json `
  --reference autotune-output/references/b15-ffn-pruned-a8-3f216ee8-fp32.json.gz `
  --accuracy-reports target/int8-pruned-study/accuracy-final-all/report.json `
    target/int8-pruned-study/accuracy-final-selective/report.json `
  --output target/my-calibration-trial
```

新建输出目录，避免覆盖历史证据。校准脚本不修改模型或线上输出，`precision_restored=false` 必须保留。原正式二进制、原 INT8 交付二进制、模型、FP16 plan 和生产 Worker 均未替换；没有提交或推送代码。

完整证据位于 `target/int8-pruned-study/`，可携带的摘要见 [int8-pruned-evidence-20260922.json](int8-pruned-evidence-20260922.json)。
