# INT8 专项优化：RMSNorm 与输入量化融合

2026-09-22，用户再次授权 INT8 专项优化。本轮在上一版 INT8 之上取得几个百分点的增益，已保留有收益的全 FFN 路径；剪枝 B15 的宽层混合配置没有达到收益门，默认保留原路径。未替换正式 FP16 二进制、计划或正在运行的 Worker。

## 交付与实测

构建：`target/int8-specialized/build/release/katago-rs.exe`。

- 最终 SHA-256：`aa3a3a80e715c19710ba1c68884831efc08df27c1b396be52790e76169088abf`。
- 直接对照为上一版 INT8：`target/int8-pruned-study/build/release/katago-rs.exe`，SHA-256 `8a084a514a88b1d37fe8e640e3ad184add52228f6f4cb66cba19e61119f6e87d`。
- 两臂相同模型、FFN 量化范围、宽度阈值、物理 batch 上限 8、单 NN 服务线程、Graph、缓存配置。真实本机 gRPC Worker，无缓存请求、ownership 关闭，每臂预热 256、测量 4096 请求，顺序 ABBA。C 是在途请求数，C1 的实际 batch 为 1，C32 的平均 batch 约为 8。
- RTX 5070 Ti、驱动 610.88、CUDA 13.3.73。最终 ABBA 没有与本轮编译、其他推理或 profiler 重叠；原生产 Worker 保持运行，不宣称独占 GPU。

| 模型 / 精度配置 | 并发 | 旧 INT8 RPC/s | 优化后 RPC/s | 收益 | 最大重复波动 |
|---|---:|---:|---:|---:|---:|
| B11 / 全 FFN INT8 | 1 | 362.75 | 373.60 | **+2.99%** | 0.58% |
| B11 / 全 FFN INT8 | 32 | 977.81 | 993.23 | **+1.58%** | 0.13% |
| 剪枝 B15 / 全 FFN INT8 | 1 | 283.28 | 294.79 | **+4.06%** | 0.32% |
| 剪枝 B15 / 全 FFN INT8 | 32 | 691.68 | 706.52 | **+2.15%** | 0.20% |

四组均通过至少 1% 收益、至多 5% 波动的门槛。表中是配对 ABBA 的几何均值，比较对象是同精度配置的旧 INT8；本轮没有重新测量最优 FP16 认证计划，也不将这些比例乘到历史 FP16 比较结果上。

B15 的 `cudaInt8MinFfnWidth=384` 只量化 12 层较宽 FFN。对它强制融合的探索结果为 C1 **+0.97%**、C32 **−0.10%**，未采纳；该探索的 C1 阶段有一次约 8 秒的测试编译重叠，不能作为收益认证。最终自动模式对所有非零宽度阈值保留原路径。此次全 FFN 路径的提升不代表超过已有宽层混合方案的最佳吞吐。

## 改了什么

新增 C384 / C512 两个 CUDA kernel，将 FFN 前的 RMSNorm、half 舍入和动态逐行 INT8 量化合并。每个 warp 处理一行，每个 block 处理四行，归一化值保存在寄存器里，省掉一次 kernel 启动和中间 half 张量的写回、读回。

保留原 RMS 的 FP32 求和顺序、half 存储边界的舍入、RNE、零行与非有限值处理。整数 GEMM 仍为 INT8 × INT8 → INT32；SwiGLU、下投影、反量化残差、注意力和输出头保持原路径。量化版本仍为 `w8a8-row-out-rne-v1`，没有启用 fast math，也没有改变精度配置或加载校准参数。

新的 shape 检查与每 stream 的独立工作区继续用于 Graph 重放。显式 RMS v1、split-K 待归约路径以及逐层 debug dump 保留原实现。新路径有 `[cuda-tactic] name=int8_rms_quantize` 标记；逐层计时中融合段标为 `int8_rms_ffn_fused`。

独立 Nsight B8 追踪确认：五次前向中，原来的 450 次 RMS 与 225 次输入量化，变为 225 次 RMS 和 225 次融合 kernel；每次前向少 45 次启动。相应单 kernel 中位数：原 RMS 4.544 μs、输入量化 5.760 μs，融合后 5.120 μs，合并段约减半。这是带 profiler 的局部诊断，不是 Worker 吞吐。

同一份旧版追踪中，注意力核心和前五项里的浮点矩阵乘法合计约占 kernel 时间的 67.5%。剪枝已经大幅缩小 FFN，所以局部量化处理减半只带来几个百分点的整网增益。

## 数值与功能验证

- `int8_kernels` 四项通过：独立 CPU INT32/RNE 对照、剪枝补齐、零输入、极小/极大值、NaN/Inf、非整 block 行数、B11/B15 宽度、流隔离和动态输入 Graph 重放。
- `dump_int8_rms_ab` 在独立进程中运行融合关、强制开、自动三种模式。B11 全 FFN、B15 全 FFN、B15 阈值 384，各 B1/B8，五个原始输出张量共 **68,958 个 FP32 值**，三种模式逐位相同。
- 最终真实 Worker 数值采集共 1536 请求：三种配置 × FP16/INT8 × W1/W32 × 128。128 是 16 个语义局面及其 8 个对称，不是 128 盘独立棋谱。六组 FP16 对照均通过原 C++ FP32 门。
- 与旧 INT8 的 W1 输出比较，三种配置共 384/384 完全一致。全 FFN 的 W32 输出也共 256/256 一致。混合配置 W32 为 103/128 一致，异步凑批不是固定形状对照；固定 B1/B8 的原始输出逐位对照另行通过，不能把这一差异用作校准恢复的证据。
- Compute Sanitizer 对 `int8_` kernel 的 racecheck 为 0 hazards / 0 errors / 0 warnings。memcheck 完整 API 日志有 556 次 `cuModuleGetFunction` 的 `CUDA_ERROR_NOT_FOUND`，逐条核对均为现有跨 PTX 模块查找过程，初次退出码 99，原日志保留。关闭 API 错误报告后的专项内存检查退出码 0、0 errors、0 bytes leaked。此处范围是 INT8 kernel，不宣称检查了所有供应商内核。
- B11/B15 的 GTP、JSON analysis、ownership、错误配置、禁止复用 FP16 plan、融合关闭与非法融合参数均有冒烟验证，见证据文件。

**本轮没有恢复 INT8 相对 FP32 的精度。** 最终六组 INT8 仍未通过原 FP32 门。以下胜率误差单位为百分点；不是对局胜率或 Elo。

| 配置 | 请求窗口 | 平均胜率输出绝对误差 | 最大误差 | policy top-1 |
|---|---:|---:|---:|---:|
| B11 全 FFN | 1 | 0.3076 | 2.1063 | 116/128 |
| B11 全 FFN | 32 | 0.2789 | 1.9569 | 114/128 |
| B15 全 FFN | 1 | 0.1321 | 1.0841 | 115/128 |
| B15 全 FFN | 32 | 0.1091 | 0.6498 | 123/128 |
| B15 阈值 384 | 1 | 0.1339 | 1.6772 | 123/128 |
| B15 阈值 384 | 32 | 0.1398 | 1.5984 | 122/128 |

此前使用用户提供的 `0831.sgf` 进行的简单输出校准没有通过独立验证，本轮没有重拟合或安装该参数，详见 [上轮诊断](RustGoB15剪枝INT8诊断.md)。上述融合对拍通过说明本轮在所测输入上没有增加误差，不能替代更大样本的精度验证或等时对局。

## GEMM 探索的边界

新增离线 `probe_int8_algos`：分别查询当前单候选算法与较大候选池，在 16 种 B1/B8、B11/B15 剪枝矩阵尺寸上，共 149 个候选先通过完整 CPU INT32 输出对照，再做每图 512 次 GEMM 的五次 CUDA event 计时。

个别形状有更快候选，例如 B8、N=2320/K=512 的上投影约 52.86→37.25 μs。但它只对应个别层，部分其他计时波动仍较大；本轮没有接入这些算法，也没有将算子收益算作整网收益。早期过短计时和 opaque descriptor 字节比较失败的探索日志保留，最终探针使用单独查询的生产算法作为基线。

```powershell
cargo test -p kata_nn --test probe_int8_algos --features cuda --release -- --ignored --nocapture --test-threads=1
```

## 使用与复现

默认全 FFN INT8 自动启用此次融合，无需新增配置项：

```powershell
./target/int8-specialized/build/release/katago-rs.exe gtp `
  --model models/b15-ffn-pruned-a8.bin.gz --config configs/gtp_int8.cfg `
  --override-config nnBackend=cudaint8backend,cudaInt8Scope=ffn,cudaInt8MinFfnWidth=0
```

B11 更换为 `models/b11c768h12nbt3tflrs-fson-silu.bin.gz`。已有宽层混合配置继续使用 `cudaInt8MinFfnWidth=384`，默认不会启用此次未获收益的融合。原生 FFN 结构化剪枝、Worker 接口与独立 INT8 精度身份保持。

诊断环境变量 `KATAGO_CUDA_INT8_RMS_FUSION=0` 强制关闭，`=1` 强制打开可支持的 C384/C512 路径；非法值报错。未设置时按上述自动规则选择。它经 `tactic_plan::tactic_var()` 读取，但不复用 FP16 的认证 plan；强制给混合配置启用也不代表已经获得性能认证。

```powershell
cargo build -p katago --features cuda --release --target-dir target/int8-specialized/build
cargo test -p kata_nn --test int8_kernels --features cuda --release -- --nocapture --test-threads=1
```

真实 Worker 数值与速度继续使用 `scripts/validate_int8_backend.py`、`scripts/benchmark_int8_backend.py`。后者提供 `--baseline-binary` 和 `--baseline-config` 时可与旧 INT8 配对，数值报告必须绑定被测二进制及模型 SHA。原始证据保存在 `target/int8-specialized/`；可携带的结果、边界与文件哈希见 [本轮证据](int8-specialized-evidence-20260922.json)。
