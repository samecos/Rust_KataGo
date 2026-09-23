# INT8：注意力分块与按尺寸选 GEMM

2026-09-23。本轮用户重新授权优化，允许混合精度，并选择独立验收口径：**相对同模型 FP16，胜率输出最大绝对偏差 ≤6 个百分点**，另报落点变化与目差。原 FP32 认证门保持不变。

已实现按形状选择 INT8 GEMM 算法，并验证它与既有 q64 注意力的组合。完整调优有稳态收益，但 B15 启动约需 90 秒，因此追加有明确原因的轻量 q64 候选：保持原 INT8 GEMM 算法，省去启动计时搜索。**同一模式两者都通过时，默认选启动更快的轻量版**，不以跨轮绝对吞吐推断谁更快。以下是实际交付选择；启动器按每个模型/模式选择对应环境。

## 默认交付的两模式实测

| 模型 | 工作负载 | 选择 | 原 INT8 策略 → 候选 | 收益 | 裁决 |
|---|---|---|---:|---:|---|
| B11 | 单机 8 线程 NN rows/s | 轻量 q64 | 815.38 → 839.30 | +2.93% | 通过 |
| B11 | Worker C1 RPC/s | 轻量 q64 | 375.47 → 381.39 | +1.58% | 通过 |
| B11 | Worker C32 RPC/s | 轻量 q64 | 992.34 → 1035.43 | +4.34% | 通过 |
| B15 | 单机 8 线程 NN rows/s | 轻量 q64 | 586.30 → 600.03 | +2.34% | 通过 |
| B15 | Worker C1 RPC/s | 轻量 q64 | 295.04 → 300.73 | +1.93% | 通过 |
| B15 | Worker C32 RPC/s | 轻量 q64 | 707.89 → 727.71 | +2.80% | 通过 |

单机实际搜索同时提高：B11 982.46→1006.19 visits/s（+2.42%）；B15 712.67→729.85 visits/s（+2.41%）。单机采纳要求 NN rows/s 和 visits/s 都达到 1% 收益、两臂重复波动均 ≤5%；Worker 对无缓存 RPC/s 应用同一性能门。

比较条件为同一新二进制、相同模型、FFN INT8、宽度阈值 0、物理 batch 上限 8、单 NN 服务线程、Graph。原策略为 q128+首个 cuBLASLt 算法；轻量版只改 q64；完整调优版再按 M/N/K 计时选整数算法。**基线不是历史最优 FP16 认证计划，未证明超过它；不累加历史百分比。**

RTX 5070 Ti / Windows WDDM / 驱动 610.88 / CUDA 13.3.73 / cuBLASLt 130600。每种模型/候选/负载顺序 ABBA。Worker 使用真实本机 gRPC，无缓存、ownership 关闭，每臂预热 256、测量 4096 请求，核计数和 Drain；C 是在途数，C1 物理 batch=1，C32 平均 batch 接近 8。单机每臂 8 个局面×1024 visits、8 线程，报告实际 NN 数与平均 batch。性能测试无本轮编译、其他 GPU 实验或 profiler 重叠；桌面仍运行，未锁频。

## 对 FP16 的误差与校准

| 模型 / 所用候选 | 窗口 | 最大胜率偏差 / 百分点 | 平均偏差 / 百分点 | Policy 首选一致 | 最大目差偏差 / 目 |
|---|---|---:|---:|---:|---:|
| B11 / light | W1 | 1.8496 | 0.2993 | 118/128 | 0.1258 |
| B11 / light | W32 | 1.8035 | 0.2832 | 118/128 | 0.1532 |
| B15 / light | W1 | 1.5947 | 0.1323 | 125/128 | 0.0571 |
| B15 / light | W32 | 0.9848 | 0.1213 | 125/128 | 0.0583 |

候选**未经额外校准即满足本次 6 个百分点门**，没有应用此前失败的 SGF 输出仿射校准，不能称为校准恢复。每窗口是 16 个语义局面×8 对称，共128请求，不是128盘棋，也不构成所有未来局面的误差保证。原始分歧与 Policy KL 保留。FP16 对照仍通过 C++ FP32 原门；INT8 仍未通过原严格门。没有测等时对局或 Elo，网络胜率输出误差不是实际对局胜率损失。

两模式共用 NnEvaluator/CUDA，数值由真实 Worker 请求采集，单机另测真实 MCTS 搜索。若继续降低精度，校准/留出集应按整盘分离，保留恒等校准基线。

## 完整调优与轻量配置的独立对照

各行都是对原 INT8 策略的配对 ABBA；下列不构成 full/light 两者的直接 ABBA，也不据此精确推断两者差距。

| 模型 | 候选 | 单机 NN 吞吐收益 | Worker C1 | Worker C32 | 初始化预热 |
|---|---|---:|---:|---:|---:|
| B11 | full | +3.10% | +2.26% | +3.66% | 4.939–4.974 秒 |
| B11 | light | +2.93% | +1.58% | +4.34% | 0.318–0.344 秒 |
| B15 | full | +2.30% | +2.06% | +2.66% | 88.225–88.344 秒 |
| B15 | light | +2.34% | +1.93% | +2.80% | 0.451–0.459 秒 |

`light` 是 `KATAGO_CUDA_INT8_GEMM_TUNE=0` + `KATAGO_CUDA_ATTN_TILE=q64`；`full` 把前者设为 `1`。GEMM 调优默认关闭。轻量数值/性能复现使用下方命令，测完整调优时将 `--candidate-gemm-tune 0` 改成 `1` 并用全新输出目录。

## 实现和启动代价

- `KATAGO_CUDA_INT8_GEMM_TUNE=1` 显式开启，默认关闭；所有读取经 `tactic_var`，非法值报错。每个 workspace/stream/shape 缓存选择，不序列化供应商 opaque descriptor。
- 单独查询原生产算法与 16 候选池。实际量化输入和权重复制到私有调优流，候选直接执行及 Graph 重放均在 poison 填充后与原生产 INT32 完整输出逐位核对。运行时此参考不是 CPU；独立 CPU oracle 来自专项组件测试。
- 每候选每图 128 次 GEMM，3 轮 ABBA，局部中位耗时至少下降 3%、两臂波动至多 5%、每轮更快才采用；不满足则保留原算法。最后正常执行实际 GEMM，不使用调优残留输出。
- 显式开关 0/1 均在接收工作前预热 B1..B8，公平排除首次中间 batch 初始化成本；未设置开关保持原轻量预热。捕获前先上传真实输入，避免预热读取新分配的未初始化缓冲。
- q64 是现有 FA2 分块，适配本项目 D32/C384/C512。保留 FP32 MMA、精确激活及 half 边界，但其归约顺序与 q128 不同；按本轮独立误差门验证。INT8 权重和激活格式仍是 `w8a8-row-out-rne-v1`。

仅完整调优版的初始化预热：B11 两次约 **4.9–5.0 秒**；B15 约 **88.2–88.3 秒**。每次启动重新选择，进程内复用，尚无跨进程调优缓存。适合持续运行的引擎/Worker；不要把稳态吞吐收益理解为启动更快。

## 使用

独立二进制：`target/int8-gemm-study/build/release/katago-rs.exe`，SHA-256：`977fb4571471a989745c73bc31f8f8fe54666dc0d15231e02bf8f4c2cf1f42e4`。原正式二进制、模型和 FP16 plan 未替换；未提交或推送。

启动器核对模型/二进制/配置 SHA、所选候选的 W1/W32 精度门和对应模式的性能采纳结果，固定本次 B8/S1/全 FFN/8 搜索线程配置；失败模式拒绝启动。环境变量只用于子进程，退出后恢复调用者的 `KATAGO_*`。它读取实验记录，不安装 FP16 认证计划。

```powershell
# 单机 GTP；需要 JSON analysis 时将 -Mode 改为 analysis。
./scripts/run_int8_tuned.ps1 -Model b11 -Mode gtp
./scripts/run_int8_tuned.ps1 -Model b15 -Mode gtp

# Worker 连接 Server 实际的 --grpc 地址；本机 B15 Server 已监听 50051。
./scripts/run_int8_tuned.ps1 -Model b15 -Mode worker -Server 127.0.0.1:50051 -Capacity 32
```

启动脚本只启动 Worker，不会启动 Server。`-Server` 必须与服务端实际的 `--grpc` 监听地址对应，且服务端模型 SHA 必须匹配。2026-09-23 排查确认本机 `go-server.exe` 使用 `--grpc 0.0.0.0:50051`，绑定上述 B15 模型；本机 Worker 使用 `127.0.0.1:50051` 连接。此前示例中的 50052 没有对应监听服务，不能直接照用；只有另行启动监听 50052 的 Server 时才填该端口。

若预热完成后出现 `tcp connect error ... os error 10061`，表示目标 TCP 连接被拒绝，应检查 Server 是否运行及其监听地址。INT8、q64 或 Graph 的启动标记不代表 Worker 已连接成功。脚本不会自动改端口，现有断线重连行为保留。

仍使用原始模型 SHA，但精度策略不同；应使用独立 Worker 池/会话，切换时清空旧评估缓存，不与 FP16 或旧注意力策略在同一搜索树中混用。不要传旧 `cudaTacticPlan`。其他 batch、lane、搜索线程数和并发未获本轮性能认证。

## 验证和复现

- `int8_kernels` 4 项通过；新 `int8_gemm_tuning` 通过 CPU INT32 oracle、剪枝尾部、换输入/权重、流隔离、动态 Graph、capture 冷形状拒绝，以及新 shape 调优后旧 Graph 回放。
- 仅切换 GEMM 调优时，B11 全 FFN、B15 全 FFN、B15 阈值 384，各 B1/B8，五个原始输出共 **68,958 个 FP32 值**与关闭模式及上轮发布构建逐位一致。此证据不包含 q64，不混称整个组合逐位一致。
- 本轮两模型 × FP16/原 INT8/组合候选 × W1/W32，每种候选共 **1,536 请求**，两种合计 3,072 请求（重复同一组 16 个语义局面，不增加独立样本数）；真实路径、精度身份、有限值、计数和 Drain 检查完成。
- 启动器真实 GTP 落子通过：B11 保持 stdin 打开逐条请求/响应，B15 PowerShell 管道输入；q64/关闭 GEMM 调优路径命中，退出后调用者环境恢复，启动提示只写 stderr。

```powershell
cargo build -p katago --features cuda --release --target-dir target/int8-gemm-study/build
$env:KATAGO_CUDA_INT8_GEMM_TUNE = '1'
cargo test -p kata_nn --features cuda --release --target-dir target/int8-gemm-study/build --test int8_gemm_tuning -- --ignored --nocapture --test-threads=1
Remove-Item Env:KATAGO_CUDA_INT8_GEMM_TUNE

. ./scripts/tuning_python.ps1
$studyPython = Resolve-RustGoTunePython
& $studyPython scripts/validate_int8_gemm_tuning.py --stage accuracy --binary target/int8-gemm-study/build/release/katago-rs.exe --model models/b11c768h12nbt3tflrs-fson-silu.bin.gz --candidate-gemm-tune 0 --candidate-attention-tile q64 --output target/my-int8-accuracy
& $studyPython scripts/validate_int8_gemm_tuning.py --stage worker --binary target/int8-gemm-study/build/release/katago-rs.exe --model models/b11c768h12nbt3tflrs-fson-silu.bin.gz --accuracy-report target/my-int8-accuracy/report.json --output target/my-int8-worker
& $studyPython scripts/validate_int8_gemm_tuning.py --stage local --binary target/int8-gemm-study/build/release/katago-rs.exe --model models/b11c768h12nbt3tflrs-fson-silu.bin.gz --accuracy-report target/my-int8-accuracy/report.json --threads 8 --positions 8 --visits 1024 --output target/my-int8-local
```

B15 换模型，并给 accuracy 加 `--reference autotune-output/references/b15-ffn-pruned-a8-3f216ee8-fp32.json.gz`。输出目录必须全新。重建后二进制 SHA 会变化，原启动器不会把新构建当作已验收版本，需要重新生成验证记录。

可携带摘要：[int8-gemm-evidence-20260923.json](int8-gemm-evidence-20260923.json)。完整日志、原始协议输出与每臂时序在 `target/int8-gemm-study/`，应与摘要一起保留。联网方案和一手来源见 [低精度优化新方案调研](RustGo低精度优化新方案调研.md)：更大幅度的后续路线是 MXFP8、按层敏感度与实际延迟联合选择精度；NVFP4/2:4/SageAttention 不可直接把其他架构的收益套到本模型。
