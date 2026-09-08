# TF3 Worker 推理性能审计（2026-09-08）

前半部分保留本轮 Fork 对标优化开始前，对官方 C++ v1.18.2 Worker 的既有成果。
本轮 G0–G4 证据在文末另行追加；Fork GPU event 比例不能替换前面的 Worker 比例。

## 比较对象与测量边界

这次比较的是 Go Server 实际使用的 C++ CUDA Worker 与 Rust 原生 CUDA Worker，
不是两个程序各自的搜索速度。审计开始时 Rust checkout 为 `0a10279`；
保留的原始 release 二进制 SHA-256 为
`73d7224198aaaf0e2e487498562a5cfe73ba2a3ed23f5344152c7049b8bd3171`，
其内嵌构建信息仍为此前编译时的 `deaf266+dirty`，以二进制 hash 标识本次基线。
本机 C++ 为 KataGo v1.18.2，revision
`231e1c4b938f068628a5e3e59a3e842ad5fc92cd-dirty-cuda`。
其本地差异是构建/命令入口和 nnworker 接入，所审阅的 CUDA 数值实现来自该基座。

- GPU：同一张 RTX 5070 Ti，70 SM，50,331,648 bytes L2。
- C++：`D:/Go/Server/worker/build-windows-CUDA/Release/katago.exe`。
- C++ 可执行文件 SHA-256：`e7dbd8b8af6341932001b5ce0ce149ea9d63e61a69a9a184ee446c903bb72d86`。
- 模型：`D:/Go/Server/models/kata1-tf3-b11c768-s11001M-d5973M.bin.gz`。
- 模型 SHA-256：`1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6`。
- 19 路，modelVersion 17，11 个 nested block，每个有 3 对 Attention/FFN；
  trunk 768、bottleneck 384、FFN 1152、12 heads × 32。
- 生产速度比较使用 C++ `cuda-5070ti.cfg` 的 FP16 路径。正确性比较另用
  `configs/worker_cpp_fp32.cfg`，不混用两者作为同一个性能基准。

`scripts/benchmark_workers.py` 使用预先生成的同一组 128 个请求模板，
固定物理 batch 上限 16，以持续补充的 gRPC 在途窗口供数。
预热、模型加载、最后的统计心跳、Drain 和 JSON 写盘均不计时。
uncached 模式显式 `skip_cache=true`，并断言实际 NN rows 增量等于请求数；
cached 模式另外报告 NN rows，不能把缓存命中算成神经网络吞吐。
各进程顺序运行，默认 C++/Rust/Rust/C++ ABBA，不同时占用 GPU。
报告保存模型/配置/二进制/请求摘要、RTT、Worker 分段耗时和真实 batch。
计时边界之外等待两个连续、空闲、计数一致的心跳，避免结果已返回而统计尚未更新。

最初 C32 ABBA：C++ **1325.6 / 1325.2 次/秒**，Rust **882.4 / 882.4 次/秒**，
Rust/C++ = **66.58%**。四轮均实际计算 4096 行，平均 batch 15.88–15.94。
计数、并发、模型、热身及客户端供数均不足以解释差距。
证据：`target/tf3-perf-audit/worker-baseline/report.json`。

最终单线程认证配置的同口径 ABBA（每轮 4096 个 uncached 请求，表中为两轮
吞吐的几何平均）：

| 在途请求上限 | C++ NN rows/s | Rust NN rows/s | Rust / C++ | Rust 实际平均 batch |
|---|---:|---:|---:|---:|
| 1 | 236.4 | 379.3 | 160.5% | 1.00 |
| 8 | 852.3 | 808.5 | 94.9% | 4.00 |
| 16 | 1151.1 | 931.9 | 81.0% | 7.98 |
| 32 | 1332.2 | 1078.2 | 80.9% | 15.94 |
| 64 | 1330.6 | 1076.6 | 80.9% | 15.94 |
| 128 | 1331.1 | 1073.5 | 80.6% | 15.94 |

默认 C32 相对原始 882.4 提高约 **22.2%**；单请求已更快，饱和吞吐仍低约
19%。因此不能笼统说所有负载都已追平，也不能把更大的 capacity 当作更大
物理 batch：C64/C128 仍是 B16，吞吐基本持平而排队延迟上升。
证据：`target/tf3-perf-audit/worker-final-sweep/report.json`。

另做 C++ / 原 Rust / 吞吐配置 / 吞吐配置 / 原 Rust / C++ 的对称顺序，
各轮同为 B16、4096 个 uncached 请求：

| 在途请求上限 | C++ | 原 Rust | 最终双线程 Rust | 对原 Rust 提升 | Rust / C++ |
|---|---:|---:|---:|---:|---:|
| 64 | 1338.1 | 883.3 | **1163.8** | **31.75%** | **86.97%** |
| 128 | 1334.5 | 882.6 | 1100.0 | 24.63% | 82.43% |

C128 两轮 Rust 分别为 1038.9 / 1164.7，波动明显；C64 为 1166.2 / 1161.3。
因此推荐吞吐配置配 **capacity 64**，没有把 128 当作默认升级，也没有只摘其快轮。
本次满载仍低于 C++ 约 **13%**，尚未完全追平。
证据：`target/tf3-perf-audit/worker-throughput-final/report.json`。

## 已确认的原因

### 1. 原 TF3 配置漏用了已有 Dual-FFN 优化

`worker_cuda.cfg` 使用默认 tactic。已有 `best-tactic-plan.json` 绑定另一个 ONNX
模型，不能直接套在 TF3 上，因此 TF3 实际走了未融合的 FFN 投影。
独立启用 `KATAGO_CUDA_DUALFFN=1`、保持 q128 后，eval B16/W32 ABBA 为
883.1 → 1038.4 → 1040.4 → 882.8 行/秒，约 **+17.7%**。
日志确认 CUTLASS 编译、运行探针和有效路径均成功，确实执行 DualGemm + SwiGLU。
TF3 128 个局面/对称请求对 C++ FP32 的所有误差门及 top-1 128/128 均通过；
旧 ONNX 16 局面对 ORT FP32 的整图门也通过。

### 2. 缓存键在推理输出处漏写

后端生成张量，不知道局面 hash。原同步/流水线完成路径直接缓存后端输出，
没有把请求的 NN hash 附回去，真实推理的结果因此写入零 hash 槽。
这影响重复局面的缓存命中，不能归因于 GPU 算子慢。

修复集中在 evaluator 的结果发布前：补齐请求 hash；若缓存已有 policy/value，
本次仅为补 ownership，则保留旧 policy/value 和标量，只补 owner map，
与 C++ 的语义一致。失败不污染已有缓存，`skip_cache` 仍跳过读取但允许更新缓存。
同步及流水线均有实际请求回归覆盖，包含语义参数键分离和失败恢复。
128 个模板来自 16 个上下文 × 8 个对称；NN cache 与 C++ 一样不把 symmetry
作为独立缓存键，因此 cached 模式预热后的主要工作是 16 个键的重放、查表和传输。

同样 C32、8192 个重复请求的对称复测：

| 实现 | RPC 请求/秒 | 计时区间实际 NN rows |
|---|---:|---:|
| C++ | 8373.3 | 0 |
| 原 Rust | 884.2 | 8192 |
| 最终 Rust | 8784.1 | 0 |

修复后重复请求处理约为原版 **9.93×**，比 C++ 高约 4.9%；这是缓存与请求层
吞吐，不是 NN 算力增长。当前 Server 的 `crates/go-server/src/worker.rs`
确实发送 `skip_cache: false`，所以修复适用于实际接入，但实际收益仍取决于重复率。
证据：`target/tf3-perf-audit/worker-cache-final/report.json`。

### 3. Attention 和 FFN 解释了主要 GPU 差距

在已启用 DualFFN、仍使用 q128 的 Rust 上，对同一 TF3/B16 做 Nsight Systems
CUDA 跟踪。截取最后 2 秒，以实际 attention 调用数 / 33 归一化，并用窗口内
完整前向复核。不能将 C++ 全 trace 除以 305：其初始化还预热了 batch 1..16，
实际包含 321 次前向；Rust 为 305 次。

| GPU 阶段 | C++ ms/前向 | Rust ms/前向 | Rust 多用时间 |
|---|---:|---:|---:|
| Attention：QKV、RoPE、core、out | 4.095 | 6.169 | +2.074 |
| FFN：gate/up、SwiGLU、down | 4.771 | 6.341 | +1.570 |
| Bottleneck 两次投影 | 0.681 | 1.158 | +0.477 |
| Trunk gate/SiLU | 0.338 | 0.615 | +0.277 |
| Stem / 输入 | 0.100 | 0.242 | +0.142 |
| 额外 FP32→half 转换 | 0 | 0.137 | +0.137 |
| Policy/value/ownership heads | 0.125 | 0.245 | +0.120 |
| RMS | 0.660 | 0.374 | −0.286 |
| **GPU kernel 时间合计** | **10.770** | **15.280** | **+4.510** |

Attention + FFN 解释约 **80.8%** 的 GPU 时间差。Rust RMS 更快，拷贝只有约
0.015 ms/前向，C++ 约 0.021 ms，均不是主要瓶颈。每前向 kernel 数是
C++ 378、Rust 302，差距也不是 Rust 发射了更多 kernel。
此处是 direct 模式的 GPU 活动时间，不与 gRPC 吞吐直接相除。

C++ attention core 约 48.27 µs，加独立 RoPE 约 11.19 µs；Rust q128 融合
RoPE 约 93.06 µs。两者都是 FP32 QK/PV 累加，但 tile/寄存器/shared memory
组织、exp 实现和归约顺序不同。C++ 使用近似 exp2 指令；Rust 保持精确 expf。
单靠 trace 不能把全部差距归因于 occupancy 或 exp，也不能声称换一个指令就能补齐。

### 4. C++ 的部分 GEMM 速度来自不同计算/存储路径

C++ 使用 `cublasHgemm` / `cublasHgemmStridedBatched`，half 输出/残差；
当前 Windows 构建没有启用 CUTLASS fused FFN，gate/up 后另跑 SwiGLU。
Rust 的 GEMM 使用 `CUBLAS_COMPUTE_32F` 并保留 FP32 residual。
不能只看 kernel 名字就断定硬件内部如何累加。

`probe_tf3_accumulation` 在相同 GPU、相同 half 输入、相同形状下独立改变
NN/TN 布局及 compute/output 类型；CUDA event 计时不含分配/传输，
50 次预热、200 次计时、正反顺序两轮，另用 CPU f64 抽样检查误差。

| 矩阵（M=5776） | C++ NN，FP32 compute / FP32 out | Rust TN，同精度 | 布局差 |
|---|---:|---:|---:|
| out，N384/K384 | 23.47 µs | 26.78 µs | NN −12.4% |
| QKV，N1152/K384 | 62.59 µs | 73.93 µs | NN −15.3% |
| packed gate/up，N2304/K384 | 121.40 µs | 142.78 µs | NN −15.0% |
| FFN down，N384/K1152 | 62.71 µs | 63.37 µs | NN −1.0% |

同一 FFN down/NN 布局改用 FP32 compute / half out 是 59.82 µs，
FP16 compute / half out 是 35.51 µs；后者抽样最大误差 0.0009792，
前者 0.0001223。受控测试表明这里主要是计算模式，布局只解释很小部分。
这些是 classic cuBLAS、beta=0 的诊断结果，不能直接当成生产 Lt/residual 的收益。
本次不改为 FP16 累加、不启用 fast_math，也不放宽数值门。

## 未采纳的候选

- 原 q64 和 DualFFN+q64 均通过标量/数组误差门，但 TF3 top-1 只有
  126/128；两个失败是同一局面的不同对称、参考概率差约 0.000555。
  接近并列仍按失败处理，不能直接复用旧 ONNX 的 q64 认证。
- 复用 queue placeholder 减少空轮询分配：独立 ABBA C32 只有 +0.14%，
  C1 +0.72%，未过 1% 采纳线，代码已回退。缓存语义修复保留。
- q128 的 P-register / Q-hoist 已在此前同形状实验分别慢 12% / 3%，
  见 `cuda-optimization-plan.md` 的 B3 证伪，本次不重复实现。

## 新增的 q64-serial 与两种运行配置

普通 q64 把 softmax 的 8 组部分和改成了树形归约，原 q128 则逐项相加。
虽然都是 FP32，这个变化足以在后续 half 边界累计成最佳落点翻转。
新 `q64-serial` 保留 q64 的四 warp、寄存器片段组织，恢复 q128 的逐项和顺序；
QK/PV 累加、RoPE、精确 expf 和 P/output 的 half 边界保持。
编译后的 normalizer 更新与 q128 都是同一 FMA 收缩行为。

- 3 个随机种子/幅度 × batch 1/2/4/8/16，共 15 组、12,892,032 个 half 输出，
  与 q128 **逐位一致**；旧 q64 在同一诊断中确实有差异。
- TF3 128 请求对 C++ FP32 全字段通过、top-1 **128/128**。
- 单独 attention ABBA，C32/8192：q128 1036.9 / 1041.4，
  q64-serial 1075.1 / 1076.5 次/秒，**+3.53%**。
- 原 q128 和原 q64 的 PTX SHA 与修改前完全相同。三个 Windows ONNX plan
  经各自原 tactic 的 ORT FP32 16/16 整图重验后更新构建指纹；WSL 历史 plan
  没有伪造本机认证，新的 WSL 构建仍需在对应环境重新验证。

新内核有独立 `attention_q64_serial` 编译能力，缺失时认证 plan 明确拒绝。
环境诊断开关在能力缺失时打印 q128 回退标记；不能用普通 q64 的能力冒充 serial。
验证脚本按完整 token 检查标记，且数值进程本身也必须执行选中的 attention；
plan 的优先级覆盖同样落实到数值 dump 环境。

`worker_tf3_sm120.cfg` 为默认配置：一个 NN 服务线程、batch 16、capacity 32，
使用 `plans/worker-tf3-sm120.json`。高并发可选
`worker_tf3_sm120_throughput.cfg` / `plans/worker-tf3-sm120-throughput.json`：
两个 NN 服务线程、关闭 Graph、capacity 至少 64。
C64/8192 的独立 ABBA 是单线程 1074.9 / 1075.5，双线程 1157.7 / 1160.4，
**+7.80%**，平均 batch 仍接近 16。
此收益限定持续高并发 Worker，不推广到 GTP 搜索或低并发场景。

## 已完成的验收与剩余差距

- 最终两种认证配置各自 TF3 全字段对拍 128/128；默认配置还验证 C1、
  plan 覆盖相反环境开关，以及 optimism=1 / temperature=1.3 的 128 请求。
- ONNX 的 q128、原 q64、新 q64-serial 各自 ORT FP32 16/16 整图通过；
  CUDA Graph warmup/首次发射回归通过。
- 默认无 CUDA 的 NN/Worker/CLI 测试：279 passed、0 failed、4 项既有 ignored；
  CUDA feature 的 NN 单元测试 130/130；两组 Python 回归共 16/16，
  8 个脚本语法检查通过；最终 release 构建通过。
- 26 个 tactic 路径/拒绝用例完成验收，17 个带 4 局面 ORT 数值检查。
  首轮 25 个通过，补批用例因旧基准的隐含超额容量假设失败；修正为明确的
  `--batch 3,4 --workers 3` 后实际出现 `launch=on phys=4` 并单独通过。
  没有放宽引擎容量限制。缺少 serial 内核的运行时分支在本完整构建上未执行，
  该拒绝条件已有 CPU 单测；日志明确记录此项，不把它当作实机覆盖。
- 真实 Go Server + C++/Rust 混合池，在 capacity64 吞吐配置下完成注册、两次
  genmove、undo、双 pass 终局和 Drain；双方零 failures，Drain 后池为空。
  此用例只证明接入与收尾，完成任务比例不作为速度排名。

剩余优化应继续围绕 attention core/RoPE 及大 GEMM。受控 probe 表明 K384 的
NN 权重布局值得在真实 Lt/residual 路径尝试，但尚未证明整网收益；K1152 的
FFN down 布局几乎无收益，其计算模式差异不能用转置解决。精确 expf、FP32
累加/残差与 C++ 快路径的差异也不能通过关闭正确性门掩盖。
目前保留了可复现且通过验收的改动，未把这些未验证方向写进生产 plan。

## 复现工具

```powershell
# 必须使用有 grpcio / grpcio-tools 的 Python；本机如下。
$py = 'D:/Go/Server/worker/.venv-windows/Scripts/python.exe'
& $py scripts/benchmark_workers.py --concurrency 32 --requests 4096 `
  --output target/tf3-perf-audit/reproduce

# 另测重复局面，务必查看 NN rows，不能作为 GPU 吞吐。
& $py scripts/benchmark_workers.py --workload cached --concurrency 32 `
  --requests 16384 --output target/tf3-perf-audit/reproduce-cache

# 精度/布局诊断，不修改生产推理。
$env:KATAGO_RUN_ACCUM_PROBE = '1'
cargo test -p kata_nn --test probe_tf3_accumulation --features cuda --release -- --nocapture
Remove-Item Env:KATAGO_RUN_ACCUM_PROBE

# 离线分析已有 Nsight SQLite；此分析器严格检查本次架构和 launch sequence。
& $py scripts/analyze_worker_kernels.py `
  --cpp-sqlite target/tf3-perf-audit/cpp-direct-trace.sqlite `
  --rust-sqlite target/tf3-perf-audit/rust-direct-dual-trace.sqlite `
  --output target/tf3-perf-audit/kernel-window-analysis.json
```

`nnbench --mode direct/kernel` 现在可直接加载原生 TF3；eval 模式将真实物理
batch 上限和请求并发分开，预热完成后再清统计，并按实际 rows/batches 计速。
direct/kernel 是底层诊断，当前不安装配置中的 `cudaTacticPlan`；如需对 tactic
做这两种测量，必须显式设置环境并检查 `[cuda-tactic]` 标记。
生产 Worker 的 plan 则在创建 compute context 时校验并安装。

## 本轮 Fork 对标追加（2026-09-08，G0–G4 阶段快照）

本阶段沿用同一张 RTX 5070 Ti 和上述 TF3 模型 SHA，比较对象另设为本机 WSL
KataGomo Fork 的认证 B14/S2 前向。此前 **1078.2 / 1163.8** Worker rows/s、
DualFFN、q64-serial 和缓存修复仍是既有成果，不计作本阶段的新收益。
以下记录本阶段已完成的测量；G1 已限定在独立持续 C32 配置验收，G2 尚未验收。

### G0：稳定空盘前向参考已建立

`target/fork-parity-20260908/fork-rust-wsl-empty-abba/report.json` 状态为
`STABLE`。固定物理 B14、两 lane、80 次预热、每 lane 1000 次计时、无 graph，
顺序为 Fork/Rust/Rust/Fork；所有样本均参与汇总。

| 实现 | 第一次 event NN rows/s | 第二次 event NN rows/s | 几何平均 | 重复轮次 spread |
|---|---:|---:|---:|---:|
| Fork 认证 B14/S2 | 2231.37 | 2357.02 | 2293.33 | 5.63% |
| Rust TN + DualFFN + q64-serial | 1125.07 | 1128.85 | 1126.96 | 0.34% |

spread 使用 `max/min−1`，预先固定最大门为 **10%**，不是从样本中挑选最快轮。
Rust/Fork 的 **49.14%** 仅表示这两个执行路径的 lane event 吞吐比，**不能作为
Worker、evaluator、搜索或端到端速度之比，也不能当作同精度实现的效率比**。

每次 forward 的 CUDA event 成对包围 `apply`，所有迭代入队后统一同步；每 lane
先取耗时中位数，再计算 `sum(14 / lane_median_seconds)`。Rust
`--timing cuda-event` 保留并复算全部 lane 样本；Fork JSON 只提供每 lane
中位数/吞吐与迭代数，向量长度检查由已核查的源码保证。预热后两 lane 使用屏障，
Fork 显式 `phase-offset-us=0`。H2D/D2H、特征生成、输出后处理和模型加载不计入
event 区间；Fork 的另一个 `actualWallSeconds` 还包含预热等阶段，Rust 的 wall
不含这些阶段，因此报告不计算两者 wall 比例。

两边使用 19 路空盘、黑先、TrompTaylorish、komi 7.5、encore 0、symmetry 0、
policy optimism 0，以及等价的默认 MiscNNInputParams；每个 batch 行重复同一
特征。这是**同语义空盘**，尚未验证跨实现特征逐字节一致。Rust 记录输入描述及
NCHW spatial/global 的 little-endian f32 SHA，并校验重复轮次一致。

报告锁定实际模型/可执行文件/plan 路径和 SHA，复核 warmup、batch、lane、
tactic 生效标记及 median 吞吐。此轮身份为：

- Fork：`/root/katagomo-fullflow/.final-migration-env/results/sm120-top3-from-4-32-s2-gpu0/build/katago`；
  二进制 SHA `5df63305086e42f67e3171874976bd8c7d441c36e8603e899d051419c6c00868`。
- Fork 同目录 `best-tactic-plan.json` 的文件 SHA：
  `b5b4701b58f94b5878e38f40062435d22dad60de07ae374d1dee810829e3162b`。
- Rust 此轮 WSL 二进制 SHA：
  `12b9ced69c4ad30f49d745b5c5bd6f7fc7607dfc96b01176954dbc41d99e7236`。
  后续重建不能沿用这个 hash 冒充本轮二进制。

还存在两项不可省略的条件差异：Fork 是 CUDA **13.0.88 / cuDNN 9.24** 认证
构建，Rust 此轮是 CUDA **13.3.73**，不声称运行库相同；Fork qk16 使用 QK
FP16 累加，且部分残差 GEMM 也采用 FP16 累加，Rust 保留 FP32 累加/归约、
精确 expf 及既有存储边界。Fork 历史认证值 2418.95 和其数值门不替代本轮实测
或 Rust 数值门。

首次 `fork-rust-wsl-baseline-abba/report.json` 的 Rust 输入为不同局面，且 Fork
两轮为 1749.64 / 2398.04，spread **37.06%**。该轮完整保留为 **diagnostic**；
其中约 **55.6%** 的比例不得引用为有效基线，也不能摘取快轮替换本次稳定结果。

复现工具如下；它会重新检查当前二进制，产出新一轮证据，不保证重建后数值等于上表：

```powershell
.venv/Scripts/python.exe -B scripts/benchmark_fork_parity.py `
  --rust-platform wsl --input empty --layout tn `
  --iterations 1000 --warmup 80 `
  --output target/fork-parity-20260908/fork-rust-wsl-empty-reproduce
```

### 输出头工作量审计

双方都计算 policy、value/score 和 ownership。Fork `NNResultBuf.includeOwnerMap`
默认 false，benchmark 的 `whiteOwnerMap=NULL` 只跳过计时外的 CPU 输出复制；
计时路径仍无条件执行 ownership 卷积。policy optimism 0 同样只影响计时外的
CPU base/optimistic 混合，未令 GPU 省掉 optimistic 通道。

| 计算 | Fork 认证路径 | Rust 原生 TF3 经当前执行器 |
|---|---|---|
| Policy 和 pass 输出 | 模型原生 2 通道，均计算 | 固定 6 通道；native 0/1 映射到 0/5，其余 4 通道为零，但仍计算 |
| Value / score 终端 | 3 value + 6 score，共 9 通道 | 3 value + 10 misc + 8 moremisc，共 21 通道；其中 12 通道为零，但仍计算 |
| Ownership | 无条件卷积；FP16 路径另做 half→float | 无条件 ownership GEMM |
| 头部首层投影 | partial C288 合并 G1 96 + V1 192，P1 96 单独计算 | 合并 P1 96 + G1 96，V1 192 单独计算 |

因此 `partial-c288-g1-v1` 是投影组织优化，**没有裁掉 P1 或 ownership**。
Fork 认证 plan 的 `cudaUseFusedValueTerminalSm120=false` 只选择 value/score
分开执行，仍计算两者；`cudaUseHeadBNHalfToFloat=true` 也是数据路径选择。
现有 benchmark CLI 没有能消除 2/6 policy 和 9/21 terminal 形状差异的开关，
本阶段未修改 Fork 认证 plan 来追求表面一致。Rust 的额外零通道约每盘
141,312 MAC；这是数学工作量，不能按其比例估算或扣减实测延迟。

代码证据：Fork `cpp/neuralnet/nneval.cpp:1160`、`:1172`，
`cudabackend.cpp:3351`、`:3706`、`:5381`、`:5532`，
`cudabackend_sm120.cpp:803`；Rust `crates/kata_nn/src/native_model.rs:235`、
`:267`、`:392`，`backends/cuda_exec.rs:1065`、`:1100`，
`cuda-kernels/executor.cu:529`。本机 Windows Fork 与 WSL 对应的
`cudabackend.cpp`、`nneval.cpp`、`benchmarknn.cpp` 在统一换行后的 SHA 相同，
这些源码边界与实际执行环境相符。

### G1：B8 阈值未采纳，B16 仅用于独立持续 C32 配置

`KATAGO_CUDA_GEMM_LAYOUT=nn_k384_b8` 在加载期同时准备所需权重布局，
运行时只在 K384 且实际 batch 至少 8 时选 NN，小 batch 保留 TN；不改变
FP32 累加和既有输出/残差边界。这里的 B8 是**启用阈值**，Worker 物理 batch
上限仍是 16，不能将候选名称理解为固定 B8。

修复 plan 生命周期之后的干净 ABBA：
`target/fork-parity-20260908/worker-layout-b8-abba/report.json`。
下表为每个并发档两轮 uncached RPC 吞吐的几何平均；报告同时验证实际 NN rows，
这里不是 Fork 比较，也不是 GPU event 计时。

| 在途请求 C | 原 Rust TN | B8 阈值候选 | 相对变化 | 实际平均 batch / 路径 |
|---|---:|---:|---:|---|
| 1 | 377.07 | 378.36 | +0.34% | 1.00，均为 TN |
| 8 | 802.58 | 798.43 | −0.52% | 4.00，均为 TN |
| 16 | 934.21 | 926.32 | −0.84% | 7.99，候选在 B8 启用 NN |
| 32 | 1073.39 | 1093.90 | +1.91% | 15.97，候选在 B16 启用 NN |

实际 B8 没有稳定收益，故 **B8 阈值未采纳**；随后收窄到
`KATAGO_CUDA_GEMM_LAYOUT=nn_k384_b16`，只在实际 B16 及以上选 NN。
最终单线程 ABBA 每轮 8192 个 uncached 请求，同 binary、同配置，比较 TN
与 B16 阈值布局；另测双线程吞吐配置的 C64。以下仍为 RPC/s 几何平均，实际
NN rows 计数也在报告中验证，不是 G0 的 event 吞吐。

| 范围 | 原 Rust TN | B16 阈值候选 | 相对变化 | 裁决 |
|---|---:|---:|---:|---|
| 单线程 C16 | 930.547 | 923.816 | −0.72% | 均主要为实际 B8/TN，无收益依据，保留旧默认 |
| 单线程 C32 | 1073.479 | 1093.684 | **+1.88%** | 实际平均 batch 15.97，超过 1% 门，独立配置采纳 |
| 双线程 C64 | 1159.036 | 1165.877 | +0.59% | 未过 1% 门，保留原吞吐配置 |

证据：`worker-layout-b16-abba/report.json`、
`worker-layout-b16-throughput-abba/report.json`，均位于
`target/fork-parity-20260908`。C32 baseline 两轮为 1072.742 / 1074.217，
候选为 1093.313 / 1094.056；没有挑快轮。被测 Windows 二进制 SHA 为
`1d0895baba596c817e192d13445dbca0a154429698c716ba02044a85ec0ab271`。
G0 的旧 WSL 二进制 hash 和 49.14% 参考保持原样，不能用这里的新 binary 身份
重写旧基线。先前 12–15% 的受控 GEMM probe 也不等于这里的整链收益。

最终只新增持续 C32 专用的 `configs/worker_tf3_sm120_c32.cfg` 与
`plans/worker-tf3-sm120-c32.json`：一个 NN 服务线程、物理 batch 上限 16、
保留 CUDA Graph，plan 启用 DualFFN、q64-serial 和 `nn_k384_b16`。
原 `worker_tf3_sm120.cfg` / `worker-tf3-sm120.json` 的默认策略，以及
`worker_tf3_sm120_throughput.cfg` / `worker-tf3-sm120-throughput.json`
的双线程策略均不变；不把 C32 的收益推广到低并发、双线程或 GTP 搜索。

持续 C32 部署方式：

```powershell
.\scripts\run_go_worker.ps1 -Server 127.0.0.1:50051 `
  -Config configs/worker_tf3_sm120_c32.cfg -Capacity 32
```

该 profile 已完成下面的最终 plan 对拍与覆盖验证；独立名称保留其负载适用范围。

### 主机函数缓存：对拍通过，性能门未过，已撤回

独立候选曾将 `CudaRuntime::get_func` 的成功查找缓存为 `CudaFunction`，
不改变 kernel、精度、算法或 tactic。C32 与双线程配置各通过 128 局全字段
数值门；C32 的 128 个完整输出与改动前完全一致，graph 回归也通过。
这些正确性证据不代表有性能收益。

首次 `function-cache-forward-abba` 的旧版两轮为 561.86 / 507.49，
相差 10.71%，已标为 `UNSTABLE_DIAGNOSTIC_ONLY`；紧接的
`function-cache-worker-throughput-abba` 旧版为 593.28 / 1134.62，
因此该报告的 +36.38% 比值也只作诊断，不能采纳。脚本正常路径确实等待
子进程结束，日志时间表明两组顺序执行；未发现本轮进程泄漏证据，异常原因
尚未确定，不能把猜测的 GPU 竞争写成已证实原因。初轮 C32 候选也有 4.53%
重复波动，未据此裁决。

确认没有其他引擎进程、GPU 空闲后，以新目录完整重跑，未挑选初轮样本：

| 口径 | 旧版几何平均 | 缓存候选几何平均 | 相对变化 |
|---|---:|---:|---:|
| WSL B14/S2、空盘、CUDA event 中位数求和 | 1108.920 | 1100.384 | −0.770% |
| Windows uncached C64、B16、双线程 Worker RPC/s | 1142.174 | 1133.523 | −0.757% |

证据分别为 `function-cache-forward-abba-r2/report.json` 和
`function-cache-worker-throughput-abba-r2/report.json`，均位于
`target/fork-parity-20260908`。前向两组重复 spread 为 1.98% / 2.34%；
Worker 为 0.22% / 0.60%，均过预先固定的 5% 稳定门，但没有超过 1% 收益门。
故撤回缓存的三处代码改动，保留 G1 布局及 plan 生命周期修复。
拒绝候选的源码和 Windows/WSL 二进制分别保存为
`function-cache-candidate-cuda.rs`、`rust-function-cache-rejected.exe`、
`rust-function-cache-rejected-wsl`，保留复核能力。

回退后 Windows/WSL release 均重新构建成功；
`numeric-function-cache-reverted/comparison.json` 全字段 PASS、top-1 128/128。
测试期间外部新增提交 `eb1093f` 包含当时仍待性能裁决的缓存候选，本轮未改写
提交历史，只在当前工作区撤回该候选。新 binary 身份受构建版本信息影响，
不把新 hash 回填到旧测量记录。

`scripts/benchmark_workers.py` 已增加预先固定的双样本及 5% 稳定门：工作负载
`status=PASS` 与 `performance_status` 分开，异常比值仅保留
`diagnostic_only_*` 字段；原报告禁止覆盖。11 项 CPU 回归覆盖本次 593/1134
异常、单样本、阈值边界与旧证据保护，见 `worker-performance-gate-tests.log`。

这说明静态统计的重复 `cuModuleGetFunction` 调用不能直接推导整网收益。
描述符缓存也暂不据源码数量采纳；下一候选优先测 Fork 分块在 TF3 实际
残差形状上的效果，继续使用 FP32 累加、FP32 残差及输出。

### TF3 残差分块：独立算子有候选，尚未进入生产

新增 `scripts/b4_residual_probe/tf3_residual_bench.cu`，保留旧 ONNX 探针。
新探针覆盖原生 TF3 的 N384、K1152（FFN down）/K384（outproj），
M5054（B14）/M5776（B16）；CUTLASS 输入/权重 half，累加、epilogue、
残差 C=D 均 FP32，alpha=beta=1，直接使用原 TN 权重。Fork 的 half 残差
模板未直接搬入。Lt 基线保持生产 TN、FP32 compute/scale/output、32 MiB
workspace 和 heuristic 首选；描述符及 CUTLASS Params 均在计时前准备。

`tf3-residual-report.json`（同一证据根目录）使用预热 100、迭代 1000、
每 tile 三轮完整 ABBA。4 个形状 × 3 个 tile 均通过全输出对 Lt 与 256 点
FP64 oracle，固定误差门 `5e-5 + 5e-5*abs(reference)`，随后才计时。
三个 tile 中，128×64×32 / warp64×32 / s3 的结果如下：

| 形状 | 生产 Lt 微秒/次 | CUTLASS 微秒/次 | 算子吞吐变化 |
|---|---:|---:|---:|
| FFN down B14 | 60.24 | 60.04 | +0.33% |
| FFN down B16 | 65.88 | 60.30 | +9.26% |
| outproj B14 | 22.77 | 24.79 | −8.16% |
| outproj B16 | 29.02 | 25.09 | +15.67% |

这是 CUDA event 整段耗时除以迭代数的独立算子数据，不是 G0 的逐前向
event 中位数，也不是整网或 Worker 收益；B16 outproj 的 TN 结果不覆盖
G1 C32 profile 的 NN 布局。全部 tile/重复数据保留，未只报告最快轮。

同次单轮 top-8 诊断发现 Lt 的其他候选接近上述 B16 分块速度，且 B14
FFN down 也存在更快候选。因此优先对预先固定的 Lt 候选做完整配对复测，
再决定是否需要新增 host kernel；单轮 top-8 数据没有替换生产 heuristic。
当前未启用新 residual tactic，未改写 kernel 指纹或迁移认证 plan。

固定 Lt 候选的后续证据为 `tf3-residual-lt-abba-report.json`：候选名单在
复测前由前次诊断固定，不重新挑选；预热 100、迭代 1000、三轮 ABBA，
三形状全输出及 FP64 点检查均 PASS，重复稳定。
B14 FFN down 的 filtered index 1 为 60.28→54.34 微秒（+10.92%），
B16 FFN down 的 index 3 为 66.08→61.88（+6.78%），
B16 outproj 的 index 2 为 29.08→24.91（+16.73%）。这些仍是 TN、beta=1
独立算子收益，下一步需绑定库版本/算法身份，并做整网与 Worker 验证。

### FP16 舍入缺陷：已确认，生产修复与重新认证待完成

strict RoPE 的新增逐位门揭示 CPU `f32_to_f16_bits` 次正规分支多右移一位，
且未正确执行 ties-to-even。例如 `2^-24` 应为 half `0x0001` 却得到零，
`2^-15` 应为 `0x0200` 却得到 `0x0100`。这不是新的 GPU 算法误差；
同一函数也用于生产 TN/NN/DualFFN 的权重打包。

纯 CPU 按原 TF3 文法与 SHA 解析真实模型至 EOF：265 个将上传为 half 的
源矩阵共 70,361,856 个元素，其中 713,115 个位模式错误（1.0134966%），
涉及 263 个矩阵，最大旧/正确 half 差为 `3.0517578125e-5`；82 个值本应
舍入至最小正规 half。计数未重复包含 NN 副本或 padding。受影响样本由
NumPy FP16 与 Python `struct '<e'` 交叉核验；证据目录为
`target/fork-parity-20260908/fp16-conversion-audit`。

此时只修正了 probe 的独立 CPU 舍入参考，生产转换仍待单独修复，不能
把旧模型数值门直接视为修复后的认证。后续必须同时处理 plan 的全局数值
契约：CUDA kernel hash 不覆盖 Rust 权重编码，且目前可选 host revision
允许部分旧 plan 缺失版本。修复需拒绝未重新验证的旧编码计划，完成 TF3、
ONNX 及相关负载检查后再迁移；不能静默重用旧 PASS 或旧性能结论。

### 剖析计时修复

`KATAGO_CUDA_PROFILE=1` 原先让 Attention 子段与外层共用事件，重录子段
起点会覆盖整层起点，导致层表及其总计低估。现已为层/子段分配独立事件对，
日志延后到采样完成；新增 FFN up/SwiGLU 与 down 子段。非 profiling 及
graph capture 不分配这些事件，算子与精度边界保持原样。

`check_profile_events.py` 对实际三次前向验证：原日志有 98 处层表/子段
不一致；修复后每次 180 层、33 Attention + 33 FFN 子段，全部父层时间
不小于其子段总和（考虑打印舍入）。证据为 `profile-events-before-check.json`
及 `profile-events-after-check.json`；Windows/WSL release 构建及
`numeric-profile-events/comparison.json` 的 128/128 整网数值门通过。
剖析模式含逐段同步和主机提交开销，修复后的约 25 ms 层时间总和也不能
替代无插桩的 G0/Worker 吞吐，或与旧错误总计计算性能变化。

### G2/G4：已完成的检查与尚未完成的验收

- **G2 现成 FA4 参考**：Fork B14/tn96 的 FP32 MMA AOT 已用于独立 opt-in
  probe，并保存来源、许可证、产物 hash 和 ABI。这里的 FP32 不代表符合本仓库
  精确 softmax：现成产物使用近似 exp2，不能凭微基准进入生产。
  证据位于 `target/fork-parity-20260908/fa4-reference` 与 `fa4-probe-full.*.log`。
- **G2 strict 候选**：`scripts/probe/build_fa4_strict.py` 的首轮在 MLIR 解析
  遇 undeclared SSA value，失败记录保留于 `fa4-strict-reference`。
  r2 修复 SSA 作用域，未改变数学运算，源码 AST 审计通过；
  `fa4-strict-reference-r2/manifest.json` 已记录离线导出成功，状态为
  **`EXPORTED_UNVERIFIED`**。后续静态审计确认全部 196 处 exp 的完整依赖图与
  Rust expf 一致，归一化为 rcp.rn，QK/PV 均 FP32；证据为 r2 的
  `codegen-audit.md`。Windows 上三组幅度 0.5/2/6 的独立 attention 数值探针
  已通过（各 1,940,736 个输出，最大绝对误差 0.001202 内），但与 q128 的
  half 输出并非逐位一致，尚未进行 TF3/ONNX 整图验收或生产集成。
  `fa4-strict-probe.log` 与 r2 `probe-fork-fa4-report.json` 保留全部结果。
  同次诊断中，strict FA4 本身中位数约 50.3 微秒，加入独立 RoPE 后约
  66.7–68.6 微秒，与当前 q64-serial 含 RoPE 的 66.8–68.8 微秒接近。
  该算子诊断不构成整网收益；下一步优先评估 RoPE 融合，再过整图门。
- **G4 B8 数值证据**：7 个布局/tactic 组合各自通过 4 局 ORT FP32 检查，
  包含大小 batch、plan 优先级、手写回退和 rank=time；最终 TF3 对 C++ FP32
  全字段通过，top-1 **128/128**；graph warmup/首次发射回归通过。
  证据：`layout-tactics-final.log`、`numeric-final-layout-b8.log`、
  `numeric-final-layout-b8/comparison.json`、`graph-final-layout-b8.log`，
  均位于 `target/fork-parity-20260908`。正确性通过不覆盖 B8 的性能负收益。
- **G4 生命周期修复**：原先权重上传发生在 tactic plan 安装之前，可能先按环境/
  默认布局准备权重，之后推理按 plan 选择另一布局。现在先安装并验证 plan，
  再上传相应布局权重；新增 `host_tactic_revision=1` 纳入构建兼容性边界，使
  旧主机逻辑构建被拒绝。`old-binary-rejects-layout-plan.log` 记录旧 binary
  实际因不认识 `host_tactic_revision` 字段而拒绝新布局 plan；这与新 binary
  对缺失/错误 revision 的拒绝分别验证。不能把错误生命周期下的测量当采纳证据。
- **G4 B16 与独立 C32 profile 验收完成**：`numeric-layout-b16` 和
  `numeric-layout-b16-throughput` 各自全字段 PASS、top-1 **128/128**。
  最终 `numeric-certified-c32-plan/comparison.json` 同样 PASS、128/128；
  该次故意设置环境布局 `tn`，`worker.log` 仍在 rows=5776 明确执行 plan 指定的
  `nn_k384_b16`，证实 plan 安装与权重准备顺序、环境覆盖实际生效。
- **G4 路径与 CPU 回归**：`layout-b16-tactics/summary.json` 通过 7 case：
  5 个布局/plan 优先级用例各有 4 局 ORT FP32 检查，另 2 个验证缺失/错误
  host revision 拒绝，不能将它们称为 7 个数值用例。`b16-revision-tests.log`
  的 16 个 tactic plan CPU 单测通过；Python paired 6/6、tactics 9/9 回归通过。
- **尚未完成**：G2 strict attention 的 TF3/ONNX 整图数值和整网性能验收、RoPE 融合，
  以及 G3 B12–B16/lane/graph 联合调优。C32 profile 的 G1/G4 通过不代表
  这些方向已通过，也不改变原有精度门。
