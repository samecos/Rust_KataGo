# RustGo 统一推理优化量化后端

2026-09-24。目标规格与实施路线，来源于用户对 TF3 B11 / 剪枝 B15 的 PTQ、混合精度和 SM120 Tensor Core 讨论。

用户于 2026-10-01 调整的目标：**一个统一的推理优化量化后端，兼容目前已有模型，代理只验证实际推理速度是否提升；模型精度全部由用户验证。**

## 2026-10-02 当前交付与启动

后续用户授权：先提交当前代码到本地 Git 仓库，然后由代理校验精度。确认的门限为**同模型 FP16 基线下，白方胜率最大绝对偏差 ≤5 个百分点**，例如 50%→55% 恰好到门限。这取代此前“精度由用户完成”的任务范围；新阶段另存实际输出与报告，旧 6 个百分点结果仍是历史数据，目前未产生新 5 个百分点通过结论。

用户确认推理优化完全放在 Rust_KataGo，授权执行。**`D:/Go/Server` 已恢复原提交 `b9873656b5d158664ae79dd12e7b71309881d54c` 的源码，工作区干净。Server 按原方式启动，不需要 `--execution-profile`。** 此前同步的 7 个文件及 2 个新增测试/示例已撤回并归档，回执为 `D:/Go/Server/target/execution-profile-rollback-r2/result.json`；旧同步与测量原件保留。

当前 Worker 恢复原 v1 Hello / Welcome / EvalRequest / EvalResult，不要求网络 profile 字段或 ACK。模型 SHA、任务/lease、重复请求和取消处理保留；实际量化身份、本地 `cudaQuantExpectedProfile`、模型/配方和 NN / CUDA Graph 缓存检查继续由 Rust 推理侧负责。网络仍按原模型 SHA 路由；`backend_info` 中的 profile 是诊断信息，Server 不将其作为调度条件。

最新程序：`D:/code/Rust_KataGo/target/worker-original-protocol-native-r1/cargo/release/katago-rs.exe`，SHA256 **`0cdafa6f90094a11dbd604642c50980ad0bc28887e5fcf2ffcc4fd85c21186ed`**。编译实际 exit 0，3,357 项显式构建来源首末一致；新 6 项原协议 CPU 检查、本地量化身份 2 项和准备入口 5 项均通过。准备入口测试子进程 exit 0；外层首次日志解析因 Windows 双 CR 行尾返回 1，原失败保留并只读复核确认 5 PASS，未重跑测试。

B15/B8 新程序已连接**原 Server WorkerPool**，完成一个真实请求：模型与本地 profile 一致，completed=1、failed=0、in_flight=0，Worker 与探针由各自持有的 Popen 正常 wait，均 exit 0。实际 profile 为 `rustgo-quant-v1:1e797d8f9e7323ffb69668474aa494bd27f9523d23ca647bbbb9ab9cc8281f35`。探针使用原 Server 库，代码位于 Rust_KataGo 的 `tools/original_worker_probe`，不修改 Server 业务源码。结果在 `target/worker-original-protocol-composition-r1/result.json`。原协议没有显式 Drain ACK，incoming 是否干净半关闭仍未知，不由流结束推断进程正常退出。

### 启动已准备的 Worker

本机 RTX 5070 Ti 的两个配置已经准备好，服务地址为 `http://127.0.0.1:50051`。在 PowerShell 执行以下之一，默认前台持续运行，并使用原 Worker 重连机制：

```powershell
# B15，batch 上限 8，并发容量 32；本次真实接入验证使用这组配置
& D:\code\Rust_KataGo\target\worker-original-protocol-preparation-b8-r1\launch-worker.ps1

# B15，batch 上限 3，并发容量 32；本次只完成元数据准备，未另启动模型
& D:\code\Rust_KataGo\target\worker-original-protocol-preparation-b3-r1\launch-worker.ps1
```

脚本启动前核对程序、模型、配方、配置与环境 SHA，设置固定环境。添加 `-Once` 可限制为一次连接。B3 的本地 expected profile 为 `rustgo-quant-v1:941b0ffce0368db6e75eea7c41549257bd211e57119d63aa02226b5336ba6aac`，本次未观察它在新 Worker 中实际加载；加载时仍须通过后端身份硬门。B11 保留原后端与原配置使用方式，本轮没有发布已确认提速的 B11 新方案。

若服务地址或 Worker ID 不同，使用新入口重新准备到一个不存在的目录，例如：

```powershell
Set-Location D:\code\Rust_KataGo
python -S -B scripts/prepare_quant_worker_v1.py --binary target/worker-original-protocol-native-r1/cargo/release/katago-rs.exe --binary-sha256 0cdafa6f90094a11dbd604642c50980ad0bc28887e5fcf2ffcc4fd85c21186ed --build-result target/worker-original-protocol-native-r1/verified-build.json --build-result-sha256 b4f3e4b34045768cd6241f210ac532c3b68ad7b2f2e4459834e7aa39444a8fe8 --candidate b15-b8 --environment target/unified-quant-b15-compat-partial-tests-cpu-r1/environment.json --environment-sha256 897ec6666f323209f4aa3949e780252211f3c01d4199b9847fa07a28bb3192ff --gpu-uuid GPU-c1842c67-b9c6-3eb5-4592-ebb8111d931b --server http://127.0.0.1:50051 --worker-id my-b15-worker --output target/my-original-v1-worker
& .\target\my-original-v1-worker\launch-worker.ps1
```

`--candidate b15-b3` 选择另一配方。准备阶段只有 CPU `quant-inspect` 与 CUDA 设备/库元数据查询，不上传模型或做推理，也不启动 Worker；真实加载仍由本地 expected-profile 门复核。目前这个准备入口限定已有两份 B15 配方与本机 5070 Ti，不认证其它 GPU 性能。

### 编译和 TUNE

原 Server 无需为此次优化重新编译。Rust Worker 的正常编译命令仍为 `cargo build -p katago --features cuda --release --locked --offline`，需要 MSVC x64、CUDA 13.3 和 CUTLASS。下面复用本机本轮成功构建的环境，并将输出放入新目录：

```powershell
Set-Location D:\code\Rust_KataGo
$buildEnvironment = Get-Content .\target\worker-original-protocol-native-r1\environment.json -Raw | ConvertFrom-Json
foreach ($property in $buildEnvironment.values.PSObject.Properties) {
    [Environment]::SetEnvironmentVariable($property.Name, [string]$property.Value, 'Process')
}
$env:CARGO_TARGET_DIR = 'D:\code\Rust_KataGo\target\my-worker-build'
cargo build -p katago --features cuda --release --locked --offline
```

现有启动配置只绑定上方已验证的新程序。重编译后需新的构建回执、程序 SHA 和元数据准备，重新计算 `cudaQuantExpectedProfile`；不能把已有配置中的 profile 原样转给另一二进制。实际构建与来源回执见 `target/worker-original-protocol-native-r1/verified-build.json`。

当前直接复用既有 q64、FP16/INT8 混合配方与 heuristic tactic，`KATAGO_CUDA_INT8_GEMM_TUNE=0`。原 Server 不参与 TUNE。重新调优应通过独立离线性能入口对照同模型、同输入、同 batch/并发下的原后端，用 A/B/B/A 和独立重复测量裁决；改变 tactic 或精度配方后必须重新生成本地身份。时间采样式 INT8 GEMM TUNE 尚不能提供最终算法身份，Worker 继续拒绝该模式接入；当前准备入口也拒绝修改其固定策略。本次保留原测试预算，累计模型 native 启动已达到 96/96，没有重新 TUNE 或增加 ABBA。

**新程序完成的是原协议兼容验证，未重新获得速度认证。** 下方 1.84% / 1.75% 与 1.7873% / 1.5831% 均是各自旧构建、设备和负载的历史实测结果，不能转成新程序的提速结论。本次不检查输出精度，精度仍由用户验证；没有替换生产服务/Worker、提交或推送。

以下为历史架构和交付记录。2026-10-01 typed Worker/Server 字段、`--execution-profile` 和 `scripts/launch_quant_worker.py` 接线已被本次原协议方案取代，旧入口和证据原样封存，不按其历史命令启动当前 Worker。

2026-10-01 历史路径更正（已被上方撤回取代）：按用户要求，Server 协议、执行身份隔离及相关测试/示例曾同步到 **`D:/Go/Server`** 仓库。7 个源码文件与首次已测版本逐字节一致；当时主仓库 13 项执行身份 CPU 测试通过，公开导出工具 18 项通过、4 项跳过，差异格式检查通过。下文与历史报告中的 `target/unified-quant-server-profile-worktree-r1` 是首次验证的独立工作树，继续保留为证据来源，不作为日常 Server 源码入口。

本条调整优先于下文历史路线和验收要求。代理不再执行输出误差、policy/胜率/目差、棋力、selection 精度门或 holdout 精度验证，精度结果也不再是性能测试的前置条件。已有精度记录作为历史证据保留，不补写通过结论。现有 PTQ 参数及候选配置优先复用；仅因性能测量不重新采集校准或精度语料。

代理的交付是 B11、B15 等现有模型的可复现性能对比及可复用配置：同模型、同输入、同物理 batch/并发、同缓存策略下，对照现有 FP16/INT8 后端；分别报告相对各基线及该负载最快现有基线的延迟、吞吐和加速比例。预热后交替重复测量，完成 GPU 同步，记录设备、构建、配方、实际 batch 分布及测量波动。正常推理路径的端到端结果是速度结论的依据；算子耗时和 Tensor Core profiling 用于解释结果，单独测量。波动覆盖的微小差异只报告为尚不能确认提升。模型/配方身份、真实执行、请求完成及计时有效性仍需检查。

速度通过只表示该设备和已测负载的性能提升，精度状态统一标为「用户验证／本次未评估」。旧失败、单次消费、锁、预算和截止保持，新范围不授权重启旧入口或修改历史。用户随后明确“继续执行吧。重启目标”，已恢复工作并完成独立速度测量，不继续精度采集链。

2026-10-01 新范围结果：48 个配置初筛、5 组正式 ABBA 和 2 组独立更长确认完成。5070 Ti / C32 上，B15/B3 与 B15/B8 混合配置相对各负载最快已测旧基线的确认吞吐提高 **1.84% / 1.75%**；B11 已测候选未确认提速。完整条件、负结果、原始证据及复测方式见 [速度验证](RustGo统一后端速度验证.md)，配置/配方/环境/程序身份交付在 `target/unified-quant-speed-only-final-r1`。这是已测条件下的小幅推理吞吐提升，不代表精度、其它设备或搜索速度通过，生产 Worker 未替换。

本地运行入口 `scripts/launch_quant_inference.py` 已完成 8 项 CPU 检查及 B15/B3、B15/B8 两次真实推理。入口按模型内容 SHA、GPU UUID/型号/驱动、batch 上限与并发匹配已测配置，校验程序、配方和环境来源，并在启动时设置 `cudaQuantExpectedProfile`。同内容模型可以换路径；匹配配置的来源损坏会拒绝启动。未匹配时须由调用者同时指定原程序及原配置作为回退，不宣称回退提速。默认 `--mode prepare` 只准备，`eval` 测量推理链，`gtp` 透传标准输入输出；目前不连接共享 nnworker Server。

例如本机已测 B15/B8 的准备命令（全路径身份仍取自 handoff，输出目录须不存在）：

```powershell
python scripts/launch_quant_inference.py --profiles target/unified-quant-speed-only-final-r1/speed-profiles.json --profiles-sha256 bdb6ff87cfe6f3b208208f58571f393dd09c1fb8093cc09bb0570f55ac73fd3e --model models/b15-ffn-pruned-a8.bin.gz --batch-cap 8 --capacity 32 --gpu-index 0 --output target/my-b15-b8-prepare --mode prepare
```

两次入口验证均实际 exit 0，完成 16,384 行，实际 profile 与已测配置一致，来源首末稳定。单次吞吐 B3 为 587.7、B8 为 721.9 eval/s；这些非配对样本只验证入口组合，不重新估算加速比例。结果在 `target/unified-quant-runtime-launcher-r2/result.json`。旧测速计划已截止，不续跑或修改；此入口验证是独立有限阶段。

2026-10-01 本轮交付完成：Worker/Server 的 `execution_profile_id` 已贯穿模型加载、Hello、Welcome、请求、结果及 Search 生命周期。服务端执行身份和模型固定后不随 Worker 断线或空池改变，Worker 拒绝不匹配的 Welcome 和请求，Search 在首次请求前绑定身份。Server 首次验证在 `target/unified-quant-server-profile-worktree-r1` 独立工作树完成，随后按用户要求同步到 `D:/Go/Server` 主工作树；现有生产进程保持不变。默认 Server 只接受旧协议空 profile；启用 `--execution-profile` 后只接受精确指定的非空 profile，B3、B8 使用各自独立的服务实例。

新 CUDA 程序是 `target/unified-quant-typed-worker-native-r1/cargo/release/katago-rs.exe`，SHA256 为 `9a1f42f7083fe65cd1aed717e33b5d2f4a776f533af6b30b84cc4b85d3a6272f`。新构建已单独完成两组 A/B/B/A；没有继承旧程序的速度认证。5070 Ti、驱动 610.88、并发 32 下，A 是前轮筛选所得各负载最快已测旧 INT8 q64 基线，B 是新构建的混合方案；每次完成 32,768 行，预热 128 轮，正式 1,024 轮。

| B15 负载 | FFN 混合方案 | A 吞吐几何均值 | B 吞吐几何均值 | 吞吐提升 | 每个已完成逻辑 batch 的墙钟成本 A → B |
|---|---|---:|---:|---:|---:|
| batch 上限 3 / C32 | 10 INT8 + 35 FP16，q64 | 581.90 eval/s | 592.30 eval/s | **1.7873%** | 5.1540 → 5.0635 ms |
| batch 上限 8 / C32 | 23 INT8 + 22 FP16，q64 | 710.70 eval/s | 721.95 eval/s | **1.5831%** | 11.2540 → 11.0785 ms |

B3 的 A/B 组内极差相对均值为 0.2753% / 0.1014%，B8 为 0.3665% / 0.1525%；两组最慢候选仍快于最快基线。结论是这一次 ABBA 的描述性重复测量结果，没有统计置信区间，新程序没有另做独立更长确认。计时覆盖 NnEvaluator 的编码、排队、CUDA 完成和解码，排除模型加载与预热；表中墙钟成本不是单请求延迟分位数。实际平均逻辑 batch 接近 3 / 8，并非每批满载，物理 batch 分布未跟踪。真实 RPC 和搜索请求只验证正常完成，不据此声称 RPC 或搜索提速。

速度原件见 `target/unified-quant-typed-worker-speed-r1/run/report.json`，实际外层退出为 0，8 个推理进程均正常 0；有限独立审查见 `target/unified-quant-typed-worker-speed-review-r1`。新程序还完成 B11 的 64 行正常推理；该短样本没有配对基线，只证明本次加载和执行完成。前轮 B11 候选未确认提速，入口保留调用者提供的原程序、原配置，不替 B11 发布新的提速方案。

真实 INT8 执行日志记录了 `cublaslt-int8`、`accumulator=int32`、`imma=true`。NVIDIA 的 [cuBLAS 数值实现标志文档](https://docs.nvidia.com/cuda/cublas/index.html#cublasltnumericalimplflags-t) 将 IMMA 定义为整数 Tensor 运算指令；这些标志支持已观察算子的 Tensor Core 路径判断。日志仅记录首个形状，不能推断全网覆盖率或利用率；硬件计数器仍因 `ERR_NVGPUCTRPERM` 未测。

统一 Worker 入口是 `scripts/launch_quant_worker.py`。它读取新速度报告与计划，按模型内容、GPU UUID/型号/驱动、batch 上限和并发选择通过本轮速度条件的配置；对所有绑定来源复核，将精确 profile 写入 `cudaQuantExpectedProfile`。默认只准备，不上传模型或启动 Worker。以下命令在仓库根目录准备 B15/B3，输出目录必须尚不存在；`--server` 应填写拟使用的服务地址：

```powershell
python -S -B scripts/launch_quant_worker.py --report target/unified-quant-typed-worker-speed-r1/run/report.json --report-sha256 ad6d495a0109eff9e3d8f1cd5bc654df072df6f09d32f0a43c1b76fe530ee33f --plan target/unified-quant-typed-worker-speed-r1/plan.json --plan-sha256 1acdd507d74f1be91b563023637499dcdde4fbf44091ce55ec0f0a8facdc034d --model models/b15-ffn-pruned-a8.bin.gz --batch-cap 3 --capacity 32 --gpu-index 0 --server http://127.0.0.1:50051 --worker-id b15-b3-local --output target/my-typed-b15-b3 --mode prepare
```

入口生成 `manifest.json`、`selected.cfg` 和 `receipt.json`。在 `D:/Go/Server` 执行 `cargo build -p go-server --release --locked` 构建服务，并把 manifest 的 `selection.expected_profile` 原样传入 Server 的 `--execution-profile`。B3 profile 是 `rustgo-quant-v1:411a535d9d2e8dffb890a8729e06bcaf826e7a47a7a6e14da3acd0d441781479`；B8 是 `rustgo-quant-v1:2ef91ab24f6f64f62265c4a2ecb15c1fb08558123a7f238d44d83681ac639ddc`。用新的输出目录和 `--mode run --timeout-seconds 300` 可以执行一次有时限的 Worker 生命周期。未匹配时，入口退出 2；只有同时提供 `--fallback-binary`、`--fallback-config` 才运行显式回退，旧空 profile Worker 必须连接 legacy 模式 Server。绑定来源损坏、未完成报告及启动失败会拒绝继续，不自动尝试另一个方案。

本轮新增 33 项 CPU 检查通过（Server 13、已加载执行身份 7、Worker 6、Worker 入口 7）。B15/B3、B15/B8 各完成一次真实 Worker/Server 请求：精确模型/profile 匹配，completed=1、failed=0、in_flight=0，Worker 与探针均实际正常 exit 0；结果见 `target/unified-quant-typed-worker-composition-r1/result.json`，有限独立审查见 `target/unified-quant-typed-worker-composition-review-r1`。Drain 已排队，服务转发和服务任务均正常结束；传入流是否干净半关闭仍未知，Worker 正常退出依据独立持有进程句柄的 wait，不从流结束推断。该组合验证没有检查输出精度。

交付限制：支持一个不可变模型的 Worker；本轮不认证同进程多模型、运行时切换配方、5060 Ti / 5080 性能或全部网络的 Tensor Core 利用率。时间调优后尚未形成最终算法身份的 CUDA 模式拒绝 typed Worker 接入。所有精度验证由用户完成，当前速度配置没有精度通过声明。新范围累计 95/96 次 native 启动，旧失败、锁、原消费及截止均保留；本轮没有重启旧 campaign 或替换生产进程。

以下内容保留此前架构路线、阶段结果与旧验收历史。涉及未完成状态或数值/selection/holdout 门时，以本页上方最新速度范围和交付结果为准。

统一入口及逐投影 FP16 / INT8 / MXFP8 已实现，现有 B11、剪枝 B15 和 ONNX 的执行、身份与 Graph 回归通过。RTX 5070 Ti 的小语料诊断中，全 FFN MXFP8 在低并发和并发 32 均慢于 INT8。旧 C1 及 B11 q64 续接累计 10 份数值采集、16 组 ABBA，已终态并完成独立审计，没有统一候选通过完整旧控制矩阵；原负结果与永久消费继续保留。

80 份单组／参考采集及 78 份比较已完成，提名出 12 个 B3/B8 混合配方，B1 成本门没有合格组。14 份混合／参考采集共 28,616 行及独立审计已完成；12 个候选各 2,044 个 **B1** 校准局面均满足相对同模型 FP16 的最大胜率偏差 ≤6 个百分点门，B11/B15 各候选最大偏差最高为 2.406883/1.530261 个百分点。该结果不替代 B3/B8 selection、其他输出指标或性能验收。

正常 `katago-rs` 已在隔离构建上完成 B11 FP16、B15 首 FFN INT8、既有 ONNX FP16 三例跨进程 record/restore：capacity3、B1/B2/B3 双槽，共 54 次前向／108 物理行；90 对五头原始输出、18 对解码结果与 3 对重导出包全部一致，独立 CPU 审计 852 份来源稳定，见 §8.45～8.46。后续 DualFFN 探针预算与固定包 selection 接口仍在隔离目录推进，当前进度见后续小节。

后续 selection 的两模型各 2,004 个局面已完成 CPU 输入导出与独立审计。新增 catalog、继承历史消费的 continuation 和 collection 外层启动器已在隔离源码接通，177 项 Python 基础检查与 14 项 Rust 父证明检查通过；超时清理修复的 9 项相关检查及一次真实 CPU 超时检查也已通过。历史临时目录隔离的 20 项 mock 与唯一只读重放已完整通过，长流程调用方的有限 CPU 补检也已完成，原失败记录保留。最终 CUDA CLI 已实际编译成功，独立只读核验通过；新版本的执行包、真实 B3/B8 selection 和性能矩阵尚未运行。详见 §8.50，不能将输入或构建验证写成模型精度与性能通过。

最新续接：长流程调用方剩余 5 项 CPU 检查及末核通过，原两次检查流程失败保留。最终同一 CUDA CLI 的 **16/16 真实 CPU preflight** 全通过，覆盖 B11/B15×容量3/8、12候选和4参考；10项构造器检查也通过。已据真实回执生成16份未 seal 的 campaign 草案，0 GPU、0模型上传、0真实账本预约。CPU预检和运行计划不代表GPU执行包已录制或恢复成功。

完整候选控制矩阵、独立确认、最终留出、Worker/Server profile 与缓存隔离仍未完成，正式二进制和生产 Worker 未替换。NVFP4、随 batch 改变精度及其它显卡专属调优属于后续可选范围；已有 SM120 实现不等于完成 5060 Ti／5080 实机认证。

## 1. 范围与完成标准

统一后端在同一模型加载与执行架构中选择精度、布局、内核和物理 batch 策略。用户提供现有模型及运行模式，后端读取模型真实结构；不要求用户按 B11 / B15 或 FP16 / INT8 分别挑选执行器。

兼容基线固定为源码提交 `c171ad60093929da47978645afb2c7b04c5b8c6a` 已支持的 CUDA 模型范围，下表列出重点对象。G1 开始时建立回归清单，记录本地已有模型的原始 SHA、格式/版本、真实层形状与对应测试；基线所有已支持结构均需保持兼容，不以模型名称替代清单，也不承诺任意未知模型都已实测。

| 对象 | 要求 |
|---|---|
| 原生 TF3 v17 B11 `.bin.gz` | 保留原文件、原始 SHA、输入特征、输出语义和模型元信息 |
| 当前 FFN 通道剪枝 B15 `.bin.gz` | 按每层实际 hidden 执行，支持当前合法宽度，不补回完整模型后冒充剪枝加速 |
| 当前 CUDA 支持的 ONNX，包括 `b11fix.onnx` | 复用已有解析和 lower 路径，不扩张为任意 ONNX 支持 |
| 单机 GTP / 搜索、nnworker | 使用同一执行层；分别验收延迟、吞吐与实际搜索收益 |
| RTX 5060 Ti / 5070 Ti / 5080 | 共用 SM120 实现，按实际设备、显存、库版本和负载选择执行计划 |

继续限定 19 路，以及当前 CUDA 层图支持的结构（包括 D32）。模型版本、算子或形状不支持时明确报错。模型加载成功、量化数值通过、获得性能收益是三个分别记录的状态。

完成标准不是所有层都使用最低位宽，而是：同一入口可以可靠加载上述现有模型；存在校准、选择、验证、保存和重载的完整流程；候选与本机同负载最佳已验证 FP16 / 既有混合精度方案竞争；没有稳定收益的候选不进入发布计划。没有某型号实机证据时，标记未验证，不由 SM 数比例推算认证结果。

## 2. G1 实施前的基础与缺口

- `CudaBackend { int8: bool }` 与 `CudaInt8Backend` 已共享 `LayerGraph` / `CudaModel`。统一工作应扩展此执行器，不复制一套 TF3、ONNX 或 Worker 后端。
- `native_model::lower_model` 与 `onnx_parser` 已汇入同一层图。
- 现有 INT8 为 W8A8，权重逐输出通道静态 absmax，激活逐 token 动态 absmax，INT32 累加；已有 RMSNorm / 量化和 SwiGLU / 再量化融合。
- 现有层选择只有 `scope` 与 FFN 宽度阈值，不能表达独立层和投影精度。
- 当前 INT8 入口会全局排除多项 FP16 专用 tactic。新架构必须改为逐层能力判定：量化部分 FFN 后，其他合法 FP16 层仍可参与专用快路径竞争。必须重新验证混合图，不能继承旧整图认证。
- 当前 Graph 主要按物理 batch 缓存，模型上传后不允许改变量化策略。v1 保留这一不可变性。
- Worker 保留原模型 SHA，通过 `backend_info` 描述 INT8。文本描述本身不等于服务端已强制精度隔离；正式接入必须核实服务端任务、路由及缓存的实际隔离机制。

源码入口：

- `crates/kata_nn/src/backends/cuda.rs`
- `crates/kata_nn/src/backends/cuda_exec.rs`
- `crates/kata_nn/src/backends/int8.rs`
- `crates/kata_nn/src/onnx_parser/mod.rs`
- `crates/kata_nn/src/native_model.rs`
- `crates/kata_nn/src/tactic_plan.rs`
- `crates/kata_program/src/setup.rs`
- `crates/kata_worker/src/evaluator.rs`

## 3. 统一架构

```text
原生 TF3 / 已支持的 ONNX
          │
          ▼
现有 parser / lower → LayerGraph（稳定层标识、真实形状、原始权重）
          │
          ├── 离线校准与敏感度分析 → 模型精度配方
          │                             │
          └── 设备与内核能力查询 ────────┤
                                        ▼
                            编译执行计划 / 权重打包
                                        │
                          数值门 → 本机 ABBA → 最终复验
                                        │
                                        ▼
                          保存已验证产物 → 统一 CUDA 执行器
                                        │
                              单机搜索 / nnworker
```

设计分为三个对象，三者绑定后才构成可部署产物：

1. **模型描述**：原始模型 SHA、格式/版本、输入输出语义、层拓扑与逐层形状。名称中的 B11/B15 不作为唯一判据。
2. **精度配方**：逐层/层组的格式、权重及激活量化方式、scale 和变换参数、累加类型、存储与舍入边界、校准数据身份、精度验收证据。配方拥有独立哈希。
3. **执行计划**：绑定模型及配方哈希、GPU/软件/构建指纹，按形状与物理 batch 指定算法、布局、融合、workspace 和 Graph 策略，并附性能与最终数值证据。

层标识须由 lower 的稳定结构路径与投影名称生成，并校验 shape；禁止只按当前向量下标或同名层模糊匹配。QKV 共享输入、FFN 双上投影等融合组同时参与约束，避免逐层选择破坏融合而失去整体收益。

v1 在加载时固定层精度与量化参数，运行时只按已覆盖的 batch/shape 选择计划内算法；不在每次请求中重新校准或试跑候选。更换配方必须重新加载，清理相应 Graph 和 NN cache。未来若引入随 batch 改变精度，必须将完整路由表纳入配方身份，并验证全部可达路径。

已实现的统一入口为 `cudaquantbackend`，精度配方配置键为 `cudaQuantPlan`；保留旧 `cudabackend` / `cudaint8backend` 配置语义，旧调用不因升级自动接受降精度。正常启动消费固定产物，耗时校准和调优由显式离线流程完成；当前实验产物的身份校验不等于完成精度或性能认证。

无匹配产物时，可以选择当前受支持的基础 FP16 路径，并清楚报告未调优状态。用户显式指定的计划若损坏、身份不符或能力不满足，则报错；不静默换成另一种精度。不能回退 dummy 后仍声称是真实推理。

## 4. 精度候选与 PTQ 路线

| 部位 | 首轮候选 | 选择依据 |
|---|---|---|
| B11 宽 FFN | 既有 FP16 / INT8，再加入 MXFP8 | 完整融合组耗时、最终输出误差 |
| B15 窄 FFN | FP16 与现有 INT8 | 量化、padding、启动成本是否抵消 GEMM 收益 |
| B15 宽 FFN | FP16 / INT8 / MXFP8 | 逐层误差与实测耗时，不只用宽度阈值 |
| QKV / attention 输出投影 | FP16 起步，逐层加入 8 位候选 | 与 attention 核心分开评估敏感度 |
| QK / PV attention 核心 | 现有 FP16 输入、FP32 累加 | 优先保留现有融合和 D32 路径 |
| RMSNorm / Softmax / SiLU / 残差 / 输出头 | 当前精度与 half 边界 | 保留 FP32 激活、归约和残差计算 |
| 少量低敏感宽 FFN | 后续加入 NVFP4 W4A4 | 校准可接受且端到端有稳定收益 |

格式本身不决定误差排序。MXFP8 的 32 元素块缩放改善局部范围适配，不增加 E4M3 尾数位数，仍须与 INT8 实测竞争。普通 FP8 可以作为对照候选。只压缩权重的 W4A16 不能按原生 FP4 乘法峰值估计收益。

校准流程：

1. 从真实合法棋局与实际搜索叶节点采样，覆盖双方、开中终盘、均势/优势、攻杀/劫争，以及实际使用的规则与贴目。
2. 按整盘划分校准、选型与最终留出集合。同局相邻局面及八种对称不得拆开伪装为独立样本。
3. B11 和每个不同 SHA 的剪枝 B15 独立统计。B15 新增量化误差对照该 B15 自身的 FP16，剪枝损失另计。
4. 从当前 absmax 基线起步；对有稳定通道离群值的层比较 SmoothQuant 类缩放。共享输入分支一致变换，不随意跨越非线性。
5. NVFP4 比较 max、权重误差和 Local-Hessian 输出误差选 scale。后者属于离线 PTQ 候选，不要求重新训练；围棋适用性尚待验证。
6. 用前向逐层/层组消融筛选，再验证完整混合图；不能把各层误差简单相加作为整网证明。

当前 Model Optimizer 以 PyTorch / ONNX 为入口。可以借鉴其算法，但原生 TF3 需要校准导出/适配；不承诺一键转换。v1 以引擎真实执行图采样为基础，避免未验证的复刻图引入另一套语义。

## 5. 性能目标与 Tensor Core 实现原则

优化目标为精度约束下的实际延迟/吞吐。低并发延迟与饱和吞吐分别选择计划，记录显存与启动成本；不把所有工作负载压成一个峰值数字。

完整成本包括量化、scale 生成、布局转换、padding、GEMM、激活/反量化、中间结果读写，以及融合得失。权重离线量化和打包；动态激活量化优先融合到 RMSNorm / SwiGLU 等生产者中。

- 主干 GEMM 的 M 通常为 `361 × physical_batch`，batch=1 不能套用 LLM 单 token decode 的带宽结论。
- 先使用匹配 SM120 的 cuBLASLt / CUTLASS 路径，再根据实际热点增加定制融合。真实路径标记及必要的 profiler / SASS 证据确认命中 Tensor Core；性能裁决运行不同时开启 profiler。
- SM120 采用专门的 `mma.sync` 低精度路径；不借用 SM100 的 tcgen05/TMEM/多 SM 能力假设。
- B15 删通道后的矩阵仍为稠密 GEMM，不自动获得 2:4 稀疏乘法收益。
- 各 GPU 重新选择算法和 batch。调优结果可持久化，但不跨版本盲目反序列化供应商 opaque algorithm descriptor；保存可复建的选择条件并重新做能力/身份校验。
- tactic 决策继续通过计划解析与 `tactic_var()` 纪律；新状态应按模型/执行上下文持有，避免当前进程级状态污染多个模型。

## 6. 精度、身份与验收

沿用用户已允许的独立实验门：**相对同模型 FP16，最终留出集最大胜率输出绝对偏差 ≤ 0.06，即 6 个百分点**。同时报告平均/P95/最大偏差、policy KL、首选与候选排序及其概率差、目差和 ownership。该门只约束采样集中的胜率输出，不代表所有未来局面或对局胜率保证。

沿用 `scripts/validate_int8_backend.py::quantization_metrics` 的定义：比较输出处理后的白方 `white_win_prob`，不比较原始 value logit，不改为当前落子方视角，也不把 no-result 概率并入胜率。参考 FP16 的模型、二进制、配置和计划哈希全部冻结。NaN/Inf、policy 合法点集合改变、输出缺失直接失败；policy 等其余误差目前为报告项，不擅自新增用户未确定的误差阈值。

首版最终留出集设计目标为至少 4,096 个不同语义局面、来自至少 128 盘独立棋局；这是工程覆盖起点，不是统计意义上的普适误差保证，对称变换不增加独立局面数。校准集、选型集与该集合不重叠。冻结请求清单及其哈希，覆盖所有发布配方和可达物理 batch 路径，并通过真实 Worker 和单机集成验证输出处理一致。

原 C++ FP32 / ORT FP32 认证门保留。对有损候选如实记录其 PASS/FAIL，不复用或改写原 FP16 认证。G1 纯 FP16 重构必须重新通过原数值门，不能改用 6 个百分点门；固定既有 INT8 配方的纯重构须保持既有输出。新低精度候选经过独立算子、整图和真实执行链路验证后再测速。

原模型 SHA 不改。量化配方、变换权重、scale、构建和执行计划拥有可追溯身份；不同配方不能误用彼此的 NN cache / Graph / 性能证据。算法和归约顺序也可能影响输出，因此每个最终执行计划都要做完整数值复验。

Worker 发布需验证服务端实际能区分精度配置及其任务/cache 范围。若现有协议无法表达，先使用明确隔离的任务/服务配置，或实施兼容的能力协商；不能只追加 `backend_info` 就声称协议已 fail-closed。

性能验收使用同 GPU、同模型、同计时边界的顺序 ABBA。算子与固定 batch 的完整前向保持相同物理 batch；真实 Worker / 搜索固定请求集合、并发/搜索条件与资源限制，允许计划采用不同 batch 策略，并报告实际 batch、延迟、吞吐和显存，避免为了凑大 batch 改变请求条件。比较对象包括本机该负载的最佳已验证 FP16 方案与现有混合精度方案，不能只挑弱默认配置。

沿用 `scripts/tune_runtime.py::decide` 的基本统计口径：一组完整 ABBA 中每臂两个有效样本，分别取几何平均，吞吐收益为 `candidate_geo / baseline_geo - 1`，每臂波动为 `max / min - 1`；至少 1% 收益且两臂波动均 ≤5% 才可采纳。以延迟为主目标时预先声明 P50/P95 等主指标，采用 `baseline_latency_geo / candidate_latency_geo` 判断加速，同时报告原始延迟。发布前再做独立确认组，不丢弃不利样本；不稳定结果只能诊断。没有稳定收益则保留更合适的既有路径。

分别报告算子/完整前向、evaluator、真实无缓存 Worker 和单机搜索；低并发报告延迟，高并发报告实际 batch 与吞吐。如需声明棋力或等时对局收益，另行进行对局验证，不能把 NN rows/s 换算为 Elo；棋力认证不作为交付实验统一后端的前置条件。

## 7. 实施顺序

| 阶段 | 产物 | 完成条件 | 当前状态 |
|---|---|---|---|
| G0 目标与架构 | 本文、兼容边界、验收口径 | 独立审阅消除关键歧义 | 已完成，独立复核通过 |
| G1 统一执行与身份 | 固定逐层 FP16/INT8 配方、加载/校验、按层能力判定、旧入口兼容 | 现有模型/调用方式回归通过，错误计划拒绝，真实路径可审计 | 已实现并通过当前三模型回归，保持实验身份 |
| G2 PTQ 与选型闭环 | 校准集管理、敏感度/耗时表、配方与计划保存重载 | 留出集通过，完整成本选型，真实两模式 ABBA | 旧C1/q64完整矩阵无胜者；80采集/78比较/12混合提名已终态，14case混合校准继承续接已全部采集验收、实际exit0，独立终审及12份B1比较/只读P95均通过（仅calibration）；执行包143CPU及六case模型API GPU/独立审计通过，正常运行、部署、留出和收益验收仍未完成 |
| G3 MXFP8 | SM120 真正块缩放 GEMM、权重打包和量化融合 | 原型数值通过；B11/B15 按层竞争；仅有收益的计划发布 | 三模型 B1～B8 回归通过；r8 全 FFN 在 C1/C32 诊断均慢于 INT8，未采纳；完整语料精度未认证 |
| G4 局部 NVFP4 | Local-Hessian 等校准候选、低敏感层 W4A4 | 完整混合图通过同一精度门并优于已验证方案 | 后续可选，不阻塞本版交付 |
| G5 多卡交付 | 各型号已验证计划、启动入口、报告和等时对局证据 | 有实卡的型号完成复验；未测型号明确标注 | 待验证 |

第一项工程重点是让低精度层与现有快速 FP16 层真正共存，而不是给原 INT8 后端增加一个别名。G3/G4 的候选是否采纳由误差和效率裁决；未胜出的精度无需为了格式齐全进入部署。

## 8. G1 实验实现与当前证据

新增入口为 `nnBackend=cudaquantbackend`（别名 `cudaquant`），可选 `cudaQuantPlan=<recipe.json>`。省略配方时使用全 FP16；显式配方错误时拒绝启动。G1 支持逐投影 `fp16` / `int8`，G3 新增显式 v2 的 `mxfp8` 实验路径。FFN 双上投影保持一个融合组，下投影、QKV、attention 输出投影可独立选择。

`quant-inspect --model <现有模型> --output <新目录>` 是 CPU 命令，输出 `model-manifest.json` 和 `fp16-recipe.json`，不覆盖已有目录。配方绑定原始文件 SHA、完整图结构摘要、量化语义版本及投影形状，未列出的投影默认 FP16。`ResolvedRecipe` 即使由 Rust 调用方修改公开字段，也会在执行器上传前重新验证规范哈希。

G1 的 `rustgo-quant-v1:<sha>` 是已加载配方与执行环境的身份，**不是认证状态**。它绑定实际二进制、设备、构建/库/驱动、tactic 和解析后的 batch 容量。`cudaQuantExpectedProfile` 在统一后端核对实际身份，Worker 要求显式填写；它仍不能代替尚待完成的服务端任务与缓存隔离。

当前 executor 仍有进程级 tactic。G1 在量化初始化时冻结这些值，拒绝与旧 FP16 plan 双向混用；修改 tactic 需新进程。`CUBLASLT_RANK=time` 和 `INT8_GEMM_TUNE=1` 暂不允许进入统一入口，待 G2 能保存实际算法后开放。请求的 DualFFN 或 q64 内核不可用时明确失败。FP16/INT8 原入口保留原有配置语义；当前 Worker 另有已加载执行身份准入，因此上述两个时间调优开关用于旧 CUDA FP16/INT8 Worker 时也会被拒绝，本机 heuristic 配方不受影响。

已验证的本地清单：

| 模型 | 原始文件 SHA-256 | 实测结构 |
|---|---|---|
| B11 原生 | `1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6` | 11 blocks，trunk 768，mid 384，12 heads |
| 剪枝 B15 原生 | `3f216ee88226ee49ca826eaa5f7e1e9982ff48a96625914bfc4e5ad20ee095a7` | 15 blocks，trunk 1024，mid 512，16 heads，各 FFN 按实际宽度 |
| 既有 `b11fix.onnx` | `f2fc09fdf58a3e8a4b97addceab0c853874c52709152e61935b6ac6b9f474f94` | 11 blocks，trunk 768，mid 384，12 heads |

2026-09-24 当前证据（RTX 5070 Ti）：

- 配方 CPU 测试 8 项通过；tactic 冻结、有效 plan 双向互斥及并发安装测试通过。
- 三模型分别比较全 FP16、全 FFN INT8、全 Transformer INT8 与对应旧执行路径。在固定物理 B1/B8、两组输入下，五个原始输出张量逐位一致。
- 每条路径另用独立参考 workspace 验证 CUDA Graph 原址输入 A→B→A；回放与当前输入的直连结果逐位一致。
- 三模型均执行四种独立投影组成的混合配方，有限输出及 Graph 更新检查通过。这是路径检查，不是对同模型 FP16 的精度验收。
- 相关命令行、setup、Worker CPU 测试首轮 194 项通过。
- B11 扩展到所有物理 B1～B8，新旧三种完整配方及混合图 Graph 检查通过。混合图中的 FP16 DualFFN 与基线逐位相同；fusion up/down/all、Split-K 的原始输出误差及 policy top-1 通过原严格门，日志也断言实际内核发射。
- `b11fix.onnx` 的 16 个局面与 ORT FP32 对拍 `RESULT: PASS`，policy top-1 16/16。
- B11 的 GTP/analysis、错误配方拒绝、实际 batch 容量与 ExpectedProfile 校验共 14 项集成检查通过。

真实 Worker 实验构建 r4 SHA-256 为 `2670004342ae6c66fef9b53a17ed7ce38cf494bc7ee8f532dd6bc364cecf366a`。B11/B15 均在容量 8、请求窗口 1/32 下完成固定 128 个回归请求。该集合是 **16 个语义局面 × 8 对称**，不能视为 128 盘独立棋局或完整留出集。

| 模型/候选 | 相对同模型 FP16 最大胜率偏差，窗口 1 / 32 | 原 FP32 门 |
|---|---|---|
| B11 全 FFN INT8 | 1.986 / 1.921 个百分点 | FAIL，符合已知有损身份，不改门 |
| B11 四投影混合 | 0.680 / 0.672 个百分点 | FAIL，同上 |
| B15 全 FFN INT8 | 1.096 / 0.708 个百分点 | FAIL，同上 |
| B15 四投影混合 | 1.420 / 1.431 个百分点 | FAIL，同上 |

两模型的新旧纯 FP16 均通过原 C++ FP32 门。窗口 1 的纯 FP16 / 全 FFN INT8 新旧输出全部逐位一致；并发追踪中，按输入 SHA、物理 batch 和行位置匹配后，B11 INT8 91/91、B15 INT8 106/106 个匹配请求的原始五张量 SHA 也完全一致。其余请求的 batch 或行位置不同，不能据此宣称逐位等价。

B11 首次无追踪并发运行曾未通过新旧 INT8 迁移门（policy / ownership 超门），保留 `worker-b11/report.json` 的失败状态。追踪复验通过且固定 B1～B8 逐位一致，支持“不同请求落入不同物理 batch”解释，但没有反推首次运行的实际分组，**不将该失败改写为通过**。追踪运行只供诊断，不作性能证据。

对应证据为 `target/unified-quant-g1/worker-b11-trace/`、`worker-b15-trace/`、`target/unified-quant-g1-fast-paths.log`、`target/unified-quant-g1-onnx-fp32.log` 和 `target/unified-quant-g1/smoke-b11/`。后续独立审计另发现统一后端 Graph 捕获失败仍可回退 direct 的身份缺口，已改为统一入口明确失败，旧后端保留原行为；该修正后的新二进制需单独记录构建及检查，不能冒用 r4 身份。

进一步使用现有 PADBATCH 的诊断仍未通过严格迁移门：该 tactic 保留 B1 单例，不能保证不同运行的请求都落到相同物理 batch。`worker-b11-pad-trace/row-independent-raw-diagnostic.json` 中 128 个输入的 126 个原始输出完全相同（包括只改变行位置的请求），仅两个请求从 B1 换为 B8 或反向，且仅这两个输出不同。此轮明确定位到不同 batch 的比较，原失败报告照常保留；不把放宽数值门用作解决办法。

Graph 捕获修正后的 r5 二进制 SHA-256 为 `eadce66b7f0c37def087af431e2b453ffd310389d2ec56d4a8c377a84ea92b68`，B11/B15 各 14 项 GTP/analysis/身份集成检查通过，报告为 `target/unified-quant-g1/smoke-{b11,b15}-r5/`。

其余证据位于 `target/unified-quant-g1/`、`target/unified-quant-g1-gpu-r2.log`、`target/unified-quant-g1-schema.log`、`target/unified-quant-g1-freeze.log`。这些都是阶段性结果；尚无该统一后端的完整 PTQ 留出验证、ABBA 性能收益或棋力认证。未替换正式二进制或原计划，未声称完整目标已完成。

MXFP8 独立原型位于 `experiments/mxfp8_probe.cu`。首轮对预量化 E4M3 数据和块 32 UE8M0 scale，使用实际 cuBLASLt 块缩放 GEMM，覆盖尾维填充、窄输出、FP32 输出与 beta=1 残差；六组 GPU 用例逐位对拍及保护区检查通过。首轮冻结二进制为 `target/mxfp8-probe/mxfp8_prequantized_v1.exe`，SHA-256 `a09a9dbd948edeb129277d16903ed02c0ee58ab442eede497d858f01aeeb487b`；随后加入实际量化器并单独构建，不能混用两个阶段的证据。该原型不证明模型量化误差或端到端提速。

动态量化扩展初跑为 8 PASS / 2 FAIL，原因是原型默认流上传和 nonblocking 计算流缺少依赖；将初始化、上传、计算及检查绑定同一显式 stream 后，CPU oracle 和 GPU **10/10** 用例通过。日志保留在 `target/mxfp8-probe/quantizer-{cpu,gpu}.log` 与 `quantizer-r2-{cpu,gpu}.log`。新增用例涵盖原 FP32 权重、half 激活边界、RNE/SATFINITE、零块/尾维/非有限拒绝，以及动态量化→GEMM 的独立缓冲 Graph A→B→A。FP8 数据与 scale 逐位验证；随机 GEMM 使用独立 decoded FP64 参考及 FP32 累加误差界，量化误差另报。尚未接入生产执行器，尚无完整模型精度或性能结论。

通过的动态量化原型为 `target/mxfp8-probe/mxfp8_probe_quantizer_r2.exe`，SHA-256 `25aa63b77e1c3f3a98399a782da9e39e61234e49b1bbf8692d80884d91181663`。整网接入需保留同流与生命周期约束，并改为每次 forward 仅清零一次非有限错误状态，各投影只作 sticky OR，避免后层覆盖前层错误。Lt descriptor 持有 scale 指针，不能只按形状复用而混用不同权重。

### 8.1 语料与采集工具

`cargo run -p kata_nn --example sgf_quantization_corpus -- <SGF目录> --output <新目录> --seed rustgo-ptq-v1` 在 CPU 上完成原生 SGF 解析、完整合法历史检查、整盘划分和跨集合去重。保留规则、贴目、初始子和初始行棋方；缺少必要元信息时拒绝，只有显式指定的假设才可使用且写入 provenance。当前仅接纳已支持的 19 路 Chinese Worker 规则。完整请求在各 split 的 `*.requests.jsonl`；旧脚本兼容子集另存 `*.worker-fixture.json`，不伪造不兼容局面的初始状态。

CPU 测试 7/7 通过。Windows 调试构建处理完整棋局时曾主线程栈溢出，按现有引擎惯例使用显式 256 MiB 工作线程栈后修复。真实两盘本地 SGF 导出 `target/ptq-corpus-two-games-r2/`：calibration 32 个点、selection 31 个点（跨盘重复点剔除并留记录）、holdout 0，状态 **NOT_READY**。这仅证明工具链可用。官方公开评级棋谱归档直接下载返回 HTTP 403，未取得该数据；尚无合格的校准/选型/留出语料。

`scripts/collect_quantization_corpus.py` 直接消费完整请求，通过临时本机真实 gRPC Worker 采集；逐请求 `skip_cache=true`，生成配置关闭 NN cache，并核对实际 NN 完成行数，避免缓存误当推理。可用 `--config` 采集现有 CUDA/INT8 基线：保留该配置的 batch、NN server 线程、cache 和 INT8 范围，冻结并绑定实际加载的 tactic plan；配置内容、执行文件或身份不符时拒绝。导入配置即使保留 cache 容量，请求仍显式跳过缓存。记录模型、二进制、实际精度身份、请求/来源/协议 SHA、原始 protobuf 结果及 JSONL，允许后续按 semantic SHA 对齐，不能按异步完成的行号对齐。当前 collector 只保存最终输出，不产生逐层激活校准统计，也不拟合 PTQ 参数。

r5 下 B11 的全 FP16、全 FFN INT8、四投影混合配方均完成该 32 点 calibration 请求的采集，结果在 `target/unified-quant-g1/corpus-b11-{fp16,ffn,mixed}-r5/`，均为 `COLLECTED_UNVERIFIED`。这不是独立留出精度通过；后续比较器必须检查完整 provenance 与原始浮点值，并将样本不足的结果保留为诊断状态。

`scripts/compare_quantization_corpus.py --reference <FP16采集目录> --candidate <候选采集目录> --output <新目录>` 已实现原始 protobuf 浮点比较、纯 FP16 配方规范哈希校验，以及来源/请求/输出的完整性核对。32 点自比较全部逐位相同；改写结果、混淆请求、以 INT8 冒充 FP16 参考均被拒绝。B11 全 FFN INT8 与四投影混合的最大胜率偏差分别为 1.719 / 0.911 个百分点，B15 全 FFN INT8 在窗口 32 下为 0.770 个百分点，结果都仅为 `DIAGNOSTIC_PASSED`；policy、目差、ownership 见对应 `corpus-*-compare-*` 报告。所有这些结果都来自同一盘 32 个 calibration 局面，不能用于最终配方验收或证明泛化。

Worker 的 `capacity` 是协议 `max_in_flight` 硬上限，和 `nnMaxBatchSize` 的物理 batch 容量不同。一次误用 `--capacity 8 --window 32` 的采集因 `CAPACITY_EXCEEDED` 如实失败并保留报告；正确组合为 `--capacity 32 --window 32 --batch 8`。collector 在参数阶段拒绝请求窗口大于 Worker capacity，避免启动 GPU 后才发现错误。

### 8.2 有界候选与后续选型

`scripts/plan_quantization_search.py` 是纯 CPU 研究计划生成器，输入原模型、`quant-inspect` 的完整 manifest/FP16 模板、语料 manifest，生成配方和 `search-plan.json`。先核对原始文件 SHA、投影 ID/形状/完整性与全部语料来源，冻结完整 selection 请求；不读取模型输出或 holdout 输出，不拟合参数、不推理、不测性能。真实图摘要仍由后续 Rust loader 从模型重新校验，Python 检查不能替代它。

首批仅生成全 FP16、全 FFN INT8、FFN hidden≥384 INT8 三个确定的种子，并按规范配方 SHA 去重。B11 得到 2 个唯一候选（后两个相同，33 组 FFN）；B15 得到 3 个候选（分别 0 / 45 / 12 组 FFN），attention 均保持 FP16。这些是试验起点，不是自动选出的优胜方案。

最终初始账本位于 `target/unified-quant-g2-search-{b11,b15}-r2/`，固定最多 8 个候选、2 个进入完整比较的 finalist、1 个冻结后接受独立 holdout 的最终候选。当前语料不足，账本均为 `BLOCKED_CORPUS`、`publish_allowed=false`。21 项 CPU 正负例、重复生成逐字节一致、输出 SHA 和已有目录拒绝均通过；规范配方 SHA 与现有 Rust 解析结果一致。

后续用实测完整 RMSNorm+FFN 操作组耗时安排少量分组回退候选，再以完整前向、真实 Worker 和 local 搜索的顺序 ABBA 决定采纳。逐层 profile 的同步开销会扰动实际执行，只作候选缩减；不能用 MAC 数、宽度阈值或单个 GEMM 的峰值代替端到端证据。最终比较必须纳入本机对应负载已验证的 FP16 和 INT8 配置，见 `docs/int8-gemm-evidence-20260923.json` 及既有 FULL/Worker 认证报告。

新增 `scripts/quantization_selection_ledger.py` 的四个 CPU 子命令为 `init/status/ingest-numeric/ingest-performance`。初始化重建 planner 完整契约并冻结源文件，登记数值时重读原始 protobuf，登记性能时重算四臂原始 timing/计数/执行文件/实际 profile 和 ABBA 判断；独占锁、顺序哈希链和证据快照记录失败，已预留的失败尝试消耗预算，改名或复制不能恢复额度。19 项 CPU 正负例通过。哈希链不是外部签名，需另存 head SHA 并通过 `--expect-head` 检查截尾回滚。

首版仅接受 v1 FP16/INT8 种子和统一后端实际 profile，不生成新候选，不登记 legacy 最优 baseline 资格，不提供 finalist/holdout/自动发布。真实 r8 B11 的新 `target/unified-quant-g2-ledger-r1/real-r8/` 已登记纯 FP16 与全 FFN INT8 各 32 个连续采集请求；原始 protobuf 重算通过，INT8 最大胜率差 1.71948 个百分点。固定 calibration/SMOKE、C1/capacity8、warmup128/cycles32，消耗 2/2 数值预算、0/1 性能预算，holdout 读取 0，仍为 `SMOKE_ONLY`。最终 head 为 `438b05ea45ea82f2defae5d3b8a0f466a9242cadbd270f98e3b152c5cd2f86e9`。这证明真实证据可登记，未完成自动实测选优；G3 的 v2 诊断配方也不能冒称已纳入该 v1 账本。

后续新增纯 CPU `summarize-selection --ledger <目录> --expect-head <独立保存的head> --output <新目录>`。从账本快照重读 PB、原始四臂 timing 和实际执行文件，不消耗预算或修改 head，输出 `EVIDENCE_INCOMPLETE`、`BASELINE_RETAINED` 或 `CANDIDATE_REQUIRES_CONFIRMATION`。未测候选不排名，失败或耗尽预算不提示原样重试；正收益候选也不授予最佳基线、独立确认、holdout 或发布资格。`SMOKE_DIAGNOSTIC_ONLY` 不能正式采纳，输出始终带完整缺口与身份。

33 项 CPU 检查通过。真实 r8 B11 两份原 numeric 证据在新 `target/unified-quant-g2-ledger-r2/real-r8-current-sources/` 重新登记后，summary 正确报告缺性能证据，head 与预算前后完全相同（2 numeric / 0 performance / 0 holdout）。新 head 为 `4c4827a5e75c488e7e6433bc0b70ac8ff66e1121b8bef26eda822ed56524af61`，见 `verification.json` 和 `summary/selection-summary.json`。初次使用旧 helper SHA 的计划被拒绝，失败保留；随后只重新生成当前源码的 CPU plan，模型、语料及候选配方未变，未补跑 GPU，也未改写旧账本。

### 8.3 完整请求 Worker 测速工具

`scripts/benchmark_quantization_corpus.py` 消费完整 FP16 参考、基线和候选采集目录。在任何 GPU 测速前，重新验证三者的模型/配方/请求/结果 provenance，并要求两侧相对同模型 FP16 的完整采集输出通过 6 个百分点实验门。拒绝读取 holdout 进行性能选择；两侧数值采集的请求窗口和协议容量必须与测速一致。正式输入只接受合格 selection 集；`--smoke` 仅用于工具检查，始终不能发布。

同一完整语料循环和相同并发，依次启动、预热、测量和结束四个独立 Worker，固定顺序 ABBA；临时服务隔离且逐请求跳过缓存。计时包含 RPC、特征处理、推理以及首批填充/尾批排空，不包含日志落盘。报告吞吐、P50/P95、NN 行数与批次数和主机开销；不将逻辑平均 batch 冒充物理 batch 分布。该工具当前未采集峰值显存，也不证明基线已达到该负载的最优状态。

每臂开始及结束复核二进制、模型、实际配置、配方/计划与工具源码 SHA。配置引用的执行文件必须确实指向冻结 artifact；只验证复制文件而仍加载原绝对路径的情况会拒绝。CPU 负例覆盖旧绝对路径、未登记执行文件与文件篡改。旧 FP16 plan 仍由原 loader 做认证身份检查。

两次小语料冒烟流程通过：`target/unified-quant-g2-abba-smoke-b11-r1/` 为新入口 FP16/全 FFN INT8 的 C1 流程检查；`target/unified-quant-g2-abba-smoke-b11-c32-r2/` 为已有正式 FP16 B16 计划和新入口全 FFN INT8 B8，在相同 C32 下的流程检查。后者四臂为 1054.227 / 975.736 / 974.795 / 1070.757 RPC/s，候选在该短测中没有收益。两者都使用不足的 calibration 语料且测量短，状态为 `SMOKE_ONLY`，不作为正式性能或配方选型结论。历史报告保留，不舍弃不利样本。

外层选择器还须绑定有界候选账本、最佳已验证 FP16 与 INT8 基线、独立 ABBA 确认组、冻结后的一次 holdout，以及 local 搜索与显存证据。当前工具的 `deployment_adopted`、`publish_allowed`、`production_certified` 均固定为 false。

采集器新增 `--window-mode continuous`，默认仍为原 `chunked`。连续模式在基本结果身份/错误检查后立即补发，完整 protobuf 留存于有界缓冲，成功 Drain 后才进行详细数值校验、序列化和落盘；失败先停止本轮自有 Worker，再保留已收到的原始结果与故障消息。默认 256 MiB 限制的是 protobuf payload，不是进程 RSS；另有完整请求数上限。每请求独立截止时间，心跳和其它结果不能无限延长丢失请求的等待。

`continuous-buffer.json` 绑定每个任务的原始结果 SHA/大小、累计 payload 和窗口边界；比较器与账本共用验证器，旧记录缺少 `window_mode` 时解释为 `chunked`，不能只改标签冒充连续采集。正式 Worker ABBA 在解析输出及启动 GPU 前要求参考/基线/候选均为 continuous，旧 chunked 证据只允许 `--smoke`。连续模式缩小采集与测速的调度差别，仍不证明不同运行的物理 batch 或计时完全相同。10 项 fake-Peer/clock CPU 检查及旧 r5 的 32/32 自比较通过，见 `target/unified-quant-g2/continuous-cpu-check-r1/`。

### 8.4 MXFP8 Rust/CUDA 算子模块

独立生产算子模块为 `crates/kata_nn/src/backends/mxfp8.rs` 与 `cuda-kernels/mxfp8.cu`，复用本仓库 cudarc/CUDA runtime。已在 RTX 5070 Ti、CUDA 13.3 / cuBLASLt 13.6 环境完成构建与 `test_mxfp8` 验证：2 项 CPU 参考自检、5 项显式 GPU 测试均通过；默认未设置 `KATAGO_TEST_MXFP8=1` 时 GPU 测试只跳过，不能将该运行记成 GPU 通过。

覆盖原始 FP32 权重量化、half 激活、FP8/scale 字节、K31/32/33/65/344 和非紧凑 stride、三种输出、不同权重 scale 绑定、变 K 共用 scratch、非首层 sticky 错误、逐槽 pinned 状态快照，以及保活 token 下的 Graph A→B→A 和工作区 wrapper 释放后的重放。原六组可精确表示矩阵的最终存储逐位一致；随机 GEMM 按预先固定的 decoded FP64/FP32 累加界检查，量化误差单独报告。

热路径不分配、不选算法、不同步；准备阶段用 `complete_warmup()` 同步验证当前 forward 的真实状态后才允许捕获。每次 forward 清零一次错误状态，各投影只能追加错误；每个在途输出槽拥有独立状态快照，后续 forward 不得抹除前一批的失败。cuBLASLt 实际算法的 numerical flags 验证 E4M3 输入、Tensor Core 类执行和 FP32 累加，关闭 FAST_ACCUM；这不是 Tensor Core 利用率或整网提速报告。

证据为 `target/unified-quant-g3-mxfp8-{build,cpu,gpu}-r1.log`。该轮测试二进制 SHA-256 为 `6744cf935687c22a050f8a6b5d4aa59f8f118316eb23609eb99be0932cd266ee`。此处记录的是独立算子证据；后续执行器、配方与调用链的新增修改须另记整网验证，不能由算子通过推导模型精度通过。

随后新增逐投影 MXFP8 整图接入、每物理 batch 预备描述符、异步双槽状态保存和 nnbench 逐轮状态环。MXFP8 源配方须显式使用 `(version=2, quantization_semantics_version=2)`，规范身份加入固定数值契约及原 FP32 权重位模式摘要；FP16/INT8 的 v1 路径不变。r7 的 15 项配方 CPU 测试、4 项 quant-inspect 测试通过；此前 6 项 nnbench admission/状态历史/barrier 测试通过。

整图首轮 `target/unified-quant-g3-graph-r1.log` 如实失败：B11 全 FFN MXFP8 的 B1 Graph A→B→A 通过，但 NaN 拒绝后的合法输入恢复仍失败。定位到原手写 stem GEMM 的 `beta=0` 仍读旧 C，`0×NaN` 使共享 `gemm_out` 持续污染；不是将错误状态清零即可修复的问题。修复前主二进制冻结在 `target/unified-quant-g3/frozen-r6-before-beta-fix/katago-rs.exe`，SHA-256 `81ea5ff7d4ebee249bd3f73030ad0b5b7c96aa04d4db901b780cdc71a8f10780`。该失败记录保留。

修复后的 r7 冻结在 `target/unified-quant-g3/frozen-r7/katago-rs.exe`，SHA-256 `2b0904a13b11deaa76ba8ae49f8a81d48e5b712a1ad484d8dce5e7701e4a1863`。手写 GEMM 的 beta=±0 路径不再读旧 C，非零 beta 保留原表达式。有限输出中的有符号零可能因此规范化，不能假定所有输入都逐位等价。实测证据如下：

- `test_beta_zero` 的 3 项 GPU 回归通过：五种普通 tile 的 N1/131/136、融合 gate/SiLU/SwiGLU、Split-K 和融合 RMS reducer，覆盖 ±0 beta、NaN/Inf 旧 C、独立可精确表示矩阵参考及非零 beta。r1/r2 测试曾因旧 half 辅助转换器把 NaN 当成 Inf 而失败；改为直接检查 IEEE binary16 指数/尾数后，`target/unified-quant-g3-beta-zero-r3.log` 通过，未改变生产转换器。
- 三模型 × 全 FFN MXFP8 / 全 Transformer MXFP8 / FP16+INT8+MXFP8 三精度配方 × B1/B8，独立 workspace 的直连与 Graph A→B→A 逐位相同，NaN forward 拒绝后恢复合法 A 也逐位相同。证据为 `target/unified-quant-g3-graph-r2.log` 及同名报告目录。
- 公共 Backend 的双槽流水线在 Graph/direct、三模型、B1/B8 下通过：先提交坏批再提交好批，等待好批 GPU 完成后才读取坏批，坏批仍拒绝且不修改调用方输出哨兵，好批与独立 handle 逐位相同。见 `target/unified-quant-g3-pipeline-r1.log`。
- 旧 FP16 / 全 FFN INT8 / 全 Transformer INT8 的三模型 B1/B8 迁移对拍及混合 Graph 回归通过。ONNX B1/B8 各 16 局面再次通过原 ORT FP32 门，policy top-1 各 16/16；与修复前对应 batch 的共 192 个输入/输出文件逐字节相同。见 `target/unified-quant-g3-legacy-recipes-r1.log`、`target/unified-quant-g3-onnx-fp32-b{1,8}.log` 和 `target/unified-quant-g3/beta-zero-numeric-r7.json`。

这些检查证明调用路径与错误恢复正确，不是 MXFP8 相对 FP16 的模型精度、性能或棋力验收。原正式二进制及认证计划未替换。

r7 的 B11/B15 GTP、analysis 和身份准入各 15 项通过，见 `target/unified-quant-g3/smoke-{b11,b15}-mxfp8-r7/`。独立 beta=0 修复 ABBA 为旧 r6 与 r7、同一 ONNX/固定 B1/B8/单 handle/400 次完整前向，不是 MXFP8 测速。B8 几何均值 784.912→787.265 rows/s（+0.30%，未达 1%）；B1 为 283.771→262.905（−7.35%），候选两臂波动 11.37%，稳定性不通过。原始样本全保留于 `target/unified-quant-g3/beta-zero-abba-r7/`，`deployment_adopted=false`；不将该轮记为性能通过，也没有替换旧正式内核二进制。

后续独立审查发现 r7 还缺少最后 MX 投影之后 FP16 输出头的非有限值终检，以及公开算子对同 raw stream handle、不同 CUDA context 的区分；异常取消时最后一个 compute handle 的销毁也需先等待在途工作再释放混合精度模型权重。这些边界由下述 r8 单独修正，r7 上述通过项不代表它们已验证。

`quant-inspect --recipe <文件>` 新增 CPU 导出，四个 artifact 包括精确源 recipe 字节和带固定字段顺序的 `recipe-identity.json`；在创建新输出目录前先完成校验。B11/B15 的全 FFN MXFP8 与只显式指定 MXFP8 dual + INT8 down 的稀疏配方，共 4 份真实导出由 Python 重新补全默认 FP16 并计算规范摘要，与 Rust 全等；66 个篡改负例拒绝。旧 v1 模板/manifest 保持逐字节一致。见 `target/unified-quant-g3/cpu-recipe-exports-r7/cpu-export-audit.json` 和 `target/unified-quant-g3/python-identity-verification-r7/report.json`。Python 校验器没有独立解析原模型权重来重算源 FP32 位模式摘要；该边界由同一冻结二进制的 CPU 导出、原模型 SHA 和后续 GPU loader 的实际 recipe SHA 共同绑定，不宣称完全独立的权重解析验证。

### 8.5 r8 完整输出与资源生命周期

r8 主二进制为 `target/unified-quant-g3/frozen-r8/katago-rs.exe`，SHA-256 `6413e963b557023bc4119e7d98a9728630e976477221c99c779aa557296e9309`。MXFP8 完整前向在所有输出头之后、状态快照/完成事件之前，增加一次融合 GPU 扫描，覆盖 policy/value/misc/moremisc/ownership；仅追加 sticky 错误，最后输出错误标记为 `final_raw_heads`。热路径没有新增同步或分配，非 MX 配方不执行该扫描。

公开 MXFP8 接口同时核对 `cu_ctx` 和 `cu_stream`；仅比较默认流 0 会混淆不同 context。compute handle 析构在释放任何字段前同步自身流，避免未完成推理继续读取已释放的非 MX 权重；析构失败如实记录且不 panic。窄范围独立审查通过。

- 新 context 测试实际在同一 GPU 创建 non-primary context，以相同 raw 默认流 0 验证输入、两种输出、最终输出扫描和 pinned 状态均被拒绝，合法 owner 控制通过。当前 cudarc 的 safe 构造函数未覆盖所用 feature，首次测试构建失败保留；测试改用已绑定的 Driver API 构造再交回 cudarc 管理，未修改生产依赖。
- 完整模型新测试使用具有独立测试身份的全有限合成末端权重，真正产生 `MAX×2` 溢出，确认 apply 尾部检查会拒绝；逐一覆盖五个输出头的 NaN/Inf、Graph A→坏输入→A 和独立 pinned 快照。三模型三类 MX 配方和旧 FP16/INT8 路径全部扩展到物理 B1～B8，通过独立直连/Graph 逐位比较及恢复检查。见 `target/unified-quant-g3-graph-r8-r2.log`。首次将三个集成 case 并行运行时，共享 primary context 的全设备同步干扰另一 case 捕获并失败；加入测试间互斥后使用默认测试线程设置通过，保留首轮日志，不归因于模型数值。
- Graph/direct、三模型、B1/B8 的真实双槽隔离再次通过；六个末尾取消测试都实际记录 `pending_when_dropped=true`，未调用 finish 直接销毁最后模型/运行时持有者后无异步错误。见 `target/unified-quant-g3-pipeline-r8.log`。
- 独立 MX 算子 2 CPU + 5 GPU、context 1 GPU 通过。nnbench 的 kernel wall/CUDA-event、双 handle direct 和旧 FP16/INT8 控制共 5 个短流程通过，逐轮状态环记录完整；该三次迭代冒烟不作性能证据。见 `target/unified-quant-g3/nnbench-mxfp8-smoke-r8/`。
- ONNX B1/B8 各 16 局面通过原 FP32 门，192 个原输入/输出文件与 r7 逐字节一致。见 `target/unified-quant-g3/fp32-and-legacy-bits-r8.json`。

r8 验证覆盖执行、恢复与生命周期，不等价于完整语料的精度通过或部署收益。

r8 B11/B15 的 GTP/analysis/身份准入各 15 项通过。在同一 r8、容量 8、连续窗口 1、关闭结果缓存并启用 batch trace 的条件下，六组 Worker 收集及四组比较完成。每组 32 个请求均实际为 B1；候选与对应 FP16 参考的 encoded input、物理 batch、行号逐一配对 32/32，四份 v2 CPU 规范身份与实际 Worker loader 相同。独立重读原 protobuf 复算结果如下：

| 模型/配方 | 胜率偏差最大/平均/P95，百分点 | 最大目差 | policy top-1 | ownership RMSE |
|---|---|---|---|---|
| B11 全 FFN MXFP8 | 4.4454 / 1.2043 / 3.7108 | 1.4054 | 32/32 | 0.007514 |
| B11 一个 FFN dual MXFP8 + down INT8，其余 FP16 | 0.7360 / 0.2104 / 0.5651 | 0.3017 | 32/32 | 0.000972 |
| B15 全 FFN MXFP8 | 2.1947 / 0.4688 / 1.9721 | 0.4149 | 32/32 | 0.006217 |
| B15 一个 FFN dual MXFP8 + down INT8，其余 FP16 | 0.0841 / 0.0148 / 0.0697 | 0.0580 | 32/32 | 0.000223 |

四组均为 `DIAGNOSTIC_PASSED`。完整 policy KL/误差、逐例结果、模型/recipe/profile/源码 SHA 和连续缓冲证据见 `target/unified-quant-g3/corpus-diagnostic-summary-r8.json` 及其链接的采集/比较目录。两个模型共用的仍是**同一盘棋的 32 个独立局面**，不是 192 个独立样本；没有拟合校准参数、独立 holdout、原 FP32 有损候选认证或棋力结论。另补同条件全 FFN INT8 控制：B11 最大 1.71948、B15 最大 0.94474 个百分点，均仅诊断通过。

### 8.6 r8 MXFP8 与 INT8 的端到端诊断比较

同一 r8 二进制、统一后端、全 FFN 配方、最大 batch 8、单 NN server 线程，在 C1 与 C32 分别比较 INT8 基线和 MXFP8 候选。每侧先完成相同连续请求窗口的 FP16 数值参考及候选采集，再运行固定 ABBA；每臂预热 128、测量 1024 请求，均跳过结果缓存。C1 预先指定 P95 延迟为主指标，C32 指定 RPC 吞吐；不得用另一个更好看的指标替换判断。

| 模型/并发 | 指标（两臂几何均值） | INT8 | MXFP8 | MXFP8 相对变化 |
|---|---|---:|---:|---:|
| B11 / C1 | P95 往返延迟，µs | 2858.232 | 3340.242 | 延迟增加 16.864% |
| B15 / C1 | P95 往返延迟，µs | 3600.351 | 3980.751 | 延迟增加 10.566% |
| B11 / C32 | RPC/s | 995.084 | 683.371 | 吞吐下降 31.325% |
| B15 / C32 | RPC/s | 709.703 | 595.370 | 吞吐下降 16.110% |

四组两侧波动均低于 5% 门，候选均无收益，不作采纳或原样重测。P95 的延迟增幅与其倒数的速度降幅不同，上表采用直接延迟比；完整原始四臂、执行文件和源码身份见 `target/unified-quant-g3/abba-{b11,b15}-mxfp8-vs-int8-c{1,32}-r8/report.json`。另以原始 run 记录复算时间、计数、几何均值并核对执行文件 SHA，汇总为 `target/unified-quant-g3/mxfp8-vs-int8-summary-r8.json`。

C32 数值采集相对该窗口的纯 FP16 参考：B11 INT8/MXFP8 最大胜率差为 2.11709 / 5.26454 个百分点，policy top-1 为 32/32 与 31/32；B15 为 0.76967 / 2.34865 个百分点，二者 top-1 均 31/32。完整目差、policy KL、ownership 和逐例值保留在对应比较报告。数值采集的 trace 不代表未启用 trace 的计时运行具有相同物理 batch；平均逻辑 batch 不作为物理 batch 分布报告。

这些是同一盘 32 个 calibration 局面的诊断，全部 `SMOKE_ONLY`，不是最佳 legacy FP16/INT8 对照、独立 holdout、local 搜索或棋力验收。当前实现的全 FFN MXFP8 在这两种条件下应保留为关闭的候选；该结果不否定少数层混合配方或未来融合实现，但后者须以独立数值和性能证据重新判断。

### 8.7 r9 缩放缓冲清零融合实验

r9 冻结主程序为 `target/unified-quant-g3/frozen-r9/katago-rs.exe`，SHA-256 `40f1c226ff6f0e7b0a17892ee0de111fb1092afe04d5842f7e5862518e908dd0`。新增 `KATAGO_CUDA_MXFP8_SCALE_CLEAR_FUSION=0|1`，默认 0，走 `tactic_var()`，纳入进程冻结及实际执行 profile；没有 MXFP8 投影时不能启用，也不能借旧 FP16 plan 启用。配方数值契约与 recipe SHA 不变。

融合只影响 half 激活量化：通过 128×4 打包布局的逆映射，量化 kernel 清除整个最大 scale 分配中当前非活动的字节，与 lane 0 写入的活动 scale 严格互斥。FP32 权重仍走原路径；FP32 amax/舍入、half 边界、GEMM、逐投影及最终输出扫描不变。每次量化少一次清零 kernel 启动，热路径不增加分配、同步或算法选择；启动次数减少不是收益证明。

数值与执行证据为 `target/unified-quant-g3/scale-clear-validation-r9.json`，保存源文件、测试程序、模型及主程序 SHA：

- 4 项新 tactic CPU 准入测试通过；独立反向地址集合审查覆盖全容量和大 K→小 K 尾部。
- 新算子测试含 2 项 CPU oracle 与独立进程 flag 0/1 GPU 比较，每路径保留 16,311,877 字节完整证据并逐位相同。覆盖 102 次形状/K 往返、完整 FP8/scale 分配、非紧凑 stride、三种输出、Graph A→B→A、非首层 NaN/±Inf sticky 与恢复。首次测试构建因错误 import 路径失败；仅修正测试 import 后构建通过，原失败日志保留，未改生产语义。
- 三模型 × 全 FFN / 全 Transformer / 三精度配方 × B1～B8，r9 flag 0、flag 1 与 r8 九组报告的原始输出逐位相同。两种 flag 的最终五头非有限值检查均通过，flag 0 的旧 FP16/INT8 回归通过；flag 1 的跨 context 拒绝和双槽 Graph/direct/取消生命周期检查通过。
- collector 新增显式 `--mxfp8-scale-clear-fusion 0|1`，清除继承环境后记录指定值，并核对 CPU 配方、实际 profile 与实际 launch 标记。11 项 CPU 工具测试通过；计时继承该已验证环境但去掉 batch trace。

同一 r9 flag 0/1 的四组初始 ABBA 已完成，每臂预热 128、测量 4,096 请求。C1 以 P95 延迟为主指标，C32 以 RPC/s 为主指标；只有通过初组的 B11 C1 增加一次独立确认，其它组没有原样重测。

| 模型/负载 | 关闭融合 | 开启融合 | 主指标速度比变化 | 裁决 |
|---|---:|---:|---:|---|
| B11 C1，P95 µs，初组 | 3299.487 | 3255.711 | +1.345% | 进入确认 |
| B11 C1，P95 µs，确认 | 3324.372 | 3275.235 | +1.500% | 同 MX 配方的局部诊断收益确认 |
| B15 C1，P95 µs | 3925.168 | 3889.225 | +0.924% | 未达 1% |
| B11 C32，RPC/s | 685.808 | 684.154 | −0.241% | 无收益 |
| B15 C32，RPC/s | 594.955 | 594.675 | −0.047% | 无收益 |

全部两侧波动低于 5%。B11 C1 的直接延迟降幅分别为 1.327% / 1.478%，与表中倒数速度比区分。完整证据见 `target/unified-quant-g3/scale-clear-measurements-r9.json` 与 `scale-clear-b11-c1-confirmation-r9/report.json`。两模型 W1 的 32/32 个最终处理后输出逐字节相同；W32 按相同 encoded input、物理 batch 和行号配对，B11 18/18、B15 28/28 个原始输出摘要相同，未配对项不冒充逐位相等。

独立 CPU 审计重算 20 个计时臂的 81,920 条原始记录，以及 12 组 collection、8 组原始 PB 数值门，全部吻合；见 `target/unified-quant-g3/scale-clear-audit-r9.json`（SHA-256 `599d992da1a8ac65dc1ff67146f845903f80914a5fd4be2786cdbae22085bfec`）。该索引保存审计时现存报告/日志/timing 的 SHA，不冒充新增的运行时签名。C32 始终指请求并发 32，最大 NN batch 仍为 8。

开关保留供显式实验，默认仍为 0；仅 B11 C1 观察到达到门限的可重复局部收益。以上验证同一 MXFP8 配方实现的局部效率，不改写 r8 与 INT8 的负收益记录，也不能替代完整语料和最佳基线比较。所有组仍为 `SMOKE_ONLY`，32 个原始 calibration 局面的重复循环和再次 ABBA 不增加独立精度样本。

可复制的本机加载、配方导出和隔离 Worker 命令见 [统一后端实验使用](RustGo统一后端实验使用.md)。

### 8.8 自动选型接入前的旧基线核对

`target/unified-quant-g2-legacy-baseline-audit-r1/report.json` 是四个历史执行方案的 CPU 清点，SHA-256 `16d6b5d94caf3e43cd01b1bd23a34f633fd814f4f01badaf6ae010819aae3c9b`。脚本实际读取二进制、模型、配置、plan 与历史报告，并核对 SHA；未启动 GPU，也未把历史成绩重新登记为本轮通过。

- B11 FP16：`configs/worker_tf3_sm120_c32.cfg` 对应物理最大 B16、单 NN server 线程与 C32 请求负载；正式二进制仍为 `d218b98d…`，plan 的迁移范围保持原声明。
- B15 FP16：`autotune-output/pruned-worker-plan-20260920/result/` 对应 B8、单线程，但历史测试为 **C64、CUSTOM_GROUPS**。它是需要加入比较的已有方案，不能由此宣布是 C1/C32 或全局最优。
- B11/B15 旧 INT8：`target/int8-gemm-study/worker-light-{b11,b15}/` 的 q64 候选需要显式 `KATAGO_CUDA_ATTN_TILE=q64`、`KATAGO_CUDA_INT8_GEMM_TUNE=0`。清点时 collector 的 `--config` 保留配置，但会清除继承环境，且没有这两个参数的导入入口；直接套用旧配置会丢失候选执行条件，不能这样冒充旧最佳 INT8 对照。

后续基线接入须冻结允许的显式执行参数、核对真实路径，先在同一语料上重做数值资格，再用相同请求并发、输出头与计时口径比较。历史 INT8/B15 FP16 的计时请求为 `ownership=false`，不能与本轮完整输出的吞吐直接排位。当前 ledger v1 仅登记统一后端种子，清点结果不是它的已通过基线；统一 FP16 对照组也不代表已证明的最佳 FP16。没有候选证明收益时保留合格基线。

### 8.9 有限种子的自动执行与实机流程冒烟

新增 `scripts/run_quantization_search.py`：`prepare` 只用 CPU 重建计划并冻结源文件、二进制、环境、完整命令和预算；`run` 一次执行采集→比较→账本→ABBA→汇总。每次启动前先持久化尝试；候选数值失败登记后跳过其 ABBA，FP16 参考失败或工具运行错误停止。没有重试/续跑入口，也不会打开 holdout 输出或发布部署计划。Windows 私有 Job 在子进程门释放前完成绑定，退出时只清理自身子孙。

13 项 CPU 测试通过，含真实 Windows 子孙进程、超时清理、退出码/输出、Job 归属及无关哨兵存活。见 `target/unified-quant-driver-cpu-r1/tests-final.log`。冻结 driver SHA-256 为 `e92d55be60c81c51ba89193f0f77376603bc47ed503ec8d9001bff2843ba710f`。B11/B15 真实 CPU 准备分别产生 2/3 个去重种子；B11 本轮仅准备，没有另跑 GPU。

仅 B15 执行了一次完整接线冒烟，证据为 `target/unified-quant-driver-prepare-r1/b15/prepared/`。使用 r9、B1/C1/容量 1、每臂预热 1 和 32 个计时请求，主指标在准备阶段固定为 RPC 吞吐。3 次数值比较和 2 组 ABBA 完成，driver 为 `FINITE_EXECUTION_COMPLETE_REQUIRES_REVIEW`，汇总为 `BASELINE_RETAINED` / `SMOKE_DIAGNOSTIC_ONLY`。没有重试、不利样本丢弃或胜出确认。

同一盘 32 个 calibration 局面中，宽层 INT8 / 全 FFN INT8 的最大胜率输出差分别为 0.623265 / 0.944743 个百分点，最大目差为 0.264998 / 0.671834，policy top-1 均 32/32；完整 policy、目差和 ownership 指标保留在比较报告。微型计时仅用于验证工具链，本次不据此给出加速结论、采纳或独立精度认证。

账本头为 `ae4afd2b85a6b55fe41bf55854a564c1c67c3a8f7740744ef867778beb368194`，driver 头为 `b97412ad7305f33f76df797218d66c5e1bcb64ebfa9f8d81f054f30a0145bca0`，holdout 输出读取数为 0。操作说明与新的可复制实验命令见[统一后端离线选型](RustGo统一后端离线选型.md)。自动执行固定种子不等于完成了逐层搜索、历史最佳基线比较或最终发布流程。

独立 CPU 审计复算原 PB 数值、两组四臂 timing、47 个 driver 事件及 15 条命令，全部一致；prepared 下 846 个文件的哈希清单在审计前后完全相同。审计禁止启动子进程/GPU，见 `target/unified-quant-driver-prepare-r1/b15/audit-r1/audit.json`（SHA-256 `5076f9d30009c5df55c09d996d2837dcc6024ea1a873e46001ecb670cb4b2654`）。原正式二进制 SHA 和原生产 Worker 存活检查另保存在 `post-run-system-check.json`。

### 8.10 公开实战语料的来源与原字节检查

本轮下载并保留来源/原包 SHA 的两个小包，仅做 CPU 检查，未读取任何模型输出。原两盘棋语料及所有已完成实验的输入保持不变。

- [OGS 发布者的 1,000 局研究样本](https://za3k.com/ogs/)：其中明确公开、已结束的 19 路 Chinese 棋局有 111 局，但记录的超级劫规则为 59 个 `ssk`、52 个 `csk`，与当前 Worker 的 Chinese/simple-ko 不同，没有直接导入或改写规则。
- [Fox Go Dataset 的 Pro2 小包](https://github.com/featurecat/go-dataset)：510,585 字节、680 份 SGF。最初 5 份抽样均为 Japanese，仅能说明抽样范围。随后完整扫描的 661 份完整语法树中，200 份明确 Chinese、461 份 Japanese；另 19 份缺少起始括号，根元数据旁证为 Chinese。不能据最初抽样否定整个包。提取前验证全部成员路径和 regular/directory 类型，拒绝路径越界与链接；原始字节完整保留。
- 当前 Rust SGF 解析器已在 `crates/kata_data/src/sgf.rs` 对 `AP[foxwq]` / `AP[SGFC:2.0]` 实现特定贴目映射，包含 325/650→6.5、375/750→7.5。原字节交给现有解析器使用该既有语义，没有手工修改 KM、RU、摆子或括号。

原字节直接交给 CPU builder 时接纳 **3 局、96 个 calibration 局面**；677 个拒绝中，621 个为非 UTF-8 编码、55 个规则不支持、1 个中途 setup/rules 变化。产物 `target/ptq-source-discovery-r1/fox-pro2-corpus-audit-r1/manifest.json` 仍为 `NOT_READY`，selection/holdout 均空。该次直接导入没有转码或修补内容；尚不能用这 3 局替代独立留出集。

该次索引为 `target/ptq-source-discovery-r1/discovery-final-full-r2.json`，SHA-256 `a7124afc4e368d86b056af7855164cb70da67e4ccbdca8473d168cfe75d699ee`。它取代最初仅抽样时“没有确认新语料”的结论，保留原始报告。证据包括下载记录、原包、完整元数据检查、builder 日志及原始 SGF；builder 的现存二进制 SHA 与当前源文件 SHA 分别记录，不冒充已证明二者的编译对应关系。该结果为后续数据导入定位了编码边界，不构成模型精度通过或棋力认证。

### 8.11 缺省编码的可逆导入证据

[SGF FF4 的 CA 定义](https://www.red-bean.com/sgf/properties.html#CA)规定，缺省字符集为 ISO-8859-1。针对 Pro2 包全部缺少 CA 的实际文件，按该默认值生成明确标记 `CA[UTF-8]` 的副本；没有猜测 GBK，也不声称恢复了人名和注释的历史真实编码。原始文件保持不变。

661 份完整单树全部通过严格字节反转、完整树与全部属性值等价，以及 NN 语义字段 ASCII 不变检查；19 份缺括号文件仍拒绝。现有未修改 builder 接纳 **93 份棋谱、2960 个 calibration 局面**，其余拒绝为规则不支持 281、中途 setup/rules 287。之前 3 份棋谱的 96 个请求语义全部保持，新增 90 份原非 UTF-8 来源。selection/holdout 仍为零，状态 `NOT_READY`，没有模型输出或 GPU 实验。

独立复核另用字节 tokenizer 验证全部 680 个源 SHA、661 个副本的原始转义拼写和树结构，以及 93 份完整历史与 2960 个请求前缀。93 份中 57 份含分支，44 份的原生最长分支经过非首子分支，因此只证明真实来源棋谱的最长分支样本，不能将全部请求称为实战主线。输入中的重复 AP 是现有原生兼容行为，也不能称作严格 FF4 合法性认证。

编码候选索引为 `target/ptq-encoding-audit-r1/encoding-audit-final.json`，SHA-256 `81f19163bfc3c7fb79f39c8bd2d2665c07873a6a8129315eb0a179406153d644`；独立审查为 `review-r1/review.json`，SHA-256 `3e50c3c1a84d18329d74ca46601565f78d32c2fb556586a7c1a354ff9fe5ca26`。这些是固定输入包的导入证据，不是模型精度、完整语料或最终留出验收。正式导入工具还须执行全树 CA/FF、US/AP 语义依赖、路径及显式错误门，不能把实验脚本的断言直接用作发布入口。

### 8.12 旧执行方案的显式绑定与接口验证

`collect_quantization_corpus.py --execution-spec` 现可读取 `rustgo-legacy-quantization-execution-spec-v1`。显式 spec 绑定原二进制、模型、配置、plan 的字节 SHA，以及 batch 和允许的环境。FP16 必须使用原 plan 与空显式环境；首版 INT8 范围为原生模型全 FFN、min width 0、q64、关闭运行时 GEMM 调优。继承的 `KATAGO_*` 仍会清除，不接受 batch trace、调试或临时调优环境混入该身份。

配置和 plan 冻结到新目录，只替换有效配置中的 plan 路径。采集和每个 ABBA 臂各运行前后两次同二进制 `cuda-fingerprint`；命令、原 stdout/stderr、结果、Worker Hello 与实际日志都绑定到独立的 `rustgo-legacy-evidence-v1` 身份。FP16 只声明原 loader 安装了原 plan；INT8 另检查 q64/tune0、量化语义、FFN 层数和 IMMA 标记。它们不证明全部 kernel 的实际利用率。旧指纹不含驱动 API 和物理 GPU UUID，报告明确保留未知值，不冒充统一量化 runtime profile 或旧 FP32 认证。

19 项自包含模块测试、5 项接线测试和原 collector/ledger/driver 的 57 项 CPU 回归通过。日志见 `target/unified-quant-g2-execution-binding-r1/cpu-integration-suite.log`；该次共 103 项还包含当时的编码导入测试，其中 1 项符号链接测试因系统权限跳过。模块冻结 SHA-256 为 `04b8f7b398f06a51b343e47cccdc294cfbb4395c191b95e1689f5a20ee6dc806`。四份真实 B11/B15 旧配置的 CPU 冻结和独立审查位于 `module-cpu-r1/real-specs-report.json` 与 `NEWreview-r1/report.json`。

随后仅执行一次 B15 两路径接口验证，事前 intent SHA-256 为 `38dec64034611e574b2c3dac5e80c681ff1cfe161cc2caa6d734b8823a30076a`。旧 FP16、旧 INT8 各采集同一盘 32 个 calibration 局面，物理容量 B8、请求并发和协议容量均为 1；已有 r9 统一后端完整 FP16 配方的 B1/C1 采集作为数值参考。两个新采集均通过原始输出和执行证据检查：

| 旧执行方案 | 最大胜率输出差，百分点 | 最大目差 | policy top-1 | ownership RMSE |
|---|---:|---:|---:|---:|
| B15 FP16 原 plan | 0.061187 | 0.008203 | 32/32 | 0.000209 |
| B15 全 FFN INT8 q64 | 0.886053 | 0.822455 | 32/32 | 0.004331 |

原始 spec、前后指纹、日志、PB、比较与单次命令记录保存在 `target/unified-quant-g2-execution-binding-r1/b15-interface-smoke-r1/`，结果为 `INTERFACE_DIAGNOSTIC_PASSED`。没有性能 ABBA、最佳基线资格、独立精度或棋力结论；正式二进制 SHA 保持不变，原生产 Worker PID 和启动时间也未改变。

独立 CPU 审计重载三组共 96 份请求及 96 份结果 PB，复算两个比较的完整指标和逐例值，并复核 intent、四条命令、102 项依赖与 190 项原始证据 SHA；全部一致，见 `NEWaudit-r1/report.json`。两组均不是逐位等价；计数为每组 32 行/32 个 NN batch，不能用这个 C1 冒烟代替完整 batch 范围或 C32 性能资格。审计封禁了子进程和 gRPC server 启动。

当前比较器的 reference 仍要求统一后端完整 all-FP16 配方；旧 FP16 collection 只能作为候选或计时基线。ledger v1 仍只接收统一种子，没有把这些旧方案伪装成已通过的 v1 配方。完整自动选型仍需单独的执行方案登记、统一工作负载、独立语料及最终留出验证。

### 8.13 正式编码导入工具与扩大来源

新增 `scripts/import_quantization_sgf_encoding.py --input <原目录> --output <新目录>`。首版只接受明确 FF[4]/GM[1]/SZ[19]、CA 缺省或单值 ISO-8859-1；全树检查 CA/FF 的位置、唯一性和单值。除 PB/PW/BR/WR/GN/C 外，非 ASCII 属性一律拒绝，US/AP 等原生语义依赖保持 ASCII。唯一内容操作是记录的根 CA 插入或替换；完整原始转义拼写、顺序、分支、重复 AP 与精确字节反转换必须通过。原件不修改，非法路径、链接、重复路径或已有输出目录拒绝；异常不发布完整 manifest。

初版 128 层深度限制在真实 Pro2 上额外拒绝 84 份文件，原失败产物保留于 `target/ptq-encoding-import-tool-r1/`。最终版改为显式栈解析、遍历及平铺摘要，保留 8 MiB/文件、100,000 节点及 4096 树层的资源界。24 项作者测试为 23 通过、1 项因 symlink 权限跳过；Windows junction 测试通过。独立 12 项检查包含 4096/4097 层边界、失败不发布和 128 例浅树转换差分，全部通过。再次转换 Pro2 得到 661 份副本、19 个拒绝，全部副本 SHA 与原独立审计一致。

工具 SHA-256 为 `350f0798eaf9ca8d8e30c31811d6d84c896516ef02a7e5284e195187e1755737`；作者证据为 `target/ptq-encoding-import-tool-r2/verification.json`，独立审查索引为 `target/ptq-encoding-import-review-r1/review-final.json`（SHA-256 `39e6ca83330bf384a7f7ce666dff9612655ddfe832d4cd79c71f3c01820247d9`）。目录总量由本次冻结来源清单限制，导入器本身不宣称可无界遍历任意目录。

另外单次下载同一发布仓库固定 commit 下的 Pro.7z，3,657,149 字节、9669 份 SGF，Git blob SHA-1 `f4520e3fce5ef72206b639eb4e55d77e541afba9` 与清单匹配；归档 SHA-256 `4b87b49adac5626d78ef48e05a122341b262d229e713d974338d573bff0ec735`。来源索引 `target/ptq-source-pro1-r1/source-manifest.json` 记录提取前路径/类型检查及全部原件 SHA。此前树查询响应的 `acace506…` 已解析为 commit，实际 tree 为 `cf3d2c97…`，纠正记录在 `ref-correction.json`，未改旧证据。

后续语料使用隔离新构建的 CPU builder：`target/ptq-corpus-builder-r1/build/debug/examples/sgf_quantization_corpus.exe`，SHA-256 `f5daa7605878faef022110fe25d6f61a24f307a94c456bf452be17f1c91b8018`。`build-provenance.json` 保存构建日志、编译器、二进制及构建后 238 份稳定源码快照；这不是完整工具链的可复现构建认证。扩大来源仍须经过原生合法回放、去重和固定分区，文件数本身不代表合格棋局或最终精度样本数。

### 8.14 原生初始局面支持与新分区声明

合并 Pro1/Pro2 的 10,349 个原件，经正式导入器得到 10,326 个副本和 23 个语法拒绝。未修改的 builder 仍只接纳 93 份棋谱，64 局面/盘配置得到 5925 个 calibration 请求，selection/holdout 为空。最终只读验证为 `target/ptq-pro-corpus-r1/verified-result.json`，状态 `CPU_ASSEMBLY_VERIFIED_NOT_READY`；组装脚本在成功生成后曾错误解读三分区列表，原错误保留，修正报告未重跑导入或生成器。

对 6198 个 `midgame_setup_or_rules` 拒绝的逐节点审计发现，全部为根元数据后独立 AB 节点，再开始白方首手，没有中途修改。仅其中 107 份声明当前支持的中国规则；其余规则保持拒绝。审计索引为 `setup-audit-r1/index.json`，SHA-256 `e2a43b44302e40a6ab1870ea4561ab88af3f8e8875301f1ef8fe9fb8bbd59e49`。原生 CompactSgf 仅采集根摆子，不能通过单纯放宽拒绝条件导入这些局面；生成器将直接从已选最长分支提取受支持的首手前 AB，并继续完整合法回放。

**在新生成器回放和任何模型输出采集之前，预先固定新语料分区为 calibration 32 盘、selection 32 盘、holdout 128 盘，每盘至多 64 个局面，seed 继续使用 `rustgo-ptq-pro-20260924-r1`。** 128/128/128 是旧命令的请求配额；本规格 §6 的最终留出门仍为至少 128 盘、4096 个不同语义局面，没有降低。32 盘校准和 32 盘选型只作为首版有限候选的工程起点，不保证覆盖搜索叶节点或所有棋型。分区按原有整盘语义哈希确定，不能根据模型误差、性能或是否恰好凑足数目重排；不足即保持 `NOT_READY`。已经用于旧模型诊断的两盘棋如与新 selection/holdout 重合，同样不接受为独立验收集。

生成器改动通过 11 项 CPU 测试与独立静态复核，隔离构建的二进制 SHA-256 为 `e8ff332ec004a7ceaaf9ae494fa75c30fbb3f08cb896307ff9cf39047b90377a`；238 份工作区源码在构建前后保持相同。证据在 `target/ptq-pre-move-setup-r1/`。单次完整回放接纳 **195 盘**，即旧 93 盘加新增 102 盘；另外 5 份中国规则候选因非法落子或轮次问题拒绝。

| 冻结分区 | 棋局数 | 不同语义局面数 |
|---|---:|---:|
| calibration | 32 | 2044 |
| selection | 32 | 2004 |
| holdout | 128 | 8049 |
| reserve | 3 | 不参与请求 |

旧 93 盘的完整语义、采样与来源保持不变；新旧共有的 5797 份请求除分区名外逐字段相同，旧两盘模型诊断棋局未进入新 selection/holdout。生成结果 `target/ptq-pro-corpus-r2/result.json` 的 SHA-256 为 `77ecd9a0bd042940379d0ee23326b39eb74bf1440ebe94a8b50460a217faccc7`，corpus manifest 为 `c7b80205d26d5dd7c7cf38dceb8b88bcc8b3b452b764f7a83113d156fc88c842`。此时状态仅为 `CORPUS_COMPOSITION_READY_REQUIRES_REVIEW`，独立语料审计和模型精度均另行记录。

当前样本包括较多初始布子局面：calibration 为 19 盘两颗黑子/贴目 6.5 和 13 盘空盘/7.5；selection 为 21 和 11 盘；holdout 为 60 盘两颗黑子/6.5、67 盘空盘/7.5，以及 1 盘两颗黑子/7.5。原始元数据和局面均保留，不为了平衡分布修改棋谱或重排分区。全部样本仍是来源棋谱的原生最长分支，未冒充已验证实战主线或真实搜索叶节点。

用该语料生成的 B11/B15 CPU 候选计划位于 `target/unified-quant-qualified-corpus-plans-r1/{b11,b15}/search/`，分别为 2 和 3 个去重后 v1 配方。它们尚未执行，不读取留出输出、不表示精度通过或选定部署计划。

独立 CPU 审计随后通过：从全部 195 份接受原件重新解析完整树和最长分支，以另一套 Python 落子/提子/简单劫逻辑重放 36,198 手，重算 12,416 个采样位置；12,097 份保留请求、127 次排重、三个分区及 3 盘 reserve 均吻合。旧两盘诊断与新 selection/holdout 在整盘、历史语义和棋盘键三个层面均无交集。审计没有读取模型输出或启动子进程。主报告 `target/ptq-pro-corpus-r2/NEWaudit-r1/report.json` 的 SHA-256 为 `717cfd0d4cb41f8f8a759c45006e8d14008ba0d9aa6a4f2a44fcc2957bf82b3a`；`scope-and-distribution.json` 另记录上述分布和适用范围。语料组成现已完成独立复核，模型精度/性能验收仍未完成。

### 8.15 有限执行方案目录

新增 `scripts/quantization_execution_catalog.py`，在 CPU 上登记统一配方、显式 batch 和旧 FP16/INT8 两类必需对照，数值 reference 固定为统一完整 FP16。当前范围仍是已有 v1 种子，不执行新搜索策略、GPU 或发布。每个统一候选须登记与两类旧控制的完整比较；预算不足以覆盖全部方案时拒绝准备，不能省略较强控制。

独立审查发现初版 key 会因相同 timeout 的 JSON 数字表示、统计指标、第三方数值 reference 或请求排版变化而重置。r2 改为按有序 Position/EvalParameters 的确定性 protobuf 字节及真实执行参数生成 key；数值采集使用独立 `collection_attempt_key`，纯 CPU 比较使用 `comparison_key`。更换数值参考只改变比较键，不解锁同一采集；warmup/cycles 只影响 ABBA，不影响数值采集；原始文件和判定设置仍绑定在完整来源记录中。

30 项 CPU 测试及独立复审通过。真实 B15 的 CPU 目录登记 5 个执行方案、5 次数值采集和 8 组 ABBA 计划，尚未执行或消耗这些实验。r2 catalog 为 `target/unified-quant-execution-catalog-r1/real-b15-cpu-r2/catalog/catalog.json`，SHA-256 `3720b7c5901e195dbd995369e5ed5e7fef563371b9298882257d9890ec90c070`；独立报告为 `target/unified-quant-execution-catalog-r1/NEWreview-r2/report.json`。该目录自身不证明历史预算未用，后续必须接入跨目录的持久尝试账本，在启动前登记消费。工具升级后旧准备产物按源码绑定拒绝运行，不能更新旧 SHA 绕过。

### 8.16 持久执行尝试账本与历史证据接线

`quantization_selection_ledger.py` 新增显式 `rustgo-quantization-execution-ledger-v2` 模式，保留 v1 schema 和命令。新入口包括 `execution-init`、`execution-register-catalog`、`execution-reserve-collection`、`execution-ingest-collection`、`execution-compare`、`execution-reserve-abba`、`execution-fail`、`execution-inspect`。同一个持久账本对注册的多个 catalog 统一计次；启动方应在外部进程启动前取得落盘 reservation。失败或中断不归还预算，相同实际 key 不得再次 reserve。不能新建空账本来重试已消费实验。

数值采集与 CPU 比较分别记录。采集验收重新核对实际统一 profile 或旧 spec/probe/log、二进制/模型/配置/batch、完整有序 Position/EvalParameters、请求 lease、全部原始 PB 与来源；不接受外部声称的 PASS。换数值参考时复用已验证的原 PB，只重算比较，不新增采集计数。ABBA reserve 要求完整统一 FP16 reference 及两臂的原始数值比较均通过，并绑定各自观测身份。

该阶段尚未连接完整 ABBA 结果入账和执行驱动，后续接线见 §8.18。它不排名、采纳或发布计划，旧 collection 报告也不能证明实际进程在 reservation 之后启动。catalog、原始采集目录、依赖和快照均须保持可用且不变；单独保留快照不能代替旧绝对路径配置和原始证据。

21 项新增 CPU 测试、33 项原 v1 账本回归、30 项 catalog 回归和 13 项原 driver 回归通过，共 97 项。冻结 ledger SHA-256 为 `1bb7668985ec5c181df88a37ce4f6c8f3ace34cbf480b386fab0c02db0596f6b`，新增测试文件 SHA-256 为 `ec26b4b838293d5a7bb9877d451a9d63db7052e91ec63f7a18e9ff78b2844a10`。日志均位于 `target/unified-quant-execution-ledger-r1/`。

随后进行一次 **只读原始数据的 CPU 接线检查**，向独立 test registry 登记此前已有的 B15 三份统一配方与两份旧执行采集，每份仍是旧同一盘 32 个 calibration 局面。5 份采集和 5 次 CPU 比较通过，8 对比较的数值前提可重算；另一个只改指标和等价 timeout 的 catalog 无法重新登记同一采集尝试。没有新增推理，没有 reserve 或执行 ABBA。这份测试账本不作为未来 GPU 实验预算，登记发生在历史运行之后，不伪称启动顺序证明。

接线结果为 `target/unified-quant-execution-ledger-r1/real-b15-existing-evidence-r1/result.json`，SHA-256 `b6210a34e0bafc247009f6a3a0c7e74c735b2607982d93b914c98a62e65c969e`，test registry 头为 `fad93d3ae3adc1deb939965d51e67cda2f1960d48239a9760001e324e7e6b300`。该结果不属于新 195 盘语料的模型精度证据。阶段结束检查确认正式二进制 SHA、生产 Worker PID/启动时间保持不变，且没有遗留测试 katago 进程，见 `post-system-check.json`。

独立复审通过，没有未解决阻断项：复核 23 个旧业务函数的 AST 保持相同，另用 15 项合成反例检查损坏 PB 的失败终态、实际 spec 字段篡改、重哈希伪造 PASS 和 ABBA 失败后的跨 catalog 重试拒绝。真实接线的 18 事件链、冻结文件与 5 份采集/比较、8 对数值前提绑定均核对。索引为 `target/unified-quant-execution-ledger-r1/NEWreview-r1/index.json`，SHA-256 `11f6326e6e49561abecf125e1f69928971f9de2cd2d2b8bf151f2c67648d9a1c`。审查没有运行模型、Worker 或 GPU；上述未接入的 ABBA 和 driver 边界保持不变。

### 8.17 ABBA 时钟边界与首轮正式 C1 负载

在本机使用的 Python 3.11.9 中，`monotonic_ns` 对应 GetTickCount64，`perf_counter_ns` 对应 QueryPerformanceCounter，两者不能直接比较时间戳。旧指纹探针使用前者，计时窗口使用后者。新探针的命令和结果统一使用 `perf_counter_ns`，同时记录 API、实现、单调性、可调整性和分辨率；新 ABBA 顶层及各臂也记录同一时钟。验收时须先核对时钟，再检查探针位于计时窗口之外。

旧无时钟标记的成对探针仍可用于既有数值身份验证，但不能作为新 ABBA 的同钟时序证据；混用新旧探针、缺少一端标记、时钟变化或非整数时间戳均拒绝。24 项模块测试与 5 项接口测试通过；独立审查另外验证了 5 个边界，并确认旧数值观测身份保持一致。源码前后快照和日志位于 `target/unified-quant-execution-clock-r1/`，独立报告 SHA-256 为 `234365d29de9552e43fb34f49037c462bd0083a163fbe9f40c434c7338f83bd8`。这不是旧进程启动顺序的追认。

首轮正式选型负载已在任何新模型输出前声明于 `target/unified-quant-selection-c1-r1/intent.json`，SHA-256 为 `d86f5854ce8d0cb0263b17bae7bf283b16c33d074b32238bc01a0e3e8162005f`：完整 selection 2004 个局面/32 盘、C1/协议容量 1、128 个预热请求、完整分区计时一遍、P95 主指标、180 秒请求超时。B11 后 B15 顺序执行；统一方案 B1，旧 B11 FP16 原配置为 B16，另外三个旧对照为 B8，均原样保留。当前共享预算预定为 9 次采集与 13 组 ABBA，不包含独立确认或留出验收。

各方案须先通过同模型统一完整 FP16 参考的最大胜率输出差 ≤6 个百分点门，再进行依赖它的性能比较，另报 policy、目差和 ownership。现有 P95 收益门为速度比 `baseline P95 / candidate P95 >=1.01`，即延迟降低约 0.9901%；报告应同时给出实际延迟降幅，不能误称为字面上的降低至少 1%。两个同方案计时样本的相对极差均须 ≤5%。旧 C32/C64 优化配置只是已知对照，不代表已经证明 C1 最优；本轮也不推及 5060 Ti、5080 或搜索叶分布。

### 8.18 完整 ABBA 入账与持久预算执行驱动

catalog-v2 已接通 `execution-ingest-abba` 和 `execution-summary`。验收重新读取四臂的实际命令、依赖、环境、Worker Hello、统一 profile 或旧执行探针/日志、有序完整请求、原始 timing、NN 计数和时钟边界，复算分位数、几何均值、极差和收益门。有效的负收益或不稳定结果也形成不可重试的终态；证据错误消耗已预约的尝试并保留失败。重用已消费计时、重叠窗口、更换统计指标或方向均不能解锁另一次实验或改写原判定。

`run_quantization_search.py prepare-execution` 使用外部既有持久账本及显式 contract SHA/head，冻结完整命令、文件和环境；`run-execution` 要求准备文件 SHA，每次启动 helper 前先写入持久 reservation 和 driver 事件。统一 reference 先采集；有效数值失败只跳过依赖它的 ABBA，运行错误则停止并保留锁和原始日志。已验证的终态证据可以只读复用，pending/failed 不可重跑。旧 v1 入口及 Windows Job 子进程隔离实现保持不变。

汇总列出每个统一候选对完整 FP16 reference 和两类旧控制的完整矩阵。缺项、失败、波动超限或数值失败不能被隐藏；完整正向矩阵也只列为未排名、仍需独立确认的候选，不采纳、发布或读取 holdout。

本轮 31 项 v2 账本测试、33 项原 v1 账本回归、30 项 catalog 回归、15 项新 driver 测试、13 项原 driver 回归和 29 项时钟/旧接口测试全部通过，共 151 项。主实现冻结 SHA-256 为 ledger `679a711a3750328649920e3c4ce615122b9026ed293efd901a548dc6fcaf5cb1`、driver `c36057fde85656ec477a2f51791f4b395ae5a0cb8bcb863406dc6318571e1d96`；日志分别位于 `target/unified-quant-execution-abba-r1/`、`target/unified-quant-execution-driver-r1/`、`target/unified-quant-execution-clock-r1/`。这些是合成 CPU 接线与回归结果，不能替代 §8.17 预登记的真实数值/性能实验。最终独立复审与实机结果另行记录。

最终独立复审通过 13 项针对性检查：使用真实 `run_arm` / `drive_window`、仅替换进程与 peer 的 CPU fixture，生成统一及旧 INT8 四臂原格式后可被新验收器接受；缺时钟、窗口重叠、reservation 之后写启动事件失败等反例均拒绝，后者没有放行 helper 且消耗一次预算。索引 `target/unified-quant-execution-abba-r1/NEWreview-r1/index.json` 的 SHA-256 为 `64db01de49fcdaa273db59f5414922e78e08bef4199ac0786a4c947b49586227`，无未解决阻断。根准备脚本另关闭了跨模型 TOFU：B15 必须使用 setup 时保存的 source/catalog/ledger 锚和 B11 完成后的 head，不接受运行间重新计算的新来源身份。

真实目录随后完成 CPU 冻结，`target/unified-quant-selection-c1-r1/setup-anchors.json` 的 SHA-256 为 `21726ad2973eb68050d84b58eaaef0e88bb9c7de49df847bd3591dc9958ce641`；初始共享账本 head 为 `fec3799e204796bdd4c4776fc204b388de0eb7e418f515f0cc38617c5561e711`。B11 准备文件 SHA-256 为 `f39eb87cdc2e2320acf38d7f8540a068097e039ce7478b1d04a36181b1314dde`，预计 4 次采集、5 组 ABBA。此为启动前记录，最终尝试数、数值与性能结论以各自 `prepared/result.json` 和独立原始证据复核为准。

### 8.19 正式 C1 实机流程（首轮及独立审计完成，无合格候选）

在同一台 RTX 5070 Ti 上已完成 B11 四份 selection 数值采集，每份为完整 2004 局面/32 盘。结果已由持久账本从原始 PB 验收和比较，并完成数值与全部性能臂的独立复核。以下误差均相对本轮同模型统一完整 FP16，单位“百分点”已将概率差乘以 100。

| B11 执行方案 | 最大胜率输出差，百分点 | 最大目差 | policy 首选一致 | ownership RMSE | 本轮数值门 |
|---|---:|---:|---:|---:|---|
| 统一完整 FP16 B1（参考） | 0 | 0 | 2004/2004 | 0 | PASS |
| 统一全 FFN INT8 B1 | 5.304781 | 0.788608 | 1974/2004 | 0.003490 | PASS |
| 旧全 FFN INT8 q64 B8 | 5.014604 | 1.518077 | 1977/2004 | 0.003473 | PASS |
| 旧 FP16 原 C32 plan / B16 | 0 | 0 | 2004/2004 | 0 | PASS |

统一与旧 INT8 的实际日志均记录 cuBLASLt `imma=true`、INT32 累加；这确认对应 GEMM 路径，不等于 Tensor Core 达到峰值利用率。全部方案用 C1/协议容量 1 和跳过缓存的请求，表中的 B1/B8/B16 是各自保持的配置容量。未使用计时采样来反推不可观测的物理 batch 分布。

预登记的五组 B11 ABBA 已测量并从原始计时验收入账，共 20 个臂；每臂 128 个预热、2004 个完整 selection 请求。两个重复臂的相对极差全部 ≤1.258%，均通过 5% 稳定性门。下表速度收益为 `baseline P95 / candidate P95 - 1`，不能与延迟降幅混用。

| B11 对照 → 候选 | 对照 P95，ms | 候选 P95，ms | P95 速度收益 | 1%/5% 门 |
|---|---:|---:|---:|---|
| 统一 FP16 → 统一 INT8 | 2.856 | 2.862 | −0.227% | NO_GAIN |
| 旧 FP16 → 统一 FP16 | 2.911 | 2.839 | +2.521% | PASS |
| 旧 INT8 q64 → 统一 INT8 | 2.793 | 2.870 | −2.665% | NO_GAIN |
| 旧 INT8 q64 → 统一 FP16 | 2.802 | 2.837 | −1.229% | NO_GAIN |
| 旧 FP16 → 统一 INT8 | 2.880 | 2.858 | +0.758% | NO_GAIN |

两种统一候选均未通过完整控制矩阵；单独对旧 FP16 的正收益不能替代对旧 INT8 的比较。本轮不采纳统一候选，不对这些已关闭的 q128 方案追加相同工作负载的重试。B11 driver 已成功退出，状态 `FINITE_EXECUTION_COMPLETE_REQUIRES_REVIEW`，完整矩阵结论为 `NO_CANDIDATE_PASSED_COMPLETE_CONTROL_MATRIX`。实际消耗 4 次采集、5 组 ABBA，没有失败、重试、确认或留出读取；共享预算剩余 5 次采集和 8 组 ABBA，供冻结的 B15 流程使用。

B11 `prepared/result.json` 的 SHA-256 为 `ebd51317b3b36cc4f36a799a9ff4424a2d6dd4730855ec9780dbf9ef374016c2`，summary SHA-256 为 `bebdb82eb8b0a17ee9ae2bc03506f99994e7dea2c29058d555452b67659a0b2a`；截至该次完成的 ledger head 为 `a82681749403bbd622b46c0da67c6052e11d4791e67b1410c50b6530a53d475a`，driver head 为 `158fca03a022c958781797f1d6dce7a6975a103c6ad39ee6a4b7482e38cde637`。

B11 独立审计首次执行即通过：重新核对 4×2004 份原始 PB、20 个性能臂及 9 次 reserve→gate→exit 顺序，16,477 个绑定文件的审计前后 SHA 完全相同。统一 INT8 的胜率差平均/P95 为 0.104352/0.493842 个百分点；旧 FP16 与统一 FP16 的 2004 份输出 protobuf 原字节完全一致。审计索引 `b11/NEWaudit-r1/index.json` 的 SHA-256 为 `1a19172d951169754cb94769dda3b437373271e69b6ce338fd7591c36b259f44`，报告为 `554bb1f121a6299d2a2515a02dc8a9c7f39a2e8701ff3ff27e6f294b2fc24d38`。审计未启动模型或读取留出输出；独立结论同为没有完整矩阵胜者。

原始数据位于 `target/unified-quant-selection-c1-r1/b11/prepared/collections/` 和 `performance/`，入账结果位于该实验的共享 `ledger/events/`。程序在验证原始证据时开销较大，全部位于 ABBA 计时窗口外；不能拿整体选型耗时充当推理速度。原生产 Worker 和正式二进制的启动前身份核对保存在 `pre-system-check.json`。

结束检查 `post-b11-system-check.json` 确认原正式二进制 SHA、生产 Worker PID/启动时间不变，仅原 PID 78408 仍在运行；实验 helper 已退出且共享 driver guard 已释放。

B15 沿用原冻结锚和 B11 终态 head 完成 CPU 准备，`b15/prepared/prepared-execution.json` SHA-256 为 `6ff186c9259a006d9430cb02a33775ddb3abb91ed6c0c299e2cd0c3efe247d44`。5 次采集/8 组 ABBA 的实机流程于 2026-09-24 13:30 UTC 后启动；启动前 `pre-b15-system-check.json` 确认 GPU 空闲、原正式二进制和生产 Worker 身份不变。三份统一方案为完整 FP16、全 FFN INT8、宽度至少 384 的 FFN 使用 INT8；先完成数值门再执行对应性能比较，结果待终态独立复核。

五份 B15 采集的数值比较均已入账为 PASS，每份覆盖 selection 的 2004 局面/32 盘；完整 FP16 reference 的自比通过。其余四份相对该统一 FP16 reference 的结果如下，尚待完整终态独立审计：

| B15 执行方案 | 最大胜率差（百分点） | 最大目差 | 平均目差 / P95 | policy 首选一致 | ownership RMSE |
|---|---:|---:|---:|---:|---:|
| 旧 FP16 控制 | 0.219029 | 0.199661 | 0.002273 / 0.006388 | 2001/2004 | 0.000203987 |
| 旧全 FFN INT8 q64 | 1.836920 | 1.097666 | 0.037522 / 0.101280 | 1977/2004 | 0.00416609 |
| 统一全 FFN INT8 | 2.592498 | 2.230920 | 0.039395 / 0.103196 | 1980/2004 | 0.00418952 |
| 统一宽 FFN ≥384 INT8 | 1.530784 | 0.820869 | 0.032154 / 0.088011 | 1981/2004 | 0.00404928 |

宽 FFN 配置实际包含 12 个 INT8 FFN 与 33 个 FP16 FFN；采集日志记录物理 B1、对应逐层精度以及 Graph launch。该样本集上宽 FFN 配置的最大胜率差和最大目差低于全 FFN INT8；性能需要另看下述完整 8 组 ABBA。两份 FP16 输出不是逐位相同，不能仅因同为 FP16 就视为纯入口迁移等价；旧 FP32 认证与当前实验门保持独立。

首组 B15 ABBA 已入账为稳定 `NO_GAIN`：旧 FP16 对照的 P95 几何均值为 3350.675 µs，统一全 FFN INT8 为 3579.605 µs，速度比下降 6.3954%（延迟增加 6.8324%）；两侧重复臂极差分别为 0.0349% 与 0.0324%。原始四臂证据在 `b15/prepared/performance/00-9f6566d5b8222423/`，driver 已写入 `PAIR_VALIDATED`。该组拒绝全 FFN INT8 相对这个控制的收益资格；其它登记比较继续执行，完整终态与独立审计仍待完成。

第二组同样入账为稳定 `NO_GAIN`：旧 INT8 q64 与统一全 FFN INT8 的 P95 几何均值分别为 3558.320 / 3582.791 µs，速度比下降 0.6830%；两侧重复臂极差为 1.8246% / 0.1736%。证据在 `b15/prepared/performance/01-6af5765adc91017c/`，driver 事件序号 28 为 `PAIR_VALIDATED`。

第三组入账为稳定 `PASS`：旧 INT8 q64 与统一宽 FFN ≥384 INT8 的 P95 几何均值分别为 3540.987 / 3348.257 µs，速度比提高 5.7561%（延迟降低 5.4428%）；两侧重复臂极差为 0.6326% / 0.4334%。证据在 `b15/prepared/performance/02-904d4357f5d73b96/`，driver 事件序号 32 为 `PAIR_VALIDATED`。这是宽 FFN 候选相对一个旧控制的收益，尚未通过完整控制矩阵或独立确认。

第四组已写入账本 `ABBA_OBSERVED`，为稳定 `NO_GAIN`：统一 FP16 与统一全 FFN INT8 的 P95 几何均值分别为 3434.503 / 3617.706 µs，速度比下降 5.0641%（延迟增加 5.3342%）；两侧重复臂极差为 0.9919% / 0.5588%。证据在 `b15/prepared/performance/03-ceac03b6eb1a9875/`，driver 事件序号 36 为 `PAIR_VALIDATED`。全 FFN INT8 对三种预设对照均无收益，已不能取得本轮 C1 完整矩阵资格；宽 FFN 候选的其余对照仍待完成。

第五组为稳定 `NO_GAIN`：旧 FP16 与统一 FP16 的 P95 几何均值分别为 3345.020 / 3415.388 µs，速度比下降 2.0603%（延迟增加 2.1037%）；两侧重复臂极差为 0.0858% / 0.4815%。证据在 `b15/prepared/performance/04-8b7fe2ec27078a1e/`，driver 事件序号 40 为 `PAIR_VALIDATED`。

第六组为稳定 `PASS`：统一 FP16 与宽 FFN ≥384 INT8 的 P95 几何均值分别为 3429.892 / 3357.317 µs，速度比提高 2.1617%（延迟降低 2.1160%）；两侧重复臂极差为 0.2654% / 0.1048%。证据在 `b15/prepared/performance/05-2587efbc185ead9c/`，driver 事件序号 44 已为 `PAIR_VALIDATED`。

第七组为稳定 `PASS`：旧 INT8 q64 与统一 FP16 的 P95 几何均值分别为 3510.166 / 3416.801 µs，速度比提高 2.7325%（延迟降低 2.6598%）；两侧重复臂极差为 0.6220% / 0.5313%。证据在 `b15/prepared/performance/06-dafad3799a7e1526/`，driver 事件序号 48 已为 `PAIR_VALIDATED`。统一 FP16 已在第五组慢于旧 FP16，不能由本组正结果取得完整矩阵资格。

第八组为稳定 `NO_GAIN`：旧 FP16 与宽 FFN ≥384 INT8 的 P95 几何均值分别为 3357.417 / 3342.144 µs，速度比提高 0.4570%（延迟降低 0.4549%），未达 1% 门；两侧重复臂极差为 0.1188% / 0.4476%。证据在 `b15/prepared/performance/07-713f8ff0a539327d/`，driver 事件序号 52 为 `PAIR_VALIDATED`。至此全部 8/8 对照入账，三个统一候选均不能取得本轮 C1 完整矩阵资格，没有候选可据此进入收益确认或采用。没有改变主指标或重试已完成负结果。

B15 主进程随后正常退出，session 59237 为 exit 0，原 driver 锁已释放。`prepared/result.json` 状态为 `FINITE_EXECUTION_COMPLETE_REQUIRES_REVIEW`，SHA-256 `446964b6b8ae1238d6a8f1b6be81a01a0cd3994c91b289c02ffaec9625f733f0`；最终账本 head 为 `d3f3740f696fb04b2fd7d6f618b2733d42607eb07eb1c0ac781b4de6fd98c54e`，driver head 为 `4abb45b123be6d824d21de26a0629f492250fdecaf06d5acbfdef322f282587d`。完整 summary SHA-256 为 `14c363fb0bdbc878e6af0df5d04ec22b0f29483bb430b824dd508e7de320ff75`，正候选列表为空。全部 13 个子命令返回 0，实际消耗 5 次采集/8 组 ABBA。

原冻结版本独立审计首轮通过、exit 0，报告 `b15/NEWaudit-r1/report.json` SHA-256 为 `ae35ee6411c59115a2b0ccfbe4af9f4a3fb0ea7e3470a3a780aea45d9a88d76a`。20,577 份绑定文件前后哈希一致，hash 清单 SHA-256 为 `c71e25942a0833b25457958c3e4b0ef3237f961ebd0e36e07901736b6cc18adc`；5×2004 数值输出、32 个计时臂及持久 reservation/gate 计次均复核一致，保留已审 B11 前缀。补充打包的 B15 index SHA-256 为 `b80cd04aad5aea8833b7dc7e4597efe6c652278f878b86976f45585f4c1b511e`。审计只证明证据完整和结论可复核，不把无收益候选改判为通过；尚无确认、留出或正式采用。

实际日志需按模型区分：统一 FP16/INT8 的 attention 为 q128，旧 INT8 为 q64。B11 的旧 FP16 为 q64-serial 且启用融合 dual-FFN；B15 的旧 FP16 则为 q64、`dual_ffn launch=unfused effective=0`，其 `collections/01-fcc82fa3c75747a3/worker.log` 和 `runtime-cudaTacticPlan.json` 一致，不能套用 B11 的融合结论。`cuda_exec.rs` 的 launch grid 为 `(ceil(S/tile), batch*heads, 1)`；因此 B11 B1、361 tokens、12 heads 的 q128 为 36 个 CTA、每 CTA 256 threads，q64 为 72 个 CTA、每 CTA 128 threads。本机指纹为 70 SM，总发射线程数两者相同。这是并行分布差异的证据，不能仅凭 CTA 数量认定 occupancy 或总性能的因果。

[NVIDIA 矩阵乘性能指南](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)说明较大分块的数据复用与较小分块的并行度存在权衡；[CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#thread-and-block-heuristics)也要求结合寄存器、共享内存与线程块数量评估配置。本文据此将 q64 视为有源码依据的候选假说，不把其他架构的示例性能或未经测量的 occupancy 写成本机收益。

现有 Rust runtime 已支持显式 q64，并将它纳入冻结 tactics/profile；当前 Python catalog/collector 的统一路径只登记空显式环境，所以本轮尚未搜索该维度。若后续决定扩展，可保留 q128 完整 FP16 reference 和两个旧控制，仅新增一个 q64 执行身份；不能修改本轮源码绑定或新建空账本重测已关闭的 q128 比较。续接必须绑定本轮最终 head、旧工具源码与已消费 key，并单独声明新增预算。本段为设计分析，不是已登记或已执行的后续实验。

### 8.20 q64 工具扩展与旧方案迁移（续接与独立复核完成，无完整收益候选）

2026-09-27 重新核实实际状态：session 85010 句柄已不存在，系统进程清单没有该实验的 Python/Worker；原生产 Worker 78408 保持。编排 events 12 为 `STAGE_COMPLETED`，终态 head `82d00b342089661c375da7dea1fa64303fe42300d77984568014187c7f6d4793`。新增 1 次采集和 3 组 ABBA 已全部终态，`prepared/result.json` SHA-256 为 `7a1d324fd72fdecd352b85c13afe388d835eba4cd8ab3beeee9e3eaf073a5eda`，child ledger head 为 `a5fca1fed106c39b0986ab382f1b8ecd307f2e91b3f518d325abc840b5b0a227`，summary SHA-256 为 `a9e64edcabebde8b3d7683ec76de9ca2a319be02bb6bfe4396d7ce95bc62d3e7`。原进程 OS 退出码未观察到；终态文件不代替该退出码。汇总仍为无候选通过完整控制矩阵，正在独立读取原始数值和计时复核，不重跑 GPU、不进入确认或留出。下文准备/执行中的记录均为此前阶段历史。

随后独立审计 r3 完成，审计进程实际 exit 0、217.49 秒，81,693 份绑定文件前后 SHA 一致。报告 `NEWaudit-r3/report.json` SHA-256 为 `4310b1b2dfceb7109f8a939dc4a72f5c5c1da5b1d3e122fd63eb6ca0912a1bc4`，索引 SHA-256 为 `f26d3755a933cbeeeb7a3a1cfe1f2d2217055f09bfc08d36b738ac951f6cde23`。首次 r2 审计因错误假定 catalog 登记字段而失败，原件保留；r3 修正绑定 snapshot 的读取并预核对真实 JSON 结构，没有重新运行模型或性能测量。

新增 q64 在 2004 局面上的最大白方胜率误差为 5.014604 个百分点，目差最大/平均/P95 为 1.518077/0.049716/0.137633，policy 首选 1977/2004 一致，ownership RMSE 为 0.00347303。与旧 INT8 q64 的 2004 份输出原始字节、1,466,928 个浮点字段位均相同。三组 P95 ABBA 均稳定：旧 FP16 2880.779810→2788.024673 µs（速度比 +3.3269%），统一 FP16 2857.717212→2788.702198 µs（+2.4748%），旧 INT8 q64 2796.868299→2791.129565 µs（+0.2056%，未达 1%）。因此完整矩阵仍无正候选；输出逐位相同不改写性能门，未进入确认、留出或采用。总消耗为 10 次采集/16 组 ABBA。

隔离目录 `target/unified-quant-q64-extension-r1/staged-scripts/` 已完成 catalog、collector、driver 工具副本及首版 11 项 CPU 测试，交付索引为 `staged-delivery.json`；账本续接仍在准备。根在 B15 数值阶段运行这 11 项测试，1.496 秒全部通过，日志 `tests-first-q64.log` 的 SHA-256 为 `7e579950da81de4400f199702a472a5f5db9661502138e373e8c11a7b354923b`。前后窗口记录确认此时 B15 尚无 performance 目录；原三个工具 SHA 不变。测试使用显式模块 overlay，独立静态审阅随后发现真实 CLI 的依赖路径及 `_child` 模块缓存来源仍待解决，因此不作集成通过或可执行发布。

工具 r2 补显式依赖路径，并将 `_child` 分流移至业务模块导入之前；独立审阅随后发现续接 facade 会让旧 planner 引用新 collector，从而改变冻结 search 的来源身份。r3 与 ledger-r2 已将原 JSON helper、collector、planner 捕获为保留真实源码身份的私有依赖图，新 q64 API 则通过 `q64_helpers()` 显式加载。原 benchmark 的子进程不预加载新版业务模块；原 `scripts/` 与正在执行的 B15 工具保持不变。

根在确认 B15 尚未进入 performance 的窗口完成首次 **8 项冷启动检查、16 项 q64 接线测试、17 项续接测试**，全部通过；两组单元测试分别耗时 1.670 秒和 1.083 秒。冷检查验证真实 `-E -s` CLI、两个子进程 helper 的实际源码、拒绝错误 argv SHA/非白名单，以及旧 B11 search 对实际函数 globals 的来源绑定。续接测试包含成功加载合成 protobuf 的父状态重建、缓存复用前文件哈希复验和篡改拒绝。41 项验证的索引 `cpu-r3-verification.json` SHA-256 为 `9f56a0d277b179aa19994ba398ff392505784ea96ffa70fc17442a2d2346a171`；绑定来源前后一致，没有实际 Worker/GPU 启动或真实 registry 续接。

联合冻结交付为 `revisions/r3/staged-delivery.json`，SHA-256 `4ecdbf44c94498e301114f4b8d41274dbcde61e0a73437641371a04d8b8cfc50`。随后独立的 CPU 准备实际从原模型、语料和来源重新验证冻结 B11 search，生成 `b11-catalog-preparation-r3/catalog/catalog.json`，SHA-256 `59ed36984213783c94a5d1dd1187ab13d3965bdaf320d3e7436dc7e1eb9fdc09`。新目录保留旧四个执行身份及其全部比较，只增加全 FFN INT8 B1/q64；目录总量为 5 次采集/8 组 ABBA，拟新增量严格为 **1 次采集/3 组 ABBA**。这是未执行的目录准备，尚未在真实账本分配额度；不重复旧尝试、不构成 GPU 性能或发布结论。

该真实 catalog 随后通过 37 项独立只读元数据核验：旧规格/绑定/所有 key 保持，新增执行 ID、采集 key 与三组 pair key 由标准库独立复算一致；配置和配方相同，仅显式 attention 环境为 q64。新 workload 只修改目录最大计数 4/5→5/8，实际 C1/容量 1、预热 128、循环 1、P95/1%/5% 门不变。报告 `b11-catalog-preparation-r3/NEWmetadata-review-r1/report.json` SHA-256 为 `9aeeb39746d33229cb8c6a3af5c534596c7f4b24e6ff1cc8dfb08c8c88af9d63`；此审查没有调用账本验证器、读取 PB 或分配预算。

当前正式 `scripts/`、B15 冻结来源和实机命令不变。扩展仅允许显式 `all-ffn-int8-q64` 执行方案，沿用同一全 FFN INT8 配方 SHA，新增 execution/profile 身份，保留原 q128 FP16 reference 与两类旧控制。collector 清理继承环境后设置白名单 q64 并核验实际路径；不得用任意环境变量绕过目录约束。此时尚未登记续接或新增模型采集，首版测试记录保留，真实入口修复使用新的 revision 记录。

账本续接实现使用同一 registry 下的显式继承 segment：父段 contract/events/evidence 和原工具永久保留，绑定父段最终 head、审计锚及两代工具 SHA，继承全部已消费 key 与终态；只给新 B11 q64 执行方案增加 1 次采集和 3 组完整对照 ABBA。父段必须终态、无 pending 和活动 driver，续接唯一且旧写入被 seal 阻止。隔离实现、合成 CPU 反例和定点静态复核已完成；真实父账本重放与续接仍须等待 B15 终态及独立审计，本段不分配或消费新增预算。

后续顺序固定为：确认当前 driver 终态并保存外部 result/head SHA → 运行冻结的原版本 B15 独立审计 → 绑定 B11/B15 审计报告及原 contract SHA 生成 parent review → 完成续接预登记的原始证据重放 → 封存父段并准备/执行新增尝试。必须在 seal **之前**完成 B15 审计，因为原审计拒绝活动 driver 锁，seal 会永久保留该锁。B15 审计不自动生成 index，可在成功后另行打包；需要持续校验的辅助证据须把其 file record 直接加入 parent review，不能只依赖嵌套 index 文本。r3 driver 与 ledger-r2 的目录、head、锁和继承状态接线已定点静态复核，未发现阻断；真实续接集成尚未执行。

单进程三阶段编排器另存于 `target/unified-quant-q64-orchestrator-r2/`，交付 SHA-256 `56b81a6224df627dde05a2deb54f54a997e685ea6549c9e06b69dfbeba6c0ef8`。它在同一个真实 facade 实例上串接预登记、封存/准备、执行，阶段间由 root 核验产物并显式传入 SHA gate；不替换原缓存、不削减逐次文件哈希检查、不扩大预算。r1 的父审计附件仅限 JSON 问题已修复：直接证据只接受有界本地普通 `.json/.py/.log`，控制文件仍只接受严格 JSON，避免遗漏审计脚本和日志。独立与 root 静态复审无阻断；B15 GPU 子命令全部退出后，首轮 33 项合成 CPU 单测全部通过，exit 0、耗时 87.12 秒，80 条绑定文件的前后 SHA 一致。`NEWcpu-r1/result.json` SHA-256 为 `bf67a637cedde6fb01027acc13f5d7f978788ddde74ccaaf8661e71b0d780543`，日志 SHA-256 为 `607c0cf30a13422bbea9ef6fc456645ad65a4484ae262ceb01e1688a0b7476c9`。真实 freeze/serve、父重放和新实验仍为零；不会因工具测试通过而提前封存仍活动的 B15 父段。

B11/B15 独立审计完成后，实际续接元数据已冻结于 `target/unified-quant-q64-continuation-real-r1/`。parent review 直接绑定 B11 的 9 份、B15 的 11 份审计文件，SHA-256 为 `4fab111f2b40d7f8c3af0a5e85cb25138e570163216b1743c161d720f0b632b5`；orchestration manifest SHA-256 为 `2c36fcb46bf30fce2496aac1a530ddd120df0bdce4a5114bc4f3b05eeef8707d`。同一 session 85010 已完成真实父原始证据重放和第一阶段预登记，audit SHA-256 为 `05f4e33a12d99b0ebf8ca4c7776b8e6541a6c7d3e18d98f1246936b874d4fd65`；第二次 root gate 随后完成父段封存与 child prepare。原根锁现为永久 seal，不得清除或重启旧 driver。

活动 contract SHA-256 为 `8f349af938a347723ca0ad2af93a3bcf4808460a5732982cfcbce6bc5fc73669`，初始 child head 为 `73ef22b5bd0d8203e74727ea63b9854f000b688f3c6b3135508e429108331afe`。prepared SHA-256 为 `fcd96c8e92f464116223c3580628e7163333e9d1c603706f8fb9b3ef2ff365a3`；继承已消耗 9/13，仅新增 1 次采集和 3 组 ABBA，旧 B11 四份采集与五组比较全部标记 `REUSE_VALIDATED`。四条新命令使用冻结 r9、旧 all-FFN INT8 recipe、显式 q64、同一 2004 selection 请求、C1/capacity1、warmup128、cycles1 和原 1%/5% 门。等待期间 r4 的 44 项 CPU 单测与 7 项真实进程检查均完成；根重新核验 prepared/contract SHA 和 GPU 空闲后已向同一 session 发送第三 RUN gate，canonical gate SHA 为 `91991793fc41b4861db3427b2e0d282304f07798217c7ecfd41484cfe70a6b24`。编排事件已到 `STAGE_STARTED: RUN`，新数值/性能尚无结论；禁止重发 gate 或重启此会话，计时窗口内不并行测试/构建/重放。运行状态辅助记录在 `ROOT-LIVE-STATE.json`，不能代替实际 ledger/driver receipts。

统一入口还需区分两个结论。**新增加速**继续要求原完整控制矩阵的至少 1% 收益、每侧至多 5% 波动及独立确认；**保留既有方案的迁移**必须另有在采样前冻结的协议，不能事后把 `NO_GAIN` 改为通过。后者只适用于相同模型、相同投影量化语义、q64/tune0、融合、Graph 与实际算法可证明一致，且全部原始输出逐位保持的既有路径。共享 `CudaModel` 或误差小于 6 个百分点均不能代替该证明。

迁移性能协议拟采用零容忍观察退步门：每组完整 ABBA 分别要求 `baseline_P95_geo / candidate_P95_geo ≥ 1.000` 和 `candidate_RPC_geo / baseline_RPC_geo ≥ 1.000`，两个指标的基线、候选重复臂相对极差各自均 ≤5%。首组通过后才运行预先声明的独立确认，第二组须独立满足所有条件，不合并平均抵消失败，不重试到通过。请求、并发、缓存语义、预热与计时边界固定，另外记录启动和显存成本。结论只能为“该冻结负载的两组测量中未观察到退步”，不称统计等价或新增加速。

新配置最大 B1、旧配置最大 B8 时，即使 C1 下观察到全部 physical B1，也仍为不同配置/profile。只有实际形状、Graph、算法和输出均有对应证据，才可限定为 C1 计算路径迁移，不能推广至 C32。迁移不免除最终 ≤6 个百分点留出门、重新加载、单机与 Worker 集成、精度身份隔离等后续验收。上述两种资格须写入未来冻结契约后才可使用；既有 B11 五组 q128 比较和完整矩阵结论保持关闭，B15 当前实验也没有迁移豁免。

源码检查另确认当前 `int8.rs` 仅一次性报告首个 GEMM 的 numerical flags 和 M/N/K，未逐形状记录完整算法配置。因此 `imma=true` 不足以授予迁移的算法一致资格；没有更强证据时只能保留为新执行候选。NVIDIA 提供算法 ID、tile、stages 等配置属性，以及 `cublasLtMatmulAlgoCheck`；后者用于设备/矩阵适配检查，成功仍不保证实际缓冲区对齐等运行条件满足。[算法配置属性](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmulalgoconfigattributes-t)、[AlgoCheck](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmulalgocheck)。未来持久化应绑定库版本并重新检查运行条件；不据此改动当前冻结二进制或登记额外实验。

### 8.21 独立确认准备模块（静态修订，未执行）

`target/unified-quant-confirmation-r2/` 隔离保存确认准备与一次性 receipt 模块，交付 `delivery.json` 的 SHA-256 为 `cfdb5742d28090bb7af5ff13b8fc5ef1e785a6c50dc6722be87d1d8efa394b1a`，模块 SHA-256 为 `3852f9b52ae43436210952ef730369c9a6646d2aed14c82d89024e0f1d9aebdd`。旧 r1 保留不变，原 `scripts/` 与 q64 r3/ledger-r2 来源不修改。当前只完成 AST 和静态审查；38 项 CPU 测试已写但未运行，没有真实父账本重放或确认任务登记。

准备入口使用固定的原 v2 或 q64-v3 verifier 冷 CPU 进程，从原始证据重建完整控制矩阵，不接受外部 PASS 摘要。正候选必须明确指定并固定所有原必需对照、工作负载、实际执行身份、环境和源文件；无正候选时只产生保留基线或证据未齐报告，不消费确认尝试，也不证明基线最优。receipt 为同一模型/实际工作负载固定一个 finalist，整组一次预留，失败或取消不退款。

r1 静态审查发现父快照竞态、v2 可误读半封存父段，以及纯校验拒绝误留全局锁。r2 按 active driver → active ledger → confirmation 的顺序获取自有临时锁，持锁重放后在提交前复核代际、contract 和最终 head；q64 原根永久 seal 只核对，活动锁取自 child。预写校验拒绝释放自有锁，发生 receipt 持久写后失败则只保留 confirmation 锁。两份独立静态复审已关闭原阻断，另核对真实冻结 schema、模块图和源码 SHA；汇总为 `root-static-review.json`。实际冷适配和测试结果仍待补齐。

该模块没有 runner、成功证据准入、holdout 或发布入口，不能把 PREPARED/RESERVED 视为独立确认通过。真实 registry 暂不创建这种仅准备的 receipt；完整执行器须另行冻结命令和进程门、重算四臂证据并验证 Windows Job 接线后才执行。静态接线说明在 `target/unified-quant-confirmation-runner-r1/README.md`：原 ABBA verifier 需要完整 `state.loaded`，q64 还须走实际路径标记包装；确认记录用独立 key 加入查重视图，不能覆盖已消费的 selection 记录。没有完整正候选时不运行确认或打开留出输出。

后续完整执行器已隔离冻结于 `target/unified-quant-confirmation-r3/`，交付 SHA-256 为 `8722bfff589bac0e242dfeb8ae09165aa095b260d9553ca443101bce3097937a`；r1/r2 与原工具均保持不变。r3 增加完整固定组的一次性预留、原 helper 的私有 Job 启动与持久进程门、冷进程原始结果准入。原始验证保留全部 selection 计时身份，以独立确认 key 追加查重，q64 明确经过实际路径检查；所有对照均通过才输出 `CONFIRMED_REQUIRES_HOLDOUT`。独立与 root 静态复审均未发现所审范围内的阻断，分别记录于 `NEWreview-audit-r1/report.json`（SHA-256 `da1b0b47f9bd61b3c5a1b0ed74cf0d2a7dac0c71b6aebe988a917512de9da7ac`）和 `root-static-review.json`。

B15 GPU 子命令结束后，r3 首轮 38 项合成 CPU 单测全部通过，exit 0、总耗时 132.69 秒，冻结来源前后 SHA 一致。`NEWcpu-r1/result.json` SHA-256 为 `d6e9842639cdb5c2c7d1727428162e9ef634f04fd435c28a45bec202fa3fc8cb`，日志 SHA-256 为 `e9cd3162e11946ce7d0077f4acb3262d9287b344a75c28013b4c334d8c076b12`。其中包含合成 PB/原始计时的 CPU 校验，但父状态、子进程及部分 q64 边界使用 mock；不替代实际冷原始重放、q64 marker 或 Windows Job 测试。实际三项冷 help 的结果和 native 工具失败另见下文；未创建真实确认 receipt 或运行确认。

补充静态检查曾指出取消及时性风险：r3 `_execute` 对子进程使用一次完整时限的 `process.wait()`；本机 Python 3.11.9 的 Windows `_wait` 直接调用 `WaitForSingleObject`，可能延后 Python 中断处理。所读 `subprocess.py` SHA-256 为 `4f08d583a95b415762d888fff499c19103040d4b7027e25a73d46c7e3d777d04`。随后下述真实 base Python 进程检查已观察到延迟，不能再将它仅列为未验证风险。

隔离冷集成检查器 `target/unified-quant-confirmation-cold-integration-r1/` 定义了 3 项固定 help 冷加载与 4 项真实私有 Job 检查，未运行原版。独立审查发现 ready JSON 创建即暴露、写完前可能被读取的竞态，报告 `NEWreview-r1/findings.md` SHA-256 为 `c02e571a67e8ffab59abca67f0533440eadd552c02a826173ab1daba6ce79a0b`。新 `target/unified-quant-confirmation-cold-integration-r2/` 已以同目录独占临时文件完整写入、同步并关闭后，以 Windows 不覆盖 rename 发布；delivery SHA-256 为 `100bb61166b4288ad359adb987e2efbe726f5183117ec0f548298a14015dc502`，独立静态复审通过，报告 SHA-256 为 `8d3fc212fa3e2e7a2605ed30225d9493c3039d349bf401c4839567fe9f53a678`。

B15 全部 GPU 子命令退出后，r2 首次真实 CPU 检查已执行：confirmation CLI、raw verifier 和 q64 原 benchmark `_child` 的三项冷 help 均通过；native success 在 `_execute` 调用前失败，`sentinel.ready.json` 的运行进程 PID 与 `Popen.pid` 不同。失败保留于 `runs/first/result.json`，SHA-256 `4924881add5c24c35d1be00dd70fe81813f7b2594bca98998019c5783ffc4ebc`；native 记录 SHA-256 `6769f989cae84e1f7a4618ac7d7bfd711b612facb83657646b0edc0f9e5f6cdd`。首轮未保存两个 PID 的完整对应链，不能从该记录单独证明 redirector 的进程关系。r2 其余三项 native 检查未执行，原失败记录保留。

新 harness r3 显式把纯标准库 native supervisor/fixture 固定为 base Python，冷业务 help 仍用原 venv，并核对实际 executable/base executable、进程镜像、PID、精确 Job 成员和保留的活动句柄。交付 SHA-256 为 `40a6ffc3d0280bbe1e9467297827d3a34d8f3b534f29fa76c317b29d5b404e76`；独立静态复核报告 SHA-256 为 `f0ee27e66c4d70644fa83551b874ec2ef14157c66c2c8207c679d84f35043563`。首轮真实执行的三项 cold help、正常退出、超时清理、放行前取消共六项通过；在途取消失败。请求时刻为 924235.609，Job close 为 924242.375，延迟 6.766 秒；close 时 `KeyboardInterrupt` 已生效且三个私有进程都仍存活，触发原有 ≤2.5 秒和超时前至少 0.5 秒的门。结果 SHA-256 为 `ef566089b17ddbde00e82b9c7296992812d543b9806c11a9d3ce41d28daf2aeb`。没有放宽阈值或把超时改名为取消；后续新 r4 修复需采用短间隔等待及同一总截止时刻，保留 r3 原件和本次失败。此 native 覆盖仍不证明 venv redirector 在途取消，也不替代原始父重放、q64 marker 或真实确认。

新 confirmation r4 仅改等待循环，采用每片至多 0.25 秒与同一个 `perf_counter` 截止时刻，到期仍以 `wait(0)` 检查刚退出的子进程；保留原 gate 顺序、Job 清理、异常传播和总 timeout。交付 SHA-256 为 `f57e022a766c7966ffaf0dc2b09d9ae4ff2498ecc57c3ae21681623382283359`，独立静态复审通过。首轮 44 项单测全部通过，91.83 秒、exit 0、121 条前后 SHA 一致；报告 SHA-256 为 `21c30a2c1f8ed833c139afe968fba633759accbaccdfb11a9432b2a6618e4e7e`。cold harness r4 只重绑定核心路径/SHA，原 probe、fixture 和七项条件保持，交付 SHA-256 为 `b11616c5f44b286fc012826f094b5a2366634657cc7e08a947730a0bee641ae5`。七项真实检查首轮全部通过、exit 0，结果 SHA-256 为 `b8080724698d625881c6af6db329d9dea0ae2f1189f5e81e30d9b914bacf020d`；在途中断从 924873.515 到 Job close 924873.781，延迟约 0.266 秒，三个私有进程当时均存活且 sentinel 未被内层 Job 结束。该结果仍限定为 base Python fixture，不外推到未测的 venv redirector 链或真实确认任务；未创建真实确认 receipt。

### 8.22 并发与单机验收的接口缺口（原型验证与实际接线待完成）

`target/unified-quant-next-workload-r1/README.md` 盘点 C32 后续范围：请求并发、配置最大 batch 与实际 physical batch 必须分别绑定；C1 正结果不借作 C32 数值或性能资格。目前无追踪计时没有物理形状分布和显存峰值证据，原 catalog 单份最多 8 组性能比较，也不能容纳任意多容量的完整矩阵。后续若增加范围，须按完整控制矩阵推导新 key 与有限额度，沿原逻辑账本继承历史；q64 固定的新增 1/3 额度不能用于 C32。

`target/unified-quant-local-parity-r1/README.md` 已完成独立静态接口核对：`kata-raw-nn all` 为八次同步对称计算，不能证明物理 B8；打印的后处理结果经过格式化舍入，不能与 Worker PB 宣称逐位一致。GTP 的让子推断默认与 Worker 不同，`loadsgf` 又使用宽松合法性回放，因此验收需要从严格验证的冻结请求生成线性 SGF、显式统一参数，并检查完整输出数量。在同 physical B1/row 0 比较实际网络输入和五头原始输出 hash 后，再分别核对后处理与非法 policy 集合；trace 输入 hash 不覆盖所有后处理参数，不能省略逐字段核对。这些验证与单机搜索、保存重载均尚未执行，不构成当前性能测试通过或新实验登记。

逐形状算法计划的接入位置和重载契约另见 `target/unified-quant-algorithm-plan-r1/README.md`。源码复核确认，当前 profile 在实际 GEMM 暖机选型前生成，INT8 标记又只记录进程中首个形状且早于可选 tune；因此这些身份和日志不能单独证明全图最终算法一致。后续应分别绑定精度 recipe 与执行算法表，覆盖实际描述符、布局、workspace、缓冲区对齐、库/设备/build 及非 LT 内核路径，在 Graph capture 前检查所有可达形状。此处仅完成静态接点梳理，没有修改运行时或解除统一入口现有的计时调优禁令；不得把该设计写成已保存或已重载的计划。

独立 Rust schema/纯验证原型首版冻结于 `target/unified-quant-execution-plan-prototype-r1/`，delivery SHA-256 为 `f4bc66e1d834c26db7c9b2c47a1700f9624d6b435ce75c6e1923d500bc78b2c3`。独立静态审查发现 `batch_count`/`sm_count_target` 缺少有符号 32 位 ABI 上限检查，报告 SHA-256 为 `86a1ed3b30d51c1dd94b4ebd67e201585348a43db0ba5c180f32487e9e706fc0`。新 `target/unified-quant-execution-plan-prototype-r2/` 已修正，并明确负零位模式、按实际执行内核分配验证义务、LT 属性只作原宽度 getter 的比较值，以及静态 CUDA runtime 身份来源的未实现边界。r2 delivery SHA-256 为 `dacb68e0d01091afbeff9511832273c72d0a6042dc161f5b555417fd9ae86b3d`，模块 SHA-256 为 `578baa397c217544310fad859cb652eb6b2af27d66fd3412dd65b6b5457c53d3`；独立静态复审通过，报告 SHA-256 为 `8df476dec353b53003374cebe6781174f1d0fda769ad762df619faa2f923381b`。21 项 CPU 测试仅有定义，尚未编译运行或接入后端。它仅表达单 GEMM，返回静态准入；多矩阵 DualFFN/SwiGLU、额外 scale operands 等尚未表达，不能称为现有全网完整算法计划。实际库查询、AlgoCheck、子视图/资源检查和保存重载仍待实现。

已有 FP16 融合路径的源码接线另见 `target/unified-quant-fp16-fusion-integration-r1/README.md`（SHA-256 `2ad923cc6199bc07e2ac5738dc2fce43e100cba9140899bdc496748e56607d79`）。统一执行器已按解析后的 FP16 dual 精度复用旧 DualGemm，并非因 recipe 存在而一律禁用；本轮普通统一目录清理环境且没有登记 DualFFN 显式变体，运行默认未请求。B11 的 mid384/hidden1152 符合现有固定内核，B15 mid512 不符合。后续应先验证已有路径的受控执行变体，不能在 B15 强开同一固定形状内核，也不能把开关设置当实际融合证据。该审计未改内核、扩展既定预算或启动新实验。

逐层敏感度与完整成本的真实接口盘点见 `target/unified-quant-sensitivity-interface-r1/README.md`（SHA-256 `d584c884baf968df84193dccdb02306fbe81a020790f8e997bfcf1edae37f06b`）。第一步可复用现有 recipe、collector、comparator，在 calibration 上每次只改变一个 FFN 的 dual/down；现有模拟脚本不能充当生产量化数值基准。成本侧仍缺按稳定 FFN 组记录的结构化接口，必须包含前置 RMS、实际量化/padding、GEMM、反量化残差及可能融合的后继 Gate。当前同步文本 profiler、会关闭部分融合的 debug 和未实现的 `KATAGO_CUDA_DUMP_INPUT` 都不能补足此证据。成本与敏感度只用于提出有限候选，单层误差不能相加作为完整混合图 6 个百分点的保证；本次仍未采集这两张表。

单 FFN 校准配方提案原型已准备于 `target/unified-quant-ffn-ablation-planner-r2/`，delivery SHA-256 为 `81f2b019a0e037c804880a37e659a8c21c4d1d0b1ae76c63578ec8dfea700c12`，模块 SHA-256 为 `1cec2d070c01ec311b752d77ba6d58ad9a4a4df5106d93f2e1c083d94faf8919`。它仅从绑定的 manifest/template 对象生成完整 FP16 参考和逐组 dual/down 同时 INT8 的配方，按固定来源 SHA 精确复用原纯校验与 canonical 编码；不读取模型/语料/PB、不生成执行命令或分配账本预算。原 r1 的合成 B15 用例曾仍用 mid384，r2 改为 mid512 并覆盖 hidden16/1088/1160，同时明确形状字段仅是逻辑维度。独立静态复核通过，报告 SHA-256 为 `a4e6cdf10dd6f414c0b2adc1f24845e29a1ab2043c98f30f5885ddb913bbcb31`；14 项测试仅定义并过 AST 检查，尚未导入或运行。全部敏感度、实际 batch/输入配对与耗时均未测，不能把配方提案当作已完成校准。

上述两个原型随后完成首轮实际 CPU 验证。Rust execution-plan r2 在原字节副本和独立 target 中编译，21 项单测全部通过，cargo 与包装器均 exit 0；报告 SHA-256 为 `6592d449d58edc4eb5adcb762c1845b3da42df5283062613dda560fc31ca0535`，原稿和副本未变。编译本身 9.83 秒、单测 0.87 秒，Windows 包装器因等待编译器后台子进程而延至 911.33 秒自然返回；未重跑或结束进程。它仍只验证静态 schema，尚未接入实际库/AlgoCheck/后端。FFN 提案 r2 的 14 项单测全部通过、exit 0、0.68 秒，报告 SHA-256 为 `46431d31a49fc7448da17d9ff22a2c18aa95fed1541a246363ab881b119d1d4d`。

实际输入提案已独立保存于 `target/unified-quant-ffn-ablation-proposals-real-r1/`，delivery SHA-256 为 `52c389924c143e993fa7ffc1f3eb2ed9fadd61e898695da696065207a5d3fa00`，index SHA-256 为 `986bc618cfdc796855c28306c8dc1da1c34b79c0da355abdd3758bc02b97103f`。B11 的 33 组生成 34 份完整配方，B15 的 45 组生成 46 份，含各自 FP16 参考；其余每份仅改一个 FFN 的 dual/down。四份实际图/FP16 metadata 与既有冻结 catalog 一致；只读取校准请求并核对 2044 条、32 盘、全部 calibration、name/semantic 唯一及原文件 SHA，全语料 request-set 身份继承已审 manifest，本次未重读其它分区。14 份生成来源前后 SHA 相同。根另用纯 JSON 检查所有 80 份规范 SHA、完整投影与每次仅两项精度变化，报告 SHA-256 为 `a2f5d26ad4fbf3281495269b8394cc128017b11fb967b3f665c2ac9cba294568`；首版 reviewer 误要求导出模板数组保持原顺序，其失败保留，新 reviewer 按规范稳定 ID 排序核对，提案原件未变。全部敏感度仍 UNKNOWN，未读取实际模型或 PB、未运行推理或分配实验预算。

最终留出入口的实际限制见 `target/unified-quant-holdout-boundary-r1/README.md`。现有选择 catalog/ledger/driver 拒绝 holdout，但 collector 自身可以读取该分区，comparator 的 holdout 标签也不证明候选已独立确认或未提前用于调优。后续需要同一逻辑账本中的一次性准入，先原始复验全部确认凭据并固定 finalist、参考及完整覆盖，再产生最终模型输出；还需将请求、实际物理 batch/行、编码输入、原始张量和后处理 PB 明确关联。语料合法性与分区构建检查不等于读取留出模型评估结果。当前未执行最终留出；如果补接口会改变二进制/profile，应先在非留出数据完成对应复验和确认，再冻结最终产物，不能临时改二进制后继承旧确认。

### 8.23 完整 FFN 成本接口（已接入，有限实机诊断通过）

实际源码 `crates/kata_nn/src/backends/group_cost.rs`、`cuda_exec.rs`、`int8.rs` 现提供显式 Direct 诊断接口，CLI 为 `crates/kata_nn/examples/ffn_group_cost.rs`。按稳定组 ID 记录前置 RMS、实际量化与 padding、双上投影、SwiGLU、下投影和相邻 midGate，识别实际融合路径；提前分配事件，每次前向仅在末端同步后读取时间。暖机必须达到无新计划/缓存分配，再进入测量。普通 apply 不创建计时器；它仍有新增 owner 检查及空观察器分支，尚无零开销或端到端性能结论。

模型加载、工作区和 recorder 绑定模型实例与 CUDA context。诊断入口在任何前向提交前校验精确输入长度、形状、stream、workspace 和配方；跨模型或 context 拒绝。17 项运行时 CPU 测试、5 项 CLI CPU 测试及 1 项真实 GPU 拒绝测试通过。独立构建目录为 `target/unified-quant-group-cost-build`；本轮诊断 executable SHA-256 为 `3db46ae4029049acc572ffdd619dc1b46f0e9c890ec5afff90dc3c54daa36f42`，正式 binary 未替换。

有限合成诊断 `target/unified-quant-group-cost-cli-r1/gpu-first/` 首轮 8 case 全部通过，包装器与各子进程 exit 0。覆盖原生 B11、剪枝 B15、已有 ONNX；每 case 物理 B1/B3/B8，3 次暖机、2 次事件采集和采集前后普通前向，共 168 次前向。实际覆盖 CUTLASS DualFFN、下投影 Gate 融合、TN/NN、剪枝 padded FFN、逐投影混合 FP16/INT8、融合/分离 RMS 量化、CTA/warp 量化、手写 GEMM 回退。测量组有完整 role、形状、setup、sequence 和实际执行路径证据；同一输入五个原始输出头在观察前/中/后逐位相同，来源、输入和 executable 前后 SHA 一致。报告 SHA-256 为 `e0eb29e783f5dd0e31963ed24544e63a295857b1bc0fd7e69d364da433a6ae7d`。

独立 CPU 只读复核首轮通过，322 个绑定文件前后 SHA 一致；重算完整 recipe 的规范 SHA、投影精度/形状、实际分支与生命周期，核对 72 份暖机、48 份测量、1800 个完整 FFN 区间、120 份原始参考头和 240 份测量头 hash。报告 `independent-review-r1/report.json` SHA-256 为 `ea06302d4f65e3caddac2e1bc48e368edab8b97509a18ca729879f2dccb1a48c`。最后一次普通 apply 的输出没有另存张量，其逐位一致来自冻结 CLI 的实际比较与成功收据，不是独立重算该份张量。后续源码改动前，将本次 10 份绑定的 Rust/host/build/Cargo.lock 原字节保存至 `frozen-source-r1/`，manifest SHA-256 为 `118c9b51e8c0e243deed0dca7c1004f02629610f4b14353833194b3e4e343e30`；原独立构建的 executable 继续保留。

这些结果仅证明采集接口的执行与观察器一致性。合成输入不代表校准分布，Direct 事件时间不等于 Graph 或端到端延迟；同新 binary 的逐位一致也不证明相对旧 r9 binary 数值等价。未据此采用新性能方案、消耗 selection 尝试或读取 holdout 输出。

构建后可先生成合成输入，再显式提供完整 recipe 采集到新目录：

```powershell
$env:CARGO_TARGET_DIR = 'D:/code/Rust_KataGo/target/unified-quant-group-cost-build'
cargo build -p kata_nn --example ffn_group_cost --features cuda --release
& "$env:CARGO_TARGET_DIR/release/examples/ffn_group_cost.exe" --make-smoke-input <新的输入目录>
& "$env:CARGO_TARGET_DIR/release/examples/ffn_group_cost.exe" --model <现有模型> --recipe <完整配方.json> --inputs <输入目录/inputs.json> --output <新的采集目录> --warmup 3 --iterations 8
```

输入 manifest 必须绑定实际 f32LE 张量文件 SHA、物理 batch 和 provenance；CLI 不把 `purpose=calibration` 自报标签当作语料合法性证明。真实校准请求到相同编码张量的关联、敏感度表、候选成本比较及端到端复验仍待完成。

### 8.24 INT8 cuBLASLt 算法缓存持久化（算子级审计通过，有限实际模型往返通过）

`int8_algorithm_plan.rs`、`int8_algorithm_native.rs` 与 `int8.rs` 现有实际 record/export/restore 接点，显式 CudaModel 薄适配也已接入并编译。每个 workspace 在首次推理前显式安装；正常模型加载和现有配置不会自动启用。缓存计划绑定完整外部身份与精确形状目录；记录实际选中的算法原生字节、矩阵描述、配置属性、能力与 workspace，重载时重建描述符、执行 AlgoCheck、重读核对属性并检查实际缓冲区地址/对齐，不另调用 heuristic 或重新调优。默认路径保留原 count=1 heuristic 和现有计算。

2026-09-27 再核 NVIDIA 官方文档：算法描述符可序列化后在相同 cuBLAS 版本中恢复；AlgoCheck 成功仍不保证实际缓冲区对齐合法。因此保存库身份和检查真实指针都不可省略。依据为 [算法结构说明](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmulalgo-t) 与 [AlgoCheck 契约](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmulalgocheck)。在线页面当前标为 cuBLAS 13.4，本机实现仍绑定实际 CUDA 13.3 头文件、加载库版本和文件 SHA，没有随在线文档升级本机库。

源码由隔离 r2 接入，交付 SHA-256 为 `2a0504679570428b0e44942e3f66202a9572c1989e5eec5f7567f40fe0b2a741`。独立静态复审后，24 项 CPU 测试及带明确 host stub 的 native 类型检查通过；报告 SHA-256 为 `5df8d92bcd616f3d73211564146a9634c80c913eb3bcf6081e16773259801836`。实际 crate 首次编译失败于 `DeviceFingerprint` 没有 Serialize，说明 stub 类型检查不能代替真正集成。随后只在实际 INT8 模块增加共用的四字段设备指纹编码 helper，未修改旧 tactic plan 类型或冻结 r2 原件。修正后的实际 CUDA release crate 编译及 24 项 CPU 测试全部通过，原失败与后续通过日志均保留于 `target/unified-quant-algorithm-integration-r2/`，构建目录为 `target/unified-quant-algorithm-build`。

session 保留 owning stream Arc，并检查实际 scratch/input/weight/output 的 context。四个公开 INT8 算子及两个 CudaModel 前向入口的任一错误使已安装 session 失效；未完整覆盖、capture 内首次遇到形状或失败后导出均拒绝。模型薄适配已提供完整 INT8 目录的显式安装，以及外部 Graph replay/copy 结果的 completion 边界；外部调用者必须主动报告错误，该接口不会拦截 CUDA API。该表仅覆盖 INT8 LT cache，没有 FP16、手写/CUTLASS 或 attention kernel 的执行身份，不能宣称是完整模型执行计划，更不能继承旧精度或性能认证。

随后有限原生诊断首轮通过：`crates/kata_nn/examples/int8_algorithm_probe.rs` 按冻结来源构建，executable SHA-256 为 `898f1ffe018c82425edce591ee160cbb90a3d44567f094ff626b73f35b918bf9`。`target/unified-quant-algorithm-native-probe-r1/gpu-first/result.json` SHA-256 为 `02f740c0b7c91a606d7ff6c1ce78321739c827c9ce6a9fd4f7052e745ab0b643`；runner 与 native 子进程均 exit 0。两个独立 stream 各覆盖 B1/B3 下三种投影形状 `(N,K)=(384,1152),(512,1160),(512,1168)`，后两者共享 Kp1168。每个 slot 六个逻辑目录项、四个实际缓存键；slot 1 也成功恢复 slot 0 的表。legacy、record、restore、Graph replay 的 48 份完整 u16 输出逐位一致，各阶段先填 NaN 排除旧结果残留。四份导出表原字节相同，SHA-256 为 `6c94d910027fd9041fd6e23325e9c1e0741a3dd6a2f69c5aad3ec6976955d8b8`。实际记录的六个形状 numerical flags 均为 `0x200804`（库报告 IMMA）；这不是 Tensor Core 指令或利用率 profiler 证据。

固定共 67 次公开投影调用尝试，包含 13 次 capture 内的主机调用及预期拒绝，两个 Graph 各重放一次；没有计时调优或重复运行。错误 SHA/绑定/目录、已有缓存上安装、覆盖不足、完成后短输入、跨 stream、真实非 primary context 的输入，以及 capture 内首次逻辑 K 别名均按约定拒绝；适用的已安装 session 后续推理与导出也拒绝。此处两个 stream 串行执行，不能称并发重叠认证。所有输入/权重为有来源的合成投影，不是 B11/B15 模型权重或整网准确率结果。

来源以真实 DLL 符号定位已加载库文件，并绑定实际 executable、头文件、静态 cudart 构建输入和源码；静态库输入没有冒充已加载 DLL。为后续模型适配保留了 22 份来源原字节快照，`target/unified-quant-algorithm-native-probe-r1/frozen-source-r1/manifest.json` SHA-256 为 `2dc4c87dd6882c8a55fbb07c6dcff0e1d37c3a19008318e39ef3aa6d70b72bd5`。

独立 CPU 只读复审已通过，149 份绑定文件前后 SHA 一致；48 份原始 u16 输出共 16,265,216 个 half 值，legacy/record/restore/Graph 各阶段及逻辑 K 别名逐位一致，两 slot 输出不同。输入、FP32 权重、量化 INT8 权重和 scale 均独立重建一致；四份算法表的 canonical bytes、六个逻辑形状的描述符/算法属性/workspace 绑定，以及 22 条拒绝记录（10 项主负例、12 条后续 poison 拒绝）均核对通过。报告 `target/unified-quant-algorithm-native-gpu-review-r1/report.json` SHA-256 为 `e67822a7bfdf3ec4f5be618c6ae0459548ebecb95c81ab1d39ec8bcba079b003`。22 份来源快照也独立核实与原 provenance 及审计前记录逐字节一致，补审 `frozen-source-verification.json` SHA-256 为 `6d47eabf291b8039265e4038f413a2ec91a6c619775f32a61d5d35dcc842975a`。首次审计启动因选错 Python 环境缺少 numpy，在读取结果前失败；该工具环境失败原件保留，切换已有仓库环境后审计 exit 0，未重跑 GPU。

新的 CudaModel 薄适配入口经静态审查后已接入实际源码，记录见 `target/unified-quant-algorithm-model-api-r1/integration.json`。实际 CUDA release 首次构建 session6717 已 exit 0，用时 21.70 秒；`actual-build-first.log` 与 `actual-build-first-exit.txt` 位于同目录。适配层从实际上传权重和 resolved recipe 推导完整 INT8 投影目录，FFN dual 使用实际 `N=2H`；record/install 不接受外部缩小的形状目录。它返回 stable op ID 列表，但该列表不等于逐 op 执行跟踪，底层仍按去重形状证明缓存覆盖；正常配置未自动启用，仍仅为 partial INT8 cache。

实际模型诊断 example `int8_model_algorithm_probe` 经独立静态复审、5 项 Rust 与 3 项 Python CPU 检查后，实际 CUDA release 首次构建通过（session47774 exit 0）。冻结交付 SHA-256 为 `d3c6ee65ce1384ad93c6a70941859c3732494a6f89b372604c392845ff22d38f`，CPU 报告 SHA-256 为 `6a1fcb1ad95a0d1934063a41bbefe668b225593550a6cff4af9b6b02a99b8416`；CPU 检查器首次环境命令引号失败发生在 rustc/测试之前，原件保留，随后直接链接已核配套的实际 CPU rlib 完成，未使用 stub。实际 GPU executable SHA-256 为 `e527c73df45fc1f385e7f914c44b545fba5d1b51569f21a2ddd57b63b7635597`，构建前 provenance SHA-256 为 `22a6ec31f3c7e418f2742a669eff2a00e33bf6c917746abc2fb3a39446736979`，构建后复核来源未变。

有限 GPU 首轮已通过，session19558 与原生子进程均 exit 0、无超时。仅使用实际 B11/B15 的全 FFN INT8、FP16 attention 配方和冻结合成输入 B1/B3，每个 case-batch 在同一已加载模型上使用三个新 workspace，恢复发生在另一存活 stream；各操作串行。四组检查共 16 次 host apply（含 4 次 capture）与 4 次 Graph replay，实际执行 16 次完整前向。ordinary/record/restore/Graph 的五个完整输出头均在 NaN 预填后保存并逐位比较，通过全部 80 份 f32LE 输出；每组 record/restored/post-Graph 的算法表原字节相同。B11 每个 batch 有 66 个静态 INT8 投影、2 个去重形状，B15 为 90/62。结果位于 `target/unified-quant-algorithm-model-probe-r1/gpu-first/`：runner SHA-256 `a392f58a2b1b26d62904442b9c1f2b3e381a9050d405d240c5372fa82b7071fd`，native SHA-256 `ec50c0a65f913f6efd1781779089a702d35da463eafb45afcb6682e5827eb5ec`。独立原件审计通过：80 份原始头输出共 81,728 个 f32 全部有限且四阶段逐字节一致；222 份原件前后 SHA 相同，保存了 53 份来源、头文件和静态 archive 快照。报告 `target/unified-quant-algorithm-model-probe-gpu-audit-r1/evidence/result.json` SHA-256 为 `d26bc535c01dcd3b0c41511410faa075ea88ebb6d7fba2a222abdb5a47f29519`，快照清单 SHA-256 为 `18317fd3571b242a810e66659b99cbcbfb9b14e98a86c46c54c76c24a68a183c`。去重目录仍含 logical K：B11 每批 2 个 logical keys 对应 2 个 physical keys，B15 为 62/57；五组 K 别名共享的序列化实现完全一致，补充 SHA-256 为 `ab445a6c91fa27e7b23bccdba818d1de818818f21318e66ccd0d33984c5c7759`。没有重跑 GPU。

上述模型比较始终使用同一量化配方，不构成相对 FP16/FP32 的精度认证；只覆盖固定 B1/B3，不能外推其它 batch、混合配方或并发重叠。FP16、自定义 kernel 和 attention 仍未持久化为完整执行计划，没有吞吐测量或正式采用。

### 8.25 校准输入编码（完整 2044 局面导出与独立审计通过）

`kata_worker::evaluator::input_encoding` 与 CPU example `export_calibration_inputs` 已接入实际源码。它直接复用 Worker 的 `prepare_context`、v7 特征函数与空间对称变换，导出 NCHW f32LE 空间/全局输入；保留初始布子、完整合法历史、贴目、15 个 PB 参数及逐行来源。新的薄 Python 入口复用原 corpus loader/PB constructor，原采集脚本与协议未修改。只接受冻结的 2044 条 calibration 源，可显式导出有独立身份的前缀；旧 832 条 fixture 不替代完整校准集。

隔离 encoder r2 静态复审通过，交付 SHA-256 为 `bcefc6f6075551bcdfc4d3b3a43cd42a9ccaa7ffdf8c6f8b15baa329f902df35`。两个完成标记先写入同目录 pending 文件并同步、关闭，再用 hard-link 无覆盖发布；协议清理在发布前完成。首轮 wrapper 测试有一项预期漏算中国规则的显式初始让子补偿，实际共享逻辑正确；仅修正测试为 `(6.5+2)/20` 并断言原贴目及规则，未改生产特征计算。随后 8 项 wrapper、9 项 example 全通过，继承未变化的 9 项 Python 边界检查，共 26 项；报告 `target/unified-quant-calibration-encoder-cpu-checks-r2/result.json` SHA-256 为 `6a427d1fac7a44a8ea110f2366d35b6a500a5b1d715b7370fdb8e0d84b98c17a`。原失败保留。

输入约定另用实际 parser/version/lower_model/graph_manifest 独立核验：B11/B15 均为 model v17、inputs v7、22 spatial/19 global、meta0、19×19/D32，两个历史输入偏好标志均 false。129 份模型、源码与 CPU 依赖绑定前后相同，编译和解析均 exit 0；`target/unified-quant-model-input-contract-r1/result.json` SHA-256 为 `b1558db24e8a5eef6465a639be8c0e5a6eed04993f974b07357ab3701d71ecc3`。此结果没有创建 CUDA context 或运行模型。

首次真实 prefix9 导出在 B11 PB 准备成功后遇到 Windows 主线程栈溢出，exit `3221225725`，原失败 executable 和日志保留。入口比照主 CLI 改为 256 MiB 专用线程，原命令解析、编码函数及错误结果保持；独立静态复核及 CPU build 通过，当前 encoder executable SHA-256 为 `946d3640c4d5f21ee367f26275d6601c142c74c650430294bced100445856017`。新 `target/unified-quant-calibration-encoding-real-r2/prefix9/` 中两个模型的 PB 准备、真实 CPU 编码全部 exit 0，result SHA-256 为 `5cf63937ac2cddf9d7e9c78ca4873f2af1d2aefc1c88598e218b0575b7c4454b`。

prefix9 独立审计首次通过，191 份绑定文件前后相同；每模型 B1/B3/B8 分别为 9/3/2 个 Inputs，B8 尾 B1，无 padding。两模型共 54 条行映射、56 份 f32LE 原件及 1,719,576 字节，原 PB 独立重建、JSONL 参数来源、行切片与 tensor hash 域、跨 packing 和跨模型逐位一致均通过；报告 `target/unified-quant-calibration-input-audit-r1/prefix9-r2-first/report.json` SHA-256 为 `ff253d9f7e08ae8ce34a98cb10e773de72600328dc1ac7adfaa2c77530852cb8`。审计没有重新实现全部围棋历史/v7 算法，没有重跑 encoder，也不是 NN 数值门。

完整 2044 条校准输入随后在同一修复版 encoder 下导出成功，session13195 exit 0；两个模型的 PB prepare/encode 四阶段均 exit 0，分别约 122.57/315.25/81.63/368.21 秒。这些是 CPU 文件生成耗时，不是推理性能。结果 `target/unified-quant-calibration-encoding-real-r2/complete2044/result.json` SHA-256 为 `f76ca8e1684cb045ce0bc064ba0298ff748140e2e5aa766362a218c6e8984de0`，31 份运行绑定输入/源码前后不变。B11/B15 的完整 provenance SHA-256 分别为 `f51a70490737f37488c218a94f34d61e1bde73834f82b5825c27cd96f5dcd19d`、`e64881299585c8aaaa16a5b1b662b36292b5c39463533963324d5a6aab0d3e37`。

冻结的同一独立 auditor 首次全量审计通过，实际 exit 0、57.45 秒；20,245 份绑定文件前后相同。每模型 B1/B3/B8 均完整覆盖 2044 条，分别为 2044/682/256 个 Inputs、16/6/2 个 shards，B3 尾 B1、B8 尾 B4，无 padding。两模型共 12,264 条行映射、11,928 份 f32LE 原件及 390,534,816 字节，4,088 份 PB 独立重建、完整参数来源、原始切片/row/tensor 域哈希，以及跨 packing 和跨模型特征逐位一致全部通过。报告 `target/unified-quant-calibration-input-audit-r1/complete2044-r2-first/report.json` SHA-256 为 `1e8af90f2329cc13d2013fab47b84323389a1aa0aa0630b52a8242469b657e80`；没有重跑 producer、GPU 或读取其它分区模型结果。尾批不能计为目标 B 的耗时样本；这些仍是 CPU 输入来源与打包证据，没有产生敏感度、成本表、NN 数值或性能收益。

为后续共享后处理提取保留了当前 23 份源码原字节及成功 encoder executable，目录为 `target/unified-quant-calibration-encoding-real-r2/frozen-source-r1/`；源码清单 SHA-256 为 `96d7adb3e11e8daca497895c5135c1d91e6474aa32fe6d9debea023601385044`，复制时逐项核对完整运行的前置 SHA。没有重新生成 PB 或输入张量。

此次新增 example 的 serde dev-dependency 对应 Cargo.lock 中 kata_worker 依赖列表新增已有 serde 一行。该锁文件变化在接入后、首次 CPU 检查快照前发生，写入者未观察到；不归因于后续 `--locked` 命令。逐字重建去掉该行后恢复原 SHA，证明无其它依赖变化，记录在 `target/unified-quant-calibration-encoder-integration-r2/cargo-lock-reconciliation.json`；实际检查及 encoder 均绑定新 lock SHA，前后稳定。

实际敏感度与成本采集的接口草案见 `target/unified-quant-ffn-calibration-collection-plan-r1/README.md`，交付 SHA-256 为 `d686ef8ffb76163d74bea270d932265ca1be34c7720c6b089bf0118370245035`。80 份配方由 2 份 FP16 参考及 78 份单 FFN 候选组成。草案区分完整 B1 校准数值和固定少量 B1/B3/B8 tensor 的完整 FFN 成本样本，复用 workspace，避免按每个局面重复 setup/预热。其调用次数仅为静态算术，不是已登记或已消费实验。共享 raw 五头→NNOutput→生产后处理→Worker PB 接口已接入并通过下节 CPU 检查，实际 GPU 路径回归仍待完成；手写 softmax 或只比较 raw logits 不能替代胜率、policy、目差语义。这一采集流程尚未实现或执行，输入完成不等于校准完成。

### 8.26 共享输出后处理（CPU、实际模型契约与 CUDA 构建通过）

`kata_nn::output_postprocess` 共享原 CUDA `finish_output` 的 CPU 五头解码与原 `SharedState::postprocess_output` 数学；既有生产调用在原同步、下载、配置锁位置委托共享函数。policy 六通道各自含 pass 的 stride、对称 5/6 的单次逆变换、optimism、合法着点与 passing hack、白方视角、no-result 条件、score 第二矩、ownership 和旧版本分支保持。原 value/score 的 f64 中间计算及其 f32 边界没有改成另一种精度。TRT 自身 raw 解码不动，其通用 evaluator 后处理仍经过同一共享函数。

Worker 新增 `NativeModelOutputContract`，只可从 SHA 校验后的真实 native 模型字节经现有 parser/lower 构造；契约绑定模型、graph、版本和全部后处理参数的原始位。`prepare_request_pb` 从完整原始 PB 字节和 Worker 原严格 `prepare_context` 重建私有完整历史，并验证编码后的特征行 SHA。完整 PB 身份不可被仅包含特征的 hash 代替。`postprocess_raw_row` 校验整个物理 batch 五头的精确形状与有限性，包括未选行、未用通道及未请求的 ownership，然后复用原 Worker `convert_output`。它不证明 raw 来自哪次 GPU；外层 reader 仍须绑定实际输入、batch/row、recipe、executable 与原始输出文件。

隔离交付 `target/unified-quant-output-postprocess-r1/delivery.json` SHA-256 为 `fb6c480dda405d565a5877c17ff34f2c528a20c042ae93db93d525d8ced832e6`。独立静态审查核对 22 个实际来源、23 个交付文件与 49 份前后相同的绑定文件，确认原数学及调用顺序按显式参数搬移后相等；发现既有 `worker_failure_tests` 缺少 `KoRule` 导入。接入时只在该测试模块补导入，冻结原稿保留。静态审查报告 SHA-256 为 `849b3ef76b408c2bc2001f9c1cf5138e8e1ad716239040e5a9912444b7195ed4`；审计脚本自身首次调用计数断言错误也保留，不是产品运行失败。五个旧文件和四个新文件的接入记录 `target/unified-quant-output-postprocess-integration-r1/result.json` SHA-256 为 `293b072cdc7185bf715265f6eccd92c0d02baa151ce660eee6f8345747f9a6d3`。

实际 CPU crate 检查 session99944 exit 0：共享语义 8 项、evaluator 29 项、Worker evaluator 28 项、encoder example 9 项，共 74 项全部 PASS、无跳过；184 份绑定文件前后相同。结果 `target/unified-quant-output-postprocess-cpu-r1/result.json` SHA-256 为 `bfa9d18a24b835ea1112b488d9786f14b35769734c2c55e6c1f11d8e95cb2152`。新增 Worker 单测使用内部 synthetic contract，实际模型构造按下一段分别验证。未读取 NN 输出或运行 GPU，没有改变正式 executable、原认证计划或正在工作的 Worker。

实际 B11/B15 原始字节的公开契约构造另用真实 CPU rlib 检查通过，session9611 exit 0；rustc 与 probe 均 exit 0，stderr 均为空。每个模型使用三行合成 B3 五头，分布覆盖黑白、symmetry 5/6、ownership 开关与非默认温度/optimism；6 行成功、2 个模型 SHA 拒绝及 30 个行边界拒绝通过。温度改变而特征 hash 不变时，完整原 PB SHA 拒绝实际触发。443 份源码、模型与依赖绑定前后相同；结果 `target/unified-quant-output-contract-cpu-r1/result.json` SHA-256 为 `3fd4065f099e3defe5332402d7b554b8ebaadbae3924aac5377f7cec015f75ee`。B11/B15 contract SHA-256 分别为 `1eabd2a78af00cdd42cc828f23fd9f80b144b21d032ee153a161877424641767`、`886d89dc10a4bd2469422ffdc6e6c760b98c26e4da396cc20a965fbc1ef91849`。初稿探针误把 PB policy/ownership 当作 f64 的两处类型问题在执行前修正，原稿与补丁保留；生产 PB 的 policy/ownership 仍是 f32，9 个标量才是 f64。这只是实际模型契约加合成头边界，不是真实前向或完整 wire admission 认证。

全新 `target/unified-quant-output-postprocess-build` 中实际 CUDA release crate 首次构建通过，session38993 exit 0、117.87 秒；186 份绑定文件前后相同。明确使用本机 CUDA 13.3 与 CUTLASS 3.9.2，没有升级库。结果 `target/unified-quant-output-postprocess-cuda-build-r1/result.json` SHA-256 为 `ca6b631e6bbdf69efefa71542275582ae22080219e3e00b49af5567a6646c289`，`libkata_nn.rlib` SHA-256 为 `de452d9a1531a9addca682974f2a2a5909d86b9926ad00483451525d8d7aa3fa`。没有启动 GPU 推理。26 份当前契约与提取相关源码原字节保存在 `target/unified-quant-output-postprocess-integration-r1/frozen-source-r1/`，清单 SHA-256 为 `2d3530597ed6e574b434c466c24acd2f8b3424806165f130df9bd74b4ad5a592`；旧数学原件仍在 r1 baseline。

实际路径比较须使用同一输入及同一 raw 五头，将抽取前实际后处理与共享包装逐位核对；两次调用同一新函数仅能说明接线一致。固定 Backend + test-only CPU Replay 的隔离设计已冻结在 `target/unified-quant-output-path-probe-plan-r2/README.md`，delivery SHA-256 为 `1327b19b7ef813d13c150a1be4f06317ec911a025a1c347aa621bf925fdb28d4`。设计分开验证固定 B1/B3 的真实 CUDA `finish_output` 与旧/新 Worker 对同 raw 的 CPU 语义，不修改生产调度。该设计后续已实现并执行，实际消费与验证见 §8.27；完整校准数值、敏感度与成本采集仍待完成。

### 8.27 固定 B1/B3 输出路径回归

隔离实现位于 `target/unified-quant-output-path-probe-impl-r1/`。两模型各 4 个完整 Worker PB，覆盖 B1 一行及 B3 三行、黑白、symmetry 5/6、非默认温度/optimism、ownership 开关；输入来自真实 Worker 编码器。本轮输入是有限合成局面，不替代 2044 条校准语料。CPU 输入准备及公共序列化边界检查通过，结果 SHA-256 为 `0350ca3001a3f6bb4f6401f7d3a95f5169e6c6d97c0ea9087e32687418defdf1`。

`target/unified-quant-output-path-build-r1/` 保存抽取前与抽取后的两棵隔离源码树。旧树的五个实际原文件来自冻结 baseline；新树使用当前五文件和四个新增共享模块，诊断 example、ReplayBackend fixture 与 common 字节相同。两套 CPU test executable 和 CUDA release example 首次实际构建通过，旧/新构建各 3,254/3,258 份绑定文件前后相同；结果 SHA-256 分别为 `d25fed371f8c57ab93b3374b44d2a6ad2fea460a4b6e554fd27f1d482aec1739`、`118035436bbe71b5b72123435208ec02f6d984ccac4c35e53a44b1c313ca20f1`。没有替换正式 binary 或向生产路径加入回放 hook。

在使用真实 GPU 输出前，旧/新 Worker 的 `Engine.evaluate` 用显式合成 DecodedBits 各回放两模型四行；4 个 CPU 进程均 exit 0，真实 parser、精确输入匹配、无预热及 4 行/4 批计数检查通过，最终 PB 和 typed-bits 全部逐字节相同。452 个绑定文件前后相同，结果 SHA-256 为 `d01d61915e43948a010477084030bc846467844cfa89d7af16eee67d5c55f6d9`。共享 raw reader 使用实际 CPU rlib 编译，新增 4 项测试通过，检查未用通道、其它行非有限值、五头字段完整性及 f64 位模式；结果 SHA-256 为 `80030c704f43975af6c38b88542bf47655fe888874dd7abb64e71e10afe3d926`。这些合成检查本身不证明 GPU 输出等价。

GPU 使用每模型一份已绑定的全 FFN INT8、attention FP16 配方；两者均显式启用 RMS fusion，B15 历史文件名中的 unfused 不代表实际配置。direct 给所有物理行完整五头 device buffer 预填 NaN；Backend 私有 device 五头没有预填，仅其公开返回 NNOutput 的消费字段预填。两条 Backend 路径必须通过原有完整 raw-row trace 与 direct 全五头 SHA 一致来建立对应。关闭 Graph、pipeline、padding 与 GEMM tuning，无额外 warmup/capture/replay。实际加载的 cuBLAS/cuBLASLt DLL 通过进程内符号地址定位并核前后 SHA，外层在进程结束后核对预绑定库来源。

首轮 `gpu-first` 中 B11 direct 正常 exit 0、完成两次前向；只读比对器随后错误地把 Windows `\\?\D:\...` 与 `D:\...` 当成不同文件，流程立即停止。原 FAIL SHA-256 `018795e0ee508e00ea49c407ff72d234d44108634aab72575a443c34eceaa7ad`、两次消费、全部输出和原 reader 均保留。r2 仅规范化严格解析后的 DOS/UNC 前缀，7 项 CPU 路径检查通过；UNC 两项只验证字符串，不宣称实际 UNC 文件系统检查。新续接先只读验证原 B11 direct，再执行剩余 5 个进程、10 次前向，没有重采旧 D 或放宽任何数值门。

`gpu-continuation-r2/result.json` 已通过，session33626 exit 0，SHA-256 为 `af07dabad1c16cbc72fd10e6f8bec4a608d3adc628614a22a012419fd77a0326`。聚合原一次与新增五次，共 6 次 GPU 模型加载、12 次前向、24 个逻辑行；3,537 份绑定文件前后相同。两模型 direct 共 20 份完整五头文件、20,432 个 f32 全部有限；旧/新各两条 B1/B3 trace 的输入与完整 raw SHA 均精确匹配对应 direct，8 行旧/新 DecodedBits 逐位一致。GPU 阶段只标记 `PASS_RAW_AND_DECODED_CPU_PB_PENDING`，真实 Worker/共享包装最终 PB 比较另行完成。

真实输出 CPU 收尾 `real-output-cpu-first/result.json` 已通过，session46819 exit 0，SHA-256 为 `a08da6ee712e2669054183707452ac99fdc90d82c017c6013c1a266ef0828115`。两模型分别运行旧 Worker、新 Worker、共享 raw reader，共 6 个 CPU 进程全部 exit 0。CPU 收尾先验证 GPU 比较报告及全部 bound_sources，再沿用报告中原 fixture/raw/captured 的 Source，不重新生成预期 SHA 接纳跨阶段改写。每模型 4 行最终 PB 与 typed-bits 在三路间逐字节相同；policy/ownership 使用 f32 位，9 个标量保留完整 f64 位。1,054 份输入/源码绑定及新生成的 54 份 PB、typed-bits、receipt 均前后复核通过，没有新增 GPU 调用。

独立终态审计已 PASS，实际审计进程 PID25004 / session44612 观察到 exit 0。报告 `target/unified-quant-output-path-terminal-audit-r1/first/report.json` SHA-256 为 `b0c9d455476e27f9e61a82b63ad2fd411dc4d7ce6eae96f66767fc8156dea091`，4,489 份去重绑定文件前后 SHA 一致。审计未导入或执行原 comparator，独立核对 20 份 raw heads / 20,432 个有限 f32、旧新 trace/decoded，以及 24 份最终 PB 与 24 份 typed-bits；独立按 protobuf wire 解码后确认三路输出全字节相同。原 reader FAIL、已消费的 B11 direct 2 次及仅续接的 10 次均保留，无重复采集。审计器自身 AST 预检首次缩进错误及原件已保留，修正后首次正式审计通过；审计未新增 GPU 调用、构建或测试。

该诊断仅覆盖这两份 native 模型的固定 B1/B3 输出提取路径，不证明 public Worker GPU 调度形成 B3，也不涉及 gRPC、完整校准、量化质量门、棋力或性能收益。正式 executable SHA-256 仍为 `d218b98dfdf2eb75820465cbf625dfd3f690b91c3bb112f3be1a56d2cc644dcc`，生产 Worker PID78408 继续运行。完整 FFN 校准敏感度与成本仍为 UNKNOWN，原 selection/ABBA 负结果和永久 seal 保持。

下一阶段静态实施交接位于 `target/unified-quant-ffn-calibration-collection-plan-r1/implementation-handoff-r2.md`，SHA-256 为 `387f3f8896459f53d9702b7cde8f8a01fefcb99a423cd4fedfc5b9989b143e34`。待实现严格校准来源连接、分块五头/PB 输出采集、指标适配和有限成本驱动；B11/B15 共 80 份配方含两份 FP16 参考，不等于 80 个待采用候选。单 FFN 误差与耗时仍需完整组合图及完整旧控制矩阵复验。交接文档未登记或执行新的 GPU 预算，历史 171,440 次前向仅为草案。

### 8.28 单配方校准采集器实现与前置检查

实际 example `crates/kata_worker/examples/collect_ffn_calibration.rs` 已接入五个诊断模块：严格输入来源连接、显式调用清单、无覆盖产物与消费日志、分块完整五头/PB 输出、单次上传的 CUDA 运行适配，并附 CPU 边界检查。`kata_worker` 的可选 `cuda` feature 增加直接使用已有同版本 `cudarc` 的依赖，没有修改生产数学或 kernel。单进程只运行一个模型/配方；完整 numeric 模式要求严格有序的 2044 个 B1 输入，诊断模式须显式列索引；成本按物理 B 复用工作区，预算为每 B `W + S×(M+2)`，不在失败后补预热或重试。每次 apply 前持久记录消费；运行库、模型、配方、输入、运行身份产物和日志均绑定并在完成前复核。

首轮实际 CPU 编译成功，但 24 项测试为 23 PASS / 1 FAIL：测试错误假定 protobuf 标量负零 round-trip 会保留符号。原失败位于 `target/unified-quant-calibration-collector-r1/cpu-first`，结果 SHA-256 为 `7628e736c15ec2acc334734fabbdf938b09abc389dcf4a557cd5ee20f18d81f1`。新分块记录明确保留 shared-output 编码前的 f32/f64 bits，并另存实际 PB 解码的 `wire_bits`；标量零的默认值省略遵循既有 prost 行为，不改变生产协议或数学。修正后 `cpu-r2` 24 项全部 PASS、实际 CPU example 构建 exit 0，397 份绑定源前后相同；结果 SHA-256 为 `aa5a7eb77fd806808ec8f8fa5bcb685919e39c793582a3a2ad970de96d69a39f`。该检查包含错 SHA/重复与缺行/尾批/原 PB 全语义与真实 Worker 编码、消费上限、完整头 finite、f64 不窄化等边界，不代表 GPU 失败注入已通过。

全新 `target/unified-quant-calibration-collector-build-r1` 中 CUDA release 首次构建 exit 0，3,455 份绑定源前后相同；`cuda-first/result.json` SHA-256 为 `44f126ccc5930624aab280ec566df76799bad47085e6eb9c3393b1ab50853ed1`，实际 executable SHA-256 为 `784766e823eaf50fe29c5aa924aa0303de1e438855fb5fab6e468fce3e65d0ea`。B11 FP16 参考与 B15 首个单 FFN 配方的真实 CPU preflight 均 exit 0，重新核对各 2044 请求、全部三种 packing 以及选定 17 个物理行的共享模型契约；`preflight-first/result.json` SHA-256 为 `ce25e59ad9bc1a8da997320319616fd8ac378d540a748a4f02fed1e5de7884e1`。截至该前置结果没有 GPU 前向。

全部 80 配方真实 CPU 准入已 PASS：每模型各一次实际 parse/lower，B11 34 份、B15 46 份完整配方通过实际 resolve、完整投影与 canonical identity 检查，14 项错误配方拒绝。`target/unified-quant-calibration-recipes-check-r2/result.json` SHA-256 为 `d4e1d35f3ebdb4b6819e4af63e1dfca93cb5984d3cfe66043619edfe53c36bfd`，598 份绑定文件前后相同，实际编译与检查均 exit 0。r1 因两个同名 serde_json artifact 未能唯一选择而在编译前停止，原件保留；r2 依据实际 collector Cargo fingerprint 选库，probe 字节未改。

有限采集接口诊断已实际完成：B11/B15 各参考与首个单 FFN 共四例，四个 collector 进程及外层 session93099 均 exit 0；各 26 次前向 / 101 物理行，合计 104 次 / 404 行 / 4 次模型上传，没有重试。`finite-gpu-first/result.json` SHA-256 为 `5a5d6613a4a81fb10b8fe8a519af32ab5a4fcf3b9b99740bda7aad25686bb66b`，原前后来源列表 4,084 项相同（列表含重复绑定，不宣称去重文件数）。每例 numeric 为 5 个输入 / 17 个物理行 / 12 个唯一请求，刻意含同请求在不同 packing 的观察与 B3/B8 尾批；四例共保存 68 行完整 raw/PB。成本仅每 B=1/3/8 一份固定输入、W=3/M=2，完整五头 before/measure/after 逐位门通过。这些有限诊断成本不进入完整校准选型，也不是 ABBA 或端到端速度证据。

指标适配器 `scripts/compare_ffn_calibration.py` 已实现：沿显式 Source 核对输入 placement、完整 raw/PB 与前后源，标准库解码 PB，并从绑定源码 AST 提取原 `quantization_metrics` / `compare_outputs` 纯函数，不启动旧 Worker。最终稿 17 项 CPU 检查 PASS/exit 0，结果 `target/unified-quant-calibration-metrics-r1/final-r4/result.json` SHA-256 为 `ad8c91b80b18b7912a8708cea02bfb01a0ba06fd4265e78e2e809e61abe94377`。首轮测试 fixture 的语法错误及后续 Windows 等价路径修正均保留。最终 adapter SHA-256 为 `0ef621cdb37b7ac1d143a9b6766775c40e1788056ed213c70d1560c9f7d02414`。

两个真实有限输出比较进程均 exit 0，`real-metrics-first/result.json` SHA-256 为 `8ae420e8aa8761965d39ac8aa597ca831f09234f916919978f088d68ebc7f764`。仅对首个 FFN 的这 17 个 placement 观察，B11/B15 相对同策略 FP16 最大胜率差分别为 0.04630685 / 0.01388192 个百分点，policy top-1 各 17/17；mean KL 为 8.98227e-6 / 1.04470e-6，最大目差为 0.05403519 / 0.04866791。指标按 placement 权重报告并单列 game/phase/target_batch；两份 `calibration_6pp_observation` 均为 `NOT_FULL_CALIBRATION`，不把重复 packing 当独立请求，也不据此接受完整或组合配方。

独立有限采集终审已 PASS，CPU 审计 session89370 实际 exit 0；报告 `target/unified-quant-calibration-collector-terminal-audit-r1/r2/report.json` SHA-256 为 `e62ea1754166eb21462d958c54cc4ff1d9ff3145373bf9a63927ee5bc050d3b3`。原 4,084 项前后来源列表完全相同，去重为 3,676 个文件；本次审计共核对 3,843 个去重绑定文件，前后 SHA 均相同。四例实际退出、104 次消费、404 物理行、100 份完整五头切片中的 173,672 个 finite f32、68 份 PB 与 typed/wire bits，以及 60 个成本样本中的 2,340 个完整 FFN 组记录（936 个 measured）均通过。首轮审计把 FP16 down 路由限定得过窄，在 B15 padded FFN 停止；失败与原审计器保留。r2 只按实际 kp/activation_rows 分支修正 CPU 审计器，没有改采集源码或重跑 GPU。这是接口与来源一致性检查，不构成完整校准、精度或性能认证。

下一阶段静态交接为 [完整单 FFN calibration 采集交接](../target/unified-quant-calibration-collector-r1/full-run-handoff.md)，SHA-256 为 `e620b344866bf609a5270f61e5e1baa18e6f34e808a45536813b674f12243609`。现 collector 和指标适配器不需重做；仍须实现全 80 配方的有限外层 driver，冻结具体成本样本、W/M 与总调用预算，并连接完整组成本表读取和汇总。完整 numeric 的 80×2044=163,520 次前向只是算术，尚未登记或执行；成本另计。当前统一 FP16 参考固定 DUALFFN=0，不能代表旧最强 FP16 对照。完整敏感度/成本采集、组合选型、完整旧控制矩阵性能确认与 holdout 均未完成；正式 binary、原认证门和历史负结果保持。

### 8.29 完整单 FFN 校准驱动与首次采集

新增 `scripts/ffn_calibration_inputs.py`、`scripts/prepare_ffn_calibration.py`、`scripts/run_ffn_calibration.py` 和 `scripts/summarize_ffn_calibration_cost.py`，沿用 §8.28 的 collector executable，没有修改生产 kernel 或输出数学。输入索引器读取完整校准 provenance 的显式关联，两模型各核对 8,044 份来源、2,982 份物理输入，并保留原 2,044 个 B1 输入顺序。全部张量与 PB 来源保持原始绑定；这里不重新实现 Rust 的 PB 语义校验或特征编码。

驱动 r2 的 17 项 CPU 检查、输入索引器首次 9 项检查、成本汇总器 r2 的 21 项检查均通过，共 47 项。覆盖完整输入缺失/重复/重排、成本尾批拒绝、全局预算、原始配方对象与实际准入身份、先消费后启动、失败后停止、只回收自己的子进程，以及清理异常保留未知退出状态。成本测试含每 B 的 S8/W3/M4 连续序列，不把测量中的 setup 或缺组静默排除。对应结果 SHA-256 分别为 `8e9de15eae10eb939c13850528434be56ca12c2588cbcec148d9d9112b90501e`、`0d16660f16314bc0a6224561ce21e9d088eab13c23d56465f1c12f42bfe71674`、`fa10336a0a1c92d2a17b30a3a075c70648d016c51f55a99713049f8b95d3649c`。独立静态复审及辅助模块接线复核无剩余阻断。四份既有有限实录的成本只读汇总均 exit 0，结果 SHA-256 为 `8a9e50c2476e4da93f17f5e6f43eb7b17cb460e4d02c3f2587e60a041528f77a`；没有重复 GPU 采集，也没有把这些有限诊断成本加入完整选型。

首次完整计划固定 B11 的 34 份与 B15 的 46 份配方，含两份 FP16 参考和 78 份单 FFN INT8 候选。numeric 每配方为全部 2,044 个 B1 输入；成本在 B1/B3/B8 各选 8 份完整输入，每 B 预热 W=3，每输入 M=4。排序键为 SHA256(`rustgo-ffn-calibration-cost-sample-v1\0` + u64LE(B) + tensor SHA bytes)，固定取前 8，再以输入标识和索引消除同分；不读取误差或耗时决定样本。两模型得到相同 tensor 清单，合计 24 份成本输入、96 个行 placement，不能据此宣称覆盖完整 32 盘或搜索叶分布。

每配方预算为 2,197 次前向、2,656 个物理行；全局为 80 个进程/上传、175,760 次前向、212,480 个物理行，其中 numeric 163,520 行。chunk_inputs=64；单配方 1,800 秒、全局 43,200 秒上限；输出预算每配方 256 MiB、总量 20 GiB，启动前要求至少 25 GiB 空闲。失败或不确定中断保留已消费记录并停止，不自动重试、补预热、替换样本或清除永久消费标记。

`target/unified-quant-calibration-full-r1/registration.json` 已冻结，SHA-256 为 `a3f8e70649d1ca9cd8fd540d2c42fdb378efe0cd7215e4679f6b492a20112837`。准备器实际 exit 0，随后全 80 份计划的 CPU 校验与 19,963 份来源复核通过；准备结果 `target/unified-quant-calibration-full-driver-r1/prepare-first/result.json` SHA-256 为 `c3e882015337eb859f3e251e5321cbe87433e23823ba54c808e5bb2093ed0f18`。该准备没有 GPU 调用。

完整采集已首次启动：根观察 session80213，外层 driver PID44160，首个 B11 FP16 collector PID56880；启动与消费原件分别位于 `target/unified-quant-calibration-full-driver-r1/gpu-first/` 和 `target/unified-quant-calibration-full-r1/capture/`。`execution-consumed.json` 是永久消费标记，不得删除或再次运行该 registration。每例退出后先核完整 forward journal、原始产物 Source 与完整 FFN 成本的绑定/路线/setup/五头 bitgate，再进入下一例。当前只登记启动事实，完整终态、78 份指标比较、混合配方选择及端到端收益尚待后续实际证据；没有正式方案发布。运行期间保持冻结源码不变，只允许静态并行工作。

首次进度观察：`b11-000` FP16 参考与 `b11-001` 首个单 FFN 候选均 actual exit 0、各 2,197 次前向 / 2,656 物理行，完整 2,044 numeric 行及成本记录核对后标为 `CAPTURE_VERIFIED`。两份 raw capture 结果 SHA-256 分别为 `69303a69d4399f319e5b01a0e1f0ec5d3d1ca49a85a88b9701ecfa78a1927235`、`c039c0c59abe9aaf3a4dcfd6fccb7845aab853fb1716b84719f8ff423dc6cae4`。`b11-002` 已启动，collector PID41016；这是在途观察，后续应查同一 session 和原始 receipts，不重启。以上没有进行数值差异比较或性能采纳，单进程包含校验/上传/文件写入的 wall time 不作为推理速度。

GPU 窗口中仅静态新增 `scripts/compare_all_ffn_calibration.py`，SHA-256 为 `5409f7e7c7b9e6b391b37f146ba671de8ad06e0642cf4fc446f6f9223eb9bffe`；相应测试定义 SHA-256 为 `618bde9822eb43cbd22c9b47cf4d1f59a229608db8c8df0105b0ba4b9023e9b3`。它要求完整 `FULL_CAPTURE_METRICS_PENDING` 终态，绑定全部 80 份采集和成本摘要，按固定顺序调用既有比较器 78 次并复用两份参考。6pp 不通过保留为数值负观测，不当作工具执行失败或自动采用条件。该后处理驱动在本次 GPU 来源集合之外单独绑定自身与解释器；独立静态接线复审无阻断，但 8 项测试仅定义、尚未运行，也尚未读取全量输出。须待 GPU 采集终态后完成 CPU 检查再使用。

### 8.30 有限混合配方提名与 v2 校准接口静态稿

完整单 FFN 采集继续使用 §8.29 的同一 session80213 / PID44160。最新一次来源日志观察已到 `b11-019`，即 20/80 份均 `CAPTURE_VERIFIED` / actual exit 0；当时进程仍存活、stderr 无输出，尚无完整终态。该进度不是数值比较或性能结果，不能据单进程 wall time 推断加速。运行中未启动新的测试、构建或 GPU 实验，冻结采集源未改。

新增 `scripts/plan_ffn_mixtures.py` 静态稿，固定从完整78份单组比较及80份成本摘要提出有限候选：每模型/物理B=1、3、8，在同一8输入、每输入4次测量中，目标完整FFN的等权平均耗时必须下降且至少6/8输入下降，同时该组全部2044个B1数值观察通过6个百分点门。按“目标组节省毫秒 / max(单组最大胜率差百分点, 0.01)”排序，以policy KL、目差和稳定group ID打破同分；分别在1.5/3/6个百分点的风险代理预算下贪心加入，超过预算的组跳过后继续。单组误差之和只用于可复现提名，不是组合误差上界或加法假设。

全局固定18个提名槽，按模型与完整配方SHA去重；每模型最多9份非空配方，空集合指向FP16，单组集合关联已有单组证据。注意力保持FP16，每个选中FFN的dual/down均为INT8。B3/B8仅有固定成本样本，不继承B1完整数值结论；DUALFFN=0/NOGRAPH=1下的局部成本不代表最强旧FP16，也不能加总为整网提速。脚本不启动模型/Worker/GPU、不写账本、不分配新实验预算，尚未读取本轮实际数值或生成实际提案。19项测试仅定义，未执行；planner与测试SHA-256分别为 `eaf3971db8154781ce0ac989639a969fb6acb64cb2d67e81942107a213dd8884`、`406bf4b4cf5807f517242d13c8e7317b694ba78c707cadc0968584cf9835ee86`。

跨语言准入使用独立的 `admission` / `admission_sha256`：仅包含模型/图/契约、完整配方、role/group IDs与配方Source，递归排序的UTF-8紧凑JSON只含字符串、u64、布尔、null及容器，不含浮点。完整统计清单仍由文件Source和Python `proposal_sha256`绑定；Rust不假定两种JSON库对浮点数的编码一致。v2计划的 `proposal_object_sha256` 明确指向admission SHA，与旧v1语义区分。

Rust多组采集改稿位于 `target/unified-quant-mixture-collector-static-r1/mixture-collector-r2.patch`，SHA-256为 `e214eda5332e7c03f8428a3f03fb6251be7c82391280bdf1d011724121cf0a9d`；stage-r2清单SHA为 `562115de6ec77a4bb47bb91b7385b5331226b770ab18a0d59bdcb511ac5d72b7`。它把共同采集主体抽到子目录中的单份文件，复用现有输入、输出和runtime；新scheduler包装原调用清单验证器，支持完整FP16及最多9份单组/多组候选，真实parse/lower/resolve和完整Source校验仍保留。13项新Rust测试仅定义，未构建、未执行。初稿补丁保留但不得应用，最终使用r2。

对应Python改稿位于 `target/unified-quant-mixture-comparator-static-r1/`：原单组规则提取为默认policy，共享原raw/PB读取与指标代码，新policy识别v2计划、严格核全部33/45组及INT8双投影，要求同一manifest、compiled_sources及采集器executable字节SHA。17项新测试仅定义。独立静态复审发现并修正Python数值类型等值、不同二进制误配、完整manifest上限不一致；新读入上限与Rust均为128 MiB。具体文件和集成顺序见两目录的 `handoff.md`，不把静态复审写成测试通过。

这些补丁尚未应用。历史Source指向原始绝对路径，后续应在隔离checkout叠加当前dirty源码快照后集成，保留原路径；先完成原版本全80采集、78次比较与候选生成，再验证新版本。新collector的参考不能直接复用旧compiled_sources的FP16捕获，需要独立登记的新版本参考与有限候选整网校准，保留所有旧消耗和负结果。

源码还确认两个执行边界：统一backend已支持同一模型多个FFN的FP16/INT8混合，B11未量化FFN仍可走现有CUTLASS DualGemm/SwiGLU；B15的mid512没有对应DualFFN专用实现。当前每个CUDA handle在上传时固定配方，Graph/workspace/异步token依赖该固定模型身份。**这符合§3的v1范围：按batch/shape选择已保存的算法，精度在加载时固定。** 动态按batch切换精度是未来可选扩展，并非本版完成前置；若扩展才需要有限模型/配方集合、不可变batch映射和独立workspace/Graph/token。v1仍须补齐完整执行计划的保存、恢复与生产接线，不能以partial INT8 cache代替。完整旧控制矩阵、确认与holdout仍未完成，正式binary和原FP32认证门保持。

### 8.31 有限混合整网采集驱动静态接线

同一full80进程已观察到 `b11-033`，B11全部34份 `CAPTURE_VERIFIED` / actual exit0，外层PID44160仍存活，B15尚待完成；不能在此时运行数值后处理或追加测试干扰在途成本窗口。B11的34份采集不等于组合精度通过，也尚未比较其单组误差。

`target/unified-quant-mixture-driver-static-r1/staged/scripts/` 新增准备器与运行器静态稿。准备器先从完整父采集、78比较及有限提案产生有序计划，再要求每份计划由同一个新CUDA binary实际CPU preflight，核命令、PID、六项Source、13项编译来源、真实模型resolve和输出契约后才能冻结registration。最多18份去重多FFN候选，加有候选模型各一次FP16参考；每例固定2044有序B1、chunk64、cost=[]，全局上限20上传/40,880前向与物理行。空/单组不自动重采。对应15项准备器与16项运行器测试仅定义，尚未运行。

一次性逻辑根由原full80 registration真实父目录派生为 `mixture-calibration-v1/`，不由新输出目录或binary身份重置；发布registration前先完成准入，运行前再持久写consume及case-consumed。运行器直接复用旧子进程管理，仅处理自己的child，失败或不确定尝试保留并停止；完整journal、五头/PB切片、实际精度目录及runtime逐项绑定后只标 `MIXTURE_CAPTURE_METRICS_PENDING`，不自动采用配方或宣称性能通过。独立静态复审发现并修正了原提名/父模型与输入Source脱链以及遗漏有候选family的问题；静态复审不替代实际测试。

历史继承锚为同目录 `history-anchor-spec.json`，SHA-256 `7cf8b6c2dd0934dea584d3490484055dc345b174563b09323bc351b62fe16f42`，仅定点核17份小文件。原10份collection/16组ABBA、q64最终head和两份永久seal均保留；三份既有C1负配方不会因新calibration、路径或binary而得到重复性能额度。没有重放旧大证据、清锁、初始化空账本或把当前full80误列为完成。新校准不宣称已经接入旧sealed selection API。

collector r3为r2之上的独立静态增量：`target/unified-quant-mixture-collector-static-r1/revisions/r3/backend-build-r3.patch`，SHA-256 `c60a97ba88c62bce3c368e85f4179ef6a6fef4fa35b7c4ea617796fc701fe3ab`。新CUDA preflight在intent中返回实际编译指纹及cuBLASLt版本，旧root和非CUDA版本返回Null；没有context、模型或前向。新增1项非CUDA测试仅定义，本轮未执行指纹函数。准备器拒绝Null，以旧实测device/libraries和新CPU build形成明确期待，未来每次capture再核实际runtime。r2原件未覆盖；集成顺序和尚未产生的真实preflight bundle接口见 `preparation-handoff.md` 与r3交接。

按§3复核，v1缺口是固定精度配方下完整执行计划的持久化和生产接线，动态按batch切换精度不是前置。已有INT8的实际模型记录/恢复链可复用；FP16 LT目前仍在cache miss后重新选择算法，完整DualFFN/attention/非LT路由也未纳入统一计划。FP16按workspace/stream独立记录恢复的代码正在隔离静态目录准备，保留当前双缓冲及TF3 residual preset；尚无构建、执行或收益证据。当前没有新GPU登记，正式binary、生产Worker和原数值门未变。

### 8.32 FP16 持久化与混合校准 CPU 流程静态稿

同一 full80 原进程继续运行，最新定点日志已观察到 `b15-008`，即 B11 34 份及 B15 9 份，共 43/80 份 `CAPTURE_VERIFIED` / actual exit0；PID44160 仍存活，stderr 为空。没有新增 GPU 尝试、读取在途数值或运行 CPU 测试/构建。该在途数量不代表完整校准或效率通过。

FP16 LT 持久化静态交付位于 `target/unified-quant-fp16-algorithm-static-r1/`，delivery SHA-256 为 `f77d0ff75576ea66a97d7614e8ee802fe7b429b982915fa37cc3a2d68e818e7c`，forward.patch SHA 为 `5b99aa14123509fcf87dbc0446795a396580518ddfa3529918f44a8fe2f223dd`。九份暂存源包括独立 schema/state/native 会话、共享 LT 属性读取和 model 薄接口；21项 CPU 检查仅定义。原始冻结源未改，没有构建、原生调用或 GPU 验证。静态打包首次基线 SHA literal 抄写错误的记录保留，修正的是打包脚本，不是运行期算法结果。

每个 workspace/stream 拥有独立 FP16 描述符、算法表和32 MiB工作区，允许原同一runtime/model的双流；两个会话会额外占64 MiB，尚未做内存复用。记录保留原top8启发式选择及TF3 residual preset，恢复时核实际设备/驱动/库/二进制、完整descriptor/opaque/config/capability及AlgoCheck，不重新选择算法。原生launch再核实际buffer/context/alignment；普通legacy路径不改变。当前仍由调用者给叶级shape目录，不能据此声称完整模型plan；自动模型目录正在独立增量实现，须从真实权重、布局和共享分支条件生成每op稳定身份，保留DualFFN及其它非LT路径。叶级shape观察也不等于逐op实际覆盖。

有限混合校准新增静态 CPU 聚合入口 `target/unified-quant-mixture-driver-static-r1/staged/scripts/compare_all_ffn_mixtures.py`，对应12项测试仅定义。它只接纳新混合采集完整终态，按登记顺序将最多18份多FFN候选与各模型同版本FP16参考配对，使用捕获前已绑定的比较器/指标/协议来源；逐份6个百分点PASS和FAIL都保留。只有全部比较完成且来源不变才发布 `COMPLETE_MIXTURE_CALIBRATION_COMPARISONS`；失败保留消费与可用child receipt，不触发GPU重采。输出仅为B1完整校准观察，不是B3/B8数值、selection、holdout或性能验收。独立静态审查发现父编码input/PB原件及外部运行库未显式加入新authority、CPU receipt只保留未核完整链两处缺口，现从父registration继承原SHA并在比较前固定来源，逐child核helper返回与实际exit/launch/log。修正后入口/测试SHA为 `a48184e22de5d00c34b113edde3d4dcb4ef343f62d43aa0afb8bca0d36370d16` / `5febb5060171ca0ff734a59b267d7b8e08db6731bc898c1431d4ae24a62fcbd7`。

同目录 `preflight_ffn_mixtures.py` 静态实现已完成，入口SHA-256为 `ebaa86d9106bf18ce3b97a190f6ee1ef007e3e3ad98505f417b193f9cec0cbb1`，17项CPU测试仅定义。固定检查11个相关Python模块和两个非CUDA example，再从实际Cargo JSON产物取得新CUDA collector，对每个计划只执行真实 `--mode preflight`。最终bundle须再次通过准备器准入后发布；不会在此入口freeze、collect或覆盖旧尝试。独立静态阅读发现实际编排器解释器未绑定的缺口，现要求 `--python` 与当前 `sys.executable` 解析同一路径。Cargo失败只确认原run_child持有的直接子进程，后代清理未知时明确标UNKNOWN并停止，不声称进程树已退出。具体来源/资源边界及CLI见 `preflight-handoff.md`，不是完整传递依赖认证。所有新代码和测试须待原采集终态后，在隔离checkout叠加当前dirty源并集成验证。

后续同一session80213轮询仍在运行；最新定点观察到 `b15-013`，共48/80份已验证，PID44160仍live，尚无完整终态。该观察只更新进度，不追加或重启尝试。混合比较聚合器上述修正版已完成独立静态复核，未发现剩余阻断；未执行12项测试或真实多配方报告集成。

### 8.33 自动矩阵目录、同 stream 工作区与生产接线静态审计

同一 full80 原进程 PID44160 仍存活，定点日志最新观察到 `b15-027`，即 B11 34 份与 B15 28 份，共62/80份 `CAPTURE_VERIFIED` / actual exit0；stderr为空，尚无driver终态。没有重启、新登记、在途数值读取、CPU测试或构建。以下工作均为静态源副本、文档及小范围命名文件打包。

自动模型 LT 目录增量已封装在 `target/unified-quant-model-lt-inventory-static-r1/`：delivery SHA `4eec720809dafa256c975d7d0c22dfb4493aa9dbf115ded1aec2f019c746d75e`，在FP16 r1之后应用的增量patch SHA `274f0784cb2ee92904555669b6a13ff66b224f1e9511fd3de67db0472bd8e833`。从实际layers、loaded weights、recipe和共享路由函数生成预期矩阵操作目录，包含FP16/INT8 LT键及custom/DualFFN等非LT原因；双GEMM实际次数/列数与SwiGLU后输出列数区分，stem.global显式记录为FP32融合dot+broadcast。DualFFN按实际effective enabled绑定，UNKNOWN→READY不改变身份，disabled或实际路由变化拒绝。新增16项CPU测试仅定义。它不覆盖完整attention QK/AV，也不是逐op执行追踪；叶级去重shape表不能替代完整目录SHA或完整计划。

正常backend只读审计见 `target/unified-quant-production-integration-static-r1/integration-handoff.md`，SHA `28a61000f704f1cd0c4f54e28901239266c03a3bd31a0dc536118410f9266956`。现加载流程在Uploaded转换时丢掉graph/局部recipe，生产接线要先保留不可变元数据；安装点在每physical batch workspace创建后、两slot首次warm/capture之前。PADBATCH=0需覆盖1..=capacity每个实际batch，PADBATCH=1覆盖1与capacity；B1/B3/B8诊断不代表完整运行覆盖。Graph重放绕过普通apply，需提交前检查及所有copy/event/sync失败上报；首次batch初始化失败须毒化handle，避免隐式重建重试。profile还要绑定完整execution package SHA，模型SHA协议本身不证明Server缓存已隔离。以上仍未接入正常backend。

审计同时发现每batch独立32MiB FP16 workspace的乘法：单handle容量32、所有实际batch出现时可新增1GiB。新静态增量 `target/unified-quant-fp16-shared-scratch-static-r1/` 提供单owner stream的共享 `Arc<Fp16LtScratch>`，各batch保留独立描述符/算法表，按数量推导可将该部分改为每handle32MiB；其它缓冲区不合并，尚无显存或速度实测。delivery SHA `9bda23476accedd8708e481e6f9b41c2875d2d7a01f9054a569f027a57ebc8a1`，overlay SHA `b93e777350d6ffb500d939987ba8a5cb278296cb1e06094b1c207e2184dbd45f`。这是前两份patch之后、校验base SHA再复制五份完整staged源的静态交付，不是已应用补丁；新增9项CPU测试未执行。

共享guard锁住整次Lt主机调用，按实际Arc对象核stream/context、共享错误状态；模型安装前失败也将传入scratch标坏。新模型提交API接收闭包，先核model/routes、INT8/FP16状态，再在同锁内调用Graph/slot提交；旧complete(result)只是后置报告，不能拦截已求值的launch。opaque闭包必须实际使用同一owner stream，禁止重入共享scratch或执行capture切换。由于外层借用mutable workspace，五头DtoH/status不能放入同一闭包；正常handle可保持原单提交锁分段处理并完整上报错误。warm/capture、在途清理与Graph销毁仍属正常backend待接责任，不能因持有scratch Arc就声称整个生命周期已闭合。两个独立窄静态审查未发现限定范围内阻断，编译、实际CPU/CUDA/Graph验证和生产接线均未完成。

[NVIDIA CUDA13.3 cuBLAS文档](https://docs.nvidia.com/cuda/archive/13.3.0/cublas/index.html#cublassetworkspace)将sm12x的用户workspace推荐值列为32MiB，并指出单次库调用可能包含多个依赖同workspace的kernel，多主机线程共享用户工作区需串行库调用；[cuBLASLtMatmul](https://docs.nvidia.com/cuda/archive/13.3.0/cublas/index.html#cublasltmatmul)要求至少256-byte对齐。同stream跨batch复用及Graph共锁是本项目据此选择的实现，仍需实机证明；不改变既有精度门、有限预算及负结果。

后续仍先等待原80采集的真实终态，再用冻结原比较器完成78份CPU比较及有限混合提名。执行计划各静态增量须在隔离dirty快照中验证，不能为提前集成而修改旧Source锚。当前没有完整混合模型精度验收、端到端收益或holdout通过。

### 8.34 完整单 FFN 采集终态与组合执行包

原 full80 首次采集已完成全部 80 份 `CAPTURE_VERIFIED`。2026-09-27 根观察 session80213 返回实际 exit0（工具 chunk `b7bce7`），原 driver PID44160 已不存在；`gpu-first/exit.json` 记录 actual_exit_code=0、failure=null、stderr 为空，SHA-256 为 `b5c08991bfab123df90b9be1b093ca147414c49998bcf5fcf34294cd7f4b0f4c`。完整 `capture/result.json` SHA 为 `1341821a66fb302ca61a33b339dd3e3a0df919ec28d438b4269041f619f99320`，状态 `FULL_CAPTURE_METRICS_PENDING`、verified_cases=80、sources_unchanged=true，error/source_error 均为 null。6,157.38 秒是含上传、记录和校验的外层 wall time，不是推理速度。原 registration 与永久消费标记保留，不重采。

后续 CPU 入口位于 `target/unified-quant-calibration-comparison-driver-r1/run_once.py`，SHA-256 为 `bd5a41cbd9fd04b056047bd1cd89f2a781529b0477df4e60851a17417ec9a6db`。它绑定上述两个实际终态 SHA，先执行原比较器 8 项和提名器 19 项测试，再调用冻结原版本完成 78 份比较及固定 18 槽混合提名；输出目录必须首次创建，逐 child 绑定实际 command/PID/launch/exit/log，失败停止，不调用 GPU、模型或 Worker。首次启动 session22049、wrapper PID78888、比较子进程 PID35220；8+19 项实际 CPU 测试均 exit0，78 比较已开始。后续检查同一 session 与 `cpu-first/` / `target/unified-quant-calibration-comparison-full-r1/` 原件，不重复执行入口；完整比较和提名终态尚未观察。

attention 路由静态增量已封装于 `target/unified-quant-attention-inventory-static-r1/`，delivery SHA 为 `063e37117420bcd002a440932373972585f538a35dde805f0a3758324c1f484c`，补丁 SHA 为 `f084cefe59dc304a59ace086b6623a54adcebba8453b18c03447d1edf353c5a5`。七份暂存源、13 项 CPU 测试定义，将实际共享分支选择与 launch shape 用于目录：attention core 只计一次，关联 QKV/out projection，保存实际 B/S/H/D、qk scale 的 FP32 bits 和 QKV 精度；覆盖 v3/q128/q64/serial/H12/generic 与现有 strict 路由。非统一普通路径保持原行为，目录拒绝不支持的固定配方；不是逐 op GPU 实际追踪。此交付封存时仅经静态复审，没有编译或执行。

组合执行包静态交付位于 `target/unified-quant-execution-package-static-r1/`：delivery SHA 为 `5aa384dc548637d89d67c7ed065281bd070f94e7f171d922a89497fe679a2512`，composition SHA 为 `1a4f70d9580c464f39cf7ad4c21d0070daf3a1170b27c8fbba14feab3947b324`，source receipt SHA 为 `71c44c486ea7954607aef87e6da81fddfd5c5f363813cbc23a935f075b1d786f`。composition 给出 FP16 基础、自动模型目录、共享 scratch、attention 增量及本包的最终源覆盖顺序；后续只能在新隔离目录中的当前 dirty 源快照上核每份 base/source SHA 后一次应用，不能再重复套用父补丁。首次静态打包因未读取 attention receipt 中独立依赖字段而 exit1，失败脚本与观察保留；补充读取原冻结字段后 exit0，没有重绑源 SHA 或运行实现。

包 schema 将实际运行容量、PADBATCH/Graph/pipeline/input layout 与全部可达物理 batch 的目录、FP16/INT8 叶计划一起绑定，普通覆盖 1..capacity、padding 覆盖 B1/capacity。目录由实际 model/graph/recipe 生成；叶的存在性与实际形状集合一致，INT8 同时核逐 op ID/key。全 custom 路径也核 owner/context、环境、policy 与路由。安装前置失败标记防止早退后回落普通执行或重装，输入长度与分配 stream 身份在 kernel 提交前验证；导出要求成功非 capture 前向、owner 同步和叶表完整，恢复后回导须相同。23 项新 CPU 测试在封存时仅定义，未执行；不把 schema、代码审阅或提交成功当成数值/性能证明。

随后隔离 CPU 验证已完成：从当前 dirty 工作树复制的 340 文件快照核 base/overlay 后仅应用 composition 一次，以无 stub 的 rustc harness 直接引用六个实际 CPU 模块及其原测试。r1 在依赖选择阶段发现 Cargo JSON 含两份 serde artifact，尚未调用编译器即停止，失败结果 SHA `6c13eead2b62df267266d4397d8dfb8f930bf7260fa615973ba5495089bb0280` 保留。r2 仅修驱动，依据实际 Cargo fingerprint 的依赖整数身份选择产物，复用同一组合快照而不再覆盖源码；106/106 实际通过（INT8 24、FP16 21、scratch 9、LT routes 16、attention 13、package 23），编译与测试均 exit0，947 份绑定来源前后相同。`target/unified-quant-execution-package-cpu-r2/result.json` SHA 为 `7d4db90653eca6b950fe741054f66baaada451d79ef57ecfcc849cdd450bf992`。纯 CPU 快照的后缀过滤未包含 AOT 参数模板 .bin，不能直接当完整 CUDA 构建源；后续 native 验证须在新派生快照补齐实际编译输入，保留本次原件。

有限混合采集的独立源码集成也已完成，位置为 `target/unified-quant-mixture-integration-r1/r2/source`，未混入上述执行包改动。499 份当前 dirty/untracked 源文件共 12,834,800 字节前后不变，最终 513 份文件，保留 AOT 参数 .bin 等实际构建输入；collector r2→r3→比较 policy 后叠加准备/运行/preflight/聚合脚本。r1 因 git 保留 CRLF、输出 bytes 不等于冻结 staged 源而在严格核验处停止，原件完整保留；r2 逐 patch 保存实际输出，确认仅换行差异后写入已核 staged 精确字节，没有改实现。helper 实际 exit0（session10557、chunk `9fe6a3`）；integration SHA 为 `4fba7dc5642112cfb97ebfeb1049ab18128fa4e002afbbf73aa53cd5980c8284`，final-source-manifest SHA 为 `7868c7754e7e574b39dde038ad14d9e4977a87171629219bf8cd271185caeae3`。这只是源码集成，尚无新测试、构建、计划、preflight 或 GPU；必须等待原 78 比较及真实提名锚再继续。

正常 backend 仍需保留 graph/recipe、接入 package profile 与实际 handle 容量、完整 batch 安装、Graph replay 前检查和生命周期清理。现有 INT8 adapter 仍明确拒绝 ffn_compact，组合包 v1 不含 MXFP8；原统一 MXFP8 路径未改。CUDA 构建、实际恢复/Graph 验证及生产接线均需后续真实证据，正式 binary、生产 Worker、原 FP32 门和历史负结果保持。

### 8.35 执行包 CUDA 库编译与正常加载接线增量

从 §8.34 已组合的 CPU 快照建立独立 `target/unified-quant-execution-package-native-r1/snapshot`，补入三份实际 `params-template.bin`，没有重复应用 composition。首次 `cargo build -p kata_nn --features cuda --release --locked --offline --message-format=json` 实际 exit0、110.187 秒；退出 receipt SHA 为 `f457eb1a2cf14200f8d2642e95f9ef3a139333a40871ad949d4c760289725b3b`。外层后检查误用了 `cuda_build_info.rs` 文件名，实际生成文件为 `cuda_build.rs`，因此 r1 runner 保留 FAIL 原件（SHA `d947d2ac642792570b9e3ee2eb53072a8e9d82dc4b498177510f6f8312906049`）；不是 Rust 编译失败。

新 `target/unified-quant-execution-package-native-readback-r2` 只读取已产生的 Cargo receipt/库/构建元数据，没有重编译或改源，实际 exit0/PASS。结果 SHA 为 `754390af9447f8a101868b7cb977a8a595590667219b48459212fe9dab2fbdf5`，5,059 份绑定来源前后相同；343 份构建源为原 340 份加三份参数输入，`libkata_nn.rlib` SHA 为 `48cb44ceda6b0ab82f91a5d1572faeaf7819709289a222515308b74552d5680b`。实际生成资料核实 CUDA13.3.73、CUTLASS3.9.2、SM120/SM89 共30份PTX及 DualFFN/q64/serial 编译能力。这里只是库编译与产物核验，没有运行模型、GPU、Worker 或证明应用级链接/恢复成功。

正常后端增量在 `target/unified-quant-backend-execution-static-r1/staged/` 静态准备，未应用或构建。新增配置对 `cudaQuantExecutionPackage` / `cudaQuantExecutionPackageSha256` 必须同时提供；只读一次有界精确字节，包不覆盖 tactic，要求匹配现有冻结策略。加载时共享保留实际 graph/recipe，验证全部可达 batch；包 SHA 加入现有 profile，省略包的旧 profile JSON 保持原样，已上传模型拒绝换包。每 handle 共用一份必要的 FP16 scratch，每个实际 batch 在首次 warm 前安装叶计划，Graph upload/replay 使用前置 guard，输出拷贝在原 handle 锁内分段完成并传播错误；执行尝试失败保持 handle 失效，新 batch 初始化失败不自动重建。局部清理 guard 在各槽资源首次提交前建立，成功 handle 的 Graph/session 显式早于 model/runtime/scratch 析构。空批次/超容量等公开参数拒绝仍沿用原语义，不表述为“任意 API 错误均毒化”。

该增量尚依赖正在实现的共同实际 Binding 工厂，不能从包自报 Binding 反推实际来源；主机源/所选 cudart 静态链接输入由构建期嵌入，runtime 复用已有环境读取。四项新增配置读取 CPU 测试仅定义，静态审阅未发现当前限定范围内阻断，不替代编译或真实 normal backend 验证。诊断 recorder 生成的包绑定其 example executable SHA，不能给不同的 `katago-rs` 二进制部署；后续可部署离线记录必须在同一正式程序的命令/共享记录器内执行，不放宽 executable 身份。

原 CPU 比较仍是同一 session22049/PID78888/PID35220；最新日志定点观察为 45/78，B11 全33及 B15 前12份均 `COMPARED` / 6pp PASS。完整终态与提名尚未观察，单组通过不证明组合误差，也不提供性能采用资格。原完整采集永久消费、历史负结果和生产 Worker 保留。

### 8.36 单 FFN 比较终态、有限混合提名与执行包集成检查

原 CPU 比较/提名链已完整终态，`target/unified-quant-calibration-comparison-driver-r1/cpu-first/result.json` 状态为 `CPU_COMPARISONS_AND_PROPOSALS_COMPLETE`，SHA `9bb59fdc588ea847c2e3f05336001dda72e9610a82cf8271f4c7618231da7b1a`。实际比较进程 PID35220 和提名进程 PID83080 的退出凭据均为 exit0；后续观察时 session22049 句柄已不存在，不能据此补写外层 wrapper 的 OS exit0。78 份单 FFN 比较（B11 33、B15 45）均在完整2044个B1校准局面上通过当前6个百分点实验门，index SHA `7d9aba0b0ab14f828da0ab89f0a4baee8d268bf6442ff51c6d72842e32509d6f`。这不是完整混合图、搜索叶分布、棋力或性能验收。

原固定规则提名得到12个非空混合配方，proposals SHA `fbb1e3a042e11d5c2c8e5491e86bd5ddf1bbb825d754ee8f69e6c9701eacfae5`：两模型各6个，来自B3/B8的三档风险预算；B1的六个槽均为空。组合误差仍未知，单层误差相加只是提名启发式。隔离 mixture 源首次 plans 实际 exit0，生成12个候选与2个同构建FP16参考共14case；plan-set SHA `6f28fa44ba8222368292cd2919efea5e334c01bbb22f8ad89fd9fac6b481f1f3`。不重建旧提名，不复用另一 executable 的FP16捕获。

独立只读核验上述index/proposals及模型提案、对应最大项report的SHA：B11最大胜率偏差1.113915个百分点，来自 `b11-015` / `trunk.block04.pair02.ffn`；B15为1.213586个百分点，来自 `b15-037` / `trunk.block12.pair00.ffn`。B1虽分别有2/3组平均耗时节省为正，但最高仅4/5个输入胜出，没有组达到8个输入至少胜6个的固定成本门。B3成本合格组分别14/10，B8为33/23，不能将这些局部结果写成整网收益。

| 模型/选型batch | 风险预算1.5pp：INT8 FFN数/代理和 | 3pp：数量/代理和 | 6pp：数量/代理和 |
|---|---:|---:|---:|
| B11/B3 | 3 / 1.094988 | 6 / 2.900088 | 11 / 5.927199 |
| B11/B8 | 5 / 1.446468 | 9 / 2.997929 | 15 / 5.908102 |
| B15/B3 | 3 / 1.233259 | 6 / 2.920336 | 10 / 5.542365 |
| B15/B8 | 4 / 1.469681 | 7 / 2.996477 | 13 / 5.945291 |

表内代理和是各单FFN的B1最大偏差之和，不是组合误差上界；选型使用B3/B8局部成本不代表已验证这些batch的数值精度。

另逐份读取并核验78份report的bytes/SHA，以下为每模型全部单FFN候选中的最坏观察，仍以同模型FP16、2044请求、物理B1为参考：

| 指标 | B11 | B15 |
|---|---:|---:|
| 候选平均policy KL的最大值 | 2.93043e-5 | 6.65952e-5 |
| 任一请求policy KL最大值 | 8.63675e-4 | 6.58501e-4 |
| 最低policy top-1一致数 | 2035/2044（99.5597%） | 2031/2044（99.3640%） |
| 合法policy概率元素最大绝对差 | 0.0136237 | 0.0140074 |
| white_score_mean最大绝对差（目） | 0.233250 | 0.246700 |
| 候选目差平均绝对误差的最大值（目） | 0.0125808 | 0.0164135 |
| 后处理ownership元素最大绝对差 | 0.0521161 | 0.0803699 |
| 候选ownership RMSE的最大值 | 0.00213104 | 0.00356459 |

KL为自然对数 `KL(FP16 || candidate)`，policy元素误差排除非法落点负标记（每候选545,064有效元素）；ownership为后处理值（737,884元素），不是胜率百分点。B11首选分歧均属旧诊断定义的near-tie（参考前两名差<0.01），B15另有1个非near-tie分歧，来自b15-045，参考间隔0.0126955。旧阈值诊断B11 33/33和B15 45/45均FAIL，ownership均有超限元素；“78份胜率6pp通过”不得写成全部输出头通过，更不能写成原FP32认证。原始头指标另存 `raw_heads`，不与后处理表混用；这些汇总没有P95或棋力结论。

该隔离源第一次CPU预备流程已保留失败：前四Python suite共54项PASS，第五 `test_ffn_calibration_inputs` 因隔离 `ROOT/target` 缺历史编码fixture而在setUpClass失败，实际0项；未进入Rust、CUDA build或模型preflight。preflight result SHA `86afa0c99f8542fb1976913864eb8f6f0cf314a4a606a7de9e7a9804331433d0`，外层result SHA `ae78bb9569ad0414c0c0359b5d5a03edb167406f5248e69114dbe52c48bfce3f`。只读核验原1,403直接Source及513冻结源码未变；所需fixture精确清单为16,088文件、521,851,654字节，SHA `b4e3d01c2ffcf046a2d7176d1988fab3764fab50b035a8a6c80f9693e40df0b0`。整改仅复制该清单的精确输入到隔离target，不建可变目录junction、不重写plan。修复后允许在新输出目录重验原完整CPU流程（包含重复54项），明确累计CPU消耗，不重置GPU/性能预算；尚无新混合采集登记或前向。

fixture精确复制已完成，16,088文件/521,851,654字节原/目标一致，复制清单SHA `d87c5e1bb63fc26a61551afefa68dc5798dac13f2ac991a3a676a79a641fe217`。第二次原入口session74328实际exit1，前10suite共151项PASS，第11suite的17项中16通过、1失败；完整成功suite累计205次，通过individual累计221次。失败测试的合成case只给name而漏plan，导致controller准备intent时KeyError('plan')，未调用预期失败的mock child；真实14case计划均有plan。inner失败SHA `fefb0e1fa789dda330426949593a2d9bc7b94bdbb5c3f5efed571bae459f03dd`，outer SHA `bf5a6bd7e5f9f52dbcc387daf448df1be3967b1f390f242138eb1668b539c5c4`；仍未进入Rust/CUDA build/preflight/GPU。单项CPU诊断真实复现提前异常，后续仅在新派生源修该fixture并断言确为child失败，原513冻结源不改，旧失败/计划不重写。

共同 Binding 工厂静态交付 `execution-binding-static-r1/delivery-r2.json` SHA `9d182b6ebf116c00739f8a9b781632439006266f4d848a528176801e83ba722b`，保留22字段，嵌入构建时有限来源集合与实际选定静态cudart archive；不宣称完整工具链传递依赖或最终linker成员认证。正常backend静态交付SHA `9a9cc58f1d6256fe1d02d9a2af1b96b42f61096d5f9ec7c3a8c5272babeaeb94`，诊断probe静态交付SHA `0cfb7d42801ca13c80d3b41156db25934c52d53551943ea669f55836afcb0c17`。三份增量共14个唯一目的文件，已与343文件native基线组成352文件完整源图；仍未改原冻结工作树源码。

执行包集成runner r1的 `--nocapture` 会让构建输入测试的日志打断Rust harness单行结果，静态审查后保留r1，新r2去除此flag并增加实际embedded provenance可用检查。r2首次执行在build script实际编译失败，E0277来自archiver参数迭代错误类型 `&str` 与别名 `String` 不匹配；actual Cargo exit101，0项新测试、0GPU，5,454绑定文件前后不变。失败result SHA `e09c43fab9e1aaa080b8d6dd38b04247790e168c4f843cc5235d5a23bfa87d74`，session45301实际exit1。修订只将该处转换为拥有所有权的String，在新final-composition-r2明确覆盖来源，SHA `49b263fae4e65c844176f81e48954ce8881015ea835e47e343e5db7b0ba6e3b3`；原失败源/manifest完整保留。新 integrated-checks-r3 同一有限检查范围已首次启动 session2929，目标为133项库CPU检查、10项probe纯CPU检查及应用链接，尚待真实终态；不会执行probe或GPU。

随后同一session2929真实返回exit0（chunk `438473`）。集成r3最终PASS，result SHA `8ec148782cb7a5beef2499f04af0cf529b09f5a0cd03f6a245fd32e1f7c9760e`，143/143纯CPU检查通过（库133+example10）；实际三次Cargo链接均exit0，耗时175.278/6.812/24.173秒。实际构建的embedded provenance非Null、ERROR=None、CUDA组摘要、静态cudart archive和workspace源字节核对通过，5,525绑定文件前后不变。普通probe exe SHA `92ca23d9eded73331c5bb5be7451b0dae2023a16bc70812862875d138769f6c8`；尚未运行该普通程序、模型、GPU或Worker。无须重跑已通过的143项检查。

诊断example包仍绑定其自身exe，不能部署到另一katago-rs。后续同正式程序离线录制的静态接线方案在 `target/unified-quant-production-recorder-static-r1/handoff.md`，SHA `d1e8dbbe59ba81cc7a23c228f39a4f787c2c7488011b6a6b6f95ad319af3537c`：应直接复用正常Backend handle的record/restore/Graph双槽，不再另造CudaModel录制路径。当前Worker协议仅model SHA，backend_info仅诊断，服务端profile/缓存隔离仍需实际部署边界或协议接线与验收。该文档没有新增实现或GPU预算。最终精度/性能验收仍未完成，当前未确认新整网收益，正式binary与生产Worker保持原状。

### 8.37 执行包六模型配方预检与混合采集测试夹具修复

执行包普通probe已完成首次真实CPU预检：`target/unified-quant-execution-package-probe-cpu-r1/result.json` PASS，SHA `025e285e797dad41a71eade9e00733b233ee8edeb0148d7896d7d89168ae11cc`；session5355 wrapper实际exit0，probe PID27800实际exit0/18.250秒。使用§8.36同一exe与final-composition-r2，实际解析B11/B15/ONNX各FP16和首FFN两投影INT8，共六case；生成36个合成输入文件，5,553绑定文件前后相同，352编译源未改，所有GPU计数0。registration SHA `3d1330bf5a72db8a325bc989d23855b37e1f354ff9c436469b65ee0d199482e3`，明确不是校准或性能证据。

该registration固定后续一次合成GPU诊断：6次上传、12条stream/12份scratch、36对输入上传、54个workspace、66次host apply（54非capture+12capture）、12次Graph重放、144物理行、330个raw头文件，及有限guard/注入/组包检查。已准许外置driver在保持注册/二进制/源码不变下进行首次运行，单总deadline1200秒，首失败保留，不自动重试；不得停止生产Worker或重置旧消费。本段写入时尚待该GPU driver的实际launch/终态，不把CPU预检或登记视为GPU已执行。

混合采集测试夹具修复采用独立 `target/unified-quant-mixture-integration-r1/r4/source`：保持原513源，仅修 `test_preflight_ffn_mixtures.py` 的合成case plan Source，并增加实际child失败类型/消息断言；测试源SHA `b5278b9c21d6d7e3141195f11453ee465e2e5049758ab338af5ccb70ad291c4a`，完整513源清单SHA `d775c4671850da56237146362dc0d9dcc4ca7f561e49f71dacec31ea7142dfdb`。r3派生helper误用CRLF替换锚而实际源为LF，在测试/构建之前停止，其失败原件SHA `848ff875012070afcec34c8f85cc1e3b48a434ca77a0d14a605ecfa341a52ce8`保留；r4只修封装锚，不改变已确定修复范围。

r4已复制精确fixture并完成修复后的17项实际CPU检查/exit0，receipt SHA `72aaaa32e68bb62b42c2020d242faec55c7ac495ce0f8d62179deac558bed3ae`，integration SHA `6c6e0c801312e10b92be4f21ac46aaf12eac05559fc96939745ccdff58a86381`。原完整preflight流程随后首次在该派生树启动，session33886、wrapper PID65796、preflight PID74936，输出 `r4/source/target/mixture-cpu-third`；继续沿用原14case plan-set，不重新提名/生成计划。该实际会话尚待终态，不启动混合GPU采集。

后续同一r4流程已实际通过11个Python suite共168项、`collect_ffn_calibration` 24项及 `collect_ffn_mixtures` 38项Rust CPU检查，共230项；专门修复验证17项另计。CUDA release collector首次构建已启动，Cargo PID67248，真实preflight Python PID12244（PID74936为venv redirector）；根CIM核PID65796/12244/67248仍live，未重启。构建与14case CPU preflight仍待终态，未进行混合GPU采集。

执行包原注册的首次GPU诊断也已启动：`target/unified-quant-execution-package-probe-gpu-driver-r1`、session42692、probe PID21156，实际仅一次 `--mode run --output registration.output`（工具chunk `2ea996`）。启动前重新核CPU退出凭据、注册、exe与所有绑定Source；仍为1200秒总deadline和原固定预算，不做性能测量。当前尚待实际计数及终态，不能重启或删除其 `gpu-execution-consumed.json`；生产Worker PID78408仍在运行。

两条原会话随后均已终态：r4混合CPU流程session33886实际exit0（chunk `e87842`），inner状态 `PASS_CPU_CHECKS_BUILD_AND_ALL_PREFLIGHTS`、SHA `101086d3c72aaad6482ea5082c68c762c646079da47ce27393f33598c8500f6e`，outer SHA `1cd60a659cf472ba4c4568b91f2b15def26b2993291cdf761e88f9aa43d3628d`。230项完整流程CPU检查及额外17项修复检查通过；CUDA release实际124.029秒/exit0，14/14 CPU preflight，3,573构建来源前后不变。collector exe SHA `0367870585a65902fe86246792779d356957fbd848a206076dae0e1bef34755c`，checks SHA `7d956d304b6e7963e577e49d48da188253d106bfdfab3ff44bd6e79b45955488`，preflights SHA `185576139795ef2ce3d3afdaed2c01125a894bd701eba137b75e4792f87b7366`。累计Python通过执行406次（包含明确重复与失败suite内16次），Rust62次；不将累计执行数当唯一测试数。混合GPU采集尚未执行。

执行包诊断session42692实际wrapper exit0（chunk `ac48fc`），probe PID21156实际exit0/138.252秒，exit SHA `a7515f2c47a767ec9c041909221195307ca3211d775c5985240b025d4518b594`。driver PASS结果SHA `9d43f75d4ad22b68dc31daf8f8a5f872c6a82c7950b5ee03dcfc9731e836e2a4`，所有固定16维预算恰好耗尽：330 raw含367,776有限f32，240次完整头字节比较通过，六case各四份package字节一致；5,557原绑定文件前后不变。probe聚合结果SHA `e1e5c74ae96fae98397976dff7ccf61930b92655e85a9f41d2803547fdc0167f`，永久consume SHA `afe536119c5b000a7074733e8cf57e1d10ba29946ea83f617de2c88ddff55eac`。这是模型API的合成诊断，Graph只覆盖B3、两stream顺序使用；不是正常Backend双槽流水线、Worker、校准精度或性能证明。独立静态审阅要求继续用只读CPU补核每case/batch journal分布、新产物完整前后指纹及实际DLL Source，不重跑GPU。

### 8.38 执行包独立终审与完整混合图首次采集

执行包独立只读CPU补审首轮PASS/实际exit0，session45035（终态chunk `5e95c7`），`target/unified-quant-execution-package-probe-audit-r1/report.json` SHA `06d0f04b96d4eef81587285da74110abbd2a8de14a9c4c846dc491b0d858cdcc`。732条journal事件的逐case/batch/类型/数量/累计值/顺序与独立展开序列完全一致；核六case recipe、完整B1/B2/B3、负例标志及固定输入，330raw/367,776有限f32/240整头逐位比较和六套24份包全部通过。6,107绑定文件前后相同，包含原GPU/driver全部547个新产物，目录集合不变。实际记录的cuBLASLt/cuBLAS模块Source与当前磁盘字节、六份Binding一致；未伪造它们属于原wrapper前置清单的凭据。无GPU重跑、模型解析或旧文件修改，范围仍不延伸至正常Backend/Worker/性能。

混合校准已通过原冻结准备器的真实CPU准入，freeze进程PID75068实际exit0（session19224、chunk `c4aeee`）。唯一逻辑registration位于父full80的 `target/unified-quant-calibration-full-r1/mixture-calibration-v1/registration.json`，SHA `8bb08018e6e260090ff53205a662ae3d2cba792f006e5d2e62d0f27b99398e1e`；`target/unified-quant-mixture-calibration-real-r1`只保存外层命令/退出/准入凭据，不移动逻辑预算根。17个history锚保留旧10collection/16ABBA和三份完整矩阵负结果，父永久consume与两seal保持。预算严格为14上传/28,616个B1 numeric前向与物理行，cost为空、无预热/ABBA，单例1800秒、总25800秒，输出上限4,026,531,840字节；冻结result SHA `c283fca07c48b719ccb11c44d3bc2d16dccf15c71b9a35e7e96af85a4d8dbc5b`。

新GPU采集已首次启动：session75714，wrapper PID80208（2026-09-27 13:29:50 +08:00）、实际driver PID60444（13:29:52），首例collector PID82620；根CIM已核driver/collector live。外置入口 `target/unified-quant-mixture-calibration-real-r1/run_gpu_once.py` SHA `d812c51f069dd12314303a38422b5dc3ef925477080dd818334d05903df4011b`，实际日志/launch/intent位于 `gpu-first/`。它只调用原r4 `run_ffn_mixtures.py` 和上述registration，启动前4,075来源核验，25800秒单总deadline包含prelaunch（启动剩余25797.624秒）；原14计划、来源和二进制不改，失败停止不自动重试，仅管理自己持有child，不能声称后代全清理。生产Worker PID78408保留。

当前14例仅在途，尚无完整混合图精度结论；继续观察同一session/PID与原capture，不重启或删除新永久consume。实际完整终态后才能用此次构建的FP16参考完成12份CPU比较，不能按在途结果追加候选、重采或宣称效率收益。

**后续真实终态已停止，覆盖上述在途观察：** session75714实际exit1（chunk `ee26ab`），driver60444实际exit1/67.951秒；首例B11参考collector82620实际exit0/58.613秒，producer状态 `CAPTURE_COMPLETE_METRICS_PENDING`、2044 numeric/前向、1次上传、cost为空、sources_unchanged=true，result SHA `8fdade245705a10470fab44020e4e74ff0b8478a11dacb01d7853795e25df2ee`。失败发生在外层CPU `descriptor_equal`：期待Python输入索引也有Rust描述的 `manifest_path`，实际索引用 `manifest` Source并附带tensor/PB Source，触发KeyError。原测试只比较了两份相同Rust形状的合成字典，未覆盖真实Python→Rust描述投影契约。

父capture失败结果SHA `65cb83abaca2679163d51bcdb6e03d5552e6d3649ee45b5ae14df8fbb467e73f`，reserved_cases=1、observed_forward_attempts=2044、verified_cases=0、sources_unchanged=true；外层result SHA `4b468f159a00857b9b8000e1e1f36a7b4ed855a20a98fbfaafea27fc6c9cb3a1`，driver exit receipt SHA `1c4e0f5cd3f2839cd609529dd97c9c2c33d3cfc33799e76e46d93d31b9489af4`。三个实验进程80208/60444/82620均已不存在，Worker78408仍live。首例与总消费永久保留，不能重采；后13例未启动。

下一步已限定为新 `target/unified-quant-mixture-descriptor-recovery-r1` CPU读回修复：显式核两种描述的字段/类型/顺序、manifest规范路径与SHA、PB Source冗余证明，增加真实双形状与篡改/行交换等拒绝测试，再只读完整验收原首例32chunks/2044rows。必须记录实际新reader身份，不能冒充原冻结校验器；此工作不修改r4源/registration/consume，不调用GPU。未完成只读恢复和继承原已消费1例的合法续接前，不启动后13例。

### 8.39 混合采集首例只读恢复与剩余预算续接

首例只读恢复首次实际完成：session17257 actual exit0（chunk `36dcf0`），`target/unified-quant-mixture-descriptor-recovery-r1/cpu-first/result.json` SHA `793539e99f5612dc305ccd02607e5f0759d0513b21764ac50aa689fa67dab811`，状态 `PASS_READ_ONLY_CONSUMED_REFERENCE_RECOVERY`。16项新增CPU测试全部通过，原B11参考2044行/32chunks完整读回，零新增GPU/模型上传/collector调用。reader SHA `98b68e2444b7f4e2269f7c48f8aca497d186961a3f44f620b738e6468922532d`，严格映射模块SHA `d9cbf35ca3faff1a088e8cb01b0dae77863d9ef511e2ac2d873459581103cebf`；只在派生reader中显式映射input descriptor及row identity，原其余验收检查保留，并裁掉采集execute/main入口。

映射核Python/Rust各自字段全集、严格整数类型、batch/index/row顺序、manifest/PB/tensor实际Source和冗余SHA，重算tensor与row-feature哈希；未知字段、行置换、Source篡改和bool/float伪整数被拒绝。最终10,338个已绑定文件逐件recheck稳定；source-before的4,086条在after中保持同一身份，后者在读回过程中扩展来源集，不能写成两份清单完全相同。独立只读复审已核实际Rust字段和哈希域，未发现首例恢复阻断，后13例仍无目录/消费/pending。

剩余13例续接正在新独立脚本中准备：保留原registration、永久consume、FAIL result和首例原产物，继续原14例顺序及原case输出路径；新汇总显式记录恢复校验器身份、原已耗1次上传/2044前向与剩余13次/26,572前向。原数值比较器保持，另适配续接汇总准入，不能伪装原driver成功。时间门保守收紧至2026-09-27T12:30:00Z（本地20:30），早于此前观察的首次wrapper起点加原25,800秒；不补造原wrapper没有记录的UTC起点。原wrapper实际70.375865秒、原driver67.951219秒，续接记录69秒作为已耗下界，不将它写成完整实测耗时。当前尚未冻结或启动该续接，不构成完整混合精度或性能通过。

独立恢复复审已封存于 `target/unified-quant-mixture-descriptor-review-r1`，evidence SHA `2a01498a44c893ce69d500bb869e75f2369348785eacba51f0b8f25dd78bcc0b`；仅静态与元数据复审，没有重跑16测试、reader或全量10,338文件。新CPU比较聚合适配器明确接纳continuation v1新schema及全局/新增消费，调用原冻结数值比较器。r1首次9项测试有6通过、1失败、2错误，原因均为把要求存在的 `resolved_path(strict=True)` 用于尚未创建的新输出目录；未执行真实比较/GPU，失败result SHA `6f4c805ebba872d93141c3154be4f7f7639d23881b38f64678937f941272047b`保留。新r2仅改为核现存父目录再拼接新目录名，九项实际全部通过/exit0（chunk `6e482e`），result SHA `26e6f70531fd486507cd0301a6d60c87cedf94801a4e2817d3a6dac439f90938`，测试exit SHA `f362cee1a27b9a68fb0bc3a28abc0f1d69f053ea1560ed8f513732bee6f06a0c`。聚合器SHA `4f9f48e8a6ad2ea1ba91205d81a59cb507bd1b6e538cb516be44196c75c2746d`，路径 `target/unified-quant-mixture-comparison-continuation-r2`；完整真实准入/12比较仍待续接终态，不能将合成边界检查作为实际比较结果。

续接控制首轮20项实际CPU检查PASS/exit0（chunk `495a0b`），result SHA `08dd85b986c85ff4a8e894f77579462f213c5032019213d5af564a686faed4ef`，test receipt SHA `2054f558e6e42368871525041db3f4be9f9f70e4d08bc49fa9c963119bbbde6d`。检查包含在真实campaign父目录下验证尚不存在的owned output/guard路径；静态复审提前修复了严格路径函数不能接新leaf的问题。CPU freeze实际exit0（session28699/chunk `cbefd4`），登记位于 `target/unified-quant-mixture-continuation-r1/registration.json`，SHA `e39844d14e8ef9c90e4e31652cc43cfe4d6e2f31e84f7686bdb2d3c3b8ec47b5`，driver SHA `119c4d7bfc27329cd5541dcf42f9d82d22d9948772c62989abfc362bb1722fd1`，10,357份继承来源清单SHA `609a72d9f88dd07ec61cb15529147c514e0c266f6545af8974ad2b40e0fb14bf`。新登记只是原campaign的显式续接说明，原registration/FAIL/consume不变。

独立静态launch审阅无确定阻断，evidence SHA `604935bd9d89f2b0ce4ef82579874c4c8678e1fa45e1d3c7ff80e2047e52e4fc`；另发现比较r2未交叉核若干重复继承字段，故新r3增加aggregate及guard的原consume/FAIL/recovery/counts/limits/deadline一致性检查。r3共11项实际CPU检查首轮PASS/exit0（chunk `3bb7fa`），result SHA `5991fbafde14d5450ae66d3c204bfbf5e9d419d136023b1295950408519cb576`，exit receipt SHA `a555c90826a7a71a3ac175d3065c76ea0a32d18ebffae3931b5e211dd0f5040c`；原r1/r2记录均保留，没有真实比较。

剩余13例已由根唯一启动：session38519（chunk `19195a`），wrapper PID69984、driver PID4632（2026-09-27 13:55:01 +08:00），首个新collector PID76368（13:55:29）。入口 `target/unified-quant-mixture-continuation-launch-r1/run_once.py` SHA `aded2533dc5d4cc64c395e02d0dc342a16e894229c9903021371ca7d81d53ca3`，日志 `gpu-first/`。原capture路径继续保存各新case；新汇总只能写父campaign的 `continuation-r1/`，新增 `continuation-consumed.json` 永久保留。只能观察同一session，不重启或另发run；运行窗口仅做静态并行工作，不跑测试/构建。首B11参考仍复用原捕获，全球上限14上传/28,616前向，新增13/26,572；当前尚无完整终态/混合精度结论，生产Worker78408保留。

同一续接 session38519 已实际 exit0（tool chunk `be09af`），driver PID4632 的持久退出记录 actual exit0/1227.8193863s；wrapper69984/driver4632 均已不存在，生产 Worker78408 仍live。14/14有序收齐，全球14上传/28616前向，新增13/26572；首B11参考没有重采。`continuation-r1/result.json` SHA为 `0870195ba290fc9b44c64d1e905150a77ad3fd0ace3cb5be0f882851534f5ff6`，launch wrapper result SHA为 `4b90b36ef435da9358ccd82989e24cf8a205878831d6e5a84bebd79cffb3f868`，driver exit SHA为 `b1252c020e076e09063cc2a55817e4dee375f65024a9f3b4439dbcdb9b94c3a0`，真实tool退出观察SHA为 `cca801ea921427c6d62c96652deaa6e47dd28b77b5774c1ffd9bebd4cac6797d`。耗时是整个采集driver wall time，不是推理性能。两次永久消费、原失败、旧registration均保留，禁止重启或重采。独立CPU完整性审计已获上述终态锚的一次执行授权，12份数值比较尚未启动；本段不预写其通过。

### 8.40 正常录制接口与共用配置的静态增量

新增 `target/unified-quant-production-recorder-implementation-r1` 以实际352文件基线派生五份源，delivery SHA `90a578510f241cebeaf11ba58f946313583245d0a84f543a9437268feaf5c0cd`。typed Recording session 复用正常后端的加载/context、workspace、双槽warm/capture/提交/完成路径；Observer在真实动作前记attempt，并在真实完成后接收完整物理五头和原decoder的逻辑输出。Record无最终服务profile，Restore继续按实际package SHA发布身份；全物理batch/两槽完成且无pending才能seal/export。14项纯CPU测试仅定义，尚未执行；没有新的模型或GPU预算。

独立静态窄审发现第二次SlotCreate observer panic会使已完成slot/workspace早于同步析构，报告 `target/unified-quant-production-recorder-review-r1` evidence SHA `3014332ad1458f280f5135982dbdc8352840d917c9c4d2f65bae5c446734f47f`。保留r1，r2在资源拥有者声明后增加跨整个batch构建/移交的RAII guard；delivery SHA `a6376fb6ae1608060325d460861c221f00e4a94b6e30f0ace8680552d86c4ae5`，composition SHA `652799b16b3c9005d6ed934bcf60f592bbc3271cd233c294ef42f5a990edb0ed`，cuda.rs SHA `5fe4b47277ff56b51cc8c7396ffda9cf4e5284b059ba91721f6ad3ceb5aed901`。r2尚未构建/运行；panic仍要求整次进程失败退出，不能catch后在同进程恢复。

共用setup/eval resolver增量最终为 `target/unified-quant-execution-config-static-r1/r2`，delivery SHA `9f955397c7c58d89215cde486c99301a5f7e939d04e6c0e688307221ea18dd33`。实际setup和eval改为调用同一个已解析参数模块，保留模型/线程/GPU/FP16优先级、普通batch范围及旧profile字符串存在性；Worker capacity仍仅并发提示，没有偷换实际batch。独立静态复审22来源稳定，report SHA `b5d2d88cdd3fa4a00ebfdc94b2c8c4d563da5a9d56c6bd7ae110413a8a857333`。集中解析前移会改变坏配置的首错、seed日志和used-key集合，明确不保证全错误路径等价。14项新CPU检查仅定义；最终新隔离composition必须精确合并两个lib.rs模块导出，不能覆盖丢失一方。正常CLI preflight/record/verify、持久预算observer、同exe新进程基线/恢复、生产Worker缓存隔离和性能验收仍待完成。

recording r2独立窄复审无新增确定阻断，review SHA `bfb6ac8d3e603ee48bf056868f631251e97898141481cccf5508845f44367579`，evidence SHA `10079fd39fa32e60ac4e0480863b980a4dfa48d84ab6b3b1a1bcbbc5091c1237`。新的隔离集成编排位于 `target/unified-quant-production-recording-integration-r1`，delivery SHA `7e84c30470f5e7477da2de7078791b3dec03af4f5a4d2e6269fa106af97f0f32`，runner SHA `85260428259c5ca76cbd6b4f1204b96cddcccbc89ddd2de3e3b9fd328cfb955a`，完整frozen-plan SHA `aa15e9a509f3278fb1ce22e18ea4c10d43bd9a605fbf0f4fc0a98b90b6395ec6`。独立静态审阅34来源前后稳定，review SHA `4a83ddedc05c571277ff9c349761754ff651350cd112587e84b4082c61aff29e`。

同一runner已首次启动session21051（初始chunk `60e0b8`），精确352→356隔离源已落地，保留三份AOT `.bin`；两个lib导出合并后641字节/SHA `58b0d60aab5564a3c9408bdff8ee1bec7b563354c144b618e4c538104e84bc7d`。固定四阶段：非CUDA真实库测试链接及28CPU、实际kata_program check、CUDA真实库测试链接及78CPU、CUDA库构建与嵌入来源核验。106是测试执行次数，28项新测试分别在两种feature下执行；没有重复原143全套。已观察五项工具链命令exit0，实际构建/测试终态待核，勿重启。只管理直接自有子进程，中断后Cargo后代状态UNKNOWN；不会运行模型、Graph、Worker或部署CLI，不据成功编译预写正常运行通过。

该同一集成session21051随后完整actual exit0（chunk `840915`），result SHA `7860e4efe0b412b0166be5952f5954208456e26426d4651030c6a66eec183cba`。106/106纯CPU检查全部PASS；四个Cargo阶段均actual exit0，耗时分别71.1829841/7.2900607/148.8926788/27.7824727s。实际356源清单前后SHA均为 `971ab4879136b7be623277e0cef6a1be22239f6b5b71273d33d286656e7cc24a`，4,531绑定文件稳定、CUDA嵌入provenance通过。venv redirector36004/实际Python45832均已退出；实际tool观察SHA `370b6aabf820a2e4151c74cdf229a993a0f6eead2043e2ef41ac4c29e7f6eb2d`。没有模型/GPU/正常双槽运行、完整CLI或生产部署；无需重跑该106项。

### 8.41 完整混合图校准比较终态

完整性独立审计首次session78638 actual exit0（chunk `575e65`），报告SHA `bc365b1988aedfbcfcb99ce5e3dfa76c46dbedf05f0fcf8a2f046429ac98ae01`。独立核14上传/28616前向、448chunks、73,085,264个raw f32全finite、143080个五头片段及28616个PB片段；22,363个绑定Source前后SHA一致，1527产物文件/14目录全集稳定。复用哪些准入/解码函数已在报告披露；这是完整性与消费核验，没有在审计中重复数值比较或GPU。

随后唯一启动12份数值比较，session56071 actual exit0（chunk `6032a2`），比较driver PID50268实际退出0，12/12对每对2044个B1局面均为6pp PASS；没有增加GPU调用。聚合 `target/unified-quant-mixture-comparisons-real-r1/index.json` SHA `e99c674dd228e86135d92619dcfce3d1904900876f5939c3cba85bc885477ca4`，wrapper result SHA `b0ad305979020faa2b307725f5e6e69049c28b5914f9dcdfd6c20b6cc2affd7b`，driver-exit SHA `90e4c0bfb6613c7f1cf67243bc111f953dd3df777d52bb73457fc338af286f0a`。旧CPU验收失败不覆盖、首参考不重采；全部新配方仍未采用。

额外只读统计脚本 `target/unified-quant-mixture-statistics-r1/summarize.py` SHA `57ab839261d0289fa036ab886f0fe4f14bf7e6e39fb4b4c067ace15d06bb5eca` 首次执行session4447 actual exit0（chunk `bcd04a`），报告SHA `27122475fa57b913b59010187e30958f587b6ecc7c0992cd45370c27e5a4fe24`。它绑定完整比较index和独立审计，只重读已绑定chunk/PB及原report，复用原PB/typed-bits解码，逐项精确重现原mean/max/KL/top1后补P95；未重复原始五头比较、模型或GPU。P95采用nearest rank，即排序后第ceil(0.95×N)项；胜率与目差每语义局面一权，ownership每局面每交点一权。所读来源末尾再验SHA一致，不将此局部重读冒称第二次全量独立审计。

下表batch是**提出该配方时使用的算子成本batch**，实际本次数值验证统一是B1；表内误差都相对同模型的本次FP16参考。FFN数表示dual和down同时INT8的完整组，其余组保留FP16。

| 候选 | 成本batch | INT8 FFN组 | 胜率平均/ P95 /最大偏差（百分点） | 目差最大绝对偏差 | policy首选一致（/2044） |
|---|---:|---:|---:|---:|---:|
| b11-mix-001 | 3 | 3 | 0.02281 / 0.11181 / 0.64304 | 0.15763 | 2040 |
| b11-mix-002 | 3 | 6 | 0.06161 / 0.29648 / 1.10676 | 0.30094 | 2033 |
| b11-mix-003 | 3 | 11 | 0.08647 / 0.42071 / 2.40688 | 0.29869 | 2027 |
| b11-mix-004 | 8 | 5 | 0.02866 / 0.13731 / 0.66327 | 0.14855 | 2042 |
| b11-mix-005 | 8 | 9 | 0.04480 / 0.21251 / 0.79633 | 0.31108 | 2040 |
| b11-mix-006 | 8 | 15 | 0.07142 / 0.32898 / 1.96654 | 0.33542 | 2036 |
| b15-mix-001 | 3 | 3 | 0.03009 / 0.14816 / 0.73912 | 0.13624 | 2033 |
| b15-mix-002 | 3 | 6 | 0.04647 / 0.21650 / 0.98893 | 0.24742 | 2021 |
| b15-mix-003 | 3 | 10 | 0.06471 / 0.31019 / 1.53026 | 0.70178 | 2019 |
| b15-mix-004 | 8 | 4 | 0.03640 / 0.18766 / 0.80270 | 0.16572 | 2035 |
| b15-mix-005 | 8 | 7 | 0.04303 / 0.21431 / 0.88662 | 0.33695 | 2030 |
| b15-mix-006 | 8 | 13 | 0.06782 / 0.33982 / 1.45017 | 0.68786 | 2021 |

统计补充又经有限独立复审：首r1审查器遗漏Windows extended DOS前缀而在首chunk读取前误拒绝，失败原件保留；r2只修审查器路径处理，session19330 actual exit0（chunk `346c11`），result SHA `6eb159593a0d20efd6954c5870b47e0b7e662db5f37103d00f229f8302536755`，review SHA `59b899db309b5c71a0c83fe1900c5e4d0bab81bac46c570f402ec0218698def8`。12份原report字段精确一致；仅对B11 mix003/B15 mix006独立重算各2044个win/score/KL分布与top1，全部精确复现，P95为第1942排序项。292实际读取来源稳定，0 raw-head/原比较器/GPU；PB decoder共享，ownership RMSE和policy max来自原报告、ownership分布未独立重算，不外推为第二次完整数值审计。

本次最大误差明显低于原单层风险代理相加值，但这只说明这些实际候选在这份calibration/B1样本中的结果，不能证明误差可相加、外推B3/B8、真实搜索叶或未来局面。12份旧阈值诊断仍均FAIL，不是原FP32认证；policy/ownership等继续作为报告项，未新增或放宽用户的实验门。完整混合图成本、最终算法包下的全部发布batch、selection/独立确认/holdout、Worker和搜索收益仍待后续有限登记与实际验证。B1原成本门仍无候选；不能根据这次精度通过覆盖历史C1负结果。

### 8.42 正常 CLI 预检与持久计数日志的隔离增量

基于§8.40实际356文件快照，正常CLI静态交付位于 `target/unified-quant-production-cli-static-r1`，delivery SHA `6604807b9b508530c635c3fbf5b102c53de12cf0d8ab5a0a4d464ab375af488f`。新增 `execution-package preflight`，共用原CUDA/quant-inspect模型解析、setup/eval配置及Worker输入编码；原生B11/B15与ONNX都绑定原始模型字节、实际exe和嵌入构建来源。预检只解析模型/配方与校验CPU输入，不创建CUDA运行时。record/verify目前在读取来源或创建输出前明确拒绝。独立静态窄审39来源稳定、无确定阻断，review SHA `b1c656a94d645d6c64e82272134aaed3c67adf639e6b61df616f9252f9fa7093`，evidence SHA `22476902c40463387e7a11ce983f111c0ed2a6117f15bf4f185ad8f8acabfc7a`；20项新CPU测试此时仅定义。

持久attempt日志静态交付 `target/unified-quant-execution-journal-static-r1`，delivery SHA `4b369d9a945438eae82d9182053a188d911542b42fe027aa10c68bb2429919db`。它在创建日志前独占创建并同步永久consume；16类typed attempt逐次检查额度、写序号/前项hash后flush+sync，返回成功才允许实际动作。失败保持停止状态，不删除消费、不续跑；15类exact与scratch_allocate上界分开。末尾只证明attempt accounting，不证明动作成功、输出完整或性能。16项CPU测试仅定义，尚未接Observer、Completion持久化或CLI record/verify。

两份增量将在新隔离目录合成为363文件，明确合并model_cpu/execution_journal导出，保留三份AOT参数.bin。有限CPU方案在 `target/unified-quant-production-cli-integration-plan-r1`：五个Cargo阶段，90个选定CPU测试执行（两种feature各45项），随后同一新exe对B11内置FP16、B15首FFN INT8、ONNX显式FP16进行真实CPU预检；不重复原106项。三例声明完整B1/B2/B3两槽序列，但预检实际GPU/上传/前向/Graph均为0；ONNX三条PB/特征将由现有CPU编码入口产生新来源。此段记录准备范围，尚未启动或预写通过。

后续选择静态交接在 `target/unified-quant-mixture-selection-handoff-r1`，C32补充SHA `c0f964aa7b3f5fa49817ba592d451005590f6c36529ed56f3aafa834536418c0`。建议把每模型六候选、两种容量FP16参考及旧控制组纳入同一C32负载；现catalog需支持候选对应不同参考，20次collection/44组ABBA只是静态范围，尚未冻结执行登记，不能重置旧10/16消耗或覆盖C1负结果。实际physical batch覆盖与最终执行包数值门仍需建立。

Worker隔离只读交接 `target/unified-quant-worker-profile-isolation-static-r1/handoff.md` SHA `feb4d7e42f8f94eccdd1687302b87535c6f1ca3bc00425c1b112a0b9d9fbafdd` 确认：现服务端按模型SHA接纳Worker，backend_info里的profile只是文本，新端口/Worker名本身不保证拒绝不同profile。静态建议临时单profile池在注册与ack核对实际profile，并使用新搜索/缓存；尚无协议修改、服务启动或部署，不触及生产Worker。

该有限CPU集成已由root首次启动session34943（chunk `daba15`），venv redirector PID78656/实际Python PID40764。静态delivery SHA `10f2ae67aa775182bd54857604e8db7cb9cd543196acc4200826e82b82c5877a`，runner SHA `8901cc2e4c28ce32b9e5d82c98cd5696f465a3baa48da6b20ac414dcef8e8dd6`，frozen-plan SHA `dec3b920dd68449d34b05c54c55a142fc4cbc93246eaaf88cf61583ca728833d`。363文件源图已物化到该目录run/snapshot，合并lib687字节/SHA `311a011db85a9b9c7182651918f97a770b8539e3b1046e1c9af870c5292bdb8b`。四工具链命令actual exit0，首Cargo PID22872在途，最终检查和三模型CPU预检尚待实际结果；只观察同一session，不重启入口。

同一session34943随后actual exit0（chunk `216dbc`），result SHA `e683b971b4b9bd71bd6270d3547477a1f13ecde1ef888c755292a4ac0440cd7a`；90/90选定CPU检查、五Cargo阶段和三模型CPU preflight均actual exit0。实际363文件清单前后SHA相同：`2c7cbac9c052a8f93be2d4d5d43533b07966f305644b9e5fb1880261ca50d7d8`，4,637绑定来源稳定。CUDA CLI为17,192,448字节，SHA `737a13eba4d20ecbc6edf8b6262554f572ebab9a7259b8ecc4fd5852cd7ce009`，构建provenance SHA `e2c06221f95f8a54211a03d0c168de0cffd0c27cc979d7ba656aea03a7ecba17`。三份预检结果SHA依次为B11 `5afee55060ae1996683cf01b47cdcc398c914a37fc4c819001365c0d1f060fe2`、B15 `dd57a42ce3b310bca5652642a5b14f78c8a8c7b7b0822e86128de1e89e33769e`、ONNX `f781fdbbdae9d35ad5d0115651fb53fc21a10cc8e2570d21828225d794e562bf`。ONNX两次新输入CPU准备也actual exit0，仅三条独立PB/特征；真实tool退出观察SHA `264ea3c2ae8f8090f2fd00a8ca0ee373efb479789d4451991f6835f9e8be0c80`。无需重跑这90项或五次构建。

这些预检没有创建CUDA运行时，GPU/上传/前向/Graph实际计数均0，不能作为正常Backend双槽实机或性能证据。下一增量静态设计已确认两处实质接口缺口：期望设备/driver/Binding必须在ModelUpload前核对；normal restore启动观察必须接到共享实际构造点，不能在observer attach之后用外层估算补写scratch等消费。跨进程record/verify还需同一campaign根seal下的固定单次stage claim与预冻结总预算；现AttemptJournal仅独立单次consume，不具备该跨进程流程。尚无新GPU登记或运行。

### 8.43 正常输出持久化与跨进程录制的后续静态增量

以§8.42已验证363文件为基线，`target/unified-quant-execution-observer-static-r1` 已封存3新增源与一个lib导出，计划共366文件；delivery SHA `5df5bb9547f566172108adfd3f81f7c23ebae18622692e029c5f99c54dd12b22`，composition SHA `5346c07a89ca653e6086997a79c8652cfc451398c6a2461a87b114c6003d22e4`。20项CPU测试仅定义，尚未应用/编译/测试/GPU。

Observer复用已验证AttemptJournal，在真实操作前持久预留；用注册logical handle/tag/有序行关联进程内opaque token，支持两pending逆序finish。每个Completion保存五份完整physical raw u32LE、一份全部NNOutput字段的typed-bits及manifest，检查shape/finite；回调保存仅为provisional，必须等真实finish返回成功后confirm_finished。终态校验两日志完整chain/sequence/head与内存及AccountingReceipt一致、consume原字节不变、七文件SHA和目录全集一致；错误永久停止并保留部分原件。回执只到normal Backend共享decoder之后，尚未到evaluator后处理或Worker PB，也不证明package export成功、数值达标或速度收益。

同exe跨进程record/verify静态方案在 `target/unified-quant-production-record-verify-static-r1/handoff.md`，SHA `5d8c3fbf3341741c9eb8658b316cc078484f93916d62db93dd0918b4246be313`，source receipt SHA `97182ceb4becefe849e94d9310b6daeef5842ceed7d913c8fc482315118d6a90`。它复用CPU Prepared、正常context/handle与现执行包工厂，要求唯一campaign根seal、固定record/verify单次stage claim、预冻结总上限；verify必须由record数据终态加外层actual exit0和实际package Source派生，不能由部分文件或进程内第二handle冒充新进程恢复。

Observer独立静态窄审无范围内确定阻断，review SHA `ebfd5a79ea0fb210f441e030800986b3133ec21deb51d79a7b320c80f44c16e4`，evidence SHA `16eacf84bee95fbd7e764aa5a074672d5e70cca3643df6835f8995bcded14ad1`；18有限来源稳定，重建363→366来源图一致，不代替实际编译/CPU/GPU。

上传前准入与normal restore精确startup增量已封存：`target/unified-quant-startup-observation-static-r1` delivery SHA `b420dbc2b69ff5c5aa335ac20d80281f615de41925bbbf539a97fb1283b3fe2e`，composition SHA `3eb785d722f67184f951a09292888868723039f2ff17b7bdb4a5e004775bad10`，3替换/2新增，12项CPU测试仅定义。新 `start_execution_observation` 明确区分Record/ObserveRestored，actual Binding准入在ModelUpload前，精确HandleCreate/条件ScratchAllocate回调沿共享构造点；LoadedModel一旦尝试offline context即禁止失败后再次复用。普通None路径与既有同步/Graph RAII保持原实现；旧兼容start_execution_recording仍没有expected-runtime gate，新campaign必须用新API。尚未构建或实机验证。

CPU Prepared抽取已封存：`target/unified-quant-execution-prepared-static-r1` delivery SHA `9e0702d8b9a09683be440006016619424453bbabb4779f0703fdca4380e03361`，composition SHA `9857ab902f286c00efff79d484f3d69b1d4c0319331c4fb181b34ad80134f714`。仅替换CLI两源，保留原7 command/10 protocol测试，另定义3项CPU检查；私有Prepared只能由原完整校验构造，保留typed配置/输入/序列但不持有大图权重。Preflight依然先拒绝已存在输出，纯prepare本身不要求新输出，供未来record/verify复用既有receipt。root已读完整diff无确定阻断，未运行检查。

三增量合成预计368文件，正在准备五Cargo/84CPU执行（两feature各库32+CLI10）及三模型真实CPU preflight的有限集成，ONNX复用§8.42已有三条输入而不重生成。当前尚无冻结集成执行、StagePermit或可执行record/verify。后续编入功能会改变exe/provenance，必须用最终同一新exe产生新的CPU preflight，不把已通过的363版receipt重标为新构建；旧PB/特征来源可继续绑定复用。不因静态方案追加GPU实验或重跑已通过90/106项。

startup独立窄审已封存，review SHA `b0ef65f0a6b466e38bff8efd5aa3a934e18a929c7e3fc4f8f45907431a347caf`、evidence SHA `c20bd98baaa61f122faa4868fde32a6f6f44cde317912891875958508d21e213`。18来源前后相同，范围内无确定阻断；12测试此时仍只是定义。

三增量的有限集成已冻结并由root唯一启动session29352（初始chunk `656d6e`），目录 `target/unified-quant-observer-startup-prepared-integration-r1`。delivery SHA `12df39ed5364018ef728197606da27423e4ec0dad05b54272124603e93d279ba`，runner SHA `2f86bb6b77a3717ce4238d7c3fa1cb1c0d544cbc793269db347e1d37e3fa0bd8`，plan SHA `95eda601d759b1a78ed29bf5d0ace6ab7ee10f58e0d4d0367d1961002a5af9e9`。固定368源、五Cargo、84CPU及三个真实CPU preflight；ONNX复用原prefix3，不生成/重编码新输入子进程。首非CUDA库链接和32项CPU检查已actual0，其余待同一session终态；不得重启或修改冻结源。

StagePermit草图 `target/unified-quant-stage-permit-static-r1/handoff.md` SHA `a6b7f47e469dd4ea9a6f0c3d8cd304322c1e0e0cc5b3c4feb5249cf1121d630a` 只规定双阶段预算/Source与永久claim边界，不替代CLI私有Prepared。record数据终态由实际export/输出确认生产，进程exit必须来自supervisor真实Child等待；普通JSON/SHA不单独证明进程或GPU执行。静态代码增量正在实现，尚未测试/接通。为避免身份循环，固定argv不含campaign自身或未知verify产物SHA，实际Source摘要通过一次有界启动gate传入；verify使用独立派生envelope，不改写原只接受record语义的CPU注册。

同一session29352完整actual exit0（chunk `3eaf3d`），result SHA `0cecaeb0f74a39a4eb4d32a42946657637aa7f74e06031b556c3c9809d5bb3d8`。84/84有限CPU检查、五Cargo阶段及B11/B15/ONNX三个真实CPU preflight均actual exit0；368编译源前后清单SHA相同 `ddb4bf26f7ca7aa31b21cdd20a352fd840aa58ecae2e82037100442b68035770`。最终4,651绑定Source稳定（初始4,563，新产物按产生时间加入），error/source_recheck_error均null。新CLI为17,207,296字节，SHA `732e99c12dc94adb2d4fda7d1f1d0ec33955aea0d9368bd14a407c489d477c21`，嵌入provenance SHA `3520a32ee46f7b372f02df5a9813517079a994070a0044a46568c450d1e2bf91`。三个预检结果SHA依次为 `2af2ee74b89c75d4a7e3c73efecae2f8f3fd05f4c7ca14ef73259f1bb64514b6`、`e8c681c340511b68b13dce0dfa2498985499ed942d41b11b8be6d7737223e71c`、`c8a310f546a9dc20bb5a163d4d7d945469575b4af2f7b24d38885348d26f5244`。实际tool退出观察SHA `805dc7ac612b5e8cda0f9aedac5397f9dcb411295c44b0cff5f411458e4c556c`，不从进程消失推断。后续基线为该run/snapshot；无需重跑84/原90/106。0GPU、没有record/verify CLI运行、性能或采用证据。

限定性能路径复核发现：普通统一后端仍可为合格FP16 dual投影启用现有CUTLASS DualGemm+SwiGLU；目前v1录制入口因未登记真实启动能力probe而拒绝DUALFFN=1。该开关进入execution_policy身份、目录记录实际route，故不能关闭录包后开启部署。B11未量化的合格组受此录制限制，B15不可把任意剪枝宽度视作同一快路。正式选择矩阵前须补真实每进程一次probe的有限记账，继续保留最快旧FP16控制；不得以受限统一FP16参考替代它。此复核仅静态，不新增GPU或修改现有16类计数协议。

### 8.44 固定阶段许可与正常录制驱动的静态接线

在§8.43实际368基线上，阶段许可与私有正常record驱动分别在 `target/unified-quant-stage-permit-source-r1`、`target/unified-quant-production-record-driver-static-r1` 准备；截至本段记录尚未最终封存/编译/测试，CLI仍拒绝record/verify。许可核固定campaign/两角色预算/Source与实际launch，Observer和Journal按值消费同一许可；原CLI Prepared仍唯一负责模型、输入、配置与sequence语义。私有驱动复用正常Backend submit/finish/export，不另建前向实现。后续应把完整接线合并后做有限编译，不能把静态源描述成可运行功能。

独立私有stdin gate已source-only封存：`target/unified-quant-execution-gate-static-r1`，delivery SHA `27206c711e72d20f28b02f309e32c35d55441687e6e0dead76b266eca3e17a8c`，composition SHA `e4a2e1ae2b768825292b87a601a903b554c7cce2bc1fa281dc781e66ede07c83`。两新增源、8项纯内存/channel测试仅定义，未挂接父模块、编译或执行。限定静态窄审无确定阻断。帧最多16KiB、单行LF并等EOF，一次消费，接收等待30秒；超时后读取线程可能仍占stdin，CLI必须错误退出，不能继续后端。gate仅核动态Source，claim后还须将permit的campaign/launch Source与gate再次精确匹配；进程PID/argv/producer及实际GPU准入属于后续StagePermit和正常启动工厂。本次freezer实际exit0仅代表来源散列和文件封存。

DualFFN录制缺口的7源静态交接已封存于 `target/unified-quant-dualffn-recording-gap-static-r1`，handoff SHA `dedc77db8f012a33394524c6fc944c0f7340a77d6dec3b915a38f54c1a20708e`，source receipt SHA `2a02c5d4484c0dc5532d22cf860d257e931f4a394f781881049949f548980667`。Record原顺序已是probe后建目录；Restore首次probe在包验证之后，但UNKNOWN状态也被视为effective，失败会拒绝发布context，未确认现存误路由。建议离线恢复将同一有界probe放到实际Binding/admission之后、首次目录之前，并在OnceLock实际initializer前记账；后续ensure不重复计。尚未实现或修改16类schema，不以静态分析预告性能收益。

StagePermit record-only源码随后已封存：delivery SHA `0ca19626a727dcab454301a4a5d305733dcdfe5eb2d71f4ee955e76bced0e695`，composition SHA `e31e4e7ae39050984a54331c2bf1abe583b2aa904c51067853b724e21af2f16c`，source receipt SHA `0ed75c1bf19a00b44f87b8bf83234fe3aec28c4d50a760af67afc46ce7ab3d9d`。3替换/2新源，18测试定义，0编译/测试/GPU。current exe/provenance/PID/argv来自实际进程读取；environment Source为绑定的受控supervisor声明，完整OS环境未在此模块独立重建，Prepared仍核tactics/配置，runtime Binding核设备/实际库。verify claim立即拒绝，没有public bool/closure逃生接口。

私有正常record驱动已封存：delivery SHA `bf0c922d30e89cca8b5150adae6e448eb3d999a5eba6cb2627c01034d9d0d0ac`，composition SHA `de455a5bf3bf8e42b0f524ec4092b987bca72ae58c851e1f2979585b14b480f5`，含permit的计划372源map SHA `d0a44889a669357f8ab641a4df222f48852b86be01face05e6d17091a1fa949a`。1父模块替换/2新源，16测试定义，尚未应用/编译/执行；root已读完整driver及与实际包canonical序列化的关系，无确定阻断，不代替编译验证。它消费Prepared和permit，经真实normal session/Backend执行固定序列，逐handle实际导出后才保存相同包与数据终态；Err/panic返回失败并尝试abort，未来CLI必须直接失败退出而不能catch后续跑。仅写normal Backend decoded输出，不额外做evaluator/Worker后处理。父CLI的record/verify硬拒绝仍保持。

加gate两源后预计374文件；还需一次合并父CLI模块声明、公开入口/真实supervisor、record终态与实际exit读取、受控verify恢复和逐位比较。当前没有该374合成快照、检查runner或新GPU注册。实际已通过的构建仍是§8.43的368版本，不能把静态增量写成它已实现的能力。

StagePermit独立限定静态复审随后封存于 `target/unified-quant-stage-permit-review-r1`：review SHA `3ebe6ebe3aa765f78714b5052b2ee3180145fa0e8f632ccb60e252a27bfe1cd2`，source receipt SHA `4592359727c69deb410a0af6060875f626e35c6f0afbc6b1ace1346471ef81e1`。19限定来源前后稳定，范围内无确定阻断；journal结账3356字节和Observer初始化主体/生命周期21764字节与actual368完全相同。复审没有执行18测试/构建/GPU，也不证明未接通的CLI或OS完整环境。

### 8.45 同一二进制正常 record/verify 的完整接线与有限 CPU 集成

StagePermit r2 与公开 CLI r2 已直接基于 actual368 封存，分别位于 `target/unified-quant-stage-permit-source-r2` 与 `target/unified-quant-production-record-verify-cli-r2`，delivery SHA 为 `8c1603dd1b683eca94c8d60e0a21e251b7b1ed858211228be1e3e83d9c8c06bf`、`f613ea8cd934cba6550710dd98ba0f31a40b0350b633c92d2761792bc4d51e58`。两份自包含增量共14个不重叠目标（7替换、7新增），合成375源，保留三个AOT参数二进制；不先应用旧r1再覆盖。公开seal/derive-verify为CPU入口，record/verify共享正常Backend执行循环；verify只增加包路径/SHA两个受控配置，使用实际正常恢复工厂的profile，并核重新导出包及五头/decoded typed bits逐位相同。

库内严格读取固定record终态、Observer完整原件和受控supervisor的实际退出证据，一次派生verify登记；子进程再次读取并消费不可克隆的许可。campaign v2将四个固定argv、单总deadline及独占attempt目录纳入producer。阶段stdin gate提供未知的Source SHA，避免argv含自身campaign SHA的循环依赖。profile仍是执行身份，不是精度/性能认证。实际模型路由/叶表验证继续由正常导出与恢复路径负责，CPU reader不从包本身构造“实际模型”证据。

独立Python supervisor封存于 `target/unified-quant-execution-supervisor-static-r1`，delivery SHA `30fb7832b273e2b0ebd33f399764091120435568af3b9daf458928ccadbcfbab`。仅拥有自己创建的Popen，固定四阶段、单次gate+EOF、并发有界日志、单总时限和失败停止。root窄审发现的迟到日志线程错误竞态已在join后复核并补测试。其21项mock和两项直接base Python子进程检查由下述唯一runner执行，不能据此声称GPU后代清理或实际Rust campaign通过。

有限集成目录 `target/unified-quant-record-verify-integration-r1` 已静态冻结：delivery SHA `70de15682b1fecb218dd9c33c3903e6906224b340028890cd48f4a969a3d45b9`、runner SHA `28523d0d66867dcdc56027092bb010ec6d04316d5444eb0f584333004a099abd`、plan SHA `21dd3dbf37f089d0688ffec3be0ba63e963891130d49f27eef8ffd2417f18cfa`。计划为226项Rust CPU（每feature：permit30、受影响Observer20/Journal16、CLI13/gate12/driver22）、21项mock、2项真实CPU子进程、五Cargo与三模型CPU preflight；ONNX复用原prefix3，不重新编码，不含GPU入口。来源路径只在bookkeeping规范化，冻结metadata原字节保留。root静态核读后唯一启动session75122（初始chunk3b20fb）；两项Python检查进程均actual exit0，Rust/Cargo终态待观察，禁止重启或修改在途源码。独立runner窄审对同一SHA未发现确定阻断，但收口在启动之后，不记成启动前已完成审查。

同一session75122随后实际exit1（terminal chunk4d0356），result SHA `d42fae8d9ae51d6bd0e0e3673a9959275980de6ed5deaae6b87c4d83033183f2`。21项mock和两项真实CPU child成功；首个non-CUDA kata_nn库编译exit0，permit组实际29 PASS/1 FAIL，失败为 `parent_components_are_rejected_before_creation`，尚未运行余下Rust组、CUDA构建或三模型preflight。375来源前后稳定，原报告和日志永久保留。初步定位为Windows verbatim路径的join提前规范化`..`，需在新派生测试夹具中确认原始ParentDir确实存在；不得修改在途/已冻结源或放宽运行时拒绝规则，也不把29项部分通过登记成整组通过。0 GPU。

路径夹具根因已由本机Rust源码确认。单测试修复在 `target/unified-quant-stage-permit-path-fixture-r3` 封存，delivery SHA `1da3a0870b3df2826d69d4c0517ae725a52fc13d9f3cddc2eaa8cdefec2525dc`：仅用OsString保留base内的child/../escaping-output，先断言ParentDir存在，再核原拒绝及无目录创建；其余测试与运行时代码未改。修正集成 `target/unified-quant-record-verify-integration-r2` 在新375快照/新build上继承r1已通过的21+2证据、不重跑Python检查，受影响permit30重新验证，其余未完成Rust/Cargo/三模型CPU preflight继续。root读完三源后首次freeze actual exit0，delivery SHA `ca2b613b9d8c2f63759b2bceb826a788e2c9be516cce4c29ab0b0ff984985356`，runner SHA `33b6030829e67bd7ee95f7626ce37b2092abf6b871cbba08101ba43f6591012f`，plan SHA `a5ed741af13fceca90bc44c34d0b36c6e58c0aec616f3e1bf894b81b7f7402c8`。唯一启动session98926（初始chunk811e15），继承证据检查已完成，Cargo终态待观察，禁止重启/改冻结源；仍0GPU。

后续三例正常record/restore的只读交接在 `target/unified-quant-normal-campaign-preparation-r1/README.md`；没有新GPU登记。可用旧独立实机设备/库原件作预期，由正常runtime admission在上传前核当前身份，不需额外GPU fingerprint进程；exe/provenance/kernel_build必须来自新最终产物，不能复制旧摘要。拟定六stage共54次实际前向、108物理行，但storage/time等尚未冻结，不构成执行授权登记或性能证据。

同一修正集成r2现已完整终态：session98926 actual exit0（terminal chunk96fa1f），result SHA `84256a913c2b80e3413b072249970416f2665af1284b86dba106d3c23f5cd67b`。226项Rust CPU检查、五次Cargo和三模型实际CPU preflight全部通过；此前21项监督器mock与2个直接CPU子进程凭据继承，未重跑。375份编译源码前后相同（清单SHA `cd08be23ed03f7fbb9d59490d41148d64918ee7038544a632ed2489b7a8c9c25`），4,691份绑定来源稳定，error/recheck_error均null。实际最终exe为 `target/unified-quant-record-verify-integration-r2/run/build/release/katago-rs.exe`，19,060,736字节，SHA `a9fa63ae8e864d6a4793fa5fea220405538befebd306d9f7a3c553cce9fb4a9e`；embedded provenance SHA `074b4e6fe428d192c5aa16a9e6627eaf05197fe472c02761b70c9539192ffb80`。三模型预检结果SHA分别为B11 `bb39801cd3f7ccd4bef12355cf224bfc47ae68b1c487beaae210af6454ab8dba`、B15 `2e4cbfc88cab28be0dfc8678c1e3f6a9b3c85aedc0c643115df63e2be478cd6e`、ONNX `c4617e270c405108c50747044b3d3eef051e0d1e316457f2987d1b0aa3dfda0a`；均record_verify_implemented=true、gpu_calls=0、authorizes_gpu=false。真实外层退出观察已另存 `run/tool-exit-observation.json`，SHA `d0a0b8fcb0344a7a61d51d54fab422c57c808962ed17937cef5ba35cfc750529`。失败r1、单测试修复和全部消费原件保留；正式binary未替换，原Worker PID78408仍运行。此结果只是CPU/native接线与兼容性预检，不是正常后端GPU复现、精度或性能认证。

### 8.46 正常后端三例跨进程保存／恢复的有限登记

专用CPU assembler在 `target/unified-quant-normal-campaign-assembler-static-r1` 静态封存，delivery SHA `4c43ecde8a1b129a2150eea56d64ba76ef5aaf288916d052c8062f47d6f50ca9`，entry SHA `b7fb1005d3a299d07a1c2874bef4e752ed1a4826ca6a7bd8af88a6e537b10d49`。独立只读复核确认BackendBuildFingerprint／四AOT／provenance与StagePlan／sequence序列化、Prepared输入闭集及producer四阶段契约未见阻断；没有新增launcher、ledger或测试框架。实际回调预算另经窄审，报告 `target/unified-quant-normal-budget-review-r1/handoff.md`，SHA `15a2a0aa6c1643c28d4a458512ccd7c6867beb4f94e3531c42fee913fb817a2d`。

root首次CPU准备actual exit0（chunk92a9ed），产物 `target/unified-quant-normal-backend-record-restore-r1/preparation-result.json` SHA `8fc4e5104ed367f7c79c0ef78529e1d8a2f0d245579d863367b38345f9be2b15`。固定B11 FP16、B15首FFN INT8、ONNX FP16三例；campaign SHA依次为 `d11da8f04d6ac1d993697200a6bbff24a00f94f5e8ffaa4fe11d6386ed81d59f`、`7cc2d96f21d83b6c64009a6b8e09e4456fd37f153a2b5a09ab9d99b1f53e64f6`、`d119c245309f9611c9232deaa6cff1a3d81b838b5040deb4a901470371856964`。环境SHA `ff0f2e641980f515d1009031cdbdd54fae9f892cb241cb7587a535c727b821db`，producer清单SHA `be95732e3d3e843f72e925e6e2fd579b1b4e5495b40fd65a2fd5a97327fb4625`。Binding只从旧独立实机证据继承设备／driver／库／ABI预期，其余均投影自新actual产物；实际上传前仍核当前完整Binding。

新登记只供同包record/restore正确性：每例capacity3、一handle、物理B1/B2/B3双slot、同一15step分别执行record与verify；每stage55 exact attempt加scratch至多1、57 journal事件，host apply9／GPU物理行18／logical12／raw30。三例共六stage、最多336 attempt／342事件，18次warm与36次Graph replay即54次真实前向、108物理行；180份raw／270份Observer数据文件，五头raw共735,552字节。每stage Observer数据8MiB、journal1MiB、package64MiB、stage／supervisor receipt及stdout／stderr各4MiB；每例四阶段共1800s，三例声明上限5400s，失败不自动加额或重试。原有监督器以固定environment.values启动base Python，唯一顺序B11→B15→ONNX，首失败停止；CPU准备本身未seal／claim／调用GPU。该登记不构成FP16精度门、Worker、成本、ABBA或采用；DualFFN能力探针的独立预算缺口仍待后续解决。

首次B11监督器已actual exit1（chunk035a96）：`attempts/b11/result.json` SHA `3af9b9aad33a22854f062545b2707e64cbb0c70cfd8122206e156a6f6690f7cc`，`reason=supervisor environment differs`、phases空、seal与verify_registration均null。`open_attempt`先于环境核验，故原B11外层调用永久CONSUMED，不能删除或重启；尚未启动Rust seal／model子进程，GPU消耗0。B15／ONNX未启动，整组已按首失败规则停止。root真实退出记录在 `first-launch-tool-observation.json`。

有限CPU环境诊断在 `target/unified-quant-normal-launcher-env-r1`：原PowerShell/native传递方式实际90键、预期38键，额外52键且2值不同；明确以Python subprocess的env传字典后，实际38键与预期全部MATCH。两次环境观察工具分别actual exit0（b019bf／a52704），没有启动监督器、模型或GPU。此证据支持修正外层环境传递，不放宽或改写既有监督器。后续只允许明确替代尚未执行的B11控制登记并绑定本次失败；B15／ONNX原登记不重建，合并总GPU预算仍为原六stage／54前向／108物理行。此时还没有成功seal、record或restore。

环境控制续接已首次CPU准备actual exit0（chunk461b4f），脚本 `target/unified-quant-normal-env-continuation-r1/prepare_continuation.py` SHA `ed6ffa426f3340cc89361652078b15418eabdf0f77145596dd4f4a20fd875e27`。replacement-index SHA `a1e1063866e726c46382626b01cfb3adecacc044ada0a8953b52e961bd8ee9e2`，新B11登记SHA `a4bd7350a36e05772f190fbec5c28f5d8bb0142e9fb730ebddc03f2e0b35777a`；保留原479个producer成员及全部失败原件，B11复用原未创建的reserved root，仅另开attempts/b11-env-r2，原入口永久弃用。GPU限额完全不变，控制预算保守消费1s，B11余1799s、三例余5399s。B15／ONNX保持原登记。

以显式env字典唯一启动后，B11 session61351 actual exit0（terminal c0536f），四阶段各exit0，supervisor result SHA `61a88176fbe7f807dba46b6c3c6a2c6e3fe814d48a3f0f4d7756a284083d7e20`，verify result SHA `c0d5fc935c3ed07e693593ff461a9bce5b8bc9b8e0aa940d21b4adfe4859c5ba`。随后原B15 session1359 actual exit0（terminal bf7ca9），四阶段各exit0，supervisor result SHA `b0845fb6359832759191670d7365b3385db0ade0aec75b4ac7a690a80d4df002`，verify result SHA `43d4915d0a989bb345c5c3957dbcd27fd0ace2505e37bcefceb61868e082e237`。每例30对raw／30,648个f32、6对typed-bits／12逻辑行全等、重新导出包字节一致；B15实际日志含layer5 FFN dual=int8/down=int8/hidden240/rms_fused1。两例正常恢复均保留实际生产profile，但仅此离线诊断使用。ONNX已唯一启动session34347（初始chunkb4e68f），终态待观察；禁止重复该入口。所有root实际退出观察另存于continuation目录，此时没有新的性能或FP16精度认证。

ONNX随后完整终态：session34347 actual exit0（terminal ab153f），四phase均0；supervisor result SHA `79ad2fe6c126e6d572a8e0d50ebc69b2d1dcf5ef86244d5cdd83e0e0199357f2`，verify result SHA `7d6919746b74216659c91d307fdc0d6ac50069a35c31b56a38abb7e128008468`。三例的实际GPU执行现已全部结束，原失败与全部seal／claim原件永久保留，不追加／重放模型。三例合计90对raw／91,944个配对f32、18对decoded typed-bits均全等，3份包re-export字节全等；六stage实际范围仍54次前向／108物理行，180份raw存量共183,888个f32／735,552字节。三个verify均返回实际独立profile、全部finish和export成功。仅三局面来源与capacity3的有限一致性诊断，不是完整2044精度、搜索叶、Worker或速度认证。独立CPU证据审计尚待首次执行；正式Worker PID78408原启动时间仍live。

下一性能前置的只读设计已核actual375：DualFFN真实GPU能力探针须在observed分支的完整Binding准入→ModelUpload→既有同步之后、首次执行目录之前触发，并在同一OnceLock initializer内、C++ FFI之前持久记账。建议新精确kind `dual_ffn_capability_probe`，每个独立record／verify进程DUALFFN1时1次、关闭时0次；外部提前初始化的observed会话拒绝，缓存后续调用不重复计，普通None路径保持原时机／fallback。四项model派生计数不变，单stageattempt上界56→57、journal57→58；须显式区分17类新登记与旧16类原件，联动journal／Observer／stage reader／CPU预算与preflight。此处仅设计，没有补丁、测试、构建或新GPU登记，也不代表DualFFN或混合精度已取得收益；最快旧FP16控制仍是必要比较对象。

独立CPU证据审计现已首次执行并通过：`target/unified-quant-normal-record-restore-audit-r1/first-run/result.json` SHA `3ff1889bfffc5487e8000d0edcb43c849811ceec388af09128bc1b4fe6fab374`，root实际exit0（chunk3ff7fe）。852份绑定来源前后完全相同，before／after清单SHA均为 `4e181e36e6fd175675126fe07450343166518f8e9a1e048413715a54f44eaaa3`。审计核对原失败和一对一替代、三例实际进程退出、seal／claim、六份journal的真实回调及预算、54次前向／108物理行、90对五头原始文件／91,944个配对f32、18对decoded和3对包字节全部一致；180份存量raw的183,888个f32均finite。审计自身0子进程／0GPU，没有重跑或改动采集原件；真实工具退出另存该目录 `tool-exit-observation.json`。当前正常后端跨进程恢复一致性已得到有限实机与独立读回证据，完整精度、端到端性能、自动选择及生产接线仍未完成。

DualFFN后续增量现仅在新 `target/unified-quant-dualffn-{observed,budget}-source-r1` 静态开发，以actual375为基线分离真实probe hook与17kind计数／版本接线。没有修改已测快照或三例登记；新代码尚未应用、编译、测试或运行GPU，不能继承旧exe的实机通过身份。本轮不再执行GPU。

### 8.47 从恢复一致性到混合方案选型的最小接线

只读差距核查在 `target/unified-quant-selection-gap-review-r1/handoff.md`，SHA `fb7729a1c04600f915d38c5b989462d857fdc6a40cacfd21c1f67e3a017a2f18`；来源清单SHA `da50ceb8c5bbac394811086bde9f3bd695c2319bdec71c964598dccac8999560`。旧交接中“正常record/verify CLI尚不可用”已被§8.46实机结果替代，但当前有限三例不是12混合配方或capacity8的证明。

最少实施顺序为：补DualFFN与单次restore-only采集所需源码，冻结最终同exe身份；沿现catalog/ledger接原12配方及候选所属capacity3/8两套FP16参考；构造并复验有限执行包；直接对真实selection做目标物理B3/B8数值门，再进入完整旧控制矩阵。包构造使用已有有限输入，覆盖容量内全部可达batch与双槽，不把2004个selection局面机械放进record和verify两次执行。现底层`start_execution_observation`/`ObserveRestored`/DurableObserver可复用，公开CLI和许可仍强制两阶段同序列；需在既有collection预约内补一次restore-only入口及共享PB后处理，不另开空consume根。DualFFN增量的中间exe不能先全量采集再因新增入口重采。

12份B1/2044混合数值证据只继承为固定候选依据，保持原exe/Direct/DUALFFN0身份；不改称新exe/Graph/包/B3/B8/selection通过，不追加一轮完整calibration。现catalog仍限旧seed与单参考，collector的统一配置尚缺包与expected profile接线；原账本累计10 COLLECTION/16 ABBA、余量0，新后继段必须保留全部旧终态、负结果与永久seal。C32建议拓扑仍20唯一collection/44 ABBA，只是静态建议；包构造、恢复、probe、warm、capture与selection上传/前向须按最终命令分别推导再合并，20次collection不等于20次模型上传。仅完整矩阵收益候选才进入独立确认和holdout。该核查0测试/构建/GPU/新登记。

### 8.48 DualFFN实际探针与版本化预算的静态增量

真实probe hook已在 `target/unified-quant-dualffn-observed-source-r1` 封存：delivery SHA `55ef531757d02cbd43f8cbf9b0fd8211a50a15f8bd822fbe01cfea8235c17560`，composition SHA `bc2ab2ff645e9fff3774babf3ae76201eec7753d02c07aa4ffa28315de98391e`，source receipt SHA `041f9d67b9816e8e85e7e92ec0bae13d51114ec198164cee85b2c27727bde6d9`。基于actual375，5替换/1新增；20份限定来源前后稳定。8个局部OnceLock/闭包测试及3个startup测试仅定义，未执行；旧startup12定义保留。C++ M16/N1152/K384真实probe原字节不变。

有required RuntimeAdmission且DUALFFN1时，完整Binding准入→权重上传→既有同步之后、首次目录之前进入观察路径；同一OnceLock initializer先持久记账，再调用原FFI。外部已初始化的cache拒绝，后续普通ensure共享同一结果；hook Err保留失败且不调用FFI。panic仍有OnceLock未初始化语义，必须退出整个离线进程，不能据此声称同进程新模型恢复安全。普通None、旧legacy录制与DUAL0保持原ensure时机；legacy没有新的有界probe证据。新增读取遵守tactic_var，旧普通测试钩子行为保留。

root已读全部相关差异，独立限定复审未发现hook/cache阻断；这不替代编译。新增enum须与17kind journal/Observer/strict reader/CLI配套一起合成，单独叠此六源不是完整可编译交付。配套版本约定为旧preflight-v1/result-v1、campaign-v2保留16类且拒绝DUAL1；新preflight-v2/result-v2、campaign-v3显式17类，DUAL0/1分别精确0/1次probe，四项模型派生计数不变。两份增量均仅源码交付；没有应用、测试、构建、包构造、新登记或GPU，不继承旧exe实机通过身份。

预算配套在 `target/unified-quant-dualffn-budget-source-r1` 封存：delivery SHA `79525bda9eb6cff83301beba7b8d0bf2d840d9916ab91fa0259dd9eeabf03f61`，composition SHA `e9012fdb64f8394cdf7a703b2243cdb9f33ff6a076a2ae923f3f7e0e9b5c53e6`，source receipt SHA `dc7cea08ed6491556fa213d9f79e7cf7e2a687a0eed98579ac990f2c759c9353`。13个Rust替换、15项新CPU定义未执行，51份限定绑定来源稳定。与hook增量共18替换/1新增、26项新CPU定义；联合376源map SHA `b32ce8258f75c69515b5dd5afb89c5d2a797021ca68da24b81601f097eda610a`。root另只读核验19个目标不重叠、37份基线/替换来源字节匹配（chunkec2b81，actual exit0），没有物化快照或执行测试。

独立supervisor派生文件只增加campaign-v3识别，SHA `3050d8616dbf9cbf7d6038f6f77dab282fa02cadfe962ce5bc28b5b62d708f89`；不计入376个编译源。原21项mock和2个CPU子进程检查不能宣称此新脚本已通过，后续须绑定新producer并执行受影响CPU检查。所有旧失败、永久消费和负结果原件保留；下一步须按§8.47补一次restore-only selection接口并完成有限集成，不能直接以静态新增17类登记运行GPU。

### 8.49 固定执行包的 selection 采集接口与有限 CPU 验证

本轮在 `target/unified-quant-restore-selection-source-r1` 隔离集成 §8.48 的 DualFFN 探针记账、父 COLLECTION 证明和正常 Backend 的 restore-only 采集接口，共 386 份编译输入。根工作区正式运行源码、正式二进制和生产 Worker 未由这份快照替换。新接口不重复执行 2,004 行 package record/verify；它复用已经完成同 executable/provenance 验证的固定包，在父 COLLECTION 预约内生成一次子 claim，并执行自己的 selection 序列。

`CollectionPermit` 校验实际 executable/provenance、PID、argv、环境和 cwd，以及成功 CPU preflight、固定包完整 record/verify 证据和父账本的有限证明。Rust/Python 两端逐项核对 execution spec 与实际模型、配方、包、profile、语料和物理 batch；共享 numeric workload 不混入 batch 或 RPC 并发。现有 q64-v3 allowance 不允许新增候选 key，synthetic parent fixture 也不构成真实预约。

CLI 新增 collection preflight/collect，复用私有 Prepared 和正常双槽执行循环，固定 2,004 个 selection 局面、物理 B3 或 B8；B8 尾批为 4，不 padding、不 export。运行前逐条核对语料完整 Position、EvalParameters、PB 和输入身份。保存的完整五头通过原有共享后处理生成 Worker PB 和 typed bits；Python reader 显式识别这种离线恢复采集，只在外层真实 `Popen.wait` 退出成功后接纳产物，不补造 RPC/heartbeat/timing 字段。

Rust 首轮 CPU runner 的相对构建路径导致产物写进 snapshot，首个库编译 exit0 后被路径检查拒绝；386 原源码字节未变，额外 1,250 个构建文件及失败原件保留。r2 只改绝对路径，随后 76 项完整组检查通过，startup 组 14 通过／1 失败；失败是新测试预期错误消息漏掉既有 `offline execution: ` 前缀。r3 只改该测试断言，运行时代码不变。两份失败结果 SHA 分别为 `badc39fb2804f6ecdd45b227bd0f2cde8ab3d7dde74490b7f0282f359f4787cc`、`0c7a325249284635da38a3d74f7826d895f3e8bc929754297c5c7153a3e7707b`，均保留实际 exit1。

Rust r3 已完整终态：session39906 actual exit0（chunk `c9b5cd`），四次 Cargo test-link 和 **421/421 CPU 检查全部通过**。结果 `target/unified-quant-restore-selection-source-r1/cpu-r3/run/result.json` SHA `bee19b8c123e7883fdbb20722de4750b33f0404897530a3732ff0e4361597f62`，实际 tool 退出观察 SHA `4df0c4c38a768f4a271275d3b0f7c027712b7dd8ba3d7f9e24a83dc9d7da03d2`。386 份编译源前后清单均为 `377adf7485e5c91c3ecab99614e555cc94a0503f89f8173a9e9d9a9b8c077fe3`；最终 5,477 个绑定来源均重核通过，初始 5,393 为最终清单的不变子集，新增 84 个实际产物，不把两张总清单写成相同。421 项包括两种 feature 各 116 库＋89 CLI、CUDA 配置独有的 8 项纯 CPU 缓存检查，以及三个仅执行一次的指定读回。三个读回分别检查原三模型完成包证据、两语言共用父证明样例、历史 B11 物理 B3 的一个真实 raw 行到共享 Worker PB 的逐字节一致性，不创建模型前向。后续源码基线为 **`cpu-r3/run/snapshot`**，不能直接复用仍保留原测试断言的初始 `source`；本次链接产物是 Cargo test executable，不是最终发布 CLI。

Python r3 已 actual exit0（chunk `7ddfd2`），精确 15 项父证明＋17 项离线 reader 检查通过。结果位于 `target/unified-quant-restore-selection-python-cpu-r3/run/result.json`，SHA `6bea90ec6bd52c63618b2179752d54d3eec87dcbb22474961ea724fab9c3032b`；前后 2,153 份来源清单相同，总 2,155 另含实际 stdout/stderr。实际 imports 从 264 扩至 408，均已前置绑定。独立只读复核逐份重哈希 2,155 文件及核对九份脚本通过，报告 `target/unified-quant-restore-selection-python-independent-readback-r1/readback.md`，SHA `086bde2286deae5d665819e2b0257e268df18d04788823b9c338c51b41b8b5c9`。r1 驱动 SHA 字面量误改的 0-test 失败、r2 的 32 项测试通过但漏登记 `typing_extensions.pyc` 而整体失败均保留；累计 64 次 unittest 和一次 import-only 检查，不将失败改写为成功。

本轮无新 GPU 调用、COLLECTION 预约、性能窗口或留出执行。受控 collection 外层启动器、新有限 continuation、最终应用 executable／同版本执行包和 B3/B8 真实 selection 尚待接线；完整 C32 控制矩阵仍未登记。后续必须继承旧 C1 的 10 次 COLLECTION、16 组 ABBA 及负结果，不能另开零消费账本。20/44 仍是静态建议；新 supervisor v3 也不能复用旧 21+2 检查冒充已验证。续接边界见 `target/unified-quant-restore-selection-source-r1/continuation-boundaries.md`。

### 8.50 Selection 输入、采集控制链与最终 CUDA 应用

本轮在独立 `target` 目录续接 §8.49，未替换正常配置、正式二进制或生产 Worker。旧 C1 的 10 次 COLLECTION／16 组 ABBA、负结果和永久消费继续保留，没有新 GPU 预约、模型前向或性能窗口。

**完整 selection 输入。** `target/unified-quant-selection-inputs-r1` 的 CPU 编码器检查为 15 项 Rust＋15 项 Python，实际 exit0；result SHA `a76578c157eaaf2a33f4df4f65ed19cde3df3fbed67d51b8069ea31162091887`。编码器仍使用原 v7 特征实现，新增精确 2,004 行的 selection 导出；不创建 CUDA 模型。真实导出 session90657 actual exit0（chunk `921852`），两模型各 prepare／encode 子进程均实际 exit0，result SHA `f8404009991f594b1195e4d47a8cdb5e1d9f7c3582e388f73d026ac3653d703c`。文件位于 `target/unified-quant-selection-inputs-r1/cpu-r1/run/snapshot/target/selection-complete2004-r1/{b11,b15}/{pb,encoded}`；后续复制源码必须按显式 386 文件清单，不能递归把生成输入混入编译源。

独立只读审计 `target/unified-quant-selection-input-audit-r1/first-run/report.json` actual exit0（session7170／chunk `b72f4a`），SHA `9227aa3d613d758ff7fa27cf33d17e8ec17365e08a89b55ac074c76a579d722e`。18,688 份绑定文件前后相同，独立重建 4,008 个 PB；两模型共 8,016 个 f32LE 张量文件、127,630,752 字节，31,907,688 个 f32 全 finite，同局面两模型特征逐位相同。核对完整 Position/EvalParameters、初始布子、历史、lease、源顺序、PB input_hash、行／张量域哈希及逆空间对称和基本棋盘平面。这里只完成 **B1 存储**；真实 B3/B8 打包推理未运行，审计没有独立重写完整 Go/v7 特征算法，也没有模型输出、收益或留出结论。

**有限 catalog 与历史续接。** 隔离 catalog 支持 `unified_v1_restored_package`，将实际 executable／provenance／固定包／配方／物理 batch 绑定到 spec；共享 numeric workload 不含 batch。四个 catalog 必须覆盖 B11/B15×B3/B8，每个保留三个提名候选、统一 FP16 参考及原两类控制，完整性能比较对不删减。本次数值阶段性能状态为 `DEFERRED_RPC_DEPLOYMENT`，新增 ABBA allowance 为 0；这不代表放弃后续完整性能矩阵。真实 catalog、v4 contract、COLLECTION 预约尚未创建。

continuation-r1/r2 通过固定历史锚继承旧消费，不能另开零消费账本。原 q64-v3 最终 head 为 `a5fca1fed106c39b0986ab382f1b8ecd307f2e91b3f518d325abc840b5b0a227`，contract SHA `8f349af938a347723ca0ad2af93a3bcf4808460a5732982cfcbce6bc5fc73669`。新增 r3 为历史 v3 内部的唯一 v2 子进程补显式 `-B` 和共享绝对 deadline，只替换隔离导入模块的局部 subprocess 绑定，不改历史原件或全局 subprocess；delivery SHA `d400a180183f19256d0051d00c728ce91f7a518d0eeaa64c5fb16c2755e3961e`。该 r3 当时仅定义 12 项新 mock，实际检查和只读重放另行记账；不能用已通过的 r2 检查代替。输出大小在 communicate 返回后检查，不是流式内存上限；仅清理自有 Popen，不能声称整个后代树都已观察退出。

实际启动前的副作用审查又发现，原 v2/v3 `execution_replay` 会经 `validate_catalog → validate_search` 在历史 evidence 目录创建 `ledger-plan-check-` 临时目录；正常退出清理不改变既有文件，但中断可能留存。因此 r3 真实 replay 未启动，派生 r4，将三个精确旧调用目录及 Protocol 的临时生成文件定向到显式独立 receipt 根，并保留失败产物；没有全局修改 tempfile 或历史源码。r4 delivery SHA `1ad905b1e5b6e0ac09de23e83f4c498c883c5f5ba6170be4f3a94c388d636c0c`。此发现不影响已完成的旧结果，不构成重新采集旧尝试的理由。

`target/unified-quant-continuation-history-cpu-r1` 的首次有限 CPU 执行现已完整终态：session67562 actual exit0（chunk `fce01f`），20 项 mock 全通过；唯一只读历史重放也通过。result SHA `5974daa31c7b5cee37bdaa05acc57ac446cd9d182c3290c4bf032249d9416f83`，tool-exit SHA `c7478d14835c3e3677f23a9811fa415d11113a7bdaedbc0f494ec4aa9bcdf355`。原 81,693 个历史来源稳定，旧目录前后完全相同；最终 84,554 个绑定来源全部复核稳定，新增 78 个 scratch 文件保留在独立输出根，不能把新增产物后的全集称为初始全集。history wrapper owned PID74200 actual exit0／334.945s，v3 owned PID15252 与 v2 owned PID21988 均实际 exit0；venv redirector 与实际 base PID 单独记录，不声称观察了整个后代树。实际重放保持 66 个旧事件、10 次 COLLECTION、16 组 ABBA 及原最终 head，0 预约、0 真实账本修改、0 GPU。原历史 orchestrator 的退出码仍为 `NOT_OBSERVED_UNCHANGED`，不能用本次读回退出补写。

长流程调用方另在 `target/unified-quant-history-caller-static-r5` 窄派生，delivery SHA `8fe3fb5e1ba5183a699e8d669689d3838fdfe16f0f537b1bf952663c54108321`。首次历史重放仍是一次性、有界、失败不重试；后续操作复核同一完整来源清单并使用当前操作时限，不因首次 1,800s 窗口过期而必然阻断长 driver。prepare、validate 和各账本调用共享显式 receipt 上下文，动态时钟和 PID 不进入确定性 contract。12 项新 mock 与 17 项受影响旧检查尚待有限 CPU 验证，不重复真实历史重放。当前没有整个 driver 的总 deadline；逐操作协作式检查也不是阻塞 I/O 的硬中断保证。

该有限检查首次 session39675 actual exit1（chunk `c554dc`），`target/unified-quant-history-caller-cpu-r1/run/result.json` SHA `ae9839a5312839d03c95cbb119684365b25ad860ab14ae987b6914eabf1d6072`。12 项新 mock 各有实际 ok；随后旧 driver fixture 的 setUpClass 因缺少隔离 snapshot/target 父目录失败，末核又发现新生成 worker_pb2 模块未绑定，聚合 FAIL 保留。17 项旧检查尚未执行，不把局部 ok 改写成整体通过。只派生检查环境补父目录、把这两个精确 protobuf 生成文件保存在新 receipt 根并绑定实际来源，业务源码不改；不重复 12 新 mock、真实历史重放或 timeout。此前成功旧 suite 正常关闭临时目录，import 检查仅扫描尚存在文件，不能追认其覆盖已删除生成模块的来源。

只修检查环境的 caller-cpu-r2 已首次执行，session2708 actual exit1（chunk `189239`），result SHA `90d5d21163a182e39ec7ab8335a7f73fbb0bdaffa95af659c04089dc867c24aa`。前 12 个旧方法逐项 ok，第 13 个完整合成 catalog 方法在途达到原 300s unit 上限，未延时、未重启；最后 4 个尚未执行。owned redirector PID35908 的实际等待退出为 null、清理退出为 1，后代状态仍 UNKNOWN；实际 Python 身份记录 PID32524，root 后续精确命令进程查询未见存活，但没有补写其 OS 退出码。4,795 绑定来源末验稳定不等于 unit 导入末核完成，tests=null、聚合 FAIL 保留。正在只读诊断剩余测试的 CPU 成本，不扩大真实历史或 GPU 范围。

只读诊断确认前 12 项及启动约用 170s，第 13 项仅余约 130s，仍推进到 4/5 个合成 ABBA；892 次 Protocol 生成的 1,784 文件保留，没有发现该路径无限循环。文件时间戳只支持次序和近似耗时，不是分项 profiler 证据。`target/unified-quant-history-caller-cpu-r3` 保留同一 91 源和原 checker，只执行剩余 5 项；从新合成 fixture 开始，原中断 fixture 不续写。unit≤600s、总900s内含90s清理末核，没有延长原 r2 尝试。

该 r3 首次 session81427 actual exit0（chunk `51d2f8`），5/5、零错误/跳过，unit owned 实际退出0／429.142s；result SHA `4c6d5297b1df1175738e950b3cc2e92c4ad89e5d39fd9f822a35689a5c6d82b5`，tool-exit SHA `35b1e00fd86a189537e70dd1146d4a711695ea7ad71deacec1f0177776293b05`。实际 imports438→448，7,481 绑定来源末核稳定。原 r1 的12新方法、r2 的12旧方法作为局部成功继承，未重复；r3补齐5项，不能把前两份聚合 FAIL 改为PASS，也不能声称29项在一次完整末核中通过。没有重复真实历史重放、gate/timeout烟测、Rust构建或GPU。

**Python 检查与保留的失败。** `target/unified-quant-restored-selection-integration-cpu-r1` 因缺少既有 pyc 来源在 0-test 阶段失败；r2 仅冻结，独立审查发现 staged 目录缺少完整导入上下文，未启动。r3 改为显式 85 文件完整 Python 快照，前 11 项通过，第 12 项暴露 private Path 传入严格 string 接口的问题。catalog-r2 只修四处调用端 `str()`，未放宽 JSON 或父证明验证。r4 继承前 11 项，通过剩余 **166/166**，unit receipt SHA `6e2938844b2a8962cec1998ecc0cae144861300b628c7731a79a412c9bdd6ae8`；累计 177 个唯一单测通过，含失败重执行共 178 次。r4 外层仍 actual exit1：base Python 的 `.pth` 启动来源未前置绑定，真实直接子进程检查尚未进入。

r5 只补启动来源后执行两例真实 CPU 子进程，不重复单测。gate／LF／EOF 正常退出已通过；owned timeout 已实际终止自有子进程、观察 cleanup_exit_code=1，但 supervisor 单次 join 后过早返回 UNKNOWN，聚合 actual exit1。失败 result SHA `843e50ca5aa6730be40432bd40f671a18faad6aef6fdf3d0faa44bbf3cea517b` 保留。launcher-r3 只在原总 deadline 内继续短片等待未结束的自有线程；未结束或晚报错仍为 UNKNOWN，不延长时限、不重发 gate。delivery SHA `445c396f6c2e22acf5f9011ec210dedd2855351d3b7cc4982e5065ca601e7f33`；新增验证固定四项新 mock＋五项相关旧检查及一个 timeout-only 子进程，正常 gate 成功不重跑。原失败没有线程时间戳，不能反推其实际 drain 延迟。

r6 首次实际验证已经完成：外层 actual exit0（chunk `38dfa0`，首次 exec 内完成，无异步 session ID），9/9 unit PASS；唯一真实 timeout helper PID12892 正确返回 `timed_out`、terminated=true、cleanup_exit_code=1、exit_code=null，正常 gate／EOF 未重复。result SHA `fa1352eb2981c11f0e95d0fb39aee8ce7313b8d6572fdc7b0d4149e790bc3fa8`，tool-exit SHA `163f1c6241a7a2293130818f6398398993b80d0a761acb8fb72f7131d41ff369`，2,673 份绑定来源末验稳定；真实 smoke wrapper actual exit0／2.7045s，实际 imports 97→155 均在绑定闭包内。源码为 `target/unified-quant-restored-selection-integration-cpu-r6/snapshot` 的 87 文件快照；它只含 continuation-r2，历史读取 r4 由上述独立 89 文件快照验证。新四项使该 Python 组合累计 181 个唯一单测通过，不能把五项重复回归再算成新用例。真实直接 base Python 的退出证据不覆盖 venv redirector 的整个后代树。

**Rust 父证明与最终 native 构建。** `target/unified-quant-restored-parent-rust-cpu-r1` 单次 CPU test-link＋14 项普通父证明检查 actual exit0（session98387／chunk `06cc89`），保留原一项 ignored 测试不执行；result SHA `81790528993b04423d39666d5c1303d3c723389c426525ea9b388575d04c41c0`。1,433 份绑定来源和 386 份编译源分别前后相同。该次没有 CUDA 或真实账本操作。

`target/unified-quant-restored-selection-cli-native-r1` 首次实际 `cargo build -p katago --features cuda --release --locked --offline --message-format=json` 已 Cargo exit0／199.53s；生成 exe 20,071,424 字节，SHA `dd665d03fe31739712a205eaa6f63852dbc27fee339dfe91f38067237e57bfdc`。外层因误要求 embedded workspace 集合等于全部 386 源而 actual exit1（session77826／chunk `590e7b`），原 FAIL result SHA `21bb3ef08444d9862055c4f8ff9bfb68c7ab03340656d5ae203be4da19942f4b` 不覆盖。实际 build.rs 只包含 workspace 根与 crates；差集精确为两个未启用的 TensorRT `cpp-shim/src/trt_shim.{cpp,h}`，CUDA-only artifact 不应要求它们进入 embedded 集合。

独立 `target/unified-quant-restored-selection-cli-native-readback-r2` 不重编、不运行 CLI，以现有 Cargo artifact、canonical provenance 和 exe 原字节读回 actual exit0（chunk `727668`），result SHA `f1cdaf64cb780482b8e1201dedb533034a604166889f50281591e14b14bdd8df`。4,514 份绑定来源前后相同，embedded 的 3,559 个物理条目逐一映射核验、24 个 synthetic build/context 条目单列；384 workspace 源精确等于 386 减上述两项。provenance SHA `72bb2282d427a7d9290b1efc15b99be5c0807605a644de823aa129cfd83dbb2f`、ERROR=None，556,751 字节在 actual exe 中完整找到。CUDA DLL 仅核文件身份，不声称实际进程已加载；也不声称覆盖整个 Cargo registry 或 OS 工具链。

**下一阶段包矩阵只完成静态推导。** `target/unified-quant-candidate-package-wiring-static-r1/handoff.md` SHA `2ebbe43fe9ba90e4a648897348bdf717d33321466a0b8d5a9692d23df53c95d3`，67 份有限读源核验，Source 清单 SHA `cd3d553bffe8b3b4c943b99e80c80126d7d8a157a583e35abe83211275de6324`。12 个候选 recipe identity 各异，加两模型 FP16 共 14 配方；由于 collection 要求 `max_batch_size == batch`，四个 catalog 实际需要 16 个独立新版本包，参考配方的 cap3/cap8 包不能互换。

| 拟定成功路径，尚未登记／执行 | 上传 | 模型 GPU 执行，含 warm | 物理行，含 warm | 独立 DualFFN 能力探针 |
|---|---:|---:|---:|---:|
| 16 包各 record＋verify，全容量双槽 | 32 | 528 | 2,016 | 32 |
| 16 次 restore-only selection | 16 | 7,376 | 32,184 | 16 |
| 合计，不含 legacy RPC 控制 | 48 | 7,904 | 34,200 | 48 |

Selection 的正式提交为 7,352 次／32,064 行，另有 24 次 warm／120 行；48 次 capture 只构图，不重复计为 GPU 模型执行。每个包 stage 的 exact attempt 是 C3=56／C8=141，另 ScratchAllocate≤1；完整 17-kind 分项见 handoff。DualFFN 的 M16/N1152/K384 原生能力执行单独记录，不折算模型行；本方案 GEMM 计时调优为 0。另四个共享 legacy control 合成 20 个拟新增数值 COLLECTION，不能由该数量推断 RPC 实际上传／预热次数；44 组 ABBA 仍为后续性能阶段。上述都是源码预算推导，不是新额度或消费记录。

旧三例 campaign assembler 仍固定旧 exe、cap3、16-kind/v2 与 DUAL0，实际新版本的 16 项 v3 构造器在独立目录静态派生；现有 supervisor 的 v3 接口可复用。各包必须先由新 exe 生成真实 CPU preflight，再冻结来源／预算并完成 record/verify；cap8 与这些混合配方在新 exe 尚无运行证据。剩余工作包括长流程调用方接线、真实有限 continuation／catalog 和物理 B3/B8 selection。catalog 的真实注册需要完整已验证 package，不能先用占位 Source 登记。旧 exe 的三模型恢复验证不能通过改写 SHA 转用为新 exe 包。性能、RPC profile／缓存隔离、完整旧控制矩阵、独立确认及 holdout 均仍未完成；任何 CPU 输入、编译或账本工具检查均不能替代这些结果。

构造器静态交付为 `target/unified-quant-candidate-package-assembler-static-r1/delivery-r2.json`，SHA `4ad76afbce05744a57600cf6611d2617906f04abd04656d58466433dd59c5166`；入口 SHA `cc5c34f8ede3c1e06b090ba8b35307949faea94073a8ef156c6885205ccbeae7`。独立静态窄审的 50 来源稳定，report SHA `e4440bf45b3b325b41484a69d0307187cab1af2507bf84f308665bfe6a38688a`；审查修复显式 recipe 缺 cudaQuantPlan 配置和 Windows extended DOS/UNC Source 比较问题，未修改历史原件。10 项 synthetic 检查此时仅定义。prepare 只生成固定 16 项 CPU 输入登记；真实预检与外层实际退出回执齐备后 freeze 才生成未 seal 的 v3 campaign。构造器本身没有 subprocess，不替代后续 GPU 运行；静态阶段尚未执行 prepare/preflight/freeze。

**实际16项CPU预检与未执行计划。** 首次 producer-r1/session3309 actual exit1、0 child/0测试/0 prepare：既存的 Attempt helper `.pyc` 未预绑定，result SHA `b4b5a1d60fd120020476f9c6a2abea0211ac430920f289bf76bbc049336ed021` 保留。`-B`禁止新写缓存，不禁止引用既存缓存。r2保持runner原字节，仅补确定的53,334B缓存Source并使用新输出；其余五个明确模块的缓存不存在，未知import拒绝未放宽。窄审 SHA `772ca1d87cf4d58351b4859f16238626e2bbd0ddbf8e82383f9259479f83d826`。

`target/unified-quant-candidate-package-cpu-r2` 首次 session19770 actual exit0（chunk `3b0806`）：10项synthetic、一次prepare及16个真实CLI preflight共18个owned child均实际exit0，总243.453s。control result SHA `79ebd2dcc9478764cb6233bac5f0eccb77d18bd3804433fae041aab09223cec7`；3,188控制来源与3,194 CPU输入闭集末核稳定。实际Prepared为 `target/unified-quant-candidate-package-prepared-r2/prepared.json`，SHA `df2c40e84416497bb67303386c6c97236755c4a229397543d0c474c8d9b5a25b`；CPU结果SHA `3d06dea47e134cb077a79086ce3529b2a2d37b27722c8eaba2eb3da149d3cf26`。root观察外层exit0后另写固定tool-exit，SHA `7986e31caa7f540f6a3684ccd25fb697d725ff137092e4d5689929923647e664`，没有让producer自证外层退出。

同一已审assembler首次freeze/session99734 actual exit0（chunk `7941d5`），生成 `target/unified-quant-candidate-campaigns-r1` 的16个v3草案。freeze result SHA `607d24543ff6592e0ce434a2660594bdac16185f550447dde32480d2a4619e9d`，producer-sources为692,390B／SHA `ddc79c54ee4d2b9c1d48b7e7f2c9359980b0139a94b0be912ec55faf1efb5a49`；root退出观察SHA `af142be0a72f0ec01c59f2f54fc3c64a01dcd35980c3a9a32e63841c0d0351f2`。状态严格为 `CAMPAIGNS_PREPARED_NOT_SEALED_OR_LAUNCHED`，runtime child、seal与GPU调用实际均0。草案中的32上传/528模型执行等仍为未来声明预算，不是已消费记录；真实package、record/verify与selection尚不存在。所有CPU耗时都是工具耗时，不是模型推理基准。后续RPC隔离仍需核对能否保留同一exe身份，不能在必须重编的前提下盲目先制作GPU包。

**同一exe的后续C32路径。** 窄静态核对 `target/unified-quant-same-exe-c32-isolation-review-r1/handoff.md` SHA `6e120c59a30f4c29b7460b591ba6f9ae419f5e261cbb3f4bc5f650c0a3117a2e`，27个有限读源清单SHA `00499db35ce22e21045c3497905656f29b78a27357b8ad85c1e16cfaa5785d8e`。actual386已有完整包恢复及Worker本地ExpectedProfile门；`--capacity32`是请求并发，显式 `nnMaxBatchSize=3/8`仍优先，因而不必仅为C32基准重编或构造容量32包。未来真实verify终态提供actual_inference_profile，不能预写或另用GTP上传猜测它。

当前还需三处Python接线：临时harness在Welcome前独占唯一Worker连接并严格核ID/模型/backend/profile；性能runtime文件允许包路径/SHA及实际verify profile且保持四个执行配置值；将offline selection数值证据与本臂实际RPC Hello/heartbeat资格分开绑定。不能给offline报告补假C32窗口或握手字段。各臂使用全新本机harness及owned Worker，旧控制保留原配置，完整44性能比较不缩减。该路径只是源码可行性，尚未实现/测试/运行；当前冻结合同ABBA额度仍0，性能阶段需要同一历史根上的独立有限登记。这是流程额度边界，不撤销用户的统一优化目标，也不要求重跑已消费负结果。

本机单Worker基准的profile文本交叉核对不是通用生产认证。生产Go Server的typed profile与缓存分池缺口仍在；若未来改Rust/proto重编，现有包不能改SHA复用，需新exe下的包和最终确认。该生产改造不构成当前受控C32基准必须先换exe的理由。本轮静态核对没有服务启动、CLI执行、GPU或性能登记。

### 8.51 同一二进制的包部署、RPC 准入与性能报告适配

§8.50 的三处 Python 缺口已在隔离增量中实现，不改 Rust、Go 或 protobuf，也不更换已构建的 executable。三份静态交付分别为：`target/unified-quant-rpc-admission-static-r1/delivery.json`（SHA `f20d5a511f5602d97e3a05fdb41ec78c3282d944a9f3bfe0261c08d036502191`）、`target/unified-quant-rpc-package-benchmark-static-r1/delivery.json`（SHA `5cc95a6d6b370024b583a7994409d7ba4ce5b2397c8efec361a9a805dc00c4c6`）和 `target/unified-quant-rpc-ledger-static-r1/delivery.json`（SHA `6f6ea0efdd4ac98bcd9d782a8bb24308bab321d11f07c362487bfd5c003a47ad`）。相对于 actual91 快照，合计三处替换、五个新源；根工作区正常运行脚本尚未由它们替换。

临时 WorkerHarness 的可选准入门在读取首个 Hello 前占用本臂唯一连接槽，核对 worker/model/backend/profile/协议与能力后才发送 Welcome。第一连接失败不能换一个补测；第二连接、重复 Hello、提前断流会令本臂永久失败，实际 Drain、owned-child wait 和关闭后仍检查该状态。未指定新准入参数的旧工具行为保留。此门用于受控本机测试，不宣称能认证恶意本机对端。

新 benchmark 模式显式为 `--restored-package-runtime`，报告类型为 `rustgo-restored-package-corpus-abba-v1`。离线 selection 继续没有 Hello、RPC 窗口和 heartbeat；三角色的 `numeric_sources` 分别绑定 `offline_restored_selection` 或原 `rpc_continuous`。旧控制仍要求 C32 数值采集的原 window/capacity，不能将历史 C1 结果改成 C32 资格。实际各臂独立记录真实 Hello、heartbeat、完整计时和退出。包 runtime 从已成功的 native verify/selection/父预约来源重建，保留完整环境、配方与四个执行配置值，添加实际 package/SHA/ExpectedProfile；本模式固定原 executable 路径，旧 v1 的 relocation 行为不变。

账本仅为这个显式新报告增加读取分支，继续重算原数值资格、四臂原始 timing、旧控制前后指纹、完整请求、时间重叠与重复消费；offline runtime 的派生配置和 receipt 必须能从验证来源重建。原 v1 的 continuous 门不放宽。当前 v4 `ABBA=0`、性能 driver 的 deferred 状态和旧 10 COLLECTION/16 ABBA 消耗均未改变；此源码不发放新的性能额度。

独立有限静态审查 `target/unified-quant-rpc-contract-review-r1/review.md` SHA `558bc1a0d05414477ea92ce33e7843fda96ca18b19596eef0938ac2255deaa15`，evidence SHA `86c59c2ce996e47b3a1d6ab1e4e7cc21d678c26511f666c7c8892eb06ecf1315`，33 份来源末核稳定。执行前发现并修复实际 producer/launch 对象字段错配、legacy profile 为 null、重复删除 comparator 字段，以及测试共用可变返回字典等问题；审查本身没有执行测试。

握手部分首次有限 CPU 检查已实际通过：`target/unified-quant-rpc-admission-cpu-r1`，外层 direct actual exit0/chunk `355504`，producer PID78248、owned child PID45132 实际 wait0。19/19 检查全部通过，无跳过；unit imports333→333、producer imports107→107，2,136 个绑定来源前后稳定，禁止操作尝试为 0。result SHA `169e23e5317aba7dd9698ebcae9989a501ebe0eddd1de2ab1ade69a7c52261ea`，tests-result SHA `53a18abfa741eb494d34770367cb5310002d8001afef7da59d74d1dd49e3ae38`，外部 tool-exit SHA `bf38cd361b5a4a9be8d38f6d70c205a636462b00797d6d06b0d799f1b1ee5db1`。这是 synthetic admission 和 fake service 的 CPU 结果，没有 Protocol/protoc、真实 socket、Worker、模型、GPU 或账本操作。该入口不得重启。

运行配置的 20 项与报告 reader 的 14 项已另行组合验证。首次 `target/unified-quant-rpc-package-reader-cpu-r1` direct actual exit1/chunk `c2900d`：reader 测试文件多一个右括号，导入阶段 SyntaxError、0项执行；原 result SHA `a80adb28a86f9b80d7718cf64ddb50072d6516cf6d561e44b6f17c54556be97b`、外部 exit SHA `104e522a44f2d59777d0bc5e8e4515fd17305725a9ccc3dc3a642a96b1662da0` 保留。静态审查没有发现此语法错误，不能将其报告当作编译通过。

`target/unified-quant-rpc-ledger-static-r2` 仅派生测试文件少一个右括号，delivery SHA `10386fd89e37b6416b9a3c004fbdd8b541285047bacd70949c777be3b0a181f8`；运行代码全部保持原字节。三个测试模块先完成有限 AST 解析（actual exit0/chunk `d067a5`，没有业务 import），随后 `target/unified-quant-rpc-package-reader-cpu-r2` 组合96源，首次执行尚未运行的34项方法全部PASS：direct actual exit0/chunk `fb59cc`，producer62720、owned venv54824实际wait0；unit basePID83160仅由identity报告，未另外观察OS wait。unit imports426→426、producer107→107，2,493份绑定来源前后相同，禁止操作为0。result SHA `70553b907c74714440053be33dcb339975e5367d78de87abf9bfe2bcb019de1a`，外部exit SHA `936c1ab3b4d9e682f116463157d39672e6db642e1eaaf591aaf7953fd53eb549`，摘要SHA `3d1db98ca615ac5ec3441ffbbdf0914b09e4bc3faaf68ab9b5917d71af44ce23`。两组共53项实际CPU检查通过，没有重跑19项；34项的r1聚合FAIL不改写。reader 的合成测试在已独立验证的数值/package 公共边界使用 mock，四臂计时、准入、退出字段、旧控制指纹分支和重复证据检查使用实际 reader，不能替代真实包或 RPC 演练。

本轮只读 `nvidia-smi` 观察仍为 RTX 5070 Ti、UUID `GPU-c1842c67-b9c6-3eb5-4592-ebb8111d931b`、driver610.88；16,303MiB 总显存、3,961MiB 占用和3%利用率是瞬时读数，不是独占或性能窗口证据。生产 Worker PID78408仍在运行，未停止。

**首组 pre-seal 启动失败。** 在全部CPU检查终态后，只启动 `b11-c3-mix-000` 原冻结 supervisor 一次。PowerShell 对本进程环境逐项清空/设入冻结38键，再以 `&` 调用原 argv；direct工具实际exit1/chunk `25808b`，reason=`supervisor environment differs`。`attempts/b11-c3-mix-000/attempt.json` 为 `CONSUMED_NO_RETRY`；result为 `FAILED_NO_RETRY`、phases为空、seal与verify_registration均null，GPU root目录不存在。外部观察 `target/unified-quant-candidate-campaign-launch-observation-r1/tool-exit-observation.json` SHA `4ce540f0c0cbe8f32682e027513f5cd108bb166855cbf3d65f7558685bc6b3c6`。没有seal、native child、模型上传或GPU调用；原attempt永久保留且不得重启，后15份未启动。

只读诊断 `target/unified-quant-candidate-preseal-env-diagnosis-r1/handoff.md` SHA `f53feac5a683621b2f143ebfe25c5bf7ff29bd2e644970a6427a2ad706859fca`：冻结环境无空值/null/大小写冲突；本次实际子环境没有保存，具体差集仍UNKNOWN。历史 `target/unified-quant-normal-launcher-env-r1` 已有实际CPU证据：PowerShell native继承不匹配，而显式 `subprocess.Popen(env=values)` 完全匹配38项。该旧结果支持复用已审 `Attempt.run` 的显式环境入口，不能把旧差集写成本次事实。新建的 `target/unified-quant-campaign-environment-diagnosis-r1/inspect_environment.py` 未执行，没有再次重做同型诊断。首例补齐须形成绑定这次0phase/0GPU失败的一对一控制续接，不能用新目录偷偷重置已消费attempt或直接重启原入口；剩余15组也尚未运行。新的实际 record/verify、B3/B8 selection、完整44组ABBA、确认和holdout均未发生，没有新增推理收益或可采用方案。

### 8.52 显式环境续接与 16 组执行包的首次实际运行

本阶段保留 §8.51 的首例永久失败，不重启旧入口。`target/unified-quant-candidate-preseal-continuation-r1` 只替换首例的控制身份、attempt、producer manifest 与四处 campaign argv，原 GPU root、配方、输入、工作和限额不变；尾15份精确继承。原失败0.031秒向上扣除1秒，首例剩余1799秒，串行总限28799秒。新增失败证据只进入 producer manifest，原 `external_inputs` 仍与真实 CPU preflight receipt 闭集一致。首次构造 actual exit0/chunk `a4a946`，replacement-index SHA `6b43ca1d1c6d8d190a69dd8c4391ce33cd2179fb88d5726fd1ffb2409936da93`，没有新GPU预算。

七文件 AST 实际通过（chunk `9ae45b`）。独立一次当前环境观察 actual exit0/chunk `f07534`，显式 `Attempt(env=values)` 的直接 base Python child PID70824 实际 wait0，38项完全相等；result SHA `535c8b637b3f8d5a8e3c73822f2f72192b9f84cbeec6351003492a0c59612931`。随后续接5项及launcher8项合成CPU检查全部通过，direct actual exit0/chunk `fb41f3`，owned venv child67604 wait0，336→336 imports、2518绑定稳定、禁止操作0；result SHA `a5add430af7bdb972ee31a0b4e7a31140a4d9baa80466018757b4a1e9f643857`。没有重跑之前的53项。

launcher-r1 的首次CPU freeze因根PowerShell参数列表拼接导致 argparse exit2（chunk `9ce398`），修正argv后首次实际freeze又因Windows路径大小写规范化误判Source而exit1（chunk `f1546a`），未生成启动计划或启动GPU。原配方 `D:/code/rust_katago/target/unified-quant-calibration-full-r1/b11-000-recipe.json` 实际18409B/SHA `f928bc48dda3bce59006e0ed8af06b64bf3a8c41a92472a7e8a91bd485309a61` 未变；Python将路径规范化为 `Rust_KataGo`。隔离r2仅让 `verify` 打开原路径后比较实际bytes/SHA，并返回原Source保留拼写；freezer字节未改。8项受影响检查加2项大小写/extended DOS别名与篡改拒绝检查全部通过，actual exit0/chunk `09869a`，334→334 imports、2529绑定稳定；result SHA `b3052de61a9a72861fe3bdf0141bd44a5644093aad9cb1aaaca06a3ede1934d5`。原r1及失败记录保留，未重跑5项续接或环境观察。

r2 CPU freeze 的同一 session37178 actual exit0/chunk `ea0fd1`，计划 `target/unified-quant-candidate-campaign-launcher-r2/frozen-plan.json` 为14758209B/SHA `9f2b4479d9b235aa1de0d1bcac6b792fe2abc611c2a66ac3ca81abe87550e464`。只使用已有r6 supervisor和原32上传/528模型执行/2016物理行等有限预算。显式环境、每组来源前后核验、实际owned wait0与四阶段终态检查全部在单一总限内；首错停止，禁止重启。

**首次真实启动正在进行，已完整通过1/16。** session69973，controller PID40860。首例 `b11-c3-mix-000` 的 supervisor PID77852 实际 wait0/172.979秒；四阶段 seal42556、record38104、derive_verify36196、verify39296均由owned handle观察exit0。controller组结果为 `COMPLETE_OBSERVED_PROCESS_SEQUENCE_ONLY`、来源稳定，SHA `79e4350d944723638f9c4f65a65f761b92c8f4ad82070c473c2dbcc97c156319`；native verify-stage 为 `VERIFY_STAGE_COMPLETE_BITWISE_EQUAL`，SHA `977611c054f99f4a8febcc7dc3e3e5be59661817ef6cd5367cf75eaad4bb4ea8`。这是B11 FP16参考配方capacity3的同配方记录/恢复逐位一致，不是对FP32或校准门的证明。当前已自动进入第2组 `b11-c3-mix-001`，尚未观察串行总终态。

最初在途工具观察见 `target/unified-quant-candidate-campaign-launch-observation-r2/observations.json`。必须继续观察同一session和原attempt，不能因一次等待无输出而重启。GPU窗口内仅做静态并行工作；真实B3/B8 selection、端到端ABBA、独立确认和holdout仍未执行，不据执行包一致性通过声称性能或精度验收通过。

后续同一session实际观察第2组 `b11-c3-mix-001` 的supervisor owned wait0/218.441秒（chunk `4b3e98`），controller组结果完整、来源稳定，native verify-stage仍为 `VERIFY_STAGE_COMPLETE_BITWISE_EQUAL`。至此2/16完成，覆盖capacity3参考及首混合候选；当前第3组 `b11-c3-mix-002`，没有观察串行总退出。当前交接入口为 `target/unified-quant-candidate-campaign-launcher-r2/run/ROOT-LIVE-STATE.json`，其中完整绑定两份组终态与第2份native stage结果。

同一 session 后续实际推进到 **8/16 完成**（chunk `cc1fbd`）：B11 capacity3 与8各一个参考及三个混合候选的 supervisor 全部 owned wait0，native verify-stage 全部 `VERIFY_STAGE_COMPLETE_BITWISE_EQUAL`，`restore_verified`、`numeric_comparison_passed`、`all_handle_package_bytes_equal` 为true，`accuracy_or_performance_certified` 为false。root仅读取已完成的八份组结果及八份native终态，没有提前执行独立raw审计。进度原件 `run/progress-08-observation.json` SHA `6341af4783fdc1ecb2b405f8695a8d2d15353a5fcc0da7c3143ca777e3f4bbed`；同一controller已进第9组 `b15-c3-mix-000`，实际外层总退出仍未观察。`ROOT-LIVE-STATE.json`是可更新交接指针，不代替冻结计划或实际总退出回执。

### 8.53 后续 selection 的薄组装器与独立审计静态准备

GPU窗口内仅进行静态源码工作。`target/unified-quant-restored-selection-assembler-static-r1` 新增一份薄组装器，复用实际386的collection-preflight/collect、actual96的父预约与catalog API，不修改Rust或原执行包。prepare保留原16配方/model/config/四身份键/环境，映射已完成CPU审计的2004局面输入；B3按668个整批，B8按250个整8与末尾4，完整覆盖且不padding。freeze必须读取真实16 CPU回执、16成功包链以及root实际外层退出回执，才发布未预约的collection登记。delivery SHA `b85d3824dc9310173b0abab484b36f46523ad934d8f278cf3f23a4749586b85f`，assembler SHA `5c1dc9428672de0040315f347c09b76af82f5eca0639170f2f4344450e67bbc2`；独立静态review SHA `f096404980e88816881d0e53185614de7f7beb10e68734cb9a7a439c1fe7f363`。10项CPU检查仅定义，未执行。

对应最小CPU runner在 `target/unified-quant-restored-selection-cpu-static-r1`，delivery SHA `8869576d6b305c0f58fa3023df466d8bb94cfee72dced92431c1905bf17fa4cd`，entry SHA `5f693132a11e73b73c2f7abfbeb00abd6f351b6105e092618f1435e861ecca39`。它复用已审Attempt，固定10项unit、一次prepare与16个真实collection-preflight，共18个owned child；单一1200秒总限、child至多120秒，30秒cleanup预留包含在总限内。prepare-child继承父截止值；不自行延长。独立窄审 SHA `5457870a47cbe775f9b302b6d379e05169fd516f3a79f6e1080fc61d55c66642`，17个有限来源稳定。实际plan/test-plan、venv/import/cache与输入闭包仍待GPU窗口结束后冻结，不能把静态review写成准入或PASS。

16包独立审计器由旧三包审计器最小派生，当前封存版本为 `target/unified-quant-candidate-record-restore-audit-static-r1/r2`，delivery SHA `62419373d189df0da70d08de14e283085256bd0793af2245b9bc63650af70cad`，audit SHA `8589de48cf7d7b645750f23321067db890236cf3cf551cdd5cc1ed69ec3fca8a`。静态独审发现r1残留attempt journal v1与实际v2不符，以及Windows规范化Source别名被误拒；r2仅修两处并增加定义，旧r1保留。独立review SHA `2427eac533110739e6b51b0ade5e9d43102e69101f22e90db836b58561956197`，47来源稳定，5项检查仅定义。实际审计必须等待同一session69973的16组总终态和真实root退出回执，当前未运行或读在途原始输出。

两份C32控制模板命令与workload仅静态准备在 `target/unified-quant-c32-control-templates-static-r1`，commands SHA `69b658befbfc88b77244a13c661c4f0f3af86cea5bd12e77c79e764f0d12e87a`。原旧控制spec/model/config身份保留，新模板用于实际C32数值资格，不能借用旧C1资格；未来四restored catalog各6数值/11性能配对是完整矩阵描述，不是执行额度。模板、四catalog、同一历史根v4接续、B3/B8 selection均未物化/执行，当前ABBA仍0；原10 COLLECTION/16 ABBA消耗及负结果不能重置。

**本轮GPU最终失败终态。** 随后同一session69973 actual exit1/chunk `6fec6b`，controller结果 `FAILED_STOPPED_NO_RETRY`、completed_campaigns=8、来源末核稳定，elapsed1702.609秒；result SHA `69a5a18111de1483fcd2dd9f54ac18f2f7250d7c3f09611464f4ae77399e6715`。root已实际观察退出并写 `run/tool-exit-observation.json`，SHA `aa53534db791b5a833552e0f3a693ec85e76b1f25aee5bac74ede240ea0f2dcf`，明确actual_tool_exit_code=1；`ROOT-LIVE-STATE.json`同步终态。首B15 `b15-c3-mix-000` 的seal PID77204实际exit0，record PID40488实际exit1；后7份 `b15-c3-mix-001/002/003` 及 `b15-c8-mix-000/004/005/006` 未启动。不得继续轮询当作live、重启原controller或清除失败seal。

实际stderr为 `CUDA model upload failed: quantization recipe has no eligible FP16 projection for DUALFFN`。已定位旧候选组装器 `target/unified-quant-candidate-package-assembler-static-r1/staged/assembler.py:340` 将两模型统一由DUAL0改为1；actual386 `cuda_exec.rs:1931` 只接受存在FP16 dual且mid_channels384、hidden1152，:1962–1967对不满足却请求开启的配置拒绝。B15所有45个dual的K=512，因此此配置不合法，不能放宽固定CUTLASS形状或静默把强制请求解释成成功。现exe已有DUAL0→Projection及17类中probe exact0的路径，修复可以位于新配置派生层，保留同exe；尚未执行修复后的GPU验证。

原CPU preflight只核tactic格式与配方ID/shape/precision，没有调用上传阶段的tactic eligibility检查，不能将其通过解释为所有native优化路径均可用。失败journal只出现model_load及model_upload两项预约，均必须计入消费；当前没有handle/warm/replay/Dual probe记录。拒绝发生在逐层权重上传前，但不能据此把CUDA运行时初始化或已预约上传改记为零消耗。新独立审计须保留8成功/1失败/7未启动的真实状态，已完成B11不重做；原要求16-all-success的selection assembler/freezer前置不成立，暂不执行。所有失败原件与旧10/16账本消耗保留，后续合法B15配置仍须新CPU实际preflight及明确继承失败消费的有限登记。

root另完成16份原registration→真实CPU preflight→已绑定recipe的只读元数据交叉核对：`target/unified-quant-b15-dual-failure-handoff-r1/shape-policy-diagnosis.json` SHA `8ea282c4df4d5c6c3aeee57b4ab0965561bd898105545020ab311cf70399eb57`。B11八份均存在FP16 K384/H1152 dual，B15八份均没有；没有重新解析/上传模型、CLI或GPU执行。新B15 DUAL0的17类预算应为C3 exact55/events57、C8 exact140/events142，原四项derived计数不变；完整环境与Binding.execution_policy_sha256必须改变，不能称完整运行身份不变。旧selection工具的DUAL1/单一全局环境假设也须逐case改正，具体断点见 `target/unified-quant-b15-dual-failure-handoff-r1/next-integration.md`。上述是根因及接续边界，不是修复后的实际preflight/GPU成功或新增额度。

### 8.54 B15 配置修正的 CPU 检查与部分终态独立审计

新 `target/unified-quant-b15-dual-compatibility-static-r1` 仅派生8份B15 CPU preflight输入，按已绑定recipe的FP16 dual形状派生DUAL0，不改原模型、配方、配置文本、exe、完整1..capacity序列。新probe exact0与event C3=57/C8=142由原预算函数派生；完整环境只改DUAL值。旧8份B11、失败claim/load/upload及1702.609秒既有控制时间保留；条件总load/upload各33明确包含原失败，不创建新增额度。delivery SHA `0c866f80ec4cf97eef25f2c9160c65e999e4351694de36a2b1679d0ab27e0ca8`，entry SHA `35a8463c699664b090906faf3938b86488b9aee2c1b6126c827ddf557a5b7959`，独立review SHA `d76bd713012daacc6bfbbd1243cd8712de2971c810d52ff45063826138ca4bdc`，86份有限代码/元数据来源稳定。

partial auditor r3保留r2原件，成功case的20962字符检查段逐字继承，外层严格要求真实exit1/chunk6fec6b与8成功/1失败/7未启动。它新增核失败seal/owned wait/永久claim/两项journal预约和observer失败事件，并将未观察的CUDA启动活动保持UNKNOWN；没有把上传预约改成零消耗。delivery SHA `a755a56dd0603af4e72988a0737350a96f23c2fd20c1c0c8dc22a6dfcf19fc7a`，entry SHA `25bc4fa40bb1a604b5cb0aa35a13814d25ee0ad708d783006f193353ed7702b7`；独审review SHA `8987e71401bb6139ff54b397580c417939b2000e346853629cd2241994284289`。

5份Python源码首次AST实际exit0/chunk `13557e`，只解析源码不业务import。复用原 `run_once.py`/`check_tests.py` 和已通过的venv/import/cache闭包，计划冻结为 `target/unified-quant-b15-compat-partial-tests-cpu-r1/frozen-plan.json`，SHA `9fc1c0eede4ea5790588192095ec96e09736c8f63966da51699c93e8a828df4a`。freezer独审发现仅hash交付JSON却未核复制文件对应声明的缺口，已在首次执行前补Source→delivery逐项内容核验；最终review SHA `38d33868c89099803b246e9ad60a52bf7dc3e6532e4488910e315af7651bec1b`。没有重跑此前53或其它已完成测试。

16项首次CPU实际exit0/chunk `5985a2`，producer PID17504、owned venv PID51812实际wait0；组成是4项从未运行的原审计helper、4项partial规则、8项B15兼容规则。全部PASS、0错误/失败/跳过，imports335→336、禁止操作0，2524份绑定来源前后稳定。result SHA `c8b41d340f980203d9e7aace79930dc7b3f86ddd0ae5fd0299e9abcd694db1a4`，unit result SHA `a085c7ed43d1771575dce71cc03534907f60c3a341930982c8f3affea4419a9c`，root实际外层退出SHA `f56d80b442c4b6efe1cd073c5ceaa3dabab2ccefce92a8c4503a06ac68b9f2ef`。复用运行器保留旧schema/scope文本；本次实际内容以精确test names和冻结plan为准，不是实际RPC握手。

随后首次真实只读partial审计actual exit0/chunk `550b42`，输出 `target/unified-quant-candidate-record-restore-audit-static-r1/r3/first-run/result.json`，SHA `0792400f9883c0e23c4bb2cd452d097c920cfcd4d8fde78a4866f35bc81e91a4`；root外部退出回执SHA `1fed72022b7c49ae2689d9e7623b9b142b271ca0d448cc37b92c871fb5b2ed03`。状态严格为 `PASS_PARTIAL_TERMINAL_EVIDENCE`：4898来源前后相同，440 raw pairs/858144 paired f32、88 decoded pairs、8 package byte pairs一致；实际共model_load/model_upload各17（16成功阶段加1失败）、warm_forward88、graph_replay176、Dual probe16。B15失败observer四derived均0，但底层未观测CUDA活动仍UNKNOWN。审计本身没有GPU、native child或新预约。

旧16-all-success selection工具仍不能直接执行。下一步是受控一次prepare和8个B15原exe CPU preflight；9-child runner/新环境与完整来源冻结正在静态准备，不重复16项unit，也不重做8份B11。尚无新B15 GPU登记、上传或额度；完整混合执行包、2004 selection、44组性能配对、独立确认与holdout仍未完成。正式binary与生产Worker保持原状。

### 8.55 B15 DUAL0 的八组真实 CPU 前置验证

在 §8.54 的实际16项CPU检查和partial审计之后，9-child控制器及显式环境启动层完成独立静态复审。新入口复用原 `Attempt`，只启动一个持有完整38键DUAL0环境的控制器；控制器仅执行一次prepare和八个原exe `execution-package preflight`，不重复unit或B11。启动层review SHA `f2f326143ed1d3a0875db5fec5960c2f4e1a8ef45ca75f38f63d4ad05b447bc5`；Python AST与PowerShell语法首次actual exit0/chunk `fa5670`。freezer首次actual exit0/chunk `dc6035`，生成 `target/unified-quant-b15-compatibility-cpu-r1/plan.json`，SHA `3a637a7666120d4312726d2113e44edae73c3e472b76da9973abec4b6f544196`。环境与原38键相比仅 `KATAGO_CUDA_DUALFFN` 从1改0，模型、recipe、配置文本与exe字节保持。

唯一实际执行session91222已正常终态：root观察actual exit0/chunk `351452`，outer PID81588、controller PID78760，owned controller实际wait0/64.7591928秒。一次prepare及八组B15 C3/C8 CPU preflight全部actual exit0；3277份控制来源前后稳定，两个实际环境检查通过。控制器result SHA `85d1f0f9add3fa65cd8b5c29e8d43db79acdc71c62e52532a460bfd2292d6070`，八例CPU结果SHA `a9c2bfc35848d09533468b94819836ad8eca913f15358934f1496b5e6122a57d`；prepared SHA `725f71da51dfd6cdcba5754c2e45c1791be65ce94354f7e35cad395ac016c66f`。外层result SHA `6b320bd57919b0b64de3f099d62d1f2ac4a49f47d86f657dc18cd489f3e4f1df`，root独立实际退出回执 `target/unified-quant-b15-compatibility-cpu-r1/tool-exit-observation.json` SHA `b37359d16201a5b3049d7926b186ad8ee3617575ff8fa9e3f2b713345aefd760`，同时绑定内外控制器与真实结果，不能把控制器自报替代tool退出。

这只证明新B15配置的实际CPU准备、模型解析与前置契约通过。没有GPU包构造、上传、Graph、数值、性能、selection或holdout认证，也没有新GPU额度。原8份B11成功包和B15失败永久claim保持；后续新B15接续必须继承原失败load/upload各1、1702.609秒已消费控制时间及更早preseal控制消耗，以真实新CPU回执和新环境派生Binding/profile，不能沿用旧DUAL1身份或重启旧入口。全目标尚未完成，正式binary与生产Worker未更换。

### 8.56 B15 有限接续的派生器、运行入口与十四项 CPU 检查

新的接续派生器位于 `target/unified-quant-b15-campaign-continuation-static-r1`，delivery SHA `f4f8e913114e54ca67cc9e42f2790f322a82acc32952034e4b1d909692591cf8`，entry SHA `f58a8871a78e9623469c6e4ff00bdec6309d8ed5dd8e2a7b3416eb0fa0b1559f`。它核真实8组CPU结果及内外退出链，复用原stage/storage/Binding工厂，以DUAL0派生新B15登记；旧8份B11和1失败/7未启动的原件保留。原失败load/upload各1显式计入：条件累计33次load/upload、528次模型执行/2016物理行、Dual probe16，exact/max attempts/journal上限分别3138/3170/3202。新八组模型执行额度未增加；仅失败setup需要显式增加各1次累计加载/上传预约，不能宣称累计限制仍是原32。控制时间扣原preseal 1秒及ceil(1702.609)=1703秒后剩27096秒，新八组进一步限定总14400秒。上述是源码中的待发布登记，不是已执行GPU消耗。

薄运行入口位于 `target/unified-quant-b15-campaign-launcher-static-r1`，delivery SHA `4e24facc7ba0a825c40b2197724c2d6f675eb15d41613dc74279f065a9e2cac7`。它只复用原launcher的group projection、single campaign、terminal检查、显式环境Attempt与清理边界，新增八组顺序、旧失败/时间继承和DUAL0检查；不调用旧16项execute。未来结果schema为 `rustgo-b15-continuation-controller-result-v1`，成功状态为 `EIGHT_B15_PROCESS_SEQUENCES_COMPLETE_NOT_PERFORMANCE_CERTIFICATION`，与旧16项终态分开。两份源码联合独审无确定阻断，review SHA `d4229c70f7f7d94d621b2f691079fec3b36d713066b7c51140c675b08ef045df`，32份有限来源稳定。

四文件首次AST实际exit0/chunk `428bb4`。复用旧run_once/check_tests与依赖闭包，新14项CPU计划SHA `68a87337d897623823f64dd1ae9c99cbe550e2d0a9a46caa26b358b909a8cd32`，freeze实际exit0/chunk `d50ae6`。唯一unit执行actual exit0/chunk `6e215a`，producer22444、owned venv47248实际wait0；派生器6项和新入口8项全部PASS，0错误/失败/跳过/禁止操作，imports333→336、2546来源前后稳定。result SHA `5f0e7565ececdaca8e4d0d007400ed19d972cff59381fd8fe1614b8b34da3a89`，unit result SHA `07659bb7aa4731420233d9a1672614c05e872a8f3cf92c1b804147f813362ac0`，独立root实际退出SHA `7108a5894f0f6609fd006927a2077f49ea343b9cc92243101a8faeb5784382c9`。入口为 `target/unified-quant-b15-continuation-tests-cpu-r1`，不得重跑已通过14项；复用运行器保留旧RPC schema文本，实际检查范围以冻结test names为准。

派生器完整derive尚未执行，八份新GPU登记/index与新launcher frozen-plan尚不存在。下一步仅一次有界CPU派生和实际结果读取，再冻结GPU运行计划，不能直接启动未冻结静态入口。后续selection逐case环境适配已交付静态helper，路径 `target/unified-quant-restored-selection-mixed-environment-static-r1`，delivery SHA `f76e4c0e454bd584effa966221a364caf7e7814398ab6b3ab69b4ab8e6794537`；10项CPU检查仅定义，尚未应用到完整assembler/runner或执行。新B15实际record/verify、组合selection、完整性能矩阵和holdout仍待完成。

### 8.57 B15 接续实际派生与新八组 GPU 首次启动

CPU 派生首次冻结实际exit0/chunk `73cfe8`；原入口 `target/unified-quant-b15-continuation-derive-cpu-r1` 的 session29070 最终actual exit1/chunk `6b6621`。子进程PID24516实际exit1/0.0704545秒，原因是派生器用 `base.key(output)` 的 `resolve(strict=True)` 检查按设计尚不存在的新输出目录。失败发生在输出创建前，index=null、3771来源前后稳定、GPU/launcher均0；result SHA `ad9d33a5acf583036a22bbbf6c68288fb66c2cab925a3cc03fa8739ae5b936d3`，root真实退出SHA `aead7dce350329eb0f111ce54ffe7661d47b8f574a8d91ca5a93ec4cf58c564e`。原尝试保留且不得重启。

隔离r2仅先调用已有 `base.future` 检查新目录，再比较规范化固定路径；原有文件Source的strict检查不变。新pins/index明确继承上次CPU失败，原GPU配方、模型、exe、预算和未创建的publication路径保持。deriver SHA `d92caa0b8ed95299a35fea61febe8e330f13f64e7d00eb26bcaf5bf6df224ee5`；窄审SHA `c4d2c54be2e2fcc5f3f8834b194dac333a06cc5c6e0a89a86fda747907da1142`。针对真实修改的两条语句，正确缺失路径接受、错误缺失路径拒绝、已有路径拒绝三项CPU检查全部PASS，3文件Python AST和PowerShell parser实际exit0/chunk `6bb569`；检查结果SHA `95482392413992fc0a6a0182d26d2df995a3f7b99c1f7d74de0a4843c892beaa`。未重跑原14项。

新CPU控制目录 `target/unified-quant-b15-continuation-derive-cpu-r2` 冻结actual exit0/chunk `655e10`，plan SHA `3b71a7f5827555329a44d93f6e07429a5dd4b5e6b3f30509bf5fa98b8afbad47`。唯一执行session50660最终actual exit0/chunk `8cd8e9`，controller72932/owned child59308，子进程实际exit0/22.5831467秒，总50.609秒；3771控制来源、5024产物及引用来源核验通过，GPU0。result SHA `0f4ad517bc4a4ea30d8bdd0d7d9e43eaad22661fc28ff41519d6353d745cbb17`，root退出SHA `d24a8accb37337ce40a1719c516d744ef6d65dbd818c93daf9444b969c623c30`。实际八组index位于 `target/unified-quant-b15-campaign-continuation-r1/continuation-index.json`，SHA `b4085991960021d414596dc2d1495b717876331b17916b51e01f59456585497a`。

GPU计划冻结同一session62565最终actual exit0/chunk `eae47e`；freezer只写计划，没有启动子进程或GPU。`target/unified-quant-b15-campaign-launcher-r1/frozen-plan.json` SHA `5530d8a8be88330806d0ba2bef9e64abca1409df4fa162a786aa56f98039c267`，5486绑定来源；freeze-result SHA `0e584cf9e33eadf5bf7f0069bc9adeee6b5049af59dfc6d3429b22138bb5fa2e`。保留旧8份B11、旧B15失败和CPU派生失败，仅新8组B15，16次加载/上传、88 warm+176 replay=264模型执行/1008物理行、Dual probe0、ABBA0，总限14400秒。加上旧失败后条件历史累计加载/上传各33，不把原失败当零消耗。

**八组B15 GPU验证已首次实际启动，尚无总终态。** session35450（初始chunk `4268a9`）、controller PID3196，首组 `b15-c3-mix-000` supervisor PID57152 已由实际Popen创建；实时状态须续看同一session及各组原件，禁止重启。GPU窗口内仅静态并行工作，不跑测试或构建。完整B15保存/恢复、B3/B8 selection、端到端完整控制矩阵、独立确认和holdout仍未通过；正式binary与生产Worker未替换。

后续同一session35450实际观察首组完整通过（chunk `23450a`）：supervisor PID57152 owned wait0/214.5561255秒，seal81628、record66476、derive_verify24164、verify44220四阶段均actual exit0。native verify为 `VERIFY_STAGE_COMPLETE_BITWISE_EQUAL`，`restore_verified`、`numeric_comparison_passed`、`all_handle_package_bytes_equal` 全true，`accuracy_or_performance_certified=false`；stage-result SHA `c982c49805d7ef84151af72e0a69322282032b33d7a417171d16b1c1bb2ccc58`。独立只读进度记录 `target/unified-quant-b15-campaign-launcher-r1/run/progress-01-observation.json` SHA `9b34e4dd82f2e40152f50e28817015b83d953b4fd9fe7fa0f5705036aa168cd5`；当前已进入第二组 `b15-c3-mix-001`，不能把1/8观察当作总退出或独立raw审计。最新交接仍为旧launcher目录的可更新 `ROOT-LIVE-STATE.json`，其 `b15_dual0_continuation` 单列新会话，不覆盖原失败终态。

GPU窗口内，逐case完整环境的selection CPU runner已实际写入新隔离目录 `target/unified-quant-restored-selection-mixed-cpu-static-r1`，delivery SHA `c9559ea557008dbb04c46e39db4814e90eaa0527b61c9d66c3689b99c800b52a`、entry SHA `a14552953126efa055ebfc011bcec32db2f93d9882a1be7339be23f14b7633f5`。仅17个CPU子进程（prepare+16 preflight），单位检查另绑定实际回执、不机械复跑旧10；6项纯CPU接线检查仅定义。与新assembler Prepared/CPU-results v2和真实旧unit回执字段的静态接口核对无确定阻断；完整assembler封存、有限新增unit与实际依赖/运行计划仍待完成。没有执行新selection CPU/GPU或新增ABBA。

### 8.58 双来源 selection 接线与 B15 审计静态交付

同一GPU session35450/controller3196继续运行，先后实际观察第二组 `b15-c3-mix-001` owned wait0/236.883992秒（chunk `cd6ba9`）、第三组 `b15-c3-mix-002` wait0/217.2202156秒（chunk `4d25e2`）、第四组 `b15-c3-mix-003` wait0/220.6415267秒（chunk `02482e`）。四份C3参考/候选native verify均为 `VERIFY_STAGE_COMPLETE_BITWISE_EQUAL`，restore/numeric/package-byte三门true、accuracy_or_performance_certified=false。四组只读元数据观察 `target/unified-quant-b15-campaign-launcher-r1/run/progress-04-observation.json` SHA `e92fbc3cd834958715e67eb329b500c8e3cbc2c0756315bcdb7e4587b64fbd15`；第四组stage SHA `dd6345faf2f46258a2c4fff8f90081791f48d0e71693b3506bcd6821e0375744`。当前4/8已观察，正在第五组 `b15-c8-mix-000`，没有总退出，不重启或在窗口内执行测试/构建。

完整selection assembler现已在 `target/unified-quant-restored-selection-mixed-assembler-static-r1` 源码封存，delivery SHA `10dd7bcbd664c5f44414702245c426e598036701a303769f638d17861522e872`、entry SHA `6a5fc698480b895eceee635cdca7ead4d3fe05003c487eb61a5f548128548652`。它实际接入每case的原registration、CPU receipt、campaign和完整环境；Prepared/CPU-results改为v2，producer环境与native case环境分开。prepare读取真实接续index与旧CPU来源；后续freeze必须接上旧8份B11真实部分成功链和新8份B15真实总终态/root退出、CPU派生链，不伪造原16全成功。`package_identity` 原完整检查段经root只读文本核对与旧版一致；正常Rust入口仍在采集GPU初始化前再次检查成功验证链。与 §8.57 的17-child CPU runner API完成静态核对；13项assembler与10项environment检查仅定义。

B15独立审计器在 `target/unified-quant-b15-record-restore-audit-static-r2` 封存，delivery SHA `0b9cf688f1539f07246d5ef964717541015cc21ae69c568bfa869d275cedd84c`、entry SHA `160c64b2f72e7e1915c2350330837f0b640fb22768e0389ca50ae715fb6776d4`。原r3成功case检查段文本保持；DUAL0只去掉能力probe并核两个启动上下文事件。新B15的五头raw、decoded、包字节、17类attempt及四阶段owned退出仍实际读取核对。旧B11精确880 raw+176 decoded来源委托给已有partial audit，同时交叉核新controller前后库存，单列匹配数，不冒称重新独立hash旧输出。r1 root PID可缺省的验收缺口由独审指出，r2仅改为必需整数且匹配实际terminal，并新增一项缺失/错误/boolean PID拒绝定义，合计7项；r1原件保留。独审review SHA `47bade87a9e6bfaf045ab76a93821c959ec3e813424183ca88baa6ebcad8b86f`，未执行审计或读取GPU在途数据。

首次有限CPU检查冻结入口位于 `target/unified-quant-restored-selection-mixed-tests-static-r1`，delivery SHA `2b305ddc6415010ee2fcb2086481f5cb7227b368863a0e201039877abd45ee9a`、freezer SHA `52a05b4971ca8d680303b6744d80b6efc5405114fa934b020f42c19f428c00dc`。固定13+10+6+7=36项首次定义，单一snapshot的8个Python源码；复用原已通过run_once/check_tests、旧14实际依赖库存和精确模块cache候选，不重跑旧14。静态独审review SHA `ee557e9b2a25edd24c630fe8ac8bf4c4dbb3b92cb4a49c5c46ca2c31742ec00e`。未来实际八源AST回执是freeze输入，GPU窗口内未执行AST、freeze或测试，`target/unified-quant-restored-selection-mixed-tests-cpu-r1` 尚未创建。本段源码就绪不等于36项通过、17真实CPU preflight通过、B15独立审计通过或selection/性能通过；后续17-child运行计划仍在静态准备。

### 8.59 B15 八组终态、独立审计与 selection CPU 接线

2026-09-28，同一 B15 session35450 已实际 exit0（`functions.write_stdin` chunk `42fe45`），controller PID3196 的八组全部完成，1969.468 秒，5487 份控制来源前后相同。总结果 SHA `0a008391d531b34b9de27793447f922fe0ff2236f8596c1f242b62f4d769ded9`，实际根退出回执 SHA `556bcf6d0cb33ad95db9a1cbad8c5a858abf35c1949ac6deca5effa7700de4cf`；不得重启原入口。所有八组正常保存／跨进程恢复逐位一致，旧 B11 八组与原 B15 已消费失败继续保留。

GPU 窗口结束后，八份 Python 源 AST 首次通过（receipt SHA `e61b8490cf40dd609a880bf23024229c9f22118b8370362e5ca07166da9b5c2e`）。首次 mixed36 检查实际 exit0/chunk `30511f`，13 assembler＋10 environment＋6 runner＋7 auditor 全 PASS，0 skip/forbidden；2589 份来源稳定，producer68784、owned child29832。目录 `target/unified-quant-restored-selection-mixed-tests-cpu-r1`，plan SHA `b0349aa34c1590f846df860959534dc0bcef181f76910e1c27b46e0e8ec1effa`，result SHA `c0e114f3cc97649655d2658456d77b53607d67e2c2b4f7200f661b8722694ca2`，根退出 SHA `56d596103d5fb7980bd69952688045fd1a4a398a904cb25fc6893d2ce65e5718`。没有重跑旧十四项或原 GPU 尝试。

B15 auditor r2 首轮独立实际 exit0（session55881/chunk `90d0ad`），报告 `target/unified-quant-b15-record-restore-audit-static-r2/first-run/result.json` SHA `a3439302ec85fdd8b324c22d5b9aa6693544371359cfe223e247006f1389fcba`，根退出 SHA `4dfafd99e1f5ed19352d28869a1a998951830ffa7aac7f88d9af3f26ba7e5ff8`。独立读取 5972 份来源前后稳定，440 对新 B15 五头原始输出／858144 个配对 f32、88 对解码结果、8 对包字节全部相同。旧 B11 精确 1056 份输出来源沿用既有独审，未重新读取；这与本次独立核验数量分开记录。新 B15 16 次模型上传、264 次模型执行／1008 物理行、DualFFN probe0、ABBA0；累计上传33包含旧失败各一次，旧未计量 CUDA 活动仍为 UNKNOWN。本结果仅证明保存／恢复一致性，不是 selection 精度或性能认证。

17-child CPU freezer 的最终源码经独审（delivery SHA `4d66aa2fc962a9873afef973709ce0cfc980f6fc408c48f74caf127014421ec1`，review SHA `c0258bf42fff1c4cceca03d0520860753ceebb1473d71a7a260ea8d9ed908d8f`），在实际36检查后首次冻结成功，session67051 exit0/chunk `3232b7`。实际新 plan 位于 `target/unified-quant-restored-selection-mixed-cpu-r1/plan.json`，SHA `337ba9b9569dd085c7286f6227efe9957fd79bf776dd01de24bbde2751745f74`，20271 份来源核对稳定。runner 与 prepare 同用实际 venv，prepare 显式完整 producer 环境，16 native preflight 分别继承真实 B11 DUAL1／B15 DUAL0 环境；总1200秒、每子120秒、清理保留30秒，不重跑unit、不触及GPU/账本。该计划已首次启动 session20024／producer68584，终态待观察，不得重启。后续 selection GPU 登记、完整性能控制矩阵、确认、holdout 和生产发布仍未完成。

### 8.60 Selection 十六项 CPU 终态与采集登记的路径兼容缺口

同一 session20024 已实际 exit0（chunk `f30e77`），一次 prepare／16 次原生 collection-preflight 共17个子进程全部正常退出，311.5秒，20271份控制来源和12298份核心输入来源复核通过。控制结果 `target/unified-quant-restored-selection-mixed-cpu-r1/run/result.json` SHA `d65f85f09d99c3b37f0a114cb4dbc95a55b6a2753014405cfb682e5be3896505`；实际 Prepared SHA `9d7f0353d8d848cb7600eea4917e03a524f91058cf53d85617b19cb7da6d55d4`，CPU-results SHA `e2ebd5556cc009b72f062784ea211a0d3eca774c573a0f0c08083240dbf9936c`，独立根退出 SHA `da7b515c1236cb38c008f28613dd48b7ec9effc5bee48cae0d9fdbe4c5f428b7`。根退出中的 plan_sha256 指向 Prepared，另列真实外层控制计划。GPU、collection、seal、账本修改均为0，不重跑这些预检。

采集登记的首次直接 CLI 调用错误使用 `-I`，顶层同目录 `per_case_environment` 导入被拒，freeze函数未进入，工具实际 exit1/chunk `7fad30`；失败原件 `target/unified-quant-restored-selection-freeze-invocation-r1/failure.json` SHA `89d7ca310d4dc1520fa57e84f6cf85814addc8c74eb71c0f6bed0f404e88e2c8`。保留原件并仅修为该既有CLI支持的 `-B` 后，session94549实际 exit1/chunk `15dcc1`，在 `restored_collection_parent.execution_projection` 拒绝 `preflight producer Source differs`。body失败回执 SHA `c31cf0bff9c310dcb031925b22055844826e6989f9a43ffb48d45e366e056a40`，预定 `mixed-registrations-r1` 输出不存在，没有新增GPU或账本预约。初步定位为原生receipt的 `\\?\D:\...` 与普通绝对路径在该Python父校验器中的等价比较缺口；修复须保留原始Source字节/SHA和旧Prepared生产者身份，不能改写原件或为修读取逻辑重做16项原生预检。当前仅准备单模块路径修复与新的freeze消费者，尚未完成登记或采集。

另有生产profile隔离静态增量 `target/unified-quant-production-profile-static-r1`（delivery SHA `efe362bdf1e559896c60d4a15aedb4a3fe8c46fd5800ed254c34050ae37e905d`，独审review SHA `151fd8fad2d8799eed3bde61586b18a913d912747321222465b71c173ad652a7`）：两份proto追加typed Hello/Welcome，Worker使用真实loaded-model profile，Server显式model/profile在Welcome及同ID替换前检查，pool/evaluator/Search生命周期内保持身份。8替换＋2新测试源、14项CPU检查仅定义；未应用、生成协议、构建、测试或部署。旧空profile兼容域不因此获得新的精度隔离保证；真正构建后仍须绑定新的exe/包与验收，不能迁移旧包SHA冒充。此增量不阻塞原exe的受控C32选型。

### 8.61 路径修复通过与十六份采集登记生成

父校验器仅修 Windows 普通／extended DOS／UNC 路径的等价规范化，保留原绝对路径、父目录穿越和已有链接拒绝规则。首轮七项 CPU 检查中六项通过，UNC 测试错误假定小写 namespace 可越过 Python 原有绝对路径门，actual exit1/chunk `63a3f8`；原件保留，仅修测试预期。修正后完整七项 actual exit0/chunk `ec9e60`，21份来源稳定，真实首例四对 Source 全通过。结果 `target/unified-quant-restored-parent-path-cpu-r2/run/result.json` SHA `436a03a53ac8bd3dfe0d90b5b96494a7b0399015224281c729866d671d96bb08`，根退出 SHA `88d2648742383649f664c565a863bc880293d1773d9972d300fe0fb0fc781885`；父实现 SHA `7ff61991e7fd9c2aa597a66af88b357d4d43d4cc88cb5287501f95c0f11aa460` 未因测试修正而变化。

新的 freeze-only 消费者保留旧 Prepared 的原生产者／pins，提前显式加载实际修复父模块；22个保留函数文本与原组装器一致。独立静态复审无范围内阻断（review SHA `f3cc57ceeb5f36fc3be1386a6e47df8c368bf56f690b11b237de84455df5d3b7`）。实际同一 session37895 正常 exit0/chunk `fcb204`，未重做16原生预检或执行包验证；`target/unified-quant-restored-selection-mixed-registrations-r1/freeze-result.json` 为 v3／`REGISTRATIONS_PREPARED_NOT_RESERVED_OR_LAUNCHED`，16份登记、19106项来源复核一致，结果 SHA `3409e2cb550d06bdfb8fd65f3392ab4203b7289e1f41840307748ef6ed9ae2aa`，独立根退出 SHA `e7c1ba921427dcde67ca17df5d5fbdf89e0089e66a79294cdc56f17b4059ebe7`。来源清单 SHA `40504252d2e3e894024bb7541ecbbe7635f68b5addba2c7f95a08d76afd9a5fe`；既有两次失败原件继续绑定。GPU、账本创建、预约和 ABBA 均为0。

后续固定2控制模板＋4 restored catalog的CPU运行器r1尚未执行，独审发现其将Windows venv启动器owned PID与实际Python PID强制相等，会误拒绝成功子进程；review SHA `c8af5ca817ae7901c11a4534d0880030a677642c6c5f8e0f5f88dcc498c7902d`。新r2正分别绑定实际owned wait退出和内层自报进程身份，四项有限CPU检查待实际运行。真实catalog、原根账本v4续接、selection数值、完整性能矩阵、确认及holdout仍未完成；没有新加速或生产采用结论。

### 8.62 六步目录首例环境检查失败与有限修复依据

六步运行器r2已在独审中修复venv启动器PID与实际Python PID误等同问题（review SHA `2aa963a70fe0487dec54e821f9f9995f6ad9569ffe96b23a00cefa6287c75012`）。原三项未执行定义加一项PID边界检查首次全部通过，session58828 actual exit0/chunk `b010b5`，2607来源稳定；结果 SHA `6b0902d08b9224b5a17fbe14a3bc8bcc763048423b4546ad8e4b86dbec1e42ed`，根退出 SHA `cfc5b767edeba48086ef13165d95ccfd8de7d1f78ac4a3ec641c0ef4539b3fdf`。实际六步计划首次冻结成功，SHA `5e36f23f20d7d6d16e8d048348d130e79f33e403dec8f14faa8c9528d92dffa8`，未重跑旧36项。

随后同一session54368 actual exit1/chunk `b74869`：首例catalog-00在进入原catalog／Protocol之前，因直接比较Windows进程环境与原冻结字典被拒；原stderr记录 `actual child environment differs`。finally又读取尚不存在的Protocol凭据，内层result错误字段被后者覆盖，两个原件均保留。19389来源稳定，目录发布0、Protocol0、GPU0、账本0；`target/unified-quant-selection-catalog-cpu-r1/run/result.json` SHA `31ec255ef9eeb122292ad5decca18f742d397bd686cddb3e9e2e68b7e03407e1`，根退出 SHA `861ff35fbea808a40790339076e179a2f92e94d40627a2ef88f354b66b6e7a08`。原control永久保留且不得重启。

一次独立有限CPU环境观察使用相同venv、原环境和原Attempt，actual exit0/chunk `cfa9d7`。两侧75键，19个原键大小写不同，按Windows键名规范化后完整相同；额外／缺失／值变化／大小写冲突均0，未输出环境值。`target/unified-quant-catalog-environment-probe-r1/run/observation.json` SHA `6c8cb0b5feb9c777328d82a6cc32ace784bf495c716a86fe0fbd7f6384b3413d`，根退出 SHA `e08b18ada3c3db2afd74ec86cb5178ceb7d1d1d4e9c318121d9fbbbad87d2398`。新r3仅准备规范化比较、独立保留首错和末核错误，Popen环境原件不改；新源和检查尚待完成。

v4单进程原API接续入口的三个首次注入API检查已全部通过，session94830 actual exit0/chunk `d187f8`，2606来源稳定；结果 `target/unified-quant-v4-entry-tests-cpu-r1/run/result.json` SHA `96e7d0804b689ada2e730512397189d39e979a6bf9fc17881c2d97a3d8345320`，根退出 SHA `801b5fca4c0aa63dd28a450b3a8279a42ed873b60c87c4dadd379ec8a93cf5f4`。这不是实际历史重放或v4账本发布；仍须完整实际六目录终态和独立根退出。没有新selection数值、性能或holdout结论。

### 8.63 环境修复实机越过与控制模板预算字段不兼容

r3环境与首错保留补充两项首次CPU检查实际通过，exec exit0/chunk `60de86`，2608来源稳定；结果 `target/unified-quant-selection-catalog-repair-tests-cpu-r1/run/result.json` SHA `0710fcecd916c4ef6b4fef3c3f33aa97db125a4a6579a0e58cff5faa90221f1f`，根退出 SHA `58c93f1b630a3b774346c9362913ea9703f51d7c9b225bfd39d59956ab1aea73`。旧四项不重跑。新六步CPU计划 SHA `88ef5c5bc65d3b50eee2148311776bb94f36b5fca4dc26ed00b15853f742ee41` 首次冻结成功。

同一session32297实际 exit1/chunk `bf9170`：首child正常越过环境比较和341项实际imports，进入原catalog CLI后，因原静态workload的 `max_performance_evaluations=11` 超出旧 `validate_workload` 上限8被拒。首错和Protocol收尾错误分别保留，证明r3没有再次覆盖第一错误。结果 `target/unified-quant-selection-catalog-cpu-r2/run/result.json` SHA `c14e97b7ef5c66031aa3f026f935544281ee4044e33127bc3a7f7321449c7e45`，独立根退出 SHA `a5d562ae8905033031904b0df4c6c5e71225b1e8e5581c4d0111668447d79e4a`；19404来源稳定、0目录发布／Protocol调用／GPU／账本，原control和已创建的空catalog父目录保留，禁止重启。

实际源码与search-plan的静态核对表明：旧B11控制模板需4个numeric／5个performance项，B15需5／8，均在旧上限内；restored四统一方案＋两旧控制则需6／11，其专用构造器允许仅两个预算字段与control不同。新r4因此须分别绑定合法control workload和原restored(6,11) workload；不能放宽旧validator或删掉完整11组比较。该源派生尚在进行，仍未执行真实六目录成功流程。

v4有界CPU控制准备器已静态封存并独审（delivery SHA `ae435689a3856cda4d02346762143d4d9dc8f94a21c584bb8e786437548b5489`，review SHA `578d82341d7c14034e861ca08552e2cdc9db4f4fac2f0a2a954b514a280ca3d8`），等待真实successor六目录及根退出。它复用三个实际CPU检查、原Attempt和单个原v4入口；未执行freeze、history或账本。后续driver的独立_child仍需明确传递新parent来源，顶层preload本身不解决子进程接线，尚无GPU驱动或新ABBA。

### 8.64 控制预算检查通过，实际目录暴露旧配置相对路径基底问题

r4 分离控制模板与 restored 工作量的两项 CPU 检查首次实际 PASS，工具 exit0（chunk `b40191`），2,611 份来源前后相同。结果 `target/unified-quant-selection-catalog-workload-tests-cpu-r1/run/result.json` SHA `bb8c072113299dc701fca3d636e16bdd1c30325ad2333ef9141e1dc57ab48458`，根退出 SHA `1769018a8e350ae25aadb83af8c8ad87adfebb0d7d52e1290a5d96f859cc246a`。只运行一项受影响检查和一项新增预算检查，旧四项及环境两项不重跑。

实际六步计划 SHA `a39276364c249d00c82e8d495e3686b19a76962f64abf55cfda8e371e3bc35a3` 随后首次执行，session20922／producer47904 actual exit1（chunk `a63ca9`）。首个 B11 ordinary catalog 已越过环境与预算门，并完成一次 Protocol 生成，随后拒绝 `source config does not reference the declared original plan`。19,427 份来源前后一致；0 目录发布、0 账本、0 GPU、0 ABBA。结果 SHA `cf8880540cd130a2b0179a3f42324231c7b9bfb9fd3bcef12dd4864f1b61e475`，根退出 SHA `68eb80f7089287f612b82ff5261c2229a165e0610c90c6b9c3deb703fb1b4bcd`；三次失败的控制目录和永久原件均保留，禁止重启。

只读诊断确认：B11 原配置仍为 SHA `f1367abcc4e156a3bd2466e5c0f1a1a1ae71c17888a1ac4594413ba63c759874`，原计划仍为 `e132a249949488de8eb3ae8afbc98f30ac89065397797c6a11acf060e7bf5d7f`，两者与原 spec 一致。相对值 `plans/worker-tf3-sm120-c32.json` 被复制到 snapshot 的 `legacy_quantization_execution.py` 按其 `__file__` 推导的 snapshot ROOT 解析，造成位置误判；不是配置内容改变，也不是 Windows 路径别名。原根与 snapshot 的该模块字节相同，SHA `fd8570e746764c1ee7d80eb3819beef8707070e5302ee14b283191d27b7298f4`。后续修复须显式绑定旧模块的真实原根来源，并检查 catalog、v4、driver 与 `_child` 的导入闭包；不能修改旧 config/spec 或全局替换 actual96 的 ROOT/proto。

driver 最薄入口已静态封存并独审（delivery `72fb4ecd…`／review `4d60cdde…`），四项检查仅定义，原 parent 预加载和逐组账本 head 继承已有源码；新 legacy 来源接线及全流程时限尚待集成。v4 尚未发布，selection GPU、性能、holdout 与生产部署仍未完成。

### 8.65 原根路径检查与六目录实际通过，driver 接线检查通过

原根 legacy 两项首次 CPU 检查 actual exit0/chunk `d649a0`，2,625 份来源前后稳定。真实调用原 load_spec，核对 B11 相对计划与 B15 绝对计划及其模型、binary、配置、计划 SHA；负例拒绝同字节 snapshot 路径、错误来源和已占用 canonical 模块。结果 `target/unified-quant-selection-catalog-legacy-tests-cpu-r1/run/result.json` SHA `4c6fe231e02cbcbb559cf311c54bc437662303966659725c3128634484bc846e`，独立根退出 SHA `54a57d459e5447e0a98ddd54affa9dc236c293249e8f06d2143c262e0518976d`。

r5 目录运行器独审通过（review SHA `8d2807244bb46294a1488fcd86c29e5cfc42e1537e80ed2b413ba753c925afb3`），首次冻结新计划 SHA `359b522f95b55cbd01bbd228267acdba74b07d8b8948361a4cb9d3fbfb649dd7`。同一 session45554／producer60672 实际 exit0/chunk `24997d`，六个串行 child 全部 actual exit0；19,447 份来源前后相同。结果 `target/unified-quant-selection-catalog-cpu-r4/run/result.json` SHA `a1ca3bff964385f553ac66e3ff56757e7330a3b103ef7557d332a886b6718011`，根退出 SHA `bc61af3f3eb288989f32984d773c1b329d3f00c9c30ae697120de72ec8983e6d`。两个旧控制目录和四个 restored 目录已实际生成；四组仍各 2,004 个 selection 局面、6 numeric／11 performance，未缩减完整矩阵。目录生成的 GPU、ledger、ABBA 均为0，三次旧失败完整继承且不重启。

driver entry r2 已封存（delivery SHA `a41d22bbde5ddf0c31a5283042fb56ccc7b7cb09485005c94ea4a1c5fc9b4318`）。它显式安装原根 legacy，仅对旧采集器脚本 argv 和对应 digest 作投影，保留原 driver 完整契约验证；六个指定 snapshot 依赖预加载用于保留实际协议及其它模块 ROOT。独审发现旧外层 controller 仅接受 delivery v1、会拒绝该 v2，后续新控制器必须明确接通；旧封存件不修改。

该入口十项首次 CPU 检查已 actual exit0/chunk `dd0021`，2,641 份来源稳定，实际 imports 334→416，10 PASS／0 errors／0 skips／0 forbidden。检查包括原 B11/B15 配置读取、命令投影、根退出约束、环境比较与 late import 来源，不执行完整 bootstrap／driver main／Protocol／history／ledger／GPU。结果 `target/unified-quant-restored-selection-driver-entry-tests-cpu-r1/run/result.json` SHA `2ce179a2cc13c37df6a472d3b746d308532dd805d9910a4f07ba90adee9fd1f0`，根退出 SHA `593e12a9400c66c8c215639a09810e0a1085f045580742839e892957c6a241f8`。

v4 entry/control r2 独审通过（review SHA `dca282c1921b7d47c3ecfc01ca9a26a1370308b2dde762fa01cf582d0b8c7a8d`）。真实六目录及其根退出已被首次 freezer 消费，计划 `target/unified-quant-restored-selection-v4-publish-cpu-r1/plan.json` SHA `5bddc1284cc068d65c0e92961a8e0e45286ffa8d879496edb6145e5e07c993d0`。首次启动 session71940／producer7892，初始工具 chunk `9676f3`；此处尚未观察总退出，必须续看原 session，不能重启或据计划声称账本成功。CPU 历史与发布正在同一操作中进行，无 GPU／collection／ABBA 启动。

外层 GPU driver 的完整进程树 containment 尚未实测通过。新的 base-Python gate helper／Windows Job 方案正在隔离源码中实现，必须在放行前纳入 Job、验证嵌套后代以及超时后 ActiveProcesses 归零，不能以只回收 venv redirector 或 UNKNOWN 字段声称完整清理。selection 数值、性能、独立确认、holdout 与生产采用仍未完成。

## 8.66 v4 实际发布终态与进程树 CPU 首次失败（2026-09-30）

§8.65 的同一 v4 会话 session71940 已实际 exit0／chunk `f4434e`，producer7892，耗时 1675.343s，2,606 份外层来源前后稳定。`target/unified-quant-restored-selection-v4-publish-cpu-r1/run/result.json` SHA `3e7dc0d1d5b563513afa5fe955c7a1de4dc513245f30c655466dbe83ddd8aeb6`；根工具退出回执 SHA `881aae5ce452fdffce5157a4f51e07a143617cc59b3a1425fa527485c9ff17cc`。发布结果 `target/unified-quant-restored-selection-v4-control-r1/result.json` SHA `6b76f60ddf14c8bcb5d44393d444eee5055b9e3d553bc94c8be4f8fee307e747`，状态 `V4_REGISTERED_NO_COLLECTION_LAUNCHED`；实际历史 v3／v2 两个已观察 child 均 exit0，不以此声称整个后代树已取得退出资格。

实际 active directory 为原 ledger 的 `continuations/restored-selection-v4`，contract SHA `110e92e2d0ee25d92f885abc2182edb78e03f82e8b688aa23112775cf00dfbd8`。新事件文件 `events/000066.json` SHA `5ce80c385ae60ba30e65e1c6194100863775cab2bed50f354571e80ff88cfb05`，语义 head SHA `cb54af140c21dfc0ff6ac194d8d7abb086aabadec145e8835c973c19d351f04f`，两者不混用。67 个事件继承旧 consumed COLLECTION10／ABBA16，remaining COLLECTION20／ABBA0，pending reservation0。四个新 restored catalogs 已登记，status 为 `CPU_EXECUTION_REGISTRY_ONLY`、performance 为 `DEFERRED_RPC_DEPLOYMENT`；无新 GPU、collection 或 ABBA。该操作已终态，不重放 history、不重跑发布、不修改 contract/head。

外层控制 r2 已封存，delivery SHA `af19201264b66849ad0d745545c9030a471e92ed01bbb9d5fcdba6459dc777d2`。它接通 driver entry v2，以 base Python gate 和不可继承的 Windows Job 句柄处理子树，源码审查不等同运行资格。首次有限 CPU 计划 `target/unified-quant-selection-containment-tests-cpu-r1/frozen-plan.json` SHA `ba374959e7051e45d3b34735082acd90293b0c4bb69ff07fc1abac72d665fbae`，session39597／producer70148 实际 exit1／chunk `ed019b`。结果 SHA `e279b6782f22c981f0d0e241c35f7b3179a6200104d40162fc98f68a04a548dc`，根退出 SHA `6c16570c1a2b64f367c89830dbc281e496ce90755d386f337665824f21c16717`，保持 `FAIL_STOPPED_NO_RETRY`。

12 项实际执行：5 项纯检查和前 6 项真实检查局部 ok，最后 EOF 检查将 Job `TotalProcesses` 预期为1却观察为2，聚合失败。EOF owned PID62296 实际 wait1，stderr 停在 gate 空输入检查、尚未进入 Popen；该分支没有独立 helper self-PID 回执。outer 最终 ActiveProcesses0／TotalProcesses40；记录中的23个不同 PID 不是完整进程普查，不能据此证明真实进程总数≤23，也不能把40解释为40个新 OS 进程或推导重复计数公式。breakaway 创建实际成功、叶 PID77292 完成，但旧检查没有直接以叶进程句柄验证具体 Job 成员关系，因此此前局部 ok 不构成完整 containment 资格。原 `sources_unchanged=false` 保留；独立根观察确认前后两份2,671来源列表内容相同，不能用该相同值改写聚合失败。

有限现场证据为 `target/unified-quant-selection-containment-failure-evidence-r1/report.md` SHA `e515a52e80b13d30919b6f7683f61d3ee303ab0ff9ac417b667f6d1af0614deb`，66 个有限原件前后稳定；另只读诊断 `target/unified-quant-selection-containment-diagnosis-r1/diagnosis.md` SHA `fdf34b346af176bcac52463e5a82a2435a5c6b5e2ccc6f0b0f7e0a4f46afb55b`。后续仅在新隔离两 case 诊断中增加有屏障的 ProcessIdList、精确进程句柄及出生身份、outer／inner成员检查与最终 empty／Active0；未归属 Job 的 owned leaf 也必须用原 Popen 句柄回收。诊断尚未实际执行；不修改原 r2／23 门、不重启失败 CPU、不启动生产 driver 或 GPU。selection、性能、独立确认及 holdout 继续未完成。

## 8.67 两项直接成员诊断的实际结果（2026-09-30）

独立诊断包 `target/unified-quant-selection-containment-membership-static-r1/delivery.json` SHA `870dc8291c7125516675cbd9a5ccbcc51ec6debabae53573b523e4d2cb3571b4` 已封存，8 份交付文件及4个Python语法经 root 核验；独立静态复审21份有限来源稳定，报告 SHA `e057765270a01ea22d43aeedf33f08e4c0fb60f638d1f29fc0ebea17ffbc4e36`。freezer 首次 actual exit0／chunk `c06d0e`，新计划 `target/unified-quant-selection-membership-diagnostics-cpu-r1/frozen-plan.json` SHA `c891ddaceb17f28694e2675d1ee37107fb2f995394342fa96337d948f3e9caad`，只含 EOF、breakaway 两项，90 秒总限，原 Job／gate 字节不变。

随后首次 direct actual exit1／chunk `52b6f3`，producer72608，2,751 份来源前后稳定。结果 SHA `4c58d1f2bcf45b742f1d6260413203a2374f16a2502f5fc7954f1a57f9621ab9`；根退出 `run/tool-exit-observation.json` SHA `484728b26a0893312f23c85f903ca656149d8cced7c8dfc16a33905f9ec8203d`，保持 `FAIL_STOPPED_NO_RETRY`，无异步在途会话、禁止重启。

EOF helper42100 实际 exit1，stderr 命中原空 gate 拒绝且没有 helper-result，两个 Job 在 gate 前的完整列表均只有该 held helper；终态均完整列表为空／ActiveProcesses0，局部 `passed=true`，case SHA `effd46cffcebc0d0d510a1d87feffb84f060a2ea4c7b45ab5f8897b739a29843`。最终 TotalProcesses2 仍只保留原始观察，不能据此解释第二个进程身份。

breakaway 的 held helper43880、redirector84428、fixture54756、leaf37660 都已取得精确句柄及出生／镜像身份；四者在 outer、inner 两 Job 的 IsProcessInJob 均实际为 true。但是，两份未截断 ProcessIdList 均为七个不同 PID，ActiveProcesses 同为7，额外成员60888、65496、67028尚无镜像／出生身份。此处是实际活跃成员集合差额，不能解释成单纯 TotalProcesses 重复显示，也不能猜测为 conhost 或其它系统辅助进程。原“最多5新进程”的显式启动拓扑未涵盖所有实际 Job 成员，不能作为完整 OS 进程总数上限。

由于集合不符，释放前停止并执行清理：leaf 精确 duplicate handle 和 fixture 原 Popen 均实际 wait3758096386，fixture cleanup_error=null；原 helper cleanup wait3758096385、actual normal exit仍为null，不补正常退出。两 Job 最终完整列表为空／ActiveProcesses0，cleanup_errors为空，watchdog未触发。该实际负结果及 EOF 局部通过均保留；生产控制器不因此取得资格。

后续仅在新隔离 r2 中进行一次 breakaway 观察：先保存所有实际成员的精确句柄、出生时间、镜像及可验证父关系，再判定集合。EOF不重跑，原两项失败入口和23门均不改；新诊断的显式Python创建次数与可能的系统成员数量必须分别记录，不能用前者代替后者。该新观察尚未运行，原账本20次采集／0 ABBA仍未消费，selection、性能与holdout继续未完成。

## 8.68 单项成员身份诊断实际通过（2026-09-30）

新 `target/unified-quant-selection-containment-membership-static-r2/delivery.json` SHA `c85ae930029b33444feb5e542b9c5427d7e94ef476d32f5245389a0212ec3aea` 仅定义一次 breakaway，原 fixture／Job／gate 字节不变，EOF只读继承。90秒总限／15秒工作窗保持；显式Python创建次数4与完整Job成员查询容量16分开记录，后者不冒充累计进程额度。root核对8份交付和4个Python语法；独立静态复审24份有限来源稳定，报告 SHA `db5921452db51befdd97ae489114a5aa5cf01a20f4544751e8f44e216bfb23ba`。

freezer 首次 actual exit0／chunk `ce5d26`，计划 `target/unified-quant-selection-membership-diagnostics-cpu-r2/frozen-plan.json` SHA `6b439a99d63a5c8255fe96e7a434b5d126ca732230068197585873d41204da7f`。随后单项首次 direct actual exit0／chunk `2f3553`，producer13336，2,786份来源稳定。`run/result.json` SHA `27bbe89b21c3c5d13684775e8a9968e63cade3b57a400da4f02661dbc177658a`，状态 `BREAKAWAY_ALL_MEMBER_IDENTITIES_OBSERVED`；根工具退出回执 SHA `62d3d59bab416793ef6c158d5fc325b55f0b74d0b63ba6c1cfcfd288d4e38c62`，case SHA `2485f6ad78acbb1a2cc259b4036b0e7c0e79df4d30a290ae7bdf130528d693a9`。已终态，不重启；无GPU、Worker或账本操作。

本次七个完整活跃成员由四个Python角色和三个 `C:\Windows\System32\conhost.exe` 组成。三个extra PID9128／52044／58404已取得持有句柄、出生时间与镜像Source（1,015,808B，SHA `a29c38fe5566650658c1d6377af7f5115d7c54f3af7de0a10e867db25cb68f9f`）；Toolhelp报告父PID分别对应redirector84348／helper51332／leaf78332，父句柄身份和出生先后核对通过。七个成员在outer、inner的逐句柄成员查询均为true，身份观测前后完整列表相同；身份及父记录均无缺失。镜像来源末核未变。该证据限于本次，不能给§8.67未记录身份的三个旧PID补写conhost身份，也不能将更早TotalProcesses40推导成完整唯一进程普查。

原helper51332和owned venv84348均正常exit0；fixture68948的原Popen和root duplicate对leaf78332均实际wait0。两个Job正常及清理末态完整列表为空、ActiveProcesses0，无watchdog或cleanup error。EOF沿用§8.67实际局部PASS而未重跑。新诊断明确保留 `production_control_qualified=false`：下一步仅修control r3资格消费者，按具体来源继承原pure5、normal／orphan／nested-timeout／assign／expired局部证据，加上实际EOF与完整breakaway证据；不改写旧聚合FAIL、不重跑旧real7或模型验证。

首组生产plan freezer和r3资格消费者尚在静态准备，未实际准备或运行selection。v4原登记20次采集／0 ABBA继续未消费；完整性能矩阵、独立确认、holdout及生产采用仍未完成。

## 8.69 控制器资格消费者四项实际通过（2026-09-30）

最小控制器 `target/unified-quant-selection-driver-control-static-r3/delivery.json` SHA `3cd061b6f999a213084e5d25500c4accb75b5b369b03796425d2a590357aa582` 已封存。资格消费者按来源分别继承旧局部成功、r1 EOF 和 r2 完整成员观察；旧两份聚合失败及未知 PID 保持原样。原 `contained_attempt.py`、`gate_helper.py`、预算／总钟／owned-wait 段和从总钟构造到末尾的执行段均未改写。独立静态审查报告 SHA `e63f820ef9cc0f5c0d006b475c2a3ad7e983eaf1d2445f659d034657c22e973e`，160 份有限来源前后相同；146 项资格输入全部位于152项作者来源清单中。

root 核对7份交付、152份来源及3份Python AST后，pure4 freezer 首次 actual exit0／chunk `f26074`。新计划 `target/unified-quant-selection-driver-qualification-tests-cpu-r1/frozen-plan.json` SHA `143d1f11978d82a136c3f8fef4e10b1f35c38f603e138e4fdb3c761b02bc4130`；随后首次直接运行 actual exit0／chunk `b458e8`，producer66624，唯一unit子进程实际wait0／0.627258秒。四项全部PASS，0错误／失败／跳过／禁止操作，imports334→334，2,748份绑定来源前后一致。结果 SHA `98463891c23a3effa3c837120872fc104896b4815097e1820644eea8f66ca922`，unit SHA `75dfd6428b22455cfa6739e822eb17a17d43241135213e87c849528d07583f26`，独立根工具退出 SHA `3797c4fe0b9536359e0e5d39d6fbfafd90c6762b6ab6f3a3df2306cd339a8f61`。该入口已终态，不重跑。

前三项读取固定实际证据并在内存副本中验证错误拒绝，第四项验证新四检查自身的root／result／unit／runner／test来源合同。没有新Job或模型运行、Worker、GPU、账本预约。首组plan freezer正在将实际pure4、v4及六目录绑定到原16份registration锁定的driver输出根；四组时限仍需首次执行前明确固定，沿用同一总单调时钟。原pre-release ActiveProcesses==1门保持，已有观察不能排除其更早出现辅助进程时的严格拒绝。完整B3/B8选型、性能矩阵、独立确认和holdout仍未完成。

## 8.70 首组完整数值选型首次启动（2026-09-30）

首组freezer已封存，delivery SHA `5abdc19d555646df267c9884050171306ff45e18190d6bc0805a51ff43a2645a`，源码 SHA `0eedd99a6c4f715e60b9398737da04da6e8e5126ceab2b9e4a3c6a7e206d3f21`。root核对7份交付、48份有限来源并通过AST；独立静态报告 SHA `4a124bf00b4dc607f3878ebae4c93295c5db311d97da8c8b77684aeb5f5bb515`、最终交付绑定 SHA `b47ada4b81803f4f2717d765ebf1da0b2b66f61bec7f408262a15bb9b91c643b`。16份原registration逐项锁定四组driver根、execution/attempt身份与完整2004行。B3每份668次提交；B8每份250个B8加一个B4尾批。旧控制按原C32连续RPC请求采集，实际物理分批不能由其上限推断。

freezer首次session32450 actual exit0／chunk `e09693`，`target/unified-quant-selection-b11-c3-driver-r1/frozen-plan.json` SHA `4f40b8b801cb7f3e1b933c994ee98381b73b7bd889e4a3a3375999c735d09061`。15,344份输入来源前后相同，完整control来源15,349、runtime pins2,929；freeze-result SHA `7f56a39f0f5aed438bf3ffbf788e99e9ef21cd2e5e5f842b98c3b59d15f82864`，独立freeze工具退出 SHA `19dc813b8f8b2c0271349a50f61d8bbd60b0ecf8ce95ef50d9c617c8c39e8c5f`。freezer没有运行history／账本API／GPU或创建原driver根。

root明确固定history CPU上限1800秒，四组运营上限28800／21600／28800／21600秒，总100800秒（28小时）。实际已知历史只读owned调用334.9452773秒、v4发布总1675.343秒；缓存命中单操作和新完整集合耗时仍未知。这些限时不作运行预测，原restored1200秒／legacy362010秒命令上限保持。总单调时钟从第一控制器进入开始，包含后续组间间隔，不重置。

随后首组B11/C3唯一首次启动于session28076，初始chunk `b3ddd4`；精确命令查询chunk `031940`确认producer23464。原登记driver根为 `target/unified-quant-restored-selection-mixed-prepared-r1/driver-roots/b11-c3`，独立control根为上述新目录。当前尚无终态，只继续观察同一session，不重启、不补写成功；实际采集预约与输出以原账本和driver原件为准。预登记新采集投影6／4／6／4共20，ABBA为0。窗口内不运行测试或构建，仅静态推进后组续接freezer与性能来源投影。完整数值选型、44对性能矩阵、独立确认、holdout及生产采用仍未完成。

同一session后续仍live（chunk `298d61`）。控制器已保存sources-before、imports-before、intent及prepare-contained；后者helper50716在放行前已分配Job、Active1。精确命令查询chunk `1f1e69`观察到prepare owned venv21076及其实际inner73144，均执行原 `prepare-execution` 入口。该进程观察不是退出证据；当前仅到准备阶段，尚无prepare或整组终态。

## 8.71 原首组在途观察与后续静态接线（2026-09-30）

原session28076仍live（本阶段最后工具poll chunk `c9c503`）。prepare历史重放回执 `target/unified-quant-selection-b11-c3-driver-r1/run/history-prepare/replay-receipt.json` 为242,781B，SHA `8dda9ff8af14e46e7a3156f99398d4f0c230cebfc40d73e1b446063562ba5ed1`；其中v3 owned16652、v2 owned71148均实际exit0，cleanup exit为空、error为空。该记录只证明这两个历史子进程正常等待完成，不是prepare或整组终态；继续同一入口，不能据此重启或补写root exit0。

后续三组逐组freezer静态交付位于 `target/unified-quant-selection-driver-next-plan-freeze-static-r1`，delivery SHA `b569112acdb41ae9068303a9c8711e902da2e52630c50e264fe9774d2532b779`，源码 SHA `3e2a9180144fd1cb622adb0aff62901f1b8a3b2cf6172ef2d00d34f09468f60c`；独立报告 SHA `d38c105c4b40a3d2a5f98125ffe0de43de586033ef6eb44dc6b8a726812c54b7`，20份有限来源稳定。它由即时前组实际root反查control／Prepared／driver／gate／head链，继承同一100800秒总deadline及四组上限，保持原登记driver根；数值FAIL作为原结果保留，不逼候选变成PASS或重采。每个新control根必须选target下独立目录，避开所有旧driver根。尚未AST、import或freeze，没有下一组启动。

性能来源投影位于 `target/unified-quant-restored-performance-sources-static-r1`，delivery SHA `0fe9b648a6d8ed27561473de77bd3fca4d5ea42882f1a1d2ee2c0471d8bb2ac0`，126份作者输入来源前后相同。基于原96文件清单，94复制／2替换／2新增形成完整98文件的显式映射；没有扫描或改写旧运行树。producer与reader通过同一helper绑定具名实际Source表，保留原parent修复Source和原根legacy Source。已有执行与判定主体三个原文段保持一致。独立报告 SHA `9acbbff041ca5311029eaa31c570285552bc204e355189af1e8d481db455572a`，29份有限来源稳定、12份交付匹配。

新增四项Source检查仅定义，使用合成canonical模块身份和真实文件Source；不实际导入业务parent／legacy，不调用installer，不能证明真实bootstrap、ABBA reader或性能。root有限CPU freezer源码为10,564B／SHA `326e1c4be823783f7ca98fb48867730fe3bd2d1d895bdce48a39d79c89f88b52`，静态交付 SHA `5564f919d9d809a81b7cf9218757ca447c3f6491d37218914314e7268678ca32`。它拟复用既已PASS的180／30／30秒harness，完整物化98源后只运行这四项；当前未AST、import、freeze或测试，须等现有执行窗口结束。

v5仅性能continuation正以独立新源准备：未来必须绑定四组实际终态及最终v4 head，复用原父replay、`execution_qualification`和ABBA判定函数；原v4继续拒绝ABBA。计划只新增0 COLLECTION及完整44 pair的有限ABBA白名单，数值不合格者保留UNQUALIFIED且不预约。未来父段永久seal沿用既有continuation协议，现未实际创建任何seal或改账本。现有四份C32旧控制在数值阶段完成后应复用，不因性能接线另采。当前仍无新增ABBA登记或实测性能结论。

同一session28076随后完成prepare：`run/prepare-exit.json` 为2599B／SHA `2c5202be28029dc6ecbe70249a4b0537e73772a3f1ebe4b37c25cfeb05c28e16`，helper50716及owned venv21076均实际exit0，1120.469秒，failure／cleanup／Job close error为空，watchdog未触发，完整子树退出标记为true且Active0。raw TotalProcesses9仅原样保留，不解释为唯一OS进程普查。原登记driver根已生成2,665,897B的 `prepared-execution.json`，SHA `b86616f75c6e078e3feeb9dddd70a8664cd0e2ca24a33d55093030de2b7dad5b`；6条原命令／6份采集、0 ABBA，完整11对性能保持DEFERRED。初始逻辑head仍为 `cb54af140c21dfc0ff6ac194d8d7abb086aabadec145e8835c973c19d351f04f`。

20:10:05本地时间，同一controller沿原总钟进入 `run-execution`，owned wait上限27476.203秒。精确命令查询chunk `c605fc`观察run helper55156、owned venv47808及实际inner47840；该run进程另有自己的前置history核验。主producer23464／session28076仍未终态，尚无数值结果，不能把prepare PASS或运行态写成整组成功。最新状态已记入ROOT-LIVE-STATE。

## 8.72 首组运行核验通过与性能矩阵调用接线（2026-09-30）

原 `session28076` 在本轮实际轮询 `dacaf6`、`c77af3` 均仍live，不重启。run阶段只读历史回执 `run/history-run/replay-receipt.json` 为242618B／`bb7c631ab1fecaa11450aa54f443a61eab00803ae1f29a8c9dad80bf435a899b`，其owned redirector60908实际exit0、error=null；这只证明该历史子步骤，不证明主driver退出。精确进程查询仍见producer23464、gate helper55156、venv47808、inner47840。原driver已写 `events/0000.json`（4751B／`ca3fae75a80694b427cf1cbfbdbc11f0df44ab92a8efc5f939244e1bfc938eb6`），事件为 `RUN_STARTED`；本轮末观察事件数仍1，无driver/controller终态或采集完成回执。

root观察位于 `target/unified-quant-selection-b11-c3-driver-r1/run-history-observation-02.json`，2073B／`431676329f913b148928eb34aba16cba0a0cbc947f0b25caa1fdba2f6fe2b6a5`。原01的 `held_process_ids_observed` 命名已在02明确更正为精确PID查询观察，不声称root持有这些进程句柄；01原件保留。真实owned子进程退出证据仍来自绑定回执。ROOT-LIVE-STATE已指向02，不能据缺少结果推断失败或重新启动。

性能专用续接库 `target/unified-quant-restored-performance-continuation-static-r1` 已静态封存，delivery为11576B／`39f14ec33234255885a6525e87422987aa6d32aea67e553df109ddfb9f1c2386`，来源回执13071B／`d66e85eb4aa504c601b813bf752cda6cea5fa05ade943de4e6f2bc1cf59e3f5c`。6份Python源码、12份产物、24个有限外部来源在封存窗前后稳定；没有重读大库存或执行代码。库要求完整四组实际终态与v4末head，继承全部原消费和负结果；30 COLLECTION／16 ABBA是未来必须核实的父终态条件，不是当前计数。新段仅允许0 COLLECTION／44 ABBA白名单，原 `execution_qualification` 决定ELIGIBLE／UNQUALIFIED，后者保留但不预约；原v4保持只数值消费。旧96模块guard覆盖父重放完整生命周期，实际临时模块的执行来源被追踪；新98来源树由同一producer/reader投影读取。调用方的1～16个显式Python `caller_sources` 同时必须进入runtime库存与v5工具合同，避免来源遗漏。10项policy/loader检查仅定义，尚未AST、导入、freeze或执行，不能视为真实bootstrap或性能资格。

新增 `target/unified-quant-restored-performance-driver-static-r1/run_matrix.py` 为静态库草稿，尚未封存或执行。它在同一owned venv及活跃 `entry.session` 内调用原canonical `benchmark_quantization_corpus.main()`，只供应原CLI参数和显式 `--restored-package-runtime`，保留原数值读取、ABBA计算、结果读取及完整四目录summary。全部44格按原顺序保留，直接复用已有数值目录；每个合格组合先完成持久预约和日志事件，再进入main。main返回值明确记录为函数返回，不伪称独立进程OS退出；外层实际OS退出、Job清理和library scope completion仍是必须的终态证据。失败原件／预约／run lock保留，不重试；不新增数值采集或holdout读取。4项新adapter检查仅定义。

后续caller静态设计收敛为两次owned venv：首次显式prepare在同一library session中派生preregistration、保存外部audit、发布已由root冻结授权的v5段并准备矩阵，不启动ABBA；第二次由root另行启动run。两阶段共享首次绝对总时限并包含间隔，复用已有Job/gate原件，避免四次重复历史重放。完整进程／工作／时限预算、新入口实际CPU资格、四组选型实际终态均仍是前置条件；当前没有v5登记、真实锁或新ABBA消费。生产binary与Worker不变，完整性能、独立确认及holdout仍未完成。

本节续更：v5库独审已封于 `target/unified-quant-restored-performance-continuation-review-r1`，review3332B／`ba29b80cc3a19cf2fc70a2ae57ec929df00ac257d3bcff1737978cfd625a949f`、evidence11548B／`d7410a4c6155260905191aa5c4abc826712cedb7431f68ab264d4e362e7147ed`；12产物一致、18有限已读来源稳定。原generic checker会预载 `worker_protocol_tools`，与r1 mock的dirty-slot guard冲突；保留封存r1，新增仅测试overlay `target/unified-quant-restored-performance-continuation-tests-static-r2`，delivery SHA`0cc9f79b8ebd2708201ec49f127ffe71942e93d21189e75a1614a4c1ee15f08f`，仅setUp暂存／移除／finally恢复8个明确mock槽，原4测试正文及生产guard不改。overlay独审review SHA`8829ed77f5a9d85d81c096eaaaa40f6584006af4d0edbf9b20214a97657f9c15`，10有限来源稳定、4产物一致。该冲突在运行前静态修正，没有执行一次已知必失败的检查；pure10 freezer改绑overlay后仍须首次实际验证。

矩阵driver现已静态封存：delivery5113B／`3888764b3ee705d7044c315c0ff832009880f5d02ad380e670ad0bc7d80f3add`，`run_matrix.py`20224B／`cebce16494b40c0d8d7360bbf32560f97067c9edbf864cea72d135760f1de34e`，4检查定义7614B／`ff851759b33b8ad3d592a2c3ec6d7f7431727d25f05b4b71cc21761770ebc21f`，12外部有限来源前后稳定。静态审查修复了事件payload与result共用对象造成末核失败的问题，事件先深复制再hash和保存；输出根同时拒绝非规范化路径。4检查仅覆盖完整矩阵投影、参数/数量、UNQUALIFIED与直接main/argv适配，不覆盖完整run账本/事件流，后续caller检查需覆盖成功组合与失败消费路径。当前无AST/import/tests/benchmark执行；完整外层Job、实际CPU资格及真实性能仍未完成。20:41实际轮询 `3229d8` 仍见原session live；driver事件1条、v4扩展事件仍只有初始000066，尚未出现新预约。

矩阵driver最终独审位于 `target/unified-quant-restored-performance-driver-review-r1`，review1957B／`8c758841363cda1288864064dab884752d8146da8f40d7b04675f0022b26900a`，evidence6051B／`cebe96fcd73a2e62e2fd02a7a4765e236c0d8997997a4209aff9e8c1feabbdf7`；9有限来源稳定、4产物一致，无新增确定静态阻断。ROOT-LIVE-STATE已收录库、测试overlay和矩阵driver的最终交付／独审Sources，仍明确全部新检查未运行、v5未发布、外层caller在制。

## 8.73 首份数值预约与性能入口的历史截止边界（2026-09-30）

同一 `session28076` 在20:54的实际轮询 `c09e3b` 仍live。`cpu-000002-register-catalog/result.json`（1301B／`6b0aa2f4c107782c5723bc627aea07489ec2840f63a83e791669d9c73816583b`）与 `cpu-000005-reserve/result.json`（1292B／`f75076a86cf81492bd542614b1832d58ff6a10f3b3775082bed631c1e8b4a8fd`）均实际PASS／error=null。v4扩展新增 `events/000067.json`，2068B／`d41451dac5dbdbb902f40d3a42a4548ca08a8c884061e7990f3d78de4eea7bc6`；其逻辑head是 `4d7eab029bf17b45de80ffeee95cb24fbd272f47d8f0da3b97e63cd277d98194`，不要与文件SHA混同。

该事件为首个COLLECTION预约，reservation id `49c07923aef4b5067ad6bf4754f53a61cf6acb452282cd0bad8d4ba892dc8986`，对应原Prepared中的 `000-collect`／B11-C3参考 `b11-c3-mix-000`。采集命令和1200秒原上限不变。这份尝试已永久消费，即使尚未观察子进程开始也不能重启或当作零消耗。root观察 `first-reservation-observation.json` 为2170B／`bd9192b5eeaf74ec3f62e8877d96666eaf9a1df9b9193c0cfc507592a521161b`；当时reservation-head操作尚无终态，driver事件仍只有RUN_STARTED，没有采集完成或整组退出证据。ROOT-LIVE-STATE已改为首预约在途。

静态预算核查发现一个跨层约束：已封性能entry直接把多小时outer deadline传给parent loader，而原 `restored_history_replay.py` 的 `MAX_SECONDS=1800`、`_remaining()` 和原facade的两个入口都拒绝剩余时间大于1800秒。因此新长phase不能直接使用r1 entry。独立新 `target/unified-quant-restored-performance-continuation-entry-static-r2` 仅把parent范围收紧为 `min(outer_deadline, parent_start + first_plan.history_cpu_seconds)`，并核原冻结字段为1800；outer／library／ABBA总截止保持原值。单独的parent-scope admission属于每次运行证据，不进入跨进程确定性的父audit。旧r1库、旧history与当前在途数值运行均不改；该新overlay仍只静态，未实跑。

首次库CPU资格相应改为12项：原6项policy、独立fixture overlay的4项、entry截止的2项。已封pure10 freezer原件保留且从未执行；新12 freezer需要绑定最终entry overlay与独审，不先制造一次已知不合法的10项组合执行。完整矩阵流程另已在 `target/unified-quant-restored-performance-matrix-flow-tests-static-r1` 定义并封存：delivery2962B／`8532b4d3192f7dbf140471df0574f40627a8c7127e64e3811ad9abe1befc6c9a`，测试源码18526B／`0e84f1fc9777484201bcd3b9426907afc62ba879b7bfb4d280ea3ec599e76846`。两个方法、三个场景调用真实sealed `prepare/run`，使用小型真实文件和mock Session/main，覆盖4合格＋40不合格的完整44格、预约先于main、终态事件独立副本、返回1／抛错消费与禁止重试；只在未来caller组合suite中执行一次，不另跑重复场景。尚未执行，不能代替实际账本或GPU资格。

有限来源工作量说明位于 `target/unified-quant-restored-performance-work-budget-static-r1/projection.json`，24933B／`0672ffd32647a2ee60c3e6ea8b18ac264f29c9596c530fbcbd3321a924a8238b`，14份有限来源前后稳定。44对全部合格时，原路径最多176个Worker臂、352704个计时RPC请求、22528个warmup请求以及128次legacy指纹调用；实际按N个合格pair、其中L个legacy基线pair，分别取4N、4N×2004、4N×128、4L。Protocol直接调用进程内 `grpc_tools.protoc.main()`，不能算作protoc子进程。两阶段已知应用调用314的说明包含controller、gate、owned venv及历史child，明确不是Windows唯一OS进程总数，也不推断binary内部模型加载／上传／Graph启动执行数。墙钟预算尚待root显式冻结；该说明没有登记新额度、启动GPU或修改生产程序。完整性能、独立确认、holdout及Tensor Core利用率实测仍未完成。

## 8.74 首份原生采集启动与两阶段性能调用器封存（2026-09-30）

原session28076在21:08的实际轮询 `30854b` 仍live。原driver新增 `events/0002.json`，2049B／`b0359ac2479426c159283abc35497e6e178aedf35f3ea3b4d8ca849518d3c4f7`，类型为ATTEMPT_STARTED，绑定同一永久预约和 `000-collect`。参考 `b11-c3-mix-000` 的 `collection.claim.json` 为1772B／`4a3217a46905f74a682b58480a7ada9da2541aa495efa0a29e6c7997117906b8`，状态CONSUMED_NO_RETRY。后续只读观察见supervisor、原生attempt/observer日志和输出文件；文件数量不能当作已完成batch数或数值PASS。尚无本份采集或整组完成回执。root观察 `target/unified-quant-selection-b11-c3-driver-r1/first-collection-started-observation.json` 为1726B／`cef1d1cd478931d5010d4d0cc4dbfdc67ffb06cb753e142cacc63d97db019e79`，ROOT-LIVE-STATE已同步；继续原session，不重启或重置消费。

性能entry-only r2现已封存：delivery5825B／`3d703b58945abf84cbae24c181ecc96d15033565aaf79fe7ab704aaa43545c6c`，entry22524B／`92daeeb02ca6582a732cf88d1a07b43e06795559de78f0543ca55ca461aa1a66`，独审review1874B／`c71df7fcc3b3c49dd3455b9ab7a6ec7f1aa05b3a28ffba83c579ef13f10a1a8e`。单次parent范围从原开始点计算更紧的1800秒截止并保存admission，整个library及ABBA继续使用原outer截止。新12项freezer位于 `target/unified-quant-restored-performance-continuation-tests-freeze-r2`，delivery5731B／`919c1fffca933049b24f9b9f109d0bbbefb67809dd5d4fc2503548db59b418d3`，独审review1362B／`0c650c97e942add3d77283cc96d0603eaf30c2189b009a9255c4ca550962abb2`。它绑定七份Python源码及三份delivery，保留6＋4＋2精确检查与原180／30／30秒harness；固定新CPU输出根r2尚未生成。原10项包及旧r1原件不改，均未运行。

两阶段caller已封存于 `target/unified-quant-restored-performance-caller-static-r1`，delivery10430B／`036317c99e146501d1bd36fc8eecb54ab13a16a882829a347190a43c8a4523a0`，60份有限外部来源前后稳定。`caller.py`、`controller.py`和`qualifications.py`分别接同一library session、原Job/gate/总钟以及四组实际CPU证据；root读过主要调用与资格链，最终独立审查绑定仍在收尾。prepare只发布已明确冻结授权的v5和矩阵准备结果；run必须另有真实prepare根exit0、完整子树退出和library scope completion才可进入。工作量说明固定绑定0672ffd3…，两个phase与总时限仍要求root另行显式冻结，静态delivery不产生登记授权。

caller首次suite为4项控制检查加2项完整矩阵flow检查，各只执行一次；flow独审review1561B／`adaa50b9c7025b34e64efd0fa429c8a8fd1cfcdea6ce7b471dedad7efd13d884`已完成。四组未来CPU资格必须分别实际通过4／12／4／6并绑定原源码、复制源码、真实import清单和owned退出；静态交付不能替代这些证据。6项不覆盖真实history、v5发布、Job、网络或GPU，也不声称覆盖资格消费者每个谓词的完整正向合成链。matrix4及caller6的首次freezer仍在静态准备，当前窗口没有AST/import/tests/freeze执行、新v5发布或ABBA。正式binary与生产Worker保持原状态。

本节续更：21:13实际轮询 `fbfd02` 仍见原session28076 live。首参考的native PID64944已由原supervisor持有句柄观察exit0，未超时／未被终止。`process-result.json`1029B／`166986de3ec2d6fdcea860cf13ff7f0a7a92d806ef3344250a19ea903286386b`；`collection-exit.json`5215220B／`58d1aa4a5bac5353f96a4c654220ab2453e6ed9d8bab129958ccd6472e338e25`；native `stage-result.json`4647276B／`285439fe293001d7ef281c5bb990960ac8547c451b2a6cb3db7d4ce100fe63cf`。stage为COLLECTION_COMPLETE_NOT_ACCURACY_OR_PERFORMANCE：668次提交／2004物理输出行／4009文件／41789986字节，全部backend finish返回，selection顺序一次覆盖；numeric_comparison_passed及accuracy_or_performance_certified仍为false。来源回执前后各10456条，文本列表不同来自一个recipe根路径的字母大小写，按Windows规范化后的path→bytes/SHA映射相同；此root观察未重哈希全部10456个文件。root观察 `first-collection-native-exit-observation.json`2820B／`dc8fd3d1cd4fe57c762af071214f9142a93c1d15cb770bff83419191cdb72b1f` 已保留。外层整理／入账与整组终态仍未观察，不据原生成功重启任何阶段。

caller最终独审现已封存：`target/unified-quant-restored-performance-caller-review-r1/review.md`2222B／`a40be4afa71b62bd64245f9657ac07e069f5973755cf4e068fcbafba9ce3c44d`，evidence12296B／`3ee4c5847eca0334217bedde6961443f4a53b32abf612721345afc62f9e198e4`，17份有限来源前后稳定、7产物一致。matrix4首次freezer已封存于 `target/unified-quant-restored-performance-driver-tests-freeze-r1`，delivery5697B／`1670b63bcd82afdd1d5a7df7781338606fab96ccea69bee7c9c6617659238e88`，独立审查仍待最终绑定；caller6首次freezer源码正在独审。两者都未生成CPU输出根或执行测试，后续仍需实际4／12／4／6的独立根证据。

21:17同一session实际轮询 `829535` 仍live，原driver新增PROCESS_EXITED事件 `0003.json`（981B／`8a5e0666863d02ba305ea9811712b22a538a286a04a8c9d09ee2fcf7391f311b`），采集launcher返回0。首参考 `report.json`7713B／`5fa25a0ac03b60f9a5fb717fdf27198b193262cbe2dff8525c493055b7dc6563` 已发布：32盘、2004／2004请求、668个B3批次，无padding，状态仍COLLECTED_UNVERIFIED。root观察 `first-collection-report-observation.json`1623B／`ac90eee43205b75485fe39b551673960539c82407c37fa319b6f2ba49a034e29` 已绑定原生回执、launcher退出事件和报告。尚未观察数值验证入账或整组退出，不把报告发布当作资格PASS。

matrix4 freezer终审review1351B／`7cfe0b005a2d2afc217e85d0449d7a929ea74847edf890dd1c09693c06a6c28b`已完成，12个有限来源稳定。caller6 freezer使用 `target/unified-quant-restored-performance-caller-tests-freeze-r1/delivery-r2.json`6339B／`bad266467d7bfeb5729bccf2ee46ffa08707b49a6dd1312d3e8148cfce9dedd5`，代码12506B／`0284a98c0453101c773c630a01149e11c1a24f0c9b63cda4a7f93181fc067390`；独审review1605B／`1abd17ec580e4efcdc60bf1f67162d4a3e5d363efe3f7d5364f6f096da335fd5`，15个有限来源稳定。首版封存清单的PowerShell去重错误把输入折成1条，原receipt/delivery完整保留；修正receipt-r2明确30个来源前后稳定，源码及README未变。这是静态元数据修正，没有发生CPU执行或重跑。四套首次CPU准备器与资格消费者已完成静态接线，尚未freeze、运行检查或取得真实资格；当前性能阶段没有登记或启动。

## 8.75 首份采集观察入账与最终性能计划组装边界（2026-09-30）

同一session28076在21:26实际轮询 `6170ab` 仍live。首份B11/C3参考已写入v4 `events/000068.json`，6191318B／`e7c27cefef362a68870f8bb75a4639e436b46fad3304b3ad2afaa04f7cecd711`，事件COLLECTION_OBSERVED，逻辑head `7ba4a7d6c11ed82e6443882f651e7cb4f5c228935f35c0fdb1a872d4c6a34556`；前驱仍为该同一预约的4d7eab02…。事件保留原报告5fa25a0a…、恢复包98d165a9…及模型／recipe／实际推理profile身份；validation为RAW_PROTOBUF_AND_RESTORED_EXECUTION_IDENTITY_VALIDATED，numeric_gate为NOT_COMPARED。其9份主要原件、9份冻结evidence及19143条runtime Sources仍按原协议处理，不能据文件生成声称独立重验全部来源或通过精度门。

root只读验证了该事件逻辑摘要，观察 `target/unified-quant-selection-b11-c3-driver-r1/first-collection-ledger-event-observation.json`1651B／`0c36a8480d61bbb61f9ff7b95f2cf42b92fbc35c421b24423328097c43156c4b` 已绑定前次报告观察。`cpu-000008-check`已PASS，`cpu-000009-ingest-collection`在该次观察仍无result，属于事件已写而操作末核未终态；driver及controller也无总结果。只继续原session，不重启、清seal或重复采集。ROOT-LIVE-STATE已更新该阶段。

后续最终组装入口正在独立 `target/unified-quant-restored-performance-plan-assembler-static-r1` 静态准备：library组件消费四组真实终态、最后不可变v4 event及四项CPU资格组合，按封存投影物化完整98源并构造entry r2要求的15字段library plan和9字段parent配置；outer组件消费精确4／12／4／6实际CPU根、原Job资格和已封caller，产出campaign与显式启动说明。装配不调用library session、history、Protocol、账本或benchmark，控制／矩阵运行根只声明路径、不提前创建。prepare／run及一次CPU组装时限均要求root未来显式给出，无默认值；性能首次总钟仅由实际controller入口创建，包含两phase间隔。共享IO仅去重本次初读，发布前仍须真实末核，不能用缓存证明来源稳定。当前尚未执行装配、任何新CPU检查、v5登记或ABBA。

## 8.76 首份参考自比较完成与性能发布前边界（2026-09-30）

原session28076在21:45实际轮询`a8eb0d`仍live。首份B11/C3参考的`cpu-000009-ingest-collection/result.json`1302B／`d28887bb31739bee8cfbb4f896f52cb90593c69abf1d9a159d7ef1099360e390`实际PASS，来源末核两次；随后`cpu-000010-compare/result.json`1292B／`d20d76e3c315fae8c48c0714213e905b85d08f28d1eea1bf0939d3abc3c3743c`也实际PASS。原流程已进入`cpu-000011-check`，尚无controller或driver整组result，不重启、不重复任何已消费采集。

v4新增不可变`events/000069.json`3201B／`3cbecd6a0643f5f151ababfa9db485bbe052016cd2c29e7680ffde2e48f782be`，NUMERIC_COMPARED，记录逻辑head`901074fcb6f37ccd43db842f6512df38f67525a2b760b06b87159cec0b278c81`。其中reference与candidate是同一attempt；32盘、2004局面、2004/2004输出字节相等，胜率／分数／policy／ownership误差均0，实验门PASS。必须明确这是参考自比较，不能据此称任何混合候选通过、原FP32门通过、棋力或性能通过。driver `events/0004.json`653B／`f51402e4d0ce03568076f8b1227a292e489ff8cca2372295a7d4be2913269ab3`记录NUMERIC_VALIDATED。root观察`first-reference-self-compare-observation.json`2315B／`6ac5ed37c9897bfeb2c334b46cb529903f95f58a0d2d432746f4e46cd8c01308`绑定原件和实际工具轮询；root未重新推理或重读全部原始输出。

最终性能计划组装器已仅静态封存：`target/unified-quant-restored-performance-plan-assembler-static-r1/delivery.json`8053B／`2c52746f283cfa6bef5645f42ea2bf85c8d7d5589474fe7000315a0b9d82ef6f`；assemble22645B／645a6501…，library_plan24733B／b3d7ad65…。作者22份有限Source稳定，独审18份有限Source稳定，5项artifacts一致；root另以有限哈希核对5产物绑定相符（253338），没有运行装配。独审`target/unified-quant-restored-performance-plan-assembler-review-r1/review.md`3781B／`81f8d721c51e2b8ba4f509ab1e3afc99ca1463064c69b7d92d963ce2f78c2cf4`，evidence13798B／`58cdff37d459e3158ae9151fe8a3d81e323374b0d28ce4d1ee959c445f5be57a`。初稿缺Inventory.key的确定接线问题已在封存前修正。四数值实际根及4/12/4/6实际CPU根尚不齐备，静态组装交付不是实际资格或发布许可。

独审确认一个发布顺序缺口：原continue_once先写两份永久parent seal，随后才构造包含完整preregistration、matrix、tools和snapshots的contract，而最终读取限128MiB；单独audit≤128MiB不足以保证完整contract可读。当前大小不证明未来必超，但不能用永久发布试错。contract无当前时间字段，snapshot Source可由冻结目标路径、bytes和SHA预构造；因此正在独立`target/unified-quant-restored-performance-contract-capacity-static-r1`静态派生最小修复，将原pretty的最终UTF-8字节大小和同一截止检查放在所有seal之前，后续复制逐项核Source，再写同一已检查字节。阈值、原事件协议、失败消费和不重试规则保持；旧封存代码不改。新增检查仅定义，首次CPU冻结和消费端需绑定最终新字节。此时没有新测试／导入／AST／history／v5发布／ABBA或生产变更。

## 8.77 首个混合候选预约与完整contract发布前修复（2026-09-30）

同一session28076在22:07实际轮询`a556ae`仍live，仍只继续原入口。首个混合候选`b11-c3-mix-002`已新增v4 `events/000070.json`2069B／`8564a16bd3d18d831a2746e7d7998f3617097319383c73713c2a60254ab9aad0`，ATTEMPT_RESERVED，逻辑head `e2ac9b3684b619a5edfc5c6841872c6a6b9a6d4b4e4c5b971f8a7c0ba4466d01`，继承前一参考自比较head901074fc…。候选attempt `09141d817a4e07ca1bed7969a738876de532f13575074c12e8b96033087e5008`，reservation `8a99b89d7148d71e4d59aec1f5d5d05e237b05647212741a2e4dac3dc6bcab1d`；绑定原登记`registrations/b11-c3-mix-002.json`1900711B／3cba698e…。`cpu-000013-reserve/result.json`1292B／`052f28c3a1d08249a204bdc7476975085d89ffc4eefe4f051bd226e66c7aac6c`实际PASS，inventory_checks1。随后进入cpu-000014-reservation-head，新COLLECTION永久预约累计2；这不是两次已完成采集或候选精度通过。root观察`first-mixed-candidate-reservation-observation.json`2489B／`085f00931b0b28e2209c8306b9ccc68d329d78a385431078400d6b1fb007453e`绑定原件与实际轮询c4b41f、元数据2a07df。整组及候选采集均尚无完成观察，不得重启或重复预约。

完整contract发布顺序修复已独立静态封存于`target/unified-quant-restored-performance-contract-capacity-static-r1`：delivery6933B／`1d276cb7f1bf467340056145bc721befea1b174c3b406486acd9ab4032047734`，新continuation31292B／`dab25fb38544b6cb9080d5ba9df7ee0ef4e3ed559579fde06a2ea9ea31d5e41e`，test_capacity12710B／64ed3de9…。两份永久parent seal之前，先预测snapshot的目标路径／bytes／SHA，使用原pretty（UTF-8、ensure_ascii=False、indent2、拒非有限数、末尾换行）序列化完整contract，按原128MiB读取上限检查并复核原绝对截止；后续每份实际snapshot须与预测Source一致，contract用同一已检查raw独占写入、flush/fsync并核Source。原`_active`起全部后缀逐字保留，数值／ABBA／继承失败与不可重试规则不改。三项新增纯CPU检查仅定义，覆盖精确字节边界、audit可容纳而完整contract不可容纳、序列化／截止失败时零seal和零目录写入、成功raw与Source以及复制失败后的永久部分消费；合成小界限不分配128MiB样本，不运行真实history／Protocol／Worker。

消费接线独立封存于`target/unified-quant-restored-performance-capacity-wiring-static-r1`：delivery18693B／`db886463ac05ad96e2a922f87acd6afa8429581c6d129bf460975f6ee37598ae`，caller-delivery13136B／`66e2e5efe94bb138505610e5856a5917b48b59f1c5aab9005dff0478a6ec3617`。新qualifications19674B／ec21656f…；assembler22601B／74f6047b…，library_plan26165B／296458b3…。仅源选择、资格名单和固定身份派生；原caller／controller／test、entry r2、Job／gate、完整98树及公开字段接口保持原路径字节。新continuation贯通测试snapshot、预期模块、实际导入来源消费、runtime库存和最终library plan，不能拿旧字节资格冒认新代码。

首次资格现在是四个实际根4／15／4／6，共29方法。库15保留原policy6＋parent4＋entry2，再加capacity3；不新建第五资格根。`freeze_library.py`20363B／11c3edba…未来只创建`unified-quant-restored-performance-continuation-tests-cpu-r3`；`freeze_caller.py`12516B／0d41e4e2…未来只创建`unified-quant-restored-performance-caller-tests-cpu-r2`。另两组仍为projection cpu-r1和matrix cpu-r1。root有限核对capacity4项与wiring10项artifact均与交付Source匹配，并核实四新目标及旧library cpu-r2／caller cpu-r1均不存在（77e954）。因此没有重复任何已执行测试；旧pure10／pure12／caller6准备器和旧描述永久保留，未来不直接运行这些旧配置。

合并独审`target/unified-quant-restored-performance-contract-capacity-review-r1/review.md`4380B／`ad8c1372584d7422961e768566f2fffd543ea43aac4cae1b4e9bdc635bcd27b4`，evidence23820B／`5ffab1b7fdf3fd96e3704fa9c263e8e951ef1c74e4b6615be343361f8aa82ef4`。38份明确小Source前后稳定，capacity／wiring／caller的4／10／7产物绑定相符；作者11／88／73来源回执元数据一致，独审不冒称重哈希全部继承大库存。未见确定静态阻断，但尚未执行Python导入、AST、首次29项、任何freezer／assembly／v5发布或ABBA，也不证明未来完整contract实际能容纳。真实数据过大必须在新门拒绝；四数值终态、实际CPU资格、显式未来时限及运行时来源核验仍是后续前置。ROOT-LIVE-STATE已选择新消费入口，同时保存旧描述的引用。

## 8.78 首个混合候选原生启动与路由证据范围（2026-09-30）

原session28076在22:17实际轮询`25357e`仍live。`cpu-000015-check`实际PASS；driver新增`events/0006.json`2049B／`729fcd701fb42460f8f55c8b239c938cdc296c196aab91c2f13ca7710e6ef0da`，ATTEMPT_STARTED，对应同一已消费的`b11-c3-mix-002`预约。候选`collection.claim.json`为CONSUMED_NO_RETRY，supervisor `collection-launch.json`2291B／`d65766109a3adae690d511c923f1f71aa4394a78217b613cac4084a009c06b46`记录native child PID81736。此PID仅为启动记录，不能替代当前held liveness或真实退出。root `first-mixed-candidate-started-observation.json`2762B／`1d43d766eb42a3514688d2e1fe074c9beb5261555fdb07053e3c90c5556c85d2`绑定启动、claim、前置CPU结果及原预约观察；尚未观察native退出、完整采集、比较或整组终态。只续看原会话，不重启。

另完成该候选既有包的有限只读路由检查：package458233B／`6450ec0054e8b317600133dcb1af7b4246d73c76afae8297a44777c984140fb2`与登记一致；B3内嵌directory_json原文UTF-8 SHA `7df28fb35832ecf988860269c06a1436c6a91505c400670788b0bd6523570e9d`与保存摘要一致。契约graph／pipeline、max_batch3、pad_to_max=false，rows1083；165项operation中33组FFN由6组INT8和27组FP16构成。INT8上下投影层索引同为57／61／101／105／121／125；首例dual为N2304/K384的一次native GEMM，down为N384/K1152的一次native GEMM。其余27个FP16 dual选择custom `cutlass_dual_ffn_swiglu`，FP16 down为Lt。量化与融合的组合收益仍需完整FFN成本和正式端到端对照，不能从6/33比例推断速度。

root `first-mixed-candidate-route-observation.json`5522B／`994c360c1572d650a50d421dd6afec361091a2460b5253ad11d4e7308d21b743`保存上述有限来源和统计。actual386快照的`crates/kata_nn/src/backends/model_lt_inventory.rs:92`明确：目录根据已上传权重、prepared工件和有效runtime推导expected routes，本身不做GPU调用，也不是execution observation。因此此次只证明保存路由契约及摘要一致，没有新增模型执行、Tensor Core计数器或性能测量。Lt算法表与原保存恢复证据也不等同于动态指令占比。后续诊断可复用已有独立`ffn_group_cost`及INT8算法probe能力，但前者目前只覆盖Direct FP16/INT8，拒MXFP8／compact，没有现成NVTX范围控制；后者同输出目录不能直接重复application replay。原execution_package_probe写永久consumption_marker，更不能当可重放入口。所有后续profiler仍需独立有限执行范围与输出绑定，本轮未新增预算、实现或运行。

## 8.79 首个混合候选采集完成，数值入账在途（2026-09-30）

原session28076在22:37实际轮询`48875b`仍live，整组driver和controller均尚无终态结果。首混合候选`b11-c3-mix-002`的native PID81736已由supervisor owned handle观察正常exit0，`process-result.json`1029B／`97e3f88ead6b59e683de8f5c0e778f2e9c807dca1348ce69ed6b6524818516b5`，无超时或supervisor终止；collection-exit5205106B／`ebb1e8287895b7ad3b3d7b79cc82f1dd08a4d00bf1fcb3d6ec9354649d37cb4a`也记录exited/code0。driver事件0007为PROCESS_EXITED、launcher returncode0，981B／`aa1d7a1f1890963286765366fb45ede678c5c277dc18f71bf31b8e6c211fd2a5`。这是原已消费预约的结果，不能据此重启、重做或写整组exit0。

stage-result4642238B／`bac3912a908a9b84d30befcb1c3160e90f1f74826349e5f8ad2cdc751eb338b2`记录668次三行提交、2004输出行、4009文件／41789978字节，所有backend finish均返回，selection行按登记顺序各一次；状态明确为COLLECTION_COMPLETE_NOT_ACCURACY_OR_PERFORMANCE。manifest3835910B／`19df10c052b2fd43014b869fe04bb8bc03cbd84268cb3a31f0e5d112794473d9`；report7713B／`4643c10b639a531097a0e341881af1796f39fe40210933f5aec242848ddb22af`为COLLECTED_UNVERIFIED，expected/completed均2004、32盘、B3、capacity3，performance_measured=false。这里的numeric_comparison_passed=false表示采集阶段没有通过比较的声明，不是已判定候选精度失败。

采集后`cpu-000016-check/result.json`1290B／`91732bb039141ca2c941b634673fc8ae2a2d180231be9a9c6502b589a3274beb`实际PASS；随后`cpu-000017-ingest-collection/started.json`227B／`29dfeedfb71b4226b8754f19d5fa5cbbd7ddb80baba687ed18531311d4a98c48`已生成，观察时尚无ingest终态或候选比较事件。root `first-mixed-candidate-native-exit-observation.json`4229B／`25a832a04385cd8d1a6532802d592a650090210e8fc2c8be232353bced2541de`绑定上述有限原件及先前启动观察；本次没有重读raw、复哈全来源或替代正式ingest审计。完整四组数值选型、性能矩阵、独立确认及holdout均仍未完成；29项首次CPU、v5发布、ABBA和profiler未执行，窗口内仅静态工作。

22:41续看同session28076（`f3dc6b`）仍live。原流程已写v4第71号COLLECTION_OBSERVED事件，6184911B／`f0b72ea1c478d1fa2ac8e973466b1059f787c9f174529b4687a9349f289759ab`，logical head为`0207589d65ccd96f7e8c599cdc924f2ff9f1781237a6505b61a5eecd91160c2e`，previous为原第70号预约head `e2ac9b3684b619a5edfc5c6841872c6a6b9a6d4b4e4c5b971f8a7c0ba4466d01`。root按原canonical规则独立计算摘要相同；事件绑定同一reservation及report，validation为RAW_PROTOBUF_AND_RESTORED_EXECUTION_IDENTITY_VALIDATED，numeric_gate为NOT_COMPARED，original_files9、runtime_sources19118、evidence9。root有限观察`first-mixed-candidate-ledger-event-observation.json`2065B／`d30ab1040951a6ee1309bbf5b28486f5ebd7c5ced684e6bcb31c458242f95c45`保留此范围；未独立复哈全部事件引用。此时CPU17仍无终态、CPU18比较尚未观察启动，不能把事件已写当作ingest最终PASS、候选精度通过或整组终态。四份采集回执另经有限独立复核，SHA、报告引用stage及668×3／2004／4009计数吻合；该复核未读取raw、manifest或其库存。

22:48同session28076实际轮询`aa74c4`仍live。CPU17入账末核现已生成PASS回执，1302B／`c13bad8cd6fcef278d1168a8a8dc79d69d1ac85bd1b476b6e5bfd9ff73860ffb`，inventory_checks2、历史来源81619，来源集合SHA保持`9904b46b50e43fe469c2a82d5f741561bd96255e97aeef70d25ea94294206799`。随后CPU18 compare已启动，started227B／`95166c4c74202e447527b090fc2aa855e014093dd98e473d47e7290c412f7d2d`；此时尚无比较结果。root `first-mixed-candidate-ingest-complete-observation.json`1598B／`f0106be0112e7befd2f44978d8066d2b17034757b5dd0301fca5108cc54c2a72`绑定两个原件及前一事件观察。ingest PASS只证明该阶段结束，不是混合候选数值裁决、性能资格或整组实际退出；此前NOT_COMPARED事件原件保持不变。

## 8.80 首个混合候选通过selection实验数值门（2026-09-30）

22:55同session28076实际轮询`1d0b77`仍live，CPU18 compare尚无最终result，整组driver／controller也未终态。原流程已写v4第72号NUMERIC_COMPARED事件，3524B／`5e95820bb35da4c43866bb0d952a4b00a3d3db80472c1585a359a0ee24961f1f`，logical head `711d1fc2f1f363e4ddc139bd44eeaf30da62c5163febe7f64a1a9d0c4394de3d`，previous为第71号采集观察head `0207589d65ccd96f7e8c599cdc924f2ff9f1781237a6505b61a5eecd91160c2e`。canonical摘要独立计算一致。候选`b11-c3-mix-002`的attempt `09141d…`及comparison key `5bbe614…`匹配第70号登记，参考attempt `5d8771…`匹配第69号参考自检；两端不同，登记明确is_reference_self_check=false，不能把本次结果混同于先前参考自比较。

事件记录selection 32盘／2004局面、status DIAGNOSTIC_PASSED，experimental_gate.result为PASS，measured最大白方胜率概率差0.013682335615158081，低于既定0.06门。概率差换算为百分点须乘100；该候选的主要报告如下。

| 指标 | 最大 | 平均 | P95 |
|---|---:|---:|---:|
| 白方胜率绝对差（百分点） | 1.3682335615 | 0.0405495763 | 0.2094894648 |
| 白方目差均值绝对差（目） | 1.7043190002 | 0.0204949968 | 0.0583243370 |
| policy KL（nats） | 0.0008283370 | 0.0000552766 | 0.0001681925 |

策略Top1为1990／2004一致（99.301397%），合法集合2004／2004一致；参考Top1在候选中的最差名次为2。ownership原始数值最大绝对差0.0377013981、平均0.0006341648，不另转成胜率百分点。完整输出逐位相同0／2004，非有限值及合法性硬门均PASS。现有实验门要求输出有效、合法集合不变及最大白方胜率差≤6个百分点；目差、KL、ownership等为报告项，没有另行加设通过阈值。因此这只是selection诊断门通过，不是原FP32门、棋力、holdout、性能或生产认证，不能从INT8比例和该数值结果推断速度。

root `first-mixed-candidate-numeric-event-observation.json`3738B／`e66f87dcd205bccbe595d6a374a57eca9535391d57ab3d04536aca9a06689389`绑定当前事件、参考自检、候选预约与先前ingest完成观察。三事件另经有限独立只读复核，SHA前后稳定、身份与单位换算相符；没有重新计算raw指标或执行项目程序。事件存在与gate PASS不代替CPU18末核、完整控制矩阵资格或整组OS退出；原v4仍不允许ABBA。29项首次CPU、v5性能续接、无profiler正式ABBA、独立确认及holdout均未完成，保留所有既有失败／负收益和永久消费，继续原session，不重启。

23:01同session28076实际轮询`99d90c`仍live，首混合候选的CPU18 compare末核已实际PASS：result1292B／`870116e20ab7327b70bfa70944e113dbda2fac329e3cbf3e6383a0cb7e030691`，inventory_checks2、历史来源81619及集合SHA `9904b46b50e43fe469c2a82d5f741561bd96255e97aeef70d25ea94294206799`保持。driver事件0008已写NUMERIC_VALIDATED／gate PASS，653B／`50343defb16d5ae57aa4099db41417af3106d6cb31642338403352dc25476440`，明确绑定候选spec `156bc8…`、comparison `5bbe614…`及v4 head `711d1fc2…`。CPU19 check已启动，started227B／`7afc0edecbde8f309ec59d63feb01e2fab84612878a01e9d4640c781f0993b6e`。root `first-mixed-candidate-numeric-complete-observation.json`2359B／`419740913ff5e8817dc220f93eb80a296756ad7ba1f6eba13e3ba87c62abc57e`保存该阶段完成证据；这结束了首混合候选的采集／入账／数值比较，不是整组OS退出或完整性能矩阵资格。原会话继续，不重启或重采。

另对B11/C3剩余两个既有候选包完成有限只读路由核对。三个候选在B3／rows1083下均有33组FFN，量化位置形成嵌套集合；下表仅描述保存的路由契约，不是kernel执行trace或性能排序。

| 候选 | INT8 FFN | FP16 FFN | INT8 dual／down共同层索引 |
|---|---:|---:|---|
| mix-001 | 3 | 30 | 57、61、105 |
| mix-002 | 6 | 27 | 57、61、101、105、121、125 |
| mix-003 | 11 | 22 | 57、61、69、85、101、105、121、125、149、165、173 |

mix-001包SHA `00531d6ddf37f07e549b2100385d9b50b4ce971db38c6a93d70705c99bd4fb1f`，recipe `2679f0300d0eba795148f4e030564282c775c580cfa5624644922c6f68ca4a3f`，B3 directory `b6269cb3952daf48e59b03ba3e5900300804df421cb1193021770473219e115a`；mix-003包SHA `592f2bc662eb2b922edca2b60194ea674a014b268397dfc6580bd05d7f8cfeba`，recipe `ad4b263d00b37314531979779087e06a0e9b1dff3029bc1a49e871d29456386c`，B3 directory `64226dbb88b70dffb86d46ad0c81896efa00b886801338c69420a577792dc276`。两包读取前后摘要一致，directory原文UTF-8摘要匹配，目录model／graph／recipe／policy与各自binding一致；共同execution_policy为`3e98968a9de0083c55acbea896caa82b91f68dea35d74e3db91c6c30761a6c57`，包内该字段不能冒充外部actual_inference_profile核对。

两个候选的FP16 dual均为custom `cutlass_dual_ffn_swiglu`，INT8 dual为`int8_lt`，down分别为FP16／INT8 Lt。因此增加INT8层会改变这些层的融合路径，不能把INT8层比例直接换算为加速比。目录的native_gemm_count在融合FP16 dual上记录2、native_gemm_n为1152，在INT8 dual上记录1、N2304；这是描述字段，尤其不能解读为FP16执行了两次kernel launch。root `b11-c3-remaining-candidate-route-observation.json`3708B／`d8561fd4ca13457a35a48baa05a1e3772925b07f6e4a8162ff66051926acd8fd`保存有限核对范围，未运行GPU、profiler、测试或重读raw。冻结Prepared摘要b86616f7…保持，下一项002-collect为mix-001，后续仍保留旧控制组与mix-003；23:16同session28076轮询`934c4e`仍live，CPU20检查已PASS、CPU21 reserve已启动但尚无预约事件或终态。

## 8.81 第二个B11/C3混合候选完成永久预约（2026-09-30）

23:24同session28076实际轮询`99be43`仍live，原驱动新增v4第73号ATTEMPT_RESERVED／COLLECTION事件，2069B／`fd201d61de7bc9e8674e7f1099560075de7d3eda6be2c951a4fc78e236b93a12`，logical head `aa385f94d012b4b14409ffe35515f5e33523ef7b9f6294181d5e06bcef25f307`，previous为首混合候选比较head `711d1fc2f1f363e4ddc139bd44eeaf30da62c5163febe7f64a1a9d0c4394de3d`。新预约对应`b11-c3-mix-001`，attempt `ecc1cf3d…`、reservation `2547b5a643994e63978da7a90195735f20d19a049604a6fe3ce1114aaf04a708`、candidate spec `375d02df…`、comparison `ca3e7e1c…`，明确非参考自比较。事件canonical摘要、原Prepared b86616f7…中的002-collect身份与登记Source均经有限核对相符；登记1900711B／`e7dd137f21f6c8ca18f09edab3b204afce153b21882cec40e0510a0ba84179ea`。

CPU21 reserve已实际PASS，result1292B／`adfd0f61cb016f9aafc03c4fab6c2cc2fd4766a80d76a6eb1da4145b9e4af3c6`，inventory_checks1、历史来源81619。CPU22 reservation-head已启动，started227B／`90431a3c29cb628a6f43c5ccab4830382b04248cf78902e54c966398c9227efa`，观察时尚无末核result。该候选的claim、native launch、process-result及report均尚未观察到。root `second-mixed-candidate-reservation-observation.json`3518B／`0801b1e656c0b0856007e744fb7f6cc7ae08675eb1253e8db0de587f9b641ff7`绑定上述事实与先前mix-002数值完成观察。新COLLECTION永久预约累计3，不是已完成3份采集；首参考和mix-002完成证据保留，mix-001的3 INT8／30 FP16保存路由不构成精度或速度结果。整组未终态，所有已消费项不得重启；后续完整矩阵、性能、独立确认及holdout仍待完成。

23:31同session28076轮询`d3f37f`仍live；CPU22 reservation-head现已PASS，result1301B／`1fb1391b52721ec731c1dcbd02f2f52485ca856cbf864a820774c97adef05a86`。driver事件0009为ATTEMPT_RESERVED，4278B／`5dd0e18bfba93382afab138e9f36926f7c8cffe32c28d63f31f43260e4d8ab2a`，receipt为RESERVED_BEFORE_EXTERNAL_LAUNCH／consumed=true／retry_allowed=false，绑定原reservation2547b5…与command002-collect，execution_started=false。CPU23启动前检查已开始，started227B／`61488ca1118bf2e5c91a71b1d63401c24120c7aa6424552173c68f5387258060`；claim、native launch及report仍未观察到。该预约的执行尚待原流程继续，不能把预约核对通过当作GPU采集完成。

## 8.82 第二个B11/C3混合候选原生采集启动（2026-09-30）

原session28076在23:39实际轮询`be9dac`仍live。第二混合候选`b11-c3-mix-001`的CPU23启动前检查已PASS，result1290B／`e818b518a9c56f2e4e2974b46b1801af67f45a18a876627513905bd681b0c327`；driver事件0010为ATTEMPT_STARTED，2050B／`2824101fd99efa758bf016c1835c32ac003661e889daf5ca83dd321955bef70e`，对应原reservation2547b5…、command002-collect及原1200秒child上限，没有新预约或重置截止。

supervisor `collection-launch.json`2290B／`5ea438ec531e3c36f29fb13f0b824904e32c47c2a4b6967e6ac01815761398fd`记录native child PID8544，冻结exe SHA仍为dd665d03…、collection登记仍为e7dd137f…。精确PID查询`865a9b`在当时观察到8544、parent52208、出生时间23:37:28.062178，镜像路径与launch记录相同；这不等于root持有native进程句柄，也不能替代实际退出回执。此前短暂未观察到claim不作失败裁决；23:39实际已观察`collection.claim.json`1772B／`9814d683a43599182c64ae0afda77f14c5c41b44a928782ebf7a78b151477d1c`及supervisor `attempt.json`644B／`b084c8de143e24285a55b0ce56689b5fe0b98584a64220ae9055c2a411d98814`，两者均为CONSUMED_NO_RETRY。

root `second-mixed-candidate-started-observation.json`3187B／`5c35075e81a8a3c78f81d2e72c2e560aa8809f4e09b9c38406f7871aed1635f3`绑定前一预约观察、检查、driver事件、native launch及两个永久消费原件。当时尚无native process-result、采集report、候选数值比较或整组终态；只能确认启动，不能宣称已完成2004行、精度通过、Tensor Core利用率或性能收益。新永久预约仍累计3，原首候选mix-002数值完成证据保持；继续同一session，不重启，窗口内仅静态工作。

## 8.83 第二个B11/C3混合候选采集完成与入账在途（2026-10-01）

原session28076于北京时间2026-10-01 00:01（UTC 2026-09-30 16:01:00）经同一会话轮询`929892`确认仍LIVE。第二混合候选`b11-c3-mix-001`的保存路由契约为3组INT8、30组FP16 FFN；native PID8544已由supervisor持有并观察exit0，launcher driver事件0011的returncode为0。这是native子进程终态，不是外层或driver整体退出。

本次采集保存32盘、668×3=2004局面，共4009个输出文件、41,789,958字节；report状态为`COLLECTED_UNVERIFIED`。`target/unified-quant-selection-b11-c3-driver-r1`下的native退出观察4204B／`d511d95b766b269172e223401d2170363226bd33684d4cf0da0d24972fd5e996`绑定该完成事实。采集退出0及文件数量不能替代数值比较、逐位验证或性能证据。

root有限只读`511d36`确认control `run/history-run`中的CPU24检查result为PASS，1290B／`4b4f2b12ba3a2196c68c41f88bc7be66933ee9e9ee860fd47f115c7b14b0731f`，记录`inventory_checks=1`及history来源数81619。同目录链的CPU25 `ingest-collection`已启动，started227B／`3b59774763ef5422c73888f250ad6bb493d2da7e89e9e80b0cef4226debe4dc3`，尚无result；不能把启动记录写成入账成功或候选数值通过。root的`target/unified-quant-selection-b11-c3-driver-r1/second-mixed-candidate-postcheck-complete-observation.json`为1673B／`56e61248df0072dad5806210a2f536663af85ffeadac57aa908c624448015840`，绑定CPU24 PASS、CPU25 started与同一session轮询`929892`；ROOT当前嵌套状态已更新，历史顶层FAILED保留。

本次观察时v4仍为73事件，逻辑head为`aa385f94d012b4b14409ffe35515f5e33523ef7b9f6294181d5e06bcef25f307`，新永久预约累计3，外层result与driver result均不存在。首混合候选mix-002的数值PASS保持；第二候选仍待入账和数值比较，整组未终态。继续同一session，不重启、不重置消费或截止；完整性能、holdout及生产未通过，29项首次CPU、v5、ABBA、profiler均未执行，运行窗口内仅静态工作。

在途更新：北京时间2026-10-01 00:03:54（UTC 2026-09-30 16:03:54），同一session28076轮询`307021`仍LIVE；CPU25尚无result，CPU26未启动，但v4事件`000074.json`已生成，6,184,911B／`c2b90d2c209e7e0061d1123fc91800b819f298669ab5002f401c685def65b800`，类型为`COLLECTION_OBSERVED`，逻辑head为`3f2dcaaca3517b02eb16c1b1f611b858588b607b5d67fbe4df6eb558f1448fd7`，prev为上述第73事件head。root有限复核`798bec`确认第73、74事件canonical摘要和前序链，以及同一attempt `ecc1cf3d…`、reservation `2547b5a6…`、执行身份`375d02df…`与实际report SHA `e14828a4…`相符。第74事件的validation为`RAW_PROTOBUF_AND_RESTORED_EXECUTION_IDENTITY_VALIDATED`，numeric_gate为`NOT_COMPARED`；记录9个original_files、19118个runtime_sources和9个evidence，此处仅统计引用数量，未复核全部引用文件。事件生成不等于CPU25终态，也不构成候选数值裁决。

root将本次第74事件观察保存为`target/unified-quant-selection-b11-c3-driver-r1/second-mixed-candidate-ledger-event-observation.json`，1813B／`4aa82aef0b3fd1b78020ebcca1d628c3dbb3cd6f2caaccd986698da87caca561`；ROOT嵌套latesthead同步为`3f2dcaac…`，状态为`RUN_EXECUTION_ACTIVE_SECOND_MIXED_COLLECTION_OBSERVED_INGEST_PENDING`，历史顶层状态未改。独立`progress_metrics_check`仅对第73、第74事件和report三文件完成有限只读复核：canonical摘要、前序链及attempt/reservation/execution/workload一致，report的actual_profile、登记和child-exit引用逐字段相符，三文件前后SHA稳定；未复核raw或19118个runtime来源，未观察到CPU25终态。

续更：北京时间2026-10-01 00:11:49（UTC 2026-09-30 16:11:49），原session28076轮询`0bb979`仍LIVE。root实际只读`a5b5d6`确认`target/unified-quant-selection-b11-c3-driver-r1/run/history-run/cpu-000025-ingest-collection/result.json`已生成，1302B／`7428a7e7088ebc9144d989a21441c176b08cb0ac5a43e62b1139349c102b63db`，status为PASS，`inventory_checks=2`、history来源数81619、source_set为`9904b46b…`、error为null，并绑定原started `3b597747…`。这证明该次入账末核通过，不是候选数值比较通过。

同一`run/history-run`中的`cpu-000026-compare/started.json`为227B／`30ead42394c921abf356cccfd7c281a0fd5af8b703238035a36bb06be0cea28c`，数值比较已开始；compare result、driver result及controller result均不存在。v4仍为第74事件，`numeric_gate=NOT_COMPARED`，继续原会话，不重启或重置消费与截止。root观察`target/unified-quant-selection-b11-c3-driver-r1/second-mixed-candidate-ingest-complete-observation.json`为1585B／`d4f1a26d15f36328832a95f0f95c4fedbf26158dad19299927a147b38072b117`，ROOT当前阶段已同步为数值比较在途。首mix-002数值PASS及全部旧记录保持；完整性能、holdout及生产未通过，29项首次CPU、v5、ABBA、profiler均未执行。

后续控制身份静态核对：冻结的第4份`003-collect`为`b11-legacy-int8-q64`／`cudaint8backend`，登记`semantic.execution.batch=8`、`config.nnMaxBatchSize=8`，命令为`--batch 8 --capacity 32 --window 32`；第6份`005-collect`为`b11-legacy-fp16-c32`／`cudabackend`，相应batch与nnMaxBatchSize均为16，capacity/window仍为32。两者都是`REQUIRED_LEGACY_CONTROL`，分别绑定自己的显式legacy spec，共同参考为执行身份`d71c3cea…`，均非自比较、最大尝试1。目录中的C3标签不能替代这些控制各自的固定配置；`numeric_workload.packing=per_execution_spec_fixed_batch_with_tail`及命令参数也不能证明旧RPC实际物理batch分布。旧spec的`semantic.execution`没有统一后端的`inference_profile_id`字段，不能补写参考profile。

上述有限核对只读取Prepared与catalog及ROOT中的Prepared Source，未读取spec等引用文件、环境值或运行项目程序；两份JSON前后SHA稳定，独立`progress_metrics_check`核对结果一致。root保存静态观察`target/unified-quant-selection-b11-c3-driver-r1/b11-c3-legacy-control-identity-observation.json`，6790B／`0086ca698868414c6e9ae80f1d48fac9378dc0fc52754cef52c0a3cc11c3ac36`，记录精确argv与catalog绑定的spec Source。此观察不证明控制已启动、已完成、实际batch分布、数值门或性能收益；冻结计划未修改。

## 8.84 第二个B11/C3混合候选数值门通过与比较末核在途（2026-10-01）

北京时间2026-10-01 00:20:35，原session28076实际轮询`cfb28b`仍LIVE。第二混合候选`b11-c3-mix-001`（3组INT8、30组FP16 FFN）的v4第75事件`NUMERIC_COMPARED`已生成，3527B／`f103414017e4692a9e344a06e1e3567bbcb90ff4d96b71ce82ac3a9cab165097`，逻辑head为`ba3e7d9ba61fe80f23c0382f5eba95adf895bcb882e312dff199d06e36aafab1`，prev为第74事件head `3f2dcaaca3517b02eb16c1b1f611b858588b607b5d67fbe4df6eb558f1448fd7`。candidate attempt `ecc1cf3d…`与reference attempt `5d8771a5…`不同，comparison为`ca3e7e1c…`；root有限3事件canonical摘要及身份核对已实际PASS／`7bb8bf`，未重算raw。

实际metrics gate为`PASS/DIAGNOSTIC_PASSED`，覆盖32盘、2004局面。最大白方胜率差为0.87857246个百分点，平均0.01637967个百分点，P95为0.07740855个百分点，门限为6个百分点；Top1一致1998/2004（99.7005988%），合法性2004/2004，最大score差0.70106506目。exact输出为0/2004，不能将数值门通过写成逐位相同。

| 已完成数值事件的候选 | 保存FFN路由 | 最大白方胜率差（百分点） | Top1一致 |
| --- | --- | ---: | ---: |
| 首混合`b11-c3-mix-002` | 6 INT8 / 27 FP16 | 1.36823356 | 1990/2004（约99.3014%） |
| 第二混合`b11-c3-mix-001` | 3 INT8 / 30 FP16 | 0.87857246 | 1998/2004（99.7005988%） |

此表仅比较已经保存的数值结果，不据误差大小选择性能胜者。第二候选性能为`NOT_MEASURED`，FP32为`NOT_EVALUATED`，holdout及生产未通过。CPU26 `compare`结果、driver总结果及controller总结果均不存在，故第75事件的数值裁决不等于CPU比较末核或整组终态。新永久预约仍累计3，继续原session，不重启、不重置消费或截止；29项首次CPU、v5、ABBA及profiler均未执行。

root数值事件观察保存为`target/unified-quant-selection-b11-c3-driver-r1/second-mixed-candidate-numeric-event-observation.json`，4912B／`e0eb598519bc480f7b5284eb2aa0ee2169acc4a03cba0b8fd6db626e734ba440`；ROOT已同步第75事件head，CPU26仍未终态。独立`progress_metrics_check`仅对第73、74、75三事件完成有限只读复核，三文件前后SHA稳定，canonical摘要、前序链、candidate attempt、第73/74事件的execution及第73/75事件的comparison一致，实际gate PASS及指标换算与root一致。第75事件没有execution或self-check字段，`is_reference_self_check=false`来自第73事件登记，不能说第75事件直接提供该字段。本次未重算raw或全部引用文件，未据此证实CPU26末核、整组终态或性能。

续更：北京时间2026-10-01 00:28:19，原session28076轮询`cb1872`仍LIVE。第二候选`b11-c3-mix-001`的`target/unified-quant-selection-b11-c3-driver-r1/run/history-run/cpu-000026-compare/result.json`现为PASS，1292B／`cd647d817684734ac3d650de6de377b10f022a49130a86d4490c9ca46695047a`，`inventory_checks=2`、history来源数81619、error为null，原started `30ead423…`与source_set `9904b46b…`一致。driver事件0012为`NUMERIC_VALIDATED`、gate PASS，654B／`b51f8e00d0e2da78034beda3b206cb6c1851cb888ddf077e25377b190212183c`，逻辑SHA为`50d2dec88177888fefdc6aba1b87d902b0bed6a4233cfb138852a613100961d3`；execution `375d02df…`、comparison `ca3e7e1c…`及registry head `ba3e7d9b…`与第75事件绑定。由此记录比较末核及driver数值阶段完成，不补写独立OS退出证明。

同一`run/history-run`中的CPU27 `check`已启动，started227B／`32dd7e210cf9a1eb8b74bd7cb3f4395e6c4e37dbca51090564116621a761462c`，整组仍未终态。root已保存`target/unified-quant-selection-b11-c3-driver-r1/second-mixed-candidate-numeric-complete-observation.json`，2570B／`c20f2c02359df3b5c858bfcf613045f508afdeba912c35f5a67bbd56d4662652`。独立`progress_metrics_check`仅对CPU26 result、原started、driver0012及v4第75事件四文件完成有限复核，前后SHA稳定、相关canonical摘要一致；未复算raw或全部来源，CPU result不是独立OS退出证明。

当前已完成FP16参考自验及两混合候选数值阶段，剩余旧控制、第三混合候选及整组、完整性能、holdout、生产仍未完成。新永久预约仍累计3，继续原会话，不重启、不重置消费或截止；29项首次CPU、v5、ABBA及profiler均未执行。

## 8.85 旧INT8 q64控制永久预约与预约后检查在途（2026-10-01）

北京时间2026-10-01 00:52:25，原session28076轮询`c18b2d`仍LIVE。旧控制`b11-legacy-int8-q64`／Prepared命令`003-collect`已写入v4第76事件`ATTEMPT_RESERVED`，1772B／`d56d53af969c587a9d6d6a12af0f8cc8340dc08cdff5bed306ec639b0cd70ab5`，逻辑head为`b414112424fe4c3202a819bafa4e72fdb89f92287071d53df44870be29445540`，prev为第75事件head `ba3e7d9ba61fe80f23c0382f5eba95adf895bcb882e312dff199d06e36aafab1`。reservation为`997180f6d927adf30f0fbb675ac438e8e83e36bd38ba0c620d8d3ccac165d8eb`，attempt `a8fbee0d…`、execution `4142fbfb…`、comparison `7a7d08ff…`、reference `d71c3cea…`；非自比较，maxattempt=1。root有限canonical核对及第67、70、73、76四个已知COLLECTION预约事件核对通过，新增永久预约累计4，不能因后续尚未采集而视为未消费。

CPU29 `reserve`的result为PASS，1292B／`9b2d1e87dd5f17eafc867ea1b2448bd9023e743891f3f75fe2fdc24757571ee8`，`inventory_checks=1`、history来源数81619、error为null。CPU30 `reservation-head`已启动，started227B／`b3a7f30a9c86e4df356384cf5056c4f265847f21c401b493aacb0f84fef609a0`，尚无该步骤终态、实际采集启动或采集结果，也无整组终态。该控制的静态配置为backend `cudaint8backend`、batch8、nnMaxBatchSize8、capacity32、window32；这些是配置值，实际batch分布尚未观察。

root已保存`target/unified-quant-selection-b11-c3-driver-r1/legacy-int8-control-reservation-observation.json`，3876B／`fccd6df27a2e5eecbe2f8a66f8f902e1f989dcbc7dee9c9a922dc10e7a923b28`，并同步ROOT当前进度。独立`progress_metrics_check`仅对第75、76事件、冻结Prepared、catalog及CPU29 result五文件完成有限只读复核，前后SHA稳定；第76事件registration与catalog对应项完全相等，Prepared `003-collect`匹配。未读取其引用文件或raw，不据此证明CPU30终态、独立OS退出或实际采集已经发生。

FP16参考自验及两混合候选数值完成证据保留；整组、完整性能、holdout及生产仍未完成，29项首次CPU、v5、ABBA及profiler均未执行。继续原session，不重启、不重置永久消费或截止。

续更：北京时间2026-10-01 01:01:06，原session28076实际轮询`481db4`仍LIVE。CPU30 `reservation-head`已PASS，result1301B／`53799d835599c057c07a8ff9b793ec04d05b5cf9a51fb2a09ecb98e233415fbd`，`inventory_checks=1`、history来源数81619、error为null。driver事件0013为`ATTEMPT_RESERVED`，4670B／`7c81570f8beeed3e4717b3eb593880e748afb9a9bd8c210e976037bb6baec27f`，逻辑SHA为`160baebcd57e15c635b228ba51d985bb1bd8f794b17872a44748993f78a05f96`，prev为driver0012逻辑SHA `50d2dec88177888fefdc6aba1b87d902b0bed6a4233cfb138852a613100961d3`。receipt为`RESERVED_BEFORE_EXTERNAL_LAUNCH`，`consumed=true`、`retry_allowed=false`、`execution_started=false`、`publish_allowed=false`；head `b4141124…`及reservation `997180f6…`与v4第76事件完全绑定。此处证明启动前预约确认，不是实际采集启动。

CPU31 `check`已启动，started227B／`5cb4cc2479d6bf1a11eafae5070f4ce7437a8d3ea71a799fc97f98ada24039d1`，尚无result或driver0014，未观察实际采集启动或整组终态。root保存`target/unified-quant-selection-b11-c3-driver-r1/legacy-int8-control-prelaunch-observation.json`，2519B／`cb729024e731ff029cf722b7294824aa5b7716e72fe01d8fc39a32d71ea4e8b4`，ROOT当前进度为启动前检查在途。

独立`progress_metrics_check`有限复核CPU30 result、driver0012/0013、冻结Prepared及v4第76事件五文件，前后SHA稳定；driver链canonical摘要、receipt.reservation与第76事件payload、command与Prepared `003-collect`逐字段匹配。未读started原件、raw或引用文件，不据此宣称实际启动或独立OS退出。新增永久预约仍累计4，参考及两混合候选数值完成证据保留，完整矩阵、性能、holdout及生产仍未完成；继续原session，不重启、不重置消费或截止。

采集完成续更：root观察`target/unified-quant-selection-b11-c3-driver-r1/legacy-int8-control-collection-exit-observation.json`为3326B／`e3e41b9f8d62ce0bf86e1f08c381704f3690251fe6611bb28055cd329537b592`，记录旧`b11-legacy-int8-q64`／`003-collect`真实采集完成。driver0014启动事件2758B／`02fb898ff8331b9d91d891b1f5ba07cca89a0f7de4353c4af05dadbd56c00576`之后，driver0015为`PROCESS_EXITED`、returncode0，983B／`a1c0d57a05101fe9ad9c0c7c48d86d7085fd0f15358cc1a79bdbee25cead51e4`。采集report位于`target/unified-quant-restored-selection-mixed-prepared-r1/driver-roots/b11-c3/collections/03-4cd88c59f6565930/report.json`，9678B／`ab2d8df52180cc1b66fbcfa5c5e0fcf416626402c114e97fd08e03f1aa04373a`，状态仍为`COLLECTED_UNVERIFIED`。32盘2004/2004请求完成，最终heartbeat记录`nn_rows=2004`、`nn_batches=252`、`failed_requests=0`、`in_flight=0`，`drained=true`；这些是本次采集与排空证据，不是旧控制的候选数值比较或性能认证。

CPU31启动前检查已PASS，result1290B／`917866c13a99bfe631e51c15176867f6144faca6063fc614035122ad151036b0`；CPU32采集后检查已PASS，result1290B／`50cd942227de8a1c07372d86acd7cfa046e9bffdcfc26898db54dd7bba0ee32a`。CPU33 `ingest-collection`已启动，started227B／`84a91e946b6c47e607ff2913a36db8ceca306efc8996e4719c5a00aa86adcce2`，尚无result。北京时间2026-10-01 01:19:48（UTC 2026-09-30 17:19:48），原session28076实际轮询`848b0d`仍LIVE；root有限metadata观察`08376a`确认CPU33未终态、v4第77/78事件和driver0016及总result均不存在，未观察CPU34启动。

独立五文件有限复核覆盖driver0013→0014→0015、report及CPU32 result，前后SHA稳定，driver链canonical摘要和同一reservation一致；未读raw、独立Worker PID退出凭据或CPU33终态，不把launcher退出记录扩写成独立Worker退出证明。新增永久预约仍累计4，FP16参考及两混合候选既有数值PASS保留；旧控制尚未数值比较或性能认证，无整组终态，29项未来CPU、v5、ABBA及holdout未执行，生产未通过。继续原session，禁止重启或改动消费与截止。

采集事件续更：北京时间2026-10-01 01:25:58（UTC 2026-09-30 17:25:58），原session28076实际轮询`1d74da`仍LIVE。root有限metadata观察`7bda54`及canonical核对`679941`确认v4第77事件`COLLECTION_OBSERVED`已生成，2,024,623B／`93789ec5b282272d106e5ff8d577d95495ad55d6e4c5bd9cd90ce3beec834e37`，逻辑head为`84d2718b8fa3bc52de9bbb55fc718ace74dd447e62ab9b22b43eebcd905bf01e`，prev为第76事件head `b414112424fe4c3202a819bafa4e72fdb89f92287071d53df44870be29445540`。同一旧INT8控制attempt `a8fbee0d…`、reservation `997180f6…`、execution `4142fbfb…`及numeric workload `8d260e4a…`对应，report `ab2d8df5…`与实际legacy执行身份`rustgo-legacy-evidence-v1:9bd863564f0619668b8bb744e0272fc7082933fb90573f76f7cff761481cce29`绑定。

第77事件记录original_files与evidence各4027项、runtime_sources 14项，validation为`RAW_PROTOBUF_AND_EXECUTION_IDENTITY_VALIDATED`，numeric_gate为`NOT_COMPARED`。`process_start_after_reservation`仍为`NOT_PROVEN_BY_COLLECTION_REPORT`：旧控制报告不证明进程在预约之后启动，也没有restored路径的`actual_child_exit` Source，不能以另一条路径的证明补齐此缺口。

root观察`target/unified-quant-selection-b11-c3-driver-r1/legacy-int8-control-ledger-event-observation.json`为2822B／`de79d02818ffc3586dc94bf59598acb0a8895eaed84c2018a3b917e3e3294b76`。独立三文件有限复核仅覆盖第76、第77事件及report，前后SHA稳定，canonical摘要、链及身份核对通过，未读引用或raw。CPU33 result仍不存在，本次新增事实是事件写入，不是CPU33末核完成；旧控制尚无数值比较或性能终态，整组未终态。新增永久预约仍4，29项未来CPU、v5、ABBA、profiler及holdout未执行，继续原session，不重启或改动消费与截止。

入账完成续更：北京时间2026-10-01 01:32:17（UTC 2026-09-30 17:32:17），原session28076轮询`74cbaa`仍LIVE；root有限metadata `ccc7c8`及只读`d8df14`确认CPU33 `ingest-collection`的result为PASS，1302B／`6bd6c2682c36c9e0166021256e2c7c9d75df6c816bd524854c34d2a5e7930fe1`，`inventory_checks=2`、history来源数81619、source_set为`9904b46b50e43fe469c2a82d5f741561bd96255e97aeef70d25ea94294206799`、error为null。CPU34 `compare`已启动，started227B／`61c6e74846a04538b889686f8087a2b2f939ac722a1a7d45dbd8dc84a80552eb`，尚无result、第78事件或整体终态。root观察`target/unified-quant-selection-b11-c3-driver-r1/legacy-int8-control-ingest-complete-observation.json`为1750B／`73867ad18f00eb1e33f0eb8890a6990b93411d144718a5035f0d77c2d43d55ee`。独立三小文件有限复核前后SHA稳定，started Source精确匹配；未读引用或raw，CPU33操作result不是独立OS退出证明，旧replay两次exit不能代替新CPU33退出。第77事件及其`numeric_gate=NOT_COMPARED`不改写；新增永久预约仍4、旧消耗完整继承，参考及两候选既有数值PASS保持，整体、性能及holdout未通过。继续原session，不重启或改动消费与截止。

## 8.86 旧INT8 q64控制selection数值事件与比较末核在途（2026-10-01）

北京时间2026-10-01 01:42:14（UTC 2026-09-30 17:42:14），原session28076轮询`0f0bbd`仍LIVE。root有限metadata `8d9e9d`及只读`c742ef`确认v4第78事件`NUMERIC_COMPARED`已生成，3515B／`493900b244a26ee5a88adf35aaa1f02ea1783392c418dc7a3c8526615f326bb7`，逻辑head为`964dbe10a57ba7ec14dd200783d5e7837939933bed00fe528a1a792246bbdaf4`，prev为第77事件head `84d2718b8fa3bc52de9bbb55fc718ace74dd447e62ab9b22b43eebcd905bf01e`。比较键为`rustgo-planned-pair-v2:7a7d08ff4cc760ae4511fe7acfe91e8f3003157a9fd11ae4dbd111e5b13c2403`；共享统一FP16 reference attempt `5d8771a5…`与旧INT8 q64 candidate attempt `a8fbee0d…`不同。非自比较标志来自第76事件登记，第78事件没有该字段，不能说第78事件直接提供。

实际metrics状态为`DIAGNOSTIC_PASSED`、experimental gate为`PASS`，覆盖selection的32盘、2004局面。白方胜率绝对差最大2.85074413个百分点、平均0.10202194个百分点、P95为0.54038167个百分点，门限为6个百分点；最大score差1.05678844目。Top1一致1976/2004（98.60279441%），合法性2004/2004，nonfinite及policy legality硬门均PASS、failures为空。exact输出0/2004、`all_exact=false`，数值门通过不代表逐位相同。此处仅为selection诊断，`original_fp32_gate=NOT_EVALUATED`、`playing_strength=NOT_MEASURED`、`performance=NOT_MEASURED`、`independent_holdout_eligible=false`、`production_certified=false`。

root观察`target/unified-quant-selection-b11-c3-driver-r1/legacy-int8-control-numeric-event-observation.json`为4970B／`038ecb13604c0fff6d97f5cd2536f8b88dded885b91b978fbe3565e84fff7854`；root第77/78事件canonical核对`ff158e`通过。独立第76、77、78三事件有限只读复核前后SHA稳定，身份及canonical链一致；未读引用文件或重算raw指标。CPU34 result、driver0016及controller/driver总result仍未见，不能把数值事件PASS写成CPU34末核完成、独立OS退出或整组终态。新增永久预约仍4，29项未来CPU、v5、ABBA及profiler尚未执行；原进程继续，不重启或改消费与截止。

比较完成续更：北京时间2026-10-01 01:49:48（UTC 2026-09-30 17:49:48），原session28076轮询`19d467`仍LIVE；root有限metadata `899ab7`确认CPU34 `compare`完整返回PASS，result1292B／`0df43297b45a2e5a8b10033ec48920a437e1fcbdfb2eb83cdeb123730aab6feb`，`inventory_checks=2`、history来源数81619、source_set `9904b46b…`、error为null。driver0016为`NUMERIC_VALIDATED`、gate PASS，654B／`65f93f19473f320ac72c1ce2a8acbda6fec4ac12ea890789010df996ef58f62b`，逻辑SHA为`012f38e1ba51f1bbcf73b0d75c6379e61b04b1638b5aa68de0b7eaa3adb9b47b`，绑定v4第78事件head `964dbe10…`；CPU35 `check`已启动，started227B／`e525c03d3434081139fe0126b217d99984e2ca1c80932ee0cfed440243899691`，尚无整体result。root完成观察`target/unified-quant-selection-b11-c3-driver-r1/legacy-int8-control-numeric-complete-observation.json`为3794B／`aa0a19d6b323cf1f3a253eea487baaf9abee56faa9086b373d04c565cf26a633`；独立CPU34 result/started、driver0015/0016及v4第78事件五文件有限复核前后SHA稳定、canonical链与身份一致，未读引用或raw，CPU操作结果不是独立OS退出证明。当前已完成参考、两混合候选及旧INT8数值阶段；冻结Prepared `b86616f7…`已核对下一项`004-collect`／`b11-c3-mix-003`的身份，其11 INT8/22 FP16 FFN为静态保存路由观察，尚未观察新预约或启动。新增永久预约仍4，第三混合、旧FP16、全组、其余负载、性能及holdout尚未完成，不据数值完成授予生产资格；继续原session，不重启或改消费与截止。

## 8.87 第三个B11/C3混合候选永久预约与预约后检查在途（2026-10-01）

北京时间2026-10-01 02:14:49（UTC 2026-09-30 18:14:49），原session28076轮询`471342`仍LIVE。第三混合候选`b11-c3-mix-003`／`004-collect`已写入v4第79事件`ATTEMPT_RESERVED`，2069B／`c3c7c402a6624120db7a4047afadae7b5dba31c608d71eceea383756179d1542`，逻辑head为`d7276090b03161a84240940b87622d5909b42490c360620d53af90ab39d72cb8`，prev为第78事件head `964dbe10a57ba7ec14dd200783d5e7837939933bed00fe528a1a792246bbdaf4`。reservation为`12e391132ef6b1c66d5d204378eb4e939325028a07115f27ae9ce48010edf928`，attempt `bbb5e6a8…`、execution `47d9ddcd…`、comparison `c6c45bd3…`；非自比较、maxattempt=1。注册Source为`target/unified-quant-restored-selection-mixed-registrations-r1/registrations/b11-c3-mix-003.json`，1,900,711B／`e3520f7e0a7cce72db69e17a77b099c638157c25741480f6babfb529b76f0fc9`。

CPU37 `reserve`已PASS，result1292B／`4b241d655b95b212494758187e6584e388be39224ecd239f826d2e4e1d89c11d`，`inventory_checks=1`、history来源数81619、error为null。CPU38 `reservation-head`已启动，started227B／`ca5c32fc9eccd416952de710e47fb64e1af1b0bab4b6a5eb4bd91890d4535a3b`，尚无终态或实际采集启动。root有限canonical及第67、70、73、76、79五个已知预约事件核对，确认新增永久COLLECTION预约累计5，旧消耗继续继承；预约不等于实际启动、模型上传或GPU工作。

root观察`target/unified-quant-selection-b11-c3-driver-r1/third-mixed-candidate-reservation-observation.json`为6190B／`d530292a23ba5c26c3b59e2eb676176d056fa6b57cfae8a0a5906807ec2f305f`。独立第78/79事件、冻结Prepared、catalog及CPU37 result五文件有限复核前后SHA稳定，registration与catalog对应项完全相等，Prepared `004-collect`身份匹配；未读引用文件、raw或started原件，不据此证明GPU启动、CPU38终态或独立OS退出。候选B3的11 INT8/22 FP16 FFN是此前保存包的静态路由观察，不是本次运行trace或性能测量。

参考、两混合候选及旧INT8数值完成证据保留；第三混合、旧FP16、整组、其它负载、性能及holdout仍未完成，29项未来CPU、v5、ABBA及profiler未执行。继续原session，不重启、不重置预算、消费或截止。

启动前续更：北京时间2026-10-01 02:23:35（UTC 2026-09-30 18:23:35），原session28076轮询`493391`仍LIVE。CPU38 `reservation-head`已PASS，result1301B／`c85276159e15fb053f946e9b62331823d535b4b996c12587b1373973b44bcac6`，`inventory_checks=1`、history来源数81619、error为null；driver0017为`ATTEMPT_RESERVED`，4279B／`e2efdf426adb61c1b3bd5ac14afd0cde46ff36a153aba5313306b93764dbec8b`，逻辑SHA为`4d6b3ac82989f0bf9242a813de0620ef84c725c55235ace8b17d69d6a18b2238`，prev为driver0016逻辑SHA `012f38e1…`。receipt为`RESERVED_BEFORE_EXTERNAL_LAUNCH`，`consumed=true`、`retry_allowed=false`、`execution_started=false`、`publish_allowed=false`，绑定v4第79事件head `d7276090…`及reservation `12e39113…`。CPU39 `check`已启动，started227B／`4167db3b10d9d8b211b57a638eb5ffd9fa160a27a8e9f99a362ff4922735f6b9`，尚未观察其result、driver0018或实际启动/整组终态。root观察`target/unified-quant-selection-b11-c3-driver-r1/third-mixed-candidate-prelaunch-observation.json`为3444B／`e1dcaff36384aeaa53c6878d728ad4afedbc925f21b81cbee8d8592f40503aeb`；独立CPU38 result、driver0016/0017、Prepared及第79事件五文件有限复核前后SHA稳定，canonical链及receipt/command/artifacts逐项相等，未读started原件、raw或引用文件，不是CPU39终态或独立OS退出证明。新增永久预约仍5，参考、两混合候选及旧INT8数值完成证据保留，第三混合、旧FP16、完整矩阵、性能、holdout及生产未完成；继续原session，不重启或改消费与截止。

## 8.88 第三个B11/C3混合候选原生采集启动（2026-10-01）

第三混合候选`b11-c3-mix-003`的CPU39启动前检查已PASS，result1290B／`cd418c12a81d1d7572d5f007760c50eab0ec98e9701a3a50b95614a326f971e4`。driver0018为`ATTEMPT_STARTED`，2050B／`998bd7746105129f22a7003ba946eaa9c4a0be622c8dbe6474dadeede47ac30c`，逻辑SHA为`c21e499076f131d1d7c3a8413796f3e8f1ec9e9c4c69dd904425a01d371d398c`。该driver事件先于launcher Popen，单独不能证明进程或GPU执行；实际原生启动另由`collection-launch.json`及root既有精确PID观察支持。

native launch为2291B／`711bec1c64bb986541b43e620d0108c6acdc574e44a2efa2096b643222082b7e`，记录PID46672，冻结exe SHA为`dd665d03fe31739712a205eaa6f63852dbc27fee339dfe91f38067237e57bfdc`。root精确PID CIM观察`9ff9c4`记录parent63824、creation `2026-09-30T18:30:36.3723580Z`，镜像路径与launch一致；root不持有native进程句柄，这不是退出证明。后续已生成的`collection.claim.json`1772B／`241f4e60e7dc945d6730761184be224e1a4bd4889b6091fdce84d0ca90c3b713`及supervisor `attempt.json`644B／`f04a7ec00391b7eff975c3e4d4b29619e33355cfd6a55ab8055aaf823d2f4c23`均为`CONSUMED_NO_RETRY`，绑定原registration `e3520f7e…`、reservation `12e39113…`及`004-collect`，不得重做已消费入口。

root观察`target/unified-quant-selection-b11-c3-driver-r1/third-mixed-candidate-started-observation.json`为3584B／`146f69077e4e97c2ea176c2981706ca1ddb3a5ce7c38cf4dd140a06e04249be5`。独立五文件有限复核覆盖driver0017/0018、native launch、supervisor attempt及CPU39 result，前后SHA稳定，canonical链、registration及parent绑定一致；未读claim、report或查询OS，不能把该有限复核扩写成独立原生退出观察。

最新北京时间2026-10-01 02:36:20（UTC 2026-09-30 18:36:20），原session28076轮询`d54e9b`仍LIVE；root有限metadata `0ac119`未见native退出、report、stage、driver0019、CPU40及整体result。新增永久预约仍5，原参考、两混合候选及旧INT8数值完成证据保持；第三候选GPU推理完成、数值、性能及holdout均未认证。继续原session，不重启、不重做已消费入口或改动消费与截止。

采集完成续更：第三混合候选`b11-c3-mix-003`的native PID46672已由supervisor持有并观察exit0，`process-result.json`为1029B／`1b4adc78262b15e79ed32149909a76f134481bdafcfd8924fe560a09862f74e5`；driver0019为`PROCESS_EXITED`、returncode0，982B／`c83bf603632e02590ca6daf4f36409dfa7873f85226eeb404c0caa2ddd882aad`。本次完成32盘2004局面、668个物理B3，保存4009输出文件、41,789,924字节；实际inference profile为`rustgo-quant-v1:7cac2d6e5b89fbe51e956bdfa0ccc76a149718af587c5069a1b25eeda137f3d3`，与catalog匹配。report为7713B／`e76eadbee4a4a478e7a347432f09744a24885513c4b7c5e4eb885188f264aa10`，状态仍为`COLLECTED_UNVERIFIED`，不把采集退出0或文件计数写成数值、性能通过。root观察`target/unified-quant-selection-b11-c3-driver-r1/third-mixed-candidate-collection-exit-observation.json`为3423B／`74127efa78dc5ce5c5133384856fa13c121bec76c90b91919bf7bbf7707b5c65`，记录原session28076轮询`fb0553`仍LIVE；最新同session轮询`4e9e11`仍LIVE，随后clock工具记录UTC 2026-09-30 18:50:33（北京时间2026-10-01 02:50:33）。独立driver0018/0019、native process result、report及catalog五文件有限复核前后SHA稳定，仅验证canonical链、身份/profile与计数，未复算raw、扩展读取引用或查询OS。第三候选数值与性能仍未通过，整组未终态；保留原启动观察及参考、两混合候选、旧INT8的既有数值结果，继续原session，不重启或重做已消费入口。

采集后检查与事件续更：第三混合候选的CPU40已PASS，result1290B／`f8dafb2381c2c062892d728fb70639a0355f533969e07f469a9213dc97c17928`，`inventory_checks=1`、history来源数81619、source_set `9904…`；root postcheck观察为1659B／`5d7c44708b447088625484defb83cec78c7c91e16d30bd7ab325a1b376a0a0b2`。原session28076轮询`0b5b27`仍LIVE，随后clock记录UTC 2026-09-30 18:59:56（北京时间2026-10-01 02:59:56）；root有限metadata `719f6a`确认v4第80事件`COLLECTION_OBSERVED`已生成，6,184,911B／`33198de150575ef4718cc2db28586a3561bc60803b252a1c6c0cd1b2f8acf835`，逻辑head为`d1202041cab5fc830561cce5d940b25aaae0e47054820956c458a2582931e57a`，prev为第79事件head `d7276090b03161a84240940b87622d5909b42490c360620d53af90ab39d72cb8`。validation为`RAW_PROTOBUF_AND_RESTORED_EXECUTION_IDENTITY_VALIDATED`，numeric_gate为`NOT_COMPARED`，记录9个original_files、19118个runtime_sources及9个evidence。root已保存`target/unified-quant-selection-b11-c3-driver-r1/third-mixed-candidate-ledger-event-observation.json`，2579B／`0f7a96540016e89a5c1b22a1253fa0d0d129270a649e748a6c124b7989f22ec8`；独审仅对第79、第80事件及report三文件完成有限复核，前后SHA稳定，canonical链与身份、report/profile/registration/child-exit Source一致，未跟读引用或raw，不证明CPU41末核、数值、性能或整组终态。CPU41 ingest result及第81数值比较事件仍未见，事件写入不等于入账操作终态或数值比较通过；整组未终态，新增永久预约仍5、新ABBA为0。继续原session，不重启或改消费与截止。

入账完成续更：CPU41 `ingest-collection`已PASS，result1302B／`99c73c3aa9178c8fc14ccaedd1f65b4a06955d61fc99865e5d778b5991e38b8a`，error为null、`inventory_checks=2`、history来源数81619、source_set `9904…`；CPU42 `compare`已启动，started227B／`ccbed92f4a8f7b7610db5fb46f88b9d80da4319e5617cf33440cb79918b553cc`，数值结果尚未观察。原session28076轮询`1c9535`仍LIVE，随后时钟记录UTC 2026-09-30 19:08:20（北京时间2026-10-01 03:08:20）；root有限metadata为`fd261f`，观察`target/unified-quant-selection-b11-c3-driver-r1/third-mixed-candidate-ingest-complete-observation.json`为1881B／`c1adcd509ca1f98e7ac247454d75d265d8b5fa0adbcb6745a329aef7e902c341`。独立复核仅覆盖CPU41 result/started及CPU42 started三小文件，前后SHA稳定、绑定匹配，不把历史replay退出当成新OS退出，不证明数值、性能或整组终态。新增永久预约仍5、新ABBA为0，继续原session，不重启或改消费与截止。

## 8.89 第三个B11/C3混合候选数值事件与比较末核在途（2026-10-01）

原session28076轮询`595adc`仍LIVE，随后时钟记录UTC 2026-09-30 19:19:18（北京时间2026-10-01 03:19:18）。root有限metadata `e4212b`／`899b38`确认v4第81事件`NUMERIC_COMPARED`已生成，3515B／`9f290e11a96620b4da235baf4033c070107699bf5c477d67463e6bc3dab70fda`，逻辑head为`92f89418a707c40f10e53790e98f7f08851b4191a04de8c69cd6aaf8e838e966`，prev为第80事件head `d1202041cab5fc830561cce5d940b25aaae0e47054820956c458a2582931e57a`。第三混合候选为`b11-c3-mix-003`，此前保存路由为11组INT8、22组FP16 FFN；数值事件覆盖32盘2004局面，状态为`DIAGNOSTIC_PASSED`、gate为`PASS`。

事件记录的白方胜率绝对差最大值为2.1202921867个百分点、平均0.0576440670个百分点、P95为0.2727955580个百分点，门限为6个百分点；score绝对差最大1.40966796875目、平均0.032250103293目、P95为0.089168548584目。Top1一致1987/2004（99.1516966068%），finite及policy legality硬门均PASS，exact输出0/2004，非逐位相同。这些是已保存事件中的数值，不是本次独立raw复算；不能笼统称所有指标优于旧INT8控制，例如第三候选最大目差约1.410目，高于旧INT8的约1.057目，也不能据此选择性能胜者。

root观察`target/unified-quant-selection-b11-c3-driver-r1/third-mixed-candidate-numeric-event-observation.json`为5395B／`148acb63fe7e2ddc2b6dd05d207f7eb7993895038fe2864fb38e08d1b1448995`。独立第79、第80、第81三事件有限复核前后SHA稳定，canonical链及身份匹配；非自比较标志来自第79事件登记，不能写成第81事件直接提供。未独立复算raw，CPU42末核、driver0020及整组result尚未见，数值事件PASS不等于比较操作终态、独立OS退出或整组完成，更不等于性能、棋力、holdout及生产通过。旧FP16仍未完成，新增永久预约5、新ABBA为0保持；继续原session，不重启或改消费与截止。

比较完成续更：原session28076轮询`8cc0d5`仍LIVE，随后时钟记录UTC 2026-09-30 19:27:28（北京时间2026-10-01 03:27:28）；root有限metadata `14387e`／`e49e54`确认CPU42 `compare`实际PASS，result1292B／`619ed4a5e04287466e579baeb61c3652e594e0a18c36bd2ba0e5ba07a4b7d04e`，`inventory_checks=2`、history来源数81619、source_set `9904…`。driver0020为`NUMERIC_VALIDATED`、gate PASS，654B／`d8225cb312eb5d8343b17908710423836425042f1e6fbbac706677cc8ea7431f`，逻辑SHA为`bb7d56879156166fa7c6e4a58e06018ed1389868916e7c3081089f2842c81d94`，绑定第81事件head `92f89418…`及comparison `c6c45…`；CPU43 `check`已启动，started227B／`603df40221d438b34881e50c1f9331245b32c5a11a89ad69bb63278bb5404943`，整组未终态。root观察`target/unified-quant-selection-b11-c3-driver-r1/third-mixed-candidate-numeric-complete-observation.json`为2928B／`2278bde1a49b88d35b647c1300c9065445db19b82ed3935113a84e0790da64b0`。独立CPU42 result/started、driver0019/0020及第81事件五文件有限复核前后SHA稳定，链与绑定匹配，未读raw、引用文件或CPU43，不证明新OS退出、整组完成或性能/holdout采用。当前参考、3个混合候选及旧INT8均有通过数值记录，旧FP16待完成；新增永久预约5、新ABBA为0保持，继续原session，不重启或改消费与截止。

比较后检查续更：第三混合候选的CPU43 `check`已PASS，result1290B／`f5d08ae88255993f20a6ff03c221b975640b696fc882342fa93199927a07a8b6`，`inventory_checks=1`、history来源数81619、source_set `9904…`；旧FP16 `b11-legacy-fp16-c32`／`005-collect`的CPU44 `check`已启动，started227B／`875cc5c9aed1fe1dadbbe8cd697921e5ebdd4bfdb38e4b5f8df6d2552463e41a`，未见第82预约事件、实际启动或整体终态。原session28076轮询`437ed0`仍LIVE，随后时钟记录UTC 2026-09-30 19:36:09（北京时间2026-10-01 03:36:09）；root观察`target/unified-quant-selection-b11-c3-driver-r1/third-mixed-candidate-numeric-postcheck-observation.json`为1973B／`05820b25d14488c416b40767dafc03b4a7c66c0b79d924a21e9608386d253d58`。独立CPU43 result/started及CPU44 started三文件有限复核前后SHA稳定、绑定一致，未跟读引用或raw，历史exits不作新OS退出证明。现有参考、3个混合候选及旧INT8数值通过记录保留，新增永久预约5、新ABBA为0保持；继续原session，不重启或改消费与截止。

旧FP16检查续更：CPU44 `check`已PASS，result1290B／`0ed3692fc6d0b3cff66c1dbf9d6a32f3e0bedfbbc4b6b9bed393aaa6f4dea956`，`inventory_checks=1`、history来源数81619、source_set `9904…`；CPU45 `reserve`操作已开始，started227B／`55a6feba7ff05d2a4e9da10aae0cd5662110a3ebcb4ea4938ff1b60cda1de476`。这不构成已确认的永久预约消费，第82事件及实际启动尚未观察，新增永久预约仍5、新ABBA为0。原session28076轮询`7a594d`仍LIVE，随后时钟记录UTC 2026-09-30 19:44:16（北京时间2026-10-01 03:44:16），root有限metadata为`f87bed`；root观察`target/unified-quant-selection-b11-c3-driver-r1/legacy-fp16-control-precheck-observation.json`为2296B／`18d3ffcfb6d5df6d6df32f038809abd046b6ba8d89a29ab59cb7da55c0a415b9`。独立复核仅覆盖CPU44 result/started及CPU45 started三小文件，前后SHA稳定、绑定匹配，未审第82事件，不宣称预约已消费，也不代表新OS退出或整组终态。继续原session，不重启或改消费与截止。

## 8.90 B11/C3 首组达到工作截止后的部分终态（2026-10-01）

原session28076已由同一工具句柄返回实际exit1（chunk `68b5cb`），北京时间2026-10-01 03:48:07。controller结果为`FAIL_STOPPED_NO_RETRY`，耗时28746.437秒；run阶段达到冻结的绝对work deadline，watchdog触发。原入口已经终态，不再轮询、重启或重做prepare；此前§8.89的LIVE记录仅代表当时观察。

三份终态依据为：

- `target/unified-quant-selection-b11-c3-driver-r1/root-run-tool-exit-observation.json`：3332B，SHA256 `f337d878ec3ac7842e150da2e65c14177c93d13875a19ee5675d6a5d9c46e799`。
- `run/result.json`：15171B，SHA256 `83b6b378c483265812cdb5cfb29becc260ecfab36d1680f59edaac956601f4b5`。
- `run/run-exit.json`：3061B，SHA256 `def16e7236cabb3db2cb0a0e85d23fc868ad0f61874ba519bc44d46854fc22d0`。

独立三文件有限复核已确认前后SHA稳定：run Job终态`active_processes=0`、`whole_descendant_exit_certified=true`，cleanup exit为3758096385，无cleanup/job-close错误。正常helper及owned venv退出码仍为null，driver整体result缺失，不补写exit0或完整成功。controller的`sources_unchanged=true`只证明其绑定控制来源末核，不能替代被中断CPU45的81,619份历史来源末核。

本组已有参考、3个混合候选及旧INT8共5份数值通过记录，新增永久COLLECTION预约为5。旧FP16的CPU44 check已PASS、CPU45 reserve存在started而无result；当前v4 events浅目录为66至81、driver events为0000至0020、collections只有上述5目录，未见第82事件或旧FP16目录。完整限定范围的部分账本审计另行保存，尚不据这次浅目录检查宣称所有引用/raw已复核。数值通过不代表整组、性能、棋力或holdout通过；新ABBA为0，其余3组尚未执行。

失败记录、永久锁、已消费尝试、原全局及组截止全部保留。现有下一组freezer只接受完整成功，不能用于当前失败或放宽其成功门。后续需要显式处理部分结果继承与剩余工作准入，不能把新目录当成重置预算或原尝试的理由。

CPU来源核验的并行原型位于`target/unified-quant-source-verify-parallel-static-r1`，默认4、最多8线程，保留每次完整读取、原绝对截止和全部任务join，无缓存。8项合成CPU测试已定义；首轮独立静态复核未见确定阻断，但尚未执行或接入业务，未获得性能数字。受阻OS读取仍由外层owned进程/Job截止强停，不能用线程取消冒称读取已停止。

部分终态独审续更：`target/unified-quant-selection-b11-c3-partial-audit-r1/report.json`已生成，109885B，SHA256 `b35610e9534d8b83e0b4140b9a08055b5ec70920c284abf30ca36d2140c1d449`，状态`PASS_PARTIAL_TERMINAL_EVIDENCE`；实际一次只读审计exit0／chunk `e4b8a1`。204项有限一致性检查均通过，确认v4第66–81事件及driver第0000–0020事件的canonical链、6份Prepared命令绑定、5次预约对应5份采集和数值PASS（含1份参考自比较）。005旧FP16在完整已审新事件范围内没有发布预约或driver启动，采集目录不存在，CPU45无result；全部3把锁及失败原件保留。

审计只浅枚举5个指定目录，读取42份证据、AGENTS和审计实现，未递归、跟读引用、读取raw或81,619份历史库存。42份证据及实现初末稳定；AGENTS在审计期间由root更新，初/中/末SHA及`changed=true`已单列，不能称全部43输入稳定。报告不将原失败改为整组成功、不补正常子exit0、不提供性能/holdout资格或新的时间与重试授权。首次纯stdlib审计命令曾因Windows命令行长度限制在CreateProcess前拒绝，未执行业务；短命令完成的本次审计只读取原件并在新目录保存报告。

## 8.91 并行来源核验的首次有限 CPU 检查（2026-10-01）

为处理selection控制阶段重复来源核验的开销，隔离原型保留原Source的逐次完整读取和原绝对截止，只并发执行独立回调。首轮四文件独立静态复核见`target/unified-quant-source-verify-parallel-review-r1/review.md`，没有发现原型本身的确定静态阻断；原生产Source回调未修改。

首次测试固定8项合成CPU方法、1次owned venv运行、180秒外层上限，复用原Windows Job及stdin gate，未导入selection driver或执行账本操作。实际工具返回exit0（chunk `6a25c0`），controller PID48148，耗时0.953秒；helper PID35540与owned venv PID69616均实际exit0，Job终态active0，无watchdog或清理错误。8PASS、0失败/错误/跳过，15份绑定来源前后稳定。

- `target/unified-quant-source-verify-parallel-tests-cpu-r1/frozen-plan.json`：7450B，SHA256 `4af783ae193538d86a8a7880796ea3d64b12b2496b214245187846702c6dbff7`。
- `run/result.json`：3892B，SHA256 `c95d3de0773dee8eeb177b39379c1f726c72918ce94a00cda345818dbda4ecc9`。
- `run/unit-result.json`：2231B，SHA256 `fbe2226a7514b4db01a7cdd2f27b74a65e4d8fb2ebc2173fb5d10d25ec61cdad`。
- `root-tool-exit-observation.json`：1737B，SHA256 `30e97836a0ce501ed0b143a08120b62ee80859d480075f154fbf4d46d8a22d44`。

覆盖范围为worker/pending上界、成功后join、独立深快照、输出与异常输入顺序、初始和在途截止、取消无法强停已阻塞回调、每次重复完整核验及并发调用隔离。测试回调为合成对象；未读取真实Source清单，不能声称实际文件回调/调用方集成已认证，亦没有读取提速或推理性能数字。

独立审查另指出新CPU适配器的一个潜在缺口：checker和controller在最后结果写入前判断截止，写入/fsync/打印后直接返回。原r1本次实际8项成功、结果及源均保留；`target/unified-quant-source-verify-parallel-tests-cpu-r2`只为这两处增加返回成功前的原绝对截止复核，change记录2247B／`7459e89b7343fc94d1d5d73738b9b417b72df158f2bfd65ebd8b9c066da68960`。r2尚未冻结或运行，不重跑原8项，不把静态修正称为运行资格。写出的PASS仍需外层实际退出与截止证据，不能单凭正文采用。

原session28076保持失败终态，5份已完成数值结果与全部消费/截止保留；本次未创建新COLLECTION或ABBA。后续先验证实际Source回调与最小调用方接线，再处理部分结果的合法继承；不直接修改旧冻结模块、旧来源身份或放宽成功门。

共享控制的静态续接分析：原B11/C3与B11/C8两个catalog绑定同一旧FP16采集attempt `ca0674…`和execution `8cf384…`，但comparison/reference分别不同。由于该共享采集尚未预约，可研究由新显式部分结果消费者将其首次采集放到C8：已完成5＋C8待5＋B15/C3待6＋B15/C8待4仍为原20次；C3余下比较另按原comparison与C3参考核验，不能复用C8比较结论。原C3不重启，后3组原6/8/6小时上限、原28小时全局截止及间隔耗时均保留；额外比较/源核验须另作剩余时间投影，尚未证明可行。现有success-only freezer及其[6,4,6,4]成功前置保持，不能直接消费该分配。静态交接`target/unified-quant-selection-partial-transfer-static-r1/handoff.md`为3646B／`8e76f16d421daca94cc5e0bce6c5b9158694194d5caf60b5cd07eeb377764b48`，来源表2114B／`3f414722112e5d9b5c5cf199eb9a5ee352819771ce2e878e39bf41532d071bc6`；没有新执行计划、时间授权、账本操作或GPU调用。

## 8.92 两处来源核验循环的隔离接线与真实文件检查（2026-10-01）

`target/unified-quant-source-verify-integration-static-r1`保存新facade、原并行helper的精确副本、4项新增测试、handoff、Source映射与精确diff。facade只增加helper import、私有phase包装，并替换`_parent_inventory`末尾及`_active_contract`末尾两处独立Source循环；反向恢复四处diff后与原facade全文相同。原parent/history只绑定原Source而不复制修改，旧contract的来源身份检查保持。

phase要求显式active history operation；传入截止只能等于或收紧该operation截止，前/每个回调前/join后均检查同一绝对截止。每个回调仍执行原`p.source`；不使用返回副本替代原库存，原冲突拒绝、清单顺序、JSON二次读取、计数和摘要保持。没有active operation时拒绝是新消费者的明确前置，不能静默套入旧冻结调用方。

来源表`source-map.json`为6187B／`97342c80e731efda7a55b55b70a3502115d59d9cda24a876f51c55ab76374584`；facade为45678B／`46360f9d46177ee505ad872e8746fdb832fda3fac5635d7fd35ea82a8456e2ab`，4项测试为10705B／`816e03a321da91844b27b4647408eb4d708760c108c4569d45f2a9a89df194f7`。差异文件2889B／`783fd32816c4a24383d5b7faa725d001f9687f36f2f1264231aa4f8ae2c108d4`，原4份Source末核稳定。root已有限静态读取接线与4项测试；不将静态稿描述为完整campaign资格。

新独立CPU适配器继承§8.91尾写后截止修正，仅将测试集合改为以下4项，固定单次owned venv、180秒总上限；原8项不重跑：

- 原Source回调读取空文件、跨64KiB文件、Unicode路径及重复Source，两次完整phase后修改同长度内容，第三次拒绝旧SHA，原输入对象与内容保持。
- Source别名冲突、缺active operation及扩大截止在派发前拒绝。
- parent inventory保留事件检查并向并行phase传递更紧截止；改变合成旧事件会在派发前拒绝。
- active contract使用当前operation截止；真实小文件核验并join后模拟达到截止，拒绝继续，不触发写入哨兵。

首次实际工具exit0／chunk `4d028a`，controller PID69320、耗时0.5秒；helper PID11904与owned venv PID81312均实际exit0，Job终态active0，无watchdog或清理错误。4PASS、0失败/错误/跳过，20份绑定来源前后稳定。结果位于`target/unified-quant-source-verify-integration-tests-cpu-r1`：

- `frozen-plan.json`：8391B／`8b1a513d971ea511e3710195e83503ed23958a30a59e528b1ab59988888ed65c`。
- `run/result.json`：3928B／`4d59f6a1c449db7fcedc5f148da568f4901af016100c2eea8cf978176cf7d544`。
- `run/unit-result.json`：3402B／`a40e3b43d7689ff664561c0fb16f4b6c9f2b942e866a87238bee014da0342eec`。
- `root-tool-exit-observation.json`：1817B／`b84a82c9f516c4834a1e90f67de6510c75afa000b3c981347d99dbd2d38caee5`。

这些检查使用真实临时文件和合成历史/contract夹具；没有重放真实历史、核验完整81,619来源、测得读取加速、运行新selection消费者或新增GPU/账本操作。实际读取收益及新消费者完整身份/截止绑定仍待验证。原B11/C3失败、5份数值结果、全部消费和原全局时间继续保留，新ABBA为0。

## 8.93 来源读取有限诊断与原清单构建单次剖析（2026-10-01）

固定样本诊断位于`target/unified-quant-source-verify-io-diagnostic-cpu-r1`。从原81,619条库存按四个预定尺寸层、精确路径UTF-8 SHA顺序固定选择812条唯一来源，共55,110,863B；93份大于1MiB的来源未入样，不能按此尺寸组成推算全量表现。先串行预热，再固定S-P4-P4-S，每次调用完整原`p.source`，无应用层核验缓存或失败替换。

首次实际exit0／chunk `456a49`，五阶段共4,060次完整核验；controller1.922秒、16份元数据来源前后稳定、Job终态active0。四个计时阶段依次212.3696／237.7356／229.9913／284.0586ms，串行均值/并行均值为1.061363，但串行首尾相对漂移28.8819%。OS/设备缓存未控制，结论为`EXECUTION_PASS_PERFORMANCE_INCONCLUSIVE`，没有采用四线程或宣称全量/推理加速。`run/diagnostic-result.json`为3371B／`15a081cbf02e718ff5f92881b4aa864506645863423adca74967eceaa9a30c33`，根实际退出记录为1588B／`203c4c84581abd24c8b2b7cbe791c63ec28ff874a3020c02dfbe59f15e9bd05d`。

为继续定位，使用原facade和原parent，只调用一次`_records(contract)`，未调用`_parent`、history replay、ledger或GPU。输入为原v4 contract（22,140,533B／`110e92e2d0ee25d92f885abc2182edb78e03f82e8b688aa23112775cf00dfbd8`）。r1已冻结但未执行；静态审查发现控制器第二次读取计划未绑定raw长度/SHA，以及调用前截止失败时空pstats可能遮盖原错误。新隔离r2修复这两点，保留r1原件，不将未执行版本记作失败运行。

`target/unified-quant-source-inventory-profile-cpu-r2`首次session41698实际exit0／chunk `c97218`。controller22.078秒，helper17416、owned venv31208均实际exit0，Job终态active0；16份元数据来源稳定，没有watchdog或清理错误。一次调用返回81,661条规范化来源，摘要`0a826b922c2d73ff3f4c2e03f68855681dc8a7e1a298cc0860fd65bc7a0a6936`。

带cProfile的清单构建耗时21.4719131秒；`path_key`81,874次、累计21.1164609秒，`nt.stat`989,176次、自身耗时9.7600165秒，`nt._getfinalpathname`163,748次、自身耗时3.6721576秒。累计时间相互包含，不能相加；profile包含仪器开销，计时不含输入JSON读取、导入、结果摘要和写出。它定位了该函数的路径元数据开销，未证明整轮8小时的全部耗时来源，也没有验证枚举文件内容或给出优化收益。

- `frozen-plan.json`：7591B／`c3472978020ea81f5710eee84d5b93a4f37368c682cdc8944ea6deec09874106`。
- `run/result.json`：3908B／`e361c4520b792e66775f538f77ac4cca20cecce21f07c6c064c828ec1e1ff196`。
- `run/diagnostic-result.json`：13129B／`159e79a63aa92ce80f1bf75b38df36bdb91892f7bd3721ccdc77bb91811ce0e9`。
- `root-tool-exit-observation.json`：1596B／`4e77bb81c56a1aaf07aedc8788ce72e07132b685de570602c599ad3287950c38`。

部分结果续接方面，`target/unified-quant-selection-partial-consumer-static-r1/select_partial.py`只定义了九份固定输入的纯选择器，保留5已完成+5+6+4的静态投影及C3/C8各自比较身份，未运行或生成计划。进一步源码核对确认Python及实际编译的native父证明均固定原v4路径/版本与前代身份；新namespace不能直接使用旧exe和登记。新消费者的锁所有权及身份兼容协议仍待具体实现，不能仅放宽success-only检查或换目录绕锁。原selection失败、消费、截止、五份数值结果及新ABBA0保持。

续接进一步收敛为保留原v4 contract、路径及reservation身份的显式successor协议，尚未实现执行转移，因此当前没有必需重编native的结论。root只读原三锁，driver guard为266B／`3defd6789a9848b846996a412be00447ec981ee506756c30e55a16f8fa6e2bfa`，绑定原Prepared与PID47840；ledger/run锁均为11B／`b8c983b48d214b6ac6d3149f63a129729cc4ec4dec2889c6db3091896c11e9ec`，内容`pid=47840`。观察`target/unified-quant-selection-partial-lock-observation-r1/observation.json`为1728B／`838ac3620ba8c09e72d567e4e43d9545a9be5e267e841dbafff1f5fccab4f99a`，锁字节本身不证明进程死亡。

随后从本会话原始工具日志找回2026-09-30T12:11:57.977Z的实际`c605fc`输出和对应查询调用，明确观察到55156→47808→47840的父子PID链，两层Python命令均匹配原entry/selfSHA/runtimeSHA/Prepared/history路径。原始两段按日志字节范围提取并前后重读一致；整个持续追加的会话日志没有当成稳定Source。提取记录`target/unified-quant-selection-parent-process-recovery-r1/extraction.json`为2938B／`67770e8c0a3c9a72e424d13670eb40386ee6c26c54f4e4bbf2d228a20fb89790`。结合原helper放行前的Job分配、无breakaway启动代码及终态Job为空，可以建立源码与同期观察支持的归属推导；没有补造内层直接IsProcessInJob、进程出生时间或正常退出码。原锁未改、successor claim未创建，新准入函数及调用方仍需验证。

## 8.94 路径候选有限差分、全清单诊断与所有权证据资格（2026-10-01）

路径候选`target/unified-quant-path-key-optimization-static-r1/path_key_candidate.py`仅将`(p, *p.parents)`改为逐级`.parent`遍历相同数量的祖先，保留顺序、每级检查、完整resolve、扩展/UNC前缀处理与normcase，不减少文件系统查询，不缓存验证结果。原Python311没有`Path.is_junction`，候选保留原fallback行为，不能声称新增了对全部junction的拒绝。

首次6项差分CPU检查actual exit0／`1f054e`，controller0.359秒，helper19188/owned venv36252均正常exit0，Job终态active0；6PASS/0skip，15份来源稳定。真实临时普通文件/目录/缺失路径fixture保留；UNC、symlink、junction和错误传播使用模拟路径类，不当作真实网络共享或重解析点测试。结果在`target/unified-quant-path-key-tests-cpu-r1`：`run/result.json`为3769B／`d4c1406b41dc2200ebc23b2819248f63a0c32bd6c2a5aad24ddfcafa1394109d`，`run/unit-result.json`为2582B／`f0b1ec47b1f67e0a42973be3936815725dea805b0083cd64998bfe15df3b4f32`，根退出记录1288B／`753f77ea84473302ae4ab1f7e72927660b2b3d90679c68fe7ba4ebc5f3db4b3e`。

随后只做一次固定原contract的清单构建对照：原函数预热一次，再A1/B1/B2/A2。B通过独立FunctionType globals适配路径函数和same_source调用，原parent模块不修改；计时包含B适配开销，两侧均只计`_records`，摘要、输入解析、导入和写出在计时外。每阶段返回81,661条，完整清单摘要均为`0a826b922c2d73ff3f4c2e03f68855681dc8a7e1a298cc0860fd65bc7a0a6936`。

首次session98051实际exit0／`820cb6`，controller75.688秒，helper67988/venv82020正常exit0、Job终态active0，17元数据来源稳定。预热14.8812588秒；四阶段14.9720811／14.7271770／14.9604119／15.2800558秒。描述性原均值/候选均值1.0190163；A首尾漂移2.0361%，B漂移1.5713%。无profiler，但OS缓存和系统噪声未受控；结果仅为`MEASUREMENT_PASS_SMALL_EXPLORATORY_DIFFERENCE_NOT_ADOPTED`。这点差异不足以解释或解决整轮超时，不重复本基准，不采用候选或推断推理收益。

诊断目录`target/unified-quant-path-key-measure-cpu-r1`的`frozen-plan.json`为8203B／`128dc5997cdc58ae1743ccbc4669a19d6e13ab4e5cdfe427fb91804401c509cf`；`run/result.json`为3836B／`6b01b8cb9c7dd57caed8d8a1bbf373ee48b64bf319a9291681a2e972c2e3d264`，详细诊断2479B／`cd567b72ebc0606cd7f69944cc65a669555877ecbb2f48c9a821d05972204b1f`，根退出1287B／`7b08a9bcf00a411d19a02191461382a09aaa6c96e08e1f9183b201aea69cc743`。

所有权纯准入位于`target/unified-quant-selection-partial-segment-static-r1/ownership_transfer.py`，绑定原九份selection元数据以及十份锁、恢复工具片段、Job/gate源码证据，返回`QUALIFIED_STATIC_EVIDENCE`。它保留原组失败、未观察到的正常退出为null、缺少CreationDate及直接inner Job查询为false；以同期PID/命令链、原Job分配和非breakaway源码支持归属推导。原scope固定总100800秒、剩余组21600/28800/21600秒、累计30/16、已消费15/16、剩余15/0及5/5/6/4分配；不创建claim或授予执行资格。

首次4项CPU检查actual exit0／`781630`，controller0.781秒，helper51252/venv59188正常0，Job终态active0；4PASS/0skip，31来源稳定。覆盖完整固定输入成功、缺证据/原raw变化、错误PID链/命令/调用关联、非空Job和预算/截止/key扩大拒绝；语义负例直接测试公开入口实际调用的校验函数，避免仅由SHA拒绝掩盖语义缺口。结果目录`target/unified-quant-selection-transfer-tests-cpu-r1`：计划15849B／`ce4bc34de4213952de067a95c9f40a26229150edffeed47215da5927a02575c9`，result3871B／`014228688c10c0995df5fea86df6339c4015cf4fc0cb0f652e15a00e65374e16`，unit11328B／`947999796927f9f2105ea844f0e473749076dabd0ac0b797c30f3fb690f1d0ba`，根退出1309B／`dfe1ddbfe49f41fc3c0efb51da8d72f1008cf0c3d7b1a8e587d9c8c8ef521133`。

下一消费者采用一个新outer owner贯穿后3组的方向，仍需每组受控子进程实际终态与原group/global截止。旧三锁永久保留；新固定唯一claim、跨进程owner通道、driver prepare/run/check和facade mutation接线仍在静态实现，必须实际完整replay确认head/pending/剩余消费并通过剩余时间准入后才能运行。上述CPU结果不是转移、重放、GPU、性能矩阵或生产资格。

## 8.95 全部暂缓矩阵阶段的有限 CPU 验证（2026-10-01）

对原 actual96 驱动和 v4 facade 的静态调用图核对发现：父包已缓存时，一条成功 COLLECTION 路径包含约20次完整 active-contract Sources核验、10次parent-inventory核验及10次v4 replay；这些不是10个新的历史子进程。原11个 `DEFERRED_RPC_DEPLOYMENT` 行各执行完整replay，返回状态未用于任何后续账本操作。调用图保存在 `target/unified-quant-facade-callgraph-static-r1`，仅为静态计数，不是整轮耗时归因。

新隔离 `target/unified-quant-deferred-matrix-stage-static-r1` 为全部11行暂缓的固定矩阵增加明确阶段：严格检查行数、顺序、唯一pair身份、空command及完整行内容；阶段前后各做一次原完整check，每行保留原本地Sources、driver历史、guard及Prepared核验，写出全部原PAIR_DEFERRED事件。不缓存replay结果、不增加预约或账本写入；非全暂缓分支保留原循环，原final check、inspect、summary及summary后check不变。

这是新的观察时序，不能声称与原11次中间全量来源观察完全等价：若某个传递来源在阶段内改变又恢复，新阶段可能无法观察到该瞬时变化。所有局部事件在阶段末及原finalization成功前均不能作为整体成功；结构上减少9次完整replay，不代表已测得运行时间改善。

首次运行前独审发现并已修复两处问题：逐行local核验越过截止后仍可能写事件；以及在history-operation退出和回执发布之前更新outcomes。现在helper在full边界及每次append前后检查同一原绝对截止；driver绑定原active-operation对象及deadline，不允许替换或延长；整个context成功退出并再次检查原截止后才发布outcomes。helper已列入两个contract构造器的Source集合。freezer转换旧runner/entry的实际二次raw也核长度与SHA后才解码。

固定四项helper检查首次actual exit0／chunk `f8cebf`，controller PID33452、0.406秒；helper15620与owned venv67864实际exit0，Job终态active0，4PASS/0skip，18份绑定来源前后稳定。覆盖真实固定Prepared/catalog中的11行身份及事件顺序、非完整或不合法矩阵拒绝、真实小文件和内存内容变化、full边界失败，以及local／append／post-full跨截止。full-replay回调为明确的测试替身；driver仅AST解析，没有导入或实际业务运行。

- `target/unified-quant-deferred-matrix-tests-cpu-r1/frozen-plan.json`：8409B／`acd5ebe608cb8d8d172a86159b5f6b7d8870e2632315e60bbca7e2af9296c1ce`。
- `run/result.json`：3840B／`35efebf925566511c2f0f838b22407fa5c5eb4181540866026bf5d41304730ac`。
- `run/unit-result.json`：3316B／`ce08debf5643e16ae28f3be04dcbbbbb6a60c03b62d49360aab04b07034da966`。
- 根实际退出记录：1295B／`70e83121130001b4c077d98fa91a2a6622dc7c848d56a6225fc5f02ab44d035d`。

这四项不资格化新driver完整import／ROOT／Source接线或接续执行协议。真实history、successor claim、账本、GPU及新ABBA均未执行；原首组失败、五份数值结果、旧消费和原总钟继续保留。

## 8.96 接续执行控制核心的首次五项夹具检查（2026-10-01）

`target/unified-quant-selection-successor-guard-static-r1` 已实现固定唯一claim的exclusive创建、flush/fsync和失败保留，outer同进程串行操作边界、原replay调用的适配以及同一边界内append准入。它复用原mutator本来执行的完整replay返回值，不在进入或退出时再增加一次历史扫描；原写前完整来源核验及ingest/compare写后replay仍需由真实接线保留。消费、剩余15个COLLECTION key、5/6/4分组和原总钟沿用，不启用ABBA。

首次检查前修复了四类静态问题，旧候选均未用于CPU冻结或执行：事件摘要必须与原facade一致使用`ensure_ascii=True`；成功close必须核验旧锁、consumer Sources及claim；create_claim和操作边界的最后一次来源核验之后必须复查同一工作截止；成功close在原60秒收尾余量中检查原最终截止，终态写入后也检查来源及截止。若写后失败，已写candidate原件保留且调用抛错，不能仅凭该文件声称进程成功。

最终guard为19655B／`d3c3132ab1a29267e99e6ca67de87ca72946379fdc9f9fc6550794be1947746c`，adapter为3037B／`3337606ab634f56cc492e660b01139405bcf69d2c57099277887846ad843f364`；五方法测试为19535B／`43fb5f5189e753da4bba5940ad18785023ff5fbcdd7886db25fcef6a0720278f`，来源表8553B／`3aaaecdabbb4024667e4ff31358ba9ea3196d1db7afd509f5c597c8ac722b118`。独立静态复核关闭上述问题；root三份Python AST实际exit0／`bb1623`。

`target/unified-quant-successor-guard-tests-cpu-r1` 首次actual exit0／chunk `835d12`。controller PID40060、0.875秒；helper18808与owned venv39440正常exit0，Job终态active0。5PASS/0skip、24份绑定来源前后稳定。检查覆盖：真实小文件的唯一创建/fsync/失败保留；缺replay或错误head/pending拒绝；同一边界原调用次数不增加、中文payload独立摘要、失败append保留消费；来源变化和各个写后/末核截止；外进程对象及未实现通道拒绝。成功close正例与变源、变claim、回调/写后越界负例均包含在原五方法内。

- `frozen-plan.json`：10692B／`80233f9a0f5709229c57a57694e884164db18e42dc7c5081ca476cdbecaeba48`。
- `run/result.json`：3841B／`16e623e7f5f0632fac6ece164439a2e6fd0c2ac132cfe53b77e8acfb180da2b9`。
- `run/unit-result.json`：3859B／`7aad21c40649db2bfe709e38909e98d6b8a15130e32bb9b12773041972acfd28`。
- 根实际退出记录：1296B／`852fde1d13a106b04d450941ff6ace3281d774da8e087d275c78aa4ee43960ef`。

这些是保留的OS临时目录及真实小文件检查，admission、replay、append和完成状态为明确的合成夹具，未重跑此前四项固定证据检查。没有创建业务claim、改动旧锁、执行真实ledger/history或调用GPU，也不证明实际三组执行。当前`ChildOwnerChannel`仍明确拒绝：保留peer进程句柄、出生时间/镜像及同一Job认证的通信通道、driver/facade生命周期和实际owned组终态尚未完成。后续通道只应传绑定代码中原replay同步调用产生的必要状态，不复制完整contract/loaded输出，不额外replay；该证据来自绑定代码与认证进程，不能描述成内核证明Python函数运行。

§8.95和本节共9项首次CPU检查完成；原失败、五份数值结果、历史消费和原全局截止不变。完整选型、性能矩阵、独立确认、holdout及生产接线仍未完成。

## 8.97 路由元数据与传递来源扫描拆分的首次检查（2026-10-01）

新隔离候选位于 `target/unified-quant-facade-routing-split-static-r1`。保留原 `_active_contract` 的全量含义、私有 mutator、锁内完整 replay、append 与操作后 replay；仅为 reserve/compare 的外层路由增加独立元数据入口，逐项保留原 sidecar、v3 seal、contract Source、identity 和 audit 校验。原完整调用参数不变。旧锁路径的结构计数为 reserve 的 A4→3、compare 的 A5→4，R/P 未减少；未实测运行收益，未接入新 campaign。

首次固定四项 CPU 检查实际 exit0（工具 `fb6e7a`），4 PASS／0 skip，20 份绑定来源前后稳定。controller PID53364、0.375s，helper83604 与 owned venv34708 均实际 exit0，Job 终态 active0。原 parent/history/facade 按固定 Source 真实导入；元数据、传递扫描及私有 mutator 的执行为有限替身，未做真实历史重放、ledger 操作或 GPU 调用。四项覆盖原元数据观察序列、控制文件变化拒绝、原参数传递与写前／写后失败。

结果：

- plan：`efff9e39adf4e9fe0fe8c4f2cacb29fdd3b6c06ebb3ea6f97a1c47805d92923b`。
- `run/result.json`：`2e4ae00e53a872f78b1bf54d1d450b59dd5898f7778531e3f16e6e29b70aa9d5`。
- `run/unit-result.json`：`90f2faec7ee161b3442b30377f3c4ee0bb1a9beaf724fc27377e5321481de669`。
- `root-tool-exit-observation.json`：`2f2c07606c4565b417ce72d2b42d8f666503edba251e61b0502af7a1cb56fba5`。

与前两节合计 13 项首次 CPU 检查通过，不能外推为完整消费者资格。拆分明确改变部分来源的观察时序：无效策略可能先被拒绝，合法操作由后续原全量核验拒绝变更，短暂变更后恢复的观察不保证与旧路径等价。未采用此前未证明收益的并行读取或路径遍历候选。

Windows 通道已静态交付，双端校验 held PID／创建时间／镜像，server 校验明确的 owned Job；取消必须实际完成才释放 native 缓冲区。真实 IPC 三项检查尚未执行，Owner／driver／facade 接线尚未完成。原 session28076 失败、五份数值结果、全部历史消费和原总截止保持，业务 claim 未创建，新 GPU／ABBA 为 0。

## 8.98 Windows 通道首次三项真实 CPU 验证（2026-10-01）

固定通道源码 `target/unified-quant-selection-successor-channel-static-r1/owner_pipe.py` 为 19092 B／`3a45fbc25a566ee6acce6ab30b0077485ca1bbd997e0e9c62f6b68a20bc96cff`。首次验证独立于业务 claim，使用固定 CPU-only pipe 名和三个单次 direct base-Python peer；原 owned runner、显式环境、gate 前 Job 分配保持，父镜像与子镜像都必须匹配冻结的 base_python Source。

`target/unified-quant-successor-channel-real-tests-cpu-r1` 首次实际 exit0（工具 `f6d12c`），3 PASS／0 skip，20 份绑定来源前后稳定。controller PID14716、10.688s；helper41680 与 owned venv7716 均实际 exit0，外层 Job active0。每阶段固定 10s work／额外15s cleanup，整个检查180s、内部120s且受原campaign总截止约束，遇失败即停，未复跑已消费入口。

三个真实结果：

- 双端 held PID／创建时间／镜像身份匹配，server 确认同一明确 Job；144080 字节 Unicode 消息双向一致。随后读取未发送帧，在原 work 截止触发真实 CancelIoEx；request10 的记录顺序为 cancel_called→cancel_returned→completed(error995)→release_called→released，均在 cleanup 截止前，无未完成请求留存。该时钟粒度下时间戳相同，仅证明记录顺序，不推断取消耗时。
- 错误创建时间准确返回 `peer birth/image differs from held expected process`；错误但有效的空 Job 准确返回 `peer not in exact held group Job`。两子进程在 hello 实际收到 WinError109，未把任意异常当作成功。
- 三个 held direct child PID11764／46272／65580 均由实际 wait 观察 exit0，各阶段 Job active0，无 watchdog、清理或最终来源错误。每阶段 Job total_processes 实际为2，外层为11；这些是原始 accounting 值，不推断辅助进程原因，也不声称完整 OS 进程普查。

冻结计划 SHA `c8b0811dbb6be58fffdffb9772d3147d8be38f9099c4b8efca5fc015dbc3334a`；结果 `0f396264f97f4f2dd2aef084f930cc42028c72b876a33ee9383f8c5eb9fd139a`；含所有观察的 unit-result 为 `dd25ebb2813c2a22237e65ceda80970a59b825cedc509d99ab6e19a04ec7f4e3`；根实际退出为 `71e512ffb042b0e0a0b08fecb60b101cc69b64fee96544cb2834027740e54f0e`。独立窄审读取21份小结果／日志，前后稳定并确认上述取消、精确拒绝及真实退出；未重新核验整个来源库存。

此次证明实际管道传输与失败清理，不证明跨进程业务 Owner 或完整消费者已通过。新派生 Owner 必须接收原 replay 的有限投影、执行原同 lease／head／消费门后再 ack；facade 保留原锁前 full A 和所有原 R，driver 尚需完整 prepare/run 生命周期与剩余三组接线。旧三锁、原失败、五份数值、所有消费和原总钟不变。业务 claim／新ledger操作／history／GPU／ABBA均0，性能与holdout仍未通过。

## 8.99 Owner／facade／driver 实际接线源码的联合静态复核（2026-10-01）

三份新目录完成可读源码候选，尚未导入、AST、冻结、测试或业务执行：

| 组件 | 文件／交付 | 当前 SHA256 |
|---|---|---|
| RemoteOwner | `target/unified-quant-successor-owner-integration-static-r1/successor_guard.py` | `9099abdf453f75d95fc334903324bfb2a9e5d8ad049c0693c70fbd2aca98b2d7` |
| RPC adapter／phase dispatcher | 同目录 `call_adapters.py` | `34d2567bc8fcdbc43c91482ff59f520ffab98ae491f9581f438c12104ea7e638` |
| Owner 静态清单 | 同目录 `source-map.json` | `54996074cce4a69450f3bb801a384fb20ba7316ef6f2c459d4ab71d5c0f9ca02` |
| 完整 facade | `target/unified-quant-successor-facade-integration-static-r1/restored_selection_continuation.py` | `07e320fc7a019a188a29005d931a4e28f6d36612d5cd8c5538aad7e276a0a1f5` |
| 完整 driver | `target/unified-quant-successor-driver-integration-static-r1/run_quantization_search.py` | `8223bbb0348a3d4f8853a11b4d55fd8e2b86e386625993c529adba83c33faa7f` |
| driver 静态交付 | 同目录 `delivery.json` | `f6c76cc081da83d4c9032bb1b6aa1fa019670a373c60705190fac9b54847fadc` |

RemoteOwner 接收已认证、绑定来源的原 replay 有限投影，执行原 lease／head／消费门后 ack，不构造假 callable 或把完整输出跨进程复制。append 前后以实际完整 canonical 摘要和有限语义字段衔接；新增 head 由实际 append ACK 更新，原 post-R 照常执行。phase complete 消息不代表 OS 退出，成功关闭必须有六个真实 prepare/run 终态。

新 facade 的原锁前 full A、append 前 full A、snapshot、全部原 replay 与 post-R 保留；只将已过 routing4 的外层元数据拆分接入。旧 raw lock 在新副本中明确拒绝，缺 client 不回退碰旧 marker。旧 v4 audit、Source pins、ROOT 和私有 legacy 来源不伪造、不放宽。

新 driver 保留 history 嵌套原截止，RPC lease 与 history depth 分开；metadata failure-terminal 容器内部允许独立 inspect/fail lease。首次静态复核发现 prepare/run admission 晚于首 operation 的必拒顺序，已前移到 public wrapper。首次原 replay 通过后仅 outer 建 claim；child 获取 Source 后才写 Prepared。Prepared 顶层合同不变，新来源通过 source_records 绑定。C8 原数值阶段后另执行固定 C3 旧 FP16 comparison，不重采已完成结果、不重开原失败组。

根侧首个字符串派生因重复匹配拒绝，实际 exit1／`fe34b1`，没有发布候选；错误记录原件保留。限定到 restored prepare 后派生 actual0／`c4b3cf`。这属于静态作者工具，不是消费者试跑或新实验。独立联合阅读绑定上述实际源，未见新增确定阻断，但不能据此执行任何业务。

具体 entry/controller、venv redirector 到实际 inner 的 held 身份链、完整 OS 终态 validator 与消费者资格仍未完成。后续入口须保留原 `_child` 和 actual96 依赖；phase nonce／当前时钟不进入 Prepared 的静态 source_records，避免 prepare/run 重建不一致。现有 guard5、routing4、IPC3 的证据范围不能外推新组成。

原三把锁实际再次读验 SHA 不变，固定业务 claim 仍不存在。原 session28076 失败、五份数值、全部消费及原 global／组上限保持；新 history／ledger／GPU／ABBA 为0，端到端性能和holdout仍未认证。

## 8.100 Successor完整接线候选、真实终态校验与首次语法检查（2026-10-01）

上一goal轮仅进度汇报，无新增实现或验证；本轮继续实际源码。新增entry/bootstrap模板、held venv身份链、controller与phase lifecycle；controller内合入独立作者的terminal validator，并由另一peer及root对照真实driver/facade schema独审。当前完整接线均为源码候选，未导入执行业务，尚无bootstrap manifest、生成entry、业务plan或完整CPU资格。

启动顺序为原base gate helper入Job/观察Active1后放gate，ready只作PID locator，经实际helper→venv redirector→inner两父边、birth/image/同Job核验，再在同一controller线程认证pipe并serve。修复两个静态确认问题：不得同步wait child后才serve形成死锁；sealed Pipe.send末尾还核peer存活，所以child收到最后ACK后等待controller在serve返回后一次性发布exit-release，随后才close/退出。release用pending xb/fsync再Windows rename，拒覆盖/重试。held链接管先于截止检查，消除hold_inner返回后过期时新句柄不可达的清理缺口。

正常终态分别保存helper Popen实际wait、helper报告的owned venv wait、retained inner与venv句柄实际wait和Job Active0；均须在原work界内，cleanup退出不能填正常exit0。清理后validator对照本次对象实际观察、Prepared/result、5/6/4新key、所有原11个deferred项与累计COLLECTION20/26/30、ABBA16；不新增full replay或重复原数值判定。实际phase/current/previous callbacks定义在controller文件内，co_filename不冒充其它源。写后失败的Owner终态仅保留candidate Source，不据文件推断成功；末核每项前后使用原final截止。

八个新组成文件的首次AST+compile实际exit0/chunk `1c4b52`，8项 `PASS_SYNTAX_ONLY`、10输入来源稳定；plan2524B/`4726c25525ac37007ab75bfa972ebfb35e2a05cce1524d3cf9a4d5b98b8ba52c`，result8120B/`f530d2b45c3b098fa478a2dcf46a6d5d1a9899e6df7ff2b3f4c7a2311c072507`，根退出851B/`14aa19e211bc9a7ac3e6dae9f1ac6d8e0aa856f508a8f0c07ce218e3b49b8409`。仅解析/编译内存code，不执行或导入业务模块。

随后peer发现qualification读取的helper_result Source未加入最终末核。原r1已固定在上项证据，保留不改；仅派生controller-static-r2补返回列表一项，另一peer核唯一差异。新文件首次单项AST+compile实际exit0/chunk `b96040`，其它七项继承不重跑；controller39895B/`8d99b47438508ef59f39a5b1814f889477339aa9c64490684bc4538cf6b5d23d`，result1026B/`f32d9dd196f2125ddaf187125bfd4bfe0ddc06b93349ff027bac9229be2c0fe5`，根退出873B/`d288d63cd7fbdd4ab77890241e6e281f2dfa884952a63fe2b3355a02c973e4d5`。完整候选delivery10749B/`281e82c3bf24c1c62a7455d40bdc304b756028dff969a08de43c614b52328b8c`，16来源前后稳定。

当前使用r2 controller，加r1 lifecycle20779B/45f840b2…、entry模板22806B/ff7d454f…与held链16923B/5d0a3ba3…。完整CPU资格reader只是未来接口合同，未有对应producer或PASS。它已关联root/result/唯一owned terminal、test-plan/SHA/work argv、helper producer/venvwait、实际被测来源和scope；不能拼接其它成功退出。真实OS、真实_child搭配无业务payload、隔离历史/collector的prepare/run夹具分别记范围，不把history=0的测试冒称真实固定catalog端到端通过。下一步定义原gate+真实venv的held-chain有限三项检查，不重跑旧IPC3，也不替代后续整体资格。

原三锁本轮只读SHA相同（chunk `65343e`），业务claim与新controller运行目录均不存在。原失败、五份数值、15/16消费及FIRST/GLOBAL总钟保持；该次观察原剩余约60860秒是历史观察，不是续期或运行预测。新history/ledger/GPU/ABBA均0，生产不变；完整性能、确认和holdout仍未完成。

## 8.101 真实held进程链首轮失败、共享终态修复与三项首次通过（2026-10-01）

原gate→真实venv redirector→fixture inner的首轮三项检查在第一项停止，根实际exit1/chunk `cd6021`。live两父边、同Job、PID/birth/image核验已通过，helper实际wait0；随后对已退出的inner执行QueryFullProcessImageNameW返回WinError31，正常inner/venv wait均未取得。1项error、两负例未运行，外层Job active0/total10；旧正常wait未知保持。失败根1545B/`9c67a91fc99f9b9b346a8e0e950901bdfeba3d3ddefa6004b2fa13884311cdc4`，result5070B/`51d28a1a494166716c3509067fe8ca942827439ed3a4bce3175b73cf465daed1`，unit22621B/`993c5d4aec6b184f72680e47e15b3bbe11a20becead49dd5aca4ea3f801aba2a`。未重启原入口或补造成功退出。

新共享`target/unified-quant-successor-held-terminal-static-r1/held_terminal.py`为3572B/`0a48cf973c200db36559c35d6df2127ebf5df388ca9c4605f2c77e20f8a7d098`。调用者在进程live时认证完整身份并保留同一handle；终态等待signaled后从该handle读取GetProcessId/GetProcessTimes/GetExitCodeProcess，核相同PID/birth和实际退出时间。输出明确标记image来自同handle的preexit证据，不在退出后再次查询image，不重新按PID打开进程。正常与cleanup保留各自原deadline，cleanup退出不升级正常成功。

新lifecycle-r2为20169B/`e07f1f7a84d86a5d8b75b21cd291c3d67844cd8bfb7aa5f5c58dbbf188852135`，controller-r3为40553B/`39350355341ee878e45a054085de240e8f05d9e61c8f1b5a4a96480dcb222a5c`；它们使用同一共享函数及显式Source，terminal validator追加真实PID/birth/exit-time/signaled/preexit-image标签核验。旧已封lifecycle/controller不修改。独立有限Source审查未见新增确定阻断，不构成完整controller已运行。

修复版`target/unified-quant-successor-held-chain-real-tests-cpu-r2`首次实际exit0/chunk `dd2cb1`，3PASS/0error/0failure/0skip，25外层Sources稳定、10个unit Sources稳定。修复第一项，加原两项未运行的wrong-helper-birth/wrong-empty-Job负例；原fixture与三个case主体不变，仅接共享终态函数。六次独立inner/venv正常held读取均actual0，三helper正常0，各case Job active0/total5、watchdog和cleanup无错误。外层helper79740及ownedvenv32304均actual0，outer Job active0/total20；这是实际accounting值，不声称完整OS进程普查。仍使用每树15s work+5s cleanup、suite120s、outer180s和原global截止，不重跑旧IPC3。

| 新实际证据 | SHA256 |
|---|---|
| frozen-plan，15151B | `893f9cfb51bc38f1547a4af4ee05d1bf1b0eb919e34f51ee2a86fd0591fafb44` |
| run/result，3944B | `caf54ea532b3f37cd470d10f07cdfa8fba0300eccfbe64aab93551ce1683ded1` |
| run/unit-result，66898B | `d42a265cb6cbc0335e991625c5898aa488efd24834c2063616b2a12d5084409a` |
| root实际退出记录，3261B | `853c27a7ffdd304a04a7bec2866d64bf1913af4eb69bdacdf62e591ae263f569` |

两个changed production consumer及新bootstrap freezer的首次三项AST/compile实际exit0/chunk `dd9f54`，仅语法检查，不重跑其余已固定源码。新final9 delivery11590B/`c3b8593827a5354ed38700a05bd8be72ec80b62fd39e6b9008d758d120073c96`；bootstrap request9844B/`21b36169622343a49568d6a9b9e46f03cd7d32c9f096a82498537885016f4d76`，明确child10/outer10/8roles/4deps和原2929-source库存，未将未来产物SHA硬写成循环。bootstrap尚未生成、完整consumer资格未运行。准备的整体资格区分真实OS lifecycle、实际generated entry `_child` inert payload与隔离历史/collector边界的实际prepare/run函数调用，不能用held3成功替代整体资格。

上一轮已实际取得held3修复PASS后按用户要求暂停，子任务停止；目标后来恢复active，本轮从这些已完成证据续接，未重跑。原session28076失败、五份数值、15/16消费、原FIRST/GLOBAL截止、旧三锁均保持。新真实history/ledger/GPU/ABBA0，生产不变；完整性能矩阵、独立确认、holdout及剩余时间准入仍未完成。

## 8.102 固定bootstrap与generated entry首次owned构造成功（2026-10-01）

最终九件组成明确包含shared terminal、lifecycle-r2及controller-r3，其delivery与request已在上一节绑定。新增有限owned wrapper/runner只加载原IO/containment与source-only freezer；原base gate先入Job后放行，wrapper直接读取本次原Attempt保存的`freeze-intent.json`，将同一个实际work绝对截止传入`freezer.freeze`，没有重新计时。外层360s总界、10s cleanup和10s final reserve受原GLOBAL截止钳制；异常保留固定输出和部分写入，禁止重跑同输出根。

首次实际owned freeze根exit0/chunk `38d8bc`，controller48952、4.781s；helper37252和ownedvenv59584分别实际正常exit0，Job active0/total5，无watchdog/cleanup/最终来源错误。31个外层绑定Sources稳定。freezer对原2929源库存及明确新来源逐块实际前后读取，2947静态Sources一致，原2929路径/长度/SHA记录全部保留；bootstrap加自身manifest和generated entry后共2949，不接近4096上限。只构造启动材料，未运行正常generated entry、原history、真实账本、native Worker或GPU。

| 固定产物 | 字节数 | SHA256 |
|---|---:|---|
| bootstrap-r1/bootstrap.json | 720146 | `b0ca4432ea5f2857a0cd2f3d92bdfb2bf61f2b061c68bc2031e992d5032afb88` |
| bootstrap-r1/entry.py | 22992 | `940dfc59ce1ab2302f41228c8b555f7c3a99cacf5cd98c9c5505061bcb045e44` |
| bootstrap-r1/freeze-record.json | 1430694 | `0ab452e58d1784b532188d2ab224b23c85c8049128d80ef916ca7cc5c8ff4044` |
| bootstrap-control-r1/frozen-plan.json | 9779 | `72c468d4d94601dd85cf293eb6b977dd1ef844f6c3b0de786a910d8625e43924` |
| bootstrap-control-r1/run/result.json | 4562 | `00369cfa781ab1d22dfd13c26a1e622b0972f1df3adfb79af3ec9481000a8c81` |
| bootstrap-control-r1/root-tool-exit-observation.json | 2387 | `8af9ec4b34637075be4af73ba8766eab5bb774aaca36daadf742596aa1243ae1` |

根侧在实际退出后有限核对产物：generated entry逐字等于sealed模板唯一`BOOTSTRAP_SOURCE`字面替换，Source键顺序保持path/bytes/sha256；manifest保持精确13字段、child10/outer10/8roles/4deps，无新manifest/entry/phase自引用。此有限核对只读产物和原runtime metadata，不重复2947内容全量读取；实际首末读取证据来自source-bound freezer及本次owned正常退出。独立harness静态审查未见确定接口阻断。

构造状态仅为`PASS_FREEZE_ONLY`／`ARTIFACTS_WRITTEN_NOT_RUNTIME_QUALIFIED`。完整qualification源码正在实现，须实际调用production `run_phase`/PhaseLifecycle/PhaseDispatcher及fixed generated entry `_child`，并将真实OS与隔离历史/collector效应边界的实际prepare/run函数测试分别记scope。不得用冻结记录、held3或dummy flags替代该资格，也不冒称真实固定catalog的history=0端到端通过。

原session28076失败、五份数值采集、15/16历史消费、FIRST/GLOBAL及旧三锁保持。业务claim/controller运行目录尚未创建，新真实history/ledger/GPU/ABBA0；完整矩阵、独立确认、最终holdout和发布接线仍未完成。

## 8.103 完整消费者一次性计划冻结与首方法放行前失败（2026-10-01）

四个有序方法和八项语义检查已静态交付，生产Source库存与隔离业务fixture明确分离。suite/helper/fixture child/inert payload封存于`target/unified-quant-successor-consumer-check-static-r1`；source-map为20411B／`7b86fc6299ee503b8d497595faebd3bb3776afc7f7610c0278d89629231e1c8d`。原真实历史、Protocol和collector入口被fixture拒绝，真实driver/facade函数使用合成前缀及小数据，append次数与production操作分开记录。该scope不覆盖真实collector中途失败的所有分支、三生产组、精度或性能。

root的unit/outer producer与source-only freezer经有限独审收口；可信plan之后的启动错误保留阶段、原异常及已落失败unit，不能因报告缺失补造正常wait。source-only freezer最终为13415B／`7aeaeb910c7dc19e9ec86f5f825013b62b48d38a9495211f60cb4dbc2e9a4bc8`，180s同一有限截止钳制于原GLOBAL，最终输出后再次检查该界。首次freeze actual exit0／`08e484`、6.547s，六份此前未检查新源AST+compile通过且没有import；2962个显式输入首末一致。qualified stock按原`_source_stock`代数精确为2949，仅原manifest静态库存加实际bootstrap和entry。request精确26字段、CPU plan为23必需字段加no_retries；request没有自身或未来phase/claim/output Source，run/fixtures在实际owned启动后才创建。

| 本轮固定原件 | 字节数 | SHA256 |
|---|---:|---|
| consumer-tests-cpu-r1/frozen-plan.json | 1435705 | `ec16407049076314262cd8ac0067d32e947858b35bf495eb8d9eb3893ba4fdfb` |
| consumer-tests-cpu-r1/freeze-result.json | 1435651 | `37b3a17ac93d3536213968fc8c23dbd6a5b4fe5817d8a58a06442461afc20173` |
| consumer-tests-cpu-r1/run/result.json | 5554 | `9c6bc30555c700b30a6ee484bd071094eb18a2ef2d91a4af0cb9e962650eaeed` |
| consumer-tests-cpu-r1/run/unit-result.json | 1435981 | `e44e68f34724d51e8a77070d1965adf2b631473c1c183adae0f1e848f8a47aa3` |
| consumer-tests-cpu-r1/root-tool-exit-observation.json | 3519 | `ec9ce497208c53741ff45dd24cf03696b478bb4751dfb6b890eea070a95a7a91` |

首次suite启动session69622，后续同一session观察实际exit1／`0d2f03`、34.719s，禁止重启。unit完成实际configure后进入首方法，但prepare的`PhaseLifecycle.start`在gate前观察Job active2/total2，拒绝`unreleased helper is not sole Job process`；测试实际1 error、0 failure/skip，余三方法未运行，完整report未调用。actual configure返回只证明该次调用控制流，不能当作完整消费者资格。

首phase helper27656已经入原Job，held helper身份已取得；gate_release_attempted/released/write_completed全部false，ready/held_chain/output及所有正常wait均null。失败清理wait为3758096385、末Job active0，无cleanup/Job/pipe/chain错误。第二个Job成员没有PID/birth/image原件，身份为UNKNOWN，不据旧案例猜conhost，不放宽active==1门。新lifecycle比旧Attempt在assign与accounting之间多做held identity/alive/image核验，这是观察时点差异，尚未证明根因。fixture未替换WindowsJob；同活树wrong-Job负例尚未运行。

outer helper56740实际exit1；原outer receipt的owned PID/wait为null并保留，另由helper-result明确记录ownedvenv42492实际exit1，不回填原件。outer Job active0/total7、无watchdog/cleanup错误，unit与outer实际2964绑定来源首末稳定。这些是失败和清理事实，不能提升为PASS；未生成qualification或production plan，也没有执行实际generated entry的`_child`路径。

独立闭gate单helper成员身份诊断正在源级准备，须在同原Job现场取得成员列表及retained handle身份，保持原拓扑、来源与截止、永不写gate packet。仅此新诊断能改变后续修复选择，不能重复已终态suite或旧IPC/held检查。原三锁SHA再次读验一致，业务claim仍不存在；原session28076失败、五份数值、COLLECTION15/ABBA16及FIRST/GLOBAL继续保持，新真实history/productionledger/GPU/ABBA0。

## 8.104 原拓扑下闭gate成员诊断完成与晚观察缺口（2026-10-01）

新增有限probe只使用原base gate/WindowsJob、IO及显式环境，固定一内层helper；外层仍由原Attempt启动owned venv unit，另有一原gate helper。内层永不写stdin packet或启动entry；两次QueryInformationJobObject(class3)成员列表均≤32，用同retained handles核PID/birth/image及精确Job，Toolhelp父快照最多16384行、仅作父PID locator。原singleton门、sealed lifecycle与全部失败原件不变。工作≤100s、cleanup10s/final10s受owned120s及原GLOBAL钳制；外层180s总界与原30+30 reserve保持。

probe29505B／`099f68fc637401cfd858cc7f4bcd089f9019db0d8fdba815c69c9bae6bcb326a`，联合有限独审13 Source稳定。首次source-only freeze实际0／`f715c6`，仅probe和新outer两项AST+compile通过；旧消费套件或组件不重跑。首次owned诊断实际0／`2331ce`、0.406s，outer helper61248和ownedvenv22184分别正常wait0、Job active0/total7、无watchdog/cleanup或末核错误。16个外层绑定来源、8个内层来源分别前后稳定。

| 固定诊断原件 | 字节数 | SHA256 |
|---|---:|---|
| pregate-job-probe-cpu-r1/frozen-plan.json | 6077 | `9533b59f77f8607d52a89d8485dc396372c2508ca6782090ab5a013e25780f25` |
| pregate-job-probe-cpu-r1/run/result.json | 4069 | `2d0005de0295334768c1d0f292fb5dbd61b54f84c6407c96fdd2711ab9fc38cd` |
| pregate-job-probe-cpu-r1/run/inner/result.json | 11804 | `abb7ff69306c1e5bb48e6dfd9f6551ddd57c268712acd85e4b40a21e3e5dbaa2` |
| pregate-job-probe-cpu-r1/root-tool-exit-observation.json | 2654 | `0b6234f9272090219adbe0c157f7f62cd38759d54ee4cb1d1e40e49f9da0be1c` |

probe内层PID77896，closed helper63196；两即刻样本在1471434687000000均active1/total1/list[63196]。该已知helper的同held birth/image/Job及跨父快照活性均核实；父快照429行在1471434703000000完成，随后同Job清理，helper actual wait3758096385、held对象signaled，终态不再查询image。内层Job最终active0/total2。这说明当前两个采样窗没有取得后续累计新增成员的身份；不能将两窗的完整列表外推为整个生命周期成员已完整审计。

状态`PASS_DIAGNOSTIC_ONLY_NOT_CONSUMER`只表示本次有界观察及清理完成。第二成员仍UNKNOWN；不能映射到旧phase27656的另一个成员、猜测conhost、修改计数门或提升wholeconsumer资格。r2新源只补≤2s、10ms间隔、全部≤201次accounting/list的晚观察；发现extra PID后停止等待并持有核验全部union，最终累计数仍超已证实成员时必须保留UNKNOWN。该r2尚未冻结/执行，r1不再运行。

本轮仍无production plan/claim、真实history/ledger、native Worker、GPU或ABBA；原session28076与wholeconsumer首错、五份数值、COLLECTION15/ABBA16及FIRST/GLOBAL均保留。完整精度选型、端到端收益、独立确认、holdout及生产接线尚未完成。

## 9. 依据

项目证据：

- [INT8 后端](RustGoINT8后端.md)
- [B15 剪枝诊断](RustGoB15剪枝INT8诊断.md)
- [INT8 专项优化](RustGoINT8专项优化.md)
- [注意力与算法调优](RustGoINT8注意力与算法调优.md)
- [低精度方案调研](RustGo低精度优化新方案调研.md)

本次核对的 NVIDIA 官方资料：

- [CUTLASS SM120 GEMMs](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html#blackwell-sm120-gemms)
- [cuBLASLt 算法保存与恢复](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmulalgo-t)
- [cuBLASLt AlgoCheck](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmulalgocheck)
- [cuBLAS 缩放支持表](https://docs.nvidia.com/cuda/cublas/index.html#scaling-mode-support-overview)
- [CUDA 12.8 Update 1：GeForce 块缩放 FP8/FP4 支持](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html#cublas-release-12-8-update-1)
- [TensorRT 数值精度](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/accuracy-considerations.html)
- [Model Optimizer AutoQuantize](https://nvidia.github.io/Model-Optimizer/announcements/autoquantize.html)
- [NVFP4 Local-Hessian](https://nvidia.github.io/Model-Optimizer/announcements/local-hessian.html)
- [cuSPARSELt 稀疏格式](https://docs.nvidia.com/cuda/cusparselt/types.html)
- [矩阵分块与并行度的权衡](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)
- [CUDA 线程块与 occupancy](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#thread-and-block-heuristics)

官方库的能力、其他模型的论文结果和本项目的实测结果分别记录。在线文档会更新，实现时需要冻结所用版本。

2026-09-27再次核对实际363快照的Tensor Core路径：7份GEMM/INT8/LT源与此前356快照SHA一致，cuda.rs仅共享CPU parser接线改变。令T=361×physical batch、O为输出通道，则INT8 Lt按列主序TN解释已有缓冲，库视角M=O、N=T、K=ceil(Klogical/16)×16；INT8输入，INT32计算/输出后反量化。FP16 Lt采用FP16输入、FP32计算，输出half或FP32由路径决定；kp不等于k时转入已有手写mma.sync路径，仍使用Tensor Core。B15剪枝形状不能根据“未进入Lt”就判为CUDA标量计算，也不能根据存在mma指令就宣称高利用率。

NVIDIA的[cuBLASLt IMMA要求](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul)说明regular ordering需要TN、矩阵指针及leading dimension满足4对齐条件、M/K为4倍数；旧COL32特殊布局限于Turing/Ampere，不能移用于SM120。当前代码的O%4准入和Kp16对齐满足静态形状条件，实际算法仍需观测。[CUTLASS SM120说明](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html#blackwell-sm120-gemms)与SM100部分分别描述mma和tcgen05路线，不能把数据中心Blackwell专属机制直接套用消费级显卡。

最终独立profiler应核关键形状的实际HMMA/IMMA、tile/grid/波次尾部、padding后有效工作量及完整FFN外围开销；ABBA则在无profiler状态测完整normal Backend、固定负载和完整旧控制矩阵。B1的token维已经是361，不等于GEMV；B15 H1160的down投影K填到1168，H16则可能因tile数量少而利用率不足，后者只是待测解释。NVIDIA的[Nsight测量说明](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)指出采集可能多pass重放，application replay甚至重跑整个应用；因此profiler需单独有限登记，不能穿过永久单次consume自动重试，也不把其耗时混入ABBA。本机仅只读确认Nsight Compute 2026.2.1.0/build38283040可用，尚未运行profiler或查询实际SM120指标；不据工具安装声称利用率已测。

2026-09-27 再核 [NVIDIA AutoQuantize 文档](https://nvidia.github.io/Model-Optimizer/announcements/autoquantize.html)：当前搜索成本使用 effective bits，硬件实测延迟列为后续方向；算子分组体现运行时耦合，单层敏感度求和仍忽略量化误差的组合影响。因此本项目继续用实际完整 FFN 耗时提出候选、用完整混合图复验胜率/policy/目差，不能把平均位宽或单层误差之和写成 SM120 加速或整网精度证明。这是本项目的实施选择，不代表已直接接入 ModelOpt 的梯度评分。

2026-09-30再次核对 [CUTLASS SM120 GEMM约束](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html#blackwell-sm120-gemms)：该文所述窄精度builder只支持TN布局，GeForce cluster固定1×1×1；NVFP4合法tile从128×128×128起。后续若评估NVFP4，需要把B15窄FFN的tile浪费、缩放／格式转换及尾部成本纳入完整组成本，不能从位宽直接推出比现有混合FP16／INT8更快。这是针对本项目剪枝形状的推论，尚未运行新NVFP4候选。

同日核对 [Nsight Compute测量边界](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#range-and-precision)：默认时钟／缓存控制及kernel串行化会改变测量条件，小kernel的跨pass比率也可能失真。后续Tensor Core诊断须单独记录replay模式、pass数、时钟及缓存设置，用最小必要指标解释实际路径、tile和瓶颈；完整C32收益仍取无profiler的ABBA。文档中的application replay建议不能直接套到永久单次consume入口，需另有显式重放预算和独立输出。此次仅核对官方文档，未修改GPU时钟、运行profiler或查询实际SM120指标。

同日追加核对 [Source动态指标](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#source-page)：`Instructions Executed`按warp统计实际执行，`Predicated-On Thread Instructions Executed`区分有效线程；`Metric Pipelines`仅说明指令可能使用的管线。项目后续需将非零MMA动态计数绑定到实际launch／模块／形状，再观察本机支持的Tensor管线吞吐，不能只凭kernel名称、dtype或静态MMA文本认定利用率。

[Graph profiling](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#graph-profiling)整图模式不提供指令级Source指标；[range replay支持表](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#supported-apis)列出Graph Management不支持，application-range又会重跑应用。后续节点指令证据与整图并发诊断需分别选模式。[指标语义](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-guide)中的active与elapsed百分比分母不同；具体SM120指标、单位、pass数仍待本机实际查询。这里仅记录官方边界，无profiler/query执行或新增预算，正式收益仍取无profiler的完整ABBA。
