# RustGo 统一后端速度验证

2026-10-01。用户将验收范围调整为只验证速度，全部模型精度验证由用户完成。本轮没有执行输出误差、policy、胜率、目差、棋力或 holdout 精度检查；性能结论不代表精度通过。

本轮已完成：48 个配置初筛、五组正式 ABBA、两个 B15 候选的独立更长确认。RTX 5070 Ti / C32 下，B15/B3、B15/B8 相对最快已测旧基线的确认轮吞吐分别提高 **1.84%、1.75%**；B11 已测候选没有确认提速。机器可读结果及配置交付位于 `target/unified-quant-speed-only-final-r1`。

## 测量条件

- 实机为 RTX 5070 Ti，SM120，驱动 610.88。没有在 5060 Ti 或 5080 上运行本轮测量。
- B11 原模型 SHA256：`1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6`；剪枝 B15：`3f216ee88226ee49ca826eaa5f7e1e9982ff48a96625914bfc4e5ad20ee095a7`。模型原件不改。
- `nnbench --mode eval`，19 路，32 个请求线程；同组模型、生成的局面、batch 上限、线程数和测量长度一致。随机对称关闭，强制对称为 0；评估请求绕过 NN 缓存。
- 每个进程只测一个 batch 上限。正式轮每线程预热 128 次、计时 512 次，共 16,384 次请求；独立确认每线程计时 1,024 次，共 32,768 次请求。
- 正式与确认均按基线 A1、候选 B1、候选 B2、基线 A2 顺序串行运行，避免候选和基线争抢 GPU。每个样本必须完成全部请求、native 正常退出、源文件和模型/配方身份稳定。
- 不在计时中运行 profiler。使用系统正常时钟和功耗设置；另有用户已有的空闲 nnworker 进程，未修改或停止。样本前后 GPU 状态保存在原始回执。

这里的吞吐计时覆盖 NnEvaluator 的输入编码、排队、CUDA 完成和输出解码，以及测量线程启动/汇合，排除模型加载、局面生成和预热。CUDA 调用是异步的，现有 evaluate 完成链和线程汇合等待了 GPU 工作完成，符合 [NVIDIA 对 CPU 计时的说明](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#using-cpu-timers)。它是推理评估器的端到端吞吐，不是 GTP/网络 RPC 延迟或整局搜索速度。

CLI 的 `perBatchMs` 是这段墙钟时间除以完成批数，不是单请求 P50/P95 延迟。实际平均逻辑 batch 小于上限，存在尾批；没有记录完整物理 batch 分布。不得把 `1000 / nnEvals/s` 在 C32 下解释为单请求响应时间。

四组 B3/B8 对比使用同一 `dd665d03…` 程序及输入生成实现。B11/B16 旧 FP16 调优计划需要其原始兼容程序 `d218b98d…`，本轮保留了该构建；两边具有相同配置和 CLI，但没有跨构建的 encoded-input 摘要，B16 结果保留这一限制，不据它声称严格同输入的正收益。

## 正式 ABBA

基线是该模型、batch 上限下初筛中最快的现有 FP16/INT8 后端，候选是已测集合中最快的统一量化配置。这是有限候选集合的选择，不表示全局最优。表中吞吐为两个重复样本的几何均值，波动定义为 `max / min - 1`。

| 模型 / batch 上限 | 最快旧基线 eval/s | 统一候选 eval/s | 吞吐变化 | 基线 / 候选重复波动 |
|---|---:|---:|---:|---:|
| B11 / 3 | 803.60 | 805.55 | +0.24% | 0.65% / 0.41% |
| B11 / 8 | 1013.30 | 1011.55 | −0.17% | 0.20% / 0.51% |
| B11 / 16 | 1059.30 | 1039.25 | −1.89% | 0.15% / 0.22% |
| B15 / 3 | 581.45 | 592.55 | +1.91% | 0.05% / 0.15% |
| B15 / 8 | 712.35 | 721.95 | +1.35% | 0.04% / 0.12% |

速度筛选要求：正常完成，重复波动不超过 5%，候选几何均值至少提高 1%，且最慢候选仍快于最快基线；正收益再做独立更长的 ABBA 确认。B11 三组没有通过速度筛选。B15 两组正式轮及独立确认均通过。该规则是有限重复样本的描述性筛选，不是统计置信区间。

B15/B3 候选在 45 个 FFN 中选择 10 个 INT8、35 个 FP16；B15/B8 候选选择 23 个 INT8、22 个 FP16。attention 保留 FP16，attention tile 为 q64，Graph 和双缓冲 pipeline 开启，padding 关闭。B8 配方仅根据已有 FFN 成本数据提名，没有新增校准或精度检查。

## 独立确认

候选在确认开始前固定，确认数据没有用于再次选择配方。每臂 32,768 个真实请求，八次 native 均实际退出 0，确认进程实际退出 0。

| 负载 | 基线两个样本 eval/s | 候选两个样本 eval/s | 几何均值，基线→候选 | 加速比 | 基线 / 候选重复波动 |
|---|---|---|---|---:|---:|
| B15 / B3 / C32 | 581.6、583.4 | 591.0、595.5 | 582.50→593.25 | 1.01845× | 0.31% / 0.76% |
| B15 / B8 / C32 | 707.3、707.2 | 720.3、719.0 | 707.25→719.65 | 1.01753× | 0.014% / 0.18% |

相同口径的每批完成墙钟成本几何均值：B3 为 5.1485→5.0555 ms，降低 1.81%；B8 为 11.3095→11.1140 ms，降低 1.73%。这是吞吐对应的批完成成本，不是独立测得的请求延迟分位数。平均逻辑 batch 接近 3/8，但尾批数量有轻微差异，完整计数保留在回执。

在该已测条件下，统一后端应按层形状与 batch 选择精度，而不统一强制全部 FFN 使用 INT8。B11 保留现有最快执行配置；B15 两份混合配置可供用户进一步验证精度。未自动替换生产 Worker，也没有把收益推广到未测 batch、并发、显卡或整局搜索。

## Tensor Core 依据

实际正常完成的 INT8 运行日志记录了 `launch=cublaslt-int8 accumulator=int32 imma=true numerical_flags=0x200804`。这来自运行时对所选 cuBLASLt 算法属性的查询；[NVIDIA 将 IMMA 标志定义为整数 Tensor 运算实现](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmulnumericalimplflags-t)。日志只首次打印，能够证明已观察的算法选择，不能证明全网所有形状的 Tensor Core 覆盖或利用率。

当前路径使用 INT8 输入、INT32 累加与 TN 布局，随后反量化回原执行路径；混合配方中的 FP16 是显式选择。B15 的剪枝宽度和量化/反量化、补齐、kernel 启动成本会影响收益，不能仅用 INT8 峰值预测整网加速。旧 FP16 路径也使用 Tensor Core，比较对象不是纯 CUDA core 实现。

Nsight 查询性能计数器返回 `ERR_NVGPUCTRPERM`，因此硬件利用率没有测得。该错误表示计数器访问权限受限，见 [NVIDIA 官方说明](https://developer.nvidia.com/ERR_NVGPUCTRPERM)。没有修改驱动权限。常规无 profiler 的墙钟测量不依赖这些计数器。

[CUTLASS 的 SM120 文档](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html#blackwell-sm120-gemms) 区分了 Blackwell 的数据类型及执行条件。现有统一实现提供 FP16、INT8、MXFP8；没有 NVFP4 执行路径，不能把显卡 NVFP4 峰值当成本次 INT8 PTQ 的收益。

## 原始证据与复现

- 测量程序：`scripts/benchmark_quant_speed.py`，元数据准备程序：`scripts/prepare_quant_speed.py`。
- 初筛：`target/unified-quant-speed-only-r5/screen/report.json` 的前 6 个正常样本、`target/unified-quant-speed-only-r6/screen-remaining/report.json` 的 28 个样本、`target/unified-quant-speed-only-extra-r1/screen/report.json` 的 12 个样本，以及 `target/unified-quant-speed-only-tile-r1/screen/report.json` 的 2 个样本。
- 正式轮：`target/unified-quant-speed-only-extra-r1/abba-r1/run/report.json`。
- 独立确认：`target/unified-quant-speed-only-tile-r1/confirmation-r1/run/report.json`，已实际正常完成。
- 汇总：`target/unified-quant-speed-only-final-r1/results.json`；可复用配置/配方/环境/二进制身份：`target/unified-quant-speed-only-final-r1/speed-profiles.json`。
- 正式轮与确认轮的独立只读审计：`target/unified-quant-speed-formal-abba-review-r1`、`target/unified-quant-speed-confirmation-review-r1`；计数、日志、配置/profile 身份和独立算术核对通过，没有执行精度检查。
- 每次 native 的 stdout、stderr、PID、实际退出码、吞吐、请求/批数、profile 和前后 GPU 状态均与 report 同目录保存。计划绑定二进制、模型、配置、配方和环境摘要。

此前新入口中的元数据准备失败、旧 FP16 计划与新程序不兼容失败、profile 前缀解析失败，以及全 INT8 请求不兼容 DualFFN 的失败原件全部保留；成功样本作为独立事实使用，没有把失败的整轮改写为成功。旧 campaign、精度账本、单次入口、锁、预算和截止没有重启或重置。

已冻结计划的截止时间属于本轮测量，不能在过期后重用。后续复测应创建新的有限计划和输出目录。候选运行需要配置文件、完整环境覆盖和配方身份一起使用，不能只复制配置文件并继承其它实验的 `KATAGO_*` 环境。

本轮共 82 次 native 模型启动，其中 80 次 native 正常退出、2 次失败原件保留；包含 48 个正常初筛样本、20 个正式样本、8 个确认样本，以及 4 次加载冒烟。没有新增精度检查。复测元数据准备程序的真实构造检查也正常完成，未启动额外 GPU 推理。

复测已确认的两份 B15 配置，可在仓库 PowerShell 中执行以下命令。第一步只生成新的八次 ABBA 元数据，第二步才执行推理；输出目录必须是新的。此命令保留原实验，使用独立的新计划。

```powershell
python -S -B scripts/prepare_quant_speed_from_profiles.py --profiles target/unified-quant-speed-only-final-r1/speed-profiles.json --profiles-sha256 bdb6ff87cfe6f3b208208f58571f393dd09c1fb8093cc09bb0570f55ac73fd3e --output target/speed-check-new-r1
$quantSpeedPlan = Get-FileHash target/speed-check-new-r1/plan.json -Algorithm SHA256
$quantSpeedSchedule = Get-FileHash target/speed-check-new-r1/schedule.json -Algorithm SHA256
python -S -B scripts/benchmark_quant_speed.py --plan target/speed-check-new-r1/plan.json --plan-sha256 ($quantSpeedPlan.Hash.ToLowerInvariant()) --schedule target/speed-check-new-r1/schedule.json --schedule-sha256 ($quantSpeedSchedule.Hash.ToLowerInvariant()) --output target/speed-check-new-r1/run
```

若二进制、模型、配方、配置、runner 或固定环境已改变，准备/运行会拒绝，而不是把不同身份的结果混为本次结果。
