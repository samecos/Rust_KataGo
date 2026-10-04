# CUDA 训练与原型权重

2026-10-03：四头 `dense` / `compact` 研究模型已具备 CUDA 训练、FP16/BF16 混合精度、完整 epoch 恢复和原生 `.rgmodel` 自动导出。训练后的文件使用程序已有的 `--model` 入口；程序按文件内容识别新格式，无需模型类型参数。此页描述训练与权重生成，CUDA 引擎整图对拍与 GTP 结果见本轮兼容性报告。

本阶段代理未下载新的真实训练数据。下面的合成验证只证明训练到导出链可运行。它不证明棋力、CLM 收益或真实训练数据的输出质量。此前 M5 的训练和性能数字仍只适用于其原始研究记录。

## 安装 CUDA 环境

在项目根目录执行 PowerShell。本机已经建立该环境，无需重复安装。

```powershell
python -m venv research/jev-clm-go/.venv-cuda
research/jev-clm-go/.venv-cuda/Scripts/python.exe -m pip install -r research/jev-clm-go/requirements-cuda.txt
research/jev-clm-go/.venv-cuda/Scripts/python.exe -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

环境固定为 Python 3.11.9、官方 PyTorch `2.12.0+cu130`、NumPy `2.4.4`；本机 NVIDIA 驱动 `610.88`，RTX 5070 Ti，CUDA capability `12.0`，wheel 支持列表包含 `sm_120`。[PyTorch 官方安装矩阵](https://pytorch.org/get-started/previous-versions/)提供 CUDA 13.0 wheel，[2.12 发布说明](https://pytorch.org/blog/pytorch-2-12-release-blog/)说明 Blackwell 应使用 CUDA 13.0 及更高版本，并要求相应驱动。本机实际执行过 CUDA 卷积前向和反向。

本机还安装了不同版本的系统 cuDNN。官方 PyTorch 2.12 的 wheel 在加载可选 engine DLL 时可能混入系统版本，首次卷积出现 `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`；[上游记录](https://github.com/pytorch/pytorch/issues/188892)解释了相同问题。训练入口自动隔离当前 Python 进程的系统 cuDNN 搜索目录，保留 wheel 的 DLL，并将移除目录记录在 `manifest.json`；系统环境不变。其他 CUDA Python 工具可用通用启动器：

```powershell
& research/jev-clm-go/run_cuda_python.ps1 research/jev-clm-go/benchmark_student.py --help
```

旧 [`requirements.txt`](requirements.txt) 保留 M5 实验的依赖记录；Windows CUDA 使用 [`requirements-cuda.txt`](requirements-cuda.txt)。`.venv-cuda` 被 Git 忽略。

## 下载数据后准备缓存

原始数据需为本仓审计器支持的 KataGo v7 NPZ。归档需保留下载时的文件名，其 NPZ 成员路径必须是 `<归档去掉 .tar/.tar.gz/.tgz 后的 basename>/<一层模型目录>/<文件.npz>`，例如 `2026-09-27npzs/模型目录/文件.npz` 对应 `2026-09-27npzs.tgz`。准备器流式读取归档、校验形状和目标、筛选完整 19×19 policy 行、按整局哈希切分，并保存各监督头的有效权重。源数据不被修改；缓存目录必须不存在。

```powershell
research/jev-clm-go/.venv-cuda/Scripts/python.exe research/jev-clm-go/prepare_model_data.py `
  --archive D:/GoData/2026-09-27npzs.tgz `
  --out-dir research/jev-clm-go/data/prototype-v1 `
  --split-seed student-prototype-v1
```

已经解开的 NPZ 目录可将 `--archive` 替换为 `--input-dir D:/GoData/selfplay`；改名后与内部根目录不匹配、或采用其它嵌套布局的归档，也应先解包再用此入口。若复用以前探针涉及的归档，应依 [07 的命令](07-新模型训练与本机测速.md)提供 `--exclude-manifest` / `--prior-sample-dir`，隔离旧探针文件与棋局。归档未提供合法变化序列，这个入口训练的是现有四头研究网络，不能据此称为完成 CLM 变化编码。

## CUDA 训练与自动导出

默认使用 CUDA，CUDA 不可用会明确报错。默认混合精度为 CUDA FP16，损失按 FP32 计算，AdamW 保留 FP32 参数；GradScaler 在梯度溢出时跳过该步并降低 scale，每轮记录实际优化器更新与溢出跳步数。可显式选择 `--precision bf16` 或 `--precision fp32`；CPU/MPS 需显式选择设备且使用 FP32。

```powershell
research/jev-clm-go/.venv-cuda/Scripts/python.exe research/jev-clm-go/train_student.py `
  --cache-dir research/jev-clm-go/data/prototype-v1 `
  --out-dir research/jev-clm-go/runs/prototype-cuda-v1 `
  --epochs 5 --batch-size 128
```

该命令训练两档模型，使用相同选定数据、种子、优化器和顺序。默认选取最多 200,000 / 20,000 / 20,000 条 train/validation/test 行；用 `--train-rows 0 --val-rows 0 --test-rows 0` 取各 split 全部行。只训练轻量原型可加 `--variants compact`。batch 128 是示例配置，真实数据规模、耗时与显存需要在下载数据后测量。

各模型按 validation total loss 选择 best epoch，随后评估 test。每次恢复并延长训练会重新评估最终选定 checkpoint 的 test，因此不可通过反复查看同一 test 来选择超参数。输出包含：

| 文件 | 用途 |
| --- | --- |
| `compact.pt` / `dense.pt` | 验证集选出的 best checkpoint；可供 PyTorch 和导出器读取 |
| `compact.last.pt` / `dense.last.pt` | 最近完整 epoch 的模型、AdamW、GradScaler、数据顺序 RNG、Torch/CUDA RNG 与 best checkpoint |
| `compact.rgmodel` / `dense.rgmodel` | 自动导出的原生 CUDA 模型，FP32 权重；可直接交给已有 `--model` 入口 |
| `*.rgmodel.receipt.json` | 格式、输入版本、参数数、源 checkpoint 与模型 SHA，以及精确读回结果 |
| `manifest.json` | 数据/分片/选中行指纹、环境、各轮损失、溢出跳步、best epoch、权重 SHA 和 CUDA 显存峰值 |

新训练不覆盖已有非空输出目录。恢复时沿用同一配置与目录，加 `--resume`，`--epochs` 表示训练完成后的总 epoch 数：

```powershell
research/jev-clm-go/.venv-cuda/Scripts/python.exe research/jev-clm-go/train_student.py `
  --cache-dir research/jev-clm-go/data/prototype-v1 `
  --out-dir research/jev-clm-go/runs/prototype-cuda-v1 `
  --epochs 10 --batch-size 128 --resume
```

恢复严格核对缓存 manifest、分片 SHA、选中数据、种子、batch、学习率、精度和梯度裁剪设置。从最近完整 epoch 继续；中断 epoch 的未提交更新重做。双变体中已达到总 epoch 的模型跳过，尚未完成首 epoch 的模型按原 seed 开始。若中断发生于 best 文件更新成功、last 更新尚未提交的窗口，会保留新的 best 文件为 `*.best-before-resume.<sha>.pt`，从 last 中恢复对应的完整 best，不丢失恢复依据。

## 本轮真实 CUDA 验证

最终训练链回执为 [`target/student-cuda-training-smoke-r5/result.json`](../../target/student-cuda-training-smoke-r5/result.json)。在 RTX 5070 Ti 上实际使用 FP16，随机生成 16 / 8 / 8 条 train/validation/test 合成行，各模型先训练 1 epoch，再恢复到总计 2 epoch。两档模型的 40 / 32 个参数 tensor 均在恢复轮更新且有限，AdamW 状态延续至 4 步，GradScaler 状态恢复，`.rgmodel` 参数与 best checkpoint 精确一致。

另有 8 项训练专项测试全部通过：连续训练与分段恢复权重逐位一致、输入变化拒绝、已有输出不覆盖、双变体部分完成恢复、默认 CUDA 缺失报错、探针两种头实际 CUDA 优化器更新、best/last 中断窗口恢复、实际 FP16 溢出跳步降 scale 后正常更新。最终研究目录整体回归运行 58 项，54 项通过、4 项按平台条件跳过，包含上述训练测试和 14 项导出测试；日志为 [`target/student-cuda-engine-r1/python-final-tests.txt`](../../target/student-cuda-engine-r1/python-final-tests.txt)，不将 skip 计为执行通过。

首次 CUDA smoke `r1` 因上述系统 cuDNN 混库失败，原目录保留；修复后成功回执在独立目录生成。全部都是新的合成训练验证，不使用旧性能 campaign、旧精度预算或旧模型的认证身份。用户下载真实数据后，需另建训练目录，获得真实原型权重并执行输出质量与棋力验证。

## 2026-10-03 真实数据原型

用户提供的 `train-datas/2026-09-29npzs.tgz` 已完成本轮训练：4,035,076,636 字节，SHA-256 `52b176828d081432c918e1619ece32276208fd03cd5f6e10fad0591786739e09`。完整 gzip CRC/EOF 与 tar 路径检查通过，147,976 个 NPZ 均经准备器解析，8,649,817 条源行中筛出 6,098,286 条完整 19 路监督行。按棋局哈希划分后，训练/验证/测试各有 4,882,220 / 602,932 / 613,134 行、72,029 / 8,921 / 9,018 局；独立复算确认跨集合没有同局。

训练缓存为 `data/2026-10-03-sep29-cuda-prototype-r1/`。固定 `selection_seed=20261003` 选取 200,000 / 20,000 / 20,000 行，两个变体共用这些数据，CUDA FP16、batch 128、AdamW lr 0.001、seed 29，各训练 5 epoch。仅按 validation total loss 选择 best，两档均为第 5 轮；最终数据身份为 `c44dfffdaec177f8cb13e9c48f0fc9d3fdeda39a2d83549dcb71a1a3a2ea4f8d`。

可直接加载的真实权重：

- [compact.rgmodel](runs/2026-10-03-sep29-cuda-prototype-r1/compact.rgmodel)，181,562 参数，模型 SHA `0b00c12b5fceeea8b338fba0b84e356186ef5ba5b38c68bac02f8a2e4ab2d30d`。
- [dense.rgmodel](runs/2026-10-03-sep29-cuda-prototype-r1/dense.rgmodel)，463,002 参数，模型 SHA `a86c7108975626646f9494554e177056dfbe8ed6949ea4a3acb33516fb862755`。

```powershell
target/student-cuda-build-r1/release/katago-rs.exe gtp `
  --config configs/gtp_student_cuda.cfg `
  --model research/jev-clm-go/runs/2026-10-03-sep29-cuda-prototype-r1/compact.rgmodel
```

独立审计确认 `.pt` 与 `.rgmodel` 的 FP32 参数逐 tensor 完全一致，权重有限、数据指纹匹配。在固定 20,000 条测试行上，用 CUDA FP32（禁 TF32）重新计算官方 MCTS/TD/终局标签的加权指标，未以这些测试结果重新选择模型或修改训练配置：

| 指标 | compact | dense |
| --- | ---: | ---: |
| Policy CE（nats） | 3.19568 | 3.15328 |
| Policy teacher Top1 | 28.1474% | 28.4824% |
| Value CE | 0.69358 | 0.69443 |
| Value Brier | 0.45918 | 0.46003 |
| Score MAE（目） | 5.87841 | 5.89247 |
| Ownership MAE | 0.54793 | 0.53715 |

Policy 和 ownership 优于预先固定的朴素基线，value 小幅改善；score MAE 略差于仅从训练集计算的常数目差基线 `5.84118` 目，目差头仍需进一步研究。这些是标签拟合指标，尚无真实对局棋力结论。

两档权重的原生 CUDA 与 CPU PyTorch FP32 四头对拍（固定测试输入，batch 1/8）全部通过，最大原始输出绝对偏差 `8.58306884765625e-6`；各完成 GTP 落子与 16 条原 Worker evaluator 请求。8 次 native 启动均正常退出，无重试或清理；未连接或启动 Server，也未替换生产进程。汇总和完整证据见 [`target/student-real-prototype-r1/result.json`](../../target/student-real-prototype-r1/result.json)。

## 2026-10-03 三份新档案 CUDA 试玩模型

从 [KataGo 官方训练数据目录](https://katagoarchive.org/kata1/trainingdata/index.html) 下载 9 月 28、27、26 日三份新档案到 `D:/code/Rust_KataGo/train-datas`，共 12,154,787,867 字节。完整 gzip EOF/CRC、tar 路径与全部 446,669 个 NPZ 解析通过。SHA 为本地计算，官方目录未提供独立 SHA-256。

| 新档案 | 字节数 | 本地 SHA-256 |
| --- | ---: | --- |
| 2026-09-28npzs.tgz | 3,898,624,826 | `8aeed7a9e64d7d797f7953e0217a5115264c3ad5caed7dfb7a409ed5e6649d5b` |
| 2026-09-27npzs.tgz | 4,157,453,979 | `f95eaf34eb4cb6a626ee038834c984857452649438abd3cbcd99bd7bfcf4cf32` |
| 2026-09-26npzs.tgz | 4,098,709,062 | `f37bbe6ab4549b5d7ccff9bafccd94a4e96edeebfd2c2dc857106dc747c5f0a5` |

18,340,848 条完整 19 路候选行去除 2,106,941 条完整行重复后保留 16,233,907 行、270,575 局。训练/验证/测试为 12,998,437 / 1,627,512 / 1,607,958 行，216,752 / 26,944 / 26,879 局，独立全量哈希拆分复算无跨集同局。缓存位于 `data/2026-10-03-playable-merged-r1`，manifest SHA `17988df180ecdb50655bcbb428863ba16c85ec3070873a975126bf00e9d2c608`。沿用固定全局拆分 seed；这些日期中的棋局可能曾在历史日期中出现，不称新的盲测。

单个 dense 四头 CNN 学生模型，463,002 参数，从随机初始化开始。固定 selection seed 20261003、训练 seed 29、200 万 / 10 万 / 10 万行、CUDA FP16、batch 256、20 epoch、AdamW、余弦学习率 0.001→0.00001、max grad norm 10。完成 156,189 次优化器更新，71 次 AMP 溢出跳步均被记录，后续正常更新，权重和指标有限。完整训练正常退出 0，第 19 轮验证损失最低，best 与导出权重的 40 个 FP32 tensor 完全一致。Selected data SHA `9ba404e32da12f7eda0911d290b26a7dcc8141d0be37103999d114cc3d6b1945`。

权重在 [dense.rgmodel](runs/2026-10-03-playable-dense-r1/dense.rgmodel)，相同副本在 [试玩模型](../../models/student-playable-20261003/dense.rgmodel)，SHA `fb3b77e796db158d3fb6a836291f7efa0fb33bc5b7ac0f0f5f9e45ae8061721a`。原 CUDA 入口按内容自动识别，无需新增模型类型参数。

| 指标 | 最终模型（CUDA FP32） | 固定基线 |
| --- | ---: | ---: |
| Policy teacher Top1 | 35.40424% | 0.85785% |
| Policy teacher Top5 | 66.77451% | 2.24202% |
| Policy CE | 2.73920 | 5.86653 |
| Value CE | 0.66805 | 0.69824 |
| Value Brier | 0.44021 | 0.46732 |
| Score MAE（目） | 5.80685 | 5.88538 |
| Ownership MAE | 0.51309 | 0.99867 |

固定 10 万条测试行由独立 CUDA FP32（TF32 关闭）工具重新评估，不改变权重选择。策略基线采用训练集加权位置频率，其余为训练集胜负概率/均值目差、零 ownership。结果为离线标签拟合，未认证等级、Elo、CLM 收益或推理提速。

新程序修复编号 GTP 响应、SGF 双响应标记、SGF 载入后的悔棋起点及中国规则已计分终局输出。18 项独立 GTP CPU 检查实际通过，构建见 [verified-build.json](../../target/student-playable-build-r1/verified-build.json)。新权重完成 B1/B8 四头对拍，maxabs `2.86102294921875e-05`，以及 GTP/有界自弈共 5 次原生验收，全部正常退出 0，无清理或重试。selfplay-short：64 手，capped_controller_passes，控制器追加 2 次停一手；selfplay-longer：160 手，capped_controller_passes，控制器追加 2 次停一手。控制器追加停一手仅用于流程验证。使用方法见 [学生模型试玩](../../docs/学生模型试玩.md)。

新数据准备的三个统计包装器在已发布完整缓存后因 len(int) 报告错误退出 1，原失败保留；独立只读资格核查通过，未重做缓存。两次 HDD SQLite 合并进程经确认所有权后中止，部分目录和回执保留；实际完成的 r3 用有界 2 GiB SQLite 内容页的 RAM 索引，最终备份到 D 盘，CPU 差分通过且全部源首末身份稳定。未启动或连接 Server、未替换生产进程、未重启旧 campaign 或重置锁/预算，未提交推送。汇总 [result.json](../../target/student-playable-delivery-r1/result.json)，SHA `c4aa954baab00d164d00bdb42cd5227bf722e8b54285386220d940f731560d1d`。
