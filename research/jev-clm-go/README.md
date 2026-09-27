# Jev / CLM 与 Rust_KataGo：研究入口

状态：**两份归档的 CLM 固定着法头探针已完成；八月归档的新四头轻量模型与 M5 Pro 推理测速已完成；尚无棋力或部署认证，RTX 5070 Ti 待后续复测。**日期：2026-09-27。

本目录是本次工作的唯一新增位置。本轮不改 Rust/C++ 引擎、CUDA kernel、模型格式、Server/Worker 协议、正式二进制或认证 plan。原有 Fork 性能优化已结项；这里建立的是一项新的模型研究，不继续旧性能实验。

## 一句话结论

Jev 是 TypeSafe 的闭源「状态 → 有类型决策」服务；这里的 CLM 是开源 **Contrastive Language Models**，以状态和候选行动的对比表示做排序。它们针对文本决策，**没有已验证的围棋棋力**。CLM-8B 原样放进 KataGo 每叶推理既不满足现有 policy/value/目差/领地输出合同，也不适合本机单卡常驻。值得研究的是 **Go 原生、训练侧 CLM 辅助目标**：在现有棋盘网络上学习局面与着法/后续变化的相容性，最后仍由兼容的围棋主网络提供全部既有输出。是否采用，以同数据同计算量的对照实验与等时棋力判定。

## 目录导航

| 文件 | 内容 |
| --- | --- |
| [01-原始资料与证据.md](01-原始资料与证据.md) | Jev/CLM 机制、来源、发布日期、宣传数字与证据边界 |
| [02-项目适配与目标模型.md](02-项目适配与目标模型.md) | 本仓 I/O 和数据审计、应用场景排序、最终模型编排和规模 |
| [03-训练与验证协议.md](03-训练与验证协议.md) | 数据准备、损失、对照组、阶段门、棋力/数值/性能验收 |
| [04-推进记录.md](04-推进记录.md) | 已完成工作、当前阻碍、后续任务顺序和停机条件 |
| [05-真实数据试验.md](05-真实数据试验.md) | 用户归档全包清点、19 路筛选、MPS 三种子对照的结果和证据边界 |
| [06-八月归档复验.md](06-八月归档复验.md) | 新 `.tgz` 的 v2/v3 数据审计、相同预算复验和 Q 候选覆盖 |
| [07-新模型训练与本机测速.md](07-新模型训练与本机测速.md) | 全包筛选、四头模型权重、独立测试质量和 M5 Pro ABBA 推理速度 |
| [08-5070Ti移交说明.md](08-5070Ti移交说明.md) | 可同步模型/输入、RTX 复测命令和正式 RustGo 基线边界 |
| `audit_npz.py` | 自对弈训练文件的只读数据审计工具；不更改引擎或训练文件 |
| `census_tar.py`、`prepare_tar_sample.py` | 对 tar 作只读全包清点，安全地抽取可复现的训练子集 |
| `audit_q_candidates.py` | 在已通过审计的新包子集上统计有访问记录的 Q 候选覆盖；不证明 Q 标签可靠性 |
| `train_probe.py` | 独立 P1 小型 CNN 探针：等参数量的 CLM 相似度头与普通 policy 头对照；不接入引擎 |
| `prepare_model_data.py`、`student_models.py`、`train_student.py`、`benchmark_student.py` | 八月归档的独立四头模型数据准备、训练、checkpoint 和 MPS/CUDA 同机测速 |
| [`runs/2026-09-27-multitask-m5/`](runs/2026-09-27-multitask-m5/) | 两份训练后模型、便携 32 行真实输入、训练/测速 JSON 和数据摘要；这些文件可随项目源码同步 |
| `test_*.py` | 合成 NPZ、数据分组和探针的轻量验证 |

## 目前能立即做的事情

有真实自对弈 `.npz` 后，先运行数据审计：

```bash
research/jev-clm-go/.venv/bin/python research/jev-clm-go/audit_npz.py \
  /path/to/selfplay-data --output /path/to/audit.json
```

这个命令只验证数据格式、覆盖和按棋局分组切分；它**不代表完成 CLM 训练或证明棋力**。本 checkout 现有两份真实自对弈归档，均混合棋盘尺寸；第一份缺少 Q，第二份包含 Q 且混合 v2/v3 目标格式。相应的数据门和结果见 [首次试验](05-真实数据试验.md)与[新数据复验](06-八月归档复验.md)。

审计通过后，在本目录的独立 PyTorch 环境运行 P1 探针：

```bash
research/jev-clm-go/.venv/bin/python research/jev-clm-go/train_probe.py \
  /path/to/selfplay-data --out-dir research/jev-clm-go/runs/p1 --device mps
```

本机 macOS arm64 的隔离环境位于 `research/jev-clm-go/.venv/`，依赖版本记录在 [`requirements.txt`](requirements.txt)；重建命令：`/opt/homebrew/bin/python3 -m venv research/jev-clm-go/.venv && research/jev-clm-go/.venv/bin/python -m pip install -r research/jev-clm-go/requirements.txt`。`--device auto`（默认）会优先选 CUDA、再选 Apple MPS、最后选 CPU；显式 `--device mps` 会在 MPS 不可用时报错。默认以棋局哈希 80/10/10 切分、3 个种子分别训练两种精确等参数量的头，验证集选 epoch，测试集只在选定后评一次；结果写入 `manifest.json`。若只核查可用训练行与切分，可加 `--inspect-only`。该脚本用小 CNN 和 362 全槽访问目标，**没有合法手重放、B11 权重共享或棋力认证**。

第一份归档的 24 文件子集有 73,359 行可用 19 路数据；相似度头的测试交叉熵比等参数线性头平均低 `0.04228` nats（3 种子）。新 `.tgz` 的 1,800 文件子集有 75,726 行可用 19 路数据，同口径平均只低 `0.00378` nats，且 1 个种子更差。固定着法头的初始信号**未稳定复现**，不能以此推动 CLM 模型接入；详情见 [两次试验](05-真实数据试验.md)与[新数据复验](06-八月归档复验.md)。

随后按用户的新要求，用八月全包准备 1,418,024 条完整 19 路局面，训练了两个相同输出合同的独立四头 PyTorch 模型；分别从同局隔离的数据中取 200,000 / 20,000 / 20,000 行训练/验证/测试。`compact` 权重与同机 `dense` 基线相比，MPS 纯前向快 **1.28–1.59 倍**，含解包与输出回传快 **1.10–1.39 倍**，但测试 policy 交叉熵差 **+0.07590 nats**、top-1 差 **−0.695 个百分点**。这些是卷积主干缩小带来的速度与质量权衡，**不归因于 CLM**；权重、方法和限制见 [新模型报告](07-新模型训练与本机测速.md)。

## 目标和边界

- 近期目标：验证「对比目标是否比等参数普通 policy 辅助头提供额外信息」。先用独立训练代码和真实棋局数据做离线实验。
- 条件式最终目标：训练期在 TF3 B11 量级的围棋主干上加约 0.5–2M 参数的双投影/变化编码辅助分支；推理期只保留能映射到原有输入输出合同的主干及 policy/value/目差/ownership 头。参数是**设计预算**，不是本仓现有模型的实测值。
- 如果想把候选变化重排器在线接入搜索根节点，未来必须另开引擎/协议/后端任务，先通过数值和等时棋力门。当前只保留离线研究路径。

主要一手来源：[Jev 官方发布](https://typesafe.ai/blog/introducing-system-one-models-and-jev)、[CLM 原仓库](https://github.com/Contrastive-LM/CLM)、[CLM 模型卡](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B)。
