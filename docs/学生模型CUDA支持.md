# 学生模型 CUDA 支持

2026-10-03：合并 `origin/main` 的研究提交 `71dc36ab11deda063d9d2bbd0b5f4ad03c95fe47`，保留本地两条推理/精度提交。新的工程支持对象是 `research/jev-clm-go/student_models.py` 中两个已定义的四头网络：`compact`（48 通道、4 个残差块）和 `dense`（64 通道、6 个残差块）。原始研究和 M5 结果仍保留。

三份新官方档案的 dense 模型现已完成 200 万条训练数据、20 轮 CUDA 训练和实际 GTP 验收。当前可用权重、程序与 Sabaki 接入步骤见 [学生模型试玩](学生模型试玩.md)，训练及质量指标见 [CUDA 训练与原型权重](../research/jev-clm-go/09-CUDA训练与原型权重.md)。

## 自动识别与执行

沿用 `nnBackend=cudabackend` 和已有 `--model`，程序读取权重内容的 16 字节 magic `RustGoStudentV1\n` 自动识别。文件扩展名不参与学生模型路由；不需要 `--model-type`、变体开关或新的后端参数。模型变体、输入版本、目差尺度和张量清单均来自文件，并在创建 CUDA 设备前验证。

训练完成会自动生成 `compact.rgmodel` / `dense.rgmodel`。现有 PyTorch `.pt` 检查点也可转换一次：

```powershell
research/jev-clm-go/.venv-cuda/Scripts/python.exe research/jev-clm-go/export_student.py `
  --checkpoint research/jev-clm-go/runs/2026-09-27-multitask-m5/compact.pt `
  --output target/student-model/compact.rgmodel
```

`.pt` 是训练检查点，`.rgmodel` 是独立推理权重。引擎不执行 Python 或 PyTorch，也不需要 ONNX / TensorRT runtime。格式为 magic、little-endian u32 JSON 长度、版本化 JSON 清单和连续 little-endian FP32 参数。解析器拒绝缺失/重复参数、错维度、非有限值、错偏移、截断、尾随字节和未知架构。模型 SHA 绑定实际读取的完整文件。

CUDA 推理使用手写算子、im2col 和 cuBLAS SGEMM，权重、激活和累加均为 FP32，明确关闭 TF32。每次前向使用独立 CUDA stream、cuBLAS handle 和工作区；新路径支持多行 batch 和并发调用。旧 TF3/ONNX 的执行器、INT8 配方和原 Worker protobuf 协议保持原接口。

## 构建与运行

本阶段使用独立构建目录：

```powershell
cargo build -p katago --features cuda --release --target-dir target/student-cuda-build-r1

target/student-cuda-build-r1/release/katago-rs.exe gtp `
  --config configs/gtp_student_cuda.cfg `
  --model research/jev-clm-go/runs/prototype-cuda-v1/compact.rgmodel
```

同一个配置可以加载两个变体。`configs/gtp_student_cuda.cfg` 设置 19 路、NCHW 和标准 CUDA 后端，不引用旧模型的 tactic plan。旧模型的量化配方或认证 plan 不能用于学生网络；加载时遇到这些选项会明确报错。

已有原协议 Worker 也使用同一模型参数，例如：

```powershell
target/student-cuda-build-r1/release/katago-rs.exe nnworker `
  --server 127.0.0.1:50051 --worker-id student-cuda `
  --model research/jev-clm-go/runs/prototype-cuda-v1/compact.rgmodel `
  --config configs/gtp_student_cuda.cfg --capacity 8
```

Server 仍按模型 SHA 区分池。学生路径提供包含实际模型、可执行文件、CUDA build、GPU/驱动/cuBLAS、batch 和布局的执行身份；不冒用已有 TF3 profile。上述 Worker 命令用于用户自己的测试服务，本阶段没有替换或停止生产进程。

## 四头与引擎语义

| 训练原始输出 | 引擎处理 |
| --- | --- |
| `policy_logits [B,362]` | 361 个行优先交点和 pass；逆转输入对称，按真实历史过滤非法手并 softmax。只有一个 policy 头，optimism 不改变其 logits。 |
| `value_logits [B,3]` | 行棋方的胜/负/无结果 logits；沿用规则约束、softmax 和白方视角转换。 |
| `score [B]` | 行棋方的 20 目单位；乘 20，再沿用无结果概率和白方视角处理。`lead` 复用同一估计。 |
| `ownership_logits [B,361]` | 逆转输入对称、tanh 并转白方视角。 |

学生权重格式版本为 1，输入是 v7（22×19×19 空间特征 + 19 全局特征）。引擎使用 v8 基础输出接口语义，使时间/短期误差头明确不可用：`supports_shortterm_error=false`，对应字段为 -1。没有训练 score stdev 头，条件目差方差取 0；无结果事件按现有规则取 0 目，因此最终二阶矩仍可能包含无结果混合产生的方差。这里的 v8 是引擎接口兼容版本，不是把学生网络称为 TF3 或官方 KataGo v8 权重。

这是远端研究中已实现的轻量 CNN 原型，没有在线 CLM-8B 分支或候选变化重排器。原研究尚未证明 CLM 辅助目标带来稳定收益，本次工程接入不新增此类结论。新的学习目标或更大主干需要独立模型格式版本和训练实验。

## 验证与数据下一步

本阶段验证模型导出/读取、PyTorch FP32 与原生 CUDA 原始四头对拍、8 种对称、batch/并发/错误拒绝、GTP 搜索，以及原 Worker 的评估与后处理。CUDA 训练用明确标注的合成监督数据验证梯度、混合精度、断点恢复和自动导出。合成模型用于执行链检查，不能作为棋力评估模型。

RTX 5070 Ti 上最终四头对拍完成 12 次正常退出：两个变体 × 真实研究输入/固定合成输入 × batch 1/3/8，34,896 个原始输出值最大绝对偏差 `1.430511474609375e-5`，全部满足预设容差。两档合成训练权重均通过 GTP 实际落子及每档 16 条 Worker 评估请求。Rust CPU 回归为 `kata_nn` 217 项、`kata_worker` 35 项通过；Python 为 54 项通过、4 项跳过。汇总与原始证据索引见 [`target/student-cuda-delivery-r1/result.json`](../target/student-cuda-delivery-r1/result.json)。

用户准备真实 `.npz` 或 `.tgz` 后，按 [CUDA 训练与原型权重](../research/jev-clm-go/09-CUDA训练与原型权重.md) 筛选 19 路、按棋局切分并训练，再将自动导出的 `.rgmodel` 交给同一个程序。真实数据训练、测试质量和棋力验证属于这个后续数据阶段。没有新性能认证或旧 campaign 重跑。
