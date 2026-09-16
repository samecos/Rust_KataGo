# RustGo 对照 Fork 性能优化结项记录

结项日期：2026-09-16。用户明确确认主体工作完成，要求停止追加测试，保留当前可用版本、整理文档和使用说明，并关闭本轮目标。后续优化设想归入可选后续工作，不再作为本轮待办或自动继续的理由。

## 交付版本

正式使用 `target/release/katago-rs.exe`，对应本机 Windows RTX 5070 Ti / SM120。

| 项目 | 结项状态 |
|---|---|
| 原生 TF3 加载 | 直接读取指定 `.bin.gz`，模型 SHA 绑定原始文件 |
| 真实 Go Server Worker | 已接入；Server 持有搜索图，Rust 提供 NN 评估 |
| 高吞吐配置 | `configs/worker_tf3_sm120_throughput.cfg` + `plans/worker-tf3-sm120-throughput.json`，B14/S2/C64 |
| 已采纳优化 | strict Attention、stem 三维映射、C768 四通道门、只读 QKV/RoPE、QKV classic N128、既有 half 全零 FFN 通道压缩、Attention 输出投影 classic N128 |
| 日常/C32 配置 | 保留各自认证配置，未强制改为高吞吐组合 |
| 精度纪律 | FP16 存储；FP32 累加、关键激活/归约、残差与最终输出；精确激活及数值门保持 |
| 配置身份 | 模型、GPU、构建、能力和 AOT artifact 绑定；不匹配明确拒绝启动 |

正式二进制 SHA-256：

```text
d218b98dfdf2eb75820465cbf625dfd3f690b91c3bb112f3be1a56d2cc644dcc
```

TF3 模型：`D:/Go/Server/models/kata1-tf3-b11c768-s11001M-d5973M.bin.gz`。

```text
1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6
```

吞吐计划 SHA-256：

```text
7910f765134322f264722c7533d659f5ee0f87828c48f444c3eb6cd1770dc7e4
```

## 已取得的性能结果

- 当前认证 Worker：**1574.822297 uncached RPC/s**。最近采纳的输出投影 N128 在同 CLI 的配对 ABBA 中为 1557.022297 → 1574.822297 RPC/s，提升 **1.143208%**；对应完整前向共同墙钟提升 **1.456147%**。
- 最新 Fork 参考：同原始模型、同 GPU、B14、双流、统一共同墙钟，Rust Windows **1595.982821 NN rows/s**，Fork WSL 测试副本 **1563.247273 NN rows/s**，比值 **102.094074%**；两臂重复波动为 0.176416% / 1.125760%。
- Fork 对照仍存在 Windows/WSL、运行库和累加精度差异；这是已明确边界的参考，不声称同精度优势，也不是 Fork Worker 吞吐比较。Fork 历史 2418.947898 是另一种 lane event 统计，不能直接与 Rust Worker 相除。
- 历次优化收益来自不同配对基线，不能相加为总提升。详情保留在 [性能审计](tf3-worker-performance-audit.md)。

## 验证与最新候选裁决

正式输出投影 N128 版本已完成整网数值、真实 Worker、配置身份、旧 ONNX、Graph、计划迁移及安装后实际路径验收。

最后一个额外候选是 compact FFN down NN 布局，独立于正式版本：

| 检查 | 已完成结果 |
|---|---|
| 组件数值/内存/竞争 | 34 组、131,970,048 个保存 FP32 输出与基线逐位一致；FP64 参考门通过；memcheck 0 错误/0 泄漏，racecheck 0 错误/0 警告 |
| 整网 | B1/B14/B16、两臂双流六组；612,960 个输出值与现用版本逐位一致 |
| 真实 Worker 数值 | W1/W32/W64 两臂共 768 请求，首选 768/768，11 个 C++ FP32 数值字段通过；原始协议和 Drain 复核通过 |
| 实际路径 | 7,248 次内核执行；每次前向仅 33 个目标 down 内核变化，其余 269 个保持 |
| 兼容与身份 | 40 项计划单测、2 项转置/批量单测、20 项 CLI 拒绝用例、旧 ONNX/ORT 及 B1/B14 Graph 通过 |
| 唯一完整前向 ABBA | 共同墙钟 **+1.264881%**，event **+0.430978%**；均稳定，但 event 未达到预先登记的 1% 收益门 |
| 裁决 | **不采纳**；未进入 Worker 性能测试，未重测，未迁移计划 |

“内存检查”不等于发现内存泄漏。上述已完成检查没有发现泄漏或非法内存访问；其范围是这些已测路径和用例，不能扩大为所有未来输入与无限运行时间的证明。早前检查器连接问题和已解释的 API 查找报告保留在历史记录中，不混写为本次内存错误。

这项候选的组件性能失败结论也保留；独立整网实验没有将其改成组件通过。其他已关闭的慢项、不稳定项和未通过数值门的候选均不进入正式计划。

## 记录与使用入口

- [RustGo 使用指南](RustGo使用指南.md)：本机直接启动、配置选择、Server、GUI/GTP、停止和常见问题。
- [TF3 部署说明](TF3权重适配与Worker部署.md)：模型范围、部署参数及维护者参考流程。
- [CUDA 主规划](cuda-fork-parity-plan.md)：历次机制、测试边界和裁决。
- [TF3 Worker 性能审计](tf3-worker-performance-audit.md)：历史原始结果和计时口径。

本机证据目录（位于被 Git 忽略的 `target/`，应与交付环境一起保留，不能当作可随意删除的临时数据）：

- [`outproj-classic-tn-n128-integration-r1/final-review.json`](../target/fork-parity-20260908/outproj-classic-tn-n128-integration-r1/final-review.json)：正式版本采纳与迁移。
- [`outproj-current-fork-reference-r1/`](../target/fork-parity-20260908/outproj-current-fork-reference-r1/)：当前 Fork/Rust 统一边界参考。
- [`compact-down-nn-integration-r1/final-review.json`](../target/fork-parity-20260908/compact-down-nn-integration-r1/final-review.json)：最新未采纳候选完整裁决。
- [`goal-closeout-r1/closeout.json`](../target/fork-parity-20260908/goal-closeout-r1/closeout.json)：本轮结项身份与文档清单。

用户要求收尾后，没有新增构建、GPU 运行、基准或引擎测试。收尾只修改文档并核对现有文件身份；引擎源码、正式二进制和认证计划保持原样。工作区原有未提交修改保留，本轮未额外提交、推送或清理实验档案。

## 本轮结束后的边界

更广泛的 batch/并发调优、同平台同精度对照、长期压测、其他 GPU/模型及 Linux 新组合均属于以后另行提出的工作；未测项目不宣称通过。它们不影响用户确认本轮主体交付完成，也不会在本目标关闭后继续执行。

历史阶段报告里的 `active` 和“下一步”表示当时状态；以本结项记录为当前决定。本轮目标关闭，后续只有收到新的明确任务才继续优化。
