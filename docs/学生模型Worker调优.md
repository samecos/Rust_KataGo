# CUDA 学生模型 Worker 调优

2026-10-04 已修复学生模型的批量调度和 CUDA 资源复用。RTX 5070 Ti 上，当前试玩 dense 模型使用 **batch 32、capacity 64、1 个 NN 线程**。程序仍通过普通 `--model` 参数按权重内容识别模型。

在旧 Worker 的终端按 `Ctrl+C`，然后从任意目录执行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:\code\Rust_KataGo\scripts\run_student_worker.ps1"
```

默认连接 `127.0.0.1:50051`，Worker ID 为 `student-dense-5070ti-tuned`。现有 Server 的 target inflight 64 已足够，直接使用现有 Server。连接远端时可在上述命令末尾追加 `-Server "服务器地址:50051"`。`-PrintCommand` 仅检查文件并打印参数。

等价的完整命令：

```powershell
& "D:\code\Rust_KataGo\target\student-worker-tuning-20261004-r1\optimized-build\release\katago-rs.exe" nnworker `
  --server 127.0.0.1:50051 `
  --worker-id student-dense-5070ti-tuned `
  --model "D:\code\Rust_KataGo\models\student-playable-20261003\dense.rgmodel" `
  --config "D:\code\Rust_KataGo\configs\worker_student_cuda.cfg" `
  --capacity 64
```

必须使用这个新程序才能获得调度修复。配置文件控制物理 batch 上限；CLI capacity 控制最多接纳多少请求，两者分别为 32 和 64。配置启用正常 NN 缓存；性能测试关闭缓存，以确保每个请求都实际执行网络。

## 原因与修改

观察到旧 Worker 的累计 NN rows 和 NN batches 均为 51,739，历史平均物理 batch 为 1。旧命令只声明 capacity 8，现有 Server 的 target 64 因此被限制为 8。观察时没有活动任务，GPU 0% 的空闲快照不能用于判断运行速度。

学生模型的 CUDA 调用同步完成推理，但后端整体声明支持异步流水线。流水线过早提交单条请求，无法正常聚合。这次增加按实际 compute handle 判断能力：学生模型走正常批量聚合，原 TF3 的异步流水线保持原入口。

CUDA 每次调用还会创建 stream、cuBLAS handle 和临时显存。现在每次调用独占一套可复用资源，工作区按需增容，只计算和复制实际 batch 的前缀。最多保留 8 套空闲资源；执行期间不持资源池锁。执行失败或输出非有限数时丢弃资源。推理仍为原来的 FP32 pedantic 运算和算子顺序。

## 本机实测

当前模型 SHA-256 为 `fb3b77e796db158d3fb6a836291f7efa0fb33bc5b7ac0f0f5f9e45ae8061721a`；新程序 SHA-256 为 `0d4ed32a8c46b968048552a4f67fda28dc8bc374304217ba6762bd3da26ae557`。同一模型、同一 64 个合法历史局面，经过初筛后用独立旧／新／新／旧四进程复测，每进程预热 2 秒、测量 3×5 秒。

| 指标 | 旧 B8 / C8 / T1 | 新 B32 / C64 / T1 |
| --- | ---: | ---: |
| Worker 评估链吞吐 | 428.91 请求/秒 | 11,958.15 请求/秒 |
| 平均请求延迟 | 18.620 ms | 5.345 ms |
| 六个窗口各自 P95 的范围 | 20.669–21.692 ms | 5.606–5.819 ms |
| 实际平均物理 batch | 1.000 | 31.941 |

确认吞吐为 **27.88 倍**；两次旧进程漂移 0.72%，两次新进程漂移 1.75%。初筛的新 B8 / C64、B16 / C64、B32 / C64 分别为 8,179、9,960、11,984 请求/秒。新程序在旧 B8 / C8 设置的三次诊断值约 6,670–6,696 请求/秒，这些诊断不参与选优或独立确认。

计时覆盖 Worker Evaluator 的历史重放、NN 排队、CUDA 推理、后处理与响应解码；预热和加载不计入吞吐，截止前接纳的尾部请求全部完成。没有计入 protobuf 序列化、RPC 或 Server 搜索，实际对局的速度还取决于 Server 供给请求的速度。

确认阶段每组各有 34 个 device 0 的 `nvidia-smi` 全进程辅助样本，GPU 利用率均值约 **35.1% → 61.4%**。样本覆盖加载、预热与测量，缺少精确窗口锚点，只用于辅助观察。这个 463,002 参数的小网络在请求较少或等待 Server 时仍会空闲。

## 验证与证据

批量调度的 14 项 CPU 检查、资源池的 8 项 CPU 检查、测量工具的 7 项 CPU 检查通过。原两槽异步调度、错误唤醒、批量上限及独占资源的回归均通过。

新 CUDA 程序在单进程中执行 `32→1→8→32` 两轮和两个线程调用，共 10 组、40 个输出头、152,670 个值。四头与固定 CPU FP32 参考比较通过，最大绝对差 `2.86102294921875e-05`。串行 8 次调用只创建 1 套执行资源、增容 1 次，另外 7 次复用。线程同时起跑的输出验证通过；这不单独证明 GPU 内核重叠。

本轮独立范围为 1 次 CUDA 兼容进程和 8 次性能进程，全部 owned 正常退出 0，无清理或自动重试。构建时基准示例的两处颜色类型转换错误、测量工具首次模拟测试失败和启动回执说明更正均保留原件；它们未消耗 GPU 性能额度。未调用旧量化 campaign，也未重置其锁、消费或截止。

48 次只读 HTTP GET 的生产快照均成功，原 Worker 连接身份和计数保持一致且无活动任务。收尾 CIM 仍观察到原 Server PID 3380 和旧 Worker PID 67244，出生时间和命令一致；没有停止或替换它们。`D:\Go\Server` 保持原 HEAD 和干净工作区。模型未重新训练，未新增棋力或等级结论。

完整汇总：[delivery.json](../target/student-worker-tuning-20261004-r1/delivery.json)。性能：[result.json](../target/student-worker-tuning-20261004-r1/result.json)，实际退出：[performance-owned-final.json](../target/student-worker-tuning-20261004-r1/performance-owned-final.json)，辅助统计：[postrun-observations.json](../target/student-worker-tuning-20261004-r1/postrun-observations.json)，四头与复用：[cuda-reuse-result.json](../target/student-worker-tuning-20261004-r1/cuda-reuse-result.json)。
