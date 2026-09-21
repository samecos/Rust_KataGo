# Go Server 大搜索树落子优化

2026-09-20。依据《大搜索树落子卡顿研究与解决方案》实施 Server 第一阶段优化。实现位于 `D:/Go/Server`，独立输出位于 `target/move-latency-v1/`。本次不修改 RustGo 的模型、CUDA 内核或 PLAN。

## 使用入口

新 Server：`D:/Go/Server/target/move-latency-v1/build/release/go-server.exe`。

将原启动命令的可执行文件路径换成上述路径，其余 HTTP/gRPC、模型 SHA、会话和 Worker 调度参数沿用。后台回收默认开启；`--synchronous-graph-reclamation` 可以只关闭后台析构，用于诊断对照。它并不恢复旧的换根扫描算法。

源码构建：

```powershell
cd D:/Go/Server
cargo build -p go-server --release --locked --target-dir target/move-latency-v1/build
```

既有 `target/release/` 和 `target/server-32g/` 程序未替换；没有接管既有会话或重启用户 Worker。验证服务使用独立临时端口，并只清理本次启动的子进程。

## 实现和边界

- 换根仍在单一会话所有者的安全点执行：先取消旧任务并撤销虚拟占用、增加核心 generation，然后标记可达图并完成编号映射。后台线程不遍历活动图，旧 EvalToken 的迟到结果不能写入新树。
- 使用密集编号数组去重标记，维护按原候选顺序排列的已连接边索引。共享节点和环只标记一次，候选边顺序及重复指向同一子节点的不同边保留。此索引也用于已有统计扫描，浮点运算顺序保持原样。
- 从旧 Vec 按旧编号倒序 `swap_remove` 保留节点，再恢复原顺序；死亡节点留在旧分配中。所有保留边编号和父索引在前台改写完成。父索引从子边重建，不能仅过滤旧 parents，因为重新绑定后的历史反向链接可能残留。
- 退休批次唯一拥有旧节点 Vec 和旧 HashMap。懒启动每会话一个回收线程，最多两个运行中/排队批次，同时检查剩余字节与节点额度。新搜索分配计入待回收字节；临时不足返回等待并加速回收，不永久进入 `memory_limited`。小批次、预算不足或线程创建失败走同步释放。退出统一排空并 join。
- 普通回收每256个节点休息约1ms，减轻与新搜索的分配器争用；队列/内存压力和退出时取消节奏延时。没有使用不受限的线程池或每落子创建一个回收线程。
- 图预算预留换根所需的标记/映射/栈、新 Vec 和新索引空间。退休计费另含旧容器 capacity；整批销毁后扣减，计费偏保守。它不是进程 RSS 硬上限，也不包含 Worker 的显存。

本阶段仍重编号并重建保留图，复杂度仍随图规模和保留边数增长。高保留率、快速连续落子导致的预算追赶仍可能停顿；稳定槽位/带代数 NodeId、并发增量 GC 尚未实现。

## 已完成的图性能验证

release，真实 Node/Edge 分配，每节点256个候选槽，含共享子节点和环；每组两轮 ABBA，即每侧4个样本。对照是测试中冻结的旧 `collect_unreachable` 实现。以下为前台标记、搬迁、重映射与提交回收批次的均值，不含后续异步析构，不是 WebSocket 往返 p95。

| 节点数 | 保留比例 | 原同步方法 ms | 新方法前台 ms |
|---:|---:|---:|---:|
| 10,000 | 10% | 14.05 | 0.30 |
| 10,000 | 90% | 19.42 | 2.11 |
| 100,000 | 10% | 158.26 | 2.53 |
| 100,000 | 90% | 204.44 | 48.17 |
| 300,000 | 10% | 599.71 | 13.54 |
| 300,000 | 90% | 664.89 | 159.03 |
| 1,000,000 | 10% | 6,024.35 | 61.31 |
| 1,000,000 | 90% | 2,077.32 | 516.91 |

64个样本全部通过内容摘要与结构断言，包括节点 key/统计/raw/ownership、子边及顺序、parents/index/稀疏索引；正常单元测试另覆盖编号空洞、100%保留、残留父链接、迟到结果和虚拟占用。

百万节点/10%保留的基线波动很大，范围2.10–8.60秒；候选范围54.35–71.50ms。因此这组只用作规模边界、正确性和明显延迟改善证据，不能当成稳定性能认证或足量 p95/p99 统计。百万节点实验进程峰值工作集13.46GiB，100ms间隔采样的峰值私有内存14.55GiB；这是单进程合成实验的峰值，不是长期 RSS 无增长证明。真实释放仍可能需要数秒，不能省略退休内存计费与背压。

第一次不设回收节奏的候选使紧接着的3000次新搜索评估约从77ms升到155ms，未采纳。最终版本加入节奏控制、复用已连接边进行统计扫描后，两轮 ABBA 的新搜索均值为同步对照74.84ms、后台候选67.04ms；换根本身为604.96ms、0.05ms。新搜索请求 Hash 和最终快照相同。该实验是丢弃整棵30万节点旧树后启动新搜索，不能冒充90%保留子树的结果。排空回收在搜索计时之后执行，报告另保留析构经过时间。

证据：`D:/Go/Server/target/move-latency-v1/abba-sparse-gc.json`、`abba-million.json`、`million-memory.json`、`abba-paced-search-final.json`。早期失败候选 `abba-background-search.json` 和中间版本 `abba-paced-search.log` 保留，未选择性删除。64个图样本使用相同的前台标记/搬迁/重映射算法；其中48个样本早于后台回收节奏调整，百万节点样本已包含节奏控制。最终版的后台争用和普通搜索吞吐另行复验；图数据不冒充每次后续构建的重新测量。

## 搜索吞吐与语义验证

普通搜索对照是实施前保存的 `go-core` 源码副本，原 `search.rs` SHA256 为 `7957c4913d0dee51df4dc1de516949e51a0f411e457542f482ee1f5efb3fc8e0`。固定 opening/fight/pass，窗口1/32，启用合成共享局面策略，每样本6000次评估，两轮 ABBA 共48样本。

所有请求 Hash 轨迹、最终搜索结果相等；只排除新增的诊断字段及调整后的内存计费。最终 CPU 搜索耗时的几何平均下降 **14.76%**，6组中位数下降范围 **13.34%–16.30%**。早期仅维护索引而未复用它进行统计扫描的版本增加约1.2%耗时，已被最终版本替代。完整结果：`throughput-final-summary.json` 与 `throughput-final-*.jsonl`，复现脚本 `run_search_abba.py`。

既有 b11 真实 NN 捕获 `reports/real-replay-20260917/full/nn.bin` 的 **120,000条请求 input hash** 全部匹配，12个阶段的搜索结果与原 capture 快照完全相等（排除内存计费和新诊断字段）。这验证搜索语义；并非本轮重新执行 Worker/CUDA 的数值金标。证据：`real-replay-final.jsonl`、`replay-summary.json`。

## 真实模型流程与工程检查

同一份新 Server，沿用已有、未改动的 RustGo Worker 二进制，分别加载 b11 和剪枝 b15。每个模型先完成8192 visits，然后执行22条生命周期命令；真实图规模约5千节点，用于接口与回收协议验证。

| 模型 | 初始图节点 | 首次主分支落子往返 | 保留节点 | 结果 |
|---|---:|---:|---:|---|
| 原生 b11 | 5,255 | 3.38ms | 783 | PASS |
| 剪枝 b15 | 5,218 | 2.73ms | 759 | PASS |

两组都实际启动了后台回收，覆盖新根继续分析、pass、undo、seek、改 komi、清盘、连续落子、搜索中换根、genmove、队列归零、正常退出。两组初次主分支落子的单样本时延不能外推为大树 p95。原始报告：`live-b11/report.json`、`live-b15/report.json`；模型 SHA 分别为 `1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6`、`3f216ee88226ee49ca826eaa5f7e1e9982ff48a96625914bfc4e5ad20ee095a7`。

- 默认 workspace：89项通过、9项按既有条件或显式基准要求忽略。
- 全特性 workspace：91项通过、9项忽略。
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`、`cargo fmt --all -- --check`、`git diff --check` 通过。
- 手动 release 基准之外的默认忽略项包括外部 C++ oracle 和前端环境测试；没有将它们计为本轮已通过。
- 新 Server SHA256：`027a0e0aa387a83175774b0cd7c1aed117eabe61b9c5f72c593ce6ed2d958968`。
- 原 `go-server-32g.exe` 仍为 `ef2d8f85ba38aa645152dc06b3f233c317517394e4e486414fc0df07db964ab8`；Worker 仍为 `6ca478ab0065f66140a71fdc26a1b1e253475467ff5971d92ba57b41f9821d32`。

本轮完成第一阶段实现与独立构建。**没有宣称达到所有规模的落子 p95 < 50ms / p99 < 100ms，也没有完成双模型真实大树端到端 ABBA 与长期 RSS 正式验收。** 下一阶段若继续压低高保留率停顿，应针对仍占前台时间的保留图搬迁、编号改写和父索引重建，评估稳定槽位/带代数 NodeId；不能直接把当前搬迁循环放到后台。

## 遥测和复现

WebSocket 的 `analysis.reclamation` 提供待回收节点/字节/批次数、峰值、已回收数量、同步回收次数和析构/预算等待耗时。`analysis.rootChange` 提供最近核心换根的取消、标记、搬迁、重建、提交回收和总耗时，以及首个新 NN 请求、完成、开始构造新分析快照的时间。核心 generation 与会话 generation 分别解释。详细字段见 `D:/Go/Server/docs/json_api_v1.md`。

```powershell
cd D:/Go/Server
cargo test --workspace --locked
cargo test --workspace --all-features --locked
$env:MOVE_PROBE_NODES = '10000,100000,300000'
$env:MOVE_PROBE_KEEP = '10,90'
$env:MOVE_PROBE_ROUNDS = '2'
cargo test -p go-core --release --lib root_collection_tests::abba_collection -- --ignored --nocapture --test-threads=1
$env:MOVE_PROBE_NODES = '300000'
cargo test -p go-core --release --lib root_collection_tests::abba_reset_and_search -- --ignored --nocapture --test-threads=1
```

合成图使用64GiB的逻辑预算，以覆盖百万节点；不是程序默认预算调整。普通 Server 默认仍为32GiB。大型实验不宜与其它性能测试或编译同时运行。

真实 Worker 生命周期检查脚本：`D:/code/Rust_KataGo/scripts/move_latency/verify_server.py`。传入 `--server/--worker/--model/--worker-config/--out-dir`；输出目录必须不存在，需要安装 websockets 的 Python。脚本验证已搜索主分支落子、pass、undo、seek、komi、清盘、连续落子、搜索中换根、genmove、退休队列排空和 Server 正常退出。它不修改模型/PLAN，也不能代替两个模型在10万/30万/100万真实图上的正式延迟分位数门。
