# KataGo Rust 移植方案

> 版本：v2.0
> 日期：2026-07-16
> 范围：基于当前目录 `KataGo/` 下的 C++ 代码库，制定逐步迁移到 Rust 的工程方案。
> 状态：core / game / dataio / search / program / command / book / distributed 已大体完成；当前聚焦 neuralnet 后端骨架与 NNEvaluator 异步批量推理。

---

## 1. 目标与范围

### 1.1 总体目标
将 KataGo C++ 引擎以**可验证、增量、最小可用优先**的方式迁移到 Rust，最终达到：
- 核心棋类游戏逻辑、搜索、NN 评估等价于原 C++ 实现。
- 至少支持 CPU/Eigen 后端作为默认后端，CUDA/TensorRT/OpenCL/Metal 后端后续接入。
- 提供与原 C++ 版本可对比的 GTP、analysis、benchmark 命令行入口。
- 保持与现有训练生态（PyTorch 导出模型、配置、SGF、selfplay 数据格式）兼容。

### 1.2 范围边界
| 范围 | 说明 |
|------|------|
| **纳入移植** | `cpp/core`、`cpp/game`、`cpp/search`、`cpp/program`、`cpp/dataio`、`cpp/neuralnet`（抽象层 + 各后端骨架 + ONNX 模型加载）、`cpp/tests`。 |
| **延后或封装** | CUDA/TensorRT/OpenCL/Metal 后端：初期通过 FFI / 独立子进程或 `cust`/`tch`/`wgpu` 逐步替换；`distributed` 客户端已用 `reqwest` 重写。 |
| **不移植** | `python/` 训练脚本、CoreML 导出器（`katagocoreml`）、外部 vendored 第三方库如 `cudnn-frontend`、`tclap` 等，用 Rust 生态替代。 |

---

## 2. 现状盘点

### 2.1 代码规模（基于当前目录统计）
- `cpp/` 总文件数：约 780
- `.cpp`：145，`.h`：211，`.hpp`：31
- `python/*.py`：约 56

### 2.2 主要模块

```text
KataGo/cpp/
├── core/          # 基础工具：日志、配置解析、随机数、哈希、文件、线程、并发容器
├── game/          # 围棋规则、棋盘、落子历史、图哈希
├── search/        # MCTS 搜索树、节点、PUCT、多线程搜索、评估缓存、时间控制
├── neuralnet/     # NN 描述、输入编码、模型加载、后端接口、各硬件后端
├── program/       # 高层程序：GTP、对弈、selfplay 管理、设置
├── dataio/        # SGF、训练数据读写、numpy 导出、home 目录处理
├── command/       # 各 CLI 子命令入口
├── book/          # 定式书
├── distributed/   # 分布式 selfplay 客户端
└── tests/         # 单元/回归测试
```

### 2.3 关键外部依赖映射

| C++ 依赖 | 作用 | Rust 候选方案 |
|----------|------|---------------|
| 原生线程 / 条件变量 | 并发 | `std::sync` / `parking_lot` / `tokio`（仅分布式 I/O） |
| `half.hpp` | FP16 | `half` crate |
| `nlohmann/json` | JSON | `serde_json` |
| `tclap` | CLI 参数 | `clap` |
| `httplib` | HTTP 客户端 | `reqwest` |
| `ghc::filesystem` | 文件系统 | `std::fs` / `std::path` |
| Eigen | CPU 矩阵 / NN | `ndarray` + `matrixmultiply`；或 `tch-rs`（LibTorch） |
| CUDA/cuDNN/TensorRT | GPU 推理 | `cust`、`cudarc`、`tensorrt-rs`（社区）；或保留 C++ shim |
| OpenCL | GPU 推理 | `ocl` |
| Metal | Apple GPU | `metal-rs` |
| ONNX 模型解析 | 加载权重 | `tract-onnx` 或 `onnxruntime-rs` |

---

## 3. 移植策略

### 3.1 核心原则
1. **增量迁移**：先让 Rust 版本能跑起来，再逐步替换模块。不要一次性重写全部 145 个 `.cpp`。
2. **测试先行**：每个被移植模块必须配套 Rust 单元测试，并与原 C++ 模块输出做差异对比。
3. **模块隔离**：按 `core → game → dataio → neuralnet(抽象+CPU) → search → program → command` 顺序推进。
4. **FFI 桥接**：在搜索、NN 后端等复杂模块未完全迁移前，可先用 `cxx`/`cbindgen` 让 Rust 调用稳定的 C++ 逻辑，反之亦然。
5. **避免过早抽象**：不引入通用围棋引擎框架，先保持与原 C++ 结构 1:1 对应，降低心智负担。

### 3.2 推荐架构：Rust 为主 + C++ shim 为辅
```text
+---------------------+
|  katago-rs CLI      |  Rust (clap)
+---------------------+
|  program / command  |  Rust
+---------------------+
|  search             |  Rust (优先)
+---------------------+
|  neuralnet          |  Rust 抽象层 + Eigen/ONNX CPU 后端 + GPU 后端骨架
+---------------------+
|  game / dataio      |  Rust
+---------------------+
|  core               |  Rust
+---------------------+
|  C++ shim (可选)    |  CUDA/TRT/OpenCL/Metal 后端临时 FFI
+---------------------+
```

### 3.3 与 C++ 代码共存策略
- 在 `KataGo/` 旁新建 `katago-rs/` 目录（或 `rust/` 子目录），不破坏原 `cpp/`。
- 使用 `cargo` 管理 Rust 子项目；通过 `build.rs` 将 C++ shim 编译为静态库并链接。
- 当某模块 Rust 版本通过全部对应测试后，删除对应 C++ shim。

---

## 4. 模块级移植计划

### 阶段 0：基础设施（已完成）
目标：建立 Rust 工程、CI、基准测试 harness。

| 任务 | 输出 | 验收标准 |
|------|------|----------|
| 创建 `katago-rs/` Cargo workspace | `Cargo.toml`、workspace 结构 | `cargo build/test/clippy` 通过 |
| 配置 CI | `.github/workflows/rust.yml` | PR 触发 fmt/clippy/test |
| 建立 `kata-go-test-harness` | 与原 C++ 可执行文件输出对比脚本 | 可对比 GTP 命令输出 |
| 引入基础 crate | `anyhow`、`thiserror`、`serde`、`serde_json`、`clap`、`log`、`env_logger`、`rand`、`parking_lot`、`ndarray` | `Cargo.lock` 提交 |

### 阶段 1：core + game（已完成）
这是风险最低、最独立的模块，优先完成以建立信心。

#### 4.1 core → `kata-core`
| C++ 文件 | Rust crate/模块 | 注意点 |
|----------|-----------------|--------|
| `global.h/cpp` | `kata_core::global` | 断言、宏、平台差异 |
| `logger.h/cpp` | `kata_core::logger` | 用 `log` facade + 自定义格式化 |
| `config_parser.h/cpp` | `kata_core::config` | 保持与原 `.cfg` 解析语义一致 |
| `rand.h/cpp`, `rand_helpers.h/cpp` | `kata_core::rng` | 对齐 Zobrist / PCG / xoshiro 种子行为 |
| `hash.h/cpp` | `kata_core::hash` | `std::collections::hash_map::DefaultHasher` 不行，需定制 |
| `fancymath.h/cpp` | `kata_core::math` | SoftPlus、LogSumExp、Dirichlet 噪声 |
| `fileutils.h/cpp`, `makedir.h/cpp`, `datetime.h/cpp` | `kata_core::fs`, `kata_core::time` | 用 `std`/`chrono` 替代 |
| `multithread.h/cpp`, `parallel.h/cpp`, `threadsafecounter.h/cpp`, `threadsafequeue.h/cpp` | `kata_core::thread` | `std::thread` + `crossbeam` 队列 |
| `test.h/cpp`, `threadtest.h/cpp` | `kata_core::test` | 用 `#[cfg(test)]` + `insta` 做快照 |

#### 4.2 game → `kata-game`
| C++ 文件 | Rust crate/模块 | 注意点 |
|----------|-----------------|--------|
| `rules.h/cpp` | `kata_game::rules` | 中日韩规则、贴目、座子 |
| `board.h/cpp` | `kata_game::board` | 位运算棋盘，性能关键；考虑 `u64`/`u128` 或 `BitVec` |
| `boardhistory.h/cpp` | `kata_game::history` | 状态栈、superko、pass 次数 |
| `graphhash.h/cpp` | `kata_game::graph_hash` | 与搜索图哈希对齐 |

**阶段 1 验收**（已完成）：
- 全部 `testboard*`、`testrules`、`testmisc`、`testsgf` 等价的 Rust 测试通过。
- Rust 棋盘状态机与原 C++ 对同一 SGF 序列产生一致最终状态。

### 阶段 2：dataio + 配置持久化（已完成）
| C++ 文件 | Rust crate/模块 | 注意点 |
|----------|-----------------|--------|
| `sgf.h/cpp` | `kata_data::sgf` | SGF 解析/生成，注意编码 |
| `files.h/cpp` | `kata_data::files` | 训练数据目录管理 |
| `numpywrite.h/cpp` | `kata_data::numpy` | 输出 `.npy`/`.npz`，可用 `npz` crate 或手写 |
| `trainingwrite.h/cpp` | `kata_data::training` | TFRecord 格式或兼容 numpy 数据流 |
| `poswriter.h/cpp` | `kata_data::pos_writer` | 位置写入器 |
| `loadmodel.h/cpp` | `kata_data::model_loader` | 模型文件发现、修改时间更新、旧模型清理 |

**验收**（已完成）：能正确解析 KataGo 自带测试 SGF；能写出训练脚本可读取的 numpy 文件。

### 阶段 3：neuralnet 抽象层 + CPU/GPU 后端骨架（进行中）
这是移植的难点之一。先完成**描述 + 输入 + 后端接口 + 后端骨架**，保证搜索可运行；真实推理逐步替换 dummy 占位。

| C++ 文件 | Rust crate/模块 | 状态 | 注意点 |
|----------|-----------------|------|--------|
| `nninterface.h` | `kata_nn::backend::Backend` trait | done | 定义 `eval` 接口、marker traits、dummy backend 测试 |
| `nninputs.h/cpp` | `kata_nn::inputs` | done | 特征平面编码，务必与原 C++ 逐通道比对 |
| `desc.h/cpp`, `modelversion.h/cpp` | `kata_nn::desc`, `kata_nn::version` | done | 模型结构描述、版本校验；istream 模型解析留作后续切片 |
| `activations.h` | `kata_nn::activations` | done | ReLU、GELU、Mish、SoftPlus |
| `dummybackend.cpp` | `kata_nn::backends::dummy` | done | 占位后端，用于测试搜索 |
| `trtbackend.cpp` | `kata_nn::backends::trt` | wip | 已创建模块与 `trt` feature；真实推理待 FFI 集成 |
| `eigenbackend.cpp` | `kata_nn::backends::eigen` | pending | 用 `ndarray` 重写，或 `tch-rs` 走 LibTorch CPU |
| `cudabackend.cpp` | `kata_nn::backends::cuda` | pending | C++ shim 或 `cudarc` |
| `openclbackend.cpp` | `kata_nn::backends::opencl` | pending | `ocl` 或 shim |
| `metalbackend.mm` | `kata_nn::backends::metal` | pending | `metal-rs` 或 shim |
| `nneval.h/cpp` | `kata_nn::eval` | wip | 评估队列、批量、缓存；server-thread 与异步批量推理为当前重点 |
| `debugprint.h/cpp` | `kata_nn::debug` | done | NN 调试输出 |
| `sgfmetadata.h/cpp` | `kata_nn::sgf_meta` | done | SGF 元信息嵌入 |
| `onnxmodelbuilder.h/cpp` | `kata_nn::onnx_builder` | skip | 已创建 API stub；实际 ONNX 序列化待后续 |

**关键决策**：是否使用 `tch-rs`？
- **方案 A（推荐）**：用 `tch-rs` + LibTorch CPU/CUDA 作为初期后端。优势：直接运行 ONNX/ TorchScript 模型，省去手写算子；劣势：引入大依赖。
- **方案 B（纯 Rust）**：用 `tract-onnx` CPU + 自定义 CUDA kernel。更“Rust 原生”，但模型兼容工作量大。

建议**阶段 3 采用方案 A**，待稳定后再引入方案 B 的纯 Rust CPU 后端。

### 阶段 4：search（已完成）
移植最大、最复杂的模块。

| C++ 文件 | Rust crate/模块 | 注意点 |
|----------|-----------------|--------|
| `search.h/cpp` | `kata_search::search` | MCTS 主循环 |
| `searchnode.h/cpp` | `kata_search::node` | 节点内存布局，用 arena allocator |
| `searchhelpers.cpp`, `searchexplorehelpers.cpp`, `searchnnhelpers.cpp`, `searchupdatehelpers.cpp`, `searchtimehelpers.cpp`, `searchmultithreadhelpers.cpp` | `kata_search::helpers` | 拆分子模块 |
| `searchpuct.cpp`, `searchresults.cpp`, `searchprint.cpp`, `searchmirror.cpp` | `kata_search::policy`, `kata_search::results`, `kata_search::print`, `kata_search::symmetry` | 与原算法逐行对齐 |
| `evalcache.h/cpp` | `kata_search::eval_cache` | LRU/锁-free cache |
| `mutexpool.h/cpp` | `kata_search::mutex_pool` | 锁池 |
| `localpattern.h/cpp` | `kata_search::local_pattern` | 局部模式 |
| `patternbonustable.h/cpp` | `kata_search::pattern_bonus` | 模式奖励表 |
| `distributiontable.h/cpp` | `kata_search::distribution` | 分布采样 |
| `subtreevaluebiastable.h/cpp` | `kata_search::subtree_bias` | 子树价值偏差 |
| `asyncbot.h/cpp` | `kata_search::async_bot` | 异步 GTP/分析接口 |
| `reportedsearchvalues.h/cpp` | `kata_search::reported_values` | 搜索结果上报 |
| `timecontrols.h/cpp` | `kata_search::time_control` | 读秒、包干 |
| `analysisdata.h/cpp` | `kata_search::analysis` | JSON 分析输出 |

**风险点**：
- 多线程 MCTS 的内存序与 C++ 版本必须等价。
- 建议先做单线程搜索移植，再扩展多线程。
- 使用 `crossbeam-epoch` 或 `parking_lot` 管理节点生命周期。

**验收**：
- 在相同模型、相同随机种子下，Rust 与 C++ 搜索树在多个 benchmark 位置上的胜率/选点分布无显著差异（统计检验）。
- 通过 `runsearchtests.sh` 等价的 Rust 测试套件。

### 阶段 5：program + command（已完成）
| C++ 文件 | Rust crate/模块 | 注意点 |
|----------|-----------------|--------|
| `setup.h/cpp` | `katago::setup` | 全局初始化 |
| `play.h/cpp`, `playutils.h/cpp`, `playsettings.h/cpp` | `katago::play` | 对弈引擎、 humanSL、规则设置 |
| `gtpconfig.h/cpp` | `katago::gtp_config` | GTP 默认配置 |
| `selfplaymanager.h/cpp` | `katago::selfplay` | selfplay 流水线 |
| `command/gtp.cpp` | `katago::cmd::gtp` | GTP 命令循环 |
| `command/analysis.cpp` | `katago::cmd::analysis` | 分析 JSON 输入输出 |
| `command/benchmark.cpp` | `katago::cmd::benchmark` | 性能基准 |
| `command/contribute.cpp` | `katago::cmd::contribute` | 分布式客户端 |
| `command/*.cpp` 其他 | `katago::cmd::*` | 逐步移植 |

### 阶段 6：GPU 后端与性能优化（约 6–10 周，可并行）
- CUDA/TensorRT：保留 C++ shim，通过 `cxx` 暴露；或评估 `cudarc`/`tensorrt-rs`。
- OpenCL：用 `ocl` crate 重写 tuning 与 kernel 加载。
- Metal：用 `metal-rs` 重写。
- 性能目标：在相同硬件上，Rust 版本与 C++ 版本 NN eval 吞吐差距 < 10%。

### 阶段 7：book + distributed（已完成）
- `book/` 定式书逻辑已移植到 `kata_book`。
- `distributed/` HTTP 客户端已用 `reqwest` + `tokio` 重写到 `kata_distributed`。

---

## 5. 目录结构建议

```text
KataGo-Lite/
├── KataGo/                       # 原始 C++ 代码库（保持不变）
│   ├── cpp/
│   └── python/
├── katago-rs/                    # Rust 移植工程
│   ├── Cargo.toml                # workspace root
│   ├── crates/
│   │   ├── kata_core/            # 通用工具
│   │   ├── kata_game/            # 围棋规则与棋盘
│   │   ├── kata_data/            # SGF / 训练数据 I/O
│   │   ├── kata_nn/              # 神经网络抽象与后端
│   │   ├── kata_search/          # MCTS 搜索
│   │   ├── kata_program/         # 程序逻辑（play/selfplay/GTP 设置）
│   │   ├── kata_book/            # 定式书
│   │   ├── kata_distributed/     # 分布式 selfplay 客户端
│   │   └── katago/               # CLI 入口
│   ├── cpp-shim/                 # 临时 C++ FFI shim
│   ├── models/                   # 测试用 tiny 模型（git-lfs 或脚本下载）
│   ├── tests/                    # 跨 crate 集成测试 / 回归测试
│   └── benches/                  # Criterion 性能基准
├── doc/                          # 本方案与补充文档
│   ├── rust-port-plan.md         # 本文件
│   ├── module-mapping.md         # C++ → Rust 文件级映射（随进度更新）
│   └── ffi-guide.md              # FFI 桥接规范
└── scripts/
    ├── compare-gtp.sh            # GTP 输出对比脚本
    └── compare-search.sh         # 搜索结果对比脚本
```

---

## 6. 当前重点工作（v2.0 新增）

### 6.1 近期目标
1. **TensorRT 后端骨架就绪**：`kata_nn::backends::trt` 模块已创建，需完成编译接入、feature gate、单元测试。
2. **NNEvaluator 异步批量推理**：实现 `spawn_server_threads` / `kill_server_threads` / `serve` / `wait_for_next_nn_eval_if_any` 等核心方法，使 NN 评估从“同步占位”变为“异步批处理”。
3. **模型文件解析器**：补全 `desc.rs` 中基于 `std::io::Read` 的 KataGo 二进制模型流解析，为真实后端提供权重。
4. **Eigen/CPU 后端**：在 dummy 后端之后，实现第一个可跑真实模型的纯 Rust/ndarray CPU 后端。

### 6.2 TensorRT 后端实施步骤
| 步骤 | 输出 | 验收 |
|------|------|------|
| 在 `kata_nn/src/lib.rs` 声明 `pub mod backends;` | 模块参与编译 | `cargo check` 通过 |
| 为 `trt` feature 添加单元测试 | `tests` 模块 | `cargo test -p kata_nn --features trt` 通过 |
| 验证无 `trt` feature 时编译通过 | default build | `cargo test --workspace` 通过 |
| 后续接入真实 TensorRT FFI | `cpp-shim/trtbackend.cpp` + Rust bindings | 可用真实模型做 smoke inference |

### 6.3 NNEvaluator 异步批量推理实施步骤
| 步骤 | 输出 | 验收 |
|------|------|------|
| 实现 `NNResultBuf` 的条件变量等待 | `Condvar` + `Mutex` | 单线程阻塞等待结果通过测试 |
| 实现 `serve` 主循环 | 后台线程消费 `query_queue` | 多个 `evaluate` 调用被聚合为 batch |
| 实现 `spawn_server_threads` / `kill_server_threads` | 线程生命周期管理 | 启动/停止无泄漏、无 panic |
| 接入真实后端 `get_output` | 调用 `Backend::get_output` | dummy 后端产生非均匀输出也可运行 |
| 实现对称平均与随机对称选择 | `average_multiple_symmetries` | 与 C++ 语义对齐 |

---

## 7. 关键技术决策

### 7.1 数值精度
- 棋盘、搜索用 `f32` 保持与 C++ 一致。
- 需要严格按位一致的随机/哈希：移植原算法，不依赖 Rust 默认 hasher。

### 7.2 内存管理
- 搜索节点数量巨大，使用 arena allocator：`bumpalo` 或自定义 lock-free pool。
- 多线程访问节点用 `AtomicPtr` + `parking_lot::RwLock` 组合，避免 `Arc` 带来的引用计数开销。

### 7.3 并发模型
- 搜索内部：工作窃取线程池（`rayon` 或 `crossbeam` deque）。
- NN server threads：每个 GPU/线程一个 `std::thread`，通过 `ThreadSafeQueue` 收集请求。
- I/O/GTP：单线程 + 非阻塞，或少量 `tokio` 任务；不推荐把 tokio runtime 带入搜索热路径。
- 分布式：独立 `tokio` runtime 处理 HTTP。

### 7.4 错误处理
- 库 crate：使用 `thiserror` 定义错误类型。
- 二进制 crate：使用 `anyhow` 兜底。
- 保持与原 C++ 错误信息格式一致，方便上游脚本解析。

### 7.5 unsafe 边界
- 允许在以下场景使用 `unsafe`：
  1. 与 C++ shim FFI 交互。
  2. 搜索节点 arena 的原始指针操作（需完整 Miri/loom 测试）。
  3. SIMD 优化（可用 `std::simd` 或 `portable_simd`）。
- 其他场景优先 Safe Rust。

---

## 8. 测试策略

### 8.1 三层测试
1. **单元测试**：每个 crate 内部 `#[cfg(test)]`，覆盖公共 API。
2. **集成测试**：`katago-rs/tests/`，调用原 C++ 可执行文件与 Rust CLI 对比输出。
3. **回归测试**：
   - 固定随机种子，跑 1000+ 个 SGF 棋谱，对比落子合法性、终局结果。
   - 固定模型与搜索配置，对比 `benchmark` 棋盘的 top  policy、value、ownership。

### 8.2 与原 C++ 的等价性验证
- 为每个被移植模块编写 **oracle test**：用 subprocess 调用原 C++ 对应测试/命令，记录输出；Rust 版本产生同样输出即通过。
- 对搜索等随机过程，使用相同伪随机序列与确定性种子。

### 8.3 性能基准
- 使用 `criterion.rs` 持续跟踪：
  - 棋盘落子/撤销
  - NN 输入编码
  - 单线程/多线程搜索 nodes/sec
  - GTP 响应延迟

---

## 9. 开发路线图（总估算 8–12 个月）

| 阶段 | 时间 | 里程碑 | 可运行产物 |
|------|------|--------|------------|
| 0 | 2 周 | 工程与 CI 就绪 | `cargo test` 通过空 workspace |
| 1 | 4–6 周 | core + game 完成 | Rust 棋盘 + SGF 解析可独立使用 |
| 2 | 3–4 周 | dataio 完成 | 训练数据读写 Rust 化 |
| 3 | 6–8 周 | NN 抽象 + 后端骨架 | TensorRT 骨架可编译；NNEvaluator 异步批处理可运行 |
| 4 | 8–12 周 | search 完成 | 单/多线程 MCTS 等价 |
| 5 | 4–6 周 | program + command | `katago-rs gtp`、`katago-rs analysis` |
| 6 | 6–10 周 | GPU 后端/优化 | 性能接近 C++ |
| 7 | 3–5 周 | book + distributed | 功能补齐 |

> 实际周期取决于人力与后端方案选择；阶段 6 可与阶段 4/5 并行推进。当前（2026-07-16）阶段 1/2/4/5/7 已完成，阶段 3 进行中。

---

## 10. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 搜索多线程行为不一致 | 高 | 先做单线程；引入 `loom`/`Miri` 测试；关键路径对照 C++ 源码 |
| NN 后端依赖复杂 | 高 | 先用 `tch-rs`/LibTorch；GPU 后端保留 C++ shim |
| 模型格式变更 | 中 | 单独维护模型加载层，密切关注 KataGo 版本更新 |
| 性能不达预期 | 中 | 设立 Criterion 基准；热点用 SIMD/unsafe/arena 优化 |
| 团队 Rust 经验不足 | 中 | 阶段 1 做培训；代码审查强制 clippy + fmt |
| 与原 C++ 生态脱节 | 低 | 保持配置/GTP/SGF/训练数据格式兼容 |

---

## 11. 工具链与 CI

- **Rust 版本**：`stable` + 按需 `nightly`（portable SIMD）。
- **格式化**：`rustfmt`。
- **静态检查**：`clippy -- -D warnings`。
- **测试**：`cargo test --workspace`。
- **覆盖率**：`cargo tarpaulin` 或 `llvm-cov`。
- **文档**：`cargo doc --no-deps`。
- **CI 示例**（`.github/workflows/rust.yml`）：

```yaml
name: Rust
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo fmt --check
      - run: cargo clippy --workspace --all-targets -- -D warnings
      - run: cargo test --workspace
```

---

## 12. 下一步行动清单

1. 在 `katago-rs/crates/kata_nn/src/lib.rs` 中声明 `pub mod backends;`。
2. 运行 `cargo fmt --all`、`cargo clippy --workspace --all-targets -- -D warnings`、`cargo test --workspace` 验证 TensorRT 骨架。
3. 更新 `doc/module-mapping.md`，将 `neuralnet/trtbackend.cpp` 状态从 `shim` 改为 `wip`。
4. 实现 `NNEvaluator::spawn_server_threads` / `kill_server_threads` / `serve` / `wait_for_next_nn_eval_if_any`。
5. 实现 `NNResultBuf` 的条件变量等待与异步批处理聚合逻辑。
6. 补全 `desc.rs` 二进制模型流解析器，为真实后端提供权重。
7. 实现 `kata_nn::backends::eigen` CPU 后端，替换 dummy 占位推理。

---

## 13. 附录

### 13.1 参考资料
- KataGo 原仓库：`https://github.com/lightvector/KataGo`
- Rust FFI：`https://cxx.rs/`
- `tch-rs`：`https://github.com/LaurentMazare/tch-rs`
- `tract`：`https://github.com/sonos/tract`

### 13.2 命名约定
- Crate 名：`kata_*` 为库，`katago` 为 CLI。
- 模块名：小写 + 下划线。
- 类型名：PascalCase；方法/函数：snake_case。
- 与原 C++ 类同名时加 `Rust` 前缀或保持原名并在文档中说明。

### 13.3 文件级映射模板
详细映射表见 `module-mapping.md`，应随移植进度持续更新。初期示例：

| C++ | Rust |
|-----|------|
| `cpp/game/board.h` | `katago-rs/crates/kata_game/src/board.rs` |
| `cpp/search/search.h` | `katago-rs/crates/kata_search/src/search.rs` |
| `cpp/neuralnet/nneval.h` | `katago-rs/crates/kata_nn/src/eval.rs` |
| `cpp/neuralnet/trtbackend.cpp` | `katago-rs/crates/kata_nn/src/backends/trt.rs` |
