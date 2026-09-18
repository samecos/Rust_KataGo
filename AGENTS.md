# Rust_KataGo

> 2026-09-16：本轮 Fork 对标性能优化已按用户要求结项，停止追加测试和实验。
> 当前交付与使用入口：`docs/RustGo性能优化结项记录.md`、`docs/RustGo使用指南.md`。
> 主规划和实验报告中的历史“active/下一步”不构成继续执行授权；后续优化需新的明确任务。下列开发与数值纪律继续适用于后续任务。

KataGo 围棋引擎的 Rust 移植(基座:KataGo-Lite/katago-rs,约 10 万行,含与 C++ 对拍的 oracle 测试)
+ 针对本机 RTX 5070 Ti(Blackwell SM120)的 CUDA 推理后端。

## 构建与运行

- 构建(默认特性,dummy 后端):`cargo build --workspace`
- TRT 后端:`cargo build -p katago --features trt`(需本机 CUDA + TensorRT 头文件)
- CUDA 后端(手写 kernel,nvcc 编译):`cargo build -p katago --features cuda`;
  冒烟 `cargo test -p kata_nn --test test_cuda --features cuda`
- CUTLASS(B2 dual-FFN 主机侧 CUTLASS DualGemm):build.rs 按
  `KATAGO_CUTLASS_ROOT` → `third_party/cutlass` → `D:/code/cutlass` 查找
  (本机已克隆 v3.9.2 到 D:/code/cutlass);找不到则跳过该 tactic
  (plan 里的 KATAGO_CUDA_DUALFFN=1 静默回退现有路径,不报错)
- 测试:`cargo test --workspace`
- GTP 冒烟:`--model /dev/null`(dummy 后端);真实推理:`--model D:/code/b11fix.onnx
  --override-config nnBackend=cudabackend`(CUDA)或 `nnBackend=trtbackend`(TRT,
  首次构建引擎较慢,plan 缓存在模型旁)
- genconfig(交互式生成 GTP 配置,对齐上游 benchmark.cpp 的 MainCmds::genconfig):
  问答规则/搜索限制/后端(多出 nnBackend 选择写入配置,Rust 端运行时选后端所需)/
  显存缓存/设备,然后 ternary 搜索调优 numSearchThreads 并测半 batch,产物可直接
  供 `gtp --config` 使用;不测"每 GPU 2 个 NN server 线程"(本仓库单消费者凑批设计,
  `NnEvaluator::set_num_threads` 为 no-op)。复用 benchmark.rs 的调优件
  (create_nneval/do_auto_tune_threads 等,pub(crate))
- 数值对拍:`cargo test -p kata_nn --test dump_nn_io_cuda --features cuda --release`
  生成转储 → `.venv/Scripts/python.exe scripts/compare_nn_output.py
  crates/kata_nn/target/nn_io_dump_cuda`(ORT FP32 黄金参考;gates:policy 5e-2、
  value 2.5e-2、misc 1e-2、ownership 5e-3、policy top-1 100%)
- 性能基准(release):`./target/release/katago-rs.exe benchmark --config
  configs/gtp_smoke.cfg --model D:/code/b11fix.onnx --override-config
  nnBackend=cudabackend --override-config numSearchThreads=1 -v 20 -n 1 -t 1,4`
  (不带 -t 时走 auto-tune;模型只加载一次但逐档搜索仍慢,
  务必用 -t 指定 1-2 个配置)
- 离线 autotune:`python scripts/autotune.py --threads both`(8 决策组 ABBA,
  产出 plans/best-tactic-plan.json + 可直接 `gtp --config` 使用的
  configs/gtp_autotuned.cfg——按 nnEvals/s 线程扫描选 numSearchThreads 并做
  GTP genmove 冒烟验证;`--model/--out-plan/--out-cfg/--threads-sweep` 可改目标,
  `--out-cfg ""` 回到只产 plan 的旧行为);接入选认证 plan:`--override-config
  nnBackend=cudabackend,cudaTacticPlan=D:/code/Rust_KataGo/plans/best-tactic-plan.json`
  (fail-closed:指纹/模型不匹配即报错);设备指纹:`katago-rs cuda-fingerprint --model <f>`
- per-layer 剖析:`KATAGO_CUDA_PROFILE=1` + GTP `kata-raw-nn all`(逐层/attention
  子段耗时,graph 模式下被跳过需配 `KATAGO_CUDA_NOGRAPH=1`);输入 dump:
  `KATAGO_CUDA_DUMP_INPUT=<dir>`;逐层 dump:`KATAGO_CUDA_DEBUG_LAYER=<i>`
- nnbench(固定物理 batch 吞吐,对齐 C++ benchmarknn):`./target/release/katago-rs.exe
  nnbench --model D:/code/b11fix.onnx --override-config nnBackend=cudabackend
  --mode eval --batch 1,2,4,8,12,16,24,32 --iterations 400`(`--mode direct`=
  CudaModel::apply 直连含拷贝,`--mode kernel`=纯前向;eval 的 --workers 默认
  2×当前 batch——serve target 是发射阈值非上限,worker 过多会把 avgBatch 抬高);
  cuBLASLt 候选池探针:`cargo test -p kata_nn --test probe_cublaslt_algos
  --features cuda --release -- --nocapture`
- tactic 组合验证(仿官方 runcudaopttests.sh,对齐 v1.18.1 审计结论):
  `scripts/validate_cuda_tactics.py`(24 case:路径标记断言 + plan
  fail-closed 负例;`--with-numeric` 加数值门——每组合 dump+ORT 对拍)。
  所有 tactic 路径决策点会打一次性 `[cuda-tactic]` 标记(dual_ffn/
  attention/gemm kind×engine/splitk/fusion/rms/padbatch/cublaslt_rank/
  graph),供脚本与人工确认开关真实生效(M2 DUALFFN 断裂先例的制度化)
- graph 回归测试:`cargo test -p kata_nn --test graph_launch_repro
  --features cuda --release`(2026-08-24 修复:无 pre-capture warm 时
  首次捕获的 graph exec 会被后续同流捕获作废,首批发射恒
  CUDA_ERROR_INVALID_VALUE——warmup/首批 eval 静默丢批;
  `REPRO_NO_WARM=1` 可复现)

## 后端选择

- Go Server Worker：`katago-rs nnworker --server 127.0.0.1:50051
  --worker-id rustgo-5070ti
  --model D:/Go/Server/models/kata1-tf3-b11c768-s11001M-d5973M.bin.gz
  --config configs/worker_cuda.cfg --capacity 32`；真实后端当前仅 CUDA。
  `crates/kata_worker` 实现纯 NN gRPC Worker，Server 持有搜索图；协议与
  `D:/Go/Server/proto/worker.proto` 同步。接入/模型身份边界与验证见
  `docs/Go-Server-Worker.md`。合成测试必须显式 `--allow-dummy` + `/dev/null`。
  原生 TF3 v17 直接解析并 lower 为 CUDA 层图，SHA 绑定原始压缩文件；
  `worker_cuda.cfg` 不引用旧 b11fix ONNX 的认证 plan。

- 配置键 `nnBackend`(每模型可用 `nnBackend{i}` 覆盖):`dummybackend`(默认)/
  `trtbackend` / `cudabackend` / `eigenbackend`(未实现,选择即报错)
- `cudabackend` 需要 katago 的 `cuda` feature(透传 `kata_nn/cuda` +
  `kata_program/cuda`);CUDA 后端仅支持 19 路、强制 NCHW 输入
- 命令行示例:`--override-config nnBackend=cudabackend`

## 目录

- `crates/kata_*`:引擎各层(core/game/data/search/nn/program/book/distributed);
  `crates/katago` 为 CLI(bin `katago-rs`)
- `crates/xuanping`:玄枰 GUI(egui/eframe 0.36,冷感东方风;视图:
  Play(分析覆层 `a`)/ Review `r` / Settings「器」`e` / 帮助「键」`?`,
  棋谱条 `k`、玄墨/雪宣双主题 `t`、←/→ 步进;
  运行 `cargo run -p xuanping`(无模型=演示模式);引擎模式
  `cargo run -p xuanping --features cuda -- --model D:/code/b11fix.onnx
  --backend cuda --threads 8`(同 workspace kata_* 直连,AsyncBot 分析回调
  0.25s 喂候选/领地/根胜率,用户执黑引擎执白,中国规则 komi 7.5);
  帧截图自验 `XUANPING_SHOT=<ppm>`(配 `XUANPING_VIEW/THEME/HELP/MODEL/
  BACKEND/THREADS` 截任意状态)、`XUANPING_DEBUG=1` 打布局指标;
  设计稿与生成脚本在 `docs/design/`、`scripts/design/xuanping_mockup.py`)
- `crates/kata_nn`:NN 抽象层与后端(`backends/`:dummy、trt、cuda/cuda_exec);
  `cuda-kernels/*.cu` 为手写 kernel(basic/gemm/gemm_v2/elementwise/attention/
  executor),由 build.rs 按 `configs/sm-targets.json`(sm_120 优先)用 nvcc
  编译成 PTX 嵌入
- `cpp-shim/`:TensorRT C++ FFI shim
- `configs/`:sm-targets.json、gtp_smoke.cfg、gtp_cuda.cfg、gtp_trt.cfg、
  gtp_benchmark.cfg
- `scripts/`:冒烟/对拍/对局脚本(compare_nn_output.py、play_match.py、autotune.py)
- `docs/`:需求汇总.md(含进度 §9)、cuda-fork-parity-plan.md(**对齐/超越
  fork 主规划,跨对话对齐入口,CUDA 性能动工前必读其 §1/§2/§8**)、
  cuda-optimization-plan.md(M4 路线图与 SM120 时效资料)、
  fork-sm120-kernel-notes.md(KataGomo_fork 认证 plan 提炼)、
  搜索算法调研与改进计划.md(kata_search 现状审计、官方 KataGo 演进与学术前沿、
  搜索树改进 P0-P2 路线;含 wideRootNoise 随机项/pondering/subtreeValueBias 写侧/
  evalCache 四处移植断裂清单)、使用与GUI接入.md(GTP GUI 接入与配置用法)
- `plans/`:autotune 产物(best-tactic-plan.json、autotune-history.json)

## 约定

- 仅支持 19 路标准棋盘
- 数值纪律:激活/归约一律 FP32 计算、half 存储边界与官方 KataGo 逐位对齐
  (正确性门前提,见 `docs/需求汇总.md` §7);SiLU=x/(1+exp(-x)) 精确形式;
  禁 fast_math
- 后端路线:手写 CUDA C++/PTX kernel 为主线(nvcc 编译 + cudarc 加载),
  TensorRT 兜底,plan JSON fail-closed
- tactic 开关(KATAGO_CUDA_* 环境变量)一律经 `tactic_plan::tactic_var()`
  读取(优先级 plan > env > 默认;直接 env::var 会绕过认证 plan);
  默认值变更必须 autotune ABBA + 对拍双证据。当前 plan(r1)仅启用
  `KATAGO_CUDA_DUALFFN=1`(CUTLASS DualGemm + SwiGLU epilogue;M2 落地,
  2026-08-17 修复提交断裂后实测 +26.5% eval B16 / +28.0% 搜索 t=32);
  `KATAGO_CUDA_CUBLASLT_RANK=time` 曾于 M1 采纳(+2.55%),2026-08-17
  复审下架(B2 接管 ffn_up 后边际归零,搜索口径净负 -4%)
- cuda-host 源码编译失败 = 构建失败(fail-loud,CUTLASS 缺失仍静默跳过;
  逃生门 KATAGO_ALLOW_BROKEN_CUDA_HOST=1)——M2 曾提交编译不过的
  dual_ffn_cutlass.cu 被 build.rs 静默跳过,DUALFFN 空转数小时无人察觉
- 依赖真实模型的测试用 `KATAGO_TEST_MODEL_DIR` 环境变量定位模型,缺失时自动跳过
- 新后端实现需实现 `kata_nn::backend::Backend` trait 并接入 `kata_program::setup`
  的 backend 选择
- 性能变更必须 ABBA 实测(基准命令见上),慢于基线即回退;数值变更必须过
  整图对拍(compare_nn_output.py RESULT: PASS);启用休眠代码路径(如
  长期未跑的 tactic 组合)前同样先过对拍(FUSION=none 的 act384f16
  断裂 bug 即先例)
