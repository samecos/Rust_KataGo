# Rust_KataGo

KataGo 围棋引擎的 Rust 移植(基座:KataGo-Lite/katago-rs,约 10 万行,含与 C++ 对拍的 oracle 测试)
+ 针对本机 RTX 5070 Ti(Blackwell SM120)的 CUDA 推理后端。

## 构建与运行

- 构建(默认特性,dummy 后端):`cargo build --workspace`
- TRT 后端:`cargo build -p katago --features trt`(需本机 CUDA + TensorRT 头文件)
- CUDA 后端(手写 kernel,nvcc 编译):`cargo build -p katago --features cuda`;
  冒烟 `cargo test -p kata_nn --test test_cuda --features cuda`
- 测试:`cargo test --workspace`
- GTP 冒烟:`--model /dev/null`(dummy 后端);真实推理:`--model D:/code/b11fix.onnx
  --override-config nnBackend=cudabackend`(CUDA)或 `nnBackend=trtbackend`(TRT,
  首次构建引擎较慢,plan 缓存在模型旁)
- 数值对拍:`cargo test -p kata_nn --test dump_nn_io_cuda --features cuda --release`
  生成转储 → `.venv/Scripts/python.exe scripts/compare_nn_output.py
  crates/kata_nn/target/nn_io_dump_cuda`(ORT FP32 黄金参考;gates:policy 5e-2、
  value 2.5e-2、misc 1e-2、ownership 5e-3、policy top-1 100%)
- 性能基准(release):`./target/release/katago-rs.exe benchmark --config
  configs/gtp_smoke.cfg --model D:/code/b11fix.onnx --override-config
  nnBackend=cudabackend --override-config numSearchThreads=1 -v 20 -n 1 -t 1,4`
  (注意:不带 -t 时走 auto-tune,每个线程配置都重新加载模型 ~25s,很慢;
  务必用 -t 指定 1-2 个配置)
- 离线 autotune:`python scripts/autotune.py --threads both`(8 决策组 ABBA,
  产出 plans/best-tactic-plan.json);接入选认证 plan:`--override-config
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

## 后端选择

- 配置键 `nnBackend`(每模型可用 `nnBackend{i}` 覆盖):`dummybackend`(默认)/
  `trtbackend` / `cudabackend` / `eigenbackend`(未实现,选择即报错)
- `cudabackend` 需要 katago 的 `cuda` feature(透传 `kata_nn/cuda` +
  `kata_program/cuda`);CUDA 后端仅支持 19 路、强制 NCHW 输入
- 命令行示例:`--override-config nnBackend=cudabackend`

## 目录

- `crates/kata_*`:引擎各层(core/game/data/search/nn/program/book/distributed);
  `crates/katago` 为 CLI(bin `katago-rs`)
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
  使用与GUI接入.md(GTP GUI 接入与配置用法)
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
  默认值变更必须 autotune ABBA + 对拍双证据。当前 plan 已启用
  `KATAGO_CUDA_CUBLASLT_RANK=time`(cuBLASLt top-8 计时重排,M1:+2.55%)
- 依赖真实模型的测试用 `KATAGO_TEST_MODEL_DIR` 环境变量定位模型,缺失时自动跳过
- 新后端实现需实现 `kata_nn::backend::Backend` trait 并接入 `kata_program::setup`
  的 backend 选择
- 性能变更必须 ABBA 实测(基准命令见上),慢于基线即回退;数值变更必须过
  整图对拍(compare_nn_output.py RESULT: PASS);启用休眠代码路径(如
  长期未跑的 tactic 组合)前同样先过对拍(FUSION=none 的 act384f16
  断裂 bug 即先例)
