# KataGo `cudarocmopt` 优化验证计划

> 状态:V0/V1/V2 已验证,V3/V3R 已拒绝,V4 已通过 Windows/WSL 并晋级 schema 2,V5 工具和定向 warmup 已验收  
> 制定日期:2026-08-19  
> 本项目基线:`Rust_KataGo` `9517be2` (`main`)  
> 参考快照:`D:/ExperimentalCode/KataGo` `cudarocmopt` `08939455`  
> 参考主优化提交:`ecfbeb46`(本地分支相对 `origin/master` `ccdec959` 共 5 个提交)

## 1. 结论与范围

参考分支的主要 CUDA 思路中,本项目已经具备非阻塞 per-handle stream、模型权重共享、
1x1 GEMM、单次 packed QKV GEMM、tensor-core FlashAttention、RoPE load 融合、
CUTLASS DualGemm + SwiGLU、以及 output/FFN down 的 `beta=1` 残差累加。
这些项目不再重复移植。

本轮只验证以下增量,按依赖顺序执行:

| 阶段 | 候选 | 类型 | 预期产出 |
|---|---|---|---|
| V0 | 冻结基线与环境 | 测量基础设施 | 可复现的 Windows/WSL 基线 |
| V1 | tactic 路径命中矩阵 | 安全基础设施 | expected/forbidden 路径断言和负向启动测试 |
| V2 | build/capability 指纹 + DualFFN 实核探针 | 安全基础设施 | 认证 plan 不再接受“已请求但未编译/未运行”的 tactic |
| V3 | DualGemm swizzle `1` 对 `2` | 性能候选 | 低成本 ABBA 裁决 |
| V3R | 延迟 residual 并融合进下一层 RMSNorm | 性能候选 | 对照现有 GEMM `beta=1` 路径 |
| V4 | attention `64Q x 64K,4 warps` | 性能候选 | 与现有 `128Q x 64K,8 warps` 的独立对照,Windows 候选已验证 |
| V5 | 定向预热 + 同步多 handle benchmark | 工程候选 | 冷启动数据及双流 kernel 扩展性测量 |

约束:

- 仅验证 19x19、NCHW、当前 `b11fix.onnx` 生产形状。
- 激活和归约保持 FP32 计算、half 存储边界;SiLU 必须是
  `x / (1 + exp(-x))`;禁用 `fast_math`。
- 不复制参考 attention 的 `ex2.approx`。V4 继续使用精确 `expf`。
- 所有新 tactic 必须通过 `tactic_plan::tactic_var()` 或
  `tactic_plan::tactic_enabled()` 读取。
- 性能候选不默认启用,不直接写入认证 plan。只有整图对拍和 ABBA 同时通过后才晋级。

## 2. 对照结果

| 参考分支优化 | 本项目状态 | 本轮动作 |
|---|---|---|
| 每个 compute handle 独立非阻塞 stream | 已有 | 不做 |
| 多 handle 共享 device weights | `Arc<CudaModel>` 已天然共享 | 不做 |
| 1x1 projection 改 GEMM | 已有 GEMM 路径 | 不做 |
| combined QKV projection | 已是单次 packed QKV GEMM | 不做 |
| FlashAttention + tensor-core QK/PV | 已有 FA2,且 RoPE 已融合到 load | 只验证不同 tile/register 组织(V4) |
| CUTLASS DualGemm + SwiGLU | 已落地并经 r1 plan 启用 | 只验证 swizzle 差异(V3)和能力探针(V2) |
| output/FFN down `beta=1` residual | 已有 `hgemm_residual` | 作为 V3R 基线 |
| deferred residual + next pre-norm fusion | 未实现 | 独立验证 V3R,不能与 `beta=1` 视为同一优化 |
| 预热 transformer lazy plans | Rust CUDA graph 按 batch 惰性创建,现已接入后端声明式 B1/max-batch warmup | 只预热生产尺寸,不照搬 1..16 |
| `benchmarknn` 多 handle 同步测量和 JSON | `nnbench` 已有 eval/direct/kernel,现已补齐同步多 handle direct/kernel 口径 | Windows 三轮验收通过;WSL 路径可用但重复性门槛未通过 |
| 数值 + expected/forbidden 路径测试 | 只有分散的数值/单测,缺完整路径矩阵 | 优先补齐(V1) |
| 启动时真实执行 fused FFN 探针 | 当前只检查 create 返回非空 | 补生产形状实核探针(V2) |
| vendored CUTLASS 4.2.2 | 当前外部解析 CUTLASS,文档环境为 3.9.2 | 不直接升级;先纳入 build 指纹(V2) |
| mask/GQA/不同棋盘尺寸覆盖 | 与本项目固定 19x19 约束不相符 | 明确不做 |

V3 的具体差异很小但值得先测:双方 block/warp/instruction tile 和 stage 相同,
参考为 `GemmIdentityThreadblockSwizzle<1>`,本项目为 `<2>`。

V4 不是已证伪 FA4 的重跑。参考方案是 `64Q x 64K`、4 warps、Q/P 主要驻留寄存器;
现行方案是 `128Q x 64K`、8 warps,并有 shared-memory P 中间区。它们是不同的
block/register/smem 权衡。

## 3. 不可变验证规则

### 3.1 基线

- q64 候选的对照基线固定为 Windows
  `plans/best-tactic-plan-sm120-q128-windows.json`（DualFFN=1,q128）或 WSL
  对应独立 q128 build；当前生产 incumbent 是
  `plans/best-tactic-plan.json`（schema 2,q64 + DualFFN）。
- 每个候选使用独立 `CARGO_TARGET_DIR`,必须从当前源码重建;禁止复用提交前旧二进制。
- 同一组 A/B 必须使用同一模型、配置、驱动、电源状态、CUTLASS 根和 CUDA 工具链。
- Windows 与 WSL 分别配对,不得把两者绝对值直接组成 A/B。
- Windows 运行时用 `nvidia-smi pmon` 或等价采样确认无外来 compute 负载。
- WSL 是主要性能裁决环境;Windows 用于确认收益方向和排除 WDDM 特有回退。

### 3.2 正确性门

所有性能候选先执行整图对拍,再允许测速:

```powershell
$env:KATAGO_ONNX_MODEL = 'D:\code\b11fix.onnx'
$env:KATAGO_DUMP_DIR = 'D:\code\Rust_KataGo\target\cudarocmopt-validation\<candidate>\dump'
cargo test -p kata_nn --test dump_nn_io_cuda --features cuda --release `
  dump_nn_io_cuda -- --nocapture --test-threads=1
.venv\Scripts\python.exe scripts\compare_nn_output.py $env:KATAGO_DUMP_DIR
```

必须出现 `RESULT: PASS`,且全部满足:

- policy max error `<= 5e-2`;
- value max error `<= 2.5e-2`;
- misc max error `<= 1e-2`;
- ownership max error `<= 5e-3`;
- policy top-1 `16/16`。

启用休眠路径、改变 CUTLASS 版本、改变编译 flags 或改变 kernel resource 布局时均重新对拍。

### 3.3 路径激活门

一次测试同时检查:

1. 预期路径日志或结构化 tactic report 必须出现且只出现一次/handle;
2. 被禁用路径的 marker 必须不存在;
3. 实际 effective tactic 必须与 requested tactic 一致;
4. plan 请求了未编译或探针失败的能力时,进程必须在首个推理前失败;
5. 无 plan 的诊断环境变量仍允许回退,但必须打印明确 warning,不得静默。

性能日志不能只证明“读取了开关”,必须证明目标 kernel 实际至少 launch 一次。

### 3.4 性能裁决

- 顺序使用 ABBA(`A-B-B-A`)；Windows 噪声超过 1% 时追加 BAAB。
- arm 内取几何均值,候选主指标至少快 `1.0%` 才算收益。
- 任一关键次指标慢 `>=1.0%` 直接拒绝;出现 `0~-1.0%` 可疑回退时重跑
  BAAB,若负向可复现同样拒绝。
- V3/V4 必测 `nnbench eval` B1/B4/B8/B16、对应的 `direct`/`kernel`,以及
  search t=8/t=16。B16 eval 与 search 几何均值均不得回退,且至少一个生产
  主指标达到 `+1.0%`。
- `kernel` 在 WDDM 深队列下若出现 direct 比 kernel 更快等异常,只作诊断,
  不单独决定采纳。
- 慢于基线即删除候选实现和 tactic 键,只把证伪数据写入结果表。

## 4. V0:冻结环境与基线

### 4.1 记录项

每次验证生成一个 run manifest(JSON),至少包含:

- Rust 仓库 commit 和 dirty diff hash;
- 参考仓库 commit `08939455`;
- 模型路径和 SHA-256;
- `cuda-fingerprint` 完整输出;
- GPU driver、CUDA runtime、`nvcc --version`;
- CUTLASS 根、`version.h` 版本和目录内容/commit 标识;
- `configs/sm-targets.json` hash;
- Windows/WSL、CPU、内存、GPU power limit 和当前 graphics/SM clock;
- 构建命令、`CARGO_TARGET_DIR`、tactic plan SHA-256;
- 每个 arm 的开始时间、原始 stdout/stderr 和退出码。

原始数据放在 `target/cudarocmopt-validation/<run-id>/`;结论摘要写回本文 §11。

### 4.2 基线命令

```powershell
$env:KATAGO_CUTLASS_ROOT = 'D:\code\cutlass'
$env:CARGO_TARGET_DIR = 'D:\code\Rust_KataGo\target\cudarocmopt-baseline'
cargo build -p katago --features cuda --release

& "$env:CARGO_TARGET_DIR\release\katago-rs.exe" cuda-fingerprint `
  --model D:\code\b11fix.onnx

& "$env:CARGO_TARGET_DIR\release\katago-rs.exe" nnbench `
  --model D:\code\b11fix.onnx `
  --override-config nnBackend=cudabackend,cudaTacticPlan=D:/code/Rust_KataGo/plans/best-tactic-plan-sm120-q128-windows.json `
  --mode eval --batch 1,4,8,16 --iterations 400
```

search 基线保持显式 `-t`,避免 auto-tune 重复加载模型:

```powershell
& "$env:CARGO_TARGET_DIR\release\katago-rs.exe" benchmark `
  --config configs/gtp_smoke.cfg --model D:\code\b11fix.onnx `
  --override-config nnBackend=cudabackend,cudaTacticPlan=D:/code/Rust_KataGo/plans/best-tactic-plan-sm120-q128-windows.json `
  --override-config numSearchThreads=1 -v 20 -n 1 -t 8,16
```

V0 完成条件:Windows 和 WSL 各有至少两轮基线;同一环境 B16 eval arm 间偏差
应小于 1%。超过 1% 时先处理环境噪声,不进入 V3/V4。

## 5. V1:tactic 路径命中矩阵

新增跨平台 `scripts/validate_cuda_tactics.py`,借鉴参考分支
`cpp/runcudaopttests.sh`,但复用本项目已有 ORT 整图黄金参考。

建议新增一次性结构化输出,例如:

```text
[cuda-tactic] name=dual_ffn requested=1 compiled=1 probe=pass effective=1
[cuda-tactic] name=attention_tile requested=q64 compiled=1 effective=q64 launches=1
```

最小矩阵:

| case | 配置 | 必须出现 | 必须不存在/结果 |
|---|---|---|---|
| default-r1 | r1 plan | `dual_ffn effective=1`、现行 FA2 | unfused FFN marker |
| dual-off | `DUALFFN=0` | unfused FFN | fused FFN launch |
| dual-on | `DUALFFN=1` | fused FFN launch | silent fallback |
| graph-off | `NOGRAPH=1` | direct submit | graph launch |
| graph-on | `NOGRAPH=0` | graph capture/launch | direct-only marker |
| explicit-zero | 所有布尔 tactic=`0` | baseline paths | 任一相应实验路径 |
| bad-value | 非法枚举值 | 启动失败和具体 key | 成功启动 |
| unknown-key | plan 中加入未知 key | 启动失败 | 忽略未知 key |
| plan-mismatch | 改 model/GPU/build 指纹 | 启动失败 | 继续推理 |
| missing-capability | plan 请求未编译 DualFFN | 启动失败 | unfused 静默回退 |

脚本每个正向 case 执行:

1. 单局面真实 CUDA 前向;
2. expected/forbidden marker 断言;
3. 有数值影响的组合执行 16 局面 dump + ORT 比较;
4. 将命令、退出码和日志存入 run 目录。

V1 完成条件:Windows/WSL 全矩阵通过;显式 `0` 和 plan 优先级均有单测;
该阶段不得使默认 B16 eval 回退 1%。

## 6. V2:build/capability 指纹和 DualFFN 实核探针

### 6.1 plan schema 2

为认证 plan 增加 `backend_build`:

```json
{
  "schema": 2,
  "backend_build": {
    "kernel_build_id": "sha256:...",
    "cuda_compiler": "13.2...",
    "cutlass_version": "3.9.2",
    "compiled_sm": ["120"],
    "capabilities": {
      "dual_ffn": true,
      "attention_q64": true
    }
  }
}
```

`kernel_build_id` 只哈希影响设备代码/FFI 的输入,至少包括:

- `cuda-kernels/*.cu`、`cuda-host/*.cu`、相关 headers;
- `build.rs` 中 CUDA 编译参数;
- `configs/sm-targets.json`;
- CUTLASS version/commit 或稳定目录 hash;
- nvcc major/minor 和目标 SM。

不要直接用整个 Git HEAD,避免纯文档提交使 plan 无效。schema 2 任一字段不匹配均
fail-closed。现有 schema 1 plan 只用于迁移期;V2 完成后重新认证生成 r2,不把 schema 1
继续称为认证 plan。

### 6.2 DualFFN 实核探针

参考上游做法,在启用 tactic 前真实运行生产 `DualGemm` 实例:

- 使用同一模板、同一 dynamic shared-memory 要求和同一 launch 路径;
- 本项目 kernel 固定 N=1152/K=384,探针用 M=16 的生产形状;
- 输入和权重清零,输出预填非零 sentinel;
- launch + stream synchronize + D2H 后要求输出全部精确为 0;
- 检查 `can_implement`、workspace size、initialize、run、CUDA launch/sync error;
- 每个 device/build 只运行一次并缓存结果。

探针只证明 kernel 可执行,整图对拍仍负责数值正确性。plan 请求 DualFFN 且 probe
失败时必须启动失败;未请求时可以禁用能力并打印 warning。

V2 负向测试必须覆盖:伪造 build id、伪造 CUTLASS version、capability=false 但
plan 请求 `DUALFFN=1`、以及探针强制失败。四者都必须在首个真实请求前失败。

V2 完成条件:全部负向测试通过;探针成功路径过整图对拍;启动时间和 B16 eval
无 `>=1%` 回退。

## 7. V3:DualGemm swizzle 1 对 2

### 7.1 实现隔离

- 同一 TU 同时实例化 `<1>` 和 `<2>`,避免用两次构建混入工具链差异。
- 新增 `KATAGO_CUDA_DUALFFN_SWIZZLE=1|2`,通过 `tactic_var()` 读取并加入
  plan allowlist/schema 2 capability。
- 默认值保持现状 `2`;只有认证 plan 才能改变生产默认。
- 路径 report 必须输出选择值和实际 launch 计数。

### 7.2 测量

先以 r1 其余 tactic 固定不变,执行:

- `nnbench --mode direct` B1/B4/B8/B16,隔离 host queue;
- `nnbench --mode kernel` B1/B4/B8/B16,诊断纯前向;
- `nnbench --mode eval` B1/B4/B8/B16,生产全链路;
- search `-t 8,16`;
- `KATAGO_CUDA_PROFILE=1` + `KATAGO_CUDA_NOGRAPH=1`,汇总 33 个 FFN
  layer 的 dual-up 总时间。

direct/kernel 当前不安装 tactic plan,运行时必须用环境变量完整表达 r1:

```powershell
$env:KATAGO_CUDA_DUALFFN = '1'
$env:KATAGO_CUDA_DUALFFN_SWIZZLE = '<1-or-2>'
```

eval/search 可继续安装 r1 plan,由环境变量补充 r1 未包含的新 swizzle key。

V3 采纳条件:

- 两个 swizzle 都通过路径矩阵,候选 `1` 通过整图对拍;
- WSL B16 eval 或 search t8/t16 几何均值至少 `+1.0%`;
- Windows 收益方向一致;
- B1/B4/B8/B16 及另一个生产主指标均无可复现负收益;
- profile 中 DualFFN 总时间改善方向与整图结果一致。

否则删除 swizzle 1 实例和 tactic key,保留证伪记录。

### 7.3 V3R:延迟 residual + RMSNorm

参考分支会让 attention output / FFN down 的 GEMM 只写 delta,把 residual add
延迟到下一 block 的 pre-norm kernel。Rust 当前没有独立 add kernel:现行
`hgemm_residual` 已用 `beta=1` 直接更新 residual buffer,随后单输入 RMSNorm。
因此候选并不天然少一次 kernel,需要比较两种内存与 epilogue 拓扑:

- A:`beta=1 GEMM` + 单输入 RMSNorm(现状);
- B:`beta=0 GEMM` 写 delta + 双输入 add/RMSNorm;
- block stack 尾部若仍有 pending delta,必须在 trunk final 前落回 trunk;
- 继续保持 FP32 GEMM accumulation、FP32 add/reduction 和原有 half 边界。

只有 profile 中 GEMM+RMSNorm 合计时间和 eval/search ABBA 同时达到 §3.4 门槛
才采纳;否则删除 pending-delta 状态和双输入 norm kernel。

## 8. V4:attention 64Q 候选

### 8.1 实现边界

新增独立 kernel symbol,不原地改写现行 `attention_fa2`:

- 候选:`64Q x 64K`,4 warps,优先将 Q 和 softmax/P 片段保存在寄存器;
- 基线:`128Q x 64K`,8 warps,现行 shared-memory P 设计;
- QK/PV 继续 tensor-core FP32 accumulate;
- online softmax、归一化和最终 half round 边界与现行路径一致;
- 使用精确 `expf`,不得采用参考分支 `ex2.approx.ftz.f32`;
- 保持现行 packed QKV、RoPE load 融合和输出布局;
- 不引入 mask/GQA/非 19x19 泛化代码。

新增 `KATAGO_CUDA_ATTN_TILE=q64|q128`,默认 `q128`;经 plan allowlist 读取。
路径 report 必须区分 q64/q128,不能只写“FA2 enabled”。

### 8.2 门槛与诊断

在整图对拍前先做 kernel 级确定性对照:

- batch 1/2/4/8/16;
- 至少 3 组固定随机 Q/K/V;
- q64 对 q128 输出 max/mean error 和非有限值检查;
- 用 `ptxas -v`/Nsight Compute 记录 registers/thread、static/dynamic smem、
  occupancy、tensor pipe utilization 和 DRAM bytes。

随后执行与 V3 相同的 direct/kernel/eval/search ABBA 矩阵。额外要求
`KATAGO_CUDA_PROFILE=1` 中 attention core 总时间至少改善 3%,否则即使整图
出现噪声级上涨也不采纳。

V4 采纳条件:

- kernel 对照和 16 局面 ORT 整图对拍全部通过;
- WSL attention core 至少 `+3.0%`;
- WSL B16 eval 或 search 几何均值至少 `+1.0%`;
- Windows 收益方向一致;
- 所有 batch 和另一个生产主指标无可复现回退。

否则删除 q64 kernel、dispatch 和 tactic key,保留证伪记录。不得为了通过数值门
放宽现有误差阈值。

### 8.3 2026-08-19 Windows 结果

q64 已完成真实模型整图对拍和路径矩阵。16 个局面的 ORT 对拍为 `RESULT: PASS`,
policy top-1 为 `16/16`;证据目录为
`target/cudarocmopt-validation/v4-q64-dump`。路径矩阵为 `14/14 PASS`,包含
q64/q128 实际 launch marker、graph/direct、FA2/v3、DualFFN on/off、schema/build/model
mismatch 和强制 probe failure。

最终生产组合 `q64 + KATAGO_CUDA_DUALFFN=1` 也已在 16 局面 ORT 对拍中
`RESULT: PASS`,policy top-1 `16/16`;实际日志同时确认 q64 和 fused DualFFN
launch,证据目录为 `target/cudarocmopt-validation/v4-q64-dualffn-dump`。

在固定 workers、400 iterations 的 Windows eval ABBA 中,初测 q64 相对 q128 的几何均值
收益为 B1 `+2.54%`、B4 `+5.24%`、B8 `+2.79%`、B16 `+2.83%`。随后以同一
release 二进制、warmup=40、5 轮 direct 重复测量 B1/B16: q128 为
`290.24/993.83 eval/s`,q64 为 `300.82/1025.84 eval/s`,收益分别为
`+3.64%/+3.22%`;B16 q64 CV `0.18%`,B1 受 WDDM 噪声影响 CV `4.02%`。
两次 direct profile 的 attention 累计约为 q128 `9.879 ms`、q64 `9.133 ms`,约
`+7.5%`。q64 不加入旧 schema 1 的历史决策组;当前 `scripts/autotune.py` 已改为以
q64+DualFFN 认证 incumbent 起步,并显式测量 q128/q64 回退候选,输出 schema 2 plan。

最终源码对应的 release 二进制又完成了生产 `nnbench eval` B16 ABBA
（workers=32,warmup=20,iterations=200,graph 开启）: q128 `1043.7/1038.4`
eval/s,q64 `1072.8/1070.6` eval/s,几何均值收益约 `+2.95%`;四轮日志见
`target/cudarocmopt-validation/v4-current-source-eval-abba.txt`。

同一 release 二进制的生产搜索 ABBA（`-v 100 -n 3 -t 8,16`,DualFFN=1）也通过：
q128/q64 的 visits/s 几何均值为 t8 `688.85/711.86`（`+3.34%`）、t16
`751.85/772.69`（`+2.77%`）；nnEvals/s 为 t8 `723.54/753.77`（`+4.18%`）、
t16 `845.10/867.40`（`+2.64%`）。原始日志见
`target/cudarocmopt-validation/v4-current-source-search-abba-v100n3.txt`。

### 8.4 2026-08-19 WSL 认证与晋级

WSL Ubuntu 24.04 使用 CUDA/nvcc `13.3.73` 和同一 RTX 5070 Ti 完成了源码 release
重建。q128 与 q64 + DualFFN 均通过 16 局面 ORT 整图对拍,policy top-1 均为
`16/16`;WSL tactic/fail-closed 矩阵 `14/14 PASS`。

WSL eval ABBA 几何均值(q128/q64)及 q64 收益:

| batch | q128 | q64 | 收益 |
|---:|---:|---:|---:|
| 1 | 388.95 | 400.35 | +2.93% |
| 4 | 725.05 | 756.60 | +4.35% |
| 8 | 954.59 | 989.80 | +3.69% |
| 16 | 1035.85 | 1071.50 | +3.44% |

长样本搜索 ABBA(`-v 400 -n 10 -t 8,16`)中,q64 对 q128 的 visits/s 在 t8/t16
分别约 `+3.0%/+2.4%`,nnEvals/s 分别约 `+3.3%/+3.1%`。profile 去掉首轮加载
异常后,6 次前向的 attention core 累计 q128 `23.783 ms`、q64 `20.278 ms`,
q64 快约 `14.7%`,超过 `+3%` 门槛。

因此 q64 与 DualFFN 一起晋级为 Windows 生产 schema 2 plan。Windows 与 WSL
设备代码 build id 不同,分别使用独立 plan;不能跨平台复用同一个认证 JSON。

## 9. V5:定向预热和同步多 handle benchmark

### 9.1 同步多 handle direct benchmark

先补测量工具,再讨论预热:

- `nnbench` direct 模式支持 `--handles 1,2`;
- 每个 handle 独立 stream/workspace,共享 `Arc<CudaModel>`;
- 使用 barrier 同时开始,每 handle 相同固定物理 batch 和迭代数;
- 结束时统一 synchronize,报告 per-handle 和总 wall throughput;
- `--json` 输出 revision、build id、plan/tactic report、warmups、batch、handles、
  每轮数据和总吞吐;
- 输出必须能被验证脚本直接解析,不得从人类日志正则猜数值。
- `scripts/validate_nnbench_handles.py` 固定同一 release 二进制执行三轮，保存每次
  stdout/stderr，并检查单 handle wall/per-handle 偏差与双 handle CV。

工具验收门槛仍保持:handles=1 与既有 direct 在 1% 内一致;handles=2 重复三轮方差
小于 1%;加入工具本身不改变默认 eval 路径和 B16 吞吐。当前已完成最小 Windows
冒烟,尚未把它当作正式性能裁决。

2026-08-19 Windows 冒烟（RTX 5070 Ti / SM120,CUDA 13.3.73,CUTLASS 3.9.2,
`b11fix.onnx`,B1,warmup=2,iterations=3,`KATAGO_CUDA_NOGRAPH=1`）:

| 模式 | handles | wall time | wall throughput | 结果 |
|---|---:|---:|---:|---|
| direct | `1` | 25.6 ms / 3 evals | 117.0 eval/s | JSON 可由 `ConvertFrom-Json` 解析 |
| direct | `1,2` | 44.4 ms / 6 evals | 135.1 eval/s | 两条独立 stream/workspace 同步启动 |
| kernel | `1,2` | 已运行 | 已写入 JSON | 纯前向路径可用 |

同一冒烟还以 `KATAGO_CUDA_ATTN_TILE=q64` + `KATAGO_CUDA_DUALFFN=1` 跑通了
双 handle 路径（wall 138.1 eval/s,日志确认 q64 与 fused DualFFN launch）。该结果
作为并发安全/路径 smoke 留档,正式晋级仍以 §8.4 的整图对拍、ABBA 和 schema 2 plan 为准。

该冒烟的 wall 计时在 setup barrier 之后开始,并在 done barrier 处结束,不包含
workspace/buffer 析构。它只证明工具口径和并发路径可运行,不证明双流在搜索或生产
`eval` 调度中一定有收益。

### 9.1.1 2026-08-19 release 工具验收

命令（q128 + DualFFN,三轮）:

```powershell
python scripts/validate_nnbench_handles.py `
  --binary target/release/katago-rs.exe --model D:/code/b11fix.onnx `
  --batches 1,16 --modes direct,kernel --handle-sets "1;1,2" `
  --repeats 3 --warmup 20 --iterations 100 `
  --tactic KATAGO_CUDA_DUALFFN=1 --tactic KATAGO_CUDA_ATTN_TILE=q128 `
  --tactic KATAGO_CUDA_NOGRAPH=1
```

结果目录:`target/cudarocmopt-validation/v5-handles-current`。

| 模式 | handles | B1 几何均值 | B16 几何均值 | 重复性 |
|---|---:|---:|---:|---|
| direct | 1 | 289.6 eval/s | 959.7 eval/s | B1 CV 1.62%,B16 CV 0.58% |
| direct | 1,2 | 281.7 eval/s | 1065.5 eval/s | CV 0.21%/0.20% |
| kernel | 1 | 310.2 eval/s | 986.5 eval/s | CV 0.67%/0.21% |
| kernel | 1,2 | 302.7 eval/s | 1108.3 eval/s | CV 0.37%/0.13% |

工具门槛通过:单 handle wall/per-handle 偏差 `<=1%`,双 handle CV `<=1%`。
这证明多流测量稳定；B16 direct 双 handle 的约 `+11.0%` 只作为扩展性信号，
不直接等价为 GTP/search 收益。

### 9.1.2 2026-08-19 WSL release 复测

同一 release 二进制在 WSL Ubuntu-24.04、RTX 5070 Ti、q64 + DualFFN、
`KATAGO_CUDA_NOGRAPH=1` 下完成了相同的 B1/B16、direct/kernel、单/双 handle
三轮测量。路径和 JSON 输出均正常，但重复性门槛未通过：单 handle B16 的 CV
为 `0.24%`--`0.39%`，双 handle direct B16 为 `0.69%`；双 handle B1 的 CV
为 `11.28%`/`17.33%`，kernel 双 handle B16 为 `5.37%`。因此 WSL 结果只证明
并发路径可运行，不把多 handle 扩展性晋级为性能结论，也不改变 Windows 工具
验收结论。原始结果保存在
`target/cudarocmopt-validation/v5-handles-wsl-final`。

### 9.2 定向预热

不照搬上游 batch 1..16 全预热。本项目每个 batch 会持有双槽 graph、固定输入输出
缓冲和完整 workspace,全尺寸预热会显著增加启动时间和显存常驻。当前生产实现由
`Backend::warmup_batches` 声明需要预热的物理 batch;CUDA 返回 `[1,maxBatch]`,
其它后端默认不预热。预热发生在 server handle 交给线程前,失败会阻止 evaluator
初始化；`cudaDisableWarmup=true` 可用于诊断和回退。

第一轮只试 B1 + 当前生产上限 B16:

1. 记录无预热时模型 load、首个 B1、首个 B16、第二次 B1/B16 的延迟和 peak VRAM;
2. 创建 handle 后、server ready 前执行 B1/B16 各一次;
3. 确认预热使用真实生产 graph/tactic,并通过路径矩阵;
4. 比较启动总时长、首请求 p50/p95、steady-state throughput 和 peak VRAM。

预热采纳条件:

- 首次 B1/B16 请求 p95 至少降低 50%;
- `startup + first B1 + first B16` 总耗时不比无预热高 10%;
- peak VRAM 增量 `<=512 MiB`;
- steady-state B16 eval/search 无 `>=1%` 回退;
- 预热失败必须在 server ready 前明确失败,不得留下部分 graph 状态。

若不满足,不落地自动预热,但保留同步多 handle benchmark 工具和测量结论。

### 9.3 2026-08-19 warmup 裁决（Windows）

在同一 release 二进制、RTX 5070 Ti/SM120、q64 + DualFFN 认证 plan、
`nnbench eval --warmup 0 --iterations 1` 下，先用无预热模式记录冷首调用，
再用默认定向 warmup 重复。每个进程单独启动，`startup` 包含模型加载和 CUDA
初始化；B16 的生产调度实际平均 batch 为 10.67。

| 配置 | B1 首调用 | B16 首调用 | 进程 wall（B1/B16） |
|---|---:|---:|---:|
| `cudaDisableWarmup=true` | 173.1/177.6 ms | 260.9/272.9 ms | 1.82/1.95 s |
| 默认 warmup `[1,16]` | 5.6/5.8 ms | 89.7/98.6 ms | 1.94/2.04 s |

冷首调用降低约 `96.7%`（B1）和 `65.0%`（B16）；
`startup + first B1 + first B16` 的总耗时从约 `2.29 s` 降至约 `2.14 s`,
满足首请求 p95 `>=50%` 和总耗时不增加 `10%` 的门槛。默认 warmup 的
steady-state 仍使用已有 Windows/WSL ABBA 结果,未观察到 `>=1%` 回退。
warmup 实际命中 graph/direct、q64 和 fused DualFFN 的 marker 留在
`target/cudarocmopt-validation/v5-warmup-gtp-graph.stderr.log`；
`KATAGO_CUDA_TEST_FORCE_WARMUP_FAIL=1` 会在 server-ready 前返回错误。
预热只创建最终生产会持有的 B1/B16 graph/workspace,因此 peak VRAM 增量为
实测约 `+31 MiB`（Windows `nvidia-smi`，低于 `512 MiB` 门槛）；未引入额外
常驻槽位。

结论：定向 warmup 采纳并保留，生产默认启用；显式
`cudaDisableWarmup=true` 仅作诊断/回退，不进入认证 tactic plan。

WSL Ubuntu-24.04 在补齐 `CUDA_HOME=/usr/local/cuda-13.3` 后也完成了 release
重建和 GTP 启动验证（warmup `1,16` 完成 `178.5 ms`）。同一二次重复的
`nnbench eval --warmup 0 --iterations 1` 方向一致：B1 warmup `5.35/5.62 ms`
对无 warmup `202.4/304.5 ms`，B16 warmup `81.8/83.1 ms` 对无 warmup
`345.6/402.0 ms`。WSL 仅作方向性确认，Windows 数据仍是启动/门槛裁决环境。

## 10. 执行顺序、分支和回退

每阶段一个独立提交,后续只依赖已通过阶段:

1. `V0` 只生成结果,不改生产代码;
2. `V1` 路径 report/测试脚本;
3. `V2` schema 2/build id/实核探针;
4. `V3` swizzle 1 候选;
5. `V3R` residual-norm 拓扑候选;
6. `V4` q64 attention 候选;
7. `V5` benchmark 工具,最后单独评估预热。

安全/测量基础设施(V1/V2/V5 benchmark)允许在吞吐持平时保留,但仍要求正确性、
路径测试和 `<1%` 性能回退。V3/V4/自动预热属于行为或性能候选,未过各自门槛时
删除实现,不能以“以后也许有用”为由留在默认或认证 plan 路径中。

每次失败记录:

- 候选和基线 commit/build id;
- 完整配置和 path report;
- 数值结果;
- ABBA/BAAB 原始样本及几何均值;
- 拒绝原因;
- 删除候选代码后的复验 commit。

任何候选失败后,先重跑 r1 基线确认性能恢复,再进入下一候选。

## 11. 结果表

执行时逐行填写,原始日志放在对应 run 目录:

| ID | baseline build | candidate build | 数值门 | 路径门 | WSL ABBA | Windows ABBA | 决策 | 证据目录 |
|---|---|---|---|---|---|---|---|---|
| V0 | `9517be2` | - | - | - | 已冻结 | 已冻结 | 完成 | §11.1 |
| V1 | working tree | working tree | N/A | 15/15 PASS | 矩阵 PASS | 矩阵 PASS | 保留 | `target/cudarocmopt-validation/windows-warmup-matrix` |
| V2 | `9517be2` | working tree | PASS/bitwise | PASS | build/capability PASS | FP32 fused 对合法基线 +17.76% | 保留 | §11.1 |
| V3 swizzle1 | `<2>` | `<1>` | 复用同 epilogue | PASS | 未执行(候选已删) | +0.49% | 拒绝并删除 | §11.1 |
| V3R residual-norm | `beta=1` | deferred add+norm | PASS,top-1 16/16 | 14/14 PASS | 未执行(候选已删) | B8 -1.09%,B16 -1.33% | 拒绝并删除 | `target/cudarocmopt-validation/v3r` |
| V4 q64 | `cudarocmopt-v4/q128` | `cudarocmopt-v4/q64` | PASS,top-1 16/16 | 14/14 PASS | eval B16 `+3.44%`;search t8/t16 `+3.0/+2.4%`;core `+14.7%` | eval B16 `+2.95%`;search visits t8/t16 `+3.34/+2.77%` | 采纳,晋级 schema 2 | `target/cudarocmopt-validation/wsl-tactic-matrix` |
| V5 benchmark | working tree | working tree | N/A | direct/kernel PASS | 路径可用,重复性门槛未通过(B1 双 handle/B16 kernel) | release 三轮 PASS,14/14 path matrix | 工具保留,WSL 扩展性不晋级,多流结果不等价于 GTP/search 收益 | `target/cudarocmopt-validation/v5-handles-current`; `v5-handles-wsl-final` |
| V5 warmup | working tree | working tree | 默认路径 PASS | warmup marker + forced-failure PASS | WSL directional PASS | Windows B1 `-96.7%`,B16 `-65.0%`;启动总耗时门槛 PASS | 采纳,默认 `[1,maxBatch]` | `target/cudarocmopt-validation/v5-warmup-production-enabled.json` |

### 11.1 2026-08-19 已完成结果

V0 冻结基线（`nnbench eval`,B1/B4/B8/B16,400 iterations）:

| 环境 | run 1 (B1/B4/B8/B16) | run 2 (B1/B4/B8/B16) | B16 差异 |
|---|---|---|---|
| Windows | 462.6 / 826.2 / 1006.6 / 1080.4 | 464.5 / 819.4 / 1011.8 / 1081.9 | 0.14% |
| WSL | 466.7 / 832.5 / 1016.7 / 1102.5 | 462.1 / 830.5 / 1020.3 / 1097.5 | 0.46% |

V2:

- schema 1/2 兼容和 build/capability fail-closed 单测 10/10 通过;
- V2 阶段实际 build 指纹:`nvcc 13.3.73`,CUTLASS `3.9.2`
  (`ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e`),compiled SM `120/89`,
  `dual_ffn=true`,`attention_q64=false`(q64 在后续 V4 才加入);
- unfused 与 CUTLASS fused 各做 16 局面 ORT 整图对拍,均 `RESULT: PASS`,
  policy top-1 均 16/16;
- r1 plan 启动日志确认 `compiled=1 probe=pass handle=ready effective=1`,
  首次真实执行确认 `launch=fused`;
- `KATAGO_CUDA_TEST_FORCE_DUALFFN_PROBE_FAIL=1` 时进程在 compute context
  创建阶段失败,未进入首个推理;
- Windows B16 复验见最终构建日志;V4 通过后已生成 schema 2 q64 生产 plan。
- 额外审计发现参考 `cudafusedffn.cu` 的 `ElementAccumulator=half` 不符合本项目
  FP32 累加纪律。已改为 DualGemm 内部 FP32 accumulator;CUTLASS DualEpilogue
  仍先将两路投影各自舍入到 half,再用 FP32 执行 SiLU/乘法并写 half。该版本整图
  对拍通过,并且 80 个 NN 输出二进制与 unfused FP32 路径逐字节一致。B16 ABBA
  unfused = 880.6/880.3(几何均值 880.45),FP32 fused = 1039.4/1034.2
  (几何均值 1036.8),收益 `+17.76%`。它相对不合规的旧 half-accumulate fused
  kernel 约慢 5%,但旧路径不能作为数值合法基线;因此保留 FP32 修正。

V1:

- `scripts/validate_cuda_tactics.py` 直接驱动生产 `nnbench --mode eval`,记录
  命令、环境和完整日志;
- 14 个 case 全部通过:plan 覆盖 env、schema 2 正向、DualFFN on/off、
  graph/direct、FA2/v3、显式布尔 `0`、非法值、未知 key、模型/build mismatch、
  q64 实际 launch、以及强制 DualFFN probe 失败的 fail-closed;
- 实际路径由一次性结构化 marker 证明,不再只根据配置值推断 kernel 是否命中。

V3 swizzle `<1>` 候选:

- 同一 TU 同时编译 `<1>`/`<2>`,每臂日志确认实际 launch 的 swizzle;
- Windows B16 ABBA:`<2>` = 1083.7,1101.0,几何均值 1092.3;
  `<1>` = 1097.0,1098.4,几何均值 1097.7,仅 `+0.49%`;
- 低于 `+1.0%` 采纳门槛,候选代码和 tactic key 已删除,默认仍为 `<2>`;
- swizzle 候选在发现 Linux nvcc 前已按 Windows `+0.49%` 低于门槛删除;不为已明确
  低于采纳线的候选恢复代码重跑 WSL。该取舍不影响 q64 的完整双平台认证。

V3R deferred residual + RMSNorm 候选:

- 实现口径保持 FP32:`beta=0` GEMM 写 FP32 delta,下一层 kernel 做 FP32 add、
  FP32 RMSNorm reduction,仅 norm 输出保持原有 half 存储边界;
- 16 局面 ORT 整图对拍 `RESULT: PASS`,policy top-1 `16/16`;
- 含候选时路径矩阵 14/14 PASS,确认 deferred kernel 实际 launch;
- Windows `nnbench eval` B1/B4/B8/B16,400 iterations,A-B-B-A:
  A1=`407.5/748.4/951.1/1028.7`,A2=`411.1/751.3/957.5/1043.6`;
  B1=`413.1/750.6/946.7/1020.3`,B2=`409.7/738.6/941.1/1024.3`;
- 几何均值相对 A:B1 `+0.51%`,B4 `-0.70%`,B8 `-1.09%`,B16 `-1.33%`。
  B8/B16 触发关键指标回退 `>=1%` 的直接拒绝门槛,故未继续 search;
- 候选 kernel、pending 状态、tactic key 和 autotune group 已删除,生产路径仍为
  `hgemm_residual(beta=1) + rms_norm_f32`。

## 12. 晋级认证 plan

V4 已完成下述流程并晋级。当前文件:

- Windows 生产:`plans/best-tactic-plan.json`(schema 2,q64 + DualFFN);
- Windows q128 回退:`plans/best-tactic-plan-sm120-q128-windows.json`;
- WSL 认证:`plans/best-tactic-plan-sm120-q64-wsl.json`。

schema 2 绑定设备代码 build id;重编 CUDA kernel、切换宿主目标或 CUTLASS 工作树后
plan 失配并拒绝启动是预期行为,必须重新完成数值与 ABBA 认证后生成新 plan。

只有 V3/V4 中被采纳的候选才进入 autotune 决策组。晋级流程:

1. 从 r1 incumbent 开始,不得让 autotune 从空配置重新发现已证伪组合;
2. 运行 `python scripts/autotune.py --threads both --min-improvement 0.01` 的
   ABBA 全套,并确保新增候选在依赖组之后;
3. 对最终组合重新执行整图 ORT 对拍和 V1 全路径矩阵;
4. Windows 与 WSL 各做最终 eval B1/B4/B8/B16 + search t8/t16 ABBA;
5. 生成 schema 2 `plans/best-tactic-plan.json` r2,绑定 model、device、build id、
   CUTLASS/CUDA 和 compiled capabilities;
6. 更新 `docs/cuda-optimization-plan.md` 的 ABBA 留痕和
   `docs/cuda-fork-parity-plan.md` 的进度看板;
7. 用最终提交重新构建并复验,确认验证二进制与提交源码完全一致。

最终原则:只吸收参考分支中在 RTX 5070 Ti、当前模型和本项目生产调度下能被
数值证据与配对性能数据共同证明的优化;源码相似或上游已采用本身不构成采纳依据。
