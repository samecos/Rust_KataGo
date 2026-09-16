# Strict FA4 r2 codegen audit — 2026-09-08

**结论：在本次导出的 IR/PTX 范围内，exp 与归一化符合当前 Rust
q64serial 的非 fast-math FP32 路径。数值验证尚未完成。** 本次只读分析已有产物，
未编译、未运行 GPU，也未改变 manifest 的 `EXPORTED_UNVERIFIED` 状态。

审计对象：B14/S361/H12/D32，M128/N96，QK/PV FP32 累加。产物完整路径见
[manifest.json](manifest.json)；下面的 P/C 分别指该目录唯一 PTX / `*_clean.mlir`。

| 对象 | SHA256 |
| --- | --- |
| r2 PTX | `0856940c9f8a2054468c83b04614825ad68c2ea65b98e2083cc9e294aeb84011` |
| r2 cubin（manifest） | `90341b6d5c54a69e983a7e7d8b62956748785ad44c3e4e1b2734b92f3da46749` |
| Rust q64serial PTX | `dc462cc9cc832d69b72b3c009f690e7aa5c0cd2dbcc15bfcd9df3347b7b460dc` |

Rust 对照文件为
`target/cudarocmopt-wsl/release/build/kata_nn-09ad76d4c77a4f6c/out/sm_120_attention_fa2_q64_serial.ptx`。
r2 NVVM 标头为 CUDA 13.3.27，Rust 为 CUDA 13.3.73；均为 NVVM 23 / PTX 9.3 / SM120。

## 完整 exp / div 行为

- 源码两处 `cute.math.exp(..., fastmath=False)` 生成 FP32 `math.exp`；
  `cute.math.div(..., fastmath=False)` 生成 FP32 `arith.divf`。IR 没有 fast-math
  属性、`math.exp2`、log/LSE 运算。默认无属性即无 fast-math flags。
- 检查了 **全部 196 处** r2 exp 和 **全部 34 处** Rust expf，追溯每处输入至
  输出乘法的依赖图，并消去寄存器命名和常量搬运差异：全部得到同一个图。
  结果及逐文件统计保存在 [codegen-structural-scan.json](codegen-structural-scan.json)。
- 具体序列为范围规约（FMA / 饱和转换 / 定向舍入 FMA）、指数位构造，使用
  `0f3FB8AA3B` 与 `0f32A57060` 高低两段常量的两次 FMA 校正，然后
  `ex2.approx.ftz.f32` 和最终 `mul.f32` 重建结果。r2 示例 P:965–976，
  Rust:1109–1120；在线旧输出缩放也采用同一序列（P:4590–4599）。
  因而这里的 `ex2.approx` 是普通 expf 完整实现中的一步，不能单独认定为
  Fork 旧版的直接近似 exp2 路径。此结论基于整个依赖图，而非仅搜索指令名称。
- r2 归一化 P:5045–5064 为四个 `rcp.rn.f32`，Rust:1732/1772 为两个同类
  指令；**没有 `rcp.approx` 或 `div.approx`**。r2 保留零/NaN 分母返回 1
  的保护，PTX 用 `setp.equ` / `selp` 实现。`.rn` 是最近偶数舍入，且该
  reciprocal 未带 `.ftz`。[NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-rcp)

这里的“精确形式”指普通 CUDA 数学语义；普通 `expf` 本身仍有有限误差，
不能解释成所有输入都精确舍入。NVIDIA 的精度表列出 `expf` 为 2 ULP，
并将快速 `__expf` 单列为随输入增大的误差界。
[CUDA 浮点函数精度表](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html)

## scale、累加与 half 边界

- 自然 FP32 scale 为参数 4，未在 host/kernel 入口乘 `LOG2_E`。
  C:1516–1518/1785–1787 先以 FP32 乘 score，再做 max；C:1602/1873 做减法，
  随后 exp。P:867–880、901–944、950–976 保持这一顺序。
  `mul/sub.f32x2` 是两路 FP32 运算，不是 half2。
  [NVIDIA packed FP32 指令](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-mul)
- 全部 **192 条 MMA** 均为 `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`；
  C:93/122 的 QK、PV tiled MMA 及 accumulator 类型均为 FP32。
- exp 与 row sum 先保持 FP32（C:1603–1607）；随后 P 仅在 PV 前通过
  `arith.truncf f32→f16`（C:1627/1898，P:2363 起 `cvt.rn.f16x2.f32`）存入
  half fragment，PV 仍 FP32 累加。最终输出也转 half（C:1999）。
- N96 分块、遍历和归约顺序与 Rust q64serial 不同；half P 的舍入时刻依赖
  各块在线最大值。因此上述 codegen 相符 **不能推出 attention 或整图逐位相同**。

仍须完成新 ABI probe、attention 参考及原生 TF3/ONNX 全头对拍，最后再做
无并发干扰的 ABBA。没有新的数值门、性能收益或生产采纳结论。
