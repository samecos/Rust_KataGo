# C2 tcgen05 可行性探针(2026-08-17,结论:硬件证伪)

M3/C2(tcgen05 主 GEMM 通路)的可行性裁决探针。结论与证据链落档于
`docs/cuda-optimization-plan.md`「M3-pre:C2 tcgen05 硬件证伪」节。

## 文件

- `tcgen05_probe.cu` — ptxas 裁决探针:tcgen05.alloc / dealloc / mma(kind::f16
  与 kind::f8f6f4)三条指令。
- `mma_f8f6f4_probe.cu` — 正向对照:窄精度 `mma.sync.kind::f8f6f4` 与经典
  FP16 `mma.sync.m16n8k16`。
- `sm120_fp8_gemm_bench.cu` — CUTLASS 4.7.0 SM120 dense FP8 GEMM 完整模板
  (TMA + f8f6f4 mma.sync + Sm120 epilogue,tile 128x64x64),含 128³ 数值自检
  (CPU 双精度参考)与 cublas FP16 同形状对照计时(ffn_up/qkv/ffn_down/
  outproj 生产形状)。

## 复现命令(Windows,Git Bash)

```bash
CLBIN="/c/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.50.35717/bin/Hostx64/x64"

# 1) tcgen05 证伪:sm_120f / sm_120a 均报 "Instruction ... not supported"
nvcc -cubin -arch=sm_120f -ccbin "$CLBIN" -o probe.cubin tcgen05_probe.cu
nvcc -cubin -arch=sm_120a -ccbin "$CLBIN" -o probe.cubin tcgen05_probe.cu
#    对照:sm_100f 编译通过(探针良构)
nvcc -cubin -arch=sm_100f -ccbin "$CLBIN" -o probe.cubin tcgen05_probe.cu

# 2) 窄精度 mma.sync:sm_120f 可编,plain sm_120 拒(.kind::f8f6f4)
nvcc -cubin -arch=sm_120f -ccbin "$CLBIN" -o mma.cubin mma_f8f6f4_probe.cu   # OK
nvcc -cubin -arch=sm_120  -ccbin "$CLBIN" -o mma.cubin mma_f8f6f4_probe.cu   # 报错

# 3) CUTLASS 4.7 SM120 FP8 GEMM(device 侧通过;Windows host 发射受 C2719 阻塞)
nvcc -cubin -O3 -std=c++17 --expt-relaxed-constexpr -arch=sm_120f -ccbin "$CLBIN" \
  -I/d/code/cutlass4/include -I/d/code/cutlass4/tools/util/include \
  -Xcompiler "/EHsc /bigobj /std:c++17 /Zc:preprocessor /Zc:__cplusplus" \
  sm120_fp8_gemm_bench.cu -o bench.cubin        # ✅ 25.8s,cubin 96KB
# 可执行链接(-o bench.exe -L"$CUDA_PATH/lib/x64" -lcublas)在 MSVC 下
# 触发 C2719(alignas(128) 内核参数按值传递,SM90+ TMA 通病);
# 跑数值/计时需 WSL 构建、clang-cl 作 -ccbin,或自写 driver-API launcher。
```

依赖:CUTLASS v4.7.0 在 `D:/code/cutlass4`(与 3.9.2 并存,后者仍服务
DualGemm);nvcc CUDA 13.2;MSVC 经 `-ccbin` 指定。
