// Rust_KataGo 手写 CUDA kernel（M3 起步：基础逐元素 kernel）。
//
// 编译：build.rs 按 configs/sm-targets.json 用 nvcc 生成 PTX 并嵌入二进制；
// 运行时由 cudarc 按设备能力选择目标。所有符号保持 extern "C" 便于稳定加载。
//
// 数值纪律（对齐 KataGo fork 结论）：逐元素/归约类 kernel 保持 FP32 计算与
// 官方的 half 存储边界一致；不要使用 --use_fast_math 或近似激活。

#include <cuda_fp16.h>

// 逐元素 f32 加法（CI 冒烟/自检用）。
extern "C" __global__ void f32_add_kernel(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

// f32 -> f16 转换（输入特征转换用）。
extern "C" __global__ void f32_to_half_kernel(const float* in, __half* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

// f16 -> f32 转换。
extern "C" __global__ void half_to_f32_kernel(const __half* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(in[i]);
}
