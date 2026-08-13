// 注意力 kernel v1（正确优先版本；M4 换 FlashAttention 风格 both16 tile 版）。
//
// 输入 Q/K/V：[B*H, S, D] 行主序 FP16（S=361, H=12, D=32），non-causal 无掩码。
// 计算：out[i,:] = Σ_j softmax_j(Q_i·K_j · scale) · V[j,:]
//
// `scale` 由调用方指定：ONNX 图里 q/k 各乘 qk_scale=1/∜d，乘积即 1/√d；
// 传统 1/√d 语义则传 rsqrtf(d)（两者数值等价，缩放挪到 score 上做）。
//
// 每个 block 处理一个查询行 i：blockDim = S（向上取 384），
// 线程 j 先算 score_j，shared 树归约做两遍 softmax 归一，再算输出。

#include <cuda_fp16.h>
#include <cstdint>

#define ATT_BLOCK 512

extern "C" __global__ void attention_row_kernel(
    const __half* __restrict__ q,   // [B*H, S, D]
    const __half* __restrict__ k,   // [B*H, S, D]
    const __half* __restrict__ v,   // [B*H, S, D]
    __half* __restrict__ out,       // [B*H, S, D]
    int s, int d, float scale) {
    const int i = blockIdx.x;   // query 行
    const int bh = blockIdx.y;  // batch*head
    const int j = threadIdx.x;  // key 索引

    __shared__ float s_score[ATT_BLOCK];  // 原始 score（保留）
    __shared__ float s_red[ATT_BLOCK];    // 归约工作区
    __shared__ float s_p[ATT_BLOCK];
    __shared__ float s_maxv;
    __shared__ float s_sumv;

    const __half* qrow = q + ((size_t)bh * s + i) * d;

    // --- score_j = Q_i · K_j * scale ---
    float score = 0.0f;
    if (j < s) {
        const __half* krow = k + ((size_t)bh * s + j) * d;
        for (int dd = 0; dd < d; ++dd) {
            score += __half2float(qrow[dd]) * __half2float(krow[dd]);
        }
        score *= scale;
    }
    s_score[j] = (j < s) ? score : -1e30f;
    s_red[j] = s_score[j];
    __syncthreads();

    // --- 最大值归约（树形，shared）---
    for (int off = ATT_BLOCK / 2; off > 0; off >>= 1) {
        if (j < off) {
            float a = s_red[j];
            float b = s_red[j + off];
            s_red[j] = fmaxf(a, b);
        }
        __syncthreads();
    }
    s_maxv = s_red[0];
    __syncthreads();

    // --- p_j = exp(score - max) / sum ---
    s_red[j] = (j < s) ? expf(s_score[j] - s_maxv) : 0.0f;
    __syncthreads();
    for (int off = ATT_BLOCK / 2; off > 0; off >>= 1) {
        if (j < off) {
            s_red[j] += s_red[j + off];
        }
        __syncthreads();
    }
    s_sumv = s_red[0];
    __syncthreads();
    if (j < s) {
        s_p[j] = expf(s_score[j] - s_maxv) / s_sumv;
    }
    __syncthreads();

    // --- out[i][j] = Σ_jj p[jj] * V[jj][j]（线程 j < d 负责输出维 j）---
    if (j < d) {
        float acc = 0.0f;
        const __half* vcol = v + (size_t)bh * s * d + j;
        for (int jj = 0; jj < s; ++jj) {
            acc += s_p[jj] * __half2float(vcol[(size_t)jj * d]);
        }
        out[((size_t)bh * s + i) * d + j] = __float2half(acc);
    }
}
