// ptxas 裁决实验:sm_120f 是否接受 tcgen05 指令
#include <cstdint>

__global__ void probe_tmem_alloc(uint32_t* out) {
  // tcgen05.alloc:分配 tensor memory
  __shared__ uint32_t dst;
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
               :: "r"(uint32_t(__cvta_generic_to_shared(&dst))), "r"(128));
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" :: "r"(0), "r"(128));
  *out = dst;
}

__global__ void probe_mma_f16(uint64_t da, uint64_t db, uint32_t tmem_c, uint32_t idesc) {
  // tcgen05.mma kind::f16 (FP16 输入 FP32 累加)
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %3, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %4, p;\n\t"
      "}\n"
      :: "r"(tmem_c), "l"(da), "l"(db), "r"(1u), "r"(idesc));
}

__global__ void probe_mma_f8f6f4(uint64_t da, uint64_t db, uint32_t tmem_c, uint32_t idesc) {
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %3, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %4, p;\n\t"
      "}\n"
      :: "r"(tmem_c), "l"(da), "l"(db), "r"(1u), "r"(idesc));
}
