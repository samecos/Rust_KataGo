// 正向对照:SM120 消费级张量核的窄精度 mma.sync 通路
#include <cstdint>
__global__ void probe_mma_f8f6f4(float* d, uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3, uint32_t b0,uint32_t b1) {
  float d0=0,d1=0,d2=0,d3=0;
  asm volatile(
    "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
  d[0]=d0+d1+d2+d3;
}
// FP16 基准通路(已知可用)
__global__ void probe_mma_f16(float* d, uint32_t a0,uint32_t a1, uint32_t b0) {
  float d0=0,d1=0,d2=0,d3=0;
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
    : "r"(a0),"r"(a1),"r"(a0),"r"(a1),"r"(b0),"r"(b0));
  d[0]=d0+d1+d2+d3;
}
