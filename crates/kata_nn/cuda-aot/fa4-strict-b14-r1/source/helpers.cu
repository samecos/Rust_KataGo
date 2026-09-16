
__device__ float h2f(unsigned short h) {
    float v; asm("cvt.f32.f16 %0, %1;" : "=f"(v) : "h"(h)); return v;
}
__device__ unsigned short f2h(float v) {
    unsigned short h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(v)); return h;
}
extern "C" __global__ void probe_rope(
    const unsigned short* x, const float* co, const float* sn, unsigned short* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= 14 * 361 * 192) return;
    int row = i / 192, pair = i % 192, r = (row % 361) * 192 + pair;
    int offset = row * 1152 + pair * 2;
    float c = co[r], s = sn[r];
    #pragma unroll
    for(int qk=0;qk<2;qk++) {
        int p = offset + qk * 384;
        float a=h2f(x[p]), b=h2f(x[p+1]);
        y[p]=f2h(a*c-b*s); y[p+1]=f2h(a*s+b*c);
    }
    y[offset+768]=x[offset+768]; y[offset+769]=x[offset+769];
}
extern "C" __global__ void probe_reference(
    const unsigned short* x, float* out, float scale) {
    int qi=blockIdx.x, bh=blockIdx.y, b=bh/12, h=bh%12, t=threadIdx.x;
    __shared__ float scores[361];
    int qoff=(b*361+qi)*1152+h*32;
    for(int ki=t;ki<361;ki+=blockDim.x) {
        int koff=(b*361+ki)*1152+384+h*32;
        float dot=0;
        #pragma unroll
        for(int d=0;d<32;d++) dot=__fadd_rn(dot,__fmul_rn(h2f(x[qoff+d]),h2f(x[koff+d])));
        scores[ki]=__fmul_rn(dot,scale);
    }
    __syncthreads();
    if(t==0) {
        float mx=-3.402823466e38F, sum=0;
        for(int i=0;i<361;i++) mx=fmaxf(mx,scores[i]);
        for(int i=0;i<361;i++) { scores[i]=expf(scores[i]-mx); sum=__fadd_rn(sum,scores[i]); }
        for(int i=0;i<361;i++) scores[i]=__fdiv_rn(scores[i],sum);
    }
    __syncthreads();
    if(t<32) {
        float acc=0;
        for(int ki=0;ki<361;ki++) {
            int voff=(b*361+ki)*1152+768+h*32+t;
            acc=__fadd_rn(acc,__fmul_rn(scores[ki],h2f(x[voff])));
        }
        out[(b*361+qi)*384+h*32+t]=acc;
    }
}


extern "C" __global__ void probe_rope_contract(
    const unsigned short* x, const float* co, const float* sn, unsigned short* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= 14 * 361 * 192) return;
    int row=i/192, pair=i%192, r=(row%361)*192+pair;
    int offset=row*1152+pair*2;
    float c=co[r], s=sn[r];
    #pragma unroll
    for(int qk=0;qk<2;qk++) {
        int p=offset+qk*384;
        float a=h2f(x[p]), b=h2f(x[p+1]), ac, bs, as, u, v;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(ac) : "f"(a), "f"(c));
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(bs) : "f"(b), "f"(s));
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(as) : "f"(a), "f"(s));
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(u) : "f"(ac), "f"(bs));
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(v) : "f"(b), "f"(c), "f"(as));
        y[p]=f2h(u); y[p+1]=f2h(v);
    }
    y[offset+768]=x[offset+768]; y[offset+769]=x[offset+769];
}


extern "C" __global__ void probe_rope_no_vcopy(
    const unsigned short* x, const float* co, const float* sn, unsigned short* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= 14 * 361 * 192) return;
    int row=i/192, pair=i%192, r=(row%361)*192+pair;
    int offset=row*1152+pair*2;
    float c=co[r], s=sn[r];
    #pragma unroll
    for(int qk=0;qk<2;qk++) {
        int p=offset+qk*384;
        float a=h2f(x[p]), b=h2f(x[p+1]), ac, bs, as, u, v;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(ac) : "f"(a), "f"(c));
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(bs) : "f"(b), "f"(s));
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(as) : "f"(a), "f"(s));
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(u) : "f"(ac), "f"(bs));
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(v) : "f"(b), "f"(c), "f"(as));
        y[p]=f2h(u); y[p+1]=f2h(v);
    }
}
