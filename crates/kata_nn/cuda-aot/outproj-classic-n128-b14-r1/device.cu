#include "kernel.cuh"
extern "C" __global__ void outproj_classic_tn_n128(
 CUTLASS_GRID_CONSTANT outproj_n128::Kernel::Params const params) {
 extern __shared__ int storage[];
 outproj_n128::Kernel op;
 op(params,*reinterpret_cast<outproj_n128::Kernel::SharedStorage*>(storage));
}
