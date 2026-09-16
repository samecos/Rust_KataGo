#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <type_traits>
namespace outproj_n128 {
using Half=cutlass::half_t;
using Row=cutlass::layout::RowMajor;
using Col=cutlass::layout::ColumnMajor;
using Tile=cutlass::gemm::GemmShape<128,128,32>;
using Warp=cutlass::gemm::GemmShape<64,64,32>;
using Instruction=cutlass::gemm::GemmShape<16,8,16>;
using Gemm=cutlass::gemm::device::Gemm<Half,Row,Half,Col,float,Row,float,
 cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,Tile,Warp,Instruction,
 cutlass::epilogue::thread::LinearCombination<float,4,float,float>,
 cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,3,8,8,false>;
using Kernel=Gemm::GemmKernel;
using Iterator=Kernel::Epilogue::OutputTileIterator;
// The installed QKV core uses these same A/B, warp and stage parameters.
using ReferenceGemm=cutlass::gemm::device::Gemm<Half,Row,Half,Col,Half,Row,float,
 cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,Tile,Warp,Instruction,
 cutlass::epilogue::thread::LinearCombination<Half,8,float,float>,
 cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,3,8,8,false>;
static_assert(std::is_same<Kernel::Mma,ReferenceGemm::GemmKernel::Mma>::value,"same installed N128 FP32 mainloop type");
static_assert(std::is_same<Gemm::ElementAccumulator,float>::value,"FP32 accumulation");
static_assert(std::is_same<Gemm::ElementC,float>::value,"FP32 residual and output");
static_assert(Kernel::kThreadCount==128 && Kernel::Mma::Detail::kStages==3,"four warps and three stages");
static_assert(sizeof(Kernel::SharedStorage)==49152,"same three-stage shared storage");
static_assert(!Kernel::kSplitKSerial && Kernel::Epilogue::kPartitionsK==1,"no split K");
}
