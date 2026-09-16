// CPU-only ABI exporter using the exact operator probe's instantiated types.
#define main original_probe_main
#include "../qkv-immutable-params-rope-r1/probe.cu"
#undef main
#include <new>
#include <array>
static_assert(sizeof(void*)==8 && sizeof(size_t)==8,"Win64 ABI");
std::vector<unsigned char> parameter_image(std::array<uint64_t,6> tags){
  alignas(Kernel::Params) unsigned char bytes[sizeof(Kernel::Params)]{};
  Gemm::Arguments a({M,N,K},{reinterpret_cast<cutlass::half_t*>(tags[0]),K},{reinterpret_cast<cutlass::half_t*>(tags[1]),K},{reinterpret_cast<cutlass::half_t*>(tags[2]),N},{reinterpret_cast<cutlass::half_t*>(tags[3]),N},{1.0f,0.0f},1);
  need(Gemm::can_implement(a)==cutlass::Status::kSuccess,"alignment ABI sample");
  Gemm::ThreadblockSwizzle swizzle;auto tiled=swizzle.get_tiled_shape(a.problem_size,{128,64,32},1);
  new(bytes)Kernel::Params{a.problem_size,tiled,a.ref_A.non_const_ref(),a.ref_B.non_const_ref(),a.ref_C.non_const_ref(),a.ref_D,a.epilogue,nullptr,a.gather_A_indices,a.gather_B_indices,a.scatter_D_indices};
  auto p=reinterpret_cast<Kernel::Params*>(bytes);
  need(p->problem_size.m()==M&&p->problem_size.n()==N&&p->problem_size.k()==K&&p->grid_tiled_shape.m()==40&&p->grid_tiled_shape.n()==18&&p->gemm_k_size==384&&p->swizzle_log_tile==0,"fixed ABI dimensions");
  need(p->params_C.co==nullptr&&p->params_C.sn==nullptr&&p->params_D.co==nullptr&&p->params_D.sn==nullptr&&p->semaphore==nullptr,"null optional pointers");
  need(p->output_op.alpha==1.0f&&p->output_op.beta==0.0f,"fixed FP32 scale");
  p->params_D.co=reinterpret_cast<const float*>(tags[4]);p->params_D.sn=reinterpret_cast<const float*>(tags[5]);
  return std::vector<unsigned char>(bytes,bytes+sizeof(bytes));
}
int main(int argc,char** argv){try{
  need(argc==2,"new-output-directory");fs::path out=argv[1];need(!fs::exists(out),"fresh ABI output");fs::create_directories(out);
  std::array<uint64_t,6>a={0x1111222233334400ull,0x5555666677778800ull,0x9999aaaabbbbcc00ull,0xddddeeeeffff0000ull,0x2468ace13579bd00ull,0x98765432fedcba00ull};
  std::array<uint64_t,6>b={0x0123456789abcd00ull,0xfedcba9876543200ull,0x13579bdf2468ac00ull,0xfdb97531eca86400ull,0x777788889999aa00ull,0xbbbbaaaa99998800ull};
  auto x=parameter_image(a),y=parameter_image(b);std::array<size_t,6> offsets{};auto templ=x;
  for(int role=0;role<6;++role){int found=0;for(size_t i=0;i+8<=x.size();++i)if(!memcmp(x.data()+i,&a[role],8)){offsets[role]=i;++found;need(!memcmp(y.data()+i,&b[role],8),"tagged ABI slot");memset(templ.data()+i,0,8);}need(found==1&&offsets[role]%8==0,"unique aligned pointer slot");}
  auto compare=y;for(auto i:offsets)memset(compare.data()+i,0,8);need(compare==templ,"only six data addresses differ");
  raw(out/"sample-a.bin",x);raw(out/"sample-b.bin",y);raw(out/"params-template.bin",templ);
  std::ofstream f(out/"abi.json");f<<"{\"status\":\"EXPORTED_EXACT_CUTLASS_PARAMS_CPU_ONLY\",\"gpu_used\":false,\"parameter_size\":"<<sizeof(Kernel::Params)<<",\"parameter_alignment\":"<<alignof(Kernel::Params)<<",\"pointer_offsets\":{\"input\":"<<offsets[0]<<",\"weight\":"<<offsets[1]<<",\"source\":"<<offsets[2]<<",\"output\":"<<offsets[3]<<",\"cos\":"<<offsets[4]<<",\"sin\":"<<offsets[5]<<"},\"shape\":[5054,1152,384],\"grid\":[40,18,1],\"threads\":128,\"shared_bytes\":"<<sizeof(Kernel::SharedStorage)<<",\"alpha\":1,\"beta\":0}";need(bool(f),"ABI write");std::cout<<"EXPORTED_EXACT_CUTLASS_PARAMS_CPU_ONLY\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
