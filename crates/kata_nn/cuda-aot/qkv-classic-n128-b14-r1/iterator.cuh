// A register-fragment transform at the existing CUTLASS half store boundary.
// All other epilogue operations and destination predicates remain CUTLASS's.
template<class BaseIterator> struct RopeStoreIterator : BaseIterator {
  using Base=BaseIterator;
  using Fragment=typename Base::Fragment;
  using ThreadMap=typename Base::ThreadMap;
  using Layout=typename Base::Layout;
  using TensorCoord=typename Base::TensorCoord;
  using Element=typename Base::Element;
  using IteratorParams=typename BaseIterator::Params;
  struct Params : IteratorParams {
    const float *co=nullptr,*sn=nullptr;
    CUTLASS_HOST_DEVICE Params(){}
    CUTLASS_HOST_DEVICE Params(Layout const& l):IteratorParams(l){}
  };
  const float *co_,*sn_;
  CUTLASS_DEVICE RopeStoreIterator(Params const& p,Element* output,TensorCoord extent,int thread,TensorCoord offset=TensorCoord(),int const* indices=nullptr)
    :Base(p,output,extent,thread,offset,indices),co_(p.co),sn_(p.sn){}
  CUTLASS_DEVICE void store(Fragment const& fragment)const {
    // LinearCombination has already performed the first FP32 -> RN half.
    // Transform the half register fragment, without reading the output tensor.
    Fragment rotated=fragment;
    if(co_ && this->thread_start_column()<768){
      CUTLASS_PRAGMA_UNROLL
      for(int cl=0;cl<ThreadMap::Iterations::kCluster;++cl){
        CUTLASS_PRAGMA_UNROLL
        for(int gr=0;gr<ThreadMap::Iterations::kGroup;++gr){
          CUTLASS_PRAGMA_UNROLL
          for(int rr=0;rr<ThreadMap::Iterations::kRow;++rr){
            int row=this->thread_start_row()+rr*ThreadMap::Delta::kRow+gr*ThreadMap::Delta::kGroup+cl*ThreadMap::Delta::kCluster;
            int fragrow=rr+ThreadMap::Iterations::kRow*(gr+ThreadMap::Iterations::kGroup*cl);
            CUTLASS_PRAGMA_UNROLL
            for(int cc=0;cc<ThreadMap::Iterations::kColumn;++cc){
              int col=this->thread_start_column()+cc*ThreadMap::Delta::kColumn;
              int start=(fragrow*ThreadMap::Iterations::kColumn+cc)*Base::kElementsPerAccess;
              if(row<this->extent_row() && col<768){
                CUTLASS_PRAGMA_UNROLL
                for(int e=0;e<Base::kElementsPerAccess;e+=2){
                  int i=start+e,r=(row%361)*192+((col+e)%384)/2;
                  cutlass::half_t ah=fragment[i],bh=fragment[i+1];
                  float c=co_[r],s=sn_[r],a=exact_h2f(ah.raw()),b=exact_h2f(bh.raw()),ac,bs,as,u,v;
                  asm("mul.rn.f32 %0, %1, %2;":"=f"(ac):"f"(a),"f"(c));
                  asm("mul.rn.f32 %0, %1, %2;":"=f"(bs):"f"(b),"f"(s));
                  asm("mul.rn.f32 %0, %1, %2;":"=f"(as):"f"(a),"f"(s));
                  asm("sub.rn.f32 %0, %1, %2;":"=f"(u):"f"(ac),"f"(bs));
                  asm("fma.rn.f32 %0, %1, %2, %3;":"=f"(v):"f"(b),"f"(c),"f"(as));
                  rotated[i]=cutlass::half_t::bitcast(exact_f2h(u));
                  rotated[i+1]=cutlass::half_t::bitcast(exact_f2h(v));
                }
              }
            }
          }
        }
      }
    }
    Base::store(rotated);
  }
};
using OriginalKernel=Gemm::GemmKernel;
using OriginalEpilogue=OriginalKernel::Epilogue;
using OriginalIterator=OriginalEpilogue::OutputTileIterator;
using FusedIterator=RopeStoreIterator<OriginalIterator>;
using FusedEpilogue=cutlass::epilogue::threadblock::Epilogue<
  OriginalEpilogue::Shape,OriginalEpilogue::WarpMmaOperator,OriginalEpilogue::kPartitionsK,
  FusedIterator,OriginalEpilogue::AccumulatorFragmentIterator,OriginalEpilogue::WarpTileIterator,
  OriginalEpilogue::SharedLoadIterator,OriginalEpilogue::OutputOp,OriginalEpilogue::Padding,
  OriginalEpilogue::Base::kFragmentsPerIteration,1>;
using Kernel=cutlass::gemm::kernel::Gemm<OriginalKernel::Mma,FusedEpilogue,Gemm::ThreadblockSwizzle,false>;
static_assert(OriginalIterator::kElementsPerAccess==8,"four adjacent half pairs per vector");
static_assert(FusedEpilogue::kPartitionsK==1 && !Kernel::kSplitKSerial,"no split-K");
static_assert(sizeof(Kernel::SharedStorage)==sizeof(OriginalKernel::SharedStorage),"unchanged shared storage");
// All iterator parameters, including RoPE pointers, are populated on the host.
// A grid-constant read-only parameter has no per-thread mutable parameter copy.
template<bool ROTATE>__global__ void immutable_qkv_rope(CUTLASS_GRID_CONSTANT Kernel::Params const params){
  extern __shared__ int storage[];
  Kernel op;op(params,*reinterpret_cast<Kernel::SharedStorage*>(storage));
}

// CPU-only export of the actual instantiated ThreadMap. The row progression
// follows PredicatedTileIterator::operator++; the checker verifies coverage.
void export_mapping(fs::path const& out){
  need(!fs::exists(out),"fresh mapping directory");fs::create_directories(out);
  using T=OriginalIterator::ThreadMap;std::vector<int32_t> records;
  for(int t=0;t<OriginalIterator::kThreads;++t){
    auto initial=T::initial_offset(t);int rowbase=initial.row(),colbase=initial.column();int state[3]={0,0,0};
    for(int it=0;it<OriginalIterator::kIterations;++it){
      for(int cl=0;cl<T::Iterations::kCluster;++cl)for(int gr=0;gr<T::Iterations::kGroup;++gr)for(int rr=0;rr<T::Iterations::kRow;++rr)for(int cc=0;cc<T::Iterations::kColumn;++cc){
        int row=rowbase+rr*T::Delta::kRow+gr*T::Delta::kGroup+cl*T::Delta::kCluster;
        int col=colbase+cc*T::Delta::kColumn;
        int start=((rr+T::Iterations::kRow*(gr+T::Iterations::kGroup*cl))*T::Iterations::kColumn+cc)*OriginalIterator::kElementsPerAccess;
        for(int x:{t,it,start,row,col,OriginalIterator::kElementsPerAccess})records.push_back(x);
      }
      ++state[0];rowbase+=T::Shape::kRow;
      if(state[0]==T::Count::kRow){state[0]=0;++state[1];rowbase+=(T::Shape::kGroup-1)*T::Shape::kRow*T::Count::kRow;
        if(state[1]==T::Count::kGroup){state[1]=0;++state[2];rowbase+=T::Count::kGroup*T::Shape::kGroup*T::Count::kRow*T::Shape::kRow;
          if(state[2]==T::Count::kCluster){state[2]=0;rowbase+=T::Shape::kGroup*T::Shape::kRow*T::Shape::kCluster*T::Shape::kTile;}
        }
      }
    }
  }
  raw(out/"vectors.i32le",records);std::ofstream f(out/"report.json");
  f<<"{\"status\":\"EXPORTED_COMPILED_THREAD_MAP_CPU_ONLY\",\"threads\":"<<OriginalIterator::kThreads<<",\"epilogue_iterations\":"<<OriginalIterator::kIterations<<",\"fragment_elements\":"<<OriginalIterator::Fragment::kElements<<",\"elements_per_access\":"<<OriginalIterator::kElementsPerAccess<<",\"records\":"<<records.size()/6<<",\"shared_bytes\":"<<sizeof(Kernel::SharedStorage)<<",\"gpu_used\":false}";
  need(bool(f),"mapping report");std::cout<<"EXPORTED_COMPILED_THREAD_MAP_CPU_ONLY\n";
}
