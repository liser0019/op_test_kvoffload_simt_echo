// CPU semantic execution of the actual generated VFs/controller. No timing or
// address-space/real cross-AIV publication simulation is claimed.
#define GM_ADDR void*
#include "sorted_cann_cpu.h"
template<typename T>inline T asc_ldcg(const T* p){return *p;}
#include "plan.h"
extern "C" __attribute__((visibility("default"))) void fusion_run(void** p,const int64_t* a){
  using namespace qli_fused_plan;
  TilingData d{};
  d.maxRows=a[0];d.topk=a[1];d.capacity=a[2];d.maxToken=a[3];d.maxRequests=a[4];
  d.blockSize=a[5];d.hashCapacity=4096;d.workspaceRowElements=qli_fusion_resource::PrivateWords(d.capacity,d.topk);
  d.scratchElements=qli_fusion_resource::ScratchWords(d.capacity);
  d.localMemoryBytes=qli_fusion_resource::LocalBytes(d.capacity);d.jointHashCapacity=1;d.maxJointRows=8;
  Buffers b{p[0],p[1],p[2],p[3],p[4],nullptr,p[5],p[6],p[7],p[8],p[9],p[10],p[11],p[12],p[13],p[14],p[15],p[16],nullptr};
  for(int64_t first=0;first<d.maxRows;){
    int64_t end=first+1;while(end<d.maxRows&&((int32_t*)p[4])[end]==((int32_t*)p[4])[first])++end;
    AscendC::TPipe pipe;AscendC::TBuf<AscendC::TPosition::VECCALC> storage;pipe.InitBuffer(storage,d.localMemoryBytes);
    // Exercise hybrid reading from the last producer, as well as all-GM.
    auto* local=a[6]?((int32_t*)p[1]+(end-1)*d.topk):nullptr;
    RunGroup(b,d,first,end,d.maxRows,pipe,storage.Get<int32_t>(),local,end-1);
    first=end;
  }
}
