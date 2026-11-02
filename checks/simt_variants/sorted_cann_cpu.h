// CPU-only orchestration model. This is NOT a CANN header or instruction emulator.
// Executes real VF helpers; models documented MrgSort exhausted-prefix semantics.
#pragma once
#include <algorithm>
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <limits>
#include <cmath>
#include "cpu_runtime.h"
inline uint32_t SortedCpuBlock=0,SortedCpuBlocks=1;
using event_t=uint32_t;
struct dim3 { uint32_t x; explicit dim3(uint32_t n):x(n){} };
inline std::unique_ptr<CpuWorkers> SortedWorkers;
inline uint32_t SortedWorkerCount=0;
template <auto F,typename... A> inline void asc_vf_call(dim3 d,A...a){
  if(!SortedWorkers||SortedWorkerCount!=d.x){SortedWorkers.reset();SortedWorkers=std::make_unique<CpuWorkers>(d.x);SortedWorkerCount=d.x;}
  SortedWorkers->Run([&]{F(a...);});
}
namespace AscendC {
enum class HardEvent { V_S,S_V };
enum class TPosition { VECCALC };
inline uint32_t GetBlockIdx(){return SortedCpuBlock;}
inline uint32_t GetBlockNum(){return SortedCpuBlocks;}
template <HardEvent E>inline void SetFlag(event_t){}
template <HardEvent E>inline void WaitFlag(event_t){}
template <typename T>struct LocalTensor {
  T* p=nullptr;uint32_t n=0;
  void* GetPhyAddr(){return p;}
  LocalTensor operator[](uint32_t off)const {assert(off<=n && off%8U==0U);return {p+off,n-off};}
};
class TPipe {public:template<typename B>void InitBuffer(B&b,uint32_t n){b.Init(n);}bool allocated[2]={false,false};
  template<HardEvent E>uint32_t AllocEventID(){auto i=static_cast<unsigned>(E);assert(!allocated[i]);allocated[i]=true;return 0;}
  template<HardEvent E>void ReleaseEventID(uint32_t id){auto i=static_cast<unsigned>(E);assert(id==0&&allocated[i]);allocated[i]=false;}
  ~TPipe(){assert(!allocated[0]&&!allocated[1]);}};
template <TPosition P>class TBuf {
  std::vector<int32_t> backing;uint32_t bytes=0;
 public:
  void Init(uint32_t n){bytes=n;backing.assign(n/4+32,0x1234567);}
  template<typename T>LocalTensor<T> Get(){return {reinterpret_cast<T*>(backing.data()+16),static_cast<uint32_t>(bytes/sizeof(T))};}
  ~TBuf(){for(int i=0;i<16;++i){assert(backing[i]==0x1234567);assert(backing[backing.size()-1-i]==0x1234567);}}
};
template<typename T>struct MrgSortSrcList {LocalTensor<T> src1,src2,src3,src4;};
template<typename T,bool Exhaust>void MrgSort(LocalTensor<T> dst,const MrgSortSrcList<T>&l,
    const uint16_t lengths[4],uint32_t used[4],uint16_t valid,int repeats){
  assert(repeats==1);LocalTensor<T>s[4]={l.src1,l.src2,l.src3,l.src4};
  uint32_t n=valid==3?2:valid==7?3:valid==15?4:0;assert(n>=2);
  for(uint32_t k=0;k<n;++k){used[k]=0;assert(lengths[k]>0&&s[k].n>=2U*lengths[k]);}
  uint32_t out=0;
  while(true){uint32_t best=n;
    for(uint32_t k=0;k<n;++k)if(used[k]<lengths[k]&&(best==n||s[k].p[2U*used[k]]>s[best].p[2U*used[best]]))best=k;
    if(best==n)break;
    assert(2U*out+1U<dst.n);
    std::memcpy(dst.p+2U*out,s[best].p+2U*used[best],2*sizeof(T));++out;++used[best];
    if(Exhaust&&used[best]==lengths[best])break;
  }
}
}
