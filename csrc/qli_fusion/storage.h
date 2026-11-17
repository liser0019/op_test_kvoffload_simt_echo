// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "resources.h"

namespace qli_fused_plan {
// 当前producer行来自UB；其他producer已经发布的行从GM读，沿用原融合协议。
struct FusionTopk {
  __gm__ int32_t* gm;
  __ubuf__ int32_t* local=nullptr;
  int64_t begin=0, count=0, offset=0;
  __simt_callee__ __aicore__ inline FusionTopk(__gm__ int32_t* p):gm(p){}
  __simt_callee__ __aicore__ inline FusionTopk(__gm__ int32_t* p,__ubuf__ int32_t* l,int64_t b,int64_t n):gm(p),local(l),begin(b),count(n){}
  __simt_callee__ __aicore__ inline int32_t operator[](int64_t i)const{
    i+=offset;return local&&i>=begin&&i<begin+count?local[i-begin]:asc_ldcg(gm+i);
  }
  __simt_callee__ __aicore__ inline FusionTopk operator+(int64_t i)const{auto v=*this;v.offset+=i;return v;}
  __simt_callee__ __aicore__ inline operator __gm__ int32_t*()const{return gm+offset;}
};

// A/B存query位置，不截断token ID。C只在后续阶段读已经完整写入的GM unique。
struct FusionUnique {
  __ubuf__ int32_t* data;
  __gm__ int32_t* gm;
  FusionTopk topk;
  int64_t queryBase;
  uint32_t c,offset=0;
  bool single=false;
  __ubuf__ int32_t* singleBegin=nullptr;
  __simt_callee__ __aicore__ inline int32_t operator[](uint32_t uid)const{
    uid+=offset;
    // Strictly increasing singleton: UID already is its position in the valid
    // prefix. Preserve the parent's direct path without adding atomic packing.
    if(single)return topk[queryBase+static_cast<uint32_t>(*singleBegin)+uid];
    if constexpr(QLI_PLAN_STORAGE==3)return asc_ldcg(gm+uid);
    else return topk[queryBase+reinterpret_cast<__ubuf__ uint16_t*>(data+c/2U)[uid]];
  }
  __simt_callee__ __aicore__ inline FusionUnique operator+(uint32_t i)const{auto v=*this;v.offset+=i;return v;}
  __simt_callee__ __aicore__ inline void Set(uint32_t uid,int32_t token,uint32_t position)const{
    if(single)return;
    if constexpr(QLI_PLAN_STORAGE==3)gm[uid]=token;
    else {
      // Only the singleton/SIMT compatibility producer uses Set. Adjacent halfwords
      // may share one B32 store, so publish them with CAS rather than a racing RMW.
      auto* word=data+c/2U+uid/2U;
      const uint32_t shift=(uid&1U)*16U,mask=65535U<<shift;
      uint32_t old=static_cast<uint32_t>(asc_atomic_cas(word,0,0)),seen;
      do{seen=old;old=static_cast<uint32_t>(asc_atomic_cas(word,static_cast<int32_t>(seen),static_cast<int32_t>((seen&~mask)|(position<<shift))));}while(old!=seen);
    }
  }
};

__simd_callee__ inline void FusionStoreUnique(
    __ubuf__ int32_t* data,AscendC::Reg::RegTensor<int32_t>& token,
    AscendC::Reg::RegTensor<uint32_t>& payload,AscendC::Reg::RegTensor<uint32_t>& uid,
    AscendC::Reg::RegTensor<uint32_t>& lane,AscendC::Reg::MaskReg& writeHead,
    AscendC::Reg::MaskReg& full,uint32_t mapWords,uint32_t capacity,uint32_t initialCount){
  using namespace AscendC;
  Reg::RegTensor<uint32_t> index,safe;
  if constexpr(QLI_PLAN_STORAGE==3){
    Reg::Adds(index,uid,mapWords-initialCount,full);
    Reg::Adds(safe,lane,mapWords+2048U,full);Reg::Select(index,index,safe,writeHead);
    Reg::Scatter<int32_t,uint32_t>(data,token,index,writeHead);
  }else{
    Reg::Adds(index,uid,2U*mapWords,full);
    Reg::Adds(safe,lane,2U*mapWords+capacity+64U,full);
    Reg::Select(index,index,safe,writeHead);
    Reg::Adds(safe,lane,2U*mapWords+capacity+128U,full);
    Reg::ShiftLefts(safe,safe,int16_t{16},full);Reg::Or(index,index,safe,full);
    Reg::Scatter<uint16_t,uint16_t>(reinterpret_cast<__ubuf__ uint16_t*>(data),
      reinterpret_cast<Reg::RegTensor<uint16_t>&>(payload),
      reinterpret_cast<Reg::RegTensor<uint16_t>&>(index),writeHead);
  }
}
} // namespace qli_fused_plan
