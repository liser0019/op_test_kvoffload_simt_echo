// SPDX-License-Identifier: Apache-2.0
#pragma once
// 两条16K候选共用收尾。512线程；保留32位owner的LRU-last原子仲裁。
// 1: map暂存GM，释放32KiB UB作uint16 flags；2: 全UB三位图。
// 此版分类压缩采用SIMT Scan，不能把与旧BD的差值全部归于存储介质。
__simt_callee__ __aicore__ inline void Capacity16Finish(const JointPlanArgs& a,
    int64_t first,int64_t end,__ubuf__ int32_t* data,__ubuf__ int32_t* ctx,
    __ubuf__ int32_t* scan,__gm__ int32_t* workspace) {
  const uint32_t t=threadIdx.x,c=a.capacity,topk=a.topk,q=(end-first)*topk,u=ctx[1];
  auto* map=reinterpret_cast<__ubuf__ uint16_t*>(data);
  auto* unique=data+8192U;
  auto* owner=unique+c;
  auto* part=reinterpret_cast<__ubuf__ uint16_t*>(owner+c);
  auto* bits=reinterpret_cast<__ubuf__ uint32_t*>(part+c);
  auto* flags=map;
  const int64_t qb=first*topk,cb=first*c;
  const uint32_t rows=end-first;
  const int32_t req=a.tokenToReq[first];const int64_t reqid=a.reqIds[req];
  const bool reset=a.rowModes[first]!=(rows==1?ROW_PRIVATE:ROW_JOINT)||a.lastReqIds[first]!=reqid;
  if(reset){asc_syncthreads();FinishColdSorted<SORTED_THREADS,true>(a,first,end,EightMap16View{map},unique,ctx,t);return;}
  int32_t stable=a.stablePrefixLens[first];
  for(int64_t r=first+1;r<end;++r)if(a.stablePrefixLens[r]<stable)stable=a.stablePrefixLens[r];
  if(stable<0)stable=0;if(stable>a.maxToken)stable=a.maxToken;
#if JOINT_LAYOUT16 == 1
  // One packed32-bit store per pair. No overlapping halfword GM writers.
  for(uint32_t i=t;i<q/2;i+=SORTED_THREADS)workspace[qb/2+i]=data[i];
  asc_syncthreads(); // all map reads precede flags overwrite
#endif
  for(uint32_t i=t;i<u;i+=SORTED_THREADS)owner[i]=-1;
  asc_syncthreads();
  for(uint32_t tile=0;tile<c;tile+=SORTED_THREADS){
    const uint32_t order=tile+t;int32_t slot=order<c?a.lruSlots[cb+order]:-1;
    const bool valid=slot>=0&&static_cast<uint32_t>(slot)<c;
    bool hit=false,invalid=false;
    if(valid){const int32_t token=a.slotToToken[cb+slot];invalid=token>=stable;
      if(token>=0&&token<stable){uint32_t uid=SortedLower(unique,u,token);
        if(uid<u&&unique[uid]==token){hit=true;asc_atomic_max(owner+uid,static_cast<int32_t>((order<<16)|slot));}}}
#if JOINT_LAYOUT16 == 1
    if(order<c)flags[order]=valid?static_cast<uint16_t>(slot|(hit?0x4000U:0U)|(invalid?0x8000U:0U)):uint16_t{65535};
#else
    // Full warp votes, including inactive lanes. One writer owns each bitmap word.
    const uint32_t vh=asc_ballot(hit),vi=asc_ballot(invalid),vv=asc_ballot(valid);
    if((t&31U)==0&&order<c){bits[order/32]=vh;bits[c/32+order/32]=vi;bits[2*c/32+order/32]=vv;}
#endif
  }
  asc_syncthreads();
  uint32_t evicts=0,hits=0,parity=0;
  for(uint32_t tile=0;tile<c;tile+=SORTED_THREADS){
    const uint32_t order=tile+t;bool valid=false,hit=false,invalid=false;uint32_t slot=0;
    if(order<c){
#if JOINT_LAYOUT16 == 1
      uint32_t f=flags[order];valid=f!=65535U;hit=valid&&(f&0x4000U);invalid=valid&&(f&0x8000U);slot=f&0x3fffU;
#else
      const uint32_t mask=1U<<(order&31U);valid=bits[2*c/32+order/32]&mask;
      hit=bits[order/32]&mask;invalid=bits[c/32+order/32]&mask;
      if(valid)slot=static_cast<uint32_t>(a.lruSlots[cb+order]);
#endif
    }
    const int32_t evict=valid&&!hit,totalFlag=evict+(hit?(1U<<COUNT_SHIFT):0U);int32_t total=0;
    const uint32_t rank=SortedScan(scan,totalFlag,t,parity,total);
    const uint16_t entry=static_cast<uint16_t>(slot|(invalid?0x4000U:0U));
    if(evict)part[evicts+(rank&COUNT_MASK)]=entry;
    if(hit)part[c-1-(hits+(rank>>COUNT_SHIFT))]=entry;
    asc_syncthreads();evicts+=static_cast<uint32_t>(total)&COUNT_MASK;hits+=static_cast<uint32_t>(total)>>COUNT_SHIFT;
  }
  if(evicts+hits!=c){if(t==0)ctx[0]=0;return;}
  uint32_t misses=0;parity=0;
  for(uint32_t tile=0;tile<u;tile+=SORTED_THREADS){
    uint32_t uid=tile+t;bool missing=uid<u&&owner[uid]<0;int32_t total=0;
    uint32_t rank=RefineBoolScan(scan,missing,t,parity,total)+misses;
#if JOINT_LAYOUT16 == 1
    if(uid<u)flags[uid]=missing?static_cast<uint16_t>(rank):uint16_t{65535};
#endif
    if(missing&&rank<evicts){uint32_t slot=part[rank]&0x3fffU;
#if JOINT_LAYOUT16 == 1
      owner[uid]=slot;
#else
      // Old owners use bits0..29. Bit30 marks new miss AFTER all hit atomics.
      owner[uid]=static_cast<int32_t>(0x40000000U|(rank<<16)|slot);
#endif
    }
    asc_syncthreads();misses+=total;
  }
  if(misses>evicts){if(t==0)ctx[0]=0;return;}
  // All rejection checks above precede persistent state mutation.
  for(uint32_t i=t;i<c;i+=SORTED_THREADS)if(part[i]&0x4000U)a.slotToToken[cb+(part[i]&0x3fffU)]=-1;
  asc_syncthreads();
  for(uint32_t uid=t;uid<u;uid+=SORTED_THREADS){
#if JOINT_LAYOUT16 == 1
    const bool missing=flags[uid]!=65535U;const uint32_t rank=flags[uid];
#else
    const bool missing=(static_cast<uint32_t>(owner[uid])&0x40000000U)!=0;
    const uint32_t rank=(static_cast<uint32_t>(owner[uid])>>16)&0x3fffU;
#endif
    if(missing){int32_t slot=owner[uid]&0xffff;a.slotToToken[cb+slot]=unique[uid];a.missTokens[qb+rank]=unique[uid];a.missSlots[qb+rank]=slot;}
  }
#if JOINT_LAYOUT16 == 1
  // Same thread reads the exact32-bit word it stored. No cross-warp GM exchange,
  // no Scalar/DMA reader, and no packed16-bit store race or cache handoff.
  for(uint32_t pair=t;pair<q/2;pair+=SORTED_THREADS){
    const uint32_t packed=static_cast<uint32_t>(workspace[qb/2+pair]);
    const uint32_t lo=packed&65535U,hi=packed>>16;
    a.currentSlots[qb+2*pair]=lo==65535U?-1:owner[lo]&0xffff;
    a.currentSlots[qb+2*pair+1]=hi==65535U?-1:owner[hi]&0xffff;
  }
#else
  for(uint32_t i=t;i<q;i+=SORTED_THREADS){uint32_t uid=map[i];a.currentSlots[qb+i]=uid==65535U?-1:owner[uid]&0xffff;}
#endif
  for(uint32_t i=misses+t;i<q;i+=SORTED_THREADS){a.missTokens[qb+i]=-1;a.missSlots[qb+i]=-1;}
  const uint32_t unused=evicts-misses;
  for(uint32_t order=t;order<c;order+=SORTED_THREADS){uint32_t index=order<unused?misses+order:order<evicts?order-unused:c-1-(order-evicts);a.lruSlots[cb+order]=part[index]&0x3fffU;}
  const FastGroupSource src=MakeFastGroupSource(a,req);const uint32_t pages=src.Pages(c);
  for(uint32_t r=0;r<rows;++r){int64_t row=first+r;
    for(uint32_t p=t;p<pages;p+=SORTED_THREADS)a.residentBlockTable[row*pages+p]=first*pages+p;
    if(t==0){a.residentRows[row]=first;int64_t left=static_cast<int64_t>(misses)-r*topk;a.missCount[row]=left<=0?0:left<topk?left:topk;}
    if(t==1)a.rowModes[row]=rows==1?ROW_PRIVATE:(r==0?ROW_JOINT:ROW_FOLLOWER);
  }
  if(t==0){a.lastReqIds[first]=reqid;ctx[0]=2;}asc_syncthreads();
}
