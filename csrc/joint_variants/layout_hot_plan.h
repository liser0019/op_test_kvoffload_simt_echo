// SPDX-License-Identifier: Apache-2.0
#pragma once
// A/B/C/D候选收尾。512线程；保留32位owner的LRU-last原子仲裁。
// 1: map暂存GM，映射区复用uint16 flags；2: 全UB三位图。
// B: stage由独立SIMD VF完成压缩；全UB直接排名位图字；其余沿用Scan。
template<int Phase>
__simt_callee__ __aicore__ inline void LayoutHotFinish(const JointPlanArgs& a,
    int64_t first,int64_t end,__ubuf__ int32_t* data,__ubuf__ int32_t* ctx,
    __ubuf__ int32_t* scan,__gm__ int32_t* workspace) {
  const uint32_t t=threadIdx.x,c=a.capacity,topk=a.topk,q=(end-first)*topk,u=ctx[1];
  auto* map=reinterpret_cast<__ubuf__ uint16_t*>(data);
  auto* unique=data+c/2U;
  auto* owner=unique+c;
  auto* part=reinterpret_cast<__ubuf__ uint16_t*>(owner+c);
  auto* bits=reinterpret_cast<__ubuf__ uint32_t*>(part+c);
  auto* flags=map;
  const int64_t qb=first*topk,cb=first*c;
  const uint32_t rows=end-first;
  const int32_t req=a.tokenToReq[first];const int64_t reqid=a.reqIds[req];
  uint32_t evicts=Phase==9?ctx[43]:0,hits=Phase==9?ctx[44]:0,misses=Phase==9?ctx[45]:0,parity=0;
  if constexpr(Phase!=9){
  const bool reset=a.rowModes[first]!=(rows==1?ROW_PRIVATE:ROW_JOINT)||a.lastReqIds[first]!=reqid;
  if(reset){asc_syncthreads();FinishColdSorted<SORTED_THREADS,true>(a,first,end,EightMap16View{map},unique,ctx,t);return;}
  int32_t stable=a.stablePrefixLens[first];
  for(int64_t r=first+1;r<end;++r)if(a.stablePrefixLens[r]<stable)stable=a.stablePrefixLens[r];
  if(stable<0)stable=0;if(stable>a.maxToken)stable=a.maxToken;
#if JOINT_LAYOUT16 == 1
  for(uint32_t i=t;i<q/2;i+=SORTED_THREADS)workspace[qb/2+i]=data[i];
  asc_syncthreads();
#endif
  // C: partition is not live yet; cache physical-slot lookup results here.
  // D: owner is not live yet; its first1025 words temporarily hold an exact directory.
  uint32_t bucketShift=0;
#if (JOINT_LAYOUT_HOT_BITS & 4)
#if (JOINT_LAYOUT_HOT_BITS & 8)
  while((uint64_t{1024}<<bucketShift)<static_cast<uint32_t>(a.maxToken))++bucketShift;
  for(uint32_t i=t;i<=1024U;i+=SORTED_THREADS)owner[i]=SortedLower(unique,u,static_cast<int32_t>(i<<bucketShift));
  asc_syncthreads();
#endif
  for(uint32_t slot=t;slot<c;slot+=SORTED_THREADS){
    const int32_t token=a.slotToToken[cb+slot];uint32_t encoded=token>=stable?0x8000U:0U;
    if(token>=0&&token<stable){uint32_t pos;
#if (JOINT_LAYOUT_HOT_BITS & 8)
      const uint32_t bucket=static_cast<uint32_t>(token)>>bucketShift;
      const uint32_t lo=owner[bucket],hi=owner[bucket+1U];
      pos=lo+SortedLower(unique+lo,hi-lo,token);
#else
      pos=SortedLower(unique,u,token);
#endif
      if(pos<u&&unique[pos]==token)encoded=0x4000U|pos;
    }
    part[slot]=static_cast<uint16_t>(encoded);
  }
  asc_syncthreads(); // finish ALL directory reads before owner init
#endif
  for(uint32_t i=t;i<u;i+=SORTED_THREADS)owner[i]=-1;
  asc_syncthreads();
  for(uint32_t tile=0;tile<c;tile+=SORTED_THREADS){
    const uint32_t order=tile+t;int32_t slot=order<c?a.lruSlots[cb+order]:-1;
    const bool valid=slot>=0&&static_cast<uint32_t>(slot)<c;
    bool hit=false,invalid=false;
    if(valid){
#if (JOINT_LAYOUT_HOT_BITS & 4)
      const uint32_t encoded=part[slot];invalid=(encoded&0x8000U)!=0;hit=(encoded&0x4000U)!=0;
      if(hit)asc_atomic_max(owner+(encoded&0x3fffU),static_cast<int32_t>((order<<16)|slot));
#else
      const int32_t token=a.slotToToken[cb+slot];invalid=token>=stable;
      if(token>=0&&token<stable){const uint32_t uid=SortedLower(unique,u,token);
        if(uid<u&&unique[uid]==token){hit=true;asc_atomic_max(owner+uid,static_cast<int32_t>((order<<16)|slot));}}
#endif
    }
#if JOINT_LAYOUT16 == 1
    if(order<c)flags[order]=valid?static_cast<uint16_t>(slot|(hit?0x4000U:0U)|(invalid?0x8000U:0U)):uint16_t{65535};
#else
    // Full warp votes, including inactive lanes. One writer owns each bitmap word.
    const uint32_t vh=asc_ballot(hit),vi=asc_ballot(invalid),vv=asc_ballot(valid);
    if((t&31U)==0&&order<c){bits[order/32]=vh;bits[c/32+order/32]=vi;bits[2*c/32+order/32]=vv;}
#endif
  }
  asc_syncthreads();
  if constexpr(Phase==6)return;
#if JOINT_LAYOUT16 == 2 && (JOINT_LAYOUT_HOT_BITS & 2)
  // B: scan512 word counts rather than32 serial512-slot block scans.
  auto* prefixes=bits+3U*c/32U;uint32_t carry=0;
  for(uint32_t base=0;base<c/32U;base+=SORTED_THREADS){
    uint32_t seg=base+t,value=0;if(seg<c/32U){const uint32_t valid=bits[2U*c/32U+seg],hit=bits[seg];value=__popc(valid&~hit)+(__popc(valid&hit)<<16U);}
    int32_t total=0;uint32_t rank=SortedScan(scan,static_cast<int32_t>(value),t,parity,total);
    if(seg<c/32U)prefixes[seg]=carry+rank;asc_syncthreads();carry+=static_cast<uint32_t>(total);
  }
  evicts=carry&65535U;hits=carry>>16U;
  for(uint32_t order=t;order<c;order+=SORTED_THREADS){
    const uint32_t seg=order/32U,mask=1U<<(order&31U),before=mask-1U;
    const uint32_t valid=bits[2U*c/32U+seg],hit=bits[seg]&valid,evict=valid&~hit,prefix=prefixes[seg];
    if(valid&mask){const uint16_t entry=static_cast<uint16_t>(a.lruSlots[cb+order]|((bits[c/32U+seg]&mask)?0x4000U:0U));
      if(hit&mask)part[c-1U-((prefix>>16U)+__popc(hit&before))]=entry;
      else part[(prefix&65535U)+__popc(evict&before)]=entry;}
  }
  asc_syncthreads();
#else
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
#endif
  if(evicts+hits!=c){if(t==0)ctx[0]=0;return;}
  parity=0;
#if JOINT_LAYOUT16 == 2 && (JOINT_LAYOUT_HOT_BITS & 2)
  for(uint32_t base=0;base<u;base+=SORTED_THREADS){const uint32_t uid=base+t;const uint32_t votes=asc_ballot(uid<u&&owner[uid]<0);
    if((t&31U)==0U&&uid<u)bits[uid/32U]=votes;}
  asc_syncthreads();uint32_t carryMiss=0,segments=(u+31U)/32U;
  for(uint32_t base=0;base<segments;base+=SORTED_THREADS){uint32_t seg=base+t;int32_t total=0;
    uint32_t rank=SortedScan(scan,seg<segments?__popc(bits[seg]):0,t,parity,total);
    if(seg<segments)prefixes[seg]=carryMiss+rank;asc_syncthreads();carryMiss+=static_cast<uint32_t>(total);}
  misses=carryMiss;
  for(uint32_t uid=t;uid<u;uid+=SORTED_THREADS){uint32_t mask=1U<<(uid&31U),votes=bits[uid/32U];
    if(votes&mask){uint32_t rank=prefixes[uid/32U]+__popc(votes&(mask-1U));if(rank<evicts)owner[uid]=static_cast<int32_t>(0x40000000U|(rank<<16U)|(part[rank]&0x3fffU));}}
  asc_syncthreads();
#else
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
#endif
  if(misses>evicts){if(t==0)ctx[0]=0;return;}
  } // Phase9 enters here after SIMD completion and status checks
  // All rejection checks above precede persistent state mutation.
  for(uint32_t i=t;i<c;i+=SORTED_THREADS)if(part[i]&0x4000U)a.slotToToken[cb+(part[i]&0x3fffU)]=-1;
  asc_syncthreads();
  for(uint32_t uid=t;uid<u;uid+=SORTED_THREADS){
#if JOINT_LAYOUT16 == 1 && !(JOINT_LAYOUT_HOT_BITS & 2)
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
