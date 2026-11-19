// SPDX-License-Identifier: Apache-2.0
#pragma once
// Rank-pipeline候选：冻结exec ABCD之上，A精确位图、B分块Scan、C批量输出。
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
#if JOINT_MTP3_MAP
  const bool hotUbMap=c==8192U && topk==2048U && end-first==4;
  // This region of firstUid is dead once slotLookup has been built.
  auto* flags=hotUbMap?reinterpret_cast<__ubuf__ uint16_t*>(owner+12288U):map;
#else
  auto* flags=map;
#endif
  const int64_t qb=first*topk,cb=first*c;
  const uint32_t rows=end-first;
  const int32_t req=a.tokenToReq[first];const int64_t reqid=a.reqIds[req];
  uint32_t evicts=Phase==9?ctx[43]:0,hits=Phase==9?ctx[44]:0,misses=Phase==9?ctx[45]:0,parity=0;
  if constexpr(Phase!=9){
  const bool reset=a.rowModes[first]!=(rows==1?ROW_PRIVATE:ROW_JOINT)||a.lastReqIds[first]!=reqid;
  if(reset){asc_syncthreads();
#if (JOINT_RANKPIPE_BITS & 4)
    RankPipeCold(a,first,end,map,unique,owner,ctx,t);
#else
    FinishColdSorted<SORTED_THREADS,true>(a,first,end,EightMap16View{map},unique,ctx,t);
#endif
    return;
  }
  int32_t stable=a.stablePrefixLens[first];
  for(int64_t r=first+1;r<end;++r)if(a.stablePrefixLens[r]<stable)stable=a.stablePrefixLens[r];
  if(stable<0)stable=0;if(stable>a.maxToken)stable=a.maxToken;
  // A's index and slot UID table use the dead merge packet region, at most96KiB.
  const bool directRank=(JOINT_RANKPIPE_BITS&1U) && a.maxToken<=262144;
  auto* slotLookup=directRank?reinterpret_cast<__ubuf__ uint16_t*>(owner+16384U):part;
#if JOINT_LAYOUT16 == 1
#if JOINT_MTP3_MAP
  if(!hotUbMap)
#endif
  for(uint32_t i=t;i<q/2;i+=SORTED_THREADS)workspace[qb/2+i]=data[i];
  asc_syncthreads();
#endif
  if(directRank){
    auto* masks=reinterpret_cast<__ubuf__ uint32_t*>(owner);
    auto* firstUid=reinterpret_cast<__ubuf__ uint32_t*>(owner+8192U);
    // Tokens at or above the stable prefix are invalidated before lookup. In a
    // short-context request they need no bitmap word, so bound the mandatory
    // clear by the actual stable prefix while retaining maxToken as the gate.
    const uint32_t bitmapExtent=JOINT_RANKPIPE_BOUNDED_BITMAP?static_cast<uint32_t>(stable)
                                                           :static_cast<uint32_t>(a.maxToken);
    const uint32_t words=(bitmapExtent+31U)/32U;
    for(uint32_t w=t;w<words;w+=SORTED_THREADS)masks[w]=0U;
    asc_syncthreads(); // zero bitmap before the single-writer heads publish words
    for(uint32_t uid=t;uid<u;uid+=SORTED_THREADS){
      const uint32_t token=static_cast<uint32_t>(unique[uid]),word=token>>5U;
      if(uid==0U || (static_cast<uint32_t>(unique[uid-1U])>>5U)!=word){
        uint32_t value=0U;
        // Sorted unique tokens make each word a contiguous run of at most32 tokens.
        // One head owns BOTH32-bit writes; no atomic OR or halfword RMW required.
        for(uint32_t j=uid;j<u && (static_cast<uint32_t>(unique[j])>>5U)==word;++j)
          value|=uint32_t{1}<<(static_cast<uint32_t>(unique[j])&31U);
        firstUid[word]=uid;masks[word]=value;
      }
    }
    asc_syncthreads();
    for(uint32_t slot=t;slot<c;slot+=SORTED_THREADS){
      const int32_t token=a.slotToToken[cb+slot];uint32_t encoded=token>=stable?0x8000U:0U;
      if(token>=0 && token<stable){
        const uint32_t word=static_cast<uint32_t>(token)>>5U,bit=uint32_t{1}<<(static_cast<uint32_t>(token)&31U);
        const uint32_t mask=masks[word];
        if(mask&bit){const uint32_t uid=firstUid[word]+__popc(mask&(bit-1U));encoded=0x4000U|uid;}
      }
      slotLookup[slot]=static_cast<uint16_t>(encoded);
    }
    asc_syncthreads(); // index is dead before owner initialization reuses its first64KiB
  }else{
  // C: partition is not live yet; cache physical-slot lookup results here.
  // D: owner is not live yet; its first1025 words temporarily hold an exact directory.
  // D: 调整目录粒度；仍是精确二分，不改变查找语义或 UB 大小。
  const uint32_t directoryBins=(JOINT_STAGE_EXEC_BITS&8U) && c==4096U ? 512U : 1024U;
  uint32_t bucketShift=0;
#if (JOINT_LAYOUT_HOT_BITS & 4)
#if (JOINT_LAYOUT_HOT_BITS & 8)
  while((static_cast<uint64_t>(directoryBins)<<bucketShift)<static_cast<uint32_t>(a.maxToken))++bucketShift;
  for(uint32_t i=t;i<=directoryBins;i+=SORTED_THREADS)owner[i]=SortedLower(unique,u,static_cast<int32_t>(i<<bucketShift));
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
  }
  for(uint32_t i=t;i<u;i+=SORTED_THREADS)owner[i]=-1;
  asc_syncthreads();
  for(uint32_t tile=0;tile<c;tile+=SORTED_THREADS){
    const uint32_t order=tile+t;int32_t slot=order<c?a.lruSlots[cb+order]:-1;
    const bool valid=slot>=0&&static_cast<uint32_t>(slot)<c;
#if (JOINT_STAGE_EXEC_BITS & 2)
    // B 的位置推导依赖每个 LRU 位置均为合法槽。沿用现有拒绝语义，
    // 在已有分类及其末尾同步中提前发布失败；不读取未填满的 partition。
    if(order<c && !valid)asc_atomic_min(ctx,0);
#endif
    bool hit=false,invalid=false;
    if(valid){
#if (JOINT_LAYOUT_HOT_BITS & 4)
      const uint32_t encoded=slotLookup[slot];invalid=(encoded&0x8000U)!=0;hit=(encoded&0x4000U)!=0;
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
#if !(JOINT_STAGE_EXEC_BITS & 4)
  for(uint32_t i=t;i<c;i+=SORTED_THREADS)if(part[i]&0x4000U)a.slotToToken[cb+(part[i]&0x3fffU)]=-1;
  asc_syncthreads();
#endif
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
#if JOINT_MTP3_MAP
    const uint32_t packed=hotUbMap?reinterpret_cast<__ubuf__ uint32_t*>(data)[pair]
                                  :static_cast<uint32_t>(workspace[qb/2+pair]);
#else
    const uint32_t packed=static_cast<uint32_t>(workspace[qb/2+pair]);
#endif
    const uint32_t lo=packed&65535U,hi=packed>>16;
    a.currentSlots[qb+2*pair]=lo==65535U?-1:owner[lo]&0xffff;
    a.currentSlots[qb+2*pair+1]=hi==65535U?-1:owner[hi]&0xffff;
  }
#else
  for(uint32_t i=t;i<q;i+=SORTED_THREADS){uint32_t uid=map[i];a.currentSlots[qb+i]=uid==65535U?-1:owner[uid]&0xffff;}
#endif
#if !(JOINT_RANKPIPE_BITS & 4)
  for(uint32_t i=misses+t;i<q;i+=SORTED_THREADS){a.missTokens[qb+i]=-1;a.missSlots[qb+i]=-1;}
#else
  // All readers of owner/unique have finished before output staging overwrites them.
  asc_syncthreads();
  for(uint32_t i=t;i<q-misses;i+=SORTED_THREADS)unique[i]=-1;
#endif
  const uint32_t unused=evicts-misses;
  for(uint32_t order=t;order<c;order+=SORTED_THREADS){
    const uint32_t index=order<unused?misses+order:order<evicts?order-unused:c-1-(order-evicts);
    const uint32_t entry=part[index],slot=entry&0x3fffU;
#if (JOINT_STAGE_EXEC_BITS & 4)
    // C: part[0..misses) 已分给新 token，直接覆盖即可。其余槽与 miss 槽
    // 不相交，在写 LRU 的同次遍历中失效；依赖原有 LRU 为槽排列的契约。
    if(index>=misses && (entry&0x4000U))a.slotToToken[cb+slot]=-1;
#endif
#if (JOINT_RANKPIPE_BITS & 4)
    owner[order]=static_cast<int32_t>(slot);
#else
    a.lruSlots[cb+order]=slot;
#endif
  }
  const FastGroupSource src=MakeFastGroupSource(a,req);const uint32_t pages=src.Pages(c);
  for(uint32_t r=0;r<rows;++r){int64_t row=first+r;
    for(uint32_t p=t;p<pages;p+=SORTED_THREADS)a.residentBlockTable[row*pages+p]=first*pages+p;
    if(t==0){a.residentRows[row]=first;int64_t left=static_cast<int64_t>(misses)-r*topk;a.missCount[row]=left<=0?0:left<topk?left:topk;}
    if(t==1)a.rowModes[row]=rows==1?ROW_PRIVATE:(r==0?ROW_JOINT:ROW_FOLLOWER);
  }
  if(t==0){
#if (JOINT_RANKPIPE_BITS & 4)
    ctx[46]=q-misses;ctx[47]=misses;ctx[120]=1;
#endif
    a.lastReqIds[first]=reqid;ctx[0]=2;
  }asc_syncthreads();
}
