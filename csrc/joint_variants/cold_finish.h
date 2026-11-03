// SPDX-License-Identifier: Apache-2.0
// SIMT-only reset completion. Caller has captured uniform reset and crossed
// a block barrier BEFORE any mode/request-ID write. Never call on mere all-miss.
// Sorted union and capacity checks remain in the unchanged parent stages.
static_assert(JOINT_OPT_MASK==72 || JOINT_OPT_MASK==90,"cold finish requires frozen AC512 or union base512");
template <uint32_t Threads,bool SingleAllowed,typename P>
__simt_callee__ __aicore__ inline void FinishColdSorted(const JointPlanArgs& a,
    int64_t first,int64_t end,P map,P unique,__ubuf__ int32_t* ctx,uint32_t thread) {
  const uint32_t u=static_cast<uint32_t>(ctx[1]),rows=static_cast<uint32_t>(end-first);
  if(u>a.capacity){if(thread==0U)ctx[0]=0;asc_syncthreads();return;}
#ifdef JOINT_COLD_CPU_MODEL
  if(thread==0U)++ColdCpuCalls; // CPU semantic-test observer; absent from device build.
#endif
  const int64_t qb=first*static_cast<int64_t>(a.topk),cb=first*static_cast<int64_t>(a.capacity);
  const uint32_t unused=a.capacity-u,q=rows*a.topk;
  // Reset has no valid resident/hit. Assign uid directly to identity slot uid.
  // This fuses cache initialization and fill; no intermediate owner/flags/partition.
  for(uint32_t slot=thread;slot<a.capacity;slot+=Threads){
    const int32_t token=slot<u?unique[slot]:-1;
    a.slotToToken[cb+slot]=token;
    a.lruSlots[cb+slot]=static_cast<int32_t>(slot<unused?u+slot:slot-unused);
    if(slot<u){a.missTokens[qb+slot]=token;a.missSlots[qb+slot]=static_cast<int32_t>(slot);}
  }
  for(uint32_t i=thread;i<q;i+=Threads){
    const int32_t uid=map[i];a.currentSlots[qb+i]=uid>=0?uid:-1;
  }
  // Both candidates retain the full original descriptor-tail contract (no D).
  for(uint32_t i=u+thread;i<q;i+=Threads){a.missTokens[qb+i]=-1;a.missSlots[qb+i]=-1;}
  const int32_t req=a.tokenToReq[first];
  const FastGroupSource src=MakeFastGroupSource(a,req);const uint32_t pages=src.Pages(a.capacity);
  for(uint32_t r=0;r<rows;++r){const int64_t row=first+r;
    for(uint32_t page=thread;page<pages;page+=Threads)
      a.residentBlockTable[row*pages+page]=static_cast<int32_t>(first*pages+page);
    if(thread==0U){
      a.residentRows[row]=static_cast<int32_t>(first);
      const int64_t left=static_cast<int64_t>(u)-static_cast<int64_t>(r)*a.topk;
      a.missCount[row]=left<=0?0:left<a.topk?static_cast<int32_t>(left):static_cast<int32_t>(a.topk);
    }
    // Preserve the empirical thread-1 mode-publication workaround. No extra output.
    if(thread==1U)a.rowModes[row]=SingleAllowed&&rows==1U?ROW_PRIVATE:(r==0U?ROW_JOINT:ROW_FOLLOWER);
  }
  if(thread==0U){a.lastReqIds[first]=a.reqIds[req];ctx[0]=2;}
  asc_syncthreads(); // finish all writes before this AIV reuses scratch for another group
}
