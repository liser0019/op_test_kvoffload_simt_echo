// SPDX-License-Identifier: Apache-2.0
#pragma once
// C reset producer: same UID/slot and protocol, stage LRU and descriptor tails for MTE3.
__simt_callee__ __aicore__ inline void RankPipeCold(const JointPlanArgs& a,
    int64_t first,int64_t end,__ubuf__ uint16_t* map,__ubuf__ int32_t* unique,
    __ubuf__ int32_t* owner,__ubuf__ int32_t* ctx,uint32_t thread){
  constexpr uint32_t Threads=SORTED_THREADS;
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
    owner[slot]=static_cast<int32_t>(slot<unused?u+slot:slot-unused);
    if(slot<u){a.missTokens[qb+slot]=token;a.missSlots[qb+slot]=static_cast<int32_t>(slot);}
  }
  for(uint32_t i=thread;i<q;i+=Threads){
    const uint32_t uid=map[i];a.currentSlots[qb+i]=uid!=65535U?static_cast<int32_t>(uid):-1;
  }
  // Both candidates retain the full original descriptor-tail contract (no D).
  asc_syncthreads(); // finish every unique token consumer before reusing unique as -1 buffer
  for(uint32_t i=thread;i<q-u;i+=Threads)unique[i]=-1;
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
    if(thread==1U)a.rowModes[row]=rows==1U?ROW_PRIVATE:(r==0U?ROW_JOINT:ROW_FOLLOWER);
  }
  if(thread==0U){a.lastReqIds[first]=a.reqIds[req];ctx[46]=q-u;ctx[47]=u;ctx[120]=1;ctx[0]=2;}
  asc_syncthreads(); // finish all writes before this AIV reuses scratch for another group
}
