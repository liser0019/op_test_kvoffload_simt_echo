// SPDX-License-Identifier: Apache-2.0
// 新BD的8行候选：512线程，四路分组→两路前缀归并；冻结BD源码不改。
#pragma once
#ifndef JOINT_MTP7_OPT
#define JOINT_MTP7_OPT 0
#endif
#ifndef JOINT_MTP7_REUSE
#define JOINT_MTP7_REUSE 0
#endif
#ifndef JOINT_MTP7_TILE
#define JOINT_MTP7_TILE 0
#endif
static_assert(JOINT_MTP7_OPT>=0 && JOINT_MTP7_OPT<=2,"MTP7 candidate selector");
#if JOINT_MTP7_CAPACITY8K
// Capacity8192 adapter only. No algorithm or thread-layout change.
constexpr uint64_t EIGHT_UB_WORDS=196U*1024U/4U;
constexpr uint32_t EIGHT_CTX=128U;
static_assert(196U*1024U+288U+8U*1024U+32U*1024U<=256U*1024U,"reserve compiler area and minimum DCache");
#else
constexpr uint64_t EIGHT_UB_WORDS=192U*1024U/4U;
constexpr uint32_t EIGHT_CTX=128U;
static_assert(192U*1024U+288U+8U*1024U+32U*1024U<=256U*1024U,"reserve compiler area and minimum DCache");
#endif
static_assert(JOINT_HOT_BITS>=0 && JOINT_HOT_BITS<32,"hot factor bits");
constexpr bool HOT_FILTER=(JOINT_HOT_BITS&1U)!=0;
constexpr bool HOT_PART=(JOINT_HOT_BITS&2U)!=0 || JOINT_HOT_PART_ONLY;
constexpr bool HOT_MISS=(JOINT_HOT_BITS&2U)!=0 || JOINT_HOT_MISS_ONLY;
constexpr bool HOT_PACKET=(JOINT_HOT_BITS&4U)!=0;
constexpr bool HOT_REUSE=(JOINT_HOT_BITS&8U)!=0;
constexpr bool HOT_FIXED=(JOINT_HOT_BITS&16U)!=0;
static_assert(REFINE_A && REFINE_B && REFINE_D && !REFINE_C,"frozen ABD parent");

// Hardware sweep: A=control fast paths, B=32-bin resident index,
// C=group-sized map/adaptive packets, D=retain unconsumed UB heads,
// E=256-thread union/planner VFs. Original 1024-thread private fallback retained.
constexpr uint32_t HW_BITS = JOINT_OPT_MASK <= 69 ? JOINT_OPT_MASK-38U :
    JOINT_OPT_MASK==70 ? 15U : JOINT_OPT_MASK==71 ? 14U : JOINT_OPT_MASK==72 ? 5U : 4U;
constexpr bool HW_A=(HW_BITS&1U)!=0, HW_B=(HW_BITS&2U)!=0;
constexpr bool HW_C=(HW_BITS&4U)!=0, HW_D=(HW_BITS&8U)!=0;
#if JOINT_MTP7_REUSE
static_assert(!HW_D,"reused input cannot retain old packet tails");
#endif
constexpr uint32_t HW_DIV = JOINT_OPT_MASK>=70 ? 2U : (HW_BITS&16U)!=0 ? 4U : 1U;
constexpr uint32_t SORTED_THREADS = PLAN_THREADS/HW_DIV >= PLAN_WARP_SIZE ? PLAN_THREADS/HW_DIV : PLAN_WARP_SIZE;
constexpr uint32_t SORTED_WARPS = SORTED_THREADS/PLAN_WARP_SIZE;
static_assert(SORTED_THREADS<=PLAN_THREADS && (SORTED_WARPS&(SORTED_WARPS-1U))==0U,"warp layout");
// Uses existing (oversized) Scan storage. Warp count follows THIS VF's launch.
__simt_callee__ __aicore__ inline int32_t SortedScan(__ubuf__ int32_t* work,int32_t value,
    uint32_t thread,uint32_t& parity,int32_t& total){
  __ubuf__ int32_t* totals=work+(parity?WARP_BASES_OFFSET:WARP_TOTALS_OFFSET);parity^=1U;
  uint32_t lane=thread&(PLAN_WARP_SIZE-1U),warp=thread/PLAN_WARP_SIZE;
  int32_t inclusive=value;
#pragma unroll
  for(uint32_t d=1;d<PLAN_WARP_SIZE;d<<=1U){int32_t x=asc_shfl_up(inclusive,d,PLAN_WARP_SIZE);if(lane>=d)inclusive+=x;}
  if(lane==PLAN_WARP_SIZE-1U)totals[warp]=inclusive;asc_syncthreads();
  int32_t own=lane<SORTED_WARPS?totals[lane]:0,sum=own;
#pragma unroll
  for(uint32_t d=1;d<SORTED_WARPS;d<<=1U){int32_t x=asc_shfl_up(sum,d,PLAN_WARP_SIZE);if(lane>=d)sum+=x;}
  total=asc_shfl(sum,SORTED_WARPS-1U,PLAN_WARP_SIZE);
  return asc_shfl(sum-own,warp,PLAN_WARP_SIZE)+inclusive-value;
}
// B：只替换布尔 Scan 的 warp 内前缀；跨 warp 汇总、parity 与同步沿用参考版。
// Every logical tail lane participates with false; do not put ballot in i<n.
__simt_callee__ __aicore__ inline int32_t RefineBoolScan(__ubuf__ int32_t* work,bool flag,
    uint32_t thread,uint32_t& parity,int32_t& total){
  if constexpr(!REFINE_B)return SortedScan(work,flag?1:0,thread,parity,total);
  const uint32_t lane=thread&(PLAN_WARP_SIZE-1U),warp=thread/PLAN_WARP_SIZE;
  const uint32_t votes=asc_ballot(flag?1:0),before=(uint32_t{1}<<lane)-1U;
  const int32_t ownRank=__popc(votes&before);
  __ubuf__ int32_t* totals=work+(parity?WARP_BASES_OFFSET:WARP_TOTALS_OFFSET);parity^=1U;
  if(lane==PLAN_WARP_SIZE-1U)totals[warp]=__popc(votes);
  asc_syncthreads();
  int32_t own=lane<SORTED_WARPS?totals[lane]:0,sum=own;
#pragma unroll
  for(uint32_t d=1;d<SORTED_WARPS;d<<=1U){int32_t x=asc_shfl_up(sum,d,PLAN_WARP_SIZE);if(lane>=d)sum+=x;}
  total=asc_shfl(sum,SORTED_WARPS-1U,PLAN_WARP_SIZE);
  return asc_shfl(sum-own,warp,PLAN_WARP_SIZE)+ownRank;
}
#if JOINT_MTP7_MAP16
// Compress only supported eight-row UB groups.0xffff is the invalid UID.
__simt_callee__ __aicore__ inline bool EightMapIsPacked(uint32_t rows, uint32_t topk) {
  return rows == 8U && static_cast<uint64_t>(rows) * topk <= 65408U;
}
__aicore__ inline uint32_t EightMapWords(uint32_t queries, uint32_t topk) {
  return queries / topk == 8U && queries <= 65408U ? (queries + 1U) / 2U : queries;
}
struct EightMap16View {
  __ubuf__ uint16_t* values;
  __simt_callee__ __aicore__ inline int32_t operator[](uint32_t i) const {
    const uint16_t value = values[i];
    return value == 65535U ? -1 : static_cast<int32_t>(value);
  }
};
#define EIGHT_MAP_PARAMETER , uint32_t mapWords
#define EIGHT_MAP_ARGUMENT(q, t) , EightMapWords(q, static_cast<uint32_t>(t))
#else
#define EIGHT_MAP_PARAMETER
#define EIGHT_MAP_ARGUMENT(q, t)
#endif

// Hash-free sorted union planner. Helpers enter only through SortedStageVf.
// Lifetime: ctx[128] | queryToUid[8T] | unique[C] | work[max(3C,64)].
// work is packet input/output during union construction, then owner/flags/LRU.
// GM scratch keeps packet input/output in the separate UB staging area.
template <typename P>
__simt_callee__ __aicore__ inline uint32_t SortedLower(P p, uint32_t n, int32_t token) {
  uint32_t lo=0,hi=n;
  while(lo<hi){uint32_t mid=lo+((hi-lo)>>1U);if(p[mid]<token)lo=mid+1U;else hi=mid;}
  return lo;
}

template <int Phase, bool Fixed, typename P>
__simt_callee__ __aicore__ inline void SortedStage(const JointPlanArgs& a,
    int64_t first,int64_t end,P data,__ubuf__ int32_t* ctx,__ubuf__ int32_t* packet,
    __ubuf__ int32_t* scan) {
  const uint32_t topk=Fixed?2048U:a.topk,capacity=Fixed?4096U:a.capacity;
  const uint32_t t=static_cast<uint32_t>(threadIdx.x), rows=static_cast<uint32_t>(end-first);
  const uint32_t mapRows=HW_C&&rows<=8U?rows:8U;
#if JOINT_MTP7_MAP16
  const bool packedMap = EightMapIsPacked(rows, topk);
  const uint32_t mapWords = packedMap ? (mapRows * topk + 1U) / 2U : mapRows * topk;
  P map=data, unique=data+mapWords, owner=data+((mapWords+capacity+7U)&~7U);
  auto* map16 = reinterpret_cast<__ubuf__ uint16_t*>(ctx + EIGHT_CTX);
#else
  P map=data, unique=map+mapRows*topk, owner=data+((mapRows*topk+capacity+7U)&~7U);
#endif
  P flags=owner+capacity, partition=flags+capacity;
  const uint32_t q=rows*topk;
  const int64_t qb=first*static_cast<int64_t>(topk),cb=first*static_cast<int64_t>(capacity);
  if constexpr(Phase==0) {
    if(t==0U){
      ctx[0]=1;ctx[1]=0;ctx[2]=-1;ctx[4]=0;ctx[6]=0;ctx[7]=0;ctx[28]=0;ctx[29]=0;
      uint32_t chunk=(3U*capacity>64U?3U*capacity:64U)/16U;
      if(chunk>768U)chunk=768U;ctx[3]=static_cast<int32_t>(chunk&~3U);
      uint64_t words=16ULL*static_cast<uint32_t>(ctx[3]);
      if(HW_C && a.scratchElements<=EIGHT_UB_WORDS)
#if JOINT_MTP7_MAP16
        words=a.scratchElements-EIGHT_CTX-((mapWords+capacity+7U)&~7U);
#else
        words=a.scratchElements-EIGHT_CTX-((mapRows*topk+capacity+7U)&~7U);
#endif
      ctx[30]=static_cast<int32_t>((words/2U)&~uint64_t{7U}); // fixed input/output split

      const int32_t req=a.tokenToReq[first];
      uint32_t bits=1;while(bits<31U&&(1ULL<<bits)<q)++bits;
      if(rows<1U||rows>8U||(rows&(rows-1U))!=0U||capacity>32768U||q>65535U||a.maxToken<=0||a.maxToken>16777216||
         (static_cast<uint64_t>(a.maxToken-1)<<bits)>UINT32_MAX||req<0||req>=a.maxRequests)ctx[0]=0;
      for(uint32_t r=0;r<8U;++r){ctx[48+r]=0;ctx[56+r]=0;}
      if(ctx[0])for(uint32_t r=0;r<rows;++r){
        const int64_t b=(first+r)*static_cast<int64_t>(topk);
        int32_t upper=a.visibleSeqLens[first+r];if(upper<0)upper=0;if(upper>a.maxToken)upper=a.maxToken;
        // Only Phase0 uses endpoint checks. SortedLower for resident lookup is unchanged.
        // upper remains row-specific and clipped; equality at upper is NOT visible.
        if(topk>0U && a.topkIndices[b]>=0 && a.topkIndices[b+topk-1U]<upper){
          ctx[48+r]=0;ctx[56+r]=static_cast<int32_t>(topk);
        }else{
          ctx[48+r]=static_cast<int32_t>(SortedLower(a.topkIndices+b,topk,0));
          ctx[56+r]=static_cast<int32_t>(SortedLower(a.topkIndices+b,topk,upper));
        }
        if(ctx[48+r]<ctx[56+r])ctx[6]=1;
      }
      if constexpr(REFINE_A){
        // A：严格递增单行的 UID 就是有效区间内的相对位置，无需打包或归并。
        // Duplicate detection below still vetoes this entire attempt before commit.
        if(ctx[0] && rows==1U){ctx[1]=ctx[56]-ctx[48];ctx[6]=0;}
      }
    }
    asc_syncthreads();
    // Reject unsupported group widths before touching the 8T-sized map.
    if(ctx[0])for(uint32_t i=t;i<q;i+=SORTED_THREADS){
      // Independent semantics count duplicate query misses separately. Fall back;
      // never silently share their slot through the single-row unique list.
      if constexpr(REFINE_A){
        if(rows==1U){
          const int32_t token=a.topkIndices[qb+i];
          if(i && token==a.topkIndices[qb+i-1U])asc_atomic_min(ctx,0);
          const bool valid=i>=static_cast<uint32_t>(ctx[48]) && i<static_cast<uint32_t>(ctx[56]);
          map[i]=valid?static_cast<int32_t>(i-static_cast<uint32_t>(ctx[48])):-1;
          if(valid)unique[i-static_cast<uint32_t>(ctx[48])]=token;
#if JOINT_MTP7_MAP16
        }else {
          // One writer per32-bit word, including odd TopK shapes.
          if (packedMap) { if (i < mapWords) map[i] = -1; }
          else map[i] = -1;
        }
#else
        }else map[i]=-1;
#endif
      }else{
        if(rows==1U && i && a.topkIndices[qb+i]==a.topkIndices[qb+i-1U])asc_atomic_min(ctx,0);
#if JOINT_MTP7_MAP16
        if (packedMap) { if (i < mapWords) map[i] = -1; }
        else map[i] = -1;
#else
        map[i]=-1;
#endif
      }
    }
    asc_syncthreads();
  } else if constexpr(Phase==1) {
    const uint32_t oldChunk=ctx[3],oldActive=ctx[28],outOffset=ctx[30];
    // Output from the previous merge has been consumed; use it for nonoverlap
    // tail staging. Input and output regions have a fixed split for this group.
    if constexpr(HW_D){
      if(ctx[29])for(uint32_t z=t;z<oldActive*oldChunk;z+=SORTED_THREADS){
        uint32_t k=z/oldChunk,j=z%oldChunk;
        uint32_t used=ctx[80+k],left=static_cast<uint32_t>(ctx[72+k])-used;
        if(j<left){uint32_t src=k*oldChunk+used+j,dst=k*oldChunk+j;
          packet[outOffset+2U*dst]=packet[2U*src];packet[outOffset+2U*dst+1U]=packet[2U*src+1U];}
      }
      asc_syncthreads();
    }
    if(t==0U){
      for(uint32_t r=0;r<8U;++r){ctx[88+r]=0;ctx[96+r]=0;}
      if(HW_D&&ctx[29])for(uint32_t k=0;k<oldActive;++k){
        uint32_t r=ctx[64+k];ctx[88+r]=ctx[72+k]-ctx[80+k];ctx[96+r]=k*oldChunk;
      }
      uint32_t active=0;
      for(uint32_t r=0;r<rows;++r)if(ctx[48+r]<ctx[56+r])ctx[64+active++]=r;
      uint32_t chunk=oldChunk;
#if JOINT_MTP7_REUSE
      if(rows>4U){
        // Same total allocation. Release first-half inputs after its merge;
        // work now needs40*chunk int32s rather than48*chunk.
#if JOINT_MTP7_MAP16
        const uint64_t work = a.scratchElements - EIGHT_CTX - ((mapWords + capacity + 7U) & ~uint64_t{7U});
        const uint32_t maximum = packedMap ? 816U : 612U;
        // Eight-record alignment makes the full eight-row output a multiple of64.
        chunk = static_cast<uint32_t>(work / 40U) & (packedMap ? ~7U : ~3U);
        if (chunk > maximum) chunk = maximum;
#else
        const uint64_t work=a.scratchElements-EIGHT_CTX-((mapRows*topk+capacity+7U)&~uint64_t{7U});
        chunk=static_cast<uint32_t>(work/40U)&~3U;if(chunk>612U)chunk=612U;
#endif
        ctx[30]=24U*chunk;
      }else if(HW_C&&active){chunk=(static_cast<uint32_t>(ctx[30])/(2U*active))&~3U;if(chunk>4092U)chunk=4092U;}
#else
      if(HW_C&&active){chunk=(static_cast<uint32_t>(ctx[30])/(2U*active))&~3U;if(chunk>4092U)chunk=4092U; if(rows>4U && chunk>512U)chunk=512U;}
      if(rows>4U)ctx[30]=32U*chunk; // input8, intermediate8, final8 records per chunk
#endif
      ctx[3]=chunk;ctx[28]=active;ctx[7]=0;
#if JOINT_MTP7_MID_UNIQUE || JOINT_MTP7_FUSED_COUNT
      ctx[107]=0; // deferred original-query consumption only for the eight-row two-level path
#endif
      uint32_t total=0;
      for(uint32_t k=0;k<active;++k){uint32_t r=ctx[64+k],n=ctx[56+r]-ctx[48+r];
        if(n>chunk)n=chunk;ctx[72+k]=n;ctx[80+k]=0;total+=n;}
#if JOINT_MTP7_PREFIX
      if (rows == 8U && active > 0U) {
        // Anything beyond the minimum tail of an incomplete source could
        // precede an unseen value. Only consume the safe inclusive frontier.
        int32_t frontier = a.maxToken;
        for (uint32_t k = 0; k < active; ++k) {
          const uint32_t row = ctx[64U + k], n = ctx[72U + k];
          const uint32_t begin = ctx[48U + row];
          if (n < static_cast<uint32_t>(ctx[56U + row]) - begin) {
            const int32_t tail = a.topkIndices[qb + row * topk + begin + n - 1U];
            if (tail < frontier) frontier = tail;
          }
        }
        uint32_t selected = 0U;
        total = 0U;
        for (uint32_t k = 0; k < active; ++k) {
          const uint32_t row = ctx[64U + k];
          const uint32_t begin = ctx[48U + row];
          const uint32_t n = SortedLower(a.topkIndices + qb + row * topk + begin,
                                          static_cast<uint32_t>(ctx[72U + k]), frontier + 1);
          if (n == 0U) continue; // never pass a zero-length source to MrgSort
          ctx[64U + selected] = static_cast<int32_t>(row);
          ctx[72U + selected] = static_cast<int32_t>(n);
          ctx[80U + selected] = static_cast<int32_t>(n);
          ++selected;
          total += n;
        }
        ctx[28] = static_cast<int32_t>(selected);
        if (selected == 0U) ctx[0] = 0;
      }
#endif
      ctx[4]=total;ctx[5]=total;
    }
    asc_syncthreads();
    const uint32_t chunk=ctx[3],active=ctx[28];
    auto* score=reinterpret_cast<__ubuf__ float*>(packet);
    for(uint32_t i=t;i<active*chunk;i+=SORTED_THREADS){uint32_t k=i/chunk,j=i%chunk;
      if(j<static_cast<uint32_t>(ctx[72+k])){uint32_t r=ctx[64+k],pos=static_cast<uint32_t>(ctx[48+r])+j;
        if(HW_D&&j<static_cast<uint32_t>(ctx[88+r])){uint32_t saved=ctx[96+r]+j;
          packet[2U*i]=packet[outOffset+2U*saved];packet[2U*i+1U]=packet[outOffset+2U*saved+1U];
        }else{score[2U*i]=-static_cast<float>(a.topkIndices[qb+r*topk+pos]);packet[2U*i+1U]=r*topk+pos;}
      }
    }
    asc_syncthreads();
  } else if constexpr(Phase==2) {
    // Software packet merge is a controlled comparison for hardware MrgSort.
    const uint32_t chunk=ctx[3],active=ctx[28];
    auto* score=reinterpret_cast<__ubuf__ float*>(packet);
    auto* out=packet+static_cast<uint32_t>(ctx[30]);
    for(uint32_t z=t;z<active*chunk;z+=SORTED_THREADS){
      const uint32_t k=z/chunk,j=z%chunk,n=ctx[72+k];if(j>=n)continue;
      const float key=score[2U*z];const int32_t id=packet[2U*z+1U];uint32_t rank=j;
      for(uint32_t s=0;s<active;++s)if(s!=k){
        uint32_t lo=0,hi=ctx[72+s];
        while(lo<hi){uint32_t mid=lo+((hi-lo)>>1U),p=s*chunk+mid;
          const float other=score[2U*p];
          if(other>key||(other==key&&packet[2U*p+1U]<id))lo=mid+1U;else hi=mid;
        }rank+=lo;
      }
      out[2U*rank]=packet[2U*z];out[2U*rank+1U]=id;
      if(j+1U==n)asc_atomic_min(ctx+5,static_cast<int32_t>(rank+1U));
    }
    asc_syncthreads();
    if(t==0U){
      for(uint32_t i=0;i<static_cast<uint32_t>(ctx[5]);++i){
        uint32_t row=static_cast<uint32_t>(out[2U*i+1U])/topk;
        for(uint32_t k=0;k<active;++k)if(static_cast<uint32_t>(ctx[64+k])==row)++ctx[80+k];
      }
      for(uint32_t k=0;k<active;++k)ctx[48+ctx[64+k]]+=ctx[80+k];
      ctx[4]=ctx[5];
    }
    asc_syncthreads();
  } else if constexpr(Phase==3) {
    const uint32_t n=ctx[4],chunk=ctx[3];auto* out=packet+(ctx[7]?0U:static_cast<uint32_t>(ctx[30]));
    auto* scores=reinterpret_cast<__ubuf__ float*>(out);
    uint32_t count=ctx[1],parity=0;
    for(uint32_t tile=0;tile<n;tile+=SORTED_THREADS){
      const uint32_t i=tile+t;int32_t token=-1,prev=-1;bool added=false;
      if(i<n){token=static_cast<int32_t>(-scores[2U*i]);
        prev=i==0U?ctx[2]:static_cast<int32_t>(-scores[2U*(i-1U)]);added=token!=prev;}
      int32_t total=0;const int32_t rank=RefineBoolScan(scan,added,t,parity,total);
      if(i<n){uint32_t uid=count+static_cast<uint32_t>(rank)-(added?0U:1U);
        if(uid<capacity&&added)unique[uid]=token;
#if JOINT_MTP7_MAP16
        if (packedMap) map16[static_cast<uint32_t>(out[2U*i+1U])] = static_cast<uint16_t>(uid);
        else map[static_cast<uint32_t>(out[2U*i+1U])] = static_cast<int32_t>(uid);
#else
        map[static_cast<uint32_t>(out[2U*i+1U])]=static_cast<int32_t>(uid);
#endif
      }
      asc_syncthreads();count+=static_cast<uint32_t>(total);
    }
    if(t==0U){ctx[1]=count;if(n)ctx[2]=static_cast<int32_t>(-scores[2U*(n-1U)]);if(count>capacity)ctx[0]=0;
      ctx[29]=1;ctx[6]=0;for(uint32_t r=0;r<rows;++r)if(ctx[48+r]<ctx[56+r])ctx[6]=1;}
    asc_syncthreads();
  } else if constexpr(Phase==5 || Phase==6 || Phase==7 || Phase==8 || Phase==9) {
    const uint32_t u=ctx[1];const int32_t req=a.tokenToReq[first];
    const int64_t reqid=a.reqIds[req];
    int32_t stable=(Phase==5||Phase==6)?0:ctx[40];
    if constexpr(Phase==5 || Phase==6){
    const bool reset=a.rowModes[first]!=(rows==1U?ROW_PRIVATE:ROW_JOINT)||a.lastReqIds[first]!=reqid;
#if JOINT_MTP7_MAP16
    if (reset) {
      asc_syncthreads();
      if (packedMap) FinishColdSorted<SORTED_THREADS,true>(a,first,end,EightMap16View{map16},unique,ctx,t);
      else FinishColdSorted<SORTED_THREADS,true>(a,first,end,map,unique,ctx,t);
      return;
    }
#else
    if(reset){asc_syncthreads();FinishColdSorted<SORTED_THREADS,true>(a,first,end,map,unique,ctx,t);return;}
#endif
    stable=a.stablePrefixLens[first];
    for(int64_t row=first+1;row<end;++row)if(a.stablePrefixLens[row]<stable)stable=a.stablePrefixLens[row];
    if(stable<0)stable=0;if(stable>a.maxToken)stable=a.maxToken;
    if constexpr(Phase==6)if(t==0U)ctx[40]=stable;
    const bool filter=HOT_FILTER && a.scratchElements<=EIGHT_UB_WORDS && capacity>=1024U;
    if(filter)for(uint32_t i=t;i<1024U;i+=SORTED_THREADS)partition[i]=0;
    if constexpr(REFINE_C){
      if(a.scratchElements<=EIGHT_UB_WORDS){
        // partition is dead until resident classification finishes. Reuse it as a
        // C：连续读取 resident 快照；复用 partition 前一生命周期，不新增 UB 缓冲。
        for(uint32_t slot=t;slot<capacity;slot+=SORTED_THREADS){
          if(slot<u)owner[slot]=-1;
          partition[slot]=a.slotToToken[cb+slot];
        }
      }else for(uint32_t i=t;i<u;i+=SORTED_THREADS)owner[i]=-1;
    }else for(uint32_t i=t;i<u;i+=SORTED_THREADS)owner[i]=-1;
    asc_syncthreads();
    if(filter){
      for(uint32_t i=t;i<u;i+=SORTED_THREADS){uint32_t bit=static_cast<uint32_t>(unique[i])&32767U;
        asc_atomic_or(partition+(bit>>5U),static_cast<int32_t>(1U<<(bit&31U)));}
      asc_syncthreads();
    }
    uint32_t valueBits=0;while(valueBits<31U&&(1U<<valueBits)<static_cast<uint32_t>(a.maxToken))++valueBits;
    const uint32_t binShift=valueBits>5U?valueBits-5U:0U;
    if constexpr(HW_B){
      for(uint32_t i=t;i<=32U;i+=SORTED_THREADS){
        uint32_t key=i<<binShift;ctx[31U+i]=SortedLower(unique,u,static_cast<int32_t>(key));
      }
      asc_syncthreads();
    }
    // One resident scan, no hash construction; every duplicate owner is protected.
    for(uint32_t order=t;order<capacity;order+=SORTED_THREADS){
      int32_t slot=a.lruSlots[cb+order];
      flags[order]=-1;
      if(slot<0||static_cast<uint32_t>(slot)>=capacity)continue;
      uint32_t hit=0,invalidTag=0;
      {const int32_t token=REFINE_C&&a.scratchElements<=EIGHT_UB_WORDS?partition[slot]:a.slotToToken[cb+slot];
        if constexpr(HOT_REUSE)if(token>=stable)invalidTag=0x20000U;
        const uint32_t bit=static_cast<uint32_t>(token)&32767U;
        if(token>=0&&token<stable && (!filter || (static_cast<uint32_t>(partition[bit>>5U])&(1U<<(bit&31U))))){uint32_t pos;
          if constexpr(HW_B){uint32_t bin=static_cast<uint32_t>(token)>>binShift;
            uint32_t lo=ctx[31U+bin],hi=ctx[32U+bin];pos=lo+SortedLower(unique+lo,hi-lo,token);
          }else pos=SortedLower(unique,u,token);
          if(pos<u&&unique[pos]==token){hit=1;asc_atomic_max(owner+pos,static_cast<int32_t>((order<<16U)|static_cast<uint32_t>(slot)));}}
      }
      flags[order]=slot|static_cast<int32_t>((hit<<16U)|invalidTag);
    }
    asc_syncthreads();
    if constexpr(Phase==6)return; // original final classify barrier precedes SIMD readers
    }
    uint32_t evicts=(Phase==5||Phase==7)?0:ctx[43],hits=(Phase==5||Phase==7)?0:ctx[44],parity=0;
    if constexpr(Phase==5 || Phase==7){
    evicts=0;hits=0;
    for(uint32_t tile=0;tile<capacity;tile+=SORTED_THREADS){
      uint32_t order=tile+t;int32_t f=order<capacity?flags[order]:-1;
      int32_t hit=f>=0?(HOT_REUSE?((f>>16U)&1):(f>>16U)):0,evict=f>=0?1-hit:0,total=0;
      int32_t rank=SortedScan(scan,evict+(hit<<COUNT_SHIFT),t,parity,total);
      if(evict)partition[evicts+(static_cast<uint32_t>(rank)&COUNT_MASK)]=f&(HOT_REUSE?0x2ffff:0xffff);
      if(hit)partition[capacity-1U-(hits+(static_cast<uint32_t>(rank)>>COUNT_SHIFT))]=f&(HOT_REUSE?0x2ffff:0xffff);
      asc_syncthreads();evicts+=static_cast<uint32_t>(total)&COUNT_MASK;hits+=static_cast<uint32_t>(total)>>COUNT_SHIFT;
    }
    if(evicts+hits!=capacity){if(t==0U)ctx[0]=0;return;}
    if constexpr(Phase==7){if(t==0U){ctx[43]=evicts;ctx[44]=hits;}asc_syncthreads();return;}
    }
    uint32_t misses=(Phase==5||Phase==8)?0:ctx[45];parity=0;
    if constexpr(Phase==5 || Phase==8){
    misses=0;
    for(uint32_t tile=0;tile<u;tile+=SORTED_THREADS){
      uint32_t pos=tile+t;bool missing=pos<u&&owner[pos]<0;int32_t total=0;
      int32_t rank=RefineBoolScan(scan,missing,t,parity,total);
      if(pos<u){flags[pos]=missing?static_cast<int32_t>(misses)+rank:-1;
        if(missing&&misses+static_cast<uint32_t>(rank)<evicts)owner[pos]=partition[misses+static_cast<uint32_t>(rank)];}
      asc_syncthreads();misses+=static_cast<uint32_t>(total);
    }
    if(misses>evicts){if(t==0U)ctx[0]=0;return;}
    if constexpr(Phase==8){if(t==0U)ctx[45]=misses;asc_syncthreads();return;}
    }
    // All capacity checks precede persistent writes. Missing slots now hold final slot IDs.
    if constexpr(HOT_REUSE){
      // 依照既有LRU排列契约，partition覆盖每个物理槽；只清理第一次扫描已标出的槽。
      for(uint32_t i=t;i<capacity;i+=SORTED_THREADS){uint32_t entry=static_cast<uint32_t>(partition[i]);
        if(entry&0x20000U)a.slotToToken[cb+(entry&0xffffU)]=-1;}
    }else for(uint32_t slot=t;slot<capacity;slot+=SORTED_THREADS)
      if(a.slotToToken[cb+slot]>=stable)a.slotToToken[cb+slot]=-1;
    asc_syncthreads();
    for(uint32_t pos=t;pos<u;pos+=SORTED_THREADS)if(flags[pos]>=0){
      const int32_t slot=owner[pos]&0xffff,rank=flags[pos];
      a.slotToToken[cb+slot]=unique[pos];a.missTokens[qb+rank]=unique[pos];a.missSlots[qb+rank]=slot;
    }
#if JOINT_MTP7_MAP16
    for (uint32_t i=t; i<q; i+=SORTED_THREADS) {
      const int32_t uid = packedMap ? EightMap16View{map16}[i] : map[i];
      a.currentSlots[qb+i] = uid >= 0 ? (owner[uid] & 0xffff) : -1;
    }
#else
    for(uint32_t i=t;i<q;i+=SORTED_THREADS){int32_t uid=map[i];a.currentSlots[qb+i]=uid>=0?(owner[uid]&0xffff):-1;}
#endif
    for(uint32_t i=misses+t;i<q;i+=SORTED_THREADS){a.missTokens[qb+i]=-1;a.missSlots[qb+i]=-1;}
    const uint32_t unused=evicts-misses;
    for(uint32_t order=t;order<capacity;order+=SORTED_THREADS){
      int32_t slot=order<unused?partition[misses+order]:order<evicts?partition[order-unused]:partition[capacity-1U-(order-evicts)];
      if constexpr(HOT_REUSE)slot&=0xffff;
      a.lruSlots[cb+order]=slot;
    }
    const FastGroupSource src=MakeFastGroupSource(a,req);const uint32_t pages=src.Pages(capacity);
    for(uint32_t r=0;r<rows;++r){int64_t row=first+r;
      for(uint32_t page=t;page<pages;page+=SORTED_THREADS)a.residentBlockTable[row*pages+page]=static_cast<int32_t>(first*pages+page);
      if(t==0U){a.residentRows[row]=first;int64_t left=static_cast<int64_t>(misses)-r*topk;a.missCount[row]=left<=0?0:left<topk?left:topk;}
      if(t==1U)a.rowModes[row]=rows==1U?ROW_PRIVATE:(r==0U?ROW_JOINT:ROW_FOLLOWER);
    }
    if(t==0U){a.lastReqIds[first]=reqid;ctx[0]=2;}asc_syncthreads();
  }
}

template <int Phase>
__simt_vf__ __aicore__ LAUNCH_BOUND((Phase==4?PLAN_THREADS:SORTED_THREADS)) inline void SortedStageVf(
    __ubuf__ int32_t* scanWorkspace, __gm__ int64_t* reqIds, __gm__ int64_t* lastReqIds, __gm__ int32_t* topkIndices,
    __gm__ int32_t* stablePrefixLens, __gm__ int32_t* visibleSeqLens, __gm__ int32_t* tokenToReq,
    __gm__ int32_t* slotToToken, __gm__ int32_t* lruSlots, __gm__ int32_t* currentSlots,
    __gm__ int32_t* missCount, __gm__ int32_t* missTokens, __gm__ int32_t* missSlots, __gm__ int32_t* compactWorkspace,
    uint64_t hashCapacity, uint64_t workspaceRowElements, int64_t rowStart, int64_t rowStride, int64_t numReqs,
    int64_t topk, int64_t capacity, int64_t maxToken, int64_t maxRequests,
    int32_t blockSize, __gm__ int32_t* rowModes, __gm__ int32_t* residentRows, __gm__ int32_t* residentBlockTable,
    uint64_t jointHashCapacity, uint64_t scratchElements) {
  const JointPlanArgs args{reqIds,
                           lastReqIds,
                           topkIndices,
                           stablePrefixLens,
                           visibleSeqLens,
                           tokenToReq,
                           slotToToken,
                           lruSlots,
                           currentSlots,
                           missCount,
                           missTokens,
                           missSlots,
                           rowModes,
                           residentRows,
                           residentBlockTable,
                           static_cast<uint32_t>(topk),
                           static_cast<uint32_t>(capacity),
                           static_cast<uint32_t>(hashCapacity),
                           static_cast<uint32_t>(jointHashCapacity),
#if JOINT_OPT_MASK == 16
                           compactWorkspace, scratchElements,
#elif JOINT_OPT_MASK >= 17
                           scratchElements,
#endif
                           static_cast<int32_t>(maxToken),
                           blockSize,
                           maxRequests};

  const uint32_t thread=static_cast<uint32_t>(threadIdx.x);
  if constexpr(Phase==4) {
    if(jointHashCapacity==0U){
      for(int64_t row=rowStart;row<numReqs;row+=rowStride){
        if(scratchElements<=EIGHT_UB_WORDS)RunPrivateRow(args,row,scanWorkspace+PLAN_SCAN_STORAGE_ELEMENTS,scanWorkspace,thread);
        else RunPrivateRow(args,row,compactWorkspace+static_cast<uint64_t>(row)*scratchElements,scanWorkspace,thread);
      }
    }else for(int64_t row=rowStart;row<rowStride;++row){
      if(scratchElements<=EIGHT_UB_WORDS)RunPrivateRow(args,row,scanWorkspace+PLAN_SCAN_STORAGE_ELEMENTS,scanWorkspace,thread);
      else RunPrivateRow(args,row,compactWorkspace+static_cast<uint64_t>(row)*scratchElements,scanWorkspace,thread);
    }
  } else {
    __ubuf__ int32_t* ctx=scanWorkspace+PLAN_SCAN_STORAGE_ELEMENTS;
    if(scratchElements<=EIGHT_UB_WORDS){
      __ubuf__ int32_t* data=ctx+EIGHT_CTX;
#if JOINT_MTP7_MAP16
      const uint32_t groupRows = HW_C && rowStride-rowStart<=8 ? static_cast<uint32_t>(rowStride-rowStart) : 8U;
      const uint32_t queries = groupRows * args.topk;
      const uint32_t words = EightMapIsPacked(groupRows,args.topk) ? (queries+1U)/2U : queries;
      auto* packet=data+((words+args.capacity+7U)&~7U);
#else
      auto* packet=data+((((HW_C&&rowStride-rowStart<=8?static_cast<uint32_t>(rowStride-rowStart):8U)*args.topk+args.capacity+7U)&~7U));
#endif
      if constexpr(HOT_FIXED){if(args.topk==2048U&&args.capacity==4096U){
        SortedStage<Phase,true>(args,rowStart,rowStride,data,ctx,packet,scanWorkspace);return;}}
      SortedStage<Phase,false>(args,rowStart,rowStride,data,ctx,packet,scanWorkspace);
    }else{
      __gm__ int32_t* data=compactWorkspace+static_cast<uint64_t>(rowStart)*scratchElements+EIGHT_CTX;
      if constexpr(HOT_FIXED){if(args.topk==2048U&&args.capacity==4096U){
        SortedStage<Phase,true>(args,rowStart,rowStride,data,ctx,ctx+EIGHT_CTX,scanWorkspace);return;}}
      SortedStage<Phase,false>(args,rowStart,rowStride,data,ctx,ctx+EIGHT_CTX,scanWorkspace);
    }
  }
}

// SPDX-License-Identifier: Apache-2.0
// D：替换完整 UB Phase3；只修改临时映射，不访问 GM 或提交持久状态。
// Scatter bases are 32B-aligned (data/ctx); unique is addressed as data[q+uid],
// so odd TopK/capacity layouts do not require an aligned unique pointer.
__simd_vf__ inline void RefineUniqueMapVf(__ubuf__ int32_t* records,
    __ubuf__ int32_t* data,__ubuf__ int32_t* ctx,uint32_t n,uint32_t q,
    uint32_t capacity,uint32_t initialCount,int32_t initialLast,uint32_t more EIGHT_MAP_PARAMETER){
  using namespace AscendC;
  Reg::RegTensor<uint32_t> lane,zero,one,even,odd,previousIndex,lastIndex;
  Reg::RegTensor<uint32_t> flags,prefix,carry,uid,payload,index,safeIndex,writeFlag,status,value;
  Reg::RegTensor<float> score,previous,lastScore,shiftScore,positive;
  Reg::RegTensor<int32_t> token,lastToken;
  Reg::MaskReg full=Reg::CreateMask<uint32_t,Reg::MaskPattern::ALL>();
  Reg::MaskReg first,head,valid,fits,writeHead,chosen;
  // Unsqueeze(ALL) is exactly [0,1,...,63] for b32: no lane table in UB.
  Reg::Unsqueeze<uint32_t>(lane,full);
  Reg::Duplicate(zero,uint32_t{0});Reg::Duplicate(one,uint32_t{1});
  Reg::Duplicate(carry,initialCount);Reg::Duplicate(lastToken,initialLast);
  Reg::Duplicate(lastScore,-static_cast<float>(initialLast));
  Reg::Compares<uint32_t,CMPMODE::EQ>(first,lane,uint32_t{0},full);
  Reg::Adds(previousIndex,lane,uint32_t(-1),full);
  Reg::Select(previousIndex,zero,previousIndex,first); // no out-of-range register Gather
  static constexpr Reg::CastTrait toInt={Reg::RegLayout::UNKNOWN,Reg::SatMode::NO_SAT,
      Reg::MaskMergeMode::ZEROING,RoundMode::CAST_TRUNC};
  for(uint32_t base=0;base<n;base+=64U){
    uint32_t remaining=n-base;valid=Reg::UpdateMask<uint32_t>(remaining);
    const uint32_t count=n-base<64U?n-base:64U;
    Reg::Adds(even,lane,base,full);Reg::ShiftLefts(even,even,int16_t{1},full);
    Reg::Adds(odd,even,uint32_t{1},full);
    Reg::Duplicate(score,0.0f);
    Reg::Gather<float,float,uint32_t>(score,reinterpret_cast<__ubuf__ float*>(records),even,valid);
    Reg::Gather<float,uint32_t>(shiftScore,score,previousIndex);
    Reg::Select(previous,lastScore,shiftScore,first);
    Reg::Compare<float,CMPMODE::NE>(head,score,previous,valid);
    Reg::Select(flags,one,zero,head);Reg::Select(flags,flags,zero,valid);
    // Rebuild all mask bits: no assumption about Compare's inactive lanes.
    Reg::Compares<uint32_t,CMPMODE::NE>(head,flags,uint32_t{0},full);
    Reg::Unsqueeze<uint32_t>(prefix,head); // exclusive head count; last head not included here
    Reg::Add(prefix,prefix,flags,full);    // include every head, including lane63
    Reg::Add(prefix,prefix,carry,full);
    Reg::Adds(uid,prefix,uint32_t(-1),full);
    Reg::Neg(positive,score,full);
    Reg::Cast<int32_t,float,toInt>(token,positive,valid); // exact integers in [0,2^24)
    Reg::Compares<uint32_t,CMPMODE::LT>(fits,uid,capacity,full);
    Reg::Select(writeFlag,flags,zero,fits);
    Reg::Compares<uint32_t,CMPMODE::NE>(writeHead,writeFlag,uint32_t{0},full);
#if JOINT_MTP7_MAP16
    Reg::Adds(index,uid,mapWords,full);
    Reg::Adds(safeIndex,lane,mapWords+capacity,full);
#else
    Reg::Adds(index,uid,q,full);
    Reg::Adds(safeIndex,lane,q+capacity,full);
#endif
    Reg::Select(index,index,safeIndex,writeHead); // even masked indices are unique
    Reg::Scatter<int32_t,uint32_t>(data,token,index,writeHead);
    Reg::Duplicate(payload,uint32_t{0});
    Reg::Gather<uint32_t,uint32_t,uint32_t>(payload,reinterpret_cast<__ubuf__ uint32_t*>(records),odd,valid);
    Reg::Adds(safeIndex,lane,q,full);Reg::Select(payload,payload,safeIndex,valid);
#if JOINT_MTP7_MAP16
    if (mapWords != q) {
      // B32 mask selects even B16 lanes. Each low half stores one UID;
      // inactive odd-lane indices are distinct and inside allocated scratch.
      Reg::Adds(safeIndex,lane,q+64U,full);
      Reg::ShiftLefts(index,safeIndex,int16_t{16},full);
      Reg::Or(index,index,payload,full);
      Reg::Scatter<uint16_t,uint16_t>(reinterpret_cast<__ubuf__ uint16_t*>(data),
          reinterpret_cast<Reg::RegTensor<uint16_t>&>(uid),
          reinterpret_cast<Reg::RegTensor<uint16_t>&>(index),valid);
    } else {
      Reg::Scatter<uint32_t,uint32_t>(reinterpret_cast<__ubuf__ uint32_t*>(data),uid,payload,valid);
    }
#else
    Reg::Scatter<uint32_t,uint32_t>(reinterpret_cast<__ubuf__ uint32_t*>(data),uid,payload,valid);
#endif
    Reg::Duplicate(lastIndex,uint32_t{63});Reg::Gather<uint32_t,uint32_t>(carry,prefix,lastIndex);
    Reg::Duplicate(lastIndex,count-1U);
    Reg::Gather<float,uint32_t>(lastScore,score,lastIndex);
    Reg::Gather<int32_t,uint32_t>(lastToken,token,lastIndex);
  }
  // All progress Scatter indices are [0,63], unique and within the original ctx64.
  Reg::Compares<uint32_t,CMPMODE::EQ>(chosen,lane,uint32_t{1},full);
  Reg::Scatter<uint32_t,uint32_t>(reinterpret_cast<__ubuf__ uint32_t*>(ctx),carry,lane,chosen);
  Reg::Compares<uint32_t,CMPMODE::EQ>(chosen,lane,uint32_t{2},full);
  Reg::Scatter<int32_t,uint32_t>(ctx,lastToken,lane,chosen);
  Reg::Compares<uint32_t,CMPMODE::GE>(fits,carry,capacity+1U,full);
  Reg::Select(status,zero,one,fits);
  Reg::Compares<uint32_t,CMPMODE::EQ>(chosen,lane,uint32_t{0},full);
  Reg::Scatter<uint32_t,uint32_t>(reinterpret_cast<__ubuf__ uint32_t*>(ctx),status,lane,chosen);
  Reg::Compares<uint32_t,CMPMODE::EQ>(chosen,lane,uint32_t{29},full);
  Reg::Scatter<uint32_t,uint32_t>(reinterpret_cast<__ubuf__ uint32_t*>(ctx),one,lane,chosen);
  Reg::Duplicate(value,more);
  Reg::Compares<uint32_t,CMPMODE::EQ>(chosen,lane,uint32_t{6},full);
  Reg::Scatter<uint32_t,uint32_t>(reinterpret_cast<__ubuf__ uint32_t*>(ctx),value,lane,chosen);
}

// 热路径SIMD压缩：稳定排列与rank完全保留；只改scratch，不提交持久状态。
// data/ctx基址32B对齐；离散访问使用相对data索引。
// 未激活lane使用跳过目标数组区间的[0,64)索引，全部唯一且落在已分配UB内。
__simd_callee__ inline void HotPutCtx(__ubuf__ int32_t* ctx,
    AscendC::Reg::RegTensor<uint32_t>& value,AscendC::Reg::RegTensor<uint32_t>& lane,
    uint32_t slot,AscendC::Reg::MaskReg& full){
  AscendC::Reg::MaskReg chosen;
  AscendC::Reg::Compares<uint32_t,AscendC::CMPMODE::EQ>(chosen,lane,slot,full);
  AscendC::Reg::Scatter<uint32_t,uint32_t>(reinterpret_cast<__ubuf__ uint32_t*>(ctx),value,lane,chosen);
}
template<bool DoPartition,bool DoMiss>
__simd_vf__ inline void HotCompactVf(__ubuf__ int32_t* data,__ubuf__ int32_t* ctx,
    uint32_t q,uint32_t capacity,uint32_t u,uint32_t oldEvicts,uint32_t oldHits EIGHT_MAP_PARAMETER){
  using namespace AscendC;
#if JOINT_MTP7_MAP16
  const uint32_t ow=(mapWords+capacity+7U)&~7U,fl=ow+capacity,pa=fl+capacity;
#else
  const uint32_t ow=(q+capacity+7U)&~7U,fl=ow+capacity,pa=fl+capacity;
#endif
  Reg::RegTensor<uint32_t> lane,zero,one,maskHit,keep,raw,hflag,eflag,vflag,hprefix,eprefix;
  Reg::RegTensor<uint32_t> hi,ei,hcarry,ecarry,mc,rank,inc,index,safe,slotTag,value,sum,status,tmp;
  Reg::RegTensor<int32_t> f,ownerValue,missRank;
  Reg::MaskReg full=Reg::CreateMask<uint32_t,Reg::MaskPattern::ALL>(),valid,hit,evict,selected,fits;
  Reg::Unsqueeze<uint32_t>(lane,full);Reg::Duplicate(zero,uint32_t{0});Reg::Duplicate(one,uint32_t{1});
  Reg::Duplicate(hcarry,oldHits);Reg::Duplicate(ecarry,oldEvicts);Reg::Duplicate(mc,uint32_t{0});
  Reg::Duplicate(maskHit,uint32_t{65536});Reg::Duplicate(keep,HOT_REUSE?uint32_t{0x2ffff}:uint32_t{0xffff});
  if constexpr(DoPartition){
    Reg::Duplicate(hcarry,uint32_t{0});Reg::Duplicate(ecarry,uint32_t{0});
    for(uint32_t base=0;base<capacity;base+=64U){
      uint32_t n=capacity-base;valid=Reg::UpdateMask<uint32_t>(n);
      Reg::Adds(index,lane,fl+base,full);Reg::Duplicate(f,int32_t{-1});
      Reg::Gather<int32_t,int32_t,uint32_t>(f,data,index,valid);
      Reg::Compares<int32_t,CMPMODE::GE>(selected,f,int32_t{0},valid);
      Reg::Select(vflag,one,zero,selected);Reg::Select(vflag,vflag,zero,valid);
      Reg::And(raw,reinterpret_cast<Reg::RegTensor<uint32_t>&>(f),maskHit,full);
      Reg::Compares<uint32_t,CMPMODE::NE>(hit,raw,uint32_t{0},full);
      Reg::Select(hflag,vflag,zero,hit);Reg::Sub(eflag,vflag,hflag,full);
      Reg::Compares<uint32_t,CMPMODE::NE>(hit,hflag,uint32_t{0},full);
      Reg::Compares<uint32_t,CMPMODE::NE>(evict,eflag,uint32_t{0},full);
      Reg::Unsqueeze<uint32_t>(hprefix,hit);Reg::Unsqueeze<uint32_t>(eprefix,evict);
      Reg::Add(hi,hprefix,hcarry,full);Reg::Add(ei,eprefix,ecarry,full);
      Reg::Duplicate(tmp,capacity-1U);Reg::Sub(index,tmp,hi,full);
      Reg::Select(index,index,ei,hit);Reg::Adds(index,index,pa,full);
      Reg::Adds(safe,lane,capacity,full);Reg::Compares<uint32_t,CMPMODE::GE>(fits,lane,pa,full);Reg::Select(safe,safe,lane,fits);
      Reg::Compares<uint32_t,CMPMODE::NE>(selected,vflag,uint32_t{0},full);
      Reg::Select(index,index,safe,selected);
      Reg::And(slotTag,reinterpret_cast<Reg::RegTensor<uint32_t>&>(f),keep,full);
      Reg::Scatter<uint32_t,uint32_t>(reinterpret_cast<__ubuf__ uint32_t*>(data),slotTag,index,selected);
      Reg::Add(hi,hi,hflag,full);Reg::Add(ei,ei,eflag,full);Reg::Duplicate(tmp,uint32_t{63});
      Reg::Gather<uint32_t,uint32_t>(hcarry,hi,tmp);Reg::Gather<uint32_t,uint32_t>(ecarry,ei,tmp);
    }
  }
  if constexpr(DoPartition && DoMiss){
    // partition刚写出，miss阶段将读取；同时flags原读生命周期结束后才可覆盖。
    Reg::LocalMemBar<Reg::MemType::VEC_ALL,Reg::MemType::VEC_ALL>();
  }
  if constexpr(DoMiss){
    for(uint32_t base=0;base<u;base+=64U){
      uint32_t n=u-base;valid=Reg::UpdateMask<uint32_t>(n);
      Reg::Adds(index,lane,ow+base,full);Reg::Duplicate(ownerValue,int32_t{0});
      Reg::Gather<int32_t,int32_t,uint32_t>(ownerValue,data,index,valid);
      Reg::Compares<int32_t,CMPMODE::LT>(selected,ownerValue,int32_t{0},valid);
      Reg::Select(vflag,one,zero,selected);Reg::Select(vflag,vflag,zero,valid);
      Reg::Compares<uint32_t,CMPMODE::NE>(selected,vflag,uint32_t{0},full);
      Reg::Unsqueeze<uint32_t>(rank,selected);Reg::Add(rank,rank,mc,full);
      Reg::Duplicate(missRank,int32_t{-1});
      Reg::Select(missRank,reinterpret_cast<Reg::RegTensor<int32_t>&>(rank),missRank,selected);
      Reg::Adds(index,lane,fl+base,full);Reg::Adds(safe,lane,capacity,full);
      Reg::Compares<uint32_t,CMPMODE::GE>(hit,lane,fl,full);Reg::Select(safe,safe,lane,hit);
      Reg::Select(index,index,safe,valid);Reg::Scatter<int32_t,uint32_t>(data,missRank,index,valid);
      Reg::Compare<uint32_t,CMPMODE::LT>(fits,rank,ecarry,full);
      Reg::Select(vflag,vflag,zero,fits);Reg::Compares<uint32_t,CMPMODE::NE>(fits,vflag,uint32_t{0},full);
      Reg::Adds(index,rank,pa,full);Reg::Duplicate(tmp,pa);Reg::Select(index,index,tmp,fits);
      Reg::Duplicate(value,uint32_t{0});
      Reg::Gather<uint32_t,uint32_t,uint32_t>(value,reinterpret_cast<__ubuf__ uint32_t*>(data),index,fits);
      Reg::Adds(index,lane,ow+base,full);Reg::Adds(safe,lane,capacity,full);
      Reg::Compares<uint32_t,CMPMODE::GE>(hit,lane,ow,full);Reg::Select(safe,safe,lane,hit);Reg::Select(index,index,safe,fits);
      Reg::Scatter<uint32_t,uint32_t>(reinterpret_cast<__ubuf__ uint32_t*>(data),value,index,fits);
      Reg::Compares<int32_t,CMPMODE::GE>(selected,missRank,int32_t{0},valid);
      Reg::Select(vflag,one,zero,selected);Reg::Select(vflag,vflag,zero,valid);Reg::Add(inc,rank,vflag,full);Reg::Duplicate(tmp,uint32_t{63});
      Reg::Gather<uint32_t,uint32_t>(mc,inc,tmp);
    }
  }
  Reg::Add(sum,ecarry,hcarry,full);Reg::Compares<uint32_t,CMPMODE::EQ>(selected,sum,capacity,full);
  Reg::Select(status,one,zero,selected);
  if constexpr(DoMiss){Reg::Adds(tmp,ecarry,uint32_t{1},full);Reg::Compare<uint32_t,CMPMODE::LT>(selected,mc,tmp,full);Reg::Select(status,status,zero,selected);}
  HotPutCtx(ctx,ecarry,lane,43,full);HotPutCtx(ctx,hcarry,lane,44,full);
  HotPutCtx(ctx,mc,lane,45,full);HotPutCtx(ctx,status,lane,0,full);
}

// Five-factor hardware experiment; validated controller kept in sorted_controller.h.
// SIMD/SIMT streaming controller; each kernel handles all groups assigned to its AIV.
// Source scores are -token, exact in float32 for maxToken<=2^24; payload is original qIdx.
// A refill restages all active heads, then consumes ONLY the exhausted prefix.
// No full resident hash and no persistent token map/cache across invocations.
__aicore__ inline void SortedVectorToScalar(AscendC::TPipe& pipe) {
  auto id=pipe.AllocEventID<AscendC::HardEvent::V_S>();
  AscendC::SetFlag<AscendC::HardEvent::V_S>(id);AscendC::WaitFlag<AscendC::HardEvent::V_S>(id);
  pipe.ReleaseEventID<AscendC::HardEvent::V_S>(id);
}
__aicore__ inline void SortedScalarToVector(AscendC::TPipe& pipe) {
  auto id=pipe.AllocEventID<AscendC::HardEvent::S_V>();
  AscendC::SetFlag<AscendC::HardEvent::S_V>(id);AscendC::WaitFlag<AscendC::HardEvent::S_V>(id);
  pipe.ReleaseEventID<AscendC::HardEvent::S_V>(id);
}

// Copy is only used for a one-list half; no GM traffic or persistent writes.
__simt_vf__ __aicore__ LAUNCH_BOUND(SORTED_THREADS) inline void EightCopyVf(
    __ubuf__ int32_t* dst,__ubuf__ int32_t* src,uint32_t words){
  for(uint32_t i=threadIdx.x;i<words;i+=SORTED_THREADS)dst[i]=src[i];
  asc_syncthreads();
}
__simt_vf__ __aicore__ LAUNCH_BOUND(SORTED_THREADS) inline void EightProgressVf(
    __ubuf__ int32_t* records,__ubuf__ int32_t* ctx,uint32_t n,uint32_t topk){
  const uint32_t t=threadIdx.x;
  if(t<8U){
    uint32_t consumed=0;
    for(uint32_t i=0;i<n;++i)if(static_cast<uint32_t>(records[2U*i+1U])/topk==t)++consumed;
    ctx[48U+t]+=consumed;
  }
  asc_syncthreads();
}

#if JOINT_MTP7_OPT >= 1
// A：512线程只遍历自己负责的记录；8个局部计数按常量下标展开，再按warp汇总。
__simt_vf__ __aicore__ LAUNCH_BOUND(SORTED_THREADS) inline void EightParallelProgressVf(
    __ubuf__ int32_t* records,__ubuf__ int32_t* ctx,uint32_t n,uint32_t topk){
  const uint32_t t=threadIdx.x;
  auto* partials=records-static_cast<uint32_t>(ctx[30]);
  uint32_t bins[8]={0,0,0,0,0,0,0,0};
  for(uint32_t i=t;i<n;i+=SORTED_THREADS){
    const uint32_t qid=static_cast<uint32_t>(records[2U*i+1U]);
    const uint32_t row=topk==2048U?qid>>11U:qid/topk;
    if(row>=8U)asc_atomic_min(ctx,0);
#pragma unroll
    for(uint32_t r=0;r<8U;++r)bins[r]+=row==r?1U:0U; // constant-index registers, no dynamic private-array access
  }

  // Dead original packet input is now the warp partial-count workspace.
  // Each record is loaded once, bins are thread-private; no per-record atomics.
  const uint32_t lane=t&31U,warp=t/32U;
#pragma unroll
  for(uint32_t r=0;r<8U;++r){
    uint32_t v=bins[r];
#pragma unroll
    for(uint32_t delta=16U;delta;delta>>=1U)v+=static_cast<uint32_t>(asc_shfl(static_cast<int32_t>(v),lane^delta,32));
    if(lane==0U)partials[warp*8U+r]=static_cast<int32_t>(v);
  }
  asc_syncthreads();
  if(t<8U){uint32_t total=0;
    for(uint32_t w=0;w<SORTED_WARPS;++w)total+=static_cast<uint32_t>(partials[w*8U+t]);
    ctx[48U+t]+=total;
  }
  asc_syncthreads();
}
#endif

#if JOINT_MTP7_OPT == 2
// Sorted score is -token. Inclusive=true counts score>=key; otherwise score>key.
__simt_callee__ __aicore__ inline uint32_t EightScoreBound(
    __ubuf__ int32_t* records,uint32_t n,float key,bool inclusive){
  auto* score=reinterpret_cast<__ubuf__ float*>(records);uint32_t lo=0,hi=n;
  while(lo<hi){const uint32_t mid=lo+((hi-lo)>>1U);const float value=score[2U*mid];
    if(value>key||(inclusive&&value==key))lo=mid+1U;else hi=mid;
  }
  return lo;
}
#if JOINT_MTP7_TILE
__simt_callee__ __aicore__ inline uint32_t EightDiagonal(
    __ubuf__ int32_t* a,__ubuf__ int32_t* b,uint32_t na,uint32_t nb,uint32_t k){
  auto* x=reinterpret_cast<__ubuf__ float*>(a);auto* y=reinterpret_cast<__ubuf__ float*>(b);
  uint32_t lo=k>nb?k-nb:0U,hi=k<na?k:na;
  while(lo<=hi){
    const uint32_t i=lo+((hi-lo)>>1U),j=k-i;
    if(i>0U&&j<nb&&x[2U*(i-1U)]<y[2U*j]){hi=i-1U;continue;}
    if(j>0U&&i<na&&y[2U*(j-1U)]<=x[2U*i]){lo=i+1U;continue;}
    return i;
  }
  return lo;
}
#endif
// B：两段安全前缀并行rank-scatter归并，同时统计原始行消费量。
// No final MrgSort or separate progress VF. All persistent writes remain downstream.
__simt_vf__ __aicore__ LAUNCH_BOUND(SORTED_THREADS) inline void EightRankMergeVf(
    __ubuf__ int32_t* packet,__ubuf__ int32_t* ctx,uint32_t chunk,uint32_t topk){
  const uint32_t t=threadIdx.x;
#if JOINT_MTP7_REUSE
  auto* a=packet+16U*chunk;auto* b=packet;auto* out=packet+24U*chunk;
  auto* partials=packet+8U*chunk; // dead second-half input, disjoint from both live mids
#else
  auto* a=packet+16U*chunk;auto* b=packet+24U*chunk;auto* out=packet+32U*chunk;
  auto* partials=packet;
#endif
  if(t==0U){
    uint32_t na=ctx[104],nb=ctx[105];const uint32_t complete=ctx[106];
    float cut=-16777216.0f;
    if(!(complete&1U))cut=reinterpret_cast<__ubuf__ float*>(a)[2U*(na-1U)];
    if(!(complete&2U)){float v=reinterpret_cast<__ubuf__ float*>(b)[2U*(nb-1U)];if(v>cut)cut=v;}
    // Retain all known records equal to cut. Unseen records are <=cut (score),
    // so future equal tokens remain legal and are deduplicated by the carried lastToken.
    ctx[107]=EightScoreBound(a,na,cut,true);ctx[108]=EightScoreBound(b,nb,cut,true);
    ctx[4]=ctx[107]+ctx[108];if(ctx[4]==0)ctx[0]=0;
  }
  asc_syncthreads();
  const uint32_t na=ctx[107],nb=ctx[108],n=na+nb;
  uint32_t bins[8]={0,0,0,0,0,0,0,0};
#if JOINT_MTP7_TILE
  // Every thread owns small output tiles, never an entire queue. All ranks
  // follow stable A-before-B ties, including across8-output tile boundaries.
  for(uint32_t base=t*8U;base<n;base+=SORTED_THREADS*8U){
    uint32_t ia=EightDiagonal(a,b,na,nb,base),ib=base-ia;
    const uint32_t limit=n-base<8U?n:base+8U;
    for(uint32_t dst=base;dst<limit;++dst){
      const bool first=ia<na&&(ib>=nb||reinterpret_cast<__ubuf__ float*>(a)[2U*ia]>=reinterpret_cast<__ubuf__ float*>(b)[2U*ib]);
      auto* src=first?a:b;const uint32_t index=first?ia++:ib++;
      out[2U*dst]=src[2U*index];out[2U*dst+1U]=src[2U*index+1U];
      const uint32_t qid=static_cast<uint32_t>(src[2U*index+1U]);
      const uint32_t row=topk==2048U?qid>>11U:qid/topk;
      if(row>=8U)asc_atomic_min(ctx,0);
#pragma unroll
      for(uint32_t r=0;r<8U;++r)bins[r]+=row==r?1U:0U;
    }
  }
#else
  for(uint32_t i=t;i<n;i+=SORTED_THREADS){
    const bool first=i<na;const uint32_t j=first?i:i-na;
    auto* src=first?a:b;auto* other=first?b:a;
    const float key=reinterpret_cast<__ubuf__ float*>(src)[2U*j];
    // Equal keys from a precede equal keys from b: ranks are unique, no atomics.
    const uint32_t rank=j+EightScoreBound(other,first?nb:na,key,!first);
    out[2U*rank]=src[2U*j];out[2U*rank+1U]=src[2U*j+1U];
    const uint32_t qid=static_cast<uint32_t>(src[2U*j+1U]);
    const uint32_t row=topk==2048U?qid>>11U:qid/topk;
    if(row>=8U)asc_atomic_min(ctx,0);
#pragma unroll
    for(uint32_t r=0;r<8U;++r)bins[r]+=row==r?1U:0U; // constant-index registers, no dynamic private-array access
  }
#endif

  // Dead original packet input is now the warp partial-count workspace.
  // Each record is loaded once, bins are thread-private; no per-record atomics.
  const uint32_t lane=t&31U,warp=t/32U;
#pragma unroll
  for(uint32_t r=0;r<8U;++r){
    uint32_t v=bins[r];
#pragma unroll
    for(uint32_t delta=16U;delta;delta>>=1U)v+=static_cast<uint32_t>(asc_shfl(static_cast<int32_t>(v),lane^delta,32));
    if(lane==0U)partials[warp*8U+r]=static_cast<int32_t>(v);
  }
  asc_syncthreads();
  if(t<8U){uint32_t total=0;
    for(uint32_t w=0;w<SORTED_WARPS;++w)total+=static_cast<uint32_t>(partials[w*8U+t]);
    ctx[48U+t]+=total;
  }
  asc_syncthreads();
}
#endif

#if JOINT_MTP7_MID_UNIQUE || JOINT_MTP7_FUSED_COUNT
#include "mtp7_merge_unique.h"
#endif

#if JOINT_MTP7_FUSED_TUNE_BITS
#include "mtp7_fused_refine.h"
#endif

template <bool AllowJoint>
__aicore__ inline void SortedDirectBody(GM_ADDR reqIds, GM_ADDR topkIndices, GM_ADDR stablePrefixLens, GM_ADDR visibleSeqLens,
    GM_ADDR tokenToReq, GM_ADDR activeRows, GM_ADDR lastReqIds,
    GM_ADDR slotToToken, GM_ADDR lruSlots, GM_ADDR currentSlots, GM_ADDR missCount,
    GM_ADDR missTokens, GM_ADDR missSlots, GM_ADDR compactWorkspace, GM_ADDR rowModes,
    GM_ADDR residentRows, GM_ADDR residentBlockTable, GM_ADDR tiling) {
  JointDirectTilingData d;
  const auto* gm=reinterpret_cast<const __gm__ JointDirectTilingData*>(tiling);
  d.hashCapacity=gm->hashCapacity;
  d.workspaceRowElements=gm->workspaceRowElements;
  d.maxRows=gm->maxRows;
  d.topk=gm->topk;
  d.capacity=gm->capacity;
  d.maxToken=gm->maxToken;
  d.maxRequests=gm->maxRequests;
  d.blockSize=gm->blockSize;
  d.localMemoryBytes=gm->localMemoryBytes;
  d.jointHashCapacity=gm->jointHashCapacity;
  d.scratchElements=gm->scratchElements;
  const int32_t requested=*reinterpret_cast<__gm__ int32_t*>(activeRows);
  const int64_t rows=requested<=0?0:requested<d.maxRows?requested:d.maxRows;
  const int64_t start=AscendC::GetBlockIdx(),stride=AscendC::GetBlockNum();if(start>=rows)return;
  AscendC::TPipe pipe;AscendC::TBuf<AscendC::TPosition::VECCALC> storage;pipe.InitBuffer(storage,d.localMemoryBytes);
  auto local=storage.Get<int32_t>();auto* scan=reinterpret_cast<__ubuf__ int32_t*>(local.GetPhyAddr());
  auto* ctx=scan+PLAN_SCAN_STORAGE_ELEMENTS;
#define SORTED_CALL(PHASE) asc_vf_call<SortedStageVf<PHASE>>(dim3((PHASE)==4?PLAN_THREADS:SORTED_THREADS), scan, reinterpret_cast<__gm__ int64_t*>(reqIds), reinterpret_cast<__gm__ int64_t*>(lastReqIds), \
      reinterpret_cast<__gm__ int32_t*>(topkIndices),reinterpret_cast<__gm__ int32_t*>(stablePrefixLens), \
      reinterpret_cast<__gm__ int32_t*>(visibleSeqLens),reinterpret_cast<__gm__ int32_t*>(tokenToReq), \
      reinterpret_cast<__gm__ int32_t*>(slotToToken),reinterpret_cast<__gm__ int32_t*>(lruSlots), \
      reinterpret_cast<__gm__ int32_t*>(currentSlots),reinterpret_cast<__gm__ int32_t*>(missCount), \
      reinterpret_cast<__gm__ int32_t*>(missTokens),reinterpret_cast<__gm__ int32_t*>(missSlots), \
      reinterpret_cast<__gm__ int32_t*>(compactWorkspace),d.hashCapacity,d.workspaceRowElements, \
      first,end,rows,d.topk,d.capacity,d.maxToken,d.maxRequests,d.blockSize, \
      reinterpret_cast<__gm__ int32_t*>(rowModes),reinterpret_cast<__gm__ int32_t*>(residentRows), \
      reinterpret_cast<__gm__ int32_t*>(residentBlockTable),d.jointHashCapacity,d.scratchElements)
  if(!AllowJoint||d.jointHashCapacity==0U){int64_t first=start,end=stride;SORTED_CALL(4);return;}
  int64_t group=0;
  for(int64_t first=0;first<rows;){
    int64_t end=first+1;const int32_t req=reinterpret_cast<__gm__ int32_t*>(tokenToReq)[first];
    while(end<rows&&reinterpret_cast<__gm__ int32_t*>(tokenToReq)[end]==req)++end;
    if(group%stride==start){
      SORTED_CALL(0);SortedVectorToScalar(pipe);
      while(ctx[0]==1&&(!HW_A||ctx[6])){
        SORTED_CALL(1);SortedVectorToScalar(pipe);if(ctx[28]==0)break;
#if JOINT_OPT_MASK >= 39
        if(end-first>4 && ctx[28]>4){
          const uint32_t chunk=ctx[3];
#if JOINT_MTP7_MAP16
          const uint32_t inputOffset=PLAN_SCAN_STORAGE_ELEMENTS+EIGHT_CTX+
              ((EightMapWords(static_cast<uint32_t>(end-first)*d.topk,static_cast<uint32_t>(d.topk))+d.capacity+7U)&~7U);
#else
          const uint32_t inputOffset=PLAN_SCAN_STORAGE_ELEMENTS+EIGHT_CTX+
              ((static_cast<uint32_t>(end-first)*d.topk+d.capacity+7U)&~7U);
#endif
          auto floats=storage.Get<float>();auto* packet=scan+inputOffset;
          uint32_t lengths[2]={0,0}; bool allComplete[2]={true,true};
#if (JOINT_MTP7_FUSED_TUNE_BITS & 1)
          ctx[108]=-1; //0=completeA,1=completeB,2=both; -1=general8-row histogram
#endif
          for(uint32_t half=0;half<2U;++half){
            const uint32_t begin=half*4U, count=(static_cast<uint32_t>(ctx[28])-begin<4U?static_cast<uint32_t>(ctx[28])-begin:4U);
            uint16_t lens[4]={0,0,0,0};uint32_t used[4]={0,0,0,0};
            for(uint32_t k=0;k<count;++k){
              lens[k]=static_cast<uint16_t>(ctx[72+begin+k]);
#if !JOINT_MTP7_PREFIX
              if(ctx[72+begin+k]!=ctx[56+ctx[64+begin+k]]-ctx[48+ctx[64+begin+k]])allComplete[half]=false;
#endif
            }
#if JOINT_MTP7_REUSE
            const uint32_t mid=half==0U?16U*chunk:0U;
#else
            const uint32_t mid=16U*chunk+half*8U*chunk;
#endif
            if(count==1U){
              asc_vf_call<EightCopyVf>(dim3(SORTED_THREADS),packet+mid,packet+2U*begin*chunk,2U*lens[0]);
              lengths[half]=lens[0];
#if (JOINT_MTP7_FUSED_TUNE_BITS & 1)
              if(ctx[28]==8)ctx[112+begin]=lens[0];
#endif
              SortedVectorToScalar(pipe);
            }else{
              AscendC::MrgSortSrcList<float> lists{floats[inputOffset+2U*begin*chunk],floats[inputOffset+2U*(begin+1U)*chunk],
                floats[inputOffset+2U*(begin+(count>2?2U:0U))*chunk],floats[inputOffset+2U*(begin+(count>3?3U:0U))*chunk]};
              auto dst=floats[inputOffset+mid];
              if(allComplete[half])AscendC::MrgSort<float,false>(dst,lists,lens,used,static_cast<uint16_t>((1U<<count)-1U),1);
              else AscendC::MrgSort<float,true>(dst,lists,lens,used,static_cast<uint16_t>((1U<<count)-1U),1);
              SortedVectorToScalar(pipe);
              for(uint32_t k=0;k<count;++k){
                if(allComplete[half])used[k]=lens[k];
                if(used[k]>lens[k])ctx[0]=0;
                lengths[half]+=used[k];
#if (JOINT_MTP7_FUSED_TUNE_BITS & 1)
                if(ctx[28]==8)ctx[112+begin+k]=static_cast<int32_t>(used[k]);
#endif
              }
            }
          }
#if JOINT_MTP7_MID_UNIQUE
          // Compact both four-way prefixes without dropping any query's association.
          asc_vf_call<EightMidUniqueVf>(packet,ctx+EIGHT_CTX,ctx,
              static_cast<uint32_t>(end-first)*static_cast<uint32_t>(d.topk),chunk,lengths[0],lengths[1]);
          SortedVectorToScalar(pipe);
          lengths[0]=static_cast<uint32_t>(ctx[104]);lengths[1]=static_cast<uint32_t>(ctx[105]);
#endif
#if JOINT_MTP7_OPT == 2
          if(!lengths[0]||!lengths[1])ctx[0]=0;
          if(ctx[0]){
            ctx[104]=lengths[0];ctx[105]=lengths[1];
            ctx[106]=(allComplete[0]?1U:0U)|(allComplete[1]?2U:0U);
            SortedScalarToVector(pipe);
            asc_vf_call<EightRankMergeVf>(dim3(SORTED_THREADS),packet,ctx,chunk,static_cast<uint32_t>(d.topk));
            SortedVectorToScalar(pipe);
          }
#else
          uint16_t lens[4]={static_cast<uint16_t>(lengths[0]),static_cast<uint16_t>(lengths[1]),0,0};
          uint32_t used[4]={0,0,0,0};
          if(!lengths[0]||!lengths[1])ctx[0]=0;
          if(ctx[0]){
#if JOINT_MTP7_MID_UNIQUE
            AscendC::MrgSortSrcList<float> lists{floats[inputOffset+24U*chunk],floats[inputOffset+32U*chunk],floats[inputOffset+24U*chunk],floats[inputOffset+24U*chunk]};
            auto dst=floats[inputOffset];
#elif JOINT_MTP7_REUSE
            AscendC::MrgSortSrcList<float> lists{floats[inputOffset+16U*chunk],floats[inputOffset],floats[inputOffset+16U*chunk],floats[inputOffset+16U*chunk]};
            auto dst=floats[inputOffset+24U*chunk];
#else
            AscendC::MrgSortSrcList<float> lists{floats[inputOffset+16U*chunk],floats[inputOffset+24U*chunk],floats[inputOffset+16U*chunk],floats[inputOffset+16U*chunk]};
            auto dst=floats[inputOffset+32U*chunk];
#endif
            if(allComplete[0]&&allComplete[1]){
              AscendC::MrgSort<float,false>(dst,lists,lens,used,3,1);used[0]=lens[0];used[1]=lens[1];
            }else AscendC::MrgSort<float,true>(dst,lists,lens,used,3,1);
            SortedVectorToScalar(pipe);
            if(used[0]>lens[0]||used[1]>lens[1]||used[0]+used[1]==0U)ctx[0]=0;
            else{
              ctx[4]=used[0]+used[1];
#if (JOINT_MTP7_FUSED_TUNE_BITS & 1)
              // Only full8 active sources have contiguous row0..3 / row4..7 halves.
              // A partial second-layer prefix MUST NOT inherit first-layer counts.
              if(ctx[28]==8){
                const bool fullA=used[0]==lens[0],fullB=used[1]==lens[1];
                ctx[108]=fullA?(fullB?2:0):(fullB?1:-1);
                ctx[109]=static_cast<int32_t>(used[0]);ctx[110]=static_cast<int32_t>(used[1]);
              }
#endif
#if JOINT_MTP7_MID_UNIQUE || JOINT_MTP7_FUSED_COUNT
              ctx[107]=1; // the following SIMD/map stage will publish original-query consumption
#elif JOINT_MTP7_PREFIX
              uint32_t expected = 0U;
              for (uint32_t k=0; k<static_cast<uint32_t>(ctx[28]); ++k) {
                const uint32_t row=ctx[64U+k], n=ctx[72U+k];
                ctx[48U+row] += n;
                expected += n;
              }
              if (expected != static_cast<uint32_t>(ctx[4])) ctx[0]=0;
#else
              SortedScalarToVector(pipe);
#if JOINT_MTP7_OPT >= 1
#if JOINT_MTP7_REUSE
              asc_vf_call<EightParallelProgressVf>(dim3(SORTED_THREADS),packet+24U*chunk,ctx,static_cast<uint32_t>(ctx[4]),static_cast<uint32_t>(d.topk));
#else
              asc_vf_call<EightParallelProgressVf>(dim3(SORTED_THREADS),packet+32U*chunk,ctx,static_cast<uint32_t>(ctx[4]),static_cast<uint32_t>(d.topk));
#endif
#else
              asc_vf_call<EightProgressVf>(dim3(SORTED_THREADS),packet+32U*chunk,ctx,static_cast<uint32_t>(ctx[4]),static_cast<uint32_t>(d.topk));
#endif
              SortedVectorToScalar(pipe);
#endif
            }
          }
#endif
          SortedScalarToVector(pipe);
        }else
        if(HW_A&&ctx[28]==1){
          ctx[4]=ctx[72];ctx[80]=ctx[72];ctx[48+ctx[64]]+=ctx[72];ctx[7]=1;
          SortedScalarToVector(pipe);
        }else
        if(ctx[28]>1){
          const uint32_t chunk=ctx[3];
#if JOINT_MTP7_MAP16
          const uint32_t inputOffset=d.scratchElements<=EIGHT_UB_WORDS?
              PLAN_SCAN_STORAGE_ELEMENTS+EIGHT_CTX+((EightMapWords((HW_C?static_cast<uint32_t>(end-first):8U)*d.topk,d.topk)+d.capacity+7U)&~7U):PLAN_SCAN_STORAGE_ELEMENTS+EIGHT_CTX;
#else
          const uint32_t inputOffset=d.scratchElements<=EIGHT_UB_WORDS?
              PLAN_SCAN_STORAGE_ELEMENTS+EIGHT_CTX+((((HW_C?static_cast<uint32_t>(end-first):8U)*d.topk+d.capacity+7U)&~7U)):PLAN_SCAN_STORAGE_ELEMENTS+EIGHT_CTX;
#endif
          auto floats=storage.Get<float>();auto out=floats[inputOffset+static_cast<uint32_t>(ctx[30])];
          AscendC::MrgSortSrcList<float> lists{floats[inputOffset],floats[inputOffset+2U*chunk],
              floats[inputOffset+(ctx[28]>2?4U*chunk:0U)],floats[inputOffset+(ctx[28]>3?6U*chunk:0U)]};
          uint16_t lens[4]={0,0,0,0};uint32_t consumed[4]={0,0,0,0};
          for(int k=0;k<ctx[28];++k)lens[k]=static_cast<uint16_t>(ctx[72+k]);
          bool complete=HW_C;
#if JOINT_MTP7_PREFIX
          if (end-first != 8) {
          for(int k=0;k<ctx[28];++k)if(ctx[72+k]!=ctx[56+ctx[64+k]]-ctx[48+ctx[64+k]])complete=false;
          }
#else
          for(int k=0;k<ctx[28];++k)if(ctx[72+k]!=ctx[56+ctx[64+k]]-ctx[48+ctx[64+k]])complete=false;
#endif
          if(complete){
            AscendC::MrgSort<float,false>(out,lists,lens,consumed,static_cast<uint16_t>((1U<<ctx[28])-1U),1);
            for(int k=0;k<ctx[28];++k)consumed[k]=lens[k];
          }else AscendC::MrgSort<float,true>(out,lists,lens,consumed,static_cast<uint16_t>((1U<<ctx[28])-1U),1);
          SortedVectorToScalar(pipe);
          uint32_t total=0;
          for(int k=0;k<ctx[28];++k){if(consumed[k]>lens[k]){ctx[0]=0;break;}total+=consumed[k];ctx[80+k]=consumed[k];ctx[48+ctx[64+k]]+=consumed[k];}
          if(total==0U)ctx[0]=0;ctx[4]=static_cast<int32_t>(total);
          SortedScalarToVector(pipe);
        }else { SORTED_CALL(2); }
#else
        SORTED_CALL(2);
#endif
#if (JOINT_REFINE_BITS & 8)
          if(d.scratchElements<=EIGHT_UB_WORDS && ctx[0]==1){
            const uint32_t q=static_cast<uint32_t>(end-first)*static_cast<uint32_t>(d.topk);
#if JOINT_MTP7_MAP16
            const uint32_t inputOffset=PLAN_SCAN_STORAGE_ELEMENTS+EIGHT_CTX+((EightMapWords(q,d.topk)+d.capacity+7U)&~7U);
#else
            const uint32_t inputOffset=PLAN_SCAN_STORAGE_ELEMENTS+EIGHT_CTX+((q+d.capacity+7U)&~7U);
#endif
            auto* records=scan+inputOffset+(ctx[7]?0U:static_cast<uint32_t>(ctx[30]));
            uint32_t more=0;for(int64_t r=0;r<end-first;++r)if(ctx[48+r]<ctx[56+r])more=1;
#if JOINT_MTP7_MID_UNIQUE
            if(ctx[107]){
              auto* packet=scan+inputOffset;const uint32_t chunk=static_cast<uint32_t>(ctx[3]);
              asc_vf_call<RefineMidUniqueMapVf>(packet,ctx+EIGHT_CTX,ctx,
                  static_cast<uint32_t>(ctx[4]),q,static_cast<uint32_t>(d.capacity),
                  static_cast<uint32_t>(ctx[1]),ctx[2],1U,packet+16U*chunk,8U*chunk);
              SortedVectorToScalar(pipe);
              if(ctx[0]==1)asc_vf_call<EightResolveMidMapVf>(dim3(SORTED_THREADS),packet,ctx+EIGHT_CTX,ctx,chunk,static_cast<uint32_t>(d.topk));
            }else
#elif JOINT_MTP7_FUSED_COUNT
            if(ctx[107]){
#if JOINT_MTP7_FUSED_TUNE_BITS
#define FUSED_TUNE_CALL(KNOWN,KT,UT) asc_vf_call<RefineFusedTuneVf<JOINT_MTP7_FUSED_TUNE_BITS,KNOWN>>(records,ctx+EIGHT_CTX,ctx, \
    static_cast<uint32_t>(ctx[4]),q,static_cast<uint32_t>(d.capacity),static_cast<uint32_t>(ctx[1]),ctx[2],static_cast<uint32_t>(d.topk),KT,UT)
#if (JOINT_MTP7_FUSED_TUNE_BITS & 1)
              // Dispatch is Scalar-side: each SIMD VF has a compile-time counter layout.
              if(ctx[108]==2){FUSED_TUNE_CALL(2,0U,0U);}
              else if(ctx[108]==0){FUSED_TUNE_CALL(0,static_cast<uint32_t>(ctx[109]),static_cast<uint32_t>(ctx[110]));}
              else if(ctx[108]==1){FUSED_TUNE_CALL(1,static_cast<uint32_t>(ctx[110]),static_cast<uint32_t>(ctx[109]));}
              else
#endif
              {FUSED_TUNE_CALL(-1,0U,0U);}
#undef FUSED_TUNE_CALL
#else
              asc_vf_call<RefineFusedCountVf>(records,ctx+EIGHT_CTX,ctx,
                  static_cast<uint32_t>(ctx[4]),q,static_cast<uint32_t>(d.capacity),
                  static_cast<uint32_t>(ctx[1]),ctx[2],more,static_cast<uint32_t>(d.topk));
#endif
            }else
#endif
            asc_vf_call<RefineUniqueMapVf>(records,ctx+EIGHT_CTX,ctx,
                static_cast<uint32_t>(ctx[4]),q,static_cast<uint32_t>(d.capacity),
                static_cast<uint32_t>(ctx[1]),ctx[2],more EIGHT_MAP_ARGUMENT(q,d.topk));
          }else {SORTED_CALL(3);} // GM scratch retains SIMT, including B if enabled.
#else
        SORTED_CALL(3);
#endif
        SortedVectorToScalar(pipe);
      }
      if(ctx[0]==1){
        if((HOT_PART||HOT_MISS) && d.scratchElements<=EIGHT_UB_WORDS){
          SORTED_CALL(6);SortedVectorToScalar(pipe);
          const uint32_t q=static_cast<uint32_t>(end-first)*static_cast<uint32_t>(d.topk);
          if(ctx[0]==1){
            if constexpr(HOT_PART && HOT_MISS){
              asc_vf_call<HotCompactVf<true,true>>(ctx+EIGHT_CTX,ctx,q,static_cast<uint32_t>(d.capacity),static_cast<uint32_t>(ctx[1]),0U,0U EIGHT_MAP_ARGUMENT(q,d.topk));
              SortedVectorToScalar(pipe);
            }else{
              if constexpr(HOT_PART){
                asc_vf_call<HotCompactVf<true,false>>(ctx+EIGHT_CTX,ctx,q,static_cast<uint32_t>(d.capacity),static_cast<uint32_t>(ctx[1]),0U,0U EIGHT_MAP_ARGUMENT(q,d.topk));
                SortedVectorToScalar(pipe);
              }else{SORTED_CALL(7);SortedVectorToScalar(pipe);}
              if(ctx[0]==1){
                if constexpr(HOT_MISS){
                  asc_vf_call<HotCompactVf<false,true>>(ctx+EIGHT_CTX,ctx,q,static_cast<uint32_t>(d.capacity),static_cast<uint32_t>(ctx[1]),static_cast<uint32_t>(ctx[43]),static_cast<uint32_t>(ctx[44]) EIGHT_MAP_ARGUMENT(q,d.topk));
                  SortedVectorToScalar(pipe);
                }else{SORTED_CALL(8);SortedVectorToScalar(pipe);}
              }
            }
            if(ctx[0]==1){SORTED_CALL(9);SortedVectorToScalar(pipe);}
          }
        }else{SORTED_CALL(5);SortedVectorToScalar(pipe);}
      }
      if(ctx[0]!=2){SORTED_CALL(4);}
    }
    ++group;first=end;
  }
#undef SORTED_CALL
}
