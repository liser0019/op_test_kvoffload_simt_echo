// Hardware sweep: A=control fast paths, B=32-bin resident index,
// C=group-sized map/adaptive packets, D=retain unconsumed UB heads,
// E=256-thread union/planner VFs. Original 1024-thread private fallback retained.
constexpr uint32_t HW_BITS = JOINT_OPT_MASK <= 69 ? JOINT_OPT_MASK-38U :
    JOINT_OPT_MASK==70 ? 15U : JOINT_OPT_MASK==71 ? 14U : JOINT_OPT_MASK==72 ? 5U : 4U;
constexpr bool HW_A=(HW_BITS&1U)!=0, HW_B=(HW_BITS&2U)!=0;
constexpr bool HW_C=(HW_BITS&4U)!=0, HW_D=(HW_BITS&8U)!=0;
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
// Hash-free sorted union planner. Helpers enter only through SortedStageVf.
// Lifetime: ctx[64] | queryToUid[4T] | unique[C] | work[max(3C,64)].
// work is packet input/output during union construction, then owner/flags/LRU.
// GM scratch keeps packet input/output in the separate UB staging area.
template <typename P>
__simt_callee__ __aicore__ inline uint32_t SortedLower(P p, uint32_t n, int32_t token) {
  uint32_t lo=0,hi=n;
  while(lo<hi){uint32_t mid=lo+((hi-lo)>>1U);if(p[mid]<token)lo=mid+1U;else hi=mid;}
  return lo;
}

template <int Phase, typename P>
__simt_callee__ __aicore__ inline void SortedStage(const JointPlanArgs& a,
    int64_t first,int64_t end,P data,__ubuf__ int32_t* ctx,__ubuf__ int32_t* packet,
    __ubuf__ int32_t* scan) {
  const uint32_t t=static_cast<uint32_t>(threadIdx.x), rows=static_cast<uint32_t>(end-first);
  const uint32_t mapRows=HW_C&&rows<=4U?rows:4U;
  P map=data, unique=map+mapRows*a.topk, owner=data+((mapRows*a.topk+a.capacity+7U)&~7U);
  P flags=owner+a.capacity, partition=flags+a.capacity;
  const uint32_t q=rows*a.topk;
  const int64_t qb=first*static_cast<int64_t>(a.topk),cb=first*static_cast<int64_t>(a.capacity);
  if constexpr(Phase==0) {
    if(t==0U){
      ctx[0]=1;ctx[1]=0;ctx[2]=-1;ctx[4]=0;ctx[6]=0;ctx[7]=0;ctx[28]=0;ctx[29]=0;
      uint32_t chunk=(3U*a.capacity>64U?3U*a.capacity:64U)/16U;
      if(chunk>768U)chunk=768U;ctx[3]=static_cast<int32_t>(chunk&~3U);
      uint64_t words=16ULL*static_cast<uint32_t>(ctx[3]);
      if(HW_C && a.scratchElements<=PLAN_ROW_UB_LIMIT_ELEMENTS)
        words=a.scratchElements-64U-((mapRows*a.topk+a.capacity+7U)&~7U);
      ctx[30]=static_cast<int32_t>((words/2U)&~uint64_t{7U}); // fixed input/output split

      const int32_t req=a.tokenToReq[first];
      uint32_t bits=1;while(bits<31U&&(1ULL<<bits)<q)++bits;
      if(rows<1U||rows>4U||a.capacity>32768U||q>65535U||a.maxToken<=0||a.maxToken>16777216||
         (static_cast<uint64_t>(a.maxToken-1)<<bits)>UINT32_MAX||req<0||req>=a.maxRequests)ctx[0]=0;
      for(uint32_t r=0;r<4U;++r){ctx[8+r]=0;ctx[12+r]=0;}
      if(ctx[0])for(uint32_t r=0;r<rows;++r){
        const int64_t b=(first+r)*static_cast<int64_t>(a.topk);
        int32_t upper=a.visibleSeqLens[first+r];if(upper<0)upper=0;if(upper>a.maxToken)upper=a.maxToken;
        ctx[8+r]=static_cast<int32_t>(SortedLower(a.topkIndices+b,a.topk,0));
        ctx[12+r]=static_cast<int32_t>(SortedLower(a.topkIndices+b,a.topk,upper));
        if(ctx[8+r]<ctx[12+r])ctx[6]=1;
      }
    }
    asc_syncthreads();
    // Reject >4 rows before touching the 4T-sized query map.
    if(ctx[0])for(uint32_t i=t;i<q;i+=SORTED_THREADS){
      // Independent semantics count duplicate query misses separately. Fall back;
      // never silently share their slot through the single-row unique list.
      if(rows==1U && i && a.topkIndices[qb+i]==a.topkIndices[qb+i-1U])asc_atomic_min(ctx,0);
      map[i]=-1;
    }
    asc_syncthreads();
  } else if constexpr(Phase==1) {
    const uint32_t oldChunk=ctx[3],oldActive=ctx[28],outOffset=ctx[30];
    // Output from the previous merge has been consumed; use it for nonoverlap
    // tail staging. Input and output regions have a fixed split for this group.
    if constexpr(HW_D){
      if(ctx[29])for(uint32_t z=t;z<oldActive*oldChunk;z+=SORTED_THREADS){
        uint32_t k=z/oldChunk,j=z%oldChunk;
        uint32_t used=ctx[24+k],left=static_cast<uint32_t>(ctx[20+k])-used;
        if(j<left){uint32_t src=k*oldChunk+used+j,dst=k*oldChunk+j;
          packet[outOffset+2U*dst]=packet[2U*src];packet[outOffset+2U*dst+1U]=packet[2U*src+1U];}
      }
      asc_syncthreads();
    }
    if(t==0U){
      for(uint32_t r=0;r<4U;++r){ctx[32+r]=0;ctx[36+r]=0;}
      if(HW_D&&ctx[29])for(uint32_t k=0;k<oldActive;++k){
        uint32_t r=ctx[16+k];ctx[32+r]=ctx[20+k]-ctx[24+k];ctx[36+r]=k*oldChunk;
      }
      uint32_t active=0;
      for(uint32_t r=0;r<rows;++r)if(ctx[8+r]<ctx[12+r])ctx[16+active++]=r;
      uint32_t chunk=oldChunk;
      if(HW_C&&active){chunk=(static_cast<uint32_t>(ctx[30])/(2U*active))&~3U;if(chunk>4092U)chunk=4092U;}
      ctx[3]=chunk;ctx[28]=active;ctx[7]=0;
      uint32_t total=0;
      for(uint32_t k=0;k<active;++k){uint32_t r=ctx[16+k],n=ctx[12+r]-ctx[8+r];
        if(n>chunk)n=chunk;ctx[20+k]=n;ctx[24+k]=0;total+=n;}
      ctx[4]=total;ctx[5]=total;
    }
    asc_syncthreads();
    const uint32_t chunk=ctx[3],active=ctx[28];
    auto* score=reinterpret_cast<__ubuf__ float*>(packet);
    for(uint32_t i=t;i<active*chunk;i+=SORTED_THREADS){uint32_t k=i/chunk,j=i%chunk;
      if(j<static_cast<uint32_t>(ctx[20+k])){uint32_t r=ctx[16+k],pos=static_cast<uint32_t>(ctx[8+r])+j;
        if(HW_D&&j<static_cast<uint32_t>(ctx[32+r])){uint32_t saved=ctx[36+r]+j;
          packet[2U*i]=packet[outOffset+2U*saved];packet[2U*i+1U]=packet[outOffset+2U*saved+1U];
        }else{score[2U*i]=-static_cast<float>(a.topkIndices[qb+r*a.topk+pos]);packet[2U*i+1U]=r*a.topk+pos;}
      }
    }
    asc_syncthreads();
  } else if constexpr(Phase==2) {
    // Software packet merge is a controlled comparison for hardware MrgSort.
    const uint32_t chunk=ctx[3],active=ctx[28];
    auto* score=reinterpret_cast<__ubuf__ float*>(packet);
    auto* out=packet+static_cast<uint32_t>(ctx[30]);
    for(uint32_t z=t;z<active*chunk;z+=SORTED_THREADS){
      const uint32_t k=z/chunk,j=z%chunk,n=ctx[20+k];if(j>=n)continue;
      const float key=score[2U*z];const int32_t id=packet[2U*z+1U];uint32_t rank=j;
      for(uint32_t s=0;s<active;++s)if(s!=k){
        uint32_t lo=0,hi=ctx[20+s];
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
        uint32_t row=static_cast<uint32_t>(out[2U*i+1U])/a.topk;
        for(uint32_t k=0;k<active;++k)if(static_cast<uint32_t>(ctx[16+k])==row)++ctx[24+k];
      }
      for(uint32_t k=0;k<active;++k)ctx[8+ctx[16+k]]+=ctx[24+k];
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
      int32_t total=0;const int32_t rank=SortedScan(scan,added?1:0,t,parity,total);
      if(i<n){uint32_t uid=count+static_cast<uint32_t>(rank)-(added?0U:1U);
        if(uid<a.capacity&&added)unique[uid]=token;
        map[static_cast<uint32_t>(out[2U*i+1U])]=static_cast<int32_t>(uid);
      }
      asc_syncthreads();count+=static_cast<uint32_t>(total);
    }
    if(t==0U){ctx[1]=count;if(n)ctx[2]=static_cast<int32_t>(-scores[2U*(n-1U)]);if(count>a.capacity)ctx[0]=0;
      ctx[29]=1;ctx[6]=0;for(uint32_t r=0;r<rows;++r)if(ctx[8+r]<ctx[12+r])ctx[6]=1;}
    asc_syncthreads();
  } else if constexpr(Phase==5) {
    const uint32_t u=ctx[1];const int32_t req=a.tokenToReq[first];
    const int64_t reqid=a.reqIds[req];const bool reset=a.rowModes[first]!=(rows==1U?ROW_PRIVATE:ROW_JOINT)||a.lastReqIds[first]!=reqid;
    if (reset) {
      // Uniform reset: no thread may publish mode/ID until all have captured it.
      asc_syncthreads();
      FinishColdSorted<SORTED_THREADS,true>(a,first,end,map,unique,ctx,t);
      return;
    }
    int32_t stable=a.stablePrefixLens[first];
    for(int64_t row=first+1;row<end;++row)if(a.stablePrefixLens[row]<stable)stable=a.stablePrefixLens[row];
    if(stable<0)stable=0;if(stable>a.maxToken)stable=a.maxToken;
    for(uint32_t i=t;i<u;i+=SORTED_THREADS)owner[i]=-1;
    asc_syncthreads();
    uint32_t valueBits=0;while(valueBits<31U&&(1U<<valueBits)<static_cast<uint32_t>(a.maxToken))++valueBits;
    const uint32_t binShift=valueBits>5U?valueBits-5U:0U;
    if constexpr(HW_B){
      for(uint32_t i=t;i<=32U;i+=SORTED_THREADS){
        uint32_t key=i<<binShift;ctx[31U+i]=SortedLower(unique,u,static_cast<int32_t>(key));
      }
      asc_syncthreads();
    }
    // One resident scan, no hash construction; every duplicate owner is protected.
    for(uint32_t order=t;order<a.capacity;order+=SORTED_THREADS){
      int32_t slot=reset?static_cast<int32_t>(order):a.lruSlots[cb+order];
      flags[order]=-1;
      if(slot<0||static_cast<uint32_t>(slot)>=a.capacity)continue;
      uint32_t hit=0;
      if(!reset){const int32_t token=a.slotToToken[cb+slot];
        if(token>=0&&token<stable){uint32_t pos;
          if constexpr(HW_B){uint32_t bin=static_cast<uint32_t>(token)>>binShift;
            uint32_t lo=ctx[31U+bin],hi=ctx[32U+bin];pos=lo+SortedLower(unique+lo,hi-lo,token);
          }else pos=SortedLower(unique,u,token);
          if(pos<u&&unique[pos]==token){hit=1;asc_atomic_max(owner+pos,static_cast<int32_t>((order<<16U)|static_cast<uint32_t>(slot)));}}
      }
      flags[order]=slot|static_cast<int32_t>(hit<<16U);
    }
    asc_syncthreads();
    uint32_t evicts=0,hits=0,parity=0;
    for(uint32_t tile=0;tile<a.capacity;tile+=SORTED_THREADS){
      uint32_t order=tile+t;int32_t f=order<a.capacity?flags[order]:-1;
      int32_t hit=f>=0?(f>>16U):0,evict=f>=0?1-hit:0,total=0;
      int32_t rank=SortedScan(scan,evict+(hit<<COUNT_SHIFT),t,parity,total);
      if(evict)partition[evicts+(static_cast<uint32_t>(rank)&COUNT_MASK)]=f&0xffff;
      if(hit)partition[a.capacity-1U-(hits+(static_cast<uint32_t>(rank)>>COUNT_SHIFT))]=f&0xffff;
      asc_syncthreads();evicts+=static_cast<uint32_t>(total)&COUNT_MASK;hits+=static_cast<uint32_t>(total)>>COUNT_SHIFT;
    }
    if(evicts+hits!=a.capacity){if(t==0U)ctx[0]=0;return;}
    uint32_t misses=0;parity=0;
    for(uint32_t tile=0;tile<u;tile+=SORTED_THREADS){
      uint32_t pos=tile+t;bool missing=pos<u&&owner[pos]<0;int32_t total=0;
      int32_t rank=SortedScan(scan,missing?1:0,t,parity,total);
      if(pos<u){flags[pos]=missing?static_cast<int32_t>(misses)+rank:-1;
        if(missing&&misses+static_cast<uint32_t>(rank)<evicts)owner[pos]=partition[misses+static_cast<uint32_t>(rank)];}
      asc_syncthreads();misses+=static_cast<uint32_t>(total);
    }
    if(misses>evicts){if(t==0U)ctx[0]=0;return;}
    // All capacity checks precede persistent writes. Missing slots now hold final slot IDs.
    for(uint32_t slot=t;slot<a.capacity;slot+=SORTED_THREADS)
      if(reset||a.slotToToken[cb+slot]>=stable)a.slotToToken[cb+slot]=-1;
    asc_syncthreads();
    for(uint32_t pos=t;pos<u;pos+=SORTED_THREADS)if(flags[pos]>=0){
      const int32_t slot=owner[pos]&0xffff,rank=flags[pos];
      a.slotToToken[cb+slot]=unique[pos];a.missTokens[qb+rank]=unique[pos];a.missSlots[qb+rank]=slot;
    }
    for(uint32_t i=t;i<q;i+=SORTED_THREADS){int32_t uid=map[i];a.currentSlots[qb+i]=uid>=0?(owner[uid]&0xffff):-1;}
    for(uint32_t i=misses+t;i<q;i+=SORTED_THREADS){a.missTokens[qb+i]=-1;a.missSlots[qb+i]=-1;}
    const uint32_t unused=evicts-misses;
    for(uint32_t order=t;order<a.capacity;order+=SORTED_THREADS)
      a.lruSlots[cb+order]=order<unused?partition[misses+order]:order<evicts?partition[order-unused]:partition[a.capacity-1U-(order-evicts)];
    const FastGroupSource src=MakeFastGroupSource(a,req);const uint32_t pages=src.Pages(a.capacity);
    for(uint32_t r=0;r<rows;++r){int64_t row=first+r;
      for(uint32_t page=t;page<pages;page+=SORTED_THREADS)a.residentBlockTable[row*pages+page]=static_cast<int32_t>(first*pages+page);
      if(t==0U){a.residentRows[row]=first;int64_t left=static_cast<int64_t>(misses)-r*a.topk;a.missCount[row]=left<=0?0:left<a.topk?left:a.topk;}
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
        if(scratchElements<=PLAN_ROW_UB_LIMIT_ELEMENTS)RunPrivateRow(args,row,scanWorkspace+PLAN_SCAN_STORAGE_ELEMENTS,scanWorkspace,thread);
        else RunPrivateRow(args,row,compactWorkspace+static_cast<uint64_t>(row)*scratchElements,scanWorkspace,thread);
      }
    }else for(int64_t row=rowStart;row<rowStride;++row){
      if(scratchElements<=PLAN_ROW_UB_LIMIT_ELEMENTS)RunPrivateRow(args,row,scanWorkspace+PLAN_SCAN_STORAGE_ELEMENTS,scanWorkspace,thread);
      else RunPrivateRow(args,row,compactWorkspace+static_cast<uint64_t>(row)*scratchElements,scanWorkspace,thread);
    }
  } else {
    __ubuf__ int32_t* ctx=scanWorkspace+PLAN_SCAN_STORAGE_ELEMENTS;
    if(scratchElements<=PLAN_ROW_UB_LIMIT_ELEMENTS){
      __ubuf__ int32_t* data=ctx+64U;
      SortedStage<Phase>(args,rowStart,rowStride,data,ctx,data+((((HW_C&&rowStride-rowStart<=4?static_cast<uint32_t>(rowStride-rowStart):4U)*args.topk+args.capacity+7U)&~7U)),scanWorkspace);
    }else{
      __gm__ int32_t* data=compactWorkspace+static_cast<uint64_t>(rowStart)*scratchElements+64U;
      SortedStage<Phase>(args,rowStart,rowStride,data,ctx,ctx+64U,scanWorkspace);
    }
  }
}
