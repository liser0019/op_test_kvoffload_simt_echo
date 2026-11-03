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
      // 本轮C：仅实际单行组走融合VF，组数与协议仍由设备输入决定。
      if(EXEC_FUSE && end-first==1){
        SORTED_CALL(6);SortedVectorToScalar(pipe);
        if(ctx[0]!=2){SORTED_CALL(4);}
      }else{
      SORTED_CALL(0);SortedVectorToScalar(pipe);
      while(ctx[0]==1&&(!HW_A||ctx[6])){
        SORTED_CALL(1);SortedVectorToScalar(pipe);if(ctx[28]==0)break;
#if JOINT_OPT_MASK >= 39
        if(HW_A&&ctx[28]==1){
          ctx[4]=ctx[20];ctx[24]=ctx[20];ctx[8+ctx[16]]+=ctx[20];ctx[7]=1;
          SortedScalarToVector(pipe);
        }else
        if(ctx[28]>1){
          const uint32_t chunk=ctx[3];
          const uint32_t inputOffset=d.scratchElements<=PLAN_ROW_UB_LIMIT_ELEMENTS?
              PLAN_SCAN_STORAGE_ELEMENTS+64U+((((HW_C?static_cast<uint32_t>(end-first):4U)*d.topk+d.capacity+7U)&~7U)):PLAN_SCAN_STORAGE_ELEMENTS+64U;
          auto floats=storage.Get<float>();auto out=floats[inputOffset+static_cast<uint32_t>(ctx[30])];
          AscendC::MrgSortSrcList<float> lists{floats[inputOffset],floats[inputOffset+2U*chunk],
              floats[inputOffset+(ctx[28]>2?4U*chunk:0U)],floats[inputOffset+(ctx[28]>3?6U*chunk:0U)]};
          uint16_t lens[4]={0,0,0,0};uint32_t consumed[4]={0,0,0,0};
          for(int k=0;k<ctx[28];++k)lens[k]=static_cast<uint16_t>(ctx[20+k]);
          bool complete=HW_C;
          for(int k=0;k<ctx[28];++k)if(ctx[20+k]!=ctx[12+ctx[16+k]]-ctx[8+ctx[16+k]])complete=false;
          if(complete){
            AscendC::MrgSort<float,false>(out,lists,lens,consumed,static_cast<uint16_t>((1U<<ctx[28])-1U),1);
            for(int k=0;k<ctx[28];++k)consumed[k]=lens[k];
          }else AscendC::MrgSort<float,true>(out,lists,lens,consumed,static_cast<uint16_t>((1U<<ctx[28])-1U),1);
          SortedVectorToScalar(pipe);
          uint32_t total=0;
          for(int k=0;k<ctx[28];++k){if(consumed[k]>lens[k]){ctx[0]=0;break;}total+=consumed[k];ctx[24+k]=consumed[k];ctx[8+ctx[16+k]]+=consumed[k];}
          if(total==0U)ctx[0]=0;ctx[4]=static_cast<int32_t>(total);
          SortedScalarToVector(pipe);
        }else { SORTED_CALL(2); }
#else
        SORTED_CALL(2);
#endif
#if (JOINT_REFINE_BITS & 8)
          if(d.scratchElements<=PLAN_ROW_UB_LIMIT_ELEMENTS && ctx[0]==1){
            const uint32_t q=static_cast<uint32_t>(end-first)*static_cast<uint32_t>(d.topk);
            const uint32_t inputOffset=PLAN_SCAN_STORAGE_ELEMENTS+64U+((q+d.capacity+7U)&~7U);
            auto* records=scan+inputOffset+(ctx[7]?0U:static_cast<uint32_t>(ctx[30]));
            uint32_t more=0;for(int64_t r=0;r<end-first;++r)if(ctx[8+r]<ctx[12+r])more=1;
            asc_vf_call<ExecUniqueMapVf>(records,ctx+64U,ctx,
                static_cast<uint32_t>(ctx[4]),q,static_cast<uint32_t>(d.capacity),
                static_cast<uint32_t>(ctx[1]),ctx[2],more);
          }else {SORTED_CALL(3);} // GM scratch retains SIMT, including B if enabled.
#else
        SORTED_CALL(3);
#endif
        SortedVectorToScalar(pipe);
      }
      if(ctx[0]==1){SORTED_CALL(5);SortedVectorToScalar(pipe);}
      if(ctx[0]!=2){SORTED_CALL(4);}
      }
    }
    ++group;first=end;
  }
#undef SORTED_CALL
}
