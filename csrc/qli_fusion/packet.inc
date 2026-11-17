// C flushes at most 2048 newly unique records per pass. Importantly, this is
// DURING merge, not a spill after materializing a capacity-sized UB unique list.
__aicore__ inline void FusionRefinePacket(__ubuf__ int32_t* records,
    __ubuf__ int32_t* data,__ubuf__ int32_t* ctx,uint32_t n,uint32_t q,
    uint32_t c,uint32_t topk,uint32_t more,bool fused,int64_t first,
    __gm__ int32_t* workspace,int64_t requests,AscendC::TPipe& pipe,AscendC::LocalTensor<int32_t> local){
  const uint32_t step=QLI_PLAN_STORAGE==3?2048U:n;
  if(n==0U)return;
  for(uint32_t base=0;base<n;base+=step){
    const uint32_t count=n-base<step?n-base:step,old=ctx[1];
    const uint32_t packetMore=base+count<n?1U:more;
    if(fused){
      if constexpr(QLI_PLAN_STORAGE!=3){
        if(ctx[108]==2)asc_vf_call<RefineFusedTuneVf<3U,2>>(records+2U*base,data,ctx,count,q,c,old,ctx[2],topk,ctx[109],ctx[110],c/2U);
        else if(ctx[108]==0)asc_vf_call<RefineFusedTuneVf<3U,0>>(records+2U*base,data,ctx,count,q,c,old,ctx[2],topk,ctx[109],ctx[110],c/2U);
        else if(ctx[108]==1)asc_vf_call<RefineFusedTuneVf<3U,1>>(records+2U*base,data,ctx,count,q,c,old,ctx[2],topk,ctx[110],ctx[109],c/2U);
        else asc_vf_call<RefineFusedTuneVf<3U,-1>>(records+2U*base,data,ctx,count,q,c,old,ctx[2],topk,0U,0U,c/2U);
      }else{
        asc_vf_call<RefineFusedTuneVf<3U,-1>>(records+2U*base,data,ctx,count,q,c,old,ctx[2],topk,0U,0U,c/2U);
      }
    }else asc_vf_call<RefineUniqueMapVf>(records+2U*base,data,ctx,count,q,c,old,ctx[2],packetMore,c/2U);
    SortedVectorToScalar(pipe);
    if(ctx[0]!=1)return;
    if constexpr(QLI_PLAN_STORAGE==3){
      const uint32_t delta=static_cast<uint32_t>(ctx[1])-old;
      if(delta){
        auto source=local[PLAN_SCAN_STORAGE_ELEMENTS+EIGHT_CTX+c/2U];
        AscendC::GlobalTensor<int32_t> target;
        target.SetGlobalBuffer(workspace+qli_fusion_resource::MapWords(requests,topk)+first*c+old,delta);
        auto event=pipe.AllocEventID<AscendC::HardEvent::V_MTE3>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event);AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event);
        pipe.ReleaseEventID<AscendC::HardEvent::V_MTE3>(event);
        AscendC::DataCopyExtParams copy(1,delta*4U,0,0,0);AscendC::DataCopyPad(target,source,copy);
        // WAR: this bounded staging will be overwritten by the next VF.
        event=pipe.AllocEventID<AscendC::HardEvent::MTE3_V>();
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event);AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event);
        pipe.ReleaseEventID<AscendC::HardEvent::MTE3_V>(event);
      }
    }
  }
}
