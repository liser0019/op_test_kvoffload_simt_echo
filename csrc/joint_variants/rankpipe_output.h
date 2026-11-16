// SPDX-License-Identifier: Apache-2.0
#pragma once
// C kernel-side consumer. Producer is SIMT/V; GM map remains read by its original
// SIMT writer, not by Scalar/MTE. This helper only copies newly staged UB outputs.
__aicore__ inline void RankPipeCopyOutputs(AscendC::TPipe& pipe,
    AscendC::LocalTensor<int32_t> local,__ubuf__ int32_t* ctx,
    int64_t first,uint32_t topk,uint32_t c,GM_ADDR lruSlots,GM_ADDR missTokens,GM_ADDR missSlots){
  const uint32_t n=static_cast<uint32_t>(ctx[46]),misses=static_cast<uint32_t>(ctx[47]);
  const uint32_t uniqueOffset=PLAN_SCAN_STORAGE_ELEMENTS+EIGHT_CTX+c/2U;
  const uint32_t ownerOffset=uniqueOffset+c;
  AscendC::GlobalTensor<int32_t> lru,mt,ms;
  lru.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(lruSlots)+first*c,c);
  const int64_t tail=first*static_cast<int64_t>(topk)+misses;
  mt.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(missTokens)+tail,n);
  ms.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(missSlots)+tail,n);
  // Explicit SIMT/V -> MTE3 visibility; V_S only publishes the control values.
  auto ready=pipe.AllocEventID<AscendC::HardEvent::V_MTE3>();
  AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ready);
  AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ready);
  pipe.ReleaseEventID<AscendC::HardEvent::V_MTE3>(ready);
  AscendC::DataCopyExtParams lruCopy(1,c*4U,0,0,0);
  AscendC::DataCopyPad(lru,local[ownerOffset],lruCopy);
  if(n){
    AscendC::DataCopyExtParams tailCopy(1,n*4U,0,0,0);
    AscendC::DataCopyPad(mt,local[uniqueOffset],tailCopy);
    AscendC::DataCopyPad(ms,local[uniqueOffset],tailCopy);
  }
  // Both sources stay intact until ALL output copies finish. Includes the group
  // tail and kernel tail; no next VF/group may reuse these UB regions early.
  auto done=pipe.AllocEventID<AscendC::HardEvent::MTE3_V>();
  AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(done);
  AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(done);
  pipe.ReleaseEventID<AscendC::HardEvent::MTE3_V>(done);
}
