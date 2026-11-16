// SPDX-License-Identifier: Apache-2.0
#pragma once
// B/stage: uint16 storage, uint32 register masks/ranks. Stable partition and miss scan.
// API: CANN9.1 Reg::ShiftRights scalar=int16_t; packed B16 Scatter follows parent.
__simd_callee__ inline void LayoutPut(__ubuf__ int32_t* ctx,
    AscendC::Reg::RegTensor<uint32_t>& value,AscendC::Reg::RegTensor<uint32_t>& lane,
    uint32_t at,AscendC::Reg::MaskReg& full){
  AscendC::Reg::MaskReg chosen;AscendC::Reg::Compares<uint32_t,AscendC::CMPMODE::EQ>(chosen,lane,at,full);
  AscendC::Reg::Scatter<uint32_t,uint32_t>(reinterpret_cast<__ubuf__ uint32_t*>(ctx),value,lane,chosen);
}
__simd_vf__ inline void LayoutHotCompactVf(__ubuf__ int32_t* data,__ubuf__ int32_t* ctx,
    uint32_t c,uint32_t u){
  using namespace AscendC;
  const uint32_t ow=c/2U+c,pa=ow+c;
  Reg::RegTensor<uint32_t> lane,zero,one,idx,oddBit,packed,high,f,mask,slot,invalid;
  Reg::RegTensor<uint32_t> hf,ef,hp,ep,hc,ec,hi,ei,last,entry,safe,tmp,mc,rank,inc,owner,index16,status;
  Reg::RegTensor<int32_t> signedOwner;
  Reg::MaskReg full=Reg::CreateMask<uint32_t,Reg::MaskPattern::ALL>(),odd,valid,hit,evict,selected,fits;
  Reg::Unsqueeze<uint32_t>(lane,full);Reg::Duplicate(zero,uint32_t{0});Reg::Duplicate(one,uint32_t{1});
  Reg::Duplicate(hc,uint32_t{0});Reg::Duplicate(ec,uint32_t{0});Reg::Duplicate(last,uint32_t{63});
  Reg::Duplicate(mask,uint32_t{1});Reg::And(oddBit,lane,mask,full);Reg::Compares<uint32_t,CMPMODE::NE>(odd,oddBit,uint32_t{0},full);
  for(uint32_t base=0;base<c;base+=64U){
    Reg::ShiftRights(idx,lane,int16_t{1},full);Reg::Adds(idx,idx,base/2U,full);
    Reg::Gather<uint32_t,uint32_t,uint32_t>(packed,reinterpret_cast<__ubuf__ uint32_t*>(data),idx,full);
    Reg::ShiftRights(high,packed,int16_t{16},full);Reg::Select(f,high,packed,odd);
    Reg::Duplicate(mask,uint32_t{65535});Reg::And(f,f,mask,full);
    Reg::Compares<uint32_t,CMPMODE::NE>(valid,f,uint32_t{65535},full);
    Reg::Duplicate(mask,uint32_t{0x4000});Reg::And(tmp,f,mask,full);Reg::Compares<uint32_t,CMPMODE::NE>(hit,tmp,uint32_t{0},valid);
    Reg::Select(hf,one,zero,hit);Reg::Select(ef,one,zero,valid);Reg::Sub(ef,ef,hf,full);
    Reg::Compares<uint32_t,CMPMODE::NE>(evict,ef,uint32_t{0},full);
    Reg::Unsqueeze<uint32_t>(hp,hit);Reg::Unsqueeze<uint32_t>(ep,evict);
    Reg::Add(hi,hp,hc,full);Reg::Add(ei,ep,ec,full);
    Reg::Duplicate(tmp,c-1U);Reg::Sub(idx,tmp,hi,full);Reg::Select(idx,idx,ei,hit);
    Reg::Duplicate(mask,uint32_t{0x3fff});Reg::And(entry,f,mask,full);
    Reg::Duplicate(mask,uint32_t{0x8000});Reg::And(invalid,f,mask,full);Reg::ShiftRights(invalid,invalid,int16_t{1},full);Reg::Or(entry,entry,invalid,full);
    // Only low halfwords selected by the B32 mask; inactive high-half indices safe.
    Reg::Adds(idx,idx,2U*c,full);Reg::Select(idx,idx,lane,valid);
    Reg::Adds(safe,lane,uint32_t{64},full);Reg::ShiftLefts(index16,safe,int16_t{16},full);Reg::Or(index16,index16,idx,full);
    Reg::Scatter<uint16_t,uint16_t>(reinterpret_cast<__ubuf__ uint16_t*>(data+ow),
      reinterpret_cast<Reg::RegTensor<uint16_t>&>(entry),reinterpret_cast<Reg::RegTensor<uint16_t>&>(index16),valid);
    Reg::Add(hi,hi,hf,full);Reg::Add(ei,ei,ef,full);Reg::Gather<uint32_t,uint32_t>(hc,hi,last);Reg::Gather<uint32_t,uint32_t>(ec,ei,last);
  }
  Reg::LocalMemBar<Reg::MemType::VEC_ALL,Reg::MemType::VEC_ALL>();
  Reg::Duplicate(mc,uint32_t{0});
  for(uint32_t base=0;base<u;base+=64U){
    uint32_t remaining=u-base;valid=Reg::UpdateMask<uint32_t>(remaining);
    Reg::Adds(idx,lane,ow+base,full);Reg::Duplicate(signedOwner,int32_t{0});
    Reg::Gather<int32_t,int32_t,uint32_t>(signedOwner,data,idx,valid);
    Reg::Compares<int32_t,CMPMODE::LT>(selected,signedOwner,int32_t{0},valid);
    Reg::Select(inc,one,zero,selected);Reg::Unsqueeze<uint32_t>(rank,selected);Reg::Add(rank,rank,mc,full);
    Reg::Compare<uint32_t,CMPMODE::LT>(fits,rank,ec,selected);
    Reg::ShiftRights(idx,rank,int16_t{1},full);Reg::Adds(idx,idx,pa,full);
    Reg::Duplicate(tmp,pa);Reg::Select(idx,idx,tmp,fits);Reg::Duplicate(packed,uint32_t{0});
    Reg::Gather<uint32_t,uint32_t,uint32_t>(packed,reinterpret_cast<__ubuf__ uint32_t*>(data),idx,fits);
    Reg::Duplicate(mask,uint32_t{1});Reg::And(tmp,rank,mask,full);Reg::Compares<uint32_t,CMPMODE::NE>(odd,tmp,uint32_t{0},full);
    Reg::ShiftRights(high,packed,int16_t{16},full);Reg::Select(slot,high,packed,odd);
    Reg::Duplicate(mask,uint32_t{0x3fff});Reg::And(slot,slot,mask,full);
    Reg::ShiftLefts(owner,rank,int16_t{16},full);Reg::Or(owner,owner,slot,full);Reg::Duplicate(mask,uint32_t{0x40000000});Reg::Or(owner,owner,mask,full);
    Reg::Adds(idx,lane,ow+base,full);Reg::Scatter<uint32_t,uint32_t>(reinterpret_cast<__ubuf__ uint32_t*>(data),owner,idx,fits);
    Reg::Add(tmp,rank,inc,full);Reg::Gather<uint32_t,uint32_t>(mc,tmp,last);
  }
  Reg::Add(tmp,ec,hc,full);Reg::Compares<uint32_t,CMPMODE::EQ>(valid,tmp,c,full);
  Reg::Adds(tmp,ec,uint32_t{1},full);Reg::Compare<uint32_t,CMPMODE::LT>(selected,mc,tmp,valid);
  Reg::Select(status,one,zero,selected);
  LayoutPut(ctx,ec,lane,43,full);LayoutPut(ctx,hc,lane,44,full);LayoutPut(ctx,mc,lane,45,full);LayoutPut(ctx,status,lane,0,full);
}
