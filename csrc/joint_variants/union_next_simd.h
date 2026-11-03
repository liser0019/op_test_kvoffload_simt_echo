// SPDX-License-Identifier: Apache-2.0
// A: 64-lane SIMD inclusive head-prefix. No scalar extraction inside this VF.
// Keep score words intact; pack packet-local inclusive UID into payload[31:16].
// qIdx occupies [15:0] (Phase0 rejects q>65535). Unlike Squeeze alone, this
// retains a UID for EVERY original query, including repeats and packet seams.
__simd_vf__ inline void NextPrefixVf(__ubuf__ int32_t* packet,__ubuf__ uint32_t* lanes,uint32_t n,int32_t lastToken){
  using namespace AscendC;
  Reg::RegTensor<uint32_t> lane,even,odd,prevIndex,lastIndex;
  Reg::RegTensor<uint32_t> zero,one,flags,prefix,shifted,shiftIndex,selected;
  Reg::RegTensor<uint32_t> carry,payload,packed;
  Reg::RegTensor<float> score,previous,lastScore,shiftScore;
  Reg::MaskReg full=Reg::CreateMask<uint32_t,Reg::MaskPattern::ALL>();
  Reg::MaskReg first,head,ge,valid;
  Reg::LoadAlign(lane,lanes);
  Reg::Duplicate(zero,uint32_t{0});Reg::Duplicate(one,uint32_t{1});Reg::Duplicate(carry,uint32_t{0});
  Reg::Compares<uint32_t,CMPMODE::EQ>(first,lane,uint32_t{0},full);
  Reg::Adds(prevIndex,lane,uint32_t(-1),full);
  Reg::Duplicate(lastScore,-static_cast<float>(lastToken));
  for(uint32_t base=0;base<n;base+=64U){
    uint32_t remaining=n-base;valid=Reg::UpdateMask<uint32_t>(remaining);
    const uint32_t count=n-base<64U?n-base:64U;
    Reg::Adds(even,lane,base,full);Reg::ShiftLefts(even,even,int16_t{1},full);
    Reg::Adds(odd,even,uint32_t{1},full);
    Reg::Duplicate(score,0.0f);
    Reg::Gather<float,float,uint32_t>(score,reinterpret_cast<__ubuf__ float*>(packet),even,valid);
    Reg::Gather<float,uint32_t>(shiftScore,score,prevIndex);
    Reg::Select(previous,lastScore,shiftScore,first);
    Reg::Compare<float,CMPMODE::NE>(head,score,previous,valid);
    Reg::Select(flags,one,zero,head);
    // Explicitly zero inactive lanes; compare's inactive mask bits are not assumed.
    Reg::Select(prefix,flags,zero,valid);
    for(uint32_t distance=1U;distance<64U;distance<<=1U){
      Reg::Adds(shiftIndex,lane,uint32_t(0U-distance),full);
      Reg::Gather<uint32_t,uint32_t>(shifted,prefix,shiftIndex);
      Reg::Compares<uint32_t,CMPMODE::GE>(ge,lane,distance,full);
      Reg::Select(selected,shifted,zero,ge);
      Reg::Add(prefix,prefix,selected,full);
    }
    Reg::Add(prefix,prefix,carry,full);
    Reg::Duplicate(lastIndex,uint32_t{63});Reg::Gather<uint32_t,uint32_t>(carry,prefix,lastIndex);
    Reg::Duplicate(lastIndex,count-1U);Reg::Gather<float,uint32_t>(lastScore,score,lastIndex);
    Reg::Duplicate(payload,uint32_t{0});
    Reg::Gather<uint32_t,uint32_t,uint32_t>(payload,reinterpret_cast<__ubuf__ uint32_t*>(packet),odd,valid);
    Reg::ShiftLefts(packed,prefix,int16_t{16},full);Reg::Or(packed,packed,payload,full);
    Reg::Scatter<uint32_t,uint32_t>(reinterpret_cast<__ubuf__ uint32_t*>(packet),packed,odd,valid);
  }
}
