// SPDX-License-Identifier: Apache-2.0
// D：替换完整 UB Phase3；只修改临时映射，不访问 GM 或提交持久状态。
// Scatter bases are 32B-aligned (data/ctx); unique is addressed as data[q+uid],
// so odd TopK/capacity layouts do not require an aligned unique pointer.
__simd_vf__ inline void RefineUniqueMapVf(__ubuf__ int32_t* records,
    __ubuf__ int32_t* data,__ubuf__ int32_t* ctx,uint32_t n,uint32_t q,
    uint32_t capacity,uint32_t initialCount,int32_t initialLast,uint32_t more){
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
    Reg::Adds(index,uid,q,full);
    Reg::Adds(safeIndex,lane,q+capacity,full);
    Reg::Select(index,index,safeIndex,writeHead); // even masked indices are unique
    Reg::Scatter<int32_t,uint32_t>(data,token,index,writeHead);
    Reg::Duplicate(payload,uint32_t{0});
    Reg::Gather<uint32_t,uint32_t,uint32_t>(payload,reinterpret_cast<__ubuf__ uint32_t*>(records),odd,valid);
    Reg::Adds(safeIndex,lane,q,full);Reg::Select(payload,payload,safeIndex,valid);
    Reg::Scatter<uint32_t,uint32_t>(reinterpret_cast<__ubuf__ uint32_t*>(data),uid,payload,valid);
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
