// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <algorithm>
#include <cstdint>
inline uint64_t LayoutHotScratch(uint64_t capacity,uint64_t privateWords,int mode,int bits){
  uint64_t work=24576U;
  if(mode==2)work=std::max(work,capacity+capacity/2U+3U*capacity/32U+((bits&2U)?capacity/32U:0U));
  return std::max(privateWords,128U+capacity/2U+capacity+work);
}
