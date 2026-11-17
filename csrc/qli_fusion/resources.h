// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
// Shared by host, torch binding, CPU checks and device. Bytes include Scan/ctx.
// 0 = ordered Indexer only; 1 = representative16 + bitmap;
// 2 = representative16 + directory/binary; 3 = streamed unique32 + bitmap.
#ifndef QLI_PLAN_STORAGE
#define QLI_PLAN_STORAGE 1
#endif
namespace qli_fusion_resource {
constexpr uint32_t kScanWords=72, kContextWords=128, kWorkWords=24576;
constexpr uint32_t UniqueWords(uint32_t c) { return QLI_PLAN_STORAGE==3?2048U:c/2U; }
constexpr uint32_t OwnerOffset(uint32_t c) { return c/2U+UniqueWords(c); }
constexpr uint32_t ScratchWords(uint32_t c) { return kContextWords+OwnerOffset(c)+kWorkWords; }
constexpr uint32_t LocalBytes(uint32_t c) { return QLI_PLAN_STORAGE==0?0U:4U*(kScanWords+ScratchWords(c)); }
constexpr uint64_t MapWords(uint64_t requests,uint32_t topk) { return requests*8U*topk/2U; }
constexpr uint64_t UniqueGmWords(uint64_t requests,uint32_t c) { return requests*8U*c; }
constexpr uint64_t PrivateWords(uint32_t c,uint32_t t) { return 3U*4096U+c+t; }
constexpr uint64_t WorkspaceWords(uint64_t requests,uint32_t c,uint32_t t) {
  return MapWords(requests,t)+UniqueGmWords(requests,c)+requests*8U*PrivateWords(c,t);
}
static_assert(LocalBytes(16384)<=216U*1024U,"Plan fits default mixed-mode UB budget");
}
