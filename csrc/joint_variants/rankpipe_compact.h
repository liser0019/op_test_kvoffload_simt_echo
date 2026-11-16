// SPDX-License-Identifier: Apache-2.0
#pragma once
// B: each thread owns32 positions. Scan word counts, then use local popcount.
// Production512 threads handle all512 words in ONE pass at16K. Outer loops also
// support reduced-thread CPU semantic models, without changing the production tile.
__simt_vf__ __aicore__ LAUNCH_BOUND(SORTED_THREADS) inline void RankPipeCompactVf(
    __ubuf__ int32_t* data,__ubuf__ int32_t* ctx,__ubuf__ int32_t* scan,
    uint32_t c,uint32_t u){
  const uint32_t t=threadIdx.x;
  auto* flags=reinterpret_cast<__ubuf__ uint16_t*>(data);
  auto* owner=data+c/2U+c;
  auto* part=reinterpret_cast<__ubuf__ uint16_t*>(owner+c);
  uint32_t hits=0U,misses=0U,parity=0U;
  for(uint32_t block=0;block<c;block+=32U*SORTED_THREADS){
    const uint32_t begin=block+t*32U;uint32_t hitMask=0U;
    for(uint32_t i=0;i<32U;++i)if(begin+i<c){
      const uint32_t f=flags[begin+i];
      if(f==65535U)asc_atomic_min(ctx,0);
      else if(f&0x4000U)hitMask|=uint32_t{1}<<i;
    }
    int32_t total=0;
    const uint32_t baseHit=hits+SortedScan(scan,__popc(hitMask),t,parity,total);
    for(uint32_t i=0;i<32U;++i)if(begin+i<c){
      const uint32_t f=flags[begin+i],bit=uint32_t{1}<<i;
      const uint32_t before=baseHit+__popc(hitMask&(bit-1U));
      if(f!=65535U){
        const uint16_t entry=static_cast<uint16_t>((f&0x3fffU)|((f&0x8000U)>>1U));
        part[(hitMask&bit)?c-1U-before:begin+i-before]=entry;
      }
    }
    asc_syncthreads(); // all Scan readers and partition writers finish before reuse
    if(ctx[0]==0)return;
    hits+=static_cast<uint32_t>(total);
  }
  const uint32_t evicts=c-hits;
  for(uint32_t block=0;block<u;block+=32U*SORTED_THREADS){
    const uint32_t begin=block+t*32U;uint32_t missMask=0U;
    for(uint32_t i=0;i<32U;++i)if(begin+i<u && owner[begin+i]<0)missMask|=uint32_t{1}<<i;
    int32_t total=0;
    const uint32_t baseMiss=misses+SortedScan(scan,__popc(missMask),t,parity,total);
    for(uint32_t i=0;i<32U;++i){const uint32_t bit=uint32_t{1}<<i;
      if(missMask&bit){const uint32_t rank=baseMiss+__popc(missMask&(bit-1U));
        if(rank<evicts)owner[begin+i]=static_cast<int32_t>(0x40000000U|(rank<<16U)|(part[rank]&0x3fffU));}
    }
    asc_syncthreads();misses+=static_cast<uint32_t>(total);
  }
  if(t==0){ctx[43]=evicts;ctx[44]=hits;ctx[45]=misses;ctx[0]=misses<=evicts?1:0;}
  asc_syncthreads();
}
