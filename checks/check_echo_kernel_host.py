#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Echo v2（维护序）回归：提取实际 C++ 函数，g++/128 个 CPU 线程检查淘汰扫描与顺序重建。

覆盖 v2 的设备算法改动（详见 kernel 头注释与 deep_dive §9.17）：
  - OpEvictScanRow：位置序扫 lru、跳过本步 hit（v1 protect 语义等价物）取前 fs 个
    非 hit 槽，输出序 = (pri,slot) 字典序，与 v1 emission 逐位一致；padding/尾零
    语义与 v1 全同；
  - OpLruRebuildRow：新 lru = [幸存者(位置序)] ++ [hit 槽号升序] ++ [分配槽号升序]
    ++ 尾零——与 v1 步末 (pri,slot) 稳定排序逐位等价；
  - 跨验证组：随机场景下按 v1 pri 语义独立推导 golden（stable_sort by (pri,slot)），
    与 v2 链（EvictScan+Rebuild）逐位比对——v2 与 v1 语义等价性的 CPU 级证明；
  - OpHitMiss：hit → flags bit0（v2 替代 pri 时间戳）、warp 归约计数、[P4] freeSize
    折叠、[C1] 淘汰标志、零 atomic；
  - OpMissCumsum：两级 warp 扫描（[P3]，v2 未改动，原样回归）；
  - OpAllocPop + OpFifoAvailUpdate：直读 freeIdx + 老栈等价（[P4] 原样回归）；
    fifo 两次自增合并为 +2（[v2]）。

依赖 Python 标准库和支持 C++20 的 g++；启用 AddressSanitizer/UBSan。
这是 CPU 算法和索引检查，不能验证 Ascend 编译、GM D-cache 或 NPU 耗时。
不导入 torch，不参与 compare.py 正式计时，临时构建产物自动删除。
"""
from pathlib import Path
import tempfile, subprocess, re
root=Path(__file__).resolve().parents[1]
# 128 个 CPU 线程模拟一个 block（blockDim.x=128→4 warp；warp 协作辅助按
# blockDim.x 运行时推导 warp 数）。断言模型常量与实际头文件一致，防止以后
# 修改 tiling 后误用旧模型。
tiling=(root/'csrc/echo/echo_lru_tiling.h').read_text()
for name,value in [('ARGTOPK_GROUP_THREADS',64),('ARGTOPK_UB_NPAD_MAX',8192),
                   ('ARGTOPK_GM_NPAD_MAX',8192),('ECHO_PERF_NUM_STEPS',26)]:
    match=re.search(r'constexpr uint32_t '+name+r'\s*=\s*(\d+)\s*;',tiling)
    assert match and int(match.group(1))==value, f'{name} 已改变，请同步调整 CPU 模型'
src=(root/'csrc/echo/echo_lru_kernel.h').read_text()
def extract(name):
    for prefix in ('__aicore__ inline void ','__aicore__ inline int32_t ',
                    '__attribute__((noinline)) __aicore__ void '):
        tag=prefix+name+'('
        if tag in src:
            start=src.index(tag)
            end=src.index('\n}\n',start)+3
            return src[start:end]
    raise KeyError(name)
# ---- 整文件结构审计 (2026-10-19 教训: v2 Init 结尾多一个 '}' 把 class 提前关闭,
#      bisheng 全文件编译才暴露, 而逐函数抽取的 mock 编译抓不到 —— 此处在 CPU
#      侧先拦住这类问题): ① 大括号平衡; ② class 体完整 (Init/Process/两个行循环
#      wrapper 都在 class 开闭之间); ③ 入口函数在 class 之后。----
_open,_close=src.count('{'),src.count('}')
assert _open==_close, f'kernel.h 大括号不平衡: {{={_open} }}={_close}'
_cls=src.index('class EchoLruKernel {')
_cls_end=src.index('\n};\n',_cls)
for _m in ('void Init(','void Process(','void EvictScanAllRows(','void LruRebuildAllRows('):
    assert _cls<src.index(_m,_cls)<_cls_end, f'{_m} 不在 class 体内 —— 作用域被破坏'
assert src.index('echo_lru_kernel(')>_cls_end, '入口函数应在 class 之后'
functions=(extract('PackedBlockScan')+'\n'+extract('OpHitMiss')+'\n'+extract('OpEvictScanRow')
           +'\n'+extract('OpLruRebuildRow')+'\n'+extract('OpMissCumsum')+'\n'+extract('OpAllocPop')
           +'\n'+extract('OpFifoAvailUpdate'))
preamble=r'''
#include <algorithm>
#include <atomic>
#include <barrier>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>
#define __aicore__
struct Dim {uint32_t x;};
thread_local Dim threadIdx{0};
Dim blockIdx{0}, blockDim{128}, gridDim{1};
std::barrier<> syncBlock(128);
constexpr int32_t I32_MAX=2147483647;
constexpr uint32_t ARGTOPK_GROUP_THREADS=64, ARGTOPK_UB_NPAD_MAX=8192;
std::atomic<uint32_t> atomicCalls{0};
inline void asc_syncthreads(){syncBlock.arrive_and_wait();}
inline void asc_dcci_entire(void*){}
inline int32_t asc_atomic_add(int32_t* p,int32_t v){atomicCalls.fetch_add(1);return std::atomic_ref<int32_t>(*p).fetch_add(v);}
// warp 原语 CPU 模拟: 设备代码在统一控制流里调用 (无 warp 内发散), 用共享缓冲 +
// 双栅栏交换即可精确复现语义。asc_shfl_up: lane<delta 时返回自身值 (与 CUDA 一致,
// 基线的 lane>=offset 判断依赖此语义)。asc_reduce_add: warp 内求和返回全 lane。
int32_t warpBuf[128];
inline int32_t asc_shfl_up(int32_t var,uint32_t delta,int32_t){
    warpBuf[threadIdx.x]=var;
    syncBlock.arrive_and_wait();
    uint32_t lane=threadIdx.x&31u;
    int32_t r=(lane>=delta)?warpBuf[threadIdx.x-delta]:var;
    syncBlock.arrive_and_wait();
    return r;
}
inline int32_t asc_reduce_add(int32_t val){
    warpBuf[threadIdx.x]=val;
    syncBlock.arrive_and_wait();
    uint32_t base=(threadIdx.x>>5)<<5;
    int32_t s=0;
    for(uint32_t i=base;i<base+32u;++i)s+=warpBuf[i];
    syncBlock.arrive_and_wait();
    return s;
}
template<typename F> void launch(F fn){
    std::vector<std::thread> threads;
    for(uint32_t t=0;t<blockDim.x;++t) threads.emplace_back([=]{threadIdx.x=t;fn();});
    for(auto& t:threads)t.join();
}
void require(bool cond,const char* msg){if(!cond)throw std::runtime_error(msg);}
'''
harness=r'''
int main(){
try{
    int checks=0;
    // ---- 1) OpEvictScanRow: 位置序淘汰扫描 (跳过本步 hit, 输出序 = (pri,slot) 序) ----
    // hitMode: 0=随机分布 hit; 1=hit 集中在最旧端 (必须跳过, 选中靠后的非 hit 槽)。
    //   fsClamp ≥ 1 (fs=0 行由调用方 wrapper 跳过, 不进入本函数)。
    {
        struct EvictCase { uint32_t B, occ, fsClamp, hitMode, seed; };
        const EvictCase cases[]={
            {64,40,10,0,1},{64,40,30,1,2},{31,31,7,0,3},{256,200,64,1,4},
            {4096,4096,512,0,5},{4096,3000,512,1,6},{64,40,40,0,7},   // fs=全部非 hit
            {64,10,20,0,8},                                            // fs>非 hit 总数 → padding
            {128,128,64,1,9},{1024,513,100,0,10},{64,1,1,0,11},{512,0,1,0,12},  // occ=0 全 padding
            {4096,2048,1,0,13},{96,95,48,1,14},{2048,2000,777,0,15},  // fs 超大 → 大量 padding
        };
        for(size_t ci=0;ci<sizeof(cases)/sizeof(cases[0]);++ci){
            const auto& c=cases[ci];
            uint32_t B=c.B,occ=c.occ,fsClamp=c.fsClamp,topkDim=B;
            printf("[evict %zu] B=%u occ=%u fs=%u hitMode=%u\n",ci,B,occ,fsClamp,c.hitMode);fflush(stdout);
            std::mt19937 rng(c.seed);
            std::vector<int32_t> lruRow(B+1,0),flagsRow(B+1,0);
            std::vector<int32_t> slots(B);std::iota(slots.begin(),slots.end(),1);
            std::shuffle(slots.begin(),slots.end(),rng);
            for(uint32_t p=0;p<occ;++p)lruRow[p]=slots[p];
            std::vector<int32_t> hitFlags(B+1,0);
            uint32_t hitCount=0;
            if(occ>0){
                hitCount=(c.hitMode==1)?std::min<uint32_t>(occ,occ/3):occ/4;
                for(uint32_t k=0;k<hitCount;++k){
                    uint32_t p=(c.hitMode==1)?k:(rng()%occ);
                    hitFlags[lruRow[p]]=1;
                }
            }
            for(uint32_t s=1;s<=B;++s)flagsRow[s]=hitFlags[s];   // bit0
            // golden: 位置序非 hit 槽; 前 fsClamp 个被淘汰; 不够补 [B+1,..); 尾零。
            std::vector<int32_t> nonHit;
            for(uint32_t p=0;p<occ;++p)if(!hitFlags[lruRow[p]])nonHit.push_back(lruRow[p]);
            std::vector<int32_t> expected(nonHit.begin(),nonHit.begin()+std::min<uint32_t>(fsClamp,(uint32_t)nonHit.size()));
            for(uint32_t k=0;expected.size()<fsClamp;++k)expected.push_back((int32_t)(B+1+k));
            std::vector<int32_t> expFlags=flagsRow;
            for(uint32_t j=0;j<fsClamp&&j<(uint32_t)nonHit.size();++j)expFlags[nonHit[j]]=2;
            std::vector<int32_t> lruGm=lruRow,flagsGm=flagsRow;
            std::vector<int32_t> freeIdx(topkDim+8,-777);
            std::vector<int32_t> flagsUb(B+1+8,-777),scan(64+8,-777);
            launch([&]{OpEvictScanRow(B,topkDim,fsClamp,occ,lruGm.data(),0,
                                      flagsGm.data(),0,freeIdx.data(),flagsUb.data(),scan.data());});
            for(uint32_t j=0;j<topkDim;++j){
                int32_t e=(j<(uint32_t)expected.size())?expected[j]:0;
                require(freeIdx[j]==e,"evict freeIdx mismatch");
            }
            for(uint32_t j=topkDim;j<freeIdx.size();++j)require(freeIdx[j]==-777,"freeIdx overrun");
            for(uint32_t s=0;s<=B;++s)require(flagsGm[s]==expFlags[s],"evict flags mismatch");
            for(uint32_t i=B+1;i<flagsUb.size();++i)require(flagsUb[i]==-777,"flagsUb staging overrun");
            for(uint32_t p=0;p<=B;++p)require(lruGm[p]==lruRow[p],"lru must not be written by EvictScan");
            ++checks;
        }
    }
    // ---- 2) OpLruRebuildRow: 顺序重建 golden ----
    // hitMode: 0=随机 hit; 1=头部 hit。evictMode: 0=不淘汰; 1=随机淘汰 1/3;
    //   2=淘汰数=分配数且分配列表 = 被淘汰槽 (淘汰后重分配同槽: 旧位置丢弃 + 尾部追加)。
    {
        struct RbCase { uint32_t B, occ, hitMode, evictMode, allocCount, seed; };
        const RbCase cases[]={
            {64,40,0,0,8,21},{64,40,1,1,8,22},{4096,4096,0,1,512,23},
            {64,64,1,0,0,24},{64,0,0,0,16,25},{128,96,1,1,32,26},
            {31,31,0,0,4,27},{256,128,1,1,64,28},{64,40,0,2,8,29},
            {2048,1500,1,1,300,30},{96,0,0,0,96,31},
        };
        for(size_t ci=0;ci<sizeof(cases)/sizeof(cases[0]);++ci){
            const auto& c=cases[ci];
            uint32_t B=c.B,occ=c.occ,topkDim=B;
            printf("[rebuild %zu] B=%u occ=%u hit=%u evict=%u nm=%u\n",ci,B,occ,c.hitMode,c.evictMode,c.allocCount);fflush(stdout);
            std::mt19937 rng(c.seed);
            std::vector<int32_t> lruRow(B+1,0);
            std::vector<int32_t> slots(B);std::iota(slots.begin(),slots.end(),1);
            std::shuffle(slots.begin(),slots.end(),rng);
            for(uint32_t p=0;p<occ;++p)lruRow[p]=slots[p];
            std::vector<int32_t> hitFlags(B+1,0),evictFlags(B+1,0);
            uint32_t hitCount=(occ>0)?((c.hitMode==1)?std::min<uint32_t>(occ,occ/2):occ/4):0;
            for(uint32_t k=0;k<hitCount;++k){
                uint32_t p=(c.hitMode==1)?k:(rng()%occ);
                hitFlags[lruRow[p]]=1;
            }
            std::vector<int32_t> nonHit;
            for(uint32_t p=0;p<occ;++p)if(!hitFlags[lruRow[p]])nonHit.push_back(lruRow[p]);
            uint32_t evCount=0;
            if(c.evictMode==1)evCount=(uint32_t)nonHit.size()/3;
            if(c.evictMode==2)evCount=std::min<uint32_t>((uint32_t)nonHit.size(),c.allocCount);
            for(uint32_t j=0;j<evCount;++j)evictFlags[nonHit[j]]=1;
            std::vector<int32_t> flagsRow(B+1,0);
            for(uint32_t s=1;s<=B;++s)flagsRow[s]=(hitFlags[s]?1:0)|(evictFlags[s]?2:0);
            // 分配列表: 先包含全部被淘汰槽, 再从空槽池补足 —— 尊重真实管线不变量
            //   evicted(=fs) ≤ nm (fs = max(nm-avail,0)); 该不变量是 v2 尾零删除
            //   (occNew ≥ occOld ⟹ 尾区 ⊆ 旧零尾) 的前提 (2026-10-19 [rebuild 2] 抓获)。
            std::vector<int32_t> inLru(B+1,0);for(uint32_t p=0;p<occ;++p)inLru[lruRow[p]]=1;
            std::vector<int32_t> freePool;for(uint32_t s=1;s<=B;++s)if(!inLru[s])freePool.push_back((int32_t)s);
            std::shuffle(freePool.begin(),freePool.end(),rng);
            std::vector<int32_t> allocList;
            for(uint32_t j=0;j<evCount;++j)allocList.push_back(nonHit[j]);
            for(uint32_t j=0;allocList.size()<c.allocCount&&j<(uint32_t)freePool.size();++j)allocList.push_back(freePool[j]);
            uint32_t nm=(uint32_t)allocList.size();
            // golden: K(位置序幸存者) ++ H(槽号升序) ++ A(槽号升序) ++ 尾零
            std::vector<int32_t> K,H;
            for(uint32_t p=0;p<occ;++p){
                int32_t s=lruRow[p];
                if(hitFlags[s])H.push_back(s);
                else if(!evictFlags[s])K.push_back(s);
            }
            std::vector<int32_t> A=allocList;
            std::sort(H.begin(),H.end());std::sort(A.begin(),A.end());
            std::vector<int32_t> expected=K;
            expected.insert(expected.end(),H.begin(),H.end());
            expected.insert(expected.end(),A.begin(),A.end());
            expected.resize(B+1,0);
            std::vector<int32_t> lruGm=lruRow,flagsGm=flagsRow,raw(topkDim+8,-777);
            for(uint32_t j=0;j<nm;++j)raw[j]=allocList[j];
            std::vector<int32_t> flagsUb(B+1+8,-777),aAt(B+1+8,-777),lruUb(B+1+8,-777),
                                  hExcl(B+1+8,-777),aExcl(B+1+8,-777),scan(64+8,-777);
            launch([&]{OpLruRebuildRow(B,topkDim,occ,nm,lruGm.data(),0,
                                       flagsGm.data(),0,raw.data(),0,
                                       flagsUb.data(),aAt.data(),lruUb.data(),
                                       hExcl.data(),aExcl.data(),scan.data(),0u);});
            for(uint32_t p=0;p<=B;++p)require(lruGm[p]==expected[p],"rebuild lru mismatch");
            for(uint32_t s=0;s<=B;++s)require(flagsGm[s]==flagsRow[s],"rebuild must not write flags");
            require(raw[nm]==-777,"rawSlots overrun");
            for(uint32_t i=B+1;i<flagsUb.size();++i){
                require(flagsUb[i]==-777&&aAt[i]==-777&&lruUb[i]==-777&&hExcl[i]==-777&&aExcl[i]==-777,"UB overrun");
            }
            ++checks;
        }
    }
    // ---- 3) 跨验证: v1 pri 语义独立 golden vs v2 链 (EvictScan + Rebuild) ----
    // v1 仿真: pri = rank+1 (位置序, 无平局); hit 盖 f0; 淘汰 = 非 hit 按 (pri,slot)
    //   升序前 fsClamp 个 (== 位置序非 hit 前 fsClamp); 分配盖 f0+1; 步末 lru =
    //   全部占用槽 stable_sort by (pri,slot)。fsClamp=0 时跳过 EvictScan (wrapper 语义)。
    for(uint32_t trial=0;trial<12;++trial){
        printf("[cross %u]\n",trial);fflush(stdout);
        std::mt19937 rng(1000+trial);
        uint32_t B=(trial%3==0)?4096u:((trial%3==1)?64u:1024u);
        uint32_t occ=rng()%(B+1);
        std::vector<int32_t> lruRow(B+1,0);
        std::vector<int32_t> slots(B);std::iota(slots.begin(),slots.end(),1);
        std::shuffle(slots.begin(),slots.end(),rng);
        for(uint32_t p=0;p<occ;++p)lruRow[p]=slots[p];
        std::vector<int32_t> hitFlags(B+1,0);
        for(uint32_t p=0;p<occ;++p)if(rng()%4==0)hitFlags[lruRow[p]]=1;
        std::vector<int32_t> nonHit;
        for(uint32_t p=0;p<occ;++p)if(!hitFlags[lruRow[p]])nonHit.push_back(lruRow[p]);
        uint32_t fsClamp=nonHit.empty()?0u:(rng()%(uint32_t)nonHit.size()+1u)/2u;
        std::vector<int32_t> inLru(B+1,0);for(uint32_t p=0;p<occ;++p)inLru[lruRow[p]]=1;
        std::vector<int32_t> freePool;for(uint32_t s=1;s<=B;++s)if(!inLru[s])freePool.push_back((int32_t)s);
        std::shuffle(freePool.begin(),freePool.end(),rng);
        // nm ≥ fsClamp (真实不变量 fs ≤ nm); 分配 = 全部淘汰槽 + 老栈补足
        uint32_t extra=(uint32_t)freePool.size()?(rng()%((uint32_t)freePool.size()+1u)):0u;
        uint32_t nm=fsClamp+extra;
        std::vector<int32_t> allocList;
        for(uint32_t j=0;j<fsClamp;++j)allocList.push_back(nonHit[j]);
        for(uint32_t j=0;allocList.size()<nm&&j<(uint32_t)freePool.size();++j)allocList.push_back(freePool[j]);
        nm=(uint32_t)allocList.size();
        // v1 golden: pri 仿真 + stable_sort
        const int32_t f0=50000;
        std::vector<int32_t> pri(B+1,-1);pri[0]=I32_MAX;
        for(uint32_t p=0;p<occ;++p)pri[lruRow[p]]=(int32_t)p+1;
        for(uint32_t p=0;p<occ;++p)if(hitFlags[lruRow[p]])pri[lruRow[p]]=f0;
        std::vector<int32_t> expectedEvict(nonHit.begin(),nonHit.begin()+fsClamp);
        for(uint32_t j=0;j<fsClamp;++j)pri[nonHit[j]]=-1;
        for(uint32_t j=0;j<nm;++j)pri[allocList[j]]=f0+1;
        std::vector<int32_t> order;
        for(uint32_t s=1;s<=B;++s)if(pri[s]>=0)order.push_back((int32_t)s);
        std::stable_sort(order.begin(),order.end(),[&](int32_t a,int32_t b){
            if(pri[a]!=pri[b])return pri[a]<pri[b];
            return a<b;
        });
        std::vector<int32_t> expectedLru=order;expectedLru.resize(B+1,0);
        // v2 链
        std::vector<int32_t> lruGm=lruRow,flagsGm(B+1,0),freeIdx(B+8,-777),raw(B+8,-777);
        for(uint32_t s=1;s<=B;++s)flagsGm[s]=hitFlags[s];   // OpHitMiss 的 bit0
        std::vector<int32_t> flagsUb(B+1+8,-777),aAt(B+1+8,-777),lruUb(B+1+8,-777),
                              hExcl(B+1+8,-777),aExcl(B+1+8,-777),scan(64+8,-777);
        if(fsClamp>0u){
            launch([&]{OpEvictScanRow(B,B,fsClamp,occ,lruGm.data(),0,flagsGm.data(),0,
                                      freeIdx.data(),flagsUb.data(),scan.data());});
            for(uint32_t j=0;j<fsClamp;++j)require(freeIdx[j]==expectedEvict[j],"cross: evicted list mismatch (v1 vs v2)");
            for(uint32_t j=fsClamp;j<B;++j)require(freeIdx[j]==0,"cross: freeIdx tail zero mismatch");
        }
        for(uint32_t j=0;j<nm;++j)raw[j]=allocList[j];
        launch([&]{OpLruRebuildRow(B,B,occ,nm,lruGm.data(),0,flagsGm.data(),0,raw.data(),0,
                                   flagsUb.data(),aAt.data(),lruUb.data(),hExcl.data(),aExcl.data(),scan.data(),0u);});
        for(uint32_t p=0;p<=B;++p)require(lruGm[p]==expectedLru[p],"cross: rebuilt lru mismatch (v1 vs v2)");
        ++checks;
    }
    // ---- 3b) 尾零 spec 门控: specOn=1 清垃圾尾; specOn=0 尾原样 (不变量: 调用前尾已为零) ----
    {
        const uint32_t B=64u,occ=40u;
        std::mt19937 rng(777);
        std::vector<int32_t> slots(B);std::iota(slots.begin(),slots.end(),1);
        std::shuffle(slots.begin(),slots.end(),rng);
        std::vector<int32_t> lruRow(B+1,-777);      // 垃圾尾 (模拟 spec 后残留)
        for(uint32_t p=0;p<occ;++p)lruRow[p]=slots[p];
        std::vector<int32_t> hitFlags(B+1,0);
        for(uint32_t k=0;k<10;++k)hitFlags[lruRow[rng()%occ]]=1;
        std::vector<int32_t> flagsRow(B+1,0);
        for(uint32_t s=1;s<=B;++s)flagsRow[s]=hitFlags[s];
        std::vector<int32_t> inLru(B+1,0);for(uint32_t p=0;p<occ;++p)inLru[p]=lruRow[p];  // 零尾版
        std::vector<int32_t> raw(B+16,-777);        // nm=0
        auto mk=[&](std::vector<int32_t> l0){return l0;};
        auto run=[&](std::vector<int32_t>& lruGm,uint32_t specOn){
            std::vector<int32_t> flagsGm=flagsRow;
            std::vector<int32_t> flagsUb(B+1+8,-777),aAt(B+1+8,-777),lruUb(B+1+8,-777),
                                  hExcl(B+1+8,-777),aExcl(B+1+8,-777),scan(64+8,-777);
            launch([&]{OpLruRebuildRow(B,B,occ,0,lruGm.data(),0,flagsGm.data(),0,raw.data(),0,
                                       flagsUb.data(),aAt.data(),lruUb.data(),hExcl.data(),aExcl.data(),scan.data(),specOn);});
        };
        // specOn=1: 垃圾尾被清零; 期望 = K+H(尾零)
        std::vector<int32_t> K,H;
        for(uint32_t p=0;p<occ;++p){int32_t v=lruRow[p];if(hitFlags[v])H.push_back(v);else K.push_back(v);}
        std::sort(H.begin(),H.end());
        std::vector<int32_t> expected=K;expected.insert(expected.end(),H.begin(),H.end());
        expected.resize(B+1,0);
        std::vector<int32_t> g1=lruRow;run(g1,1u);
        for(uint32_t p=0;p<=B;++p)require(g1[p]==expected[p],"spec tail zero (specOn=1) mismatch");
        // specOn=0: 尾原样传递 (非 spec 契约: 调用前尾已为零, kernel 不写尾)
        std::vector<int32_t> g2=mk(inLru);run(g2,0u);
        std::vector<int32_t> expected2=K;expected2.insert(expected2.end(),H.begin(),H.end());
        expected2.resize(B+1,0);   // 输入尾为零 → 输出尾保持零
        for(uint32_t p=0;p<=B;++p)require(g2[p]==expected2[p],"non-spec tail passthrough mismatch");
        ++checks;
    }

    // ---- 4) OpHitMiss: warp 归约计数 + freeSize 折叠 + [v2] flags bit0, 无 atomic ----
    for(uint32_t eff:{4u,1536u,2048u}){
        constexpr uint32_t R=3,L=4096,T=2048,B=4096;
        std::vector<int32_t> htd(R*L),pos(R*T),out(R*T,-777),misses(R,-777),fsOut(R,-777),scratch(65,-777);
        std::vector<int32_t> avail(R),flags(R*(B+1),0);   // [v2] flags (prologue 已清零)
        for(uint32_t r=0;r<R;++r)avail[r]=static_cast<int32_t>(r*700+13);
        for(uint32_t i=0;i<R*L;++i)htd[i]=(i%4==0)?I32_MAX:static_cast<int32_t>(i%4096);
        for(uint32_t i=0;i<R*T;++i)pos[i]=(i%7==0)?-1:static_cast<int32_t>((i*5)%(L+100));
        atomicCalls=0;
        int32_t anyFs=0;   // [C1] 淘汰标志 golden
        launch([&]{OpHitMiss(R,T,eff,L,B,R,R,htd.data(),pos.data(),out.data(),
                             misses.data(),avail.data(),fsOut.data(),flags.data(),scratch.data());});
        for(uint32_t r=0;r<R;++r){
            int32_t expectedMisses=0;
            std::vector<int32_t> goldenFlags(B+1,0);
            for(uint32_t j=0;j<T;++j){
                int32_t expectedSlot=j>=eff?0:htd[r*L+std::clamp(pos[r*T+j],0,static_cast<int32_t>(L-1))];
                require(out[r*T+j]==expectedSlot,"hit slot mismatch");
                if(j<eff&&expectedSlot==I32_MAX)++expectedMisses;
                if(j<eff&&expectedSlot!=I32_MAX&&expectedSlot>=0&&expectedSlot<=static_cast<int32_t>(B))
                    goldenFlags[expectedSlot]=1;   // [v2] distinct token → distinct slot
            }
            for(uint32_t s2=0;s2<=B;++s2)
                require(flags[r*(B+1)+s2]==goldenFlags[s2],"[v2] hit flags bit0 mismatch");
            require(misses[r]==expectedMisses,"miss count mismatch");
            int32_t expectedFs=expectedMisses-avail[r];
            if(expectedFs<0)expectedFs=0;
            require(fsOut[r]==expectedFs,"[P4] folded free size mismatch");
            if(expectedFs>0)anyFs=1;
        }
        require(scratch[64]==anyFs,"[C1] eviction flag mismatch");
        require(atomicCalls==0,"[P0] OpHitMiss must not use atomics");
        ++checks;
    }
    // ---- 5) OpMissCumsum: 并行扫描与串行回退 (v2 未改动, 原样回归) ----
    for(uint32_t topk:{4u,64u,129u,256u,300u,512u,2048u}){
        constexpr uint32_t R=3;
        uint32_t eff=topk>7?topk-3:topk;
        std::mt19937 rng(topk*7+1);
        std::vector<int32_t> cur(R*topk),pos(R*topk),rank(R*topk,-777),mpos(R*topk,-777);
        for(uint32_t i=0;i<R*topk;++i){cur[i]=(rng()%3==0)?I32_MAX:static_cast<int32_t>(rng()%4097);pos[i]=static_cast<int32_t>(rng()%9000);}
        std::vector<int32_t> scan(std::max<uint32_t>(2*topk+16,160),-777);
        launch([&]{OpMissCumsum(R,topk,eff,R,R,cur.data(),pos.data(),rank.data(),mpos.data(),scan.data());});
        for(uint32_t r=0;r<R;++r){
            int32_t serial=0;
            for(uint32_t j=0;j<topk;++j){
                bool miss=(j<eff)&&(cur[r*topk+j]==I32_MAX);
                int32_t expectRank=miss?serial:0,expectPos=miss?pos[r*topk+j]:0;
                require(rank[r*topk+j]==expectRank,"miss rank mismatch");
                require(mpos[r*topk+j]==expectPos,"miss pos mismatch");
                if(miss)++serial;
            }
        }
        ++checks;
    }
    // ---- 6) [P4] OpAllocPop 直读 freeIdx + OpFifoAvailUpdate (fifo+=2): 与旧栈语义逐值等价 ----
    {
        constexpr uint32_t R=4,T=64,FS=96;
        const int32_t nmList[R]={20,30,10,0},availList[R]={40,10,50,25};
        std::vector<int32_t> nm(nmList,nmList+R),avail(availList,availList+R),fs(R);
        for(uint32_t r=0;r<R;++r){fs[r]=nm[r]-avail[r];if(fs[r]<0)fs[r]=0;}
        std::mt19937 rng(99);
        std::vector<int32_t> freeIdx(R*T),free(R*FS,-777),raw(R*T,-777);
        for(uint32_t i=0;i<R*T;++i)freeIdx[i]=static_cast<int32_t>(rng()%(FS-1)+1);
        for(uint32_t i=0;i<R*FS;++i)free[i]=static_cast<int32_t>(rng()%1000+1);
        launch([&]{OpAllocPop(R,T,FS,R,R,nm.data(),fs.data(),avail.data(),
                              freeIdx.data(),true,free.data(),raw.data());});
        for(uint32_t r=0;r<R;++r){
            std::vector<int32_t> stack(free.begin()+r*FS,free.begin()+(r+1)*FS);
            for(int32_t k=0;k<fs[r];++k)stack[avail[r]+k]=freeIdx[r*T+k];
            for(uint32_t j=0;j<T;++j){
                int32_t expected=0;
                if(j<static_cast<uint32_t>(nm[r])){
                    int32_t readIdx=avail[r]+fs[r]-nm[r]+static_cast<int32_t>(j);
                    if(readIdx<0)readIdx=0;
                    if(readIdx>=static_cast<int32_t>(FS))readIdx=FS-1;
                    expected=stack[readIdx];
                }
                require(raw[r*T+j]==expected,"[P4] alloc pop mismatch with stack golden");
            }
        }
        // [v2] fifo 两次自增合并为 +2; avail 折叠不变。
        std::vector<int32_t> fifo(R),avail2(avail);
        for(uint32_t r=0;r<R;++r)fifo[r]=static_cast<int32_t>(100+r);
        launch([&]{OpFifoAvailUpdate(R,R,R,fifo.data(),avail2.data(),fs.data(),nm.data());});
        for(uint32_t r=0;r<R;++r){
            int32_t expectedAvail=avail[r]+fs[r]-nm[r];
            require(fifo[r]==102+r,"[v2] fifo must increment by 2");
            require(avail2[r]==expectedAvail,"[P4] folded avail mismatch");
            require(avail2[r]>=0,"[P4] folded avail must stay non-negative");
        }
        ++checks;
    }
    std::cout<<"PASS: "<<checks<<" checks of extracted C++ functions with 128 host threads; not NPU execution\n";
}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<"\n";return 1;}
}
'''
with tempfile.TemporaryDirectory(prefix='echo_v2_check_') as directory:
    tmp=Path(directory)
    cpp=tmp/'check.cpp'; exe=tmp/'check'
    cpp.write_text(preamble+functions+harness)
    subprocess.run(['g++','-std=c++20','-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer','-pthread',str(cpp),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True,timeout=300)
