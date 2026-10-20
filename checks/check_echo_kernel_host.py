#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Echo v2.1（单趟分类）回归：提取实际 C++ 函数，g++/128 个 CPU 线程检查分类淘汰与尾段追加。

覆盖 v2.1 的设备算法改动（详见 kernel 头注释与 deep_dive §9.22，共 54 项）：
  - OpClassifyEvictRow：单趟分类（融合 v2.0 EvictScan + Rebuild 位置域扫描）——
    三产出 freeIdx（前 victims 个候选，位置序 + padding [B+1,..) + 尾零，全部由
    fsClamp>0 门控）/ K 段（幸存者保持旧序）+ H 段（hit 按旧位置序，平局松弛）/
    rbBase（t0 写 numHitsGm）；occ==0 快路径（零栅栏）；不写 flags bit1；
  - OpAllocAppendRow：分配槽按 rawSlots j 序（分配序，平局松弛）尾段追加 +
    spec 尾零；padding 索引防御性钳位（>B 钳到 B、<0 钳到 0）；
  - 跨验证组（14 试验）：随机场景（occ/hit 模式/fs/nm/bit1 混合/padding 边界/
    nm=0/fs=0/occ=0/spec 垃圾尾）按 v2.1 新规则独立推导 golden，与链式调用
    （ClassifyEvict+AllocAppend，rbBase 经"GM"单元格传递，与设备 Step2→Step4
    同构）逐位比对，另加成员集合不变量断言（sorted(新 lru[0:occNew]) ==
    期望占用槽集合，防段错位/漏槽/重槽）；
  - OpHitMiss：hit → flags bit0（v2 替代 pri 时间戳）、warp 归约计数、[P4]
    freeSize 折叠、[C1] 淘汰标志、零 atomic；
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
                   ('ARGTOPK_GM_NPAD_MAX',8192),('ECHO_PERF_NUM_STEPS',29)]:
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
for _m in ('void Init(','void Process(','void ClassifyAllRows(','void AllocAppendAllRows('):
    assert _cls<src.index(_m,_cls)<_cls_end, f'{_m} 不在 class 体内 —— 作用域被破坏'
assert src.index('echo_lru_kernel(')>_cls_end, '入口函数应在 class 之后'
functions=(extract('PackedBlockScan')+'\n'+extract('OpHitMiss')+'\n'+extract('OpClassifyEvictRow')
           +'\n'+extract('OpAllocAppendRow')+'\n'+extract('OpMissCumsum')+'\n'+extract('OpAllocPop')
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
    // ---- 1) OpClassifyEvictRow: 单趟分类 (freeIdx + K/H 段散射 + rbBase) ----
    // hitMode: 0=随机分布 hit; 1=hit 集中在最旧端 (必须跳过, 选中靠后的候选)。
    // bit1Mode: 0=无 bit1; 1=随机标 ~occ/6 个 bit1 (spec 移除语义: 不计候选不计
    //   hit; 只标非 hit 位 —— 管线内 Step0.5 早于 Step1, hit 与 bit1 不共位)。
    // fsClamp==0 用例: freeIdx 必须零写 (fs=0 行不发 freeIdx, AllocPop 恒走老栈),
    //   K/H 照散射。occ==0 用例: fsClamp==0 走快路径 (零栅栏, 全不动 + rbBase=0);
    //   fsClamp>0 (用例 12, 管线内不可达的防御组合) 落入正常路径 (空扫描循环 +
    //   padding + 尾零 + 全套栅栏), 输出与快路径内联版逐位相同 —— 两条路径都覆盖。
    {
        struct ClsCase { uint32_t B, occ, fsClamp, hitMode, bit1Mode, seed; };
        const ClsCase cases[]={
            {64,40,10,0,0,1},{64,40,30,1,0,2},{31,31,7,0,0,3},{256,200,64,1,0,4},
            {4096,4096,512,0,0,5},{4096,3000,512,1,0,6},{64,40,40,0,0,7},   // fs=全部候选
            {64,10,20,0,0,8},                                            // fs>候选总数 → padding
            {128,128,64,1,0,9},{1024,513,100,0,0,10},{64,1,1,0,0,11},{512,0,1,0,0,12},  // occ=0 全 padding
            {4096,2048,1,0,0,13},{96,95,48,1,0,14},{2048,2000,777,0,0,15},  // fs 超大 → 大量 padding
            {64,40,10,0,1,16},{4096,3000,512,1,1,17},{128,96,32,0,1,18},    // bit1 混合
            {64,40,0,0,0,19},{4096,2048,0,1,0,20},                         // fsClamp=0: freeIdx 零写
            {64,0,0,0,0,21},                                               // occ=0 且 fsClamp=0
        };
        for(size_t ci=0;ci<sizeof(cases)/sizeof(cases[0]);++ci){
            const auto& c=cases[ci];
            uint32_t B=c.B,occ=c.occ,fsClamp=c.fsClamp,topkDim=B;
            printf("[classify %zu] B=%u occ=%u fs=%u hit=%u bit1=%u\n",ci,B,occ,fsClamp,c.hitMode,c.bit1Mode);fflush(stdout);
            std::mt19937 rng(c.seed);
            std::vector<int32_t> lruRow(B+1,0),flagsRow(B+1,0);
            std::vector<int32_t> slots(B);std::iota(slots.begin(),slots.end(),1);
            std::shuffle(slots.begin(),slots.end(),rng);
            for(uint32_t p=0;p<occ;++p)lruRow[p]=slots[p];
            std::vector<int32_t> hitFlags(B+1,0),bit1Flags(B+1,0);
            if(occ>0){
                uint32_t hitCount=(c.hitMode==1)?std::min<uint32_t>(occ,occ/3):occ/4;
                for(uint32_t k=0;k<hitCount;++k){
                    uint32_t p=(c.hitMode==1)?k:(rng()%occ);
                    hitFlags[lruRow[p]]=1;
                }
                if(c.bit1Mode==1){
                    uint32_t m=std::max<uint32_t>(1u,occ/6);
                    for(uint32_t k=0;k<m;++k){
                        uint32_t p=rng()%occ;
                        if(!hitFlags[lruRow[p]])bit1Flags[lruRow[p]]=1;
                    }
                }
            }
            for(uint32_t s=1;s<=B;++s)flagsRow[s]=(hitFlags[s]?1:0)|(bit1Flags[s]?2:0);
            // golden (v2.1 新规则直接推导): hit 组=旧位置序; 候选=非 hit 非 bit1;
            //   受害者=前 victims=min(fsClamp,候选数) 个候选 (位置序)。
            std::vector<int32_t> hitSlots,cand;
            for(uint32_t p=0;p<occ;++p){
                int32_t s=lruRow[p];
                if(hitFlags[s])hitSlots.push_back(s);
                else if(!bit1Flags[s])cand.push_back(s);
            }
            uint32_t totalCand=(uint32_t)cand.size(),totalH=(uint32_t)hitSlots.size();
            uint32_t victims=std::min<uint32_t>(fsClamp,totalCand);
            uint32_t totalK=totalCand-victims;
            uint32_t rbBase=totalK+totalH;
            // expected freeIdx: fsClamp=0 → 全然不动 (哨兵); 否则受害者+padding+尾零
            std::vector<int32_t> expFreeIdx(topkDim,-777);
            if(fsClamp!=0u){
                for(uint32_t j=0;j<fsClamp;++j)
                    expFreeIdx[j]=(j<victims)?cand[j]:(int32_t)(B+1u+(j-totalCand));
                for(uint32_t j=fsClamp;j<topkDim;++j)expFreeIdx[j]=0;
            }
            // expected lru: [0,totalK)=幸存者旧序, [totalK,rbBase)=hit 旧位置序,
            //   [rbBase,B]=输入原样 (本 Op 只写 [0,rbBase); 尾区由管线不变量/
            //   AllocAppend 接管)
            std::vector<int32_t> expLru=lruRow;
            for(uint32_t p=0;p<totalK;++p)expLru[p]=cand[victims+p];
            for(uint32_t p=0;p<totalH;++p)expLru[totalK+p]=hitSlots[p];
            std::vector<int32_t> lruGm=lruRow,flagsGm=flagsRow;
            std::vector<int32_t> freeIdx(topkDim+8,-777),rbBaseOut(2,-777);
            std::vector<int32_t> flagsUb(B+1+8,-777),lruUb(B+1+8,-777),scan(64+8,-777);
            launch([&]{OpClassifyEvictRow(B,topkDim,fsClamp,occ,lruGm.data(),0,
                                          flagsGm.data(),0,freeIdx.data(),rbBaseOut.data(),
                                          flagsUb.data(),lruUb.data(),scan.data());});
            for(uint32_t j=0;j<topkDim;++j)
                require(freeIdx[j]==expFreeIdx[j],"classify freeIdx mismatch");
            for(uint32_t j=topkDim;j<freeIdx.size();++j)require(freeIdx[j]==-777,"freeIdx overrun");
            for(uint32_t p=0;p<=B;++p)require(lruGm[p]==expLru[p],"classify lru K/H mismatch");
            for(uint32_t s=0;s<=B;++s)require(flagsGm[s]==flagsRow[s],"classify must not write flags");
            require(rbBaseOut[0]==(int32_t)rbBase,"classify rbBase mismatch");
            require(rbBaseOut[1]==-777,"rbBaseOut overrun");
            for(uint32_t i=B+1;i<flagsUb.size();++i)
                require(flagsUb[i]==-777&&lruUb[i]==-777,"UB staging overrun");
            ++checks;
        }
    }
    // ---- 2) OpAllocAppendRow: 尾段追加 golden ----
    // 预置 K/H 已散射的 lru ([0,base) 随机占位, [base,B] 哨兵 -777) + rbBase;
    //   断言 A 段 = clamp(raw) 的 j 序 (分配序, 平局松弛)、非 spec [occNew,B]
    //   哨兵不动 (数学冗余论证的输入侧契约)、specOn=1 时 [occNew,B] 清零。
    // padMode=1 掺 padding 值 (>B 钳到 B, <0 钳到 0 —— 行为对齐旧 Rebuild)。
    {
        struct ApCase { uint32_t B, base, nm, padMode, specOn, seed; };
        const ApCase cases[]={
            {64,20,8,0,0,31},{64,0,16,0,0,32},     // 常规 / rbBase=0 (cold 全新分配)
            {4096,4096u-512u,512,0,0,33},          // full 形态: base=B-512, nm=512
            {64,40,0,0,0,34},                      // nm=0 (warm_all_hit 形态, 零写)
            {64,30,10,1,0,35},                     // padding 钳位
            {64,20,8,0,1,36},{128,96,32,1,1,37},   // spec 尾零 (后者 occNew=B, 尾空)
            {4096,0,2048,0,0,38},                  // 大形状冷分配
        };
        for(size_t ci=0;ci<sizeof(cases)/sizeof(cases[0]);++ci){
            const auto& c=cases[ci];
            uint32_t B=c.B,base=c.base,nm=c.nm;
            printf("[append %zu] B=%u base=%u nm=%u pad=%u spec=%u\n",ci,B,base,nm,c.padMode,c.specOn);fflush(stdout);
            std::mt19937 rng(c.seed);
            std::vector<int32_t> lruRow(B+1,-777);
            for(uint32_t p=0;p<base;++p)lruRow[p]=(int32_t)(1+rng()%B);
            std::vector<int32_t> raw(nm>0?nm:1,-777);
            {
                std::vector<int32_t> pool;for(int32_t s=1;s<=(int32_t)B;++s)pool.push_back(s);
                std::shuffle(pool.begin(),pool.end(),rng);
                for(uint32_t j=0;j<nm;++j){
                    if(c.padMode==1&&(j%3==0))raw[j]=(j%2==0)?(int32_t)(B+7+j):-5;
                    else raw[j]=pool[j];
                }
            }
            std::vector<int32_t> rawCopy=raw;
            std::vector<int32_t> expected=lruRow;
            for(uint32_t j=0;j<nm;++j){
                int32_t a=raw[j];
                expected[base+j]=a<0?0:(a>(int32_t)B?(int32_t)B:a);
            }
            uint32_t occNew=base+nm;
            if(c.specOn!=0u)for(uint32_t p=occNew;p<=B;++p)expected[p]=0;
            std::vector<int32_t> lruGm=lruRow;
            std::vector<int32_t> rbBaseGm={(int32_t)base,-777};
            launch([&]{OpAllocAppendRow(B,nm,lruGm.data(),0,raw.data(),0,rbBaseGm.data(),0u,c.specOn);});
            for(uint32_t p=0;p<=B;++p)require(lruGm[p]==expected[p],"append lru mismatch");
            for(uint32_t j=0;j<nm;++j)require(raw[j]==rawCopy[j],"append must not write rawSlots");
            require(rbBaseGm[1]==-777,"rbBaseGm cell [1] must be untouched");
            ++checks;
        }
    }
    // ---- 3) 跨验证: v2.1 新规则独立 golden vs 链式调用 (ClassifyEvict+AllocAppend) ----
    // CPU golden 直接按新序推导: 新 lru = 幸存者(旧序) ++ hit(旧位置序) ++ 分配
    //   (rawSlots j 序); 淘汰集合 = 旧序前 fsClamp 个候选 (候选=非 hit 非 bit1)。
    // 链式调用与设备 Step2→Step4 同构 (rbBase 经"GM"单元格传递, 两次 launch 间
    //   无栅栏 —— mock 的 launch() join 即全序)。
    // 管线不变量对齐: 分配 = 受害者(重用, 分配序前段) + 空槽池补足, 空槽池含
    //   bit1 槽 (spec 语义: Step0.5 已释放) ⟹ nm ≥ victims+bit1Cnt 可满足 ⟹
    //   occNew ≥ occOld (非 spec 尾零数学冗余的前提)。特例: trial 11 强制
    //   fsClamp>候选数 (padding 边界); trial 12 = occ=0; trial 13 = fsClamp=0 +
    //   nm=0 + specOn=1 + 垃圾尾 (spec 尾零, 不变量故意不满足)。
    for(uint32_t trial=0;trial<14;++trial){
        printf("[cross %u]\n",trial);fflush(stdout);
        std::mt19937 rng(2000+trial);
        uint32_t B=(trial%3==0)?4096u:((trial%3==1)?64u:1024u);
        uint32_t occ=(trial==12)?0u:(rng()%(B+1));
        std::vector<int32_t> lruRow(B+1,0);
        std::vector<int32_t> slots(B);std::iota(slots.begin(),slots.end(),1);
        std::shuffle(slots.begin(),slots.end(),rng);
        for(uint32_t p=0;p<occ;++p)lruRow[p]=slots[p];
        if(trial==13)for(uint32_t p=occ;p<=B;++p)lruRow[p]=-777;   // spec 垃圾尾
        std::vector<int32_t> hitFlags(B+1,0),bit1Flags(B+1,0);
        for(uint32_t p=0;p<occ;++p)if(rng()%4==0)hitFlags[lruRow[p]]=1;
        if(trial%2==1)for(uint32_t p=0;p<occ;++p)
            if(rng()%7==0&&!hitFlags[lruRow[p]])bit1Flags[lruRow[p]]=1;
        std::vector<int32_t> flagsRow(B+1,0);
        for(uint32_t s=1;s<=B;++s)flagsRow[s]=(hitFlags[s]?1:0)|(bit1Flags[s]?2:0);
        std::vector<int32_t> hitSlots,cand;
        uint32_t bit1Cnt=0;
        for(uint32_t p=0;p<occ;++p){
            int32_t s=lruRow[p];
            if(hitFlags[s])hitSlots.push_back(s);
            else if(bit1Flags[s])++bit1Cnt;
            else cand.push_back(s);
        }
        uint32_t totalCand=(uint32_t)cand.size(),totalH=(uint32_t)hitSlots.size();
        uint32_t fsClamp;
        if(trial==13)fsClamp=0u;
        else if(trial==12)fsClamp=3u;                              // occ=0 + padding 边界
        else if(trial==11)fsClamp=std::min<uint32_t>(totalCand+5u,B);  // padding 边界强制
        else if(totalCand==0u)fsClamp=rng()%3;
        else fsClamp=rng()%(totalCand+1u);                         // 0..全部候选
        uint32_t victims=std::min<uint32_t>(fsClamp,totalCand);
        uint32_t totalK=totalCand-victims;
        uint32_t rbBase=totalK+totalH;
        // 空槽池: 不在 lru 的槽 + bit1 槽 (spec 已释放); 分配 = 受害者重用 + 池补足
        std::vector<int32_t> inLru(B+1,0);for(uint32_t p=0;p<occ;++p)inLru[lruRow[p]]=1;
        std::vector<int32_t> freePool;
        for(uint32_t s=1;s<=B;++s)if(!inLru[s]||bit1Flags[s])freePool.push_back((int32_t)s);
        std::shuffle(freePool.begin(),freePool.end(),rng);
        uint32_t minNm=victims+bit1Cnt;                            // occNew ≥ occOld 前提
        uint32_t maxNm=victims+(uint32_t)freePool.size();
        uint32_t nm;
        if(trial==13)nm=0u;                                        // nm=0 (warm_all_hit 形态)
        else if(trial==12)nm=std::min<uint32_t>(maxNm,2u);
        else nm=minNm+(rng()%((maxNm>=minNm)?(maxNm-minNm+1u):1u));
        if(nm>maxNm)nm=maxNm;
        std::vector<int32_t> allocList;
        for(uint32_t j=0;j<victims;++j)allocList.push_back(cand[j]);
        for(uint32_t j=0;allocList.size()<nm&&j<(uint32_t)freePool.size();++j)
            allocList.push_back(freePool[j]);
        nm=(uint32_t)allocList.size();
        uint32_t specOn=(trial==13)?1u:0u;
        // golden: [0,occNew)=幸存者+hit+分配; spec 尾零; 非 spec 且 occNew<occ
        //   时 [occNew,occ) 保持旧值 (文档化行为, 管线不变量外的输入)
        uint32_t occNew=rbBase+nm;
        std::vector<int32_t> expected=lruRow;
        for(uint32_t p=0;p<totalK;++p)expected[p]=cand[victims+p];
        for(uint32_t p=0;p<totalH;++p)expected[totalK+p]=hitSlots[p];
        for(uint32_t j=0;j<nm;++j)expected[rbBase+j]=allocList[j];
        if(specOn!=0u)for(uint32_t p=occNew;p<=B;++p)expected[p]=0;
        std::vector<int32_t> expFreeIdx(B,-777);
        if(fsClamp!=0u){
            for(uint32_t j=0;j<fsClamp;++j)
                expFreeIdx[j]=(j<victims)?cand[j]:(int32_t)(B+1u+(j-totalCand));
            for(uint32_t j=fsClamp;j<B;++j)expFreeIdx[j]=0;
        }
        // 链式调用
        std::vector<int32_t> lruGm=lruRow,flagsGm=flagsRow;
        std::vector<int32_t> freeIdx(B+8,-777),raw(nm>0?nm:1,-777),rbBaseGm(2,-777);
        for(uint32_t j=0;j<nm;++j)raw[j]=allocList[j];
        std::vector<int32_t> flagsUb(B+1+8,-777),lruUb(B+1+8,-777),scan(64+8,-777);
        launch([&]{OpClassifyEvictRow(B,B,fsClamp,occ,lruGm.data(),0,flagsGm.data(),0,
                                      freeIdx.data(),rbBaseGm.data(),
                                      flagsUb.data(),lruUb.data(),scan.data());});
        launch([&]{OpAllocAppendRow(B,nm,lruGm.data(),0,raw.data(),0,rbBaseGm.data(),0u,specOn);});
        for(uint32_t p=0;p<=B;++p)require(lruGm[p]==expected[p],"cross: lru mismatch (golden vs chain)");
        require(rbBaseGm[0]==(int32_t)rbBase,"cross: rbBase mismatch");
        for(uint32_t j=0;j<B;++j)require(freeIdx[j]==expFreeIdx[j],"cross: freeIdx mismatch");
        // 成员集合不变量: [0,occNew) 恰为期望占用槽集合 (防段错位/漏槽/重槽)
        std::vector<int32_t> want(cand.begin()+victims,cand.end());
        want.insert(want.end(),hitSlots.begin(),hitSlots.end());
        want.insert(want.end(),allocList.begin(),allocList.end());
        std::vector<int32_t> got(lruGm.begin(),lruGm.begin()+occNew);
        std::sort(want.begin(),want.end());std::sort(got.begin(),got.end());
        require(got==want,"cross: member set invariant (occupied slots) mismatch");
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
