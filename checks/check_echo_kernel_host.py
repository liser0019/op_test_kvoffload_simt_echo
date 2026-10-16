#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Echo 更新回归：直接提取实际 C++ 函数，用 g++/128 个 CPU 线程检查基数选择、hit 与 miss 扫描。

覆盖 2026-10-13 手抄版的三处设备算法改动 + P0-P4 性能优化：
  - OpHitMiss：miss 计数改 warp 归约 + t0 汇总（[P0] 无 atomic，numMisses 直接写出）；
  - OpArgtopkRow：4 趟 MSB-first 基数选择 + 并行 emission（[P1] 打包块扫描得各线程
    写入基址，连续切片保槽号升序，替代 t0 串行扫描）；
  - OpMissCumsum：两级 warp 扫描（[P3] 打包 p0/p1 miss 标志，替代 Hillis-Steele），
    否则回退 t0 串行；
  - OpHitMiss 内联块求和并顺手写 freeSize（[P4] 原 OpFreeSize 折叠）；
  - OpAllocPop：直读 freeIdx + 老栈（[P4] 不依赖 FreePush 新段，与旧栈弹出语义
    逐值等价）；OpFifoAvailUpdate：fifo++/avail += fs-nm 折叠（[P4] 替代两个
    独立 t0 步，终值与旧两步序列恒等）；
  - PackedBlockScan：warp 协作辅助（shfl_up 模拟，2 次栅栏版本）。

依赖 Python 标准库和支持 C++20 的 g++；启用 AddressSanitizer/UBSan。
这是 CPU 算法和索引检查，不能验证 Ascend 编译、GM D-cache 或 NPU 耗时。
不导入 torch，不参与 compare.py 正式计时，临时构建产物自动删除。
"""
from pathlib import Path
import tempfile, subprocess, re
root=Path(__file__).resolve().parents[1]
# 当前测试使用与设备一致的 64 路直方图线程（ARGTOPK_GROUP_THREADS），
# 128 个 CPU 线程模拟一个 block（blockDim.x=128→4 warp；warp 协作辅助按
# blockDim.x 运行时推导 warp 数）。
# 断言模型常量与实际头文件一致，防止以后修改 tiling 后误用旧模型。
tiling=(root/'csrc/echo/echo_lru_tiling.h').read_text()
for name,value in [('ARGTOPK_GROUP_THREADS',64),('ARGTOPK_UB_NPAD_MAX',8192),
                   ('ARGTOPK_GM_NPAD_MAX',8192),('ECHO_PERF_NUM_STEPS',31)]:
    match=re.search(r'constexpr uint32_t '+name+r'\s*=\s*(\d+)\s*;',tiling)
    assert match and int(match.group(1))==value, f'{name} 已改变，请同步调整 CPU 模型'
src=(root/'csrc/echo/echo_lru_kernel.h').read_text()
def extract(name):
    for prefix in ('__aicore__ inline void ','__aicore__ inline int32_t '):
        tag=prefix+name+'('
        if tag in src:
            start=src.index(tag)
            end=src.index('\n}\n',start)+3
            return src[start:end]
    raise KeyError(name)
functions=(extract('PackedBlockScan')+'\n'+extract('OpHitMiss')+'\n'+extract('OpArgtopkRow')
           +'\n'+extract('OpMissCumsum')+'\n'+extract('OpAllocPop')+'\n'+extract('OpFifoAvailUpdate'))
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
    // ---- 1) OpArgtopkRow: 逐字节基数选择 (含 [方向1] 自适应跳趟与回退) ----
    // fifo 变体: 0x40000000 → z=0 全趟 (含 dbg 全量直方图 golden);
    //   4097 → z=2 跳 2 趟; 65536 → z=1; 100 → z=3 只跑 byte0。
    //   跳趟/回退路径的输出 golden 与全趟完全一致 (选择结果与起跑趟无关)。
    for(uint32_t B:{1u,8u,31u,63u,64u,127u,128u,255u,256u,4096u}){
        uint32_t n=1;while(n<B+1)n*=2;
        for(uint32_t pattern=0;pattern<4;++pattern){
         for(uint32_t fifoVal:{0x40000000u,4097u,65536u,100u}){
            std::mt19937 rng(B*13+pattern);
            std::vector<int32_t> pri(B+1);
            for(auto& p:pri){
                if(pattern==0)p=(rng()%9==0)?-1:static_cast<int32_t>(rng()&0x7fffffff);
                if(pattern==1)p=(rng()%5==0)?-1:static_cast<int32_t>(rng()%4);
                if(pattern==2)p=-1;
                if(pattern==3)p=-1;
            }
            // topkDim=B; freeSize 混合 0 / B / 越界值, 覆盖 fsClamp 钳位与提前返回.
            int32_t freeSize=pattern==2?0:(pattern==1?static_cast<int32_t>(rng()%B):static_cast<int32_t>(B));
            if(pattern==0&&B>4)freeSize=static_cast<int32_t>(rng()%B);
            uint32_t fsClamp=freeSize<0?0u:(freeSize>(int32_t)B?B:(uint32_t)freeSize);
            // 打包 golden: key[i]=I32_MAX 当 i==0 || i>B || pri[i]<0.
            std::vector<int32_t> packed(n,I32_MAX);
            for(uint32_t i=1;i<=B;++i)packed[i]=pri[i]<0?I32_MAX:pri[i];
            // 缓冲: 单大 UB 数组分段 (key/hist/globalCnt/segLocal/segBase/scalar), 越界哨兵.
            constexpr uint32_t G=ARGTOPK_GROUP_THREADS;
            uint32_t footprint=n+G*256u+256u+G*2u+G*2u+8u;
            std::vector<int32_t> ub(32768+16,-777),out(B+16,-777),dbgKey(n,-777);
            std::vector<uint32_t> dbgCount(256,0x7fffffff);
            int32_t *key=ub.data(),*hist=ub.data()+n,*globalCnt=hist+G*256u;
            int32_t *segLocal=globalCnt+256u,*segBase=segLocal+G*2u,*scalar=segBase+G*2u;
            // 跳趟路径 (fifoVal 小) 的 dbgCount 是首趟过滤直方图, 语义随起跑趟变化 → 不验证;
            //   全趟路径 (z=0) 保持旧版 pass0 全量 golden。dbgKey 与起跑趟无关, 恒验证。
            uint32_t startPass=3u;
            if((fifoVal>>24)!=0u)startPass=0u;else if((fifoVal>>16)!=0u)startPass=1u;else if((fifoVal>>8)!=0u)startPass=2u;
            bool const fullPass=(startPass==0u);
            launch([&]{OpArgtopkRow(B,n,B,G,fifoVal,pri.data(),0,out.data(),freeSize,true,
                key,hist,globalCnt,segLocal,segBase,scalar,
                dbgCount.data(),dbgKey.data());});
            if(fsClamp==0u){
                // 提前返回: 不写任何输出/UB/dbg.
                for(uint32_t i=0;i<out.size();++i)require(out[i]==-777,"free output written on early return");
                for(uint32_t i=0;i<ub.size();++i)require(ub[i]==-777,"UB written on early return");
                for(uint32_t i=0;i<n;++i)require(dbgKey[i]==-777,"dbgKey written on early return");
                ++checks;continue;
            }
            // 选择 golden: T = fsClamp-th 小 key, quota = fsClamp - |{key<T}|.
            std::vector<int32_t> keys=packed;
            std::sort(keys.begin(),keys.end());
            uint32_t T=(uint32_t)keys[fsClamp-1];
            uint32_t lessCount=(uint32_t)(std::lower_bound(keys.begin(),keys.end(),(int32_t)T)-keys.begin());
            uint32_t quota=fsClamp-lessCount;
            std::vector<int32_t> expected;
            for(uint32_t i=0;i<n;++i)if((uint32_t)packed[i]<T)expected.push_back((int32_t)i);
            for(uint32_t i=0;i<n&&expected.size()<fsClamp;++i)
                if((uint32_t)packed[i]==T)expected.push_back((int32_t)i);
            require(expected.size()==fsClamp,"golden selection size mismatch");
            // 交叉验证: 选择集合 == 稳定排序 (key, slot) 前 fsClamp 个 (顺序无关).
            std::vector<int32_t> order(n);std::iota(order.begin(),order.end(),0);
            std::stable_sort(order.begin(),order.end(),[&](int32_t a,int32_t b){return packed[a]<packed[b];});
            std::vector<int32_t> topk(order.begin(),order.begin()+fsClamp);
            std::sort(topk.begin(),topk.end());
            std::vector<int32_t> selected=expected;   // expected 按 <T/==T 分组构造, 不保证槽号升序
            std::sort(selected.begin(),selected.end());
            require(selected==topk,"selected set differs from stable-sort golden");
            for(uint32_t i=0;i<B;++i)
                require(out[i]==(i<fsClamp?expected[i]:0),"free output mismatch");
            for(uint32_t i=B;i<out.size();++i)require(out[i]==-777,"free output overrun");
            // [B] key 只写有效区 [0, B+1); padding 区不再打包 (选择时解析式计入)。
            for(uint32_t i=0;i<=B;++i)require(ub[i]==packed[i],"key region permuted or corrupted");
            for(uint32_t i=B+1;i<n;++i)require(ub[i]==-777,"padding region must stay untouched");
            for(uint32_t i=footprint;i<ub.size();++i)require(ub[i]==-777,"UB overrun past footprint");
            for(uint32_t i=0;i<=B;++i)require(dbgKey[i]==packed[i],"debug packed key mismatch");
            for(uint32_t i=B+1;i<n;++i)require(dbgKey[i]==-777,"dbgKey padding region untouched");
            // [方向1] dbgCount 仅在全趟路径 (startPass==0) 验证: golden 为完整打包键的
            //   高字节直方图 (含 padding, 由 merge 解析式计入)。跳趟路径首趟带
            //   prefix=0 过滤, 直方图语义不同, 不做断言 (输出 golden 已覆盖正确性)。
            if(fullPass){
                std::vector<uint32_t> expectedHist(256,0);
                for(uint32_t i=0;i<n;++i)++expectedHist[(static_cast<uint32_t>(packed[i])>>24)&255u];
                require(dbgCount==expectedHist,"debug high-byte histogram mismatch");
            }
            ++checks;
         }
        }
    }
    // ---- 1b) [P-C] freeIdx 走 UB 路径 (freeIdxIsGm=false): 输出与 GM 路径 golden 一致 ----
    {
        const uint32_t B=64u;
        uint32_t n=1;while(n<B+1)n*=2;
        std::mt19937 rng(4242);
        std::vector<int32_t> pri(B+1);
        for(auto& q:pri) q=(rng()%5==0)?-1:static_cast<int32_t>(rng()%4);
        int32_t freeSize=static_cast<int32_t>(rng()%B);
        uint32_t fsClamp=freeSize<0?0u:(freeSize>(int32_t)B?B:(uint32_t)freeSize);
        std::vector<int32_t> packed(n,I32_MAX);
        for(uint32_t i=1;i<=B;++i)packed[i]=pri[i]<0?I32_MAX:pri[i];
        constexpr uint32_t G=ARGTOPK_GROUP_THREADS;
        uint32_t footprint=n+G*256u+256u+G*2u+G*2u+8u;
        std::vector<int32_t> ub(32768+16,-777),outUB(B+16,-777),outGM(B+16,-777),dbgKey(n,-777);
        std::vector<uint32_t> dbgCount(256,0x7fffffff);
        int32_t *key=ub.data(),*hist=ub.data()+n,*globalCnt=hist+G*256u;
        int32_t *segLocal=globalCnt+256u,*segBase=segLocal+G*2u,*scalar=segBase+G*2u;
        uint32_t const fifoVal=4097u;
        launch([&]{OpArgtopkRow(B,n,B,G,fifoVal,pri.data(),0,outGM.data(),freeSize,true,
            key,hist,globalCnt,segLocal,segBase,scalar,dbgCount.data(),dbgKey.data());});
        std::fill(ub.begin(),ub.end(),-777); std::fill(outUB.begin(),outUB.end(),-777);
        std::fill(dbgKey.begin(),dbgKey.end(),-777); std::fill(dbgCount.begin(),dbgCount.end(),0x7fffffff);
        launch([&]{OpArgtopkRow(B,n,B,G,fifoVal,pri.data(),0,outUB.data(),freeSize,false,
            key,hist,globalCnt,segLocal,segBase,scalar,dbgCount.data(),dbgKey.data());});
        // golden: 与 GM 路径逐位一致 (同一选择语义)
        require(outUB==outGM,"[P-C] UB-path output differs from GM-path");
        ++checks;
    }

    // ---- 2) OpHitMiss: warp 归约计数 (含 [P4] 折叠的 freeSize), 无 atomic ----
    for(uint32_t eff:{4u,1536u,2048u}){
        constexpr uint32_t R=3,L=4096,T=2048;
        std::vector<int32_t> htd(R*L),pos(R*T),out(R*T,-777),misses(R,-777),fsOut(R,-777),scratch(65,-777);
        std::vector<int32_t> avail(R);
        for(uint32_t r=0;r<R;++r)avail[r]=static_cast<int32_t>(r*700+13);
        for(uint32_t i=0;i<R*L;++i)htd[i]=(i%4==0)?I32_MAX:static_cast<int32_t>(i%4096);
        for(uint32_t i=0;i<R*T;++i)pos[i]=(i%7==0)?-1:static_cast<int32_t>((i*5)%(L+100));
        atomicCalls=0;
        int32_t anyFs=0;   // [C1] 淘汰标志 golden: 本核任一行 fs>0
        launch([&]{OpHitMiss(R,T,eff,L,R,R,htd.data(),pos.data(),out.data(),
                             misses.data(),avail.data(),fsOut.data(),scratch.data());});
        for(uint32_t r=0;r<R;++r){
            int32_t expectedMisses=0;
            for(uint32_t j=0;j<T;++j){
                int32_t expectedSlot=j>=eff?0:htd[r*L+std::clamp(pos[r*T+j],0,static_cast<int32_t>(L-1))];
                require(out[r*T+j]==expectedSlot,"hit slot mismatch");
                if(j<eff&&expectedSlot==I32_MAX)++expectedMisses;
            }
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
    // ---- 3) OpMissCumsum: 并行扫描与串行回退 ----
    // blockDim=128: topk ≤ 256 (且 2*topk ≤ 8192) 走 2 元素并行路径;
    //   300/512 走 4 元素两次链式扫描路径 (topk ≤ 4*blockDim=512 且 4*topk ≤ 8192);
    //   2048 走 t0 串行回退。golden (串行参考) 逻辑不变。
    for(uint32_t topk:{4u,64u,129u,256u,300u,512u,2048u}){
        constexpr uint32_t R=3;
        uint32_t eff=topk>7?topk-3:topk;   // 尾部留 inactive 区
        std::mt19937 rng(topk*7+1);
        std::vector<int32_t> cur(R*topk),pos(R*topk),rank(R*topk,-777),mpos(R*topk,-777);
        for(uint32_t i=0;i<R*topk;++i){cur[i]=(rng()%3==0)?I32_MAX:static_cast<int32_t>(rng()%4097);pos[i]=static_cast<int32_t>(rng()%9000);}
        // [P3] 只需 warp scratch (warpTotals[32]+warpBases[32]+ctrlA/ctrlB 两个控制槽 = 66) + 哨兵余量
        std::vector<int32_t> scan(std::max<uint32_t>(2*topk+16,160),-777);  // [Scan-v2] 四缓冲 128 槽+余量
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
    // ---- 4) [P4] OpAllocPop 直读 freeIdx + OpFifoAvailUpdate 折叠: 与旧栈语义逐值等价 ----
    {
        constexpr uint32_t R=4,T=64,FS=96;   // T=topkDim, FS=freeStackSize
        // 覆盖 nm<fs / nm==fs / nm>fs / nm=0 / fs=0 组合 (fs 由 avail/nm 推出)。
        const int32_t nmList[R]={20,30,10,0}, availList[R]={40,10,50,25};
        std::vector<int32_t> nm(nmList,nmList+R),avail(availList,availList+R),fs(R);
        for(uint32_t r=0;r<R;++r){fs[r]=nm[r]-avail[r];if(fs[r]<0)fs[r]=0;}
        std::mt19937 rng(99);
        std::vector<int32_t> freeIdx(R*T),free(R*FS,-777),raw(R*T,-777);
        for(uint32_t i=0;i<R*T;++i)freeIdx[i]=static_cast<int32_t>(rng()%(FS-1)+1);
        for(uint32_t i=0;i<R*FS;++i)free[i]=static_cast<int32_t>(rng()%1000+1);   // 老栈随机内容
        launch([&]{OpAllocPop(R,T,FS,R,R,nm.data(),fs.data(),avail.data(),
                              freeIdx.data(),free.data(),raw.data());});
        for(uint32_t r=0;r<R;++r){
            // 旧版 golden: 先 push (free[avail+k]=freeIdx[k]) 再从栈尾弹 nm 个。
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
        // fifo/avail 折叠更新: 终值与旧两步序列 (+=fs 后 -=nm) 恒等且 ≥0。
        std::vector<int32_t> fifo(R),avail2(avail);
        for(uint32_t r=0;r<R;++r)fifo[r]=static_cast<int32_t>(100+r);
        launch([&]{OpFifoAvailUpdate(R,R,R,fifo.data(),avail2.data(),fs.data(),nm.data());});
        for(uint32_t r=0;r<R;++r){
            int32_t expectedAvail=avail[r]+fs[r]-nm[r];
            require(fifo[r]==101+r,"fifo increment mismatch");
            require(avail2[r]==expectedAvail,"[P4] folded avail mismatch");
            require(avail2[r]>=0,"[P4] folded avail must stay non-negative");
        }
        ++checks;
    }
    std::cout<<"PASS: "<<checks<<" checks of extracted C++ functions with 128 host threads; not NPU execution\n";
}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<"\n";return 1;}
}
'''
with tempfile.TemporaryDirectory(prefix='echo_author_check_') as directory:
    tmp=Path(directory)
    cpp=tmp/'check.cpp'; exe=tmp/'check'
    cpp.write_text(preamble+functions+harness)
    subprocess.run(['g++','-std=c++20','-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer','-pthread',str(cpp),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True,timeout=180)
