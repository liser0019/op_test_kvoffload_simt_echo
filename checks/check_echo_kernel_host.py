#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Echo 更新回归：直接提取实际 C++ 函数，用 g++/128 个 CPU 线程检查基数选择、hit 与 miss 扫描。

覆盖 2026-10-13 手抄版的三处设备算法改动：
  - OpArgtopkRow：4 趟 MSB-first 基数选择（阈值 T + 配额 quota），替代旧 4-pass LSD 全排序；
  - OpMissCumsum：Hillis-Steele 并行前缀扫描（topkDim<=2*blockDim 时），否则回退 t0 串行；
  - OpHitMiss：每线程本地累计 + 行末一次 atomic（未改动，继续回归）。

依赖 Python 标准库和支持 C++20 的 g++；启用 AddressSanitizer/UBSan。
这是 CPU 算法和索引检查，不能验证 Ascend 编译、GM D-cache 或 NPU 耗时。
不导入 torch，不参与 compare.py 正式计时，临时构建产物自动删除。
"""
from pathlib import Path
import tempfile, subprocess, re
root=Path(__file__).resolve().parents[1]
# 当前测试使用与设备一致的 64 路直方图线程（ARGTOPK_GROUP_THREADS），
# 128 个 CPU 线程模拟一个 block（blockDim.x=128；并行扫描路径 Bm=blockDim.x）。
# 断言模型常量与实际头文件一致，防止以后修改 tiling 后误用旧模型。
tiling=(root/'csrc/echo/echo_lru_tiling.h').read_text()
for name,value in [('ARGTOPK_GROUP_THREADS',64),('ARGTOPK_UB_NPAD_MAX',8192),
                   ('ARGTOPK_GM_NPAD_MAX',8192),('ECHO_PERF_NUM_STEPS',44)]:
    match=re.search(r'constexpr uint32_t '+name+r'\s*=\s*(\d+)\s*;',tiling)
    assert match and int(match.group(1))==value, f'{name} 已改变，请同步调整 CPU 模型'
src=(root/'csrc/echo/echo_lru_kernel.h').read_text()
def extract(name):
    start=src.index('__aicore__ inline void '+name+'(')
    end=src.index('\n}\n',start)+3
    return src[start:end]
functions=(extract('OpHitMiss')+'\n'+extract('OpArgtopkRow')+'\n'+extract('OpMissCumsum'))
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
    // ---- 1) OpArgtopkRow: 4 趟基数选择 ----
    for(uint32_t B:{1u,8u,31u,63u,64u,127u,128u,255u,256u,4096u}){
        uint32_t n=1;while(n<B+1)n*=2;
        for(uint32_t pattern=0;pattern<4;++pattern){
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
            launch([&]{OpArgtopkRow(B,n,B,G,pri.data(),0,out.data(),freeSize,
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
            for(uint32_t i=0;i<n;++i)require(ub[i]==packed[i],"key region permuted or corrupted");
            for(uint32_t i=footprint;i<ub.size();++i)require(ub[i]==-777,"UB overrun past footprint");
            for(uint32_t i=0;i<n;++i)require(dbgKey[i]==packed[i],"debug packed key mismatch");
            std::vector<uint32_t> expectedHist(256,0);
            for(uint32_t i=0;i<n;++i)++expectedHist[(static_cast<uint32_t>(dbgKey[i])>>24)&255u];
            require(dbgCount==expectedHist,"debug high-byte histogram mismatch");
            ++checks;
        }
    }
    // ---- 2) OpHitMiss: 每线程本地累计 + 一次 atomic ----
    for(uint32_t eff:{4u,1536u,2048u}){
        constexpr uint32_t R=3,L=4096,T=2048;
        std::vector<int32_t> htd(R*L),pos(R*T),out(R*T,-777),hits(R,0);
        for(uint32_t i=0;i<R*L;++i)htd[i]=(i%4==0)?I32_MAX:static_cast<int32_t>(i%4096);
        for(uint32_t i=0;i<R*T;++i)pos[i]=(i%7==0)?-1:static_cast<int32_t>((i*5)%(L+100));
        atomicCalls=0;
        launch([&]{OpHitMiss(R,T,eff,L,R,R,htd.data(),pos.data(),out.data(),hits.data());});
        for(uint32_t r=0;r<R;++r){
            int32_t expectedHits=0;
            for(uint32_t j=0;j<T;++j){
                int32_t expectedSlot=j>=eff?0:htd[r*L+std::clamp(pos[r*T+j],0,static_cast<int32_t>(L-1))];
                require(out[r*T+j]==expectedSlot,"hit slot mismatch");
                if(j<eff&&expectedSlot!=I32_MAX)++expectedHits;
            }
            require(hits[r]==expectedHits,"hit count mismatch");
        }
        require(atomicCalls<=R*blockDim.x,"hit atomic calls exceed active threads");
        ++checks;
    }
    // ---- 3) OpMissCumsum: 并行扫描 (topkDim<=2*blockDim) 与串行回退 ----
    for(uint32_t topk:{4u,64u,129u,256u,2048u}){
        constexpr uint32_t R=3;
        uint32_t eff=topk>7?topk-3:topk;   // 尾部留 inactive 区
        std::mt19937 rng(topk*7+1);
        std::vector<int32_t> cur(R*topk),pos(R*topk),rank(R*topk,-777),mpos(R*topk,-777);
        for(uint32_t i=0;i<R*topk;++i){cur[i]=(rng()%3==0)?I32_MAX:static_cast<int32_t>(rng()%4097);pos[i]=static_cast<int32_t>(rng()%9000);}
        std::vector<int32_t> scan(2*topk+16,-777);
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
