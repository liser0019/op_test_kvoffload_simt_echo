#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Echo 更新回归：直接提取实际 C++ 函数，用 g++/128 个 CPU 线程检查排序与 hit。

依赖 Python 标准库和支持 C++20 的 g++；启用 AddressSanitizer/UBSan。
这是 CPU 算法和索引检查，不能验证 Ascend 编译、GM D-cache 或 NPU 耗时。
不导入 torch，不参与 compare.py 正式计时，临时构建产物自动删除。
"""
from pathlib import Path
import tempfile, subprocess, re
root=Path(__file__).resolve().parents[1]
# 当前测试使用与设备相同的 64 路 histogram/scatter，128 个 CPU 线程模拟一块。
# 断言模型常量与实际头文件一致，防止以后修改 tiling 后误用旧模型。
tiling=(root/'csrc/echo/echo_lru_tiling.h').read_text()
for name,value in [('ARGTOPK_HIST_THREADS',64),('ARGTOPK_SCAT_THREADS',64),('ARGTOPK_UB_NPAD_MAX',8192)]:
    match=re.search(r'constexpr uint32_t '+name+r'\s*=\s*(\d+)\s*;',tiling)
    assert match and int(match.group(1))==value, f'{name} 已改变，请同步调整 CPU 模型'
src=(root/'csrc/echo/echo_lru_kernel.h').read_text()
def extract(name):
    start=src.index('__aicore__ inline void '+name+'(')
    end=src.index('\n}\n',start)+3
    return src[start:end]
functions=extract('OpHitMiss')+'\n'+extract('OpArgtopkRow')
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
constexpr uint32_t ARGTOPK_HIST_THREADS=64, ARGTOPK_SCAT_THREADS=64;
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
    for(uint32_t B:{1u,8u,31u,63u,64u,127u,128u,255u,256u,4096u}){
        uint32_t n=1;while(n<B+1)n*=2;
        for(uint32_t pattern=0;pattern<3;++pattern){
            std::mt19937 rng(B*13+pattern);
            std::vector<int32_t> pri(B+1);
            for(auto& p:pri){
                if(pattern==0)p=(rng()%9==0)?-1:static_cast<int32_t>(rng()&0x7fffffff);
                if(pattern==1)p=(rng()%5==0)?-1:static_cast<int32_t>(rng()%4);
                if(pattern==2)p=-1;
            }
            std::vector<int32_t> expectedKey(n,I32_MAX),expectedSlot(n);
            for(uint32_t i=1;i<=B;++i)expectedKey[i]=pri[i]<0?I32_MAX:pri[i];
            std::iota(expectedSlot.begin(),expectedSlot.end(),0);
            std::stable_sort(expectedSlot.begin(),expectedSlot.end(),[&](int32_t a,int32_t b){return expectedKey[a]<expectedKey[b];});
            std::vector<int32_t> ub(32768+16,-777),out(B+16,-777),dbgKey(n,-777);
            std::vector<uint32_t> count(64*256,0x7fffffff),scat(64*256,0x7fffffff),dbgCount(256,0);
            int32_t freeSize=pattern==2?0:static_cast<int32_t>(B);
            launch([&]{OpArgtopkRow(B,n,B,pri.data(),0,out.data(),freeSize,
                ub.data(),ub.data()+n,ub.data()+2*n,ub.data()+3*n,
                count.data(),scat.data(),dbgCount.data(),dbgKey.data());});
            for(uint32_t i=0;i<n;++i){
                require(ub[n+i]==expectedSlot[i],"sorted slot mismatch");
                require(ub[i]==expectedKey[expectedSlot[i]],"sorted priority mismatch");
                int32_t packed=i==0||i>B||pri[i]<0?I32_MAX:pri[i];
                require(dbgKey[i]==packed,"debug packed key mismatch");
            }
            for(uint32_t i=0;i<B;++i)require(out[i]==(i<static_cast<uint32_t>(freeSize)?expectedSlot[i]:0),"free output mismatch");
            for(uint32_t i=B;i<out.size();++i)require(out[i]==-777,"free output overrun");
            for(uint32_t i=32768;i<ub.size();++i)require(ub[i]==-777,"UB overrun");
            std::vector<uint32_t> expectedHist(256,0);
            for(uint32_t i=0;i<n;++i)++expectedHist[static_cast<uint32_t>(dbgKey[i])&255u];
            require(dbgCount==expectedHist,"debug histogram mismatch");
            ++checks;
        }
    }
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
    std::cout<<"PASS: "<<checks<<" checks of extracted C++ functions with 128 host threads; not NPU execution\n";
}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<"\n";return 1;}
}
'''
with tempfile.TemporaryDirectory(prefix='echo_author_check_') as directory:
    tmp=Path(directory)
    cpp=tmp/'check.cpp'; exe=tmp/'check'
    cpp.write_text(preamble+functions+harness)
    subprocess.run(['g++','-std=c++20','-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer','-pthread',str(cpp),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True,timeout=90)
