// SPDX-License-Identifier: Apache-2.0
// Synchronous CPU semantic shim: does not model NPU cache/pipelines.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>
#define __aicore__
#define __gm__
#define GM_ADDR unsigned char*
constexpr int PIPE_ALL = 0;
namespace AscendC {
inline uint32_t block_index = 0;
inline uint32_t GetBlockNum() { return 8; }
inline uint32_t GetBlockIdx() { return block_index; }
template<int> void PipeBarrier() {}
struct DataCopyExtParams {uint16_t blockCount; uint32_t blockLen; uint32_t srcStride,dstStride,rsv;};
template<class T> struct DataCopyPadExtParams {bool enable; uint8_t left,right;T value;};
template<class T> struct GlobalTensor {
  T* p=nullptr; uint64_t size=0;
  void SetGlobalBuffer(T* ptr,uint64_t n){p=ptr;size=n;}
};
template<class T> struct LocalTensor {
  T* p;size_t size;
  T GetValue(size_t i)const{if(i>=size)throw std::out_of_range("LocalTensor read");return p[i];}
  void SetValue(size_t i,T v){if(i>=size)throw std::out_of_range("LocalTensor write");p[i]=v;}
};
template<int P=0> struct TBuf {
  std::vector<uint64_t> data;size_t bytes=0;
  template<class T> LocalTensor<T> Get(){return {reinterpret_cast<T*>(data.data()),bytes/sizeof(T)};}
};
struct TPipe {
  template<int P> void InitBuffer(TBuf<P>& b,uint32_t bytes){b.bytes=bytes;b.data.resize((bytes+7)/8);}
};
template<class T> void Duplicate(LocalTensor<T> dst,T value,int64_t count){
  if(count<0||static_cast<size_t>(count)>dst.size)throw std::out_of_range("Duplicate");
  std::fill_n(dst.p,count,value);
}
template<class T> void DataCopyPad(LocalTensor<T> dst,GlobalTensor<T> src,DataCopyExtParams cp,DataCopyPadExtParams<T>){
  if(cp.blockCount!=1||cp.blockLen>src.size*sizeof(T)||cp.blockLen>dst.size*sizeof(T))throw std::out_of_range("GM to UB");
  std::memcpy(dst.p,src.p,cp.blockLen);
}
template<class T> void DataCopyPad(GlobalTensor<T> dst,LocalTensor<T> src,DataCopyExtParams cp){
  if(cp.blockCount!=1||cp.blockLen>dst.size*sizeof(T)||cp.blockLen>src.size*sizeof(T))throw std::out_of_range("UB to GM");
  std::memcpy(dst.p,src.p,cp.blockLen);
}
}
