// SPDX-License-Identifier: Apache-2.0
#include <fstream>
#include <iostream>
#include <tuple>
#include "acc_offload_lru_compact.h"
template<class T> void read(std::ifstream& f,T* p,size_t n){f.read(reinterpret_cast<char*>(p),n*sizeof(T));if(!f)throw std::runtime_error("input truncated");}
template<class T> void write(std::ofstream& f,const std::vector<T>& a){f.write(reinterpret_cast<const char*>(a.data()),a.size()*sizeof(T));}
int main(int argc,char**argv){try{
  if(argc!=3)return 2;
  std::ifstream input(argv[1],std::ios::binary);
  int64_t dims[5];read(input,dims,5);
  auto [r,t,c,l,steps]=std::tuple{dims[0],dims[1],dims[2],dims[3],dims[4]};
  std::vector<int64_t> req(r),last(r);
  std::vector<int32_t> query(r*t),prefix(r),resident(r*c),lru(r*c),current(r*t,-777),count(r,-777),tokens(r*t,-777),slots(r*t,-777);
  std::vector<int32_t> mark(8*l,0),pos(8*l,0),epochs(8,0);
  std::ofstream output(argv[2],std::ios::binary);
  for(int64_t step=0;step<steps;++step){
    read(input,req.data(),r);read(input,query.data(),r*t);read(input,prefix.data(),r);
    if(step==0){read(input,last.data(),r);read(input,resident.data(),r*c);read(input,lru.data(),r*c);}
    for(uint32_t core=0;core<8;++core){
      AscendC::block_index=core;
      LruResidentCompactKernel kernel;
      kernel.Init(reinterpret_cast<GM_ADDR>(req.data()),reinterpret_cast<GM_ADDR>(last.data()),
        reinterpret_cast<GM_ADDR>(query.data()),reinterpret_cast<GM_ADDR>(prefix.data()),
        reinterpret_cast<GM_ADDR>(resident.data()),reinterpret_cast<GM_ADDR>(lru.data()),
        reinterpret_cast<GM_ADDR>(current.data()),reinterpret_cast<GM_ADDR>(count.data()),
        reinterpret_cast<GM_ADDR>(tokens.data()),reinterpret_cast<GM_ADDR>(slots.data()),
        reinterpret_cast<GM_ADDR>(mark.data()),reinterpret_cast<GM_ADDR>(pos.data()),
        reinterpret_cast<GM_ADDR>(epochs.data()),r,t,c,l);
      kernel.Process();
    }
    write(output,last);write(output,resident);write(output,lru);write(output,current);
    write(output,count);write(output,tokens);write(output,slots);write(output,epochs);
  }
  std::cout<<"PASS: archive class executed through CPU DMA/UB shim; not NPU execution\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
