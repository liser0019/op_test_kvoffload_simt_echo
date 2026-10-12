// CPU-only SIMT semantic shim. It does not model NPU timing or address spaces.
#pragma once

#include <atomic>
#include <barrier>
#include <cstdint>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

#define __simt_callee__
#define __simt_vf__
#define __aicore__
#define __ubuf__
#define __gm__
#define LAUNCH_BOUND(...)

struct CpuThreadIndex {
  uint32_t x = 0;
};
inline thread_local CpuThreadIndex threadIdx;

struct CpuWarp {
  std::barrier<> sync{32};
  int32_t values[32]{};
};

struct CpuContext {
  std::barrier<> block;
  std::vector<std::unique_ptr<CpuWarp>> warps;
  explicit CpuContext(int threads) : block(threads) {
    for (int w = 0; w < threads / 32; ++w) {
      warps.emplace_back(std::make_unique<CpuWarp>());
    }
  }
};
inline CpuContext* cpuContext = nullptr;

inline void asc_syncthreads() { cpuContext->block.arrive_and_wait(); }

inline int32_t CpuShuffle(int32_t value, uint32_t source) {
  CpuWarp& warp = *cpuContext->warps[threadIdx.x / 32];
  const uint32_t lane = threadIdx.x % 32;
  warp.values[lane] = value;
  warp.sync.arrive_and_wait();
  const int32_t result = warp.values[source];
  warp.sync.arrive_and_wait();
  return result;
}

inline int32_t asc_shfl_up(int32_t value, uint32_t delta, int32_t width) {
  const uint32_t lane = threadIdx.x % 32;
  const uint32_t local = lane % static_cast<uint32_t>(width);
  return CpuShuffle(value, local >= delta ? lane - delta : lane);
}

inline int32_t asc_shfl(int32_t value, uint32_t source, int32_t width) {
  const uint32_t lane = threadIdx.x % 32;
  return CpuShuffle(value, lane / width * width + source % width);
}

inline int32_t asc_atomic_cas(int32_t* pointer, int32_t expected, int32_t desired) {
  std::atomic_ref<int32_t>(*pointer).compare_exchange_strong(expected, desired);
  return expected;
}

inline int32_t asc_atomic_min(int32_t* pointer, int32_t value) {
  std::atomic_ref<int32_t> atom(*pointer);
  int32_t old = atom.load();
  while (old > value && !atom.compare_exchange_weak(old, value)) {}
  return old;
}

inline int32_t asc_atomic_max(int32_t* pointer, int32_t value) {
  std::atomic_ref<int32_t> atom(*pointer);
  int32_t old = atom.load();
  while (old < value && !atom.compare_exchange_weak(old, value)) {}
  return old;
}

class CpuWorkers {
 public:
  explicit CpuWorkers(int threads) : context_(threads), begin_(threads + 1), done_(threads + 1) {
    cpuContext = &context_;
    for (int id = 0; id < threads; ++id) {
      workers_.emplace_back([this, id] {
        threadIdx.x = id;
        for (;;) {
          begin_.arrive_and_wait();
          if (stop_) return;
          task_();
          done_.arrive_and_wait();
        }
      });
    }
  }

  ~CpuWorkers() {
    stop_ = true;
    begin_.arrive_and_wait();
    for (auto& worker : workers_) worker.join();
  }

  void Run(std::function<void()> task) {
    task_ = std::move(task);
    begin_.arrive_and_wait();
    done_.arrive_and_wait();
  }

 private:
  CpuContext context_;
  std::barrier<> begin_;
  std::barrier<> done_;
  std::vector<std::thread> workers_;
  std::function<void()> task_;
  bool stop_ = false;
};
