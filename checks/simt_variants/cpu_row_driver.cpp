#include <algorithm>
#include <climits>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "cpu_runtime.h"
#include "kernel_under_test.h"

template <typename T>
struct Guarded {
  static constexpr size_t PAD = 16;
  static constexpr T MARK = static_cast<T>(0x1234567);
  std::vector<T> backing;
  size_t count;
  explicit Guarded(size_t n, T initial = 0) : backing(n + 2 * PAD, MARK), count(n) {
    std::fill(data(), data() + n, initial);
  }
  T* data() { return backing.data() + PAD; }
  const T* data() const { return backing.data() + PAD; }
  T& operator[](size_t index) { return data()[index]; }
  const T& operator[](size_t index) const { return data()[index]; }
  void Check() const {
    for (size_t i = 0; i < PAD; ++i) {
      if (backing[i] != MARK || backing[PAD + count + i] != MARK) {
        throw std::runtime_error("buffer guard overwritten");
      }
    }
  }
};

struct State {
  int rows, topk, capacity;
  int requests = 3, blocks = 16, maxToken = 128, hostBlocks = 48, blockSize = 8;
  Guarded<int64_t> reqIds, lastIds;
  Guarded<int32_t> tokens, stable, visible, mapping, table;
  Guarded<int32_t> slots, lru, current, missCount, missTokens, missSlots;

  State(int r, int t, int c)
      : rows(r), topk(t), capacity(c), reqIds(3), lastIds(r, -1), tokens(r * t, -1), stable(r, 128),
        visible(r, 128), mapping(r), table(3 * 16), slots(r * c, -1), lru(r * c), current(r * t, 71),
        missCount(r, 71), missTokens(r * t, 71), missSlots(r * t, 71) {
    for (int i = 0; i < requests; ++i) reqIds[i] = 1001 + i;
    for (int i = 0; i < requests * blocks; ++i) table[i] = i;
    for (int row = 0; row < rows; ++row) {
      mapping[row] = row % requests;
      std::iota(lru.data() + row * capacity, lru.data() + (row + 1) * capacity, 0);
    }
  }

  bool Valid(int row, int token) const {
    if (token < 0 || token >= maxToken || token >= visible[row]) return false;
    const int req = mapping[row];
    if (req < 0 || req >= requests || blockSize <= 0) return false;
    const int block = token / blockSize;
    if (block >= blocks) return false;
    const int physical = table[req * blocks + block];
    return physical >= 0 && physical < hostBlocks;
  }

  void Check() const {
    reqIds.Check(); lastIds.Check(); tokens.Check(); stable.Check(); visible.Check(); mapping.Check(); table.Check();
    slots.Check(); lru.Check(); current.Check(); missCount.Check(); missTokens.Check(); missSlots.Check();
  }
};

// Independent scalar oracle: ordered lists, no hash, no prefix-scan, no compact scratch.
void OracleRow(State& s, int row) {
  const int tBase = row * s.topk, cBase = row * s.capacity;
  const int req = s.mapping[row];
  const int64_t id = req >= 0 && req < s.requests ? s.reqIds[req] : -1;
  if (s.lastIds[row] != id) {
    std::fill(s.slots.data() + cBase, s.slots.data() + cBase + s.capacity, -1);
    std::iota(s.lru.data() + cBase, s.lru.data() + cBase + s.capacity, 0);
    s.lastIds[row] = id;
  }
  std::fill(s.current.data() + tBase, s.current.data() + tBase + s.topk, -1);
  std::fill(s.missTokens.data() + tBase, s.missTokens.data() + tBase + s.topk, -1);
  std::fill(s.missSlots.data() + tBase, s.missSlots.data() + tBase + s.topk, -1);
  std::unordered_map<int, int> first;
  for (int p = 0; p < s.topk; ++p) if (s.Valid(row, s.tokens[tBase + p])) first.emplace(s.tokens[tBase + p], p);
  std::vector<int> evict, hit, misses;
  const int prefix = std::clamp(s.stable[row], 0, s.maxToken);
  for (int order = 0; order < s.capacity; ++order) {
    const int slot = s.lru[cBase + order];
    if (slot < 0 || slot >= s.capacity) continue;
    int token = s.slots[cBase + slot];
    if (token >= 0 && token < s.maxToken && token >= prefix) s.slots[cBase + slot] = token = -1;
    const auto found = first.find(token);
    if (token >= 0 && token < s.maxToken && found != first.end()) {
      hit.push_back(slot);
      s.current[tBase + found->second] = slot;
    } else {
      evict.push_back(slot);
    }
  }
  for (int p = 0; p < s.topk; ++p) if (s.Valid(row, s.tokens[tBase + p]) && s.current[tBase + p] < 0) misses.push_back(p);
  const size_t assigned = std::min(misses.size(), evict.size());
  for (size_t i = 0; i < assigned; ++i) {
    const int slot = evict[i], p = misses[i], token = s.tokens[tBase + p];
    s.slots[cBase + slot] = token;
    s.current[tBase + p] = slot;
    s.missTokens[tBase + i] = token;
    s.missSlots[tBase + i] = slot;
  }
  s.missCount[row] = assigned;
  std::vector<int> next(evict.begin() + assigned, evict.end());
  next.insert(next.end(), evict.begin(), evict.begin() + assigned);
  next.insert(next.end(), hit.begin(), hit.end());
  std::copy(next.begin(), next.end(), s.lru.data() + cBase);
}

template <typename T>
void Compare(const Guarded<T>& actual, const Guarded<T>& expected, const char* label) {
  actual.Check(); expected.Check();
  for (size_t i = 0; i < actual.count; ++i) {
    if (actual[i] != expected[i]) {
      throw std::runtime_error(std::string(label) + " index=" + std::to_string(i) + " actual=" +
                               std::to_string(actual[i]) + " expected=" + std::to_string(expected[i]));
    }
  }
}

uint64_t HashSize(int t) {
  uint64_t h = 32;
  while (h < static_cast<uint64_t>(t) * 2) h *= 2;
  return h;
}

#ifndef CAPACITY_ARRAYS
#define CAPACITY_ARRAYS 2
#endif

void CheckState(CpuWorkers& workers, State& s, int active, bool ub, int stride = 1) {
  State expected = s;
  for (int row = 0; row < active; ++row) OracleRow(expected, row);
  const uint64_t h = HashSize(s.topk);
  const uint64_t elements = 3 * h + CAPACITY_ARRAYS * s.capacity + s.topk;
  Guarded<int32_t> local(sparse_kv_plan::PLAN_SCAN_STORAGE_ELEMENTS + (ub ? elements : 0), 0x4567);
  Guarded<int32_t> global(s.rows * elements, 0x9876);
  for (int start = 0; start < stride; ++start) {
    workers.Run([&] {
      // Use the actual row routine, with the same strided row traversal as the VF.
      for (int row = start; row < active; row += stride) {
        int32_t* scratch = ub ? local.data() + sparse_kv_plan::PLAN_SCAN_STORAGE_ELEMENTS : global.data() + row * elements;
        ProcessRuntimeRow(local.data(), row, s.reqIds.data(), s.lastIds.data(), s.tokens.data(), s.stable.data(),
                          s.visible.data(), s.mapping.data(), s.table.data(), s.slots.data(), s.lru.data(), s.current.data(),
                          s.missCount.data(), s.missTokens.data(), s.missSlots.data(), scratch, h, s.topk, s.capacity,
                          s.maxToken, s.requests, s.blocks, s.hostBlocks, s.blockSize);
      }
    });
  }
  s.Check(); local.Check(); global.Check();
  Compare(s.lastIds, expected.lastIds, "lastIds"); Compare(s.slots, expected.slots, "slots");
  Compare(s.lru, expected.lru, "lru"); Compare(s.current, expected.current, "current");
  Compare(s.missCount, expected.missCount, "missCount"); Compare(s.missTokens, expected.missTokens, "missTokens");
  Compare(s.missSlots, expected.missSlots, "missSlots");
  if (ub) for (size_t i = 0; i < global.count; ++i) if (global[i] != 0x9876) throw std::runtime_error("UB path touched GM scratch");
}

void RandomSequence(CpuWorkers& workers, bool ub, int topk, int capacity, std::mt19937& rng, int& checks) {
  State s(4, topk, capacity);
  for (int step = 0; step < 12; ++step) {
    for (int row = 0; row < s.rows; ++row) {
      s.mapping[row] = step == 8 && row == 1 ? -1 : (row + step / 4) % s.requests;
      s.visible[row] = step % 3 == 0 ? 128 : 16 + rng() % 112;
      s.stable[row] = step % 4 == 0 ? 0 : static_cast<int>(rng() % 150) - 5;
      for (int p = 0; p < s.topk; ++p) s.tokens[row * s.topk + p] = static_cast<int>(rng() % 145) - 8;
    }
    if (step == 3) s.table[2] = -1;
    if (step == 5) s.table[2] = 2;
    if (step == 7) s.reqIds[0] += 100;
    CheckState(workers, s, step == 9 ? 2 : s.rows, ub, 2);
    ++checks;
  }
}

void SpecialCases(CpuWorkers& workers, bool ub, int& checks) {
  State s(3, 9, 17);
  for (int row = 0; row < s.rows; ++row) {
    const int input[] = {5, 5, 7, -1, 127, 128, 6, 6, 8};
    std::copy(input, input + 9, s.tokens.data() + row * s.topk);
  }
  CheckState(workers, s, 3, ub); ++checks;
  CheckState(workers, s, 3, ub); ++checks;  // Warm repeated TopK, duplicate resident tokens.
  State invalidLru = s;
  invalidLru.lru[2] = -1; invalidLru.lru[3] = s.capacity;
  invalidLru.slots[4] = 5; invalidLru.slots[6] = 5;
  CheckState(workers, invalidLru, 3, ub); ++checks;  // Invalid entries are skipped, not a legal persistent layout.
  s.visible[0] = 0; s.stable[1] = -10; s.stable[2] = 999;
  CheckState(workers, s, 3, ub); ++checks;
  CheckState(workers, s, 0, ub); ++checks;
  State fullHit(1, 7, 17);
  fullHit.lastIds[0] = fullHit.reqIds[0];
  for (int slot = 0; slot < fullHit.capacity; ++slot) fullHit.slots[slot] = slot;
  for (int p = 0; p < fullHit.topk; ++p) fullHit.tokens[p] = 2 * p;
  CheckState(workers, fullHit, 1, ub); ++checks;
  if (fullHit.missCount[0] != 0) throw std::runtime_error("full-hit case generated misses");
  fullHit.stable[0] = 0;  // Same request: suffix invalidation must keep the previous LRU order.
  CheckState(workers, fullHit, 1, ub); ++checks;
  State invalidRequest(2, 5, 11);
  invalidRequest.mapping[0] = invalidRequest.requests;
  invalidRequest.mapping[1] = -1;
  for (int p = 0; p < 10; ++p) invalidRequest.tokens[p] = p;
  CheckState(workers, invalidRequest, 2, ub); ++checks;
  State truncated(1, 9, 3);
  for (int p = 0; p < truncated.topk; ++p) truncated.tokens[p] = p;
  CheckState(workers, truncated, 1, ub); ++checks;
  State mtp(4, 7, 11);
  for (int row = 0; row < 4; ++row) {
    mtp.mapping[row] = row < 3 ? 0 : 1;
    mtp.visible[row] = row < 3 ? 101 + row : 17;
    mtp.stable[row] = row < 3 ? 100 : 16;
    const int input[] = {99, 100, 101, 102, 103, -1, 100};
    std::copy(input, input + 7, mtp.tokens.data() + row * 7);
  }
  CheckState(workers, mtp, 4, ub); ++checks;
  for (int row = 0; row < 3; ++row) {
    mtp.stable[row] = 101; mtp.visible[row] = 102 + row;
  }
  CheckState(workers, mtp, 4, ub); ++checks;
  std::swap(mtp.mapping[0], mtp.mapping[3]);  // Request reordering changes row owner.
  CheckState(workers, mtp, 4, ub); ++checks;
}

int main(int argc, char** argv) {
  try {
    CpuWorkers workers(sparse_kv_plan::PLAN_THREADS);
    std::mt19937 rng(20261012);
    int checks = 0;
    for (bool ub : {false, true}) {
      SpecialCases(workers, ub, checks);
      for (const auto& shape : {std::pair{1, 3}, {17, 41}, {65, 129}}) {
        RandomSequence(workers, ub, shape.first, shape.second, rng, checks);
      }
      if (argc > 1 && std::string(argv[1]) == "--large") {
        for (const auto& shape : {std::pair{256, 512}, {2048, 4096}, {2048, 8192}}) {
          State large(1, shape.first, shape.second);
          for (int p = 0; p < large.topk; ++p) large.tokens[p] = p % 127;
          CheckState(workers, large, 1, ub); ++checks;
          large.stable[0] = 103;
          CheckState(workers, large, 1, ub); ++checks;
        }
      }
    }
    std::cout << "PASS " << checks << " oracle/state checks; threads=" << sparse_kv_plan::PLAN_THREADS << '\n';
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
