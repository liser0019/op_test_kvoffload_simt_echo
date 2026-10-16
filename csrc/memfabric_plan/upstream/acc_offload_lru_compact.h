/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
*/
#ifndef ACC_OFFLOAD_LRU_COMPACT_H
#define ACC_OFFLOAD_LRU_COMPACT_H

#include "kernel_operator.h"

#define HYBM_AICORE_KERNEL __attribute__((always_inline)) __aicore__ __inline__

// Block dim used to launch the LRU compact kernel. Must match the per-core
// workspace (token_mark / token_pos / epochs) first dim allocated by the host.
constexpr uint32_t LRU_COMPACT_BLOCK_DIM = 8;

constexpr int64_t LRU_UB_ALIGN = 32;
constexpr int32_t LRU_EPOCH_RESET_THRESHOLD = 1 << 30;
constexpr int32_t LRU_INVALID_TOKEN = -1;
constexpr int64_t LRU_CLEAR_BUF_ELEMS = 1024; // int32 elems used to chunk-clear mark/pos on epoch overflow

// Faithful (scalar-style) NPU port of the CPU lru_resident_compact in
// vllm-ascend/.../sfa_kv_offload/cpu_sparse_attn.cpp. Each AIV core owns a
// contiguous slice of rows (strided: row % blockNum == blockIdx) and a private
// mark/pos/epoch workspace in HBM. Per-row hot arrays (topk_indices, lru_slots,
// slot_to_token, current_slots, miss_tokens/slots, hit/evictable/assigned lists)
// live in UB; token_mark/token_pos (sized max_token, too large for UB) live in
// HBM and are accessed through scalar __gm__ loads/stores. epoch is mirrored to
// a UB scalar (MTE2 load at kernel start / MTE3 store at kernel end) so the 8
// per-core epochs (which share one 64B cacheline) are never written by scalar +
// dcci, avoiding the multi-core cacheline overwrite race. miss_count and
// last_req_ids are likewise written back via MTE3 (DataCopyPad UB->GM), not
// scalar + dcci.
class LruResidentCompactKernel {
public:
    HYBM_AICORE_KERNEL LruResidentCompactKernel() {}

    HYBM_AICORE_KERNEL void Init(GM_ADDR req_ids, GM_ADDR last_req_ids, GM_ADDR topk_indices,
                                 GM_ADDR stable_prefix_lens, GM_ADDR slot_to_token, GM_ADDR lru_slots,
                                 GM_ADDR current_slots, GM_ADDR miss_count, GM_ADDR miss_tokens, GM_ADDR miss_slots,
                                 GM_ADDR token_mark_workspace, GM_ADDR token_pos_workspace, GM_ADDR epochs,
                                 int64_t num_reqs, int64_t topk, int64_t capacity, int64_t max_token)
    {
        aivNum_ = AscendC::GetBlockNum();
        aivIndex_ = AscendC::GetBlockIdx();
        num_reqs_ = num_reqs;
        topk_ = static_cast<int32_t>(topk);
        capacity_ = static_cast<int32_t>(capacity);
        max_token_ = max_token;

        req_ids_ = reinterpret_cast<__gm__ int64_t *>(req_ids);
        last_req_ids_ = reinterpret_cast<__gm__ int64_t *>(last_req_ids);
        topk_indices_ = reinterpret_cast<__gm__ int32_t *>(topk_indices);
        stable_prefix_lens_ = reinterpret_cast<__gm__ int32_t *>(stable_prefix_lens);
        slot_to_token_ = reinterpret_cast<__gm__ int32_t *>(slot_to_token);
        lru_slots_ = reinterpret_cast<__gm__ int32_t *>(lru_slots);
        current_slots_ = reinterpret_cast<__gm__ int32_t *>(current_slots);
        miss_count_ = reinterpret_cast<__gm__ int32_t *>(miss_count);
        miss_tokens_ = reinterpret_cast<__gm__ int32_t *>(miss_tokens);
        miss_slots_ = reinterpret_cast<__gm__ int32_t *>(miss_slots);

        token_mark_ = reinterpret_cast<__gm__ int32_t *>(token_mark_workspace) + aivIndex_ * max_token_;
        token_pos_ = reinterpret_cast<__gm__ int32_t *>(token_pos_workspace) + aivIndex_ * max_token_;
        epochGm_ = reinterpret_cast<__gm__ int32_t *>(epochs) + aivIndex_;

        uint32_t topkBytes = AlignUpU32(static_cast<uint32_t>(topk_) * sizeof(int32_t));
        uint32_t capBytes = AlignUpU32(static_cast<uint32_t>(capacity_) * sizeof(int32_t));

        pipe_.InitBuffer(topkBuf_, topkBytes);
        pipe_.InitBuffer(lruSlotsBuf_, capBytes);
        pipe_.InitBuffer(slotToTokenBuf_, capBytes);
        pipe_.InitBuffer(currentSlotsBuf_, topkBytes);
        pipe_.InitBuffer(missTokensBuf_, topkBytes);
        pipe_.InitBuffer(missSlotsBuf_, topkBytes);
        pipe_.InitBuffer(hitSlotsBuf_, capBytes);
        pipe_.InitBuffer(evictableSlotsBuf_, capBytes);
        pipe_.InitBuffer(assignedMissSlotsBuf_, topkBytes);
        pipe_.InitBuffer(missPositionsBuf_, topkBytes);
        pipe_.InitBuffer(clearBuf_, static_cast<uint32_t>(LRU_CLEAR_BUF_ELEMS * sizeof(int32_t)));
        pipe_.InitBuffer(epochBuf_, static_cast<uint32_t>(LRU_UB_ALIGN));
        pipe_.InitBuffer(scalarStageBuf_, static_cast<uint32_t>(LRU_UB_ALIGN));
    }

    HYBM_AICORE_KERNEL void Process()
    {
        LoadEpoch();
        AscendC::PipeBarrier<PIPE_ALL>();
        for (int64_t row = aivIndex_; row < num_reqs_; row += aivNum_) {
            ProcessOneRow(row);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        StoreEpoch();
    }

private:
    HYBM_AICORE_KERNEL uint32_t AlignUpU32(uint32_t bytes)
    {
        return (bytes + static_cast<uint32_t>(LRU_UB_ALIGN) - 1) &
               ~(static_cast<uint32_t>(LRU_UB_ALIGN) - 1);
    }

    HYBM_AICORE_KERNEL bool IsValidToken(int32_t token)
    {
        return token >= 0 && token < max_token_;
    }

    HYBM_AICORE_KERNEL void ResetRow(AscendC::LocalTensor<int32_t> &slotToToken,
                                     AscendC::LocalTensor<int32_t> &lruSlots)
    {
        AscendC::Duplicate<int32_t>(slotToToken, LRU_INVALID_TOKEN, capacity_);
        AscendC::PipeBarrier<PIPE_ALL>();
        for (int32_t i = 0; i < capacity_; ++i) {
            lruSlots.SetValue(i, i);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    HYBM_AICORE_KERNEL void ClearMarkPos()
    {
        auto clearLocal = clearBuf_.Get<int32_t>();
        AscendC::Duplicate<int32_t>(clearLocal, 0, static_cast<int64_t>(LRU_CLEAR_BUF_ELEMS));
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        for (int64_t off = 0; off < max_token_; off += LRU_CLEAR_BUF_ELEMS) {
            int64_t n = (off + LRU_CLEAR_BUF_ELEMS <= max_token_) ? LRU_CLEAR_BUF_ELEMS : (max_token_ - off);
            AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(n * sizeof(int32_t)), 0, 0, 0};
            AscendC::GlobalTensor<int32_t> markGm;
            markGm.SetGlobalBuffer(token_mark_ + off, static_cast<uint64_t>(n));
            AscendC::DataCopyPad(markGm, clearLocal, cp);
            AscendC::GlobalTensor<int32_t> posGm;
            posGm.SetGlobalBuffer(token_pos_ + off, static_cast<uint64_t>(n));
            AscendC::DataCopyPad(posGm, clearLocal, cp);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    HYBM_AICORE_KERNEL void LoadEpoch()
    {
        auto epochLocal = epochBuf_.Get<int32_t>();
        AscendC::DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        AscendC::GlobalTensor<int32_t> epochGm;
        epochGm.SetGlobalBuffer(epochGm_, 1);
        AscendC::DataCopyPad(epochLocal, epochGm, cp, padParams);
        AscendC::PipeBarrier<PIPE_ALL>();
        epochVal_ = epochLocal.GetValue(0);
    }

    HYBM_AICORE_KERNEL void StoreEpoch()
    {
        auto epochLocal = epochBuf_.Get<int32_t>();
        epochLocal.SetValue(0, epochVal_);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        AscendC::GlobalTensor<int32_t> epochGm;
        epochGm.SetGlobalBuffer(epochGm_, 1);
        AscendC::DataCopyPad(epochGm, epochLocal, cp);
    }

    HYBM_AICORE_KERNEL void StoreLastReqId(int64_t row, int64_t reqId)
    {
        auto stage = scalarStageBuf_.Get<int64_t>();
        stage.SetValue(0, reqId);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(sizeof(int64_t)), 0, 0, 0};
        AscendC::GlobalTensor<int64_t> gm;
        gm.SetGlobalBuffer(last_req_ids_ + row, 1);
        AscendC::DataCopyPad(gm, stage, cp);
    }

    HYBM_AICORE_KERNEL void StoreMissCount(int64_t row, int32_t count)
    {
        auto stage = scalarStageBuf_.Get<int32_t>();
        stage.SetValue(0, count);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        AscendC::GlobalTensor<int32_t> gm;
        gm.SetGlobalBuffer(miss_count_ + row, 1);
        AscendC::DataCopyPad(gm, stage, cp);
    }

    HYBM_AICORE_KERNEL void ProcessOneRow(int64_t row)
    {
        AscendC::DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        AscendC::DataCopyExtParams topkParams{1, static_cast<uint32_t>(topk_ * sizeof(int32_t)), 0, 0, 0};
        AscendC::DataCopyExtParams capParams{1, static_cast<uint32_t>(capacity_ * sizeof(int32_t)), 0, 0, 0};

        AscendC::GlobalTensor<int32_t> topkGm;
        topkGm.SetGlobalBuffer(topk_indices_ + row * topk_, static_cast<uint64_t>(topk_));
        auto topkLocal = topkBuf_.Get<int32_t>();
        AscendC::DataCopyPad(topkLocal, topkGm, topkParams, padParams);

        AscendC::GlobalTensor<int32_t> lruSlotsGm;
        lruSlotsGm.SetGlobalBuffer(lru_slots_ + row * capacity_, static_cast<uint64_t>(capacity_));
        auto lruSlotsLocal = lruSlotsBuf_.Get<int32_t>();
        AscendC::DataCopyPad(lruSlotsLocal, lruSlotsGm, capParams, padParams);

        AscendC::GlobalTensor<int32_t> slotToTokenGm;
        slotToTokenGm.SetGlobalBuffer(slot_to_token_ + row * capacity_, static_cast<uint64_t>(capacity_));
        auto slotToTokenLocal = slotToTokenBuf_.Get<int32_t>();
        AscendC::DataCopyPad(slotToTokenLocal, slotToTokenGm, capParams, padParams);

        AscendC::PipeBarrier<PIPE_ALL>();

        auto currentSlotsLocal = currentSlotsBuf_.Get<int32_t>();
        AscendC::Duplicate<int32_t>(currentSlotsLocal, LRU_INVALID_TOKEN, topk_);
        auto missTokensLocal = missTokensBuf_.Get<int32_t>();
        AscendC::Duplicate<int32_t>(missTokensLocal, LRU_INVALID_TOKEN, topk_);
        auto missSlotsLocal = missSlotsBuf_.Get<int32_t>();
        AscendC::Duplicate<int32_t>(missSlotsLocal, LRU_INVALID_TOKEN, topk_);
        AscendC::PipeBarrier<PIPE_ALL>();

        int64_t reqId = req_ids_[row];
        if (last_req_ids_[row] != reqId) {
            ResetRow(slotToTokenLocal, lruSlotsLocal);
            StoreLastReqId(row, reqId);
        }

        int32_t stablePrefixLen = stable_prefix_lens_[row];
        if (stablePrefixLen < 0) {
            stablePrefixLen = 0;
        } else if (stablePrefixLen > max_token_) {
            stablePrefixLen = static_cast<int32_t>(max_token_);
        }

        int32_t base = epochVal_ + 1;
        if (base >= LRU_EPOCH_RESET_THRESHOLD) {
            ClearMarkPos();
            base = 1;
        }
        epochVal_ = base;

        for (int32_t pos = 0; pos < topk_; ++pos) {
            int32_t token = topkLocal.GetValue(pos);
            if (IsValidToken(token) && token_mark_[token] != base) {
                token_mark_[token] = base;
                token_pos_[token] = pos;
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        auto hitSlotsLocal = hitSlotsBuf_.Get<int32_t>();
        auto evictableSlotsLocal = evictableSlotsBuf_.Get<int32_t>();
        int32_t hitCount = 0;
        int32_t evictableCount = 0;
        for (int32_t order = 0; order < capacity_; ++order) {
            int32_t slot = lruSlotsLocal.GetValue(order);
            if (slot < 0 || slot >= capacity_) {
                continue;
            }
            int32_t token = slotToTokenLocal.GetValue(slot);
            // The speculative suffix (token >= stablePrefixLen) may have been
            // overwritten in the CPU KV pool, so invalidate the slot.
            if (IsValidToken(token) && token >= stablePrefixLen) {
                slotToTokenLocal.SetValue(slot, LRU_INVALID_TOKEN);
                token = LRU_INVALID_TOKEN;
            }
            if (IsValidToken(token) && token_mark_[token] == base) {
                int32_t pos = token_pos_[token];
                currentSlotsLocal.SetValue(pos, slot);
                hitSlotsLocal.SetValue(hitCount, slot);
                ++hitCount;
            } else {
                evictableSlotsLocal.SetValue(evictableCount, slot);
                ++evictableCount;
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        auto missPosLocal = missPositionsBuf_.Get<int32_t>();
        int32_t localMissCount = 0;
        for (int32_t pos = 0; pos < topk_; ++pos) {
            int32_t token = topkLocal.GetValue(pos);
            if (IsValidToken(token) && currentSlotsLocal.GetValue(pos) < 0) {
                missTokensLocal.SetValue(localMissCount, token);
                missPosLocal.SetValue(localMissCount, pos);
                ++localMissCount;
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        auto assignedMissSlotsLocal = assignedMissSlotsBuf_.Get<int32_t>();
        int32_t assignCount = (localMissCount < evictableCount) ? localMissCount : evictableCount;
        for (int32_t missIdx = 0; missIdx < assignCount; ++missIdx) {
            int32_t slot = evictableSlotsLocal.GetValue(missIdx);
            int32_t token = missTokensLocal.GetValue(missIdx);
            int32_t pos = missPosLocal.GetValue(missIdx);
            slotToTokenLocal.SetValue(slot, token);
            currentSlotsLocal.SetValue(pos, slot);
            missSlotsLocal.SetValue(missIdx, slot);
            assignedMissSlotsLocal.SetValue(missIdx, slot);
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        int32_t writePos = 0;
        for (int32_t idx = assignCount; idx < evictableCount; ++idx) {
            lruSlotsLocal.SetValue(writePos, evictableSlotsLocal.GetValue(idx));
            ++writePos;
        }
        for (int32_t idx = 0; idx < assignCount; ++idx) {
            lruSlotsLocal.SetValue(writePos, assignedMissSlotsLocal.GetValue(idx));
            ++writePos;
        }
        for (int32_t idx = 0; idx < hitCount; ++idx) {
            lruSlotsLocal.SetValue(writePos, hitSlotsLocal.GetValue(idx));
            ++writePos;
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        AscendC::GlobalTensor<int32_t> currentSlotsGm;
        currentSlotsGm.SetGlobalBuffer(current_slots_ + row * topk_, static_cast<uint64_t>(topk_));
        AscendC::DataCopyPad(currentSlotsGm, currentSlotsLocal, topkParams);

        StoreMissCount(row, assignCount);

        AscendC::GlobalTensor<int32_t> missTokensGm;
        missTokensGm.SetGlobalBuffer(miss_tokens_ + row * topk_, static_cast<uint64_t>(topk_));
        AscendC::DataCopyPad(missTokensGm, missTokensLocal, topkParams);

        AscendC::GlobalTensor<int32_t> missSlotsGm;
        missSlotsGm.SetGlobalBuffer(miss_slots_ + row * topk_, static_cast<uint64_t>(topk_));
        AscendC::DataCopyPad(missSlotsGm, missSlotsLocal, topkParams);

        AscendC::GlobalTensor<int32_t> slotToTokenGmOut;
        slotToTokenGmOut.SetGlobalBuffer(slot_to_token_ + row * capacity_, static_cast<uint64_t>(capacity_));
        AscendC::DataCopyPad(slotToTokenGmOut, slotToTokenLocal, capParams);

        AscendC::GlobalTensor<int32_t> lruSlotsGmOut;
        lruSlotsGmOut.SetGlobalBuffer(lru_slots_ + row * capacity_, static_cast<uint64_t>(capacity_));
        AscendC::DataCopyPad(lruSlotsGmOut, lruSlotsLocal, capParams);

        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TBuf<> topkBuf_;
    AscendC::TBuf<> lruSlotsBuf_;
    AscendC::TBuf<> slotToTokenBuf_;
    AscendC::TBuf<> currentSlotsBuf_;
    AscendC::TBuf<> missTokensBuf_;
    AscendC::TBuf<> missSlotsBuf_;
    AscendC::TBuf<> hitSlotsBuf_;
    AscendC::TBuf<> evictableSlotsBuf_;
    AscendC::TBuf<> assignedMissSlotsBuf_;
    AscendC::TBuf<> missPositionsBuf_;
    AscendC::TBuf<> clearBuf_;
    AscendC::TBuf<> epochBuf_;
    AscendC::TBuf<> scalarStageBuf_;

    uint32_t aivNum_ = 0;
    uint32_t aivIndex_ = 0;
    int64_t num_reqs_ = 0;
    int32_t topk_ = 0;
    int32_t capacity_ = 0;
    int64_t max_token_ = 0;

    __gm__ int64_t *req_ids_ = nullptr;
    __gm__ int64_t *last_req_ids_ = nullptr;
    __gm__ int32_t *topk_indices_ = nullptr;
    __gm__ int32_t *stable_prefix_lens_ = nullptr;
    __gm__ int32_t *slot_to_token_ = nullptr;
    __gm__ int32_t *lru_slots_ = nullptr;
    __gm__ int32_t *current_slots_ = nullptr;
    __gm__ int32_t *miss_count_ = nullptr;
    __gm__ int32_t *miss_tokens_ = nullptr;
    __gm__ int32_t *miss_slots_ = nullptr;
    __gm__ int32_t *token_mark_ = nullptr;
    __gm__ int32_t *token_pos_ = nullptr;
    __gm__ int32_t *epochGm_ = nullptr;
    int32_t epochVal_ = 0;
};

#endif // ACC_OFFLOAD_LRU_COMPACT_H