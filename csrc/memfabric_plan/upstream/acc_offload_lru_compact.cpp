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

#include "acc_offload_lru_compact.h"

extern "C" __global__ __aicore__ void OffloadLruCompactOps(
    GM_ADDR req_ids, GM_ADDR last_req_ids, GM_ADDR topk_indices, GM_ADDR stable_prefix_lens,
    GM_ADDR slot_to_token, GM_ADDR lru_slots, GM_ADDR current_slots, GM_ADDR miss_count, GM_ADDR miss_tokens,
    GM_ADDR miss_slots, GM_ADDR token_mark_workspace, GM_ADDR token_pos_workspace, GM_ADDR epochs, int64_t num_reqs,
    int64_t topk, int64_t capacity, int64_t max_token)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    LruResidentCompactKernel op;
    op.Init(req_ids, last_req_ids, topk_indices, stable_prefix_lens, slot_to_token, lru_slots, current_slots,
            miss_count, miss_tokens, miss_slots, token_mark_workspace, token_pos_workspace, epochs, num_reqs, topk,
            capacity, max_token);
    op.Process();
}

extern "C" void OffloadOpsLruCompact(uint64_t req_ids, uint64_t last_req_ids, uint64_t topk_indices,
                                      uint64_t stable_prefix_lens, uint64_t slot_to_token, uint64_t lru_slots,
                                      uint64_t current_slots, uint64_t miss_count, uint64_t miss_tokens,
                                      uint64_t miss_slots, uint64_t token_mark_workspace, uint64_t token_pos_workspace,
                                      uint64_t epochs, int64_t num_reqs, int64_t topk, int64_t capacity,
                                      int64_t max_token, void *stream)
{
    constexpr uint32_t blockDim = LRU_COMPACT_BLOCK_DIM;
    OffloadLruCompactOps<<<blockDim, nullptr, stream>>>(
        reinterpret_cast<GM_ADDR>(req_ids), reinterpret_cast<GM_ADDR>(last_req_ids),
        reinterpret_cast<GM_ADDR>(topk_indices), reinterpret_cast<GM_ADDR>(stable_prefix_lens),
        reinterpret_cast<GM_ADDR>(slot_to_token), reinterpret_cast<GM_ADDR>(lru_slots),
        reinterpret_cast<GM_ADDR>(current_slots), reinterpret_cast<GM_ADDR>(miss_count),
        reinterpret_cast<GM_ADDR>(miss_tokens), reinterpret_cast<GM_ADDR>(miss_slots),
        reinterpret_cast<GM_ADDR>(token_mark_workspace), reinterpret_cast<GM_ADDR>(token_pos_workspace),
        reinterpret_cast<GM_ADDR>(epochs), num_reqs, topk, capacity, max_token);
}
