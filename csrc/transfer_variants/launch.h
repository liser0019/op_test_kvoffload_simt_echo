// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "../transfer/launch.h"
// mask -1由调用者直接启动原版；0..15有各自唯一kernel名。
extern "C" void launch_sparse_kv_transfer_variant(
    int32_t mask, GM_ADDR missCount, GM_ADDR missTokens, GM_ADDR missSlots, GM_ADDR tokenToReq,
    GM_ADDR blockTable, GM_ADDR hostCacheBases, GM_ADDR activeRows, GM_ADDR residentK,
    GM_ADDR residentV, GM_ADDR tiling, uint32_t blockDim, void* stream);
