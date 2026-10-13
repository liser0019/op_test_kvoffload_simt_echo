#include "symbol_aliases.h"
#include "upstream/sparse_kv_plan_config.h"
static_assert(sparse_kv_plan_peer_04_parallel_scan::PLAN_WARP_COUNT <= 64U, "peer pair scan supports at most 64 warps");
#include "upstream/sparse_kv_plan_kernel.cpp"
