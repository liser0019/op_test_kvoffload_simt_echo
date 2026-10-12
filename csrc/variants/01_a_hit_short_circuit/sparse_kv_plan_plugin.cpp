// SPDX-License-Identifier: Apache-2.0
// 每个变体编译一个独立注册单元；共享与原 SIMT 相同的校验和异步下发代码。
#include "../variant_launch.h"
#define SPARSE_KV_VARIANT_OP_NAME "sparse_kv_plan_v01"
#define SPARSE_KV_VARIANT_LAUNCH launch_sparse_kv_plan_v01
#define SPARSE_KV_VARIANT_CAPACITY_ARRAYS 2U
#define SPARSE_KV_VARIANT_API_LABEL "SparseKvPlan_v01"
#include "../../simt/sparse_kv_plan_plugin.cpp"
