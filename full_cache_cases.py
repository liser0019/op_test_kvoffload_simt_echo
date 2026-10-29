"""满缓存命中率控制实验。独立于MTP输入：stable=visible=max_token，100%确实零miss。"""
from __future__ import annotations

import numpy as np
import compare as plan

HIT_PERCENTAGES = (75, 90, 100)
PATTERN_RECIPE = "physical token uint64 identity + K/V tag + mixed uint32 byte positions; version 1"


def make_full_cache_case(rows, topk, capacity, max_token, hit_percent, seed=2026,
                         block_size=64, shuffle_lru=True, shuffle_pages=True):
    """各命中率共享query、页表、LRU、query→resident槽位置和非query填充。

    floor(topk*percent/100)个query被放入缓存；所有capacity槽均有效且不重复。
    hit集合嵌套：75%的hit在90%、100%中仍位于同一槽。无request reset/投机失效。
    """
    if hit_percent not in HIT_PERCENTAGES:
        raise ValueError("hit_percent只支持75/90/100")
    if not (rows > 0 and 0 < topk <= capacity and max_token > topk + capacity and block_size > 0):
        raise ValueError("需要rows>0、0<topk<=capacity、max_token>topk+capacity、block_size>0")
    blocks = (max_token + block_size - 1) // block_size
    if max(rows * (capacity + topk), rows * max_token, rows * blocks, block_size) > plan.I32_MAX:
        raise ValueError("输入超过当前int32属性/索引范围")
    rng = np.random.default_rng(seed)
    query = np.stack([rng.choice(max_token - 1, topk, replace=False).astype(np.int32) + 1
                      for _ in range(rows)])
    resident = np.empty((rows, capacity), np.int32)
    hits = topk * hit_percent // 100
    for row in range(rows):
        non_query = np.setdiff1d(np.arange(1, max_token, dtype=np.int32), query[row])
        # 先填满缓存，再替换指定query；随机数消耗不依赖命中率。
        resident[row] = rng.choice(non_query, capacity, replace=False)
        query_order = rng.permutation(topk)
        query_slots = rng.permutation(capacity)[:topk]
        selected = query_order[:hits]
        resident[row, query_slots[selected]] = query[row, selected]
    lru = (np.stack([rng.permutation(capacity).astype(np.int32) for _ in range(rows)])
           if shuffle_lru else np.tile(np.arange(capacity, dtype=np.int32), (rows, 1)))
    physical = np.arange(rows * blocks, dtype=np.int32)
    if shuffle_pages:
        physical = rng.permutation(physical)
    ids = np.arange(1000, 1000 + rows, dtype=np.int64)
    return {
        "scenario": f"full_cache_hit{hit_percent}", "rows": rows, "requests": rows,
        "topk": topk, "capacity": capacity, "max_token": max_token, "block_size": block_size,
        "host_num_blocks": rows * blocks, "hit_percent_requested": hit_percent,
        "hits_per_row": hits, "misses_per_row": topk - hits,
        "hit_percent_actual": 100.0 * hits / topk, "seed": seed,
        "source_layout": "unique physical pages across requests; shuffled" if shuffle_pages else
                         "unique physical pages across requests; identity",
        "req_ids": ids, "last_req_ids": ids.copy(), "topk_indices": query,
        "stable_prefix_lens": np.full(rows, max_token, np.int32),
        "visible_seq_lens": np.full(rows, max_token, np.int32),
        "token_to_req": np.arange(rows, dtype=np.int32),
        "block_table": physical.reshape(rows, blocks), "slot_to_token": resident, "lru_slots": lru,
        "reset_mask": np.zeros(rows, np.uint8),
    }


def physical_tokens(case, logical_tokens):
    """独立CPU地址模型，返回每行token对应的全局物理token下标。"""
    tokens = np.asarray(logical_tokens, dtype=np.int64)
    if tokens.ndim != 2 or tokens.shape[0] != case["rows"] or np.any(tokens < 0):
        raise ValueError("地址模型需要[rows,width]合法token")
    requests = case["token_to_req"].astype(np.int64)[:, None]
    physical = case["block_table"][requests, tokens // case["block_size"]].astype(np.int64)
    return physical * case["block_size"] + tokens % case["block_size"]


def payload_pattern(token_ids, token_bytes, kind):
    """每个物理token/字节位置的数据不同，K/V不同；不使用全零payload掩盖错地址。"""
    ids = np.asarray(token_ids, dtype=np.uint64).reshape(-1)
    words = np.arange((token_bytes + 3) // 4, dtype=np.uint64)
    value = ((ids[:, None] * 0x9e3779b1 + (ids[:, None] >> 16) * 0x85ebca6b
              + kind * 0xc2b2ae35) ^ (words[None, :] * 0x27d4eb2d)) & np.uint64(0xffffffff)
    value = ((value ^ (value >> 15)) * 0x2c1b3c6d) & np.uint64(0xffffffff)
    value ^= value >> 12
    result = value.astype("<u4").view(np.uint8).reshape(len(ids), -1)[:, :token_bytes].copy()
    if token_bytes >= 8:
        # 前8字节显式编码物理token，避免hash碰撞使两个token的整段payload相同。
        identity = (ids ^ (np.uint64(kind) << np.uint64(56))).astype("<u8")
        result[:, :8] = identity.view(np.uint8).reshape(-1, 8)
    return result


def assert_case(case, golden):
    r, t, c = (case[key] for key in ("rows", "topk", "capacity"))
    expected_hits = case["hits_per_row"]
    for row in range(r):
        resident = case["slot_to_token"][row]
        assert len(np.unique(resident)) == c and np.all(resident >= 0), "初始缓存必须满且唯一"
        actual = np.isin(case["topk_indices"][row], resident).sum()
        assert actual == expected_hits, (actual, expected_hits)
        np.testing.assert_array_equal(np.sort(case["lru_slots"][row]), np.arange(c))
    np.testing.assert_array_equal(golden["miss_count"], np.full(r, t - expected_hits))
    np.testing.assert_array_equal(golden["slot_to_token"][
        np.arange(r)[:, None], golden["current_slots"]], case["topk_indices"])
    # 满缓存的所有miss都必须淘汰历史token；没有空槽、reset或suffix失效。
    assert np.all(case["last_req_ids"] == case["req_ids"])
    assert np.all(case["stable_prefix_lens"] == case["max_token"])


def reference_transfer_ids(case, miss_count, miss_tokens, miss_slots, token_to_req=None,
                           block_table=None, active_rows=None):
    """Transfer独立标量地址模型：由初始payload出发，仅覆盖合法miss目标。

    与Plan最终resident反推结果交叉检查；也用于非法来源/槽、count和active_rows边界。
    返回每个resident槽的物理源token身份，随后逐字节生成期望payload。
    """
    ids = physical_tokens(case, case["slot_to_token"]).copy()
    table = case["block_table"] if block_table is None else block_table
    requests = case["token_to_req"] if token_to_req is None else token_to_req
    active = case["rows"] if active_rows is None else min(case["rows"], max(0, active_rows))
    for row in range(active):
        request = int(requests[row])
        if not 0 <= request < table.shape[0]:
            continue
        count = min(case["topk"], max(0, int(miss_count[row])))
        for index in range(count):
            token, slot = int(miss_tokens[row, index]), int(miss_slots[row, index])
            if token < 0 or not 0 <= slot < case["capacity"]:
                continue
            block, offset = divmod(token, case["block_size"])
            if block >= table.shape[1]:
                continue
            physical = int(table[request, block])
            if 0 <= physical < case["host_num_blocks"]:
                ids[row, slot] = physical * case["block_size"] + offset
    return ids


def describe(case, bytes_k, bytes_v):
    misses = case["rows"] * case["misses_per_row"]
    source_tokens = case["host_num_blocks"] * case["block_size"]
    return {
        key: case[key] for key in ("scenario", "rows", "topk", "capacity", "max_token",
                                  "hit_percent_requested", "hit_percent_actual", "hits_per_row",
                                  "misses_per_row", "source_layout")
    } | {
        "total_misses": misses, "token_bytes_k": bytes_k, "token_bytes_v": bytes_v,
        "payload_bytes_copied": misses * (bytes_k + bytes_v),
        "host_pool_bytes": source_tokens * (bytes_k + bytes_v),
        "resident_payload_bytes": case["rows"] * case["capacity"] * (bytes_k + bytes_v),
        "source_memory": "aclrtMallocHost + ACL_HOST_REG_MAPPED; Host DRAM -> NPU",
        "pattern_recipe": PATTERN_RECIPE,
        "mtp": 0, "stable_equals_visible_max_token": True,
    }
