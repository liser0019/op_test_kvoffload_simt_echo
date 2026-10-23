"""MTP 性能输入：请求维与 query 行维分开；每条 query 行拥有独立 Plan 状态。

mtp 表示 draft token 数，q_len=1+mtp。这里测试 Plan，不执行 draft 模型、
accept/reject 或 KV 搬运。相同请求的各位置共享 req_id/页表，并从同一份历史
缓存快照分叉；它们不是同时读写一份 resident/LRU，也不是按位置串行调用。
"""
from __future__ import annotations

import numpy as np

SCENARIOS = {
    "cold_reset": (0.0, False), "warm_75pct": (0.75, False),
    "warm_all_hit": (1.0, False), "full_cache_75pct": (0.75, True),
}


def resolve_shape(requests=None, rows=None, mtp=0):
    """--requests 固定请求数；--rows 固定展开行数，二者同时给出时必须相符。"""
    if mtp < 0:
        raise ValueError("mtp 必须非负，表示 draft token 数")
    q_len = mtp + 1
    if rows is not None:
        if rows <= 0 or rows % q_len:
            raise ValueError("--rows 必须是正数且能被 q_len=1+mtp 整除")
        derived = rows // q_len
        if requests is not None and requests != derived:
            raise ValueError("--requests 与 --rows/--mtp 不一致")
        requests = derived
    requests = 32 if requests is None else requests
    if requests <= 0:
        raise ValueError("requests 必须为正数")
    return requests, requests * q_len


def make_mtp_case(requests, topk, capacity, max_token, scenario, seed,
                  mtp=0, block_size=64, shuffle_lru=False, context=None):
    """固定请求数时，mtp=0/3 分别产生 requests/4*requests 条 query 行。

    context 是已提交长度。位置 p 的 visible=context+p+1，覆盖当前 token
    与该位置之前的 draft。MTP 的 stable=context；关闭 MTP 则当前可见内容
    全部稳定。每条 query 尾部包含可见的新 token，不能把 MTP 仅实现为标签。

    warm_all_hit 指历史稳定 TopK 全命中；新增可见 token 仍然 miss。报告必须
    使用 golden 产生的实际 miss 数，不能继续假定此场景 miss=0。
    """
    if requests <= 0 or mtp < 0 or not 0 < topk <= capacity or block_size <= 0:
        raise ValueError("需要 requests>0、mtp>=0、0<topk<=capacity、block_size>0")
    q_len = mtp + 1
    # MTP=0/3 使用同一 committed context，避免仅切换MTP就改变哈希输入分布。
    context = max_token - max(16, q_len) if context is None else int(context)
    if context <= topk or context + q_len > max_token:
        raise ValueError("token 值域不足以容纳稳定 TopK 与完整 MTP 可见后缀")
    fraction, full = SCENARIOS[scenario]
    rng = np.random.default_rng(seed)
    nominal_hits = int(topk * fraction)
    base_query = np.stack([rng.choice(context - 1, topk, replace=False).astype(np.int32) + 1
                           for _ in range(requests)])
    base_resident = np.full((requests, capacity), -1, np.int32)
    base_resident[:, :nominal_hits] = base_query[:, :nominal_hits]
    if full:
        if context - 1 - topk < capacity - nominal_hits:
            raise ValueError("稳定 token 值域不足以构造满缓存且避开 query")
        for req in range(requests):
            others = np.setdiff1d(np.arange(1, context, dtype=np.int32), base_query[req])
            base_resident[req, nominal_hits:] = rng.choice(
                others, capacity - nominal_hits, replace=False)

    # 预埋上一轮投机 token：request ID 匹配，确保不会被 cold reset 掩盖。
    stale = min(mtp, capacity - nominal_hits) if scenario != "cold_reset" else 0
    if stale:
        base_resident[:, nominal_hits:nominal_hits + stale] = (
            context + np.arange(stale, dtype=np.int32))
    base_lru = (np.stack([rng.permutation(capacity).astype(np.int32) for _ in range(requests)])
                if shuffle_lru else np.tile(np.arange(capacity, dtype=np.int32), (requests, 1)))
    row_to_req = np.repeat(np.arange(requests, dtype=np.int32), q_len)
    position = np.tile(np.arange(q_len, dtype=np.int32), requests)
    rows = requests * q_len
    query = np.empty((rows, topk), np.int32)
    for row, (req, pos) in enumerate(zip(row_to_req, position)):
        new_tokens = min(topk, int(pos) + 1)
        query[row, :topk - new_tokens] = base_query[req, :topk - new_tokens]
        query[row, topk - new_tokens:] = np.arange(
            context + int(pos) + 1 - new_tokens, context + int(pos) + 1, dtype=np.int32)
    req_ids = np.arange(1000, 1000 + requests, dtype=np.int64)
    visible = (context + position + 1).astype(np.int32)
    stable = np.full(rows, context, np.int32) if mtp else visible.copy()
    last = (np.full(rows, -1, np.int64) if scenario == "cold_reset"
            else req_ids[row_to_req].copy())
    blocks = (max_token + block_size - 1) // block_size
    return {
        "scenario": scenario, "rows": rows, "requests": requests, "mtp": mtp,
        "q_len": q_len, "context": context, "topk": topk, "capacity": capacity,
        "max_token": max_token, "block_size": block_size, "host_num_blocks": blocks,
        "hits": nominal_hits, "initial_speculative_slots_per_row": stale,
        "req_ids": req_ids, "last_req_ids": last, "topk_indices": query,
        "stable_prefix_lens": stable, "visible_seq_lens": visible,
        "token_to_req": row_to_req, "query_position": position,
        "block_table": np.tile(np.arange(blocks, dtype=np.int32), (requests, 1)),
        "slot_to_token": np.repeat(base_resident, q_len, axis=0),
        "lru_slots": np.repeat(base_lru, q_len, axis=0),
        "reset_mask": (last != req_ids[row_to_req]).astype(np.uint8),
    }


def make_miss_sweep_case(requests, topk, capacity, max_token, seed, mtp,
                         miss_count, distribution="front", block_size=64, shuffle_lru=False):
    """Scan 控制实验：query只含稳定token，精确指定每行miss数及位置。

    与默认四场景区分：此处不强制query包含新token，因此MTP=3也能构造0 miss。
    仍保留逐位置visible、stablePrefix与匹配ID的历史投机槽失效。
    resident初始满缓存；MTP失效后可能空出少量槽，实际淘汰需求随之变化。
    """
    if not 0 <= miss_count <= topk or distribution not in ("front", "back", "uniform"):
        raise ValueError("需要0<=miss_count<=topk，分布为front/back/uniform")
    case = make_mtp_case(requests, topk, capacity, max_token, "full_cache_75pct",
                         seed, mtp, block_size, shuffle_lru)
    context = case["context"]
    rng = np.random.default_rng(seed)
    query = np.stack([rng.choice(context - 1, topk, replace=False).astype(np.int32) + 1
                      for _ in range(requests)])
    missing = np.zeros(topk, dtype=bool)
    if miss_count:
        positions = (np.arange(miss_count) if distribution == "front" else
                     np.arange(topk - miss_count, topk) if distribution == "back" else
                     np.arange(miss_count, dtype=np.int64) * topk // miss_count)
        missing[positions] = True
    hits = topk - miss_count
    if context - 1 - topk < capacity - hits:
        raise ValueError("稳定token值域不足以构造指定miss数的满缓存")
    resident = np.empty((requests, capacity), np.int32)
    for request in range(requests):
        resident[request, :hits] = query[request, ~missing]
        others = np.setdiff1d(np.arange(1, context, dtype=np.int32), query[request])
        resident[request, hits:] = rng.choice(others, capacity - hits, replace=False)
    stale = min(mtp, capacity - hits)
    if stale:
        resident[:, hits:hits + stale] = context + np.arange(stale, dtype=np.int32)
    case.update({"topk_indices": np.repeat(query, mtp + 1, axis=0),
                 "slot_to_token": np.repeat(resident, mtp + 1, axis=0), "hits": hits,
                 "initial_speculative_slots_per_row": stale,
                 "controlled_miss_count": miss_count, "miss_distribution": distribution,
                 "query_kind": "stable tokens only; controlled Scan workload"})
    return case


def describe_case(case, expected):
    """报告实际执行语义；实际 miss 可与历史名义命中率不同。"""
    counts = expected["miss_count"][:case.get("active_rows", case["rows"])]
    prefix = np.clip(case["stable_prefix_lens"], 0, case["max_token"])
    resident = case["slot_to_token"]
    invalidated = ((resident >= prefix[:, None]) & (resident < case["max_token"])
                   & (resident >= 0)).sum(axis=1)
    return {
        "requests": case.get("requests", len(case["req_ids"])),
        "query_rows": case["rows"], "mtp": case.get("mtp", 0),
        "q_len": case.get("q_len", 1),
        "query_kind": case.get("query_kind", "stable prefix plus visible new tokens"),
        "controlled_miss_count": case.get("controlled_miss_count"),
        "miss_distribution": case.get("miss_distribution"),
        "visible_min": int(case["visible_seq_lens"].min()),
        "visible_max": int(case["visible_seq_lens"].max()),
        "stable_min": int(prefix.min()), "stable_max": int(prefix.max()),
        "miss_count_per_row": counts.astype(int).tolist(),
        "total_misses": int(counts.sum()),
        "resident_suffix_slots_before_reset": invalidated.astype(int).tolist(),
        "reset_rows": int(case["reset_mask"].sum()),
        "state_layout": "independent cache per query row; shared request identity and block table",
    }
