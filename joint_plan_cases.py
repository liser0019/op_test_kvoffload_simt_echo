"""联合Plan输入与独立语义检查；不模拟设备哈希/Scan，不规定联合miss的桶顺序。"""
from __future__ import annotations
import numpy as np
from compare import reference_simt, array_hash

INPUTS = ("req_ids", "topk_indices", "stable_prefix_lens", "visible_seq_lens", "token_to_req")
PERSISTENT = ("last_req_ids", "slot_to_token", "lru_slots", "row_modes")
OUTPUTS = ("current_slots", "miss_count", "miss_tokens", "miss_slots", "resident_rows", "resident_block_table")
SCENARIOS = ("cold_reset", "full_cache_75pct", "full_cache_90pct", "full_cache_stable_hit")
OVERLAPS = ("shared", "partial", "independent", "disjoint")
POISON = -777777


def layout(topk, capacity, rows, joint, max_token=None, bitmap=False, turbo=False):
    def hash_size(n):
        return max(32, 1 << (2 * n - 1).bit_length())
    h, j = hash_size(topk), hash_size(capacity) if joint else 0
    row = 3 * h + capacity + topk
    scratch = max(row, 2 * j + capacity + (4 * topk if turbo and joint else 0))
    if bitmap and joint:
        scratch = max(row, (max_token + 31) // 32 + 2 * capacity)
    aligned = (scratch * 4 + 31) & ~31
    ub = aligned <= 112 * 1024
    return {"hash_capacity": h, "joint_hash_capacity": j, "scratch_elements": scratch,
            "workspace_elements": rows * (scratch + max_token) if bitmap else (1 if ub else rows * scratch),
            "dynamic_ub_bytes": 288 + (aligned if ub else 0), "scratch_path": "UB" if ub else "GM"}


def groups(case, joint):
    active = min(case["rows"], max(0, case.get("active_rows", case["rows"])))
    row = 0
    while row < active:
        end = row + 1
        if joint:
            while end < active and case["token_to_req"][end] == case["token_to_req"][row]:
                end += 1
        yield row, end
        row = end


def source_valid(case, row, token):
    req = int(case["token_to_req"][row])
    return (0 <= req < len(case["req_ids"]) and 0 <= token < case["max_token"]
            and token < int(case["visible_seq_lens"][row]))


def valid_sets(case):
    return [{int(t) for t in case["topk_indices"][row] if source_valid(case, row, int(t))}
            for row in range(case["rows"])]


def make_case(requests=32, topk=2048, capacity=4096, max_token=262144, mtp=0,
              scenario="full_cache_75pct", overlap="shared", seed=2026, block_size=64):
    if mtp not in (0, 1, 3) or scenario not in SCENARIOS or overlap not in OVERLAPS:
        raise ValueError("MTP只取0/1/3，工况/重合模式必须在目录内")
    if requests <= 0 or not 0 < topk <= capacity or block_size <= 0 or capacity % block_size:
        raise ValueError("需要requests>0、0<topk<=capacity且capacity可被block_size整除")
    if max_token > np.iinfo(np.int32).max or capacity > np.iinfo(np.int32).max // 2:
        raise ValueError("token/capacity超出kernel的int32范围")
    context, width = max_token - 16, mtp + 1
    if context <= 4 * topk + capacity + 1:
        raise ValueError("稳定token值域需大于4*topk+capacity+1；提高max-token")
    rows = requests * width
    req = np.repeat(np.arange(requests, dtype=np.int32), width)
    positions = np.tile(np.arange(width, dtype=np.int32), requests)
    query, base = np.empty((rows, topk), np.int32), np.empty((requests, capacity), np.int32)
    percentage = {"cold_reset": 0, "full_cache_75pct": 75, "full_cache_90pct": 90,
                  "full_cache_stable_hit": 100}[scenario]
    hits = min(topk - 1, topk * percentage // 100)
    shift = {"shared": 0, "partial": max(1, topk // 4), "disjoint": topk, "independent": 0}[overlap]
    lru = np.empty_like(base)
    for request in range(requests):
        # 全值域排列使MTP=0/1/3的第0行query、初始resident和LRU严格相同。
        rng = np.random.default_rng(seed + request * 101)
        pool = (rng.permutation(context - 1) + 1).astype(np.int32)
        base[request, :hits] = pool[:hits]
        base[request, hits:] = pool[4 * topk:4 * topk + capacity - hits]
        # 相同旧投机后缀，三种MTP均失效；不把旧答案当作本轮新KV。
        stale = min(3, capacity - hits)
        base[request, hits:hits + stale] = context + 1 + np.arange(stale, dtype=np.int32)
        lru[request] = np.random.default_rng(seed + 70001 + request).permutation(capacity)
        for p in range(width):
            row, new = request * width + p, min(topk, p + 1)
            if overlap == "independent" and p > 0:
                # Indexer 的每条 query 独立选择历史 token；不要求相同前缀/有序。
                select = np.random.default_rng(seed + 300001 + request * 17 + p)
                query[row, :topk - new] = select.choice(pool[:2 * topk], topk - new, replace=False)
            else:
                query[row, :topk - new] = pool[p * shift:p * shift + topk - new]
            query[row, topk - new:] = np.arange(context + p + 1 - new, context + p + 1)
            if overlap == "independent":
                np.random.default_rng(seed + 400001 + request * 17 + p).shuffle(query[row])
    ids = np.arange(1000, 1000 + requests, dtype=np.int64)
    visible = context + positions + 1
    return {"rows": rows, "requests": requests, "mtp": mtp, "q_len": width,
            "topk": topk, "capacity": capacity, "max_token": max_token, "context": context,
            "block_size": block_size,
            "scenario": scenario, "overlap": overlap, "seed": seed,
            "req_ids": ids, "topk_indices": query, "token_to_req": req,
            "visible_seq_lens": visible.astype(np.int32),
            "stable_prefix_lens": (np.full(rows, context, np.int32) if mtp else visible.copy()),
            "last_req_ids": (np.full(rows, -1, np.int64) if scenario == "cold_reset" else ids[req].copy()),
            "slot_to_token": np.repeat(base, width, axis=0), "lru_slots": np.repeat(lru, width, axis=0)}


def initial_arrays(case, joint, max_group_queries=None):
    rows, topk, capacity = (case[k] for k in ("rows", "topk", "capacity"))
    out = {key: case[key].copy() for key in PERSISTENT if key != "row_modes"}
    out["row_modes"] = np.zeros(rows, np.int32)
    owners = np.arange(rows, dtype=np.int32)
    if joint and case["scenario"] != "cold_reset":
        sets = valid_sets(case)
        for first, end in groups(case, True):
            request = int(case["token_to_req"][first])
            fits_queries = max_group_queries is None or (end - first) * topk <= max_group_queries
            if (end - first > 1 and 0 <= request < len(case["req_ids"])
                    and len(set().union(*sets[first:end])) <= capacity and fits_queries):
                out["row_modes"][first:end] = 2
                out["row_modes"][first] = 1
                owners[first:end] = first
    pages = capacity // case["block_size"]
    out.update({key: np.full((rows, topk), POISON, np.int32)
                for key in ("current_slots", "miss_tokens", "miss_slots")})
    out["miss_count"] = np.full(rows, POISON, np.int32)
    out["resident_rows"] = owners
    out["resident_block_table"] = owners[:, None] * pages + np.arange(pages, dtype=np.int32)
    return out


def describe(case):
    sets = valid_sets(case)
    unions = [len(set().union(*sets[first:end])) for first, end in groups(case, True)]
    return {"scenario": case["scenario"], "overlap": case["overlap"], "mtp": case["mtp"],
            "requests": case["requests"], "query_rows": case["rows"], "q_len": case["q_len"],
            "topk": case["topk"], "capacity": case["capacity"], "max_token": case["max_token"],
            "row0_intersections": [[len(sets[first] & sets[row]) for row in range(first, end)]
                                    for first, end in groups(case, True)],
            "selection_semantics": "per-query tokens; no sorted/shared-prefix assumption",
            "union_sizes": unions, "union_fits": [u <= case["capacity"] for u in unions],
            "valid_positions_per_row": [sum(source_valid(case, r, int(t)) for t in query)
                                        for r, query in enumerate(case["topk_indices"])],
            "query_kind": "stable history plus causally visible new tokens; historical hit fraction is nominal",
            "logical_input_sha256": array_hash({k: case[k] for k in (*INPUTS, "last_req_ids", "slot_to_token", "lru_slots")})}


def validate(case, before, actual, joint, max_group_queries=None):
    """Private对独立golden逐字段比较；joint对集合、LRU、miss填充和逐行寻址完整校验。"""
    expected = {key: value.copy() for key, value in before.items()}
    sets, group_info = valid_sets(case), []
    capacity, topk = case["capacity"], case["topk"]
    pages = capacity // case["block_size"]
    for first, end in groups(case, joint):
        request = int(case["token_to_req"][first])
        union = set().union(*sets[first:end])
        fits_queries = max_group_queries is None or (end - first) * topk <= max_group_queries
        shared = end - first > 1 and 0 <= request < len(case["req_ids"]) and len(union) <= capacity and fits_queries
        group_info.append({"first": first, "end": end, "union": len(union),
                           "path": "joint" if shared else "overflow_private"
                           if end - first > 1 and (len(union) > capacity or not fits_queries)
                           else "private"})
        if not shared:
            for row in range(first, end):
                one = dict(case, rows=1, active_rows=1)
                for key in ("topk_indices", "stable_prefix_lens", "visible_seq_lens", "token_to_req"):
                    one[key] = case[key][row:row + 1].copy()
                for key in ("last_req_ids", "slot_to_token", "lru_slots"):
                    one[key] = before[key][row:row + 1].copy()
                if before["row_modes"][row] != 0:
                    one["last_req_ids"][0] = -1
                gold = reference_simt(one)
                for key in ("last_req_ids", "slot_to_token", "lru_slots", "current_slots", "miss_count"):
                    expected[key][row] = gold[key][0]
                count = int(gold["miss_count"][0])
                for key in ("miss_tokens", "miss_slots"):
                    np.testing.assert_array_equal(actual[key][row, :count], gold[key][0, :count], err_msg=key)
                expected["row_modes"][row] = 0
                expected["resident_rows"][row] = row
                expected["resident_block_table"][row] = row * pages + np.arange(pages)
            continue

        req_id = case["req_ids"][request]
        reset = before["row_modes"][first] != 1 or before["last_req_ids"][first] != req_id
        resident = np.full(capacity, -1, np.int32) if reset else before["slot_to_token"][first].copy()
        lru = np.arange(capacity) if reset else before["lru_slots"][first]
        stable = max(0, min(int(v) for v in case["stable_prefix_lens"][first:end]))
        resident[resident >= stable] = -1
        evict = [int(s) for s in lru if int(resident[s]) not in union]
        hit = [int(s) for s in lru if int(resident[s]) in union]
        hit_tokens = {int(resident[s]) for s in hit}
        missing = union - hit_tokens
        count = len(missing)
        expected_counts = [min(topk, max(0, count - (row - first) * topk)) for row in range(first, end)]
        np.testing.assert_array_equal(actual["miss_count"][first:end], expected_counts, err_msg="joint miss counts")
        tokens = np.concatenate([actual["miss_tokens"][row, :n] for row, n in zip(range(first, end), expected_counts)])
        slots = np.concatenate([actual["miss_slots"][row, :n] for row, n in zip(range(first, end), expected_counts)])
        if len(tokens) != len(set(map(int, tokens))) or set(map(int, tokens)) != missing:
            raise AssertionError("联合miss必须与独立计算的missing集合相等，且每token仅一条记录")
        np.testing.assert_array_equal(slots, evict[:count], err_msg="joint eviction slots / allocation order")
        for token, slot in zip(tokens, slots):
            resident[int(slot)] = token
        expected["slot_to_token"][first] = resident
        expected["lru_slots"][first] = evict[count:] + evict[:count] + hit
        expected["last_req_ids"][first] = req_id
        token_slots = {int(resident[s]): int(s) for s in hit + evict[:count]}
        for row, n in zip(range(first, end), expected_counts):
            expected["miss_count"][row] = n
            expected["row_modes"][row] = 1 if row == first else 2
            expected["resident_rows"][row] = first
            expected["resident_block_table"][row] = first * pages + np.arange(pages)
            expected["current_slots"][row] = [token_slots[int(t)] if source_valid(case, row, int(t)) else -1
                                                 for t in case["topk_indices"][row]]

    # follower缓存内容及last_req_ids也全量核对；非活动行应完全保留调用前值。
    for key in (*PERSISTENT, "current_slots", "miss_count", "resident_rows", "resident_block_table"):
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
    active = min(case["rows"], max(0, case.get("active_rows", case["rows"])))
    for key in ("miss_tokens", "miss_slots"):
        np.testing.assert_array_equal(actual[key][active:], before[key][active:], err_msg="inactive/" + key)
    for row in range(active):
        owner = int(actual["resident_rows"][row])
        np.testing.assert_array_equal(np.sort(actual["lru_slots"][owner]), np.arange(capacity), err_msg="LRU permutation")
        for pos, token in enumerate(case["topk_indices"][row]):
            slot = int(actual["current_slots"][row, pos])
            if source_valid(case, row, int(token)):
                if slot == -1 and expected["current_slots"][row, pos] == -1:
                    continue  # 原private接口允许因resident重复占槽而未分配；golden已核对。
                if not 0 <= slot < capacity or actual["slot_to_token"][owner, slot] != token:
                    raise AssertionError(f"逐行KV语义错误 row={row}, position={pos}, token={token}, slot={slot}")
            elif slot != -1:
                raise AssertionError("不可见/非法token必须输出-1，不能使用其他行加载的未来token")
    return {"status": "passed", "groups": group_info,
            "joint_groups": sum(g["path"] == "joint" for g in group_info),
            "overflow_groups": sum(g["path"] == "overflow_private" for g in group_info),
            "miss_count_per_row": actual["miss_count"][:active].astype(int).tolist(),
            "total_transfer_descriptors": int(actual["miss_count"][:active].sum())}


def validate_abcdf(case, before, actual):
    """独立14-tensor无来源页表接口：不伪造row_modes/owner/page输出，不走联合协议归一化。"""
    one = dict(case)
    for key in ("last_req_ids", "slot_to_token", "lru_slots"):
        one[key] = before[key]
    gold = reference_simt(one)
    active = min(case["rows"], max(0, case.get("active_rows", case["rows"])))
    for key in ("last_req_ids", "slot_to_token", "lru_slots", "current_slots", "miss_count"):
        np.testing.assert_array_equal(actual[key][:active], gold[key][:active], err_msg="abcdf/" + key)
        np.testing.assert_array_equal(actual[key][active:], before[key][active:], err_msg="abcdf/inactive/" + key)
    for key in ("miss_tokens", "miss_slots"):
        for row in range(active):
            count = int(gold["miss_count"][row])
            np.testing.assert_array_equal(actual[key][row, :count], gold[key][row, :count], err_msg="abcdf/" + key)
        np.testing.assert_array_equal(actual[key][active:], before[key][active:], err_msg="abcdf/inactive/" + key)
    return {"status": "passed", "protocol": "original_abcdf_14_tensors; no block_table or joint metadata",
            "groups": [{"first": r, "end": r + 1, "path": "abcdf"} for r in range(active)],
            "joint_groups": 0, "overflow_groups": 0,
            "miss_count_per_row": actual["miss_count"][:active].astype(int).tolist(),
            "total_transfer_descriptors": int(actual["miss_count"][:active].sum())}
