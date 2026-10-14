#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare BCDE+fused+1024 and compile-time Plan micro-optimizations.

--cpu-check checks inputs/goldens only. checks/check_plan_kernels.py executes
the actual row source with CPU shims. Neither supplies NPU performance.
Event timing includes host submission/tiling H2D gaps; use the profiler for
real kernel durations. Restore and validation stay outside measured calls.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
I32_MAX = np.iinfo(np.int32).max
BLOCK_SIZE = 64
SCENARIOS = {
    "cold_reset": (0.0, False), "warm_75pct": (0.75, False),
    "warm_all_hit": (1.0, False), "full_cache_75pct": (0.75, True),
}
SIMT_STATE = ("last_req_ids", "slot_to_token", "lru_slots")
SIMT_OUTPUT = ("current_slots", "miss_count", "miss_tokens", "miss_slots")
CATALOG = json.loads((ROOT / "plan_variants_manifest.json").read_text())
VARIANTS = {item["name"]: item for item in CATALOG["variants"]}
IMPLEMENTATIONS = tuple(VARIANTS)

def plan_layout(topk, capacity, rows, name):
    threads = VARIANTS[name]["threads"]
    hash_capacity = max(32, 1 << (2 * topk - 1).bit_length())
    row_elements = 3 * hash_capacity + capacity + topk
    scan_bytes = ((2 * (threads // 32) + 8) * 4 + 31) & ~31
    aligned_bytes = (row_elements * 4 + 31) & ~31
    use_ub = aligned_bytes <= CATALOG["row_ub_limit_bytes"]
    return {"threads": threads, "scan_bytes": scan_bytes, "capacity_arrays": 1,
            "hash_capacity": hash_capacity, "row_elements": row_elements,
            "scratch_path": "UB" if use_ub else "GM",
            "local_memory_bytes": scan_bytes + (aligned_bytes if use_ub else 0),
            "workspace_elements": 1 if use_ub else rows * row_elements}


def execution_order(names, iteration):
    """逐轮轮转；每满 len(names) 轮，每个实现都在每个执行位置出现一次。"""
    offset = iteration % len(names)
    return (*names[offset:], *names[:offset])

def make_case(rows, topk, capacity, max_token, scenario, seed, block_size=BLOCK_SIZE, shuffle_lru=False):
    """保持 main 性能用例的输入语义和 RNG 顺序。

    本基准只使用正数、合法、不重复的 query；不模拟 MTP、非法来源或重复 token。
    stable/visible 都是 max_token；MTP 与非法输入由额外审计覆盖。
    """
    fraction, full_cache = SCENARIOS[scenario]
    rng = np.random.default_rng(seed)
    query = np.stack([rng.choice(max_token - 1, topk, replace=False).astype(np.int32) + 1
                      for _ in range(rows)])
    hits = int(topk * fraction)
    resident = np.full((rows, capacity), -1, np.int32)
    resident[:, :hits] = query[:, :hits]

    if full_cache:
        # 补满缓存的 token 必须避开整条 query，否则实际命中率会超过设定值。
        if max_token - 1 - topk < capacity - hits:
            raise ValueError("token 值域不足：无法填满缓存并保留指定 miss 数")
        for row in range(rows):
            others = np.setdiff1d(np.arange(1, max_token, dtype=np.int32), query[row])
            resident[row, hits:] = rng.choice(others, capacity - hits, replace=False)

    lru = (np.stack([rng.permutation(capacity).astype(np.int32) for _ in range(rows)])
           if shuffle_lru else np.tile(np.arange(capacity, dtype=np.int32), (rows, 1)))
    req_ids = np.arange(1000, 1000 + rows, dtype=np.int64)
    # 只有冷场景换 request。warm/full 场景必须匹配，才能实际使用预埋的缓存。
    last_req_ids = req_ids.copy() if hits or full_cache else np.full(rows, -1, np.int64)
    num_blocks = (max_token + block_size - 1) // block_size
    return {
        "scenario": scenario, "rows": rows, "topk": topk, "capacity": capacity,
        "max_token": max_token, "block_size": block_size, "hits": hits, "host_num_blocks": num_blocks,
        "req_ids": req_ids, "last_req_ids": last_req_ids,
        "topk_indices": query,
        "stable_prefix_lens": np.full(rows, max_token, np.int32),
        "visible_seq_lens": np.full(rows, max_token, np.int32),
        "token_to_req": np.arange(rows, dtype=np.int32),
        "block_table": np.tile(np.arange(num_blocks, dtype=np.int32), (rows, 1)),
        "slot_to_token": resident,
        "lru_slots": lru,
        "reset_mask": (req_ids != last_req_ids).astype(np.uint8),
    }

def reference_simt(case):
    """Independent scalar lists/dictionaries; no hash, scan or UB algorithm.

    Includes duplicate TopK/resident tokens, suffix invalidation, partial active
    rows and insufficient slots, preserving the baseline's exact output order.
    """
    rows, topk, capacity = (case[key] for key in ("rows", "topk", "capacity"))
    out = {key: case[key].copy() for key in SIMT_STATE}
    poison = case.get("output_poison", -1)
    out.update({key: np.full((rows, topk), poison, np.int32)
                for key in ("current_slots", "miss_tokens", "miss_slots")})
    out["miss_count"] = np.full(rows, poison, np.int32)
    block_size = case.get("block_size", BLOCK_SIZE)
    active = min(rows, max(0, case.get("active_rows", rows)))
    for row in range(active):
        request = int(case["token_to_req"][row])
        valid_request = 0 <= request < len(case["req_ids"])
        request_id = case["req_ids"][request] if valid_request else -1
        resident, lru = out["slot_to_token"][row], out["lru_slots"][row].copy()
        if out["last_req_ids"][row] != request_id:
            resident.fill(-1)
            lru = np.arange(capacity, dtype=np.int32)
            out["last_req_ids"][row] = request_id
        for key in ("current_slots", "miss_tokens", "miss_slots"):
            out[key][row].fill(-1)
        valid = []
        for token in case["topk_indices"][row]:
            token = int(token)
            allowed = (0 <= token < case["max_token"] and token < case["visible_seq_lens"][row]
                       and valid_request and block_size > 0)
            block = token // block_size if allowed else -1
            allowed = allowed and block < case["block_table"].shape[1]
            physical = int(case["block_table"][request, block]) if allowed else -1
            valid.append(allowed and 0 <= physical < case["host_num_blocks"])
        first = {}
        for pos, token in enumerate(case["topk_indices"][row]):
            if valid[pos]: first.setdefault(int(token), pos)
        evict, hit = [], []
        prefix = max(0, min(case["max_token"], int(case["stable_prefix_lens"][row])))
        for slot in lru:
            slot = int(slot)
            if not 0 <= slot < capacity: continue
            token = int(resident[slot])
            if 0 <= token < case["max_token"] and token >= prefix:
                resident[slot] = token = -1
            if 0 <= token < case["max_token"] and token in first:
                hit.append(slot)
                out["current_slots"][row, first[token]] = slot
            else:
                evict.append(slot)
        misses = [pos for pos in range(topk) if valid[pos] and out["current_slots"][row, pos] < 0]
        assigned = min(len(misses), len(evict))
        for miss in range(assigned):
            pos, slot = misses[miss], evict[miss]
            token = case["topk_indices"][row, pos]
            resident[slot] = token
            out["current_slots"][row, pos] = slot
            out["miss_tokens"][row, miss] = token
            out["miss_slots"][row, miss] = slot
        out["miss_count"][row] = assigned
        next_lru = evict[assigned:] + evict[:assigned] + hit
        lru[:len(next_lru)] = next_lru
        out["lru_slots"][row] = lru
    return out


def make_audit_cases():
    """Small adversarial cases, outside performance capture; also used on NPU."""
    cases = []
    case = make_case(4, 7, 11, 128, "cold_reset", 17, 8)
    case["topk_indices"][:] = [99, 100, 101, 102, 103, -1, 100]
    case["token_to_req"][:] = [0, 0, 0, 1]
    case["visible_seq_lens"][:] = [101, 102, 103, 17]
    case["stable_prefix_lens"][:] = [100, 100, 100, 16]
    cases.append(("mtp_cold_duplicates", case))
    previous = reference_simt(case)
    for step in range(2):
        warm = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in case.items()}
        warm.update({key: previous[key].copy() for key in SIMT_STATE})
        warm["visible_seq_lens"][:3] = np.arange(102 + step, 105 + step, dtype=np.int32)
        warm["stable_prefix_lens"][:3] = 100 + step
        cases.append((f"mtp_rewritten_suffix_{step}", warm))
        previous = reference_simt(warm)
        case = warm
    invalid = make_case(3, 9, 17, 128, "warm_75pct", 18, 7)
    invalid["visible_seq_lens"][:] = [-1, 0, 128]
    invalid["token_to_req"][1] = -1
    invalid["topk_indices"][2] = [0, 5, 5, 7, -1, 127, 128, 6, 8]
    invalid["block_table"][2, 0] = -1
    invalid["block_table"][2, 1] = invalid["host_num_blocks"]
    invalid["stable_prefix_lens"][:] = [-3, 999, 128]
    cases.append(("invalid_sources_non_power_two_pages", invalid))
    short = make_case(1, 9, 17, 128, "cold_reset", 19, 8)
    short["capacity"] = 3
    short["slot_to_token"] = np.full((1, 3), -1, np.int32)
    short["lru_slots"] = np.arange(3, dtype=np.int32).reshape(1, 3)
    cases.append(("insufficient_slots", short))
    for active in (0, 2):
        padded = make_case(3, 9, 17, 128, "cold_reset", 20, 8)
        padded["active_rows"] = active
        cases.append((f"padded_active_{active}", padded))
    for capacity in (32768, 32769):
        owner = make_case(1, 7, capacity, 65536, "warm_all_hit", 21, 64)
        owner["slot_to_token"].fill(-1)
        owner["slot_to_token"][0, -7:] = owner["topk_indices"][0]
        cases.append((f"owner_pack_boundary_{capacity}", owner))
    for _, case in cases:
        case["output_poison"] = -777
    return cases


def prepare_npu(torch, case, unused, device, names=IMPLEMENTATIONS):
    def upload(array):
        return torch.from_numpy(np.ascontiguousarray(array)).to(device)
    inputs = {key: upload(case[key]) for key in (
        "req_ids", "topk_indices", "stable_prefix_lens", "visible_seq_lens", "token_to_req", "block_table")}
    inputs["active_rows"] = torch.tensor([case.get("active_rows", case["rows"])], dtype=torch.int32, device=device)
    snapshot = {key: upload(case[key]) for key in SIMT_STATE}
    initial = {name: snapshot for name in names}
    states = {name: {key: value.clone() for key, value in snapshot.items()} for name in names}
    rows, topk, capacity = (case[key] for key in ("rows", "topk", "capacity"))
    for name in names:
        for key in SIMT_OUTPUT:
            shape = (rows,) if key == "miss_count" else (rows, topk)
            states[name][key] = torch.full(shape, case.get("output_poison", -777), dtype=torch.int32, device=device)
        layout = plan_layout(topk, capacity, rows, name)
        states[name]["layout"] = layout
        states[name]["workspace"] = torch.empty(layout["workspace_elements"], dtype=torch.int32, device=device)
    torch.npu.synchronize()
    return inputs, initial, states


def invoke(torch, name, case, inputs, state):
    getattr(torch.ops.sparse_kv_plan_op, VARIANTS[name]["op_name"])(
        inputs["req_ids"], inputs["topk_indices"], inputs["stable_prefix_lens"],
        inputs["visible_seq_lens"], inputs["token_to_req"], inputs["block_table"],
        inputs["active_rows"], state["last_req_ids"], state["slot_to_token"],
        state["lru_slots"], state["current_slots"], state["miss_count"],
        state["miss_tokens"], state["miss_slots"], state["workspace"],
        case["topk"], case["capacity"], case["max_token"],
        case.get("block_size", BLOCK_SIZE), case["host_num_blocks"])


def audit_case(torch, case, names, device, repeats=3):
    expected = {name: reference_simt(case) for name in names}
    inputs, initial, states = prepare_npu(torch, case, None, device, names)
    for iteration in range(repeats):
        for name in names:
            restore_state(torch, name, initial, states[name])
            for key in SIMT_OUTPUT: states[name][key].fill_(case.get("output_poison", -777))
            states[name]["workspace"].fill_(I32_MAX)
            invoke(torch, name, case, inputs, states[name])
            torch.npu.synchronize()
            check_npu_result(name, states[name], expected, f"audit-{iteration}")
    return inputs, initial, states, expected


def restore_state(torch, name, initial, state):
    """每次调用前还原同一份持久状态；workspace 是临时区，沿用 main 的复用方式。"""
    for key, source in initial[name].items():
        state[key].copy_(source)
    torch.npu.synchronize()

def check_npu_result(name, state, expected, phase):
    """逐字段全等检查，读取发生在同步之后、计时区间之外。"""
    actual = {key: state[key].cpu().numpy() for key in expected[name]}
    for key, reference in expected[name].items():
        np.testing.assert_array_equal(actual[key], reference, err_msg=f"{phase}/{name}/{key}")
    return actual

def measure_one(torch, name, case, inputs, initial, state, events):
    restore_state(torch, name, initial, state)
    start, end = events
    before = time.perf_counter_ns()
    start.record()
    invoke(torch, name, case, inputs, state)
    end.record()
    end.synchronize()
    host_us = (time.perf_counter_ns() - before) / 1000.0
    event_us = start.elapsed_time(end) * 1000.0  # elapsed_time 返回毫秒。
    if not np.isfinite(event_us) or event_us <= 0:
        raise RuntimeError(f"{name}: 无效的 Event 耗时 {event_us}")
    return event_us, host_us

def summary(samples):
    return {
        "count": len(samples), "p50_us": float(np.percentile(samples, 50)),
        "p95_us": float(np.percentile(samples, 95)),
        "min_us": float(np.min(samples)), "max_us": float(np.max(samples)),
    }

def array_hash(arrays):
    """保存输入指纹，不把实际测试数据写入结果 JSON。"""
    digest = hashlib.sha256()
    for key, array in sorted(arrays.items()):
        digest.update(key.encode())
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key, default in [("rows", 32), ("topk", 2048), ("capacity", 4096), ("max-token", 262144),
                         ("block-size", 64), ("seed", 2026), ("device", 1), ("warmup", 10),
                         ("repeat", 68), ("audit-repeat", 3)]:
        parser.add_argument("--" + key, type=int, default=default)
    parser.add_argument("--scenario", choices=("all", *SCENARIOS), default="all")
    parser.add_argument("--implementations", nargs="+", choices=IMPLEMENTATIONS, default=list(IMPLEMENTATIONS))
    parser.add_argument("--reference", choices=IMPLEMENTATIONS, default="baseline")
    parser.add_argument("--library", type=Path, default=ROOT / "build/libsimt_plan_variants_ops.so")
    parser.add_argument("--json", type=Path, default=Path("results/plan_variants.json"))
    parser.add_argument("--cpu-check", action="store_true")
    parser.add_argument("--shuffle-lru", action="store_true", help="Randomize LRU order for owner gathers")
    parser.add_argument("--audit-edges", action="store_true", help="MTP/duplicates/invalid/owner boundaries before timing")
    parser.add_argument("--verify-each-sample", action="store_true")
    args = parser.parse_args()
    names = tuple(args.implementations)
    if len(set(names)) != len(names) or args.reference not in names:
        parser.error("Implementations must be unique and include --reference")
    if not (args.rows > 0 and 0 < args.topk <= args.capacity and args.max_token > args.topk
            and args.block_size > 0 and args.warmup >= 0 and args.repeat > 0 and args.audit_repeat > 0):
        parser.error("Invalid shape, block size, warmup or repeat")
    if max(args.rows * (args.capacity + args.topk), args.rows * args.max_token, args.block_size) > I32_MAX:
        parser.error("Shape exceeds supported int32 indexing")
    from checks.plan_build_info import catalog_identity
    identity = catalog_identity()
    scenarios = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    results = {"metadata": {"source_identity": identity, "implementations": list(names),
                           "reference": args.reference, "shuffle_lru": args.shuffle_lru}, "scenarios": {}}
    results["metadata"]["shape"] = {key: getattr(args, key) for key in
                                   ("rows", "topk", "capacity", "max_token", "block_size")}
    if not args.cpu_check:
        import torch
        import torch_npu
        if not args.library.is_file() or not torch.npu.is_available():
            parser.error("Build this branch's library and use an available NPU")
        torch.npu.set_device(args.device)
        device = torch.device(f"npu:{args.device}")
        torch.ops.load_library(str(args.library.resolve()))
        if json.loads(torch.ops.sparse_kv_plan_op.plan_variants_info()) != identity:
            parser.error("Compiled source identity is stale; rebuild")
        results["metadata"].update({"library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
                                    "device": torch.npu.get_device_name(args.device),
                                    "torch": torch.__version__, "torch_npu": torch_npu.__version__,
                                    "timing": "Event stream interval, not pure kernel duration"})
    if args.audit_edges:
        for label, case in make_audit_cases():
            reference_simt(case)
            if not args.cpu_check: audit_case(torch, case, names, device, args.audit_repeat)
            print("Edge audit:", label, "CPU golden only" if args.cpu_check else "NPU passed", flush=True)
    for scenario in scenarios:
        case = make_case(args.rows, args.topk, args.capacity, args.max_token, scenario, args.seed, args.block_size, args.shuffle_lru)
        expected_once = reference_simt(case)
        if args.cpu_check:
            assert np.all(expected_once["miss_count"] == args.topk - case["hits"])
            print(scenario, "CPU golden/layout passed; no kernel execution or performance measurement")
            print(plan_layout(args.topk, args.capacity, args.rows, names[0]))
            continue
        inputs, initial, states, expected = audit_case(torch, case, names, device, args.audit_repeat)
        events = {name: (torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)) for name in names}
        for iteration in range(args.warmup):
            for name in execution_order(names, iteration):
                measure_one(torch, name, case, inputs, initial, states[name], events[name])
                if args.verify_each_sample: check_npu_result(name, states[name], expected, "warmup")
        samples = {name: {"event_us": [], "host_us": []} for name in names}
        orders = []
        for iteration in range(args.repeat):
            order = execution_order(names, iteration)
            orders.append(list(order))
            for name in order:
                event_us, host_us = measure_one(torch, name, case, inputs, initial, states[name], events[name])
                samples[name]["event_us"].append(event_us)
                samples[name]["host_us"].append(host_us)
                if args.verify_each_sample: check_npu_result(name, states[name], expected, "sample")
        for name in names: check_npu_result(name, states[name], expected, "after-measurement")
        stats = {name: {"event": summary(samples[name]["event_us"]), "host": summary(samples[name]["host_us"]),
                        "raw_samples": samples[name], "layout": states[name]["layout"]} for name in names}
        baseline = stats[args.reference]["event"]["p50_us"]
        for name in names:
            stats[name]["speedup_vs_reference_event_p50"] = baseline / stats[name]["event"]["p50_us"]
            print(f"{scenario}/{name}: Event P50={stats[name]['event']['p50_us']:.2f} us, "
                  f"vs {args.reference}={stats[name]['speedup_vs_reference_event_p50']:.3f}x", flush=True)
        results["scenarios"][scenario] = {"correctness": "passed", "statistics": stats,
                                          "execution_orders": orders,
                                          "case_sha256": array_hash({key: value for key, value in case.items()
                                                                     if isinstance(value, np.ndarray)})}
    if not args.cpu_check:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
        print("Results:", args.json.resolve())
        if args.verify_each_sample: print("Audit reads affect caches; use profile_plan_variants.py for performance.")


if __name__ == "__main__":
    main()
