#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""阅读入口：main() → make_case() → 两份 reference → NPU 状态 → 校验 → 计时。

比较原 SIMT、作者新版 Echo 和 11 个 SIMT 候选；设备源码在本目录 csrc/。
先 bash build.sh，再运行本文件.

运行 NPU：python3 compare.py --device 1 --warmup 10 --repeat 50
运行 CPU：python3 compare.py --cpu-check
CPU 模式只检查测试数据和参考结果，绝不产生 NPU 性能数据。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

I32_MAX = np.iinfo(np.int32).max
BLOCK_SIZE = 64
SCENARIOS = {
    "cold_reset": (0.0, False),
    "warm_75pct": (0.75, False),
    "warm_all_hit": (1.0, False),
    "full_cache_75pct": (0.75, True),
}
SIMT_STATE = ("last_req_ids", "slot_to_token", "lru_slots")
SIMT_OUTPUT = ("current_slots", "miss_count", "miss_tokens", "miss_slots")
ECHO_STATE = ("htd", "dth", "pri", "free_slots", "avail", "fifo")
ECHO_OUTPUT = ("current_slots", "miss_host_pos", "miss_alloc_flat", "miss_mask")
VARIANTS = {v["name"]: v for v in json.loads(
    (Path(__file__).resolve().parent / "variants_manifest.json").read_text())["variants"]}
IMPLEMENTATIONS = ("simt", "echo", *VARIANTS)


def plan_layout(topk, capacity, rows, name):
    """每版自己的真实布局：C 系列只有一份 capacity 分类数组，其他版为两份。"""
    arrays = VARIANTS[name]["capacity_arrays"] if name in VARIANTS else 2
    hash_capacity = max(32, 1 << (2 * topk - 1).bit_length())
    row_elements = 3 * hash_capacity + arrays * capacity + topk
    aligned_bytes = (row_elements * 4 + 31) & ~31
    use_ub = aligned_bytes <= 112 * 1024
    return {"capacity_arrays": arrays, "hash_capacity": hash_capacity,
            "row_elements": row_elements, "scratch_path": "UB" if use_ub else "GM",
            "local_memory_bytes": 544 + (aligned_bytes if use_ub else 0),
            "workspace_elements": 1 if use_ub else rows * row_elements}


def execution_order(names, iteration):
    """逐轮轮转；每满 len(names) 轮，每个实现都在每个执行位置出现一次。"""
    offset = iteration % len(names)
    return (*names[offset:], *names[:offset])


# =============================================================================
# 1. 一份共同输入：相同 query、resident token 集合和历史 LRU 顺序
# =============================================================================
def make_case(rows, topk, capacity, max_token, scenario, seed):
    """保持 main 性能用例的输入语义和 RNG 顺序。

    本基准只使用正数、合法、不重复的 query；不模拟 MTP、非法来源或重复 token。
    stable/visible 都是 max_token，所以两个 kernel 都不会失效投机后缀。
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

    req_ids = np.arange(1000, 1000 + rows, dtype=np.int64)
    # 只有冷场景换 request。warm/full 场景必须匹配，才能实际使用预埋的缓存。
    last_req_ids = req_ids.copy() if hits or full_cache else np.full(rows, -1, np.int64)
    num_blocks = (max_token + BLOCK_SIZE - 1) // BLOCK_SIZE
    return {
        "scenario": scenario, "rows": rows, "topk": topk, "capacity": capacity,
        "max_token": max_token, "hits": hits, "host_num_blocks": num_blocks,
        "req_ids": req_ids, "last_req_ids": last_req_ids,
        "topk_indices": query,
        "stable_prefix_lens": np.full(rows, max_token, np.int32),
        "visible_seq_lens": np.full(rows, max_token, np.int32),
        "token_to_req": np.arange(rows, dtype=np.int32),
        "block_table": np.tile(np.arange(num_blocks, dtype=np.int32), (rows, 1)),
        "slot_to_token": resident,
        "lru_slots": np.tile(np.arange(capacity, dtype=np.int32), (rows, 1)),
        "reset_mask": (req_ids != last_req_ids).astype(np.uint8),
    }


def make_echo_initial(case):
    """同一初始历史转换为 Echo 状态。SIMT 的槽从 0 开始，Echo 从 1 开始。

    Echo 的 token 0 / slot 0 是保留项，因此共同 query 不使用 token 0。
    lru_slots 最旧→最新；对应 pri 必须递增，不能使用 capacity-rank。
    """
    rows, topk, capacity, max_token = (case[k] for k in ("rows", "topk", "capacity", "max_token"))
    initial = {
        "htd": np.full((rows, max_token), I32_MAX, np.int32),
        "dth": np.full((rows, capacity + 1), I32_MAX, np.int32),
        "pri": np.full((rows, capacity + 1), -1, np.int32),
        "free_slots": np.zeros((rows, capacity + topk), np.int32),
        "avail": np.zeros(rows, np.int32),
        "fifo": np.full(rows, capacity + 1, np.int32),
    }
    initial["htd"][:, 0] = 0
    initial["pri"][:, 0] = I32_MAX
    for row in range(rows):
        resident = case["slot_to_token"][row]
        occupied = np.flatnonzero(resident >= 0)
        initial["dth"][row, occupied + 1] = resident[occupied]
        initial["htd"][row, resident[occupied]] = occupied + 1
        rank = np.empty(capacity, np.int32)
        rank[case["lru_slots"][row]] = np.arange(capacity, dtype=np.int32)
        initial["pri"][row, occupied + 1] = rank[occupied] + 1
        empty_slots = np.flatnonzero(resident < 0) + 1
        initial["free_slots"][row, :len(empty_slots)] = empty_slots[::-1]
        initial["avail"][row] = len(empty_slots)
    return initial


# =============================================================================
# 2. CPU 参考：仅覆盖上述共同性能输入，按语义推导，不复用 kernel 的 hash/radix
# =============================================================================
def reference_simt(case):
    """SIMT：按已有 LRU 顺序稳定分出 hit/evictable，再按 query 顺序分配 miss。

    此处输入全合法、不重复、所有行激活且没有投机失效，因此可以省掉通用
    golden 的非法输入/重复 token/MTP 分支；LRU 重建及输出顺序保持 main 语义。
    """
    rows, topk, capacity = (case[k] for k in ("rows", "topk", "capacity"))
    out = {key: case[key].copy() for key in SIMT_STATE}
    out.update({key: np.full((rows, topk), -1, np.int32)
                for key in ("current_slots", "miss_tokens", "miss_slots")})
    out["miss_count"] = np.zeros(rows, np.int32)
    for row in range(rows):
        if case["reset_mask"][row]:
            out["slot_to_token"][row].fill(-1)
            out["lru_slots"][row] = np.arange(capacity, dtype=np.int32)
            out["last_req_ids"][row] = case["req_ids"][row]

        resident, lru = out["slot_to_token"][row], out["lru_slots"][row].copy()
        queries = case["topk_indices"][row]
        query_set = set(queries.tolist())
        hit_slots = [int(slot) for slot in lru if int(resident[slot]) in query_set]
        evictable = [int(slot) for slot in lru if int(resident[slot]) not in query_set]
        token_to_slot = {int(resident[slot]): slot for slot in hit_slots}
        misses = []
        for position, token in enumerate(queries):
            if int(token) in token_to_slot:
                out["current_slots"][row, position] = token_to_slot[int(token)]
            else:
                misses.append(position)

        # topk <= capacity 且 query 不重复，所有 miss 均可分配。
        if len(misses) > len(evictable):
            raise AssertionError("共同输入不满足可分配条件")
        for index, position in enumerate(misses):
            slot, token = evictable[index], queries[position]
            resident[slot] = token
            out["current_slots"][row, position] = slot
            out["miss_tokens"][row, index] = token
            out["miss_slots"][row, index] = slot
        out["miss_count"][row] = len(misses)
        # main 的顺序：剩余 evictable → 新分配 miss → 原 hit。
        out["lru_slots"][row] = evictable[len(misses):] + evictable[:len(misses)] + hit_slots
    return out


def reference_echo(case, initial):
    """Echo：hit 提升时间戳，必要时淘汰最低优先级，最后从 free 栈尾分配。

    CPU 用排序表达淘汰语义；作者新版设备实现即使无需淘汰仍执行完整 radix 排序。
    CPU 参考可以省掉无效计算，因为参考只验证结果，不参与 NPU 计时。
    """
    rows, topk, capacity = (case[k] for k in ("rows", "topk", "capacity"))
    out = {key: initial[key].copy() for key in ECHO_STATE}
    out.update({key: np.zeros((rows, topk), np.uint8 if key == "miss_mask" else np.int32)
                for key in ECHO_OUTPUT})
    for row in range(rows):
        htd, dth, pri, free = (out[key][row] for key in ("htd", "dth", "pri", "free_slots"))
        avail, fifo = int(out["avail"][row]), int(out["fifo"][row])
        if case["reset_mask"][row]:
            htd.fill(I32_MAX)
            htd[0] = 0
            dth.fill(I32_MAX)
            pri.fill(-1)
            pri[0] = I32_MAX
            free.fill(0)
            free[:capacity] = np.arange(capacity, 0, -1, dtype=np.int32)
            avail, fifo = capacity, 1

        tokens = case["topk_indices"][row]
        before = htd[tokens].copy()
        misses = before == I32_MAX
        num_misses = int(misses.sum())
        pri[before[~misses]] = fifo
        fifo += 1

        free_size = max(num_misses - avail, 0)
        if free_size:
            candidates = [slot for slot in range(1, capacity + 1) if pri[slot] >= 0]
            released = sorted(candidates, key=lambda slot: (int(pri[slot]), slot))[:free_size]
            for slot in released:
                htd[dth[slot]] = I32_MAX
                dth[slot], pri[slot] = I32_MAX, -1
            free[avail:avail + free_size] = released
            avail += free_size

        # free 是栈；这里取出的顺序决定每个 miss 最终拿到哪个 1 基槽。
        allocated = free[avail - num_misses:avail].copy()
        avail -= num_misses
        pri[allocated] = fifo
        fifo += 1
        for token, slot in zip(tokens[misses], allocated):
            htd[token], dth[slot] = slot, token
        out["current_slots"][row] = htd[tokens]
        out["miss_host_pos"][row] = np.where(misses, tokens, 0)
        out["miss_alloc_flat"][row] = np.where(misses, row * capacity + htd[tokens], 0)
        out["miss_mask"][row] = misses
        out["avail"][row], out["fifo"][row] = avail, fifo
    return out


def check_common_semantics(case, simt, echo):
    """独立核对命中率、最终 query 映射和被淘汰的历史 token 集合。

    不要求两个算法输出相同槽号；每种算法的所有状态/输出另与自己的参考全等。
    """
    expected_misses = np.full(case["rows"], case["topk"] - case["hits"])
    np.testing.assert_array_equal(simt["miss_count"], expected_misses)
    np.testing.assert_array_equal(echo["miss_mask"].sum(axis=1), expected_misses)
    for row, queries in enumerate(case["topk_indices"]):
        np.testing.assert_array_equal(simt["slot_to_token"][row, simt["current_slots"][row]], queries)
        np.testing.assert_array_equal(echo["dth"][row, echo["current_slots"][row]], queries)
        np.testing.assert_array_equal(echo["htd"][row, queries], echo["current_slots"][row])
        historical = case["slot_to_token"][row]
        historical = historical[historical >= 0]
        queries_set = set(queries.tolist())
        # 直接从原始 LRU 推导淘汰集合，避免只用两份 golden 互相佐证。
        free_before = case["capacity"] if case["reset_mask"][row] else int((case["slot_to_token"][row] < 0).sum())
        count = max(int(expected_misses[row]) - free_before, 0)
        oldest = [int(case["slot_to_token"][row, slot]) for slot in case["lru_slots"][row]
                  if case["slot_to_token"][row, slot] >= 0 and int(case["slot_to_token"][row, slot]) not in queries_set]
        expected_evicted = np.sort(oldest[:count])
        for after in (simt["slot_to_token"][row], echo["dth"][row, 1:]):
            np.testing.assert_array_equal(np.setdiff1d(historical, after), expected_evicted)


# =============================================================================
# 3. 分配 NPU 张量：初始快照与工作状态分开，避免 copy_ 的源被 kernel 改写
# =============================================================================
def prepare_npu(torch, case, echo_initial, device, names=("simt", "echo")):
    def upload(array):
        return torch.from_numpy(np.ascontiguousarray(array)).to(device)

    inputs = {key: upload(case[key]) for key in (
        "req_ids", "topk_indices", "stable_prefix_lens", "visible_seq_lens",
        "token_to_req", "block_table", "reset_mask")}
    inputs["active_rows"] = torch.tensor([case["rows"]], dtype=torch.int32, device=device)
    # SIMT 各路共享只读初始快照；工作状态分别 clone，绝不共享被 kernel 写入的张量。
    simt_snapshot = {key: upload(case[key]) for key in SIMT_STATE}
    echo_snapshot = {key: upload(echo_initial[key]) for key in ECHO_STATE} if "echo" in names else None
    initial = {name: echo_snapshot if name == "echo" else simt_snapshot for name in names}
    states = {name: {key: value.clone() for key, value in snapshot.items()}
              for name, snapshot in initial.items()}
    rows, topk, capacity = (case[k] for k in ("rows", "topk", "capacity"))
    for name in names:
        if name == "echo":
            for key in ECHO_OUTPUT:
                dtype = torch.uint8 if key == "miss_mask" else torch.int32
                states[name][key] = torch.empty((rows, topk), dtype=dtype, device=device)
        else:
            for key in SIMT_OUTPUT:
                shape = (rows,) if key == "miss_count" else (rows, topk)
                states[name][key] = torch.empty(shape, dtype=torch.int32, device=device)
            layout = plan_layout(topk, capacity, rows, name)
            if layout["workspace_elements"] > I32_MAX:
                raise ValueError(f"{name} workspace 超过 int32 索引范围")
            states[name]["layout"] = layout
            states[name]["workspace"] = torch.empty(layout["workspace_elements"], dtype=torch.int32, device=device)

    if "echo" not in names:
        torch.npu.synchronize()
        return inputs, initial, states

    # Echo 一核处理连续几行；尾核必须非空，不能简单使用 min(rows, cores)。
    cores = torch.ops.sparse_kv_plan_op.echo_lru_core_count()
    budget = min(rows, cores)
    rows_per_core = (rows + budget - 1) // budget
    blocks = (rows + rows_per_core - 1) // rows_per_core
    npad = 1 << capacity.bit_length()  # next_pow2(capacity+1)，4096→8192。
    base = rows * (capacity + 1 + 4 + 3 * topk)
    # GM 中仍保留 main 的 ping-pong 占位段，虽然排序实际使用静态 UB。
    elements = ((base + 1) & ~1) + blocks * (4 * npad + 64 * 256 + 64 * 256 + 256 + 8192)
    if elements > I32_MAX:
        raise ValueError("Echo workspace 超过 int32 索引范围")
    states["echo"]["workspace"] = torch.empty(elements, dtype=torch.int32, device=device)
    states["echo"]["blocks"] = blocks
    torch.npu.synchronize()
    return inputs, initial, states


def restore_state(torch, name, initial, state):
    """每次调用前还原同一份持久状态；workspace 是临时区，沿用 main 的复用方式。"""
    for key, source in initial[name].items():
        state[key].copy_(source)
    torch.npu.synchronize()  # copy 完成后才能开始计时。


def invoke(torch, name, case, inputs, state):
    """计时区间内只有这个调用；参数顺序直接对应 main 的 TORCH_LIBRARY schema。"""
    if name != "echo":
        op_name = VARIANTS[name]["op_name"] if name in VARIANTS else "sparse_kv_plan"
        getattr(torch.ops.sparse_kv_plan_op, op_name)(
            inputs["req_ids"], inputs["topk_indices"], inputs["stable_prefix_lens"],
            inputs["visible_seq_lens"], inputs["token_to_req"], inputs["block_table"],
            inputs["active_rows"], state["last_req_ids"], state["slot_to_token"],
            state["lru_slots"], state["current_slots"], state["miss_count"],
            state["miss_tokens"], state["miss_slots"], state["workspace"],
            case["topk"], case["capacity"], case["max_token"], BLOCK_SIZE, case["host_num_blocks"])
    else:
        torch.ops.sparse_kv_plan_op.echo_lru(
            inputs["topk_indices"], inputs["stable_prefix_lens"], inputs["reset_mask"],
            state["htd"], state["dth"], state["pri"], state["free_slots"],
            state["avail"], state["fifo"], state["current_slots"],
            state["miss_host_pos"], state["miss_alloc_flat"], state["miss_mask"],
            state["workspace"], state["blocks"], False)  # 所有 stable_prefix 都是 max_token。


def check_npu_result(name, state, expected, phase):
    """逐字段全等检查，读取发生在同步之后、计时区间之外。"""
    actual = {key: state[key].cpu().numpy() for key in expected[name]}
    for key, reference in expected[name].items():
        np.testing.assert_array_equal(actual[key], reference, err_msg=f"{phase}/{name}/{key}")
    return actual


# =============================================================================
# 4. 测一次调用：恢复 → start Event → invoke → end Event → 等待 → 返回微秒
# =============================================================================
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


# =============================================================================
# 5. 主流程：读下面这段即可看到所有测试步骤，没有隐含的 runner/callback 嵌套
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--capacity", type=int, default=4096)
    parser.add_argument("--max-token", type=int, default=262144)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=int, default=1, help="逻辑 NPU 卡号")
    parser.add_argument("--scenario", choices=("all", *SCENARIOS), default="all")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--audit-repeat", type=int, default=3, help="计时前重复正确性审计次数")
    parser.add_argument("--verify-each-sample", action="store_true", help="短跑审计：在每次预热/测量后全量校验")
    parser.add_argument("--cpu-check", action="store_true", help="只检查共同输入和 CPU 参考，不加载共享库、不计时")
    parser.add_argument("--implementations", nargs="+", choices=IMPLEMENTATIONS, default=list(IMPLEMENTATIONS),
                        help="参与对比的实现；默认原 SIMT、Echo 和全部 11 个候选")
    parser.add_argument("--reference", choices=IMPLEMENTATIONS,
                        help="统计加速比的基准；默认优先 simt，其次 00_baseline，否则第一路")
    parser.add_argument("--library", type=Path,
                        default=Path(__file__).resolve().parent / "build/libsimt_vs_echo_ops.so",
                        help="本目录构建的共享库；默认 build/libsimt_vs_echo_ops.so")
    parser.add_argument("--json", type=Path, default=Path("results/simt_vs_echo.json"))
    args = parser.parse_args()
    names = tuple(args.implementations)
    if len(set(names)) != len(names):
        parser.error("implementations 不得重复")
    reference = args.reference or ("simt" if "simt" in names else "00_baseline" if "00_baseline" in names else names[0])
    if reference not in names:
        parser.error("reference 必须在 implementations 中")
    if not (args.rows > 0 and 0 < args.topk <= args.capacity and args.max_token > args.topk):
        parser.error("需要 rows>0、0<topk<=capacity、max_token>topk")
    if "echo" in names and args.capacity > 4096:
        parser.error("Echo capacity 上限为 4096；更大尺寸请在 implementations 中仅选择 SIMT 实现")
    # 两个 kernel 内有 int32 索引，形状乘积必须可表示。
    if max(args.rows * args.max_token, args.rows * (args.capacity + args.topk), args.max_token) > I32_MAX:
        parser.error("形状超过本基准支持的 int32 索引范围")
    if args.warmup < 0 or args.repeat <= 0 or args.audit_repeat <= 0:
        parser.error("warmup>=0、repeat>0、audit-repeat>0")
    scenarios = list(SCENARIOS) if args.scenario == "all" else [args.scenario]

    # CPU 检查不需要 torch；共享库只在正式 NPU 路径加载。
    if not args.cpu_check:
        try:
            import torch
            import torch_npu
        except ImportError as exc:
            parser.error(f"需要匹配的 torch/torch_npu 环境：{exc}")
        if not torch.npu.is_available():
            parser.error("没有可用 NPU；可用 --cpu-check 检查参考结果")
        torch.npu.set_device(args.device)
        device = torch.device(f"npu:{args.device}")
        library_path = args.library.resolve()
        if not library_path.is_file():
            parser.error(f"共享库不存在：{library_path}；请先在本目录运行 bash build.sh")
        try:
            # dlopen 本目录的库：两个 C++ plugin 的静态初始化器注册 torch.ops。
            # 不导入原仓库的 sparse_kv_plan_op 包，也不会回退到已安装的旧扩展。
            torch.ops.load_library(str(library_path))
        except (OSError, RuntimeError) as exc:
            parser.error(f"无法加载 {library_path}：{exc}")
        required_ops = [VARIANTS[n]["op_name"] if n in VARIANTS else
                        "echo_lru" if n == "echo" else "sparse_kv_plan" for n in names]
        if "echo" in names:
            required_ops.append("echo_lru_core_count")
        for op_name in required_ops:
            if not hasattr(torch.ops.sparse_kv_plan_op, op_name):
                parser.error(f"共享库缺少 {op_name}，请重新构建本目录")
        manifest_path = Path(__file__).resolve().parent / "source_manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        metadata = {
            "device_index": args.device, "device": torch.npu.get_device_name(args.device),
            "torch": torch.__version__, "torch_npu": torch_npu.__version__,
            "library_path": str(library_path),
            "library_sha256": hashlib.sha256(library_path.read_bytes()).hexdigest(),
            "source_snapshot_commit": manifest.get("source_commit", "unknown"),
            # 基线提交不能独自区分后续 Echo 补丁，因此同时记录修订名和当前源码哈希。
            "echo_revision": manifest.get("echo_revision", "main-original"),
            "echo_kernel_sha256": hashlib.sha256(
                (Path(__file__).resolve().parent / "csrc/echo/echo_lru_kernel.h").read_bytes()).hexdigest(),
            "timing": "Event: public torch.ops stream interval including tiling H2D and host submission gaps; restore/validation excluded. Host: record+invoke+record+synchronize.",
            "shape": {k: getattr(args, k) for k in ("rows", "topk", "capacity", "max_token")},
            "seed": args.seed, "warmup": args.warmup, "repeat": args.repeat,
            "audit_repeat": args.audit_repeat, "verify_each_sample": args.verify_each_sample,
            "implementations": list(names), "reference": reference,
            "variants": {n: VARIANTS[n] for n in names if n in VARIANTS},
        }
        print("实际加载共享库:", library_path, flush=True)
        results = {"metadata": metadata, "scenarios": {}}

    for scenario in scenarios:
        case = make_case(args.rows, args.topk, args.capacity, args.max_token, scenario, args.seed)
        simt_expected = reference_simt(case)
        echo_initial = make_echo_initial(case) if "echo" in names else None
        echo_expected = reference_echo(case, echo_initial) if "echo" in names else None
        expected = {name: echo_expected if name == "echo" else simt_expected for name in names}
        if "echo" in names:
            check_common_semantics(case, simt_expected, echo_expected)
        miss_count = args.topk - case["hits"]
        available = args.capacity - case["hits"] if scenario != "full_cache_75pct" else 0
        free_size = max(miss_count - available, 0)
        print(f"{scenario}: 每行 miss={miss_count}, 初始空闲={available}, 需淘汰={free_size}", flush=True)
        if args.cpu_check:
            for name in names:
                if name != "echo":
                    print(f"  {name}: {plan_layout(args.topk, args.capacity, args.rows, name)}")
            print("  CPU 参考与布局检查通过（未执行候选 kernel；未运行 NPU）", flush=True)
            continue

        inputs, initial, states = prepare_npu(torch, case, echo_initial, device, names)

        # A. 重复审计：持久状态恢复；输出及 scratch 污染，排除“读到旧答案”的假通过。
        for iteration in range(args.audit_repeat):
            for name in names:
                state = states[name]
                restore_state(torch, name, initial, state)
                output_keys = ECHO_OUTPUT if name == "echo" else SIMT_OUTPUT
                for key in output_keys:
                    state[key].fill_(255 if key == "miss_mask" else -777)
                state["workspace"].fill_(I32_MAX)
                invoke(torch, name, case, inputs, state)
                torch.npu.synchronize()
                check_npu_result(name, state, expected, f"audit-{iteration}")

        # B. 每路固定复用两个 Event；上一轮 end 已完成，避免反复初始化 Event 的干扰。
        events = {name: (torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True))
                  for name in names}
        for iteration in range(args.warmup):
            order = execution_order(names, iteration)
            for name in order:
                measure_one(torch, name, case, inputs, initial, states[name], events[name])
                if args.verify_each_sample:
                    check_npu_result(name, states[name], expected, f"warmup-{iteration}")
        # 最后一次预热的工作状态仍是“从初始状态调用一次”的结果。
        if args.warmup:
            for name in names:
                check_npu_result(name, states[name], expected, "after-warmup")

        # C. 正式测量：全部实现轮转执行位置；每路每轮仍从相同初始状态调用一次。
        samples = {name: {"event_us": [], "host_us": []} for name in names}
        orders = []
        for iteration in range(args.repeat):
            order = execution_order(names, iteration)
            orders.append(list(order))
            for name in order:
                event_us, host_us = measure_one(torch, name, case, inputs, initial, states[name], events[name])
                samples[name]["event_us"].append(event_us)
                samples[name]["host_us"].append(host_us)
                if args.verify_each_sample:
                    check_npu_result(name, states[name], expected, f"sample-{iteration}")

        # D. 测量后校验所有实现最后的状态及输出，再生成统计。失败就不输出性能结论。
        actual = {name: check_npu_result(name, states[name], expected, "after-measurement")
                  for name in names}
        if "echo" in names:
            for name in names:
                if name != "echo":
                    check_common_semantics(case, actual[name], actual["echo"])
        stats = {name: {"event": summary(samples[name]["event_us"]),
                        "host": summary(samples[name]["host_us"]), "raw_samples": samples[name]}
                 for name in names}
        reference_us = stats[reference]["event"]["p50_us"]
        for name in names:
            stats[name]["speedup_vs_reference_event_p50"] = reference_us / stats[name]["event"]["p50_us"]
            if name != "echo":
                stats[name]["layout"] = states[name]["layout"]
        if "simt" in names and "echo" in names:
            stats["echo_over_simt_event_p50"] = stats["echo"]["event"]["p50_us"] / stats["simt"]["event"]["p50_us"]
        stats.update({"execution_orders": orders,
                      "correctness": "passed", "expected_misses_per_row": miss_count,
                      "echo_blocks": states["echo"]["blocks"] if "echo" in names else None,
                      "case_sha256": array_hash({k: v for k, v in case.items() if isinstance(v, np.ndarray)})})
        results["scenarios"][scenario] = stats
        for name in names:
            print(f"  {name}: Event P50={stats[name]['event']['p50_us']:.2f} μs, "
                  f"相对 {reference} 加速比={stats[name]['speedup_vs_reference_event_p50']:.2f}×", flush=True)

    if not args.cpu_check:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("结果与原始样本:", args.json.resolve())
        if args.verify_each_sample:
            print("本次为逐样本正确性审计；频繁读回改变缓存条件，正式性能对比请关闭该选项。")


if __name__ == "__main__":
    main()
