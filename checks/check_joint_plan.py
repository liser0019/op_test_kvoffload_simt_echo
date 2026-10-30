#!/usr/bin/env python3
"""仅服务器NPU执行：联合/逐行语义、溢出回退、连续状态、UB/GM和上下文门禁。"""
from pathlib import Path
import argparse
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joint_plan_common as common
from joint_plan_cases import make_case
from harness_io import preflight_args, write_json, record_failure


def run_one(torch, case, label):
    inputs = common.upload_inputs(torch, case)
    records = []
    for name in common.IMPLEMENTATIONS:
        bundle = common.prepare(torch, case, name, inputs)
        result = common.audit(torch, case, bundle, repeats=2)
        records.append({"implementation": name, "layout": bundle["layout"], "audit": result})
        bundle["context"].close()
    print("联合Plan门禁通过:", label, flush=True)
    return {"label": label, "implementations": records}


def continuous_gate(torch):
    """下一步直接续用上一轮NPU状态；不从golden/初始化快照写回。"""
    case = make_case(2, 7, 16, 512, 3, block_size=8)
    records = []
    for name in common.IMPLEMENTATIONS:
        bundle = common.prepare(torch, case, name)
        for step, (pattern, active) in enumerate((("shared", 8), ("disjoint", 8), ("shared", 8),
                                                ("partial", 5), ("shared", 0), ("shared", 8))):
            current = make_case(2, 7, 16, 512, 3, overlap=pattern, block_size=8)
            current["active_rows"] = active
            if step >= 2:
                current["req_ids"] += 99
            if step == 5:
                # 同一请求的行不连续：按连续片段分别规划，不越片段共享。
                current["token_to_req"][:] = [0, 1, 0, 1, 0, 1, 0, 1]
            common.update_inputs(torch, bundle, current)
            common.poison(bundle)
            torch.npu.synchronize()
            before = common.snapshot(bundle)
            common.invoke(bundle)
            torch.npu.synchronize()
            result = common.check(current, bundle, before)
            records.append({"name": name, "step": step, "pattern": pattern, "active": active, "audit": result})
        bundle["context"].close()
    return records


def context_gate(torch):
    case = make_case(1, 7, 16, 256, 3, block_size=8)
    bundle = common.prepare(torch, case, "joint")
    ctx, p, attrs = bundle["context"], bundle["ordered"], bundle["attrs"]
    def reject(operation):
        try:
            operation()
        except (RuntimeError, ValueError):
            return
        raise AssertionError("非法prepared上下文未在下发前拒绝")
    reject(lambda: ctx.run(p[:-1], attrs))
    altered = list(attrs)
    altered[-1] = 0
    reject(lambda: ctx.run(p, altered))
    rebound = list(p)
    rebound[1] = p[1].clone()
    reject(lambda: ctx.run(rebound, attrs))
    rebound[1] = p[1].view(-1)
    reject(lambda: ctx.run(rebound, attrs))
    ctx.close()
    reject(lambda: ctx.run(p, attrs))
    return ["list", "attributes", "rebound", "shape", "closed"]


def run_gate(torch):
    records = []
    for mtp in (0, 1, 3):
        for pattern in ("shared", "partial", "disjoint"):
            for scenario in ("cold_reset", "full_cache_75pct"):
                case = make_case(2, 7, 16, 512, mtp, scenario, pattern, block_size=8)
                records.append(run_one(torch, case, f"{mtp}/{pattern}/{scenario}"))
    for count in (16, 17):
        case = make_case(1, 7, 16, 512, 3, block_size=8)
        for row in range(4):
            case["topk_indices"][row] = [(row * 5 + p) % count + 1 for p in range(7)]
        records.append(run_one(torch, case, f"union_size_{count}"))
    invalid = make_case(2, 7, 21, 512, 3, block_size=7)
    invalid["topk_indices"][0] = [-1, invalid["max_token"], invalid["context"] + 2, 5, 5, 7, 8]
    invalid["token_to_req"][4:] = -1
    invalid["block_table"][0, 0] = -1
    invalid["block_table"][0, 1] = invalid["host_num_blocks"]
    invalid["stable_prefix_lens"][:2] = [-1, 512]
    records.append(run_one(torch, invalid, "invalid_sources_visibility_duplicates_block7"))
    duplicate = make_case(1, 7, 16, 256, 3, block_size=8)
    duplicate["topk_indices"][:] = [5, 5, 6, 6, 7, 7, 7]
    records.append(run_one(torch, duplicate, "duplicate_query_without_invalid_pages"))
    for active in (-1, 0, 3, 5, 99):
        case = make_case(2, 7, 16, 512, 3, block_size=8)
        case["active_rows"] = active
        records.append(run_one(torch, case, f"active_rows_{active}"))
    # More than64 groups forces one AIV to reuse its scratch for multiple requests.
    records.append(run_one(torch, make_case(65, 7, 16, 256, 3, block_size=8), "multi_group_per_aiv"))
    # Joint 160KiB scratch takes GM; private still takes UB for this shape.
    records.append(run_one(torch, make_case(1, 2048, 8192, 65536, 3), "joint_gm_private_ub"))
    return {"status": "passed", "cases": records, "continuous": continuous_gate(torch),
            "context": context_gate(torch)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_runtime_arguments(parser)
    parser.add_argument("--report", type=Path, default=Path("/dev/shm/joint_plan_gate/report.json"))
    args = parser.parse_args()
    storage = preflight_args(args, args.report.parent)
    report = {"status": "running", "storage": storage}
    try:
        torch, _, environment = common.load_runtime(args)
        report["environment"] = environment
        report["gate"] = run_gate(torch)
        report["status"] = "passed"
        write_json(args.report, report)
    except BaseException as exc:
        record_failure(args.report, report, exc)
        raise
    print("联合Plan全部门禁通过:", args.report, flush=True)


if __name__ == "__main__":
    main()
