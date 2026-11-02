#!/usr/bin/env python3
"""仅服务器NPU执行：联合/逐行语义、溢出回退、连续状态、UB/GM和上下文门禁。"""
from pathlib import Path
import argparse
import json
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joint_plan_common as common
from joint_plan_cases import make_case, describe
from harness_io import preflight_args, write_json, record_failure


def run_one(torch, case, label, failure_dir=None, implementations=common.DEFAULT_IMPLEMENTATIONS):
    inputs = common.upload_inputs(torch, case)
    records = []
    for name in implementations:
        bundle = common.prepare(torch, case, name, inputs)
        result = common.audit(torch, case, bundle, repeats=5, failure_dir=failure_dir)
        records.append({"implementation": name, "layout": bundle["layout"], "audit": result})
        common.close(bundle)
    print("联合Plan门禁通过:", label, flush=True)
    return {"label": label, "implementations": records}


def continuous_gate(torch, failure_dir=None, implementations=common.DEFAULT_IMPLEMENTATIONS):
    """下一步直接续用上一轮NPU状态；不从golden/初始化快照写回。"""
    case = make_case(2, 7, 16, 512, 3, block_size=8)
    records = []
    for name in implementations:
        bundle = common.prepare(torch, case, name)
        for step, (pattern, active) in enumerate((("independent", 8), ("disjoint", 8), ("shared", 8),
                                                ("partial", 5), ("shared", 0), ("shared", 8))):
            current = make_case(2, 7, 16, 512, 3, overlap=pattern, block_size=8)
            current["active_rows"] = active
            if step >= 2:
                current["req_ids"] += 99
            if step == 5:
                # 同一请求的行不连续：按连续片段分别规划，不越片段共享。
                current["token_to_req"][:] = [0, 1, 0, 1, 0, 1, 0, 1]
            bundle["phase"] = f"continuous-step-{step + 1}/{pattern}"
            bundle["workload"] = {key: current.get(key) for key in bundle["workload"]}
            common.update_inputs(torch, bundle, current)
            common.poison(bundle)
            torch.npu.synchronize()
            before = common.snapshot(bundle)
            try:
                common.invoke(bundle)
                torch.npu.synchronize()
                result = common.check(current, bundle, before)
            except BaseException:
                if failure_dir is not None:
                    common.dump_failure(current, bundle, failure_dir)
                raise
            records.append({"name": name, "step": step, "pattern": pattern, "active": active, "audit": result})
        common.close(bundle)
    return records


def context_gate(torch, implementations=common.DEFAULT_IMPLEMENTATIONS):
    case = make_case(1, 7, 16, 256, 3, block_size=8)
    records = []
    def reject(operation):
        try:
            operation()
        except (RuntimeError, ValueError):
            return
        raise AssertionError("非法prepared上下文未在下发前拒绝")
    for name in implementations:
        if name in ("abcdf", "abcdf_pages"):
            continue
        bundle = common.prepare(torch, case, name)
        ctx, p, attrs = bundle["context"], bundle["ordered"], bundle["attrs"]
        reject(lambda: ctx.run(p[:-1], attrs))
        altered = list(attrs)
        altered[0] += 1
        reject(lambda: ctx.run(p, altered))
        rebound = list(p)
        rebound[1] = p[1].clone()
        reject(lambda: ctx.run(rebound, attrs))
        rebound[1] = p[1].view(-1)
        reject(lambda: ctx.run(rebound, attrs))
        if name in common.VARIANTS:
            for mask in (0, max(common.VARIANTS.values()) + 1, -1):
                reject(lambda: torch.classes.mtp_joint_plan.PreparedVariant(p, attrs, common.is_joint(name), mask))
        ctx.close()
        reject(lambda: ctx.run(p, attrs))
        records.append({"implementation": name, "checks": ["list", "attributes", "rebound", "shape", "closed"],
                        "mask_range": name in common.VARIANTS})
    return records


def input_write_regression(torch, repeats=100, failure_dir=None, implementations=common.DEFAULT_IMPLEMENTATIONS):
    """每个待测协议变体各做一轮100次已知故障尺寸回归，检查输入原值，不恢复输入。"""
    if repeats <= 0:
        raise ValueError("regression-repeat必须>0")
    case = make_case(32, 2048, 4096, 262144, 1, "full_cache_75pct", "shared", 2026, 64)
    records = {}
    for name in implementations:
        if name in ("abcdf", "abcdf_pages"):
            continue
        print(f"输入错写回归 implementation={name}, repeat={repeats}", flush=True)
        bundle = common.prepare(torch, case, name)
        try:
            records[name] = common.audit(torch, case, bundle, repeats, failure_dir, progress_every=10)
        finally:
            common.close(bundle)
    return {"status": "passed", "checked_iterations": repeats, "workload": describe(case),
            "per_implementation": records, "input_restore": False}


def validate_gate_report(path, environment, minimum_repeats=100, implementations=common.DEFAULT_IMPLEMENTATIONS):
    report = json.loads(Path(path).read_text())
    gate, original = report.get("gate", {}), report.get("environment", {})
    if (report.get("status") != "passed" or gate.get("status") != "passed"
            or not set(implementations).issubset(gate.get("implementations", []))
            or not (set(implementations) - {"abcdf", "abcdf_pages"}).issubset(
                gate.get("input_write_regression", {}).get("per_implementation", {}))
            or any(gate.get("input_write_regression", {}).get("per_implementation", {}).get(name, {}).get("status") != "passed"
                   for name in implementations if name not in ("abcdf", "abcdf_pages"))
            or gate.get("input_write_regression", {}).get("status") != "passed"
            or gate.get("input_write_regression", {}).get("input_restore") is not False
            or gate.get("input_write_regression", {}).get("checked_iterations", 0) < minimum_repeats
            or any(original.get(key) != environment.get(key)
                   for key in ("device", "device_name", "hostname", "library_sha256", "identity"))):
        raise RuntimeError("门禁报告不匹配本设备/库/源码，或缺少三方完整门禁及足量输入错写回归")
    return {"status": "passed", "reused_report": str(Path(path).resolve()),
            "input_write_regression": gate["input_write_regression"], "implementations": gate["implementations"]}


def run_gate(torch, regression_repeat=100, failure_dir=None, implementations=common.DEFAULT_IMPLEMENTATIONS):
    regression = input_write_regression(torch, regression_repeat, failure_dir, implementations)
    records = []
    for mtp in (0, 1, 3):
        for pattern in ("shared", "partial", "independent", "disjoint"):
            for scenario in ("cold_reset", "full_cache_75pct"):
                case = make_case(2, 7, 16, 512, mtp, scenario, pattern, block_size=8)
                records.append(run_one(torch, case, f"{mtp}/{pattern}/{scenario}", failure_dir, implementations))
    for count in (16, 17):
        case = make_case(1, 7, 16, 512, 3, block_size=8)
        for row in range(4):
            case["topk_indices"][row] = [(row * 5 + p) % count + 1 for p in range(7)]
        records.append(run_one(torch, case, f"union_size_{count}", failure_dir, implementations))
    invalid = make_case(2, 7, 21, 512, 3, block_size=7)
    invalid["topk_indices"][0] = [-1, invalid["max_token"], invalid["context"] + 2, 5, 5, 7, 8]
    invalid["token_to_req"][4:] = -1
    invalid["stable_prefix_lens"][:2] = [-1, 512]
    records.append(run_one(torch, invalid, "invalid_sources_visibility_duplicates_block7", failure_dir, implementations))
    duplicate = make_case(1, 7, 16, 256, 3, block_size=8)
    duplicate["topk_indices"][:] = [5, 5, 6, 6, 7, 7, 7]
    records.append(run_one(torch, duplicate, "duplicate_query_without_invalid_pages", failure_dir, implementations))
    for active in (-1, 0, 3, 5, 99):
        case = make_case(2, 7, 16, 512, 3, block_size=8)
        case["active_rows"] = active
        records.append(run_one(torch, case, f"active_rows_{active}", failure_dir, implementations))
    for pattern in ("independent", "disjoint"):
        case = make_case(1, 2048, 4096, 65536, 3, overlap=pattern)
        records.append(run_one(torch, case, "multi_tile_" + pattern, failure_dir, implementations))
    # 真实1024线程配置的双tile重复、未命中、未来/非法token；检查新标记解包与因果性。
    case = make_case(1, 2048, 4096, 262144, 3, overlap="shared", scenario="cold_reset")
    for row in range(4):
        case["topk_indices"][row, 1024:] = case["topk_indices"][row, :1024]
        case["topk_indices"][row, 0] = case["context"] + 1
        case["topk_indices"][row, 1024] = -1
    records.append(run_one(torch, case, "cross_tile_duplicates_and_future_token", failure_dir, implementations))
    # More than64 groups forces one AIV to reuse its scratch for multiple requests.
    records.append(run_one(torch, make_case(65, 7, 16, 256, 3, block_size=8), "multi_group_per_aiv", failure_dir, implementations))
    # Joint 160KiB scratch takes GM; private still takes UB for this shape.
    records.append(run_one(torch, make_case(1, 2048, 8192, 65536, 3), "joint_gm_private_ub", failure_dir, implementations))
    return {"status": "passed", "implementations": list(implementations),
            "input_write_regression": regression, "cases": records,
            "continuous": continuous_gate(torch, failure_dir, implementations),
            "context": context_gate(torch, implementations)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_runtime_arguments(parser)
    parser.add_argument("--implementations", nargs="+", choices=common.IMPLEMENTATIONS,
                        default=list(common.DEFAULT_IMPLEMENTATIONS))
    parser.add_argument("--regression-repeat", type=int, default=100)
    parser.add_argument("--regression-only", action="store_true")
    parser.add_argument("--report", type=Path, default=Path("/workspace/joint_plan_results/correctness.json"))
    args = parser.parse_args()
    if len(set(args.implementations)) != len(args.implementations):
        parser.error("implementations不得重复")
    if args.regression_repeat <= 0:
        parser.error("regression-repeat必须>0")
    storage = preflight_args(args, args.report.parent)
    report = {"status": "running", "storage": storage}
    try:
        torch, _, environment = common.load_runtime(args)
        report["environment"] = environment
        if args.regression_only:
            report["input_write_regression"] = input_write_regression(torch, args.regression_repeat, args.report.parent / "failure_state", args.implementations)
        else:
            report["gate"] = run_gate(torch, args.regression_repeat, args.report.parent / "failure_state", args.implementations)
        report["status"] = "passed"
        write_json(args.report, report)
    except BaseException as exc:
        record_failure(args.report, report, exc)
        raise
    print("联合Plan全部门禁通过:", args.report, flush=True)


if __name__ == "__main__":
    main()
