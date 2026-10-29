#!/usr/bin/env python3
"""同一Host池/输入下，对照原Transfer与A/B/C/D全部组合；以kernel CSV为准。"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
sys.dont_write_bytecode = True

import plan_transfer_common as common
import profile_plan_transfer as profile
import transfer_variants as catalog
from checks.check_transfer_variants import run_gate
from full_cache_cases import describe, assert_case
from harness_io import preflight_args, write_json, record_failure, diagnostic_log, capture_error


def order_configs(configs, scenario_index, reverse=False):
    shift = scenario_index % len(configs)
    result = configs[shift:] + configs[:shift]
    return list(reversed(result)) if reverse else result


def compare_runs(runs):
    """只在相同工况、启动方式、实际核数、输入指纹和工作量之间做比值。"""
    comparisons = []
    for run in runs:
        v = run["transfer_variant"]
        baseline = next((candidate for candidate in runs
            if candidate["transfer_variant"]["name"] == "baseline"
            and candidate["transfer_variant"]["actual_blocks"] == v["actual_blocks"]
            and candidate["mode"] == run["mode"] and candidate["launch_mode"] == run["launch_mode"]
            and candidate["case_sha256"] == run["case_sha256"]
            and candidate["buffer_addresses_sha256"] == run["buffer_addresses_sha256"]
            and candidate["workload"] == run["workload"]), None)
        if baseline:
            comparisons.append({"variant": v["name"], "actual_blocks": v["actual_blocks"],
                "scenario": run["workload"]["scenario"], "mode": run["mode"],
                "fallback": v["fallback"],
                "baseline_over_variant_transfer_p50": baseline["stats"]["transfer"]["p50_us"] /
                                                     run["stats"]["transfer"]["p50_us"]})
    return comparisons


def write_comparison(path, report):
    lines = ["# Transfer优化组合对照", "",
        "成功条件：退出码0、complete元数据、COMPLETE.json与report.md同时存在。", "",
        "| 工况 | 版本 | 实际核数 | 路径 | 样本数 | Transfer P50 μs | P95 μs | payload GB/s | baseline/本版 P50 |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|"]
    for run in report["runs"]:
        v, metric = run["transfer_variant"], run["stats"]["transfer"]
        ratio = next((x["baseline_over_variant_transfer_p50"] for x in report["comparisons"]
                      if x["variant"] == v["name"] and x["actual_blocks"] == v["actual_blocks"]
                      and x["scenario"] == run["workload"]["scenario"] and x["mode"] == run["mode"]), None)
        bw = run["stats"]["effective_payload_GBps"]
        bandwidth = f"{bw:.3f}" if bw is not None else "—"
        lines.append(f"| {run['workload']['scenario']}/{run['mode']} | {v['name']} | {v['actual_blocks']} | "
                     f"{v['effective_path']} | {metric['count']} | {metric['p50_us']:.3f} | "
                     f"{metric['p95_us']:.3f} | {bandwidth}")
        lines[-1] += f" | {ratio:.3f} |" if ratio is not None else " | — |"
    lines += ["", "吞吐为payload字节/kernel时间；固定地址且预热，不等于冷DDR或链路峰值。",
              "原版baseline与control都保留：control用于分离模板重构开销。",
              "每个配置按块采集；场景间轮换先后，可用--reverse-order整轮反序复测。",
              "单kernel统计及逐对busy/span/gap的完整P50/P95/count见report.md和每例stage_summary.csv。"]
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_arguments(parser, transfer_selectors=False)
    parser.set_defaults(launch_mode="prepared", transfer_variant="baseline", transfer_core_limit=64)
    parser.add_argument("--variants", nargs="+", choices=tuple(catalog.VARIANTS),
                        default=list(catalog.VARIANTS))
    parser.add_argument("--core-limits", nargs="+", type=int, default=[64])
    parser.add_argument("--modes", nargs="+", choices=("transfer", "chain"), default=["transfer"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--verify-each-sample", action="store_true")
    parser.add_argument("--reverse-order", action="store_true")
    parser.add_argument("--describe-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("/dev/shm/transfer_factorial"))
    args = parser.parse_args()
    preset = common.resolve_payload(args, parser)
    if (args.warmup < 0 or args.repeat <= 0 or not all(1 <= n <= 64 for n in args.core_limits)
        or len(set(args.core_limits)) != len(args.core_limits)
        or len(set(args.variants)) != len(args.variants) or len(set(args.modes)) != len(args.modes)):
        parser.error("次数/核数无效或配置重复")
    cases = [common.case_from_args(args, p) for p in args.hit_percentages]
    for case in cases:
        assert_case(case, common.plan.reference_simt(case))
    requested = [(name, cores) for name in args.variants for cores in args.core_limits]
    print(json.dumps({"cases": [describe(c, args.token_bytes_k, args.token_bytes_v) for c in cases],
        "configs": requested, "launch_mode": args.launch_mode, "modes": args.modes,
        "note": "Host内存需求与profiling磁盘余量分别计算；无实测前不预设胜者/90GBps阈值"},
        ensure_ascii=False), flush=True)
    if args.describe_only:
        return
    storage = preflight_args(args, args.output)
    session = Path(tempfile.mkdtemp(prefix="matrix_", dir=storage["output"]["path"]))
    args.detailed_log, args.current_capture = session / "profile.log", session
    metadata = session / "capture_metadata.json"
    report = {"status": "running", "payload": preset, "storage": storage, "runs": [],
              "catalog": catalog.CATALOG, "launch_mode": args.launch_mode, "execution_order": [],
              "warmup": args.warmup, "repeat": args.repeat,
              "cache_condition": "fixed mapped Host addresses, warmed; restore payload outside targets",
              "sampling_order": "blocked config captures; rotate by scenario, optional reverse"}
    print("矩阵结果目录:", session, flush=True)
    try:
        write_json(metadata, report)
        with diagnostic_log(args.detailed_log):
            torch, torch_npu, environment = common.load_runtime(args)
            report["environment"] = environment
            report["gate"] = run_gate(torch, args.variants, (args.launch_mode,))
            # 仅申请一次实际Host池：所有版本复用同一源地址、页表与数据。
            sources = common.prepare_sources(torch, cases[0], args.token_bytes_k, args.token_bytes_v)
            configs, seen = [], set()
            for name, cores in requested:
                actual = int(torch.ops.sparse_kv_plan_op.transfer_block_dim_info(cores))
                if (name, actual) in seen:
                    continue  # 平台裁剪后的重复核数不重复采集，保留requested元数据。
                seen.add((name, actual))
                configs.append((name, cores))
            report["requested_configs"] = requested
            report["effective_configs"] = [
                {"variant": name, "requested_core_limit": cores,
                 "actual_blocks": int(torch.ops.sparse_kv_plan_op.transfer_block_dim_info(cores))}
                for name, cores in configs]
            for index, case in enumerate(cases):
                bundle = common.prepare_case(torch, case, sources, args.token_bytes_k, args.token_bytes_v)
                order = order_configs(configs, index, args.reverse_order)
                report["execution_order"].append({"scenario": case["scenario"], "configs": order})
                for name, cores in order:
                    selected = copy.copy(args)
                    selected.case_bundle = bundle
                    selected.transfer_variant, selected.transfer_core_limit = name, cores
                    for mode in args.modes:
                        destination = session / case["scenario"] / f"{name}_cores{cores}" / mode
                        args.current_capture = destination
                        run = profile.capture_case(torch, torch_npu, case, sources, selected, mode, destination)
                        report["runs"].append(run)
                        write_json(metadata, report)
            report["comparisons"] = compare_runs(report["runs"])
            write_comparison(session / "comparison.md", report)
            profile.finalize_report(session, report)
    except BaseException as exc:
        record_failure(metadata, report, exc)
        try:
            print(capture_error(args.current_capture, args.detailed_log, exc), file=sys.stderr, flush=True)
        except BaseException:
            pass
        raise
    print("矩阵采集完成:", session, flush=True)


if __name__ == "__main__":
    main()
