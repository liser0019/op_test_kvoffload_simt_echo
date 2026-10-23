#!/usr/bin/env python3
"""串行采集 MTP×Scan 策略矩阵；排名只读 kernel_details.csv 的设备耗时。

固定请求数衡量开启MTP后的整次Plan代价；固定query行数是辅助控制实验。
两者分别报告，禁止把128行/32行的比值当作算法本身的退化率。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import compare as bench
from mtp_cases import resolve_shape

ROOT = Path(__file__).resolve().parent
GROUPS = {
    "local": ("abcdf", "v100", "v010", "v001", "v110", "v101", "v011", "v111"),
    "structural": ("abcdf", *(f"s{i:02d}" for i in range(1, 10))),
    "cold": ("abcdf", "cold_ballot"),
    "all": bench.IMPLEMENTATIONS,
}


def validate_gate(path, names, library, device):
    gate = json.loads(path.read_text())
    if (gate.get("status") != "passed" or gate.get("mode") != "npu"
            or not gate.get("production") or gate.get("device") != device
            or not set(names).issubset(gate.get("implementations", []))
            or gate.get("library_sha256") != hashlib.sha256(library.read_bytes()).hexdigest()
            or gate.get("source_manifest_sha256") != hashlib.sha256(
                (ROOT / "plan_variants_manifest.json").read_bytes()).hexdigest()):
        raise RuntimeError("正确性报告不是本库/本源码/本设备完整生产门禁，或未覆盖所选版本")


def interaction(records):
    """二因素交互项（单位us）；只在同一场景/MTP/行数内计算，负值表示额外降低耗时。"""
    p = {r["implementation"]: r["p50_us"] for r in records}
    result = {}
    for label, together, left, right in (("AB", "v110", "v100", "v010"),
                                         ("AC", "v101", "v100", "v001"),
                                         ("BC", "v011", "v010", "v001")):
        if all(k in p for k in ("abcdf", together, left, right)):
            result[label] = p[together] - p[left] - p[right] + p["abcdf"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--library", type=Path, default=ROOT / "build/libsimt_plan_variants_ops.so")
    parser.add_argument("--group", choices=GROUPS, default="all")
    parser.add_argument("--implementations", nargs="+", choices=bench.IMPLEMENTATIONS)
    parser.add_argument("--mtp", nargs="+", type=int, default=[0, 3])
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--fixed-query-rows", type=int, default=32)
    parser.add_argument("--comparison", choices=("same_requests", "same_query_rows", "both"), default="both")
    parser.add_argument("--scenario", choices=("all", *bench.SCENARIOS), default="all")
    parser.add_argument("--miss-counts", nargs="+", type=int,
                        help="改跑精确miss数矩阵，例如0 1 16 256 1024 2048；query只含稳定token")
    parser.add_argument("--miss-distributions", nargs="+", choices=("front", "back", "uniform"),
                        default=["front", "uniform"])
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--capacity", type=int, default=4096)
    parser.add_argument("--max-token", type=int, default=262144)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, help="每版采样数；默认每个执行位置轮转4次")
    parser.add_argument("--shuffle-lru", action="store_true")
    parser.add_argument("--correctness-report", type=Path, help="复用同库同设备通过的生产门禁")
    parser.add_argument("--output", type=Path, default=Path("/dev/shm/abcdf_mtp_scan"))
    args = parser.parse_args()
    names = tuple(args.implementations or GROUPS[args.group])
    if ("abcdf" not in names or len(set(names)) != len(names) or len(set(args.mtp)) != len(args.mtp)
            or 0 not in args.mtp or any(m < 0 for m in args.mtp) or args.warmup < 0
            or (args.repeat is not None and args.repeat <= 0) or not args.library.is_file()):
        parser.error("版本须去重且包含abcdf，MTP须去重且包含0，库必须存在，采样参数必须合法")
    if args.miss_counts is not None and (len(set(args.miss_counts)) != len(args.miss_counts)
                                        or any(n < 0 or n > args.topk for n in args.miss_counts)):
        parser.error("miss数须去重且在[0,topk]内")
    if len(set(args.miss_distributions)) != len(args.miss_distributions):
        parser.error("miss分布须去重")
    for mtp in args.mtp:
        if args.comparison != "same_query_rows":
            resolve_shape(args.requests, None, mtp)
        if args.comparison != "same_requests":
            resolve_shape(None, args.fixed_query_rows, mtp)
    args.output.mkdir(parents=True, exist_ok=True)
    destination = Path(tempfile.mkdtemp(prefix="matrix_", dir=args.output.resolve()))
    env = os.environ.copy()
    env.pop("ASCEND_LAUNCH_BLOCKING", None)
    env.pop("ASCEND_RT_VISIBLE_DEVICES", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    report = {"status": "running", "device": args.device, "library": str(args.library.resolve()),
              "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
              "timing": "device kernel duration; not Event or stage timing", "runs": [], "mtp_comparison": []}
    report_file = destination / "matrix_summary.json"
    def save():
        report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    save()
    print("矩阵目录:", destination, flush=True)
    try:
        gate_path = args.correctness_report or destination / "correctness.json"
        if args.correctness_report is None:
            subprocess.run([sys.executable, "-X", "faulthandler", str(ROOT / "checks/check_mtp_scan.py"),
                            "--device", str(args.device), "--library", str(args.library.resolve()),
                            "--production", "--report", str(gate_path), "--implementations", *names],
                           cwd=ROOT, env=env, check=True)
        validate_gate(gate_path, names, args.library, args.device)
        report["correctness_report"] = str(gate_path.resolve())
        modes = ("same_requests", "same_query_rows") if args.comparison == "both" else (args.comparison,)
        if args.miss_counts is None:
            scenarios = list(bench.SCENARIOS) if args.scenario == "all" else [args.scenario]
            tasks = [(scenario, scenario, None, None) for scenario in scenarios]
        else:
            tasks = [(f"miss{count}_{distribution}", "full_cache_75pct", count, distribution)
                     for count in args.miss_counts for distribution in args.miss_distributions]
            scenarios = [task[0] for task in tasks]
        cache = {}
        pairs = {}
        for scenario_index, (scenario, input_scenario, miss_count, distribution) in enumerate(tasks):
            selected = names
            if args.group == "all" and args.implementations is None:
                selected = (("abcdf", "cold_ballot") if scenario == "cold_reset"
                            else tuple(name for name in names if name != "cold_ballot"))
            for mode in modes:
                mtps = args.mtp if scenario_index % 2 == 0 else list(reversed(args.mtp))
                for mtp in mtps:
                    requests, rows = (resolve_shape(args.requests, None, mtp) if mode == "same_requests"
                                      else resolve_shape(None, args.fixed_query_rows, mtp))
                    key = (scenario, mtp, requests, selected)
                    reused = key in cache
                    if not reused:
                        capture_root = destination / f"{scenario}_mtp{mtp}_req{requests}"
                        repeat = args.repeat or 4 * len(selected)
                        cmd = [sys.executable, "-X", "faulthandler", str(ROOT / "profile_plan_variants.py"),
                               "--project", str(ROOT), "--device", str(args.device),
                               "--library", str(args.library.resolve()), "--requests", str(requests),
                               "--mtp", str(mtp), "--topk", str(args.topk), "--capacity", str(args.capacity),
                               "--max-token", str(args.max_token), "--seed", str(args.seed),
                               "--scenario", input_scenario, "--warmup", str(args.warmup), "--repeat", str(repeat),
                               "--output", str(capture_root), "--reference", "abcdf",
                               "--implementations", *selected]
                        if args.shuffle_lru:
                            cmd.append("--shuffle-lru")
                        if miss_count is not None:
                            cmd.extend(["--miss-count", str(miss_count), "--miss-distribution", distribution])
                        print(f"采集 {mode}/{scenario}: MTP={mtp}, requests={requests}, rows={rows}", flush=True)
                        subprocess.run(cmd, cwd=ROOT, env=env, check=True)
                        metadata_files = list(capture_root.rglob("capture_metadata.json"))
                        if len(metadata_files) != 1:
                            raise RuntimeError(f"采集元数据数量异常: {metadata_files}")
                        metadata = json.loads(metadata_files[0].read_text())
                        if metadata.get("status") != "complete":
                            raise RuntimeError("采集未完整通过，不生成排名")
                        cache[key] = (metadata_files[0], metadata["scenarios"][input_scenario])
                    meta_path, data = cache[key]
                    run = {"mode": mode, "scenario": scenario, "mtp": mtp, "requests": requests,
                           "rows": rows, "reused_identical_capture": reused, "capture_metadata": str(meta_path),
                           "workload": data["workload"], "case_sha256": data["case_sha256"],
                           "layouts": data["layouts"], "stats": data["kernel_stats"],
                           "factor_interactions_us": interaction(data["kernel_stats"])}
                    report["runs"].append(run)
                    pairs[(mode, scenario, mtp)] = run
                    save()
        for mode in modes:
            for scenario in scenarios:
                off = pairs[(mode, scenario, 0)]
                off_stats = {r["implementation"]: r for r in off["stats"]}
                for mtp in args.mtp:
                    if mtp == 0:
                        continue
                    on = pairs[(mode, scenario, mtp)]
                    for record in on["stats"]:
                        baseline = off_stats[record["implementation"]]
                        report["mtp_comparison"].append({
                            "mode": mode, "scenario": scenario, "implementation": record["implementation"],
                            "mtp": mtp, "off_rows": off["rows"], "on_rows": on["rows"],
                            "off_requests": off["requests"], "on_requests": on["requests"],
                            "off_p50_us": baseline["p50_us"], "on_p50_us": record["p50_us"],
                            "on_over_off_p50": record["p50_us"] / baseline["p50_us"],
                            "on_over_off_per_query_row": record["us_per_query_row_p50"] / baseline["us_per_query_row_p50"]})
        report["status"] = "complete"
        save()
    except BaseException as exc:
        report["status"] = "failed"
        report["failure"] = str(exc)
        save()
        raise
    lines = ["# ABCDF MTP / Scan 性能矩阵", "", "数值来自设备 kernel duration；全部调用处理当前组的所有query行。", "",
             "| 模式 | 场景 | MTP | 版本 | 请求/行数 | P50 μs | P95 μs | 相对 ABCDF 耗时下降 |", "|---|---|---:|---|---|---:|---:|---:|"]
    for run in report["runs"]:
        for stat in run["stats"]:
            lines.append(f"| {run['mode']} | {run['scenario']} | {run['mtp']} | {stat['implementation']} | "
                         f"{run['requests']}/{run['rows']} | {stat['p50_us']:.3f} | {stat['p95_us']:.3f} | "
                         f"{stat['gain_vs_abcdf_percent']:.2f}% |")
    (destination / "report.md").write_text("\n".join(lines) + "\n")
    print("完成:", report_file, flush=True)


if __name__ == "__main__":
    main()
