#!/usr/bin/env python3
"""Compare 475K resident lookup and split AIV Plan kernels with per-sample gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joint_plan_common as common
from harness_io import capture_error, diagnostic_log, preflight_args, record_failure, write_json
from joint_plan_cases import make_case, sort_topk
from profile_joint_mtp import capture

FROZEN = "joint_mtp3_full_512"
DIR = "joint_mtp3_dir475_512"
BITMAP = "joint_mtp3_bitmap475_512"
PIPELINES = tuple(f"joint_mtp3_lookup{shards}_512" for shards in (1, 2, 4))
TEAM = ("joint_mtp3_bitmap_ubmap_512", "joint_mtp3_bt_share_512",
        "joint_mtp3_bt_direct_512", "joint_mtp3_bt_direct_odd_512")
IMPLEMENTATIONS = (DIR, BITMAP, *PIPELINES)
SELECTABLE = (*IMPLEMENTATIONS, *TEAM, FROZEN)
CONTEXTS = (102400, 262144, 358400, 486400)


def report_lines(report):
    out = [
        "# MTP3 resident 查询与多 AIV 对照",
        "",
        "固定 MTP3、TopK2048、容量8192。475K = 486400 token。输入为共同升序的合成 TopK。",
        "单位 μs。单 kernel 使用 Duration，多 kernel 使用首个设备 kernel 开始到最后结束的 span。",
        "lookup1/2/4 采用准备、查槽、提交三个 kernel，lookup1 是单 AIV 查槽对照。",
        "bt_share 与 bt_direct 在一个 kernel 中让两个 AIV 合作。bitmap_ubmap 是保持映射片上的单 AIV 对照。",
        "每个样本恢复同一初态并完整校验。冷启动保留全部实际设备 kernel 和同步开销。",
        "",
        "| 上下文 | Batch | 工况/重合 | 实现 | P50/P95 | 样本 | 联合/回退 | 各段 P50 |",
        "|---:|---:|---|---|---:|---:|---|---|",
    ]
    for run in report["runs"]:
        w = run["workload"]
        for name, stat in run["stats"].items():
            c = stat["correctness"]
            stage = "/".join(f'{x["p50_us"]:.3f}' for x in stat.get("stages", {}).values()) or "—"
            out.append(f'| {w["max_token"]} | {w["requests"]} | {w["scenario"]}/{w["overlap"]} | {name} | '
                       f'{stat["p50_us"]:.3f}/{stat["p95_us"]:.3f} | {stat["count"]} | '
                       f'{c["joint_groups"]}/{c["overflow_groups"]} | {stage} |')
    return "\n".join(out + ["", "逐样本耗时、各段 Block Num 与 CSV 路径见 capture_metadata.json。"]) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_runtime_arguments(parser)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests", nargs="+", type=int, default=[1, 4, 8, 9, 10, 16, 32])
    parser.add_argument("--contexts", nargs="+", type=int, choices=CONTEXTS, default=[486400])
    parser.add_argument("--scenarios", nargs="+", choices=("cold_reset", "full_cache_75pct", "full_cache_90pct", "full_cache_stable_hit"),
                        default=["cold_reset", "full_cache_75pct"])
    parser.add_argument("--overlaps", nargs="+", choices=("shared", "overlap40", "disjoint"), default=["overlap40"])
    parser.add_argument("--implementations", nargs="+", choices=SELECTABLE,
                        default=list(IMPLEMENTATIONS))
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.repeat < 1 or args.warmup < 0 or any(not 1 <= n <= 32 for n in args.requests):
        parser.error("batch 需在 1～32，warmup>=0 且 repeat>0")
    for options in (args.requests, args.contexts, args.scenarios, args.overlaps, args.implementations):
        if len(options) != len(set(options)):
            parser.error("配置项不能重复")
    if FROZEN in args.implementations and args.contexts != [262144]:
        parser.error(f"{FROZEN} 的冻结 Host 门禁仅支持 --contexts 262144")
    args.fast_fixed_check = True
    args.capture_audit_repeat = 2
    storage = preflight_args(args, args.output)
    args.output = Path(storage["output"]["path"])
    metadata = args.output / "capture_metadata.json"
    if metadata.exists():
        raise FileExistsError(f"已有采集目录，请选择新的输出位置 {args.output}")
    report = {"status": "running", "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "storage": storage, "runs": []}
    try:
        with diagnostic_log(args.output / "profile.log"):
            torch, torch_npu, env = common.load_runtime(args)
            report["environment"] = env
            write_json(metadata, report)
            for context in args.contexts:
                for requests in args.requests:
                    for scenario in args.scenarios:
                        for overlap in args.overlaps:
                            print(f"context={context} batch={requests} {scenario}/{overlap}", flush=True)
                            case = sort_topk(make_case(requests, 2048, 8192, context, 3, scenario, overlap, args.seed, 128))
                            output = args.output / f"ctx{context}" / f"bs{requests}" / scenario / overlap
                            try:
                                result = capture(torch, torch_npu, case, args.implementations, args, output)
                            except Exception as exc:
                                raise capture_error(output, args.output / "profile.log", exc) from exc
                            for name, stat in result["stats"].items():
                                if (stat["correctness"]["joint_groups"], stat["correctness"]["overflow_groups"]) != (requests, 0):
                                    raise RuntimeError(f"{name} failed to use the joint path for all requests")
                            report["runs"].append(result)
                            write_json(metadata, report)
            (args.output / "report.md").write_text(report_lines(report))
            report["status"] = "complete"
            write_json(metadata, report)
            write_json(args.output / "COMPLETE.json", {"status": "complete"})
    except BaseException as exc:
        record_failure(metadata, report, exc)
        raise
    print(args.output / "report.md", flush=True)


if __name__ == "__main__":
    main()
