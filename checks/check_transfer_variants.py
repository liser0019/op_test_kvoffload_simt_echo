#!/usr/bin/env python3
"""服务器NPU门禁：全部因子组合、两种启动、批次尾部与原算法回退。"""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
import json
import plan_transfer_common as common
import transfer_variants as catalog
from checks.check_plan_transfer import run_small_gate, negative_transfer_gate
from full_cache_cases import make_full_cache_case
from harness_io import preflight_args, write_json, record_failure


def run_gate(torch, names, launches=("prepared",)):
    result = []
    # 含非对齐、超16KiB回退、非法描述符、active_rows和一核多行。
    for name in names:
        item = {"variant": name, "small": run_small_gate(torch, launches, name, 64), "batch_edges": [],
                "batch_invalid": {launch: negative_transfer_gate(torch, launch, name, 1, topk=65)
                                  for launch in launches}}
        # 小core数使每核真正处理满8项及尾批；同时走C生产特化/非生产通用路径。
        for widths in ((64, 96), (1024, 128)):
            prototype = make_full_cache_case(2, 65, 97, 1024, 75, seed=110)
            sources = common.prepare_sources(torch, prototype, *widths)
            for percent in (75, 90, 100):
                case = make_full_cache_case(2, 65, 97, 1024, percent, seed=110)
                for cores in (1, 2):
                    for launch in launches:
                        bundle = common.prepare_case(torch, case, sources, *widths, launch, name, cores)
                        inputs, initial, state, golden, ids = bundle
                        for mode in ("transfer", "chain"):
                            common.audit(torch, case, inputs, initial, state, golden, ids, mode, repeats=2)
                        item["batch_edges"].append({"widths": widths, "percent": percent,
                                                    "cores": cores, "launch": launch,
                                                    "path": state["transfer_info"]})
        item["status"] = "passed"
        result.append(item)
        print("变体门禁通过:", name, flush=True)
    # 验证矩阵实际使用的“同一缓冲换上下文”路径，而不只测试独立分配。
    reused = []
    for percent in (75, 100):
        case = make_full_cache_case(2, 65, 97, 1024, percent, seed=111)
        sources = common.prepare_sources(torch, case, 1024, 128)
        inputs, initial, state, golden, ids = common.prepare_case(torch, case, sources, 1024, 128)
        fingerprint = common.binding_hash(inputs, state)
        for name in names:
            for launch in launches:
                common.configure_transfer(torch, case, inputs, state, launch, name, 1)
                assert common.binding_hash(inputs, state) == fingerprint, "切换版本改变了工作缓冲地址"
                for mode in ("transfer", "chain"):
                    common.audit(torch, case, inputs, initial, state, golden, ids, mode, repeats=2)
        reused.append({"hit_percent": percent, "buffer_addresses_sha256": fingerprint, "status": "passed"})
    return {"variants": result, "reused_buffers": reused}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_arguments(parser)
    parser.add_argument("--variants", nargs="+", choices=tuple(catalog.VARIANTS), default=list(catalog.VARIANTS))
    parser.add_argument("--launches", nargs="+", choices=("legacy", "prepared"), default=["legacy", "prepared"])
    parser.add_argument("--report", type=Path, default=Path("/dev/shm/transfer_variant_checks/report.json"))
    args = parser.parse_args()
    common.resolve_payload(args, parser)
    storage = preflight_args(args, args.report.parent)
    report = {"status": "running", "storage": storage, "variants": args.variants, "launches": args.launches}
    write_json(args.report, report)
    try:
        torch, _, environment = common.load_runtime(args)
        report["environment"] = environment
        report["results"] = run_gate(torch, args.variants, tuple(args.launches))
        report["status"] = "passed"
        write_json(args.report, report)
    except BaseException as exc:
        record_failure(args.report, report, exc)
        raise
    print("全部所选版本通过:", args.report, flush=True)


if __name__ == "__main__":
    main()
