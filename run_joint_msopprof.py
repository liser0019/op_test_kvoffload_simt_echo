#!/usr/bin/env python3
"""外部msprof op入口；内部不嵌套torch profiler，不提前执行同名kernel门禁。"""
from pathlib import Path
import argparse
import sys
sys.dont_write_bytecode = True
import joint_plan_common as common
from joint_plan_cases import SCENARIOS, OVERLAPS, make_case, describe
from harness_io import preflight_args, write_json, record_failure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_runtime_arguments(parser)
    parser.add_argument("--implementation", choices=common.IMPLEMENTATIONS, default="joint")
    parser.add_argument("--mtp", type=int, choices=(0, 1, 3), default=3)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--scenario", choices=SCENARIOS, default="full_cache_75pct")
    parser.add_argument("--overlap", choices=OVERLAPS, default="shared")
    for name, value in (("topk", 2048), ("capacity", 4096), ("max-token", 262144), ("block-size", 64), ("seed", 2026)):
        parser.add_argument("--" + name, type=int, default=value)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--mstx", action="store_true")
    parser.add_argument("--report", type=Path, default=Path("/workspace/joint_plan_results/msopprof_runner/report.json"))
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat <= 0:
        parser.error("warmup>=0、repeat>0")
    case = make_case(args.requests, args.topk, args.capacity, args.max_token, args.mtp,
                     args.scenario, args.overlap, args.seed, args.block_size)
    storage = preflight_args(args, args.report.parent)
    report = {"status": "running", "storage": storage, "workload": describe(case),
              "implementation": args.implementation, "warmup": args.warmup, "repeat": args.repeat,
              "kernel_name": common.KERNELS[args.implementation]}
    bundle = None
    try:
        torch, torch_npu, environment = common.load_runtime(args)
        bundle = common.prepare(torch, case, args.implementation)
        report.update(environment=environment, layout=bundle["layout"])
        marker = None
        if args.mstx:
            marker = getattr(torch_npu.npu, "mstx", None)
            if marker is not None and not hasattr(marker, "range_start"):
                marker = getattr(marker, "mstx", None)
            if marker is None or not hasattr(marker, "range_start"):
                raise RuntimeError("当前torch_npu没有可用MSTX range接口")
        for step in range(args.warmup + args.repeat):
            common.restore(torch, bundle)
            common.poison(bundle)
            torch.npu.synchronize()
            before = common.snapshot(bundle)  # 范围外准备完整审计；无同名kernel预调用。
            identifier = marker.range_start("joint_plan_target") if marker is not None and step >= args.warmup else None
            if marker is not None and step >= args.warmup and not identifier:
                raise RuntimeError("MSTX未返回有效range")
            try:
                common.invoke(bundle)
                torch.npu.synchronize()
            finally:
                if identifier is not None:
                    marker.range_end(identifier)
            report["correctness"] = common.check(case, bundle, before)
        common.close(bundle)
        report["status"] = "passed"
        write_json(args.report, report)
    except BaseException as exc:
        if bundle is not None:
            common.dump_failure(case, bundle, args.report.parent / "failure_state")
        record_failure(args.report, report, exc)
        raise
    print("单目标调用已完成语义审计:", args.report, flush=True)


if __name__ == "__main__":
    main()
