#!/usr/bin/env python3
"""外部msprof op单Transfer入口；内部不创建torch_npu.profiler，不运行额外同名kernel门禁。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import tempfile
sys.dont_write_bytecode = True

import plan_transfer_common as common
from full_cache_cases import describe
from harness_io import preflight_args, write_json, record_failure


def get_marker(torch_npu):
    marker = getattr(torch_npu.npu, "mstx", None)
    if marker is not None and not hasattr(marker, "range_start"):
        marker = getattr(marker, "mstx", None)
    if marker is None or not hasattr(marker, "range_start"):
        try:
            import mstx as marker
        except ImportError as exc:
            raise RuntimeError("需要MSTX接口；BasicInfo可不加--mstx") from exc
    return marker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_arguments(parser)
    parser.set_defaults(launch_mode="prepared")
    parser.add_argument("--hit-percent", choices=(75, 90, 100), type=int, default=75)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--mstx", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("/dev/shm/transfer_msopprof_runner"))
    args = parser.parse_args()
    preset = common.resolve_payload(args, parser)
    if args.warmup < 0 or args.repeat <= 0:
        parser.error("warmup>=0、repeat>0")
    case = common.case_from_args(args, args.hit_percent)
    print("Host内存需求与用例:", json.dumps(describe(case, args.token_bytes_k, args.token_bytes_v),
                                         ensure_ascii=False), flush=True)
    storage = preflight_args(args, args.output)
    session = Path(tempfile.mkdtemp(prefix="runner_", dir=storage["output"]["path"]))
    report = {"status": "running", "payload": preset, "storage": storage,
              "launch_mode": args.launch_mode, "warmup": args.warmup, "repeat": args.repeat,
              "mstx_range": "transfer_target" if args.mstx else None}
    path = session / "report.json"
    write_json(path, report)
    try:
        torch, torch_npu, environment = common.load_runtime(args)
        marker = get_marker(torch_npu) if args.mstx else None
        sources = common.prepare_sources(torch, case, args.token_bytes_k, args.token_bytes_v)
        inputs, initial, state, golden, ids = common.prepare_case(
            torch, case, sources, args.token_bytes_k, args.token_bytes_v, args.launch_mode,
            args.transfer_variant, args.transfer_core_limit)
        # 只生成Plan输出，不提前执行任何Transfer目标kernel。
        common.prime_transfer(torch, case, inputs, initial, state, golden)
        report.update(environment=environment, transfer_variant=state["transfer_info"],
                      workload=describe(case, args.token_bytes_k, args.token_bytes_v),
                      case_sha256=common.plan.array_hash({k: v for k, v in case.items()
                                                       if isinstance(v, common.np.ndarray)}),
                      buffer_addresses_sha256=common.binding_hash(inputs, state))
        for step in range(args.warmup + args.repeat):
            common.restore(torch, initial, state, "transfer")
            active = step >= args.warmup
            range_id = marker.range_start("transfer_target") if marker is not None and active else None
            if marker is not None and active and not range_id:
                raise RuntimeError("MSTX没有返回有效range")
            try:
                common.invoke(torch, "transfer", case, inputs, state)
                torch.npu.synchronize()
            finally:
                if range_id is not None:
                    marker.range_end(range_id)
            common.check_result(case, state, golden, ids, "transfer", f"msopprof/{step}")
        report["status"] = "passed"
        write_json(path, report)
    except BaseException as exc:
        record_failure(path, report, exc)
        raise
    print("Transfer入口验证完成:", path, flush=True)


if __name__ == "__main__":
    main()
