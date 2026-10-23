#!/usr/bin/env python3
"""外部 msopprof 的单路入口：内部不创建 torch_npu.profiler。

BasicInfo 首次探测使用 --warmup 0，确保第一个目标调用就是被校验的调用。
正式采集可启用 --mstx：预热/恢复在范围外，只有目标调用在 abcdf_target 内。
MSTX 只是Host筛选，与 --replay-mode=range 是不同概念。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import compare as bench
from mtp_cases import make_mtp_case, make_miss_sweep_case, describe_case, resolve_shape


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=bench.IMPLEMENTATIONS, default="abcdf")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--requests", type=int)
    parser.add_argument("--rows", type=int)
    parser.add_argument("--mtp", type=int, default=0)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--capacity", type=int, default=4096)
    parser.add_argument("--max-token", type=int, default=262144)
    parser.add_argument("--scenario", choices=bench.SCENARIOS, default="full_cache_75pct")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--shuffle-lru", action="store_true")
    parser.add_argument("--miss-count", type=int)
    parser.add_argument("--miss-distribution", choices=("front", "back", "uniform"), default="front")
    parser.add_argument("--mstx", action="store_true")
    parser.add_argument("--library", type=Path, default=bench.ROOT / "build/libsimt_plan_variants_ops.so")
    parser.add_argument("--report", type=Path, default=Path("results/msopprof_runner.json"))
    args = parser.parse_args()
    try:
        requests, rows = resolve_shape(args.requests, args.rows, args.mtp)
        case = make_mtp_case(requests, args.topk, args.capacity, args.max_token,
                             args.scenario, args.seed, args.mtp, shuffle_lru=args.shuffle_lru)
        if args.miss_count is not None:
            if args.scenario != "full_cache_75pct":
                raise ValueError("--miss-count需要--scenario full_cache_75pct")
            case = make_miss_sweep_case(requests, args.topk, args.capacity, args.max_token,
                                       args.seed, args.mtp, args.miss_count, args.miss_distribution,
                                       shuffle_lru=args.shuffle_lru)
    except ValueError as exc:
        parser.error(str(exc))
    if args.warmup < 0 or args.repeat <= 0 or not args.library.is_file():
        parser.error("库必须存在，warmup>=0，repeat>0")
    if max(rows * args.max_token, rows * (args.capacity + args.topk), args.max_token) > bench.I32_MAX:
        parser.error("Shape超过int32索引范围")
    import torch
    import torch_npu
    if not torch.npu.is_available():
        parser.error("没有可用 NPU")
    torch.npu.set_device(args.device)
    torch.ops.load_library(str(args.library.resolve()))
    from checks.plan_build_info import catalog_identity
    if json.loads(torch.ops.sparse_kv_plan_op.plan_variants_info()) != catalog_identity():
        parser.error("当前源码与共享库身份不一致，请重建")
    marker = None
    if args.mstx:
        marker = getattr(torch_npu.npu, "mstx", None)
        if marker is not None and not hasattr(marker, "range_start"):
            marker = getattr(marker, "mstx", None)
        if marker is None or not hasattr(marker, "range_start"):
            try:
                import mstx as marker
            except ImportError:
                parser.error("本环境没有可用 MSTX Python 接口；BasicInfo 可不用 --mstx")
    name = args.implementation
    golden = bench.reference_simt(case)
    if args.miss_count is not None:
        bench.np.testing.assert_array_equal(golden["miss_count"], bench.np.full(rows, args.miss_count))
    inputs, initial, states = bench.prepare_npu(torch, case, None, torch.device(f"npu:{args.device}"), (name,))
    state = states[name]
    report = {"status": "running", "implementation": name,
              "kernel_name": bench.VARIANTS[name]["kernel_name"], "device": args.device,
              "workload": describe_case(case, golden), "layout": state["layout"],
              "case_sha256": bench.array_hash({k: v for k, v in case.items() if isinstance(v, bench.np.ndarray)}),
              "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
              "compiler": json.loads(torch.ops.sparse_kv_plan_op.plan_compiler_info()),
              "warmup": args.warmup, "repeat": args.repeat, "mstx": args.mstx}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for _ in range(args.warmup):
        bench.restore_state(torch, name, initial, state)
        bench.invoke(torch, name, case, inputs, state)
        torch.npu.synchronize()
    if args.warmup:
        bench.check_npu_result(name, state, {name: golden}, "msopprof/after-warmup")
    for iteration in range(args.repeat):
        bench.restore_state(torch, name, initial, state)
        for key in bench.SIMT_OUTPUT:
            state[key].fill_(-777)
        torch.npu.synchronize()
        token = None
        if marker is not None:
            token = marker.range_start("abcdf_target", None)
            if token is None or token == 0:
                raise RuntimeError("MSTX 没有返回有效range；检查外部采集的 --mstx=on 和接口配套")
        try:
            bench.invoke(torch, name, case, inputs, state)
            torch.npu.synchronize()
        finally:
            if marker is not None and token is not None:
                marker.range_end(token)
        bench.check_npu_result(name, state, {name: golden}, f"msopprof/target-{iteration}")
    report["status"] = "passed"
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("目标及状态校验通过:", args.report, flush=True)


if __name__ == "__main__":
    main()
