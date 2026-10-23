#!/usr/bin/env python3
"""上机正确性门禁；默认必须有 NPU，绝不将 SKIP 当作通过。

--cpu-only 只检查 fixture/golden，不验证 kernel。开发机禁跑；本脚本交给
服务器运行。所有候选比较7个输出/状态字段，并以自身上一轮设备状态续跑。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import compare as bench
from mtp_cases import make_mtp_case, make_miss_sweep_case


def fixtures(production=False):
    for item in bench.make_audit_cases():
        yield item
    # 两轮总数不同，避免读到上轮 total 仍然看似正确。
    uneven = bench.make_case(2, 2048, 4096, 65536, "cold_reset", 900)
    uneven["topk_indices"][:, 1536:] = -1
    yield "scan_tiles_1024_then_512", uneven
    zero = bench.make_case(2, 1025, 4096, 65536, "cold_reset", 901)
    zero["visible_seq_lens"].fill(0)
    yield "scan_all_zero", zero
    for width in (1, 31, 32, 33, 1023, 1024, 1025):
        yield f"tail_{width}", bench.make_case(2, width, max(width + 3, 64),
                                              16384, "cold_reset", width)
    for capacity in (4095, 4096, 4097):
        empty = bench.make_case(2, 33, capacity, 16384, "warm_all_hit", capacity)
        empty["slot_to_token"].fill(-1)
        yield f"whole_lru_all_evict_{capacity}", empty
    dup = bench.make_case(2, 33, 65, 512, "warm_all_hit", 19, shuffle_lru=True)
    dup["topk_indices"][:, 1] = dup["topk_indices"][:, 0]
    dup["slot_to_token"][:, -1] = dup["topk_indices"][:, 0]
    dup["lru_slots"][:, 3] = -1
    dup["lru_slots"][:, 7] = 65
    yield "duplicates_owner_invalid_lru", dup
    # H=128，强制33个合法token落入同一初始桶，覆盖线性探测与最后owner语义。
    colliders = []
    for token in range(1, 65536):
        mixed = token ^ (token >> 16)
        mixed = (mixed * 0x7feb352d) & 0xffffffff
        mixed ^= mixed >> 15
        mixed = (mixed * 0x846ca68b) & 0xffffffff
        mixed ^= mixed >> 16
        if mixed & 127 == 0:
            colliders.append(token)
            if len(colliders) == 33:
                break
    assert len(colliders) == 33
    conflict = bench.make_case(2, 33, 65, 65536, "warm_75pct", 24)
    conflict["topk_indices"][:] = colliders
    conflict["slot_to_token"].fill(-1)
    conflict["slot_to_token"][:, :24] = colliders[:24]
    yield "hash_linear_probe_chain", conflict
    yield "gm_fallback", bench.make_case(2, 4096, 8192, 65536, "warm_75pct", 22)
    yield "strategy_changes_fallback", bench.make_case(2, 2048, 8192, 65536, "warm_75pct", 23)
    for mtp in (0, 3):
        for scenario in bench.SCENARIOS:
            yield f"mtp{mtp}_{scenario}", make_mtp_case(2, 33, 65, 512, scenario, 31, mtp,
                                                        shuffle_lru=True)
    # 68行>64核上限，覆盖同核多行复用 scratch 与尾部 query 行。
    yield "mtp_multiline_68_rows", make_mtp_case(17, 33, 65, 512, "full_cache_75pct", 32, 3)
    if production:
        for mtp in (0, 3):
            for scenario in bench.SCENARIOS:
                yield f"production_mtp{mtp}_{scenario}", make_mtp_case(
                    32, 2048, 4096, 262144, scenario, 2026, mtp, shuffle_lru=True)
        # 小batch保留生产T/C；极端/不均衡位图及精确miss数由golden再断言。
        for mtp in (0, 3):
            for count in (0, 1, 16, 256, 1024, 2048):
                for distribution in ("front", "uniform"):
                    yield f"miss_sweep_mtp{mtp}_{count}_{distribution}", make_miss_sweep_case(
                        2, 2048, 4096, 262144, 2026, mtp, count, distribution, shuffle_lru=True)


def assert_fixture(case, golden):
    active = min(case["rows"], max(0, case.get("active_rows", case["rows"])))
    counts = golden["miss_count"][:active]
    assert np.all((counts >= 0) & (counts <= case["topk"]))
    assert set(golden) == set(bench.SIMT_STATE + bench.SIMT_OUTPUT)
    if "controlled_miss_count" in case:
        np.testing.assert_array_equal(counts, np.full(active, case["controlled_miss_count"]))
    if "mtp" in case:
        assert case["rows"] == case["requests"] * (case["mtp"] + 1)
        np.testing.assert_array_equal(case["visible_seq_lens"],
                                      case["context"] + case["query_position"] + 1)
        assert np.all(case["topk_indices"] < case["visible_seq_lens"][:, None])
        for request in range(case["requests"]):
            rows = np.flatnonzero(case["token_to_req"] == request)
            for row in rows[1:]:
                np.testing.assert_array_equal(case["slot_to_token"][rows[0]], case["slot_to_token"][row])
        if case["mtp"] and case["scenario"] != "cold_reset":
            assert not np.any(case["reset_mask"])
            invalid = (case["slot_to_token"] >= case["stable_prefix_lens"][:, None]).sum(axis=1)
            assert np.all(invalid == case["initial_speculative_slots_per_row"])


def trace(torch, names, device, mtp):
    """不从 golden 重写设备状态；覆盖接受、拒绝、部分接受及 request reset。"""
    previous = None
    states = None
    context = 400
    reports = []
    for step, accepted in enumerate((0, 0, 1, 2, 0, 1)):
        old_context = context
        context += accepted
        case = make_mtp_case(2, 33, 65, 512, "full_cache_75pct", 61 + step, mtp,
                             shuffle_lru=True, context=context)
        if accepted and previous is not None:
            # 上一轮的context token已被接受：强制query再次访问，不能只靠随机碰到。
            case["topk_indices"][:, 0] = old_context
        if step >= 4:
            case["req_ids"][0] += 10000
        if previous is not None:
            case.update({key: previous[key].copy() for key in bench.SIMT_STATE})
        golden = bench.reference_simt(case)
        if accepted and previous is not None and step < 4:
            assert np.all(golden["current_slots"][:, 0] >= 0)
            assert np.all(case["slot_to_token"][np.arange(case["rows"]),
                                                golden["current_slots"][:, 0]] == old_context)
        if mtp and not accepted and previous is not None and step < 4:
            # 被拒绝的context token不能作为历史hit复用，必须再次出现在miss输出中。
            assert np.all(np.any(golden["miss_tokens"] == context, axis=1))
        if torch is None:
            previous = golden
            continue
        if states is None:
            inputs, _, states = bench.prepare_npu(torch, case, None, device, names)
        else:
            inputs = {key: torch.from_numpy(np.ascontiguousarray(case[key])).to(device)
                      for key in ("req_ids", "topk_indices", "stable_prefix_lens", "visible_seq_lens",
                                  "token_to_req", "block_table")}
            inputs["active_rows"] = torch.tensor([case["rows"]], dtype=torch.int32, device=device)
        for name in names:
            for key in bench.SIMT_OUTPUT:
                states[name][key].fill_(-777)
            bench.invoke(torch, name, case, inputs, states[name])
            torch.npu.synchronize()
            bench.check_npu_result(name, states[name], {name: golden}, f"trace/mtp{mtp}/step{step}")
        previous = golden
        reports.append({"step": step, "accepted": accepted, "total_misses": int(golden["miss_count"].sum())})
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--library", type=Path, default=ROOT / "build/libsimt_plan_variants_ops.so")
    parser.add_argument("--implementations", nargs="+", choices=bench.IMPLEMENTATIONS,
                        default=list(bench.IMPLEMENTATIONS))
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--report", type=Path, default=ROOT / "results/mtp_scan_correctness.json")
    args = parser.parse_args()
    names = tuple(args.implementations)
    if len(names) != len(set(names)) or "abcdf" not in names:
        parser.error("版本必须去重并包含 abcdf 基线")
    from checks.plan_build_info import catalog_identity
    identity = catalog_identity()
    for scenario in bench.SCENARIOS:
        off = make_mtp_case(2, 33, 65, 512, scenario, 73, 0)
        on = make_mtp_case(2, 33, 65, 512, scenario, 73, 3)
        np.testing.assert_array_equal(off["topk_indices"], on["topk_indices"][::4])
        assert off["context"] == on["context"]
    torch = None
    device = None
    if not args.cpu_only:
        import torch
        import torch_npu
        if not args.library.is_file() or not torch.npu.is_available():
            parser.error("需要本分支共享库及可用NPU；不会跳过设备校验")
        torch.npu.set_device(args.device)
        device = torch.device(f"npu:{args.device}")
        torch.ops.load_library(str(args.library.resolve()))
        if json.loads(torch.ops.sparse_kv_plan_op.plan_variants_info()) != identity:
            parser.error("共享库身份过期，请重建")
    report = {"status": "running", "mode": "fixture_only" if args.cpu_only else "npu",
              "device": args.device, "implementations": names, "production": args.production,
              "library_sha256": None if args.cpu_only else hashlib.sha256(args.library.read_bytes()).hexdigest(),
              "source_manifest_sha256": hashlib.sha256((ROOT / "plan_variants_manifest.json").read_bytes()).hexdigest(),
              "cases": [], "traces": {}}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    # 失败时留下 running，禁止把部分完成的报告作为通过凭据。
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for label, case in fixtures(args.production):
        golden = bench.reference_simt(case)
        assert_fixture(case, golden)
        if not args.cpu_only:
            bench.audit_case(torch, case, names, device, repeats=2)
        report["cases"].append({"name": label, "status": "fixture_checked" if args.cpu_only else "passed"})
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(label, report["cases"][-1]["status"], flush=True)
    for mtp in (0, 3):
        report["traces"][str(mtp)] = trace(torch, names, device, mtp)
    report["status"] = "fixture_only" if args.cpu_only else "passed"
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("报告:", args.report, flush=True)
    if args.cpu_only:
        print("只检查了fixture/golden；没有执行或验证设备kernel")


if __name__ == "__main__":
    main()
