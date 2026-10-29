#!/usr/bin/env python3
"""仅在服务器执行的Plan/Transfer正确性门禁。失败立即退出；不把无卡/跳过当通过。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import plan_transfer_common as common
import compare as plan
from full_cache_cases import make_full_cache_case, reference_transfer_ids


def negative_transfer_gate(torch):
    """Transfer单独验证边界；这些非法描述符不伪称为Plan的正常输出。"""
    case = make_full_cache_case(2, 7, 17, 256, 75, seed=91)
    sources = common.prepare_sources(torch, case, 64, 96)
    inputs, initial, state, golden, _ = common.prepare_case(torch, case, sources, 64, 96)
    common.prime_transfer(torch, case, inputs, initial, state, golden)
    rows, topk = case["rows"], case["topk"]
    counts = np.full(rows, topk, np.int32)
    tokens = case["topk_indices"].copy()
    slots = np.tile(np.arange(topk, dtype=np.int32), (rows, 1))
    specs = []
    for label, count in (("zero", 0), ("negative_count", -7), ("clamped_count", topk + 9)):
        specs.append((label, np.full(rows, count, np.int32), tokens, slots, None, None, rows))
    invalid_tokens = tokens.copy()
    invalid_tokens[:, 1] = -1
    invalid_tokens[:, 2] = case["block_table"].shape[1] * case["block_size"]
    specs.append(("invalid_tokens", counts, invalid_tokens, slots, None, None, rows))
    invalid_slots = slots.copy()
    invalid_slots[:, 3] = -1
    invalid_slots[:, 4] = case["capacity"]
    specs.append(("invalid_slots", counts, tokens, invalid_slots, None, None, rows))
    specs.append(("invalid_requests", counts, tokens, slots, np.array([-1, rows], np.int32), None, rows))
    specs.append(("remapped_requests", counts, tokens, slots, np.array([1, 0], np.int32), None, rows))
    bad_pages = case["block_table"].copy()
    bad_pages[0].fill(-1)
    bad_pages[1].fill(case["host_num_blocks"])
    specs.append(("invalid_pages", counts, tokens, slots, None, bad_pages, rows))
    for active in (-1, 0, 1, rows + 5):
        specs.append((f"active_rows_{active}", counts, tokens, slots, None, None, active))
    specs.append(("uneven_prefix_mod", np.array([1, topk], np.int32), tokens, slots, None, None, rows))
    device = inputs["token_to_req"].device
    for label, count, token, slot, req, table, active in specs:
        req = case["token_to_req"] if req is None else req
        table = case["block_table"] if table is None else table
        common.restore(torch, initial, state, "transfer")
        for key, array in (("miss_count", count), ("miss_tokens", token), ("miss_slots", slot)):
            state[key].copy_(torch.from_numpy(np.ascontiguousarray(array)).to(device))
        inputs["token_to_req"].copy_(torch.from_numpy(np.ascontiguousarray(req)).to(device))
        inputs["block_table"].copy_(torch.from_numpy(np.ascontiguousarray(table)).to(device))
        inputs["active_rows"].fill_(active)
        expected = reference_transfer_ids(case, count, token, slot, req, table, active)
        common.invoke(torch, "transfer", case, inputs, state)
        torch.npu.synchronize()
        common.check_payload(state, expected, "negative/" + label)
        for key, array in (("miss_count", count), ("miss_tokens", token), ("miss_slots", slot)):
            np.testing.assert_array_equal(state[key].cpu().numpy(), array, err_msg=label + "/metadata-write")
        for key in plan.SIMT_STATE:
            np.testing.assert_array_equal(state[key].cpu().numpy(), golden[key], err_msg=label + "/state-write")
    print(f"Transfer边界门禁通过: {len(specs)}组", flush=True)
    return [entry[0] for entry in specs]


def run_small_gate(torch):
    records = []
    for rows, topk, capacity, length, block, bytes_k, bytes_v in (
        (1, 1, 3, 64, 7, 1, 31),
        (2, 33, 65, 512, 64, 32, 33),
        (9, 31, 65, 512, 7, 64, 128),
        (1, 7, 17, 128, 32, 16384, 16385),
    ):
        prototype = make_full_cache_case(rows, topk, capacity, length, 75, 72, block)
        sources = common.prepare_sources(torch, prototype, bytes_k, bytes_v)
        for percent in common.HIT_PERCENTAGES:
            case = make_full_cache_case(rows, topk, capacity, length, percent, 72, block)
            inputs, initial, state, golden, ids = common.prepare_case(torch, case, sources, bytes_k, bytes_v)
            for mode in ("plan", "transfer", "chain"):
                common.audit(torch, case, inputs, initial, state, golden, ids, mode, repeats=2)
            records.append({"rows": rows, "topk": topk, "capacity": capacity, "block_size": block,
                            "token_bytes_k": bytes_k, "token_bytes_v": bytes_v, "hit_percent": percent,
                            "modes": ["plan", "transfer", "chain"], "status": "passed"})
            print("正确性通过:", records[-1], flush=True)
    return {"cases": records, "transfer_boundaries": negative_transfer_gate(torch)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_arguments(parser)
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--report", type=Path, default=ROOT / "results/plan_transfer_correctness.json")
    args = parser.parse_args()
    preset = common.resolve_payload(args, parser)
    torch, _, environment = common.load_runtime(args)
    report = {"status": "running", "environment": environment, "payload": preset, "production": args.production}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    save()
    try:
        report["small_gate"] = run_small_gate(torch)
        if args.production:
            prototype = common.case_from_args(args, args.hit_percentages[0])
            sources = common.prepare_sources(torch, prototype, args.token_bytes_k, args.token_bytes_v)
            report["production_cases"] = []
            for percent in args.hit_percentages:
                case = common.case_from_args(args, percent)
                inputs, initial, state, golden, ids = common.prepare_case(
                    torch, case, sources, args.token_bytes_k, args.token_bytes_v)
                for mode in ("plan", "transfer", "chain"):
                    common.audit(torch, case, inputs, initial, state, golden, ids, mode)
                report["production_cases"].append(common.describe(case, args.token_bytes_k, args.token_bytes_v))
        torch.npu.synchronize()
        report["status"] = "passed"
    except BaseException as exc:
        report["status"] = "failed"
        report["failure"] = str(exc)
        raise
    finally:
        save()
    print("完整正确性报告:", args.report, flush=True)


if __name__ == "__main__":
    main()
