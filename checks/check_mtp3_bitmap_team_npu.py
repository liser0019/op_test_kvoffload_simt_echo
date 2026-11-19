#!/usr/bin/env python3
"""Exact NPU gate for old and team MTP3 bitmap candidates.

All selected implementations are checked on disjoint, resident, boundary and
continuous-state workloads.  The report contains metadata only; numeric state
is written only by joint_plan_common.dump_failure when a check fails.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joint_plan_common as common
from checks.check_mtp3_candidates import before_for, target_case, union_size
from harness_io import write_json
from joint_plan_cases import initial_arrays, make_case, sort_topk, valid_sets
from joint_variants import MTP3_475_MAX_TOKEN, mtp3_config

STANDARD_CONTEXT = 262144
HIGH_CONTEXT = MTP3_475_MAX_TOKEN
CAPACITY = 8192


def source_id(environment):
    identity = environment.get("identity") if environment else None
    payload = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _copy_before(torch, bundle, before):
    device = bundle["state"]["last_req_ids"].device
    for key, value in before.items():
        target = bundle["state"][key]
        target.copy_(torch.from_numpy(np.ascontiguousarray(value)).to(device=device, dtype=target.dtype))
    torch.npu.synchronize()


def _invoke_and_check(torch, case, bundle, before, phase, apply_before=None):
    bundle["phase"] = phase
    if apply_before is not None:
        _copy_before(torch, bundle, apply_before)
    common.poison(bundle)
    torch.npu.synchronize()
    common.invoke(bundle)
    torch.npu.synchronize()
    result = common.check(case, bundle, before)
    _check_bitmap_exchange(case, bundle, before)
    return result


def _check_bitmap_exchange(case, bundle, before):
    config = mtp3_config(common.VARIANTS.get(bundle["name"], 0)) or {}
    if not config.get("bitmap_ubmap"):
        return
    workspace = bundle["state"]["workspace"].cpu().numpy()
    poison = np.uint32(np.iinfo(np.int32).max)
    map_words = case["rows"] * case["topk"] // 2
    if not np.all(workspace[:map_words] == poison):
        raise AssertionError("packed query map was written to GM")
    team = config["bitmap_team"]
    if not team or case["requests"] * 2 > bundle["layout"]["blocks"]:
        return
    count = CAPACITY // 4
    result_offset = CAPACITY + 8 if team == 1 else 0
    exchange_words = count + result_offset
    all_sets = valid_sets(case)
    for group in range(case["requests"]):
        first = group * 4
        start = map_words + group * exchange_words
        uid_order = sorted(set().union(*all_sets[first:first+4]))
        request = case["token_to_req"][first]
        hot = before["row_modes"][first] == 1 and before["last_req_ids"][first] == case["req_ids"][request]
        results = workspace[start+result_offset:start+exchange_words].view(np.uint32)
        if not hot:
            if not np.all(results == poison):
                raise AssertionError("cold helper wrote result workspace")
            if team == 1 and workspace[start] != -1:
                raise AssertionError("cold share header was not invalidated")
            continue
        stable = max(0, min(case["max_token"], min(case["stable_prefix_lens"][first:first+4])))
        uids = {token: uid for uid, token in enumerate(uid_order)}
        slots = before["slot_to_token"][first, CAPACITY//2:]
        encoded = np.array([
            0x8000 if token >= stable else 0x4000 | uids[token] if token in uids and token >= 0 else 0
            for token in slots], dtype=np.uint32)
        expected = encoded[::2] | (encoded[1::2] << 16)
        np.testing.assert_array_equal(results, expected, err_msg="exact auxiliary bitmap slot ranks")
        if team == 1:
            if workspace[start] != len(uid_order):
                raise AssertionError("shared unique count mismatch")
            np.testing.assert_array_equal(workspace[start+8:start+8+len(uid_order)], uid_order)


def _with_bundle(torch, case, name, failure_dir, operation):
    bundle = common.prepare(torch, case, name)
    try:
        try:
            return operation(bundle)
        except BaseException:
            common.dump_failure(case, bundle, failure_dir)
            raise
    finally:
        common.close(bundle)


def _result_summary(result):
    return {
        "status": result["status"],
        "joint_groups": int(result.get("joint_groups", 0)),
        "miss_count_per_row": [int(value) for value in result.get("miss_count_per_row", [])],
    }


def _run_standard(torch, name, kind, context, high_window, failure_dir, label):
    case = target_case(kind, context, high_window=high_window)
    case["scenario"] = "full_cache_75pct"
    before = before_for(case, kind)

    def operation(bundle):
        result = _invoke_and_check(torch, case, bundle, before, label, before)
        if kind in ("disjoint", "all_hits") or high_window:
            if union_size(case) != CAPACITY:
                raise AssertionError(f"{kind} must cover the full {CAPACITY}-token union")
        if kind == "all_hits" and any(result["miss_count_per_row"]):
            raise AssertionError("all_hits workload produced misses")
        return result

    result = _with_bundle(torch, case, name, failure_dir, operation)
    return {
        "case": kind,
        "context": context,
        "high_window": bool(high_window),
        "union_size": union_size(case),
        "invocations": 1,
        **_result_summary(result),
    }


def _run_cold_then_hot(torch, name, failure_dir, label):
    case = target_case("disjoint", HIGH_CONTEXT, high_window=True)
    case["scenario"] = "cold_reset"
    case["last_req_ids"][:] = -1
    before = before_for(case, "disjoint")

    def operation(bundle):
        cold = _invoke_and_check(torch, case, bundle, before, label + "/cold", before)
        if cold["miss_count_per_row"] != [2048] * 4:
            raise AssertionError(f"cold leg did not miss the full union: {cold['miss_count_per_row']}")
        # Deliberately do not restore: snapshot the state left by the cold call.
        hot_before = common.snapshot(bundle)
        hot = _invoke_and_check(torch, case, bundle, hot_before, label + "/hot")
        if any(hot["miss_count_per_row"]):
            raise AssertionError(f"second call was not fully hot: {hot['miss_count_per_row']}")
        return {"cold": _result_summary(cold), "hot": _result_summary(hot)}

    result = _with_bundle(torch, case, name, failure_dir, operation)
    return {
        "case": "cold_then_hot",
        "context": HIGH_CONTEXT,
        "high_window": True,
        "union_size": union_size(case),
        "invocations": 2,
        "restored_between_invocations": False,
        **result,
    }


def _heterogeneous_case():
    case = sort_topk(make_case(
        requests=2, topk=2048, capacity=CAPACITY, max_token=HIGH_CONTEXT,
        mtp=3, scenario="full_cache_75pct", overlap="disjoint",
        seed=4753, history_pool_tokens=CAPACITY))
    fixture = target_case("disjoint", HIGH_CONTEXT, high_window=True)
    for key in ("topk_indices", "visible_seq_lens", "stable_prefix_lens"):
        case[key] = np.concatenate((fixture[key], fixture[key]), axis=0)
    before = initial_arrays(case, True, CAPACITY)
    before["slot_to_token"].fill(-1)
    before["last_req_ids"].fill(-1)
    before["row_modes"].fill(0)

    sets = valid_sets(case)
    hot_union = np.array(sorted(set().union(*sets[:4])), dtype=np.int32)
    cold_union = set().union(*sets[4:])
    if len(hot_union) != CAPACITY or len(cold_union) != CAPACITY:
        raise AssertionError("heterogeneous requests must each have a complete 8192-token union")

    # Request 0 is hot with every query token resident and a matching identity.
    before["slot_to_token"][0, :CAPACITY] = hot_union
    before["last_req_ids"][:4] = case["req_ids"][0]
    before["row_modes"][0] = 1
    before["row_modes"][1:4] = 2
    # Request 1 stays cold: no identity, resident rows, or cached tokens.
    return case, before, {"hot_request": 0, "cold_request": 1,
                          "hot_union_size": int(len(hot_union)),
                          "cold_union_size": int(len(cold_union))}


def _run_heterogeneous(torch, name, failure_dir, label):
    case, before, roles = _heterogeneous_case()

    def operation(bundle):
        result = _invoke_and_check(torch, case, bundle, before, label, before)
        if result["joint_groups"] != 2:
            raise AssertionError(f"expected two joint MTP3 requests, got {result['joint_groups']}")
        counts = result["miss_count_per_row"]
        if counts[:4] != [0] * 4 or counts[4:] != [2048] * 4:
            raise AssertionError(f"heterogeneous miss counts do not match exact hot/cold expectations: {counts}")
        return result

    result = _with_bundle(torch, case, name, failure_dir, operation)
    return {
        "case": "heterogeneous_two_requests",
        "context": HIGH_CONTEXT,
        "high_window": True,
        "requests": 2,
        "mtp": 3,
        "invocations": 1,
        **roles,
        **_result_summary(result),
    }


def run(args):
    report_path = args.report
    failure_dir = report_path.parent / "failure_state"
    report = {
        "status": "running",
        "scope": "MTP3 bitmap team exact NPU gate",
        "configuration": {
            "implementations": list(args.implementations),
            "standard_context": STANDARD_CONTEXT,
            "high_context": HIGH_CONTEXT,
            "capacity": CAPACITY,
            "read_only_inputs": True,
            "guards": True,
            "exact_state": True,
            "failure_dump": "joint_plan_common.dump_failure",
        },
        "implementations": list(args.implementations),
        "results": {},
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report_path, report)
    try:
        torch, _torch_npu, environment = common.load_runtime(args)
        report["environment"] = environment
        report["source_id"] = source_id(environment)
        write_json(report_path, report)

        for name in args.implementations:
            print(f"MTP3 bitmap-team gate: {name}", flush=True)
            records = []
            for kind in ("disjoint", "all_hits", "repeated_resident", "short", "empty"):
                records.append(_run_standard(
                    torch, name, kind, STANDARD_CONTEXT, False, failure_dir,
                    f"{name}/{kind}"))
            records.append(_run_standard(
                torch, name, "disjoint", HIGH_CONTEXT, True, failure_dir,
                f"{name}/high_window_475k"))
            records.append(_run_cold_then_hot(
                torch, name, failure_dir, f"{name}/cold_then_hot_475k"))
            records.append(_run_heterogeneous(
                torch, name, failure_dir, f"{name}/heterogeneous_two_requests"))
            report["results"][name] = {"status": "passed", "cases": records}
            write_json(report_path, report)

        report["status"] = "passed"
        report["gate"] = {
            "status": "passed",
            "validation_scope": (
                "fixed MTP3 TopK2048/capacity8192; disjoint/all_hits/repeated_resident/short/empty; "
                "475K final-window and continuous cold-to-hot; heterogeneous hot/cold requests"
            ),
            "implementations": list(args.implementations),
            "read_only_inputs": "passed",
            "guards": "passed",
            "exact_state": "passed",
            "continuous_state": "cold_then_hot checked without restore between invocations",
            "failure_dump": "joint_plan_common.dump_failure",
        }
        write_json(report_path, report)
        print(json.dumps({
            "status": "passed",
            "source_id": report["source_id"],
            "library_sha256": environment.get("library_sha256"),
            "report": str(report_path.resolve()),
        }, ensure_ascii=False, indent=2))
        print("===== MTP3_BITMAP_TEAM_NPU_GATE_PASSED =====")
    except BaseException as exc:
        report.update(status="failed", failure=f"{type(exc).__name__}: {exc}")
        try:
            write_json(report_path, report)
        except BaseException as secondary:
            print(f"gate report save failed: {secondary}; original error: {exc}", file=sys.stderr, flush=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_runtime_arguments(parser)
    parser.set_defaults(device=5)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--implementations", nargs="+", required=True)
    args = parser.parse_args()
    if len(args.implementations) != len(set(args.implementations)):
        parser.error("implementations must not repeat")
    for name in args.implementations:
        if name not in common.IMPLEMENTATIONS:
            parser.error(f"unknown implementation: {name}")
    run(args)


if __name__ == "__main__":
    main()
