#!/usr/bin/env python3
"""NPU correctness gate for the fixed MTP3 candidates; emits a reusable gate report."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joint_plan_common as common
from checks.check_mtp3_candidates import IMPLEMENTATIONS, target_case
from harness_io import write_json


def source_id(environment):
    payload = json.dumps(environment.get("identity"), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_runtime_arguments(parser)
    parser.set_defaults(device=5)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--audit-repeat", type=int, default=3)
    args = parser.parse_args()
    if args.audit_repeat <= 0:
        parser.error("audit-repeat must be positive")
    names = tuple(name for name, _ in IMPLEMENTATIONS)
    torch, torch_npu, environment = common.load_runtime(args)
    report = {"status": "running", "scope": "fixed MTP3 target shape, Plan only",
              "environment": environment, "source_id": source_id(environment),
              "implementations": list(names)}
    per_implementation = {}
    workspace_no_spill = {}
    try:
        for kind in ("disjoint", "overlap40", "short", "empty", "bounded100k", "bounded150k", "fallback360k"):
            case = target_case(kind)
            case["scenario"] = "full_cache_75pct"
            for name in names:
                print(f"gate {kind} {name}", flush=True)
                bundle = common.prepare(torch, case, name)
                try:
                    common.audit(torch, case, bundle, repeats=args.audit_repeat,
                                 failure_dir=args.report.parent / "failure_state")
                    workspace_no_spill[name] = None
                    if name == "joint_mtp3_map_512":
                        common.restore(torch, bundle)
                        torch.npu.synchronize()
                        before = bundle["buffers"]["workspace"].clone()
                        common.invoke(bundle)
                        torch.npu.synchronize()
                        workspace_no_spill[name] = bool(torch.equal(before, bundle["buffers"]["workspace"]))
                        if not workspace_no_spill[name]:
                            raise AssertionError("hot UB map spilled to compact workspace")
                finally:
                    common.close(bundle)
                per_implementation.setdefault(name, {"status": "passed", "cases": []})["cases"].append(kind)
        report["status"] = "passed"
        report["workspace_no_spill"] = workspace_no_spill
        report["gate"] = {
            "status": "passed",
            "validation_scope": "fixed MTP3 target shape: 4 rows x TopK2048, capacity8192, maxToken<=524288 (360K uses binary fallback)",
            "implementations": list(names),
            "input_write_regression": {
                "status": "passed",
                "input_restore": False,
                "checked_iterations": args.audit_repeat,
                "per_implementation": per_implementation,
            },
        }
        write_json(args.report, report)
        print(json.dumps({"status": "passed", "source_id": report["source_id"],
                          "library_sha256": environment.get("library_sha256"),
                          "report": str(args.report.resolve())}, ensure_ascii=False, indent=2))
        print("===== MTP3_NPU_GATE_PASSED =====")
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        write_json(args.report, report)
        raise


if __name__ == "__main__":
    main()
