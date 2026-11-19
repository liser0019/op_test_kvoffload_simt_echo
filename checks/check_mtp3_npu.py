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
from checks.check_mtp3_candidates import IMPLEMENTATIONS, MTP3_475_IMPLEMENTATIONS, target_case
from joint_variants import MTP3_475_CONTEXTS
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
    parser.add_argument("--mtp3-475-only", action="store_true")
    parser.add_argument("--contexts", type=int, nargs="+")
    args = parser.parse_args()
    if args.audit_repeat <= 0:
        parser.error("audit-repeat must be positive")
    selected = MTP3_475_IMPLEMENTATIONS if args.mtp3_475_only else IMPLEMENTATIONS
    names = tuple(name for name, _ in selected)
    contexts = tuple(args.contexts) if args.contexts is not None else (
        MTP3_475_CONTEXTS if args.mtp3_475_only else (262144,))
    torch, torch_npu, environment = common.load_runtime(args)
    report = {"status": "running", "scope": "fixed four-row MTP3 Plan only",
              "environment": environment, "source_id": source_id(environment),
              "implementations": list(names), "contexts": list(contexts)}
    per_implementation = {}
    workspace_no_spill = {}
    try:
        for context in contexts:
            for kind in ("disjoint", "overlap40", "short", "empty"):
                for name in names:
                    high_window = name in ("joint_mtp3_dir475_512", "joint_mtp3_bitmap475_512")
                    case = target_case(kind, context, high_window=high_window)
                    case["scenario"] = "full_cache_75pct"
                    print(f"gate {context} {kind} {name}", flush=True)
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
                    label = f"{context}:{kind}"
                    per_implementation.setdefault(name, {"status": "passed", "cases": []})["cases"].append(label)
        report["status"] = "passed"
        report["workspace_no_spill"] = workspace_no_spill
        report["gate"] = {
            "status": "passed",
            "validation_scope": "fixed four-row MTP3: TopK2048, capacity8192, contexts=" + ",".join(map(str, contexts)),
            "implementations": list(names), "contexts": list(contexts),
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
