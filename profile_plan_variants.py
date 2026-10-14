#!/usr/bin/env python3
"""Profile feat/simt-plan-micro-variants without changing or rebuilding kernels.

Run from this repository root, or pass --project /path/to/the/project.
Example:
  python3 profile_plan_variants.py --device 1 --scenario all \
      --implementations baseline a_no_pending ab abcdefgh \
      --warmup 13 --repeat 60 --output /dev/shm/plan_micro_profile

Each call restores the original state. Goldens and validation stay outside
the recorded steps. One profiler warm-up step precedes the recorded steps.
CSV statistics group actual kernel names and task types, not Event intervals.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile


def kernel_entry(comparison, name):
    return comparison.VARIANTS[name]["kernel_name"]


def summarize_csv(path, np, entries):
    """Keep kernel tasks separate; never sum mixed task types or call intervals."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        compact = {re.sub(r"\s+|_", "", key).casefold(): key for key in headers}
        name_key = next((compact[key] for key in ("name", "kernelname", "opname")
                         if key in compact), None)
        duration_key = next((compact[key] for key in
                             ("duration(us)", "duration(µs)", "duration(μs)", "durationus")
                             if key in compact), None)
        task_key = compact.get("tasktype")
        if not name_key or not duration_key:
            print("CSV fields need inspection:", headers, flush=True)
            print("Raw CSV:", path, flush=True)
            return []
        grouped = {}
        for row in reader:
            name = row.get(name_key, "")
            matches = [impl for impl, entry in entries.items()
                       if re.search(r"(?<![A-Za-z0-9_])" + re.escape(entry)
                                    + r"(?![A-Za-z0-9_])", name)]
            if not matches:
                continue
            if len(matches) != 1:
                raise RuntimeError(f"Ambiguous kernel name {name}: {matches}")
            duration = float(row[duration_key])
            if not np.isfinite(duration) or duration <= 0:
                raise RuntimeError(f"Invalid kernel duration: {row}")
            key = (matches[0], name, row.get(task_key, "") if task_key else "")
            grouped.setdefault(key, []).append(duration)
    records = []
    for (implementation, name, task), durations in sorted(grouped.items()):
        records.append({"implementation": implementation,
                        "kernel_name": name, "task_type": task, "count": len(durations),
                        "p50_us": float(np.percentile(durations, 50)),
                        "p95_us": float(np.percentile(durations, 95)),
                        "min_us": min(durations), "max_us": max(durations)})
    return records


def validate_records(records, names, repeat, reference):
    by_impl = {}
    for name in names:
        matching = [row for row in records if row["implementation"] == name]
        if len(matching) != 1 or matching[0]["count"] != repeat:
            raise RuntimeError(f"{name}: expected one kernel group with count={repeat}; "
                               f"got {matching}. Inspect raw kernel_details.csv before comparing.")
        by_impl[name] = matching[0]
    baseline = by_impl[reference]["p50_us"]
    for row in records:
        row["speedup_vs_reference_p50"] = baseline / row["p50_us"]
    return by_impl


def git_value(project, *args):
    result = subprocess.run(["git", "-C", str(project), *args],
                            text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--library", type=Path)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--capacity", type=int, default=4096)
    parser.add_argument("--max-token", type=int, default=262144)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--scenario", default="warm_75pct",
                        choices=("all", "cold_reset", "warm_75pct", "warm_all_hit", "full_cache_75pct"))
    parser.add_argument("--implementations", nargs="+")
    parser.add_argument("--shuffle-lru", action="store_true")
    parser.add_argument("--reference", default="baseline")
    parser.add_argument("--warmup", type=int, default=13)
    parser.add_argument("--repeat", type=int, default=26,
                        help="Recorded calls per implementation, excluding profiler warm-up")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not (args.rows > 0 and 0 < args.topk <= args.capacity and args.max_token > args.topk
            and args.block_size > 0 and args.warmup >= 0 and args.repeat > 0):
        parser.error("Invalid shape, warmup or repeat")
    project = args.project.resolve()
    source = project / "compare.py"
    if not source.is_file():
        parser.error(f"Missing {source}; run beside compare.py or pass --project")
    if not (project / "plan_variants_manifest.json").is_file():
        parser.error("Requires the SIMT Plan micro-variant project")
    sys.path.insert(0, str(project))
    spec = importlib.util.spec_from_file_location("kv_variant_comparison", source)
    comparison = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparison)
    names = tuple(args.implementations or comparison.IMPLEMENTATIONS)
    if len(set(names)) != len(names) or any(name not in comparison.IMPLEMENTATIONS for name in names):
        parser.error("Unknown or duplicate implementations")
    if not hasattr(comparison, "VARIANTS"):
        parser.error("Requires the SIMT Plan micro-variant project")
    if args.reference not in names:
        parser.error("--reference must be included in --implementations")
    entries = {name: kernel_entry(comparison, name) for name in names}
    if max(args.rows * args.max_token, args.rows * (args.capacity + args.topk),
           args.max_token, args.block_size) > comparison.I32_MAX:
        parser.error("Shape exceeds supported int32 indexing")
    library = (args.library or project / "build/libsimt_plan_variants_ops.so").resolve()
    if not library.is_file():
        parser.error(f"Missing shared library: {library}")
    try:
        import torch
        import torch_npu
    except ImportError as exc:
        parser.error(f"Use the existing torch/torch_npu environment: {exc}")
    if not torch.npu.is_available():
        parser.error("No NPU available")
    torch.npu.set_device(args.device)
    torch.ops.load_library(str(library))
    for name in names:
        op = comparison.VARIANTS[name]["op_name"]
        if not hasattr(torch.ops.sparse_kv_plan_op, op):
            parser.error(f"Library is missing {op}; rebuild this branch")
    plan_build = None
    if any(name in comparison.VARIANTS for name in names):
        from checks.plan_build_info import catalog_identity
        if not hasattr(torch.ops.sparse_kv_plan_op, "plan_variants_info"):
            parser.error("Library lacks plan_variants_info; rebuild this branch")
        plan_build = json.loads(torch.ops.sparse_kv_plan_op.plan_variants_info())
        if plan_build != catalog_identity():
            parser.error("Library Plan identity is stale; rebuild this branch")
    device = torch.device(f"npu:{args.device}")
    profiler = torch_npu.profiler
    config = profiler._ExperimentalConfig(profiler_level=profiler.ProfilerLevel.Level0,
                                          aic_metrics=profiler.AiCMetrics.AiCoreNone,
                                          export_type=[profiler.ExportType.Text])
    args.output.mkdir(parents=True, exist_ok=True)
    capture = Path(tempfile.mkdtemp(prefix="capture_", dir=args.output.resolve()))
    print("Capture directory:", capture, flush=True)
    print("Loaded shared library:", library, flush=True)
    scenarios = list(comparison.SCENARIOS) if args.scenario == "all" else [args.scenario]
    metadata = {"status": "running", "project": str(project), "library": str(library),
                "project_commit": git_value(project, "rev-parse", "HEAD"),
                "project_branch": git_value(project, "branch", "--show-current"),
                "project_status": git_value(project, "status", "--short"),
                "plan_build": plan_build, "kernel_entries": entries,
                "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                "compare_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "device": args.device, "device_name": torch.npu.get_device_name(args.device),
                "torch": torch.__version__, "torch_npu": torch_npu.__version__,
                "shape": {key: getattr(args, key) for key in ("rows", "topk", "capacity", "max_token", "block_size")},
                "seed": args.seed, "shuffle_lru": args.shuffle_lru, "implementations": list(names),
                "reference": args.reference,
                "warmup": args.warmup, "recorded_calls_per_implementation": args.repeat,
                "profiler_warmup_steps": 1, "scenarios": {}}
    metadata_path = capture / "capture_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    for scenario in scenarios:
        case = comparison.make_case(args.rows, args.topk, args.capacity, args.max_token,
                                    scenario, args.seed, args.block_size, args.shuffle_lru)
        case_digest = comparison.array_hash({key: value for key, value in case.items()
                                             if isinstance(value, comparison.np.ndarray)})
        simt_expected = comparison.reference_simt(case)
        expected = {name: simt_expected for name in names}
        inputs, initial, states = comparison.prepare_npu(torch, case, None, device, names)

        # Same contaminated-output audit as compare.py; never captured for timing.
        for iteration in range(3):
            for name in names:
                state = states[name]
                comparison.restore_state(torch, name, initial, state)
                keys = comparison.SIMT_OUTPUT
                for key in keys:
                    state[key].fill_(-777)
                state["workspace"].fill_(comparison.I32_MAX)
                comparison.invoke(torch, name, case, inputs, state)
                torch.npu.synchronize()
                comparison.check_npu_result(name, state, expected, f"profile-audit-{iteration}")
        for iteration in range(args.warmup):
            for name in comparison.execution_order(names, iteration):
                comparison.restore_state(torch, name, initial, states[name])
                comparison.invoke(torch, name, case, inputs, states[name])
                torch.npu.synchronize()
        for name in names:
            comparison.check_npu_result(name, states[name], expected, "profile-before-capture")

        destination = capture / scenario
        destination.mkdir()
        print(f"Profiling {scenario}: {args.repeat} recorded calls per implementation", flush=True)
        with profiler.profile(
            activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
            schedule=profiler.schedule(wait=0, warmup=1, active=args.repeat, repeat=1),
            on_trace_ready=profiler.tensorboard_trace_handler(str(destination)),
            record_shapes=False, profile_memory=False, with_stack=False,
            experimental_config=config,
        ) as prof:
            # One step contains all implementations. Step 0 warms the collector.
            for step in range(args.repeat + 1):
                for name in comparison.execution_order(names, max(0, step - 1)):
                    comparison.restore_state(torch, name, initial, states[name])
                    with torch.profiler.record_function(f"kv_variant/{scenario}/{name}"):
                        comparison.invoke(torch, name, case, inputs, states[name])
                        torch.npu.synchronize()
                prof.step()
        for name in names:
            comparison.check_npu_result(name, states[name], expected, "profile-after-capture")

        files = sorted(destination.rglob("kernel_details.csv"))
        if not files:
            raise RuntimeError(f"No kernel_details.csv; inspect profiler logs in {destination}")
        records = []
        for path in files:
            print("Raw kernel CSV:", path, flush=True)
            records.extend(summarize_csv(path, comparison.np, entries))
        if not records:
            raise RuntimeError(f"No recognized kernel rows; inspect raw CSV fields/names in {destination}")
        validate_records(records, names, args.repeat, args.reference)
        summary_path = destination / "kernel_summary.csv"
        with summary_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        for record in records:
            print(f"  {record['implementation']}: "
                  f"count={record['count']}, P50={record['p50_us']:.2f} us, "
                  f"P95={record['p95_us']:.2f} us, "
                  f"vs {args.reference}={record['speedup_vs_reference_p50']:.3f}x", flush=True)
        metadata["scenarios"][scenario] = {"correctness": "passed",
                                            "case_sha256": case_digest,
                                            "layouts": {name: states[name]["layout"] for name in names},
                                            "raw_csv": [str(path) for path in files],
                                            "kernel_stats": records,
                                            "summary_csv": str(summary_path)}
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        print("Kernel summary:", summary_path, flush=True)
    metadata["status"] = "complete"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print("Capture complete:", capture, flush=True)


if __name__ == "__main__":
    main()
