#!/usr/bin/env python3
"""Validate and summarize the official Indexer versus ordered+Plan matrix."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path


ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/official_vs_ordered_bt_plan_results").resolve()
TARGET = "ordered_bt_direct_odd"
BATCHES = range(8, 17)
CONTEXTS = (102400, 358400)


def fmt(summary):
    return f"{summary['p50_us']:.3f}/{summary['p95_us']:.3f}"


rows = []
environment = None
build_identity = None
artifact_index = {}

def digest(path):
    path = Path(path)
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()

for batch in BATCHES:
    metadata_path = ROOT / f"batch{batch}" / "capture_metadata.json"
    if not metadata_path.is_file():
        continue
    metadata = json.loads(metadata_path.read_text())
    if environment is None:
        environment = {
            key: metadata["device"].get(key)
            for key in ("device", "device_name", "hostname", "cann_package_home", "torch", "torch_npu")
        }
        environment["plan_library"] = metadata["device"].get("library")
        environment["plan_library_sha256"] = metadata["device"].get("library_sha256")
        build_identity = {
            "qli_source_id": metadata["build_manifest"].get("source_id"),
            "qli_upstream_commit": metadata["build_manifest"].get("upstream"),
            "packaged_operators": metadata["build_manifest"].get("packaged_operators"),
            "plan_source_commit": "88c06745b0260ef258ce3870945bdb7668d3d483",
        }
    artifact_index[str(metadata_path.relative_to(ROOT))] = digest(metadata_path)
    repeat = metadata.get("configuration", {}).get("repeat")
    assert repeat and int(repeat) > 0
    for case in metadata["cases"]:
        if case["context"] not in CONTEXTS or case["scenario"] != "full_cache_90pct":
            continue
        stats = case["statistics"]
        assert {"baseline", TARGET} <= set(stats), (batch, case["context"], sorted(stats))
        assert case["requests"] == batch and case["mtp"] == 3
        assert case["capacity"] == 8192 and case["context"] in CONTEXTS
        for role in ("baseline", TARGET):
            for metric in ("span_us",):
                assert stats[role][metric]["count"] == int(repeat), (
                    batch, case["context"], role, metric, stats[role][metric]["count"], repeat)
        for metric in ("indexer_span_us", "plan_us", "between_kernel_gap_us"):
            assert stats[TARGET][metric]["count"] == int(repeat), (
                batch, case["context"], TARGET, metric, stats[TARGET][metric]["count"], repeat)
        topk = stats["baseline"]["topk_set_comparison"]
        assert topk["min_sample_overlap"] == 1 and topk["min_row_overlap"] == 1
        assert topk["max_different_rows"] == 0
        target = stats[TARGET]
        checks = target["checks"]
        assert checks["joint_groups"] == batch and checks["overflow_groups"] == 0
        hit_positions = case["input"].get("initial_hit_positions_per_row", [])
        hit_fractions = case["input"].get("initial_hit_fraction_per_row", [])
        expected_hit_positions = case["input"]["topk"] * 90 // 100
        expected_hit_fraction = expected_hit_positions / case["input"]["topk"]
        assert len(hit_positions) == batch * 4
        assert all(value == expected_hit_positions for value in hit_positions)
        assert len(hit_fractions) == batch * 4
        assert all(math.isclose(value, expected_hit_fraction, rel_tol=0, abs_tol=1e-12)
                   for value in hit_fractions)
        evidence_paths = [case.get("samples_csv"), *case.get("csv_paths", [])]
        for evidence in evidence_paths:
            if not evidence:
                continue
            evidence_path = Path(evidence)
            assert evidence_path.is_file(), evidence_path
            artifact_index[str(evidence_path.relative_to(ROOT))] = digest(evidence_path)
        pair_p50 = case["official_vs_ordered_bt_plan_p50"]
        official_p95 = stats["baseline"]["span_us"]["p95_us"]
        ordered_p95 = target["indexer_span_us"]["p95_us"]
        plan_p95 = target["plan_us"]["p95_us"]
        gap_p95 = target["between_kernel_gap_us"]["p95_us"]
        chain_p95 = target["span_us"]["p95_us"]
        pair_p95 = {
            "official_indexer_us": official_p95,
            "ordered_indexer_us": ordered_p95,
            "plan_us": plan_p95,
            "between_kernel_gap_us": gap_p95,
            "chain_span_us": chain_p95,
            "ordering_cost_us": ordered_p95 - official_p95,
            "plan_chain_cost_us": chain_p95 - ordered_p95,
            "total_delta_us": chain_p95 - official_p95,
            "chain_over_official": chain_p95 / official_p95,
            "overhead_pct": (chain_p95 / official_p95 - 1) * 100,
        }
        rows.append({
            "batch": batch,
            "context": case["context"],
            "official_indexer": stats["baseline"]["span_us"],
            "ordered_indexer": target["indexer_span_us"],
            "plan": target["plan_us"],
            "gap": target["between_kernel_gap_us"],
            "chain_busy": target["busy_us"],
            "chain_span": target["span_us"],
            "official_vs_ordered_bt_plan_p50": pair_p50,
            "official_vs_ordered_bt_plan_p95": pair_p95,
            "descriptors": checks["total_transfer_descriptors"],
            "actual_hit_fraction": 0.9,
        })

expected = {(b, c) for b in BATCHES for c in CONTEXTS}
found = {(row["batch"], row["context"]) for row in rows}
missing = sorted(expected - found)
if missing:
    raise SystemExit(f"missing cases: {missing}")

harness_commit_path = ROOT / "harness-commit.txt"
payload = {
    "status": "complete",
    "target": TARGET,
    "scenario": "full_cache_90pct",
    "harness_commit": harness_commit_path.read_text().strip() if harness_commit_path.is_file() else None,
    "environment": environment,
    "build_identity": build_identity,
    "artifact_sha256": artifact_index,
    "cases": rows,
}
(ROOT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

lines = [
    "# 官方 Indexer 与 Ordered Indexer＋Plan 对比", "",
    "固定 MTP3、TopK2048、capacity8192、满缓存90%。每行初始命中1843，miss205。",
    "所有时间为设备数据，单位μs。链路 span 包含两条目标 kernel 的逐样本间隙。", "",
    "| Batch | Context | 官方 Indexer P50/P95 | Ordered Indexer P50 | Plan P50 | Gap P50 | 链路 P50/P95 | ΔTotal P50 | Ratio P50 | ΔOrdered P50 | Plan+Gap P50 | ΔTotal P95 | Ratio P95 |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['batch']} | {row['context']} | {fmt(row['official_indexer'])} | "
        f"{row['ordered_indexer']['p50_us']:.3f} | {row['plan']['p50_us']:.3f} | "
        f"{row['gap']['p50_us']:.3f} | {fmt(row['chain_span'])} | "
        f"{row['official_vs_ordered_bt_plan_p50']['total_delta_us']:+.3f} | "
        f"{row['official_vs_ordered_bt_plan_p50']['chain_over_official']:.3f} | "
        f"{row['official_vs_ordered_bt_plan_p50']['ordering_cost_us']:+.3f} | "
        f"{row['official_vs_ordered_bt_plan_p50']['plan_chain_cost_us']:+.3f} | "
        f"{row['official_vs_ordered_bt_plan_p95']['total_delta_us']:+.3f} | "
        f"{row['official_vs_ordered_bt_plan_p95']['chain_over_official']:.3f} |"
    )

for context in CONTEXTS:
    ratios_p50 = [row["official_vs_ordered_bt_plan_p50"]["chain_over_official"]
                  for row in rows if row["context"] == context]
    ratios_p95 = [row["official_vs_ordered_bt_plan_p95"]["chain_over_official"]
                  for row in rows if row["context"] == context]
    geo_p50 = math.exp(sum(math.log(x) for x in ratios_p50) / len(ratios_p50))
    geo_p95 = math.exp(sum(math.log(x) for x in ratios_p95) / len(ratios_p95))
    lines += ["", f"Context {context} 的链路/官方耗时比：P50 几何平均 {geo_p50:.3f}，范围 {min(ratios_p50):.3f}～{max(ratios_p50):.3f}；P95 几何平均 {geo_p95:.3f}，范围 {min(ratios_p95):.3f}～{max(ratios_p95):.3f}。"]

(ROOT / "report.md").write_text("\n".join(lines) + "\n")
print(ROOT / "report.md")
