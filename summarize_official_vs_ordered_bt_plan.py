#!/usr/bin/env python3
"""Validate and summarize official Indexer, ordered+Plan, and fused QLI+Plan."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path


ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/official_vs_ordered_bt_plan_results").resolve()
OFFICIAL = "baseline"
SPLIT = "ordered_bt_direct_odd"
FUSED = "peer"
IMPLEMENTATIONS = (OFFICIAL, SPLIT, FUSED)
BATCHES = range(8, 17)
CONTEXTS = (102400, 358400)


def fmt(summary):
    return f"{summary['p50_us']:.3f}/{summary['p95_us']:.3f}"


def digest(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def comparison(case):
    stats = case["statistics"]
    official = stats[OFFICIAL]["span_us"]
    ordered = stats[SPLIT]["span_us"]
    fused = stats[FUSED]["span_us"]
    return {
        "p50": {
            "official_us": official["p50_us"],
            "ordered_plan_us": ordered["p50_us"],
            "fused_us": fused["p50_us"],
            "ordered_plan_delta_us": ordered["p50_us"] - official["p50_us"],
            "fused_delta_us": fused["p50_us"] - official["p50_us"],
            "ordered_plan_over_official": ordered["p50_us"] / official["p50_us"],
            "fused_over_official": fused["p50_us"] / official["p50_us"],
            "ordered_plan_over_fused": ordered["p50_us"] / fused["p50_us"],
            "ordered_plan_overhead_pct": (ordered["p50_us"] / official["p50_us"] - 1) * 100,
            "fused_overhead_pct": (fused["p50_us"] / official["p50_us"] - 1) * 100,
        },
        "p95": {
            "official_us": official["p95_us"],
            "ordered_plan_us": ordered["p95_us"],
            "fused_us": fused["p95_us"],
            "ordered_plan_delta_us": ordered["p95_us"] - official["p95_us"],
            "fused_delta_us": fused["p95_us"] - official["p95_us"],
            "ordered_plan_over_official": ordered["p95_us"] / official["p95_us"],
            "fused_over_official": fused["p95_us"] / official["p95_us"],
            "ordered_plan_over_fused": ordered["p95_us"] / fused["p95_us"],
            "ordered_plan_overhead_pct": (ordered["p95_us"] / official["p95_us"] - 1) * 100,
            "fused_overhead_pct": (fused["p95_us"] / official["p95_us"] - 1) * 100,
        },
    }


rows = []
environment = None
build_identity = None
artifact_index = {}

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
        assert set(IMPLEMENTATIONS) <= set(stats), (
            batch, case["context"], sorted(stats), IMPLEMENTATIONS)
        assert case["requests"] == batch and case["mtp"] == 3
        assert case["capacity"] == 8192 and case["context"] in CONTEXTS
        for role in IMPLEMENTATIONS:
            assert stats[role]["span_us"]["count"] == int(repeat), (
                batch, case["context"], role, stats[role]["span_us"]["count"], repeat)
        for metric in ("indexer_span_us", "plan_us", "between_kernel_gap_us"):
            assert stats[SPLIT][metric]["count"] == int(repeat), (
                batch, case["context"], SPLIT, metric, stats[SPLIT][metric]["count"], repeat)
        topk = stats[OFFICIAL]["topk_set_comparison"]
        assert topk["min_sample_overlap"] == 1 and topk["min_row_overlap"] == 1
        assert topk["max_different_rows"] == 0
        for role in (SPLIT, FUSED):
            checks = stats[role]["checks"]
            assert checks["joint_groups"] == batch and checks["overflow_groups"] == 0, (
                batch, case["context"], role, checks)
        hit_positions = case["input"].get("initial_hit_positions_per_row", [])
        hit_fractions = case["input"].get("initial_hit_fraction_per_row", [])
        expected_hit_positions = case["input"]["topk"] * 90 // 100
        expected_hit_fraction = expected_hit_positions / case["input"]["topk"]
        assert len(hit_positions) == batch * 4
        assert all(value == expected_hit_positions for value in hit_positions)
        assert len(hit_fractions) == batch * 4
        assert all(math.isclose(value, expected_hit_fraction, rel_tol=0, abs_tol=1e-12)
                   for value in hit_fractions)
        for evidence in [case.get("samples_csv"), *case.get("csv_paths", [])]:
            if not evidence:
                continue
            evidence_path = Path(evidence)
            assert evidence_path.is_file(), evidence_path
            artifact_index[str(evidence_path.relative_to(ROOT))] = digest(evidence_path)
        ordered = stats[SPLIT]
        fused = stats[FUSED]
        rows.append({
            "batch": batch,
            "context": case["context"],
            "official_indexer": stats[OFFICIAL]["span_us"],
            "ordered_plan": {
                "indexer": ordered["indexer_span_us"],
                "plan": ordered["plan_us"],
                "gap": ordered["between_kernel_gap_us"],
                "busy": ordered["busy_us"],
                "span": ordered["span_us"],
                "checks": ordered["checks"],
            },
            "fused_ordered_plan": {
                "busy": fused["busy_us"],
                "span": fused["span_us"],
                "checks": fused["checks"],
            },
            "comparison": comparison(case),
            "actual_hit_fraction": expected_hit_fraction,
        })

expected = {(b, c) for b in BATCHES for c in CONTEXTS}
found = {(row["batch"], row["context"]) for row in rows}
missing = sorted(expected - found)
if missing:
    raise SystemExit(f"missing cases: {missing}")

harness_commit_path = ROOT / "harness-commit.txt"
payload = {
    "status": "complete",
    "implementations": {
        OFFICIAL: "quant_lightning_indexer",
        SPLIT: "quant_lightning_indexer_ordered + joint_mtp3_bt_direct_odd_512",
        FUSED: "quant_lightning_indexer_ordered_fused",
    },
    "scenario": "full_cache_90pct",
    "harness_commit": harness_commit_path.read_text().strip() if harness_commit_path.is_file() else None,
    "environment": environment,
    "build_identity": build_identity,
    "artifact_sha256": artifact_index,
    "cases": rows,
}
(ROOT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

lines = [
    "# Official Indexer、Ordered＋Plan 与 Fused Ordered QLI＋Plan 三方对照", "",
    "固定 MTP3、TopK2048、capacity8192、满缓存90%。每行初始命中1843，miss205。",
    "所有时间为设备数据，单位μs。Ordered＋Plan 链路 span 包含两 kernel 间隙；Fused 为单次融合算子调用。", "",
    "| Batch | Context | Official P50/P95 | Ordered+Plan P50/P95 | Fused P50/P95 | Ordered/Official P50/P95 | Fused/Official P50/P95 | Ordered/Fused P50/P95 |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    c = row["comparison"]
    lines.append(
        f"| {row['batch']} | {row['context']} | {fmt(row['official_indexer'])} | "
        f"{fmt(row['ordered_plan']['span'])} | {fmt(row['fused_ordered_plan']['span'])} | "
        f"{c['p50']['ordered_plan_over_official']:.3f}/{c['p95']['ordered_plan_over_official']:.3f} | "
        f"{c['p50']['fused_over_official']:.3f}/{c['p95']['fused_over_official']:.3f} | "
        f"{c['p50']['ordered_plan_over_fused']:.3f}/{c['p95']['ordered_plan_over_fused']:.3f} |"
    )

for context in CONTEXTS:
    subset = [row for row in rows if row["context"] == context]
    def geo(metric, ratio):
        vals = [row["comparison"][metric][ratio] for row in subset]
        return math.exp(sum(math.log(x) for x in vals) / len(vals)), min(vals), max(vals)
    ops50, ops50lo, ops50hi = geo("p50", "ordered_plan_over_official")
    fps50, fps50lo, fps50hi = geo("p50", "fused_over_official")
    sf50, sf50lo, sf50hi = geo("p50", "ordered_plan_over_fused")
    ops95, ops95lo, ops95hi = geo("p95", "ordered_plan_over_official")
    fps95, fps95lo, fps95hi = geo("p95", "fused_over_official")
    sf95, sf95lo, sf95hi = geo("p95", "ordered_plan_over_fused")
    lines += [
        "",
        f"Context {context}：",
        f"- P50 Ordered/Official 几何平均 {ops50:.3f}（{ops50lo:.3f}～{ops50hi:.3f}），Fused/Official {fps50:.3f}（{fps50lo:.3f}～{fps50hi:.3f}），Ordered/Fused {sf50:.3f}（{sf50lo:.3f}～{sf50hi:.3f}）。",
        f"- P95 Ordered/Official 几何平均 {ops95:.3f}（{ops95lo:.3f}～{ops95hi:.3f}），Fused/Official {fps95:.3f}（{fps95lo:.3f}～{fps95hi:.3f}），Ordered/Fused {sf95:.3f}（{sf95lo:.3f}～{sf95hi:.3f}）。",
    ]

(ROOT / "report.md").write_text("\n".join(lines) + "\n")
print(ROOT / "report.md")
