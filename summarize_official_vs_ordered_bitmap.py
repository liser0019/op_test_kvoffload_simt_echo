#!/usr/bin/env python3
"""Validate and summarize official Indexer, bitmap Plan, and legacy BT Plan."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path


ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/official_vs_ordered_bitmap_results").resolve()
OFFICIAL = "baseline"
BITMAP = "ordered_bitmap512"
LEGACY = "ordered_bt_direct_odd"
IMPLEMENTATIONS = (OFFICIAL, BITMAP, LEGACY)
BATCHES = range(8, 17)
CONTEXTS = (102400, 358400)
EXPECTED_HITS = 1843
EXPECTED_MISSES = 205


def fmt(summary):
    return f"{summary['p50_us']:.3f}/{summary['p95_us']:.3f}"


def digest(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def ratio(numerator, denominator):
    return numerator / denominator


def geo(values):
    return math.exp(sum(math.log(value) for value in values) / len(values))


def split_metrics(stats, role, official):
    value = stats[role]
    chain = value["span_us"]
    return {
        "indexer": value["indexer_span_us"],
        "plan": value["plan_us"],
        "gap": value["between_kernel_gap_us"],
        "busy": value["busy_us"],
        "span": chain,
        "delta_vs_official_p50": chain["p50_us"] - official["p50_us"],
        "delta_vs_official_p95": chain["p95_us"] - official["p95_us"],
        "ratio_vs_official_p50": ratio(chain["p50_us"], official["p50_us"]),
        "ratio_vs_official_p95": ratio(chain["p95_us"], official["p95_us"]),
        "checks": value["checks"],
    }


rows = []
environment = None
build_identity = None
artifact_index = {}
kernel_inventory = set()
plan_library_sha256 = None

complete_path = ROOT / "MATRIX_COMPLETE.json"
if not complete_path.is_file():
    raise SystemExit("missing MATRIX_COMPLETE.json")
complete = json.loads(complete_path.read_text())
if complete.get("status") != "complete":
    raise SystemExit("matrix is not complete")

for batch in BATCHES:
    metadata_path = ROOT / f"batch{batch}" / "capture_metadata.json"
    if not metadata_path.is_file():
        continue
    metadata = json.loads(metadata_path.read_text())
    repeat = int(metadata.get("configuration", {}).get("repeat", 0))
    if repeat != 20:
        raise SystemExit(f"batch{batch}: expected 20 formal repeats, got {repeat}")
    if environment is None:
        environment = {
            key: metadata["device"].get(key)
            for key in ("device", "device_name", "hostname", "cann_package_home", "torch", "torch_npu")
        }
        environment["plan_library"] = metadata["device"].get("library")
        environment["plan_library_sha256"] = metadata["device"].get("library_sha256")
        environment["qli_s1_base_by_implementation"] = {}
        source = metadata["device"].get("identity", {}).get("source", {})
        bitmap_source = source.get("ordered_bitmap_512", {})
        if bitmap_source.get("source_commit") != "ba4c5dd9bb6aaa432abf4822607ce224bec51862":
            raise SystemExit("bitmap Plan source commit mismatch")
        if bitmap_source.get("dynamic_ub_bytes") != 181024:
            raise SystemExit("bitmap Plan dynamic UB identity mismatch")
        build_identity = {
            "qli_source_id": metadata["build_manifest"].get("source_id"),
            "qli_upstream_commit": metadata["build_manifest"].get("upstream"),
            "packaged_operators": metadata["build_manifest"].get("packaged_operators"),
            "ordered_bitmap_plan_source_commit": bitmap_source.get("source_commit"),
            "ordered_bitmap_plan_kernel": bitmap_source.get("kernel"),
            "ordered_bitmap_plan_dynamic_ub_bytes": bitmap_source.get("dynamic_ub_bytes"),
        }
        plan_library_sha256 = environment["plan_library_sha256"]
    artifact_index[str(metadata_path.relative_to(ROOT))] = digest(metadata_path)
    environment["qli_s1_base_by_implementation"].update(metadata.get("qli_s1_base_by_implementation", {}))
    kernel_inventory.update(metadata.get("kernel_inventory") or [])
    for case in metadata["cases"]:
        kernel_inventory.update(case.get("kernel_inventory") or [])
        if case["context"] not in CONTEXTS or case["scenario"] != "full_cache_90pct":
            continue
        stats = case["statistics"]
        if not set(IMPLEMENTATIONS) <= set(stats):
            raise SystemExit(f"batch{batch}/ctx{case['context']}: missing roles {sorted(set(IMPLEMENTATIONS)-set(stats))}")
        if case["requests"] != batch or case["mtp"] != 3 or case["capacity"] != 8192:
            raise SystemExit(f"unexpected case shape: {case['tag']}")
        for role in IMPLEMENTATIONS:
            count = stats[role]["span_us"]["count"]
            if count != repeat:
                raise SystemExit(f"{case['tag']}/{role}: {count} samples, expected {repeat}")
        for role in (BITMAP, LEGACY):
            for metric in ("indexer_span_us", "plan_us", "between_kernel_gap_us"):
                count = stats[role][metric]["count"]
                if count != repeat:
                    raise SystemExit(f"{case['tag']}/{role}/{metric}: {count} samples")

        topk = stats[OFFICIAL]["topk_set_comparison"]
        if topk["min_sample_overlap"] != 1 or topk["min_row_overlap"] != 1 or topk["max_different_rows"] != 0:
            raise SystemExit(f"{case['tag']}: baseline TopK set differs from ordered Indexer")

        for role in (BITMAP, LEGACY):
            checks = stats[role]["checks"]
            if checks["joint_groups"] != batch or checks["overflow_groups"] != 0:
                raise SystemExit(f"{case['tag']}/{role}: joint/fallback={checks['joint_groups']}/{checks['overflow_groups']}")

        hit_positions = case["input"].get("initial_hit_positions_per_row", [])
        hit_fractions = case["input"].get("initial_hit_fraction_per_row", [])
        if len(hit_positions) != batch * 4 or any(value != EXPECTED_HITS for value in hit_positions):
            raise SystemExit(f"{case['tag']}: 90% hit positions are not exactly {EXPECTED_HITS}/row")
        expected_fraction = EXPECTED_HITS / case["input"]["topk"]
        if len(hit_fractions) != batch * 4 or any(
                not math.isclose(value, expected_fraction, rel_tol=0, abs_tol=1e-12)
                for value in hit_fractions):
            raise SystemExit(f"{case['tag']}: 90% hit fractions are inconsistent")

        for evidence in [case.get("samples_csv"), *case.get("csv_paths", [])]:
            if evidence:
                evidence_path = Path(evidence)
                if not evidence_path.is_file():
                    raise SystemExit(f"missing evidence: {evidence_path}")
                artifact_index[str(evidence_path.relative_to(ROOT))] = digest(evidence_path)

        official = stats[OFFICIAL]["span_us"]
        bitmap = split_metrics(stats, BITMAP, official)
        legacy = split_metrics(stats, LEGACY, official)
        rows.append({
            "batch": batch,
            "context": case["context"],
            "official_indexer": official,
            "ordered_bitmap_plan": bitmap,
            "ordered_legacy_bt_plan": legacy,
            "bitmap_vs_legacy": {
                "span_delta_p50": bitmap["span"]["p50_us"] - legacy["span"]["p50_us"],
                "span_delta_p95": bitmap["span"]["p95_us"] - legacy["span"]["p95_us"],
                "span_ratio_p50": bitmap["span"]["p50_us"] / legacy["span"]["p50_us"],
                "span_ratio_p95": bitmap["span"]["p95_us"] / legacy["span"]["p95_us"],
                "plan_delta_p50": bitmap["plan"]["p50_us"] - legacy["plan"]["p50_us"],
                "plan_delta_p95": bitmap["plan"]["p95_us"] - legacy["plan"]["p95_us"],
            },
            "actual_hit_positions_per_row": EXPECTED_HITS,
            "actual_miss_positions_per_row": EXPECTED_MISSES,
        })

expected = {(batch, context) for batch in BATCHES for context in CONTEXTS}
found = {(row["batch"], row["context"]) for row in rows}
missing = sorted(expected - found)
if missing:
    raise SystemExit(f"missing cases: {missing}")

bitmap_kernel = "sparse_kv_plan_joint_kernel_ordered_bitmap_512"
if bitmap_kernel not in kernel_inventory:
    raise SystemExit(f"target kernel did not appear in profiling inventory: {bitmap_kernel}")

ordered_cases = [
    {"tag": f"batch{row['batch']}/context{row['context']}", "metadata": f"batch{row['batch']}/capture_metadata.json"}
    for row in rows
]
payload = {
    "status": "complete",
    "implementations": {
        OFFICIAL: "quant_lightning_indexer",
        BITMAP: "quant_lightning_indexer_ordered + joint_ordered_bitmap_512",
        LEGACY: "quant_lightning_indexer_ordered + joint_mtp3_bt_direct_odd_512",
    },
    "scenario": "full_cache_90pct",
    "scenario_definition": {"hits_per_row": EXPECTED_HITS, "misses_per_row": EXPECTED_MISSES, "topk": 2048},
    "harness_commit": (ROOT / "harness-commit.txt").read_text().strip() if (ROOT / "harness-commit.txt").is_file() else None,
    "environment": environment,
    "build_identity": build_identity,
    "plan_library_sha256": plan_library_sha256,
    "kernel_inventory": sorted(kernel_inventory),
    "artifact_sha256": artifact_index,
    "cases": rows,
}
(ROOT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
(ROOT / "data_index.json").write_text(json.dumps({
    "status": "complete",
    "cases": ordered_cases,
    "artifacts": artifact_index,
}, ensure_ascii=False, indent=2) + "\n")

lines = [
    "# Official Indexer 与新 Bitmap Plan 同场对照", "",
    "固定 MTP3、TopK2048、capacity8192、`full_cache_90pct`。每个 query row 初始恰好 1843 个 resident hit、205 个 miss。",
    "单位 μs；`chain` 是同 stream 上 ordered Indexer → 独立 Plan 的 span，包含两 kernel 间偶发空闲。",
    "Official 只包含原始 Indexer；其余两路都包含保序 Indexer、独立 Plan 和两 kernel 间隙。",
    "所有数据是设备 profiling 的 kernel Duration/时间戳；未测 Transfer、SFA 或端到端框架开销。", "",
    "## 整链 P50/P95", "",
    "| Batch | Context | Official | Bitmap chain | Legacy BT chain | Bitmap/Official | ΔBitmap-Official | Overhead | New vs Old |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    official = row["official_indexer"]
    bitmap = row["ordered_bitmap_plan"]
    legacy = row["ordered_legacy_bt_plan"]
    lines.append(
        f"| {row['batch']} | {row['context']} | {fmt(official)} | {fmt(bitmap['span'])} | {fmt(legacy['span'])} | "
        f"{bitmap['ratio_vs_official_p50']:.3f}/{bitmap['ratio_vs_official_p95']:.3f} | "
        f"{bitmap['delta_vs_official_p50']:+.3f}/{bitmap['delta_vs_official_p95']:+.3f} | "
        f"{(bitmap['ratio_vs_official_p50']-1)*100:+.2f}%/{(bitmap['ratio_vs_official_p95']-1)*100:+.2f}% | "
        f"{row['bitmap_vs_legacy']['span_delta_p50']:+.3f}/{row['bitmap_vs_legacy']['span_delta_p95']:+.3f} |"
    )

lines += ["", "## Bitmap Plan 分段", "",
          "| Batch | Context | Ordered Indexer | Plan kernel | Kernel gap | Chain | Joint/Fallback | Descriptor | Old Plan | Δ Plan New-Old |",
          "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
for row in rows:
    value = row["ordered_bitmap_plan"]
    old = row["ordered_legacy_bt_plan"]
    checks = value["checks"]
    lines.append(
        f"| {row['batch']} | {row['context']} | {fmt(value['indexer'])} | {fmt(value['plan'])} | "
        f"{fmt(value['gap'])} | {fmt(value['span'])} | {checks['joint_groups']}/{checks['overflow_groups']} | "
        f"{checks['total_transfer_descriptors']} | {fmt(old['plan'])} | "
        f"{row['bitmap_vs_legacy']['plan_delta_p50']:+.3f}/{row['bitmap_vs_legacy']['plan_delta_p95']:+.3f} |"
    )

lines += ["", "## 分 context 几何平均", ""]
for context in CONTEXTS:
    subset = [row for row in rows if row["context"] == context]
    values = {
        "bitmap_chain_p50": [row["ordered_bitmap_plan"]["ratio_vs_official_p50"] for row in subset],
        "bitmap_chain_p95": [row["ordered_bitmap_plan"]["ratio_vs_official_p95"] for row in subset],
        "bitmap_plan_vs_legacy_p50": [row["ordered_bitmap_plan"]["plan"]["p50_us"] / row["ordered_legacy_bt_plan"]["plan"]["p50_us"] for row in subset],
        "bitmap_plan_vs_legacy_p95": [row["ordered_bitmap_plan"]["plan"]["p95_us"] / row["ordered_legacy_bt_plan"]["plan"]["p95_us"] for row in subset],
    }
    lines += [
        f"Context {context}:",
        f"- Bitmap chain / Official：P50 {geo(values['bitmap_chain_p50']):.3f}，P95 {geo(values['bitmap_chain_p95']):.3f}。",
        f"- Bitmap Plan / Old BT Plan：P50 {geo(values['bitmap_plan_vs_legacy_p50']):.3f}，P95 {geo(values['bitmap_plan_vs_legacy_p95']):.3f}。",
    ]

renderer = [row["ordered_bitmap_plan"]["indexer"]["p50_us"] - row["official_indexer"]["p50_us"] for row in rows]
lines += [
    "", "## 解释边界", "",
    f"- 保序 Indexer 相对 Official 的 P50 差值范围为 {min(renderer):+.3f}～{max(renderer):+.3f} μs；因此整链开销不能全部归因于 Plan。",
    "- 新 Bitmap 与旧 BT 在本轮同库、同输入、同设备运行；它们之间的 Plan 差值才是本轮最直接的新旧算法对照。",
    "- `joint/fallback` 对所有 18 个工况均为 batch/0，说明测试输入实际进入联合路径。",
    f"- 目标 kernel `{bitmap_kernel}` 已出现在本轮 profiling inventory。",
    f"- Plan 库 SHA256：`{plan_library_sha256}`。",
]

(ROOT / "report.md").write_text("\n".join(lines) + "\n")
print(ROOT / "report.md")
