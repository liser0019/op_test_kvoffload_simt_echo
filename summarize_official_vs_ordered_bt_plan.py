#!/usr/bin/env python3
"""Validate and summarize the official Indexer versus ordered+Plan matrix."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path


ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/official_vs_ordered_bt_plan_results")
TARGET = "ordered_bt_direct_odd"
BATCHES = range(8, 17)
CONTEXTS = (102400, 358400)


def fmt(summary):
    return f"{summary['p50_us']:.3f}/{summary['p95_us']:.3f}"


rows = []
for batch in BATCHES:
    metadata_path = ROOT / f"batch{batch}" / "capture_metadata.json"
    if not metadata_path.is_file():
        continue
    metadata = json.loads(metadata_path.read_text())
    for case in metadata["cases"]:
        if case["context"] not in CONTEXTS or case["scenario"] != "full_cache_90pct":
            continue
        stats = case["statistics"]
        assert {"baseline", TARGET} <= set(stats), (batch, case["context"], sorted(stats))
        topk = stats["baseline"]["topk_set_comparison"]
        assert topk["min_sample_overlap"] == 1 and topk["min_row_overlap"] == 1
        assert topk["max_different_rows"] == 0
        target = stats[TARGET]
        checks = target["checks"]
        assert checks["joint_groups"] == batch and checks["overflow_groups"] == 0
        hit = case["input"].get("initial_hit_fraction_per_row", [])
        assert len(hit) == batch * 4 and min(hit) == 0.9 and max(hit) == 0.9
        pair = case["official_vs_ordered_bt_plan_p50"]
        rows.append({
            "batch": batch,
            "context": case["context"],
            "official_indexer": stats["baseline"]["span_us"],
            "ordered_indexer": target["indexer_span_us"],
            "plan": target["plan_us"],
            "gap": target["between_kernel_gap_us"],
            "chain_busy": target["busy_us"],
            "chain_span": target["span_us"],
            "ordering_cost_us": pair["ordering_cost_us"],
            "plan_chain_cost_us": pair["plan_chain_cost_us"],
            "total_delta_us": pair["total_delta_us"],
            "chain_over_official": pair["chain_over_official"],
            "descriptors": checks["total_transfer_descriptors"],
            "actual_hit_fraction": 0.9,
        })

expected = {(b, c) for b in BATCHES for c in CONTEXTS}
found = {(row["batch"], row["context"]) for row in rows}
missing = sorted(expected - found)
if missing:
    raise SystemExit(f"missing cases: {missing}")

payload = {"status": "complete", "target": TARGET, "scenario": "full_cache_90pct", "cases": rows}
(ROOT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

lines = [
    "# 官方 Indexer 与 Ordered Indexer＋Plan 对比", "",
    "固定 MTP3、TopK2048、capacity8192、满缓存90%。每行初始命中1843，miss205。",
    "所有时间为设备数据，单位μs。链路 span 包含两条目标 kernel 的逐样本间隙。", "",
    "| Batch | Context | 官方 Indexer P50/P95 | Ordered Indexer P50 | Plan P50 | Gap P50 | 链路 P50/P95 | 总增量 | 链路/官方 | 保序增量 | Plan+Gap |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['batch']} | {row['context']} | {fmt(row['official_indexer'])} | "
        f"{row['ordered_indexer']['p50_us']:.3f} | {row['plan']['p50_us']:.3f} | "
        f"{row['gap']['p50_us']:.3f} | {fmt(row['chain_span'])} | "
        f"{row['total_delta_us']:+.3f} | {row['chain_over_official']:.3f} | "
        f"{row['ordering_cost_us']:+.3f} | {row['plan_chain_cost_us']:+.3f} |"
    )

for context in CONTEXTS:
    ratios = [row["chain_over_official"] for row in rows if row["context"] == context]
    geo = math.exp(sum(math.log(x) for x in ratios) / len(ratios))
    lines += ["", f"Context {context} 的链路/官方耗时比几何平均为 {geo:.3f}，范围 {min(ratios):.3f}～{max(ratios):.3f}。"]

(ROOT / "report.md").write_text("\n".join(lines) + "\n")
print(ROOT / "report.md")
