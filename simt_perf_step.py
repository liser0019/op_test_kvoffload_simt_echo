#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""ABCDF 分阶段诊断：每 row、线程 0 的绝对 clock 低 32 位。

SIMT_PERF_STEP=ON bash build.sh
python3 simt_perf_step.py --device 1 --scenario all --output results/simt_perf

不增加同步，不把逐阶段 max 的和冒充 kernel 总耗时。默认只报 cycles；
--freq-ghz 是使用者显式提供的换算频率，不用 Event/P50 拟合频率。
正式整 kernel 对比继续使用普通构建 + profile_kernel.sh。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from html import escape
import json
from pathlib import Path
import tempfile

import numpy as np
import compare as bench

ROOT = Path(__file__).resolve().parent
SENTINEL = -12345
STAMP_COUNT = bench.SIMT_NUM_STAMPS
PATH_SLOTS = {"warm": (0, 1, 2, 3, 4, 5, 6, 7), "cold": (0, 1, 8, 9, 10, 11)}
STAGE_NAMES = {
    "warm": ("Prepare/DecideReset", "HashInit/SourceContext", "InsertTopK",
             "ClassifyLRU/InvalidateSuffix", "ResolveHit/CompactMiss",
             "AssignMiss/ClearTail", "WriteLRU/Finalize"),
    "cold": ("Prepare/DecideReset", "ClearResidents", "Validate/CompactMiss",
             "AssignMiss/ClearTail", "WriteLRU/Finalize"),
}
STAGE_ZH = {
    "Prepare/DecideReset": "初始化 current_slots、判断 request reset",
    "HashInit/SourceContext": "初始化 hash，准备来源上下文",
    "InsertTopK": "合法 TopK 插入 hash",
    "ClassifyLRU/InvalidateSuffix": "按旧 LRU 分类 hit/evict，失效投机后缀",
    "ResolveHit/CompactMiss": "F 融合：解析 hit 与稳定压缩 miss",
    "AssignMiss/ClearTail": "分配 miss、回写输出、清理无效尾部",
    "WriteLRU/Finalize": "更新 LRU 与 miss_count，包含原有末尾同步",
    "ClearResidents": "冷路径清空 resident",
    "Validate/CompactMiss": "冷路径合法性检查与 miss 压缩",
}


def expected_paths(case):
    """与 kernel 的 ID 判断一致；支持同一次调用冷/热 row 混合。"""
    paths = []
    for row, request in enumerate(case["token_to_req"]):
        req_id = case["req_ids"][request] if 0 <= request < len(case["req_ids"]) else -1
        paths.append("cold" if case["last_req_ids"][row] != req_id else "warm")
    return paths


def decode_stamps(matrix, paths, active_rows=None):
    """只对同一 row/同一线程的 clock 作差，不比较不同核的绝对 clock。

    int32 高位可以为负。uint32 模差处理一次回绕，单区间/跨度必须 <2^31。
    冷 return 后的 warm 槽、未处理的 row 必须保持哨兵，避免误解路径。
    """
    matrix = np.asarray(matrix)
    if matrix.dtype != np.int32 or matrix.shape != (len(paths), STAMP_COUNT):
        raise RuntimeError("打点矩阵必须为 int32 [rows,12]")
    active = len(paths) if active_rows is None else max(0, min(len(paths), active_rows))
    decoded = []
    for row, path in enumerate(paths):
        if row >= active:
            if np.any(matrix[row] != SENTINEL):
                raise RuntimeError(f"inactive row {row} 的打点被改写")
            continue
        slots = PATH_SLOTS[path]
        unused = [i for i in range(STAMP_COUNT) if i not in slots]
        if np.any(matrix[row, list(slots)] == SENTINEL):
            raise RuntimeError(f"row {row}/{path}: 有效槽未写入，请检查打点构建或提前返回")
        if np.any(matrix[row, unused] != SENTINEL):
            raise RuntimeError(f"row {row}/{path}: 不应执行的路径槽被写入")
        u = np.ascontiguousarray(matrix[row, list(slots)]).view(np.uint32).astype(np.uint64)
        durations = ((u[1:] - u[:-1]) & np.uint64(0xffffffff)).astype(np.int64)
        span = int((int(u[-1]) - int(u[0])) & 0xffffffff)
        if np.any(durations >= 1 << 31) or span >= 1 << 31 or int(durations.sum()) != span:
            raise RuntimeError(f"row {row}: 时间戳次序/布局异常，或区间过长")
        decoded.append({"row": row, "path": path, "cycles": durations, "span": span})
    return decoded


def aggregate_sample(matrix, decoded, blocks):
    """row 按 row%grid 分核。多行时先在各 block 内累加，再取 block max。

    block_span 从该 block 首 row entry 到末 row end，包含行间间隙。
    不含 VF 启动之前的 kernel Init，不等同 profiler 的整 kernel 时长。
    """
    if blocks <= 0:
        raise ValueError("blocks must be positive")
    block_rows = {}
    for item in decoded:
        block_rows.setdefault(item["row"] % blocks, []).append(item)
    block_spans = []
    for block, items in sorted(block_rows.items()):
        items.sort(key=lambda item: item["row"])
        first, last = items[0], items[-1]
        start = int(matrix[first["row"], 0]) & 0xffffffff
        end = int(matrix[last["row"], PATH_SLOTS[last["path"]][-1]]) & 0xffffffff
        span = (end - start) & 0xffffffff
        if span >= 1 << 31 or span < sum(item["span"] for item in items):
            raise RuntimeError(f"block {block}: 跨行跨度异常")
        block_spans.append({"block": block, "cycles": span})
    groups = {}
    for path in PATH_SLOTS:
        items = [item for item in decoded if item["path"] == path]
        if not items:
            continue
        per_block = {}
        for item in items:
            b = item["row"] % blocks
            per_block.setdefault(b, np.zeros(len(STAGE_NAMES[path]), np.int64))
            per_block[b] += item["cycles"]
        values = np.stack(list(per_block.values()))
        row_values = np.stack([item["cycles"] for item in items])
        groups[path] = {"stage_max_block_cycles": values.max(axis=0).tolist(),
                        "stage_skew_cycles": (values.max(axis=0) - values.min(axis=0)).tolist(),
                        "stage_mean_row_cycles": row_values.mean(axis=0).tolist(),
                        "stage_total_row_cycles": row_values.sum(axis=0).tolist(),
                        "rows": len(items), "owning_blocks": len(per_block)}
    return {"groups": groups, "block_spans": block_spans,
            "max_block_span_cycles": max((item["cycles"] for item in block_spans), default=0)}


def summarize(samples, freq_ghz=None):
    records = []
    for path in PATH_SLOTS:
        groups = [s["groups"][path] for s in samples if path in s["groups"]]
        if not groups:
            continue
        peaks = np.array([g["stage_max_block_cycles"] for g in groups])
        skew = np.array([g["stage_skew_cycles"] for g in groups])
        means = np.array([g["stage_mean_row_cycles"] for g in groups])
        totals = np.array([g["stage_total_row_cycles"] for g in groups]).sum(axis=0)
        denominator = int(totals.sum())
        for i, name in enumerate(STAGE_NAMES[path]):
            peak = float(np.median(peaks[:, i]))
            records.append({"path": path, "stage": name, "description": STAGE_ZH[name],
                            "p50_max_block_cycles": peak,
                            "p95_max_block_cycles": float(np.percentile(peaks[:, i], 95)),
                            "p50_mean_row_cycles": float(np.median(means[:, i])),
                            "p50_block_skew_cycles": float(np.median(skew[:, i])),
                            "share_of_all_row_cycles_pct": (float(totals[i]) / denominator * 100
                                                             if denominator else 0.0),
                            "est_max_block_us": peak / (freq_ghz * 1000) if freq_ghz else None})
    return records


def read_matrix(state, rows):
    offset = state["layout"]["stamp_offset"]
    stride = state["layout"]["stamp_row_stride"]
    matrix = state["workspace"][offset:offset + rows * stride].cpu().numpy().copy().reshape(rows, stride)
    if np.any(matrix[:, STAMP_COUNT:] != SENTINEL):
        raise RuntimeError("打点 row 的隔离填充区被改写，检查 workspace 偏移/越界")
    return matrix[:, :STAMP_COUNT].copy()


def export_scenario(folder, summary, samples, matrices, paths, blocks):
    with (folder / "stage_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    spans = np.array([s["max_block_span_cycles"] for s in samples])
    representative = int(np.argmin(np.abs(spans - np.median(spans))))
    matrix = matrices[representative]
    decoded = decode_stamps(matrix, paths)
    with (folder / "row_stamps.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "block", "path"] + [f"stamp_{i}" for i in range(STAMP_COUNT)])
        for row, path in enumerate(paths):
            writer.writerow([row, row % blocks, path, *matrix[row].tolist()])
    with (folder / "row_stages.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "block", "path", "stage", "cycles"])
        for item in decoded:
            for name, cycles in zip(STAGE_NAMES[item["path"]], item["cycles"]):
                writer.writerow([item["row"], item["row"] % blocks, item["path"], name, int(cycles)])
    # 独立报告器：不复用 Echo 固定 27 槽名表，也不把外部 P50 当作频率。
    html = ["<!doctype html><meta charset='utf-8'><title>ABCDF 分阶段诊断</title>",
            "<style>body{font:15px sans-serif;max-width:1300px;margin:32px auto;padding:16px;color:#17202a}"
            "table{border-collapse:collapse;width:100%;margin-bottom:25px}th,td{border:1px solid #ccd3dc;padding:7px;text-align:right}"
            "th:first-child,td:first-child{text-align:left}small{color:#526070}</style>",
            "<h1>ABCDF 分阶段诊断</h1><p>单位 cycles。打点不增加同步；线程 0 到达边界的观测值，包含打点开销。</p>",
            "<p>阶段 P50 = 各 block 累计该阶段周期数的最大值，再取跨轮中位。"
            "占比 = 该路径所有 row、所有轮的周期工作量占比，不是 kernel 延迟占比。阶段 max 不可相加作总耗时。</p>"]
    for path in PATH_SLOTS:
        group = [r for r in summary if r["path"] == path]
        if not group:
            continue
        html.append(f"<h2>{path} 路径</h2><table><tr><th>阶段</th><th>最大 block P50</th><th>P95</th><th>每 row 均值 P50</th><th>skew P50</th><th>工作量占比</th></tr>")
        for record in group:
            html.append(f"<tr><td>{escape(record['stage'])}<br><small>{escape(record['description'])}</small></td>"
                        f"<td>{record['p50_max_block_cycles']:.0f}</td><td>{record['p95_max_block_cycles']:.0f}</td>"
                        f"<td>{record['p50_mean_row_cycles']:.0f}</td><td>{record['p50_block_skew_cycles']:.0f}</td>"
                        f"<td>{record['share_of_all_row_cycles_pct']:.1f}%</td></tr>")
        html.append("</table>")
        rows = [d for d in decoded if d["path"] == path]
        vmax = max((int(v) for d in rows for v in d["cycles"]), default=1) or 1
        html.append(f"<h3>代表轮 {representative}：每 row × 阶段热力表</h3><table><tr><th>row/block</th>" +
                    "".join(f"<th>{escape(n)}</th>" for n in STAGE_NAMES[path]) + "</tr>")
        for item in rows:
            html.append(f"<tr><td>{item['row']} / {item['row'] % blocks}</td>")
            for cycles in item["cycles"]:
                lightness = 98 - 25 * int(cycles) / vmax
                html.append(f"<td style='background:hsl(210,75%,{lightness:.1f}%)'>{int(cycles)}</td>")
            html.append("</tr>")
        html.append("</table>")
    (folder / "report.html").write_text("\n".join(html), encoding="utf-8")
    return representative


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--capacity", type=int, default=4096)
    parser.add_argument("--max-token", type=int, default=262144)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--scenario", choices=("all", *bench.SCENARIOS), default="all")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--shuffle-lru", action="store_true")
    parser.add_argument("--freq-ghz", type=float, help="显式频率，仅作可选 μs 换算；默认只报 cycles")
    parser.add_argument("--library", type=Path, default=ROOT / "build_simt_perf/libsimt_vs_echo_ops.so")
    parser.add_argument("--output", type=Path, default=Path("results/simt_perf"))
    args = parser.parse_args()
    if not (args.rows > 0 and 0 < args.topk <= args.capacity <= 4096 and args.max_token > args.topk
            and args.warmup >= 0 and args.repeat > 0):
        parser.error("需要 rows>0、0<topk<=capacity<=4096、max_token>topk、warmup>=0、repeat>0")
    if max(args.rows * args.max_token, args.rows * (args.capacity + args.topk)) > bench.I32_MAX:
        parser.error("形状超过 int32 索引范围")
    if args.freq_ghz is not None and not (np.isfinite(args.freq_ghz) and args.freq_ghz > 0):
        parser.error("--freq-ghz 必须为正有限数")
    import torch
    import torch_npu
    if not torch.npu.is_available():
        parser.error("没有可用 NPU")
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    library = args.library.resolve()
    if not library.is_file():
        parser.error("缺少共享库，请先 SIMT_PERF_STEP=ON bash build.sh")
    torch.ops.load_library(str(library))
    info = bench.validate_simt_library(torch, require_perf=True)
    blocks = int(torch.ops.sparse_kv_plan_op.simt_abcdf_core_count())
    args.output.mkdir(parents=True, exist_ok=True)
    capture = Path(tempfile.mkdtemp(prefix="capture_", dir=args.output.resolve()))
    metadata_path = capture / "capture.json"
    report = {"status": "running", "simt_build": info, "device": torch.npu.get_device_name(args.device),
              "device_index": args.device, "torch": torch.__version__, "torch_npu": torch_npu.__version__,
              "library": str(library), "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
              "grid_blocks": blocks, "shape": {k: getattr(args, k) for k in ("rows", "topk", "capacity", "max_token")},
              "warmup": args.warmup, "repeat": args.repeat, "seed": args.seed, "shuffle_lru": args.shuffle_lru,
              "freq_ghz": args.freq_ghz, "semantics": "per-row thread0 clock, no added barrier, uint32 modulo",
              "share_semantics": "all-row cycles workload share within path; not kernel latency share", "scenarios": {}}
    metadata_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("ABCDF 分步结果目录:", capture, flush=True)
    for scenario in (bench.SCENARIOS if args.scenario == "all" else [args.scenario]):
        case = bench.make_case(args.rows, args.topk, args.capacity, args.max_token, scenario, args.seed)
        if args.shuffle_lru:
            rng = np.random.default_rng(args.seed + 17)
            case["lru_slots"] = np.stack([rng.permutation(args.capacity).astype(np.int32) for _ in range(args.rows)])
        paths = expected_paths(case)
        expected = {"simt": bench.reference_simt(case)}
        inputs, initial, states = bench.prepare_npu(torch, case, bench.make_echo_initial(case), device)
        state = states["simt"]
        offset = state["layout"]["stamp_offset"]
        stamp_region = state["workspace"][offset:offset + state["layout"]["stamp_elements"]]
        for audit in range(3):
            bench.restore_state(torch, "simt", initial, state)
            for key in bench.SIMT_OUTPUT:
                state[key].fill_(-777)
            state["workspace"].fill_(SENTINEL)
            bench.invoke(torch, "simt", case, inputs, state)
            torch.npu.synchronize()
            bench.check_npu_result("simt", state, expected, f"{scenario}/audit-{audit}")
            decode_stamps(read_matrix(state, args.rows), paths)
        for _ in range(args.warmup):
            bench.restore_state(torch, "simt", initial, state)
            bench.invoke(torch, "simt", case, inputs, state)
        torch.npu.synchronize()
        bench.check_npu_result("simt", state, expected, f"{scenario}/after-warmup")
        samples, matrices = [], []
        for iteration in range(args.repeat):
            bench.restore_state(torch, "simt", initial, state)
            # 只污染计时区；不在正式样本前整体重写 scratch，以免额外预热 hash。
            stamp_region.fill_(SENTINEL)
            torch.npu.synchronize()
            bench.invoke(torch, "simt", case, inputs, state)
            torch.npu.synchronize()
            bench.check_npu_result("simt", state, expected, f"{scenario}/sample-{iteration}")
            matrix = read_matrix(state, args.rows)
            samples.append(aggregate_sample(matrix, decode_stamps(matrix, paths), blocks))
            matrices.append(matrix)
        summary = summarize(samples, args.freq_ghz)
        destination = capture / scenario
        destination.mkdir()
        rep = export_scenario(destination, summary, samples, matrices, paths, blocks)
        span_p50 = float(np.median([s["max_block_span_cycles"] for s in samples]))
        print(f"{scenario}: 全部样本正确，最大 block 内部跨度 P50={span_p50:.0f} cycles", flush=True)
        for record in summary:
            print(f"  {record['path']:4s} {record['stage']:<30s} "
                  f"P50(max block)={record['p50_max_block_cycles']:9.0f} cyc  "
                  f"skew={record['p50_block_skew_cycles']:7.0f} cyc  "
                  f"工作量占比={record['share_of_all_row_cycles_pct']:5.1f}%", flush=True)
        report["scenarios"][scenario] = {"correctness": "all measured samples passed", "layout": state["layout"],
              "paths": paths, "case_sha256": bench.array_hash({k: v for k, v in case.items() if isinstance(v, np.ndarray)}),
              "max_block_span_p50_cycles": span_p50, "stage_summary": summary, "samples": samples,
              "raw_stamps": [m.tolist() for m in matrices], "representative_repeat": rep}
        metadata_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    report["status"] = "complete"
    metadata_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("诊断完成:", metadata_path, "；完整 kernel 性能请另跑普通构建的 profile_kernel.sh", flush=True)


if __name__ == "__main__":
    main()
