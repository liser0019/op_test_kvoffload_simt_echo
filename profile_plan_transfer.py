#!/usr/bin/env python3
"""满缓存75/90/100%：ABCDF与真实Host→NPU Transfer的设备kernel延时。

默认chain：一次下发Plan后直接下发Transfer，按两条目标kernel分别统计。
pair_busy是每一对实际duration之和，pair_span是首kernel开始到末kernel结束；
二者均不是包含整个Python调用的端到端延时，也不将两个P50直接相加。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import tempfile
import sys
sys.dont_write_bytecode = True
import numpy as np

import plan_transfer_common as common
import compare as plan
from full_cache_cases import assert_case, describe
from checks.check_plan_transfer import run_small_gate
from harness_io import (preflight_args, write_json, record_failure, diagnostic_log,
                        diagnostic_details, capture_error)


def normalize(text):
    return re.sub(r"[\s_]", "", text).casefold()


def read_kernel_rows(paths, mode, repeat):
    targets = {"plan": common.PLAN_KERNEL, "transfer": common.TRANSFER_KERNEL}
    selected = ("plan", "transfer") if mode == "chain" else (mode,)
    records = []
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = {normalize(key): key for key in (reader.fieldnames or [])}
            def field(*names):
                return next((fields[normalize(name)] for name in names if normalize(name) in fields), None)
            name_key = field("Name", "Kernel Name", "Op Name")
            duration_key = field("Duration(us)", "Duration(µs)", "Duration(μs)")
            start_key = field("Start Time(us)", "Start Time(µs)", "Start Time(μs)")
            task_key = field("Task Type")
            stream_key = field("Stream ID", "StreamID")
            device_key = field("Device ID", "DeviceID")
            if not name_key or not duration_key or (mode == "chain" and not start_key):
                raise RuntimeError(f"CSV字段不足以确认单位/串联配对: {path}; fields={reader.fieldnames}")
            for row in reader:
                name = row.get(name_key, "")
                matches = [key for key in selected if re.search(
                    r"(?<![A-Za-z0-9_])" + re.escape(targets[key]) + r"(?![A-Za-z0-9_])", name)]
                if not matches:
                    continue  # TensorMove/tiling拷贝等不加入目标kernel时长。
                if len(matches) != 1:
                    raise RuntimeError(f"kernel匹配歧义: {row}")
                duration = float(row[duration_key])
                start = float(row[start_key]) if start_key else None
                if not np.isfinite(duration) or duration <= 0 or (start is not None and not np.isfinite(start)):
                    raise RuntimeError(f"无效设备时间: {row}")
                records.append({"stage": matches[0], "kernel_name": name,
                                "task_type": row.get(task_key, "") if task_key else "",
                                "duration_us": duration, "start_us": start, "csv": str(path),
                                "stream_id": row.get(stream_key) if stream_key else None,
                                "device_id": row.get(device_key) if device_key else None})
    for stage in selected:
        found = [record for record in records if record["stage"] == stage]
        groups = {(record["kernel_name"], record["task_type"]) for record in found}
        if len(found) != repeat or len(groups) != 1:
            raise RuntimeError(f"{stage}: 需要一个kernel分组且恰好{repeat}个样本，实际{len(found)}、groups={groups}")
    if mode == "chain":
        for key in ("stream_id", "device_id"):
            values = {record[key] for record in records if record[key] is not None}
            if len(values) > 1:
                raise RuntimeError(f"目标kernel跨{key}，不能解释为同流配对: {values}")
        records.sort(key=lambda record: record["start_us"])
        for index in range(repeat):
            first, second = records[2 * index:2 * index + 2]
            if first["stage"] != "plan" or second["stage"] != "transfer":
                raise RuntimeError("目标kernel顺序不是Plan→Transfer，不能对齐本次调用")
            if second["start_us"] < first["start_us"] + first["duration_us"] - 0.05:
                raise RuntimeError("Plan/Transfer时间有重叠，不能按同流串联解释")
            if index and first["start_us"] < records[2 * index - 1]["start_us"] + records[2 * index - 1]["duration_us"] - 0.05:
                raise RuntimeError("相邻调用重叠，无法按顺序配对")
    return records


def summarize(records, mode, payload_bytes):
    result = {}
    for stage in ("plan", "transfer"):
        samples = [record["duration_us"] for record in records if record["stage"] == stage]
        if samples:
            result[stage] = plan.summary(samples) | {"samples_us": samples}
    if mode == "chain":
        busy, span, gap = [], [], []
        for first, second in zip(records[::2], records[1::2]):
            busy.append(first["duration_us"] + second["duration_us"])
            span.append(second["start_us"] + second["duration_us"] - first["start_us"])
            gap.append(max(0.0, second["start_us"] - first["start_us"] - first["duration_us"]))
        result["pair_busy"] = plan.summary(busy) | {"samples_us": busy}
        result["pair_span"] = plan.summary(span) | {"samples_us": span}
        result["between_kernel_gap"] = plan.summary(gap) | {"samples_us": gap}
    if "transfer" in result:
        result["effective_payload_GBps"] = (payload_bytes / result["transfer"]["p50_us"] / 1000
                                           if payload_bytes else None)
    return result


def capture_case(torch, torch_npu, case, sources, args, mode, destination):
    args.current_capture = destination
    try:
        return _capture_case(torch, torch_npu, case, sources, args, mode, destination)
    except BaseException as exc:
        raise capture_error(destination, args.detailed_log, exc) from exc


def _capture_case(torch, torch_npu, case, sources, args, mode, destination):
    inputs, initial, state, golden, ids = common.prepare_case(
        torch, case, sources, args.token_bytes_k, args.token_bytes_v, args.launch_mode)
    common.audit(torch, case, inputs, initial, state, golden, ids, mode)
    for _ in range(args.warmup):
        common.restore(torch, initial, state, mode)
        common.invoke(torch, mode, case, inputs, state)
        torch.npu.synchronize()
    common.check_result(case, state, golden, ids, mode, "after-warmup")
    profiler = torch_npu.profiler
    config = profiler._ExperimentalConfig(profiler_level=profiler.ProfilerLevel.Level0,
                                          aic_metrics=profiler.AiCMetrics.AiCoreNone,
                                          export_type=[profiler.ExportType.Text])
    destination.mkdir(parents=True)
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
        schedule=profiler.schedule(wait=0, warmup=1, active=args.repeat, repeat=1),
        on_trace_ready=profiler.tensorboard_trace_handler(str(destination)),
        record_shapes=False, profile_memory=False, with_stack=False, experimental_config=config,
    ) as prof:
        for step in range(args.repeat + 1):
            common.restore(torch, initial, state, mode)
            with torch.profiler.record_function(f"full_cache/{case['scenario']}/{args.launch_mode}/{mode}"):
                common.invoke(torch, mode, case, inputs, state)
                torch.npu.synchronize()
            if args.verify_each_sample:
                common.check_result(case, state, golden, ids, mode, f"captured-step-{step}")
            prof.step()
    common.check_result(case, state, golden, ids, mode, "after-profile")
    paths = sorted(destination.rglob("kernel_details.csv"))
    if not paths:
        raise RuntimeError(f"没有kernel_details.csv，请检查完整采集日志: {destination}")
    records = read_kernel_rows(paths, mode, args.repeat)
    workload = describe(case, args.token_bytes_k, args.token_bytes_v)
    stats = summarize(records, mode, workload["payload_bytes_copied"])
    case_hash = plan.array_hash({key: value for key, value in case.items() if isinstance(value, np.ndarray)})
    result = {"status": "passed", "mode": mode, "launch_mode": args.launch_mode,
              "prepared_context": state.get("prepared_info"), "workload": workload, "case_sha256": case_hash,
              "plan_layout": state["layout"], "stats": stats,
              "raw_csv": [str(path) for path in paths], "target_kernel_samples": records}
    write_json(destination / "summary.json", result)
    rows = [{"stage": key} | {name: value for name, value in stats[key].items() if name != "samples_us"}
            for key in ("plan", "transfer", "pair_busy", "pair_span", "between_kernel_gap") if key in stats]
    with (destination / "stage_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    text = ", ".join(f"{key} P50={stats[key]['p50_us']:.3f}us" for key in ("plan", "transfer") if key in stats)
    print(f"{case['scenario']}/{mode}: {text}", flush=True)
    return result


def write_report(path, report):
    lines = ["# 满缓存：ABCDF / Transfer profiling", "",
             "整轮成功须同时存在COMPLETE.json且capture_metadata.json为complete；缺失标记的部分文件不是成功结果。", "",
             "一层、一次调用处理全部query行；单位μs，数据源为kernel_details.csv。", "",
             "| 启动方式 | 模式 | 目标命中率 | 实际命中率 | miss/行 | 指标 | 样本数 | P50 μs | P95 μs |",
             "|---|---|---:|---:|---:|---|---:|---:|---:|"]
    for run in report["runs"]:
        workload = run["workload"]
        for stage in ("plan", "transfer", "pair_busy", "pair_span", "between_kernel_gap"):
            if stage not in run["stats"]:
                continue
            metric = run["stats"][stage]
            lines.append(f"| {run['launch_mode']} | {run['mode']} | {workload['hit_percent_requested']}% | "
                         f"{workload['hit_percent_actual']:.5f}% | {workload['misses_per_row']} | {stage} | "
                         f"{metric['count']} | {metric['p50_us']:.3f} | {metric['p95_us']:.3f} |")
    lines += ["", "单kernel duration：该条kernel在设备上的执行时间。",
              "pair_busy：逐次配对后将Plan/Transfer duration相加，再统计；不能相加两个P50。",
              "pair_span：每对从Plan开始到Transfer结束，含两者之间的设备时间间隙。",
              "between_kernel_gap：逐对用Transfer开始减Plan结束计算，再统计；不能用两个P50相减。",
              "上述指标都不等于Python公共接口的端到端延时。",
              "固定Host源地址和输入、预热及计时外恢复仍为默认条件；有效payload吞吐不等于冷Host DRAM带宽。",
              "100%命中仍实际下发Transfer，但payload搬运量为0。",
              "prepared去除了两条kernel之间的tiling上传；实际间隙及其变化以本次采集为准。"]
    pending = path.with_name("report.pending.md")
    with pending.open("w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
        stream.flush()
        import os
        os.fsync(stream.fileno())
    pending.replace(path)


def finalize_report(session, report):
    """状态/标记全部写入后才发布成功报告；失败只保留partial/pending。"""
    report["status"] = "finalizing"
    write_json(session / "capture_metadata.json", report)
    write_report(session / "report.partial.md", report)
    report["status"] = "complete"
    write_json(session / "capture_metadata.json", report)
    write_json(session / "COMPLETE.json", {"status": "complete",
        "metadata": str(session / "capture_metadata.json"), "launch_mode": report["launch_mode"]})
    (session / "report.partial.md").replace(session / "report.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_arguments(parser)
    parser.add_argument("--modes", nargs="+", choices=("chain", "plan", "transfer"), default=["chain"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--verify-each-sample", action="store_true")
    parser.add_argument("--describe-only", action="store_true", help="仅服务器CPU查看用例和内存需求，不生成性能结论")
    parser.add_argument("--output", type=Path, default=Path("/dev/shm/abcdf_plan_transfer"))
    args = parser.parse_args()
    preset = common.resolve_payload(args, parser)
    if args.warmup < 0 or args.repeat <= 0 or len(args.modes) != len(set(args.modes)):
        parser.error("warmup>=0、repeat>0且modes不能重复")
    try:
        cases = [common.case_from_args(args, percentage) for percentage in args.hit_percentages]
        for case in cases:
            assert_case(case, plan.reference_simt(case))
    except ValueError as exc:
        parser.error(str(exc))
    for case in cases:
        print(json.dumps(describe(case, args.token_bytes_k, args.token_bytes_v), ensure_ascii=False), flush=True)
    print("Host池内存需求（不是profiling磁盘需求）:", describe(
        cases[0], args.token_bytes_k, args.token_bytes_v)["host_pool_bytes"], "bytes", flush=True)
    if args.describe_only:
        print("仅构造用例和golden，未运行kernel；没有性能结果")
        return

    # 必须先检查，再初始化NPU/申请Host大池；不检查源码分区剩余空间。
    storage = preflight_args(args, args.output)
    args.output = Path(storage["output"]["path"])
    session = Path(tempfile.mkdtemp(prefix=f"session_{args.launch_mode}_", dir=args.output))
    args.detailed_log = session / "profile.log"
    args.current_capture = session
    metadata = session / "capture_metadata.json"
    report = {"status": "running", "payload": preset, "storage": storage, "runs": [],
              "launch_mode": args.launch_mode,
              "warmup_calls": args.warmup, "profiler_warmup_steps": 1, "recorded_calls": args.repeat,
              "verify_each_sample": args.verify_each_sample, "kernel_entries": {
                  "plan": common.PLAN_KERNEL, "transfer": common.TRANSFER_KERNEL},
              "timing": "device kernel Duration(us), not Event or end-to-end",
              "cache_condition": "fixed addresses; restore state and payload outside targets; warmed source",
              "scenario_order": args.hit_percentages, "modes": args.modes,
              "completion_marker": str(session / "COMPLETE.json")}
    print("结果目录:", session, flush=True)
    try:
        write_json(metadata, report)
        with diagnostic_log(args.detailed_log):
            torch, torch_npu, environment = common.load_runtime(args)
            report["environment"] = environment
            # 所选启动方式过完整门禁；独立checker默认同时检查两路。
            report["correctness_gate"] = run_small_gate(torch, (args.launch_mode,))
            write_json(metadata, report)
            sources = common.prepare_sources(torch, cases[0], args.token_bytes_k, args.token_bytes_v)
            for case in cases:
                for mode in args.modes:
                    report["runs"].append(capture_case(
                        torch, torch_npu, case, sources, args, mode,
                        session / case["scenario"] / mode))
                    write_json(metadata, report)
            torch.npu.synchronize()
            report["ratios_vs_hit75"] = []
            for mode in args.modes:
                baseline = next((run for run in report["runs"] if run["mode"] == mode
                                 and run["workload"]["hit_percent_requested"] == 75), None)
                if baseline is None:
                    continue
                for run in report["runs"]:
                    if run["mode"] != mode:
                        continue
                    for stage in ("plan", "transfer", "pair_busy", "pair_span", "between_kernel_gap"):
                        if stage in run["stats"] and baseline["stats"][stage]["p50_us"] > 0:
                            report["ratios_vs_hit75"].append({"mode": mode, "stage": stage,
                                "hit_percent": run["workload"]["hit_percent_requested"],
                                "p50_over_hit75": run["stats"][stage]["p50_us"] / baseline["stats"][stage]["p50_us"]})
            finalize_report(session, report)
    except BaseException as exc:
        report["diagnostics"] = diagnostic_details(args.current_capture, args.detailed_log)
        record_failure(metadata, report, exc)
        try:
            print(capture_error(args.current_capture, args.detailed_log, exc), file=sys.stderr, flush=True)
        except BaseException:
            pass
        raise
    print("完整采集通过:", session, flush=True)
    print("详细日志:", args.detailed_log, flush=True)


if __name__ == "__main__":
    main()
