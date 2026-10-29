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
import numpy as np

import plan_transfer_common as common
import compare as plan
from full_cache_cases import assert_case, describe
from checks.check_plan_transfer import run_small_gate


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
                                "duration_us": duration, "start_us": start, "csv": str(path)})
    for stage in selected:
        found = [record for record in records if record["stage"] == stage]
        groups = {(record["kernel_name"], record["task_type"]) for record in found}
        if len(found) != repeat or len(groups) != 1:
            raise RuntimeError(f"{stage}: 需要一个kernel分组且恰好{repeat}个样本，实际{len(found)}、groups={groups}")
    if mode == "chain":
        records.sort(key=lambda record: record["start_us"])
        for index in range(repeat):
            first, second = records[2 * index:2 * index + 2]
            if first["stage"] != "plan" or second["stage"] != "transfer":
                raise RuntimeError("目标kernel顺序不是Plan→Transfer，不能对齐本次调用")
            if second["start_us"] < first["start_us"] + first["duration_us"] - 0.05:
                raise RuntimeError("Plan/Transfer时间有重叠，不能按同流串联解释")
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
    inputs, initial, state, golden, ids = common.prepare_case(
        torch, case, sources, args.token_bytes_k, args.token_bytes_v)
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
            with torch.profiler.record_function(f"full_cache/{case['scenario']}/{mode}"):
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
    result = {"status": "passed", "mode": mode, "workload": workload, "case_sha256": case_hash,
              "plan_layout": state["layout"], "stats": stats,
              "raw_csv": [str(path) for path in paths], "target_kernel_samples": records}
    (destination / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
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
             "一层、一次调用处理全部query行；单位μs，数据源为kernel_details.csv。", "",
             "| 模式 | 目标命中率 | 实际命中率 | miss/行 | Plan P50 | Transfer P50 | 实际配对busy P50 | 配对span P50 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for run in report["runs"]:
        workload, stats = run["workload"], run["stats"]
        def value(key):
            return f"{stats[key]['p50_us']:.3f}" if key in stats else "—"
        lines.append(f"| {run['mode']} | {workload['hit_percent_requested']}% | {workload['hit_percent_actual']:.5f}% | "
                     f"{workload['misses_per_row']} | {value('plan')} | {value('transfer')} | "
                     f"{value('pair_busy')} | {value('pair_span')} |")
    lines += ["", "busy对每次真实Plan/Transfer配对求和后取中位数，不是两个P50相加；",
              "span包含两kernel之间的设备时间间隙，不是整个Python接口端到端延时。",
              "Host源地址和输入在重复中固定并经过预热；有效payload吞吐不等于冷Host DRAM链路带宽。",
              "100%命中仍实际下发Transfer，但payload搬运量为0。"]
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_arguments(parser)
    parser.add_argument("--modes", nargs="+", choices=("chain", "plan", "transfer"), default=["chain"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--verify-each-sample", action="store_true")
    parser.add_argument("--describe-only", action="store_true", help="仅服务器CPU查看用例、实际miss和内存需求；不生成性能结论")
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
    if args.describe_only:
        print("仅构造用例和golden，未运行kernel；没有性能结果")
        return
    torch, torch_npu, environment = common.load_runtime(args)
    args.output.mkdir(parents=True, exist_ok=True)
    session = Path(tempfile.mkdtemp(prefix="session_", dir=args.output.resolve()))
    report = {"status": "running", "environment": environment, "payload": preset, "runs": [],
              "warmup_calls": args.warmup, "profiler_warmup_steps": 1, "recorded_calls": args.repeat,
              "verify_each_sample": args.verify_each_sample, "kernel_entries": {
                  "plan": common.PLAN_KERNEL, "transfer": common.TRANSFER_KERNEL},
              "timing": "device kernel Duration(us), not Event or end-to-end",
              "cache_condition": "fixed addresses; restore public state and payload outside targets; warm steady source",
              "scenario_order": args.hit_percentages, "modes": args.modes}
    def save():
        (session / "capture_metadata.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    save()
    print("结果目录:", session, flush=True)
    try:
        # 小尺寸先验证Host可访问性、非对齐/大于16KiB分块、页表/槽/count边界。
        report["correctness_gate"] = run_small_gate(torch)
        save()
        sources = common.prepare_sources(torch, cases[0], args.token_bytes_k, args.token_bytes_v)
        for case in cases:
            for mode in args.modes:
                report["runs"].append(capture_case(torch, torch_npu, case, sources, args, mode,
                                                   session / case["scenario"] / mode))
                save()
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
                for stage in ("plan", "transfer", "pair_busy", "pair_span"):
                    if stage in run["stats"]:
                        report["ratios_vs_hit75"].append({"mode": mode, "stage": stage,
                            "hit_percent": run["workload"]["hit_percent_requested"],
                            "p50_over_hit75": run["stats"][stage]["p50_us"] / baseline["stats"][stage]["p50_us"]})
        report["status"] = "complete"
        save()
        write_report(session / "report.md", report)
    except BaseException as exc:
        report["status"] = "failed"
        report["failure"] = str(exc)
        save()
        raise
    print("完整采集通过:", session, flush=True)


if __name__ == "__main__":
    main()
