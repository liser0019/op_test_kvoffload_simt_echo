#!/usr/bin/env python3
"""新版联合Plan的MTP=0/1/3 profiling；逐行ABCDF作同输入对照。"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import re
import sys
import tempfile
import time
import numpy as np
sys.dont_write_bytecode = True
import joint_plan_common as common
from joint_plan_cases import SCENARIOS, OVERLAPS, make_case, describe
from compare import summary
from checks.check_joint_plan import run_gate
from harness_io import preflight_args, write_json, record_failure, diagnostic_log, capture_error


def read_kernels(paths, names, repeat, expected_order):
    """精确kernel名、μs单位、样本数和交错执行顺序；恢复TensorMove一律排除。"""
    records = []
    def norm(text):
        return re.sub(r"[\s_]", "", text).casefold()
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = {norm(k): k for k in (reader.fieldnames or [])}
            def field(*keys):
                return next((fields[norm(k)] for k in keys if norm(k) in fields), None)
            nk = field("Name", "Kernel Name", "Op Name")
            dk = field("Duration(us)", "Duration(µs)", "Duration(μs)")
            sk = field("Start Time(us)", "Start Time(µs)", "Start Time(μs)")
            tk, stream_key, device_key = field("Task Type"), field("Stream ID"), field("Device ID")
            if not nk or not dk or not sk:
                raise RuntimeError(f"CSV缺少kernel名、Duration(us)或Start Time(us): {path}; {reader.fieldnames}")
            for row in reader:
                found = [n for n in names if re.search(r"(?<![A-Za-z0-9_])" +
                    re.escape(common.KERNELS[n]) + r"(?![A-Za-z0-9_])", row[nk])]
                if not found:
                    continue
                if len(found) != 1:
                    raise RuntimeError("目标kernel匹配歧义")
                duration, start = float(row[dk]), float(row[sk])
                if not np.isfinite(duration) or duration <= 0 or not np.isfinite(start):
                    raise RuntimeError("设备时间无效")
                records.append({"implementation": found[0], "kernel_name": row[nk],
                    "task_type": row.get(tk, ""), "duration_us": duration, "start_us": start,
                    "stream": row.get(stream_key), "device": row.get(device_key), "csv": str(path)})
    for name in names:
        items = [r for r in records if r["implementation"] == name]
        if len(items) != repeat or len({(r["kernel_name"], r["task_type"]) for r in items}) != 1:
            raise RuntimeError(f"{name}须为一个kernel分组、恰好{repeat}个样本，实际{len(items)}")
    for key in ("stream", "device"):
        if len({r[key] for r in records if r[key] is not None}) > 1:
            raise RuntimeError(f"目标采集混入多个{key}")
    records.sort(key=lambda r: r["start_us"])
    if [r["implementation"] for r in records] != expected_order:
        raise RuntimeError("CSV目标kernel顺序与本次实际下发顺序不一致")
    for left, right in zip(records, records[1:]):
        if right["start_us"] < left["start_us"] + left["duration_us"] - 0.05:
            raise RuntimeError("同流目标kernel出现重叠，不能按本次串行调用解释")
    return records


def execution_order(names, iteration):
    # 三路保留六种排列；多候选用循环轮换，任意连续repeat的各位置次数差至多1。
    count = len(names)
    sequence = list(names) if count != 3 or (iteration // count) % 2 == 0 else list(reversed(names))
    offset = iteration % count
    return sequence[offset:] + sequence[:offset]


def duration_ratios(stats):
    pairs = [("private", "abcdf"), ("joint", "abcdf"), ("private", "joint")]
    pairs += [(name, base) for name in stats if name in common.VARIANTS
              for base in ("abcdf", "joint" if common.is_joint(name) else "private")]
    pairs += [("joint_bitmap", "joint_abc"), ("joint_turbo", "joint_abc"),
              ("joint_turbo", "joint_bitmap"), ("private_turbo", "private_a")]
    pairs += [(name, base) for name in stats if name in ("joint_turbo_a", "joint_turbo_b", "joint_turbo_ab")
              for base in ("joint_turbo", "private_turbo")]
    return {f"{left}_over_{right}_p50": stats[left]["p50_us"] / stats[right]["p50_us"]
            for left, right in pairs if left in stats and right in stats}


def capture(torch, torch_npu, case, names, args, destination):
    bundles = {}
    try:
        return _capture(torch, torch_npu, case, names, args, destination, bundles)
    except BaseException:
        for bundle in bundles.values():
            common.dump_failure(case, bundle, destination / "failure_state")
        raise


def _capture(torch, torch_npu, case, names, args, destination, bundles):
    inputs = common.upload_inputs(torch, case)
    for name in names:
        bundles[name] = common.prepare(torch, case, name, inputs)
    audits = {name: common.audit(torch, case, bundle, failure_dir=destination / "failure_state")
              for name, bundle in bundles.items()}
    for iteration in range(args.warmup):
        for name in execution_order(names, iteration):
            bundles[name]["phase"] = f"warmup-{iteration + 1}"
            common.restore(torch, bundles[name])
            common.invoke(bundles[name])
            torch.npu.synchronize()
            common.check(case, bundles[name])
    for bundle in bundles.values():
        # audit已经至少执行过一次，warmup=0也能检查最终状态。
        if args.warmup:
            common.check(case, bundle)
    profiler = torch_npu.profiler
    config = profiler._ExperimentalConfig(profiler_level=profiler.ProfilerLevel.Level0,
        aic_metrics=profiler.AiCMetrics.AiCoreNone, export_type=[profiler.ExportType.Text])
    destination.mkdir(parents=True)
    orders, expected_order = [], []
    with profiler.profile(activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
        schedule=profiler.schedule(wait=0, warmup=1, active=args.repeat, repeat=1),
        on_trace_ready=profiler.tensorboard_trace_handler(str(destination)),
        record_shapes=False, profile_memory=False, with_stack=False, experimental_config=config) as prof:
        for step in range(args.repeat + 1):
            if step == 1:
                # schedule warmup=1: preceding prof.step() has started ACTIVE capture.
                # Acquisition workaround only; never changes correctness or CSV sample gates.
                wait_record = {"reason": "profiler_active_start_host_wait", "requested_seconds": 1.0,
                               "start_unix_seconds": time.time(), "status": "waiting"}
                write_json(destination / "capture_start_wait.json", wait_record)
                start = time.monotonic()
                print("Profiler ACTIVE started: host wait 1.0s before first target", flush=True)
                time.sleep(1.0)
                wait_record.update(status="completed", elapsed_seconds=time.monotonic() - start)
                write_json(destination / "capture_start_wait.json", wait_record)
            order = execution_order(names, step)
            if step:
                expected_order.extend(order)
                orders.append(order)
            for name in order:
                bundles[name]["phase"] = f"profiler-step-{step}/active-sample-{step}"
                common.restore(torch, bundles[name])
                with torch.profiler.record_function(f"joint_plan/mtp{case['mtp']}/{name}"):
                    common.invoke(bundles[name])
                    torch.npu.synchronize()
                common.check(case, bundles[name])  # 逐样本必检；错误输入不恢复，不继续调用。
            prof.step()
    final = {name: common.check(case, bundle) for name, bundle in bundles.items()}
    paths = sorted(destination.rglob("kernel_details.csv"))
    if not paths:
        raise RuntimeError("采集后未找到kernel_details.csv；检查此目录原始文件和详细日志")
    records = read_kernels(paths, names, args.repeat, expected_order)
    stats = {}
    for name in names:
        samples = [r["duration_us"] for r in records if r["implementation"] == name]
        stats[name] = summary(samples) | {"samples_us": samples,
            "raw_csv": sorted({r["csv"] for r in records if r["implementation"] == name}),
            "us_per_query_row_p50": float(np.percentile(samples, 50)) / case["rows"],
            "us_per_request_p50": float(np.percentile(samples, 50)) / case["requests"],
            "joint_enabled": bundles[name]["joint_enabled"], "layout": bundles[name]["layout"],
            "correctness": final[name]}
    result = {"status": "passed", "workload": describe(case), "stats": stats,
              "execution_orders": orders, "audit": audits, "target_samples": records,
              "raw_csv": [str(p) for p in paths], "capture_start_wait": wait_record}
    result.update(duration_ratios(stats))
    write_json(destination / "summary.json", result)
    with (destination / "kernel_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["implementation", "mtp", "requests", "query_rows", "count", "p50_us", "p95_us", "total_misses"])
        for name, stat in stats.items():
            writer.writerow([name, case["mtp"], case["requests"], case["rows"], stat["count"], stat["p50_us"],
                             stat["p95_us"], final[name]["total_transfer_descriptors"]])
    for bundle in bundles.values():
        common.close(bundle)
    print(f"{case['scenario']}/{case['overlap']}/MTP={case['mtp']}: " +
          ", ".join(f"{n}={s['p50_us']:.3f}us" for n, s in stats.items()), flush=True)
    return result


def mtp_comparison(runs):
    result = []
    for run in runs:
        workload = run["workload"]
        if workload["mtp"] == 0:
            continue
        off = next(r for r in runs if r["comparison"] == run["comparison"]
                   and r["workload"]["scenario"] == workload["scenario"]
                   and r["workload"]["overlap"] == workload["overlap"] and r["workload"]["mtp"] == 0)
        for name, stat in run["stats"].items():
            result.append({"comparison": run["comparison"], "scenario": workload["scenario"],
                "overlap": workload["overlap"], "implementation": name, "mtp": workload["mtp"],
                "off_requests": off["workload"]["requests"], "on_requests": workload["requests"],
                "off_rows": off["workload"]["query_rows"], "on_rows": workload["query_rows"],
                "on_over_off_p50": stat["p50_us"] / off["stats"][name]["p50_us"],
                "on_over_off_per_query_row": stat["us_per_query_row_p50"] / off["stats"][name]["us_per_query_row_p50"]})
    return result


def report_text(report):
    lines = ["# 联合Plan MTP=0/1/3 profiling", "",
        "单次调用处理所有query行；数据来自kernel_details.csv，单位μs。", "",
        "abcdf为13_abcdf独立设备路径，已移除来源页检查（14 tensors + 3 attrs）；不经过RunPrivateRow，不生成联合协议元数据。历史含页表版本数据不能直接作为本轮同场结果。",
        "private为本新版joint关闭后的逐行ABCDF路径，含新增映射发布。",
        "MTP=0时joint列也关闭合并；保持同一实现的非MTP行为。", "",
        "| 对照 | 工况/TopK重合 | MTP | 请求/行 | 实现 | P50 μs | P95 μs | 样本数 | miss记录总数 | 共享/溢出组数 |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---|"]
    for run in report["runs"]:
        w = run["workload"]
        for name, stat in run["stats"].items():
            c = stat["correctness"]
            lines.append(f"| {run['comparison']} | {w['scenario']}/{w['overlap']} | {w['mtp']} | "
                f"{w['requests']}/{w['query_rows']} | {name} | {stat['p50_us']:.3f} | {stat['p95_us']:.3f} | "
                f"{stat['count']} | {c['total_transfer_descriptors']} | " +
                ("不适用 |" if name == "abcdf" else f"{c['joint_groups']}/{c['overflow_groups']} |"))
    lines += ["", "## 同一MTP输入下：三方耗时比", "",
              "比值均为左侧P50除以右侧P50；大于1表示左侧耗时更长。", "",
              "| 对照 | 工况/重合模式 | MTP | private/abcdf | joint/abcdf | private/joint |",
              "|---|---|---:|---:|---:|---:|"]
    for run in report["runs"]:
        w = run["workload"]
        values = [f"{run[key]:.3f}" if key in run else "—"
                  for key in ("private_over_abcdf_p50", "joint_over_abcdf_p50", "private_over_joint_p50")]
        lines.append(f"| {run['comparison']} | {w['scenario']}/{w['overlap']} | {w['mtp']} | " + " | ".join(values) + " |")
    lines += ["", "## Joint A/B/C/D 候选与未优化基线", "",
              "大于1表示候选更慢；这里的字母是Joint实验，与原ABCDF命名无关。", "",
              "| 工况/重合 | MTP | 候选 | 候选/参考路径（joint或joint_abc） P50 | 候选/ABCDF P50 |",
              "|---|---:|---|---:|---:|"]
    for run in report["runs"]:
        w = run["workload"]
        for name in run["stats"]:
            if name not in common.VARIANTS:
                continue
            base = ("joint" if "joint" in run["stats"] else "joint_abc") if common.is_joint(name) else "private"
            values = [f"{run[name + '_over_' + ref + '_p50']:.3f}"
                      if name + '_over_' + ref + '_p50' in run else "—" for ref in (base, "abcdf")]
            lines.append(f"| {w['scenario']}/{w['overlap']} | {w['mtp']} | {name} | " + " | ".join(values) + " |")
    lines += ["", "## Turbo 分阶段查询 / 每行 Scan 实验", "",
              "A/B为本次Turbo实验，不是旧Joint A/B；比值>1表示候选更慢。", "",
              "| 工况/重合 | MTP | 候选 | 候选/ABCDF | 候选/private_turbo | 候选/joint_turbo |",
              "|---|---:|---|---:|---:|---:|"]
    for run in report["runs"]:
        w = run["workload"]
        for name in ("joint_turbo_a", "joint_turbo_b", "joint_turbo_ab"):
            if name in run["stats"]:
                ratios = [f"{run[name + '_over_' + base + '_p50']:.3f}"
                          if name + '_over_' + base + '_p50' in run else "—"
                          for base in ("abcdf", "private_turbo", "joint_turbo")]
                lines.append(f"| {w['scenario']}/{w['overlap']} | {w['mtp']} | {name} | " + " | ".join(ratios) + " |")
    lines += ["", "各行实际初始命中数/比例见 JSON workload.initial_hit_positions_per_row / initial_hit_fraction_per_row。",
              "75pct 是首行构造标签，independent 后续 query 不保证同一命中率；输入未因候选修改。"]
    lines += ["", "## 相对MTP关闭：一次调用耗时比", "",
        "| 对照 | 工况/重合模式 | 实现 | MTP | MTP开启 P50 / MTP关闭 P50 |",
        "|---|---|---|---:|---:|"]
    for item in report["mtp_comparison"]:
        lines.append(f"| {item['comparison']} | {item['scenario']}/{item['overlap']} | {item['implementation']} | "
                     f"{item['mtp']} | {item['on_over_off_p50']:.3f} |")
    lines += ["", "固定请求数时MTP=0/1/3展开1/2/4倍query行；固定行数仅为辅助控制。",
        "按行/请求除法是吞吐归一化，不能视作独立请求延时。场景命名指第0行历史缓存构造，实际miss见表。",
        "每行currentSlots、全部持久状态、LRU和miss列表均核对；private/joint另核对owner/page映射。",
        "联合miss顺序/槽位可不同，不强求与private的LRU逐字相等。",
        "private/joint计时含其调度、union/回退和映射发布；abcdf只测原独立kernel。",
        "abcdf保留原接口逐次tiling上传，private/joint采用prepared；上传/恢复均不计入目标kernel时长。",
        "不执行真实Transfer或SFA；miss记录减少不能直接换算端到端收益。",
        "固定输入与地址、恢复后预热；不代表冷cache或生产TopK轨迹。",
        "只在退出码0、complete元数据、COMPLETE.json和report.md均存在时认定整轮完成。"]
    env = report.get("environment", {})
    lines += ["", "## 设备、库与编译身份", "",
        f"- 设备：{env.get('device')} / {env.get('device_name')}；主机：{env.get('hostname')}",
        f"- 库：{env.get('library')}；SHA256：{env.get('library_sha256')}",
        f"- 编译器：{env.get('identity', {}).get('compiler')}",
        f"- 编译选项：{env.get('identity', {}).get('flags')}",
        "- 完整源码身份、输入指纹与原始样本：capture_metadata.json；原始数据索引：data_index.json。",
        "", "## 原始CSV索引", "",
        "| 对照 | 工况/重合模式 | MTP | 实现 | 原始CSV路径 |", "|---|---|---:|---|---|"]
    for run in report["runs"]:
        w = run["workload"]
        for name, stat in run["stats"].items():
            lines.append(f"| {run['comparison']} | {w['scenario']}/{w['overlap']} | {w['mtp']} | {name} | " +
                         "<br>".join(stat["raw_csv"]) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_runtime_arguments(parser)
    parser.add_argument("--implementations", nargs="+", choices=common.IMPLEMENTATIONS, default=list(common.DEFAULT_IMPLEMENTATIONS))
    parser.add_argument("--mtp", nargs="+", type=int, choices=(0, 1, 3), default=[0, 1, 3])
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--comparison", choices=("same_requests", "same_query_rows", "both"), default="same_requests")
    parser.add_argument("--fixed-query-rows", type=int, default=32)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=["full_cache_75pct"])
    parser.add_argument("--overlaps", nargs="+", choices=OVERLAPS, default=["independent"])
    for name, default in (("topk", 2048), ("capacity", 4096), ("max-token", 262144), ("block-size", 64), ("seed", 2026)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--verify-each-sample", action="store_true", default=True,
                        help="兼容旧命令；三方现在始终逐样本校验，不能关闭")
    parser.add_argument("--regression-repeat", type=int, default=100)
    parser.add_argument("--correctness-report", type=Path, help="复用同库/同设备完整门禁，避免重复运行100次回归")
    parser.add_argument("--describe-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("/workspace/joint_plan_results/profile"))
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat <= 0 or args.regression_repeat <= 0 or 0 not in args.mtp:
        parser.error("warmup>=0、repeat>0且mtp列表必须含0")
    for key in ("mtp", "implementations", "scenarios", "overlaps"):
        if len(getattr(args, key)) != len(set(getattr(args, key))):
            parser.error(key + "不得重复")
    modes = ("same_requests", "same_query_rows") if args.comparison == "both" else (args.comparison,)
    tasks = []
    try:
        for index, (scenario, overlap) in enumerate((s, o) for s in args.scenarios for o in args.overlaps):
            mtps = args.mtp if index % 2 == 0 else list(reversed(args.mtp))
            for mode in modes:
                for mtp in mtps:
                    requests = args.requests
                    if mode == "same_query_rows":
                        if args.fixed_query_rows <= 0 or args.fixed_query_rows % (mtp + 1):
                            raise ValueError("fixed-query-rows必须为正且能被各MTP的q_len整除")
                        requests = args.fixed_query_rows // (mtp + 1)
                    case = make_case(requests, args.topk, args.capacity, args.max_token, mtp,
                                     scenario, overlap, args.seed, args.block_size)
                    tasks.append((mode, case))
                    print(json.dumps({"comparison": mode, "workload": describe(case)}, ensure_ascii=False), flush=True)
    except ValueError as exc:
        parser.error(str(exc))
    if args.describe_only:
        return
    storage = preflight_args(args, args.output)  # NPU初始化和所有device分配之前。
    session = Path(tempfile.mkdtemp(prefix="joint_matrix_", dir=storage["output"]["path"]))
    metadata, log = session / "capture_metadata.json", session / "profile.log"
    current = session
    report = {"status": "running", "storage": storage, "runs": [], "mtp_comparison": [],
              "warmup": args.warmup, "repeat": args.repeat, "profiler_warmup_steps": 1,
              "verify_each_sample": True, "regression_repeat": args.regression_repeat,
              "timing": "torch_npu.profiler kernel duration (us)", "scope": "Plan only; no Transfer/SFA",
              "implementations": args.implementations, "mtp": args.mtp}
    print("联合Plan矩阵目录:", session, flush=True)
    try:
        with diagnostic_log(log):
            write_json(metadata, report)
            torch, torch_npu, environment = common.load_runtime(args)
            report["environment"] = environment
            if args.correctness_report is not None:
                from checks.check_joint_plan import validate_gate_report
                report["gate"] = validate_gate_report(args.correctness_report, environment, args.regression_repeat, args.implementations)
            else:
                report["gate"] = run_gate(torch, args.regression_repeat, session / "gate_failure", args.implementations)
            for mode, case in tasks:
                current = session / mode / case["scenario"] / case["overlap"] / f"mtp{case['mtp']}"
                result = capture(torch, torch_npu, case, args.implementations, args, current)
                result["comparison"] = mode
                report["runs"].append(result)
                write_json(metadata, report)
            report["mtp_comparison"] = mtp_comparison(report["runs"])
            # 原始数据索引只引用本机路径；不上传源码、输入、采集数据或二进制。
            write_json(session / "data_index.json", {"status": "data_index_only", "runs": [
                {"comparison": run["comparison"], "workload": run["workload"],
                 "implementations": {name: {"kernel_name": common.KERNELS[name],
                     "count": stat["count"], "raw_csv": stat["raw_csv"]}
                     for name, stat in run["stats"].items()}} for run in report["runs"]]})
            report["status"] = "finalizing"
            write_json(metadata, report)
            (session / "report.pending.md").write_text(report_text(report))
            report["status"] = "complete"
            write_json(metadata, report)
            write_json(session / "COMPLETE.json", {"status": "complete", "metadata": str(metadata)})
            (session / "report.pending.md").replace(session / "report.md")
    except BaseException as exc:
        record_failure(metadata, report, exc)
        try:
            print(capture_error(current, log, exc), file=sys.stderr, flush=True)
        except BaseException:
            pass
        raise
    print("联合Plan采集完成:", session, flush=True)


if __name__ == "__main__":
    main()
