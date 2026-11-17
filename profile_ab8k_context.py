#!/usr/bin/env python3
"""三路专项：32请求、MTP3、8K缓存；只改变上下文范围，不新增算法。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import joint_plan_common as common
from fixed_plan_check import exact_workload_key
from harness_io import (capture_error, diagnostic_log, preflight_args,
                        record_failure, write_json)
from joint_plan_cases import describe, make_case, sort_topk
from profile_joint_mtp import capture

ROOT = Path(__file__).resolve().parent
NAMES = ("joint_mtp7_fused_ab_8k_512", "joint_rankpipe_a_512", "joint_rankpipe_ac_512")
LABELS = dict(zip(NAMES, ("AB8K", "rankpipe A", "rankpipe AC")))
CONTEXTS = (262144, 102400)
SCENARIOS = ("cold_reset", "full_cache_75pct")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    common.add_runtime_arguments(p)
    p.set_defaults(device=0)
    p.add_argument("--max-tokens", nargs="+", type=int, choices=CONTEXTS, default=list(CONTEXTS))
    p.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeat", type=int, default=20)
    p.add_argument("--regression-repeat", type=int, default=5)
    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--describe-only", action="store_true", help="只打印用例描述，不构建/初始化NPU")
    p.add_argument("--output", type=Path, default=Path("/workspace/ab8k_context_results"))
    return p


def make_workload(max_token, scenario, requests=32):
    # 每个长度/工况只生成一次输入，三种实现读相同TopK、页映射语义及初始LRU。
    # 排序在计时外；保留重复token，不把各draft强行改成相同前缀。
    return sort_topk(make_case(requests=requests, topk=2048, capacity=8192,
        max_token=max_token, mtp=3, scenario=scenario, overlap="overlap40",
        seed=2026, block_size=64, history_pool_tokens=8192))


def workload_info(case):
    return describe(case) | {
        "committed_context": case["context"],
        "visible_seq_len_min": int(case["visible_seq_lens"].min()),
        "visible_seq_len_max": int(case["visible_seq_lens"].max()),
    }


def workloads(args):
    return [make_workload(length, scenario)
            for length in args.max_tokens for scenario in args.scenarios]


def gate(torch, args, cases, output):
    """同输入重复审计可复用于采集；连续状态仍逐步计算参考，绝不重置掩盖错误。"""
    records, continuous = [], []
    evidence = {name: {} for name in NAMES}
    failure = output / "failure_state"
    for case in cases:
        label = f"max_token_{case['max_token']}/{case['scenario']}"
        inputs = common.upload_inputs(torch, case)
        for name in NAMES:
            bundle = common.prepare(torch, case, name, inputs)
            bundle["fast_fixed_check"] = True
            try:
                result = common.audit(torch, case, bundle, args.regression_repeat, failure)
                require_joint(result, case["requests"])
                evidence[name][exact_workload_key(case)] = "passed"
                records.append({"case": label, "implementation": name,
                                "workload": workload_info(case), "layout": bundle["layout"], "audit": result})
            except BaseException:
                common.dump_failure(case, bundle, failure)
                raise
            finally:
                common.close(bundle)
        print("三路同输入门禁通过:", label, flush=True)

    # 每档长度仅1请求、4行的短连续链：首次reset→续用→更换请求→stable失效→续用。
    # next call直接读取上一调用的设备持久状态，禁止CPU golden回写设备。
    for length in args.max_tokens:
        for name in NAMES:
            case = make_workload(length, "cold_reset", requests=1)
            bundle = common.prepare(torch, case, name)
            try:
                for step in range(5):
                    if step == 2:
                        case["req_ids"] += 99
                    if step == 3:
                        case["stable_prefix_lens"][:] = length // 2
                    bundle["phase"] = f"continuous/{length}/{step + 1}"
                    common.update_inputs(torch, bundle, case)
                    common.poison(bundle)
                    torch.npu.synchronize()
                    before = common.snapshot(bundle)
                    common.invoke(bundle)
                    torch.npu.synchronize()
                    result = common.check(case, bundle, before)
                    continuous.append({"max_token": length, "implementation": name,
                                       "call": step + 1, "result": result})
            except BaseException:
                common.dump_failure(case, bundle, failure)
                raise
            finally:
                common.close(bundle)
    report = {"status": "passed", "implementations": list(NAMES),
              "regression_repeat": args.regression_repeat,
              "cases": records, "continuous": continuous, "exact_workloads": evidence}
    write_json(output / "gate.json", report)
    return evidence


def require_joint(check, requests):
    # 本轮专测4行联合；不把回退结果混入联合性能排名。
    if check.get("joint_groups") != requests or check.get("overflow_groups") != 0:
        raise RuntimeError(f"本轮要求{requests}组联合、0回退，实际: {check}")


def report_text(report):
    lines = ["# AB8K / rankpipe A / rankpipe AC：两档上下文同场对照", "",
        "固定32请求、MTP3（128条query行）、TopK2048、capacity8192、512线程、seed2026、overlap40。",
        "256K=262144，100K=102400；max_token是token位置范围上界。实际已提交/可见长度见下表。",
        "仅设备Plan kernel Duration（μs），含归并和rankpipe的GM映射暂存；不含恢复、校验、Transfer/SFA或接口端到端耗时。", "",
        "| max_token | 工况 | 已提交context | 可见长度范围 | 实现 | P50/P95 μs | 样本数 | 实现/AB8K | 共享/回退 | descriptor | Block Num |",
        "|---:|---|---:|---|---|---|---:|---:|---|---:|---|"]
    for run in report["runs"]:
        w = run["workload"]
        base = run["stats"][NAMES[0]]["p50_us"]
        for name in NAMES:
            s = run["stats"][name]
            c = s["correctness"]
            lines.append(f"| {w['max_token']} | {w['scenario']} | {w['committed_context']} | "
                f"{w['visible_seq_len_min']}–{w['visible_seq_len_max']} | {LABELS[name]} | "
                f"{s['p50_us']:.3f}/{s['p95_us']:.3f} | {s['count']} | {s['p50_us']/base:.3f} | "
                f"{c['joint_groups']}/{c['overflow_groups']} | {c['total_transfer_descriptors']} | "
                f"{','.join(map(str,s.get('block_nums',[]))) or 'CSV未导出'} |")
    lines += ["", "## 同一实现：100K / 256K 的 P50 比值", "",
              "小于1表示本轮100K较快；不同长度会重新生成token位置，不能解读为只改一个属性的相同token轨迹。", "",
              "| 工况 | 实现 | 100K/256K |", "|---|---|---:|"]
    for scenario in SCENARIOS:
        runs = {r["workload"]["max_token"]: r for r in report["runs"] if r["workload"]["scenario"] == scenario}
        if all(c in runs for c in CONTEXTS):
            for name in NAMES:
                ratio = runs[102400]["stats"][name]["p50_us"] / runs[262144]["stats"][name]["p50_us"]
                lines.append(f"| {scenario} | {LABELS[name]} | {ratio:.3f} |")
    lines += ["", "75%是首行名义命中率；draft实际命中率、跨行重合和union大小见metadata。",
        "每档长度内，三路读取同一逻辑输入，各自持有独立可变状态；每样本恢复初态并完整校验输入/状态/输出/保护区。",
        "三路轮换下发顺序；只接受CSV精确kernel名、微秒单位、样本数和真实调用顺序全部匹配的结果。",
        "固定地址预热；保留原始CSV、逐样本Duration和源码/编译/库/设备身份，不混用历史设备结果。",
        "源码未改算法。AB8K为旧32位UB映射路径；rankpipe为16位映射暂存路径，额外搬运计入目标kernel。",
        "data_index.json为原始CSV索引，capture_metadata.json含输入和环境身份及墙钟耗时。",
        f"阶段墙钟秒数：{json.dumps(report['wall_seconds'], ensure_ascii=False)}", ""]
    return "\n".join(lines)


def main(argv=None):
    args = parser().parse_args(argv)
    if args.warmup < 0 or min(args.repeat, args.regression_repeat) < 1:
        raise ValueError("warmup必须非负；repeat和regression-repeat必须为正")
    if len(set(args.max_tokens)) != len(args.max_tokens) or len(set(args.scenarios)) != len(args.scenarios):
        raise ValueError("长度/工况不能重复，避免覆盖采集目录")
    cases = workloads(args)
    if args.describe_only:
        print(json.dumps({"implementations": NAMES, "workloads": [workload_info(c) for c in cases]},
                         ensure_ascii=False, indent=2))
        return
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"请用新的输出目录，保留既有结果: {output}")
    storage = preflight_args(args, output)  # NPU初始化/构建前先检查输出、临时目录。
    report = {"status": "running", "implementations": list(NAMES), "runs": [],
              "warmup": args.warmup, "repeat": args.repeat, "storage": storage, "wall_seconds": {}}
    log = output / "launcher.log"
    try:
        with diagnostic_log(log):
            if not args.skip_build:
                start = time.monotonic()
                with (output / "build.log").open("w", encoding="utf-8") as stream:
                    subprocess.run(["bash", str(ROOT / "build_joint_plan.sh"), "--ab8k-compare-only"],
                                   cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
                report["wall_seconds"]["build"] = time.monotonic() - start
            torch, torch_npu, environment = common.load_runtime(args)
            report["environment"] = environment
            start = time.monotonic()
            args.workload_gate = gate(torch, args, cases, output)
            args.fast_fixed_check = True
            args.reuse_workload_audit = True
            report["wall_seconds"]["gate"] = time.monotonic() - start
            start = time.monotonic()
            for case in cases:
                path = output / "profile" / f"max_token_{case['max_token']}" / case["scenario"]
                run = capture(torch, torch_npu, case, NAMES, args, path)
                run["workload"] = workload_info(case)
                for stats in run["stats"].values():
                    require_joint(stats["correctness"], case["requests"])
                report["runs"].append(run)
                write_json(output / "RUN_STATUS.json", report)
            report["wall_seconds"]["profile"] = time.monotonic() - start
            report["status"] = "passed"
            write_json(output / "capture_metadata.json", report)
            write_json(output / "data_index.json", [{"workload": r["workload"], "raw_csv": r["raw_csv"]}
                                                       for r in report["runs"]])
            (output / "report.md").write_text(report_text(report), encoding="utf-8")
            write_json(output / "RUN_STATUS.json", report)
            write_json(output / "COMPLETE.json", {"status": "passed", "report": str(output / "report.md")})
    except BaseException as exc:
        report["diagnostics"] = str(capture_error(output, log, exc))
        print(report["diagnostics"], flush=True)
        record_failure(output / "RUN_STATUS.json", report, exc)
        raise
    print((output / "report.md").read_text(encoding="utf-8"), flush=True)
    print("===== AB8K_CONTEXT_ALL_STAGES_PASSED =====", flush=True)


if __name__ == "__main__":
    main()
