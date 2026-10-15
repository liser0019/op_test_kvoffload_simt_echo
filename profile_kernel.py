#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""kernel 级 profile：torch_npu.profiler 采集 echo/simt 目标 kernel 的纯耗时。

Event 计时包含 tiling H2D 与 host 下发的固定开销（两路各 ~百 μs），只能看比值；
本脚本用 profiler 的 kernel_details.csv 直接统计目标 kernel 的 P50/P95/min/max，
得到可比的设备端真实耗时。测试数据与 compare.py 完全一致（import compare 作为
bench，同尺寸/seed/场景/状态恢复/调用方式），采集前后均与 golden 全量比对。

用法（推荐经 profile_kernel.sh 包装运行，自动设置 /dev/shm 采集目录）：

    bash profile_kernel.sh                          # 全部四个场景
    bash profile_kernel.sh --scenario warm_75pct    # 单场景
    bash profile_kernel.sh --active 50              # 加大采样

产物：
    $KV_PROFILE_ROOT/traces/run-*/summary.json   # 汇总 + 原始样本
    $KV_PROFILE_ROOT/profile.log                 # 完整日志（wrapper 生成）
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# 环境准备 —— 必须在 import torch 之前执行：
#   1) 采集目录放 /dev/shm（trace 文本量大，避免写满磁盘；wrapper 已建好则复用）；
#   2) TMP/TMPDIR/TEMP 指向采集目录（profiler 中间文件）；
#   3) 去掉会影响计时/设备可见性的环境变量。
# ---------------------------------------------------------------------------
PROFILE_ROOT = Path(os.environ.get("KV_PROFILE_ROOT")
                    or tempfile.mkdtemp(prefix="kvprof.", dir="/dev/shm"))
(PROFILE_ROOT / "tmp").mkdir(parents=True, exist_ok=True)
os.environ["KV_PROFILE_ROOT"] = str(PROFILE_ROOT)
os.environ["TMPDIR"] = os.environ["TMP"] = os.environ["TEMP"] = str(PROFILE_ROOT / "tmp")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
for _name in ("ASCEND_LAUNCH_BLOCKING", "ASCEND_RT_VISIBLE_DEVICES"):
    os.environ.pop(_name, None)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare as bench  # noqa: E402

# profiler 里只统计这两个 kernel；restore 的 copy_ 等任务按名字过滤掉。
KERNEL_NAMES = {
    "simt": "sparse_kv_plan_kernel",
    "echo": "echo_lru_kernel",
}


def parse_kernel_csv(csv_path: Path, kernel_name: str, expect_count: int) -> list[float]:
    """从 kernel_details.csv 提取目标 kernel 的每次耗时（μs），列名做兼容探测。"""
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        name_keys = [k for k in ("Name", "Kernel Name", "KernelName", "Op Name", "op_name")
                     if k in fields]
        duration_key = next((k for k in ("Duration(us)", "Duration (us)", "Task Duration(us)",
                                         "task_duration") if k in fields), None)
        if not name_keys or duration_key is None:
            raise RuntimeError(f"无法识别 CSV 列名: {fields}")
        samples = [float(row[duration_key].replace(",", ""))
                   for row in reader
                   if any(kernel_name in (row.get(k) or "") for k in name_keys)]
    if len(samples) != expect_count:
        raise RuntimeError(f"{kernel_name}: 预期 {expect_count} 个 kernel，实际 {len(samples)}；"
                           f"请保留采集目录与日志排查")
    values = np.asarray(samples)
    if not np.all(np.isfinite(values)) or not np.all(values > 0):
        raise RuntimeError(f"{kernel_name}: 无效耗时: {samples}")
    return samples


def profile_one(torch, torch_npu, name, case, inputs, initial, state, expected,
                scenario, folder, warmup, active):
    p = torch_npu.profiler
    print("开始采集:", name, "目录:", folder, flush=True)
    with p.profile(
        activities=[p.ProfilerActivity.CPU, p.ProfilerActivity.NPU],
        schedule=p.schedule(wait=0, warmup=warmup, active=active, repeat=1),
        on_trace_ready=p.tensorboard_trace_handler(str(folder), async_mode=False),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        experimental_config=p._ExperimentalConfig(
            profiler_level=p.ProfilerLevel.Level0,
            aic_metrics=p.AiCMetrics.AiCoreNone,
            export_type=p.ExportType.Text,
        ),
    ) as prof:
        for _ in range(warmup + active):
            # 每次调用恢复同一份初始状态（恢复在 record_function 内、便于在 trace 里剔除）。
            with torch.autograd.profiler.record_function(f"{name}/restore"):
                bench.restore_state(torch, name, initial, state)
            with torch.autograd.profiler.record_function(f"{name}/operator"):
                bench.invoke(torch, name, case, inputs, state)
                torch.npu.synchronize()
            prof.step()
    bench.check_npu_result(name, state, expected, f"{scenario}/after-profile")

    paths = list(folder.rglob("kernel_details.csv"))
    if len(paths) != 1:
        raise RuntimeError(f"{name}: CSV 数量异常: {paths}；请保留采集目录 {folder} 和日志")
    samples = parse_kernel_csv(paths[0], KERNEL_NAMES[name], active)
    values = np.asarray(samples)
    stats = {
        "count": len(samples),
        "p50_us": float(np.percentile(values, 50)),
        "p95_us": float(np.percentile(values, 95)),
        "min_us": float(values.min()),
        "max_us": float(values.max()),
        "samples_us": samples,
        "csv": str(paths[0]),
    }
    print(f"{scenario}/{name}: kernel 次数={stats['count']}, P50={stats['p50_us']:.2f} us, "
          f"P95={stats['p95_us']:.2f} us, min={stats['min_us']:.2f} us, max={stats['max_us']:.2f} us",
          flush=True)
    print("CSV:", paths[0], flush=True)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", type=int, default=1, help="逻辑 NPU 卡号（与 compare.py 一致）")
    parser.add_argument("--scenario", choices=("all", *bench.SCENARIOS), default="all")
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--capacity", type=int, default=4096)
    parser.add_argument("--max-token", type=int, default=262144)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--active", type=int, default=20)
    parser.add_argument("--library", type=Path,
                        default=Path(__file__).resolve().parent / "build/libsimt_vs_echo_ops.so",
                        help="普通构建产物；打点构建(build_perf)不用于本测量")
    args = parser.parse_args()
    if args.warmup < 0 or args.active <= 0:
        parser.error("warmup>=0、active>0")
    scenarios = list(bench.SCENARIOS) if args.scenario == "all" else [args.scenario]

    if not torch.npu.is_available():
        parser.error("没有可用 NPU")
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    library = args.library.resolve()
    if not library.is_file():
        parser.error(f"共享库不存在：{library}；请先 bash build.sh --soc=ascend950")
    torch.ops.load_library(str(library))
    for op_name in ("sparse_kv_plan", "echo_lru"):
        if not hasattr(torch.ops.sparse_kv_plan_op, op_name):
            parser.error(f"共享库缺少 {op_name}，请重新构建本目录")

    print("设备:", args.device, torch.npu.get_device_name(args.device), flush=True)
    print("共享库:", library, flush=True)
    print("采集目录:", PROFILE_ROOT, flush=True)

    traces = PROFILE_ROOT / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=str(traces)))
    report = {
        "device_index": args.device,
        "device": torch.npu.get_device_name(args.device),
        "library": str(library),
        "shape": {k: getattr(args, k) for k in ("rows", "topk", "capacity", "max_token")},
        "seed": args.seed, "warmup": args.warmup, "active": args.active,
        "scenarios": {},
    }

    for scenario in scenarios:
        # 与正式性能测试使用相同尺寸、场景和随机种子。
        case = bench.make_case(args.rows, args.topk, args.capacity, args.max_token,
                               scenario, args.seed)
        echo_initial = bench.make_echo_initial(case)
        expected = {
            "simt": bench.reference_simt(case),
            "echo": bench.reference_echo(case, echo_initial),
        }
        inputs, initial, states = bench.prepare_npu(torch, case, echo_initial, device)

        # 采集之前先检查两版结果（防带病采集）。
        for name in ("simt", "echo"):
            bench.restore_state(torch, name, initial, states[name])
            bench.invoke(torch, name, case, inputs, states[name])
            torch.npu.synchronize()
            bench.check_npu_result(name, states[name], expected, f"{scenario}/before-profile")
        print(f"{scenario}: 两版正确性检查通过", flush=True)

        report["scenarios"][scenario] = {}
        for name in ("simt", "echo"):
            folder = run_dir / scenario / name
            report["scenarios"][scenario][name] = profile_one(
                torch, torch_npu, name, case, inputs, initial, states[name],
                expected, scenario, folder, args.warmup, args.active)
        ratio = (report["scenarios"][scenario]["echo"]["p50_us"]
                 / report["scenarios"][scenario]["simt"]["p50_us"])
        report["scenarios"][scenario]["echo_over_simt_kernel_p50"] = ratio
        simt_p50 = report["scenarios"][scenario]["simt"]["p50_us"]
        echo_p50 = report["scenarios"][scenario]["echo"]["p50_us"]
        print(f"{scenario}: kernel P50 SIMT={simt_p50:.2f} μs, Echo={echo_p50:.2f} μs, "
              f"Echo/SIMT={ratio:.2f}×", flush=True)

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    print("\n===== 汇总（kernel 级 P50，不含 tiling H2D / host 下发）=====", flush=True)
    for scenario, entry in report["scenarios"].items():
        print(f"{scenario:>18s}: SIMT={entry['simt']['p50_us']:8.2f} μs  "
              f"Echo={entry['echo']['p50_us']:8.2f} μs  "
              f"Echo/SIMT={entry['echo_over_simt_kernel_p50']:.2f}×", flush=True)
    print("汇总文件:", summary_path, flush=True)
    print("采集目录:", run_dir, flush=True)


if __name__ == "__main__":
    main()
