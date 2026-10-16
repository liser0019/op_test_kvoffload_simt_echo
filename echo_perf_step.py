#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Echo 分步耗时测试：读取 ECHO_PERF_STEP 构建写入 workspace 的每核每步 clock() 周期。

前置条件（与 compare.py 的正式测量分开构建，互不覆盖）：

    source /usr/local/Ascend/cann-9.1.0/set_env.sh
    ECHO_PERF_STEP=ON bash build.sh --soc=ascend950   # 产物在 build_perf/
    python3 echo_perf_step.py --device 1

测试数据与 compare.py 完全一致：直接 import compare.py 的 make_case /
make_echo_initial / reference_echo / prepare_npu / invoke / restore_state，
同尺寸、同 seed、同场景、同状态恢复与调用方式，因此分步耗时的各步之和
可与 compare.py 的 Event 总耗时互相印证。结果正确性在测量前后均与
reference_echo 全量比对（本脚本是诊断工具，不是正式性能结论；打点构建
会比普通构建略慢，正式对比仍用 compare.py + bash build.sh）。

kernel 侧打点：PerfBegin/PerfEnd（threadIdx.x==0 的 clock() 差值）写入
workspace 尾部 stepCycles[blockIdx.x * ECHO_PERF_NUM_STEPS + stepIdx]，
非 spec 路径共 20 步（spec 路径 28 步，本基准 spec_enabled=False）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from compare import (SCENARIOS, ECHO_OUTPUT, array_hash, check_npu_result,  # noqa: E402
                     invoke, make_case, make_echo_initial, prepare_npu,
                     reference_echo, restore_state)

# ---- 与 csrc/echo/echo_lru_tiling.h 的 ECHO_PERF_NUM_STEPS 保持一致（防漂移断言）----
_TILING = (ROOT / "csrc/echo/echo_lru_tiling.h").read_text(encoding="utf-8")
_MATCH = re.search(r"constexpr uint32_t ECHO_PERF_NUM_STEPS\s*=\s*(\d+)\s*;", _TILING)
assert _MATCH, "echo_lru_tiling.h 中找不到 ECHO_PERF_NUM_STEPS"
ECHO_PERF_NUM_STEPS = int(_MATCH.group(1))

# 非 spec 路径 Process() 内 PerfBegin/PerfEnd 的 20 个打点（顺序即 perfStepIdx_ 顺序）。
# protect 折叠后: OpProtect 已并入 OpHitMiss（原 Step2 整段删除），OpFifoInc 挪至
# alloc pop 之后（防 Argtopk 读到不一致 fifo，详见 kernel 头注释）。
# specEnabled=1 时 Step0.5 在 "Step0.SyncGrid" 之后额外插入 8 步，本基准不覆盖。
ECHO_PERF_STEP_NAMES = [
    "Step0.OpResetRows", "Step0.OpResetDthPri", "Step0.OpResetFree", "Step0.OpResetAvailFifo",
    "Step0.SyncGrid",
    "Step1.OpHitMiss", "Step1.SyncGrid",
    "Step2.ArgtopkAllRows", "Step2.SyncGrid", "Step2.OpFreeRelease", "Step2.OpFreePush",
    "Step3.OpAllocPop", "Step3.OpFifoInc", "Step3.SyncGrid",
    "Step4.OpAllocPriority", "Step4.SyncBlock", "Step4.OpFifoAvailUpdate",
    "Step5.OpMissCumsum", "Step5.SyncGrid", "Step5.OpSetUpdate",
]
ECHO_PERF_NONSPEC_STEPS = len(ECHO_PERF_STEP_NAMES)
assert ECHO_PERF_NUM_STEPS == ECHO_PERF_NONSPEC_STEPS + 8, (
    f"ECHO_PERF_NUM_STEPS={ECHO_PERF_NUM_STEPS} 与 {ECHO_PERF_NONSPEC_STEPS} 个非 spec 步名 + 8 个 spec 步不符，"
    "请同步更新 echo_perf_step.py 的 ECHO_PERF_STEP_NAMES 或 tiling.h")

SENTINEL = -12345  # 周期值恒 >= 0，负哨兵可判定“该区域未被 kernel 写入”


def echo_step_region_offset(case, blocks):
    """workspace 内 stepCycles 区的 int32 偏移，与 EchoLruKernel::Init / EchoWorkspaceElements 一致。"""
    rows, topk, capacity = (case[k] for k in ("rows", "topk", "capacity"))
    base = rows * (capacity + 1 + 4 + 3 * topk)
    return ((base + 1) & ~1) + blocks * (256 + 8192)


def read_step_cycles(state, case, blocks):
    region = state["workspace"][echo_step_region_offset(case, blocks):
                                echo_step_region_offset(case, blocks) + blocks * ECHO_PERF_NUM_STEPS]
    matrix = region.cpu().numpy().reshape(blocks, ECHO_PERF_NUM_STEPS)
    return matrix[:, :ECHO_PERF_NONSPEC_STEPS]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--capacity", type=int, default=4096)
    parser.add_argument("--max-token", type=int, default=262144)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=int, default=1, help="逻辑 NPU 卡号（与 compare.py 一致）")
    parser.add_argument("--scenario", choices=("all", *SCENARIOS), default="all")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--library", type=Path,
                        default=ROOT / "build_perf/libsimt_vs_echo_ops.so",
                        help="ECHO_PERF_STEP=ON 构建的共享库；默认 build_perf/libsimt_vs_echo_ops.so")
    parser.add_argument("--json", type=Path, default=Path("results/echo_perf_step.json"))
    args = parser.parse_args()
    if not (args.rows > 0 and 0 < args.topk <= args.capacity <= 4096 and args.max_token > args.topk):
        parser.error("需要 rows>0、0<topk<=capacity<=4096、max_token>topk")
    if args.warmup < 0 or args.repeat <= 0:
        parser.error("warmup>=0、repeat>0")
    scenarios = list(SCENARIOS) if args.scenario == "all" else [args.scenario]

    try:
        import torch
        import torch_npu
    except ImportError as exc:
        parser.error(f"需要匹配的 torch/torch_npu 环境：{exc}")
    if not torch.npu.is_available():
        parser.error("没有可用 NPU")
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")

    library_path = args.library.resolve()
    if not library_path.is_file():
        parser.error(f"共享库不存在：{library_path}；请先运行 ECHO_PERF_STEP=ON bash build.sh")
    try:
        torch.ops.load_library(str(library_path))
    except (OSError, RuntimeError) as exc:
        parser.error(f"无法加载 {library_path}：{exc}")
    for name in ("sparse_kv_plan", "echo_lru", "echo_lru_core_count"):
        if not hasattr(torch.ops.sparse_kv_plan_op, name):
            parser.error(f"共享库缺少 {name}，请重新构建本目录")

    manifest_path = ROOT / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    metadata = {
        "device_index": args.device, "device": torch.npu.get_device_name(args.device),
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "library_path": str(library_path),
        "library_sha256": hashlib.sha256(library_path.read_bytes()).hexdigest(),
        "echo_revision": manifest.get("echo_revision", "unknown"),
        "echo_kernel_sha256": hashlib.sha256(
            (ROOT / "csrc/echo/echo_lru_kernel.h").read_bytes()).hexdigest(),
        "note": "ECHO_PERF_STEP 打点构建（每步 t0 clock() 周期），非正式性能结论；正式对比用 compare.py",
        "shape": {k: getattr(args, k) for k in ("rows", "topk", "capacity", "max_token")},
        "seed": args.seed, "warmup": args.warmup, "repeat": args.repeat,
        "echo_perf_num_steps": ECHO_PERF_NUM_STEPS,
        "non_spec_steps": ECHO_PERF_NONSPEC_STEPS,
        "step_names": ECHO_PERF_STEP_NAMES,
    }
    print("实际加载共享库:", library_path, flush=True)
    results = {"metadata": metadata, "scenarios": {}}

    for scenario in scenarios:
        case = make_case(args.rows, args.topk, args.capacity, args.max_token, scenario, args.seed)
        echo_initial = make_echo_initial(case)
        expected = {"echo": reference_echo(case, echo_initial)}
        inputs, initial, states = prepare_npu(torch, case, echo_initial, device)
        state = states["echo"]
        blocks = int(state["blocks"])

        # A. 审计一次：污染输出与 workspace → 调用 → 全量比对参考；同时确认打点确实写入。
        restore_state(torch, "echo", initial, state)
        for key in ECHO_OUTPUT:
            state[key].fill_(255 if key == "miss_mask" else -777)
        state["workspace"].fill_(SENTINEL)
        invoke(torch, "echo", case, inputs, state)
        torch.npu.synchronize()
        check_npu_result("echo", state, expected, f"{scenario}/audit")
        audit_steps = read_step_cycles(state, case, blocks)
        if (audit_steps == SENTINEL).all():
            raise RuntimeError(
                f"{scenario}: stepCycles 区未被写入。请确认加载的是 ECHO_PERF_STEP=ON 构建 "
                f"(ECHO_PERF_STEP=ON bash build.sh)，而不是普通 build/ 产物。")
        if (audit_steps < 0).any():
            raise RuntimeError(f"{scenario}: stepCycles 区存在负值，workspace 布局或打点不一致")
        miss_count = args.topk - case["hits"]
        free_size = max(miss_count - int(echo_initial["avail"][0]), 0)
        print(f"{scenario}: miss/行={miss_count}, 需淘汰={free_size}, blocks={blocks}", flush=True)

        # B. 预热（不读回，不打哨兵，保持与正式调用相同的缓存条件）。
        for _ in range(args.warmup):
            restore_state(torch, "echo", initial, state)
            invoke(torch, "echo", case, inputs, state)
        torch.npu.synchronize()

        # C. 正式采样：哨兵 → 恢复 → Event 计时调用 → 读回每核每步周期。
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        samples = []
        for _ in range(args.repeat):
            restore_state(torch, "echo", initial, state)
            state["workspace"].fill_(SENTINEL)
            start.record()
            invoke(torch, "echo", case, inputs, state)
            end.record()
            end.synchronize()
            event_us = start.elapsed_time(end) * 1000.0
            if not np.isfinite(event_us) or event_us <= 0:
                raise RuntimeError(f"{scenario}: 无效的 Event 耗时 {event_us}")
            steps = read_step_cycles(state, case, blocks)
            if (steps == SENTINEL).any() or (steps < 0).any():
                raise RuntimeError(f"{scenario}: 本轮 stepCycles 未完整写入（存在哨兵/负值）")
            samples.append({"event_us": event_us, "step_max_cycles": steps.max(axis=0).tolist()})

        # D. 测量后全量校验（本脚本失败即退出，不产出性能结论）。
        check_npu_result("echo", state, expected, f"{scenario}/after-measurement")

        # 统计：每步取“最慢核”（max over blocks）的跨轮中位数；用 Event 总耗时标定频率。
        step_max = np.array([s["step_max_cycles"] for s in samples], dtype=np.float64)  # [repeat, ECHO_PERF_NONSPEC_STEPS]
        event_us = np.array([s["event_us"] for s in samples])
        median_cycles = np.median(step_max, axis=0)
        total_cycles = step_max.sum(axis=1)
        # 频率标定 (MHz) = 每轮步周期之和 / 每轮 Event 微秒；取中位数。
        freq_mhz = float(np.median(total_cycles / event_us))
        est_us = median_cycles / freq_mhz if freq_mhz > 0 else np.zeros_like(median_cycles)
        pct = median_cycles / median_cycles.sum() * 100.0

        order = np.argsort(median_cycles)[::-1]
        print(f"  频率标定 ≈ {freq_mhz:.0f} MHz (Σ步max中位 {np.median(total_cycles):.0f} cyc / "
              f"Event P50 {np.median(event_us):.2f} μs)，热点 Top5:", flush=True)
        for idx in order[:5]:
            print(f"    [{idx:2d}] {ECHO_PERF_STEP_NAMES[idx]:<24s} "
                  f"{median_cycles[idx]:>10.0f} cyc  {pct[idx]:5.1f}%  ≈{est_us[idx]:7.2f} μs", flush=True)

        results["scenarios"][scenario] = {
            "blocks": blocks,
            "expected_misses_per_row": miss_count,
            "free_size_per_row": free_size,
            "event_us": {"p50": float(np.median(event_us)), "min": float(event_us.min()),
                         "max": float(event_us.max())},
            "freq_mhz_estimate": freq_mhz,
            "steps": [
                {"idx": i, "name": ECHO_PERF_STEP_NAMES[i],
                 "median_cycles": float(median_cycles[i]),
                 "min_cycles": float(step_max[:, i].min()),
                 "max_cycles": float(step_max[:, i].max()),
                 "pct_of_total": float(pct[i]),
                 "est_us": float(est_us[i])}
                for i in range(ECHO_PERF_NONSPEC_STEPS)
            ],
            "raw_samples": samples,
            "correctness": "passed",
            "case_sha256": array_hash({k: v for k, v in case.items() if isinstance(v, np.ndarray)}),
        }

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("结果与原始样本:", args.json.resolve())


if __name__ == "__main__":
    main()
