#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""v2 trap 取证：读打点构建的 stepCycles 区域，定位 trap 发生在哪一步。

原理：ECHO_PERF_STEP 构建下，PerfEnd 在每步完成时把 t0 的 clock() 周期写入
workspace 尾部 stepCycles 区。kernel 在第 N 步 trap → 步 0..N-1 已写、N 及之后
为哨兵。即使 npuSynchronizeDevice 抛错，进程内的 workspace 张量仍尝试读回。

用法（需先构建打点版）:
    ECHO_PERF_STEP=ON bash build.sh
    python debug_v2_stepcycles.py --device 0
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import compare as bench  # noqa: E402
import echo_perf_step as eps  # noqa: E402  (ECHO_PERF_NUM_STEPS / 步名表)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--library", type=Path,
                        default=Path("build_perf/libsimt_vs_echo_ops.so"))
    parser.add_argument("--scenario", default="cold_reset", choices=tuple(bench.SCENARIOS))
    args = parser.parse_args()

    import torch
    import torch_npu  # noqa: F401
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    torch.ops.load_library(str(args.library.resolve()))
    print("实际加载共享库:", args.library.resolve(), flush=True)

    case = bench.make_case(32, 2048, 4096, 262144, args.scenario, 2026)
    initial = bench.make_echo_initial(case)
    inputs, initial_dev, states = bench.prepare_npu(torch, case, initial, device, None)
    state = states["echo"]
    bench.restore_state(torch, "echo", initial_dev, state)
    for key in bench.ECHO_OUTPUT:
        state[key].fill_(255 if key == "miss_mask" else -777)
    state["workspace"].fill_(-12345)   # 哨兵: 步周期恒 >=0, 负值=未写

    try:
        bench.invoke(torch, "echo", case, inputs, state)
        torch.npu.synchronize()
        print("!!! 调用竟然成功 (未 trap) — 检查是否加载了打点库/正确场景", flush=True)
    except Exception as exc:  # noqa: BLE001
        print("调用异常(预期):", type(exc).__name__, str(exc)[:150], flush=True)

    blocks = state["blocks"]
    rows, topk, capacity = (case[k] for k in ("rows", "topk", "capacity"))
    base = rows * (2 * (capacity + 1) + 4 + 3 * topk)
    off = (base + 1) & ~1
    nsteps = eps.ECHO_PERF_NUM_STEPS
    try:
        region = state["workspace"][off:off + blocks * nsteps].cpu().numpy()
        ws = region.reshape(blocks, nsteps)
        print(f"stepCycles 读回成功 (blocks={blocks}, steps={nsteps})，每块最后完成的步：", flush=True)
        last_written = []
        for b in range(blocks):
            written = [i for i in range(nsteps) if ws[b, i] >= 0]
            last_written.append(written[-1] if written else -1)
            if b < 6:
                tail = [(i, int(ws[b, i])) for i in written[-3:]] if written else "无"
                print(f"  block{b:2d}: 最后完成步={written[-1] if written else '无'}  尾部={tail}", flush=True)
        print("全 32 块最后完成步的分布:", {s: last_written.count(s) for s in sorted(set(last_written))}, flush=True)
        names = eps.ECHO_PERF_STEP_NAMES
        for s in sorted(set(last_written)):
            label = names[s] if 0 <= s < len(names) else "?"
            nxt = names[s + 1] if 0 <= s + 1 < len(names) else "?"
            print(f"  最后完成步 {s} ({label}) → trap 发生在步 {s+1} ({nxt}) 附近", flush=True)
    except Exception as exc:  # noqa: BLE001
        print("workspace 读回失败 (设备错误后上下文可能已毒化):", type(exc).__name__, str(exc)[:150])


if __name__ == "__main__":
    main()
