#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""v2 设备侧调试：单次 echo 调用 + 逐字段 mismatch 摘要（不 assert，全字段列出）。

用途：compare.py 的 audit 在第一个失败字段就中断（如 htd），看不到其他字段
是否正确。本脚本一次调用后把 ECHO_STATE + ECHO_OUTPUT 全部字段的首例差异
打印出来，用于定位 v2 上板问题。复用 compare.py 的全部机制（同尺寸/seed/
场景/恢复/调用）。

用法:
    python debug_v2_state.py --device 0 --scenario cold_reset \
        --library build_v2dbg/libsimt_vs_echo_ops.so
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import compare as bench  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--scenario", default="cold_reset",
                        choices=tuple(bench.SCENARIOS))
    parser.add_argument("--library", type=Path,
                        default=Path("build_v2dbg/libsimt_vs_echo_ops.so"))
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--capacity", type=int, default=4096)
    parser.add_argument("--max-token", type=int, default=262144)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    import torch
    import torch_npu  # noqa: F401
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    torch.ops.load_library(str(args.library.resolve()))
    print("实际加载共享库:", args.library.resolve(), flush=True)

    case = bench.make_case(args.rows, args.topk, args.capacity,
                           args.max_token, args.scenario, args.seed)
    initial = bench.make_echo_initial(case)
    expected = bench.reference_echo(case, initial)
    inputs, initial_dev, states = bench.prepare_npu(torch, case, initial, device, None)
    state = states["echo"]

    # 与 audit 相同的流程: 恢复 → 污染输出/workspace → 单次调用
    bench.restore_state(torch, "echo", initial_dev, state)
    for key in bench.ECHO_OUTPUT:
        state[key].fill_(255 if key == "miss_mask" else -777)
    state["workspace"].fill_(bench.I32_MAX)
    bench.invoke(torch, "echo", case, inputs, state)
    torch.npu.synchronize()

    print(f"scenario={args.scenario} 单次调用完成（无 trap），逐字段比对：", flush=True)
    for key in list(bench.ECHO_STATE) + list(bench.ECHO_OUTPUT):
        actual = state[key].cpu().numpy()
        ref = expected[key]
        if actual.shape != ref.shape:
            print(f"  {key:16s} SHAPE MISMATCH {actual.shape} vs {ref.shape}", flush=True)
            continue
        bad = (actual != ref)
        n = int(bad.sum())
        line = f"  {key:16s} mismatch={n}/{actual.size}"
        if n:
            idx = np.argwhere(bad)[:3]
            samples = ", ".join(
                f"{tuple(i)}: got={actual[tuple(i)]} want={ref[tuple(i)]}" for i in idx)
            line += f"   首例 {samples}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
