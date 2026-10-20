#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Echo 分步耗时测试：读取 ECHO_PERF_STEP 构建写入 workspace 的每核每标记点 clock() 绝对时间戳。

[v2.1-prof 2026-10-20] 打点语义升级（deep-dive §9.23）：

  - kernel 侧 PerfEnd 只在步末记 t0 的 clock() 低 32 位（int32 绝对时间戳）；本脚本
    用 numpy uint32 视图做回绕差分还原每步耗时（单次 invocation 跨度 ~160K cyc
    ≪ 2^31，回绕安全）。步耗时 = t[i+1] - t[i]，步间间隙自然归属到下一步。
    槽位 27 = entry（slot 0，Process 进入）+ 非 spec 18 步 + spec 8 步
    （名表尾部陈列；spec 构建下实际插在 Step0.SyncGrid 之后，本基准不覆盖）。
  - 频率标定修正：默认常数 1.585 GHz（msprof 实测主频，§10.3①：aiv_total_cycles
    为全核和，4.22M cyc / 32 核 / 83.5 μs，DVFS 无降频）。**旧的 Event P50 标定行
    已废弃**——Event 计时含 ~600 μs 级固定开销（§7.8），曾输出 141-245 MHz 假数；
    Event P50 仍测量并打印，但注明不可信、不参与任何换算。--p50-us 给外部锚点
    （profile_kernel.sh 的 kernel 级 P50）时改用 ΣΔ/P50 标定，并打印两档对照。
  - 对账预期：v2.1 旧口径分步 Σ 只覆盖单核 ~131.5K cyc 的 94%（msprof，full_cache），
    缺口 = 未打点 prologue ~10K + 槽间隙；v2.1-prof 纳入打点后覆盖 ~100%。[v2.2]
    prologue 两槽（OpZeroFlags + SyncGrid）已随 flags 世代号化删除（上板实测合计
    13.7-13.9K cyc、占跨度 8.5-11.9%），槽位 29→27；Init/kernel 进出仍在 slot 0
    之外（残余缺口即该项，见 §9.23）。
  - 新增 skew 统计：每步 32 核 max−min 的中位（核间不均衡首次可见）。
  - 新增 --matrix <prefix>：每场景导出 <prefix>_<scenario>_abs.csv（行=块/核，
    列=27 槽原始绝对戳）与 <prefix>_<scenario>_steps.csv（行=块，列=26 步耗时 +
    末列 entry→末戳总跨度），供 prof_report.py 生成 HTML 报告。非 spec 构建下
    8 个 spec 步列恒为空。

前置条件（与 compare.py 的正式测量分开构建，互不覆盖）：

    source /usr/local/Ascend/cann-9.1.0/set_env.sh
    ECHO_PERF_STEP=ON bash build.sh --soc=ascend950   # 产物在 build_perf/
    python3 echo_perf_step.py --device 1

测试数据与 compare.py 完全一致：直接 import compare.py 的 make_case /
make_echo_initial / reference_echo / prepare_npu / invoke / restore_state，
同尺寸、同 seed、同场景、同状态恢复与调用方式，因此分步耗时的各步之和
可与外部 kernel 级 P50（profile_kernel.sh / --p50-us 锚点）互相印证。结果正确性
在测量前后均与 reference_echo 全量比对（本脚本是诊断工具，不是正式性能结论；
打点构建会比普通构建略慢，正式对比仍用 compare.py + bash build.sh）。

kernel 侧打点（§9.23 槽位表）：PerfEntry 写 slot 0（entry），此后每个步末
PerfEnd 先自增 perfStepIdx_ 再写绝对戳；哨兵 -12345 判定"该槽未被 kernel 写入"
（绝对戳 int32 高位可负，只能等值判定，不能再沿用旧差值语义的 "<0 即异常"）。
"""
from __future__ import annotations

import argparse
import csv
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

# [v2.1-prof] 槽位名表（27 项，顺序即槽位序 = kernel 侧 perfStepIdx_ 写入序）：
#   slot 0  = entry（Process 进入；Init/kernel 进出开销在其外）
#   slot 1..18 = 非 spec 18 步（Step0×5, Step1×2, Step2×4, Step3×2, Step4×2, Step5×3）
#   slot 19..26 = 8 个 spec 步名（陈列位）
#   （[v2.2] prologue.OpZeroFlags / prologue.SyncGrid 两槽已随 flags 世代号化删除；
#     spec 的 OpFillZero 从不打点，耗时并入 entry→Step0 首 Op 的差分）
# ⚠ spec 构建下 8 个 spec 戳按 kernel 现行约定（v2 起）插在 Step0.SyncGrid 之后、
#   Step1 之前——顺序计数器语义下实际落在 slot 6..13，后续非 spec 步顺延至 14..26，
#   名表 19..26 并非其真实槽位。本基准 spec_enabled=False（compare.py invoke 固定
#   传 False），spec 8 槽恒为哨兵；读回检测到 spec 构建时直接报错（spec 场景的
#   分步报告属未来工作，§10.3⑥）。
ECHO_PERF_STEP_NAMES = [
    "entry",
    "Step0.OpResetRows", "Step0.OpResetDthLru", "Step0.OpResetFree", "Step0.OpResetAvailFifo",
    "Step0.SyncGrid",
    "Step1.OpHitMiss", "Step1.SyncGrid",
    "Step2.OpClassifyEvict", "Step2.SyncGrid", "Step2.OpFreeRelease", "Step2.OpFreePush",
    "Step3.OpAllocPop", "Step3.SyncGrid",
    "Step4.OpAllocAppend", "Step4.OpFifoAvailUpdate",
    "Step5.OpMissCumsum", "Step5.SyncGrid", "Step5.OpSetUpdate",
    "Step0.5.OpInvalidateSpecSuffix", "Step0.5.SyncGrid", "Step0.5.OpSpecCumsum",
    "Step0.5.SyncGrid", "Step0.5.OpSpecPushFree", "Step0.5.SyncGrid",
    "Step0.5.OpSpecAvailAdd", "Step0.5.SyncGrid",
]
assert ECHO_PERF_NUM_STEPS == len(ECHO_PERF_STEP_NAMES) == 27, (
    f"ECHO_PERF_NUM_STEPS={ECHO_PERF_NUM_STEPS} 与名表 {len(ECHO_PERF_STEP_NAMES)} 项不符（应为 27 = "
    "entry + 非 spec 18 + spec 8），请同步 tiling.h / kernel.h 步表 / 本名表")

# 非 spec 构建写入的槽数（entry + 18 步末 = 19 个绝对戳 → 18 个步耗时）。
ECHO_PERF_MAIN_STAMPS = 19
ECHO_PERF_SPEC_STEPS = 8
assert ECHO_PERF_MAIN_STAMPS + ECHO_PERF_SPEC_STEPS == ECHO_PERF_NUM_STEPS
# 非 spec 构建下有效步耗时名（槽 1..18 的差分，= 名表切片）。
ECHO_PERF_DURATION_NAMES = ECHO_PERF_STEP_NAMES[1:ECHO_PERF_MAIN_STAMPS]
assert len(ECHO_PERF_DURATION_NAMES) == 18

# [v2.1-prof] 默认频率常数：msprof 实测主频 1.585 GHz（§10.3①，DVFS 无降频）。
#   旧口径用每轮 Σ步周期 / Event 微秒标定——Event 含 ~600 μs 级固定开销（§7.8），
#   输出 141-245 MHz 假数，已废弃。
ECHO_FREQ_HZ = 1.585e9

SENTINEL = -12345  # 绝对戳 int32 高位可负（uint32 低 32 位回绕存储），哨兵只认等值；
                   # 与真实戳碰撞概率 2^-32/槽，忽略。
                   # [v2.2] 采样前全 workspace 填本哨兵（含 flags 区）：-12345 ≡ 3 (mod 4)
                   # 恒在 flags 世代号窗 {4k+1,4k+2} 之外 → 污染免疫，与 compare.py audit 的
                   # I32_MAX 同理。


def echo_step_region_offset(case, blocks):
    """workspace 内打点区的 int32 偏移，与 EchoLruKernel::Init / EchoWorkspaceElements 一致。

    [v2] 每行段 = wsCumsum(B+1) + 4 标量 + 3*topk 向量 + flags(B+1)；对齐后即
    打点区基址（v1 的 argtopk 诊断 dump 区已删除）。偏移与槽数无关
    （区大小 = blocks * ECHO_PERF_NUM_STEPS 随 tiling.h 常量走）。
    """
    rows, topk, capacity = (case[k] for k in ("rows", "topk", "capacity"))
    base = rows * (2 * (capacity + 1) + 4 + 3 * topk)
    return (base + 1) & ~1


def read_step_stamps(state, case, blocks):
    """读回 blocks×ECHO_PERF_NUM_STEPS 的 int32 绝对时间戳矩阵（打点区原样）。"""
    region = state["workspace"][echo_step_region_offset(case, blocks):
                                echo_step_region_offset(case, blocks) + blocks * ECHO_PERF_NUM_STEPS]
    return region.cpu().numpy().reshape(blocks, ECHO_PERF_NUM_STEPS)


def validate_stamps(matrix, where):
    """非 spec 构建的打点区校验：槽 0..18 全写、槽 19..26 恒为哨兵。

    绝对戳按 int32 回绕存储、高位可负——v2 差值语义时代的 "<0 即异常" 防御在此
    失效（会把一半合法戳误判为异常），哨兵只认等值 -12345。
    """
    if (matrix[:, :ECHO_PERF_MAIN_STAMPS] == SENTINEL).any():
        raise RuntimeError(
            f"{where}: 打点区槽 0..{ECHO_PERF_MAIN_STAMPS - 1} 存在哨兵（未完整写入）。"
            "请确认加载的是 ECHO_PERF_STEP=ON 构建（ECHO_PERF_STEP=ON bash build.sh），"
            "而不是普通 build/ 产物。")
    if not (matrix[:, ECHO_PERF_MAIN_STAMPS:] == SENTINEL).all():
        raise RuntimeError(
            f"{where}: 槽 {ECHO_PERF_MAIN_STAMPS}..{ECHO_PERF_NUM_STEPS - 1} 非全哨兵 —— "
            "检测到 spec 构建（specEnabled=true）的打点。spec 8 戳按 kernel 现行约定插在 "
            "Step0.SyncGrid 之后（实际槽位 6..13，见名表注释），本脚本的名表/差分按非 spec "
            "槽位命名，spec 场景的分步报告属未来工作（§10.3⑥）。")


def stamp_durations(matrix):
    """uint32 回绕差分：每块 18 个步耗时（列 k = t[k+1]-t[k]）+ entry→末戳跨度。

    返回 (durations[blocks,18] int64, spans[blocks] int64)。逐块看步耗时之和
    == 该块跨度（望远镜恒等式）——对账以跨度为准，跨核逐峰求和只作参考。
    """
    u = np.ascontiguousarray(matrix[:, :ECHO_PERF_MAIN_STAMPS], dtype=np.int32).view(np.uint32)
    d = (u[:, 1:] - u[:, :-1]).astype(np.int64)  # uint32 减法回绕安全
    if (d >= 0x80000000).any():
        raise RuntimeError("步差分出现 ≥2^31 的值：打点区被污染或槽位语义不一致")
    spans = (u[:, -1] - u[:, 0]).astype(np.int64)
    if (spans >= 0x80000000).any():
        raise RuntimeError("entry→末戳跨度 ≥2^31：打点区被污染或槽位语义不一致")
    return d, spans


def write_matrix_csvs(args, scenario, blocks, rep_matrix, freq_hz, freq_source):
    """--matrix 导出：代表轮（最慢核跨度最接近中位）的两个 CSV，行=块可互相对照。

    <prefix>_<scenario>_abs.csv    原始绝对戳（含哨兵列，忠实于内存）
    <prefix>_<scenario>_steps.csv  26 步耗时（名表序）+ entry→末戳跨度；非 spec
                                   构建下 8 个 spec 步列为空
    首行 '#' 注释记录口径（生成命令/代表轮/频率），prof_report.py 读取时跳过。
    """
    durations, spans = stamp_durations(rep_matrix)
    abs_path = Path(f"{args.matrix}_{scenario}_abs.csv")
    steps_path = Path(f"{args.matrix}_{scenario}_steps.csv")
    note = (f"# echo_perf_step v2.1-prof 绝对戳矩阵; scenario={scenario}; blocks={blocks}; "
            f"freq={freq_hz / 1e9:.4f}GHz ({freq_source}); 槽位名表=echo_perf_step.ECHO_PERF_STEP_NAMES; "
            f"哨兵={SENTINEL} 表示未写")
    with abs_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([note])
        w.writerow(["block"] + ECHO_PERF_STEP_NAMES)
        for b in range(blocks):
            w.writerow([b] + [int(v) for v in rep_matrix[b]])
    with steps_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([note + "; 非-spec 构建下 8 个 Step0.5 列恒为空"])
        w.writerow(["block"] + ECHO_PERF_STEP_NAMES[1:] + ["span_entry_to_last"])
        for b in range(blocks):
            row = [b] + [int(v) for v in durations[b]]
            row += [""] * ECHO_PERF_SPEC_STEPS  # spec 步列（名表尾部陈列位）恒为空
            row.append(int(spans[b]))
            w.writerow(row)
    return abs_path, steps_path


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
    parser.add_argument("--p50-us", type=float, default=None,
                        help="外部锚点：kernel 级 P50（μs，来自 profile_kernel.sh）。给了则用 "
                             "ΣΔ(最慢核跨度中位)/P50 标定频率并打印两档对照；默认用 1.585 GHz 常数")
    parser.add_argument("--matrix", type=Path, default=None,
                        help="导出前缀：每场景写 <prefix>_<scenario>_abs.csv 与 "
                             "<prefix>_<scenario>_steps.csv（代表轮，行=块），供 prof_report.py")
    parser.add_argument("--json", type=Path, default=Path("results/echo_perf_step.json"))
    args = parser.parse_args()
    if not (args.rows > 0 and 0 < args.topk <= args.capacity <= 4096 and args.max_token > args.topk):
        parser.error("需要 rows>0、0<topk<=capacity<=4096、max_token>topk")
    if args.warmup < 0 or args.repeat <= 0:
        parser.error("warmup>=0、repeat>0")
    if args.p50_us is not None and not (args.p50_us > 0):
        parser.error("--p50-us 必须为正数（μs）")
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
        "note": "ECHO_PERF_STEP 打点构建（v2.1-prof：每标记点 t0 绝对 clock 低 32 位，host 回绕"
                "差分；非正式性能结论，正式对比用 compare.py）",
        "perf_semantics": "v2.2 absolute stamps (slot 0=entry, 1..18=non-spec 18, 19..26=spec names; "
                          "prologue slots removed with flags generation numbering)",
        "shape": {k: getattr(args, k) for k in ("rows", "topk", "capacity", "max_token")},
        "seed": args.seed, "warmup": args.warmup, "repeat": args.repeat,
        "echo_perf_num_steps": ECHO_PERF_NUM_STEPS,
        "main_stamps": ECHO_PERF_MAIN_STAMPS,
        "spec_steps": ECHO_PERF_SPEC_STEPS,
        "step_names": ECHO_PERF_STEP_NAMES,
        "duration_names": ECHO_PERF_DURATION_NAMES,
        "freq_hz_default": ECHO_FREQ_HZ,
        "p50_us_anchor": args.p50_us,
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

        # A. 审计一次：污染输出与 workspace → 调用 → 全量比对参考；同时确认打点确实写入
        #    且为非 spec 槽位形态（槽 0..18 全写、spec 8 槽全哨兵）。
        restore_state(torch, "echo", initial, state)
        for key in ECHO_OUTPUT:
            state[key].fill_(255 if key == "miss_mask" else -777)
        state["workspace"].fill_(SENTINEL)
        invoke(torch, "echo", case, inputs, state)
        torch.npu.synchronize()
        check_npu_result("echo", state, expected, f"{scenario}/audit")
        validate_stamps(read_step_stamps(state, case, blocks), f"{scenario}/audit")
        miss_count = args.topk - case["hits"]
        free_size = max(miss_count - int(echo_initial["avail"][0]), 0)
        print(f"{scenario}: miss/行={miss_count}, 需淘汰={free_size}, blocks={blocks}", flush=True)

        # B. 预热（不读回，不打哨兵，保持与正式调用相同的缓存条件）。
        for _ in range(args.warmup):
            restore_state(torch, "echo", initial, state)
            invoke(torch, "echo", case, inputs, state)
        torch.npu.synchronize()

        # C. 正式采样：哨兵 → 恢复 → Event 计时调用（仅打印参考，§7.8 不可信）→ 读回绝对戳。
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        samples = []
        matrices = []  # 每轮 blocks×27 绝对戳矩阵（--matrix 代表轮挑选用；不进 JSON）
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
            matrix = read_step_stamps(state, case, blocks)
            validate_stamps(matrix, f"{scenario}/sample")
            durations, spans = stamp_durations(matrix)
            matrices.append(matrix)
            samples.append({
                "event_us": event_us,
                "step_max_cycles": durations.max(axis=0).tolist(),
                "step_skew_cycles": (durations.max(axis=0) - durations.min(axis=0)).tolist(),
                "span_max_cycles": int(spans.max()),
            })

        # D. 测量后全量校验（本脚本失败即退出，不产出性能结论）。
        check_npu_result("echo", state, expected, f"{scenario}/after-measurement")

        # 统计：每步取"最慢核"（max over blocks）的跨轮中位数；skew = 每轮 32 核
        #   max−min 的中位（核间不均衡）。对账以"最慢核跨度中位"为准（逐块步耗时
        #   之和 == 该块跨度，望远镜恒等式；跨核逐峰求和只作参考）。
        step_max = np.array([s["step_max_cycles"] for s in samples], dtype=np.float64)  # [repeat, 18]
        step_skew = np.array([s["step_skew_cycles"] for s in samples], dtype=np.float64)
        span_max = np.array([s["span_max_cycles"] for s in samples], dtype=np.float64)
        event_us = np.array([s["event_us"] for s in samples])
        median_cycles = np.median(step_max, axis=0)
        skew_cycles = np.median(step_skew, axis=0)
        span_median = float(np.median(span_max))
        sum_step_median = float(median_cycles.sum())

        # 频率口径：默认 1.585 GHz 常数（msprof 实测）；--p50-us 锚点给了则
        #   freq = ΣΔ(最慢核跨度中位)/P50（跨度不含 Init/kernel 进出，锚点口径
        #   会略高——两档对照即暴露该差）。
        freq_hz = ECHO_FREQ_HZ
        freq_source = "const 1.585GHz (msprof §10.3①)"
        if args.p50_us is not None:
            freq_hz = span_median / (args.p50_us * 1e-6)
            freq_source = f"anchor ΣΔ/P50 ({span_median:.0f}cyc/{args.p50_us}μs)"
        est_us = median_cycles / freq_hz * 1e6
        pct = median_cycles / sum_step_median * 100.0

        print(f"  频率口径: {freq_source} → {freq_hz / 1e9:.4f} GHz"
              + (f"（默认档 {ECHO_FREQ_HZ / 1e9:.3f} GHz 对照）" if args.p50_us is not None else
                 "（--p50-us 可给外部锚点对照）"), flush=True)
        if args.p50_us is not None:
            print(f"    两档对照: 最慢核跨度中位 {span_median:.0f} cyc ≈ "
                  f"{span_median / ECHO_FREQ_HZ * 1e6:.2f} μs @1.585GHz / "
                  f"{span_median / freq_hz * 1e6:.2f} μs @锚点", flush=True)
        print(f"  对账: 最慢核跨度(entry→末戳)中位 {span_median:.0f} cyc ≈ "
              f"{span_median / freq_hz * 1e6:.2f} μs; Σ步max中位 {sum_step_median:.0f} cyc "
              f"(跨核逐峰求和, 仅参考; msprof 单核均值 ~131.5K cyc @full_cache, §10.3①); "
              f"Event P50 {np.median(event_us):.2f} μs (§7.8: 含 ~600μs 级固定开销, 仅参考不参与标定)",
              flush=True)
        for idx in range(len(ECHO_PERF_DURATION_NAMES)):
            print(f"    [{idx + 1:2d}] {ECHO_PERF_DURATION_NAMES[idx]:<26s} "
                  f"{median_cycles[idx]:>10.0f} cyc  skew {skew_cycles[idx]:>7.0f} cyc  "
                  f"{pct[idx]:5.1f}%  ≈{est_us[idx]:7.2f} μs", flush=True)
        order = np.argsort(median_cycles)[::-1]
        print("  热点 Top5: " + ", ".join(
            f"{ECHO_PERF_DURATION_NAMES[i]}({median_cycles[i]:.0f}c)" for i in order[:5]), flush=True)

        # --matrix：代表轮 = 最慢核跨度最接近中位的样本（两 CSV 同轮，行=块可对照）。
        matrix_paths = None
        if args.matrix is not None:
            rep = int(np.argmin(np.abs(span_max - span_median)))
            matrix_paths = write_matrix_csvs(args, scenario, blocks, matrices[rep],
                                             freq_hz, freq_source)
            print(f"  matrix 导出(代表轮 {rep}/{args.repeat}): "
                  + ", ".join(str(p) for p in matrix_paths), flush=True)

        results["scenarios"][scenario] = {
            "blocks": blocks,
            "expected_misses_per_row": miss_count,
            "free_size_per_row": free_size,
            "event_us": {"p50": float(np.median(event_us)), "min": float(event_us.min()),
                         "max": float(event_us.max()),
                         "note": "§7.8: Event 含 ~600μs 级固定开销，仅参考，不参与频率标定"},
            "freq_hz": freq_hz, "freq_source": freq_source,
            "span_max_cycles": {"p50": span_median, "min": float(span_max.min()),
                                "max": float(span_max.max())},
            "sum_step_max_median_cycles": sum_step_median,
            "steps": [
                {"idx": i + 1, "name": ECHO_PERF_DURATION_NAMES[i],
                 "median_cycles": float(median_cycles[i]),
                 "min_cycles": float(step_max[:, i].min()),
                 "max_cycles": float(step_max[:, i].max()),
                 "skew_cycles": float(skew_cycles[i]),
                 "pct_of_sum": float(pct[i]),
                 "est_us": float(est_us[i])}
                for i in range(len(ECHO_PERF_DURATION_NAMES))
            ],
            "raw_samples": samples,
            "matrix": {"representative_repeat": (int(np.argmin(np.abs(span_max - span_median)))
                                                 if args.matrix is not None else None),
                       "abs_csv": str(matrix_paths[0]) if matrix_paths else None,
                       "steps_csv": str(matrix_paths[1]) if matrix_paths else None},
            "correctness": "passed",
            "case_sha256": array_hash({k: v for k, v in case.items() if isinstance(v, np.ndarray)}),
        }

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("结果与原始样本:", args.json.resolve())
    if args.matrix is not None:
        print("HTML 报告: python3 prof_report.py --matrix <prefix>_<scenario> "
              "[--op-summary op_summary_*.csv ...] --out report.html")


if __name__ == "__main__":
    main()
