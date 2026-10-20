#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""prof_report.py — Echo v2.1-prof 分步打点的单文件 HTML 报告器（owner 在服务器生成、浏览器打开）。

把 echo_perf_step.py --matrix 导出的两个 CSV（绝对戳矩阵 + 步耗时矩阵）渲染成
一份自包含 HTML：总览与对账条、逐核×逐步热力图、逐步聚合条形、（可选）msprof
op 级表、原始数据折叠表。单文件内联 CSS/JS，**无外部依赖、无网络请求**， scp 回
本地双击即看。

用法示例（服务器）：

    # 1) 采分步数据并导出矩阵（代表轮；--p50-us 换成本轮 profile_kernel.sh 的 kernel 级 P50）
    ECHO_PERF_STEP=ON bash build.sh
    python3 echo_perf_step.py --device 0 --scenario full_cache_75pct \
        --matrix results/perf --p50-us 82.12

    # 2) 生成 HTML 报告（--matrix 传"前缀_场景"，与上一步产物名对齐；
    #    --op-summary 可多次给 msprof task-based 的 op_summary_*.csv）
    python3 prof_report.py --matrix results/perf_full_cache_75pct \
        --op-summary /dev/shm/prof_out/profiler/.../op_summary_*.csv \
        --p50-us 82.12 --out results/perf_full_cache_75pct.html

    # 3) 浏览器打开 results/perf_full_cache_75pct.html（单文件，无需网络）

页面结构：
    §1 总览    —— 场景元信息 + 统计瓦片 + 对账条（打点跨度 vs msprof 时长 vs 外部 P50 锚点，
                  缺口即 Init/kernel 进出等打点外开销，§9.23 对账口径）
    §2 热力图  —— 行=核（blockIdx），列=26 步（名表序），色=该步该核耗时（顺序色标，
                  cyc/μs 可切换，悬停显示数值；8 个 Step0.5 列在非 spec 构建下未写=灰）
    §3 逐步聚合 —— 每步 32 核分布：P50 条形（max 作同色浅轨道、min 刻度）+ skew 列（μs 轴）
    §4 op 级表 —— （若给 --op-summary）kernel 名 / Duration / aiv_vec_ratio / scalar /
                  mte2 / mte3 / icache，echo 行高亮；列名容错嗅探，匹配不到显示 '-'
    §5 原始数据 —— 两个 CSV 的表格形式（= 各图的无悬停/键盘可达等价物）

设计原则：这是给 owner 人工看的诊断报告，宁可朴素准确、不要花哨；每张图标注数据
来源与口径。配色沿用 dataviz skill 的已验证参考调色板（light 模式）——项目铁律
禁止本机执行 node 校验脚本，无法对自改配色跑 validate_palette.js，故原样引用其
验证结论（顺序色阶=蓝 100..700 单 hue；文本/网格用 chrome 色，数据色不用于文字）。
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from datetime import datetime
from html import escape
from pathlib import Path
from statistics import median

# 默认换算频率：msprof 实测主频 1.585 GHz（§10.3①）；--freq 可覆盖。
DEFAULT_FREQ_HZ = 1.585e9

# ============================ 调色板（dataviz skill 参考实例，light 模式） ============================
# [为什么原样引用] 见模块 docstring：自改配色需重跑 validate_palette.js（本机禁跑），
#   参考调色板本身已通过全部硬性检查（CVD ΔE≥8 / 正常视觉 ΔE≥15 / 对比度门槛）。
SEQ_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SLOT1 = "#2a78d6"        # 分类槽 1（单系列条形/对账条的默认色）
SLOT1_WASH = "#cde2fb"   # 同 ramp 最浅步（max 轨道 / meter 未填充段）
SURFACE = "#fcfcfb"      # 卡片（图表表面）
PAGE = "#f9f9f7"         # 页面底色
INK = "#0b0b0b"          # 主文字
INK2 = "#52514e"         # 次文字
MUTED = "#898781"        # 轴/弱化标签
GRID = "#e1e0d9"         # 网格 hairline；兼作热力图"未写"格（无数据 ≠ 数据）
BASELINE = "#c3c2b7"     # 轴线/min 刻度
BORDER = "rgba(11,11,11,0.10)"

# msprof op_summary 的行过滤与列匹配（容错：各版本列名/单位不统一，仓内无样例可对；
#   匹配不到的列显示 '-'，不猜单位不换算——诚实优先）。
KERNEL_FILTER = re.compile(r"echo_lru_kernel|sparse_kv_plan_kernel|TensorMove", re.I)
OP_COL_PATTERNS = [
    ("name", re.compile(r"(kernel|op).{0,6}name|^name$", re.I)),
    ("duration", re.compile(r"duration", re.I)),
    ("vec", re.compile(r"vec", re.I)),
    ("scalar", re.compile(r"scalar", re.I)),
    ("mte2", re.compile(r"mte2", re.I)),
    ("mte3", re.compile(r"mte3", re.I)),
    ("icache", re.compile(r"icache", re.I)),
]


def esc(s) -> str:
    return escape(str(s), quote=True)


def fmt_int(v) -> str:
    return f"{int(round(v)):,}"


def fmt_us(v, nd=2) -> str:
    return f"{v:.{nd}f}"


def cyc_to_us(cyc: float, freq_hz: float) -> float:
    return cyc / freq_hz * 1e6


def nice_ticks(vmax: float, target: int = 5):
    """干净的轴刻度（0 起，1/2/2.5/5×10^k 步进）。"""
    if vmax <= 0:
        return [0.0]
    raw = vmax / target
    mag = 10.0 ** math.floor(math.log10(raw)) if raw > 0 else 1.0
    step = 10 * mag
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            step = m * mag
            break
    ticks, t = [], 0.0
    while t <= vmax * 1.0001:
        ticks.append(round(t, 6))
        t += step
    return ticks


def seq_color(v: float, vmax: float) -> str:
    """量级 → 单 hue 顺序色阶（light→dark，最浅=接近零）。"""
    if vmax <= 0:
        return SEQ_RAMP[0]
    t = max(0.0, min(1.0, v / vmax))
    return SEQ_RAMP[min(len(SEQ_RAMP) - 1, int(t * len(SEQ_RAMP)))]


def bar_path(x0: float, y: float, w: float, h: float, r: float = 4.0) -> str:
    """条形 path：数据端 4px 圆角、基线端方角（marks 规范）。"""
    if w <= 0:
        w = 0.001
    r = min(r, w / 2.0, h / 2.0)
    return (f"M{x0:.2f} {y:.2f} H{x0 + w - r:.2f} "
            f"A{r:.2f} {r:.2f} 0 0 1 {x0 + w:.2f} {y + r:.2f} "
            f"V{y + h - r:.2f} A{r:.2f} {r:.2f} 0 0 1 {x0 + w - r:.2f} {y + h:.2f} "
            f"H{x0:.2f} Z")


# ============================ 数据读入 ============================

def read_csv_skip_comments(path: Path):
    """读 CSV，跳过 '#' 注释行（echo_perf_step 导出带口径注释；msprof 导出常带 BOM）。"""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.reader(f)
                if r and not r[0].lstrip().startswith("#")]
    if not rows:
        raise SystemExit(f"CSV 为空或全是注释行: {path}")
    return rows[0], rows[1:]


def load_matrix(prefix: str) -> dict:
    """读 <prefix>_abs.csv 与 <prefix>_steps.csv（echo_perf_step --matrix 的产物）。"""
    abs_path, steps_path = Path(f"{prefix}_abs.csv"), Path(f"{prefix}_steps.csv")
    for p in (abs_path, steps_path):
        if not p.is_file():
            raise SystemExit(f"找不到 {p} —— 先运行 echo_perf_step.py --matrix <prefix> 生成两个 CSV")
    hdr, rows = read_csv_skip_comments(abs_path)
    if len(hdr) < 3 or hdr[0] != "block":
        raise SystemExit(f"{abs_path} 表头异常（应为 block + 27 槽名）: {hdr[:3]}")
    slot_names = hdr[1:]
    abs_vals = [[int(c) for c in r[1:]] for r in rows]
    hdr2, rows2 = read_csv_skip_comments(steps_path)
    if len(hdr2) < 3 or hdr2[0] != "block" or hdr2[-1] != "span_entry_to_last":
        raise SystemExit(f"{steps_path} 表头异常（应为 block + 26 步名 + span_entry_to_last）: {hdr2[:3]}")
    step_names = hdr2[1:-1]
    if len(step_names) + 1 != len(slot_names):
        raise SystemExit(f"两 CSV 列数不一致: abs {len(slot_names)} 槽 vs steps {len(step_names)} 步")
    steps_vals, spans = [], []
    for r in rows2:
        cells = r[1:]
        if len(cells) != len(step_names) + 1:
            raise SystemExit(f"{steps_path} 行列数异常: {len(cells)} != {len(step_names) + 1}")
        steps_vals.append([int(c) if c.strip() != "" else None for c in cells[:-1]])
        spans.append(int(cells[-1]))
    if len(steps_vals) != len(abs_vals):
        raise SystemExit(f"两 CSV 行数不一致: abs {len(abs_vals)} vs steps {len(steps_vals)}")
    # 列级"已写"判定 + 与 abs 哨兵交叉校验（steps 空列 ↔ abs 槽应全为 -12345 哨兵）。
    written = [all(row[j] is not None for row in steps_vals) for j in range(len(step_names))]
    warnings = []
    for j, w in enumerate(written):
        if w:
            continue
        if any(row[j + 1] != -12345 for row in abs_vals):
            warnings.append(f"列 {j + 1}（{step_names[j]}）steps 为空但 abs 槽非全哨兵，数据可疑")
    if not any(written):
        raise SystemExit("steps CSV 无任何已写列——数据异常")
    return {"abs_path": abs_path, "steps_path": steps_path, "slot_names": slot_names,
            "step_names": step_names, "abs": abs_vals, "steps": steps_vals,
            "spans": spans, "blocks": len(rows), "written": written, "warnings": warnings}


def parse_number(s):
    if s is None:
        return None
    s = str(s).strip().replace(",", "").replace("%", "")
    if s in ("", "-", "N/A", "NA", "n/a", "null"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_op_summary(path: Path):
    """容错解析 msprof task-based op_summary CSV：表头嗅探 + 列名模糊匹配。

    返回 (记录列表, 列映射, 表头文本, 表头是否嗅探成功)。记录只保留
    echo_lru_kernel / sparse_kv_plan_kernel / TensorMove 行。
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        all_rows = [r for r in csv.reader(f) if r and any(c.strip() for c in r)]
    if not all_rows:
        return [], {}, [], False
    header_idx, sniffed = 0, False
    for i, r in enumerate(all_rows[:40]):
        cells = [c for c in r if c.strip()]
        if len(cells) >= 3 \
                and any(re.search(r"(kernel|op).{0,6}name|^name$", c, re.I) for c in cells) \
                and any(re.search(r"dur|time|cyc", c, re.I) for c in cells):
            header_idx, sniffed = i, True
            break
    header, data_rows = all_rows[header_idx], all_rows[header_idx + 1:]
    used, colmap = set(), {}
    for key, pat in OP_COL_PATTERNS:
        for idx, h in enumerate(header):
            if idx in used or not pat.search(h):
                continue
            colmap[key], _ = idx, used.add(idx)
            break
    recs = []
    need = max(colmap.values(), default=-1)
    for r in data_rows:
        if len(r) <= need:
            continue
        name = r[colmap["name"]].strip() if "name" in colmap else ""
        if not name or not KERNEL_FILTER.search(name):
            continue
        rec = {"name": name, "source": path.name}
        for key in ("duration", "vec", "scalar", "mte2", "mte3", "icache"):
            rec[key] = parse_number(r[colmap[key]]) if key in colmap else None
        recs.append(rec)
    return recs, colmap, header, sniffed


# ============================ SVG 图 ============================

def svg_heatmap(data: dict, freq_hz: float) -> str:
    """核心页：行=核，列=26 步（名表序），色=该步该核耗时（全局线性顺序色标）。

    颜色域取全体已写格的 max（诚实呈现热点集中度；小值接近表面=接近零）；
    单位切换只影响 tooltip/图例文本，不影响颜色（μs 是 cyc 的线性变换）。
    """
    blocks, names = data["blocks"], data["step_names"]
    cw, ch, gap = 18, 16, 2
    pw, ph = cw + gap, ch + gap
    left, top = 46, 26
    width = left + len(names) * pw + 10
    height = top + blocks * ph + 10
    vmax = max((v for row in data["steps"] for v in row if v is not None), default=0)
    p = [f'<svg class="chart" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
         f'role="img" aria-label="逐核逐步耗时热力图，{blocks} 核 × {len(names)} 步">']
    for j in range(len(names)):
        p.append(f'<text x="{left + j * pw + cw / 2:.1f}" y="{top - 8}" '
                 f'text-anchor="middle" class="axnum">{j + 1}</text>')
    p.append(f'<text x="{left - 8}" y="{top - 8}" text-anchor="end" class="axnum">核\\步</text>')
    for b in range(blocks):
        y = top + b * ph
        p.append(f'<text x="{left - 6}" y="{y + ch / 2 + 3.5:.1f}" '
                 f'text-anchor="end" class="axnum">{b}</text>')
        for j in range(len(names)):
            v = data["steps"][b][j]
            x = left + j * pw
            if v is None:
                p.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" rx="2" fill="{GRID}" '
                         f'data-line="未写（非 spec 构建恒为空）" data-sub="{j + 1}. {esc(names[j])}"/>')
            else:
                p.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" rx="2" '
                         f'fill="{seq_color(v, vmax)}" data-cyc="{v}" '
                         f'data-name="{j + 1}. {esc(names[j])} · 核 {b}"/>')
    p.append("</svg>")
    # 色标图例：13 段 ramp + 未写灰（范围文本随单位切换显隐）。
    sw = "".join(f'<i style="background:{c}"></i>' for c in SEQ_RAMP)
    p.append('<div class="legend"><span class="lg-t">耗时</span>'
             f'<span class="sw-row">{sw}</span>'
             f'<span class="unit-cyc">0 … {fmt_int(vmax)} cyc</span>'
             f'<span class="unit-us" hidden>0 … {fmt_us(cyc_to_us(vmax, freq_hz))} μs</span>'
             f'<span class="lg-gap"></span><i style="background:{GRID}"></i><span>未写</span></div>')
    return "".join(p)


def step_stats(data: dict):
    """每步 32 核分布（代表轮口径）：P50/max/min/skew。"""
    out = []
    for j, name in enumerate(data["step_names"]):
        vals = [row[j] for row in data["steps"] if row[j] is not None]
        if not vals:
            out.append({"idx": j + 1, "name": name, "p50": None})
            continue
        mx, mn = max(vals), min(vals)
        out.append({"idx": j + 1, "name": name, "p50": median(vals),
                    "max": mx, "min": mn, "skew": mx - mn, "n": len(vals)})
    return out


def svg_step_bars(data: dict, freq_hz: float) -> str:
    """逐步聚合：P50 条形（max 同色浅轨道、min 刻度）+ skew 列，μs 轴。"""
    stats = step_stats(data)
    namew, plotw, skeww, rh = 250, 560, 76, 24
    top, bottom = 8, 30
    height = top + len(stats) * rh + bottom
    width = namew + plotw + skeww + 14
    written = [s for s in stats if s["p50"] is not None]
    xmax = max((cyc_to_us(s["max"], freq_hz) for s in written), default=1.0) * 1.06
    xmax = max(xmax, 0.01)

    def x_of(us_v: float) -> float:
        return namew + (us_v / xmax) * plotw

    p = [f'<svg class="chart" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
         f'role="img" aria-label="逐步聚合条形图，P50/max/min/skew（μs 轴）">']
    # 网格 + x 轴（hairline、实线、recessive）。
    for t in nice_ticks(xmax):
        xt = x_of(t)
        p.append(f'<line x1="{xt:.1f}" y1="{top}" x2="{xt:.1f}" '
                 f'y2="{height - bottom + 6}" stroke="{GRID}" stroke-width="1"/>')
        lbl = f"{t:g}"
        p.append(f'<text x="{xt:.1f}" y="{height - bottom + 20}" '
                 f'text-anchor="middle" class="axnum">{lbl}</text>')
    p.append(f'<text x="{namew + plotw / 2:.0f}" y="{height - 4}" '
             f'text-anchor="middle" class="axcap">耗时（μs，@{freq_hz / 1e9:.3f} GHz）</text>')
    # 直接标注只标 Top3 热点（selective labeling），其余靠轴/悬停/表。
    top3 = sorted((s for s in written), key=lambda s: -s["p50"])[:3]
    for i, s in enumerate(stats):
        y = top + i * rh
        label = f'{s["idx"]}. {s["name"]}'
        cls = "axname" if s["p50"] is not None else "axname dim"
        p.append(f'<text x="{namew - 10}" y="{y + rh / 2 + 4:.1f}" text-anchor="end" class="{cls}">'
                 f'{esc(label)}</text>')
        if s["p50"] is None:
            p.append(f'<text x="{namew + 4}" y="{y + rh / 2 + 4:.1f}" class="axname dim">—</text>')
            continue
        p50u, maxu, minu = (cyc_to_us(s[k], freq_hz) for k in ("p50", "max", "min"))
        # max 轨道（同 ramp 最浅步）→ P50 条形（槽 1 蓝，数据端圆角）→ min 刻度。
        p.append(f'<path d="{bar_path(namew, y + 6, x_of(maxu) - namew, 12)}" fill="{SLOT1_WASH}"/>')
        p.append(f'<path d="{bar_path(namew, y + 6, x_of(p50u) - namew, 12)}" fill="{SLOT1}"/>')
        p.append(f'<rect x="{x_of(minu) - 1:.1f}" y="{y + 3}" width="2" height="18" fill="{BASELINE}"/>')
        # 整行透明命中区（≥24px，含 2px 间隙），悬停给全量数值。
        tip = (f'P50 {fmt_us(p50u)} μs · max {fmt_us(maxu)} · min {fmt_us(minu)} · '
               f'skew {fmt_us(cyc_to_us(s["skew"], freq_hz))} μs')
        sub = f'{s["idx"]}. {s["name"]} · P50 {fmt_int(s["p50"])} cyc / max {fmt_int(s["max"])} cyc'
        p.append(f'<rect x="{namew}" y="{y}" width="{plotw}" height="{rh}" fill="none" '
                 f'pointer-events="all" data-line="{esc(tip)}" data-sub="{esc(sub)}"/>')
        p.append(f'<text x="{namew + plotw + 10}" y="{y + rh / 2 + 4:.1f}" '
                 f'class="axnum" text-anchor="start">{fmt_us(cyc_to_us(s["skew"], freq_hz))}</text>')
        if s in top3:
            bx, anchor = x_of(p50u) + 6, "start"
            if bx + 84 > namew + plotw:
                bx, anchor = x_of(p50u) - 6, "end"
            p.append(f'<text x="{bx:.1f}" y="{y + 2.5}" text-anchor="{anchor}" '
                     f'class="axval">{fmt_us(p50u)} μs</text>')
    p.append(f'<line x1="{namew}" y1="{top}" x2="{namew}" y2="{height - bottom + 6}" '
             f'stroke="{BASELINE}" stroke-width="1"/>')
    p.append("</svg>")
    return "".join(p)


def svg_recon(data: dict, freq_hz: float, p50_us, msprof_us) -> tuple[str, str]:
    """对账条：打点跨度（核中位/最慢核）vs msprof echo 时长 vs 外部 P50 锚点。

    返回 (svg, 缺口注记)。所有条同测度同轴（μs，单轴），单系列槽 1 蓝；
    少量条全直接标注。
    """
    spans_us = [cyc_to_us(c, freq_hz) for c in data["spans"]]
    items = [(f"打点跨度 P50（{data['blocks']} 核中位, entry→末戳）", median(spans_us)),
             ("打点跨度 max（最慢核, 代表轮）", max(spans_us))]
    if msprof_us is not None:
        items.append(("msprof echo_lru_kernel Duration P50", msprof_us))
    if p50_us is not None:
        items.append(("外部 P50 锚点（--p50-us）", float(p50_us)))
    labelw, plotw, rh = 300, 420, 26
    top, bottom = 6, 26
    height = top + len(items) * rh + bottom
    width = labelw + plotw + 70
    xmax = max(v for _, v in items) * 1.12
    p = [f'<svg class="chart" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
         f'role="img" aria-label="对账条形图，μs 轴">']
    for t in nice_ticks(xmax):
        xt = labelw + (t / xmax) * plotw
        p.append(f'<line x1="{xt:.1f}" y1="{top}" x2="{xt:.1f}" y2="{height - bottom + 6}" '
                 f'stroke="{GRID}" stroke-width="1"/>')
        p.append(f'<text x="{xt:.1f}" y="{height - bottom + 20}" text-anchor="middle" '
                 f'class="axnum">{t:g}</text>')
    p.append(f'<text x="{labelw + plotw / 2:.0f}" y="{height - 4}" text-anchor="middle" '
             f'class="axcap">微秒（μs）</text>')
    for i, (label, v) in enumerate(items):
        y = top + i * rh
        p.append(f'<text x="{labelw - 10}" y="{y + rh / 2 + 4:.1f}" text-anchor="end" '
                 f'class="axname">{esc(label)}</text>')
        p.append(f'<path d="{bar_path(labelw, y + 7, (v / xmax) * plotw, 12)}" fill="{SLOT1}" '
                 f'data-line="{fmt_us(v)} μs" data-sub="{esc(label)}"/>')
        p.append(f'<text x="{labelw + (v / xmax) * plotw + 6:.1f}" y="{y + rh / 2 + 4:.1f}" '
                 f'class="axval">{fmt_us(v)} μs</text>')
    p.append(f'<line x1="{labelw}" y1="{top}" x2="{labelw}" y2="{height - bottom + 6}" '
             f'stroke="{BASELINE}" stroke-width="1"/>')
    p.append("</svg>")
    note = ""
    if p50_us is not None:
        gap = float(p50_us) - median(spans_us)
        note = (f"对账缺口 = 外部 P50 {fmt_us(float(p50_us))} μs − 打点跨度 P50 "
                f"{fmt_us(median(spans_us))} μs ≈ {fmt_us(gap)} μs"
                f"（Init/kernel 进出 + slot 0 之外的打点外开销，§9.23 口径；"
                f"打点跨度每核 telescoping 恒等 = 该核各步之和）")
    return "".join(p), note


# ============================ HTML 段落 ============================

def section_overview(args, data, freq_hz, freq_source, op_echo_us, msprof_note):
    spans_us = [cyc_to_us(c, freq_hz) for c in data["spans"]]
    stats = [s for s in step_stats(data) if s["p50"] is not None]
    hot = max(stats, key=lambda s: s["p50"]) if stats else None
    n_written = sum(data["written"])
    tiles = [
        ("核数（块）", str(data["blocks"]), f"代表轮 {data['steps_path'].name}"),
        ("有效步数", f"{n_written} / {len(data['step_names'])}",
         "非 spec 构建下 8 个 Step0.5 步未写"),
        ("频率口径", f"{freq_hz / 1e9:.3f} GHz", freq_source),
        ("打点跨度 P50", f"{fmt_us(median(spans_us))} μs", f"{data['blocks']} 核中位，entry→末戳"),
        ("最慢核跨度", f"{fmt_us(max(spans_us))} μs", "代表轮；= 该核各步之和（telescoping）"),
    ]
    if hot is not None:
        tiles.append(("热点步 P50", f"{fmt_us(cyc_to_us(hot['p50'], freq_hz))} μs",
                      f'{hot["idx"]}. {hot["name"]}'))
    tiles_html = "".join(
        f'<div class="tile"><div class="lb">{esc(lb)}</div>'
        f'<div class="vl">{esc(vl)}</div><div class="nt">{esc(nt)}</div></div>'
        for lb, vl, nt in tiles)
    recon_svg, gap_note = svg_recon(data, freq_hz, args.p50_us, op_echo_us)
    warn = ""
    if data["warnings"]:
        warn = ('<p class="warn">数据一致性警告：' + "；".join(esc(w) for w in data["warnings"]) + "</p>")
    note = gap_note or ("未给 --p50-us 锚点：只显示打点口径（跨度），对账缺口不可见")
    if msprof_note:
        note += "；" + msprof_note
    return (
        '<section class="card"><h2>1. 总览与对账</h2>'
        f'<p class="cap">数据来源：{esc(data["steps_path"])} / {esc(data["abs_path"])}（代表轮 = '
        f'最慢核跨度最接近中位的一轮，见 echo_perf_step --matrix 注释）；打点口径 = v2.1-prof '
        f'绝对时间戳（§9.23），诊断构建，非正式性能结论。</p>'
        f'{warn}<div class="tiles">{tiles_html}</div>'
        f'{recon_svg}<p class="cap">{esc(note)}</p></section>')


def section_heatmap(data, freq_hz):
    n_written = sum(data["written"])
    return (
        '<section class="card"><h2>2. 逐核 × 逐步热力图</h2>'
        '<button id="unit-toggle" type="button">cyc ⇄ μs</button>'
        f'<p class="cap">行 = 核（blockIdx），列 = {len(data["step_names"])} 步（名表序，列号对应 '
        f'§3 行号）；色 = 该步该核耗时，全局线性顺序色标（最浅=接近零，最深=该图最大值）。'
        f'当前有效列 {n_written}/{len(data["step_names"])}（非 spec 构建下 Step0.5×8 恒未写 = 灰格）。'
        f'悬停显示数值（cyc/μs 可切换）；列步名全称见 §3/§5。</p>'
        f'{svg_heatmap(data, freq_hz)}</section>')


def section_steps(data, freq_hz):
    return (
        f'<section class="card"><h2>3. 逐步聚合（代表轮，{data["blocks"]} 核分布）</h2>'
        f'<p class="cap">每步取代表轮 {data["blocks"]} 核的 P50/max/min/skew：蓝色条 = P50，浅色轨道 = max，'
        f'短刻线 = min，右列 = skew（max−min）；轴单位 μs（@{freq_hz / 1e9:.3f} GHz）。'
        f'悬停显示 cyc/μs 全量数值；无悬停等价物 = §5 原始表。</p>'
        f'{svg_step_bars(data, freq_hz)}</section>')


def section_ops(op_groups, dur_header, sniffed, sources):
    # 无 op 数据也保留编号占位（§5 的交叉引用 "§3/§5" 依赖编号稳定）。
    if not op_groups:
        return ('<section class="card"><h2>4. msprof op 级表</h2>'
                '<p class="cap">未提供 --op-summary，本段留空（用法示例见 prof_report.py '
                'docstring；msprof 采集注意：勿用 msprof op，对 torch 负载不适配，§10.3①）。</p></section>')
    hdr_unit = f"（原文表头: {dur_header}）" if dur_header else ""
    rows = []
    for g in op_groups:
        hl = " class='hl'" if "echo_lru" in g["name"] else ""
        cells = [esc(g["name"]), str(g["n"])]
        for key in ("duration_p50", "duration_max", "vec", "scalar", "mte2", "mte3", "icache"):
            v = g[key]
            cells.append("—" if v is None else f"{v:,.2f}")
        cells.append(esc(g["source"]))
        # 末列（来源文件名）是文本不参与数字右对齐。
        rows.append("<tr" + hl + ">" + "".join(
            f"<td class='num'>{c}</td>" if 0 < i < len(cells) - 1 else f"<td>{c}</td>"
            for i, c in enumerate(cells)) + "</tr>")
    sniff_note = "" if sniffed else "（表头嗅探失败，按首行当表头解析，列可能错位）"
    return (
        '<section class="card"><h2>4. msprof op 级表</h2>'
        f'<p class="cap">来源：{esc("，".join(sources))}；按 kernel 名聚合（n = 实例数，数值列 = 组内'
        f'中位，Duration max 除外）{esc(hdr_unit)}{sniff_note}。Duration 原样引用 msprof 数值不换算单位；'
        f'echo_lru_kernel 行高亮。过滤：echo_lru_kernel / sparse_kv_plan_kernel / TensorMove。</p>'
        '<div class="tbl-wrap"><table><thead><tr>'
        '<th>Kernel</th><th class="num">n</th><th class="num">Duration P50</th>'
        '<th class="num">Duration max</th><th class="num">aiv_vec_ratio</th>'
        '<th class="num">scalar</th><th class="num">mte2</th><th class="num">mte3</th>'
        '<th class="num">icache</th><th>来源</th></tr></thead><tbody>'
        + "".join(rows) + "</tbody></table></div></section>")


def section_raw(data):
    def table(headers, rows, num_from=1):
        head = "".join(f'<th class="{"num" if i >= num_from else ""}">{esc(h)}</th>'
                       for i, h in enumerate(headers))
        body = []
        for r in rows:
            body.append("<tr>" + "".join(
                f'<td class="num">{esc(c)}</td>' if i >= num_from else f"<td>{esc(c)}</td>"
                for i, c in enumerate(r)) + "</tr>")
        return ('<div class="tbl-wrap"><table><thead><tr>' + head + "</tr></thead><tbody>"
                + "".join(body) + "</tbody></table></div>")

    abs_rows = [[str(b)] + [str(v) for v in row] for b, row in enumerate(data["abs"])]
    steps_rows = [[str(b)] + [("" if v is None else str(v)) for v in row]
                  + [str(s)] for b, (row, s) in enumerate(zip(data["steps"], data["spans"]))]
    return (
        '<section class="card"><h2>5. 原始数据</h2>'
        '<p class="cap">两个 CSV 的原样表格（= 各图的无悬停/键盘可达等价物）；'
        '绝对戳为 int32 低 32 位（高位可负，属正常回绕存储），-12345 = 未写哨兵。</p>'
        f'<details><summary>{esc(data["abs_path"].name)}（行=块，列=27 槽原始绝对戳）</summary>'
        + table(["block"] + data["slot_names"], abs_rows) + "</details>"
        f'<details><summary>{esc(data["steps_path"].name)}（行=块，列=26 步耗时 + 总跨度）</summary>'
        + table(["block"] + data["step_names"] + ["span_entry_to_last"], steps_rows) + "</details>"
        "</section>")


# ============================ CSS / JS ============================

CSS = """
:root{color-scheme:light}
*{box-sizing:border-box}
body{margin:0;background:#f9f9f7;color:#0b0b0b;
     font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1120px;margin:0 auto;padding:26px 20px 70px}
h1{font-size:21px;margin:0 0 2px}
.sub{color:#52514e;font-size:12.5px;margin:0 0 6px}
.card{background:#fcfcfb;border:1px solid rgba(11,11,11,.10);border-radius:10px;
      padding:16px 18px 14px;margin:0 0 18px}
.card h2{font-size:15px;margin:0}
.cap{color:#898781;font-size:11.5px;margin:6px 0 12px}
.warn{color:#52514e;font-size:12px;background:#e1e0d9;border:1px solid #c3c2b7;
      border-radius:6px;padding:6px 10px;margin:0 0 10px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin:0 0 16px}
.tile{border:1px solid rgba(11,11,11,.10);border-radius:8px;padding:9px 12px}
.tile .lb{font-size:11.5px;color:#52514e}
.tile .vl{font-size:21px;font-weight:600;margin-top:2px}
.tile .nt{font-size:11px;color:#898781;margin-top:1px}
#unit-toggle{font:600 12px/1 system-ui,sans-serif;color:#2a78d6;background:#fcfcfb;
     border:1px solid #2a78d6;border-radius:6px;padding:5px 10px;margin:4px 0 2px;cursor:pointer}
#unit-toggle:hover{background:#cde2fb}
.chart{display:block;max-width:100%;height:auto}
.axnum{font-size:9.5px;fill:#898781;font-variant-numeric:tabular-nums}
.axcap{font-size:10.5px;fill:#898781}
.axname{font-size:11.5px;fill:#52514e}
.axname.dim{fill:#b9b7b0}
.axval{font-size:10.5px;fill:#0b0b0b;font-variant-numeric:tabular-nums}
.legend{display:flex;align-items:center;gap:7px;margin-top:10px;font-size:11px;color:#52514e;flex-wrap:wrap}
.legend i{width:13px;height:13px;border-radius:2px;display:inline-block}
.legend .sw-row{display:inline-flex;gap:1px}
.legend .sw-row i{width:16px;border-radius:0}
.legend .lg-t{color:#898781}
.legend .lg-gap{width:10px}
.tbl-wrap{overflow-x:auto}
table{border-collapse:collapse;font-size:12px}
th{font-weight:600;text-align:left;color:#52514e;border-bottom:1px solid #c3c2b7;
   padding:4px 8px;white-space:nowrap}
td{border-bottom:1px solid #e1e0d9;padding:3px 8px;white-space:nowrap}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
tr.hl td{background:#cde2fb}
details{margin:10px 0 4px}
summary{cursor:pointer;font-size:12.5px;color:#52514e}
details table{font-size:11px}
#tip{position:fixed;pointer-events:none;background:#0b0b0b;color:#fff;padding:6px 9px;
     border-radius:6px;font-size:12px;max-width:300px;z-index:9;display:none}
#tip .v{font-weight:600}
#tip .l{color:#c3c2b7;margin-top:1px}
footer{color:#898781;font-size:11px;margin-top:26px}
"""

JS_TEMPLATE = """
var FREQ=__FREQ__;
var UNIT='cyc';
var tip=document.getElementById('tip');
function tipClear(){while(tip.firstChild)tip.removeChild(tip.firstChild);}
function tipAdd(strong,weak){
  var d=document.createElement('div');d.className='v';d.textContent=strong;tip.appendChild(d);
  if(weak){var l=document.createElement('div');l.className='l';l.textContent=weak;tip.appendChild(l);}
}
function fmtCyc(c){return Math.round(c).toLocaleString('en-US')+' cyc';}
function fmtUs(c){return (c/FREQ*1e6).toFixed(2)+' \\u03bcs';}
document.addEventListener('pointermove',function(e){
  var el=(e.target&&e.target.closest)?e.target.closest('[data-cyc],[data-line]'):null;
  if(!el){tip.style.display='none';return;}
  tipClear();
  if(el.hasAttribute('data-cyc')){
    var c=parseFloat(el.getAttribute('data-cyc'));
    tipAdd(UNIT==='cyc'?fmtCyc(c):fmtUs(c),el.getAttribute('data-name')||'');
  }else{
    tipAdd(el.getAttribute('data-line')||'',el.getAttribute('data-sub')||'');
  }
  tip.style.display='block';
  var w=tip.offsetWidth,h=tip.offsetHeight;
  tip.style.left=Math.max(8,Math.min(e.clientX+14,window.innerWidth-w-8))+'px';
  tip.style.top=Math.max(8,e.clientY-h-12)+'px';
});
document.addEventListener('pointerdown',function(){tip.style.display='none';});
var btn=document.getElementById('unit-toggle');
if(btn){btn.addEventListener('click',function(){
  UNIT=(UNIT==='cyc')?'us':'cyc';
  var cy=document.querySelectorAll('.unit-cyc'),us=document.querySelectorAll('.unit-us');
  for(var i=0;i<cy.length;i++)cy[i].hidden=(UNIT!=='cyc');
  for(var j=0;j<us.length;j++)us[j].hidden=(UNIT!=='us');
});}
"""


def render(args) -> str:
    data = load_matrix(args.matrix)
    freq_hz = args.freq
    freq_source = f"--freq {freq_hz / 1e9:.4f} GHz"
    if args.freq == DEFAULT_FREQ_HZ:
        freq_source = "1.585 GHz 常数（msprof 实测，§10.3①）"

    # msprof op_summary（可多次）：解析 + 按 kernel 名聚合；echo Duration 走对账条。
    op_groups, op_sources, msprof_us, msprof_note, dur_header, sniffed = [], [], None, "", "", True
    if args.op_summary:
        recs = []
        for path in args.op_summary:
            r, colmap, header, sn = parse_op_summary(Path(path))
            recs.extend(r)
            op_sources.append(Path(path).name)
            sniffed = sniffed and sn
            dur_header = header[colmap["duration"]] if "duration" in colmap else ""
        groups, order = {}, []
        for rec in recs:
            key = rec["name"]
            if key not in groups:
                groups[key] = {"name": key, "source": rec["source"], "vals": {k: [] for k in
                                 ("duration", "vec", "scalar", "mte2", "mte3", "icache")}, "dmax": []}
                order.append(key)
            for k, lst in groups[key]["vals"].items():
                if rec[k] is not None:
                    lst.append(rec[k])
            if rec["duration"] is not None:
                groups[key]["dmax"].append(rec["duration"])
        for key in order:
            g = groups[key]
            row = {"name": key, "n": len(g["dmax"]) or sum(len(v) for v in g["vals"].values()),
                   "source": g["source"]}
            # duration 聚合出 P50/max 两列，比率列只取组内中位。
            row["duration_p50"] = median(g["vals"]["duration"]) if g["vals"]["duration"] else None
            row["duration_max"] = max(g["dmax"]) if g["dmax"] else None
            for k in ("vec", "scalar", "mte2", "mte3", "icache"):
                row[k] = median(g["vals"][k]) if g["vals"][k] else None
            op_groups.append(row)
        # 对账条只用确认以 μs 为单位的 Duration（不猜单位）。
        if any("echo_lru" in g["name"] and g["duration_p50"] is not None for g in op_groups):
            if re.search(r"us|μs", dur_header or "", re.I):
                msprof_us = next(g["duration_p50"] for g in op_groups
                                 if "echo_lru" in g["name"] and g["duration_p50"] is not None)
            else:
                msprof_note = (f"msprof Duration 表头（{dur_header}）未标明 μs，"
                               f"不纳入对账条（不猜单位）")

    parts = []
    parts.append('<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">'
                 '<meta name="viewport" content="width=device-width, initial-scale=1">'
                 f'<title>{esc(args.title)}</title><style>{CSS}</style></head><body>'
                 f'<div class="wrap"><h1>{esc(args.title)}</h1>'
                 f'<div class="sub">生成于 {esc(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))} · '
                 f'prof_report.py · v2.1-prof 打点口径（§9.23）· 诊断构建，非正式性能结论</div>')
    parts.append(section_overview(args, data, freq_hz, freq_source, msprof_us, msprof_note))
    parts.append(section_heatmap(data, freq_hz))
    parts.append(section_steps(data, freq_hz))
    parts.append(section_ops(op_groups, dur_header, sniffed, op_sources))
    parts.append(section_raw(data))
    parts.append('<footer>配色：dataviz skill 已验证参考调色板（light）· 单文件自包含，'
                 '无外部依赖与网络请求 · 悬停数值 = 诊断辅助，完整数值见 §5 表格</footer>')
    parts.append('</div><div id="tip" aria-hidden="true"></div>'
                 f'<script>{JS_TEMPLATE.replace("__FREQ__", repr(freq_hz))}</script>'
                 '</body></html>')
    return "".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matrix", required=True,
                        help="echo_perf_step --matrix 的前缀_场景（读 <prefix>_abs.csv 与 "
                             "<prefix>_steps.csv，例：results/perf_full_cache_75pct）")
    parser.add_argument("--op-summary", action="append", default=None, metavar="CSV",
                        help="msprof task-based 的 op_summary_*.csv，可多次；解析 "
                             "echo_lru_kernel / sparse_kv_plan_kernel / TensorMove 行")
    parser.add_argument("--p50-us", type=float, default=None,
                        help="外部锚点：kernel 级 P50（μs，来自 profile_kernel.sh），进对账条")
    parser.add_argument("--freq", type=float, default=DEFAULT_FREQ_HZ,
                        help=f"换算频率（Hz），默认 {DEFAULT_FREQ_HZ:.3e}（msprof 实测，§10.3①）")
    parser.add_argument("--out", type=Path, default=Path("prof_report.html"))
    parser.add_argument("--title", default=None, help="报告标题（默认取 --matrix 尾名）")
    args = parser.parse_args()
    if args.p50_us is not None and args.p50_us <= 0:
        parser.error("--p50-us 必须为正数（μs）")
    if args.freq <= 0:
        parser.error("--freq 必须为正数（Hz）")
    if args.title is None:
        args.title = f"Echo 分步报告 · {Path(args.matrix).name}"
    html_text = render(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # 不用 write_text(newline=...)（Python 3.10+ 才有该参数）；内容本身无 \r（LF 仓库惯例）。
    args.out.write_text(html_text, encoding="utf-8")
    print(f"已生成: {args.out.resolve()}（{len(html_text) / 1024:.0f} KiB，单文件自包含）")


if __name__ == "__main__":
    main()
