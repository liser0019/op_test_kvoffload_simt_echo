#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# 线程数 / blockNum / batch(rows) 消融一键脚本。
#
# 完整流程：按配置逐个构建（--threads=N，产物在 build_t{N}，默认 1024 在 build/）
# → 每个配置跑一遍 kernel 级 profile（4 场景 × simt/echo，采集前后均过 golden）
# → 汇总所有 run 的 summary.json 生成对比报告（results/ablation-<时间戳>/report.md）。
#
# 用法:
#   bash ablation.sh                          # 全矩阵（线程 1024/768/512/256 + blocks 16/8 + rows 64/128）
#   bash ablation.sh --active 50              # 加大每配置采样次数
#   bash ablation.sh --threads 1024,512       # 只跑指定线程配置
#   bash ablation.sh --phase threads          # 只跑某一相: threads | blocks | rows | all
#   bash ablation.sh --skip-build             # 复用已存在的构建产物
#   bash ablation.sh --device 0
#
# 预计耗时: 4 次构建(每次数分钟) + 8 次 profile(每次 1-3 分钟) ≈ 30-40 分钟。
# 数据解读注意: 256 线程时 OpMissCumsum 走 t0 串行（并行门上限 4×256=1024 < topk 2048），
# warm 场景数据带此混杂因素（报告中已自动标注）。
set -uo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DEVICE=1
ACTIVE=20
THREADS="1024,768,512,256"
BLOCKS="16,8"
ROWS="64,128"
PHASE="all"
SKIP_BUILD=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device=*) DEVICE="${1#*=}" ;;
        --active=*) ACTIVE="${1#*=}" ;;
        --threads=*) THREADS="${1#*=}" ;;
        --blocks=*) BLOCKS="${1#*=}" ;;
        --rows=*) ROWS="${1#*=}" ;;
        --phase=*) PHASE="${1#*=}" ;;
        --skip-build) SKIP_BUILD=1 ;;
        --help|-h)
            sed -n '3,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "未知参数: $1（见 --help）" >&2; exit 2 ;;
    esac
    shift
done
[[ "$PHASE" =~ ^(all|threads|blocks|rows)$ ]] || { echo "--phase 只能是 all|threads|blocks|rows" >&2; exit 2; }
[[ "$ACTIVE" =~ ^[0-9]+$ ]] && (( ACTIVE > 0 )) || { echo "--active 必须为正整数" >&2; exit 2; }
IFS=',' read -ra THREAD_LIST <<< "$THREADS"
IFS=',' read -ra BLOCK_LIST  <<< "$BLOCKS"
IFS=',' read -ra ROW_LIST    <<< "$ROWS"
for t in "${THREAD_LIST[@]}"; do
    [[ "$t" =~ ^[0-9]+$ ]] && (( t % 32 == 0 && t >= 64 && t <= 1024 )) || { echo "线程数 $t 非法（32 的倍数, [64,1024]）" >&2; exit 2; }
done

ROOT="$SCRIPT_DIR/results/ablation-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$ROOT"
FAILED=()

lib_for_threads() {  # $1=线程数 → .so 路径
    if [[ "$1" == "1024" ]]; then echo "$SCRIPT_DIR/build/libsimt_vs_echo_ops.so"
    else echo "$SCRIPT_DIR/build_t$1/libsimt_vs_echo_ops.so"; fi
}

build_one() {  # $1=线程数
    local t="$1" lib; lib="$(lib_for_threads "$t")"
    if [[ -f "$lib" && $SKIP_BUILD -eq 1 ]]; then
        echo "[build] t=$t 复用已有产物: $lib"; return 0
    fi
    echo "[build] t=$t 开始构建..."
    if bash build.sh --soc=ascend950 --threads="$t"; then
        if [[ -f "$lib" ]]; then echo "[build] t=$t OK"; else
            echo "[build] t=$t 产物缺失: $lib" >&2; FAILED+=("build_t$t"); return 1
        fi
    else
        echo "[build] t=$t 构建失败" >&2; FAILED+=("build_t$t"); return 1
    fi
}

run_profile() {  # $1=名称, 其余参数原样传给 profile_kernel.sh
    local name="$1"; shift
    local prof_root="/dev/shm/abl-${name}-$$"
    rm -rf "$prof_root"; mkdir -p "$prof_root/tmp"
    echo "[profile] $name 开始 (active=$ACTIVE, device=$DEVICE)..."
    if KV_PROFILE_ROOT="$prof_root" TMPDIR="$prof_root/tmp" TMP="$prof_root/tmp" TEMP="$prof_root/tmp" \
       bash profile_kernel.sh --device "$DEVICE" --active "$ACTIVE" "$@"; then
        local summary
        summary="$(ls -d "$prof_root"/traces/run-*/summary.json 2>/dev/null | head -1)"
        if [[ -n "$summary" ]]; then
            cp "$summary" "$ROOT/$name.summary.json"
            cp "$prof_root/profile.log" "$ROOT/$name.log" 2>/dev/null || true
            rm -rf "$prof_root"
            echo "[profile] $name OK → $ROOT/$name.summary.json"
        else
            echo "[profile] $name 成功但找不到 summary.json（保留 $prof_root 排查）" >&2
            FAILED+=("$name")
        fi
    else
        echo "[profile] $name 失败（日志: $prof_root/profile.log）" >&2
        FAILED+=("$name")   # 保留 /dev/shm 目录供排查
    fi
}

echo "==== 消融实验 $(date '+%F %T') ===="
echo "输出目录: $ROOT"
echo "矩阵: threads=[$THREADS] blocks=[$BLOCKS] rows=[$ROWS] phase=$PHASE active=$ACTIVE device=$DEVICE"
echo

# ---- 相 1: 线程数 ----
if [[ "$PHASE" == "all" || "$PHASE" == "threads" ]]; then
    for t in "${THREAD_LIST[@]}"; do build_one "$t" || true; done
    for t in "${THREAD_LIST[@]}"; do
        lib="$(lib_for_threads "$t")"
        [[ -f "$lib" ]] || { echo "[profile] 跳过 t=$t（无产物）" >&2; continue; }
        run_profile "t$t" --library "$lib"
    done
fi

# ---- 相 2: blockNum（用 1024 线程默认构建; blocks=N 与默认 32 的对比基准来自相 1 的 t1024）----
if [[ "$PHASE" == "all" || "$PHASE" == "blocks" ]]; then
    build_one 1024 || true
    for b in "${BLOCK_LIST[@]}"; do
        [[ -f "$(lib_for_threads 1024)" ]] || break
        run_profile "blocks$b" --library "$(lib_for_threads 1024)" --blocks "$b"
    done
fi

# ---- 相 3: batch(rows)，每核多行的生产形态（基准 = 相 1 的 t1024 = rows 32）----
if [[ "$PHASE" == "all" || "$PHASE" == "rows" ]]; then
    build_one 1024 || true
    for r in "${ROW_LIST[@]}"; do
        [[ -f "$(lib_for_threads 1024)" ]] || break
        run_profile "rows$r" --library "$(lib_for_threads 1024)" --rows "$r"
    done
fi

# ---- 汇总 ----
echo
echo "==== 汇总 ===="
python3 - "$ROOT" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
files = sorted(root.glob("*.summary.json"))
if not files:
    print("没有可汇总的结果（全部失败？见上方日志与 FAILED 列表）")
    sys.exit(1)

table = []
for f in files:
    name = f.name[:-len(".summary.json")]
    data = json.loads(f.read_text(encoding="utf-8"))
    shape = data.get("shape", {})
    for scen, e in data["scenarios"].items():
        table.append((name, scen, shape.get("rows", "?"), shape.get("topk", "?"),
                      e["simt"]["p50_us"], e["echo"]["p50_us"],
                      e["echo_over_simt_kernel_p50"]))

table.sort(key=lambda x: (x[0], x[1]))
cur = None
for name, scen, rows, topk, s, e, r in table:
    if name != cur:
        cur = name
        note = "   # 注: 256 线程时 MissCumsum 走 t0 串行" if name == "t256" else ""
        print(f"\n[{name}]{note}")
        print(f"{'场景':<18} {'rows':>5} {'topk':>5} {'SIMT P50':>10} {'Echo P50':>10} {'Echo/SIMT':>9}")
        print("-" * 66)
    print(f"{scen:<18} {rows:>5} {topk:>5} {s:>9.2f}μs {e:>9.2f}μs {r:>8.2f}×")

report = root / "report.md"
with report.open("w", encoding="utf-8") as fp:
    fp.write("# 消融实验报告\n\n")
    fp.write("- 基准: t1024（= blocks 默认 min(rows,cores)、rows=32 的生产尺寸形态）\n")
    fp.write("- 比较基准: 同场景下各配置与 t1024 互比（Echo/SIMT 列为两算子之比）\n")
    fp.write("- 注: t256 的 OpMissCumsum 走 t0 串行（4×256=1024 < topk 2048），warm 场景带此混杂因素\n")
    fp.write("- rows64/128 配置的 rows 列 ≠ 32，跨配置对比时注意口径\n\n")
    fp.write("| 配置 | 场景 | rows | topk | SIMT P50 μs | Echo P50 μs | Echo/SIMT |\n")
    fp.write("|---|---|---:|---:|---:|---:|---:|\n")
    for name, scen, rows, topk, s, e, r in table:
        fp.write(f"| {name} | {scen} | {rows} | {topk} | {s:.2f} | {e:.2f} | {r:.2f}× |\n")
print(f"\n报告: {report}")
PY
rc=$?
echo
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "⚠ 以下条目失败（日志/目录见上方提示）: ${FAILED[*]}"
    exit 1
fi
echo "全部完成。"
exit $rc
