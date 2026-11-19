#!/usr/bin/env bash
# Fixed MTP3 route: CPU gate -> CANN build -> NPU5 correctness gate -> short profile.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CANNDIR="${ASCEND_CANN_PACKAGE_HOME:-/usr/local/Ascend/cann-9.2.1}"
PYTHON_BIN="${PYTHON:-/usr/local/python3.12.16/bin/python3}"
DEVICE=5
OUTPUT=/workspace/mtp3_candidates_results
SKIP_BUILD=0
CPU_ONLY=0
SKIP_PROFILE=0
MTP3_475_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --skip-build) SKIP_BUILD=1 ;;
    --cpu-only) CPU_ONLY=1 ;;
    --skip-profile) SKIP_PROFILE=1 ;;
    --mtp3-475-only) MTP3_475_ONLY=1 ;;
    --device=*) DEVICE="${arg#*=}" ;;
    --output=*) OUTPUT="${arg#*=}" ;;
    --help|-h)
      echo "用法: bash scripts/test_mtp3_candidates.sh [--skip-build] [--cpu-only] [--skip-profile] [--mtp3-475-only] [--device=5] [--output=/workspace/mtp3_candidates_results]"
      exit 0 ;;
    *) echo "未知参数: $arg" >&2; exit 2 ;;
  esac
done
if [[ -e "$OUTPUT" ]]; then
  echo "拒绝覆盖已有目录: $OUTPUT" >&2
  exit 2
fi
mkdir -p "$OUTPUT/logs" "$OUTPUT/tmp"
export ASCEND_CANN_PACKAGE_HOME="$CANNDIR"
export ASCEND_HOME_PATH="$CANNDIR"
export TMPDIR="$OUTPUT/tmp"
# shellcheck disable=SC1091
source "$CANNDIR/set_env.sh"
npu-smi info >"$OUTPUT/logs/npu_before.log" 2>&1 || true

run_log() {
  local label="$1"; shift
  echo "开始 $label"
  "$@" >"$OUTPUT/logs/$label.log" 2>&1
  echo "完成 $label"
}

CPU_GATE_ARGS=()
if [[ "$MTP3_475_ONLY" == 1 ]]; then CPU_GATE_ARGS+=(--mtp3-475-only); fi
run_log cpu_gate "$PYTHON_BIN" -B -X faulthandler \
  "$ROOT/checks/check_mtp3_candidates.py" "${CPU_GATE_ARGS[@]}" --report "$OUTPUT/cpu_gate.json"
if [[ "$CPU_ONLY" == 1 ]]; then
  echo "CPU-only完成: $OUTPUT"
  exit 0
fi
if [[ "$SKIP_BUILD" == 0 ]]; then
  run_log build env PYTHON="$PYTHON_BIN" JOBS="${JOBS:-3}" \
    bash "$ROOT/build_joint_plan.sh" --soc=ascend950 --mtp3-only
fi
LIBRARY="$ROOT/build_joint_plan/libmtp_joint_plan_ops.so"
[[ -f "$LIBRARY" ]] || { echo "缺少构建产物: $LIBRARY" >&2; exit 3; }
run_log source_identity "$PYTHON_BIN" -B - "$ROOT" "$LIBRARY" "$OUTPUT/source_identity.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
root, library, output = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(root))
from checks.joint_build_info import identity
manifest = identity()
payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
output.write_text(json.dumps({
    "manifest_sha256": hashlib.sha256(payload.encode()).hexdigest(),
    "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
    "library": str(library.resolve()),
    "manifest": manifest,
}, ensure_ascii=False, indent=2) + "\n")
PY
NPU_GATE_ARGS=()
if [[ "$MTP3_475_ONLY" == 1 ]]; then
  NPU_GATE_ARGS+=(--mtp3-475-only --contexts 102400 262144 358400 486400)
fi
run_log npu_gate "$PYTHON_BIN" -B -X faulthandler \
  "$ROOT/checks/check_mtp3_npu.py" --device "$DEVICE" --library "$LIBRARY" \
  --temp-dir "$OUTPUT/tmp" --audit-repeat "${AUDIT_REPEAT:-3}" \
  "${NPU_GATE_ARGS[@]}" --report "$OUTPUT/npu_gate.json"
if [[ "$SKIP_PROFILE" == 0 ]]; then
  run_log profile "$PYTHON_BIN" -B -X faulthandler \
    "$ROOT/profile_joint_mtp.py" --device "$DEVICE" --library "$LIBRARY" \
    --temp-dir "$OUTPUT/tmp" --output "$OUTPUT/profile" \
    --implementations $([[ "$MTP3_475_ONLY" == 1 ]] && echo "joint_mtp3_dir475_512 joint_mtp3_bitmap475_512" || echo "joint_rankpipe_ac_512 joint_mtp3_full_512 joint_mtp3_map_512 joint_mtp3_load_512 joint_mtp3_full_load_512") \
    --mtp 3 --requests 1 --topk 2048 --capacity 8192 --max-token $([[ "$MTP3_475_ONLY" == 1 ]] && echo 486400 || echo 262144) \
    --scenarios full_cache_75pct --overlaps overlap40 --sorted-topk \
    --warmup "${PROFILE_WARMUP:-2}" --repeat "${PROFILE_REPEAT:-5}" --regression-repeat 3 \
    --correctness-report "$OUTPUT/npu_gate.json"
fi
npu-smi info >"$OUTPUT/logs/npu_after.log" 2>&1 || true
echo "完成: $OUTPUT"
