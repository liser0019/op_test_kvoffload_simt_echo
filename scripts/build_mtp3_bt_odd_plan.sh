#!/usr/bin/env bash
# Build the frozen joint_mtp3_bt_direct_odd_512 Plan separate from the QLI vendor.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PLAN_COMMIT=88c06745b0260ef258ce3870945bdb7668d3d483
WORK=${MTP3_BT_ODD_PLAN_BUILD_ROOT:-"$ROOT/build_mtp3_bt_direct_odd"}
PYTHON=${PYTHON:-python3}
PYTHON=$("$PYTHON" -c 'import sys; print(sys.executable)')

PLAN_ARCHIVE=${MTP3_BT_ODD_PLAN_ARCHIVE:-}
PLAN_SOURCE=${MTP3_BT_ODD_PLAN_SOURCE:-"$ROOT"}

if [[ -e "$WORK" ]]; then
  echo "构建目录已存在，请指定新的 MTP3_BT_ODD_PLAN_BUILD_ROOT：$WORK" >&2
  exit 2
fi
mkdir -p "$WORK/src"
if [[ -n "$PLAN_ARCHIVE" ]]; then
  PLAN_ARCHIVE=$(realpath "$PLAN_ARCHIVE")
  [[ -f "$PLAN_ARCHIVE" ]] || { echo "Plan 源码归档不存在：$PLAN_ARCHIVE" >&2; exit 2; }
  tar -xf "$PLAN_ARCHIVE" -C "$WORK/src"
else
  PLAN_SOURCE=$(realpath "$PLAN_SOURCE")
  git -C "$PLAN_SOURCE" cat-file -e "$PLAN_COMMIT^{commit}"
  git -C "$PLAN_SOURCE" archive "$PLAN_COMMIT" | tar -x -C "$WORK/src"
fi

(
  cd "$WORK/src"
  PYTHONPATH="$WORK/src" "$PYTHON" -B - "$WORK/identity.json" <<'PY'
import json
import sys
from checks.joint_build_info import identity
open(sys.argv[1], 'w', encoding='utf-8').write(
    json.dumps(identity(), ensure_ascii=False, indent=2, sort_keys=True) + '\n'
)
PY
  PYTHON="$PYTHON" JOBS="${JOBS:-8}" bash build_joint_plan.sh --soc=ascend950 --mtp3-only
)

LIBRARY="$WORK/src/build_joint_plan/libmtp_joint_plan_ops.so"
[[ -f "$LIBRARY" ]] || { echo "Plan 库未生成：$LIBRARY" >&2; exit 3; }
cp "$LIBRARY" "$WORK/libmtp_joint_plan_ops.so"
sha256sum "$WORK/libmtp_joint_plan_ops.so" > "$WORK/library.sha256"
printf '%s\n' "$PLAN_COMMIT" > "$WORK/source-commit.txt"
echo "Plan库: $WORK/libmtp_joint_plan_ops.so"
echo "源码身份: $WORK/identity.json"
