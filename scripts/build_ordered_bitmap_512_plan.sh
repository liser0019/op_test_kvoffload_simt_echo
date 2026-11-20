#!/usr/bin/env bash
# Build the attached ordered-query bitmap Plan as a standalone torch extension.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
WORK=${ORDERED_BITMAP_PLAN_BUILD_ROOT:-"$ROOT/build_ordered_bitmap_512"}
PYTHON=${PYTHON:-python3}
PYTHON=$("$PYTHON" -c 'import sys; print(sys.executable)')

PLAN_ARCHIVE=${ORDERED_BITMAP_PLAN_ARCHIVE:-}
PLAN_SOURCE=${ORDERED_BITMAP_PLAN_SOURCE:-"$ROOT"}
PLAN_COMMIT=${ORDERED_BITMAP_PLAN_COMMIT:-}

if [[ -e "$WORK" ]]; then
  echo "构建目录已存在，请指定新的 ORDERED_BITMAP_PLAN_BUILD_ROOT：$WORK" >&2
  exit 2
fi
mkdir -p "$WORK/src"
if [[ -n "$PLAN_ARCHIVE" ]]; then
  PLAN_ARCHIVE=$(realpath "$PLAN_ARCHIVE")
  [[ -f "$PLAN_ARCHIVE" ]] || { echo "Plan 源码归档不存在：$PLAN_ARCHIVE" >&2; exit 2; }
  tar -xf "$PLAN_ARCHIVE" -C "$WORK/src"
else
  PLAN_SOURCE=$(realpath "$PLAN_SOURCE")
  if [[ -z "$PLAN_COMMIT" ]]; then
    PLAN_COMMIT=$(git -C "$PLAN_SOURCE" rev-parse HEAD)
  fi
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
  PYTHON="$PYTHON" JOBS="${JOBS:-8}" bash build_joint_plan.sh \
    --soc=ascend950 --mtp3-only --ordered-bitmap-plan
)

LIBRARY="$WORK/src/build_joint_plan/libmtp_joint_plan_ops.so"
[[ -f "$LIBRARY" ]] || { echo "Plan 库未生成：$LIBRARY" >&2; exit 3; }
cp "$LIBRARY" "$WORK/libmtp_joint_plan_ops.so"
sha256sum "$WORK/libmtp_joint_plan_ops.so" > "$WORK/library.sha256"
printf '%s\n' "$PLAN_COMMIT" > "$WORK/source-commit.txt"
printf '%s\n' "$WORK/libmtp_joint_plan_ops.so"
printf '%s\n' "$WORK/identity.json"
