#!/usr/bin/env bash
# Dedicated upstream checkout/package: never patch an existing user's source tree.
set -euo pipefail
usage="$0 [--indexer-compare-only | --peer-compare-only | --topk-compare-only | --fast-split-only | --ordered-standalone-only | --peer-aligned-only | --auto-rankpipe-only | --factorial-only | --factorial-compare-only | --native-plan-only | --native-affinity-only | --native-adaptive-only | --native-scale-l2-only | --native-balanced-only | --native-reuse-only]"
if [[ $# -gt 1 ]]; then
  echo "用法：$usage" >&2
  exit 2
fi
OPS=quant_lightning_indexer,quant_lightning_indexer_plan_split
case "${1:-}" in
  "") OPS+=,quant_lightning_indexer_plan_split4,quant_lightning_indexer_plan_a,quant_lightning_indexer_plan_b,quant_lightning_indexer_plan_c,quant_lightning_indexer_ordered_fused,quant_lightning_indexer_plan_cfa,quant_lightning_indexer_plan_cfb,quant_lightning_indexer_plan_cfc,quant_lightning_indexer_plan_cfab,quant_lightning_indexer_plan_cfac,quant_lightning_indexer_plan_cfbc,quant_lightning_indexer_plan_cfabc ;;
  --indexer-compare-only) OPS+=,quant_lightning_indexer_plan_split4 ;;
  --peer-compare-only) OPS+=,quant_lightning_indexer_ordered_fused ;;
  --topk-compare-only) OPS=quant_lightning_indexer_plan_split4,quant_lightning_indexer_plan_topk_control4 ;;
  --fast-split-only) OPS=quant_lightning_indexer,quant_lightning_indexer_plan_final4,quant_lightning_indexer_plan_split4 ;;
  --ordered-standalone-only) OPS=quant_lightning_indexer,quant_lightning_indexer_ordered,quant_lightning_indexer_plan_split4,quant_lightning_indexer_plan_final4 ;;
  --peer-aligned-only) OPS=quant_lightning_indexer,quant_lightning_indexer_plan_final4,quant_lightning_indexer_ordered_fused,quant_lightning_indexer_plan_peer4 ;;
  --auto-rankpipe-only) OPS=quant_lightning_indexer,quant_lightning_indexer_ordered_fused,quant_lightning_indexer_plan_c ;;
  --factorial-only) OPS=quant_lightning_indexer_plan_split,quant_lightning_indexer_plan_c,quant_lightning_indexer_plan_cfa,quant_lightning_indexer_plan_cfb,quant_lightning_indexer_plan_cfc,quant_lightning_indexer_plan_cfab,quant_lightning_indexer_plan_cfac,quant_lightning_indexer_plan_cfbc,quant_lightning_indexer_plan_cfabc ;;
  --factorial-compare-only) OPS=quant_lightning_indexer_plan_split,quant_lightning_indexer_plan_c,quant_lightning_indexer_plan_cfa,quant_lightning_indexer_plan_cfb,quant_lightning_indexer_plan_cfc,quant_lightning_indexer_plan_cfab,quant_lightning_indexer_plan_cfac,quant_lightning_indexer_plan_cfbc,quant_lightning_indexer_plan_cfabc,quant_lightning_indexer,quant_lightning_indexer_ordered_fused,quant_lightning_indexer_plan_final4 ;;
  --native-plan-only) OPS=quant_lightning_indexer_ordered,quant_lightning_indexer_plan_native_shell,quant_lightning_indexer_plan_native_fused,quant_lightning_indexer_plan_native_group ;;
  --native-affinity-only) OPS=quant_lightning_indexer_ordered,quant_lightning_indexer_plan_native_group,quant_lightning_indexer_plan_native_group4,quant_lightning_indexer_plan_native_pair4 ;;
  --native-adaptive-only) OPS=quant_lightning_indexer_ordered,quant_lightning_indexer_plan_native_group,quant_lightning_indexer_plan_native_pair4,quant_lightning_indexer_plan_native_adaptive ;;
  --native-scale-l2-only) OPS=quant_lightning_indexer_ordered,quant_lightning_indexer_plan_native_adaptive,quant_lightning_indexer_plan_native_scale ;;
  --native-balanced-only) OPS=quant_lightning_indexer_ordered,quant_lightning_indexer_plan_native_scale,quant_lightning_indexer_plan_native_balanced ;;
  --native-reuse-only) OPS=quant_lightning_indexer_ordered,quant_lightning_indexer_plan_native_scale,quant_lightning_indexer_plan_native_balanced,quant_lightning_indexer_plan_native_l2_reuse,quant_lightning_indexer_plan_native_balance_reuse ;;
  *) echo "用法：$usage" >&2; exit 2 ;;
esac
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python3}
# The upstream scripts invoke python3 themselves. Keep them in the caller's
# selected Python/torch environment, including their lazy extension builder.
PYTHON=$("$PYTHON" -c 'import sys; print(sys.executable)')
export PATH="$(dirname -- "$PYTHON"):$PATH"
export MAX_JOBS=${JOBS:-3}
"$PYTHON" -c 'import build, torch, torch_npu'
WORK=${QLI_FUSION_BUILD_DIR:-"$ROOT/build_qli_fusion"}
PIN=9ec2110969a9e8502bf6d579c027ac4ea9572016
if [[ -e "$WORK" ]]; then
  echo "构建目录已存在，为保留原产物请指定新的 QLI_FUSION_BUILD_DIR：$WORK" >&2;exit 2
fi
mkdir -p "$WORK"
if [[ -n "${QLI_UPSTREAM_SOURCE:-}" ]]; then
  git clone --local --no-checkout "$QLI_UPSTREAM_SOURCE" "$WORK/upstream"
else
  git clone --single-branch --branch wzj_qli_order_fused https://gitcode.com/sparse/ops-transformer.git "$WORK/upstream"
fi
git -C "$WORK/upstream" checkout --detach "$PIN"
if [[ -n "${QLI_THIRD_PARTY_CACHE:-}" ]]; then
  mkdir -p "$WORK/upstream/third_party"
  cp -a "$QLI_THIRD_PARTY_CACHE"/. "$WORK/upstream/third_party"/
fi
"$PYTHON" -B "$ROOT/scripts/export_fusion_plan.py" --check
"$PYTHON" -B "$ROOT/scripts/prepare_qli_fusion.py" --upstream "$WORK/upstream"
cd "$WORK/upstream"
bash build.sh --pkg --soc=ascend950 --ops="$OPS" --vendor_name=qlifusion -j"${JOBS:-3}"
PACKAGES=(build_out/cann-ops-transformer-qlifusion*.run)
[[ ${#PACKAGES[@]} -eq 1 && -f "${PACKAGES[0]}" ]] || { echo "vendor包数量异常" >&2;exit 1; }
bash "${PACKAGES[0]}" --install-path="$WORK/install"
export ASCEND_CUSTOM_OPP_PATH="${ASCEND_CUSTOM_OPP_PATH:-}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$WORK/torch_extensions}"
source "$WORK/install/vendors/qlifusion_transformer/bin/set_env.bash"
bash build.sh --torch_extension --ops="$OPS" --vendor_name=qlifusion
WHEELS=(build_out/cann_ops_transformer_qlifusion*.whl)
[[ ${#WHEELS[@]} -eq 1 && -f "${WHEELS[0]}" ]] || { echo "wheel数量异常" >&2;exit 1; }
"$PYTHON" -m pip install --force-reinstall --no-deps "${WHEELS[0]}"
cp qli_fusion_manifest.json "$WORK/manifest.json"
"$PYTHON" -B - "$WORK" "$OPS" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);path=root/'manifest.json';data=json.loads(path.read_text())
data['packaged_operators']=sys.argv[2].split(',')
data['installed_files']={str(f.relative_to(root/'install')):hashlib.sha256(f.read_bytes()).hexdigest()
    for f in (root/'install').rglob('*') if f.is_file() and f.suffix in ('.so','.o','.bin','.json')}
if not data['installed_files']:raise SystemExit('no installed CANN binaries/configs found')
path.write_text(json.dumps(data,indent=2)+'\n')
PY
cat > "$WORK/env.sh" <<EOF
export ASCEND_CUSTOM_OPP_PATH="\${ASCEND_CUSTOM_OPP_PATH:-}"
source "$WORK/install/vendors/qlifusion_transformer/bin/set_env.bash"
export QLI_FUSION_BUILD_DIR="$WORK"
export TORCH_EXTENSIONS_DIR="\${TORCH_EXTENSIONS_DIR:-$WORK/torch_extensions}"
EOF
echo "构建完成；请 source $WORK/env.sh，再运行 scripts/test_qli_fusion.sh"
