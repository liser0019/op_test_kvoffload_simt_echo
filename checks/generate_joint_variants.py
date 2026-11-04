#!/usr/bin/env python3
"""只生成符号包装和分派表；算法位于 csrc/joint_variants，可直接阅读。"""
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from joint_variants import MASKS, algorithm_mask, COLD_SWEEP,REFINE_SWEEP,EXEC_SWEEP,HOT_SWEEP


def generate(destination, merge_only=False, sorted_only=False, sweep_only=False, union_only=False, ac_threads_only=False, cold_compare_only=False, integrated_only=False,refine_only=False,exec_only=False,hot_only=False,mtp8_only=False):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    launch = (ROOT / "csrc/joint_plan/launch.h").read_text()
    declaration = launch[launch.index('extern "C" void launch_joint_plan'):].strip()
    symbols = ("JointDirectBody", "launch_joint_plan", "joint_plan_block_dim", "sparse_kv_plan_joint_kernel",
               "sparse_kv_plan_private_kernel")
    declarations, entries = [], ["nullptr"] * (max(MASKS.values()) + 1)
    for letters, mask in MASKS.items():
        if mtp8_only and letters not in ("hot_bd_512","hot_bd8_512"):continue
        if hot_only and letters not in ("integrated_bounds_512","refine_abd_512","refine_bd_512") and letters not in HOT_SWEEP:continue
        if exec_only and letters not in ("integrated_bounds_512","refine_abd_512","refine_bd_512") and letters not in EXEC_SWEEP:continue
        if refine_only and letters!="integrated_bounds_512" and letters not in REFINE_SWEEP:continue
        if integrated_only and letters not in ("union_cold_512","hw_ac_cold_512","integrated_512","integrated_bounds_512"): continue
        if cold_compare_only and letters not in ("hw_ac_512","hw_ac_256","union_base_512",*COLD_SWEEP): continue
        if ac_threads_only and letters not in ("hw_ac_512", "hw_ac_256"): continue
        if union_only and letters not in ("hw_ac_512", "hw_ac_256", "hw_ac_cold_512") and not letters.startswith("union_"): continue
        if sweep_only and letters != "sorted_mrgsort" and not letters.startswith("hw_"):
            continue
        if not sweep_only and sorted_only and letters not in ("merge", "merge_bd", "sorted_stream", "sorted_mrgsort"):
            continue
        if not sweep_only and not sorted_only and merge_only and not letters.startswith("merge"):
            continue
        replacements = {name: name + "_" + letters for name in symbols}
        text = f"// Generated symbol wrapper; Joint options={letters}, mask={mask}.\n#define JOINT_OPT_MASK {algorithm_mask(mask)}\n"
        if mask in COLD_SWEEP.values() or mask in (125,126) or mask>=127: text += "#define JOINT_COLD_FINISH 1\n"
        if mask in (125,126): text += "#define JOINT_INTEGRATED 1\n"
        if mask==126: text += "#define JOINT_PHASE0_BOUNDS 1\n"
        if 127<=mask<=142:text += f"#define JOINT_REFINE 1\n#define JOINT_REFINE_BITS {mask-127}\n"
        if 143<=mask<=158:text += f"#define JOINT_REFINE 1\n#define JOINT_REFINE_BITS 11\n#define JOINT_EXEC 1\n#define JOINT_EXEC_BITS {mask-143}\n"
        if mask>=159:
            bits=10 if mask==193 else mask-159 if mask<=190 else 0
            if mask==193:text += "#define JOINT_MTP8 1\n"
            text += f"#define JOINT_REFINE 1\n#define JOINT_REFINE_BITS 11\n#define JOINT_HOT 1\n#define JOINT_HOT_BITS {bits}\n#define JOINT_HOT_PART_ONLY {int(mask==191)}\n#define JOINT_HOT_MISS_ONLY {int(mask==192)}\n"
        text += "".join(f"#define {name} {new}\n" for name, new in replacements.items())
        text += '#include "' + str(ROOT / "csrc/joint_variants/kernel.cpp") + '"\n'
        (destination / f"variant_{letters}.cpp").write_text(text)
        declarations.append(declaration.replace("launch_joint_plan", replacements["launch_joint_plan"]))
        entries[mask] = "&" + replacements["launch_joint_plan"]
    signature = declaration.replace('extern "C" void launch_joint_plan(bool',
                                    'inline void LaunchJointVariant(int64_t mask, bool').rstrip(';')
    args = "jointEntry, reqIds, topkIndices, stablePrefixLens, visibleSeqLens, tokenToReq, activeRows, lastReqIds, slotToToken, lruSlots, currentSlots, missCount, missTokens, missSlots, compactWorkspace, rowModes, residentRows, residentBlockTable, tiling, blockDim, dynamicUb, stream"
    text = '#pragma once\n#include <stdexcept>\n' + "\n".join(declarations) + "\n"
    text += "inline decltype(&launch_joint_plan) JointVariantLaunch(int64_t mask) {\n"
    text += "  static const decltype(&launch_joint_plan) launches[] = {" + ", ".join(entries) + "};\n"
    text += "  return mask > 0 && mask <= 193 ? launches[mask] : nullptr;\n}\n"
    text += 'inline int64_t JointVariantAlgorithmMask(int64_t mask) { return mask == 122 ? 59 : mask == 123 ? 90 : (mask == 124 || mask == 125 || mask == 126 || (mask >= 127 && mask <= 193)) ? 72 : mask; }\n'
    text += signature + " {\n  auto launch = JointVariantLaunch(mask);\n"
    text += "  if (!launch) throw std::invalid_argument(\"joint mask not built\");\n"
    text += "  launch(" + args + ");\n}\n"
    (destination / "variant_dispatch.h").write_text(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--sorted-only", action="store_true")
    parser.add_argument("--sweep-only", action="store_true")
    parser.add_argument("--union-only", action="store_true")
    parser.add_argument("--ac-threads-only", action="store_true")
    parser.add_argument("--cold-compare-only",action="store_true")
    parser.add_argument("--integrated-only",action="store_true")
    parser.add_argument("--refine-only",action="store_true")
    parser.add_argument("--exec-only",action="store_true")
    parser.add_argument("--hot-only",action="store_true")
    parser.add_argument("--mtp8-only",action="store_true")
    args = parser.parse_args()
    generate(args.output, args.merge_only, args.sorted_only,args.sweep_only,args.union_only,args.ac_threads_only,args.cold_compare_only,args.integrated_only,args.refine_only,args.exec_only,args.hot_only,args.mtp8_only)
