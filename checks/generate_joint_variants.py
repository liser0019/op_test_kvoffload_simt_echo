#!/usr/bin/env python3
"""只生成符号包装和分派表；算法位于 csrc/joint_variants，可直接阅读。"""
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from joint_variants import MASKS


def generate(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    launch = (ROOT / "csrc/joint_plan/launch.h").read_text()
    declaration = launch[launch.index('extern "C" void launch_joint_plan'):].strip()
    symbols = ("JointDirectBody", "launch_joint_plan", "joint_plan_block_dim", "sparse_kv_plan_joint_kernel",
               "sparse_kv_plan_private_kernel")
    declarations, entries = [], []
    for letters, mask in MASKS.items():
        replacements = {name: name + "_" + letters for name in symbols}
        text = f"// Generated symbol wrapper; Joint options={letters}, mask={mask}.\n#define JOINT_OPT_MASK {mask}\n"
        text += "".join(f"#define {name} {new}\n" for name, new in replacements.items())
        text += '#include "' + str(ROOT / "csrc/joint_variants/kernel.cpp") + '"\n'
        (destination / f"variant_{letters}.cpp").write_text(text)
        declarations.append(declaration.replace("launch_joint_plan", replacements["launch_joint_plan"]))
        entries.append("&" + replacements["launch_joint_plan"])
    signature = declaration.replace('extern "C" void launch_joint_plan(bool',
                                    'inline void LaunchJointVariant(int64_t mask, bool').rstrip(';')
    args = "jointEntry, reqIds, topkIndices, stablePrefixLens, visibleSeqLens, tokenToReq, activeRows, lastReqIds, slotToToken, lruSlots, currentSlots, missCount, missTokens, missSlots, compactWorkspace, rowModes, residentRows, residentBlockTable, tiling, blockDim, dynamicUb, stream"
    text = '#pragma once\n#include <stdexcept>\n' + "\n".join(declarations) + "\n"
    text += signature + " {\n  if (mask < 1 || mask > 17) throw std::invalid_argument(\"joint mask\");\n"
    text += "  static const decltype(&launch_joint_plan) launches[] = {nullptr, " + ", ".join(entries) + "};\n"
    text += "  launches[mask](" + args + ");\n}\n"
    (destination / "variant_dispatch.h").write_text(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    generate(parser.parse_args().output)
