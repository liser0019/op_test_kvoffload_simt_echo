#!/usr/bin/env python3
"""构建/运行联合Plan身份校验；CPU检查仅在服务器运行。"""
import argparse
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def compute_region(text):
    start = text.index("namespace {")
    end = text.index('extern "C" __global__') if "// Direct launch adaptation only:" not in text else text.index("// Direct launch adaptation only:")
    return text[start:end].strip()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Verified workaround no longer matches preserved source")
    return text.replace(old, new, 1)


def expected_fixed_region(original):
    text = compute_region(original)
    text = replace_once(text,
        "ProcessRequestRun(args, row, end, scanWorkspace + PLAN_SCAN_STORAGE_ELEMENTS, scanWorkspace);",
        "ProcessRequestRun(args, row, end, scanWorkspace + PLAN_SCAN_STORAGE_ELEMENTS, scanWorkspace, rowModes);")
    return replace_once(text,
        "ProcessRequestRun(args, row, end, compactWorkspace + static_cast<uint64_t>(row) * scratchElements,\n                          scanWorkspace);",
        "ProcessRequestRun(args, row, end, compactWorkspace + static_cast<uint64_t>(row) * scratchElements,\n                          scanWorkspace, rowModes);")


def expected_fixed_helper(original):
    text = replace_once(original,
        "                                                      __ubuf__ int32_t* scan, uint32_t thread) {",
        "                                                      __ubuf__ int32_t* scan, __gm__ int32_t* modeOutput,\n                                                      uint32_t thread) {")
    text = replace_once(text,
        "      a.rowModes[row] = row == first ? ROW_JOINT : ROW_FOLLOWER;",
        "    }\n    // Device experiments reproduced this mode value in a read-only input.\n"
        "    // Explicit output + thread 1 is an empirical workaround; root cause is\n"
        "    // unconfirmed. This is not a synchronization repair: no barrier is added.\n"
        "    if (thread == 1) {\n      modeOutput[row] = row == first ? ROW_JOINT : ROW_FOLLOWER;")
    text = replace_once(text,
        "                                                         ScratchPtr scratch, __ubuf__ int32_t* scan) {",
        "                                                         ScratchPtr scratch, __ubuf__ int32_t* scan,\n                                                         __gm__ int32_t* modeOutput) {")
    return replace_once(text,
        "PlanJointUnion(a, first, end, keys, slots, partition, scan, thread);",
        "PlanJointUnion(a, first, end, keys, slots, partition, scan, modeOutput, thread);")

def identity():
    manifest = json.loads((ROOT / "joint_plan_manifest.json").read_text())
    for path, expected in manifest["file_sha256"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
            raise ValueError("Joint Plan source fingerprint changed: " + path)
    original = (ROOT / "csrc/joint_plan/upstream/sparse_kv_plan_apt.cpp").read_text()
    direct = (ROOT / "csrc/joint_plan/kernel.cpp").read_text()
    if expected_fixed_region(original) != compute_region(direct):
        raise ValueError("Joint VF differs beyond explicit modeOutput forwarding")
    header = "csrc/joint_plan/sparse_kv_joint_plan.h"
    if expected_fixed_helper((ROOT / "csrc/joint_plan/upstream/sparse_kv_joint_plan.h").read_text()) != (ROOT / header).read_text():
        raise ValueError("Joint helper differs beyond verified thread1 store workaround")
    if (ROOT / "csrc/joint_plan/upstream/sparse_kv_plan_config.h").read_bytes() != (ROOT / "csrc/joint_plan/sparse_kv_plan_config.h").read_bytes():
        raise ValueError("Joint config changed")
    config = (ROOT / "csrc/plan_variants/13_abcdf/variant_config.h").read_text()
    common_config = (ROOT / "csrc/plan_variants/common/sparse_kv_plan_config.h").read_text()
    if "#define PLAN_OPT_MASK 47U" not in config or "PLAN_THREADS = 1024U" not in common_config:
        raise ValueError("Independent baseline must remain ABCDF(mask47), 1024 threads")
    return manifest

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--flags", required=True)
    args = parser.parse_args()
    payload = json.dumps({"source": identity(), "compiler": args.compiler, "flags": args.flags}, sort_keys=True)
    source = '#include <torch/library.h>\n#include <string>\nnamespace { std::string Info() { return R"JOINT('
    source += payload + ')JOINT"; } }\n'
    source += 'TORCH_LIBRARY_FRAGMENT(mtp_joint_plan, m) { m.def("source_info() -> str", Info); }\n'
    if not args.output.exists() or args.output.read_text() != source:
        args.output.write_text(source)

if __name__ == "__main__":
    main()
