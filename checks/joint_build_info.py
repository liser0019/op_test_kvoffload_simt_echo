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

def identity():
    manifest = json.loads((ROOT / "joint_plan_manifest.json").read_text())
    for path, expected in manifest["file_sha256"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
            raise ValueError("Joint Plan source fingerprint changed: " + path)
    original = (ROOT / "csrc/joint_plan/upstream/sparse_kv_plan_apt.cpp").read_text()
    direct = (ROOT / "csrc/joint_plan/kernel.cpp").read_text()
    if compute_region(original) != compute_region(direct):
        raise ValueError("ABCDF/joint VF calculation differs from preserved source")
    for name in ("sparse_kv_joint_plan.h", "sparse_kv_plan_config.h"):
        if (ROOT / "csrc/joint_plan/upstream" / name).read_bytes() != (ROOT / "csrc/joint_plan" / name).read_bytes():
            raise ValueError("Joint computation/config modified: " + name)
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
