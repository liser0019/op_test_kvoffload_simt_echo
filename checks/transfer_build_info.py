#!/usr/bin/env python3
"""Transfer/满缓存实验的源码身份；仅构建服务器运行，不包含性能模拟。"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def compute_class(text):
    start = text.index("class SparseKvTransferRuntimeKernel {")
    return text[start:text.index("\n};", start) + 4]


def identity():
    manifest = json.loads((ROOT / "transfer_manifest.json").read_text())
    for path, expected in manifest["file_sha256"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Transfer source fingerprint changed: {path}")
    if hashlib.sha256((ROOT / "plan_variants_manifest.json").read_bytes()).hexdigest() != manifest["plan_manifest_sha256"]:
        raise ValueError("Plan catalog changed; regenerate Transfer identity and rebuild")
    upstream = (ROOT / "csrc/transfer/upstream/sparse_kv_transfer_apt.cpp").read_text()
    direct = (ROOT / "csrc/transfer/kernel.cpp").read_text()
    if compute_class(upstream) != compute_class(direct):
        raise ValueError("Transfer computation class differs from preserved upstream source")
    fallback = (ROOT / "csrc/transfer_variants/original_class.h").read_text()
    if compute_class(upstream) != compute_class(fallback):
        raise ValueError("Variant fallback differs from frozen upstream class")
    catalog = json.loads((ROOT / "transfer_variants.json").read_text())
    if len(catalog["variants"]) != 17 or {v["mask"] for v in catalog["variants"]} != set(range(-1, 16)):
        raise ValueError("Transfer factorial catalog must contain baseline plus all16 masks")
    entries = (ROOT / "csrc/transfer_variants/entries.inc").read_text()
    for variant in catalog["variants"]:
        if variant["mask"] >= 0 and f"TRANSFER_VARIANT({variant['mask']}, {variant['name']})" not in entries:
            raise ValueError(f"Missing compiled Transfer entry: {variant['name']}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--flags", required=True)
    parser.add_argument("--plan-transfer-only", choices=("ON", "OFF"), required=True)
    args = parser.parse_args()
    payload = json.dumps({"source": identity(), "compiler": {"compiler": args.compiler, "flags": args.flags,
                                                           "plan_transfer_only": args.plan_transfer_only}},
                         sort_keys=True)
    source = '#include <torch/library.h>\n#include <string>\nnamespace {\n'
    source += 'std::string TransferInfo() { return R"TRANSFERJSON(' + payload + ')TRANSFERJSON"; }\n}\n'
    source += 'TORCH_LIBRARY_FRAGMENT(sparse_kv_plan_op, m) { m.def("transfer_source_info() -> str", TransferInfo); }\n'
    if not args.output.exists() or args.output.read_text() != source:
        args.output.write_text(source)


if __name__ == "__main__":
    main()
