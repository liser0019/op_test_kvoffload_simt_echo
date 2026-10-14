#!/usr/bin/env python3
"""Freeze actual source identity into the extension and reject stale libraries."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "plan_variants_manifest.json"


def catalog_identity():
    catalog = json.loads(MANIFEST.read_text())
    files = dict(catalog["shared_file_sha256"])
    for variant in catalog["variants"]:
        for name, expected in variant["file_sha256"].items():
            files[str(Path(variant["path"]) / name)] = expected
    for name, expected in files.items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != expected: raise ValueError(f"Source fingerprint changed: {name}")
    baseline = ROOT / "csrc/plan_variants/00_baseline/upstream/sparse_kv_plan_kernel.cpp"
    if hashlib.sha256(baseline.read_bytes()).hexdigest() != catalog["baseline_kernel_sha256"]:
        raise ValueError("The measured baseline kernel must remain unchanged")
    return catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.dumps(catalog_identity(), sort_keys=True)
    source = '#include <torch/library.h>\n#include <string>\nnamespace {\n'
    source += 'std::string PlanInfo() { return R"PLAN(' + payload + ')PLAN"; }\n}\n'
    source += 'TORCH_LIBRARY_FRAGMENT(sparse_kv_plan_op, m) {\n'
    source += '  m.def("plan_variants_info() -> str", PlanInfo);\n}\n'
    if not args.output.exists() or args.output.read_text() != source: args.output.write_text(source)


if __name__ == "__main__":
    main()
