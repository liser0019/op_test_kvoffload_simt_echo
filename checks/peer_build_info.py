#!/usr/bin/env python3
"""Validate exact imported/adapted files and generate immutable compiled provenance."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def catalog_identity():
    catalog = json.loads((ROOT / 'peer_variants_manifest.json').read_text())
    for variant in catalog['variants']:
        directory = ROOT / variant['path']
        for name, expected in variant['file_sha256'].items():
            actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f'{variant["name"]}/{name}: source fingerprint changed')
    return catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    encoded = json.dumps(catalog_identity(), sort_keys=True)
    source = '''#include <torch/library.h>
#include <string>
namespace {
std::string PeerInfo() { return R"PEER(%s)PEER"; }
}
TORCH_LIBRARY_FRAGMENT(sparse_kv_plan_op, m) {
    m.def("peer_variants_info() -> str", PeerInfo);
}
''' % encoded
    if not args.output.exists() or args.output.read_text() != source:
        args.output.write_text(source)


if __name__ == '__main__':
    main()
