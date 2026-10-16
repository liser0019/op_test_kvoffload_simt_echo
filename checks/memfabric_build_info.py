#!/usr/bin/env python3
"""Freeze original kernel and actual adapter identity into the shared library."""
import argparse
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def identity():
    manifest = json.loads((ROOT / 'memfabric_plan_manifest.json').read_text())
    for name, info in manifest['files'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != info['sha256']:
            raise ValueError('Archive source changed: ' + name)
    return {'manifest': manifest, 'plugin_sha256': hashlib.sha256((ROOT / 'csrc/memfabric_plan/plugin.cpp').read_bytes()).hexdigest()}
if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--output', type=Path, required=True)
    path = parser.parse_args().output
    text = '#include <torch/library.h>\n#include <string>\nnamespace {\n'
    text += 'std::string Info() { return R"MEM(' + json.dumps(identity(), sort_keys=True) + ')MEM"; }\n}\n'
    text += 'TORCH_LIBRARY_FRAGMENT(sparse_kv_plan_op,m) { m.def("memfabric_plan_info() -> str", Info); }\n'
    if not path.exists() or path.read_text() != text:path.write_text(text)
