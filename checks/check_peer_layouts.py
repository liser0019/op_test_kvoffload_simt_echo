#!/usr/bin/env python3
"""Compile actual per-variant Host headers; check layout/ABI/alias isolation without CANN."""
import ast
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import tempfile

from peer_build_info import ROOT, catalog_identity

spec = importlib.util.spec_from_file_location('comparison', ROOT / 'compare.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
shapes = [(16, 32), (256, 512), (2048, 4096), (2049, 4096), (4096, 4096), (4097, 8192)]
with tempfile.TemporaryDirectory(prefix='peer_host_layout_') as temporary:
    directory = Path(temporary)
    objects = []
    names = []
    for index, variant in enumerate(catalog_identity()['variants']):
        path = ROOT / variant['path']
        names.append(variant['name'])
        code = '#include <iostream>\n#include <cstddef>\n'
        code += f'#include "{path / "symbol_aliases.h"}"\n'
        code += f'#include "{path / "upstream/sparse_kv_plan_launch.h"}"\n'
        code += 'static_assert(sizeof(SparseKvPlanTilingData) == 80);\n'
        code += f'void probe_{index}() {{\n'
        for topk, capacity in shapes:
            code += f'{{ auto t=MakeSparseKvPlanTiling(32,{topk},{capacity},262144,32,4096,4096,64); '
            code += 'std::cout<<t.hashCapacity<<" "<<t.workspaceRowElements<<" "<<t.localMemoryBytes<<"\\n"; }\n'
        code += '}\n'
        cpp = directory / f'probe_{index}.cpp'
        cpp.write_text(code)
        obj = cpp.with_suffix('.o')
        subprocess.run(['g++', '-std=c++17', '-O0', '-c', str(cpp), '-o', str(obj)], check=True)
        # At O0 the inline helpers are emitted; no mangled helper may alias another variant.
        symbols = subprocess.check_output(['nm', '-C', str(obj)], text=True)
        for helper in ['MakeSparseKvPlanTiling', 'SparseKvPlanRowElements', 'SparseKvPlanLocalMemoryBytes']:
            assert helper + '_' + variant['name'] in symbols, (variant['name'], helper)
        objects.append(str(obj))
    main = directory / 'main.cpp'
    main.write_text(''.join(f'void probe_{index}();\n' for index in range(len(names))) +
                    'int main(){\n' + ''.join(f'probe_{index}();\n' for index in range(len(names))) + '}\n')
    binary = directory / 'check'
    subprocess.run(['g++', '-std=c++17', str(main), *objects, '-o', str(binary)], check=True)
    rows = subprocess.check_output([str(binary)], text=True).splitlines()
    for name_index, name in enumerate(names):
        for shape_index, (topk, capacity) in enumerate(shapes):
            actual = list(map(int, rows[name_index * len(shapes) + shape_index].split()))
            expected = b.plan_layout(topk, capacity, 32, name)
            assert actual == [expected['hash_capacity'], expected['row_elements'], expected['local_memory_bytes']], name
# Default cached variant changes Host logic only; both imported control kernels are byte-identical.
peer = ROOT / 'csrc/peer_variants'
assert (peer / 'peer_00_baseline/upstream/sparse_kv_plan_kernel.cpp').read_bytes() == \
       (peer / 'peer_09_cached_tiling/upstream/sparse_kv_plan_kernel.cpp').read_bytes()
print(f'PASS: {len(names)} isolated Host ABIs/helpers, {len(names)*len(shapes)} workspace/UB cases; cached control kernel unchanged')

# Verify literal registered schemas against the existing shared adapter schema.
base = (ROOT / 'csrc/simt/sparse_kv_plan_plugin.cpp').read_text()
a = base.index('m.def(SPARSE_KV_VARIANT_OP_NAME')
chunk = base[a:base.index(');', a)]
expected_schema = ''.join(ast.literal_eval(x) for x in re.findall(r'"(?:[^"\\]|\\.)*"', chunk))
for variant in catalog_identity()['variants']:
    text = (ROOT / variant['path'] / 'upstream/plugin_registered.h').read_text()
    a = text.index('m.def("' + variant['op_name'] + '(')
    chunk = text[a:text.index(');', a)]
    actual_schema = ''.join(ast.literal_eval(x) for x in re.findall(r'"(?:[^"\\]|\\.)*"', chunk))
    assert actual_schema == variant['op_name'] + expected_schema, variant['name']
    assert 'm.impl("' + variant['op_name'] + '", sparse_kv_plan_meta)' in text
    assert 'm.impl("' + variant['op_name'] + '", sparse_kv_plan_npu)' in text
print(f'PASS: {len(names)} registered schema strings and backend names; static check only, not Torch dispatch')
