#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""核对导入哈希、直调 ABI、C++ host tiling 与 Python 布局；不需要 CANN/NPU。"""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('comparison', ROOT / 'compare.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
manifest = json.loads((ROOT / 'variants_manifest.json').read_text())
shapes = [(4, 8), (256, 512), (2048, 4096), (2048, 7168), (2048, 7169),
          (2048, 8192), (2048, 14336), (2048, 14337)]
base_header = ROOT / 'csrc/simt/sparse_kv_plan_launch.h'
fields = re.findall(r'\b(uint64_t|int64_t|int32_t|uint32_t)\s+(\w+);',
                    base_header.read_text().split('struct SparseKvPlanTilingData {', 1)[1].split('};', 1)[0])
variant_header = (ROOT / 'csrc/variants/variant_tiling_data.h').read_text()
assert fields == re.findall(r'\b(uint64_t|int64_t|int32_t|uint32_t)\s+(\w+);', variant_header)
for v in manifest['variants']:
    path = ROOT / 'csrc/variants' / v['name']
    for filename, field in [('sparse_kv_plan_compute.h', 'compute_sha256'), ('sparse_kv_plan_config.h', 'config_sha256')]:
        assert hashlib.sha256((path / filename).read_bytes()).hexdigest() == v[field], v['name']
    text = (path / 'sparse_kv_plan_launch.cpp').read_text()
    assert '<<<blockDim, dynUBufSize, stream>>>' in text
    assert 'GET_TILING_DATA' not in text and 'REGISTER_TILING' not in text
    for _, field in fields:
        assert f'tilingData.{field} = tilingGm->{field};' in text
    declaration = re.search(r'extern "C" void (launch_sparse_kv_plan_v\d+)\(([^)]*)\)', text)
    assert declaration and declaration.group(0) + ';' in (ROOT / 'csrc/variants/variant_launch.h').read_text()
    config = (path / 'sparse_kv_plan_config.h').read_text()
    assert 'PLAN_THREADS = 2048U' in config
    if v['capacity_arrays'] == 1:
        assert 'return 3U * hashCapacity + capacity + topk;' in config

with tempfile.TemporaryDirectory(prefix='variant_layout_check_') as temporary:
    tmp = Path(temporary)
    cpp = tmp / 'layout.cpp'
    cpp.write_text(f'#include "{base_header}"\n#include <iostream>\nint main() {{\n' +
                   '\n'.join(f'{{auto t=MakeSparseKvPlanTiling(32,{t},{c},262144,32,4096,4096,64,{a});'
                             'std::cout<<t.hashCapacity<<" "<<t.workspaceRowElements<<" "<<t.localMemoryBytes<<"\\n";}'
                             for a in (1, 2) for t, c in shapes) + '\n}\n')
    subprocess.run(['g++', '-std=c++17', str(cpp), '-o', str(tmp / 'layout')], check=True)
    output = subprocess.check_output([str(tmp / 'layout')], text=True).splitlines()
    for (a, (topk, capacity)), line in zip(((a, s) for a in (1, 2) for s in shapes), output):
        name = '03_c_compact_partition' if a == 1 else 'simt'
        layout = b.plan_layout(topk, capacity, 32, name)
        assert list(map(int, line.split())) == [layout['hash_capacity'], layout['row_elements'], layout['local_memory_bytes']]
        assert layout['workspace_elements'] == (1 if layout['scratch_path'] == 'UB' else 32 * layout['row_elements'])
    # 实际生产配置头的公式也参与编译，防止 Python/catalog 与配置不一致。
    for v in manifest['variants']:
        header = ROOT / 'csrc/variants' / v['name'] / 'sparse_kv_plan_config.h'
        assertions = ['static_assert(sparse_kv_plan::PLAN_THREADS == 2048U);',
                      'static_assert(sparse_kv_plan::PLAN_SCAN_BYTES == 544U);']
        if v['capacity_arrays'] == 1:
            for t, c in shapes:
                layout = b.plan_layout(t, c, 32, v['name'])
                assertions.append(f'static_assert(sparse_kv_plan::RowWorkspaceElements({layout["hash_capacity"]},{c},{t}) == {layout["row_elements"]});')
        cpp.write_text(f'#include "{header}"\n'+'\n'.join(assertions))
        subprocess.run(['g++', '-std=c++17', '-fsyntax-only', str(cpp)], check=True)
print('PASS: 11 source/config hashes and direct ABIs; 16 C++/Python tilings including UB/GM boundaries; 11 production configs')
