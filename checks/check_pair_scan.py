#!/usr/bin/env python3
"""Test actual pair/grouped PublishWarpBases at production/future configs; 32 CPU threads."""
import os
from pathlib import Path
import re
import subprocess
import tempfile

from peer_build_info import ROOT, catalog_identity

variant = next(v for v in catalog_identity()['variants'] if v['name'] == 'combo_bcde_pair_scan')
source = ROOT / variant['path'] / 'upstream'
helpers = ROOT / 'checks/simt_variants'
with tempfile.TemporaryDirectory(prefix='pair_scan_check_') as temporary:
    base = Path(temporary)
    for threads in [32, 96, 1024, 2048, 3072]:
        build = base / str(threads)
        build.mkdir()
        config = (source / 'sparse_kv_plan_config.h').read_text()
        config = re.sub(r'PLAN_THREADS = \d+U', f'PLAN_THREADS = {threads}U', config)
        (build / 'sparse_kv_plan_config.h').write_text(config)
        (build / 'sparse_kv_plan_launch.h').write_text((source / 'sparse_kv_plan_launch.h').read_text())
        kernel = (source / 'sparse_kv_plan_kernel.cpp').read_text()
        (build / 'kernel_under_test.h').write_text(kernel[:kernel.index('extern "C" __global__')])
        for filename in ['kernel_operator.h', 'simt_api/asc_simt.h', 'simt_api/device_atomic_functions.h',
                         'simt_api/device_sync_functions.h', 'simt_api/device_warp_functions.h']:
            path = build / filename
            path.parent.mkdir(exist_ok=True)
            path.write_text('// CPU shim in cpu_runtime.h.\n')
        binary = build / 'check'
        subprocess.run(['g++', '-std=c++20', '-O1', '-g', '-pthread', '-no-pie',
                        '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
                        '-I', str(build), '-I', str(helpers), str(helpers / 'scan_driver.cpp'),
                        '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True, env=dict(os.environ, ASAN_OPTIONS='detect_leaks=0'))
