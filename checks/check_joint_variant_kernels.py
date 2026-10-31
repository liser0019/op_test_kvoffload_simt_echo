#!/usr/bin/env python3
"""真实Joint候选VF的CPU语义仿真(64线程+UBSan)；不是NPU编译/地址空间/性能验证。"""
from pathlib import Path
import argparse
import ctypes
import json
import os
import re
import subprocess
import sys
import tempfile
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from joint_variants import MASKS
from joint_plan_cases import make_case, initial_arrays, validate, layout, INPUTS, OUTPUTS
from joint_plan_common import TENSOR_ORDER


def build(directory, mask):
    source = (ROOT / "csrc/joint_variants/kernel.cpp").read_text()
    source = source[:source.index("// Direct launch adaptation only:")]
    source = source.replace('../joint_plan/', '')
    (directory / 'kernel_under_test.h').write_text(source)
    config = (ROOT / 'csrc/joint_plan/sparse_kv_plan_config.h').read_text()
    config, count = re.subn(r'PLAN_THREADS = 1024U', 'PLAN_THREADS = 64U', config)
    assert count == 1
    (directory / 'sparse_kv_plan_config.h').write_text(config)
    for name in ('sparse_kv_joint_plan.h', 'options.h'):
        (directory / name).write_bytes((ROOT / 'csrc/joint_variants' / name).read_bytes())
    (directory / 'launch.h').write_bytes((ROOT / 'csrc/joint_plan/launch.h').read_bytes())
    for name in ('kernel_operator.h', 'simt_api/asc_simt.h', 'simt_api/device_atomic_functions.h',
                 'simt_api/device_sync_functions.h', 'simt_api/device_warp_functions.h'):
        path = directory / name
        path.parent.mkdir(exist_ok=True)
        path.write_text('// CPU definitions are in cpu_runtime.h\n')
    library = directory / 'kernel.so'
    subprocess.run(['g++', '-std=c++20', '-O1', '-pthread', '-shared', '-fPIC',
                    '-fsanitize=undefined', '-fno-sanitize-recover=all', '-fvisibility=hidden',
                    '-DJOINT_OPT_MASK=' + str(mask), '-I' + str(directory),
                    '-I' + str(ROOT / 'checks/simt_variants'),
                    str(ROOT / 'checks/joint_variant_cpu_driver.cpp'), '-o', str(library)], check=True)
    return library


def execute(library, case, enabled, force_gm, before=None):
    expected = initial_arrays(case, enabled) if before is None else before
    arrays = {k: v.copy() for k, v in expected.items()}
    arrays.update({key: case[key].copy() for key in INPUTS})
    arrays['active_rows'] = np.array([case.get('active_rows', case['rows'])], np.int32)
    info = layout(case['topk'], case['capacity'], case['rows'], enabled)
    scratch = max(info['scratch_elements'], 112 * 1024 // 4 + 1) if force_gm else info['scratch_elements']
    arrays['workspace'] = np.full(case['rows'] * scratch if force_gm else 1, -1234567, np.int32)
    # Guard every input/output/scratch. No input restoration after invocation.
    allocations = {}
    for key, array in arrays.items():
        backing = np.full(array.size + 32, 0x1234567, array.dtype)
        view = backing[16:-16].reshape(array.shape)
        view[:] = array
        arrays[key], allocations[key] = view, backing
    scan_guard = np.full(scratch + 32 + 32, 0x1234567, np.int32)
    scan = scan_guard[16:-16]
    pointers = (ctypes.c_void_p * 18)(*(arrays[key].ctypes.data for key in TENSOR_ORDER))
    dims = np.array([3, info['hash_capacity'], 3 * info['hash_capacity'] + case['capacity'] + case['topk'],
                     min(case['rows'], max(0, case.get('active_rows', case['rows']))),
                     case['topk'], case['capacity'], case['max_token'], case['requests'],
                     case['block_table'].shape[1], case['host_num_blocks'], case['block_size'],
                     info['joint_hash_capacity'], scratch], np.int64)
    library.run_variant(pointers, dims.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                        scan.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
    for name, array in allocations.items():
        assert np.all(array[:16] == 0x1234567) and np.all(array[-16:] == 0x1234567), name
    assert np.all(scan_guard[:16] == 0x1234567) and np.all(scan_guard[-16:] == 0x1234567)
    for key in INPUTS:
        np.testing.assert_array_equal(arrays[key], case[key], err_msg='read-only/' + key)
    if not force_gm:
        assert np.all(arrays['workspace'] == -1234567)
    validate(case, expected, arrays, enabled)
    return {key: arrays[key].copy() for key in expected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variants', nargs='+', choices=list(MASKS), default=list(MASKS))
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    records = []
    with tempfile.TemporaryDirectory(prefix='joint_variants_cpu_') as tmp:
        for name in args.variants:
            directory = Path(tmp) / name
            directory.mkdir()
            lib = ctypes.CDLL(str(build(directory, MASKS[name])))
            lib.run_variant.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int64),
                                        ctypes.POINTER(ctypes.c_int32)]
            lib.run_variant.restype = None
            checked = 0
            for force_gm in (False, True):
                for enabled in (False, True):
                    for overlap in ('shared', 'independent', 'disjoint'):
                        case = make_case(2, 65, 128, 2048, 3, overlap=overlap, block_size=8)
                        before = execute(lib, case, enabled, force_gm)
                        # Continue device-produced state, including joint -> private fallback -> joint.
                        case = make_case(2, 65, 128, 2048, 3, overlap='disjoint', block_size=8)
                        before = execute(lib, case, enabled, force_gm, before)
                        case = make_case(2, 65, 128, 2048, 3, overlap='independent', block_size=8)
                        case['req_ids'] += 99
                        case['topk_indices'][0, :5] = [-1, 2048, case['context'] + 3, 5, 5]
                        case['block_table'][0, 0] = -1
                        case['stable_prefix_lens'][0] = -1
                        case['active_rows'] = 5
                        execute(lib, case, enabled, force_gm, before)
                        checked += 3
            records.append({'variant': name, 'checks': checked})
            print(name, checked, 'VF semantic checks passed', flush=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({'scope': 'CPU64 shim + UBSan, not NPU', 'records': records}, indent=2) + '\n')


if __name__ == '__main__':
    main()
