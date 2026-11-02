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
from joint_plan_cases import make_case, initial_arrays, validate, layout, INPUTS, OUTPUTS, validate_abcdf
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
    for name in ('sparse_kv_joint_plan.h', 'options.h', 'bitmap_plan.h', 'turbo_plan.h', 'turbo_union_experiments.h'):
        (directory / name).write_bytes((ROOT / 'csrc/joint_variants' / name).read_bytes())
    (directory / 'launch.h').write_bytes((ROOT / 'csrc/joint_plan/launch.h').read_bytes())
    for name in ('kernel_operator.h', 'simt_api/asc_simt.h', 'simt_api/device_atomic_functions.h',
                 'simt_api/device_sync_functions.h', 'simt_api/device_warp_functions.h'):
        path = directory / name
        path.parent.mkdir(exist_ok=True)
        path.write_text('// CPU definitions are in cpu_runtime.h\n')
    if mask == 0:
        common = ROOT / 'csrc/plan_variants/common'
        source = (common / 'plan_kernel_impl.h').read_text()
        (directory / 'kernel_under_test.h').write_text(source[:source.index('extern "C" __global__')])
        config = (common / 'sparse_kv_plan_config.h').read_text().replace('PLAN_THREADS = 1024U', 'PLAN_THREADS = 64U')
        (directory / 'sparse_kv_plan_config.h').write_text(config)
        (directory / 'sparse_kv_plan_launch.h').write_bytes((common / 'sparse_kv_plan_launch.h').read_bytes())
        (directory / 'variant_config.h').write_bytes((ROOT / 'csrc/plan_variants/13_abcdf/variant_config.h').read_bytes())
    library = directory / 'kernel.so'
    subprocess.run(['g++', '-std=c++20', '-O1', '-pthread', '-shared', '-fPIC',
                    '-fsanitize=undefined', '-fno-sanitize-recover=all', '-fvisibility=hidden',
                    '-DJOINT_OPT_MASK=' + str(mask), '-I' + str(directory),
                    '-I' + str(ROOT / 'checks/simt_variants'),
                    str(ROOT / 'checks/joint_variant_cpu_driver.cpp'), '-o', str(library)], check=True)
    return library


def execute(library, case, enabled, force_gm, before=None, bitmap=False, turbo=False, joint_entry=True, abcdf=False):
    info = layout(case['topk'], case['capacity'], case['rows'], enabled, case['max_token'], bitmap, turbo)
    scratch = max(info['scratch_elements'], 112 * 1024 // 4 + 1) if force_gm else info['scratch_elements']
    max_queries = (scratch - 2 * info['joint_hash_capacity'] - case['capacity']
                   if turbo and enabled else None)
    expected = initial_arrays(case, enabled, max_queries) if before is None else before
    arrays = {k: v.copy() for k, v in expected.items()}
    arrays.update({key: case[key].copy() for key in INPUTS})
    arrays['active_rows'] = np.array([case.get('active_rows', case['rows'])], np.int32)
    gm_path = force_gm or info['scratch_path'] == 'GM'
    arrays['workspace'] = np.full(case['rows'] * (scratch + case['max_token']) if bitmap else (case['rows'] * scratch if gm_path else 1), -1234567, np.int32)
    # Guard every input/output/scratch. No input restoration after invocation.
    allocations = {}
    for key, array in arrays.items():
        backing = np.full(array.size + 32, 0x1234567, array.dtype)
        view = backing[16:-16].reshape(array.shape)
        view[:] = array
        arrays[key], allocations[key] = view, backing
    scan_guard = np.full(scratch + 32 + 32, 0x1234567, np.int32)
    scan = scan_guard[16:-16]
    pointers = (ctypes.c_void_p * 17)(*(arrays[key].ctypes.data for key in TENSOR_ORDER))
    dims = np.array([3, info['hash_capacity'], scratch if abcdf else 3 * info['hash_capacity'] + case['capacity'] + case['topk'],
                     min(case['rows'], max(0, case.get('active_rows', case['rows']))),
                     case['topk'], case['capacity'], case['max_token'], case['requests'],
                     case['block_size'],
                     info['joint_hash_capacity'], scratch, 1 if joint_entry else 0], np.int64)
    library.run_variant(pointers, dims.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                        scan.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
    for name, array in allocations.items():
        assert np.all(array[:16] == 0x1234567) and np.all(array[-16:] == 0x1234567), name
    assert np.all(scan_guard[:16] == 0x1234567) and np.all(scan_guard[-16:] == 0x1234567)
    for key in INPUTS:
        np.testing.assert_array_equal(arrays[key], case[key], err_msg='read-only/' + key)
    if not gm_path and not bitmap:
        assert np.all(arrays['workspace'] == -1234567)
    if abcdf:
        validate_abcdf(case, expected, arrays)
    else:
        validate(case, expected, arrays, enabled, max_queries)
    return {key: arrays[key].copy() for key in expected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variants', nargs='+', choices=["abcdf", *MASKS], default=["abcdf", *MASKS])
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    records = []
    with tempfile.TemporaryDirectory(prefix='joint_variants_cpu_') as tmp:
        for name in args.variants:
            directory = Path(tmp) / name
            directory.mkdir()
            lib = ctypes.CDLL(str(build(directory, (0 if name == "abcdf" else MASKS[name]))))
            lib.run_variant.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int64),
                                        ctypes.POINTER(ctypes.c_int32)]
            lib.run_variant.restype = None
            checked = 0
            if name == "abcdf":
                for gm in (False, True):
                    for mtp in (0, 1, 3):
                        for scenario in ('cold_reset', 'full_cache_75pct', 'full_cache_stable_hit'):
                            case = make_case(2, 65, 128, 2048, mtp, scenario=scenario, overlap='independent', block_size=8)
                            before = execute(lib, case, False, gm, abcdf=True)
                            execute(lib, case, False, gm, before, abcdf=True)
                            checked += 2
                records.append({'variant': name, 'checks': checked})
                print(name, checked, 'VF semantic checks passed', flush=True)
                continue
            for force_gm in (False, True):
                for enabled in (False, True):
                    for overlap in ('shared', 'independent', 'disjoint'):
                        case = make_case(2, 65, 128, 2048, 3, overlap=overlap, block_size=8)
                        before = execute(lib, case, enabled, force_gm, bitmap=name == "bitmap", turbo=name.startswith("turbo"))
                        # Continue device-produced state, including joint -> private fallback -> joint.
                        case = make_case(2, 65, 128, 2048, 3, overlap='disjoint', block_size=8)
                        before = execute(lib, case, enabled, force_gm, before, bitmap=name == "bitmap", turbo=name.startswith("turbo"))
                        case = make_case(2, 65, 128, 2048, 3, overlap='independent', block_size=8)
                        case['req_ids'] += 99
                        case['topk_indices'][0, :5] = [-1, 2048, case['context'] + 3, 5, 5]
                        case['stable_prefix_lens'][0] = -1
                        case['active_rows'] = 5
                        execute(lib, case, enabled, force_gm, before, bitmap=name == "bitmap", turbo=name.startswith("turbo"))
                        checked += 3
            if name == "bitmap":
                # 最后一个不完整bitmap字、最高token和实际256K值域。
                edge = make_case(1, 129, 512, 262145, 3, overlap='independent', block_size=8)
                edge['topk_indices'][:, 0] = 262144
                edge['visible_seq_lens'][:] = 262145
                for gm in (False, True):
                    execute(lib, edge, True, gm, bitmap=True)
                    checked += 1
            if name.startswith("turbo"):
                for scenario in ('cold_reset', 'full_cache_90pct', 'full_cache_stable_hit'):
                    for mtp in (0, 1, 3):
                        edge = make_case(5, 65, 160, 2048, mtp, scenario=scenario, overlap='independent', block_size=8)
                        execute(lib, edge, mtp > 0, False, turbo=True, joint_entry=True)
                        checked += 1
                    priv = make_case(5, 65, 160, 2048, 0, scenario=scenario, overlap='independent', block_size=8)
                    for gm in (False, True):
                        execute(lib, priv, False, gm, turbo=True, joint_entry=False)
                        checked += 1
                non_pow2 = make_case(4, 21, 63, 1024, 3, scenario='full_cache_75pct', overlap='partial', block_size=7)
                non_pow2['token_to_req'][12:] = -1
                execute(lib, non_pow2, True, False, turbo=True, joint_entry=True)
                execute(lib, non_pow2, False, False, turbo=True, joint_entry=True)
                execute(lib, non_pow2, False, False, turbo=True, joint_entry=False)
                checked += 3
                # Group with >4 rows whose queryBuckets exceed scratchElements falls back to private rows cleanly.
                wide = make_case(2, 65, 128, 2048, 3, scenario='full_cache_75pct', overlap='shared', block_size=8)
                wide['token_to_req'][:] = 0
                execute(lib, wide, True, False, turbo=True, joint_entry=True)
                checked += 1
            if name.startswith("turbo"):
                # Actual 112 KiB boundary, not just a small shape with forced GM addressing.
                for scenario in ('cold_reset', 'full_cache_75pct'):
                    edge = make_case(1, 2048, 4096, 262144, 3, scenario=scenario, overlap='independent')
                    execute(lib, edge, True, False, turbo=True)
                    checked += 1
                # Natural GM scratch selection and non-packed partition path.
                edge = make_case(1, 33, 32768, 262144, 1, overlap='shared', block_size=64)
                execute(lib, edge, True, False, turbo=True)
                checked += 1
            if name.startswith("turbo"):
                def token_hash(token):
                    v = token ^ (token >> 16)
                    v = (v * 0x7feb352d) & 0xffffffff
                    v ^= v >> 15
                    v = (v * 0x846ca68b) & 0xffffffff
                    return (v ^ (v >> 16)) & 255
                collisions = [t for t in range(1, 65520) if token_hash(t) == 0][:129]
                assert len(collisions) == 129
                for gm in (False, True):
                    for scenario in ('cold_reset', 'full_cache_75pct'):
                        for unique in (1, 128, 129):
                            edge = make_case(1, 65, 128, 65536, 3, scenario=scenario, block_size=8)
                            # 同桶冲突链，跨warp/跨tile重复；union恰好capacity及capacity+1。
                            for row in range(4):
                                edge['topk_indices'][row] = [collisions[(row * 65 + p) % unique] for p in range(65)]
                            before = execute(lib, edge, True, gm, turbo=True)
                            execute(lib, edge, True, gm, before, turbo=True)
                            checked += 2
                        # 首行无效/未来token，follower可见但仍须插入；全无效时union=0。
                        for empty in (False, True):
                            edge = make_case(1, 65, 128, 65536, 3, scenario=scenario, block_size=8)
                            edge['topk_indices'][:] = -1
                            if not empty:
                                edge['topk_indices'][:, ::2] = edge['context'] + 1
                                edge['topk_indices'][:, 1] = edge['max_token']
                            execute(lib, edge, True, gm, turbo=True)
                            checked += 1
            records.append({'variant': name, 'checks': checked})
            print(name, checked, 'VF semantic checks passed', flush=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({'scope': 'CPU64 shim + UBSan, not NPU', 'records': records}, indent=2) + '\n')


if __name__ == '__main__':
    main()
