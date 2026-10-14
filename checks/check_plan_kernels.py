#!/usr/bin/env python3
"""Actual Plan micro-variant VF dispatch with CPU shims and ASAN/UBSan; not NPU validation."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'checks/simt_variants'
sys.path.insert(0, str(ROOT / 'checks'))
from plan_build_info import catalog_identity


def driver_source():
    source = (HERE / 'cpu_row_driver.cpp').read_text()
    start = source.index('void CheckState(')
    end = source.index('\n}\n', start) + 3
    source = source[:start] + '''void CheckState(CpuWorkers& workers, State& s, int active, bool, int stride = 1) {
  State expected = s;
  for (int row = 0; row < active; ++row) OracleRow(expected, row);
  const auto tiling = MakeSparseKvPlanTiling(s.rows, s.topk, s.capacity, s.maxToken,
                                            s.requests, s.blocks, s.hostBlocks, s.blockSize);
  const uint64_t elements = tiling.workspaceRowElements;
  const bool wholeUb = elements <= sparse_kv_plan::PLAN_ROW_UB_LIMIT_ELEMENTS;
  Guarded<int32_t> local(tiling.localMemoryBytes / sizeof(int32_t), 0x4567);
  Guarded<int32_t> global(wholeUb ? 1 : s.rows * elements, 0x9876);
  for (int start = 0; start < stride; ++start) {
    workers.Run([&] {
      SparseKvPlanRuntimeVf(local.data(), s.reqIds.data(), s.lastIds.data(), s.tokens.data(),
          s.stable.data(), s.visible.data(), s.mapping.data(), s.table.data(), s.slots.data(),
          s.lru.data(), s.current.data(), s.missCount.data(), s.missTokens.data(), s.missSlots.data(),
          global.data(), tiling.hashCapacity, elements, start, stride, active, s.topk, s.capacity,
          s.maxToken, s.requests, s.blocks, s.hostBlocks, s.blockSize);
    });
  }
  s.Check(); local.Check(); global.Check();
  Compare(s.lastIds, expected.lastIds, "lastIds"); Compare(s.slots, expected.slots, "slots");
  Compare(s.lru, expected.lru, "lru"); Compare(s.current, expected.current, "current");
  Compare(s.missCount, expected.missCount, "missCount"); Compare(s.missTokens, expected.missTokens, "missTokens");
  Compare(s.missSlots, expected.missSlots, "missSlots");
  if (wholeUb) for (size_t i = 0; i < global.count; ++i)
    if (global[i] != 0x9876) throw std::runtime_error("UB path touched GM scratch");
}
''' + source[end:]
    old = '{std::pair{256, 512}, {2048, 4096}, {2048, 8192}}'
    assert old in source
    source = source.replace(old, '{std::pair{256, 512}, {2048, 4096}, {2048, 8192}, {2049, 4096}, {4096, 8192}, {4097, 8192}}')
    source = source.replace('  State mtp(4, 7, 11);', '  State negative(2, 7, 17);\n  negative.visible[0] = -1; negative.visible[1] = -100;\n  for (int p = 0; p < 14; ++p) negative.tokens[p] = p;\n  CheckState(workers, negative, 2, ub); ++checks;\n  State oddPages(2, 9, 17);\n  oddPages.blockSize = 7;\n  for (int p = 0; p < 18; ++p) oddPages.tokens[p] = p * 7;\n  CheckState(workers, oddPages, 2, ub); ++checks;\n  CheckState(workers, oddPages, 2, ub); ++checks;\n' + '  State mtp(4, 7, 11);')
    source = source.replace('    std::cout << "PASS "', '#if OWNER_TESTS\n      for (int capacity : {32768, 32769}) {\n        State owner(1, 7, capacity);\n        owner.lastIds[0] = owner.reqIds[0];\n        for (int p = 0; p < owner.topk; ++p) {\n          owner.tokens[p] = p;\n          owner.slots[capacity - owner.topk + p] = p;\n        }\n        CheckState(workers, owner, 1, false); ++checks;\n      }\n#endif\n' + '    std::cout << "PASS "')
    return source


def main():
    catalog = catalog_identity()
    names = [x['name'] for x in catalog['variants']]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variants', nargs='+', choices=names, default=names)
    parser.add_argument('--threads', type=int, choices=[64, 96, 128], default=64)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--large', action='store_true')
    args = parser.parse_args()
    records = []
    with tempfile.TemporaryDirectory(prefix='plan_micro_cpu_') as temporary:
        for variant in catalog['variants']:
            name = variant['name']
            if name not in args.variants:
                continue
            source = ROOT / variant['path']
            source = source / 'upstream' if variant['name'] == 'baseline' else ROOT / 'csrc/plan_variants/common'
            build = Path(temporary) / name
            build.mkdir()
            config = (source / 'sparse_kv_plan_config.h').read_text()
            config, count = re.subn(r'PLAN_THREADS = \d+U', f'PLAN_THREADS = {args.threads}U', config)
            assert count == 1
            (build / 'sparse_kv_plan_config.h').write_text(config)
            (build / 'sparse_kv_plan_launch.h').write_text((source / 'sparse_kv_plan_launch.h').read_text())
            kernel = (source / ('sparse_kv_plan_kernel.cpp' if variant['name'] == 'baseline' else 'plan_kernel_impl.h')).read_text()
            if variant['name'] != 'baseline':
                (build / 'variant_config.h').write_bytes((ROOT / variant['path'] / 'variant_config.h').read_bytes())
            (build / 'kernel_under_test.h').write_text(kernel[:kernel.index('extern "C" __global__')])
            for name_ in ['kernel_operator.h', 'simt_api/asc_simt.h', 'simt_api/device_atomic_functions.h',
                          'simt_api/device_sync_functions.h', 'simt_api/device_warp_functions.h']:
                path = build / name_
                path.parent.mkdir(exist_ok=True)
                path.write_text('// CPU shim defined by cpu_runtime.h.\n')
            cpp = build / 'driver.cpp'
            cpp.write_text(driver_source())
            binary = build / 'check'
            command = ['g++', '-std=c++20', '-O1', '-g', '-pthread', '-no-pie',
                       '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
                       '-DOWNER_TESTS=' + ('1' if 'D' in variant['points'] else '0'),
                       '-I', str(build), '-I', str(HERE), str(cpp), '-o', str(binary)]
            compiled = subprocess.run(command, text=True, capture_output=True)
            if compiled.returncode:
                print(compiled.stderr, file=sys.stderr)
                compiled.check_returncode()
            env = dict(os.environ, ASAN_OPTIONS='detect_leaks=0:halt_on_error=1', UBSAN_OPTIONS='halt_on_error=1')
            result = subprocess.run([str(binary), *(['--large'] if args.large else [])], text=True, capture_output=True, timeout=240, env=env)
            if result.returncode:
                print(result.stdout + result.stderr, file=sys.stderr)
                result.check_returncode()
            print(name, result.stdout.strip(), flush=True)
            records.append({'name': name, 'result': result.stdout.strip()})
            if 'F' in variant['points'] or variant['name'] == 'baseline':
                original_config = (source / 'sparse_kv_plan_config.h').read_text()
                (build / 'sparse_kv_plan_config.h').write_text(original_config)
                scan_binary = build / 'scan_check'
                subprocess.run(['g++', '-std=c++20', '-O1', '-pthread', '-no-pie',
                                '-fsanitize=address,undefined', '-I', str(build), '-I', str(HERE),
                                str(HERE / 'scan_driver.cpp'), '-o', str(scan_binary)], check=True)
                result = subprocess.run([str(scan_binary)], text=True, capture_output=True, check=True, env=env)
                print(name, result.stdout.strip(), flush=True)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({'boundary': 'CPU shims, not NPU hardware', 'checks': records}, indent=2)+'\n')


if __name__ == '__main__':
    main()
