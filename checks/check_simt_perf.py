#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""服务器侧 CPU 检查：不加载 NPU。覆盖计时解析/跨行统计/原算法来源。

遵守本仓 CLAUDE.md：开发机不执行；由 owner 在服务器运行本文件。
"""
import hashlib
from pathlib import Path
import re
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import compare as bench
import simt_perf_step as perf


def row_stamps(path, start, durations):
    matrix = np.full((1, perf.STAMP_COUNT), perf.SENTINEL, np.int32)
    values = np.array([start, *[start + x for x in np.cumsum(durations)]], np.uint64)
    matrix[0, list(perf.PATH_SLOTS[path])] = (values & 0xffffffff).astype(np.uint32).view(np.int32)
    return matrix[0]


class SimtPerfTest(unittest.TestCase):
    def test_algorithm_is_exact_abcdf_after_removing_observers(self):
        # 仅允许附加观察点/参数，计算语句、分支和原有同步必须与导入源一致。
        data = (ROOT / 'csrc/simt/sparse_kv_plan_kernel.cpp').read_bytes()
        for marker in (b'HELPERS', b'POINTER'):
            pattern = rb'// SIMT_PERF_' + marker + rb'_BEGIN\n.*?// SIMT_PERF_' + marker + rb'_END\n'
            data, count = re.subn(pattern, b'', data, flags=re.S)
            self.assertEqual(count, 1)
        data = data.replace(b'#include "sparse_kv_plan_perf.h"\n', b'')
        data = re.sub(rb'^  SIMT_STAMP\(SIMT_[A-Z_]+\);\n', b'', data, flags=re.M)
        data = data.replace(b' SIMT_PERF_PARAM', b'').replace(b' SIMT_PERF_ARG', b'')
        canonical = re.sub(rb'[ \t\r\n\v\f]', b'', data)
        self.assertEqual(hashlib.sha256(canonical).hexdigest(),
                         '981d71395f9168f6d9064b951a3f7b7fcd5a3311356e86298601772673accf38')
        originals = {
            'sparse_kv_plan_config.h': 'f5d07992fe8ecab41f9fed49a8a8f7683b38b46d19cd678e5746b7f898398620',
            'sparse_kv_plan_launch.h': '64bad93a9edc031f358eebe312ba9be8dc784a17229bffec7009ba2b410b0222',
            'variant_config.h': 'e46868124d3b5b534852f3fb6ad2044af9671e43414d43b15d677896e7a9ec43'}
        for name, digest in originals.items():
            self.assertEqual(hashlib.sha256((ROOT / 'csrc/simt' / name).read_bytes()).hexdigest(), digest)

    def test_ub_and_gm_layout(self):
        ub = bench.simt_workspace_layout(32, 2048, 4096, True)
        self.assertEqual((ub['row_elements'], ub['stamp_offset'], ub['workspace_elements']), (18432, 128, 4224))
        gm = bench.simt_workspace_layout(2, 4096, 4096, True)
        self.assertEqual((gm['row_elements'], gm['stamp_offset'], gm['workspace_elements']), (32768, 65536, 65792))
        self.assertEqual(bench.simt_workspace_layout(2, 4096, 4096, False)['workspace_elements'], 65536)

    def test_wrap_and_signed_stamps(self):
        d = [1, 2, 3, 4, 5, 6, 7]
        matrix = row_stamps('warm', 0xfffffff8, d)[None, :]
        decoded = perf.decode_stamps(matrix, ['warm'])[0]
        self.assertEqual(decoded['cycles'].tolist(), d)
        self.assertEqual(decoded['span'], sum(d))

    def test_cold_early_return(self):
        matrix = row_stamps('cold', 100, [2, 3, 4, 5, 6])[None, :]
        self.assertEqual(perf.decode_stamps(matrix, ['cold'])[0]['span'], 20)
        with self.assertRaises(RuntimeError):
            perf.decode_stamps(matrix, ['warm'])
        matrix[0, 3] = 123
        with self.assertRaises(RuntimeError):
            perf.decode_stamps(matrix, ['cold'])

    def test_sentinel_and_inactive_rows(self):
        matrix = np.stack([row_stamps('warm', 100, [1] * 7), np.full(12, perf.SENTINEL, np.int32)])
        self.assertEqual(len(perf.decode_stamps(matrix, ['warm', 'warm'], active_rows=1)), 1)
        with self.assertRaises(RuntimeError):
            perf.decode_stamps(matrix, ['warm', 'warm'])

    def test_backwards_clock_rejected(self):
        matrix = row_stamps('warm', 100, [1] * 7)[None, :]
        matrix[0, 3] = 1
        with self.assertRaises(RuntimeError):
            perf.decode_stamps(matrix, ['warm'])

    def test_multiple_rows_per_block(self):
        # block0: row0=[1]*7, row2=[3]*7，行间空隙 3 cycles；block1: row1=[2]*7。
        matrix = np.stack([row_stamps('warm', 100, [1] * 7), row_stamps('warm', 1000, [2] * 7),
                           row_stamps('warm', 110, [3] * 7)])
        sample = perf.aggregate_sample(matrix, perf.decode_stamps(matrix, ['warm'] * 3), 2)
        self.assertEqual(sample['groups']['warm']['stage_max_block_cycles'], [4] * 7)
        self.assertEqual(sample['max_block_span_cycles'], 31)
        self.assertEqual(sample['groups']['warm']['stage_skew_cycles'], [2] * 7)
        summary = perf.summarize([sample])
        self.assertAlmostEqual(sum(s['share_of_all_row_cycles_pct'] for s in summary), 100)

    def test_stage_peaks_are_not_total_span(self):
        a, b = [100, 1, 1, 1, 1, 1, 1], [1, 100, 1, 1, 1, 1, 1]
        matrix = np.stack([row_stamps('warm', 100, a), row_stamps('warm', 500, b)])
        sample = perf.aggregate_sample(matrix, perf.decode_stamps(matrix, ['warm', 'warm']), 2)
        self.assertEqual(sample['max_block_span_cycles'], 106)
        self.assertEqual(sum(sample['groups']['warm']['stage_max_block_cycles']), 205)

    def test_cpp_slot_contract(self):
        header = (ROOT / 'csrc/simt/sparse_kv_plan_perf.h').read_text()
        self.assertRegex(header, r'SIMT_PERF_NUM_STAMPS\s*=\s*12U')
        self.assertRegex(header, r'SIMT_PERF_ROW_STRIDE\s*=\s*128U')
        values = dict((k, int(v)) for k, v in re.findall(r'(SIMT_[A-Z_]+)\s*=\s*(\d+)', header))
        self.assertEqual([values[k] for k in ('SIMT_ENTRY', 'SIMT_PREPARE', 'SIMT_HASH_INIT',
            'SIMT_HASH_INSERT', 'SIMT_CLASSIFY', 'SIMT_HIT_MISS_SCAN', 'SIMT_ASSIGN', 'SIMT_LRU_END')], list(range(8)))
        self.assertEqual([values[k] for k in ('SIMT_COLD_CLEAR', 'SIMT_COLD_SCAN', 'SIMT_COLD_ASSIGN', 'SIMT_COLD_END')], [8, 9, 10, 11])


if __name__ == '__main__':
    unittest.main(verbosity=2)
