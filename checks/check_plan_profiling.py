#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline CSV and runner checks; no torch/CANN/NPU execution."""
import contextlib
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('plan_profiler', ROOT / 'profile_plan_variants.py')
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)
ENTRIES = {name: f'sparse_kv_plan_kernel_{name}' for name in
           ['baseline', 'ab']}
ENTRIES['baseline'] = 'sparse_kv_plan_kernel_combo_bcde_fused_1024'


class ProfilingChecks(unittest.TestCase):
    def parse(self, rows, fields=('Name', 'Task Type', 'Duration(us)')):
        with tempfile.TemporaryDirectory(prefix='plan_csv_check_') as temporary:
            path = Path(temporary) / 'kernels.csv'
            with path.open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(fields)
                writer.writerows(rows)
            return profile.summarize_csv(path, np, ENTRIES)

    def valid_rows(self):
        first, second = ENTRIES.values()
        return [(first, 'AI_VECTOR_CORE', 2), (first, 'AI_VECTOR_CORE', 4),
                (second, 'AI_VECTOR_CORE', 1), (second, 'AI_VECTOR_CORE', 3)]

    def test_kernel_names_and_copy_exclusion(self):
        rows = self.valid_rows() + [('TensorMove', 'MEMCPY_ASYNC', 900),
                                   (ENTRIES['baseline'] + '_tiling', 'MEMCPY_ASYNC', 800)]
        records = self.parse(rows)
        result = profile.validate_records(records, tuple(ENTRIES), 2, 'baseline')
        self.assertEqual(result['baseline']['p50_us'], 3)
        self.assertEqual(result['ab']['p50_us'], 2)
        self.assertEqual(result['ab']['speedup_vs_reference_p50'], 1.5)
        self.assertEqual(len(records), 2)

    def test_alternate_csv_fields(self):
        self.assertEqual(len(self.parse(self.valid_rows(),
                                        ('Kernel Name', 'Task_Type', 'Duration(μs)'))), 2)

    def test_missing_and_wrong_count(self):
        for rows in [self.valid_rows()[:2], self.valid_rows()[:-1]]:
            with self.subTest(rows=rows), self.assertRaises(RuntimeError):
                profile.validate_records(self.parse(rows), tuple(ENTRIES), 2, 'baseline')

    def test_duplicate_kernel_groups(self):
        rows = self.valid_rows()
        rows[1] = ('wrapped(' + ENTRIES['baseline'] + ')', 'AI_VECTOR_CORE', 4)
        with self.assertRaises(RuntimeError):
            profile.validate_records(self.parse(rows), tuple(ENTRIES), 2, 'baseline')

    def test_bad_durations(self):
        for duration in [0, -1, 'nan', 'inf']:
            with self.subTest(duration=duration), self.assertRaises(RuntimeError):
                self.parse([(ENTRIES['baseline'], 'AI_VECTOR_CORE', duration)])

    def test_unrecognized_columns(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.parse(self.valid_rows(), ('Name', 'Task Type', 'cycles')), [])

    def runner(self, device):
        with tempfile.TemporaryDirectory(prefix='plan_runner_check_') as temporary:
            root = Path(temporary)
            script = root / 'run_plan_suite.sh'
            script.write_bytes((ROOT / script.name).read_bytes())
            for name in ['compare.py', 'plan_variants_manifest.json', 'profile_mtp_matrix.py']:
                (root / name).touch()
            (root / 'build').mkdir()
            (root / 'build/libsimt_plan_variants_ops.so').touch()
            calls = root / 'calls.jsonl'
            stub = root / 'fake_python'
            stub.write_text('#!/usr/bin/env python3\nimport json,os,sys\n'
                            'with open(os.environ["PEER_TEST_CALLS"],"a") as stream:\n'
                            '    stream.write(json.dumps(sys.argv[1:])+"\\n")\n')
            stub.chmod(0o755)
            env = dict(os.environ, PYTHON=str(stub), PEER_TEST_CALLS=str(calls),
                       PLAN_PROFILE_OUTPUT=str(root / 'results'), REPEAT='68', REQUESTS='32',
                       FIXED_QUERY_ROWS='32', GROUP='all')
            env.pop('PLAN_LIBRARY', None)
            result = subprocess.run(['bash', str(script), device], env=env,
                                    text=True, capture_output=True, timeout=10)
            recorded = [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
            return result, recorded

    def test_runner_audit_before_capture(self):
        result, calls = self.runner('0')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual([Path(call[2]).name for call in calls], ['check_mtp_scan.py', 'profile_mtp_matrix.py'])
        self.assertIn('--production', calls[0])
        self.assertIn('--correctness-report', calls[1])
        for call in calls:
            self.assertEqual(call[call.index('--device') + 1], '0')
        self.assertEqual(calls[1][calls[1].index('--repeat') + 1], '68')
        self.assertEqual(calls[1][calls[1].index('--mtp') + 1:calls[1].index('--mtp') + 3], ['0', '3'])

    def test_runner_invalid_device(self):
        result, calls = self.runner('invalid')
        self.assertEqual(result.returncode, 2)
        self.assertFalse(calls)


if __name__ == '__main__':
    unittest.main()
