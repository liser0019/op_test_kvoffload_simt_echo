#!/usr/bin/env python3
"""服务器CPU辅助检查：输入契约、实际miss、地址模型、CSV样本与配对。没有NPU性能结论。"""
import csv
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import compare as plan
from full_cache_cases import make_full_cache_case, assert_case, describe, reference_transfer_ids, physical_tokens, payload_pattern
from profile_plan_transfer import read_kernel_rows, summarize
from plan_transfer_common import PLAN_KERNEL, TRANSFER_KERNEL


class InputChecks(unittest.TestCase):
    def test_full_cache_and_nested_hit_sets(self):
        cases = [make_full_cache_case(2, 33, 65, 512, p, seed=21) for p in (75, 90, 100)]
        for case, expected in zip(cases, (9, 4, 0)):
            golden = plan.reference_simt(case)
            assert_case(case, golden)
            np.testing.assert_array_equal(golden["miss_count"], expected)
        for key in ("topk_indices", "block_table", "lru_slots", "req_ids"):
            np.testing.assert_array_equal(cases[0][key], cases[1][key])
            np.testing.assert_array_equal(cases[0][key], cases[2][key])
        for row in range(2):
            old = set(cases[0]["slot_to_token"][row]) & set(cases[0]["topk_indices"][row])
            new = set(cases[1]["slot_to_token"][row]) & set(cases[1]["topk_indices"][row])
            self.assertTrue(old.issubset(new))
            for token in old:
                self.assertEqual(np.where(cases[0]["slot_to_token"][row] == token)[0][0],
                                 np.where(cases[1]["slot_to_token"][row] == token)[0][0])

    def test_production_counts_and_bytes(self):
        for percent, misses in ((75, 512), (90, 205), (100, 0)):
            case = make_full_cache_case(32, 2048, 4096, 262144, percent)
            self.assertTrue(np.all(case["slot_to_token"] >= 0))
            info = describe(case, 1024, 128)
            self.assertEqual(info["misses_per_row"], misses)
            self.assertEqual(info["payload_bytes_copied"], 32 * misses * 1152)
            self.assertEqual(info["host_pool_bytes"], 9 * 2**30)
            self.assertEqual(info["resident_payload_bytes"], 144 * 2**20)

    def test_transfer_address_model_by_hand(self):
        case = make_full_cache_case(2, 7, 17, 256, 75)
        table = np.array([[7, 6, 5, 4], [3, 2, 1, 0]], np.int32)
        # 对row0取request1，token70位于logical block1、offset6，physical page2。
        count = np.array([1, 0], np.int32)
        tokens = np.full((2, 7), -1, np.int32)
        slots = np.full((2, 7), -1, np.int32)
        tokens[0, 0], slots[0, 0] = 70, 3
        ids = reference_transfer_ids(case, count, tokens, slots, np.array([1, 0]), table)
        original = physical_tokens(case, case["slot_to_token"])
        self.assertEqual(ids[0, 3], 2 * 64 + 6)
        ids[0, 3] = original[0, 3]
        np.testing.assert_array_equal(ids, original)

    def test_plan_and_transfer_oracles_agree(self):
        for p in (75, 90, 100):
            case = make_full_cache_case(2, 33, 65, 512, p)
            golden = plan.reference_simt(case)
            np.testing.assert_array_equal(reference_transfer_ids(case, golden["miss_count"],
                golden["miss_tokens"], golden["miss_slots"]), physical_tokens(case, golden["slot_to_token"]))

    def test_pattern_has_distinct_physical_identity_and_kv(self):
        tokens = np.array([1, 257, 65537], np.int64)
        for width in (1, 31, 32, 33, 1024, 16385):
            data = payload_pattern(tokens, width, 1)
            self.assertEqual(data.shape, (3, width))
            if width >= 8:
                self.assertEqual(len(np.unique(data[:, :8], axis=0)), 3)
            self.assertFalse(np.array_equal(data, payload_pattern(tokens, width, 2)))


class CsvChecks(unittest.TestCase):
    def parse(self, rows, count=2, fields=None):
        fields = fields or ["Name", "Task Type", "Start Time(us)", "Duration(us)"]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "kernel_details.csv"
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(fields)
                writer.writerows(rows)
            return read_kernel_rows([path], "chain", count)

    def valid(self):
        return [(PLAN_KERNEL, "AI_VECTOR_CORE", 100, 10),
                ("TensorMove", "MEMCPY", 115, 200),
                (TRANSFER_KERNEL, "AI_VECTOR_CORE", 120, 30),
                (PLAN_KERNEL, "AI_VECTOR_CORE", 200, 20),
                (TRANSFER_KERNEL, "AI_VECTOR_CORE", 230, 40)]

    def test_per_pair_statistics_and_copy_exclusion(self):
        records = self.parse(list(reversed(self.valid())))
        stats = summarize(records, "chain", 1024)
        self.assertEqual(stats["plan"]["p50_us"], 15)
        self.assertEqual(stats["transfer"]["p50_us"], 35)
        self.assertEqual(stats["pair_busy"]["samples_us"], [40, 60])
        self.assertEqual(stats["pair_span"]["samples_us"], [50, 70])

    def test_reject_missing_duplicate_wrong_units_and_reordering(self):
        for rows in (self.valid()[:-1], self.valid() + [self.valid()[0]],
                     [(TRANSFER_KERNEL, "AI_VECTOR_CORE", 95, 3)] + self.valid()[1:]):
            with self.assertRaises(RuntimeError):
                self.parse(rows)
        with self.assertRaises(RuntimeError):
            self.parse(self.valid(), fields=["Name", "Task Type", "Start Time(ns)", "Duration(ns)"])

    def test_median_of_actual_pairs_is_not_sum_of_medians(self):
        rows = [(PLAN_KERNEL, "AIV", 0, 1), (TRANSFER_KERNEL, "AIV", 2, 100),
                (PLAN_KERNEL, "AIV", 200, 100), (TRANSFER_KERNEL, "AIV", 301, 1),
                (PLAN_KERNEL, "AIV", 400, 100), (TRANSFER_KERNEL, "AIV", 501, 100)]
        stats = summarize(self.parse(rows, count=3), "chain", 1024)
        self.assertEqual(stats["plan"]["p50_us"] + stats["transfer"]["p50_us"], 200)
        self.assertEqual(stats["pair_busy"]["p50_us"], 101)

    def test_reject_overlapping_targets_and_nonfinite(self):
        rows = self.valid()
        rows[2] = (TRANSFER_KERNEL, "AI_VECTOR_CORE", 105, 30)
        with self.assertRaises(RuntimeError):
            self.parse(rows)
        rows = self.valid()
        rows[0] = (PLAN_KERNEL, "AI_VECTOR_CORE", 100, "nan")
        with self.assertRaises(RuntimeError):
            self.parse(rows)


if __name__ == "__main__":
    unittest.main()
