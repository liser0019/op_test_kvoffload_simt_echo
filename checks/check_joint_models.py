#!/usr/bin/env python3
"""仅服务器CPU：输入控制、独立联合语义门禁的拒错能力、CSV与MTP统计口径。"""
from pathlib import Path
import copy
import csv
import sys
import tempfile
import unittest
import numpy as np
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from joint_plan_cases import make_case, initial_arrays, validate, layout, describe
from profile_joint_mtp import read_kernels, mtp_comparison
import joint_plan_common as common
from checks.joint_build_info import identity
from compare import reference_simt


def hand_case():
    return {"rows": 2, "requests": 1, "mtp": 1, "q_len": 2, "topk": 2, "capacity": 4,
            "max_token": 128, "block_size": 1, "host_num_blocks": 128,
            "scenario": "full_cache_75pct", "req_ids": np.array([42], np.int64),
            "topk_indices": np.array([[1, 2], [2, 3]], np.int32),
            "token_to_req": np.zeros(2, np.int32), "block_table": np.arange(128, dtype=np.int32).reshape(1, -1),
            "visible_seq_lens": np.full(2, 100, np.int32), "stable_prefix_lens": np.full(2, 100, np.int32),
            "last_req_ids": np.full(2, 42, np.int64),
            "slot_to_token": np.array([[1, 9, 7, 6], [1, 9, 7, 6]], np.int32),
            "lru_slots": np.array([[3, 1, 0, 2], [3, 1, 0, 2]], np.int32)}


def hand_result(case, reverse=False):
    before = initial_arrays(case, True)
    actual = {k: v.copy() for k, v in before.items()}
    actual["slot_to_token"][0] = [1, 2, 7, 3] if not reverse else [1, 3, 7, 2]
    actual["lru_slots"][0] = [2, 3, 1, 0]
    actual["miss_count"][:] = [2, 0]
    actual["miss_tokens"][0] = [3, 2] if not reverse else [2, 3]
    actual["miss_slots"][0] = [3, 1]
    actual["current_slots"][:] = [[0, 1], [1, 3]] if not reverse else [[0, 3], [3, 1]]
    return before, actual


class CaseChecks(unittest.TestCase):
    def test_source_identity(self):
        self.assertEqual(identity()["upstream_commit"], "bba63d05762a467e496c0f8a8ae687dc632a706f")

    def test_mtp_shapes_and_same_row0_cache_lru_pages(self):
        cases = [make_case(2, 7, 16, 512, m, block_size=8) for m in (0, 1, 3)]
        for case in cases:
            self.assertEqual(case["rows"], 2 * (case["mtp"] + 1))
            for key in ("topk_indices", "slot_to_token", "lru_slots"):
                np.testing.assert_array_equal(case[key][::case["q_len"]], cases[0][key])
            np.testing.assert_array_equal(case["block_table"], cases[0]["block_table"])
        self.assertTrue(all(describe(c)["union_fits"] == [True, True] for c in cases))

    def test_disjoint_case_really_overflows_at_mtp3(self):
        case = make_case(2, 7, 16, 512, 3, overlap="disjoint", block_size=8)
        self.assertEqual(describe(case)["union_fits"], [False, False])
        self.assertTrue(np.all(initial_arrays(case, True)["row_modes"] == 0))

    def test_layout_ub_gm_and_scan_bytes(self):
        self.assertEqual(layout(2048, 4096, 128, False)["dynamic_ub_bytes"], 74016)
        self.assertEqual(layout(2048, 4096, 128, True)["dynamic_ub_bytes"], 82208)
        gm = layout(2048, 8192, 4, True)
        self.assertEqual(gm["dynamic_ub_bytes"], 288)
        self.assertEqual(gm["workspace_elements"], 4 * 40960)


class OracleChecks(unittest.TestCase):
    def test_joint_allocation_accepts_both_legal_miss_orders(self):
        for reverse in (False, True):
            case = hand_case()
            before, actual = hand_result(case, reverse)
            self.assertEqual(validate(case, before, actual, True)["total_transfer_descriptors"], 2)

    def test_mutated_owner_miss_token_lru_slot_and_follower_are_rejected(self):
        case = hand_case()
        before, actual = hand_result(case)
        for key, index, value in (("resident_rows", 1, 1), ("row_modes", 1, 0),
                                  ("miss_tokens", (0, 0), 2), ("current_slots", (1, 1), 0),
                                  ("lru_slots", (0, 0), 0), ("slot_to_token", (1, 0), 77)):
            broken = copy.deepcopy(actual)
            broken[key][index] = value
            with self.assertRaises(AssertionError):
                validate(case, before, broken, True)

    def test_loaded_future_token_is_still_hidden_from_earlier_row(self):
        case = hand_case()
        case["topk_indices"][0] = [1, 3]
        case["visible_seq_lens"][:] = [3, 4]
        before, actual = hand_result(case)
        actual["current_slots"][0] = [0, -1]
        validate(case, before, actual, True)
        actual["current_slots"][0, 1] = 3
        with self.assertRaises(AssertionError):
            validate(case, before, actual, True)

    def test_overflow_requires_whole_group_private_fallback(self):
        case = hand_case()
        case["capacity"] = 2
        case["slot_to_token"] = case["slot_to_token"][:, :2].copy()
        case["lru_slots"] = np.array([[1, 0], [1, 0]], np.int32)
        before = initial_arrays(case, True)
        actual = {k: v.copy() for k, v in before.items()}
        golden = reference_simt(case)
        actual.update(golden)
        self.assertEqual(validate(case, before, actual, True)["overflow_groups"], 1)


class CsvChecks(unittest.TestCase):
    def test_exact_target_units_count_and_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "kernel_details.csv"
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["Name", "Duration(us)", "Start Time(us)", "Task Type"])
                writer.writerow(["TensorMove", 9000, 0, "AI_CPU"])
                writer.writerow([common.KERNELS["joint"] + "_extra", 9000, 10, "AIV"])
                writer.writerow([common.KERNELS["joint"], 10, 100, "AIV"])
                writer.writerow([common.KERNELS["private"], 20, 200, "AIV"])
            records = read_kernels([path], ["private", "joint"], 1, ["joint", "private"])
            self.assertEqual([x["duration_us"] for x in records], [10, 20])
            with self.assertRaises(RuntimeError):
                read_kernels([path], ["private", "joint"], 2, ["joint", "private"])
            with self.assertRaises(RuntimeError):
                read_kernels([path], ["private", "joint"], 1, ["private", "joint"])
            path.write_text("Name,Duration(ms),Start Time(us)\n")
            with self.assertRaises(RuntimeError):
                read_kernels([path], ["joint"], 1, ["joint"])

    def test_mtp_total_ratio_is_separate_from_per_row_ratio(self):
        runs = []
        for mtp, rows, duration in ((0, 32, 10), (1, 64, 12), (3, 128, 20)):
            runs.append({"comparison": "same_requests", "workload": {"scenario": "full_cache_75pct",
                "overlap": "shared", "mtp": mtp, "requests": 32, "query_rows": rows},
                "stats": {"joint": {"p50_us": duration, "us_per_query_row_p50": duration / rows}}})
        pairs = mtp_comparison(runs)
        self.assertEqual(pairs[1]["on_over_off_p50"], 2)
        self.assertEqual(pairs[1]["on_over_off_per_query_row"], 0.5)


if __name__ == "__main__":
    unittest.main()
