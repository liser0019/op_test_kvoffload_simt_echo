#!/usr/bin/env python3
"""仅服务器CPU：任务分配覆盖、UB布局、版本身份、CSV隔离与比较口径。"""
from pathlib import Path
import copy
import csv
import random
import sys
import tempfile
import unittest
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import transfer_variants as catalog
import plan_transfer_common as common
from profile_plan_transfer import read_kernel_rows
from profile_transfer_variants import compare_runs, order_configs
from checks.transfer_build_info import compute_class


def assignment(counts, cores, mask, topk):
    counts = [min(topk, max(0, count)) for count in counts]
    total = sum(counts)
    result = [[] for _ in range(cores)]
    for core in range(cores):
        per, extra = divmod(total, cores)
        lo = per * core + min(core, extra)
        hi = lo + per + (core < extra)
        prefix = 0
        for row, count in enumerate(counts):
            if mask & 8:
                indices = range(max(0, lo - prefix), min(count, hi - prefix))
            else:
                indices = range((core - prefix) % cores, count, cores)
            result[core].extend((row, i) for i in indices)
            prefix += count
    return result


class LayoutChecks(unittest.TestCase):
    def test_catalog_and_compiled_entries_are_one_to_one(self):
        self.assertEqual(len(catalog.VARIANTS), 17)
        self.assertEqual({x["mask"] for x in catalog.VARIANTS.values()}, set(range(-1, 16)))
        self.assertEqual(len({x["kernel_name"] for x in catalog.VARIANTS.values()}), 17)
        entries = (ROOT / "csrc/transfer_variants/entries.inc").read_text()
        for name, entry in catalog.VARIANTS.items():
            if name != "baseline":
                self.assertIn(f"TRANSFER_VARIANT({entry['mask']}, {name})", entries)
                self.assertEqual(name if name != "control" else "",
                                 "".join(c.lower() for bit, c in enumerate("ABCD") if entry["mask"] & (1 << bit)))

    def test_baseline_and_fallback_computation_are_frozen(self):
        preserved = (ROOT / "csrc/transfer/upstream/sparse_kv_transfer_apt.cpp").read_text()
        for file in ("csrc/transfer/kernel.cpp", "csrc/transfer_variants/original_class.h"):
            self.assertEqual(compute_class(preserved), compute_class((ROOT / file).read_text()))

    def test_packed_segments_do_not_overlap_or_overrun(self):
        for name, entry in catalog.VARIANTS.items():
            if entry["mask"] < 0:
                continue
            for k, v in ((1, 31), (32, 33), (64, 96), (1024, 128), (16384, 16385)):
                info = catalog.layout(name, k, v, 64)
                if info["fallback"]:
                    self.assertGreater(info["bytes_per_queue_record_limit"], 16384)
                    continue
                b = info["effective_batch"]
                pk, pv = (k + 31) & ~31, (v + 31) & ~31
                for actual in range(1, b + 1):
                    groups = ([[(i * pk, k) for i in range(actual)] +
                               [(b * pk + i * pv, v) for i in range(actual)]]
                              if entry["mask"] & 1 else
                              [[(i * pk, k) for i in range(actual)], [(i * pv, v) for i in range(actual)]])
                    for group in groups:
                        previous_end = 0
                        for offset, length in sorted(group):
                            self.assertEqual(offset % 32, 0)
                            self.assertGreaterEqual(offset, previous_end)
                            self.assertLessEqual(offset + length, 16384)
                            previous_end = offset + length

    def test_specialization_is_enabled_only_for_exact_production_layout(self):
        self.assertTrue(catalog.layout("c", 1024, 128, 64)["production_specialization"])
        for name in ("control", "a", "b", "d"):
            self.assertFalse(catalog.layout(name, 1024, 128, 64)["production_specialization"])
        for config in ((1024, 128, 63), (1024, 127, 64), (16384, 16385, 64)):
            self.assertFalse(catalog.layout("abcd", *config)["production_specialization"])
        for token in (0, 1, 63, 64, 65, 262143, 2147483647):
            self.assertEqual((token >> 6, token & 63), divmod(token, 64))


class PartitionChecks(unittest.TestCase):
    def test_global_task_coverage_and_balance(self):
        rng = random.Random(17)
        cases = [[512] * 32, [205] * 32, [0] * 32, [-9, 0, 1, 7, 8, 9, 99]]
        cases += [[rng.randrange(-3, 90) for _ in range(rng.randrange(1, 12))] for _ in range(20)]
        for counts in cases:
            expected = {(r, i) for r, count in enumerate(counts) for i in range(min(512, max(0, count)))}
            for cores in (1, 2, 3, 16, 64):
                for mask in range(16):
                    tasks = assignment(counts, cores, mask, 512)
                    flat = [task for worker in tasks for task in worker]
                    self.assertEqual(len(flat), len(set(flat)))
                    self.assertEqual(set(flat), expected)
                    sizes = [len(worker) for worker in tasks]
                    self.assertLessEqual(max(sizes) - min(sizes), 1)

    def test_config_order_rotation_keeps_every_configuration(self):
        configurations = [(name, 64) for name in catalog.VARIANTS]
        for index in range(3):
            for reverse in (False, True):
                self.assertCountEqual(order_configs(configurations, index, reverse), configurations)


class ComparisonChecks(unittest.TestCase):
    def test_ratio_requires_equal_workload_input_addresses_launch_and_cores(self):
        base = {"transfer_variant": {"name": "baseline", "actual_blocks": 64, "fallback": False},
                "mode": "transfer", "launch_mode": "prepared", "case_sha256": "input",
                "buffer_addresses_sha256": "addresses", "workload": {"scenario": "hit75"},
                "stats": {"transfer": {"p50_us": 10}}}
        candidate = copy.deepcopy(base)
        candidate["transfer_variant"]["name"] = "ab"
        candidate["stats"]["transfer"]["p50_us"] = 5
        found = compare_runs([base, candidate])
        self.assertEqual(found[-1]["baseline_over_variant_transfer_p50"], 2)
        for key in ("mode", "launch_mode", "case_sha256", "buffer_addresses_sha256", "workload"):
            wrong = copy.deepcopy(candidate)
            wrong[key] = "different"
            self.assertFalse(any(x["variant"] == "ab" for x in compare_runs([base, wrong])))
        wrong = copy.deepcopy(candidate)
        wrong["transfer_variant"]["actual_blocks"] = 32
        self.assertFalse(any(x["variant"] == "ab" for x in compare_runs([base, wrong])))

    def test_csv_selects_exact_variant_not_prefix_or_original(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "kernel_details.csv"
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["Name", "Task Type", "Duration(us)"])
                for name, duration in (("baseline", 100), ("a", 10), ("ab", 1)):
                    writer.writerow([catalog.VARIANTS[name]["kernel_name"], "AIV", duration])
            records = read_kernel_rows([path], "transfer", 1, catalog.VARIANTS["a"]["kernel_name"])
            self.assertEqual([record["duration_us"] for record in records], [10])


if __name__ == "__main__":
    unittest.main()
