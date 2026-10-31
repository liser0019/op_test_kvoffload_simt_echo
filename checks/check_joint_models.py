#!/usr/bin/env python3
"""仅服务器CPU：输入控制、独立联合语义门禁的拒错能力、CSV与MTP统计口径。"""
from pathlib import Path
import copy
import csv
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from joint_plan_cases import make_case, initial_arrays, validate, validate_abcdf, layout, describe
from profile_joint_mtp import read_kernels, mtp_comparison, execution_order, duration_ratios, report_text
import joint_plan_common as common
from checks.joint_build_info import identity, expected_fixed_region, expected_fixed_helper, compute_region
from checks.check_joint_plan import validate_gate_report
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

    def test_only_verified_workaround_changes_computation(self):
        path = ROOT / "csrc/joint_plan"
        original = (path / "upstream/sparse_kv_plan_apt.cpp").read_text()
        self.assertEqual(compute_region((path / "kernel.cpp").read_text()), expected_fixed_region(original))
        helper = (path / "sparse_kv_joint_plan.h").read_text()
        upstream = (path / "upstream/sparse_kv_joint_plan.h").read_text()
        self.assertEqual(helper, expected_fixed_helper(upstream))
        self.assertEqual(helper.count("asc_syncthreads()"), upstream.count("asc_syncthreads()"))
        self.assertEqual(helper.count("for (int64_t row = first; row < end; ++row)"),
                         upstream.count("for (int64_t row = first; row < end; ++row)"))
        self.assertIn("if (thread == 1) {\n      modeOutput[row]", helper)

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
    def test_abcdf_original_protocol_has_no_joint_metadata(self):
        case = hand_case()
        before = {k: v for k, v in initial_arrays(case, False).items()
                  if k in (*common.ABCDF_STATE, *common.ABCDF_OUTPUT)}
        actual = reference_simt(case)
        self.assertNotIn("row_modes", actual)
        self.assertNotIn("resident_rows", actual)
        self.assertEqual(validate_abcdf(case, before, actual)["status"], "passed")
        actual["current_slots"][0, 0] = -777
        with self.assertRaises(AssertionError):
            validate_abcdf(case, before, actual)

    def test_abcdf_invokes_actual_original_op_with_exact_interface(self):
        calls = []
        fake = SimpleNamespace(ops=SimpleNamespace(sparse_kv_plan_op=SimpleNamespace(
            plan_abcdf=lambda *args: calls.append(args))))
        bundle = {"name": "abcdf", "torch": fake, "ordered": list(range(15)), "attrs": list(range(5))}
        common.invoke(bundle)
        self.assertEqual(calls, [tuple(range(15)) + tuple(range(5))])
        cmake = (ROOT / "joint_plan/CMakeLists.txt").read_text()
        self.assertIn('"${ABCDF_DIR}/kernel.cpp"', cmake)
        self.assertIn('"${ABCDF_DIR}/plugin.cpp"', cmake)
        self.assertIn("13_abcdf", cmake)

    def test_input_wrong_write_is_rejected_and_left_unmodified(self):
        class CpuTensor:
            def __init__(self, values): self.values = values.copy()
            def cpu(self): return self
            def numpy(self): return self.values
        case = hand_case()
        inputs = {key: CpuTensor(case[key]) for key in common.INPUTS}
        inputs["active_rows"] = CpuTensor(np.array([2], np.int32))
        inputs["visible_seq_lens"].values[0] = 1
        bundle = {"name": "abcdf", "buffers": {}, "host_initial": {}, "inputs": inputs}
        with self.assertRaises(AssertionError):
            common.check(case, bundle)
        self.assertEqual(inputs["visible_seq_lens"].values[0], 1)
        self.assertEqual(case["visible_seq_lens"][0], 100)

    def test_audit_stops_at_first_failure_and_disk_full_preserves_original_error(self):
        class CpuTensor:
            def cpu(self): return self
            def numpy(self): return np.array([1], np.int32)
        fake_torch = SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: None))
        bundle = {"name": "joint", "inputs": {"visible_seq_lens": CpuTensor()}, "buffers": {}}
        original = AssertionError("read-only visible_seq_lens changed")
        with tempfile.TemporaryDirectory() as tmp, patch.object(common, "restore") as restored, \
             patch.object(common, "poison"), patch.object(common, "snapshot", return_value={}), \
             patch.object(common, "invoke", side_effect=original) as invoked, \
             patch.object(common.np, "savez", side_effect=OSError("disk full")), patch("sys.stderr"):
            with self.assertRaises(AssertionError) as result:
                common.audit(fake_torch, hand_case(), bundle, repeats=100, failure_dir=tmp)
            self.assertIs(result.exception, original)
            self.assertEqual(invoked.call_count, 1)
            self.assertEqual(restored.call_count, 1)

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
    def test_three_way_order_rotates_all_positions_and_permutations(self):
        names = ("abcdf", "private", "joint")
        self.assertEqual(len({tuple(execution_order(names, i)) for i in range(6)}), 6)
        orders = [execution_order(names, i) for i in range(1, 51)]
        for name in names:
            counts = [sum(row[position] == name for row in orders) for position in range(3)]
            self.assertLessEqual(max(counts) - min(counts), 1)

    def test_three_exact_kernels_and_ratio_direction(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "kernel_details.csv"
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["Name", "Duration(us)", "Start Time(us)", "Task Type"])
                for i, name in enumerate(common.IMPLEMENTATIONS):
                    writer.writerow([common.KERNELS[name], 10, 100 * i, "AIV"])
            actual = read_kernels([path], common.IMPLEMENTATIONS, 1, list(common.IMPLEMENTATIONS))
            self.assertEqual([r["implementation"] for r in actual], list(common.IMPLEMENTATIONS))
        stats = {name: {"p50_us": value} for name, value in (("abcdf", 10), ("private", 14), ("joint", 7))}
        ratios = duration_ratios(stats)
        self.assertAlmostEqual(ratios["private_over_abcdf_p50"], 1.4)
        self.assertAlmostEqual(ratios["joint_over_abcdf_p50"], 0.7)
        self.assertAlmostEqual(ratios["private_over_joint_p50"], 2)

    def test_report_contains_all_three_ratios_csv_paths_and_device_identity(self):
        stats = {name: {"p50_us": value, "p95_us": value + 1, "count": 50,
                 "raw_csv": ["/workspace/raw/kernel_details.csv"],
                 "correctness": {"total_transfer_descriptors": 1, "joint_groups": 0, "overflow_groups": 0}}
                 for name, value in (("abcdf", 10), ("private", 14), ("joint", 7))}
        run = {"comparison": "same_requests", "workload": {"scenario": "full_cache_75pct",
               "overlap": "shared", "mtp": 1, "requests": 32, "query_rows": 64},
               "stats": stats} | duration_ratios(stats)
        text = report_text({"runs": [run], "mtp_comparison": [], "environment": {
            "device": 2, "device_name": "Ascend950", "library_sha256": "library", "identity": {"flags": "-O3"}}})
        for label in ("private/abcdf", "joint/abcdf", "private/joint", "1.400", "0.700", "2.000",
                      "/workspace/raw/kernel_details.csv", "Ascend950", "-O3"):
            self.assertIn(label, text)

    def test_reused_gate_requires_same_identity_device_and100_checks(self):
        env = {"device": 2, "library_sha256": "library", "identity": {"source": "source"}}
        report = {"status": "passed", "environment": env, "gate": {"status": "passed",
                  "implementations": list(common.IMPLEMENTATIONS),
                  "input_write_regression": {"status": "passed", "checked_iterations": 100, "input_restore": False,
                      "per_implementation": {name: {"status": "passed"} for name in common.IMPLEMENTATIONS if name != "abcdf"}}}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gate.json"
            import json
            path.write_text(json.dumps(report))
            validate_gate_report(path, env)
            for key, value in (("device", 0), ("library_sha256", "old"), ("identity", {})):
                with self.assertRaises(RuntimeError):
                    validate_gate_report(path, env | {key: value})
            report["gate"]["input_write_regression"]["input_restore"] = True
            path.write_text(json.dumps(report))
            with self.assertRaises(RuntimeError):
                validate_gate_report(path, env)
            report["gate"]["input_write_regression"]["input_restore"] = False
            report["gate"]["input_write_regression"]["checked_iterations"] = 99
            path.write_text(json.dumps(report))
            with self.assertRaises(RuntimeError):
                validate_gate_report(path, env)

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


class VariantChecks(unittest.TestCase):
    def test_independent_indexer_selections_are_unsorted_and_distinct(self):
        cases = [make_case(2, 65, 160, 2048, m, overlap="independent", block_size=8) for m in (0, 1, 3)]
        for case in cases:
            np.testing.assert_array_equal(case["topk_indices"][::case["q_len"]], cases[0]["topk_indices"])
        query = cases[2]["topk_indices"]
        self.assertNotEqual(set(query[0]), set(query[1]))
        self.assertFalse(np.all(query[1][:-1] <= query[1][1:]))
        self.assertTrue(all(describe(cases[2])["union_fits"]))
        # 不因独立选词而放松逐行因果可见性。
        for row, values in enumerate(query):
            self.assertTrue(np.all(values < cases[2]["visible_seq_lens"][row]))

    def test_all_fifteen_masks_unique_symbols_and_dispatch(self):
        from joint_variants import MASKS
        from checks.generate_joint_variants import generate
        self.assertEqual(set(MASKS.values()), set(range(1, 16)))
        self.assertEqual(len(set(common.KERNELS.values())), len(common.IMPLEMENTATIONS))
        with tempfile.TemporaryDirectory() as tmp:
            generate(tmp)
            dispatcher = (Path(tmp) / "variant_dispatch.h").read_text()
            for letters, mask in MASKS.items():
                source = (Path(tmp) / ("variant_" + letters + ".cpp")).read_text()
                self.assertIn(f"#define JOINT_OPT_MASK {mask}", source)
                self.assertIn(f"#define JointDirectBody JointDirectBody_{letters}", source)
                self.assertIn(f"&launch_joint_plan_{letters}", dispatcher)

    def test_many_way_rotation_and_prefix_kernel_filter(self):
        names = common.DEFAULT_IMPLEMENTATIONS
        count = len(names)
        orders = [execution_order(names, step) for step in range(1, 51)]
        for name in names:
            counts = [sum(order[pos] == name for order in orders) for pos in range(count)]
            self.assertLessEqual(max(counts) - min(counts), 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "kernel_details.csv"
            path.write_text("Name,Duration(us),Start Time(us)\n" +
                "\n".join(f"{common.KERNELS[name]},10,{100*i}" for i, name in enumerate(names)))
            self.assertEqual(len(read_kernels([path], names, 1, list(names))), len(names))
            # joint_kernel 必须不匹配 joint_kernel_a 或 joint_kernel_abcd。
            result = read_kernels([path], ["joint"], 1, ["joint"])
            self.assertEqual(len(result), 1)

    def test_variant_ratio_direction(self):
        stats = {name: {"p50_us": value} for name, value in
                 (("abcdf", 50), ("private", 75), ("joint", 150), ("private_a", 60), ("joint_abcd", 100))}
        ratios = duration_ratios(stats)
        self.assertAlmostEqual(ratios["private_a_over_private_p50"], .8)
        self.assertAlmostEqual(ratios["joint_abcd_over_joint_p50"], 2/3)
        self.assertAlmostEqual(ratios["joint_abcd_over_abcdf_p50"], 2)


if __name__ == "__main__":
    unittest.main()
