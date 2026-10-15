#!/usr/bin/env python3
"""Hand-specified oracle fixtures and catalog/build checks, without CANN/NPU."""
import importlib.util
from pathlib import Path
import sys
import ast
import re
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("comparison", ROOT / "compare.py")
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class GoldenChecks(unittest.TestCase):
    def test_mtp_visible_rows_and_duplicate_cold_misses(self):
        case = dict(c.make_audit_cases())["mtp_cold_duplicates"]
        out = c.reference_simt(case)
        np.testing.assert_array_equal(out["miss_count"], [3, 4, 5, 0])
        np.testing.assert_array_equal(out["current_slots"][0], [0, 1, -1, -1, -1, -1, 2])
        np.testing.assert_array_equal(out["current_slots"][1], [0, 1, 2, -1, -1, -1, 3])
        np.testing.assert_array_equal(out["current_slots"][2], [0, 1, 2, 3, -1, -1, 4])

    def test_rewritten_suffix_and_duplicate_warm_queries(self):
        cases = dict(c.make_audit_cases())
        first = c.reference_simt(cases["mtp_rewritten_suffix_0"])
        second = c.reference_simt(cases["mtp_rewritten_suffix_1"])
        np.testing.assert_array_equal(first["miss_count"], [3, 4, 5, 0])
        np.testing.assert_array_equal(second["miss_count"], [3, 4, 4, 0])

    def test_insufficient_slots_and_tail(self):
        case = dict(c.make_audit_cases())["insufficient_slots"]
        out = c.reference_simt(case)
        np.testing.assert_array_equal(out["current_slots"][0], [0, 1, 2, -1, -1, -1, -1, -1, -1])
        np.testing.assert_array_equal(out["miss_slots"][0], [0, 1, 2, -1, -1, -1, -1, -1, -1])
        self.assertEqual(out["miss_count"][0], 3)

    def test_invalid_and_negative_visible_lengths(self):
        case = dict(c.make_audit_cases())["invalid_sources_non_power_two_pages"]
        out = c.reference_simt(case)
        np.testing.assert_array_equal(out["current_slots"][:2], -1)
        np.testing.assert_array_equal(out["miss_count"][:2], 0)
        self.assertGreaterEqual(out["current_slots"][2, 5], 0)
        np.testing.assert_array_equal(out["current_slots"][2, [0, 1, 2, 3, 4, 6, 7, 8]], -1)

    def test_padded_rows_are_untouched(self):
        case = dict(c.make_audit_cases())["padded_active_2"]
        out = c.reference_simt(case)
        for key in c.SIMT_STATE: np.testing.assert_array_equal(out[key][2], case[key][2])
        for key in c.SIMT_OUTPUT: np.testing.assert_array_equal(out[key][2], -777)

    def test_high_owner_orders(self):
        cases = dict(c.make_audit_cases())
        for capacity in (32768, 32769):
            out = c.reference_simt(cases[f"owner_pack_boundary_{capacity}"])
            np.testing.assert_array_equal(out["current_slots"][0], np.arange(capacity - 7, capacity))
            self.assertEqual(out["miss_count"][0], 0)

    def test_shuffled_lru_preserves_requested_hits(self):
        case = c.make_case(2, 17, 41, 256, "full_cache_75pct", 31, shuffle_lru=True)
        out = c.reference_simt(case)
        np.testing.assert_array_equal(out["miss_count"], 5)
        for row in range(2):
            np.testing.assert_array_equal(out["slot_to_token"][row, out["current_slots"][row]],
                                          case["topk_indices"][row])
            np.testing.assert_array_equal(np.sort(out["lru_slots"][row]), np.arange(41))

    def test_registration_schema_preserves_baseline_contract(self):
        base = (ROOT / "csrc/plan_variants/00_baseline/upstream/plugin_registered.h").read_text()
        candidate = (ROOT / "csrc/plan_variants/common/plan_plugin_impl.h").read_text()
        def schema(text):
            start = text.index("m.def(")
            chunk = text[start:text.index(");", start)]
            return "".join(ast.literal_eval(value) for value in re.findall(r'"(?:[^"\\]|\\.)*"', chunk))
        expected = schema(base)
        self.assertEqual(schema(candidate), expected[expected.index("("):])
        for backend in ("Meta", "PrivateUse1"):
            self.assertIn("TORCH_LIBRARY_IMPL(sparse_kv_plan_op, " + backend, candidate)

    def test_abcdfh_combines_only_abcd_f_h(self):
        variant = next(v for v in c.CATALOG["variants"] if v["name"] == "abcdfh")
        self.assertEqual(variant["points"], "ABCDFH")
        self.assertEqual(variant["threads"], 1024)
        self.assertEqual(variant["op_name"], "plan_abcdfh")
        self.assertEqual(variant["kernel_name"], "sparse_kv_plan_kernel_plan_abcdfh")
        config = (ROOT / variant["path"] / "variant_config.h").read_text()
        self.assertIn("PLAN_OPT_MASK 175U", config)

    def test_only_this_round_is_built(self):
        text = (ROOT / "CMakeLists.txt").read_text()
        start = text.index("set(PLAN_VARIANTS")
        folders = text[start:text.index(")", start)].split()[1:]
        self.assertEqual(folders, [Path(v["path"]).name for v in c.CATALOG["variants"]])
        self.assertEqual(len(c.IMPLEMENTATIONS), 18)
        self.assertFalse((ROOT / "csrc/echo").exists())
        for v in c.CATALOG["variants"]:
            if v["name"] == "baseline": continue
            mask = sum(c.CATALOG["feature_bits"][letter] for letter in v["points"])
            self.assertIn(f"PLAN_OPT_MASK {mask}U", (ROOT / v["path"] / "variant_config.h").read_text())


if __name__ == "__main__":
    unittest.main()
