"""三路专项的用例、构建选择及报告回归；仅CPU检查，非NPU验证。"""
from pathlib import Path
import json
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import profile_ab8k_context as target
from checks.generate_joint_variants import generate
from joint_variants import VARIANTS
from fixed_plan_check import exact_workload_key
from joint_plan_cases import require_sorted_topk


class Checks(unittest.TestCase):
    def test_exact_three_wrappers(self):
        self.assertEqual([VARIANTS[n] for n in target.NAMES], [213, 245, 249])
        with tempfile.TemporaryDirectory() as tmp:
            generate(tmp, ab8k_compare_only=True)
            names = {p.stem.removeprefix("variant_") for p in Path(tmp).glob("variant_*.cpp")}
            self.assertEqual(names, {n.removeprefix("joint_") for n in target.NAMES})
            text = (ROOT / "joint_plan/CMakeLists.txt").read_text()
            pattern = re.search(r'JOINT_AB8K_COMPARE_ONLY AND NOT _name MATCHES "([^"]+)"', text).group(1)
            self.assertTrue(all(re.fullmatch(pattern, n) for n in names))
            dispatch = (Path(tmp) / "variant_dispatch.h").read_text()
            entries = re.search(r'launches\[\] = \{([^}]+)\}', dispatch).group(1).split(", ")
            self.assertEqual([i for i, e in enumerate(entries) if e != "nullptr"], [213, 245, 249])

    def test_contexts_same_shapes_and_visibility(self):
        cases = target.workloads(target.parser().parse_args([]))
        self.assertEqual(len(cases), 4)
        hashes = set()
        for case in cases:
            require_sorted_topk(case)
            self.assertEqual((case["requests"], case["rows"], case["mtp"], case["capacity"], case["topk"]),
                             (32, 128, 3, 8192, 2048))
            info = target.workload_info(case)
            self.assertEqual(info["committed_context"], case["max_token"] - 16)
            self.assertEqual((info["visible_seq_len_min"], info["visible_seq_len_max"]),
                             (case["max_token"] - 15, case["max_token"] - 12))
            self.assertTrue(all(info["union_fits"]))
            self.assertTrue(all(f >= .4 for group in info["row0_overlap_fraction"] for f in group[1:]))
            hashes.add(exact_workload_key(case))
        self.assertEqual(len(hashes), 4)

    def mock_report(self):
        runs = []
        for length in target.CONTEXTS:
            w = {"max_token": length, "scenario": "full_cache_75pct", "committed_context": length-16,
                 "visible_seq_len_min": length-15, "visible_seq_len_max": length-12}
            values = (100., 80., 60.) if length == 262144 else (50., 40., 30.)
            stats = {n: {"p50_us": value, "p95_us": value+1, "count": 20, "block_nums": [56],
                         "correctness": {"joint_groups": 32, "overflow_groups": 0, "total_transfer_descriptors": 10}}
                     for n, value in zip(target.NAMES, values)}
            runs.append({"workload": w, "stats": stats})
        return {"runs": runs, "wall_seconds": {}}

    def test_report_ratio_direction(self):
        text = target.report_text(self.mock_report())
        self.assertIn("80.000/81.000 | 20 | 0.800", text)
        self.assertIn("| full_cache_75pct | rankpipe AC | 0.500 |", text)
        self.assertIn("262129–262132", text)
        self.assertIn("102385–102388", text)
        with self.assertRaises(RuntimeError):
            target.require_joint({"joint_groups": 0, "overflow_groups": 32}, 32)

    def test_describe_no_npu_and_bad_arguments(self):
        with patch.object(target.common, "load_runtime", side_effect=AssertionError("must not load NPU")):
            with patch("builtins.print") as out:
                target.main(["--describe-only"])
            result = json.loads(out.call_args.args[0])
            self.assertEqual(len(result["workloads"]), 4)
            self.assertEqual(result["implementations"], list(target.NAMES))
            for flags in (["--repeat", "0"], ["--max-tokens", "102400", "102400"]):
                with self.assertRaises(ValueError):
                    target.main(["--describe-only", *flags])

    def test_failure_keeps_original_and_no_success_report(self):
        from contextlib import nullcontext
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "results"
            def preflight(args, path):
                path.mkdir()
                return {}
            with patch.object(target, "preflight_args", side_effect=preflight), \
                 patch.object(target, "diagnostic_log", return_value=nullcontext()), \
                 patch.object(target.common, "load_runtime", return_value=(None, None, {})), \
                 patch.object(target, "gate", return_value={}), \
                 patch.object(target, "capture", side_effect=RuntimeError("original capture failure")), \
                 patch("builtins.print"):
                with self.assertRaisesRegex(RuntimeError, "original capture failure"):
                    target.main(["--skip-build", "--output", str(output), "--max-tokens", "102400",
                                 "--scenarios", "cold_reset"])
            status = json.loads((output / "RUN_STATUS.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertIn("original capture failure", status["diagnostics"])
            self.assertFalse((output / "COMPLETE.json").exists())
            self.assertFalse((output / "report.md").exists())

    def test_source_cpu_execution(self):
        # 执行真实源代码的CPU桩，3实现×2长度×冷/热；检查golden及保护区。
        # 模拟线程、地址空间和同步不同，不能据此判断CANN/NPU正确性或性能。
        import ctypes
        from checks.check_joint_variant_kernels import build, execute
        for name in target.NAMES:
            mask = VARIANTS[name]
            with tempfile.TemporaryDirectory() as tmp:
                lib = ctypes.CDLL(str(build(Path(tmp), mask)))
                for length in target.CONTEXTS:
                    for scenario in target.SCENARIOS:
                        case = target.make_workload(length, scenario, requests=1)
                        execute(lib, case, True, False, turbo=True, sorted_union=True,
                                single_sorted=True, mtp8=True, capacity8k=mask == 213,
                                layout16=0 if mask == 213 else 1,
                                layout_hot_bits=0 if mask == 213 else 15)
                        print("CPU源模型通过:", name, length, scenario, flush=True)


if __name__ == "__main__":
    unittest.main()
