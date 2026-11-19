#!/usr/bin/env python3
"""MTP3 fixed-shape CPU semantic gate: frozen rankpipe AC + four candidates."""
from __future__ import annotations
import argparse
import ctypes
from pathlib import Path
import sys
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from checks.check_joint_variant_kernels import build, execute
from joint_plan_cases import initial_arrays, layout, make_case, sort_topk, valid_sets
from joint_variants import MASKS, mtp3_config, MTP3_475_CONTEXTS, MTP3_475_MAX_TOKEN

FROZEN_IMPLEMENTATIONS = (
    ("joint_rankpipe_ac_512", 249),
    ("joint_mtp3_full_512", 252),
    ("joint_mtp3_map_512", 253),
    ("joint_mtp3_load_512", 254),
    ("joint_mtp3_full_load_512", 255),
)
MTP3_475_IMPLEMENTATIONS = (
    ("joint_mtp3_dir475_512", 256),
    ("joint_mtp3_bitmap475_512", 257),
)
IMPLEMENTATIONS = (*FROZEN_IMPLEMENTATIONS, *MTP3_475_IMPLEMENTATIONS)


def target_case(kind: str, max_token: int = 262144, high_window: bool = False):
    case = make_case(1, 2048, 8192, max_token, 3, "full_cache_75pct", "disjoint",
                     history_pool_tokens=8192)
    token_to_req = np.zeros(4, np.int32)
    visible = np.full(4, max_token, np.int32)
    stable = np.full(4, max_token, np.int32)
    query = np.empty((4, 2048), np.int32)
    high_pool = None
    if high_window:
        # Exercise the final 32-bit words of the exact bitmap/directory index.
        high_pool = max_token - 8192 + np.random.default_rng(9200).permutation(8192)
        for row in range(4):
            query[row] = high_pool[row * 2048:(row + 1) * 2048]
    else:
        for row in range(4):
            query[row] = row * 2048 + np.random.default_rng(9100 + row).permutation(2048)
    if kind in ("overlap40", "repeated_resident"):
        # Keep 8192 slots as a common pool; adjacent rows share about 40%.
        common = 819
        for row in range(1, 4):
            query[row, :common] = query[0, :common]
            if high_window:
                start = 2048 + (row - 1) * (2048 - common)
                query[row, common:] = high_pool[start:start + 2048 - common]
            else:
                query[row, common:] = 2048 + (row - 1) * (2048 - common) + np.arange(2048 - common)
    elif kind == "short":
        visible = np.array([1, 64, 511, 2048], np.int32)
        stable = visible.copy()
    elif kind == "empty":
        visible = np.array([0, 64, 511, 2048], np.int32)
        stable = visible.copy()
    case.update(token_to_req=token_to_req, visible_seq_lens=visible,
                stable_prefix_lens=stable, topk_indices=np.sort(query, axis=1).astype(np.int32),
                sorted_topk=True, _high_window=high_window)
    return case


def before_for(case, kind):
    before = initial_arrays(case, True, 8192)
    if case["scenario"] == "cold_reset":
        return before
    before["slot_to_token"][:] = -1
    if kind == "all_hits":
        start = case["max_token"] - 8192 if case.get("_high_window") else 0
        tokens = np.arange(start, start + 8192, dtype=np.int32)
        before["slot_to_token"][:, :] = tokens
    elif kind == "repeated_resident":
        tokens = np.array(sorted(set().union(*valid_sets(case))), dtype=np.int32)
        assert len(tokens) < case["capacity"]
        before["slot_to_token"][:, :len(tokens)] = tokens
        # A duplicate resident token must retain every old slot and the LRU-last owner.
        before["slot_to_token"][:, len(tokens)] = tokens[0]
    return before


def union_size(case):
    return len(set().union(*valid_sets(case)))


def run(report_path: Path | None, mtp3_475_only: bool = False):
    assert MASKS["mtp3_full_512"] == 252
    assert MASKS["mtp3_map_512"] == 253
    assert MASKS["mtp3_load_512"] == 254
    assert MASKS["mtp3_full_load_512"] == 255
    assert MASKS["mtp3_dir475_512"] == 256
    assert MASKS["mtp3_bitmap475_512"] == 257
    normal = layout(2048, 8192, 4, True, 262144, layout16=1, layout_hot_bits=15, mtp8=True)
    full = layout(2048, 8192, 4, True, 262144, layout16=1, layout_hot_bits=15,
                  mtp8=True, mtp3_full=True)
    fixed475 = layout(2048, 8192, 4, True, MTP3_475_MAX_TOKEN, layout16=1,
                      layout_hot_bits=15, mtp8=True, mtp3_full=True, mtp3_475=True)
    assert normal["dynamic_ub_bytes"] == 148256, normal
    assert full["dynamic_ub_bytes"] == 181024, full
    assert fixed475["scratch_elements"] == 46912, fixed475
    assert fixed475["dynamic_ub_bytes"] == 187936, fixed475
    assert fixed475["dynamic_ub_bytes"] - full["dynamic_ub_bytes"] == 6912
    for context in MTP3_475_CONTEXTS:
        info = layout(2048, 8192, 4, True, context, layout16=1, layout_hot_bits=15,
                      mtp8=True, mtp3_full=True, mtp3_475=True)
        assert info["dynamic_ub_bytes"] == 187936, (context, info)
    assert mtp3_config(252) == {"full": True, "map": False, "load": False}
    assert mtp3_config(253) == {"full": False, "map": True, "load": False}
    assert mtp3_config(254) == {"full": False, "map": False, "load": True}
    assert mtp3_config(255) == {"full": True, "map": False, "load": True}
    assert mtp3_config(256)["dir475"] and not mtp3_config(256)["bitmap475"]
    assert mtp3_config(257)["bitmap475"] and not mtp3_config(257)["dir475"]
    implementations = MTP3_475_IMPLEMENTATIONS if mtp3_475_only else IMPLEMENTATIONS
    contexts = MTP3_475_CONTEXTS if mtp3_475_only else (262144,)

    records = []
    with tempfile.TemporaryDirectory(prefix="mtp3_cpu_gate_") as tmp:
        for name, mask in implementations:
            directory = Path(tmp) / name
            directory.mkdir()
            library = ctypes.CDLL(str(build(directory, mask, model_threads=64)))
            cfg = mtp3_config(mask) or {}
            is475 = mask >= 256
            checks = 0
            for context in contexts:
                for kind, scenario in (("disjoint", "full_cache_75pct"),
                                       ("overlap40", "full_cache_75pct"),
                                       ("short", "full_cache_75pct"),
                                       ("empty", "full_cache_75pct"),
                                       ("all_hits", "full_cache_75pct"),
                                       ("repeated_resident", "full_cache_75pct"),
                                       ("disjoint", "cold_reset")):
                    case = target_case(kind, context, high_window=is475)
                    case["scenario"] = scenario
                    if scenario == "cold_reset":
                        case["last_req_ids"][:] = -1
                    assert case["rows"] == 4 and case["topk"] == 2048 and case["capacity"] == 8192
                    if kind in ("disjoint", "all_hits"):
                        assert union_size(case) == 8192, (kind, union_size(case))
                    before = before_for(case, kind)
                    result = execute(
                        library, case, True, False, before=before, turbo=True,
                        sorted_union=True, single_sorted=True, mtp8=True,
                        layout16=1, layout_hot_bits=15, mtp3_full=bool(cfg.get("full")),
                        mtp3_475=is475, return_aux=bool(cfg.get("map")))
                    if cfg.get("map"):
                        assert np.all(result["__workspace"] == -1234567), "hot UB map must not spill to GM"
                        result.pop("__workspace")
                    checks += 1
                    print(name, context, kind, scenario, "passed", flush=True)
            records.append({"implementation": name, "mask": mask, "checks": checks,
                            "contexts": list(contexts)})
    payload = {"status": "passed", "scope": "CPU64/UBSan semantic model, not NPU",
               "records": records, "normal_dynamic_ub_bytes": normal["dynamic_ub_bytes"],
               "full_dynamic_ub_bytes": full["dynamic_ub_bytes"],
               "mtp3_475_dynamic_ub_bytes": fixed475["dynamic_ub_bytes"],
               "mtp3_475_scratch_elements": fixed475["scratch_elements"]}
    text = __import__("json").dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(text)
    print(text, end="")
    print("===== MTP3_CPU_GATE_PASSED =====")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--mtp3-475-only", action="store_true")
    args = parser.parse_args()
    run(args.report, args.mtp3_475_only)
