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
from joint_variants import MASKS, mtp3_config

IMPLEMENTATIONS = (
    ("joint_rankpipe_ac_512", 249),
    ("joint_mtp3_full_512", 252),
    ("joint_mtp3_map_512", 253),
    ("joint_mtp3_load_512", 254),
    ("joint_mtp3_full_load_512", 255),
    ("joint_mtp3_bounded_512", 256),
    ("joint_mtp3_full_bounded_512", 257),
)


def target_case(kind: str):
    max_token = 102400 if kind == "bounded100k" else 153600 if kind == "bounded150k" else 262144
    case = make_case(1, 2048, 8192, max_token, 3, "full_cache_75pct", "disjoint",
                     history_pool_tokens=8192)
    token_to_req = np.zeros(4, np.int32)
    visible = np.full(4, max_token - 16, np.int32)
    stable = np.full(4, max_token - 16, np.int32)
    query = np.empty((4, 2048), np.int32)
    for row in range(4):
        query[row] = row * 2048 + np.random.default_rng(9100 + row).permutation(2048)
    if kind in ("overlap40", "repeated_resident"):
        # Keep 8192 slots as a common pool; adjacent rows share about 40%.
        common = 819
        for row in range(1, 4):
            query[row, :common] = query[0, :common]
            query[row, common:] = 2048 + (row - 1) * (2048 - common) + np.arange(2048 - common)
    elif kind == "short":
        visible = np.array([1, 64, 511, 2048], np.int32)
        stable = visible.copy()
    elif kind == "empty":
        visible = np.array([0, 64, 511, 2048], np.int32)
        stable = visible.copy()
    case.update(token_to_req=token_to_req, visible_seq_lens=visible,
                stable_prefix_lens=stable, topk_indices=np.sort(query, axis=1).astype(np.int32),
                sorted_topk=True)
    return case


def before_for(case, kind):
    before = initial_arrays(case, True, 8192)
    if case["scenario"] == "cold_reset":
        return before
    before["slot_to_token"][:] = -1
    if kind == "all_hits":
        tokens = np.arange(8192, dtype=np.int32)
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


def run(report_path: Path | None):
    assert MASKS["mtp3_full_512"] == 252
    assert MASKS["mtp3_map_512"] == 253
    assert MASKS["mtp3_load_512"] == 254
    assert MASKS["mtp3_full_load_512"] == 255
    assert MASKS["mtp3_bounded_512"] == 256
    assert MASKS["mtp3_full_bounded_512"] == 257
    normal = layout(2048, 8192, 4, True, 262144, layout16=1, layout_hot_bits=15, mtp8=True)
    full = layout(2048, 8192, 4, True, 262144, layout16=1, layout_hot_bits=15,
                  mtp8=True, mtp3_full=True)
    assert normal["dynamic_ub_bytes"] == 148256, normal
    assert full["dynamic_ub_bytes"] == 181024, full
    assert full["scratch_elements"] - normal["scratch_elements"] == 8192
    assert mtp3_config(252) == {"full": True, "map": False, "load": False, "bounded_bitmap": False}
    assert mtp3_config(253) == {"full": False, "map": True, "load": False, "bounded_bitmap": False}
    assert mtp3_config(254) == {"full": False, "map": False, "load": True, "bounded_bitmap": False}
    assert mtp3_config(255) == {"full": True, "map": False, "load": True, "bounded_bitmap": False}
    assert mtp3_config(256) == {"full": False, "map": False, "load": False, "bounded_bitmap": True}
    assert mtp3_config(257) == {"full": True, "map": False, "load": False, "bounded_bitmap": True}

    records = []
    with tempfile.TemporaryDirectory(prefix="mtp3_cpu_gate_") as tmp:
        for name, mask in IMPLEMENTATIONS:
            directory = Path(tmp) / name
            directory.mkdir()
            library = ctypes.CDLL(str(build(directory, mask, model_threads=64)))
            full_enabled = bool((mtp3_config(mask) or {}).get("full"))
            map_enabled = bool((mtp3_config(mask) or {}).get("map"))
            checks = 0
            for kind, scenario in (("disjoint", "full_cache_75pct"),
                                   ("overlap40", "full_cache_75pct"),
                                   ("short", "full_cache_75pct"),
                                   ("empty", "full_cache_75pct"),
                                   ("all_hits", "full_cache_75pct"),
                                   ("repeated_resident", "full_cache_75pct"),
                                   ("bounded100k", "full_cache_75pct"),
                                   ("bounded150k", "full_cache_75pct"),
                                   ("disjoint", "cold_reset")):
                case = target_case(kind)
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
                    layout16=1, layout_hot_bits=15, mtp3_full=full_enabled,
                    return_aux=map_enabled)
                if map_enabled:
                    assert np.all(result["__workspace"] == -1234567), "hot UB map must not spill to GM"
                    result.pop("__workspace")
                checks += 1
                print(name, kind, scenario, "passed", flush=True)
            records.append({"implementation": name, "mask": mask, "checks": checks})
    payload = {"status": "passed", "scope": "CPU64/UBSan semantic model, not NPU",
               "records": records, "normal_dynamic_ub_bytes": normal["dynamic_ub_bytes"],
               "full_dynamic_ub_bytes": full["dynamic_ub_bytes"]}
    text = __import__("json").dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(text)
    print(text, end="")
    print("===== MTP3_CPU_GATE_PASSED =====")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    run(args.report)
