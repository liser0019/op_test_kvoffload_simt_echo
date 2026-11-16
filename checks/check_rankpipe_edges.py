"""真实候选额外覆盖全命中/全miss、SIMD63/64/65尾部与重复resident。"""
import ctypes
from pathlib import Path
import tempfile
import numpy as np
from checks.check_joint_variant_kernels import build, execute
from joint_plan_cases import make_case, sort_topk, initial_arrays


def run():
    count = 0
    for mask in (244, 245, 246, 248, 251):
        with tempfile.TemporaryDirectory() as tmp:
            lib = ctypes.CDLL(str(build(Path(tmp), mask)))
            for cap, mtp in ((4096, 0), (4096, 1), (8192, 3), (16384, 7)):
                for unique in (0, 1, 63, 64, 65, 2048):
                    c = sort_topk(make_case(1, 2048, cap, 262144, mtp, 'full_cache_75pct',
                                           'shared', history_pool_tokens=cap))
                    c['topk_indices'][:] = -1
                    if unique:
                        c['topk_indices'][:, -unique:] = 100000 + np.arange(unique, dtype=np.int32)
                    c['stable_prefix_lens'][:] = c['max_token']
                    for all_hits in (False, True):
                        before = initial_arrays(c, True, cap)
                        if all_hits and unique:
                            before['slot_to_token'][0, :unique] = 100000 + np.arange(unique, dtype=np.int32)
                            # 同token多resident必须保护全部，owner仍为LRU-last。
                            before['slot_to_token'][0, -1] = 100000
                        actual = execute(lib, c, True, False, before=before, turbo=True, sorted_union=True,
                                         single_sorted=True, mtp8=True, layout16=1, layout_hot_bits=15)
                        if all_hits:
                            assert actual['miss_count'].sum() == 0
                        count += 1
            print(mask, 'tail/all-hit/all-miss passed', flush=True)
    print('edge calls', count, flush=True)

if __name__ == '__main__':
    run()
