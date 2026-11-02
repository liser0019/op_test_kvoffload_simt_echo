#!/usr/bin/env python3
"""CPU-only input contract: sorted common queries, guaranteed row0 overlap, causality."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from joint_plan_cases import make_case, sort_topk, describe, require_sorted_topk

def main():
    checked=0
    for topk,cap,block,max_token in ((1,8,8,128),(7,16,8,512),(65,128,8,2048),(2048,4096,64,262144)):
        for seed in (0,17,2026):
            cases=[sort_topk(make_case(2,topk,cap,max_token,m,overlap="overlap40",seed=seed,block_size=block)) for m in (0,1,3)]
            for c in cases:
                require_sorted_topk(c)
                d=describe(c)
                for request in range(2):
                    first=request*c["q_len"]
                    np.testing.assert_array_equal(c["topk_indices"][first], cases[0]["topk_indices"][request])
                    for row in range(first,first+c["q_len"]):
                        tokens=c["topk_indices"][row]
                        assert len(set(map(int,tokens)))==topk
                        assert np.all(tokens<c["visible_seq_lens"][row])
                        count=len(set(map(int,tokens)) & set(map(int,c["topk_indices"][first])))
                        if row>first: assert count>=(2*topk+4)//5
                assert all(v>=0.4 for group in d["row0_overlap_fraction"] for v in group[1:])
                checked+=1
    repairs=0
    for seed in range(100):
        old=sort_topk(make_case(1,7,16,512,3,overlap="independent",seed=seed,block_size=8))
        new=sort_topk(make_case(1,7,16,512,3,overlap="overlap40",seed=seed,block_size=8))
        if any(v<0.4 for group in describe(old)["row0_overlap_fraction"] for v in group[1:]): repairs+=1
        assert all(v>=0.4 for group in describe(new)["row0_overlap_fraction"] for v in group[1:])
        checked+=1
    assert repairs>0
    # Production default already meets 40%; new generator must not change its token selection.
    for mtp in (0,1,3):
        a=sort_topk(make_case(mtp=mtp,overlap="independent"))
        b=sort_topk(make_case(mtp=mtp,overlap="overlap40"))
        np.testing.assert_array_equal(a["topk_indices"],b["topk_indices"])
        checked+=1
    print(f"{checked} CPU input contract cases passed (not NPU)")
if __name__=="__main__": main()
