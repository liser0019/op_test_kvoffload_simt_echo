#!/usr/bin/env python3
"""Execute real Merge VF with CPU64/UBSan and paged ABCDF; not CANN/NPU validation."""
from pathlib import Path
import ctypes, tempfile, sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from checks.check_joint_variant_kernels import build, execute
from checks.check_paged_abcdf_kernels import build as build_paged
from joint_plan_cases import make_case, sort_topk, require_sorted_topk, initial_arrays, layout, OUTPUTS, POISON


def cases():
    for mtp in (0, 1, 3):
        for scenario in ("cold_reset", "full_cache_75pct", "full_cache_90pct", "full_cache_stable_hit"):
            for overlap in ("shared", "independent", "disjoint"):
                yield make_case(2, 65, 128, 262144, mtp, scenario, overlap, block_size=8)
    # Full 256K token range, actual 112KiB UB allocation; >32768 and >65535 tokens.
    for mtp in (1, 3):
        for scenario in ("cold_reset", "full_cache_75pct"):
            yield make_case(1, 2048, 4096, 262144, mtp, scenario, "independent")
    for unique in (1, 16, 17):
        c=make_case(1, 7, 16, 262144, 3, block_size=8)
        c["topk_indices"][:]=np.array([[(r*7+p)%unique+200000 for p in range(7)] for r in range(4)])
        yield c
    c=make_case(1, 65, 128, 262144, 3, block_size=8)
    c["topk_indices"][:]=200000 # duplicates across scan tile boundary
    yield c
    for empty in (True, False):
        c=make_case(2, 7, 16, 262144, 3, block_size=8)
        c["topk_indices"][:]=-1
        if not empty:
            c["topk_indices"][0]=[-1,262144,c["context"]+2,200000,200000,5,6]
            c["topk_indices"][1:,0]=c["context"]+2
        c["stable_prefix_lens"][0]=-1
        c["token_to_req"][4:]=-1
        yield c
    # Duplicate resident: last LRU owner wins, every matching slot protected.
    for blocked in (False, True):
        c=make_case(1,7,16,262144,3,block_size=8)
        c["slot_to_token"][:]=200000 if blocked else np.arange(16)+200000
        c["slot_to_token"][:,1]=200000
        c["topk_indices"][:]=[200000,200000,200001,200002,200002,200020,200021]
        yield c
    for active in (0,3,5):
        c=make_case(2,7,16,262144,3,block_size=8);c["active_rows"]=active;yield c
    # Encoding exceeds 32 bits: real private fallback, no truncation.
    yield make_case(1,7,16,1<<20,3,block_size=8) # still packable
    c=make_case(1,65,128,512,3,block_size=64);c["max_token"]=1<<24;yield c
    c=make_case(2,7,16,512,3,block_size=8);c["token_to_req"][:]=0;yield c # >4 rows
    yield make_case(1,33,32768,262144,1,block_size=64) # natural GM


def main():
    unsorted=make_case(1,7,16,512,3,block_size=8)
    try: require_sorted_topk(unsorted)
    except ValueError: pass
    else: raise AssertionError("unsorted input accepted")
    with tempfile.TemporaryDirectory(prefix="merge_vf_") as tmp:
        root=Path(tmp);(root/"merge").mkdir();(root/"baseline").mkdir()
        lib=ctypes.CDLL(str(build(root/"merge",21))); baseline=build_paged(root/"baseline",0)
        for obj in (lib,baseline):
            obj.run_variant.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(ctypes.c_int64),ctypes.POINTER(ctypes.c_int32)]
            obj.run_variant.restype=None
        count=0; joint_commits=0
        for i,raw in enumerate(cases()):
            c=sort_topk(raw)
            for gm in (False,True):
                enabled=c["mtp"]>0
                info=layout(c["topk"],c["capacity"],c["rows"],enabled,c["max_token"],turbo=True)
                before=initial_arrays(c,enabled,info["scratch_elements"]-2*info["joint_hash_capacity"]-c["capacity"] if enabled else None)
                # Outputs cannot be read as persistent state (prototype residentRows bug).
                for key in OUTPUTS: before[key].fill(POISON)
                after=execute(lib,c,enabled,gm,before,turbo=True,merge=True)
                joint_commits+=int(np.count_nonzero(after["row_modes"]==1))
                # Repeat actual state; then invalidate suffix + change identity.
                execute(lib,c,enabled,gm,after,turbo=True,merge=True)
                c2=dict(c,req_ids=c["req_ids"]+11,stable_prefix_lens=np.zeros(c["rows"],np.int32))
                execute(lib,c2,enabled,gm,after,turbo=True,merge=True)
                execute(baseline,c,False,gm,abcdf=True)
                count+=4
            print(f"case {i+1}: passed",flush=True)
        assert joint_commits>0
        print(f"{count} real VF CPU64+UBSan checks passed; {joint_commits} joint commits (NOT NPU)",flush=True)
        return count
if __name__=="__main__": main()
