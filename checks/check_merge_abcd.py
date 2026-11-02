#!/usr/bin/env python3
"""All Merge ABCD: actual CPU64 VF + UBSan, unsigned pair merge, guards and independent semantics. NOT NPU."""
from pathlib import Path
import argparse, ctypes, tempfile, sys, json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from checks.check_joint_variant_kernels import build,execute
from joint_variants import MERGE_VARIANTS
from joint_plan_cases import make_case,sort_topk,initial_arrays,layout,OUTPUTS,POISON


def pair_checks(lib):
    ptr=ctypes.POINTER(ctypes.c_int32)
    lib.run_merge_pair.argtypes=[ptr,ctypes.c_uint32,ptr,ctypes.c_uint32,ptr]
    rng=np.random.default_rng(2026)
    for size in [(0,0),(0,1),(1,0),(1,1),(7,9),(8,8),(63,65),(2048,2048),(0,8192),(4096,4096)]+[(int(rng.integers(0,70)),int(rng.integers(0,70))) for _ in range(80)]:
        for duplicate in (False,True):
            a=np.sort(rng.integers(0,8 if duplicate else 1<<32,size=size[0],dtype=np.uint32)).view(np.int32)
            b=np.sort(rng.integers(0,8 if duplicate else 1<<32,size=size[1],dtype=np.uint32)).view(np.int32)
            backing=np.full(a.size+b.size+32,1234567,np.int32);out=backing[16:-16]
            lib.run_merge_pair(a.ctypes.data_as(ptr),len(a),b.ctypes.data_as(ptr),len(b),out.ctypes.data_as(ptr))
            np.testing.assert_array_equal(out.view(np.uint32),np.sort(np.concatenate((a,b)).view(np.uint32)))
            assert np.all(backing[:16]==1234567) and np.all(backing[-16:]==1234567)
    return 180


def cases():
    for mtp in (0,1,3):
        for scene in ('cold_reset','full_cache_75pct'):
            yield make_case(1,65,128,262144,mtp,scene,'overlap40',block_size=8)
    for unique in (1,16,17):
        c=make_case(1,7,16,262144,3,block_size=8)
        c['topk_indices'][:]=[[200000+(r*7+p)%unique for p in range(7)] for r in range(4)]
        yield c
    for blocked in (False,True):
        c=make_case(1,7,16,262144,3,block_size=8)
        c['slot_to_token'][:]=200000 if blocked else np.arange(16)+200000
        c['slot_to_token'][:,1]=200000
        c['topk_indices'][:]=[200000,200000,200001,200002,200002,200020,200021]
        yield c
    c=make_case(2,65,128,262144,3,block_size=8);c['active_rows']=5
    c['topk_indices'][0,:6]=[-1,262144,c['context']+2,5,5,6];c['stable_prefix_lens'][0]=-1
    yield c
    c=make_case(1,65,128,262144,3,block_size=8);c['topk_indices'][:]=-1;yield c
    # Multi-tile duplicate queries exercise phase barriers and frozen high tags.
    c=make_case(1,65,128,262144,3,block_size=8);c['topk_indices'][:]=200000;yield c
    c=make_case(2,7,16,512,3,block_size=8);c['token_to_req'][:]=0;yield c
    c=make_case(1,65,128,512,3,block_size=64);c['max_token']=1<<24;yield c


def main(variants=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--variants',nargs='+',choices=MERGE_VARIANTS,default=list(MERGE_VARIANTS));p.add_argument('--report',type=Path);args=p.parse_args([] if variants is not None else None)
    if variants is not None: args.variants=variants
    records=[]
    with tempfile.TemporaryDirectory(prefix='merge_abcd_cpu_') as tmp:
        for name in args.variants:
            d=Path(tmp)/name;d.mkdir();lib=ctypes.CDLL(str(build(d,MERGE_VARIANTS[name])))
            lib.run_variant.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(ctypes.c_int64),ctypes.POINTER(ctypes.c_int32)];lib.run_variant.restype=None
            pairs=pair_checks(lib);count=0
            for raw in cases():
                c=sort_topk(raw);enabled=c['mtp']>0
                for gm in (False,True):
                    info=layout(c['topk'],c['capacity'],c['rows'],enabled,c['max_token'],turbo=True)
                    before=initial_arrays(c,enabled,info['scratch_elements']-2*info['joint_hash_capacity']-c['capacity'] if enabled else None)
                    for k in OUTPUTS:before[k].fill(POISON)
                    after=execute(lib,c,enabled,gm,before,turbo=True,merge=True)
                    # Consume device-produced state; request reset and suffix invalidation together.
                    c2=dict(c,req_ids=c['req_ids']+123,stable_prefix_lens=np.zeros(c['rows'],np.int32))
                    execute(lib,c2,enabled,gm,after,turbo=True,merge=True);count+=2
            for scene in ('cold_reset','full_cache_75pct'):
                c=sort_topk(make_case(1,2048,4096,262144,3,scene,'overlap40'))
                after=execute(lib,c,True,False,turbo=True,merge=True)
                execute(lib,c,True,False,after,turbo=True,merge=True);count+=2
            # Actual largest allowed cached bucket layout + natural GM selection.
            c=sort_topk(make_case(1,33,32768,262144,1,block_size=64))
            execute(lib,c,True,False,turbo=True,merge=True);count+=1
            records.append({'name':name,'vf_checks':count,'merge_pair_checks':pairs})
            print(name,count,'VF +',pairs,'pair checks passed (CPU64, not NPU)',flush=True)
    if args.report:args.report.write_text(json.dumps({'scope':'CPU64+UBSan only','records':records},indent=2)+'\n')
    return sum(item['vf_checks'] for item in records)
if __name__=='__main__':main()
