#!/usr/bin/env python3
"""Real experimental controller/VFs, CPU warp model + MrgSort API model + UBSan. NOT NPU."""
from pathlib import Path
import argparse,ctypes,json,tempfile,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from checks.check_joint_variant_kernels import build,execute
from joint_variants import HW_SWEEP
from joint_plan_cases import make_case,sort_topk,initial_arrays,OUTPUTS,POISON


def cases():
    for mtp in (0,1,3):
        for scene in ('cold_reset','full_cache_75pct'):
            yield make_case(1,65,128,262144,mtp,scene,'overlap40',block_size=8)
    # 3 active rows, irregular capacity and bin boundaries.
    c=make_case(1,21,63,1024,3,block_size=7);c['active_rows']=3;yield c
    for empty in (False,True):
        c=make_case(2,65,128,262144,3,block_size=8);c['active_rows']=5
        c['topk_indices'][0,:6]=[-1,262144,c['context']+2,5,5,6];c['stable_prefix_lens'][0]=-1
        if empty:c['topk_indices'][:]=-1
        yield c
    for blocked in (False,True):
        c=make_case(1,7,16,262144,3,block_size=8)
        c['slot_to_token'][:]=200000 if blocked else np.arange(16)+200000
        c['slot_to_token'][:,1]=200000;c['topk_indices'][:]=[200000,200000,200001,200002,200002,200020,200021]
        yield c
    yield make_case(1,65,128,262144,3,overlap='disjoint',block_size=8)
    c=make_case(4,128,128,262144,3,block_size=8);c['token_to_req'][:]=0;yield c
    # All selected tokens in one value bin, repeated across refills.
    c=make_case(1,65,128,262144,3,block_size=8);c['topk_indices'][:]=np.arange(65)%7+200000;yield c
    # Float precision limit, then unsupported maxToken (must fall back).
    for max_token in (16777216,16777217):
        c=make_case(1,7,16,262144,3,block_size=8);c['max_token']=max_token
        c['visible_seq_lens'][:]=max_token;c['stable_prefix_lens'][:]=max_token
        c['topk_indices'][:]=[0,5,5,8191,8192,16777214,16777215];yield c


def main(variants=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--variants',nargs='+',choices=HW_SWEEP,default=list(HW_SWEEP));p.add_argument('--report',type=Path)
    a=p.parse_args([] if variants is not None else None)
    if variants is not None:a.variants=variants
    records=[]
    with tempfile.TemporaryDirectory(prefix='hw_sweep_cpu_') as tmp:
        for name in a.variants:
            d=Path(tmp)/name;d.mkdir();lib=ctypes.CDLL(str(build(d,HW_SWEEP[name])))
            lib.run_variant.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(ctypes.c_int64),ctypes.POINTER(ctypes.c_int32)];lib.run_variant.restype=None
            checks=0;commits=0
            for raw in cases():
                c=sort_topk(raw);enabled=c['mtp']>0
                for gm in (False,True):
                    before=initial_arrays(c,enabled,4*c['topk'] if enabled else None)
                    for k in OUTPUTS:before[k].fill(POISON)
                    after=execute(lib,c,enabled,gm,before,turbo=True,merge=True,sorted_union=True)
                    commits+=int(np.count_nonzero(after['row_modes']==1))
                    changed=dict(c,req_ids=c['req_ids']+100,stable_prefix_lens=np.zeros(c['rows'],np.int32))
                    execute(lib,changed,enabled,gm,after,turbo=True,merge=True,sorted_union=True);checks+=2
            # Full two-queue fit; 4-way streaming; repeated keys cross packet boundaries.
            for mtp,scene in ((1,'full_cache_75pct'),(3,'full_cache_75pct'),(3,'cold_reset')):
                c=sort_topk(make_case(1,2048,4096,262144,mtp,scene,'overlap40'))
                after=execute(lib,c,True,False,turbo=True,merge=True,sorted_union=True)
                execute(lib,c,True,False,after,turbo=True,merge=True,sorted_union=True);checks+=2
            c=make_case(1,2048,4096,262144,3,overlap='overlap40');c['topk_indices'][:]=np.arange(2048)%64+200000
            execute(lib,sort_topk(c),True,False,turbo=True,merge=True,sorted_union=True);checks+=1
            c=sort_topk(make_case(1,33,32768,262144,1,block_size=64))
            execute(lib,c,True,False,turbo=True,merge=True,sorted_union=True);checks+=1
            assert commits>0
            records.append({'variant':name,'checks':checks,'joint_commits':commits})
            print(name,checks,'controller checks passed (CPU model, not CANN/NPU)',flush=True)
    if a.report:a.report.write_text(json.dumps({'scope':'CPU model + UBSan, not CANN/NPU','records':records},indent=2)+'\n')
    return sum(v['checks'] for v in records)
if __name__=='__main__':main()
