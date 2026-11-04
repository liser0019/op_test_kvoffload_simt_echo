#!/usr/bin/env python3
"""CPU actual-controller + UBSan; explicitly NOT Bisheng/Ascend validation."""
from pathlib import Path
import ctypes,json,sys,tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from checks.check_cold_finish import load
from checks.check_joint_variant_kernels import execute
from checks.mtp8_cases import cases,transitions
from joint_plan_cases import make_case,sort_topk,layout


def main():
    count=0
    with tempfile.TemporaryDirectory(prefix='mtp8_cpu_') as tmp:
        lib=load(Path(tmp)/'eight',193);old=load(Path(tmp)/'old',169)
        kw=dict(turbo=True,merge=True,sorted_union=True,single_sorted=True)
        for label,c in cases():
            before=None
            for step in range(3):
                current=dict(c,req_ids=c['req_ids']+41) if step==2 else c
                before=execute(lib,current,True,False,before,mtp8=True,**kw);count+=1
            print('passed',label,flush=True)
        before=None
        for label,c in transitions():
            before=execute(lib,c,True,False,before,mtp8=True,**kw);count+=1
        for mtp in (0,1,3):
            for scenario in ('cold_reset','full_cache_75pct'):
                c=sort_topk(make_case(1,2048,4096,262144,mtp,scenario,'overlap40'))
                before=None
                for step in range(2):
                    expected=execute(old,c,True,False,before,**kw)
                    actual=execute(lib,c,True,False,before,mtp8=True,**kw)
                    for k in expected:np.testing.assert_array_equal(actual[k],expected[k],err_msg='frozen-BD/'+k)
                    before=actual;count+=1
        for scenario in ('cold_reset','full_cache_75pct'):
            c=sort_topk(make_case(1,2048,4096,262144,7,scenario,'overlap40'))
            execute(lib,c,True,False,mtp8=True,**kw);count+=1
        rng=np.random.default_rng(7041)
        for sample in range(24):
            t=int(rng.choice([7,31,63,65,129,513]));cap=((t*int(rng.choice([1,2,4]))+7)//8)*8
            c=sort_topk(make_case(1,t,cap,262144,7,'cold_reset' if sample%2 else 'full_cache_75pct',str(rng.choice(['shared','overlap40','disjoint'])),seed=sample+211,block_size=8))
            if sample%3==0:c['topk_indices'][:]=np.sort(rng.integers(-1,max(t*3,8),(8,t),dtype=np.int32),axis=1)
            before=execute(lib,c,True,False,mtp8=True,**kw)
            execute(lib,c,True,False,before,mtp8=True,**kw);count+=2
        info=layout(2048,4096,256,True,262144,sorted_union=True,mtp8=True)
        assert info['scratch_path']=='UB' and info['workspace_elements']==1 and info['dynamic_ub_bytes']==181024
        print(json.dumps({'status':'passed','controller_calls':count,'production_layout':info,'scope':'CPU + UBSan, NOT CANN/NPU'}))
if __name__=='__main__':main()
