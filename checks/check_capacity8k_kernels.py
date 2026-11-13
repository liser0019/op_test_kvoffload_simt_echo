#!/usr/bin/env python3
"""实际8K控制流CPU/UBSan验证；独立golden+逐字段对照，不能验证NPU地址空间/同步/性能。"""
from pathlib import Path
import ctypes,json,sys,tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from checks.check_joint_variant_kernels import build,execute
from checks.check_cold_finish import load
from checks.capacity8k_cases import capacity_cases,capacity_transitions
from checks.mtp8_cases import cases,transitions
from joint_plan_cases import make_case,sort_topk


def main():
    with tempfile.TemporaryDirectory(prefix='capacity8k_cpu_') as tmp:
        root=Path(tmp);ref=load(root/'ref',212);candidate=load(root/'candidate',213)
        frozen_fused=load(root/'frozen204',204);frozen_ab=load(root/'frozen207',207)
        kw=dict(turbo=True,merge=True,sorted_union=True,single_sorted=True,mtp8=True)
        count=0;frozen_count=0;wide_count=0
        def compare(case,before=None,eight=True):
            nonlocal count
            a=execute(ref,case,True,False,before,capacity8k=eight,**kw)
            b=execute(candidate,case,True,False,before,capacity8k=eight,**kw)
            for key in a:np.testing.assert_array_equal(a[key],b[key],err_msg=key)
            count+=1;return b
        for label,c in capacity_cases():
            before=None
            for step in range(2):before=compare(c,before)
            print('passed',label,flush=True)
        before=None
        for label,c in capacity_transitions():before=compare(c,before)
        for mtp in (0,1,3,7):
            for scenario in ('cold_reset','full_cache_75pct'):
                c=sort_topk(make_case(2,2048,8192,262144,mtp,scenario,'overlap40',history_pool_tokens=8192));before=None
                for step in range(2):before=compare(c,before)
        # Frozen4096 reference comparison, including naturally falling-back shapes and actual continuations.
        for label,c in cases(include_extreme=True,include_prefix=True):
            before=None
            for step in range(2):
                b=compare(c,before)
                for lib in(frozen_fused,frozen_ab):
                    a=execute(lib,c,True,False,before,**kw)
                    for key in a:np.testing.assert_array_equal(a[key],b[key],err_msg='frozen/'+key)
                    frozen_count+=1
                before=b
        before=None
        for label,c in transitions():before=compare(c,before)
        wide_dir=root/'wide';wide_dir.mkdir();wide=ctypes.CDLL(str(build(wide_dir,213,model_threads=256)))
        wide.run_variant.argtypes=candidate.run_variant.argtypes
        for scenario in ('cold_reset','full_cache_75pct'):
            c=sort_topk(make_case(1,2048,8192,262144,7,scenario,'overlap40',history_pool_tokens=8192));before=None
            for step in range(2):
                a=compare(c,before);b=execute(wide,c,True,False,before,capacity8k=True,**kw)
                for key in a:np.testing.assert_array_equal(a[key],b[key],err_msg='wide/'+key)
                before=b;wide_count+=1
        print(json.dumps({'status':'passed','exact_controller_comparisons':count,'frozen4096_comparisons':frozen_count,'multiwarp8k_comparisons':wide_count,'scope':'CPU+UBSan; not CANN/NPU'}),flush=True)
if __name__=='__main__':main()
