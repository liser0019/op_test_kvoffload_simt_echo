#!/usr/bin/env python3
"""CPU+UBSan exact controller and SIMD-boundary checks. Not a CANN/NPU test."""
from pathlib import Path
import argparse,ctypes,json,subprocess,sys,tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from checks.check_mtp7_fast import worker
from checks.check_cold_finish import load
from checks.check_hot_kernels import aligned
from checks.check_joint_variant_kernels import execute,build
from joint_plan_cases import make_case,sort_topk


def primitives(mask):
    count=0;rng=np.random.default_rng(9016);ptr=ctypes.c_void_p
    with tempfile.TemporaryDirectory(prefix="fused_tune_primitive_") as tmp:
        root=Path(tmp);reference=load(root/'ref',204);candidate=load(root/'candidate',mask)
        reference.run_fused_unique.argtypes=[ptr]*3+[ctypes.c_uint32]*4+[ctypes.c_int32]+[ctypes.c_uint32]*2
        candidate.run_tuned_unique.argtypes=[ptr]*3+[ctypes.c_uint32]*4+[ctypes.c_int32,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint32,ctypes.c_uint32]
        for topk in (7,65,513,2048):
            q=8*topk
            for n in sorted(set(min(q,v) for v in (0,1,31,32,33,63,64,65,127,128,129,511,512,513,4896))):
                for mode in (-1,0,1,2):
                    data,dg,do=aligned(q+4096+24576);ref,rg,ro=aligned(len(data));ref[:]=data
                    ctx,cg,co=aligned(128);rc,rcg,rco=aligned(128)
                    initial=rng.integers(0,19,8,dtype=np.int32);ctx[48:56]=initial;ctx[56:64]=initial+topk
                    payload=rng.permutation(q)[:n].astype(np.int32)
                    hist=np.bincount(payload//topk,minlength=8).astype(np.int32)
                    # Only known rows contain authoritative first-layer counts; the rest are poisoned.
                    ctx[112:120]=-777
                    if mode in (0,2):ctx[112:116]=hist[:4]
                    if mode in (1,2):ctx[116:120]=hist[4:]
                    rc[:]=ctx
                    # Exactly the available prefix plus sentinel guards. Full loads may never touch the tail.
                    records,pg,po=aligned(max(2*n,1))
                    records.view(np.float32)[:2*n:2]=-(np.arange(n)//3+100).astype(np.float32)
                    records[1:2*n:2]=payload
                    kt=int(hist[:4].sum() if mode==0 else hist[4:].sum())
                    ut=int(hist[4:].sum() if mode==0 else hist[:4].sum())
                    reference.run_fused_unique(records.ctypes.data,ref.ctypes.data,rc.ctypes.data,n,q,4096,3,100,1,topk)
                    candidate.run_tuned_unique(records.ctypes.data,data.ctypes.data,ctx.ctypes.data,n,q,4096,3,100,topk,mode,kt,ut)
                    np.testing.assert_array_equal(data,ref,err_msg=f'{mask}/{topk}/{n}/{mode}/data')
                    np.testing.assert_array_equal(ctx,rc,err_msg=f'{mask}/{topk}/{n}/{mode}/ctx')
                    for backing,off,size in ((dg,do,len(data)),(rg,ro,len(ref)),(cg,co,128),(rcg,rco,128),(pg,po,len(records))):
                        assert np.all(backing[:off]==-777) and np.all(backing[off+size:]==-777)
                    count+=1
        wide=root/'wide';wide.mkdir();lib=ctypes.CDLL(str(build(wide,mask,model_threads=256)))
        for scenario in ('cold_reset','full_cache_75pct'):
            c=sort_topk(make_case(1,65,256,262144,7,scenario,'shared',block_size=8));before=None
            for step in range(2):
                kw=dict(turbo=True,merge=True,sorted_union=True,single_sorted=True,mtp8=True)
                expected=execute(reference,c,True,False,before,**kw);actual=execute(lib,c,True,False,before,**kw)
                for k in expected:np.testing.assert_array_equal(actual[k],expected[k])
                before=actual;count+=1
    return count


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int,choices=range(205,212));a=p.parse_args()
    if a.worker:
        r=worker(a.worker,reference_mask=204,include_prefix=True)
        r['primitives_and_wide_controller']=primitives(a.worker);print(json.dumps(r),flush=True);return
    for m in range(205,212):subprocess.run([sys.executable,'-B',__file__,'--worker',str(m)],check=True)
if __name__=='__main__':main()
