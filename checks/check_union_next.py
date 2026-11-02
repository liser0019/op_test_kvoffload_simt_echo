#!/usr/bin/env python3
"""Actual new controller + SIMT/SIMD bodies under CPU API models/UBSan; NOT CANN/NPU."""
from pathlib import Path
import argparse,ctypes,json,tempfile,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from checks.check_joint_variant_kernels import build,execute
from checks.check_hardware_sweep import cases
from joint_plan_cases import make_case,sort_topk
from joint_variants import UNION_SWEEP


def prefix_gate(lib):
    lib.run_next_prefix.argtypes=[ctypes.POINTER(ctypes.c_int32),ctypes.c_uint32,ctypes.c_int32]
    rng=np.random.default_rng(2026)
    for n in (1,2,31,32,63,64,65,127,128,129,767,768,769,2048,8192):
        for seam in (False,True):
            token=np.sort(rng.integers(0,100,max(n,1))).astype(np.int32)
            previous=int(token[0]) if seam else -1
            p=np.full(2*n+32,0x1234567,np.int32);a=p[16:-16].reshape(-1,2)
            a[:,0]=(-token.astype(np.float32)).view(np.int32);a[:,1]=rng.permutation(n)
            before=a.copy();lib.run_next_prefix(a.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),n,previous)
            prefix=np.cumsum(token!=np.r_[previous,token[:-1]])
            np.testing.assert_array_equal(a[:,0],before[:,0])
            np.testing.assert_array_equal(a[:,1].view(np.uint32), (prefix.astype(np.uint32)<<16)|before[:,1].astype(np.uint32))
            assert np.all(p[:16]==0x1234567) and np.all(p[-16:]==0x1234567)
    return 30


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variants',nargs='+',choices=UNION_SWEEP,default=list(UNION_SWEEP));p.add_argument('--report',type=Path)
    p.add_argument('--model-threads',type=int,default=64,choices=(64,128,256,512,1024));a=p.parse_args();records=[]
    with tempfile.TemporaryDirectory(prefix='union_next_cpu_') as tmp:
        for name in a.variants:
            mask=UNION_SWEEP[name];bits=(mask-74)%16;d=Path(tmp)/name;d.mkdir()
            lib=ctypes.CDLL(str(build(d,mask,model_threads=a.model_threads)))
            lib.run_variant.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(ctypes.c_int64),ctypes.POINTER(ctypes.c_int32)]
            prefix=prefix_gate(lib);checks=0
            extra=[]
            for scene in ('cold_reset','full_cache_75pct','full_cache_stable_hit'):
                for duplicates in (False,True):
                    c=make_case(2,65,128,262144,0,scene,block_size=8)
                    if duplicates:c['topk_indices'][:,1]=c['topk_indices'][:,0]
                    extra.append(c)
            c=make_case(1,65,128,262144,0,block_size=8);c['topk_indices'][0,:4]=[-1,-1,262144,c['context']+3];extra.append(c)
            for raw in [*cases(),*extra]:
                c=sort_topk(raw);enabled=c['mtp']>0
                for gm in (False,True):
                    kw=dict(turbo=True,merge=True,sorted_union=True,union_next=True,preserve_tail=bool(bits&8))
                    out=execute(lib,c,enabled,gm,**kw)
                    out=execute(lib,c,enabled,gm,out,**kw)
                    changed=dict(c,req_ids=c['req_ids']+100,stable_prefix_lens=np.zeros(c['rows'],np.int32))
                    execute(lib,changed,enabled,gm,out,**kw);checks+=3
            for mtp in (0,1,3):
                for scene in ('cold_reset','full_cache_75pct'):
                    c=sort_topk(make_case(1,2048,4096,262144,mtp,scene,'overlap40'))
                    kw=dict(turbo=True,merge=True,sorted_union=True,union_next=True,preserve_tail=bool(bits&8))
                    execute(lib,c,mtp>0,False,**kw);checks+=1
            records.append(dict(variant=name,controller_checks=checks,prefix_checks=prefix,model_threads=a.model_threads))
            print(name,checks,'controller +',prefix,'SIMD prefix checks passed (CPU, NOT NPU)',flush=True)
    if a.report:a.report.write_text(json.dumps({'scope':'CPU API models + actual controller/VF source + UBSan; not CANN or NPU','records':records},indent=2)+'\n')
if __name__=='__main__':main()
