#!/usr/bin/env python3
"""Actual mixed controller + VF, CPU64/UBSan and documented MrgSort model. NOT CANN or NPU."""
from pathlib import Path
import argparse,ctypes,json,tempfile,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from checks.check_joint_variant_kernels import build,execute
from checks.check_merge_plan import cases
from joint_plan_cases import make_case,sort_topk,layout,initial_arrays,OUTPUTS,POISON

def main(variants=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--variants',nargs='+',choices=('sorted_stream','sorted_mrgsort'),default=['sorted_stream','sorted_mrgsort']);p.add_argument('--report',type=Path);a=p.parse_args([] if variants is not None else None)
    if variants is not None:a.variants=variants
    records=[]
    with tempfile.TemporaryDirectory(prefix='sorted_plan_cpu_') as tmp:
        for name in a.variants:
            d=Path(tmp)/name;d.mkdir();lib=ctypes.CDLL(str(build(d,37 if name=='sorted_stream' else 38)))
            lib.run_variant.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(ctypes.c_int64),ctypes.POINTER(ctypes.c_int32)];lib.run_variant.restype=None
            count=0;commits=0
            extra=[]
            # Exact float key edge and unsupported range (no silent truncation).
            for max_token in (16777216,16777217):
                c=make_case(1,7,16,262144,3,block_size=8);c['max_token']=max_token
                c['visible_seq_lens'][:]=max_token;c['stable_prefix_lens'][:]=max_token
                c['topk_indices'][:]=[5,5,6,7,8,16777214,16777215];extra.append(c)
            # Many equal keys span repeated MrgSort refills; row-specific future bounds.
            for unique in (1,2,64,1024):
                c=make_case(1,2048,4096,262144,3,overlap='overlap40')
                c['topk_indices'][:]=np.arange(2048)%unique+200000;extra.append(c)
            wide=make_case(4,128,128,262144,3,block_size=8);wide['token_to_req'][:]=0
            extra.append(wide) # >4 rows must reject before touching a 4T map
            for i,raw in enumerate([*cases(),*extra]):
                c=sort_topk(raw);enabled=c['mtp']>0
                for gm in (False,True):
                    info=layout(c['topk'],c['capacity'],c['rows'],enabled,c['max_token'],turbo=True,sorted_union=True)
                    before=initial_arrays(c,enabled,4*c['topk'] if enabled else None)
                    for k in OUTPUTS:before[k].fill(POISON)
                    after=execute(lib,c,enabled,gm,before,turbo=True,merge=True,sorted_union=True)
                    commits+=int(np.count_nonzero(after['row_modes']==1))
                    execute(lib,c,enabled,gm,after,turbo=True,merge=True,sorted_union=True)
                    c2=dict(c,req_ids=c['req_ids']+99,stable_prefix_lens=np.zeros(c['rows'],np.int32))
                    execute(lib,c2,enabled,gm,after,turbo=True,merge=True,sorted_union=True);count+=3
                print(name,'case',i+1,'passed',flush=True)
            assert commits>0
            records.append({'variant':name,'controller_checks':count,'joint_commits':commits})
    if a.report:a.report.write_text(json.dumps({'scope':'CPU64 controller + UBSan + MrgSort model; NOT CANN/NPU','records':records},indent=2)+'\n')
    print(records,flush=True);return sum(v['controller_checks'] for v in records)
if __name__=='__main__':main()
