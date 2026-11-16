"""真实设备源CPU桩执行；UBSan/保护区/语义，不是CANN或NPU验证。"""
from pathlib import Path
import sys,tempfile,ctypes
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from checks.check_joint_variant_kernels import build,execute
from joint_plan_cases import make_case,sort_topk
import numpy as np

def cases():
 for m in (0,1,3,7):
  for s in ('cold_reset','full_cache_75pct'):
   yield f'{s}/{m}',sort_topk(make_case(1,2048,{0:4096,1:4096,3:8192,7:16384}[m],262144,m,s,'overlap40',history_pool_tokens={0:4096,1:4096,3:8192,7:16384}[m]))
 c=sort_topk(make_case(1,2048,16384,262144,7,'cold_reset','disjoint'))
 for row in range(8):c['topk_indices'][row]=100000+2048*row+np.arange(2048,dtype=np.int32)
 yield 'exact16384',c
 for cap,m in ((4096,1),(8192,3)):
  c=sort_topk(make_case(1,2048,cap,262144,m,'cold_reset','disjoint'))
  for row in range(m+1):c['topk_indices'][row]=100000+2048*row+np.arange(2048,dtype=np.int32)
  yield f'exact{cap}',c
 c={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in c.items()};c['topk_indices'][:]=100000
 yield 'duplicates',c
 c=sort_topk(make_case(1,2048,16384,262144,7,'full_cache_75pct','overlap40',history_pool_tokens=16384))
 c['stable_prefix_lens'][:]=1000
 yield 'invalidation',c
 c={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in c.items()};c['topk_indices'][:]=-1
 yield 'empty',c

def main():
 for mode in (1,2):
  with tempfile.TemporaryDirectory() as tmp:
   lib=ctypes.CDLL(str(build(Path(tmp),213+mode)))
   for label,c in cases():
    execute(lib,c,True,False,turbo=True,sorted_union=True,single_sorted=True,mtp8=True,layout16=mode)
    print(mode,label,'passed',flush=True)
   c=sort_topk(make_case(1,2048,16384,262144,7,'cold_reset','overlap40',history_pool_tokens=16384))
   before=None
   for step in range(6):
    if step==2:c['req_ids']+=17
    if step==3:c['stable_prefix_lens'][:]=1000
    if step==4:c['active_rows']=1
    if step==5:c['active_rows']=8
    before=execute(lib,c,True,False,before=before,turbo=True,sorted_union=True,single_sorted=True,mtp8=True,layout16=mode)
    print(mode,'continuous',step,'passed',flush=True)
   # Duplicate resident owners remain protected; LRU-last decides slot.
   c=sort_topk(make_case(1,2048,16384,262144,7,'full_cache_75pct','overlap40',history_pool_tokens=16384))
   from joint_plan_cases import initial_arrays
   before=initial_arrays(c,True,8*2048)
   token=int(c['topk_indices'][0,0]);before['slot_to_token'][0,:2]=token
   execute(lib,c,True,False,before=before,turbo=True,sorted_union=True,single_sorted=True,mtp8=True,layout16=mode)
   print(mode,'duplicate resident','passed',flush=True)
if __name__=='__main__':main()
