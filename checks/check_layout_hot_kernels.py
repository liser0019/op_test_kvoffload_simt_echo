"""真实候选controller/SIMD/SIMT源码CPU桩+UBSan。不是NPU验证。"""
from pathlib import Path
import sys,ctypes,tempfile,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from checks.check_capacity16 import cases
from checks.check_joint_variant_kernels import build,execute
from joint_variants import LAYOUT_HOT_MASKS,layout_hot_bits
from joint_plan_cases import make_case,sort_topk,initial_arrays

def run(masks,model_threads=64,smoke=False):
 count=0
 for name,mask in masks:
  mode=1+(mask-216)//6;bits=layout_hot_bits(mask)
  with tempfile.TemporaryDirectory(prefix='layout_hot_cpu_') as tmp:
   lib=ctypes.CDLL(str(build(Path(tmp),mask,model_threads=model_threads)))
   def check(c,before=None):
    return execute(lib,c,True,False,before=before,turbo=True,sorted_union=True,single_sorted=True,mtp8=True,layout16=mode,layout_hot_bits=bits)
   selected=[(label,c) for label,c in cases() if not smoke or label in ('cold_reset/7','full_cache_75pct/7','exact16384','empty')]
   for label,c in selected:check(c);count+=1;print(name,label,'passed',flush=True)
   for cap,m in (((16384,7),) if smoke else ((4096,1),(8192,3),(16384,7))):
    c=sort_topk(make_case(1,2048,cap,262144,m,'cold_reset','overlap40',history_pool_tokens=cap));before=None
    for step in range(6):
     if step==2:c['req_ids']+=17
     if step==3:c['stable_prefix_lens'][:]=1000
     if step==4:c['active_rows']=1
     if step==5:c['active_rows']=m+1
     before=check(c,before);count+=1
    print(name,'continuous',cap,'passed',flush=True)
    c=sort_topk(make_case(1,2048,cap,262144,m,'full_cache_75pct','overlap40',history_pool_tokens=cap))
    before=initial_arrays(c,True,8*2048);before['slot_to_token'][0,:2]=int(c['topk_indices'][0,0]);check(c,before);count+=1
   # Wide groups exceed A's compact map bound: reject before touching map, then private.
   c=sort_topk(make_case(1,2048,4096,262144,7,'cold_reset','disjoint'));check(c);count+=1
   c=sort_topk(make_case(1,2048,4096,262144,7,'cold_reset','shared'));check(c);count+=1
   print(name,'ALL PASSED',flush=True)
 print('total',count,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--masks',nargs='+',type=int);p.add_argument('--model-threads',type=int,choices=(64,128,256),default=64);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 run([(n,m) for n,m in LAYOUT_HOT_MASKS.items() if a.masks is None or m in a.masks],a.model_threads,a.smoke)
