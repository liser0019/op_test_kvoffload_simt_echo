"""16K专项：实际设备连续状态；所有调用保留完整输入/输出/保护区审计。"""
import numpy as np
from checks.check_capacity16 import cases
from joint_plan_cases import make_case,sort_topk
import joint_plan_common as common
from fixed_plan_check import exact_workload_key
from harness_io import write_json

def gate(torch,a,workloads,environment,destination):
 records=[];evidence={n:{} for n in a.implementations};failed=destination/'failure_state';regression={}
 for label,c in [*cases(),*((f"performance/{c['scenario']}/mtp{c['mtp']}",c) for c in workloads)]:
  for name in a.implementations:
   b=common.prepare(torch,c,name);b['fast_fixed_check']=True
   try:
    result=common.audit(torch,c,b,a.regression_repeat,failed)
    records.append({'case':label,'implementation':name,'audit':result,'layout':b['layout']})
    evidence[name][exact_workload_key(c)]='passed';regression[name]=result
   finally:common.close(b)
  print('16K门禁:',label,'通过',flush=True)
 continuous=[]
 for name in a.implementations:
  c=sort_topk(make_case(1,2048,16384,262144,7,'cold_reset','overlap40',history_pool_tokens=16384))
  b=common.prepare(torch,c,name)
  try:
   for step in range(7):
    if step==2:c['req_ids']+=99
    if step==3:c['stable_prefix_lens'][:]=1000
    if step==4:c['active_rows']=1
    if step==5:c['active_rows']=8
    common.update_inputs(torch,b,c);common.poison(b);torch.npu.synchronize()
    before=common.snapshot(b);b['phase']=f'continuous-{step}'
    try:common.invoke(b);torch.npu.synchronize();result=common.check(c,b,before)
    except BaseException:common.dump_failure(c,b,failed);raise
    continuous.append({'implementation':name,'step':step,'result':result})
  finally:common.close(b)
 result={'status':'passed','environment':environment,'exact_workloads':evidence,'gate':{
  'status':'passed','implementations':list(a.implementations),'validation_scope':'mtp8_v1: capacity16 exact boundary, duplicate, invalidation, continuous mode switch',
  'input_write_regression':{'status':'passed','checked_iterations':a.regression_repeat,'per_implementation':regression,'input_restore':False},
  'capacity16_boundary':'passed','cases':records,'continuous':continuous}}
 write_json(destination/'gate.json',result);return result

def report_text(report):
 lines=['# MTP动态容量：ABCDF / 映射暂存 / 全UB','',
 '固定32请求（实际请求数以每个workload为准）；TopK2048。容量按MTP动态选择：MTP0/1=4096，MTP3=8192，MTP7=16384。仅设备Plan kernel Duration，含映射暂存。',
 '两候选保留硬件归并与AB消费计数；热收尾使用压缩布局的SIMT Scan，非原BD的32位SIMD压缩。GM版197408B，全UB版203552B。',
 '', '| 工况/MTP/容量 | 实现 | P50/P95 μs | 样本数 | 实现/ABCDF | 共享/回退 | descriptor |','|---|---|---|---:|---:|---|---:|']
 for run in report['runs']:
  w=run['workload'];base=run['stats']['abcdf_pages']['p50_us']
  for n,s in run['stats'].items():
   c=s['correctness'];lines.append(f"| {w['scenario']}/{w['mtp']} | {n} | {s['p50_us']:.3f}/{s['p95_us']:.3f} | {s['count']} | {s['p50_us']/base:.3f} | {c.get('joint_groups',0)}/{c.get('overflow_groups',0)} | {c['total_transfer_descriptors']} |")
 lines+=['','实际miss、路径计数、原始CSV、逐样本数据、身份和墙钟时间见capture_metadata.json/data_index.json。固定地址预热；未测Transfer/SFA。']
 return '\n'.join(lines)+'\n'
