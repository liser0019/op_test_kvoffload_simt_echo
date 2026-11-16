#!/usr/bin/env python3
"""构建一次；18路短测→按热工况固定选2路→同库独立复测。"""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
import joint_plan_common as common
from harness_io import preflight_args,write_json,record_failure
from checks.stage_exec_suite import select,report_text,factor_text
from joint_variants import STAGE_EXEC_REFS,STAGE_EXEC_NAMES
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser(description=__doc__);common.add_runtime_arguments(p);p.set_defaults(device=0,temp_dir=None)
 p.add_argument('--output',type=Path,required=True);p.add_argument('--requests',type=int,default=32)
 p.add_argument('--screen-repeat',type=int,default=5);p.add_argument('--confirm-repeat',type=int,default=20)
 p.add_argument('--regression-repeat',type=int,default=5);p.add_argument('--skip-build',action='store_true');p.add_argument('--screen-only',action='store_true');p.add_argument('--describe-only',action='store_true')
 a=p.parse_args()
 if min(a.requests,a.screen_repeat,a.confirm_repeat,a.regression_repeat)<=0:p.error('positive counts required')
 if a.describe_only:
  print(json.dumps({'implementations':[*STAGE_EXEC_REFS,*STAGE_EXEC_NAMES[1:]],'requests':a.requests,'mtp':[0,1,3,7], 'capacities':[4096,4096,8192,16384],'screen_repeat':a.screen_repeat,'confirm_repeat':None if a.screen_only else a.confirm_repeat,'build':'--stage-exec-only','screen_target_samples':18*8*a.screen_repeat,'status':'plan only; no NPU run'},ensure_ascii=False,indent=2));return
 a.output=a.output.expanduser().resolve();a.library=a.library.expanduser().resolve();a.temp_dir=(a.temp_dir or a.output/'tmp').expanduser().resolve()
 if a.output.exists():raise RuntimeError('refuse overwrite: '+str(a.output))
 storage=preflight_args(a,a.output);status=a.output/'RUN_STATUS.json';state={'status':'running','storage':storage,'wall_seconds':{}}
 def run(cmd,label):
  log=a.output/(label+'.log');state.update(phase=label,log=str(log));write_json(status,state);start=time.monotonic()
  print('开始:',label,'详细日志:',log,flush=True)
  with log.open('w') as stream:r=subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
  state['wall_seconds'][label]=time.monotonic()-start
  if r.returncode:
   lines=log.read_text(errors='replace').splitlines();print('\n'.join(lines[-35:]),flush=True)
   raise RuntimeError(f'{label} rc={r.returncode}; full log={log}')
 def load(dest):
  if not (dest/'COMPLETE.json').is_file():raise RuntimeError('missing COMPLETE: '+str(dest))
  report=json.loads((dest/'capture_metadata.json').read_text())
  if report.get('status')!='complete':raise RuntimeError('capture not complete')
  for r in report['runs']:
   for s in r['stats'].values():
    if s['count']!=report['repeat']:raise RuntimeError('sample count mismatch')
  return report
 runtime=['--device',str(a.device),'--library',str(a.library),'--requests',str(a.requests),'--temp-dir',str(a.temp_dir),'--regression-repeat',str(a.regression_repeat),
  '--min-output-free-bytes',str(a.min_output_free_bytes),'--min-temp-free-bytes',str(a.min_temp_free_bytes),'--min-free-inodes',str(a.min_free_inodes)]
 try:
  if not a.skip_build:run(['bash','build_joint_plan.sh','--soc=ascend950','--stage-exec-only'],'build')
  run([sys.executable,'-B','-X','faulthandler','run_mtp8.py','--suite','stage_exec','--skip-build',*runtime,'--warmup','2','--repeat',str(a.screen_repeat),'--output',str(a.output/'screen')],'screen')
  screen=load(a.output/'screen');chosen,ranking=select(screen)
  (a.output/'screen_report.md').write_text(report_text(screen))
  (a.output/'factor_effects.md').write_text(factor_text(screen))
  write_json(a.output/'selection.json',{'selected':chosen,'ranking':ranking,'criterion':'two fixed candidates by all four hot-workload geometric means over frozen stage ABCD; screening only'})
  result=screen;used=a.output/'screen'
  if not a.screen_only:
   run([sys.executable,'-B','-X','faulthandler','run_mtp8.py','--suite','stage_exec','--skip-build',*runtime,'--correctness-report',str(a.output/'screen/gate.json'),'--warmup','10','--repeat',str(a.confirm_repeat),'--implementations',*chosen,'--output',str(a.output/'confirm')],'confirm')
   result=load(a.output/'confirm');used=a.output/'confirm'
   # Validate exact input, device, library identity across independent captures.
   if result['environment']!=screen['environment']:raise RuntimeError('confirmation environment differs')
   key=lambda x:(x['workload']['scenario'],x['workload']['mtp'],x['workload']['capacity'],x['workload']['logical_input_sha256'])
   if [key(x) for x in result['runs']]!=[key(x) for x in screen['runs']]:raise RuntimeError('confirmation inputs differ')
  text=report_text(result)+f"\n数据阶段：{'初筛，未独立复测' if a.screen_only else '独立复测；不合并初筛样本'}\n\n原始索引：{used/'data_index.json'}\n初筛：{a.output/'screen/capture_metadata.json'}\n"
  (a.output/'report.md').write_text(text);state.update(status='complete',selected=chosen);write_json(status,state)
  write_json(a.output/'COMPLETE.json',{'status':'complete','report':str(a.output/'report.md'),'capture':str(used)})
  print(text,flush=True);print('===== STAGE_EXEC_SWEEP_ALL_STAGES_PASSED =====',flush=True)
 except BaseException as exc:record_failure(status,state,exc);raise
if __name__=='__main__':main()
