#!/usr/bin/env python3
"""Plan-only refinement factorial: build -> strict gate -> screen -> independent confirmation."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import joint_plan_common as common
from joint_variants import REFINE_NAMES,REFINE_REFS,description
from harness_io import preflight_args,write_json,record_failure
from run_union_next_sweep import read_complete,validate_confirmation,validate_cold_capture,workload_key
DEFAULT_NAMES=('abcdf_pages','joint_integrated_bounds_512',*REFINE_NAMES)


def select_candidates(report,names,top):
    if top<=0:raise ValueError('positive top required')
    selected=set(REFINE_REFS);rankings=[]
    candidates=[n for n in names if n in REFINE_NAMES and n not in REFINE_REFS]
    if not candidates:raise ValueError('need at least one optimization candidate')
    for run in report['runs']:
        ranked=sorted(candidates,key=lambda n:(run['stats'][n]['p50_us'],n));chosen=ranked[:top]
        selected.update(chosen);rankings.append({'workload':workload_key(run),'chosen':chosen,'ranking':ranked})
    return [n for n in names if n in selected],rankings


def summary_text(path,report,screen_path=None):
    lines=['# bounds512 后续 A/B/C/D：独立正式计时','',
      '本表仅用正式采集样本；初筛样本只用于入围。所有比值为当前实现P50/参考P50，小于1表示更快。',
      'ABCDF、冻结bounds512、新control均保留；control用于观察代码组织及编译资源变化。',
      'SIMT固定512线程，原逐行回退1024线程。D在UB上替换完整Phase3；B仍作用于Phase5布尔Scan，GM上仍使用SIMT Phase3。','',
      '| 工况/MTP | 实现 | P50/P95 μs | 样本数 | 实现/ABCDF | 实现/bounds512 | 实现/control | 共享/回退 | 单行有序组 |',
      '|---|---|---|---:|---:|---:|---:|---|---:|']
    for run in report['runs']:
        w=run['workload'];stats=run['stats']
        for n,s in stats.items():
            c=s['correctness'];ratios=[s['p50_us']/stats[r]['p50_us'] for r in REFINE_REFS]
            lines.append(f"| {w['scenario']}/MTP{w['mtp']} | {n} | {s['p50_us']:.3f}/{s['p95_us']:.3f} | {s['count']} | "+' | '.join(f'{v:.3f}' for v in ratios)+f" | {c.get('joint_groups',0)}/{c.get('overflow_groups',0)} | {c.get('single_sorted_groups',0)} |")
    lines+=['','正式metadata：'+str(path),'','原始CSV与样本索引：'+str(Path(path).parent/'data_index.json')]
    if screen_path:lines+=['','初筛metadata：'+str(screen_path)]
    lines+=['','完整输入指纹、实际miss/descriptor数量、各实现相对自身MTP0、设备/库/编译身份与原始样本见正式metadata及采集子目录报告。',
      '','各输入独立判断；不能把逐工况选择的赢家当成同一个固定实现。不承诺所有组合更快。',
      '','A只作用于严格递增单行；C只作用于非reset的UB路径；A命中时不执行D，B/D在UB Phase3不重复执行。完整原有输出尾部、LRU和回退契约均保留。',
      '','固定输入与地址预热；逐样本恢复/校验不计入目标kernel。只测Plan，不含Transfer/SFA和接口端到端时间。']
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__);common.add_runtime_arguments(p);p.set_defaults(device=0,temp_dir=None)
    p.add_argument('--output',type=Path,default=Path('/workspace')/('refine512_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid())))
    p.add_argument('--implementations',nargs='+',choices=DEFAULT_NAMES,default=list(DEFAULT_NAMES))
    p.add_argument('--scenarios',nargs='+',choices=('cold_reset','full_cache_75pct'),default=['cold_reset','full_cache_75pct'])
    p.add_argument('--screen-repeat',type=int,default=5);p.add_argument('--confirm-repeat',type=int,default=50)
    p.add_argument('--top-per-workload',type=int,default=2);p.add_argument('--regression-repeat',type=int,default=5)
    p.add_argument('--full-confirm',action='store_true');p.add_argument('--skip-build',action='store_true');p.add_argument('--describe-only',action='store_true');a=p.parse_args()
    if min(a.screen_repeat,a.confirm_repeat,a.top_per_workload,a.regression_repeat)<=0:p.error('positive counts required')
    if len(a.implementations)!=len(set(a.implementations)) or not set(REFINE_REFS)<=set(a.implementations):p.error('distinct implementations and all three references required')
    if not any(n in REFINE_NAMES and n not in REFINE_REFS for n in a.implementations):p.error('select at least one optimization candidate')
    if len(a.scenarios)!=len(set(a.scenarios)):p.error('duplicate scenarios')
    if a.describe_only:
        print(json.dumps({'implementations':a.implementations,'count':len(a.implementations),'build_mode':'--refine-only',
            'mtp':[0,1,3],'scenarios':a.scenarios,'screen_repeat':None if a.full_confirm else a.screen_repeat,
            'confirm_repeat':a.confirm_repeat,'confirm_warmup':10,'regression_repeat':a.regression_repeat,
            'selection':'all selected; no screen' if a.full_confirm else 'top per scenario/MTP, then independent confirmation',
            'variants':{n:description(n) for n in a.implementations}},ensure_ascii=False,indent=2));return
    a.output=a.output.expanduser().resolve();a.library=a.library.expanduser().resolve()
    if a.output.exists():raise RuntimeError('refuse overwrite: '+str(a.output))
    a.temp_dir=(a.temp_dir or a.output/'tmp').expanduser().resolve()
    storage=preflight_args(a,a.output);status=a.output/'RUN_STATUS.json';state={'status':'running','storage':storage}
    os.chdir(ROOT);os.environ['PYTHONDONTWRITEBYTECODE']='1'
    def run(cmd,label):
        log=a.output/(label+'.log');state.update(phase=label,log=str(log));write_json(status,state)
        print('开始:',label,'详细日志:',log,flush=True)
        with log.open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,text=True)
        if r.returncode:
            lines=log.read_text(errors='replace').splitlines()
            first=next((v for v in lines if any(k in v for k in ('Traceback','error:','Error:','fatal error'))),'see full log')
            print('首个异常:',first,'\n日志末尾:\n'+'\n'.join(lines[-25:]),flush=True)
            raise RuntimeError(f'{label} failed rc={r.returncode}; log={log}; first={first}')
    try:
        if not a.skip_build:run(['bash','build_joint_plan.sh','--soc=ascend950','--refine-only'],'build')
        runtime=['--device',str(a.device),'--library',str(a.library),'--temp-dir',str(a.temp_dir),
          '--min-output-free-bytes',str(a.min_output_free_bytes),'--min-temp-free-bytes',str(a.min_temp_free_bytes),'--min-free-inodes',str(a.min_free_inodes)]
        gate=a.output/'gate.json'
        run([sys.executable,'-B','-X','faulthandler','checks/check_refine_npu.py',*runtime,'--implementations',*a.implementations,
          '--regression-repeat',str(a.regression_repeat),'--report',str(gate)],'gate')
        def profile(names,dest,repeat,warmup,label):
            run([sys.executable,'-B','-X','faulthandler','profile_joint_mtp.py',*runtime,'--implementations',*names,
              '--sorted-topk','--requests','32','--mtp','0','1','3','--topk','2048','--capacity','4096','--max-token','262144',
              '--block-size','64','--seed','2026','--scenarios',*a.scenarios,'--overlaps','overlap40','--warmup',str(warmup),
              '--repeat',str(repeat),'--regression-repeat',str(a.regression_repeat),'--correctness-report',str(gate),'--output',str(dest)],label)
            path,report=read_complete(dest,names,repeat);validate_cold_capture(report,names,a.scenarios);return path,report
        sp=None;sr=None
        if a.full_confirm:selected=a.implementations
        else:
            sp,sr=profile(a.implementations,a.output/'screen',a.screen_repeat,2,'screen')
            selected,ranking=select_candidates(sr,a.implementations,a.top_per_workload)
            write_json(a.output/'selection.json',{'screen_metadata':str(sp),'selected':selected,'ranking':ranking,
                'criterion':'screening only; union of top candidates per workload + fixed references'})
        cp,cr=profile(selected,a.output/'confirm',a.confirm_repeat,10,'confirm')
        if sr is not None:validate_confirmation(sr,cr)
        text=summary_text(cp,cr,sp);(a.output/'report.md').write_text(text)
        state.update(status='complete',selected=selected,screen=str(sp) if sp else None,confirm=str(cp));write_json(status,state)
        write_json(a.output/'COMPLETE.json',{'status':'complete','report':str(a.output/'report.md')})
        print(text,flush=True);print('===== REFINE_SWEEP_ALL_STAGES_PASSED =====',flush=True)
    except BaseException as exc:record_failure(status,state,exc);raise
if __name__=='__main__':main()
