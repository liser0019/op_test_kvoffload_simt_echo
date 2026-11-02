#!/usr/bin/env python3
"""One server invocation: build -> strict gate -> factorial screen -> independent confirmation."""
from pathlib import Path
import argparse,json,math,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from joint_variants import HW_NAMES
import joint_plan_common as common
from harness_io import preflight_args,write_json,record_failure
REFS=('abcdf_pages','joint_sorted_mrgsort')


def workload_key(run):
    w=run['workload']
    return (run.get('comparison','same_requests'),w['scenario'],w['overlap'],w['mtp'])


def read_complete(directory,expected_names=None,expected_count=None):
    paths=list(Path(directory).glob('joint_matrix_*/capture_metadata.json'))
    if len(paths)!=1:raise RuntimeError(f'exactly one capture expected: {directory}, got {paths}')
    path=paths[0];r=json.loads(path.read_text())
    if r.get('status')!='complete' or not (path.parent/'COMPLETE.json').is_file():raise RuntimeError(f'incomplete capture: {path}')
    for run in r['runs']:
        if expected_names is not None and set(run['stats'])!=set(expected_names):raise RuntimeError('implementation set mismatch')
        for n,s in run['stats'].items():
            if (expected_count is not None and s['count']!=expected_count) or len(s['samples_us'])!=s['count']:raise RuntimeError(f'sample count mismatch: {n}')
            if s['correctness']['status']!='passed' or not math.isfinite(s['p50_us']) or s['p50_us']<=0:raise RuntimeError(f'invalid result: {n}')
    return path,r


def select_candidates(report,top):
    if top<=0:raise ValueError('top must be positive')
    keep=set(REFS);rankings=[]
    for run in report['runs']:
        if run['workload']['mtp']==0:continue # no union path, never rank from this control
        candidates=sorted((n for n in run['stats'] if n in HW_NAMES),key=lambda n:(run['stats'][n]['p50_us'],n))
        if not candidates:raise ValueError('no candidates in nonzero MTP workload')
        chosen=candidates[:top];keep.update(chosen)
        rankings.append({'workload':workload_key(run),'chosen':chosen,'ranking':[
            {'name':n,'p50_us':run['stats'][n]['p50_us']} for n in candidates]})
    return [*REFS,*(n for n in HW_NAMES if n in keep)],rankings


def validate_confirmation(screen,confirm):
    old={workload_key(r):r for r in screen['runs']};seen=set()
    if len(old)!=len(screen['runs']):raise RuntimeError('duplicate screen workload')
    for run in confirm['runs']:
        key=workload_key(run)
        if key in seen:raise RuntimeError('duplicate confirmation workload')
        seen.add(key)
        if key not in old or run['workload']['logical_input_sha256']!=old[key]['workload']['logical_input_sha256']:raise RuntimeError('confirmation logical input changed')
    if seen!=set(old):raise RuntimeError('confirmation workload set changed')
    for key in ('device','device_name','hostname','library_sha256','identity'):
        if screen['environment'][key]!=confirm['environment'][key]:raise RuntimeError('confirmation environment/library changed: '+key)


def summary_text(screen_path,confirm_path,screen,confirm,selected):
    lines=['# 硬件归并五因素：独立复测结果','',
        '下表仅使用复测样本；初筛用于入围，不能与复测样本混合统计。MTP0不参与入围排名。',
        '参考硬件版是已验证的joint_sorted_mrgsort；小于1表示候选更快。','',
        '| 工况/MTP | 实现 | P50/P95 μs | 样本数 | 候选/ABCDF | 候选/参考硬件 | 共享/回退 |',
        '|---|---|---|---:|---:|---:|---|']
    for run in confirm['runs']:
        w=run['workload'];base=run['stats'][REFS[0]]['p50_us'];hardware=run['stats'][REFS[1]]['p50_us']
        for n,s in run['stats'].items():
            c=s['correctness'];lines.append(f"| {w['scenario']}/MTP{w['mtp']} | {n} | {s['p50_us']:.3f}/{s['p95_us']:.3f} | {s['count']} | {s['p50_us']/base:.3f} | {s['p50_us']/hardware:.3f} | {c.get('joint_groups',0)}/{c.get('overflow_groups',0)} |")
    lines+=['','初筛：'+str(screen_path),'','复测：'+str(confirm_path),'','入围：'+', '.join(selected),
        '','全部原始CSV路径、样本及源码/库身份分别保留在两份采集目录的data_index.json/capture_metadata.json。',
        '','每种输入单独观察P50、P95和回退；不将不同工况临时挑选出的赢家说成同一固定版本。未测Transfer/SFA。']
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__);common.add_runtime_arguments(p);p.set_defaults(device=0,temp_dir=None)
    p.add_argument('--output',type=Path,default=Path('/workspace')/('hardware_sweep_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid())))
    p.add_argument('--implementations',nargs='+',choices=(*REFS,*HW_NAMES),default=[*REFS,*HW_NAMES])
    p.add_argument('--scenarios',nargs='+',choices=('cold_reset','full_cache_75pct','full_cache_90pct','full_cache_stable_hit'),default=['full_cache_75pct'])
    p.add_argument('--screen-repeat',type=int,default=5);p.add_argument('--confirm-repeat',type=int,default=50)
    p.add_argument('--top-per-mtp',type=int,default=2);p.add_argument('--skip-build',action='store_true');p.add_argument('--describe-only',action='store_true')
    a=p.parse_args()
    if min(a.screen_repeat,a.confirm_repeat,a.top_per_mtp)<=0 or len(a.implementations)!=len(set(a.implementations)) or not set(REFS)<=set(a.implementations) or not any(n in HW_NAMES for n in a.implementations):p.error('positive counts, distinct names, both references and at least one candidate required')
    if len(a.scenarios)!=len(set(a.scenarios)):p.error('duplicate scenarios')
    if a.describe_only:
        print(json.dumps({'implementations':a.implementations,'count':len(a.implementations),'mtp':[0,1,3],
            'scenarios':a.scenarios,'screen_repeat':a.screen_repeat,'confirm_repeat':a.confirm_repeat,
            'selection':'top per nonzero-MTP workload, independent confirmation; MTP0 excluded'},ensure_ascii=False,indent=2));return
    a.output=a.output.expanduser().resolve();a.library=a.library.expanduser().resolve()
    if a.temp_dir is not None:a.temp_dir=a.temp_dir.expanduser().resolve()
    if a.output.exists():raise RuntimeError('refuse overwrite: '+str(a.output))
    if a.temp_dir is None:a.temp_dir=a.output/'tmp'
    required=['checks/check_merge_abcd_npu.py','profile_joint_mtp.py','build_joint_plan.sh']
    for n in required:
        if not (ROOT/n).is_file():raise RuntimeError('missing entry: '+n)
    storage=preflight_args(a,a.output);status=a.output/'RUN_STATUS.json';state={'status':'running','storage':storage}
    os.chdir(ROOT);os.environ['PYTHONDONTWRITEBYTECODE']='1'
    def run(cmd,label):
        log=a.output/(label+'.log');state.update(phase=label,log=str(log));write_json(status,state)
        print('开始:',label,'详细日志:',log,flush=True)
        with log.open('w') as f:
            result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,text=True)
        if result.returncode:
            lines=log.read_text(errors='replace').splitlines()
            first=next((v for v in lines if 'Traceback' in v or 'error:' in v or 'Error:' in v or 'fatal error' in v),'see full log')
            print('首个异常:',first,'\n日志末尾:\n'+'\n'.join(lines[-25:]),flush=True)
            raise RuntimeError(f'{label} failed rc={result.returncode}; log={log}; first={first}')
    try:
        if not a.skip_build:run(['bash','build_joint_plan.sh','--soc=ascend950','--sweep-only'],'build')
        runtime=['--device',str(a.device),'--library',str(a.library.resolve()),'--temp-dir',str(a.temp_dir.resolve()),
                 '--min-output-free-bytes',str(a.min_output_free_bytes),'--min-temp-free-bytes',str(a.min_temp_free_bytes),
                 '--min-free-inodes',str(a.min_free_inodes)]
        gate=a.output/'gate.json'
        run([sys.executable,'-B','-X','faulthandler','checks/check_merge_abcd_npu.py',*runtime,'--implementations',*a.implementations,'--regression-repeat','5','--report',str(gate)],'gate')
        def profile(names,destination,repeat,warmup,label):
            run([sys.executable,'-B','-X','faulthandler','profile_joint_mtp.py',*runtime,'--implementations',*names,
                '--sorted-topk','--requests','32','--mtp','0','1','3','--topk','2048','--capacity','4096','--max-token','262144','--block-size','64','--seed','2026',
                '--scenarios',*a.scenarios,'--overlaps','overlap40','--warmup',str(warmup),'--repeat',str(repeat),
                '--regression-repeat','5','--correctness-report',str(gate),'--output',str(destination)],label)
            return read_complete(destination,names,repeat)
        sp,sr=profile(a.implementations,a.output/'screen',a.screen_repeat,2,'screen')
        selected,ranking=select_candidates(sr,a.top_per_mtp)
        write_json(a.output/'selection.json',{'screen_metadata':str(sp),'selected':selected,'ranking':ranking,
            'criterion':'lowest candidate P50 per nonzero-MTP workload; screening only, not final winner'})
        cp,cr=profile(selected,a.output/'confirm',a.confirm_repeat,10,'confirm')
        validate_confirmation(sr,cr)
        text=summary_text(sp,cp,sr,cr,selected);(a.output/'report.md').write_text(text)
        state.update(status='complete',selected=selected,screen=str(sp),confirm=str(cp));write_json(status,state)
        write_json(a.output/'COMPLETE.json',{'status':'complete','report':str(a.output/'report.md')})
        print(text,flush=True);print('整轮完成:',a.output,flush=True)
    except BaseException as e:record_failure(status,state,e);raise
if __name__=='__main__':main()
