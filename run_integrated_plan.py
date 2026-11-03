#!/usr/bin/env python3
"""四路固定对照：ABCDF、两条冻结冷优化参考、一个512线程整合候选；仅Plan。"""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from joint_variants import INTEGRATED_COMPARE_NAMES,description
import joint_plan_common as common
from harness_io import preflight_args,write_json,record_failure
from run_union_next_sweep import read_complete,validate_cold_capture
NAMES=INTEGRATED_COMPARE_NAMES


def summary_text(path,report):
    lines=['# 512线程单行/联合整合：四路固定对照','',
      '所有实现均正式采集，不初筛、不拼接不同实现的赢家。仅设备目标kernel Duration，单位μs。',
      '整合版实际按连续请求组行数选择算法；输入先共同排序，保留重复token和因果可见性。',
      '单行使用private协议；多行使用joint协议；不支持时回退原1024线程逐行实现。',
      '两条冻结参考的MTP0执行路径不同；AC冷参考仍走原1024线程逐行路径。','',
      '| 工况/MTP | 实现 | P50/P95 μs | 样本数 | 实现/ABCDF | 实现/单行冷参考 | 实现/联合冷参考 | 共享/回退 | 单行有序组 |',
      '|---|---|---|---:|---:|---:|---:|---|---:|']
    for run in report['runs']:
        w=run['workload'];stats=run['stats']
        for name in NAMES:
            s=stats[name];c=s['correctness']
            ratios=[s['p50_us']/stats[n]['p50_us'] for n in NAMES[:3]]
            lines.append(f"| {w['scenario']}/MTP{w['mtp']} | {name} | {s['p50_us']:.3f}/{s['p95_us']:.3f} | {s['count']} | "+' | '.join(f'{r:.3f}' for r in ratios)+f" | {c.get('joint_groups',0)}/{c.get('overflow_groups',0)} | {c.get('single_sorted_groups',0)} |")
    lines+=['','比值方向为左侧实现P50/参考P50，小于1表示更快。整合后可能改变编译资源，不能预先保证继承各参考的最优耗时。',
      '','采集前恢复初态，逐样本完整校验；固定地址预热，不代表生产访问轨迹或冷硬件缓存。',
      '','同一实现相对自身MTP0、实际miss/descriptor数、原始样本、输入/设备/编译/库身份见metadata及采集子目录报告。',
      '','原始CSV索引：'+str(Path(path).parent/'data_index.json'),'', '采集metadata：'+str(path),
      '','仅Plan，不包含Transfer/SFA或接口端到端时间。只有退出码0、COMPLETE.json及报告齐全才算整轮成功。']
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__);common.add_runtime_arguments(p);p.set_defaults(device=0,temp_dir=None)
    p.add_argument('--output',type=Path,default=Path('/workspace')/('integrated512_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid())))
    p.add_argument('--scenarios',nargs='+',choices=('cold_reset','full_cache_75pct'),default=['cold_reset','full_cache_75pct'])
    p.add_argument('--repeat',type=int,default=50);p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--regression-repeat',type=int,default=5)
    p.add_argument('--skip-build',action='store_true');p.add_argument('--describe-only',action='store_true');a=p.parse_args()
    if a.repeat<=0 or a.regression_repeat<=0 or a.warmup<0 or len(a.scenarios)!=len(set(a.scenarios)):p.error('positive repeat/regression-repeat, nonnegative warmup, distinct scenarios')
    if a.describe_only:
        print(json.dumps({'implementations':NAMES,'build_mode':'--integrated-only','mtp':[0,1,3],
            'scenarios':a.scenarios,'repeat':a.repeat,'warmup':a.warmup,'regression_repeat':a.regression_repeat,
            'selection':'fixed implementations; no screening','variants':{n:description(n) for n in NAMES}},ensure_ascii=False,indent=2));return
    a.output=a.output.expanduser().resolve();a.library=a.library.expanduser().resolve()
    if a.output.exists():raise RuntimeError('refuse overwrite: '+str(a.output))
    a.temp_dir=(a.temp_dir or a.output/'tmp').expanduser().resolve()
    storage=preflight_args(a,a.output);status=a.output/'RUN_STATUS.json';state={'status':'running','storage':storage}
    os.chdir(ROOT);os.environ['PYTHONDONTWRITEBYTECODE']='1'
    def run(command,label):
        log=a.output/(label+'.log');state.update(phase=label,log=str(log));write_json(status,state)
        print('开始:',label,'详细日志:',log,flush=True)
        with log.open('w') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,text=True)
        if result.returncode:
            lines=log.read_text(errors='replace').splitlines()
            first=next((v for v in lines if any(k in v for k in ('Traceback','error:','Error:','fatal error'))),'see full log')
            print('首个异常:',first,'\n日志末尾:\n'+'\n'.join(lines[-25:]),flush=True)
            raise RuntimeError(f'{label} failed rc={result.returncode}; log={log}; first={first}')
    try:
        if not a.skip_build:run(['bash','build_joint_plan.sh','--soc=ascend950','--integrated-only'],'build')
        runtime=['--device',str(a.device),'--library',str(a.library),'--temp-dir',str(a.temp_dir),
            '--min-output-free-bytes',str(a.min_output_free_bytes),'--min-temp-free-bytes',str(a.min_temp_free_bytes),'--min-free-inodes',str(a.min_free_inodes)]
        gate=a.output/'gate.json'
        run([sys.executable,'-B','-X','faulthandler','checks/check_cold_finish_npu.py',*runtime,
            '--implementations',*NAMES,'--regression-repeat',str(a.regression_repeat),'--report',str(gate)],'gate')
        run([sys.executable,'-B','-X','faulthandler','profile_joint_mtp.py',*runtime,'--implementations',*NAMES,
            '--sorted-topk','--requests','32','--mtp','0','1','3','--topk','2048','--capacity','4096','--max-token','262144',
            '--block-size','64','--seed','2026','--scenarios',*a.scenarios,'--overlaps','overlap40',
            '--warmup',str(a.warmup),'--repeat',str(a.repeat),'--regression-repeat',str(a.regression_repeat),
            '--correctness-report',str(gate),'--output',str(a.output/'profile')],'profile')
        path,report=read_complete(a.output/'profile',NAMES,a.repeat)
        validate_cold_capture(report,NAMES,a.scenarios)
        text=summary_text(path,report);(a.output/'report.md').write_text(text)
        state.update(status='complete',implementations=NAMES,metadata=str(path));write_json(status,state)
        write_json(a.output/'COMPLETE.json',{'status':'complete','report':str(a.output/'report.md')})
        print(text,flush=True);print('===== INTEGRATED_PLAN_ALL_STAGES_PASSED =====',flush=True)
    except BaseException as exc:record_failure(status,state,exc);raise
if __name__=='__main__':main()
