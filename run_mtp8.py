#!/usr/bin/env python3
"""小范围MTP0/1/3/7短流程：构建→一次专项门禁→一次profiling，无初筛/复测叠加。"""
from pathlib import Path
from types import SimpleNamespace
import argparse,json,os,subprocess,sys,time
import numpy as np
import joint_plan_common as common
from joint_plan_cases import make_case,sort_topk,describe
from checks.mtp8_cases import cases,transitions
from checks.check_joint_plan import context_gate
from harness_io import preflight_args,write_json,record_failure,diagnostic_log,capture_error
from fixed_plan_check import exact_workload_key
from profile_joint_mtp import capture,mtp_comparison
from joint_variants import MTP7_PACK_NAMES,MTP7_PREFIX_NAMES,MTP7_MERGE_NAMES
ROOT=Path(__file__).resolve().parent
NAMES=('abcdf_pages','joint_hot_bd_512','joint_hot_bd8_512')
FIX_NAMES=('abcdf_pages','joint_hot_bd8_512','joint_mtp7_parallel_512','joint_mtp7_rank_512')
PACK_NAMES=(*FIX_NAMES,*MTP7_PACK_NAMES)
PEER_NAMES=('abcdf_pages','joint_mtp7_reuse_parallel_512','peer_qli_plan')
PREFIX_NAMES=('joint_mtp7_reuse_parallel_512',*MTP7_PREFIX_NAMES)
MERGE_UNIQUE_NAMES=('joint_mtp7_reuse_parallel_512',*MTP7_MERGE_NAMES)


def workload_cases(a):
    return [sort_topk(make_case(a.requests,a.topk,a.capacity,a.max_token,m,s,'overlap40',2026,64))
            for s in a.scenarios for m in a.mtp]


def gate(torch,a,workloads,environment,destination):
    records=[];evidence={n:{} for n in a.implementations};regression={}
    failed=destination/'failure_state'
    def fixed(label,c,repeats):
        inputs=common.upload_inputs(torch,c)
        for n in a.implementations:
            b=common.prepare(torch,c,n,inputs);b['fast_fixed_check']=True
            try:
                r=common.audit(torch,c,b,repeats,failed)
                records.append({'case':label,'implementation':n,'audit':r,'layout':b['layout'],'check_cost':b.get('check_timing')})
                evidence[n][exact_workload_key(c)]='passed'
                if label=='known_input_write':regression[n]=r
            finally:common.close(b)
        print('专项门禁:',label,'通过',flush=True)
    fixed('known_input_write',sort_topk(make_case(a.requests,2048,4096,262144,1,'full_cache_75pct','shared')),a.regression_repeat)
    for label,c in cases(include_extreme=a.suite in ('mtp7_fix','mtp7_pack','peer','prefix','merge_unique'),include_prefix=a.suite in ('prefix','merge_unique')):fixed(label,c,1)
    continuous=[]
    for n in a.implementations:
        start=next(transitions())[1];b=common.prepare(torch,start,n)
        try:
            for label,c in transitions():
                common.update_inputs(torch,b,c);common.poison(b);torch.npu.synchronize()
                before=common.snapshot(b);b['phase']=label
                try:
                    common.invoke(b);torch.npu.synchronize();r=common.check(c,b,before)
                except BaseException:common.dump_failure(c,b,failed);raise
                continuous.append({'case':label,'implementation':n,'audit':r})
        finally:common.close(b)
    for c in workloads:fixed(f"performance/{c['scenario']}/mtp{c['mtp']}",c,a.regression_repeat)
    result={'status':'passed','environment':environment,'gate':{'status':'passed','implementations':list(a.implementations),
        'validation_scope':'mtp8_v1: exact8 group + packet seams + duplicate/causality + capacity fallback +10-step actual continuations',
        'input_write_regression':{'status':'passed','checked_iterations':a.regression_repeat,'per_implementation':regression,'input_restore':False},
        'cases':records,'continuous':continuous,'context':context_gate(torch,a.implementations)},'exact_workloads':evidence}
    write_json(destination/'gate.json',result)
    return result


def prefix_report_text(report):
    base = "joint_mtp7_reuse_parallel_512"
    lines = ["# MTP7安全前缀与16位映射四路对照", "",
             "仅设备Plan kernel Duration，单位μs。A/B/AB基于冻结reuse_parallel，512线程、相同生产动态UB预算。",
             "A提前确定安全消费前缀；B仅为8行组使用16位映射并扩大归并包。", "",
             "| 工况/MTP | 实现 | P50/P95 μs | 样本数 | 实现/冻结基线 | 耗时下降 | 共享/回退 | descriptor数 | Block Num |",
             "|---|---|---|---:|---:|---:|---|---:|---|"]
    for run in report["runs"]:
        workload = run["workload"]
        reference = run["stats"][base]["p50_us"]
        for name, stats in run["stats"].items():
            check = stats["correctness"]
            ratio = stats["p50_us"] / reference
            blocks = ",".join(stats.get("block_nums", [])) or "CSV未导出"
            lines.append(f"| {workload['scenario']}/MTP{workload['mtp']} | {name} | "
                         f"{stats['p50_us']:.3f}/{stats['p95_us']:.3f} | {stats['count']} | "
                         f"{ratio:.3f} | {(1-ratio)*100:.1f}% | "
                         f"{check.get('joint_groups',0)}/{check.get('overflow_groups',0)} | "
                         f"{check['total_transfer_descriptors']} | {blocks} |")
    lines += ["", "正的耗时下降表示候选更快。所有实现使用共同升序输入，排序在计时外。",
              f"每实现每工况{report['repeat']}个样本，初态恢复和逐样本完整校验保留。",
              "实际请求/行、同实现相对MTP0比值、输入/源码/库身份、原始CSV与墙钟统计见capture_metadata.json/data_index.json。"]
    return "\n".join(lines) + "\n"


def merge_unique_report_text(report):
    text=prefix_report_text(report)
    text=text.replace("MTP7安全前缀与16位映射四路对照","MTP7中间去重／融合消费统计三路对照")
    text=text.replace("A/B/AB基于冻结reuse_parallel","两条候选基于冻结reuse_parallel")
    text=text.replace("A提前确定安全消费前缀；B仅为8行组使用16位映射并扩大归并包。",
        "MID：四路前缀先去重，再做最终两路硬件归并；FUSED：原归并记录的消费统计合入SIMD去重。均只改8行两级归并路径，保留32位映射和612条/路包长。")
    return text


def report_text(report):
    if all(name in report['implementations'] for name in MTP7_MERGE_NAMES):
        return merge_unique_report_text(report)
    if all(name in report['implementations'] for name in MTP7_PREFIX_NAMES):
        return prefix_report_text(report)

    reference='joint_mtp7_reuse_parallel_512' if 'peer_qli_plan' in report['implementations'] else 'joint_hot_bd_512' if 'joint_hot_bd_512' in report['implementations'] else 'joint_hot_bd8_512'
    reference_label='当前reuse_parallel' if reference=='joint_mtp7_reuse_parallel_512' else '旧BD' if reference=='joint_hot_bd_512' else '上轮BD8'
    expanded=all(n in report['implementations'] for n in MTP7_PACK_NAMES)
    extra_header=' 实现/A | 实现/B |' if expanded else ''
    extra_separator='---:|---:|' if expanded else ''
    lines=['# MTP0/1/3/7 Plan对照','',
      '仅设备目标kernel Duration，单位μs；不含恢复、CPU参考、校验、Transfer/SFA。固定地址预热。',
      ('同事独立Plan保留上游UB/GM布局，具体路径见metadata。' if reference=='joint_mtp7_reuse_parallel_512' else '旧BD在MTP7是逐行回退参考。' if reference=='joint_hot_bd_512' else '上轮BD8与本轮联合候选使用相同生产动态UB预算。')+'联合实现需结合共享/回退计数判断实际路径。比值为当前实现P50/参考，小于1更快。','',
      f'| 工况/MTP | 实现 | P50/P95 μs | 样本数 | 实现/ABCDF | 实现/{reference_label} |'+extra_header+' 共享/回退 | 实际descriptor数 | Block Num |',
      '|---|---|---|---:|---:|---:|'+extra_separator+'---|---:|---|']
    for r in report['runs']:
        w=r['workload'];stats=r['stats']
        for n,s in stats.items():
            c=s['correctness']
            extra=(f" {s['p50_us']/stats['joint_mtp7_parallel_512']['p50_us']:.3f} | {s['p50_us']/stats['joint_mtp7_rank_512']['p50_us']:.3f} |" if expanded else '')
            lines.append(f"| {w['scenario']}/MTP{w['mtp']} | {n} | {s['p50_us']:.3f}/{s['p95_us']:.3f} | {s['count']} | {s['p50_us']/stats['abcdf_pages']['p50_us']:.3f} | {s['p50_us']/stats[reference]['p50_us']:.3f} |"+extra+f" {c.get('joint_groups',0)}/{c.get('overflow_groups',0)} | {c['total_transfer_descriptors']} | {','.join(s.get('block_nums', [])) or 'CSV未导出'} |")
    lines+=['','## 各实现相对自身MTP0','',
        '| 工况 | 实现 | MTP | 实际请求/行 | 自身MTP0请求/行 | P50比值 |',
        '|---|---|---:|---|---|---:|']
    for c in report['mtp_comparison']:
        lines.append(f"| {c['scenario']} | {c['implementation']} | {c['mtp']} | {c['on_requests']}/{c['on_rows']} | {c['off_requests']}/{c['off_rows']} | {c['on_over_off_p50']:.3f} |")
    lines+=['','逐样本严格校验。输入指纹、实际miss/共享/回退、源码/设备/库身份、原始样本、CSV路径与各阶段墙钟见capture_metadata.json/data_index.json。',
        f"本轮每实现每工况{report['repeat']}个正式样本；5样本是探索结果，不与后续50样本混合统计。"]
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__);common.add_runtime_arguments(p);p.set_defaults(device=0,temp_dir=None)
    p.add_argument('--suite',choices=('mtp8','mtp7_fix','mtp7_pack','peer','prefix','merge_unique'),default='mtp8')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--skip-build',action='store_true')
    p.add_argument('--describe-only',action='store_true');p.add_argument('--formal',action='store_true')
    p.add_argument('--mtp',nargs='+',type=int,choices=(0,1,3,7),default=[0,1,3,7])
    p.add_argument('--scenarios',nargs='+',choices=('cold_reset','full_cache_75pct'),default=['cold_reset','full_cache_75pct'])
    p.add_argument('--requests',type=int,default=28);p.add_argument('--topk',type=int,default=2048)
    p.add_argument('--capacity',type=int,default=4096);p.add_argument('--max-token',type=int,default=262144)
    p.add_argument('--warmup',type=int);p.add_argument('--repeat',type=int);p.add_argument('--regression-repeat',type=int,default=5)
    p.add_argument('--correctness-report',type=Path,help='仅复用同源码/库/设备、同输入且包含本轮专项门禁的报告')
    a=p.parse_args();a.warmup=(10 if a.formal else 2) if a.warmup is None else a.warmup
    a.repeat=(50 if a.formal else 5) if a.repeat is None else a.repeat
    if a.warmup<0 or min(a.repeat,a.regression_repeat,a.requests)<=0 or 0 not in a.mtp:p.error('positive counts, warmup>=0, MTP list must include0')
    if len(set(a.mtp))!=len(a.mtp) or len(set(a.scenarios))!=len(a.scenarios):p.error('duplicate scenarios/MTP')
    a.implementations=MERGE_UNIQUE_NAMES if a.suite=='merge_unique' else PREFIX_NAMES if a.suite=='prefix' else PEER_NAMES if a.suite=='peer' else PACK_NAMES if a.suite=='mtp7_pack' else FIX_NAMES if a.suite=='mtp7_fix' else NAMES
    a.build_flag='--merge-unique-only' if a.suite=='merge_unique' else '--prefix-only' if a.suite=='prefix' else '--peer-only' if a.suite=='peer' else '--mtp7-pack-only' if a.suite=='mtp7_pack' else '--mtp7-fix-only' if a.suite=='mtp7_fix' else '--mtp8-only'
    workloads=workload_cases(a)
    if a.describe_only:
        print(json.dumps({'implementations':a.implementations,'mtp':a.mtp,'workloads':[describe(c) for c in workloads],
            'warmup':a.warmup,'repeat':a.repeat,'regression_repeat':a.regression_repeat,'build':a.build_flag,
            'target_active_samples':len(workloads)*len(a.implementations)*a.repeat,'hardware_status':'not locally verified'},ensure_ascii=False,indent=2));return
    a.output=a.output.expanduser().resolve();a.library=a.library.expanduser().resolve()
    if a.output.exists():raise RuntimeError('refuse overwrite: '+str(a.output))
    a.temp_dir=(a.temp_dir or a.output/'tmp').expanduser().resolve()
    storage=preflight_args(a,a.output);status=a.output/'RUN_STATUS.json';log=a.output/'launcher.log'
    state={'status':'running','storage':storage,'wall_seconds':{},'implementations':a.implementations,'workload_count':len(workloads)}
    os.chdir(ROOT);os.environ['PYTHONDONTWRITEBYTECODE']='1'
    with diagnostic_log(log):
      try:
        if not a.skip_build:
            state['phase']='build';write_json(status,state);t=time.monotonic()
            with (a.output/'build.log').open('w') as stream:
                subprocess.run(['bash','build_joint_plan.sh','--soc=ascend950',a.build_flag],stdout=stream,stderr=subprocess.STDOUT,check=True)
            state['wall_seconds']['build']=time.monotonic()-t
        torch,npu,env=common.load_runtime(a)
        state['phase']='gate';write_json(status,state);t=time.monotonic()
        if a.correctness_report:
            from checks.check_joint_plan import validate_gate_report
            validate_gate_report(a.correctness_report,env,a.regression_repeat,a.implementations)
            g=json.loads(a.correctness_report.read_text())
            if not g['gate'].get('validation_scope','').startswith('mtp8_v1:'):raise RuntimeError('not an MTP8 targeted gate')
            if any(g.get('exact_workloads',{}).get(n,{}).get(exact_workload_key(c))!='passed' for c in workloads for n in a.implementations):raise RuntimeError('gate lacks exact workload inputs')
            write_json(a.output/'gate.json',g)
        else:g=gate(torch,a,workloads,env,a.output)
        state['wall_seconds']['gate']=time.monotonic()-t
        a.fast_fixed_check=True;a.reuse_workload_audit=True;a.workload_gate=g['exact_workloads']
        report={'status':'running','environment':env,'storage':storage,'runs':[],'mtp_comparison':[],
            'implementations':a.implementations,'suite':a.suite,'mtp':a.mtp,'warmup':a.warmup,'repeat':a.repeat,
            'scope':f'Plan kernel Duration only; same{a.requests}requests; fixed-address warmup; no Transfer/SFA',
            'gate':str(a.output/'gate.json'),'validation':'strict each sample, fixed certified reference cache, batchedD2H; continuous gate uncached'}
        metadata=a.output/'capture_metadata.json';state['phase']='profile';write_json(status,state);t=time.monotonic()
        for c in workloads:
            current=a.output/'profile'/c['scenario']/f"mtp{c['mtp']}"
            r=capture(torch,npu,c,a.implementations,a,current);r['comparison']='same_requests';report['runs'].append(r)
            write_json(metadata,report)
        report['mtp_comparison']=mtp_comparison(report['runs']);state['wall_seconds']['profile']=time.monotonic()-t
        report['wall_seconds']=state['wall_seconds'];report['status']='complete'
        write_json(metadata,report)
        write_json(a.output/'data_index.json',{'runs':[{'workload':r['workload'],'raw_csv':r['raw_csv'],'target_samples':r['target_samples']} for r in report['runs']]})
        text=report_text(report)+'\n## 本轮说明\n\n'+('旧BD在MTP7只作逐行回退对照。\n' if a.suite=='mtp8' else '同场比较独立ABCDF、reuse_parallel与同事独立Plan。\n' if a.suite=='peer' else 'MID保留完整query→MID→union映射；未消费的临时映射恢复-1，消费游标按原始query数推进。FUSED用SIMD累计行计数并在包末归约。两路不启用安全前缀或16位映射；同预算/同输入，性能只看本轮CSV。\n' if a.suite=='merge_unique' else 'A=安全消费前缀，跳过事后消费统计；B=8行组16位UID映射，归并包由612扩至816条/路。四路均512线程，生产动态UB同为181024B；兼容回退未改。\n' if a.suite=='prefix' else 'A=512线程并行消费统计；B=并行rank归并+融合消费统计；reuse=缓冲复用；tile=每线程分段归并。与上轮BD8同UB预算/同输入；性能结论仅依据本轮CSV。兼容回退未改。\n')+'所有联合实现都需查看共享/回退数量，不能把容量溢出回退当成联合执行。\n'
        if a.suite=='peer':
            text+='\npeer_qli_plan为同事分支收录的独立Plan；读取共同GM升序TopK。QLI/TopK生成和融合producer调度不在本轮范围。上游保留112KiB scratch门槛，UB/GM路径见metadata。\n'
        text+='\n阶段墙钟秒数：'+json.dumps(state['wall_seconds'],ensure_ascii=False)+'\n'
        text+='\n每工况的恢复、校验、首次参考认证、缓存命中、采集循环及profiler导出解析耗时见metadata；这些不是设备kernel耗时。\n'
        (a.output/'report.md').write_text(text)
        state['status']='complete';write_json(status,state);write_json(a.output/'COMPLETE.json',{'status':'complete','metadata':str(metadata)})
        print(('===== MTP7_MERGE_UNIQUE_ALL_STAGES_PASSED =====' if a.suite=='merge_unique' else '===== MTP7_PREFIX_ALL_STAGES_PASSED =====' if a.suite=='prefix' else '===== PEER_PLAN_ALL_STAGES_PASSED =====' if a.suite=='peer' else '===== MTP7_SWEEP_ALL_STAGES_PASSED =====' if a.suite=='mtp7_pack' else '===== MTP7_FIX_ALL_STAGES_PASSED =====' if a.suite=='mtp7_fix' else '===== MTP8_ALL_STAGES_PASSED =====')+'\n报告:',a.output/'report.md',flush=True)
      except BaseException as exc:
        record_failure(status,state,exc)
        print(capture_error(a.output,log,exc),file=sys.stderr,flush=True)
        raise
if __name__=='__main__':main()
