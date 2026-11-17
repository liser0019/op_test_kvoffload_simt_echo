#!/usr/bin/env python3
"""同输入比较：保序Indexer→独立rankpipe AC，与融合A/B/C。仅在NPU运行。"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from compare import summary
import joint_plan_common as common
from joint_plan_cases import make_case,describe,initial_arrays
from fusion.metrics import read_csvs
from harness_io import add_storage_arguments,preflight_args,write_json,record_failure,diagnostic_log,diagnostic_details
from scripts.prepare_qli_fusion import NAMES,source_id

ROOT=Path(__file__).resolve().parent
BASE='joint_rankpipe_ac_512'


def check_ordered_topk(indices,visible_seq_lens):
    """Verify this pinned ordered Indexer's contract; never sort/repair its output."""
    rows=np.asarray(indices)
    visible=np.asarray(visible_seq_lens).reshape(-1)
    if rows.ndim!=2 or len(rows)!=len(visible):
        raise RuntimeError('ordered TopK/visible_seq_lens shape mismatch')
    for r,row in enumerate(rows):
        count=min(len(row),max(0,int(visible[r])))
        valid=row[:count]
        if (np.any(valid<0) or np.any(valid>=visible[r]) or
            np.any(valid[1:]<=valid[:-1]) or np.any(row[count:]!=-1)):
            raise RuntimeError(f'row {r}: expected strictly increasing valid token IDs followed only by -1')


def load_operators(manifest):
    if manifest['source_id']!=source_id():raise RuntimeError('fusion source/manifest mismatch; rebuild required')
    operators={};identities={}
    for mode,name in NAMES.items():
        module=importlib.import_module('cann_ops_transformer_qlifusion.ops.attention.'+name)
        native=getattr(module,name+'_op_builder').load()
        if native._fusion_identity()!=manifest['source_id']:raise RuntimeError(name+': stale compiled torch extension')
        path=Path(native.__file__)
        identities[name]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        operators['split' if mode==0 else 'abc'[mode-1]]=getattr(module,name)
    return operators,identities


def fingerprint_qli(torch,values):
    h=hashlib.sha256()
    for key,value in sorted(values.items()):
        h.update(key.encode())
        if torch.is_tensor(value):
            h.update(str((tuple(value.shape),tuple(value.stride()),value.storage_offset(),str(value.dtype))).encode())
            # Bounded D2H hashing only during fixture creation, never in timing.
            byte=value.view(torch.uint8)
            for start in range(0,byte.shape[0],256):h.update(byte[start:start+256].cpu().contiguous().numpy().tobytes())
        else:h.update(repr(value).encode())
    return h.hexdigest()


def make_plan_case(inputs,indices,mtp,scenario,seed):
    s=inputs.shape
    case=make_case(s.batch,s.topk,s.capacity,max(262144,s.seq_len),mtp,scenario,'shared',seed=seed,block_size=128)
    case['max_token']=s.seq_len;case['context']=s.seq_len-s.width
    case['overlap']='indexer_actual'
    case['topk_indices']=np.ascontiguousarray(indices.reshape(s.rows,s.topk),dtype=np.int32)
    for key in ('req_ids','stable_prefix_lens','visible_seq_lens','token_to_req'):
        case[key]=inputs.plan_metadata[key].cpu().numpy().copy()
    case['last_req_ids']=np.repeat(case['req_ids'],s.width)
    case['lru_slots']=np.tile(np.arange(s.capacity,dtype=np.int32),(s.rows,1))
    case['slot_to_token']=np.full((s.rows,s.capacity),-1,np.int32)
    if scenario=='cold_reset':case['last_req_ids'][:]=-1
    else:
        # 75% refers to the FIRST query, matching the historic experiment label.
        # Other rows use real Indexer choices; their measured hit rates are reported.
        for request in range(s.batch):
            first=request*s.width;valid=case['topk_indices'][first]
            valid=valid[(valid>=0)&(valid<case['stable_prefix_lens'][first])]
            hits=np.unique(valid)[:3*s.topk//4]
            if len(hits)!=3*s.topk//4:raise RuntimeError('not enough stable first-row tokens to construct 75% input')
            excluded=np.unique(case['topk_indices'][first:first+s.width]);pool=np.arange(s.seq_len-s.width,dtype=np.int32)
            available=pool[~np.isin(pool,excluded)]
            needed=s.capacity-len(hits)
            if len(available)<needed:raise RuntimeError('context too short for full-cache 75% fixture; increase --contexts')
            rng=np.random.default_rng(seed+request);rng.shuffle(available)
            resident=np.concatenate((hits,available[:needed]));rng.shuffle(resident)
            case['slot_to_token'][first:first+s.width]=resident
    return case


def extra_workspace(torch,case,device):
    r,c,t=case['requests'],case['capacity'],case['topk']
    n=r*8*t//2+r*8*c+r*8*(3*4096+c+t)
    storage=torch.full((n+32,),common.GUARD_VALUE,dtype=torch.int32,device=device)
    view=storage[16:-16];view.fill_(np.iinfo(np.int32).max)
    return storage,view


def prepare_fused(torch,case,inputs):
    host=initial_arrays(case,True,case['capacity']);state={};buffers={};initial={}
    for k,array in host.items():
        dtype=torch.int64 if array.dtype==np.int64 else torch.int32
        storage=torch.full((array.size+32,),common.GUARD_VALUE,dtype=dtype,device=inputs['req_ids'].device)
        view=storage[16:-16].reshape(array.shape);view.copy_(torch.from_numpy(array).to(view.device))
        state[k]=view;buffers[k]=storage;initial[k]=view.clone()
    storage,view=extra_workspace(torch,case,inputs['req_ids'].device)
    state['workspace']=view;buffers['workspace']=storage;initial['workspace']=view.clone()
    return dict(name=BASE,torch=torch,inputs=inputs,state=state,buffers=buffers,initial=initial,
        host_initial=host,joint_enabled=True,context=None,layout={},page_table_host=None)


def operator_kwargs(inputs,bundle,ready,workspace=None):
    """Pass the actual metadata tensors checked after invocation, never a clone."""
    s=inputs.shape;state=bundle['state']
    metadata={k:bundle['inputs'][k] for k in ('req_ids','stable_prefix_lens','visible_seq_lens','token_to_req')}
    metadata['host_block_table']=inputs.plan_metadata['host_block_table']
    return inputs.qli|metadata|{k:v for k,v in state.items() if k!='workspace'}|{
        'active_rows':bundle['inputs']['active_rows'],'compact_workspace':state['workspace'] if workspace is None else workspace,
        'ready_counts':ready,'capacity':s.capacity,'max_token':s.seq_len,'host_block_size':128,
        'host_num_blocks':inputs.host_num_blocks,'joint_mtp':1,'max_joint_rows':8}


def main_run(args,report):
    torch,torch_npu,device=common.load_runtime(args)
    build=Path(args.fusion_build).resolve();manifest=json.loads((build/'manifest.json').read_text())
    custom=os.environ.get('ASCEND_CUSTOM_OPP_PATH','')
    if not custom or not Path(custom.split(':')[0]).resolve().is_relative_to(build/'install'):
        raise RuntimeError('source this build\'s env.sh; its dedicated vendor must be first in ASCEND_CUSTOM_OPP_PATH')
    if not manifest.get('installed_files'):raise RuntimeError('installed package identity missing; rebuild with scripts/build_qli_fusion.sh')
    for name,digest in manifest['installed_files'].items():
        f=build/'install'/name
        if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=digest:raise RuntimeError('installed CANN package changed: '+str(f))
    operators,compiled=load_operators(manifest)
    report.update(device=device,build_manifest=manifest,compiled_extensions=compiled)
    from fusion.glm52_inputs import Shape,make_inputs
    def quantize(x):return torch_npu.npu_dynamic_quant(x,dst_type=torch.float8_e4m3fn)
    names=args.implementations
    if 'peer' in names:
        m=importlib.import_module('cann_ops_transformer_qlifusion.ops.attention.quant_lightning_indexer_ordered_fused')
        operators['peer']=m.quant_lightning_indexer_ordered_fused
    out=args.output;wall={};all_results=[]
    for context in args.contexts:
      for mtp in args.mtp:
        capacity=max(4096,(mtp+1)*2048)
        shape=Shape(batch=args.requests,width=mtp+1,seq_len=context,capacity=capacity)
        begin=time.monotonic();inputs=make_inputs(shape,device=f'npu:{args.device}',seed=args.seed,quantizer=quantize)
        # Exact snapshots stay on device; comparisons occur outside target timing.
        readonly={k:v.clone() for k,v in inputs.qli.items() if torch.is_tensor(v)}
        host_table_before=inputs.plan_metadata['host_block_table'].clone()
        qli_hash=fingerprint_qli(torch,inputs.qli|{'plan_host_block_table':inputs.plan_metadata['host_block_table']})
        dummy_case=make_case(args.requests,2048,capacity,max(262144,context),mtp,'cold_reset','shared',seed=args.seed,block_size=128)
        dummy=common.prepare(torch,dummy_case,BASE);common.close(dummy)
        for key in ('req_ids','stable_prefix_lens','visible_seq_lens','token_to_req'):
            dummy['inputs'][key].copy_(inputs.plan_metadata[key])
        ds,dw=extra_workspace(torch,dummy_case,inputs.qli['query'].device)
        ready=torch.zeros(shape.rows,dtype=torch.int32,device=inputs.qli['query'].device)
        def kwargs(bundle,workspace=None):
            return operator_kwargs(inputs,bundle,ready,workspace)
        # Canonical ordered TopK is generated on NPU, never precomputed on CPU.
        canonical=operators['split'](**kwargs(dummy,dw));torch.npu.synchronize()
        canonical_cpu=canonical.cpu().numpy().reshape(shape.rows,2048)
        check_ordered_topk(canonical_cpu,inputs.plan_metadata['visible_seq_lens'].cpu().numpy())
        wall[f'fixture/context{context}/mtp{mtp}']=time.monotonic()-begin
        for scenario in args.scenarios:
            case=make_plan_case(inputs,canonical_cpu,mtp,scenario,args.seed)
            tag=f'context{context}_mtp{mtp}_{scenario}';directory=out/tag;directory.mkdir()
            bundles={};topk_guards={}
            for n in names:
                inp=common.upload_inputs(torch,case)
                guard=torch.full((shape.rows*2048+32,),common.GUARD_VALUE,dtype=torch.int32,device=inp['req_ids'].device)
                inp['topk_indices']=guard[16:-16].reshape(shape.rows,2048)
                inp['topk_indices'].copy_(canonical.reshape(shape.rows,2048))
                bundles[n]=common.prepare(torch,case,BASE,inputs=inp) if n=='split' else prepare_fused(torch,case,inp)
                topk_guards[n]=guard
            extras={};calls={n:0 for n in names};raw={n:[] for n in names}
            for n,b in bundles.items():
                b['fast_fixed_check']=True
                extras[n]=[torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)]
            current={'implementation':None,'phase':None,'call':0}
            def invoke(n,phase):
                b=bundles[n];current.update(implementation=n,phase=phase,call=calls[n]+1)
                common.restore(torch,b);ready.zero_();torch.npu.synchronize()
                b['inputs']['topk_indices'].fill_(-777);torch.npu.synchronize()
                start,end=extras[n];host=time.monotonic();start.record()
                kw=kwargs(b,dw if n=='split' else None)
                if n!='peer':kw['topk_out']=b['inputs']['topk_indices'].reshape(shape.rows,1,2048)
                result=operators['split' if n=='split' else n](**kw)
                if n=='split':
                    # Current stream; no synchronize/D2H between the two kernels.
                    common.invoke(b)
                end.record();end.synchronize();event_us=start.elapsed_time(end)*1000.;host_us=(time.monotonic()-host)*1e6
                calls[n]+=1
                np.testing.assert_array_equal(result.cpu().numpy().reshape(shape.rows,2048),canonical_cpu,err_msg='ordered TopK changed')
                if n=='peer':b['inputs']['topk_indices'].copy_(result.reshape(shape.rows,2048)) # checking only, after end Event
                for k,before in readonly.items():
                    if not torch.equal(inputs.qli[k].view(torch.uint8),before.view(torch.uint8)):raise RuntimeError('read-only QLI input modified: '+k)
                if not torch.equal(inputs.plan_metadata['host_block_table'],host_table_before):raise RuntimeError('read-only Plan host_block_table modified')
                if not torch.equal(ready,torch.zeros_like(ready)):raise RuntimeError('publication counter not returned to zero')
                for guard in (ds[:16],ds[-16:]):
                    if not torch.all(guard==common.GUARD_VALUE):raise RuntimeError('split QLI workspace guard changed')
                for guard in (topk_guards[n][:16],topk_guards[n][-16:]):
                    if not torch.all(guard==common.GUARD_VALUE):raise RuntimeError('TopK output guard changed')
                result_info=common.check(case,b)
                return event_us,host_us,result_info
            begin=time.monotonic()
            try:
                gate_info={}
                for n in names:
                    for _ in range(args.regression_repeat):gate_info[n]=invoke(n,'gate')[2]
                if args.gate_only:
                    entry={'tag':tag,'correctness':'passed','checks':gate_info,'calls':calls.copy()}
                    write_json(directory/'gate.json',entry);all_results.append(entry)
                    continue
                for i in range(args.warmup):
                    for n in names[i%len(names):]+names[:i%len(names)]:invoke(n,'warmup')
                wall[tag+'/gate_warmup']=time.monotonic()-begin
                orders=[];checks={};capture=directory/'trace'
                begin=time.monotonic()
                with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
                    record_shapes=False,profile_memory=False,with_stack=False,
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(capture))) as prof:
                    wait_start=time.monotonic();time.sleep(1)
                    write_json(directory/'capture_start_wait.json',{'requested_seconds':1,'elapsed_seconds':time.monotonic()-wait_start})
                    for iteration in range(args.repeat):
                        order=names[iteration%len(names):]+names[:iteration%len(names)]
                        for n in order:
                            event_us,host_us,info=invoke(n,'sample');raw[n].append({'event_us':event_us,'host_us':host_us});orders.append(n);checks[n]=info
                        prof.step()
                wall[tag+'/capture_check_export']=time.monotonic()-begin
                csvs=sorted(capture.rglob('kernel_details.csv'))
                if not csvs:raise RuntimeError(f'kernel_details.csv missing; raw capture={capture}, log={out/"profile.log"}')
                inventory=set()
                for path in csvs:
                    with path.open(encoding='utf-8-sig',newline='') as stream:
                        for row in csv.DictReader(stream):inventory.add(row.get('Name',row.get('Kernel Name','')))
                write_json(directory/'kernel_inventory.json',sorted(inventory))
                pairs,inventory=read_csvs(csvs,orders,args.repeat,json.loads(args.kernel_map.read_text()) if args.kernel_map else None)
                stats={}
                for n in names:
                    samples=[p for p in pairs if p['implementation']==n]
                    stats[n]={key:summary([p[key] for p in samples]) for key in ('span_us','busy_us')}
                    for key in ('indexer_span_us','plan_us','between_kernel_gap_us'):
                        if key in samples[0]:stats[n][key]=summary([p[key] for p in samples])
                    stats[n].update(event=summary([p['event_us'] for p in raw[n]]),host=summary([p['host_us'] for p in raw[n]]),checks=checks[n],raw_interface=raw[n],raw_device=samples)
                data={'tag':tag,'context':context,'mtp':mtp,'capacity':capacity,'requests':args.requests,
                    'qli_input_sha256':qli_hash,'qli_s1_base':2,
                    'input':describe(case),'logical_input_sha256':hashlib.sha256(b''.join(case[k].tobytes() for k in common.INPUTS)).hexdigest(),
                    'statistics':stats,'execution_orders':orders,'csv_paths':[str(p) for p in csvs],'kernel_inventory':inventory}
                write_json(directory/'capture_metadata.json',data);all_results.append(data)
                write_json(out/'data_index.json',{'status':'running','cases':[{'tag':r['tag'],'metadata':str(out/r['tag']/'capture_metadata.json'),'csv_paths':r['csv_paths']} for r in all_results]})
                print('完成',tag,flush=True)
            except BaseException:
                try:
                    b=bundles[current['implementation']]
                    np.savez(directory/'failure_state.npz',**common.snapshot(b),**{k:case[k] for k in common.INPUTS},
                        **{'actual_input_'+k:b['inputs'][k].cpu().numpy() for k in common.INPUTS})
                    write_json(directory/'failure_call.json',current)
                except BaseException:pass
                raise
            finally:
                for b in bundles.values():common.close(b)
        del readonly,inputs,dummy,canonical,bundles
    report.update(status='complete',wall_seconds=wall,cases=all_results)
    write_json(out/'capture_metadata.json',report)
    if args.gate_only:
        (out/'report.md').write_text('# 融合Plan边界正确性\n\n仅正确性检查，无性能结论。\n\n'+json.dumps(all_results,ensure_ascii=False,indent=2)+'\n')
        return
    write_json(out/'data_index.json',{'status':'complete','cases':[{'tag':r['tag'],'metadata':str(out/r['tag']/'capture_metadata.json'),'csv_paths':r['csv_paths']} for r in all_results]})
    lines=['# 保序Indexer＋Plan：拆分与融合对照','','设备时间单位μs。busy为目标设备区间的并集长度，避免把重叠AIC/AIV重复相加；span为首个目标开始至最后目标结束。Event包含接口下发及辅助任务，单独报告。',
           'split为同一保序Indexer实现＋冻结rankpipe AC，Indexer分块固定S1Base=2。状态恢复、校验不计入目标Duration；固定地址预热。',
           '75%标签仅约束首行初始命中；其他行使用真实Indexer选择，实际命中与descriptor见元数据。未测Transfer/SFA。','',
           '| 工况 | 实现 | busy P50/P95 | span P50/P95 | Event P50/P95 | span/拆分 | 样本数 | 联合/回退/单行有序 | descriptor数 |',
           '|---|---|---:|---:|---:|---:|---:|---|---:|']
    for r in all_results:
        base=r['statistics']['split']['span_us']['p50_us']
        for n,s in r['statistics'].items():
            fmt=lambda x:f"{x['p50_us']:.3f}/{x['p95_us']:.3f}"
            checked=s['checks']
            paths=f"{checked['joint_groups']}/{checked['overflow_groups']}/{checked.get('single_sorted_groups',0)}"
            lines.append(f"| {r['tag']} | {n} | {fmt(s['busy_us'])} | {fmt(s['span_us'])} | {fmt(s['event'])} | {s['span_us']['p50_us']/base:.3f} | {args.repeat} | {paths} | {checked['total_transfer_descriptors']} |")
        s=r['statistics']['split'];lines+=['',f"拆分详情 {r['tag']}：Indexer span P50={s['indexer_span_us']['p50_us']:.3f}，Plan P50={s['plan_us']['p50_us']:.3f}，两者间隙P50={s['between_kernel_gap_us']['p50_us']:.3f}。",'']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);common.add_runtime_arguments(p)
    p.add_argument('--fusion-build',type=Path,default=Path(os.environ.get('QLI_FUSION_BUILD_DIR',ROOT/'build_qli_fusion')))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--requests',type=int,default=32)
    p.add_argument('--mtp',nargs='+',type=int,choices=(0,1,3,7),default=[0,1,3,7]);p.add_argument('--contexts',nargs='+',type=int,default=[262144])
    p.add_argument('--scenarios',nargs='+',choices=('cold_reset','full_cache_75pct'),default=['cold_reset','full_cache_75pct'])
    p.add_argument('--implementations',nargs='+',choices=('split','a','b','c','peer'),default=['split','a','b','c'])
    p.add_argument('--warmup',type=int,default=2);p.add_argument('--repeat',type=int,default=5);p.add_argument('--regression-repeat',type=int,default=1)
    p.add_argument('--seed',type=int,default=2026);p.add_argument('--kernel-map',type=Path)
    p.add_argument('--gate-only',action='store_true',help='只验证融合A/B/C，可测短序列尾部-1；不运行独立Plan或生成性能数字')
    args=p.parse_args()
    if args.gate_only:
        args.implementations=[n for n in args.implementations if n not in ('split','peer')]
        if args.scenarios!=['cold_reset']:p.error('--gate-only requires --scenarios cold_reset')
    if args.requests<1 or args.repeat<1 or args.regression_repeat<1 or args.warmup<0 or any(x<(8 if args.gate_only else 2048) for x in args.contexts):p.error('invalid sizes/counts')
    if (not args.gate_only and 'split' not in args.implementations) or not args.implementations or len(set(args.implementations))!=len(args.implementations):p.error('split baseline is mandatory for performance; implementations must be unique')
    args.output=args.output.resolve()
    if (args.output/'RUN_STATUS.json').exists():p.error('output already contains a run; choose a new directory')
    report={'status':'running','configuration':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}}
    preflight_args(args,args.output);write_json(args.output/'RUN_STATUS.json',report)
    try:
        with diagnostic_log(args.output/'profile.log'):
            main_run(args,report)
            write_json(args.output/'RUN_STATUS.json',report)
            write_json(args.output/'COMPLETE.json',{'status':'complete','kind':'correctness_only' if args.gate_only else 'performance',
                'report':str(args.output/'report.md')})
    except BaseException as exc:
        try:report['diagnostics']=diagnostic_details(args.output,args.output/'profile.log')
        except BaseException as secondary:report['diagnostics']={'log':str(args.output/'profile.log'),'diagnostic_error':str(secondary)}
        record_failure(args.output/'RUN_STATUS.json',report,exc)
        try:
            if (args.output/'report.md').exists():(args.output/'report.md').rename(args.output/'INCOMPLETE_report.md')
        except OSError:pass
        print('失败：',args.output,'详细日志：',args.output/'profile.log','首个异常：',report['diagnostics'],file=sys.stderr)
        raise
    print((args.output/'report.md').read_text());print('QLI_FUSION_ALL_STAGES_PASSED')

if __name__=='__main__':main()
