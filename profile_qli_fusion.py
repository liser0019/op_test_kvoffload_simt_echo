#!/usr/bin/env python3
"""同输入比较原始Indexer、保序Indexer→独立rankpipe AC和融合A/B/C。"""
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
from joint_plan_cases import make_case,describe,initial_arrays,sort_topk
from fusion.metrics import read_csvs,TIMING_ARITHMETIC
from harness_io import add_storage_arguments,preflight_args,write_json,record_failure,diagnostic_log,diagnostic_details
from scripts.prepare_qli_fusion import NAMES,NATIVE_INDEXER,ORDERED4,source_id

ROOT=Path(__file__).resolve().parent
BASE='joint_rankpipe_ac_512'
INDEXER_ONLY = {'baseline', 'ordered4'}


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


def check_native_topk(indices, canonical, visible_seq_lens):
    """Validate native output and compare token sets against the ordered Indexer."""
    rows = np.asarray(indices)
    if (rows.ndim != 2 or rows.shape != canonical.shape or rows.dtype != np.int32 or
            len(visible_seq_lens) != rows.shape[0]):
        raise RuntimeError('native Indexer TopK shape or dtype mismatch')
    # The upstream Indexer does not promise order by token ID. Sorting a CPU
    # copy of the valid prefix checks uniqueness without changing NPU outputs.
    sorted_rows = rows.copy()
    shared = 0
    valid_total = 0
    min_overlap = 1.0
    different_rows = 0
    for r, visible in enumerate(visible_seq_lens):
        count = min(rows.shape[1], max(0, int(visible)))
        sorted_rows[r, :count].sort()
        valid = sorted_rows[r, :count]
        if (np.any(valid < 0) or np.any(valid >= visible) or
                np.any(valid[1:] == valid[:-1]) or np.any(rows[r, count:] != -1)):
            raise RuntimeError(f'row {r}: invalid native Indexer token IDs, duplicates or padding')
        if count:
            shared_count = np.intersect1d(valid, canonical[r, :count], assume_unique=True).size
            shared += shared_count
            valid_total += count
            min_overlap = min(min_overlap, shared_count / count)
            different_rows += shared_count != count
    return {
        'status': 'passed',
        'topk_set_overlap': shared / valid_total if valid_total else 1.0,
        'topk_min_row_overlap': min_overlap,
        'topk_rows_with_different_sets': int(different_rows),
    }


def load_operators(manifest, vendor, implementations):
    if manifest['source_id']!=source_id():raise RuntimeError('fusion source/manifest mismatch; rebuild required')
    operators={};identities={}
    names = {'split' if mode == 0 else 'abc'[mode-1]: name for mode, name in NAMES.items()}
    if 'baseline' in implementations:
        if manifest.get('native_indexer', {}).get('operator') != NATIVE_INDEXER:
            raise RuntimeError('native Indexer missing from build manifest; rebuild the fusion package')
        names['baseline'] = NATIVE_INDEXER
    if 'ordered4' in implementations:
        if manifest.get('ordered_indexer_4', {}).get('operator') != ORDERED4:
            raise RuntimeError('ordered S1Base=4 missing from build manifest; rebuild the fusion package')
        names['ordered4'] = ORDERED4
    names = {role: name for role, name in names.items() if role == 'split' or role in implementations}
    required = set(names.values())
    if 'peer' in implementations:
        required.add('quant_lightning_indexer_ordered_fused')
    missing = required.difference(manifest['packaged_operators'])
    if missing:
        raise RuntimeError(f'operators missing from the package: {sorted(missing)}')
    for role,name in names.items():
        package='cann_ops_transformer_qlifusion.ops.attention.'+name
        module=importlib.import_module(package)
        binding=importlib.import_module(package+'.'+name)
        native=getattr(binding,name+'_op_builder').load()
        if native._fusion_identity()!=manifest['source_id']:raise RuntimeError(name+': stale compiled torch extension')
        path=Path(native.__file__)
        identities[name]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        if role == 'baseline':
            api_library = (vendor/'op_api/lib/libcust_opapi.so').resolve()
            paths = [Path(p).resolve() for p in native._op_api_paths()]
            if paths != [api_library, api_library]:
                raise RuntimeError(f'native Indexer resolved outside the benchmark vendor: {paths}')
            identities[name]['op_api_library'] = str(api_library)
        operators[role] = getattr(module,name)
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
    build=Path(args.fusion_build).resolve();manifest=json.loads((build/'manifest.json').read_text())
    vendor=(build/'install'/'vendors'/'qlifusion_transformer').resolve()
    def prefer_vendor():
        parts=[p for p in os.environ.get('ASCEND_CUSTOM_OPP_PATH','').split(':') if p]
        parts=[p for p in parts if Path(p).resolve()!=vendor]
        custom=':'.join([str(vendor),*parts]);os.environ['ASCEND_CUSTOM_OPP_PATH']=custom
        return custom
    # Global sitecustomize/backend autoload can prepend fla_npu during imports.
    # Restore this dedicated vendor immediately before and after runtime loading.
    prefer_vendor();torch,torch_npu,device=common.load_runtime(args);custom=prefer_vendor()
    if Path(custom.split(':')[0]).resolve()!=vendor:
        raise RuntimeError('failed to place dedicated vendor first in ASCEND_CUSTOM_OPP_PATH')
    if not manifest.get('installed_files'):raise RuntimeError('installed package identity missing; rebuild with scripts/build_qli_fusion.sh')
    for name,digest in manifest['installed_files'].items():
        f=build/'install'/name
        if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=digest:raise RuntimeError('installed CANN package changed: '+str(f))
    names=args.implementations
    operators,compiled=load_operators(manifest,vendor,names)
    report.update(device=device,build_manifest=manifest,compiled_extensions=compiled,
                  timing_arithmetic=TIMING_ARITHMETIC)
    from fusion.glm52_inputs import Shape,make_inputs
    def quantize(x):return torch_npu.npu_dynamic_quant(x,dst_type=torch.float8_e4m3fn)
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
        dummy_case=sort_topk(make_case(args.requests,2048,capacity,max(262144,context),mtp,'cold_reset','shared',seed=args.seed,block_size=128))
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
        visible_seq_lens=inputs.plan_metadata['visible_seq_lens'].cpu().numpy()
        check_ordered_topk(canonical_cpu,visible_seq_lens)
        wall[f'fixture/context{context}/mtp{mtp}']=time.monotonic()-begin
        for scenario in args.scenarios:
            case=make_plan_case(inputs,canonical_cpu,mtp,scenario,args.seed)
            tag=f'context{context}_mtp{mtp}_{scenario}';directory=out/tag;directory.mkdir()
            bundles={};topk_guards={};topk_outputs={}
            device_query=inputs.qli['query'].device
            for n in names:
                guard=torch.full((shape.rows*2048+32,),common.GUARD_VALUE,dtype=torch.int32,device=device_query)
                topk_outputs[n]=guard[16:-16].reshape(shape.rows,1,2048)
                topk_guards[n]=guard
                if n not in INDEXER_ONLY:
                    inp=common.upload_inputs(torch,case)
                    inp['topk_indices']=topk_outputs[n].reshape(shape.rows,2048)
                    inp['topk_indices'].copy_(canonical.reshape(shape.rows,2048))
                    bundles[n]=common.prepare(torch,case,BASE,inputs=inp) if n=='split' else prepare_fused(torch,case,inp)
            extras={};calls={n:0 for n in names};raw={n:[] for n in names}
            for n in names:
                extras[n]=[torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)]
            for b in bundles.values():
                b['fast_fixed_check']=True
            current={'implementation':None,'phase':None,'call':0}
            def invoke(n,phase):
                b=bundles.get(n);current.update(implementation=n,phase=phase,call=calls[n]+1)
                if b is not None:
                    common.restore(torch,b);ready.zero_();torch.npu.synchronize()
                topk_outputs[n].fill_(-777);torch.npu.synchronize()
                start,end=extras[n];host=time.monotonic();start.record()
                if n=='baseline':
                    kw=inputs.qli
                elif n=='ordered4':
                    # The shared registration ABI expects these tensors. The
                    # generated Indexer has its Plan hook disabled at compile time.
                    kw=kwargs(dummy,dw)
                else:
                    kw=kwargs(b,dw if n=='split' else None)
                result=operators[n](**kw,topk_out=topk_outputs[n]) if n!='peer' else operators[n](**kw)
                if n=='split':
                    # Current stream; no synchronize/D2H between the two kernels.
                    common.invoke(b)
                end.record();end.synchronize();event_us=start.elapsed_time(end)*1000.;host_us=(time.monotonic()-host)*1e6
                calls[n]+=1
                indices=result.cpu().numpy().reshape(shape.rows,2048)
                if n=='baseline':
                    result_info=check_native_topk(indices,canonical_cpu,visible_seq_lens)
                else:
                    np.testing.assert_array_equal(indices,canonical_cpu,err_msg='ordered TopK changed')
                if n=='peer':b['inputs']['topk_indices'].copy_(result.reshape(shape.rows,2048)) # checking only, after end Event
                for k,before in readonly.items():
                    if not torch.equal(inputs.qli[k].view(torch.uint8),before.view(torch.uint8)):raise RuntimeError('read-only QLI input modified: '+k)
                if not torch.equal(inputs.plan_metadata['host_block_table'],host_table_before):raise RuntimeError('read-only Plan host_block_table modified')
                if not torch.equal(ready,torch.zeros_like(ready)):raise RuntimeError('publication counter not returned to zero')
                for guard in (ds[:16],ds[-16:]):
                    if not torch.all(guard==common.GUARD_VALUE):raise RuntimeError('split QLI workspace guard changed')
                for guard in (topk_guards[n][:16],topk_guards[n][-16:]):
                    if not torch.all(guard==common.GUARD_VALUE):raise RuntimeError('TopK output guard changed')
                if n=='ordered4':
                    for key in dummy['host_initial']:
                        if key!='workspace' and not torch.equal(dummy['state'][key], dummy['initial'][key]):
                            raise RuntimeError('ordered4 modified Plan state: '+key)
                    result_info={'status':'passed', 'topk_exact_match':True, 's1_base':4}
                elif b is not None:
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
                            event_us,host_us,info=invoke(n,'sample')
                            sample={'event_us':event_us,'host_us':host_us}
                            if n=='baseline':sample['topk_check']=info
                            raw[n].append(sample);orders.append(n);checks[n]=info
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
                split_span=stats['split']['span_us']['p50_us']
                for s in stats.values():
                    s['span_p50_over_split']=s['span_us']['p50_us']/split_span
                if 'baseline' in stats:
                    baseline_span=stats['baseline']['span_us']['p50_us']
                    for s in stats.values():
                        span=s['span_us']['p50_us']
                        s['span_p50_over_native_indexer']=span/baseline_span
                        s['span_p50_delta_vs_native_indexer_us']=span-baseline_span
                    topk_checks=[s['topk_check'] for s in raw['baseline']]
                    stats['baseline']['native_topk_comparison']={
                        'min_sample_overlap':min(s['topk_set_overlap'] for s in topk_checks),
                        'min_row_overlap':min(s['topk_min_row_overlap'] for s in topk_checks),
                        'max_different_rows':max(s['topk_rows_with_different_sets'] for s in topk_checks),
                    }
                data={'tag':tag,'context':context,'mtp':mtp,'capacity':capacity,'requests':args.requests,
                    'timing_arithmetic':TIMING_ARITHMETIC,
                    'qli_input_sha256':qli_hash,'qli_s1_base':2,
                    'input':describe(case),'logical_input_sha256':hashlib.sha256(b''.join(case[k].tobytes() for k in common.INPUTS)).hexdigest(),
                    'statistics':stats,'execution_orders':orders,'csv_paths':[str(p) for p in csvs],'kernel_inventory':inventory}
                if 'ordered4' in stats:
                    ordered2=stats['split']['indexer_span_us']['p50_us']
                    ordered4=stats['ordered4']['span_us']['p50_us']
                    data['indexer_comparison']={
                        'ordered4_s1_base':4,
                        'ordered2_s1_base':2,
                        'ordered2_span_p50_us':ordered2,
                        'ordered4_span_p50_us':ordered4,
                        'ordered2_minus_ordered4_p50_us':ordered2-ordered4,
                        'split_plan_p50_us':stats['split']['plan_us']['p50_us'],
                    }
                    if 'baseline' in stats:
                        data['indexer_comparison']['ordered4_minus_native_p50_us']=ordered4-baseline_span
                write_json(directory/'capture_metadata.json',data);all_results.append(data)
                write_json(out/'data_index.json',{'status':'running','cases':[{'tag':r['tag'],'metadata':str(out/r['tag']/'capture_metadata.json'),'csv_paths':r['csv_paths']} for r in all_results]})
                print('完成',tag,flush=True)
            except BaseException:
                try:
                    write_json(directory/'failure_call.json',current)
                    if current['implementation'] in INDEXER_ONLY:
                        name=current['implementation']
                        np.savez(directory/'failure_topk.npz',
                                 **{name:topk_outputs[name].cpu().numpy()},ordered=canonical_cpu,
                                 visible_seq_lens=visible_seq_lens)
                    else:
                        b=bundles[current['implementation']]
                        np.savez(directory/'failure_state.npz',**common.snapshot(b),**{k:case[k] for k in common.INPUTS},
                            **{'actual_input_'+k:b['inputs'][k].cpu().numpy() for k in common.INPUTS})
                except BaseException:pass
                raise
            finally:
                for b in bundles.values():common.close(b)
        del readonly,inputs,dummy,canonical,bundles,topk_outputs,topk_guards
    report.update(status='complete',wall_seconds=wall,cases=all_results)
    write_json(out/'capture_metadata.json',report)
    if args.gate_only:
        (out/'report.md').write_text('# 融合Plan边界正确性\n\n仅正确性检查，无性能结论。\n\n'+json.dumps(all_results,ensure_ascii=False,indent=2)+'\n')
        return
    write_json(out/'data_index.json',{'status':'complete','cases':[{'tag':r['tag'],'metadata':str(out/r['tag']/'capture_metadata.json'),'csv_paths':r['csv_paths']} for r in all_results]})
    lines=['# Indexer与Plan性能对照','','设备时间单位μs。busy为目标设备区间的并集长度，避免把重叠AIC/AIV重复相加；span为首个目标开始至最后目标结束。Event包含接口下发及辅助任务，单独报告。',
           'CSV使用十进制解析并先扣除共同时间原点；原始时间字符串与小范围时间戳重叠容差保留在逐样本元数据。',
           'split为同一保序Indexer实现＋冻结rankpipe AC，Indexer分块固定S1Base=2。状态恢复、校验不计入目标Duration；固定地址预热。',
           '75%标签仅约束首行初始命中；其他行使用真实Indexer选择，实际命中与descriptor见元数据。未测Transfer/SFA。']
    with_baseline = 'baseline' in names
    if with_baseline:
        lines+=['baseline只运行上游原始quant_lightning_indexer，使用原始tiling。冷/热标签只描述Plan的缓存初态。',
                'span/拆分用于评估融合收益。span/原始Indexer和差值反映两种不同计算范围的整段耗时。']
    baseline_columns = ' span/原始Indexer | 对原始Indexer Δspan P50 (μs) |' if with_baseline else ''
    baseline_align = '---:|---:|' if with_baseline else ''
    lines+=['',f'| 工况 | 实现 | busy P50/P95 | span P50/P95 | Event P50/P95 | span/拆分 |'
            f'{baseline_columns} 样本数 | 联合/回退/单行有序 | descriptor数 |',
            f'|---|---|---:|---:|---:|---:|{baseline_align}---:|---|---:|']
    for r in all_results:
        for n,s in r['statistics'].items():
            fmt=lambda x:f"{x['p50_us']:.3f}/{x['p95_us']:.3f}"
            checked=s['checks']
            paths,descriptors='—','—'
            if n not in INDEXER_ONLY:
                paths=f"{checked['joint_groups']}/{checked['overflow_groups']}/{checked.get('single_sorted_groups',0)}"
                descriptors=checked['total_transfer_descriptors']
            comparison=''
            if with_baseline:
                comparison=(f" {s['span_p50_over_native_indexer']:.3f} |"
                            f" {s['span_p50_delta_vs_native_indexer_us']:+.3f} |")
            lines.append(f"| {r['tag']} | {n} | {fmt(s['busy_us'])} | {fmt(s['span_us'])} | {fmt(s['event'])} | "
                         f"{s['span_p50_over_split']:.3f} |{comparison} {args.repeat} | {paths} | {descriptors} |")
        s=r['statistics']['split'];lines+=['',f"拆分详情 {r['tag']}：Indexer span P50={s['indexer_span_us']['p50_us']:.3f}，Plan P50={s['plan_us']['p50_us']:.3f}，两者间隙P50={s['between_kernel_gap_us']['p50_us']:.3f}。",'']
        if 'indexer_comparison' in r:
            indexer=r['indexer_comparison']
            lines.append(f"Indexer 对照 {r['tag']}：保序 S1Base=4 单跑 P50 "
                         f"{indexer['ordered4_span_p50_us']:.3f} μs，split 的保序 S1Base=2 段 "
                         f"{indexer['ordered2_span_p50_us']:.3f} μs，两者差 "
                         f"{indexer['ordered2_minus_ordered4_p50_us']:+.3f} μs（2行减4行）。")
            if 'ordered4_minus_native_p50_us' in indexer:
                lines.append(f"保序 S1Base=4 减原始 Indexer 的 P50 差值 "
                             f"{indexer['ordered4_minus_native_p50_us']:+.3f} μs。")
            lines.append('')
        if 'baseline' in r['statistics']:
            topk=r['statistics']['baseline']['native_topk_comparison']
            lines+= [f"原始Indexer TopK折算为集合后，{r['tag']} 的最小样本重合率为 "
                     f"{topk['min_sample_overlap']:.4%}，最小行重合率为 {topk['min_row_overlap']:.4%}，"
                     f"单个样本最多 {topk['max_different_rows']} 行集合有差异。",'']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);common.add_runtime_arguments(p)
    p.add_argument('--fusion-build',type=Path,default=Path(os.environ.get('QLI_FUSION_BUILD_DIR',ROOT/'build_qli_fusion')))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--requests',type=int,default=32)
    p.add_argument('--mtp',nargs='+',type=int,choices=(0,1,3,7),default=[0,1,3,7]);p.add_argument('--contexts',nargs='+',type=int,default=[262144])
    p.add_argument('--scenarios',nargs='+',choices=('cold_reset','full_cache_75pct'),default=['cold_reset','full_cache_75pct'])
    p.add_argument('--implementations',nargs='+',choices=('baseline','ordered4','split','a','b','c','peer'),
                   default=['split','a','b','c'],
                   help='baseline: upstream Indexer only; ordered4: ordered Indexer only, S1Base=4')
    p.add_argument('--warmup',type=int,default=2);p.add_argument('--repeat',type=int,default=5);p.add_argument('--regression-repeat',type=int,default=1)
    p.add_argument('--seed',type=int,default=2026);p.add_argument('--kernel-map',type=Path)
    p.add_argument('--gate-only',action='store_true',help='只验证融合A/B/C，可测短序列尾部-1；不运行独立Plan或生成性能数字')
    args=p.parse_args()
    if args.gate_only:
        args.implementations=[n for n in args.implementations if n not in INDEXER_ONLY | {'split','peer'}]
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
