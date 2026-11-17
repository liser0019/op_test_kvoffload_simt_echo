"""Strict invocation grouping, including overlapping AIC/AIV components."""
import csv
import math
import re
from collections import Counter

TOKENS={'split':'quant_lightning_indexer_plan_split','a':'quant_lightning_indexer_plan_a',
        'b':'quant_lightning_indexer_plan_b','c':'quant_lightning_indexer_plan_c',
        'peer':'quant_lightning_indexer_ordered_fused',
        'plan':'sparse_kv_plan_joint_kernel_rankpipe_ac_512'}

def interval_union(events):
    spans=sorted((r['start_us'],r['start_us']+r['duration_us']) for r in events)
    total=0.;end=-math.inf
    for lo,hi in spans:
        total+=max(0.,hi-max(lo,end));end=max(end,hi)
    return total

def read_csvs(paths,orders,repeat,exact=None):
    records=[];inventory=set();expected=Counter(orders)
    if any(v!=repeat for v in expected.values()):raise ValueError('invocation schedule/count mismatch')
    for path in paths:
        with open(path,encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f)
            def column(options):return next((k for k in options if k in (reader.fieldnames or [])),None)
            nk=column(('Name','Kernel Name'));dk=column(('Duration(us)','Duration(μs)','Duration(µs)'))
            sk=column(('Start Time(us)','Start Time(μs)','Start Time(µs)'))
            if not all((nk,dk,sk)):raise RuntimeError(f'{path}: missing exact microsecond columns: {reader.fieldnames}')
            for row in reader:
                name=row[nk];inventory.add(name)
                hits=[]
                for role,token in TOKENS.items():
                    if exact:
                        match=name in exact.get(role,[])
                    else:
                        # Accept build-generated suffixes, not an arbitrary substring.
                        camel=''.join(w.title() for w in token.split('_'))
                        match=bool(re.search(r'(?<![A-Za-z0-9])(?:'+re.escape(token)+'|'+re.escape(camel)+r')(?![A-Za-z0-9])',name))
                    if match:hits.append(role)
                if not hits:continue
                if len(hits)!=1:raise RuntimeError('ambiguous target kernel: '+name)
                role=hits[0];impl='split' if role=='plan' else role
                if impl not in expected:raise RuntimeError('unexpected target implementation: '+impl)
                start,duration=float(row[sk]),float(row[dk])
                if not math.isfinite(start) or not math.isfinite(duration) or duration<=0:raise RuntimeError('invalid device time')
                records.append(dict(implementation=impl,role=role,kernel=name,start_us=start,duration_us=duration,
                    task=row.get('Task Type',''),stream=row.get('Stream ID'),device=row.get('Device ID'),csv=str(path)))
    for key in ('stream','device'):
        if len({r[key] for r in records if r[key] is not None})>1:raise RuntimeError('mixed target '+key)
    schema={}
    for impl in expected:
        counts=Counter((r['role'],r['kernel'],r['task']) for r in records if r['implementation']==impl)
        if not counts or any(n!=repeat for n in counts.values()):
            raise RuntimeError(f'{impl}: target component sample mismatch: {counts}; inventory={sorted(inventory)}')
        if len(counts)>3:raise RuntimeError(f'{impl}: unexpected number of target components')
        if impl=='split' and not any(k[0]=='plan' for k in counts):raise RuntimeError('independent Plan missing')
        schema[impl]=counts
    records.sort(key=lambda r:r['start_us']);position=0;pairs=[]
    for impl in orders:
        count=len(schema[impl]);group=records[position:position+count];position+=count
        actual=Counter((r['role'],r['kernel'],r['task']) for r in group)
        if any(r['implementation']!=impl for r in group) or set(actual)!=set(schema[impl]) or any(v!=1 for v in actual.values()):
            raise RuntimeError('target invocation/component order mismatch')
        start=min(r['start_us'] for r in group);end=max(r['start_us']+r['duration_us'] for r in group)
        result=dict(implementation=impl,span_us=end-start,busy_us=interval_union(group),components=group)
        if impl=='split':
            indexer=[r for r in group if r['role']=='split'];plan=[r for r in group if r['role']=='plan']
            if not indexer or len(plan)!=1:raise RuntimeError('split Indexer/Plan cardinality mismatch')
            finish=max(r['start_us']+r['duration_us'] for r in indexer)
            gap=plan[0]['start_us']-finish
            tolerance=max(0.05,4*math.ulp(max(abs(finish),abs(plan[0]['start_us']))))
            if gap < -tolerance:raise RuntimeError('Plan starts before ordered Indexer finishes')
            result.update(indexer_span_us=finish-start,plan_us=plan[0]['duration_us'],between_kernel_gap_us=max(0.,gap))
        if pairs:
            prev=pairs[-1]
            prev_end=max(r['start_us']+r['duration_us'] for r in prev['components'])
            tolerance=max(0.05,4*math.ulp(max(abs(start),abs(prev_end))))
            if start<prev_end-tolerance:raise RuntimeError('different invocations overlap unexpectedly')
        pairs.append(result)
    if position!=len(records):raise RuntimeError('unconsumed target records')
    return pairs,sorted(inventory)
