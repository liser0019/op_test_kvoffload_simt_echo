"""Strict invocation grouping, including overlapping AIC/AIV components."""
import csv
import math
import re
from collections import Counter
from decimal import Decimal, InvalidOperation

TOKENS={'split':'quant_lightning_indexer_plan_split','a':'quant_lightning_indexer_plan_a',
        'b':'quant_lightning_indexer_plan_b','c':'quant_lightning_indexer_plan_c',
        'c0':'quant_lightning_indexer_plan_c',
        'A':'quant_lightning_indexer_plan_cfa','B':'quant_lightning_indexer_plan_cfb',
        'C':'quant_lightning_indexer_plan_cfc','AB':'quant_lightning_indexer_plan_cfab',
        'AC':'quant_lightning_indexer_plan_cfac','BC':'quant_lightning_indexer_plan_cfbc',
        'ABC':'quant_lightning_indexer_plan_cfabc',
        'peer':'quant_lightning_indexer_ordered_fused',
        'peer4':'quant_lightning_indexer_plan_peer4',
        'ordered_native':'quant_lightning_indexer_ordered',
        'ordered4':'quant_lightning_indexer_plan_split4',
        'native_shell':'quant_lightning_indexer_plan_native_shell',
        'native_fused':'quant_lightning_indexer_plan_native_fused',
        'native_group':'quant_lightning_indexer_plan_native_group',
        'native_group4':'quant_lightning_indexer_plan_native_group4',
        'native_pair4':'quant_lightning_indexer_plan_native_pair4',
        'native_pair_auto':'quant_lightning_indexer_plan_native_adaptive',
        'final4':'quant_lightning_indexer_plan_final4',
        'topk_control4':'quant_lightning_indexer_plan_topk_control4',
        'baseline':'quant_lightning_indexer',
        'plan':'sparse_kv_plan_joint_kernel_rankpipe_ac_512'}
TIMING_ARITHMETIC='decimal_relative_v1'
SPLIT_ROLES = frozenset({'split', 'split4', 'final4'})


def kernel_pattern(role, token):
    camel = (f'QuantLightningIndexerPlanCF{role}' if role in {'A','B','C','AB','AC','BC','ABC'}
             else ''.join(word.title() for word in token.split('_')))
    snake = re.escape(token)
    if role == 'baseline':
        # The native snake-case name is a prefix of several other operators.
        snake += r'(?!_(?:plan|ordered|v[0-9]+)(?:_|$))'
    elif role == 'ordered_native':
        # The independent operator name prefixes the fused registration.
        snake += r'(?!_(?:fused|plan)(?:_|$))'
    return re.compile(r'(?<![A-Za-z0-9])(?:' + snake + '|' + camel + r')(?![A-Za-z0-9])')


KERNEL_PATTERNS = {role: kernel_pattern(role, token) for role, token in TOKENS.items()}


def interval_union(events):
    """Union without changing numeric types (Decimal inside the CSV parser)."""
    spans=sorted((r['start_us'],r['start_us']+r['duration_us']) for r in events)
    if not spans:return 0.
    end=spans[0][0];total=end-end
    for lo,hi in spans:
        if hi>end:
            total+=hi-max(lo,end);end=hi
    return total


def timestamp_tolerance(*absolute_times):
    # Some profiler exporters already quantize absolute timestamps as doubles.
    # Preserve the server-tested allowance, but do not use doubles for timing
    # arithmetic. Its value and any clamped overlap are exposed in each pair.
    return Decimal(str(max(0.05,4*math.ulp(float(max(map(abs,absolute_times)))))))


def public_component(record):
    return {**record,'start_us':float(record['start_us']),
            'duration_us':float(record['duration_us'])}


def assign_implementation_roles(records, orders, launches):
    """Disambiguate a common native Indexer by its scheduled Plan partner."""
    expected = Counter(kernel for implementation in orders for kernel in launches[implementation])
    schema = {}
    for kernel, count in expected.items():
        seen = Counter((r['kernel'], r['task']) for r in records if r['role'] == kernel)
        if not seen or any(n != count for n in seen.values()):
            raise RuntimeError(f'{kernel}: expected {count} launches per component, found {seen}')
        schema[kernel] = len(seen)
    position = 0
    for implementation in orders:
        kernels = launches[implementation]
        for index, kernel in enumerate(kernels):
            end = position + schema[kernel]
            group = records[position:end]
            signatures = {(r['kernel'], r['task']) for r in group}
            if len(group) != schema[kernel] or len(signatures) != len(group) or any(
                    r['role'] != kernel for r in group):
                raise RuntimeError(f'{implementation}: unexpected kernel or order at row {position}')
            for record in group:
                record['implementation'] = implementation
                record['role'] = implementation if index == 0 else 'plan'
            position = end
    if position != len(records):
        raise RuntimeError('unconsumed ordered chain records')


def read_csvs(paths,orders,repeat,exact=None,split_roles=SPLIT_ROLES,*,launches=None,plan_kernels=None):
    records=[];inventory=set();expected=Counter(orders)
    if any(v!=repeat for v in expected.values()):raise ValueError('invocation schedule/count mismatch')
    patterns = KERNEL_PATTERNS
    expected_kernels = expected
    if launches is not None:
        if set(launches) != set(expected):
            raise ValueError('invalid ordered chain roles')
        expected_kernels = {kernel for launch in launches.values() for kernel in launch}
        patterns = {role:pattern for role,pattern in KERNEL_PATTERNS.items() if role != 'plan'}
        patterns.update({role:kernel_pattern(role, name) for role,name in plan_kernels.items()})
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
                for role,pattern in patterns.items():
                    if exact:
                        key='split4' if role=='ordered4' and 'split4' in expected else role
                        match=name in exact.get(key,exact.get(role,[]))
                    else:
                        match=bool(pattern.search(name))
                    if not match:continue
                    # c and c0 are the same historical operator exposed under two
                    # factorial role names. Select only the requested spelling.
                    if role=='c' and 'c0' in expected and 'c' not in expected:continue
                    if role=='c0' and 'c0' not in expected and 'c' in expected:continue
                    hits.append(role)
                if not hits:continue
                if len(hits)!=1:raise RuntimeError('ambiguous target kernel: '+name)
                role=hits[0]
                if role=='ordered4' and 'split4' in expected:
                    role='split4'
                if role not in expected_kernels and (role!='plan' or launches is not None):
                    raise RuntimeError('unexpected target implementation: '+role)
                try:
                    start,duration=Decimal(row[sk]),Decimal(row[dk])
                except (InvalidOperation,TypeError,ValueError) as exc:
                    raise RuntimeError('invalid device time') from exc
                if (not start.is_finite() or not duration.is_finite() or duration<=0 or
                    not math.isfinite(float(start)) or not math.isfinite(float(duration)) or float(duration)<=0):
                    raise RuntimeError('invalid device time')
                records.append(dict(implementation=role,role=role,kernel=name,start_us=start,duration_us=duration,
                    absolute_start_us=row[sk],csv_duration_us=row[dk],
                    task=row.get('Task Type',''),stream=row.get('Stream ID'),
                    device=row.get('Device ID') or row.get('Device_id'),
                    block_num=row.get('Block Num'),mix_block_num=row.get('Mix Block Num'),csv=str(path)))
    if not records:raise RuntimeError('no target kernel records')
    # Subtract the exact common origin BEFORE any duration/interval arithmetic.
    # Components returned to JSON have relative start_us plus the original CSV
    # strings, and each invocation carries this exact origin explicitly.
    origin=min(r['start_us'] for r in records)
    for record in records:record['start_us']-=origin
    for key in ('stream','device'):
        if len({r[key] for r in records if r[key] is not None})>1:raise RuntimeError('mixed target '+key)
    records.sort(key=lambda r:r['start_us'])
    if launches is not None:
        assign_implementation_roles(records, orders, launches)
    else:
        # Every split invocation launches the same Plan. Pair it with the preceding
        # Indexer on this stream before counting components by implementation.
        pending=None
        for record in records:
            role=record['role']
            if role=='plan':
                if pending is None:
                    raise RuntimeError('Plan has no preceding split Indexer')
                record['implementation']=pending
                pending=None
            elif role in split_roles:
                if pending is not None and pending!=role:
                    raise RuntimeError('split Indexer changed before Plan was recorded')
                pending=role
            elif pending is not None:
                raise RuntimeError('split Plan missing before next implementation')
        if pending is not None:
            raise RuntimeError('independent Plan missing at end of sampling')
    schema={}
    for impl in expected:
        counts=Counter((r['role'],r['kernel'],r['task']) for r in records if r['implementation']==impl)
        if not counts or any(n!=repeat for n in counts.values()):
            raise RuntimeError(f'{impl}: target component sample mismatch: {counts}; inventory={sorted(inventory)}')
        if len(counts)>3:raise RuntimeError(f'{impl}: unexpected number of target components')
        if impl in split_roles and not any(k[0]=='plan' for k in counts):raise RuntimeError('independent Plan missing')
        schema[impl]=counts
    position=0;pairs=[];previous_end=None
    for impl in orders:
        count=len(schema[impl]);group=records[position:position+count];position+=count
        actual=Counter((r['role'],r['kernel'],r['task']) for r in group)
        if any(r['implementation']!=impl for r in group) or set(actual)!=set(schema[impl]) or any(v!=1 for v in actual.values()):
            raise RuntimeError('target invocation/component order mismatch')
        start=min(r['start_us'] for r in group);end=max(r['start_us']+r['duration_us'] for r in group)
        result=dict(implementation=impl,span_us=float(end-start),busy_us=float(interval_union(group)),
            timestamp_origin_us=str(origin),component_start_basis='relative_to_timestamp_origin_us',
            components=[public_component(r) for r in group])
        if impl in split_roles:
            indexer=[r for r in group if r['role']==impl];plan=[r for r in group if r['role']=='plan']
            if not indexer or len(plan)!=1:raise RuntimeError('split Indexer/Plan cardinality mismatch')
            finish=max(r['start_us']+r['duration_us'] for r in indexer)
            gap=plan[0]['start_us']-finish
            tolerance=timestamp_tolerance(origin+finish,origin+plan[0]['start_us'])
            if gap < -tolerance:raise RuntimeError('Plan starts before ordered Indexer finishes')
            result.update(indexer_span_us=float(finish-start),plan_us=float(plan[0]['duration_us']),
                between_kernel_gap_us=float(max(Decimal(0),gap)),reported_between_kernel_gap_us=float(gap),
                gap_clamped_us=float(max(Decimal(0),-gap)),gap_tolerance_us=float(tolerance))
        if previous_end is not None:
            tolerance=timestamp_tolerance(origin+start,origin+previous_end)
            separation=start-previous_end
            if separation < -tolerance:raise RuntimeError('different invocations overlap unexpectedly')
            result.update(previous_invocation_overlap_us=float(max(Decimal(0),-separation)),
                          invocation_order_tolerance_us=float(tolerance))
        pairs.append(result)
        previous_end=end
    if position!=len(records):raise RuntimeError('unconsumed target records')
    return pairs,sorted(inventory)
