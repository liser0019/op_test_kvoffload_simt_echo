"""8K容量边界及真实设备状态续用；只生成数据，不执行NPU。"""
import numpy as np
from joint_plan_cases import make_case,sort_topk


def make_capacity_case(mtp=7,scenario='cold_reset',seed=2026):
    return sort_topk(make_case(1,2048,8192,262144,mtp,scenario,'overlap40',seed,64,history_pool_tokens=8192))


def capacity_cases():
    c=make_capacity_case()
    for row in range(8):c['topk_indices'][row]=100000+(row//2)*2048+np.arange(2048,dtype=np.int32)
    yield 'capacity8k/union_exact8192',c
    c={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in c.items()}
    c['topk_indices'][-1,-1]=108192
    yield 'capacity8k/union8193_fallback',c
    c=make_capacity_case()
    for row in range(8):c['topk_indices'][row]=100000+row*2048+np.arange(2048,dtype=np.int32)
    yield 'capacity8k/union16384_fallback',c
    c=make_capacity_case();c['topk_indices'][:]=100000
    yield 'capacity8k/repeated_packet_boundaries',c
    c=make_capacity_case(scenario='full_cache_75pct');c['topk_indices'][:]=-1
    yield 'capacity8k/no_visible_queries',c


def capacity_transitions():
    # 同一形状：联合冷启动→直接续用→并集溢出转private→重新联合→请求变更。
    base=make_capacity_case()
    for step in range(5):
        c={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in base.items()}
        if step==2:
            for row in range(8):c['topk_indices'][row]=100000+row*2048+np.arange(2048,dtype=np.int32)
        if step==4:c['req_ids']+=99
        yield f'capacity8k/continuous-{step}',c
