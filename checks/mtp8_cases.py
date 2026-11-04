"""小而有针对性的8行回归：重复/分块/可见性/回退/连续状态。"""
import numpy as np
from joint_plan_cases import make_case,sort_topk


def cases(include_extreme=False):
    for topk,cap in ((7,16),(63,128),(65,256),(513,4096),(2048,4096)):
        for scenario in ('cold_reset','full_cache_75pct'):
            c=make_case(1,topk,cap,262144,7,scenario,'shared',block_size=8 if topk!=2048 else 64)
            yield f'mtp7/shared/{topk}/{scenario}',sort_topk(c)
    for kind in ('empty','mixed_empty','five_active','six_active','repeated','uneven','overflow','capacity_exact','capacity_plus1','overwide','nonpower','invalid_request','partial_active'):
        c=make_case(2,65,256,262144,7,'full_cache_75pct','shared',block_size=8)
        if kind=='empty':c['topk_indices'][:]=-1
        elif kind=='mixed_empty':c['topk_indices'][::2]=-1
        elif kind in ('five_active','six_active'):
            n=3 if kind=='five_active' else 2
            for first in (0,8):c['topk_indices'][first:first+n]=-1
        elif kind=='repeated':c['topk_indices'][:]=np.arange(65)//3+100000
        elif kind=='uneven':
            for row in range(c['rows']):c['visible_seq_lens'][row]=100000+(row%8)*8;c['topk_indices'][row]=np.arange(65)+100000
        elif kind=='overflow':
            for row in range(c['rows']):c['topk_indices'][row]=np.arange(65)+(row%8)*65+100000
        elif kind in ('capacity_exact','capacity_plus1'):
            u=256+(kind=='capacity_plus1')
            for row in range(c['rows']):c['topk_indices'][row]=(np.arange(65)+(row%8)*65)%u+100000
        elif kind=='overwide':c['token_to_req'][:]=0
        elif kind=='nonpower':c['token_to_req'][:]=[0]*3+[1]*5+[0]*8
        elif kind=='invalid_request':c['token_to_req'][8:]=-1
        elif kind=='partial_active':c['active_rows']=11
        yield kind,sort_topk(c)
    # Long repeated runs straddle both hardware packet boundaries and SIMD64 seams.
    c=make_case(1,2048,4096,262144,7,'cold_reset','shared')
    for r in range(8):c['topk_indices'][r]=np.arange(2048)//3+100000+(r%2)*100
    yield 'production/repeated_packet_seams',sort_topk(c)

    if include_extreme:
        c=make_case(1,2048,4096,262144,7,'cold_reset','shared')
        c['topk_indices'][:]=100000
        yield 'production/all_equal_across_unknown_tails',sort_topk(c)


def transitions():
    base=make_case(2,65,256,262144,7,'cold_reset','shared',block_size=8)
    for step in range(10):
        c=make_case(2,65,256,262144,7,'cold_reset','shared',block_size=8)
        c['topk_indices'][:]=np.arange(65)+100000+(step//2)*100
        if step>=3:c['req_ids']+=99
        if step==2:
            for row in range(c['rows']):c['topk_indices'][row]=np.arange(65)+100000+(row%8)*65
        if step==4:c['token_to_req'][:]=np.arange(c['rows'])%2 # joint → independent singletons
        if step==6:c['active_rows']=11 # partial final group of3 must fall back
        if step==7:c['stable_prefix_lens'][:]=100050
        if step in (8,9):
            width=4 if step==8 else 2
            c['token_to_req'][:]=np.tile(np.repeat([0,1],width),c['rows']//(2*width))
        yield f'continuous-{step}',sort_topk(c)
