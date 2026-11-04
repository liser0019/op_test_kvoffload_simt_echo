"""固定初态采样的参考结果复用；连续状态仍调用完整模型。无全局缓存。"""
from copy import deepcopy
import hashlib
import numpy as np
from joint_plan_cases import INPUTS


def identity(case,before):
    h=hashlib.sha256()
    for k in ('topk','capacity','max_token','block_size','rows','requests','mtp','active_rows'):
        h.update(f'{k}={case.get(k)};'.encode())
    for prefix,values in (('input',case),('before',before)):
        keys=INPUTS if prefix=='input' else sorted(before)
        for k in keys:
            a=np.ascontiguousarray(values[k]);h.update(f'{prefix}/{k}/{a.dtype}/{a.shape}'.encode());h.update(a.tobytes())
    return h.hexdigest()


def exact_workload_key(case):
    return identity(case,{k:case[k] for k in ('last_req_ids','slot_to_token','lru_slots')})


class FixedExpectation:
    """首个结果经原独立模型全量认证后，复用其确定性期望。

    只用于固定输入且每次restore相同初态的sorted/independent采样。
    与原模型相同：private描述符仅有效前缀有定义；joint检查完整尾部。
    不是用上一次结果作为下一次状态，不向设备写回任何期望结果。
    """
    def __init__(self,case,before,actual,result):
        self.key=identity(case,before)
        self.expected={k:v.copy() for k,v in actual.items()}
        self.result=deepcopy(result)
        self.active=min(case['rows'],max(0,case.get('active_rows',case['rows'])))
        self.full_tail=set()
        for g in result['groups']:
            if g['path']=='joint':self.full_tail.update(range(g['first'],g['end']))

    def check(self,case,before,actual):
        if identity(case,before)!=self.key:raise RuntimeError('fixed expectation input/initial-state identity changed')
        for k,gold in self.expected.items():
            if k not in ('miss_tokens','miss_slots'):
                np.testing.assert_array_equal(actual[k],gold,err_msg='fixed-reference/'+k)
            else:
                np.testing.assert_array_equal(actual[k][self.active:],gold[self.active:],err_msg='fixed-inactive/'+k)
                for row in range(self.active):
                    n=gold.shape[1] if row in self.full_tail else int(self.expected['miss_count'][row])
                    np.testing.assert_array_equal(actual[k][row,:n],gold[row,:n],err_msg=f'fixed-reference/{k}/row{row}')
        return deepcopy(self.result)


def batched_readback(bundle):
    """每种dtype一次D2H；保护区仍逐个比对，workspace只读两端保护区。"""
    torch=bundle['torch'];items=[]
    for k,v in bundle['inputs'].items():items.append(('input/'+k,v))
    for k,v in bundle['buffers'].items():
        items.extend([('left/'+k,v[:16]),('right/'+k,v[-16:])])
    keys=('last_req_ids','slot_to_token','lru_slots','current_slots','miss_count','miss_tokens','miss_slots')
    if 'row_modes' in bundle['state']:keys+=('row_modes','resident_rows','resident_block_table')
    items.extend(('state/'+k,bundle['state'][k]) for k in keys)
    result={}
    for dtype in dict.fromkeys(v.dtype for _,v in items):
        group=[(k,v) for k,v in items if v.dtype==dtype]
        flat=torch.cat([v.reshape(-1) for _,v in group]).cpu().numpy()
        pos=0
        for k,v in group:
            result[k]=flat[pos:pos+v.numel()].reshape(tuple(v.shape));pos+=v.numel()
    return result
