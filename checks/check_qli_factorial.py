"""Structural and layout audit for c0/A/B/C/AB/AC/BC/ABC.

This is a source-level gate. NPU compilation and execution are separate gates.
No Block Num is accepted as evidence for S1Base or for the C UB layout.
"""
import argparse
import json
from pathlib import Path
from scripts.prepare_qli_fusion import FACTORIAL_NAMES,NATIVE_SOURCE,TOPK_HEADER,source_id


def one(text,anchor,label):
    count=text.count(anchor)
    if count!=1:raise RuntimeError(f'{label}: expected one anchor, got {count}: {anchor[:100]}')


def check_c_layout(root,name):
    plan=(root/'op_kernel/plan/plan.h').read_text()
    wrapper=(root/'op_kernel/plan/sparse_kv_plan_fused.h').read_text()
    if '#define QLI_HOT_UB_MAP 1' not in wrapper:
        raise RuntimeError('C wrapper does not enable QLI_HOT_UB_MAP')
    for anchor,count in (
        ('const bool hotUbMap=QLI_USE_HOT_UB_MAP(c,topk,end-first);',1),
        ('const bool hotUbMap=QLI_USE_HOT_UB_MAP(d.capacity,d.topk,end-first);',1),
        ('#define QLI_USE_HOT_UB_MAP(c,t,r) (QLI_HOT_UB_MAP && (c)==8192U && (t)==2048U && (r)==4)',1),
        ('auto* owner=data+QLI_OWNER_OFFSET(c);',2),
        ('auto* part=reinterpret_cast<__ubuf__ uint16_t*>(owner+c);',2),
        ('auto* flags=hotUbMap?reinterpret_cast<__ubuf__ uint16_t*>(owner+12288U):map;',1),
        ('directRank?reinterpret_cast<__ubuf__ uint16_t*>(owner+16384U):part',1),
        ('if(!hotUbMap){for(uint32_t i=t;i<q/2;i+=SORTED_THREADS)workspace[qb/2+i]=data[i];}',1),
        ('packed=hotUbMap?reinterpret_cast<__ubuf__ uint32_t*>(data)[pair]',1),
        ('const uint32_t flagsOffset=hotUbMap?ow+12288U:0U;',1),
        ('DataCopyPad(mt,local[PLAN_SCAN_STORAGE_ELEMENTS+EIGHT_CTX],tailCopy);',1),
    ):
        if plan.count(anchor)!=count:raise RuntimeError(f'C anchor count changed: {anchor[:100]}')
    # Relative to owner: 0/32K/48K/64K and direct firstUid 32K..64K.
    owner=(8192//2+2048)*4
    owner_bytes=8192*4
    part=owner+owner_bytes
    flags=owner+12288*4
    slot=owner+16384*4
    first_uid=owner+8192*4
    assert (owner,part,flags,slot)==(24576,57344,73728,90112)
    assert first_uid==57344 and first_uid+8192*4==90112
    assert flags==73728 and flags+8192*2==90112
    assert flags-first_uid==16384
    assert slot==90112 and slot+8192*2==106496
    assert slot+8192*2<=120*1024
    # The firstUid read/barrier must precede any flags write in the direct path.
    direct=plan[plan.index('if(directRank){'):plan.index('}else{',plan.index('if(directRank){'))]
    slot_write=direct.index('slotLookup[slot]')
    firstuid_done=direct.index('slotLookup has consumed firstUid',slot_write)
    flags_write=plan.index('if(order<c)flags[order]=valid',plan.index('if(directRank){'))
    if not slot_write<firstuid_done<flags_write:raise RuntimeError('C firstUid/flags lifetime order changed')
    # Binary slotLookup aliases part, while flags is separate. The existing V_S
    # stage handoff must remain between Phase6 and compaction.
    if 'RankPipeCompactVf>(dim3(SORTED_THREADS),ctx+EIGHT_CTX,ctx,scan' not in plan:
        raise RuntimeError('RankPipeCompactVf launch missing')
    if 'asc_syncthreads();\n  }\n  for(uint32_t i=t;i<u;i+=SORTED_THREADS)owner[i]=-1;' not in plan:
        raise RuntimeError('binary slotLookup/part phase boundary changed')
    # Phase9 must read the UB map and output staging must follow it.
    phase9=plan.index('packed=hotUbMap?')
    staging=plan.index('DataCopyPad(mt,local[PLAN_SCAN_STORAGE_ELEMENTS+EIGHT_CTX],tailCopy);')
    if not phase9<staging:raise RuntimeError('tail staging precedes Phase9 map consumption')


def check_vector(root,role,name,factors):
    vector=(root/'op_kernel/arch35/quant_lightning_indexer_service_vector_arch35.h').read_text()
    has_sort='AscendC::Sort<uint32_t>' in vector
    copyback='DataCopy(indicesOutLocal_, orderedIndices, topkCount_)' in vector
    if has_sort!=factors['final_sort'] or copyback!=factors['final_sort']:
        raise RuntimeError(f'{role}: A gate mismatch')
    if has_sort:
        one(vector,'#include "include/adv_api/sort/sort.h"','A sort include')
        one(vector,'if (topkCount_ == 2048U && qliValidS2Len >= topkCount_)','A tie/full-row gate')
        sort=vector.index('AscendC::Sort<uint32_t>')
        back=vector.index('DataCopy(indicesOutLocal_, orderedIndices, topkCount_)',sort)
        publish=vector.index('SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT)',back)
        if not sort<back<publish:raise RuntimeError(f'{role}: sort/copyback/publish order changed')
        if 'PipeBarrier<PIPE_ALL>();' not in vector[back:publish]:
            raise RuntimeError(f'{role}: copyback is not fully ordered before publication')
        if 'planHook_->PublishGroup' not in vector[publish:]:
            raise RuntimeError(f'{role}: Plan publication missing')
        if 'reinterpret_cast<__ubuf__ int32_t *>(indicesOutLocal_.GetPhyAddr())' not in vector[publish:]:
            raise RuntimeError(f'{role}: Plan must receive indicesOutLocal_, not sort scratch')
    reuse='pipe->InitBuffer(weightBuf_' not in vector
    if reuse!=(factors['vector_reuse'] or factors.get('legacy_reuse',False)):
        raise RuntimeError(f'{role}: B gate mismatch')
    if reuse:
        if 'resMm1Buf_.Get' not in vector:raise RuntimeError(f'{role}: resMm lost its independent buffer')
        if 'pipe->InitBuffer(indicesOutBuf_' not in vector:raise RuntimeError(f'{role}: indices buffer lost independence')
        one(vector,'const uint64_t miscBytes=3U*weightBytes+scaleBytes+outBytes;','B device arena')
        one(vector,'const uint64_t arenaBytes=QLICommon::Max(miscBytes+topkScratchBytes,planScratchBytes);','B device max')
        if vector.count('PipeBarrier<PIPE_ALL>();')<1:raise RuntimeError(f'{role}: K-scale pipeline handoff missing')
        if vector.count('SetFlag<HardEvent::MTE2_V>(drain);WaitFlag<HardEvent::MTE2_V>(drain);')!=2:
            raise RuntimeError(f'{role}: both normal and invalid Plan publication hooks need Drain')


def main():
    p=argparse.ArgumentParser();p.add_argument('--upstream',required=True,type=Path);a=p.parse_args()
    manifest=json.loads((a.upstream/'qli_fusion_manifest.json').read_text())
    if manifest['source_id']!=source_id():raise RuntimeError('stale generated overlay')
    if manifest['factorial_roles']!=['c0','A','B','C','AB','AC','BC','ABC']:
        raise RuntimeError('factorial role order/identity changed')
    if manifest['factorial_operators']!=FACTORIAL_NAMES:raise RuntimeError('factorial operator mapping changed')
    names=list(FACTORIAL_NAMES.values())
    if len(set(names))!=8:raise RuntimeError('factorial operator identities must be unique')
    native_topk=(a.upstream/NATIVE_SOURCE/TOPK_HEADER).read_bytes()
    for role,name in FACTORIAL_NAMES.items():
        root=a.upstream/'attention'/name
        factors=dict(manifest['factorial_factors'][role])
        tiling=(root/'op_host'/f'{name}_tiling.cpp').read_text()
        if 'fusedS1BaseSize_ = tSize_ <= 64 ? 2U : 4U;' not in tiling:
            raise RuntimeError(f'{role}: host auto S1Base rule changed')
        check_vector(root,role,name,factors)
        if ((root/TOPK_HEADER).read_bytes()==native_topk)!=factors['final_sort']:
            raise RuntimeError(f'{role}: native TopK selection mismatch')
        if factors['hot_ub_map']:check_c_layout(root,name)
        else:
            wrapper=(root/'op_kernel/plan/sparse_kv_plan_fused.h').read_text()
            if 'QLI_HOT_UB_MAP 1' in wrapper:raise RuntimeError(f'{role}: C leaked into non-C variant')
    print('QLI_FACTORIAL_STRUCTURE_PASSED')


if __name__=='__main__':main()
