"""Compile generated host resource functions and audit factorial source gates."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
from scripts.prepare_qli_fusion import FACTORIALS,NATIVE_SOURCE,NAMES,ORDERED4,TOPK_HEADER,source_id

RES_MM_S1B4=65536
INDICES=8448
MISC_S1B4=23552
TOPK=89344
PLAN_C8192=123648


def check_scale_reset(vector_source):
    """Structural WAW guard for every generated Vector-reuse path."""
    body=vector_source[vector_source.index('::ProcessVec1('):]
    clear=body.index('Duplicate(kScaleUB_')
    load=body.index('GetKeyScale(',clear)
    if 'PipeBarrier<PIPE_ALL>();' not in body[clear:load]:
        raise RuntimeError('kScale reset must complete across pipes before GetKeyScale MTE2 writes')


def check_expected_source(root,role,name,mode,factors):
    mode3=mode==3
    vector=(root/'op_kernel/arch35/quant_lightning_indexer_service_vector_arch35.h').read_text()
    wrapper=(root/'op_kernel/plan/sparse_kv_plan_fused.h').read_text()
    has_sort='AscendC::Sort<uint32_t>' in vector
    copyback='DataCopy(indicesOutLocal_, orderedIndices' in vector
    has_reuse='pipe->InitBuffer(weightBuf_' not in vector
    hot='#define QLI_HOT_UB_MAP 1' in wrapper
    if has_sort!=factors['final_sort'] or copyback!=factors['final_sort']:
        raise RuntimeError(f'{role}: final-sort gate mismatch')
    if factors['final_sort'] and (root/TOPK_HEADER).read_bytes()!=(root.parents[1]/NATIVE_SOURCE/TOPK_HEADER).read_bytes():
        raise RuntimeError(f'{role}: native score TopK header mismatch')
    if has_reuse!=(factors['vector_reuse'] or mode in (1,2)):
        raise RuntimeError(f'{role}: Vector-only reuse gate mismatch')
    if hot!=factors['hot_ub_map']:
        raise RuntimeError(f'{role}: C hot-map gate mismatch')
    if factors['hot_ub_map'] and not mode3:
        raise RuntimeError(f'{role}: C hot map must use rankpipe AC storage')
    if has_reuse:
        if 'resMm1Buf_' not in vector or 'resMm1Buf_.Get' not in vector:
            raise RuntimeError(f'{role}: Cube resMm must stay independently allocated')
        check_scale_reset(vector)
        resource=(root/'op_host/fused_resource.h').read_text()
        if 'const uint64_t fusedShared=(misc+topk)>plan?(misc+topk):plan;' not in resource:
            raise RuntimeError(f'{role}: host shared-arena formula missing')
        if 'const uint64_t miscBytes=3U*weightBytes+scaleBytes+outBytes;' not in vector:
            raise RuntimeError(f'{role}: device shared-arena formula missing')
    if hot:
        plan=(root/'op_kernel/plan/plan.h').read_text()
        anchors=(
            'const bool hotUbMap=QLI_USE_HOT_UB_MAP(c,topk,end-first);',
            '#define QLI_USE_HOT_UB_MAP(c,t,r) (QLI_HOT_UB_MAP && (c)==8192U && (t)==2048U && (r)==4)',
            'auto* owner=data+QLI_OWNER_OFFSET(c);',
            'auto* flags=hotUbMap?reinterpret_cast<__ubuf__ uint16_t*>(owner+12288U):map;',
            'directRank?reinterpret_cast<__ubuf__ uint16_t*>(owner+16384U):part',
            'slotLookup has consumed firstUid',
            'packed=hotUbMap?reinterpret_cast<__ubuf__ uint32_t*>(data)[pair]',
            'const uint32_t flagsOffset=hotUbMap?ow+12288U:0U;',
            'DataCopyPad(mt,local[PLAN_SCAN_STORAGE_ELEMENTS+EIGHT_CTX],tailCopy);',
        )
        for anchor in anchors:
            if anchor not in plan:
                raise RuntimeError(f'{role}: C layout anchor missing: {anchor}')


def expected_total(s1_base,reuse):
    if reuse:
        return RES_MM_S1B4 if s1_base==4 else 32768, RES_MM_S1B4 if s1_base==4 else 32768
    return None,None


def main():
    p=argparse.ArgumentParser();p.add_argument('--upstream',required=True,type=Path);a=p.parse_args()
    manifest=json.loads((a.upstream/'qli_fusion_manifest.json').read_text())
    if manifest['source_id']!=source_id():raise RuntimeError('stale generated upstream overlay')
    cases=[]
    for mode,name in NAMES.items():
        factors=manifest['factorial_factors']['c0'] if mode==3 else dict(final_sort=False,vector_reuse=False,hot_ub_map=False)
        cases.append((('c0' if mode==3 else ('split' if mode==0 else 'abc'[mode-1])),mode,name,factors))
    for role,(name,factors) in FACTORIALS.items():
        if role!='c0':
            cases.append((role,3,name,factors))
    cases.append(('ordered4',0,ORDERED4,dict(final_sort=False,vector_reuse=False,hot_ub_map=False)))
    records=[]
    for role,mode,name,factors in cases:
        host=a.upstream/'attention'/name/'op_host'
        root=host.parent
        s1_base=4 if name==ORDERED4 else (2 if mode==0 or (mode in (1,2)) else None)
        if name==ORDERED4:
            kernel=root/'op_kernel'
            hook=(kernel/'quant_lightning_indexer_fused_hook.h').read_text()
            if 'using OrderedFusedPlan = OrderedFusedPlanHook<false>;' not in hook:
                raise RuntimeError('ordered4 must compile with Plan disabled')
            tiling=(host/f'{name}_tiling.cpp').read_text()
            if 'fusedS1BaseSize_=4U;' not in tiling or 'fusedS1BaseSize_ = 2U;' in tiling:
                raise RuntimeError('ordered4 must use four-row tiles without falling back')
        check_expected_source(root,role,name,mode,factors)
        s1_values=(2,4) if role in FACTORIALS else ((4,) if s1_base==4 else (s1_base,))
        for probe_s1 in s1_values:
            with tempfile.TemporaryDirectory() as td:
                td=Path(td);src=td/'probe.cpp';exe=td/'probe'
                source='''#include <iostream>
#include "fused_resource.h"
#include "qli_plan_resources.h"
int main(){for(unsigned c:{4096U,8192U,16384U}){
 auto plan=qli_fusion_resource::LocalBytes(c);
 auto r=qli_fused_host::ComputeResourceBytes(S1_BASE,32,2048,plan,qli_fused_host::QliDataPath::FP8_BF16);
 std::cout<<c<<" "<<plan<<" "<<r.total<<" "<<(r.total<=216U*1024U)<<"\\n";
}}
'''
                src.write_text(source.replace('qli_fusion_resource',name+'_resource')
                               .replace('qli_fused_host',name+'_host').replace('S1_BASE',str(probe_s1)))
                subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror','-I'+str(host),str(src),'-o',str(exe)],check=True)
                for line in subprocess.check_output([str(exe)],text=True).splitlines():
                    c,plan,total,fits=map(int,line.split())
                    record={'role':role,'operator':name,'storage':mode,'capacity':c,
                            's1_base':probe_s1,'plan_bytes':plan,'fusion_bytes':total,'fits_216k':bool(fits),**factors}
                    if role in FACTORIALS and c==8192:
                        if factors['vector_reuse']:
                            if probe_s1==4 and total!=197632:
                                raise RuntimeError(f'{role}: S1Base4 B total expected 197632, got {total}')
                            if probe_s1==2 and total!=164864:
                                raise RuntimeError(f'{role}: S1Base2 B total expected 164864, got {total}')
                        else:
                            if probe_s1==4 and total!=221184:
                                raise RuntimeError(f'{role}: S1Base4 non-B total expected 221184, got {total}')
                            if probe_s1==2 and total!=184832:
                                raise RuntimeError(f'{role}: S1Base2 non-B total expected 184832, got {total}')
                    records.append(record)
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
