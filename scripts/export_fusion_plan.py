#!/usr/bin/env python3
"""Freeze the rankpipe AC source selection, then adapt only storage and dispatch.

The checked-in header is readable device source. This exporter must reproduce it;
it does not compile an Ascend kernel and requires no CANN installation.
"""
from pathlib import Path
import argparse
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
FLAGS = dict(JOINT_OPT_MASK=72, JOINT_COLD_FINISH=1, JOINT_REFINE=1,
             JOINT_REFINE_BITS=11, JOINT_HOT=1, JOINT_HOT_BITS=10,
             JOINT_HOT_PART_ONLY=0, JOINT_HOT_MISS_ONLY=0,
             JOINT_MTP8=1, JOINT_MTP7_OPT=1, JOINT_MTP7_REUSE=1,
             JOINT_MTP7_TILE=0, JOINT_MTP7_CAPACITY8K=0,
             JOINT_MTP7_MID_UNIQUE=0, JOINT_MTP7_FUSED_COUNT=1,
             JOINT_MTP7_FUSED_TUNE_BITS=3, JOINT_MTP7_PREFIX=0,
             JOINT_MTP7_MAP16=1, JOINT_LAYOUT16=1, JOINT_LAYOUT_HOT_BITS=15,
             JOINT_STAGE_EXEC=1, JOINT_STAGE_EXEC_BITS=15,
             JOINT_RANKPIPE=1, JOINT_RANKPIPE_BITS=5)


def replace(s, old, new, count=1):
    actual = s.count(old)
    if actual != count:
        raise RuntimeError(f'port anchor changed: expected {count}, got {actual}: {old[:100]}')
    return s.replace(old, new)


def selected_source():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for name in ('kernel_operator.h', 'cstdint', 'climits',
                     'simt_api/asc_simt.h', 'simt_api/device_atomic_functions.h',
                     'simt_api/device_sync_functions.h', 'simt_api/device_warp_functions.h',
                     'simt_api/device_functions.h'):
            p = tmp / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('')
        source = (ROOT/'csrc/joint_variants/kernel.cpp').read_text()
        source = source[:source.index('// Direct launch adaptation only:')]
        source = source.replace('"../joint_plan/', '"')
        for name in ('sparse_kv_plan_config.h', 'launch.h'):
            (tmp/name).write_bytes((ROOT/'csrc/joint_plan'/name).read_bytes())
        (tmp/'input.h').write_text(source)
        text = subprocess.check_output([
            'g++', '-E', '-P', '-x', 'c++', '-nostdinc', '-nostdinc++',
            *[f'-D{k}={v}' for k,v in FLAGS.items()], '-I'+str(tmp),
            '-I'+str(ROOT/'csrc/joint_variants'), str(tmp/'input.h')], text=True)
    # External declarations and standalone-only tiling are not the fusion ABI.
    start = text.index('namespace {')
    text = text[start:].replace('namespace {', 'namespace qli_fused_plan {', 1)
    return text


def port(text):
    original=(ROOT/'csrc/joint_variants/layout_hot_impl.h').read_text()
    controller = original[original.index('template <bool AllowJoint>\n__aicore__ inline void SortedDirectBody'):]
    controller = re.sub(
        r'^#if (JOINT_MTP3_LOOKUP_SHARDS|JOINT_MTP3_BITMAP_TEAM) // Omit from frozen fusion export\.\n'
        r'.*?^#endif // \1\n', '', controller, flags=re.M | re.S)
    start = controller.index("  const int64_t start=AscendC::GetBlockIdx(),stride=AscendC::GetBlockNum();")
    end = controller.index("  AscendC::TPipe pipe;", start)
    controller = controller[:start] + (
        "  const int64_t start=AscendC::GetBlockIdx(),stride=AscendC::GetBlockNum();if(start>=rows)return;\n"
    ) + controller[end:]
    # The exporter selects frozen AC. Its optional fused map has a separate ABI.
    begin = controller.index('#if JOINT_MTP3_MAP')
    keep = controller.index('#else // frozen rankpipe launch ABI\n', begin)
    end = controller.index('#endif // JOINT_MTP3_MAP\n', keep)
    controller = (controller[:begin] + controller[keep:end].split('\n', 1)[1] +
                  controller[end + len('#endif // JOINT_MTP3_MAP\n'):])
    text = text[:text.index('template <bool AllowJoint>\n__aicore__ inline void SortedDirectBody')]
    # Remove uncalled historical joint/hash entry points; retain the actual private
    # calculation and protocol fallback. No algorithm selection is changed there.
    text = text.replace('__gm__ int32_t *topkIndices, *stablePrefixLens, *visibleSeqLens, *tokenToReq;',
                        'FusionTopk topkIndices;\n  __gm__ int32_t *stablePrefixLens, *visibleSeqLens, *tokenToReq;')
    text = replace(text,'auto* unique=data+c/2U;\n  auto* owner=unique+c;',
        'auto unique=FusionUnique{data,workspace+qli_fusion_resource::MapWords(a.maxRequests,a.topk)+first*c,a.topkIndices,first*a.topk,c,0U,end-first==1,ctx+48};\n  auto* owner=data+qli_fusion_resource::OwnerOffset(c);')
    text = replace(text,'__ubuf__ uint16_t* map,__ubuf__ int32_t* unique,',
                        '__ubuf__ uint16_t* map,FusionUnique unique,')
    text = replace(text,'for(uint32_t i=thread;i<q-u;i+=Threads)unique[i]=-1;',
                        'for(uint32_t i=thread;i<2048U;i+=Threads)reinterpret_cast<__ubuf__ int32_t*>(map)[i]=-1;')
    text = replace(text,'for(uint32_t i=t;i<q-misses;i+=SORTED_THREADS)unique[i]=-1;',
                        'for(uint32_t i=t;i<2048U;i+=SORTED_THREADS)data[i]=-1;')
    text = replace(text,'(1U&1U) && a.maxToken<=262144',
                        '(QLI_PLAN_STORAGE!=2) && a.maxToken<=262144') if '(1U&1U)' in text else text
    text = replace(text,'(5&1U) && a.maxToken<=262144',
                        '(QLI_PLAN_STORAGE!=2) && a.maxToken<=262144') if '(5&1U)' in text else text
    # Scalar offsets used by hot classification/compaction/output.
    text = text.replace('const uint32_t ow=c/2U+c,pa=ow+c;',
                        'const uint32_t ow=qli_fusion_resource::OwnerOffset(c),pa=ow+c;')
    text = text.replace('auto* owner=data+c/2U+c;', 'auto* owner=data+qli_fusion_resource::OwnerOffset(c);')
    text = replace(text,'P map=data, unique=data+mapWords, owner=data+((mapWords+capacity+7U)&~7U);',
        'P map=data, owner=data+qli_fusion_resource::OwnerOffset(capacity);\n  auto unique=FusionUnique{data,mapWorkspace+qli_fusion_resource::MapWords(a.maxRequests,topk)+first*capacity,a.topkIndices,qb,capacity,0U,end-first==1,ctx+48};')
    # qb was originally declared after the pointers.
    text = text.replace('a.topkIndices,qb,capacity', 'a.topkIndices,first*static_cast<int64_t>(topk),capacity')
    text = replace(text,'if(valid)unique[i-static_cast<uint32_t>(ctx[48])]=token;',
                        'if(valid)unique.Set(i-static_cast<uint32_t>(ctx[48]),token,i);')
    text = replace(text,'if(uid<capacity&&added)unique[uid]=token;',
                        'if(uid<capacity&&added)unique.Set(uid,token,static_cast<uint32_t>(out[2U*i+1U]));')
    text = replace(text,'int64_t first,int64_t end,__ubuf__ uint16_t* map,FusionUnique unique,',
                        'int64_t first,int64_t end,__ubuf__ uint16_t* map,FusionUnique unique,')
    # New launch arguments are explicit; no shared/global mutable dispatch context.
    at=text.index('template <int Phase>\n__simt_vf__', text.index('inline void SortedStage('))
    before, stage = text[:at], text[at:]
    stage=replace(stage,'uint64_t jointHashCapacity, uint64_t scratchElements) {',
                        'uint64_t jointHashCapacity, uint64_t scratchElements, __ubuf__ int32_t* localTopk, int64_t localRow) {')
    stage=replace(stage,'                           topkIndices,',
                        '                           FusionTopk{topkIndices,localTopk,localRow*topk,topk},')
    text=before+stage
    # Only three used SIMD producers need storage changes. Preserve the parent
    # head detection, cross-packet last token, map writes and source-row counts.
    for name in ('RefineUniqueMapVf','RefineFusedTuneVf'):
        start=text.index('inline void '+name)
        brace=text.index('{',start);depth=1;end=brace+1
        while depth:
            depth += (text[end]=='{')-(text[end]=='}');end+=1
        body=text[start:end]
        # Ensure payload is loaded before publishing its representative position.
        if name=='RefineUniqueMapVf':
            load='Reg::Duplicate(payload,uint32_t{0});\n    Reg::Gather<uint32_t,uint32_t,uint32_t>(payload,reinterpret_cast<__ubuf__ uint32_t*>(records),odd,valid);'
            body=replace(body,load,'')
            anchor='Reg::Scatter<int32_t,uint32_t>(data,token,index,writeHead);'
            body=replace(body,anchor,load+'\n    FusionStoreUnique(data,token,payload,uid,lane,writeHead,full,mapWords,capacity,initialCount);')
        else:
            anchor='Reg::Scatter<int32_t, uint32_t>(data, token, index, writeHead);'
            # Pair loads already carry payload; tail/gather branches load it later.
            body=body.replace(anchor,
                'Reg::Gather<uint32_t,uint32_t,uint32_t>(payload,reinterpret_cast<__ubuf__ uint32_t*>(records),odd,valid);\n            FusionStoreUnique(data,token,payload,uid,lane,writeHead,full,mapWords,capacity,initialCount);')
            # In the aligned pair-load loop odd is not initialized. Remove that
            # extra gather in the first loop (the DINTLV load already supplied it).
            first=body.index('Reg::Gather<uint32_t,uint32_t,uint32_t>(payload,reinterpret_cast<__ubuf__ uint32_t*>(records),odd,valid);')
            body=body[:first]+body[first:].replace('Reg::Gather<uint32_t,uint32_t,uint32_t>(payload,reinterpret_cast<__ubuf__ uint32_t*>(records),odd,valid);','',1)
        text=text[:start]+body+text[end:]
    # Phase0 accepts the actual Indexer contract, as well as the old harness's
    # negative-prefix sorted fixtures. No sorting/copying of query inputs occurs.
    begin=text.index('        if(topk>0U && a.topkIndices[b]>=0')
    end=text.index('        if(ctx[48+r]<ctx[56+r])',begin)
    text=text[:begin]+'''        if(topk && a.topkIndices[b]>=0 && a.topkIndices[b+topk-1U]>=0 && a.topkIndices[b+topk-1U]<upper){
          ctx[48+r]=0;ctx[56+r]=static_cast<int32_t>(topk);
        }else{
        uint32_t validEnd=topk;
        if(topk && a.topkIndices[b]>=0 && a.topkIndices[b+topk-1U]<0){
          uint32_t lo=0,hi=topk;
          while(lo<hi){uint32_t mid=lo+(hi-lo)/2U;if(a.topkIndices[b+mid]>=0)lo=mid+1U;else hi=mid;}
          validEnd=lo;
        }
        ctx[48+r]=static_cast<int32_t>(SortedLower(a.topkIndices+b,validEnd,0));
        ctx[56+r]=static_cast<int32_t>(SortedLower(a.topkIndices+b,validEnd,upper));
        }
''' + text[end:]
    # Repeated -1 padding is not a duplicate *valid query*.
    text=text.replace('if(i && token==a.topkIndices[qb+i-1U])', 'if(i && token>=0 && token==a.topkIndices[qb+i-1U])')
    # Enlarged compact storage only applies to these three new candidates.
    text=text.replace('constexpr uint64_t EIGHT_UB_WORDS=204U*1024U/4U;',
                      'constexpr uint64_t EIGHT_UB_WORDS=216U*1024U/4U;')
    text=text.replace('((mapWords+capacity+7U)&~7U)', 'qli_fusion_resource::OwnerOffset(capacity)')
    text=text.replace('data+((words+args.capacity+7U)&~7U)',
                      'data+qli_fusion_resource::OwnerOffset(args.capacity)')
    text=replace(text,'  const uint32_t uniqueOffset=PLAN_SCAN_STORAGE_ELEMENTS+EIGHT_CTX+c/2U;\n  const uint32_t ownerOffset=uniqueOffset+c;',
                 '  const uint32_t uniqueOffset=0U; // map is dead: bounded descriptor-tail staging\n  const uint32_t ownerOffset=PLAN_SCAN_STORAGE_ELEMENTS+EIGHT_CTX+qli_fusion_resource::OwnerOffset(c);')
    begin=text.index('  if(n){\n    AscendC::DataCopyExtParams tailCopy')
    end=text.index('  auto done=',begin)
    text=text[:begin]+'''  for(uint32_t offset=0;offset<n;offset+=2048U){
    const uint32_t count=n-offset<2048U?n-offset:2048U;
    AscendC::DataCopyExtParams tailCopy(1,count*4U,0,0,0);
    mt.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(missTokens)+tail+offset,count);
    ms.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(missSlots)+tail+offset,count);
    AscendC::DataCopyPad(mt,local[PLAN_SCAN_STORAGE_ELEMENTS+EIGHT_CTX],tailCopy);
    AscendC::DataCopyPad(ms,local[PLAN_SCAN_STORAGE_ELEMENTS+EIGHT_CTX],tailCopy);
  }
''' + text[end:]
    # Controller: operate on the elected group, using caller-owned TPipe/arena.
    begin=controller.index('  JointDirectTilingData d;')
    end=controller.index('#define SORTED_CALL',begin)
    signature='''__aicore__ inline void RunFusionGroupImpl(const Buffers& b,const TilingData& d,
    int64_t first,int64_t end,int64_t rows,AscendC::TPipe& pipe,
    AscendC::LocalTensor<int32_t> local,__ubuf__ int32_t* localTopk,int64_t localRow){
  auto storage=local;
  auto* scan=reinterpret_cast<__ubuf__ int32_t*>(local.GetPhyAddr());
  auto* ctx=scan+PLAN_SCAN_STORAGE_ELEMENTS;
  auto reqIds=b.reqIds,topkIndices=b.topkIndices,stablePrefixLens=b.stablePrefixLens,visibleSeqLens=b.visibleSeqLens;
  auto tokenToReq=b.tokenToReq,lastReqIds=b.lastReqIds,slotToToken=b.slotToToken,lruSlots=b.lruSlots;
  auto currentSlots=b.currentSlots,missCount=b.missCount,missTokens=b.missTokens,missSlots=b.missSlots;
  auto compactWorkspace=b.compactWorkspace,rowModes=b.rowModes,residentRows=b.residentRows,residentBlockTable=b.residentBlockTable;
'''
    controller=signature+controller[end:]
    controller=replace(controller,'d.jointHashCapacity,d.scratchElements)', 'd.jointHashCapacity,d.scratchElements,localTopk,localRow)')
    begin=controller.index('  if(!AllowJoint')
    end=controller.index('      SORTED_CALL(0);',begin)
    controller=controller[:begin]+controller[end:]
    controller=replace(controller,'    }\n    ++group;first=end;\n  }\n#undef SORTED_CALL\n}', '#undef SORTED_CALL\n}\n')
    controller=controller.replace('((EightMapWords(static_cast<uint32_t>(end-first)*d.topk,static_cast<uint32_t>(d.topk),d.capacity)+d.capacity+7U)&~7U)',
                                  'OwnerOffset(d.capacity)')
    controller=re.sub(r'\(\(EightMapWords\([^\n]+?\)\+d.capacity\+7U\)&~7U\)',
                      'OwnerOffset(d.capacity)',controller)
    controller=controller.replace('storage.Get<float>()','storage.ReinterpretCast<float>()')
    # Replace refinement dispatch by the bounded packet wrapper below.
    begin=controller.index('\n#if JOINT_MTP7_MID_UNIQUE',controller.index('uint32_t more=0;'))
    end=controller.index('\n#else\n        SORTED_CALL(3);',begin)
    controller=controller[:begin]+'''          FusionRefinePacket(records,ctx+EIGHT_CTX,ctx,static_cast<uint32_t>(ctx[4]),q,
            static_cast<uint32_t>(d.capacity),static_cast<uint32_t>(d.topk),more,ctx[107]!=0,
            first,reinterpret_cast<__gm__ int32_t*>(compactWorkspace),d.maxRequests,pipe,local);
          }else {SORTED_CALL(3);}
''' +controller[end:]
    # A rejected group uses a nonoverlapping private workspace; never let private
    # writes race another request group's map/unique scratch.
    stageStart=text.index('template <int Phase>\n__simt_vf__',text.index('inline void SortedStage('))
    stageEnd=text.index('__simd_vf__ inline void RefineUniqueMapVf',stageStart)
    stage=text[stageStart:stageEnd]
    stage=stage.replace('compactWorkspace+static_cast<uint64_t>(row)*scratchElements',
        'compactWorkspace+qli_fusion_resource::MapWords(maxRequests,topk)+qli_fusion_resource::UniqueGmWords(maxRequests,capacity)+static_cast<uint64_t>(row)*workspaceRowElements')
    # Force private fallback into the separately reserved GM region. Compressed
    # fast-path arena is not sized for the private scratch algorithm.
    stage=stage.replace('if(scratchElements<=EIGHT_UB_WORDS)RunPrivateRow', 'if(false)RunPrivateRow')
    # Do not instantiate the compressed UB-only stage with a GM ScratchPtr.
    # Host rejects a footprint that cannot fit; Phase4 remains the GM fallback.
    begin=stage.index('    }else{\n      __gm__ int32_t* data=')
    end=stage.index('\n    }\n  }\n}',begin)
    stage=stage[:begin]+'    }'+stage[end+len('\n    }'):]
    stage=re.sub(r'      const uint32_t (?:groupRows|queries|words) = [^\n]+\n','',stage)
    text=text[:stageStart]+stage+text[stageEnd:]
    text=text.replace('  const uint32_t mapRows=HW_C&&rows<=8U?rows:8U;\n','')
    # Do not carry the old joint hash/experimental merge implementations into
    # this fusion header. The actual RunPrivateRow compatibility path remains.
    begin=text.index('template <typename ScratchPtr>\n__simt_callee__ __aicore__ inline int32_t InsertJointToken')
    end=text.index('template <typename ScratchPtr>\n__simt_callee__ __aicore__ inline void RunPrivateRow',begin)
    text=text[:begin]+text[end:]
    begin=text.index('template <typename ScratchPtr>\n__simt_callee__ __aicore__ inline void ProcessRequestRun')
    end=text.index('static_assert(11>=0',begin)
    text=text[:begin]+text[end:]
    begin=text.index('__simd_callee__ inline void HotPutCtx')
    end=text.index('__aicore__ inline void SortedVectorToScalar',begin)
    text=text[:begin]+text[end:]
    # Device code must not emit a host TPECALL for a runtime offset helper.
    text=text.replace('qli_fusion_resource::OwnerOffset(', 'QLI_OWNER_OFFSET(')
    text=text.replace('OwnerOffset(', 'QLI_OWNER_OFFSET(')
    text=text.replace('qli_fusion_resource::MapWords(', 'QLI_MAP_WORDS(')
    text=text.replace('qli_fusion_resource::UniqueGmWords(', 'QLI_UNIQUE_GM_WORDS(')
    # CANN 9.2.1 cannot cast void* across address spaces; the generated hook ABI
    # uses GM pointers directly.
    text=replace(text,'int64_t first,uint32_t topk,uint32_t c,void* lruSlots,void* missTokens,void* missSlots){',
        'int64_t first,uint32_t topk,uint32_t c,__gm__ uint8_t* lruSlots,__gm__ uint8_t* missTokens,__gm__ uint8_t* missSlots){')
    controller=controller.replace(
        'RankPipeCopyOutputs(pipe,local,ctx,first,static_cast<uint32_t>(d.topk),static_cast<uint32_t>(d.capacity),lruSlots,missTokens,missSlots)',
        'RankPipeCopyOutputs(pipe,local,ctx,first,static_cast<uint32_t>(d.topk),static_cast<uint32_t>(d.capacity),reinterpret_cast<__gm__ uint8_t*>(lruSlots),reinterpret_cast<__gm__ uint8_t*>(missTokens),reinterpret_cast<__gm__ uint8_t*>(missSlots))')
    begin=text.index('__simd_vf__ inline void RefineFusedCountVf')
    end=text.index('template <uint32_t Bits, int KnownHalf>',begin)
    text=text[:begin]+text[end:]
    controller=controller.replace('qli_fusion_resource::OwnerOffset(', 'QLI_OWNER_OFFSET(')
    controller=controller.replace('OwnerOffset(', 'QLI_OWNER_OFFSET(')
    controller=controller.replace('qli_fusion_resource::MapWords(', 'QLI_MAP_WORDS(')
    controller=controller.replace('qli_fusion_resource::UniqueGmWords(', 'QLI_UNIQUE_GM_WORDS(')
    return text,controller


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', action='store_true')
    p.add_argument('--check', action='store_true')
    args = p.parse_args()
    text = selected_source()
    if args.raw:
        Path('/tmp/qli_rankpipe_selected.h').write_text(text)
        print('selected source:', len(text.splitlines()), 'lines')
        return
    text,controller=port(text)
    header='''/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Original MemFabric_Hybrid calculation: Mulan PSL v2.
 * Experiment helpers and storage adapters: Apache-2.0.
 * SPDX-License-Identifier: MulanPSL-2.0 AND Apache-2.0
 */
// Generated from frozen rankpipe AC; storage/dispatch port: scripts/export_fusion_plan.py.
#pragma once
#include <climits>
#include <cstdint>
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_warp_functions.h"
#include "simt_api/device_functions.h"
#include "sparse_kv_plan_config.h"
#include "storage.h"
'''
    header+='\n'.join(f'#define {k} {v}' for k,v in FLAGS.items())+'\n'
    header+='#define EIGHT_MAP_ARGUMENT(q,t) , (d.capacity/2U)\n#define QLI_OWNER_OFFSET(value) ((static_cast<uint32_t>(value)/2U)+(QLI_PLAN_STORAGE==3?2048U:static_cast<uint32_t>(value)/2U))\n#define QLI_MAP_WORDS(requests,topk) (static_cast<uint64_t>(requests)*8U*static_cast<uint64_t>(topk)/2U)\n#define QLI_UNIQUE_GM_WORDS(requests,capacity) (static_cast<uint64_t>(requests)*8U*static_cast<uint64_t>(capacity))\n'
    public=(ROOT/'csrc/qli_fusion/public.inc').read_text()
    wrappers=(ROOT/'csrc/qli_fusion/packet.inc').read_text()
    output=header+text+public+wrappers+controller+'''
__aicore__ inline void RunGroup(const Buffers& b,const TilingData& d,int64_t first,int64_t end,int64_t rows,
    AscendC::TPipe& pipe,AscendC::LocalTensor<int32_t> storage,__ubuf__ int32_t* localTopk,int64_t localRow){
  if(first<0||end<=first||end>rows)return;
  RunFusionGroupImpl(b,d,first,end,rows,pipe,storage,localTopk,localRow);
}
} // namespace qli_fused_plan
'''
    # C keeps its private-completion flag in ctx[47]. The context is only
    # allocated for 120 words, recovering exactly 32 bytes at capacity 8192.
    output=replace(output,
        'constexpr uint64_t EIGHT_UB_WORDS=216U*1024U/4U;\nconstexpr uint32_t EIGHT_CTX=128U;',
        '''constexpr uint64_t EIGHT_UB_WORDS=216U*1024U/4U;
constexpr uint32_t EIGHT_CTX=qli_fusion_resource::kContextWords;
constexpr int32_t PRIVATE_DONE_FLAG=0x40000000;
__simt_callee__ __aicore__ inline void MarkPrivateDone(__ubuf__ int32_t* ctx,uint32_t unique){
  if constexpr(QLI_PLAN_STORAGE==3){ctx[47]=static_cast<int32_t>(unique|static_cast<uint32_t>(PRIVATE_DONE_FLAG));}
  else{ctx[120]=1;}
}
__aicore__ inline bool PrivateDone(const __ubuf__ int32_t* ctx){
  if constexpr(QLI_PLAN_STORAGE==3){return (static_cast<uint32_t>(ctx[47])&static_cast<uint32_t>(PRIVATE_DONE_FLAG))!=0U;}
  return ctx[120]==1;
}
__aicore__ inline uint32_t PrivateUniqueCount(const __ubuf__ int32_t* ctx){
  if constexpr(QLI_PLAN_STORAGE==3){return static_cast<uint32_t>(ctx[47])&~static_cast<uint32_t>(PRIVATE_DONE_FLAG);}
  return static_cast<uint32_t>(ctx[47]);
}''')
    output=replace(output,
        'if(thread==0U){a.lastReqIds[first]=a.reqIds[req];ctx[46]=q-u;ctx[47]=u;ctx[120]=1;ctx[0]=2;}',
        'if(thread==0U){a.lastReqIds[first]=a.reqIds[req];ctx[46]=q-u;ctx[47]=u;MarkPrivateDone(ctx,u);ctx[0]=2;}')
    output=replace(output,'ctx[46]=q-misses;ctx[47]=misses;ctx[120]=1;',
        'ctx[46]=q-misses;ctx[47]=misses;MarkPrivateDone(ctx,misses);')
    output=replace(output,'      ctx[120]=0;\n',
        '      if constexpr(QLI_PLAN_STORAGE==3)ctx[47]=0;\n      else ctx[120]=0;\n')
    output=replace(output,'const uint32_t n=static_cast<uint32_t>(ctx[46]),misses=static_cast<uint32_t>(ctx[47]);',
        'const uint32_t n=static_cast<uint32_t>(ctx[46]),misses=PrivateUniqueCount(ctx);')
    output=replace(output,'if(ctx[0]==2 && ctx[120]==1)','if(ctx[0]==2 && PrivateDone(ctx))')
    output=output.replace('qli_fusion_resource::OwnerOffset(', 'QLI_OWNER_OFFSET(')
    output=output.replace('OwnerOffset(', 'QLI_OWNER_OFFSET(')
    output=output.replace('qli_fusion_resource::MapWords(', 'QLI_MAP_WORDS(')
    output=output.replace('qli_fusion_resource::UniqueGmWords(', 'QLI_UNIQUE_GM_WORDS(')
    output='\n'.join(line.rstrip() for line in output.splitlines())+'\n'
    target=ROOT/'csrc/qli_fusion/plan.h'
    if args.check:
        if target.read_text()!=output:raise SystemExit('generated fusion plan is stale')
        print('fusion plan generation identity passed')
    else:
        target.write_text(output)
        print('wrote',target,len(output.splitlines()),'lines')


if __name__ == '__main__':
    main()
