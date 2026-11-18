#!/usr/bin/env python3
"""Install reviewed overlays into a *dedicated* pinned ops-transformer checkout.

Never edits a user's existing operator directory. Each generated operator has
its own registration so candidates can coexist in one vendor package/process.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
if __package__:
    from .export_fusion_plan import ROOT,replace
else:
    from export_fusion_plan import ROOT,replace

COMMIT='9ec2110969a9e8502bf6d579c027ac4ea9572016'
SOURCE='attention/quant_lightning_indexer_ordered_fused'
NATIVE_INDEXER='quant_lightning_indexer'
FINAL_SORT4='quant_lightning_indexer_plan_final4'
PEER4='quant_lightning_indexer_plan_peer4'
NATIVE_SOURCE=f'attention/{NATIVE_INDEXER}'
NATIVE_BINDING=ROOT/'fusion/native_indexer_binding'
NAMES={0:'quant_lightning_indexer_plan_split',1:'quant_lightning_indexer_plan_a',
       2:'quant_lightning_indexer_plan_b',3:'quant_lightning_indexer_plan_c'}
# New factorial identities are deliberately independent of the historical
# storage-only A/B names above. c0 is the existing rankpipe AC operator.
FACTORIALS={
    'c0':('quant_lightning_indexer_plan_c',dict(final_sort=False,vector_reuse=False,hot_ub_map=False)),
    'A':('quant_lightning_indexer_plan_cfa',dict(final_sort=True,vector_reuse=False,hot_ub_map=False)),
    'B':('quant_lightning_indexer_plan_cfb',dict(final_sort=False,vector_reuse=True,hot_ub_map=False)),
    'C':('quant_lightning_indexer_plan_cfc',dict(final_sort=False,vector_reuse=False,hot_ub_map=True)),
    'AB':('quant_lightning_indexer_plan_cfab',dict(final_sort=True,vector_reuse=True,hot_ub_map=False)),
    'AC':('quant_lightning_indexer_plan_cfac',dict(final_sort=True,vector_reuse=False,hot_ub_map=True)),
    'BC':('quant_lightning_indexer_plan_cfbc',dict(final_sort=False,vector_reuse=True,hot_ub_map=True)),
    'ABC':('quant_lightning_indexer_plan_cfabc',dict(final_sort=True,vector_reuse=True,hot_ub_map=True)),
}
FACTORIAL_NAMES={role:name for role,(name,_) in FACTORIALS.items()}
ORDERED4='quant_lightning_indexer_plan_split4'
TOPK_CONTROL4='quant_lightning_indexer_plan_topk_control4'
TOPK_HEADER=Path('op_kernel/arch35/vf/quant_lightning_indexer_topk.h')

def source_id():
    files=[ROOT/'scripts/export_fusion_plan.py',ROOT/'scripts/prepare_qli_fusion.py',
           ROOT/'csrc/joint_plan/sparse_kv_plan_config.h',
           *sorted(NATIVE_BINDING.iterdir()),
           *sorted((ROOT/'csrc/qli_fusion').glob('*'))]
    h=hashlib.sha256(COMMIT.encode())
    for f in files:
        if f.is_file():h.update(str(f.relative_to(ROOT)).encode()+b'\0'+f.read_bytes())
    return h.hexdigest()


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def install_native_binding(repo, operator, digest):
    dest=repo/'attention'/operator/'torch_extension'
    (dest/'csrc').mkdir(parents=True)
    native_camel='QuantLightningIndexer'
    camel=native_camel if operator==NATIVE_INDEXER else 'QuantLightningIndexerPlanFinal4'
    for source in NATIVE_BINDING.iterdir():
        if source.suffix not in ('.py','.cpp'):
            continue
        code=source.read_text().replace(NATIVE_INDEXER,operator).replace(native_camel,camel)
        code=code.replace('qli_native_indexer_baseline','qli_native_indexer_final4' if operator==FINAL_SORT4
                          else 'qli_native_indexer_baseline')
        if source.suffix=='.cpp':
            code=replace(code,'@FUSION_SOURCE_ID@',digest)
        parent=dest/'csrc' if source.suffix=='.cpp' else dest
        (parent/(operator+source.suffix)).write_text(code)


def install_final_sort_indexer(repo):
    source=repo/NATIVE_SOURCE
    dest=repo/'attention'/FINAL_SORT4
    shutil.copytree(source,dest,ignore=shutil.ignore_patterns('torch_extension','tests','docs','examples'))
    for file in sorted(dest.rglob('*'),key=lambda path:len(path.parts),reverse=True):
        if file.is_file():
            code=file.read_text().replace(NATIVE_INDEXER,FINAL_SORT4).replace(
                'QuantLightningIndexer','QuantLightningIndexerPlanFinal4')
            code=code.replace('QUANT_LIGHTNING_INDEXER','QUANT_LIGHTNING_INDEXER_PLAN_FINAL4')
            if file.is_relative_to(dest/'op_host'):
                # Both native-derived operators link into the same host library.
                code=code.replace('QLIInfoParser','QLIFinal4InfoParser')
            if file.is_relative_to(dest/'op_host') or file.is_relative_to(dest/'op_kernel'):
                code=code.replace('QLITilingData','QLIFinal4TilingData')
            file.write_text(code)
        if NATIVE_INDEXER in file.name:
            file.rename(file.with_name(file.name.replace(NATIVE_INDEXER,FINAL_SORT4)))
    vector=dest/'op_kernel/arch35'/f'{FINAL_SORT4}_service_vector_arch35.h'
    code=vector.read_text()
    anchor='#include "../arch35/vf/quant_lightning_indexer_plan_final4_topk.h"'
    code=replace(code,anchor,anchor+'\n#include "include/adv_api/sort/sort.h"')
    # The early no-visible-tokens copy and the short sequence branch stay native.
    anchor='''        SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
        SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        AscendC::DataCopyPad(indiceOutGm[info.indiceOutOffset + (qliCurS1Idx + qliRowIdx) * topkCount_],
                             indicesOutLocal_.ReinterpretCast<int32_t>(), qliCopyOutParams);'''
    epilogue=(ROOT/'csrc/qli_fusion/native_final_sort.inc').read_text().rstrip()
    copy=anchor.replace('indicesOutLocal_.ReinterpretCast<int32_t>()','orderedIndices.ReinterpretCast<int32_t>()')
    vector.write_text(replace(code,anchor,epilogue+'\n'+copy))


def install_final_sort_ordered(vector):
    """Order only the selected TopK positions, then write them back to the
    dedicated output buffer. The Plan-owned sort scratch is never passed to the
    fused Plan; short/empty rows retain their ascending index prefix and -1 tail.
    """
    s=vector.read_text()
    anchor='#include "../arch35/vf/quant_lightning_indexer_topk.h"'
    s=replace(s,anchor,anchor+'\n#include "include/adv_api/sort/sort.h"')
    anchor='''        SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
        SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        AscendC::DataCopyPad(indiceOutGm[info.indiceOutOffset + (qliCurS1Idx + qliRowIdx) * topkCount_],
                             indicesOutLocal_.ReinterpretCast<int32_t>(), qliCopyOutParams);'''
    ordered='''        // A: the original score TopK has finished. Reuse its Plan-owned
        // scratch only as sort output/temp, then copy ordered positions back to
        // the dedicated indices buffer. The token set is unchanged, including
        // every score tie chosen by the original TopK.
        if (topkCount_ == 2048U && qliValidS2Len >= topkCount_) {
            auto orderedIndices = topkSharedTmpLocal_;
            PipeBarrier<PIPE_V>();
            AscendC::Sort<uint32_t>(orderedIndices, indicesOutLocal_,
                                    mrgValueLocal_.template ReinterpretCast<uint8_t>(), topkCount_);
            PipeBarrier<PIPE_V>();
            AscendC::DataCopy(indicesOutLocal_, orderedIndices, topkCount_);
            // DataCopy completion is fully ordered before both GM and Plan read
            // indicesOutLocal_; do not publish an alias of Plan scratch.
            PipeBarrier<PIPE_ALL>();
        }
'''+anchor
    s=replace(s,anchor,ordered)
    vector.write_text(s)


def install_peer4_fused(repo):
    """Clone the pinned fused operator with request-aligned four-row tiling.

    The parent selects two rows whenever total query rows are at most 64.
    That rule makes batch8/MTP3 split each four-row request across producers.
    Keep the measured four-row launch whenever Plan fits, and retain the host
    resource fallback to two rows for larger QLI/Plan footprints.
    """
    source=repo/SOURCE
    dest=repo/'attention'/PEER4
    shutil.copytree(source,dest)
    old='quant_lightning_indexer_ordered_fused'
    camel='QuantLightningIndexerOrderedFused'
    for path in sorted(dest.rglob('*'),key=lambda item:len(item.parts),reverse=True):
        if path.is_file():
            try:
                code=path.read_text()
            except UnicodeDecodeError:
                continue
            code=(code.replace(camel,'QuantLightningIndexerPlanPeer4')
                  .replace(old,PEER4)
                  .replace('QLIOrderedFused','QLIPlanPeer4')
                  .replace('QLI_ORDERED_FUSED','QLI_PLAN_PEER4'))
            path.write_text(code)
        if old in path.name:
            path.rename(path.with_name(path.name.replace(old,PEER4)))
    tiling=dest/'op_host'/f'{PEER4}_tiling.cpp'
    code=tiling.read_text()
    code=replace(code,'fusedS1BaseSize_ = tSize_ <= 64 ? 2U : 4U;',
                 'fusedS1BaseSize_ = 4U;  // request-aligned candidate; resource fallback remains below')
    tiling.write_text(code)


def verify_topk_isolation(repo):
    """Check both source trees after normalizing generated registration names."""
    replacements = (
        (b'QuantLightningIndexerPlanTopkControl4', b'QuantLightningIndexerPlanSplit4'),
        (b'QLIPlanTopkControl4', b'QLIPlanSplit4'),
        (b'quant_lightning_indexer_plan_topk_control4', b'quant_lightning_indexer_plan_split4'),
        (b'QLI_PLAN_TOPKCONTROL4', b'QLI_PLAN_SPLIT4'),
        (b'qli_plan_resources_topkcontrol4', b'qli_plan_resources_split4'),
    )
    def normalize(value):
        for actual, ordered in replacements:
            value=value.replace(actual,ordered)
        return value

    def files_for(operator):
        root=repo/'attention'/operator
        return {normalize(str(p.relative_to(root)).encode()):normalize(p.read_bytes())
                for tree in ('op_host','op_kernel','torch_extension') for p in (root/tree).rglob('*')
                if p.is_file() and p.relative_to(root)!=TOPK_HEADER}

    if files_for(ORDERED4)!=files_for(TOPK_CONTROL4):
        raise RuntimeError('matched Indexers differ outside the TopK implementation')
    control=(repo/'attention'/TOPK_CONTROL4/TOPK_HEADER).read_bytes()
    native=(repo/NATIVE_SOURCE/TOPK_HEADER).read_bytes()
    if control!=native:
        raise RuntimeError('fixed-4 original TopK control differs from pinned native TopK')


def overlay(folder,mode,suffix,s1_base,*,final_sort=False,vector_reuse=False,hot_ub_map=False):
    op=folder.name
    kernel=folder/'op_kernel';host=folder/'op_host'
    for name in ('plan.h','storage.h','resources.h'):
        shutil.copy2(ROOT/'csrc/qli_fusion'/name,kernel/'plan'/name)
    config=(ROOT/'csrc/joint_plan/sparse_kv_plan_config.h').read_text()
    config+='\nnamespace qli_fused_plan { namespace config = sparse_kv_plan_config; }\n'
    (kernel/'plan/sparse_kv_plan_config.h').write_text(config)
    plan=kernel/'plan/sparse_kv_plan_fused.h'
    if mode:
        if final_sort:
            vector=kernel/'arch35/quant_lightning_indexer_service_vector_arch35.h'
            install_final_sort_ordered(vector)
        if hot_ub_map and mode!=3:
            raise RuntimeError('hot UB query map is only valid for rankpipe AC storage')
        plan.write_text(f'#define QLI_PLAN_STORAGE {mode}\n'+
                        ('#define QLI_HOT_UB_MAP 1\n' if hot_ub_map else '')+
                        '#include "plan.h"\n')
    else:
        plan.write_text('#include "kernel_operator.h"\n#include "sparse_kv_plan_config.h"\nnamespace qli_fused_plan {\n'+(ROOT/'csrc/qli_fusion/public.inc').read_text()+
            '\n__aicore__ inline void RunGroup(const Buffers&,const TilingData&,int64_t,int64_t,int64_t,AscendC::TPipe&,AscendC::LocalTensor<int32_t>,__ubuf__ int32_t*,int64_t){}\n}\n')
    hook=kernel/'quant_lightning_indexer_fused_hook.h';s=hook.read_text()
    if not mode:
        s=replace(s,'using OrderedFusedPlan = OrderedFusedPlanHook<true>;','using OrderedFusedPlan = OrderedFusedPlanHook<false>;')
    else:
        # Publication is at the arena tail, outside QLI Vector-only values. Its
        # result is read before RunGroup starts using the entire arena.
        s=replace(s,'reinterpret_cast<__ubuf__ int64_t *>(storage.GetPhyAddr())',
            'reinterpret_cast<__ubuf__ int64_t *>(storage.GetPhyAddr()) + (tiling_.localMemoryBytes-32U)/sizeof(int64_t)')
    hook.write_text(s)
    # Host and torch use the same resource formulas as the device.
    resource=f'#define QLI_PLAN_STORAGE {mode}\n'+(ROOT/'csrc/qli_fusion/resources.h').read_text()
    (host/'qli_plan_resources.h').write_text(resource)
    binding=folder/'torch_extension/csrc'/f'{op}.cpp'
    resource_name=f'qli_plan_resources_{suffix.lower()}.h'
    common=folder.parent.parent/'torch_extension/cann_ops_transformer/common';common.mkdir(parents=True,exist_ok=True)
    # This file lives outside dest, so the final per-operator namespace rewrite
    # does not visit it. Match the rewritten binding explicitly on a clean build.
    (common/resource_name).write_text(resource.replace('qli_fusion_resource',op+'_resource'))
    s=binding.read_text();s=f'#include "{resource_name}"\n'+s
    start=s.index('    const uint64_t scratch = PlanScratchElements(')
    end=s.index('\n    char *layoutQueryPtr',start)
    s=s[:start]+'''    TORCH_CHECK(sparseCount==2048 && (capacity==4096 || capacity==8192 || capacity==16384), "fusion experiment shape unsupported");
    TORCH_CHECK(rows<=reqIds.numel()*8,"fusion experiment supports at most 8 query rows per allocated request");
    const uint64_t requiredCompact=qli_fusion_resource::WorkspaceWords(reqIds.numel(),capacity,sparseCount);
    TORCH_CHECK(static_cast<uint64_t>(compactWorkspace.numel())>=requiredCompact,"fusion workspace too small");
'''+s[end:]
    # Unused original resource helpers are removed instead of hidden by warnings.
    start=s.index('constexpr int64_t kPlanRowUbLimitBytes')
    end=s.index('void CheckPlanTensor',start);s=s[:start]+s[end:]
    s=replace(s,'int64_t maxJointRows) {','int64_t maxJointRows, const c10::optional<at::Tensor>& topkOut) {')
    s=replace(s,'    at::Tensor output = at::empty(outputSize, query.options().dtype(at::kInt));',
        '''    at::Tensor output;
    if(topkOut.has_value()){
        output=*topkOut;
        TORCH_CHECK(output.device()==query.device() && output.scalar_type()==at::kInt && output.is_contiguous(),"topk_out dtype/device/layout mismatch");
        TORCH_CHECK(output.sizes()==at::IntArrayRef(outputSize),"topk_out shape mismatch");
        for(const auto* input:{&query,&key,&weights,&queryDequantScale,&keyDequantScale,&reqIds,
            &stablePrefixLens,&visibleSeqLens,&tokenToReq,&hostBlockTable,&activeRows,&lastReqIds,
            &slotToToken,&lruSlots,&currentSlots,&missCount,&missTokens,&missSlots,&compactWorkspace,
            &rowModes,&residentRows,&residentBlockTable,&readyCounts})
            TORCH_CHECK(!output.is_alias_of(*input),"topk_out aliases another input/state");
        if(blockTable.has_value())TORCH_CHECK(!output.is_alias_of(*blockTable),"topk_out aliases block_table");
    }else output=at::empty(outputSize,query.options().dtype(at::kInt));''')
    s=s.replace('PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {',
        'PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n    m.def("_fusion_identity", [](){return "'+source_id()+'";});')
    binding.write_text(s)
    py=folder/'torch_extension'/f'{op}.py';s=py.read_text()
    s=replace(s,'int joint_mtp=1, int max_joint_rows=4) -> Tensor',
                'int joint_mtp=1, int max_joint_rows=4, Tensor(a!)? topk_out=None) -> Tensor')
    s=s.replace('max_joint_rows: int = 4,','max_joint_rows: int = 4,\n            topk_out: Optional[torch.Tensor] = None,')
    s=replace(s,'capacity, max_token, host_block_size, host_num_blocks, joint_mtp, max_joint_rows,',
                'capacity, max_token, host_block_size, host_num_blocks, joint_mtp, max_joint_rows, topk_out,')
    py.write_text(s)
    tiling=host/f'{op}_tiling.cpp';s=tiling.read_text();s='#include "qli_plan_resources.h"\n'+s
    start=s.index('    uint64_t hash = 32;',s.index('::CheckPlanInputs()'))
    end=s.index('    qli_fused_host::QliDataPath dataPath',start)
    s=s[:start]+'''    if(topk!=2048 || rows>dim0(0)*8 || (qliParams_.capacity!=4096 && qliParams_.capacity!=8192 && qliParams_.capacity!=16384)){
        OP_LOGE(opName_,"fusion experiment supports TopK2048 and capacity4K/8K/16K");return ge::GRAPH_FAILED;
    }
    const uint64_t required=qli_fusion_resource::WorkspaceWords(dim0(0),qliParams_.capacity,topk);
    if(static_cast<uint64_t>(dim0(13))<required){OP_LOGE(opName_,"fusion workspace too small");return ge::GRAPH_FAILED;}
    planHashCapacity_=4096;planWorkspaceRowElements_=qli_fusion_resource::PrivateWords(qliParams_.capacity,topk);
    planMaxRows_=rows;planCapacity_=qliParams_.capacity;planMaxToken_=qliParams_.maxToken;
    planMaxRequests_=dim0(0);planMaxNumBlocks_=logicalMaxBlocks;planHostNumBlocks_=qliParams_.hostNumBlocks;
    planBlockSize_=static_cast<int32_t>(qliParams_.hostBlockSize);
    planLocalMemoryBytes_=qli_fusion_resource::LocalBytes(qliParams_.capacity);
    planJointHashCapacity_=qliParams_.jointMtp?1U:0U;
    planScratchElements_=qli_fusion_resource::ScratchWords(qliParams_.capacity);planMaxJointRows_=8;
    @S1_ASSIGNMENT@
'''+s[end:]
    if s1_base is None:
        s=replace(s,'@S1_ASSIGNMENT@','fusedS1BaseSize_ = tSize_ <= 64 ? 2U : 4U;')
    else:
        s=replace(s,'@S1_ASSIGNMENT@',f'fusedS1BaseSize_={s1_base}U;')
    if s1_base==4:
        # This experiment changes the tile, so a silent fallback to two rows
        # would make its timing indistinguishable from the split Indexer.
        start=s.index('    if (resources.total > qli_fused_host::kUbLimitBytes) {',start)
        end=s.index('    if (resources.total > qli_fused_host::kUbLimitBytes) {',start+1)
        s=s[:start]+s[end:]
        s=replace(s,'fused QLI and plan UB footprint exceeds 216 KiB',
                  'ordered Indexer with fixed S1Base=4 exceeds 216 KiB')
    # Original lambda and old UB-limit constant would now be unused under -Werror.
    start=s.index('    constexpr uint64_t kPlanRowUbLimitBytes',s.index('::CheckPlanInputs()'))
    end=s.index('    const auto &outShape',start)
    # checkedProduct is still used by shape-product validation below.
    s=s[:start]+s[start:end].replace('    constexpr uint64_t kPlanRowUbLimitBytes = qli_fused_plan::config::PLAN_ROW_UB_LIMIT_BYTES;\n','')+s[end:]
    tiling.write_text(s)
    if vector_reuse:
        # Keep the full Cube-owned resMm allocation pinned. Only the five
        # Vector-only arrays share the larger TopK/Plan arena.
        vf=kernel/'arch35/quant_lightning_indexer_service_vector_arch35.h';s=vf.read_text()
        start=s.index('    pipe->InitBuffer(weightBuf_')
        end=s.index('    // Topk',start)
        s=s[:start]+s[end:]
        start=s.index('    const uint64_t arenaBytes =')
        end=s.index('    pipe->InitBuffer(indicesOutBuf_',start)
        s=s[:start]+'''    const uint64_t weightBytes=2U*CeilDiv(s1BaseSize_,2)*UB_BANK_DEPTH_STRIDE;
    const uint64_t scaleBytes=2U*s2BaseSize_*16U*sizeof(SCALE_T);
    const uint64_t outBytes=2U*CeilDiv(s1BaseSize_,2)*s2BaseSize_*sizeof(SCORE_T);
    const uint64_t miscBytes=3U*weightBytes+scaleBytes+outBytes;
    const uint64_t arenaBytes=QLICommon::Max(miscBytes+topkScratchBytes,planScratchBytes);
    pipe->InitBuffer(topkPlanArenaBuf_,arenaBytes);
    auto arena=topkPlanArenaBuf_.Get<uint8_t>();
    weightUB_=arena.ReinterpretCast<W_T>();weightFloatUB_=arena[weightBytes].ReinterpretCast<float>();
    kScaleUB_=arena[2U*weightBytes].ReinterpretCast<SCALE_T>();
    qScaleUB_=arena[2U*weightBytes+scaleBytes].ReinterpretCast<SCALE_T>();
    vec1OutUB_=arena[3U*weightBytes+scaleBytes].ReinterpretCast<SCORE_T>();
    mrgValueLocal_=arena[miscBytes].ReinterpretCast<SCORE_T>();
    scoreOutLocal_=arena[miscBytes+mrgValueBytes].ReinterpretCast<SCORE_T>();
    topkSharedTmpLocal_=arena[miscBytes+mrgValueBytes+scoreOutBytes].ReinterpretCast<uint32_t>();
    planStorage_=arena.ReinterpretCast<int32_t>();

'''+s[end:]
        # TopK is called only at isLastS2InnerLoop. The next Vec1 region starts
        # at isFirstS2InnerLoop and reloads weights/Q scales/K scales. Padding in
        # K scales must also be reinitialized after Plan overwrote this area.
        anchor='    if (info.isFirstS2InnerLoop) {\n        WaitFlag<HardEvent::V_MTE2>'
        s=replace(s,anchor,'''    if (info.isFirstS2InnerLoop) {
        Duplicate(kScaleUB_,static_cast<SCALE_T>(0),2*s2BaseSize_*16);
        // The arena was borrowed by Plan. This new V write must finish before
        // GetKeyScale's MTE2 writes the same region. Existing ready flags were
        // published before this reset; PIPE_V alone cannot order the two pipes.
        // Stage-boundary initialization: avoid borrowing an active ping-pong ID.
        PipeBarrier<PIPE_ALL>();
        WaitFlag<HardEvent::V_MTE2>''')
        # Explicitly drain pending Vector-owned DMA before borrowing the arena;
        # this does not release or overwrite Cube's separate resMm region.
        anchor='            planHook_->PublishGroup(row, row + 1, planRows_, planStorage_,'
        s=replace(s,anchor,'''            auto drain=pipe_->AllocEventID<HardEvent::MTE2_V>();
            SetFlag<HardEvent::MTE2_V>(drain);WaitFlag<HardEvent::MTE2_V>(drain);
            pipe_->ReleaseEventID<HardEvent::MTE2_V>(drain);
'''+anchor,count=2)
        # The upstream class stores no pipe_ member; pass the same TPipe pointer
        # already received by InitBuffers, rather than introducing a global.
        s=s.replace('    int64_t planRows_ = 0;','    int64_t planRows_ = 0;\n    TPipe* pipe_=nullptr;')
        s=s.replace('QLIVector<QLIT>::InitBuffers(TPipe *pipe)\n{','QLIVector<QLIT>::InitBuffers(TPipe *pipe)\n{\n    pipe_=pipe;')
        vf.write_text(s)
        r=host/'fused_resource.h';s=r.read_text()
        s=replace(s,'return {qli, topk, plan, indices, qli + indices + arena};',
            'const uint64_t misc=qli-resMm;\n    const uint64_t shared=(misc+topk)>plan?(misc+topk):plan;\n    return {qli,topk,plan,indices,resMm+indices+shared};')
        # shared is already a variable name for TopK scratch in this function.
        s=s.replace('const uint64_t shared=(misc+topk)', 'const uint64_t fusedShared=(misc+topk)').replace('resMm+indices+shared};','resMm+indices+fusedShared};')
        s=s.replace('    const uint64_t arena = topk > plan ? topk : plan;\n','')
        r.write_text(s)
    cmake=host/'CMakeLists.txt';s=cmake.read_text()
    s=s.replace('            -Werror','            -Werror\n            -O3\n            -mllvm -cce-aicore-dcci-before-kernel-end=false')
    cmake.write_text(s)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--upstream',required=True,type=Path);a=p.parse_args()
    repo=a.upstream.resolve()
    head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if head!=COMMIT:raise SystemExit(f'expected upstream {COMMIT}, got {head}')
    source=repo/SOURCE
    # Enforce exact public upstream files, including the QLI/TopK implementation.
    tracked=subprocess.check_output(
        ['git','-C',str(repo),'ls-tree','-r','--name-only',COMMIT,SOURCE,NATIVE_SOURCE],
        text=True,
    ).splitlines()
    for name in tracked:
        expected=subprocess.check_output(['git','-C',str(repo),'show',f'{COMMIT}:{name}'])
        if not (repo/name).is_file() or (repo/name).read_bytes()!=expected:raise SystemExit('upstream file differs: '+name)
    baseline_binding=repo/NATIVE_SOURCE/'torch_extension'
    variants=[]
    for mode,name in NAMES.items():
        factors=FACTORIALS['c0'][1] if mode==3 else dict(final_sort=False,vector_reuse=False,hot_ub_map=False)
        variants.append((mode,name,'Split' if mode==0 else 'ABC'[mode-1],
                         None if mode==3 else 2,factors))
    for role,(name,factors) in FACTORIALS.items():
        if role!='c0':
            variants.append((3,name,'CF'+role,None,factors))
    variants.append((0,ORDERED4,'Split4',4,dict(final_sort=False,vector_reuse=False,hot_ub_map=False)))
    variants.append((0,TOPK_CONTROL4,'TopkControl4',4,dict(final_sort=False,vector_reuse=False,hot_ub_map=False)))
    destinations=[repo/'attention'/name for _,name,_,_,_ in variants]+[baseline_binding,repo/'attention'/FINAL_SORT4,repo/'attention'/PEER4]
    for dest in destinations:
        if dest.exists():raise SystemExit('refusing to overwrite existing directory: '+str(dest))
    install_final_sort_indexer(repo)
    install_peer4_fused(repo)
    for mode,name,suffix,s1_base,factors in variants:
        dest=repo/'attention'/name
        shutil.copytree(source,dest)
        old='quant_lightning_indexer_ordered_fused';camel='QuantLightningIndexerOrderedFused'
        for f in sorted(dest.rglob('*'),key=lambda x:len(x.parts),reverse=True):
            if f.is_file():
                try:s=f.read_text()
                except UnicodeDecodeError:continue
                s=s.replace(camel,'QuantLightningIndexerPlan'+suffix).replace(old,name).replace('QLIOrderedFused','QLIPlan'+suffix).replace('QLI_ORDERED_FUSED','QLI_PLAN_'+suffix.upper())
                f.write_text(s)
            if old in f.name:f.rename(f.with_name(f.name.replace(old,name)))
        overlay(dest,mode,suffix,s1_base,
                final_sort=factors['final_sort'],vector_reuse=factors['vector_reuse'] or mode in (1,2),
                hot_ub_map=factors['hot_ub_map'])
        if name == TOPK_CONTROL4 or factors['final_sort']:
            # The control isolates native selection. Factorial A also sorts the
            # selected token positions before publishing them to the fused Plan.
            shutil.copyfile(repo/NATIVE_SOURCE/TOPK_HEADER,dest/TOPK_HEADER)
        # All host tiling objects are linked into one vendor library. Distinct
        # constexpr definitions must NOT share a mangled name across A/B/C.
        for f in dest.rglob('*'):
            if f.is_file() and f.suffix in ('.h','.cpp','.inc'):
                s=f.read_text().replace('qli_fusion_resource',name+'_resource').replace('qli_fused_host',name+'_host')
                f.write_text(s)
    verify_topk_isolation(repo)
    digest=source_id()
    for operator in (NATIVE_INDEXER,FINAL_SORT4):
        install_native_binding(repo,operator,digest)
    manifest={'upstream':COMMIT,'source_id':digest,'plan_parent':'44ee66caed1ceb58fad4190887b1e07fae0d4281',
              'native_indexer':{'operator':NATIVE_INDEXER,'source':NATIVE_SOURCE,
                                'upstream':COMMIT,'tiling':'upstream'},
              'native_final_sort':{'operator':FINAL_SORT4,'source':NATIVE_SOURCE,
                                   'upstream':COMMIT,'tiling':'upstream','topk':2048},
              'peer_aligned4':{'operator':PEER4,'source':SOURCE,'upstream':COMMIT,
                               's1_base':4,'fallback_s1_base':2},
              'auto_rankpipe_c':{'operator':NAMES[3],'source':SOURCE,'upstream':COMMIT,
                                 'plan':'rankpipe_ac','tiling':'upstream_auto'},
              'factorial_roles':['c0','A','B','C','AB','AC','BC','ABC'],
              'factorial_operators':FACTORIAL_NAMES,
              'factorial_factors':{role:factors for role,(_,factors) in FACTORIALS.items()},
              'factorial_ub_bytes':{role:{'plan_c8192':123648,
                  's1_base4':197632 if factors['vector_reuse'] else 221184,
                  's1_base2':164864 if factors['vector_reuse'] else 184832}
                  for role,(_,factors) in FACTORIALS.items()},
              'operators':NAMES,'qli_s1_base':2,
              'ordered_indexer_4':{'operator':ORDERED4,'s1_base':4,'plan_enabled':False},
              'topk_control_4':{'operator':TOPK_CONTROL4,'parent':ORDERED4,'s1_base':4,
                                'plan_enabled':False,'replaced_file':str(TOPK_HEADER)},
              'files':{}}
    for _,name,_,_,_ in variants:
        for f in (repo/'attention'/name).rglob('*'):
            if f.is_file():manifest['files'][str(f.relative_to(repo))]=sha(f)
    for operator in (NATIVE_INDEXER,FINAL_SORT4,PEER4):
        for f in (repo/'attention'/operator).rglob('*'):
            if f.is_file():manifest['files'][str(f.relative_to(repo))]=sha(f)
    (repo/'qli_fusion_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('prepared experiment operators and the native Indexer binding; manifest:',repo/'qli_fusion_manifest.json')

if __name__=='__main__':main()
