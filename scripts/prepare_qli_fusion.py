#!/usr/bin/env python3
"""Install reviewed overlays into a *dedicated* pinned ops-transformer checkout.

Never edits a user's existing operator directory. Four distinct registered ops
are generated, so all candidates can coexist in one vendor package/process.
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
NAMES={0:'quant_lightning_indexer_plan_split',1:'quant_lightning_indexer_plan_a',
       2:'quant_lightning_indexer_plan_b',3:'quant_lightning_indexer_plan_c'}

def source_id():
    files=[ROOT/'scripts/export_fusion_plan.py',ROOT/'scripts/prepare_qli_fusion.py',
           ROOT/'csrc/joint_plan/sparse_kv_plan_config.h',
           *sorted((ROOT/'csrc/qli_fusion').glob('*'))]
    h=hashlib.sha256(COMMIT.encode())
    for f in files:
        if f.is_file():h.update(str(f.relative_to(ROOT)).encode()+b'\0'+f.read_bytes())
    return h.hexdigest()


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def overlay(folder,mode):
    op=NAMES[mode];suffix='Split' if mode==0 else 'ABC'[mode-1]
    kernel=folder/'op_kernel';host=folder/'op_host'
    for name in ('plan.h','storage.h','resources.h'):
        shutil.copy2(ROOT/'csrc/qli_fusion'/name,kernel/'plan'/name)
    config=(ROOT/'csrc/joint_plan/sparse_kv_plan_config.h').read_text()
    config+='\nnamespace qli_fused_plan { namespace config = sparse_kv_plan_config; }\n'
    (kernel/'plan/sparse_kv_plan_config.h').write_text(config)
    plan=kernel/'plan/sparse_kv_plan_fused.h'
    if mode:
        plan.write_text(f'#define QLI_PLAN_STORAGE {mode}\n#include "plan.h"\n')
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
    fusedS1BaseSize_=2U; // same QLI tile for split/A/B/C; report this explicitly
'''+s[end:]
    # Original lambda and old UB-limit constant would now be unused under -Werror.
    start=s.index('    constexpr uint64_t kPlanRowUbLimitBytes',s.index('::CheckPlanInputs()'))
    end=s.index('    const auto &outShape',start)
    # checkedProduct is still used by shape-product validation below.
    s=s[:start]+s[start:end].replace('    constexpr uint64_t kPlanRowUbLimitBytes = qli_fused_plan::config::PLAN_ROW_UB_LIMIT_BYTES;\n','')+s[end:]
    tiling.write_text(s)
    if mode in (1,2):
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
    tracked=subprocess.check_output(['git','-C',str(repo),'ls-tree','-r','--name-only',COMMIT,SOURCE],text=True).splitlines()
    for name in tracked:
        expected=subprocess.check_output(['git','-C',str(repo),'show',f'{COMMIT}:{name}'])
        if not (repo/name).is_file() or (repo/name).read_bytes()!=expected:raise SystemExit('upstream file differs: '+name)
    for mode,name in NAMES.items():
        dest=repo/'attention'/name
        if dest.exists():raise SystemExit('refusing to overwrite existing directory: '+str(dest))
        shutil.copytree(source,dest)
        old='quant_lightning_indexer_ordered_fused';camel='QuantLightningIndexerOrderedFused'
        suffix='Split' if mode==0 else 'ABC'[mode-1]
        for f in sorted(dest.rglob('*'),key=lambda x:len(x.parts),reverse=True):
            if f.is_file():
                try:s=f.read_text()
                except UnicodeDecodeError:continue
                s=s.replace(camel,'QuantLightningIndexerPlan'+suffix).replace(old,name).replace('QLIOrderedFused','QLIPlan'+suffix).replace('QLI_ORDERED_FUSED','QLI_PLAN_'+suffix.upper())
                f.write_text(s)
            if old in f.name:f.rename(f.with_name(f.name.replace(old,name)))
        overlay(dest,mode)
        # All host tiling objects are linked into one vendor library. Distinct
        # constexpr definitions must NOT share a mangled name across A/B/C.
        for f in dest.rglob('*'):
            if f.is_file() and f.suffix in ('.h','.cpp','.inc'):
                s=f.read_text().replace('qli_fusion_resource',name+'_resource').replace('qli_fused_host',name+'_host')
                f.write_text(s)
    manifest={'upstream':COMMIT,'source_id':source_id(),'plan_parent':'44ee66caed1ceb58fad4190887b1e07fae0d4281',
              'operators':NAMES,'qli_s1_base':2,'files':{}}
    for name in NAMES.values():
        for f in (repo/'attention'/name).rglob('*'):
            if f.is_file():manifest['files'][str(f.relative_to(repo))]=sha(f)
    (repo/'qli_fusion_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('prepared four registered operators; manifest:',repo/'qli_fusion_manifest.json')

if __name__=='__main__':main()
