#!/usr/bin/env python3
"""Minimal Plan fusion overlay for the standalone ordered Indexer device tree."""
from pathlib import Path

if __package__:
    from .export_fusion_plan import replace
else:
    from export_fusion_plan import replace


def rename_sources(root, replacements):
    """Rename generated identities while preserving all other source bytes."""
    for path in sorted(root.rglob('*'), key=lambda item: len(item.parts), reverse=True):
        if path.is_file():
            try:
                code = path.read_text()
            except UnicodeDecodeError:
                continue
            for old, new in replacements:
                code = code.replace(old, new)
            path.write_text(code)
        for old, new in replacements:
            if old in path.name:
                path.rename(path.with_name(path.name.replace(old, new)))


def ordered_names(name, camel):
    return (
        ('QuantLightningIndexerOrdered', 'QuantLightningIndexerPlan' + camel),
        ('quant_lightning_indexer_ordered', name),
        ('QLIOrdered', 'QLIPlan' + camel),
        ('QLI_ORDERED', 'QLI_PLAN_' + camel.upper()),
        ('QUANT_LIGHTNING_INDEXER_ORDERED', 'QUANT_LIGHTNING_INDEXER_PLAN_' + camel.upper()),
    )


def fused_names(name, camel):
    return (
        ('QuantLightningIndexerOrderedFused', 'QuantLightningIndexerPlan' + camel),
        ('quant_lightning_indexer_ordered_fused', name),
        ('QLIOrderedFused', 'QLIPlan' + camel),
        ('QLI_ORDERED_FUSED', 'QLI_PLAN_' + camel.upper()),
    )


def patch_entry(dest: Path, enabled: bool) -> None:
    path = dest / 'op_kernel' / f'{dest.name}.cpp'
    code = path.read_text()
    code = replace(
        code,
        '''        op.Init(query, key, weights, queryScale, keyScale, actualSeqLengthsQ, actualSeqLengthsK, blocktable, \\
                sparseIndices, user, tiling_data, &tPipe);''',
        '''        op.Init(query, key, weights, queryScale, keyScale, actualSeqLengthsQ, actualSeqLengthsK, blocktable, \\
                reqIds, stablePrefixLens, visibleSeqLens, tokenToReq, hostBlockTable, activeRows, \\
                lastReqIds, slotToToken, lruSlots, currentSlots, missCount, missTokens, missSlots, compactWorkspace, \\
                rowModes, residentRows, residentBlockTable, readyCounts, sparseIndices, user, tiling_data, &tPipe);''',
    )
    native_args = '''                                                   __gm__ uint8_t *blocktable, __gm__ uint8_t *sparseIndices,
                                                   __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    TPipe tPipe;'''
    plan_args = '''                                                   __gm__ uint8_t *blocktable, __gm__ uint8_t *reqIds,
                                                   __gm__ uint8_t *stablePrefixLens,
                                                   __gm__ uint8_t *visibleSeqLens, __gm__ uint8_t *tokenToReq,
                                                   __gm__ uint8_t *hostBlockTable, __gm__ uint8_t *activeRows,
                                                   __gm__ uint8_t *lastReqIds, __gm__ uint8_t *slotToToken,
                                                   __gm__ uint8_t *lruSlots, __gm__ uint8_t *currentSlots,
                                                   __gm__ uint8_t *missCount, __gm__ uint8_t *missTokens,
                                                   __gm__ uint8_t *missSlots, __gm__ uint8_t *compactWorkspace,
                                                   __gm__ uint8_t *rowModes, __gm__ uint8_t *residentRows,
                                                   __gm__ uint8_t *residentBlockTable, __gm__ uint8_t *readyCounts,
                                                   __gm__ uint8_t *sparseIndices, __gm__ uint8_t *workspace,
                                                   __gm__ uint8_t *tiling)
{
''' + ('    AscendC::InitSocState();\n' if enabled else '') + '''    TPipe tPipe;'''
    code = replace(code, native_args, plan_args)
    if not enabled:
        code = replace(
            code,
            '''                                                   __gm__ uint8_t *sparseIndices, __gm__ uint8_t *workspace,
                                                   __gm__ uint8_t *tiling)
{
''',
            '''                                                   __gm__ uint8_t *sparseIndices, __gm__ uint8_t *workspace,
                                                   __gm__ uint8_t *tiling)
{
    (void)reqIds; (void)stablePrefixLens; (void)visibleSeqLens; (void)tokenToReq;
    (void)hostBlockTable; (void)activeRows; (void)lastReqIds; (void)slotToToken;
    (void)lruSlots; (void)currentSlots; (void)missCount; (void)missTokens;
    (void)missSlots; (void)compactWorkspace; (void)rowModes; (void)residentRows;
    (void)residentBlockTable; (void)readyCounts;
''',
        )
    path.write_text(code)


def patch_kernel(dest: Path, camel: str, enabled: bool, fixed_four: bool = False,
                 key_scale_l2: bool = False) -> None:
    path = dest / 'op_kernel/arch35/quant_lightning_indexer_kernel_arch35.h'
    code = path.read_text()
    tiling = f'QLIPlan{camel}TilingData'
    code = replace(
        code,
        '#include "quant_lightning_indexer_service_cube_arch35.h"',
        '#include "quant_lightning_indexer_service_cube_arch35.h"\n'
        '#include "../quant_lightning_indexer_fused_hook.h"',
    )
    native_decl = f'''                                __gm__ uint8_t *actualSeqLengthsK, __gm__ uint8_t *blockTable,
                                __gm__ uint8_t *sparseIndices, __gm__ uint8_t *workspace,
                                const {tiling} *__restrict tiling, TPipe *tPipe);'''
    plan_decl = f'''                                __gm__ uint8_t *actualSeqLengthsK, __gm__ uint8_t *blockTable,
                                __gm__ uint8_t *reqIds, __gm__ uint8_t *stablePrefixLens,
                                __gm__ uint8_t *visibleSeqLens, __gm__ uint8_t *tokenToReq,
                                __gm__ uint8_t *hostBlockTable, __gm__ uint8_t *activeRows,
                                __gm__ uint8_t *lastReqIds, __gm__ uint8_t *slotToToken,
                                __gm__ uint8_t *lruSlots, __gm__ uint8_t *currentSlots,
                                __gm__ uint8_t *missCount, __gm__ uint8_t *missTokens,
                                __gm__ uint8_t *missSlots, __gm__ uint8_t *compactWorkspace,
                                __gm__ uint8_t *rowModes, __gm__ uint8_t *residentRows,
                                __gm__ uint8_t *residentBlockTable, __gm__ uint8_t *readyCounts,
                                __gm__ uint8_t *sparseIndices, __gm__ uint8_t *workspace,
                                const {tiling} *__restrict tiling, TPipe *tPipe);'''
    code = replace(code, native_decl, plan_decl)
    code = replace(
        code,
        '    QLICommon::SplitCoreInfo splitCoreInfo{};',
        '    QLICommon::SplitCoreInfo splitCoreInfo{};\n    OrderedFusedPlan planHook_{};',
    )
    native_def = f'''    __gm__ uint8_t *blockTable, __gm__ uint8_t *sparseIndices, __gm__ uint8_t *workspace,
    const {tiling} *__restrict tiling, TPipe *tPipe)'''
    plan_def = f'''    __gm__ uint8_t *blockTable, __gm__ uint8_t *reqIds, __gm__ uint8_t *stablePrefixLens,
    __gm__ uint8_t *visibleSeqLens, __gm__ uint8_t *tokenToReq, __gm__ uint8_t *hostBlockTable,
    __gm__ uint8_t *activeRows, __gm__ uint8_t *lastReqIds, __gm__ uint8_t *slotToToken,
    __gm__ uint8_t *lruSlots, __gm__ uint8_t *currentSlots, __gm__ uint8_t *missCount,
    __gm__ uint8_t *missTokens, __gm__ uint8_t *missSlots, __gm__ uint8_t *compactWorkspace,
    __gm__ uint8_t *rowModes, __gm__ uint8_t *residentRows, __gm__ uint8_t *residentBlockTable,
    __gm__ uint8_t *readyCounts, __gm__ uint8_t *sparseIndices, __gm__ uint8_t *workspace,
    const {tiling} *__restrict tiling, TPipe *tPipe)'''
    code = replace(code, native_def, plan_def)
    if key_scale_l2:
        # Both AIVs read the same K-scale pages. Keep their default L2 hint
        # while preserving the native hint for the much larger Key tensor.
        code = replace(
            code,
            '''        if (constInfo.setL2DisableFlag) {
            kScaleGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
''',
            '',
        )
    if fixed_four:
        # Set the tile in the device source as well. Native Ordered derives
        # S1Base from tSize and does not consume the fused host's S1Base field.
        code = replace(
            code,
            '''    constInfo.mBaseSize = S1_BASE_SIZE * constInfo.gSize;
    constInfo.s2BaseSize = S2_BASE_SIZE;
    constInfo.s1BaseSize = S1_BASE_SIZE;

    if (constInfo.tSize <= 64) {
        constInfo.mBaseSize /= 2;
        constInfo.s1BaseSize /= 2;
    }''',
            '''    constInfo.s1BaseSize = tilingData->fusedS1BaseSize;
    constInfo.mBaseSize = constInfo.s1BaseSize * constInfo.gSize;
    constInfo.s2BaseSize = S2_BASE_SIZE;''',
        )
    code = replace(
        code,
        '    InitBuffers();\n}',
        '''    InitBuffers();
    if ASCEND_IS_AIV {
        qli_fused_plan::Buffers planBuffers{
            reqIds, sparseIndices, stablePrefixLens, visibleSeqLens, tokenToReq, hostBlockTable, activeRows,
            lastReqIds, slotToToken, lruSlots, currentSlots, missCount, missTokens, missSlots, compactWorkspace,
            rowModes, residentRows, residentBlockTable, readyCounts};
        qli_fused_plan::TilingData planTiling{
            tiling->planHashCapacity, tiling->planWorkspaceRowElements, tiling->planMaxRows,
            static_cast<int64_t>(tiling->sparseCount), tiling->planCapacity, tiling->planMaxToken,
            tiling->planMaxRequests, tiling->planMaxNumBlocks, tiling->planHostNumBlocks, tiling->planBlockSize,
            tiling->planLocalMemoryBytes, tiling->planJointHashCapacity, tiling->planScratchElements,
            tiling->planMaxJointRows};
        planHook_.Init(planBuffers, planTiling, reinterpret_cast<__gm__ int32_t *>(readyCounts), pipe);
        vectorService.SetPlanHook(&planHook_, tiling->planMaxRows);
    }
}''',
    )
    if not enabled:
        path.write_text(code)
        return
    code = replace(
        code,
        '    GetBN2Idx(bN2LoopIdx);',
        '''    // The shared base helper sets this for a short-S1 batch. Clear it for
    // every batch so a preceding core interval cannot publish stale cleanup.
    tempLoopInfo.needDealActS1LessThanS1 = false;
    GetBN2Idx(bN2LoopIdx);''',
    )
    native_zero = '''    uint32_t tBase = 0U;
    if (constInfo.outputLayout == QLICommon::LI_LAYOUT::TND) {
        tBase = bIdx == 0 ? 0 : actualSeqLengthsGmQ.GetValue(bIdx - 1);
    }
    LIV2Common::DealActSeqLenIsZero<QLICommon::LI_LAYOUT::TND, QLICommon::LI_LAYOUT::BSND>(
        bIdx, n2Idx, s1Start, tBase, tempLoopInfo.actS1Size, constInfo.sparseCount, constInfo, vectorService);'''
    row_zero = '''    if ASCEND_IS_AIV {
        if ((tmpBlockIdx & 1U) != 0U) return;
    } else {
        return;
    }
    uint32_t tBase = 0U;
    if (constInfo.outputLayout == QLICommon::LI_LAYOUT::TND) {
        tBase = bIdx == 0 ? 0 : actualSeqLengthsGmQ.GetValue(bIdx - 1);
        for (uint32_t s1Idx = s1Start; s1Idx < tempLoopInfo.actS1Size; ++s1Idx) {
            const uint64_t row = (static_cast<uint64_t>(tBase) + s1Idx) * constInfo.kHeadNum + n2Idx;
            vectorService.ProcessInvalidRow(row, row * constInfo.sparseCount);
        }
        return;
    }
    for (uint32_t s1Idx = s1Start; s1Idx < constInfo.qSeqSize; ++s1Idx) {
        const uint64_t row = static_cast<uint64_t>(bIdx) * constInfo.qSeqSize * constInfo.kHeadNum +
                             static_cast<uint64_t>(s1Idx) * constInfo.kHeadNum + n2Idx;
        vectorService.ProcessInvalidRow(row, row * constInfo.sparseCount);
    }'''
    code = replace(code, native_zero, row_zero)
    native_invalid = '''        uint32_t aivCoreNum = GetBlockNum() * 2; // 2 means c:v = 1:2
        uint64_t totalOutputSize;
        if constexpr (Q_LAYOUT_T == LI_LAYOUT::BSND) {
            totalOutputSize = constInfo.batchSize * constInfo.qSeqSize * constInfo.kHeadNum * constInfo.sparseCount;
        } else {
            uint32_t tBase = constInfo.batchSize == 0 ? 0 : constInfo.tSize;
            totalOutputSize = tBase * constInfo.kHeadNum * constInfo.sparseCount;
        }
        uint64_t singleCoreSize =
            QLICommon::Align((totalOutputSize + aivCoreNum - 1) / aivCoreNum, GM_ALIGN_BYTES / sizeof(OUT_T));
        uint64_t baseSize = tmpBlockIdx * singleCoreSize;
        if (baseSize < totalOutputSize) {
            uint64_t dealSize =
                (baseSize + singleCoreSize <= totalOutputSize) ? singleCoreSize : totalOutputSize - baseSize;
            GlobalTensor<OUT_T> output = indiceOutGm[baseSize];
            AscendC::InitGlobalMemory(output, dealSize, constInfo.INVALID_IDX);
        }'''
    row_invalid = '''        // Each logical row has one producer (the even AIV). Keep odd AIVs
        // out so every invalid row receives exactly one Plan publication.
        if ((tmpBlockIdx & 1U) != 0U) return;
        const uint32_t aiv = tmpBlockIdx / 2U;
        const uint32_t aivCount = GetBlockNum();
        const uint64_t tokens = Q_LAYOUT_T == LI_LAYOUT::BSND
            ? static_cast<uint64_t>(constInfo.batchSize) * constInfo.qSeqSize
            : constInfo.tSize;
        const uint64_t rows = tokens * constInfo.kHeadNum;
        vectorService.AllocEventID();
        for (uint64_t row = aiv; row < rows; row += aivCount) {
            vectorService.ProcessInvalidRow(static_cast<int64_t>(row), row * constInfo.sparseCount);
        }
        vectorService.FreeEventID();'''
    code = replace(code, native_invalid, row_invalid)
    path.write_text(code)


def patch_vector(dest: Path, enabled: bool, grouped: bool = False, paired: bool = False,
                 pair_two_rows: bool = False) -> None:
    path = dest / 'op_kernel/arch35/quant_lightning_indexer_service_vector_arch35.h'
    code = path.read_text()
    code = replace(
        code,
        '#include "../arch35/vf/quant_lightning_indexer_topk.h"',
        '#include "../arch35/vf/quant_lightning_indexer_topk.h"\n'
        '#include "../quant_lightning_indexer_fused_hook.h"',
    )
    code = replace(
        code,
        '    __aicore__ inline void FreeEventID();',
        '''    __aicore__ inline void FreeEventID();
    __aicore__ inline void SetPlanHook(OrderedFusedPlan *hook, int64_t rows)
    {
        planHook_ = hook;
        planRows_ = rows;
    }''',
    )
    code = replace(
        code,
        '    topk::LITopk<SCORE_T> topkOp_;',
        '    topk::LITopk<SCORE_T> topkOp_;\n'
        '    OrderedFusedPlan *planHook_ = nullptr;\n'
        '    int64_t planRows_ = 0;',
    )
    if not enabled:
        path.write_text(code)
        return
    code = replace(
        code,
        '    __aicore__ inline void CleanInvalidOutput(int64_t invalidS1offset);',
        '    __aicore__ inline void CleanInvalidOutput(int64_t invalidS1offset);\n'
        '    __aicore__ inline void ProcessInvalidRow(int64_t row, int64_t outputOffset);',
    )
    native_buffers = '''    // tmp buff for topk
    TBuf<TPosition::VECCALC> mrgValueBuf_;
    LocalTensor<SCORE_T> mrgValueLocal_;

    TBuf<TPosition::VECCALC> indicesOutBuf_;
    LocalTensor<uint32_t> indicesOutLocal_;

    TBuf<TPosition::VECCALC> scoreOutBuf_;
    LocalTensor<SCORE_T> scoreOutLocal_;

    TBuf<TPosition::VECCALC> topkSharedTmpBuf_;
    LocalTensor<uint32_t> topkSharedTmpLocal_;

    LocalTensor<int32_t> outInvalidLocal_;'''
    arena_buffers = '''    // tmp buff for topk and the compact rankpipe AC plan. The native TopK
    // values are dead before Plan borrows this arena; per-row publication
    // drains MTE3 and later Vector work provides the lifecycle fence.
    TBuf<TPosition::VECCALC> topkPlanArenaBuf_;
    LocalTensor<SCORE_T> mrgValueLocal_;

    TBuf<TPosition::VECCALC> indicesOutBuf_;
    LocalTensor<uint32_t> indicesOutLocal_;

    LocalTensor<SCORE_T> scoreOutLocal_;

    LocalTensor<uint32_t> topkSharedTmpLocal_;'''
    code = replace(code, native_buffers, arena_buffers)
    code = replace(
        code,
        '    uint32_t trunkLen_ = 0;',
        '    uint32_t trunkLen_ = 0;\n    uint32_t planLocalMemoryBytes_ = 0;',
    )
    code = replace(
        code,
        '    topk::LITopk<SCORE_T> topkOp_;',
        '    topk::LITopk<SCORE_T> topkOp_;\n    LocalTensor<int32_t> planStorage_;',
    )
    native_topk_init = '''    // Topk
    pipe->InitBuffer(mrgValueBuf_, (topkCountAlign256_ + trunkLen_) * sizeof(SCORE_T));
    mrgValueLocal_ = mrgValueBuf_.Get<SCORE_T>();
    outInvalidLocal_ = mrgValueBuf_.Get<int32_t>();

    pipe->InitBuffer(indicesOutBuf_,
                     (topkCountAlign256_ + 64) *
                         sizeof(uint32_t)); // 大小：(topkCountAlign256_ + 64) * 4  64:duplicate刷-1需要额外空间
    indicesOutLocal_ = indicesOutBuf_.Get<uint32_t>();

    pipe->InitBuffer(scoreOutBuf_, topkCountAlign256_ * sizeof(SCORE_T));
    scoreOutLocal_ = scoreOutBuf_.Get<SCORE_T>();

    uint64_t topkSharedTmpSize = topkOp_.GetSharedTmpBufferSize();
    pipe->InitBuffer(topkSharedTmpBuf_, topkSharedTmpSize);
    topkSharedTmpLocal_ = topkSharedTmpBuf_.Get<uint32_t>();'''
    arena_topk_init = '''    // Topk scratch and Plan scratch have disjoint lifetimes inside one row.
    const uint64_t mrgValueBytes = QLICommon::Align(
        static_cast<uint64_t>(topkCountAlign256_ + trunkLen_) * sizeof(SCORE_T), static_cast<uint64_t>(32));
    const uint64_t scoreOutBytes = QLICommon::Align(
        static_cast<uint64_t>(topkCountAlign256_) * sizeof(SCORE_T), static_cast<uint64_t>(32));
    const uint64_t topkSharedTmpBytes = QLICommon::Align(
        static_cast<uint64_t>(topkOp_.GetSharedTmpBufferSize()), static_cast<uint64_t>(32));
    const uint64_t topkScratchBytes = mrgValueBytes + scoreOutBytes + topkSharedTmpBytes;
    const uint64_t planScratchBytes = static_cast<uint64_t>(planLocalMemoryBytes_);
    const uint64_t arenaBytes = QLICommon::Max(topkScratchBytes, planScratchBytes);
    pipe->InitBuffer(topkPlanArenaBuf_, arenaBytes);
    auto arena = topkPlanArenaBuf_.Get<uint8_t>();
    mrgValueLocal_ = arena.ReinterpretCast<SCORE_T>();
    scoreOutLocal_ = arena[mrgValueBytes].ReinterpretCast<SCORE_T>();
    topkSharedTmpLocal_ = arena[mrgValueBytes + scoreOutBytes].ReinterpretCast<uint32_t>();
    planStorage_ = arena.ReinterpretCast<int32_t>();

    pipe->InitBuffer(indicesOutBuf_,
                     (topkCountAlign256_ + 64) *
                         sizeof(uint32_t)); // 大小：(topkCountAlign256_ + 64) * 4  64:duplicate刷-1需要额外空间
    indicesOutLocal_ = indicesOutBuf_.Get<uint32_t>();'''
    code = replace(code, native_topk_init, arena_topk_init)
    code = replace(
        code,
        '    topkCount_ = constInfo.sparseCount;',
        '    topkCount_ = constInfo.sparseCount;\n'
        '    planLocalMemoryBytes_ = tilingData->planLocalMemoryBytes;',
    )
    code = replace(
        code,
        '    Duplicate(outInvalidLocal_, constInfo_.INVALID_IDX, constInfo_.sparseCount);',
        '    Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>(), constInfo_.INVALID_IDX, constInfo_.sparseCount);',
    )
    code = replace(
        code,
        '    AscendC::DataCopyPad(indiceOutGm[invalidS1Offset], outInvalidLocal_, dataCopyOutParams);\n}',
        '''    AscendC::DataCopyPad(indiceOutGm[invalidS1Offset], indicesOutLocal_.ReinterpretCast<int32_t>(), dataCopyOutParams);
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::ProcessInvalidRow(int64_t row, int64_t outputOffset)
{
    CleanInvalidOutput(outputOffset);
    if (planHook_ != nullptr) {
        planHook_->PublishGroup(row, row + 1, planRows_, planStorage_,
                                reinterpret_cast<__ubuf__ int32_t *>(indicesOutLocal_.GetPhyAddr()), row);
    }
}''',
    )
    if not grouped:
        publish_short = '''            SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            if (planHook_ != nullptr) {
                const int64_t row = info.indiceOutOffset / topkCount_ + qliCurS1Idx + qliRowIdx;
                planHook_->PublishGroup(row, row + 1, planRows_, planStorage_,
                                        reinterpret_cast<__ubuf__ int32_t *>(indicesOutLocal_.GetPhyAddr()), row);
            }
            continue;'''
        code = replace(
            code,
            '''            SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            continue;''',
            publish_short,
        )
        publish_full = '''        SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        if (planHook_ != nullptr) {
            const int64_t row = info.indiceOutOffset / topkCount_ + qliCurS1Idx + qliRowIdx;
            planHook_->PublishGroup(row, row + 1, planRows_, planStorage_,
                                    reinterpret_cast<__ubuf__ int32_t *>(indicesOutLocal_.GetPhyAddr()), row);
        }'''
        code = replace(
            code,
            '''        SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    }''',
            publish_full + '\n    }',
        )
    else:
        # Publish after the complete AIV-local row group. The last row remains in
        # indicesOutLocal_; RunGroup reads the sibling row from visible GM.
        publish = '''        SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    }
    if (planHook_ != nullptr && qliCurAivS1ProcNum > 0) {
        // qliCurAivS1Idx already includes the tile's qliCurS1Idx.
        const int64_t firstRow = info.indiceOutOffset / topkCount_ + qliCurAivS1Idx;
        const int64_t lastRow = firstRow + qliCurAivS1ProcNum - 1;
@PAIR@
        planHook_->PublishRows(firstRow, qliCurAivS1ProcNum, planRows_, planStorage_,
                               reinterpret_cast<__ubuf__ int32_t *>(indicesOutLocal_.GetPhyAddr()), lastRow);
    }
}
} // namespace QLIKernel'''
        paired_publish = '''        // The paired AIVs process the same complete S1 tile, two rows each.
        if (!constInfo_.isLDOpen && s1BaseSize_ == 4 && qliCurS1ProcNum == 4 &&
            planHook_->TryPublishPair(firstRow - (blockId_ % 2U) * 2,
                                      (blockId_ & 1U) != 0U, planRows_, planStorage_,
                                      reinterpret_cast<__ubuf__ int32_t *>(indicesOutLocal_.GetPhyAddr()), lastRow)) {
            return;
        }
'''
        if pair_two_rows:
            paired_publish += '''        // With a complete two-row tile, both AIVs arrive with one row.
        if (!constInfo_.isLDOpen && s1BaseSize_ == 2 && qliCurS1ProcNum == 2) {
            const bool owner = (blockId_ & 1U) != 0U;
            planHook_->PublishPairRows(firstRow - static_cast<int64_t>(owner), owner, planRows_, planStorage_,
                                        reinterpret_cast<__ubuf__ int32_t *>(indicesOutLocal_.GetPhyAddr()), lastRow);
            return;
        }
'''
        code = replace(
            code,
            '''        SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    }
}
} // namespace QLIKernel''',
            replace(publish, '@PAIR@\n', paired_publish if paired else ''),
        )
    path.write_text(code)


def patch_group_hook(dest: Path) -> None:
    """Add a grouped publication entry without cross-AIV polling."""
    path = dest / 'op_kernel/quant_lightning_indexer_fused_hook.h'
    code = path.read_text()
    vf = r"""
__simt_vf__ __aicore__ LAUNCH_BOUND(QLI_PUBLICATION_THREADS) inline void PublishRowsVf(
    __gm__ int32_t *tokenToReq, __gm__ int32_t *activeRows, __gm__ int32_t *readyCounts,
    int64_t packedRange, int64_t maxRows, uint64_t jointHashCapacity,
    __ubuf__ int64_t *result)
{
    if (threadIdx.x == 0) {
        const int64_t start = packedRange >> 8;
        const int64_t count = packedRange & 255;
        int64_t rows = maxRows;
        const int32_t requested = asc_ldcg(activeRows);
        if (requested <= 0) {
            rows = 0;
        } else if (requested < rows) {
            rows = requested;
        }
        int64_t first = -1;
        int64_t end = -1;
        int64_t winner = 0;
        int64_t eligible = 0;
        if (count > 0 && start >= 0 && start + count <= rows && jointHashCapacity != 0U) {
            const int32_t request = asc_ldcg(tokenToReq + start);
            first = start;
            end = start + count;
            while (first > 0 && asc_ldcg(tokenToReq + first - 1) == request) {
                --first;
            }
            while (end < rows && asc_ldcg(tokenToReq + end) == request) {
                ++end;
            }
            bool sameRequest = true;
            for (int64_t row = start + 1; row < start + count; ++row) {
                if (asc_ldcg(tokenToReq + row) != request) {
                    sameRequest = false;
                }
            }
            if (sameRequest && start >= first && start + count <= end) {
                eligible = 1;
                asc_threadfence();
                const int32_t old = asc_atomic_add(readyCounts + first, static_cast<int32_t>(count));
                winner = old + count == end - first ? 1 : 0;
            }
        }
        result[0] = first;
        result[1] = end;
        result[2] = winner;
        result[3] = eligible;
    }
    asc_syncthreads();
}

"""
    method = r"""
    __aicore__ inline void PublishRows(int64_t start, int64_t count, int64_t rows,
                                       AscendC::LocalTensor<int32_t> storage,
                                       __ubuf__ int32_t *localTopk, int64_t localRow)
    {
        if constexpr (!Enabled) {
            (void)start; (void)count; (void)rows; (void)storage; (void)localTopk; (void)localRow;
            return;
        }
        if (count <= 0 || start < 0 || start >= rows || readyCounts_ == nullptr ||
            buffers_.activeRows == nullptr || buffers_.tokenToReq == nullptr) return;
        if (tiling_.maxRows > 0 && rows > tiling_.maxRows) rows = tiling_.maxRows;
        if (count == 1) {
            PublishGroup(start, start + 1, rows, storage, localTopk, localRow);
            return;
        }
        auto mte3ToScalar = pipe_->AllocEventID<AscendC::HardEvent::MTE3_S>();
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(mte3ToScalar);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(mte3ToScalar);
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_S>(mte3ToScalar);
        auto *publication = reinterpret_cast<__ubuf__ int64_t *>(storage.GetPhyAddr()) +
                            (tiling_.localMemoryBytes - 32U) / sizeof(int64_t);
        asc_vf_call<PublishRowsVf>(
            dim3{QLI_PUBLICATION_THREADS, 1, 1},
            reinterpret_cast<__gm__ int32_t *>(buffers_.tokenToReq),
            reinterpret_cast<__gm__ int32_t *>(buffers_.activeRows), readyCounts_,
            (start << 8) | count, rows, tiling_.jointHashCapacity, publication);
        auto vectorToScalar = pipe_->AllocEventID<AscendC::HardEvent::V_S>();
        AscendC::SetFlag<AscendC::HardEvent::V_S>(vectorToScalar);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(vectorToScalar);
        pipe_->ReleaseEventID<AscendC::HardEvent::V_S>(vectorToScalar);
        const int64_t first = publication[0];
        const int64_t end = publication[1];
        const bool winner = publication[2] != 0;
        const bool eligible = publication[3] != 0;
        if (eligible) {
            if (!winner) return;
            qli_fused_plan::RunGroup(buffers_, tiling_, first, end, rows, *pipe_, storage,
                                     localTopk, localRow);
            if (end - first > 1 && tiling_.jointHashCapacity != 0U) {
                asc_vf_call<ResetGroupVf>(dim3{QLI_PUBLICATION_THREADS, 1, 1}, readyCounts_, first);
            }
        } else {
            // Boundary/count fallback is still per-row and uses GM for both rows.
            for (int64_t row = start; row < start + count; ++row) {
                PublishGroup(row, row + 1, rows, storage, nullptr, -1);
            }
            return;
        }
        vectorToScalar = pipe_->AllocEventID<AscendC::HardEvent::V_S>();
        AscendC::SetFlag<AscendC::HardEvent::V_S>(vectorToScalar);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(vectorToScalar);
        pipe_->ReleaseEventID<AscendC::HardEvent::V_S>(vectorToScalar);
    }
"""
    class_marker = '// The hook is intentionally a separate template parameter.'
    private_marker = '\nprivate:\n    qli_fused_plan::Buffers buffers_{};'
    if code.count(class_marker) != 1 or code.count(private_marker) != 1:
        raise RuntimeError('group hook insertion anchors changed')
    code = code.replace(class_marker, vf + class_marker)
    code = code.replace(private_marker, method + private_marker)
    path.write_text(code)


def patch_pair_hook(dest: Path, pair_two_rows: bool = False) -> None:
    path = dest / 'op_kernel/quant_lightning_indexer_fused_hook.h'
    code = path.read_text()
    method = '''    __aicore__ inline bool TryPublishPair(int64_t first, bool owner, int64_t rows,
                                         AscendC::LocalTensor<int32_t> storage,
                                         __ubuf__ int32_t *localTopk, int64_t localRow)
    {
        if (tiling_.jointHashCapacity == 0U || buffers_.activeRows == nullptr || buffers_.tokenToReq == nullptr) {
            return false;
        }
        AscendC::GlobalTensor<int32_t> active;
        active.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(buffers_.activeRows));
        const int64_t activeCount = active.GetValue(0);
        rows = rows < tiling_.maxRows ? rows : tiling_.maxRows;
        rows = rows < activeCount ? rows : activeCount;
        const int64_t end = first + 4;
        if (first < 0 || end > rows) return false;

        // Both AIVs use the same tile bounds and immutable Plan metadata.
        // A partial, split, or adjacent run with the same request must use GM publication.
        AscendC::GlobalTensor<int32_t> requests;
        requests.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(buffers_.tokenToReq));
        const int32_t request = requests.GetValue(first);
        if ((first > 0 && requests.GetValue(first - 1) == request) ||
            requests.GetValue(first + 1) != request || requests.GetValue(first + 2) != request ||
            requests.GetValue(first + 3) != request || (end < rows && requests.GetValue(end) == request)) {
            return false;
        }

        // Mode 1 is local to the two AIVs of this AI Core. Each AIV publishes
        // after its two TopK MTE3 writes and consumes exactly one flag.
        // The odd AIV owns Plan and still has its final TopK row in UB.
        constexpr uint16_t pairReadyFlag = 8U;
        AscendC::CrossCoreSetFlag<1, PIPE_MTE3>(pairReadyFlag);
        AscendC::CrossCoreWaitFlag<1, PIPE_MTE3>(pairReadyFlag);
        if (owner) {
            auto ready = pipe_->AllocEventID<AscendC::HardEvent::MTE3_S>();
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(ready);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(ready);
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_S>(ready);
            qli_fused_plan::RunGroup(buffers_, tiling_, first, end, rows, *pipe_, storage, localTopk, localRow);
            auto finished = pipe_->AllocEventID<AscendC::HardEvent::V_S>();
            AscendC::SetFlag<AscendC::HardEvent::V_S>(finished);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(finished);
            pipe_->ReleaseEventID<AscendC::HardEvent::V_S>(finished);
        }
        return true;
    }

'''
    if pair_two_rows:
        method += '''    __aicore__ inline void PublishPairRows(int64_t first, bool owner, int64_t rows,
                                           AscendC::LocalTensor<int32_t> storage,
                                           __ubuf__ int32_t *localTopk, int64_t localRow)
    {
        // Both AIVs reach this point for every complete two-row tile. MTE3
        // writes for both rows must precede the owner's single GM publication.
        // PublishRows still handles request boundaries and active-row tails.
        constexpr uint16_t pairReadyFlag = 8U;
        AscendC::CrossCoreSetFlag<1, PIPE_MTE3>(pairReadyFlag);
        AscendC::CrossCoreWaitFlag<1, PIPE_MTE3>(pairReadyFlag);
        if (owner) {
            PublishRows(first, 2, rows, storage, localTopk, localRow);
        }
    }

'''
    marker = '\nprivate:\n    qli_fused_plan::Buffers buffers_{};'
    path.write_text(replace(code, marker, '\n' + method + marker))


def patch_tiling(dest: Path, fixed_four: bool = False) -> None:
    path = dest / 'op_host' / f'{dest.name}_tiling.cpp'
    code = path.read_text()
    code = replace(
        code,
        'fusedS1BaseSize_ = tSize_ <= 64 ? 2U : 4U;',
        'fusedS1BaseSize_ = 4U;  // four-query affinity experiment' if fixed_four else
        'fusedS1BaseSize_ = tSize_ <= 64 ? 2U : 4U;  // pinned standalone Ordered rule',
    )
    marker = '    if (resources.total > qli_fused_host::kUbLimitBytes) {'
    first = code.index(marker)
    second = code.index(marker, first + 1)
    code = code[:first] + code[second:]
    code = code.replace(
        'fused QLI and plan UB footprint exceeds 216 KiB',
        'native ordered + Plan footprint exceeds 216 KiB',
    )
    path.write_text(code)


def patch_shell_workspace(dest: Path) -> None:
    """The shell carries the fused ABI but treats Plan scratch as a dummy."""
    binding = dest / 'torch_extension/csrc' / f'{dest.name}.cpp'
    code = binding.read_text()
    suffix = '::WorkspaceWords(reqIds.numel(),capacity,sparseCount);'
    candidates = [f'    const uint64_t requiredCompact=qli_fusion_resource{suffix}',
                  f'    const uint64_t requiredCompact={dest.name}_resource{suffix}']
    old = next((item for item in candidates if item in code), None)
    if old is None:
        raise RuntimeError('shell binding workspace anchor changed')
    code = code.replace(
        old,
        '    // Plan is disabled: any non-empty ABI placeholder is sufficient.\n'
        '    const uint64_t requiredCompact=1;',
    )
    binding.write_text(code)
    tiling = dest / 'op_host' / f'{dest.name}_tiling.cpp'
    code = tiling.read_text()
    suffix = '::WorkspaceWords(dim0(0),qliParams_.capacity,topk);'
    candidates = [f'    const uint64_t required=qli_fusion_resource{suffix}',
                  f'    const uint64_t required={dest.name}_resource{suffix}']
    old = next((item for item in candidates if item in code), None)
    if old is None:
        raise RuntimeError('shell tiling workspace anchor changed')
    code = code.replace(
        old,
        '    // Plan is disabled: the host still validates shape but not Plan GM scratch.\n'
        '    const uint64_t required=1;',
    )
    tiling.write_text(code)


def write_provenance(dest: Path, commit: str, ordered_tree: str, fused_source: str,
                     native_source: str) -> None:
    (dest / 'NATIVE_ORDERED_PROVENANCE.md').write_text(
        '# Native ordered provenance\n\n'
        f'- Device QLI/Cube/Vector/ordered TopK: `{native_source}` at `{commit}`.\n'
        f'- Ordered TopK tree: `{ordered_tree}`.\n'
        f'- Plan ABI/host/publication hook source: `{fused_source}` at `{commit}`.\n'
        '- Generated by `scripts/prepare_qli_fusion.py`; the upstream operator is not edited.\n'
        '- `native_shell` disables the Plan hook and leaves Plan tensors untouched.\n'
        '- `native_fused` publishes each native row after its MTE3 output write.\n'
    )
