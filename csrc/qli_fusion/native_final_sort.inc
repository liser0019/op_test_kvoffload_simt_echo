        // The original score TopK has finished. Reuse its scratch to order only
        // the selected positions. All 2048 entries are valid on this branch.
        // Short/empty rows already have an ascending prefix and -1 padding.
        auto orderedIndices = indicesOutLocal_;
        if (topkCount_ == 2048 && qliValidS2Len >= topkCount_) {
            // uint32 radix sort does not round IDs or change selection at ties.
            // With a read-only source the temporary requirement is 11 * 2048
            // + 512 bytes. Both native score types reserve at least 32 KiB in
            // mrgValueBuf_; topkSharedTmpBuf_ starts with 2048 history indices.
            orderedIndices = topkSharedTmpLocal_;
            PipeBarrier<PIPE_V>();
            AscendC::Sort<uint32_t>(orderedIndices, indicesOutLocal_,
                                   mrgValueLocal_.template ReinterpretCast<uint8_t>(), topkCount_);
        }
