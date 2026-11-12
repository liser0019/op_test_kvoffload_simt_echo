// SPDX-License-Identifier: Apache-2.0
// Two isolated experiments over frozen196. No timing output or persistent ABI changes.
#pragma once
#if JOINT_MTP7_MID_UNIQUE
// Work region, in int32 words, chunk=C<=612:
// rawA[16C,24C), rawB[0,8C) -> compactA[24C,32C), compactB[32C,40C).
// final hardware output[0,16C); dead rawA[16C,24C) becomes midUID->unionUID.
// query map temporarily stores -(midUID+2), never a persistent output.
__simd_vf__ inline void EightMidUniqueVf(__ubuf__ int32_t *packet, __ubuf__ int32_t *queryMap,
                                         __ubuf__ int32_t *ctx, uint32_t q, uint32_t chunk, uint32_t na,
                                         uint32_t nb) {
    using namespace AscendC;
    Reg::RegTensor<uint32_t> lane, zero, one, even, odd, prevIndex, lastIndex, payload, flags, prefix, carry;
    Reg::RegTensor<uint32_t> uid, midId, index, safeIndex, encoded, value;
    Reg::RegTensor<float> score, previous, shifted, lastScore;
    Reg::MaskReg full = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg valid, first, head, selected;
    Reg::Unsqueeze<uint32_t>(lane, full);
    Reg::Duplicate(zero, uint32_t{0});
    Reg::Duplicate(one, uint32_t{1});
    Reg::Compares<uint32_t, CMPMODE::EQ>(first, lane, uint32_t{0}, full);
    Reg::Adds(prevIndex, lane, uint32_t(-1), full);
    Reg::Select(prevIndex, zero, prevIndex, first);
    for (uint32_t half = 0; half < 2U; ++half) {
        const uint32_t n = half ? nb : na, input = half ? 0U : 16U * chunk,
                       output = (24U + 8U * half) * chunk;
        const uint32_t midBase = half * 4U * chunk;
        if (half) {
            // rawA is dead only after half0 has finished all its reads.
            Reg::LocalMemBar<Reg::MemType::VEC_ALL, Reg::MemType::VEC_ALL>();
            Reg::Duplicate(value, uint32_t(-1));
            for (uint32_t base = 0; base < 8U * chunk; base += 64U) {
                uint32_t rem = 8U * chunk - base;
                valid = Reg::UpdateMask<uint32_t>(rem);
                Reg::Adds(index, lane, 16U * chunk + base, full);
                Reg::Adds(safeIndex, lane, 8U * chunk, full);
                Reg::Select(index, index, safeIndex, valid);
                Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(packet), value, index,
                                                 valid);
            }
        }
        Reg::Duplicate(carry, uint32_t{0});
        Reg::Duplicate(lastScore, 1.0f); // valid scores are -token<=0
        for (uint32_t base = 0; base < n; base += 64U) {
            uint32_t rem = n - base;
            valid = Reg::UpdateMask<uint32_t>(rem);
            const uint32_t count = n - base < 64U ? n - base : 64U;
            Reg::Adds(even, lane, base, full);
            Reg::ShiftLefts(even, even, int16_t{1}, full);
            Reg::Adds(even, even, input, full);
            Reg::Adds(odd, even, uint32_t{1}, full);
            Reg::Duplicate(score, 0.0f);
            Reg::Duplicate(payload, uint32_t{0});
            Reg::Gather<float, float, uint32_t>(score, reinterpret_cast<__ubuf__ float *>(packet), even,
                                                valid);
            Reg::Gather<uint32_t, uint32_t, uint32_t>(payload, reinterpret_cast<__ubuf__ uint32_t *>(packet),
                                                      odd, valid);
            Reg::Gather<float, uint32_t>(shifted, score, prevIndex);
            Reg::Select(previous, lastScore, shifted, first);
            Reg::Compare<float, CMPMODE::NE>(head, score, previous, valid);
            Reg::Select(flags, one, zero, head);
            Reg::Select(flags, flags, zero, valid);
            Reg::Compares<uint32_t, CMPMODE::NE>(head, flags, uint32_t{0}, full);
            Reg::Unsqueeze<uint32_t>(prefix, head);
            Reg::Add(prefix, prefix, flags, full);
            Reg::Add(prefix, prefix, carry, full);
            Reg::Adds(uid, prefix, uint32_t(-1), full);
            Reg::Adds(midId, uid, midBase, full);
            Reg::ShiftLefts(index, uid, int16_t{1}, full);
            Reg::Adds(index, index, output, full);
            Reg::Adds(safeIndex, lane, 8U * chunk, full);
            Reg::Select(index, index, safeIndex, head);
            Reg::Scatter<float, uint32_t>(reinterpret_cast<__ubuf__ float *>(packet), score, index, head);
            Reg::ShiftLefts(index, uid, int16_t{1}, full);
            Reg::Adds(index, index, output + 1U, full);
            Reg::Select(index, index, safeIndex, head);
            Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(packet), midId, index,
                                             head);
            // Every raw query receives a temporary MID ID, including duplicate token positions.
            Reg::Adds(encoded, midId, uint32_t{2}, full);
            Reg::Sub(encoded, zero, encoded, full);
            Reg::Adds(safeIndex, lane, q, full);
            Reg::Select(payload, payload, safeIndex, valid);
            Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(queryMap), encoded,
                                             payload, valid);
            Reg::Duplicate(lastIndex, uint32_t{63});
            Reg::Gather<uint32_t, uint32_t>(carry, prefix, lastIndex);
            Reg::Duplicate(lastIndex, count - 1U);
            Reg::Gather<float, uint32_t>(lastScore, score, lastIndex);
        }
        Reg::Compares<uint32_t, CMPMODE::EQ>(selected, lane, 104U + half - 64U, full);
        // ctx base remains32B aligned; actual destination is104+half, inactive indices64..127.
        Reg::Adds(index, lane, 64U, full);
        Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(ctx), carry, index, selected);
    }
}

// Resolve temporary query->MID IDs after final merge, and advance only consumed original queries.
__simt_vf__ __aicore__ LAUNCH_BOUND(SORTED_THREADS) inline void EightResolveMidMapVf(
    __ubuf__ int32_t *packet, __ubuf__ int32_t *queryMap, __ubuf__ int32_t *ctx, uint32_t chunk,
    uint32_t topk) {
    const uint32_t t = threadIdx.x, active = ctx[28];
    auto *midToUnion = packet + 16U * chunk;
    auto *partials = packet + 8U * chunk;
    uint32_t bins[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (uint32_t i = t; i < active * chunk; i += SORTED_THREADS) {
        const uint32_t k = i / chunk, j = i % chunk, row = ctx[64U + k];
        if (j >= static_cast<uint32_t>(ctx[72U + k]))
            continue;
        const uint32_t qid = row * topk + static_cast<uint32_t>(ctx[48U + row]) + j;
        const int32_t code = queryMap[qid];
        if (code < -1) {
            const uint32_t mid = static_cast<uint32_t>(-code - 2);
            if (mid >= 8U * chunk) {
                asc_atomic_min(ctx, 0);
                continue;
            }
            const int32_t uid = midToUnion[mid];
            queryMap[qid] = uid;
#pragma unroll
            for (uint32_t r = 0; r < 8U; ++r)
                bins[r] += (row == r && uid >= 0) ? 1U : 0U;
        } else if (code != -1)
            asc_atomic_min(ctx, 0);
    }
    const uint32_t lane = t & 31U, warp = t / 32U;
#pragma unroll
    for (uint32_t r = 0; r < 8U; ++r) {
        uint32_t v = bins[r];
#pragma unroll
        for (uint32_t d = 16U; d; d >>= 1U)
            v += static_cast<uint32_t>(asc_shfl(static_cast<int32_t>(v), lane ^ d, 32));
        if (lane == 0U)
            partials[warp * 8U + r] = static_cast<int32_t>(v);
    }
    asc_syncthreads();
    if (t < 8U) {
        uint32_t total = 0;
        for (uint32_t w = 0; w < SORTED_WARPS; ++w)
            total += static_cast<uint32_t>(partials[w * 8U + t]);
        ctx[48U + t] += total;
    }
    asc_syncthreads();
    if (t == 0U) {
        ctx[6] = 0;
        for (uint32_t row = 0; row < 8U; ++row)
            if (ctx[48U + row] < ctx[56U + row])
                ctx[6] = 1;
    }
    asc_syncthreads();
}
__simd_vf__ inline void RefineMidUniqueMapVf(__ubuf__ int32_t *records, __ubuf__ int32_t *data,
                                             __ubuf__ int32_t *ctx, uint32_t n, uint32_t q, uint32_t capacity,
                                             uint32_t initialCount, int32_t initialLast, uint32_t more,
                                             __ubuf__ int32_t *midToUnion, uint32_t midCapacity) {
    using namespace AscendC;
    Reg::RegTensor<uint32_t> lane, zero, one, even, odd, previousIndex, lastIndex;
    Reg::RegTensor<uint32_t> flags, prefix, carry, uid, payload, index, safeIndex, writeFlag, status, value;
    Reg::RegTensor<float> score, previous, lastScore, shiftScore, positive;
    Reg::RegTensor<int32_t> token, lastToken;
    Reg::MaskReg full = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg first, head, valid, fits, writeHead, chosen;
    Reg::Unsqueeze<uint32_t>(lane, full);
    Reg::Duplicate(zero, uint32_t{0});
    Reg::Duplicate(one, uint32_t{1});
    Reg::Duplicate(carry, initialCount);
    Reg::Duplicate(lastToken, initialLast);
    Reg::Duplicate(lastScore, -static_cast<float>(initialLast));
    Reg::Compares<uint32_t, CMPMODE::EQ>(first, lane, uint32_t{0}, full);
    Reg::Adds(previousIndex, lane, uint32_t(-1), full);
    Reg::Select(previousIndex, zero, previousIndex, first);
    static constexpr Reg::CastTrait toInt = {Reg::RegLayout::UNKNOWN, Reg::SatMode::NO_SAT,
                                             Reg::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};
    for (uint32_t base = 0; base < n; base += 64U) {
        uint32_t remaining = n - base;
        valid = Reg::UpdateMask<uint32_t>(remaining);
        const uint32_t count = n - base < 64U ? n - base : 64U;
        Reg::Adds(even, lane, base, full);
        Reg::ShiftLefts(even, even, int16_t{1}, full);
        Reg::Adds(odd, even, uint32_t{1}, full);
        Reg::Duplicate(score, 0.0f);
        Reg::Gather<float, float, uint32_t>(score, reinterpret_cast<__ubuf__ float *>(records), even, valid);
        Reg::Gather<float, uint32_t>(shiftScore, score, previousIndex);
        Reg::Select(previous, lastScore, shiftScore, first);
        Reg::Compare<float, CMPMODE::NE>(head, score, previous, valid);
        Reg::Select(flags, one, zero, head);
        Reg::Select(flags, flags, zero, valid);
        Reg::Compares<uint32_t, CMPMODE::NE>(head, flags, uint32_t{0}, full);
        Reg::Unsqueeze<uint32_t>(prefix, head);
        Reg::Add(prefix, prefix, flags, full);
        Reg::Add(prefix, prefix, carry, full);
        Reg::Adds(uid, prefix, uint32_t(-1), full);
        Reg::Neg(positive, score, full);
        Reg::Cast<int32_t, float, toInt>(token, positive, valid);
        Reg::Compares<uint32_t, CMPMODE::LT>(fits, uid, capacity, full);
        Reg::Select(writeFlag, flags, zero, fits);
        Reg::Compares<uint32_t, CMPMODE::NE>(writeHead, writeFlag, uint32_t{0}, full);
        Reg::Adds(index, uid, q, full);
        Reg::Adds(safeIndex, lane, q + capacity, full);
        Reg::Select(index, index, safeIndex, writeHead);
        Reg::Scatter<int32_t, uint32_t>(data, token, index, writeHead);
        Reg::Duplicate(payload, uint32_t{0});
        Reg::Gather<uint32_t, uint32_t, uint32_t>(payload, reinterpret_cast<__ubuf__ uint32_t *>(records),
                                                  odd, valid);
        Reg::Adds(safeIndex, lane, midCapacity, full);
        Reg::Select(payload, payload, safeIndex, valid);
        Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(midToUnion), uid, payload,
                                         valid);
        Reg::Duplicate(lastIndex, uint32_t{63});
        Reg::Gather<uint32_t, uint32_t>(carry, prefix, lastIndex);
        Reg::Duplicate(lastIndex, count - 1U);
        Reg::Gather<float, uint32_t>(lastScore, score, lastIndex);
        Reg::Gather<int32_t, uint32_t>(lastToken, token, lastIndex);
    }
    Reg::Compares<uint32_t, CMPMODE::EQ>(chosen, lane, uint32_t{1}, full);
    Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(ctx), carry, lane, chosen);
    Reg::Compares<uint32_t, CMPMODE::EQ>(chosen, lane, uint32_t{2}, full);
    Reg::Scatter<int32_t, uint32_t>(ctx, lastToken, lane, chosen);
    Reg::Compares<uint32_t, CMPMODE::GE>(fits, carry, capacity + 1U, full);
    Reg::Select(status, zero, one, fits);
    Reg::Compares<uint32_t, CMPMODE::EQ>(chosen, lane, uint32_t{0}, full);
    Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(ctx), status, lane, chosen);
    Reg::Compares<uint32_t, CMPMODE::EQ>(chosen, lane, uint32_t{29}, full);
    Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(ctx), one, lane, chosen);
    Reg::Duplicate(value, more);
    Reg::Compares<uint32_t, CMPMODE::EQ>(chosen, lane, uint32_t{6}, full);
    Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(ctx), value, lane, chosen);
}

#endif // JOINT_MTP7_MID_UNIQUE
#if JOINT_MTP7_FUSED_COUNT
__simd_vf__ inline void RefineFusedCountVf(__ubuf__ int32_t *records, __ubuf__ int32_t *data,
                                           __ubuf__ int32_t *ctx, uint32_t n, uint32_t q, uint32_t capacity,
                                           uint32_t initialCount, int32_t initialLast, uint32_t more,
                                           uint32_t topk) {
    using namespace AscendC;
    Reg::RegTensor<uint32_t> lane, zero, one, even, odd, previousIndex, lastIndex;
    Reg::RegTensor<uint32_t> flags, prefix, carry, uid, payload, index, safeIndex, writeFlag, status, value;
    Reg::RegTensor<float> score, previous, lastScore, shiftScore, positive;
    Reg::RegTensor<int32_t> token, lastToken;
    Reg::MaskReg full = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg first, head, valid, fits, writeHead, chosen;
    Reg::Unsqueeze<uint32_t>(lane, full);
    Reg::Duplicate(zero, uint32_t{0});
    Reg::Duplicate(one, uint32_t{1});
    Reg::Duplicate(carry, initialCount);
    Reg::Duplicate(lastToken, initialLast);
    Reg::Duplicate(lastScore, -static_cast<float>(initialLast));
    Reg::Compares<uint32_t, CMPMODE::EQ>(first, lane, uint32_t{0}, full);
    Reg::Adds(previousIndex, lane, uint32_t(-1), full);
    Reg::Select(previousIndex, zero, previousIndex, first);
    static constexpr Reg::CastTrait toInt = {Reg::RegLayout::UNKNOWN, Reg::SatMode::NO_SAT,
                                             Reg::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};

    Reg::RegTensor<uint32_t> bin0, bin1, bin2, bin3, bin4, bin5, bin6, bin7;
    Reg::Duplicate(bin0, uint32_t{0});
    Reg::Duplicate(bin1, uint32_t{0});
    Reg::Duplicate(bin2, uint32_t{0});
    Reg::Duplicate(bin3, uint32_t{0});
    Reg::Duplicate(bin4, uint32_t{0});
    Reg::Duplicate(bin5, uint32_t{0});
    Reg::Duplicate(bin6, uint32_t{0});
    Reg::Duplicate(bin7, uint32_t{0});
    for (uint32_t base = 0; base < n; base += 64U) {
        uint32_t remaining = n - base;
        valid = Reg::UpdateMask<uint32_t>(remaining);
        const uint32_t count = n - base < 64U ? n - base : 64U;
        Reg::Adds(even, lane, base, full);
        Reg::ShiftLefts(even, even, int16_t{1}, full);
        Reg::Adds(odd, even, uint32_t{1}, full);
        Reg::Duplicate(score, 0.0f);
        Reg::Gather<float, float, uint32_t>(score, reinterpret_cast<__ubuf__ float *>(records), even, valid);
        Reg::Gather<float, uint32_t>(shiftScore, score, previousIndex);
        Reg::Select(previous, lastScore, shiftScore, first);
        Reg::Compare<float, CMPMODE::NE>(head, score, previous, valid);
        Reg::Select(flags, one, zero, head);
        Reg::Select(flags, flags, zero, valid);
        Reg::Compares<uint32_t, CMPMODE::NE>(head, flags, uint32_t{0}, full);
        Reg::Unsqueeze<uint32_t>(prefix, head);
        Reg::Add(prefix, prefix, flags, full);
        Reg::Add(prefix, prefix, carry, full);
        Reg::Adds(uid, prefix, uint32_t(-1), full);
        Reg::Neg(positive, score, full);
        Reg::Cast<int32_t, float, toInt>(token, positive, valid);
        Reg::Compares<uint32_t, CMPMODE::LT>(fits, uid, capacity, full);
        Reg::Select(writeFlag, flags, zero, fits);
        Reg::Compares<uint32_t, CMPMODE::NE>(writeHead, writeFlag, uint32_t{0}, full);
        Reg::Adds(index, uid, q, full);
        Reg::Adds(safeIndex, lane, q + capacity, full);
        Reg::Select(index, index, safeIndex, writeHead);
        Reg::Scatter<int32_t, uint32_t>(data, token, index, writeHead);
        Reg::Duplicate(payload, uint32_t{0});
        Reg::Gather<uint32_t, uint32_t, uint32_t>(payload, reinterpret_cast<__ubuf__ uint32_t *>(records),
                                                  odd, valid);
        Reg::Adds(safeIndex, lane, q, full);
        Reg::Select(payload, payload, safeIndex, valid);
        // Count every original query, not only the first record of each token run.
        Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, 1U * topk, full);
        Reg::Select(value, one, zero, chosen);
        Reg::Add(bin0, bin0, value, full);
        Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, 2U * topk, full);
        Reg::Select(value, one, zero, chosen);
        Reg::Add(bin1, bin1, value, full);
        Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, 3U * topk, full);
        Reg::Select(value, one, zero, chosen);
        Reg::Add(bin2, bin2, value, full);
        Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, 4U * topk, full);
        Reg::Select(value, one, zero, chosen);
        Reg::Add(bin3, bin3, value, full);
        Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, 5U * topk, full);
        Reg::Select(value, one, zero, chosen);
        Reg::Add(bin4, bin4, value, full);
        Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, 6U * topk, full);
        Reg::Select(value, one, zero, chosen);
        Reg::Add(bin5, bin5, value, full);
        Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, 7U * topk, full);
        Reg::Select(value, one, zero, chosen);
        Reg::Add(bin6, bin6, value, full);
        Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, 8U * topk, full);
        Reg::Select(value, one, zero, chosen);
        Reg::Add(bin7, bin7, value, full);

        Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(data), uid, payload, valid);
        Reg::Duplicate(lastIndex, uint32_t{63});
        Reg::Gather<uint32_t, uint32_t>(carry, prefix, lastIndex);
        Reg::Duplicate(lastIndex, count - 1U);
        Reg::Gather<float, uint32_t>(lastScore, score, lastIndex);
        Reg::Gather<int32_t, uint32_t>(lastToken, token, lastIndex);
    }
    Reg::Compares<uint32_t, CMPMODE::EQ>(chosen, lane, uint32_t{1}, full);
    Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(ctx), carry, lane, chosen);
    Reg::Compares<uint32_t, CMPMODE::EQ>(chosen, lane, uint32_t{2}, full);
    Reg::Scatter<int32_t, uint32_t>(ctx, lastToken, lane, chosen);
    Reg::Compares<uint32_t, CMPMODE::GE>(fits, carry, capacity + 1U, full);
    Reg::Select(status, zero, one, fits);
    Reg::Compares<uint32_t, CMPMODE::EQ>(chosen, lane, uint32_t{0}, full);
    Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(ctx), status, lane, chosen);
    Reg::Compares<uint32_t, CMPMODE::EQ>(chosen, lane, uint32_t{29}, full);
    Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(ctx), one, lane, chosen);
    Reg::Duplicate(value, more);
    Reg::Compares<uint32_t, CMPMODE::EQ>(chosen, lane, uint32_t{6}, full);
    Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(ctx), value, lane, chosen);

    // Reduce each cumulative counter once per packet. Only lane0 of Reduce<SUM> is read.
    Reg::RegTensor<uint32_t> reduced, previousCount, cumulative, delta, oldCursor, endCursor;
    Reg::Duplicate(previousCount, uint32_t{0});
    Reg::Duplicate(even, uint32_t{0});
    Reg::Compares<uint32_t, CMPMODE::EQ>(first, lane, uint32_t{0}, full);
    Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin0, full);
    Reg::Gather<uint32_t, uint32_t>(cumulative, reduced, even);
    Reg::Sub(delta, cumulative, previousCount, full);
    previousCount = cumulative;
    Reg::Adds(index, lane, 48U, full);
    Reg::Duplicate(oldCursor, uint32_t{0});
    Reg::Gather<uint32_t, uint32_t, uint32_t>(oldCursor, reinterpret_cast<__ubuf__ uint32_t *>(ctx), index,
                                              first);
    Reg::Add(oldCursor, oldCursor, delta, full);
    Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(ctx), oldCursor, index, first);
    Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin1, full);
    Reg::Gather<uint32_t, uint32_t>(cumulative, reduced, even);
    Reg::Sub(delta, cumulative, previousCount, full);
    previousCount = cumulative;
    Reg::Adds(index, lane, 49U, full);
    Reg::Duplicate(oldCursor, uint32_t{0});
    Reg::Gather<uint32_t, uint32_t, uint32_t>(oldCursor, reinterpret_cast<__ubuf__ uint32_t *>(ctx), index,
                                              first);
    Reg::Add(oldCursor, oldCursor, delta, full);
    Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(ctx), oldCursor, index, first);
    Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin2, full);
    Reg::Gather<uint32_t, uint32_t>(cumulative, reduced, even);
    Reg::Sub(delta, cumulative, previousCount, full);
    previousCount = cumulative;
    Reg::Adds(index, lane, 50U, full);
    Reg::Duplicate(oldCursor, uint32_t{0});
    Reg::Gather<uint32_t, uint32_t, uint32_t>(oldCursor, reinterpret_cast<__ubuf__ uint32_t *>(ctx), index,
                                              first);
    Reg::Add(oldCursor, oldCursor, delta, full);
    Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(ctx), oldCursor, index, first);
    Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin3, full);
    Reg::Gather<uint32_t, uint32_t>(cumulative, reduced, even);
    Reg::Sub(delta, cumulative, previousCount, full);
    previousCount = cumulative;
    Reg::Adds(index, lane, 51U, full);
    Reg::Duplicate(oldCursor, uint32_t{0});
    Reg::Gather<uint32_t, uint32_t, uint32_t>(oldCursor, reinterpret_cast<__ubuf__ uint32_t *>(ctx), index,
                                              first);
    Reg::Add(oldCursor, oldCursor, delta, full);
    Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(ctx), oldCursor, index, first);
    Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin4, full);
    Reg::Gather<uint32_t, uint32_t>(cumulative, reduced, even);
    Reg::Sub(delta, cumulative, previousCount, full);
    previousCount = cumulative;
    Reg::Adds(index, lane, 52U, full);
    Reg::Duplicate(oldCursor, uint32_t{0});
    Reg::Gather<uint32_t, uint32_t, uint32_t>(oldCursor, reinterpret_cast<__ubuf__ uint32_t *>(ctx), index,
                                              first);
    Reg::Add(oldCursor, oldCursor, delta, full);
    Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(ctx), oldCursor, index, first);
    Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin5, full);
    Reg::Gather<uint32_t, uint32_t>(cumulative, reduced, even);
    Reg::Sub(delta, cumulative, previousCount, full);
    previousCount = cumulative;
    Reg::Adds(index, lane, 53U, full);
    Reg::Duplicate(oldCursor, uint32_t{0});
    Reg::Gather<uint32_t, uint32_t, uint32_t>(oldCursor, reinterpret_cast<__ubuf__ uint32_t *>(ctx), index,
                                              first);
    Reg::Add(oldCursor, oldCursor, delta, full);
    Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(ctx), oldCursor, index, first);
    Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin6, full);
    Reg::Gather<uint32_t, uint32_t>(cumulative, reduced, even);
    Reg::Sub(delta, cumulative, previousCount, full);
    previousCount = cumulative;
    Reg::Adds(index, lane, 54U, full);
    Reg::Duplicate(oldCursor, uint32_t{0});
    Reg::Gather<uint32_t, uint32_t, uint32_t>(oldCursor, reinterpret_cast<__ubuf__ uint32_t *>(ctx), index,
                                              first);
    Reg::Add(oldCursor, oldCursor, delta, full);
    Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(ctx), oldCursor, index, first);
    Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin7, full);
    Reg::Gather<uint32_t, uint32_t>(cumulative, reduced, even);
    Reg::Sub(delta, cumulative, previousCount, full);
    previousCount = cumulative;
    Reg::Adds(index, lane, 55U, full);
    Reg::Duplicate(oldCursor, uint32_t{0});
    Reg::Gather<uint32_t, uint32_t, uint32_t>(oldCursor, reinterpret_cast<__ubuf__ uint32_t *>(ctx), index,
                                              first);
    Reg::Add(oldCursor, oldCursor, delta, full);
    Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(ctx), oldCursor, index, first);
    Reg::LocalMemBar<Reg::MemType::VEC_ALL, Reg::MemType::VEC_ALL>();
    uint32_t eight = 8U;
    valid = Reg::UpdateMask<uint32_t>(eight);
    Reg::Adds(index, lane, 48U, full);
    Reg::Duplicate(oldCursor, uint32_t{0});
    Reg::Gather<uint32_t, uint32_t, uint32_t>(oldCursor, reinterpret_cast<__ubuf__ uint32_t *>(ctx), index,
                                              valid);
    Reg::Adds(index, lane, 56U, full);
    Reg::Duplicate(endCursor, uint32_t{0});
    Reg::Gather<uint32_t, uint32_t, uint32_t>(endCursor, reinterpret_cast<__ubuf__ uint32_t *>(ctx), index,
                                              valid);
    Reg::Compare<uint32_t, CMPMODE::LT>(chosen, oldCursor, endCursor, valid);
    Reg::Select(value, one, zero, chosen);
    Reg::Select(value, value, zero, valid);
    Reg::Reduce<Reg::ReduceType::SUM>(reduced, value, full);
    Reg::Gather<uint32_t, uint32_t>(cumulative, reduced, even);
    Reg::Compares<uint32_t, CMPMODE::NE>(chosen, cumulative, uint32_t{0}, full);
    Reg::Select(value, one, zero, chosen);
    Reg::Compares<uint32_t, CMPMODE::EQ>(chosen, lane, uint32_t{6}, full);
    Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(ctx), value, lane, chosen);
}

#endif // JOINT_MTP7_FUSED_COUNT
