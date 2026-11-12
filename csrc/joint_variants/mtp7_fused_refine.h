// SPDX-License-Identifier: Apache-2.0
// Frozen FUSED refinement: A=hardware counts, B=aligned pair load, C=lean/batched cursor.
#pragma once
template <uint32_t Bits, int KnownHalf>
__simd_vf__ inline void RefineFusedTuneVf(__ubuf__ int32_t *records, __ubuf__ int32_t *data,
                                          __ubuf__ int32_t *ctx, uint32_t n, uint32_t q, uint32_t capacity,
                                          uint32_t initialCount, int32_t initialLast, uint32_t topk,
                                          uint32_t knownTotal, uint32_t unknownTotal) {
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

    // A: KnownHalf=-1 counts all rows;0/1 reuses that complete half;2 reuses both.
    // C: last cumulative counter equals the known effective record count.
    static_assert(KnownHalf >= -1 && KnownHalf <= 2);
    constexpr bool PackedCursor = (Bits & 4U) != 0;
    constexpr uint32_t CountRows = KnownHalf < 0 ? 8U : KnownHalf == 2 ? 0U : 4U;
    constexpr uint32_t Counters = CountRows - (PackedCursor && CountRows ? 1U : 0U);
    Reg::RegTensor<uint32_t> bin0, bin1, bin2, bin3, bin4, bin5, bin6, bin7;
    if constexpr (0U < Counters)
        Reg::Duplicate(bin0, uint32_t{0});
    if constexpr (1U < Counters)
        Reg::Duplicate(bin1, uint32_t{0});
    if constexpr (2U < Counters)
        Reg::Duplicate(bin2, uint32_t{0});
    if constexpr (3U < Counters)
        Reg::Duplicate(bin3, uint32_t{0});
    if constexpr (4U < Counters)
        Reg::Duplicate(bin4, uint32_t{0});
    if constexpr (5U < Counters)
        Reg::Duplicate(bin5, uint32_t{0});
    if constexpr (6U < Counters)
        Reg::Duplicate(bin6, uint32_t{0});
    if constexpr (7U < Counters)
        Reg::Duplicate(bin7, uint32_t{0});
    if constexpr ((Bits & 2U) != 0) {
        for (uint32_t base = 0; base < (n & ~63U); base += 64U) {
            uint32_t remaining = 64U;
            valid = Reg::UpdateMask<uint32_t>(remaining);
            const uint32_t count = n - base < 64U ? n - base : 64U;
            // uint32 loads preserve score bits; query IDs are never numerically cast to float.
            Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B32>(
                reinterpret_cast<Reg::RegTensor<uint32_t> &>(score), payload,
                reinterpret_cast<__ubuf__ uint32_t *>(records) + 2U * base);
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
            Reg::Adds(safeIndex, lane, q, full);
            Reg::Select(payload, payload, safeIndex, valid);
            // Every original query counts, including repeated token positions.
            if constexpr (0U < Counters) {
                constexpr uint32_t Boundary = (KnownHalf == 0 ? 5U : 1U) + 0U;
                Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, Boundary * topk, full);
                Reg::Select(value, one, zero, chosen);
                Reg::Add(bin0, bin0, value, full);
            }
            if constexpr (1U < Counters) {
                constexpr uint32_t Boundary = (KnownHalf == 0 ? 5U : 1U) + 1U;
                Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, Boundary * topk, full);
                Reg::Select(value, one, zero, chosen);
                Reg::Add(bin1, bin1, value, full);
            }
            if constexpr (2U < Counters) {
                constexpr uint32_t Boundary = (KnownHalf == 0 ? 5U : 1U) + 2U;
                Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, Boundary * topk, full);
                Reg::Select(value, one, zero, chosen);
                Reg::Add(bin2, bin2, value, full);
            }
            if constexpr (3U < Counters) {
                constexpr uint32_t Boundary = (KnownHalf == 0 ? 5U : 1U) + 3U;
                Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, Boundary * topk, full);
                Reg::Select(value, one, zero, chosen);
                Reg::Add(bin3, bin3, value, full);
            }
            if constexpr (4U < Counters) {
                constexpr uint32_t Boundary = (KnownHalf == 0 ? 5U : 1U) + 4U;
                Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, Boundary * topk, full);
                Reg::Select(value, one, zero, chosen);
                Reg::Add(bin4, bin4, value, full);
            }
            if constexpr (5U < Counters) {
                constexpr uint32_t Boundary = (KnownHalf == 0 ? 5U : 1U) + 5U;
                Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, Boundary * topk, full);
                Reg::Select(value, one, zero, chosen);
                Reg::Add(bin5, bin5, value, full);
            }
            if constexpr (6U < Counters) {
                constexpr uint32_t Boundary = (KnownHalf == 0 ? 5U : 1U) + 6U;
                Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, Boundary * topk, full);
                Reg::Select(value, one, zero, chosen);
                Reg::Add(bin6, bin6, value, full);
            }
            if constexpr (7U < Counters) {
                constexpr uint32_t Boundary = (KnownHalf == 0 ? 5U : 1U) + 7U;
                Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, Boundary * topk, full);
                Reg::Select(value, one, zero, chosen);
                Reg::Add(bin7, bin7, value, full);
            }
            Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(data), uid, payload,
                                             valid);
            Reg::Duplicate(lastIndex, uint32_t{63});
            Reg::Gather<uint32_t, uint32_t>(carry, prefix, lastIndex);
            Reg::Duplicate(lastIndex, count - 1U);
            Reg::Gather<float, uint32_t>(lastScore, score, lastIndex);
            Reg::Gather<int32_t, uint32_t>(lastToken, token, lastIndex);
        }
        for (uint32_t base = (n & ~63U); base < n; base += 64U) {
            uint32_t remaining = n - base;
            valid = Reg::UpdateMask<uint32_t>(remaining);
            const uint32_t count = n - base < 64U ? n - base : 64U;
            Reg::Adds(even, lane, base, full);
            Reg::ShiftLefts(even, even, int16_t{1}, full);
            Reg::Adds(odd, even, uint32_t{1}, full);
            Reg::Duplicate(score, 0.0f);
            Reg::Gather<float, float, uint32_t>(score, reinterpret_cast<__ubuf__ float *>(records), even,
                                                valid);
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
            // Every original query counts, including repeated token positions.
            if constexpr (0U < Counters) {
                constexpr uint32_t Boundary = (KnownHalf == 0 ? 5U : 1U) + 0U;
                Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, Boundary * topk, full);
                Reg::Select(value, one, zero, chosen);
                Reg::Add(bin0, bin0, value, full);
            }
            if constexpr (1U < Counters) {
                constexpr uint32_t Boundary = (KnownHalf == 0 ? 5U : 1U) + 1U;
                Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, Boundary * topk, full);
                Reg::Select(value, one, zero, chosen);
                Reg::Add(bin1, bin1, value, full);
            }
            if constexpr (2U < Counters) {
                constexpr uint32_t Boundary = (KnownHalf == 0 ? 5U : 1U) + 2U;
                Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, Boundary * topk, full);
                Reg::Select(value, one, zero, chosen);
                Reg::Add(bin2, bin2, value, full);
            }
            if constexpr (3U < Counters) {
                constexpr uint32_t Boundary = (KnownHalf == 0 ? 5U : 1U) + 3U;
                Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, Boundary * topk, full);
                Reg::Select(value, one, zero, chosen);
                Reg::Add(bin3, bin3, value, full);
            }
            if constexpr (4U < Counters) {
                constexpr uint32_t Boundary = (KnownHalf == 0 ? 5U : 1U) + 4U;
                Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, Boundary * topk, full);
                Reg::Select(value, one, zero, chosen);
                Reg::Add(bin4, bin4, value, full);
            }
            if constexpr (5U < Counters) {
                constexpr uint32_t Boundary = (KnownHalf == 0 ? 5U : 1U) + 5U;
                Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, Boundary * topk, full);
                Reg::Select(value, one, zero, chosen);
                Reg::Add(bin5, bin5, value, full);
            }
            if constexpr (6U < Counters) {
                constexpr uint32_t Boundary = (KnownHalf == 0 ? 5U : 1U) + 6U;
                Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, Boundary * topk, full);
                Reg::Select(value, one, zero, chosen);
                Reg::Add(bin6, bin6, value, full);
            }
            if constexpr (7U < Counters) {
                constexpr uint32_t Boundary = (KnownHalf == 0 ? 5U : 1U) + 7U;
                Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, Boundary * topk, full);
                Reg::Select(value, one, zero, chosen);
                Reg::Add(bin7, bin7, value, full);
            }
            Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(data), uid, payload,
                                             valid);
            Reg::Duplicate(lastIndex, uint32_t{63});
            Reg::Gather<uint32_t, uint32_t>(carry, prefix, lastIndex);
            Reg::Duplicate(lastIndex, count - 1U);
            Reg::Gather<float, uint32_t>(lastScore, score, lastIndex);
            Reg::Gather<int32_t, uint32_t>(lastToken, token, lastIndex);
        }
    } else {
        for (uint32_t base = 0; base < n; base += 64U) {
            uint32_t remaining = n - base;
            valid = Reg::UpdateMask<uint32_t>(remaining);
            const uint32_t count = n - base < 64U ? n - base : 64U;
            Reg::Adds(even, lane, base, full);
            Reg::ShiftLefts(even, even, int16_t{1}, full);
            Reg::Adds(odd, even, uint32_t{1}, full);
            Reg::Duplicate(score, 0.0f);
            Reg::Gather<float, float, uint32_t>(score, reinterpret_cast<__ubuf__ float *>(records), even,
                                                valid);
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
            // Every original query counts, including repeated token positions.
            if constexpr (0U < Counters) {
                constexpr uint32_t Boundary = (KnownHalf == 0 ? 5U : 1U) + 0U;
                Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, Boundary * topk, full);
                Reg::Select(value, one, zero, chosen);
                Reg::Add(bin0, bin0, value, full);
            }
            if constexpr (1U < Counters) {
                constexpr uint32_t Boundary = (KnownHalf == 0 ? 5U : 1U) + 1U;
                Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, Boundary * topk, full);
                Reg::Select(value, one, zero, chosen);
                Reg::Add(bin1, bin1, value, full);
            }
            if constexpr (2U < Counters) {
                constexpr uint32_t Boundary = (KnownHalf == 0 ? 5U : 1U) + 2U;
                Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, Boundary * topk, full);
                Reg::Select(value, one, zero, chosen);
                Reg::Add(bin2, bin2, value, full);
            }
            if constexpr (3U < Counters) {
                constexpr uint32_t Boundary = (KnownHalf == 0 ? 5U : 1U) + 3U;
                Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, Boundary * topk, full);
                Reg::Select(value, one, zero, chosen);
                Reg::Add(bin3, bin3, value, full);
            }
            if constexpr (4U < Counters) {
                constexpr uint32_t Boundary = (KnownHalf == 0 ? 5U : 1U) + 4U;
                Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, Boundary * topk, full);
                Reg::Select(value, one, zero, chosen);
                Reg::Add(bin4, bin4, value, full);
            }
            if constexpr (5U < Counters) {
                constexpr uint32_t Boundary = (KnownHalf == 0 ? 5U : 1U) + 5U;
                Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, Boundary * topk, full);
                Reg::Select(value, one, zero, chosen);
                Reg::Add(bin5, bin5, value, full);
            }
            if constexpr (6U < Counters) {
                constexpr uint32_t Boundary = (KnownHalf == 0 ? 5U : 1U) + 6U;
                Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, Boundary * topk, full);
                Reg::Select(value, one, zero, chosen);
                Reg::Add(bin6, bin6, value, full);
            }
            if constexpr (7U < Counters) {
                constexpr uint32_t Boundary = (KnownHalf == 0 ? 5U : 1U) + 7U;
                Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, Boundary * topk, full);
                Reg::Select(value, one, zero, chosen);
                Reg::Add(bin7, bin7, value, full);
            }
            Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(data), uid, payload,
                                             valid);
            Reg::Duplicate(lastIndex, uint32_t{63});
            Reg::Gather<uint32_t, uint32_t>(carry, prefix, lastIndex);
            Reg::Duplicate(lastIndex, count - 1U);
            Reg::Gather<float, uint32_t>(lastScore, score, lastIndex);
            Reg::Gather<int32_t, uint32_t>(lastToken, token, lastIndex);
        }
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
    Reg::RegTensor<uint32_t> reduced, previousCount, cumulative, delta, oldCursor, endCursor;
    Reg::RegTensor<uint32_t> knownCounts, allDeltas, updated, knownOffset;
    Reg::Duplicate(previousCount, uint32_t{0});
    Reg::Duplicate(even, uint32_t{0});
    Reg::Duplicate(allDeltas, uint32_t{0});
    uint32_t eight = 8U;
    valid = Reg::UpdateMask<uint32_t>(eight);
    if constexpr (KnownHalf >= 0) {
        Reg::Duplicate(knownCounts, uint32_t{0});
        Reg::Adds(index, lane, 112U, full);
        Reg::Select(index, index, lane, valid); // inactive addresses stay within ctx[0..127]
        Reg::Gather<uint32_t, uint32_t, uint32_t>(knownCounts, reinterpret_cast<__ubuf__ uint32_t *>(ctx),
                                                  index, valid);
    }
    Reg::Duplicate(knownOffset, KnownHalf == 0 ? knownTotal : 0U);
    if constexpr (KnownHalf == 2 || KnownHalf == 0) {
        Reg::Duplicate(lastIndex, uint32_t{0});
        Reg::Gather<uint32_t, uint32_t>(delta, knownCounts, lastIndex);
    } else {
        constexpr uint32_t J = KnownHalf < 0 ? 0U : 0U;
        if constexpr (J == CountRows - 1U && PackedCursor) {
            Reg::Duplicate(cumulative, KnownHalf < 0 ? n : unknownTotal);
        } else {
            if constexpr (J == 0U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin0, full);
            if constexpr (J == 1U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin1, full);
            if constexpr (J == 2U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin2, full);
            if constexpr (J == 3U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin3, full);
            if constexpr (J == 4U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin4, full);
            if constexpr (J == 5U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin5, full);
            if constexpr (J == 6U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin6, full);
            if constexpr (J == 7U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin7, full);
            Reg::Gather<uint32_t, uint32_t>(cumulative, reduced, even);
            if constexpr (KnownHalf == 0)
                Reg::Sub(cumulative, cumulative, knownOffset, full);
        }
        Reg::Sub(delta, cumulative, previousCount, full);
        previousCount = cumulative;
    }
    if constexpr (PackedCursor) {
        Reg::Compares<uint32_t, CMPMODE::EQ>(chosen, lane, uint32_t{0}, full);
        Reg::Select(allDeltas, delta, allDeltas, chosen);
    } else {
        Reg::Compares<uint32_t, CMPMODE::EQ>(first, lane, uint32_t{0}, full);
        Reg::Adds(index, lane, 48U, full);
        Reg::Duplicate(oldCursor, uint32_t{0});
        Reg::Gather<uint32_t, uint32_t, uint32_t>(oldCursor, reinterpret_cast<__ubuf__ uint32_t *>(ctx),
                                                  index, first);
        Reg::Add(oldCursor, oldCursor, delta, full);
        Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(ctx), oldCursor, index, first);
    }
    if constexpr (KnownHalf == 2 || KnownHalf == 0) {
        Reg::Duplicate(lastIndex, uint32_t{1});
        Reg::Gather<uint32_t, uint32_t>(delta, knownCounts, lastIndex);
    } else {
        constexpr uint32_t J = KnownHalf < 0 ? 1U : 1U;
        if constexpr (J == CountRows - 1U && PackedCursor) {
            Reg::Duplicate(cumulative, KnownHalf < 0 ? n : unknownTotal);
        } else {
            if constexpr (J == 0U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin0, full);
            if constexpr (J == 1U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin1, full);
            if constexpr (J == 2U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin2, full);
            if constexpr (J == 3U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin3, full);
            if constexpr (J == 4U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin4, full);
            if constexpr (J == 5U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin5, full);
            if constexpr (J == 6U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin6, full);
            if constexpr (J == 7U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin7, full);
            Reg::Gather<uint32_t, uint32_t>(cumulative, reduced, even);
            if constexpr (KnownHalf == 0)
                Reg::Sub(cumulative, cumulative, knownOffset, full);
        }
        Reg::Sub(delta, cumulative, previousCount, full);
        previousCount = cumulative;
    }
    if constexpr (PackedCursor) {
        Reg::Compares<uint32_t, CMPMODE::EQ>(chosen, lane, uint32_t{1}, full);
        Reg::Select(allDeltas, delta, allDeltas, chosen);
    } else {
        Reg::Compares<uint32_t, CMPMODE::EQ>(first, lane, uint32_t{0}, full);
        Reg::Adds(index, lane, 49U, full);
        Reg::Duplicate(oldCursor, uint32_t{0});
        Reg::Gather<uint32_t, uint32_t, uint32_t>(oldCursor, reinterpret_cast<__ubuf__ uint32_t *>(ctx),
                                                  index, first);
        Reg::Add(oldCursor, oldCursor, delta, full);
        Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(ctx), oldCursor, index, first);
    }
    if constexpr (KnownHalf == 2 || KnownHalf == 0) {
        Reg::Duplicate(lastIndex, uint32_t{2});
        Reg::Gather<uint32_t, uint32_t>(delta, knownCounts, lastIndex);
    } else {
        constexpr uint32_t J = KnownHalf < 0 ? 2U : 2U;
        if constexpr (J == CountRows - 1U && PackedCursor) {
            Reg::Duplicate(cumulative, KnownHalf < 0 ? n : unknownTotal);
        } else {
            if constexpr (J == 0U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin0, full);
            if constexpr (J == 1U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin1, full);
            if constexpr (J == 2U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin2, full);
            if constexpr (J == 3U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin3, full);
            if constexpr (J == 4U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin4, full);
            if constexpr (J == 5U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin5, full);
            if constexpr (J == 6U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin6, full);
            if constexpr (J == 7U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin7, full);
            Reg::Gather<uint32_t, uint32_t>(cumulative, reduced, even);
            if constexpr (KnownHalf == 0)
                Reg::Sub(cumulative, cumulative, knownOffset, full);
        }
        Reg::Sub(delta, cumulative, previousCount, full);
        previousCount = cumulative;
    }
    if constexpr (PackedCursor) {
        Reg::Compares<uint32_t, CMPMODE::EQ>(chosen, lane, uint32_t{2}, full);
        Reg::Select(allDeltas, delta, allDeltas, chosen);
    } else {
        Reg::Compares<uint32_t, CMPMODE::EQ>(first, lane, uint32_t{0}, full);
        Reg::Adds(index, lane, 50U, full);
        Reg::Duplicate(oldCursor, uint32_t{0});
        Reg::Gather<uint32_t, uint32_t, uint32_t>(oldCursor, reinterpret_cast<__ubuf__ uint32_t *>(ctx),
                                                  index, first);
        Reg::Add(oldCursor, oldCursor, delta, full);
        Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(ctx), oldCursor, index, first);
    }
    if constexpr (KnownHalf == 2 || KnownHalf == 0) {
        Reg::Duplicate(lastIndex, uint32_t{3});
        Reg::Gather<uint32_t, uint32_t>(delta, knownCounts, lastIndex);
    } else {
        constexpr uint32_t J = KnownHalf < 0 ? 3U : 3U;
        if constexpr (J == CountRows - 1U && PackedCursor) {
            Reg::Duplicate(cumulative, KnownHalf < 0 ? n : unknownTotal);
        } else {
            if constexpr (J == 0U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin0, full);
            if constexpr (J == 1U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin1, full);
            if constexpr (J == 2U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin2, full);
            if constexpr (J == 3U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin3, full);
            if constexpr (J == 4U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin4, full);
            if constexpr (J == 5U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin5, full);
            if constexpr (J == 6U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin6, full);
            if constexpr (J == 7U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin7, full);
            Reg::Gather<uint32_t, uint32_t>(cumulative, reduced, even);
            if constexpr (KnownHalf == 0)
                Reg::Sub(cumulative, cumulative, knownOffset, full);
        }
        Reg::Sub(delta, cumulative, previousCount, full);
        previousCount = cumulative;
    }
    if constexpr (PackedCursor) {
        Reg::Compares<uint32_t, CMPMODE::EQ>(chosen, lane, uint32_t{3}, full);
        Reg::Select(allDeltas, delta, allDeltas, chosen);
    } else {
        Reg::Compares<uint32_t, CMPMODE::EQ>(first, lane, uint32_t{0}, full);
        Reg::Adds(index, lane, 51U, full);
        Reg::Duplicate(oldCursor, uint32_t{0});
        Reg::Gather<uint32_t, uint32_t, uint32_t>(oldCursor, reinterpret_cast<__ubuf__ uint32_t *>(ctx),
                                                  index, first);
        Reg::Add(oldCursor, oldCursor, delta, full);
        Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(ctx), oldCursor, index, first);
    }
    if constexpr (KnownHalf == 2 || KnownHalf == 1) {
        Reg::Duplicate(lastIndex, uint32_t{4});
        Reg::Gather<uint32_t, uint32_t>(delta, knownCounts, lastIndex);
    } else {
        constexpr uint32_t J = KnownHalf < 0 ? 4U : 0U;
        if constexpr (J == CountRows - 1U && PackedCursor) {
            Reg::Duplicate(cumulative, KnownHalf < 0 ? n : unknownTotal);
        } else {
            if constexpr (J == 0U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin0, full);
            if constexpr (J == 1U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin1, full);
            if constexpr (J == 2U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin2, full);
            if constexpr (J == 3U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin3, full);
            if constexpr (J == 4U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin4, full);
            if constexpr (J == 5U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin5, full);
            if constexpr (J == 6U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin6, full);
            if constexpr (J == 7U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin7, full);
            Reg::Gather<uint32_t, uint32_t>(cumulative, reduced, even);
            if constexpr (KnownHalf == 0)
                Reg::Sub(cumulative, cumulative, knownOffset, full);
        }
        Reg::Sub(delta, cumulative, previousCount, full);
        previousCount = cumulative;
    }
    if constexpr (PackedCursor) {
        Reg::Compares<uint32_t, CMPMODE::EQ>(chosen, lane, uint32_t{4}, full);
        Reg::Select(allDeltas, delta, allDeltas, chosen);
    } else {
        Reg::Compares<uint32_t, CMPMODE::EQ>(first, lane, uint32_t{0}, full);
        Reg::Adds(index, lane, 52U, full);
        Reg::Duplicate(oldCursor, uint32_t{0});
        Reg::Gather<uint32_t, uint32_t, uint32_t>(oldCursor, reinterpret_cast<__ubuf__ uint32_t *>(ctx),
                                                  index, first);
        Reg::Add(oldCursor, oldCursor, delta, full);
        Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(ctx), oldCursor, index, first);
    }
    if constexpr (KnownHalf == 2 || KnownHalf == 1) {
        Reg::Duplicate(lastIndex, uint32_t{5});
        Reg::Gather<uint32_t, uint32_t>(delta, knownCounts, lastIndex);
    } else {
        constexpr uint32_t J = KnownHalf < 0 ? 5U : 1U;
        if constexpr (J == CountRows - 1U && PackedCursor) {
            Reg::Duplicate(cumulative, KnownHalf < 0 ? n : unknownTotal);
        } else {
            if constexpr (J == 0U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin0, full);
            if constexpr (J == 1U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin1, full);
            if constexpr (J == 2U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin2, full);
            if constexpr (J == 3U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin3, full);
            if constexpr (J == 4U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin4, full);
            if constexpr (J == 5U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin5, full);
            if constexpr (J == 6U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin6, full);
            if constexpr (J == 7U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin7, full);
            Reg::Gather<uint32_t, uint32_t>(cumulative, reduced, even);
            if constexpr (KnownHalf == 0)
                Reg::Sub(cumulative, cumulative, knownOffset, full);
        }
        Reg::Sub(delta, cumulative, previousCount, full);
        previousCount = cumulative;
    }
    if constexpr (PackedCursor) {
        Reg::Compares<uint32_t, CMPMODE::EQ>(chosen, lane, uint32_t{5}, full);
        Reg::Select(allDeltas, delta, allDeltas, chosen);
    } else {
        Reg::Compares<uint32_t, CMPMODE::EQ>(first, lane, uint32_t{0}, full);
        Reg::Adds(index, lane, 53U, full);
        Reg::Duplicate(oldCursor, uint32_t{0});
        Reg::Gather<uint32_t, uint32_t, uint32_t>(oldCursor, reinterpret_cast<__ubuf__ uint32_t *>(ctx),
                                                  index, first);
        Reg::Add(oldCursor, oldCursor, delta, full);
        Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(ctx), oldCursor, index, first);
    }
    if constexpr (KnownHalf == 2 || KnownHalf == 1) {
        Reg::Duplicate(lastIndex, uint32_t{6});
        Reg::Gather<uint32_t, uint32_t>(delta, knownCounts, lastIndex);
    } else {
        constexpr uint32_t J = KnownHalf < 0 ? 6U : 2U;
        if constexpr (J == CountRows - 1U && PackedCursor) {
            Reg::Duplicate(cumulative, KnownHalf < 0 ? n : unknownTotal);
        } else {
            if constexpr (J == 0U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin0, full);
            if constexpr (J == 1U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin1, full);
            if constexpr (J == 2U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin2, full);
            if constexpr (J == 3U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin3, full);
            if constexpr (J == 4U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin4, full);
            if constexpr (J == 5U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin5, full);
            if constexpr (J == 6U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin6, full);
            if constexpr (J == 7U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin7, full);
            Reg::Gather<uint32_t, uint32_t>(cumulative, reduced, even);
            if constexpr (KnownHalf == 0)
                Reg::Sub(cumulative, cumulative, knownOffset, full);
        }
        Reg::Sub(delta, cumulative, previousCount, full);
        previousCount = cumulative;
    }
    if constexpr (PackedCursor) {
        Reg::Compares<uint32_t, CMPMODE::EQ>(chosen, lane, uint32_t{6}, full);
        Reg::Select(allDeltas, delta, allDeltas, chosen);
    } else {
        Reg::Compares<uint32_t, CMPMODE::EQ>(first, lane, uint32_t{0}, full);
        Reg::Adds(index, lane, 54U, full);
        Reg::Duplicate(oldCursor, uint32_t{0});
        Reg::Gather<uint32_t, uint32_t, uint32_t>(oldCursor, reinterpret_cast<__ubuf__ uint32_t *>(ctx),
                                                  index, first);
        Reg::Add(oldCursor, oldCursor, delta, full);
        Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(ctx), oldCursor, index, first);
    }
    if constexpr (KnownHalf == 2 || KnownHalf == 1) {
        Reg::Duplicate(lastIndex, uint32_t{7});
        Reg::Gather<uint32_t, uint32_t>(delta, knownCounts, lastIndex);
    } else {
        constexpr uint32_t J = KnownHalf < 0 ? 7U : 3U;
        if constexpr (J == CountRows - 1U && PackedCursor) {
            Reg::Duplicate(cumulative, KnownHalf < 0 ? n : unknownTotal);
        } else {
            if constexpr (J == 0U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin0, full);
            if constexpr (J == 1U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin1, full);
            if constexpr (J == 2U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin2, full);
            if constexpr (J == 3U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin3, full);
            if constexpr (J == 4U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin4, full);
            if constexpr (J == 5U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin5, full);
            if constexpr (J == 6U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin6, full);
            if constexpr (J == 7U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin7, full);
            Reg::Gather<uint32_t, uint32_t>(cumulative, reduced, even);
            if constexpr (KnownHalf == 0)
                Reg::Sub(cumulative, cumulative, knownOffset, full);
        }
        Reg::Sub(delta, cumulative, previousCount, full);
        previousCount = cumulative;
    }
    if constexpr (PackedCursor) {
        Reg::Compares<uint32_t, CMPMODE::EQ>(chosen, lane, uint32_t{7}, full);
        Reg::Select(allDeltas, delta, allDeltas, chosen);
    } else {
        Reg::Compares<uint32_t, CMPMODE::EQ>(first, lane, uint32_t{0}, full);
        Reg::Adds(index, lane, 55U, full);
        Reg::Duplicate(oldCursor, uint32_t{0});
        Reg::Gather<uint32_t, uint32_t, uint32_t>(oldCursor, reinterpret_cast<__ubuf__ uint32_t *>(ctx),
                                                  index, first);
        Reg::Add(oldCursor, oldCursor, delta, full);
        Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(ctx), oldCursor, index, first);
    }
    if constexpr (PackedCursor) {
        Reg::Adds(index, lane, 48U, full);
        Reg::Duplicate(oldCursor, uint32_t{0});
        Reg::Gather<uint32_t, uint32_t, uint32_t>(oldCursor, reinterpret_cast<__ubuf__ uint32_t *>(ctx),
                                                  index, valid);
        Reg::Add(oldCursor, oldCursor, allDeltas, full);
        Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(ctx), oldCursor, index, valid);
    } else {
        Reg::LocalMemBar<Reg::MemType::VEC_ALL, Reg::MemType::VEC_ALL>();
        Reg::Adds(index, lane, 48U, full);
        Reg::Duplicate(oldCursor, uint32_t{0});
        Reg::Gather<uint32_t, uint32_t, uint32_t>(oldCursor, reinterpret_cast<__ubuf__ uint32_t *>(ctx),
                                                  index, valid);
    }
    // C consumes its updated register directly; no UB cursor readback required.
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
