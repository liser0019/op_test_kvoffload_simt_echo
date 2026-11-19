// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

namespace mtp3_lookup {
constexpr unsigned kRowsPerRequest = 4;
constexpr unsigned kCapacity = 8192;
constexpr unsigned kMapWords = 4096;
constexpr unsigned kHeaderWords = 8;
constexpr unsigned kDirectoryBins = 1024;
constexpr unsigned kDirectoryWords = 1032; // Includes the terminal offset and 32B alignment.
constexpr unsigned kRecordsOffset = kHeaderWords;
constexpr unsigned kDirectoryOffset = kRecordsOffset + kCapacity;
constexpr unsigned kSlotsOffset = kDirectoryOffset + kDirectoryWords;
constexpr unsigned kRequestWords = kSlotsOffset + kCapacity / 2;
constexpr unsigned kLookupUbBytes = (kCapacity + kDirectoryWords) * sizeof(int32_t);
constexpr int kPrepare = 1;
constexpr int kCommit = 2;
}
