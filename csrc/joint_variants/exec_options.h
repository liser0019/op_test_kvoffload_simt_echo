// SPDX-License-Identifier: Apache-2.0
#pragma once
#ifndef JOINT_EXEC_BITS
#error "Use generated exec wrapper"
#endif
static_assert(JOINT_EXEC_BITS>=0 && JOINT_EXEC_BITS<16,"execution-factor bits");
constexpr bool EXEC_LOAD=(JOINT_EXEC_BITS&1U)!=0;   // A: interleaved UB load
constexpr bool EXEC_PACKED=(JOINT_EXEC_BITS&2U)!=0; // B: hit/evict ballot prefix
constexpr bool EXEC_FUSE=(JOINT_EXEC_BITS&4U)!=0;   // C: singleton Phase0+5 in one VF
constexpr bool EXEC_CURSOR=(JOINT_EXEC_BITS&8U)!=0; // D: division-free packet cursor
static_assert(REFINE_A && REFINE_B && REFINE_D && !REFINE_C,"parent must be ABD512");
