// SPDX-License-Identifier: Apache-2.0
#pragma once
#ifndef JOINT_REFINE_BITS
#error "Compile through a generated refine wrapper"
#endif
static_assert(JOINT_REFINE_BITS>=0 && JOINT_REFINE_BITS<16,"refine bits A/B/C/D");
constexpr bool REFINE_A=(JOINT_REFINE_BITS&1U)!=0; // singleton direct UID
constexpr bool REFINE_B=(JOINT_REFINE_BITS&2U)!=0; // boolean ballot Scan
constexpr bool REFINE_C=(JOINT_REFINE_BITS&4U)!=0; // warm resident staging
constexpr bool REFINE_D=(JOINT_REFINE_BITS&8U)!=0; // complete SIMD Phase3, UB only
