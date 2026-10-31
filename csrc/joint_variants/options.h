// SPDX-License-Identifier: Apache-2.0
#pragma once
// A=1: 私有 VF 编译期隔离；B=2: 复用 query 桶号；
// C=4: 按需 resident 写回；D=8: 每行一次 union 计数。
#ifndef JOINT_OPT_MASK
#error "Compile through generated variant wrapper (mask 1..16)"
#endif
static_assert(JOINT_OPT_MASK > 0 && JOINT_OPT_MASK <= 16, "invalid joint experiment mask");
