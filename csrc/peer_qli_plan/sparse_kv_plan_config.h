// SPDX-License-Identifier: Apache-2.0
#pragma once

// Resolve the preserved standalone body's quoted include in its own directory,
// before the framework's same-named configuration on the compiler search path.
// kernel.cpp supplies the sparse_kv_plan_config namespace alias for this config.
#include "upstream/sparse_kv_plan_config.h"
