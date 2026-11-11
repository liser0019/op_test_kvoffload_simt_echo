"""同事独立Plan接入规格，固定来源见csrc/peer_qli_plan/upstream/source.json。"""
NAME = "peer_qli_plan"
KERNEL = "sparse_kv_plan_peer_qli_kernel"

def layout(topk, capacity, rows, max_group_rows):
    if max_group_rows not in (1, 2, 4, 8):
        raise ValueError("peer max_group_rows must be1/2/4/8")
    h = max(32, 1 << (2 * topk - 1).bit_length())
    j = max(32, 1 << (2 * capacity - 1).bit_length())
    row = 3 * h + capacity + topk
    scratch = max(row, 64 + ((max_group_rows * topk + capacity + 7) & ~7)
                  + max(3 * capacity, 64, 16 * max_group_rows))
    aligned = (4 * scratch + 31) & ~31
    ub = aligned <= 112 * 1024
    chunk = min(768, max(64, 3 * capacity) // 16) // 4 * 4
    stage = (64 + max(16 * chunk, 16 * max_group_rows)) * 4
    return {"hash_capacity": h, "joint_hash_capacity": j, "scratch_elements": scratch,
            "workspace_elements": 1 if ub else rows * scratch,
            "dynamic_ub_bytes": 288 + (aligned if ub else stage),
            "scratch_path": "UB" if ub else "GM", "max_group_rows": max_group_rows}

def description():
    return {"name": NAME, "origin": "vllmascend_sep_fif@382819673",
            "snapshot": "ops-transformer@9ec2110969a9e8502bf6d579c027ac4ea9572016",
            "scope": "standalone Plan; GM ordered TopK; excludes QLI and fused producer election",
            "threads": 512, "fallback_threads": 1024,
            "scratch_threshold_bytes": 112 * 1024}
