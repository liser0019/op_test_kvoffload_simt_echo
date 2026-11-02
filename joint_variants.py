"""Joint A/B/C/D 编译目录；与原 ABCDF 的字母命名完全独立。"""
OPTIONS = {
    "a": "non-joint VF compile-time isolation (protocol retained)",
    "b": "reuse per-query union hash bucket",
    "c": "write resident only on reset/invalidation",
    "d": "one union overflow reduction per row",
}
MASKS = {"".join(letter for bit, letter in enumerate(OPTIONS) if mask & (1 << bit)): mask
         for mask in range(1, 16)}
MASKS["bitmap"] = 16
MASKS["turbo"] = 17
MASKS.update(turbo_a=18, turbo_b=19, turbo_ab=20)
MASKS["merge"] = 21
MERGE_OPTIONS = {"a": "resident bucket forwarding in partition high bits",
                 "b": "tiled merge-path pairwise linear merge",
                 "c": "single currentSlots store per query",
                 "d": "stage-end tile barriers with ping-pong Scan"}
MERGE_VARIANTS = {"merge_" + "".join(c for i, c in enumerate(MERGE_OPTIONS) if bits & (1 << i)): 21 + bits
                  for bits in range(1, 16)}
MASKS.update(MERGE_VARIANTS)
MERGE_NAMES = ("joint_merge", *("joint_" + n for n in MERGE_VARIANTS))
TURBO_MASKS = tuple(range(17, 37))
VARIANTS = {"private_a": 1, "private_turbo": 17} | {"joint_" + letters: mask for letters, mask in MASKS.items()}
DEFAULT_IMPLEMENTATIONS = ("abcdf", "joint_abc", "joint_bitmap", "joint_turbo")


def is_joint(name):
    return name == "joint" or name.startswith("joint_")


def description(name):
    mask = VARIANTS.get(name, 0)
    if mask >= 22:
        bits = mask - 21
        return {"mask": mask, "merge_bits": bits, "structure": "Merge ABCD factorial",
                "optimizations": [v for i, v in enumerate(MERGE_OPTIONS.values()) if bits & (1 << i)],
                "baseline": "abcdf_pages", "merge_reference": "joint_merge"}
    if mask == 21:
        return {"mask": 21, "structure": "resident-first + sorted miss rank-scatter merge",
                "optimizations": ["sorted TopK contract", "resident LRU-last owner", "miss-only union",
                                  "dynamic 32-bit token/query packing", "capacity check before commit"],
                "baseline": "abcdf_pages"}
    structure = "bitmap+unique+direct_map" if mask == 16 else ("hash+compact_unique+fast_source+finish_tile" if mask in TURBO_MASKS else "hash")
    opts = ([
        "single-VF launch structure + compile-time isolated private VF",
        "fast group discovery with register-cached tokenToReq and early-break bound",
        "FastGroupSource single unsigned token bound + bit-shift resident page division",
        "dedicated UB queryBuckets array + active-bucket tracking in partition high-16-bits",
        "cold/mode-reset zero-scan Joint fast path",
        "single-barrier ping-pong TurboExclusiveScan + private UB token forwarding",
    ] if mask in TURBO_MASKS else [value for bit, value in enumerate(OPTIONS.values()) if mask & (1 << bit)])
    if mask in (18, 20):
        opts.append("A: frozen-table lookup phase, then CAS-only insertion of absent queries")
    if mask in (19, 20):
        opts.append("B: one union compaction Scan per query row, with scratch new-entry flags")
    return {"mask": mask, "structure": structure, "optimizations": opts,
            "baseline": "joint" if is_joint(name) else "private"}
