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
VARIANTS = {"private_a": 1} | {"joint_" + letters: mask for letters, mask in MASKS.items()}
DEFAULT_IMPLEMENTATIONS = ("abcdf", "joint_abc", "joint_bitmap")


def is_joint(name):
    return name == "joint" or name.startswith("joint_")


def description(name):
    mask = VARIANTS.get(name, 0)
    return {"mask": mask, "structure": "bitmap+unique+direct_map" if mask == 16 else "hash", "optimizations": [value for bit, value in enumerate(OPTIONS.values())
                                             if mask & (1 << bit)],
            "baseline": "joint" if is_joint(name) else "private"}
