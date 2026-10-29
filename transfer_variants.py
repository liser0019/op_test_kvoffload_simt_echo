"""Transfer因子目录与可审计的布局模型；不包含设备计算替代实现。"""
import json
from pathlib import Path

CATALOG = json.loads((Path(__file__).resolve().parent / "transfer_variants.json").read_text())
VARIANTS = {entry["name"]: entry for entry in CATALOG["variants"]}


def layout(name, bytes_k, bytes_v, block_size):
    entry = VARIANTS[name]
    mask = entry["mask"]
    batch = 8 if mask >= 0 and mask & 2 else 1
    pitch_k, pitch_v = (bytes_k + 31) & ~31, (bytes_v + 31) & ~31
    required = batch * (pitch_k + pitch_v if mask >= 0 and mask & 1 else max(pitch_k, pitch_v))
    fallback = mask >= 0 and required > CATALOG["buffer_bytes"]
    production = mask >= 0 and bool(mask & 4) and (bytes_k, bytes_v, block_size) == (1024, 128, 64)
    return dict(entry) | {
        "requested_features": "".join(letter for bit, letter in enumerate("ABCD") if mask >= 0 and mask & (1 << bit)),
        "requested_batch": batch, "effective_batch": None if fallback or mask < 0 else batch,
        "bytes_per_queue_record_limit": required, "ub_bytes": CATALOG["ub_bytes"],
        "fallback": fallback, "production_specialization": production and not fallback,
        "effective_path": ("baseline" if mask < 0 else "original_fallback" if fallback else
                           "factorial_production" if production else "factorial_generic"),
    }
