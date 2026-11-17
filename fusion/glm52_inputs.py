"""Synthetic GLM-5.2 C8 inputs for operator-only profiling.

The large public shape mirrors one decode step: 32 requests, four MTP rows,
256K visible tokens, 32 indexer heads, and 128 values per head.  The values are
synthetic deterministic tensors, not model activation dumps. Key pages are
written in bounded chunks so constructing the fixture does not need a second
full-size floating-point copy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Final

import torch

BLOCK_SIZE: Final = 128
HEAD_DIM: Final = 128
INDEX_HEADS: Final = 32
TOPK: Final = 2048
SEQ_LEN: Final = 256 * 1024
MTP_WIDTH: Final = 4
DEFAULT_BATCH: Final = 32
FP8_MAX: Final = 448.0

Quantizer = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


@dataclass(frozen=True)
class Shape:
    """Public benchmark dimensions; smaller values are useful for CPU checks."""

    batch: int = DEFAULT_BATCH
    width: int = MTP_WIDTH
    seq_len: int = SEQ_LEN
    heads: int = INDEX_HEADS
    head_dim: int = HEAD_DIM
    topk: int = TOPK
    capacity: int = 4096

    @property
    def rows(self) -> int:
        return self.batch * self.width

    @property
    def pages(self) -> int:
        return math.ceil(self.seq_len / BLOCK_SIZE)

    @property
    def logical_pages(self) -> int:
        return self.batch * self.pages

    @property
    def physical_blocks(self) -> int:
        return self.logical_pages

    @property
    def page_bytes(self) -> int:
        return self.head_dim * BLOCK_SIZE + 4 * BLOCK_SIZE


@dataclass(frozen=True)
class Inputs:
    """All tensors needed by either QLI variant and by the fused plan state."""

    shape: Shape
    qli: dict[str, torch.Tensor | int | str]
    plan_metadata: dict[str, torch.Tensor]
    previous_query: torch.Tensor
    previous_query_scale: torch.Tensor
    host_num_blocks: int


def _cpu_quantize(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference per-row symmetric FP8 quantization for CPU fixture tests."""
    flat = values.float().reshape(-1, values.shape[-1])
    scale = flat.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / FP8_MAX
    quantized = (flat / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return quantized.reshape(values.shape), scale.reshape(values.shape[:-1]).float()


def _quantize(values: torch.Tensor, quantizer: Quantizer | None) -> tuple[torch.Tensor, torch.Tensor]:
    if quantizer is None:
        return _cpu_quantize(values)
    quantized, scale = quantizer(values)
    return quantized, scale.reshape(values.shape[:-1]).to(torch.float32)


def _page_tables(shape: Shape, seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    physical = torch.randperm(shape.physical_blocks, generator=generator)
    qli_table = physical.reshape(shape.batch, shape.pages).to(device=device, dtype=torch.int32)
    host_table = torch.arange(shape.logical_pages, dtype=torch.int32, device=device).reshape(
        shape.batch, shape.pages
    )
    return qli_table, host_table


def _allocate_packed_key(shape: Shape, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw = torch.zeros((shape.physical_blocks, shape.page_bytes), dtype=torch.uint8, device=device)
    key = raw.view(torch.float8_e4m3fn).as_strided(
        (shape.physical_blocks, BLOCK_SIZE, 1, shape.head_dim),
        (shape.page_bytes, shape.head_dim, shape.head_dim, 1),
    )
    scales = raw.view(torch.float32).as_strided(
        (shape.physical_blocks, BLOCK_SIZE, 1),
        (shape.page_bytes // 4, 1, 1),
        storage_offset=(BLOCK_SIZE * shape.head_dim) // 4,
    )
    return raw, key, scales


def _fill_key_pages(shape: Shape, key: torch.Tensor, scales: torch.Tensor, seed: int,
                    quantizer: Quantizer | None) -> None:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    chunk_pages = min(256, shape.physical_blocks)
    for start in range(0, shape.physical_blocks, chunk_pages):
        count = min(chunk_pages, shape.physical_blocks - start)
        source = torch.randn((count, BLOCK_SIZE, 1, shape.head_dim), generator=generator,
                             dtype=torch.float32).to(device=key.device, dtype=torch.bfloat16)
        quantized, scale = _quantize(source, quantizer)
        key[start : start + count].copy_(quantized)
        scales[start : start + count].copy_(scale)


def _queries(shape: Shape, seed: int, device: torch.device,
             quantizer: Quantizer | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    base = torch.randn((shape.batch, 1, shape.heads, shape.head_dim), generator=generator)
    lane = torch.randn((shape.batch, shape.width, shape.heads, shape.head_dim), generator=generator)
    previous = (base + 0.05 * lane).to(device=device, dtype=torch.bfloat16).reshape(shape.rows, shape.heads, shape.head_dim)
    delta = torch.randn((shape.batch, 1, shape.heads, shape.head_dim), generator=generator)
    current = (base + 0.05 * lane + 0.15 * delta).to(device=device, dtype=torch.bfloat16).reshape(shape.rows, shape.heads, shape.head_dim)
    query, qscale = _quantize(current, quantizer)
    old_query, old_scale = _quantize(previous, quantizer)
    return query, qscale, old_query, old_scale


def make_inputs(shape: Shape = Shape(), *, device: str = "cpu", seed: int = 17,
                quantizer: Quantizer | None = None) -> Inputs:
    """Build deterministic, full-length C8 inputs without importing vLLM."""
    if shape.heads != INDEX_HEADS or shape.head_dim != HEAD_DIM:
        raise ValueError("GLM-5.2 C8 indexer requires heads=32 and head_dim=128")
    if shape.width not in (1, 2, 4, 8) or shape.topk > shape.capacity:
        raise ValueError("width must be one of 1, 2, 4, 8 and topk must fit capacity")
    target = torch.device(device)
    qli_table, host_table = _page_tables(shape, seed, target)
    _, key, key_scale = _allocate_packed_key(shape, target)
    _fill_key_pages(shape, key, key_scale, seed + 1, quantizer)
    query, qscale, previous, previous_scale = _queries(shape, seed + 2, target, quantizer)
    q_ends = torch.arange(1, shape.batch + 1, device=target, dtype=torch.int32) * shape.width
    key_lengths = torch.full((shape.batch,), shape.seq_len, dtype=torch.int32, device=target)
    visible = torch.arange(1, shape.width + 1, dtype=torch.int32, device=target).repeat(shape.batch) + shape.seq_len - shape.width
    weight_generator = torch.Generator(device="cpu").manual_seed(seed + 3)
    weight_base = torch.randn((shape.batch, 1, shape.heads), generator=weight_generator)
    weight_lane = torch.randn((shape.batch, shape.width, shape.heads), generator=weight_generator)
    weights = (weight_base + 0.05 * weight_lane).reshape(shape.rows, shape.heads)
    weights = (weights / math.sqrt(shape.heads * shape.head_dim)).to(target, torch.bfloat16)
    metadata = {
        "req_ids": torch.arange(shape.batch, dtype=torch.int64, device=target),
        "stable_prefix_lens": torch.full((shape.rows,), shape.seq_len - shape.width, dtype=torch.int32, device=target),
        "visible_seq_lens": visible,
        "token_to_req": torch.arange(shape.batch, dtype=torch.int32, device=target).repeat_interleave(shape.width),
        "host_block_table": host_table,
    }
    qli: dict[str, torch.Tensor | int | str] = {
        "query": query, "key": key, "weights": weights,
        "query_dequant_scale": qscale, "key_dequant_scale": key_scale,
        "actual_seq_lengths_query": q_ends, "actual_seq_lengths_key": key_lengths,
        "block_table": qli_table, "query_quant_mode": 0, "key_quant_mode": 0,
        "layout_query": "TND", "layout_key": "PA_BSND", "sparse_count": shape.topk,
        "sparse_mode": 3, "pre_tokens": 2**63 - 1, "next_tokens": 2**63 - 1,
    }
    return Inputs(shape, qli, metadata, previous, previous_scale, shape.logical_pages)
