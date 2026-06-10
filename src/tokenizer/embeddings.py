from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

import torch

from src.perception.schema import PerceptionBatch

from .sequence import SequenceBatch
from .slot_vector import build_slot_vector


def build_sequence_embeddings(
    sequence_batch: SequenceBatch,
    perception_batch: PerceptionBatch | Mapping[str, Any],
    embed_tokens: Callable[[torch.Tensor], torch.Tensor],
    slot_adapter: Callable[[torch.Tensor], torch.Tensor],
) -> SequenceBatch:
    """Build M2b embeddings from M2a layout metadata.

    Attention mask construction is M2c, not part of M2b.
    """

    base_embeds = embed_tokens(sequence_batch.input_ids)
    _validate_base_embeds(base_embeds, sequence_batch.input_ids)

    slot_vector = build_slot_vector(perception_batch)
    _validate_slot_vector(slot_vector, sequence_batch.slot_positions)

    slot_embeds = slot_adapter(slot_vector)
    _validate_slot_embeds(slot_embeds, sequence_batch.slot_positions, base_embeds)

    inputs_embeds = _scatter_slot_embeds(
        base_embeds=base_embeds,
        slot_embeds=slot_embeds,
        slot_positions=sequence_batch.slot_positions,
    )

    # SequenceBatch is frozen; return a copy with M2b fields filled.
    return replace(
        sequence_batch,
        inputs_embeds=inputs_embeds,
        slot_vector=slot_vector,
        slot_embeds=slot_embeds,
    )


def _scatter_slot_embeds(
    *,
    base_embeds: torch.Tensor,
    slot_embeds: torch.Tensor,
    slot_positions: torch.Tensor,
) -> torch.Tensor:
    """Replace slot placeholder embeddings at their flattened sequence positions."""
    B, T, S, _ = slot_embeds.shape
    input_embeds = base_embeds.clone()  # [B,L,D]
    for b in range(B):
        for t in range(T):
            for s in range(S):
                position = slot_positions[b, t, s]
                input_embeds[b, position, :] = slot_embeds[b, t, s, :]

    return input_embeds


def _validate_base_embeds(base_embeds: torch.Tensor, input_ids: torch.Tensor) -> None:
    if not isinstance(base_embeds, torch.Tensor):
        raise TypeError("embed_tokens(input_ids) must return a torch.Tensor")
    if base_embeds.ndim != 3:
        raise ValueError("embed_tokens(input_ids) must return shape [B,L,D]")
    if base_embeds.shape[:2] != input_ids.shape:
        raise ValueError("base embeddings must match input_ids [B,L] prefix")


def _validate_slot_vector(slot_vector: torch.Tensor, slot_positions: torch.Tensor) -> None:
    if slot_vector.ndim != 4:
        raise ValueError("slot_vector must have shape [B,T,S,320]")
    if slot_vector.shape[:-1] != slot_positions.shape:
        raise ValueError("slot_vector [B,T,S] prefix must match slot_positions")
    if slot_vector.shape[-1] != 320:
        raise ValueError(f"slot_vector last dim must be 320, got {slot_vector.shape[-1]}")


def _validate_slot_embeds(
    slot_embeds: torch.Tensor,
    slot_positions: torch.Tensor,
    base_embeds: torch.Tensor,
) -> None:
    if not isinstance(slot_embeds, torch.Tensor):
        raise TypeError("slot_adapter(slot_vector) must return a torch.Tensor")
    if slot_embeds.ndim != 4:
        raise ValueError("slot_adapter(slot_vector) must return shape [B,T,S,D]")
    if slot_embeds.shape[:-1] != slot_positions.shape:
        raise ValueError("slot_embeds [B,T,S] prefix must match slot_positions")
    if slot_embeds.shape[-1] != base_embeds.shape[-1]:
        raise ValueError("slot_embeds hidden dim must match base embeddings hidden dim D")
    if slot_embeds.device != base_embeds.device:
        raise ValueError("slot_embeds must be on the same device as base embeddings")
    if slot_embeds.dtype != base_embeds.dtype:
        raise ValueError("slot_embeds dtype must match base embeddings dtype")
