from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from src.perception.schema import PerceptionBatch


def build_slot_vector(perception_batch: PerceptionBatch | Mapping[str, Any]) -> torch.Tensor:
    """Build [addr32, content256, time16, role16] slot vectors."""

    batch = perception_batch.as_dict() if isinstance(perception_batch, PerceptionBatch) else perception_batch
    addr = batch["addr"]
    content = batch["content"]
    slot_time_emb = batch["slot_time_emb"]
    slot_role_emb = batch["slot_role_emb"]
    slot_valid_mask = batch.get("slot_valid_mask") #用()获取表示可选的slot_valid_mask，如果不存在则为None


    #验证组件是否符合预期的形状和类型
    _validate_component("addr", addr, 32)
    _validate_component("content", content, 256)
    _validate_component("slot_time_emb", slot_time_emb, 16)
    _validate_component("slot_role_emb", slot_role_emb, 16)
    _validate_common_shape(addr, content, slot_time_emb, slot_role_emb)

    dtype = content.dtype
    device = content.device
    #对齐几个组件的类型和设备，并将它们连接成一个大的slot_vector
    components = [
        addr.to(device=device, dtype=dtype),
        content,
        slot_time_emb.to(device=device, dtype=dtype),
        slot_role_emb.to(device=device, dtype=dtype),
    ]
    slot_vector = torch.cat(components, dim=-1)

    if slot_valid_mask is not None:
        if slot_valid_mask.shape != slot_vector.shape[:-1]:
            raise ValueError("slot_valid_mask must match slot vector prefix shape")
        slot_vector = slot_vector * slot_valid_mask.to(device=device).unsqueeze(-1).to(dtype)

    return slot_vector


def _validate_component(name: str, tensor: torch.Tensor, width: int) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != 4:
        raise ValueError(f"{name} must have shape [B,T,S,{width}]")
    if tensor.shape[-1] != width:
        raise ValueError(f"{name} last dim must be {width}, got {tensor.shape[-1]}")


def _validate_common_shape(*tensors: torch.Tensor) -> None:
    prefix = tensors[0].shape[:-1]
    device = tensors[0].device
    for tensor in tensors[1:]:
        if tensor.shape[:-1] != prefix:
            raise ValueError("slot vector components must share [B,T,S] shape")
        if tensor.device != device:
            raise ValueError("slot vector components must be on the same device")

