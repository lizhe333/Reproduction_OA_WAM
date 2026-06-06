from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class PerceptionBatch:
    """Batch emitted by mock or cached frozen perception."""

    text_t1_noun_ids: torch.Tensor #t1时刻的自然语言指令中提取的名词
    text_t2_bpe_ids: torch.Tensor
    vq_input_ids: torch.Tensor #经过VQ-GAN编码后的视觉token
    state_input_ids: torch.Tensor  #机器人状态token
    action_input_ids: torch.Tensor
    addr: torch.Tensor #后面的是OA-WAM的核心slot内容
    content: torch.Tensor
    pose: torch.Tensor
    slot_valid_mask: torch.Tensor
    slot_is_object: torch.Tensor
    slot_role_ids: torch.Tensor
    slot_time_emb: torch.Tensor
    slot_role_emb: torch.Tensor
    dino_features: torch.Tensor  #DINOv3提取的视觉特征
    sam_masks: torch.Tensor
    object_track_ids: torch.Tensor
    target_slot_ids: torch.Tensor #监督 哪些slot是当前动作的目标物体

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

