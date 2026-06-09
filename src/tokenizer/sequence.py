from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import torch

from src.perception.schema import PerceptionBatch

SLOT_TOKEN_ID = 16500
S_BOS_TOKEN_ID = 16501
S_EOS_TOKEN_ID = 16502
F_BOS_TOKEN_ID = 16503
F_EOS_TOKEN_ID = 16504
ACT_Q_TOKEN_ID = 16505


class TokenType(IntEnum):
    TEXT_T1 = 0
    TEXT_T2 = 1
    VQ = 2
    SLOT = 3
    STATE = 4
    ACTION = 5
    SPECIAL = 6


@dataclass(frozen=True)
class SequenceBatch:
    input_ids: torch.Tensor
    token_type_ids: torch.Tensor
    slot_positions: torch.Tensor
    act_q_position: torch.Tensor
    f_bos_positions: torch.Tensor
    s_bos_positions: torch.Tensor
    special_token_mask: torch.Tensor
    slot_valid_mask: torch.Tensor
    slot_is_object: torch.Tensor
    inputs_embeds: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    attention_mask_4d: torch.Tensor | None = None
    addr_cache: torch.Tensor | None = None
    slot_vector: torch.Tensor | None = None
    slot_embeds: torch.Tensor | None = None

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class SequenceTokenizer:
    """Build the M2a integer sequence layout and position metadata.不做完整的embedding和attention mask"""

    def forward(self, perception_batch: PerceptionBatch | Mapping[str, Any]) -> SequenceBatch:
        batch = (
            perception_batch.as_dict()
            if isinstance(perception_batch, PerceptionBatch)
            else perception_batch
        )
        text_ids = batch["text_t2_bpe_ids"]
        vq_ids = batch["vq_input_ids"]
        state_ids = batch["state_input_ids"]
        action_ids = batch["action_input_ids"]
        slot_valid_mask = batch["slot_valid_mask"]
        slot_is_object = batch["slot_is_object"]

        _validate_inputs(text_ids, vq_ids, state_ids, action_ids, slot_valid_mask, slot_is_object)

        device = text_ids.device
        bsz, text_len = text_ids.shape
        _, frames, vq_tokens_per_frame = vq_ids.shape
        slots = slot_valid_mask.shape[2]
        state_tokens_per_frame = state_ids.shape[2]
        action_tokens_per_transition = action_ids.shape[2] if frames > 1 else 0

        sequence_length = _sequence_length(
            text_len=text_len,
            frames=frames,
            slots=slots,
            vq_tokens_per_frame=vq_tokens_per_frame,
            state_tokens_per_frame=state_tokens_per_frame,
            action_tokens_per_transition=action_tokens_per_transition,
        )

        input_ids = torch.empty((bsz, sequence_length), dtype=torch.long, device=device)
        token_type_ids = torch.empty((bsz, sequence_length), dtype=torch.long, device=device)
        special_token_mask = torch.zeros((bsz, sequence_length), dtype=torch.bool, device=device)
        slot_positions = torch.empty((bsz, frames, slots), dtype=torch.long, device=device)
        f_bos_positions = torch.empty((bsz, frames), dtype=torch.long, device=device)
        s_bos_positions = torch.empty((bsz, frames), dtype=torch.long, device=device)

        input_ids[:, :text_len] = text_ids
        token_type_ids[:, :text_len] = int(TokenType.TEXT_T2)

        position = text_len  #下一个写入L维的位置
        for frame_idx in range(frames):
            f_bos_positions[:, frame_idx] = position
            _write_special(input_ids, token_type_ids, special_token_mask, position, F_BOS_TOKEN_ID)
            position += 1

            if vq_tokens_per_frame > 0:
                next_position = position + vq_tokens_per_frame
                input_ids[:, position:next_position] = vq_ids[:, frame_idx]
                token_type_ids[:, position:next_position] = int(TokenType.VQ)
                position = next_position

            s_bos_positions[:, frame_idx] = position
            _write_special(input_ids, token_type_ids, special_token_mask, position, S_BOS_TOKEN_ID)
            position += 1

            next_position = position + slots
            input_ids[:, position:next_position] = SLOT_TOKEN_ID
            token_type_ids[:, position:next_position] = int(TokenType.SLOT)
            slot_positions[:, frame_idx, :] = torch.arange(
                position, next_position, dtype=torch.long, device=device
            ).view(1, slots).expand(bsz, slots)
            position = next_position

            _write_special(input_ids, token_type_ids, special_token_mask, position, S_EOS_TOKEN_ID)
            position += 1

            next_position = position + state_tokens_per_frame
            input_ids[:, position:next_position] = state_ids[:, frame_idx]
            token_type_ids[:, position:next_position] = int(TokenType.STATE)
            position = next_position

            if frame_idx < frames - 1:
                #如果不是最后一帧，就写action history token
                next_position = position + action_tokens_per_transition
                input_ids[:, position:next_position] = action_ids[:, frame_idx]
                token_type_ids[:, position:next_position] = int(TokenType.ACTION)
                position = next_position

            _write_special(input_ids, token_type_ids, special_token_mask, position, F_EOS_TOKEN_ID)
            position += 1

        act_q_position = torch.full((bsz,), position, dtype=torch.long, device=device)
        _write_special(input_ids, token_type_ids, special_token_mask, position, ACT_Q_TOKEN_ID)
        position += 1

        if position != sequence_length:
            raise RuntimeError("sequence layout length accounting mismatch")

        return SequenceBatch(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            slot_positions=slot_positions,
            act_q_position=act_q_position,
            f_bos_positions=f_bos_positions,
            s_bos_positions=s_bos_positions,
            special_token_mask=special_token_mask,
            slot_valid_mask=slot_valid_mask,
            slot_is_object=slot_is_object,
            addr_cache=batch["addr"].detach(),
        )


def _write_special(
    input_ids: torch.Tensor,
    token_type_ids: torch.Tensor,
    special_token_mask: torch.Tensor,
    position: int,
    token_id: int,
) -> None:
    input_ids[:, position] = token_id
    token_type_ids[:, position] = int(TokenType.SPECIAL)
    special_token_mask[:, position] = True


def _sequence_length(
    *,
    text_len: int,
    frames: int,
    slots: int,
    vq_tokens_per_frame: int,
    state_tokens_per_frame: int,
    action_tokens_per_transition: int,
) -> int:
    per_frame = 4 + vq_tokens_per_frame + slots + state_tokens_per_frame
    action_history = max(frames - 1, 0) * action_tokens_per_transition
    return text_len + frames * per_frame + action_history + 1


def _validate_inputs(
    text_ids: torch.Tensor,
    vq_ids: torch.Tensor,
    state_ids: torch.Tensor,
    action_ids: torch.Tensor,
    slot_valid_mask: torch.Tensor,
    slot_is_object: torch.Tensor,
) -> None:
    if text_ids.ndim != 2:
        raise ValueError("text_t2_bpe_ids must have shape [B,L_t2]")
    if vq_ids.ndim != 3:
        raise ValueError("vq_input_ids must have shape [B,T,L_vq]")
    if state_ids.ndim != 3:
        raise ValueError("state_input_ids must have shape [B,T,7]")
    if action_ids.ndim != 3:
        raise ValueError("action_input_ids must have shape [B,T-1,7]")
    if slot_valid_mask.ndim != 3:
        raise ValueError("slot_valid_mask must have shape [B,T,S]")
    if slot_is_object.shape != slot_valid_mask.shape:
        raise ValueError("slot_is_object must match slot_valid_mask shape")

    bsz, frames = slot_valid_mask.shape[:2]
    if text_ids.shape[0] != bsz:
        raise ValueError("text_t2_bpe_ids batch size must match slot_valid_mask")
    if vq_ids.shape[:2] != (bsz, frames):
        raise ValueError("vq_input_ids must match [B,T] from slot_valid_mask")
    if state_ids.shape[:2] != (bsz, frames):
        raise ValueError("state_input_ids must match [B,T] from slot_valid_mask")
    if action_ids.shape[:2] != (bsz, max(frames - 1, 0)):
        raise ValueError("action_input_ids must match [B,T-1] from slot_valid_mask")

    device = text_ids.device
    for name, tensor in (
        ("vq_input_ids", vq_ids),
        ("state_input_ids", state_ids),
        ("action_input_ids", action_ids),
        ("slot_valid_mask", slot_valid_mask),
        ("slot_is_object", slot_is_object),
    ):
        if tensor.device != device:
            raise ValueError(f"{name} must be on the same device as text_t2_bpe_ids")
