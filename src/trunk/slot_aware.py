from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from src.tokenizer import SequenceBatch


@dataclass(frozen=True)
class TrunkOutput:
    last_hidden_state: torch.Tensor
    slot_hidden: torch.Tensor
    slot_hidden_before_reset: torch.Tensor
    act_q_hidden: torch.Tensor
    optional_attentions: Any | None = None


class SlotAwareTrunk(nn.Module):
    """Thin M3 wrapper around a frozen backbone plus slot/action gathers."""

    def __init__(
        self,
        backbone: nn.Module,
        mask_adapter: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.mask_adapter = mask_adapter or _identity_mask_adapter

    def forward(self, sequence_batch: SequenceBatch) -> TrunkOutput:
        #从SequenceBatch中取出input_embeds和mask，传给backbone
        inputs_embeds = sequence_batch.inputs_embeds
        attention_mask_4d = sequence_batch.attention_mask_4d  #将attention mask放在slotaware中转换可以适配更多backbone，更加灵活
        if inputs_embeds is None:
            raise ValueError("SlotAwareTrunk requires sequence_batch.inputs_embeds")
        if attention_mask_4d is None:
            raise ValueError("SlotAwareTrunk requires sequence_batch.attention_mask_4d")

        _validate_sequence_inputs(sequence_batch, inputs_embeds, attention_mask_4d)
        attention_mask = self.mask_adapter(attention_mask_4d)

        outputs = self.backbone(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        last_hidden_state = _extract_output(outputs, "last_hidden_state")
        _validate_last_hidden_state(last_hidden_state, inputs_embeds)

        slot_hidden = _gather_slot_hidden(
            last_hidden_state=last_hidden_state,
            slot_positions=sequence_batch.slot_positions,
        )
        act_q_hidden = _gather_act_q_hidden(
            last_hidden_state=last_hidden_state,
            act_q_position=sequence_batch.act_q_position,
        )

        return TrunkOutput(
            last_hidden_state=last_hidden_state,
            slot_hidden=slot_hidden,
            slot_hidden_before_reset=slot_hidden,
            act_q_hidden=act_q_hidden,
            optional_attentions=_extract_output(outputs, "attentions", required=False),
        )


def _identity_mask_adapter(attention_mask_4d: torch.Tensor) -> torch.Tensor:
    return attention_mask_4d


def chameleon_attention_mask_adapter(
    attention_mask_4d: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert project bool mask to Chameleon additive attention mask.

    Project convention: True means query may read key.
    Chameleon eager attention adds the mask to attention logits, so allowed
    entries must be 0 and blocked entries must be a large negative value.
    """
    if attention_mask_4d.dtype != torch.bool:
        raise ValueError("chameleon_attention_mask_adapter expects a bool mask")

    additive_mask = torch.zeros(
        attention_mask_4d.shape,
        dtype=dtype,
        device=attention_mask_4d.device,
    )
    return additive_mask.masked_fill(~attention_mask_4d, torch.finfo(dtype).min)


def _gather_slot_hidden(
    *,
    last_hidden_state: torch.Tensor,
    slot_positions: torch.Tensor,
) -> torch.Tensor:
    #将经过backbone处理的输出序列中，提出出来slot对应的embeds
    batch_size = last_hidden_state.shape[0]
    batch_index = torch.arange(batch_size, device=last_hidden_state.device).view(
        batch_size,
        1,
        1,
    )
    return last_hidden_state[batch_index, slot_positions]


def _gather_act_q_hidden(
    *,
    last_hidden_state: torch.Tensor,
    act_q_position: torch.Tensor,
) -> torch.Tensor:
    batch_size = last_hidden_state.shape[0]
    batch_index = torch.arange(batch_size, device=last_hidden_state.device)
    return last_hidden_state[batch_index, act_q_position]


def _validate_sequence_inputs(
    sequence_batch: SequenceBatch,
    inputs_embeds: torch.Tensor,
    attention_mask_4d: torch.Tensor,
) -> None:
    if inputs_embeds.ndim != 3:
        raise ValueError("inputs_embeds must have shape [B,L,D]")
    batch_size, sequence_length, _ = inputs_embeds.shape
    if sequence_batch.input_ids.shape != (batch_size, sequence_length):
        raise ValueError("input_ids must match inputs_embeds [B,L] prefix")
    if sequence_batch.slot_positions.ndim != 3:
        raise ValueError("slot_positions must have shape [B,T,S]")
    if sequence_batch.slot_positions.shape[0] != batch_size:
        raise ValueError("slot_positions batch size must match inputs_embeds")
    if sequence_batch.act_q_position.shape != (batch_size,):
        raise ValueError("act_q_position must have shape [B]")
    if attention_mask_4d.shape != (batch_size, 1, sequence_length, sequence_length):
        raise ValueError("attention_mask_4d must have shape [B,1,L,L]")
    if attention_mask_4d.dtype != torch.bool:
        raise ValueError("attention_mask_4d must be bool")
    _validate_positions_in_range(
        sequence_batch.slot_positions,
        sequence_length,
        name="slot_positions",
    )
    _validate_positions_in_range(
        sequence_batch.act_q_position,
        sequence_length,
        name="act_q_position",
    )


def _validate_positions_in_range(
    positions: torch.Tensor,
    sequence_length: int,
    *,
    name: str,
) -> None:
    if positions.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if positions.min().item() < 0 or positions.max().item() >= sequence_length:
        raise ValueError(f"{name} entries must be within [0, L)")


def _validate_last_hidden_state(
    last_hidden_state: torch.Tensor,
    inputs_embeds: torch.Tensor,
) -> None:
    if not isinstance(last_hidden_state, torch.Tensor):
        raise TypeError("backbone output last_hidden_state must be a torch.Tensor")
    if last_hidden_state.shape != inputs_embeds.shape:
        raise ValueError("last_hidden_state must have shape [B,L,D] matching inputs_embeds")


def _extract_output(outputs: Any, name: str, *, required: bool = True) -> Any:
    if isinstance(outputs, dict):
        value = outputs.get(name)
    else:
        value = getattr(outputs, name, None)
    if required and value is None:
        raise ValueError(f"backbone output must include {name}")
    return value
