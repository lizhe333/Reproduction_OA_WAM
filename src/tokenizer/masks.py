from __future__ import annotations

from dataclasses import replace

import torch

from src.tokenizer.sequence import SequenceBatch


def build_sequence_attention_mask(sequence_batch: SequenceBatch) -> SequenceBatch:
    """Build M2c boolean 4D attention mask from M2a layout metadata.

    The returned `attention_mask_4d` must use shape [B,1,L,L], where
    mask[b,0,query_pos,key_pos] == True means query may read key/value.
    """

    B, L = sequence_batch.input_ids.shape
    device=sequence_batch.input_ids.device
    T, _ = sequence_batch.slot_positions.shape[1:]

    positions = torch.arange(L, device=device)
    query_pos = positions.view(L, 1)
    key_pos = positions.view(1, L)
    mask_2d = key_pos <= query_pos
    attention_mask_4d = mask_2d.view(1, 1, L, L).expand(B, 1, L, L).clone()

    for b in range(B):
        for t in range(T):
            padding_positions = sequence_batch.slot_positions[b, t][
                ~sequence_batch.slot_valid_mask[b, t]
            ]

            valid_positions = sequence_batch.slot_positions[b, t][
                sequence_batch.slot_valid_mask[b, t]
            ]
            for query_pos in valid_positions:
                for key_pos in valid_positions:
                    attention_mask_4d[b, 0, query_pos, key_pos] = True
            for pad_pos in padding_positions:
                attention_mask_4d[b, 0, :, pad_pos] = False

    return replace(sequence_batch, attention_mask_4d=attention_mask_4d)
