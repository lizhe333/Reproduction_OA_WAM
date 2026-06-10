import pytest
import torch

from src.perception import MockPerceptionCache, MockPerceptionConfig

try:
    from src.tokenizer import SequenceTokenizer, TokenType, build_sequence_attention_mask
except ImportError:
    SequenceTokenizer = None
    TokenType = None
    build_sequence_attention_mask = None


pytestmark = pytest.mark.xfail(
    SequenceTokenizer is None or build_sequence_attention_mask is None,
    reason="M2c attention mask scaffold is not available yet",
    strict=False,
)


def test_attention_mask_ordinary_causal_rule():
    sequence_batch = _build_toy_sequence_batch()

    result = build_sequence_attention_mask(sequence_batch)
    mask = result.attention_mask_4d

    assert mask is not None
    assert mask.dtype == torch.bool
    assert mask.shape == (
        sequence_batch.input_ids.shape[0],
        1,
        sequence_batch.input_ids.shape[1],
        sequence_batch.input_ids.shape[1],
    )
    assert mask[0, 0, 5, 3].item() is True
    assert mask[0, 0, 5, 7].item() is False


def test_attention_mask_same_frame_slots_bidirectional():
    sequence_batch = _build_toy_sequence_batch()
    first_slot = sequence_batch.slot_positions[0, 0, 0].item()
    later_slot = sequence_batch.slot_positions[0, 0, 1].item()

    result = build_sequence_attention_mask(sequence_batch)
    mask = result.attention_mask_4d

    assert sequence_batch.slot_valid_mask[0, 0, 0].item() is True
    assert sequence_batch.slot_valid_mask[0, 0, 1].item() is True
    assert mask[0, 0, first_slot, later_slot].item() is True
    assert mask[0, 0, later_slot, first_slot].item() is True


def test_attention_mask_padding_slots_are_not_readable():
    sequence_batch = _build_toy_sequence_batch()
    padding_slot = sequence_batch.slot_positions[0, 0, 2].item()

    result = build_sequence_attention_mask(sequence_batch)
    mask = result.attention_mask_4d

    assert sequence_batch.slot_valid_mask[0, 0, 2].item() is False
    assert mask[0, 0, :, padding_slot].any().item() is False


def test_attention_mask_action_reads_slot_but_slot_does_not_read_action():
    sequence_batch = _build_toy_sequence_batch()
    slot_pos = sequence_batch.slot_positions[0, 0, 0].item()
    action_pos = _first_position_of_type(sequence_batch.token_type_ids, TokenType.ACTION)

    result = build_sequence_attention_mask(sequence_batch)
    mask = result.attention_mask_4d

    assert mask[0, 0, action_pos, slot_pos].item() is True
    assert mask[0, 0, slot_pos, action_pos].item() is False


def test_attention_mask_padding_columns_are_batch_specific():
    sequence_batch = _build_toy_sequence_batch(batch_size=2, force_last_slot_padding=False)
    slot_valid_mask = sequence_batch.slot_valid_mask.clone()
    slot_valid_mask[0, :, -1] = False
    slot_valid_mask[1, :, -1] = True
    sequence_batch = sequence_batch.__class__(
        **{
            **sequence_batch.as_dict(),
            "slot_valid_mask": slot_valid_mask,
        }
    )
    last_slot_b0 = sequence_batch.slot_positions[0, 0, -1].item()
    last_slot_b1 = sequence_batch.slot_positions[1, 0, -1].item()

    result = build_sequence_attention_mask(sequence_batch)
    mask = result.attention_mask_4d

    assert mask[0, 0, :, last_slot_b0].any().item() is False
    assert mask[1, 0, :, last_slot_b1].any().item() is True


def _build_toy_sequence_batch(batch_size: int = 1, force_last_slot_padding: bool = True):
    perception_batch = MockPerceptionCache(
        MockPerceptionConfig(
            batch_size=batch_size,
            history_frames=2,
            object_slots=2,
            text_t2_len=2,
            vq_tokens_per_frame=1,
            seed=29,
        )
    ).generate()
    sequence_batch = SequenceTokenizer().forward(perception_batch)
    if not force_last_slot_padding:
        return sequence_batch
    return _with_one_padding_slot(sequence_batch)


def _with_one_padding_slot(sequence_batch):
    slot_valid_mask = sequence_batch.slot_valid_mask.clone()
    slot_valid_mask[:, :, -1] = False
    return sequence_batch.__class__(
        **{
            **sequence_batch.as_dict(),
            "slot_valid_mask": slot_valid_mask,
        }
    )


def _first_position_of_type(token_type_ids: torch.Tensor, token_type: TokenType) -> int:
    positions = torch.nonzero(token_type_ids[0] == int(token_type), as_tuple=False)
    assert positions.numel() > 0
    return positions[0, 0].item()
