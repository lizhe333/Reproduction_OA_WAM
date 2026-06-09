import pytest
import torch

from src.perception import MockPerceptionCache, MockPerceptionConfig

try:
    from src.tokenizer.sequence import (
        ACT_Q_TOKEN_ID,
        SLOT_TOKEN_ID,
        SequenceTokenizer,
        TokenType,
    )
except ImportError:
    ACT_Q_TOKEN_ID = 16505
    SLOT_TOKEN_ID = 16500
    SequenceTokenizer = None
    TokenType = None


pytestmark = pytest.mark.xfail(
    SequenceTokenizer is None,
    reason="M2a SequenceTokenizer layout implementation is not available yet",
    strict=False,
)


def test_sequence_slot_positions_match_token_type():
    sequence = _build_toy_sequence()
    input_ids = _field(sequence, "input_ids")
    token_type_ids = _field(sequence, "token_type_ids")
    slot_positions = _field(sequence, "slot_positions")

    expected_slot_positions, _, expected_length = _expected_toy_layout()

    assert input_ids.shape == (1, expected_length)
    assert token_type_ids.shape == (1, expected_length)
    assert slot_positions.shape == (1, 2, 3)
    assert torch.equal(slot_positions.cpu(), expected_slot_positions)

    slot_type = _token_type("SLOT")
    for position in slot_positions[0].flatten().tolist():
        assert input_ids[0, position].item() == SLOT_TOKEN_ID
        assert token_type_ids[0, position].item() == slot_type


def test_act_q_position_matches_special_token():
    sequence = _build_toy_sequence()
    input_ids = _field(sequence, "input_ids")
    token_type_ids = _field(sequence, "token_type_ids")
    act_q_position = _field(sequence, "act_q_position")

    _, expected_act_q_position, expected_length = _expected_toy_layout()

    assert act_q_position.shape == (1,)
    assert act_q_position[0].item() == expected_act_q_position
    assert act_q_position[0].item() == expected_length - 1

    q_position = act_q_position[0].item()
    assert input_ids[0, q_position].item() == ACT_Q_TOKEN_ID
    assert token_type_ids[0, q_position].item() == _token_type("SPECIAL")


def test_sequence_has_exactly_t_times_s_slot_tokens():
    sequence = _build_toy_sequence()
    token_type_ids = _field(sequence, "token_type_ids")
    slot_positions = _field(sequence, "slot_positions")

    slot_type = _token_type("SLOT")
    slot_token_count = token_type_ids[0].eq(slot_type).sum().item()
    flattened_positions = slot_positions[0].flatten()

    assert slot_token_count == 2 * 3
    assert flattened_positions.unique().numel() == 2 * 3


def _build_toy_sequence():
    if SequenceTokenizer is None:
        pytest.xfail("M2a SequenceTokenizer layout implementation is not available yet")

    batch = MockPerceptionCache(
        MockPerceptionConfig(
            batch_size=1,
            history_frames=2,
            object_slots=2,
            text_t2_len=2,
            vq_tokens_per_frame=0,
            seed=23,
        )
    ).generate()

    return SequenceTokenizer().forward(batch)


def _expected_toy_layout() -> tuple[torch.Tensor, int, int]:
    text_len = 2
    frames = 2
    slots = 3
    vq_tokens_per_frame = 0
    state_tokens_per_frame = 7
    action_tokens_per_transition = 7

    position = text_len
    slot_positions: list[list[int]] = []

    for frame_idx in range(frames):
        position += 1  # F_BOS
        position += vq_tokens_per_frame
        position += 1  # S_BOS
        slot_positions.append(list(range(position, position + slots)))
        position += slots
        position += 1  # S_EOS
        position += state_tokens_per_frame
        if frame_idx < frames - 1:
            position += action_tokens_per_transition
        position += 1  # F_EOS

    act_q_position = position
    sequence_length = act_q_position + 1

    return torch.tensor([slot_positions], dtype=torch.long), act_q_position, sequence_length


def _field(sequence, name: str):
    if isinstance(sequence, dict):
        return sequence[name]
    return getattr(sequence, name)


def _token_type(name: str) -> int:
    value = getattr(TokenType, name)
    if hasattr(value, "value"):
        return int(value.value)
    return int(value)
