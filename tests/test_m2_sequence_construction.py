import pytest
import torch

from src.perception import MockPerceptionCache, MockPerceptionConfig

try:
    from src.tokenizer import (
        SequenceTokenizer,
        TokenType,
        build_sequence_attention_mask,
        build_sequence_embeddings,
    )
except ImportError:
    SequenceTokenizer = None
    TokenType = None
    build_sequence_attention_mask = None
    build_sequence_embeddings = None


pytestmark = pytest.mark.xfail(
    SequenceTokenizer is None
    or build_sequence_embeddings is None
    or build_sequence_attention_mask is None,
    reason="M2 sequence construction components are not available yet",
    strict=False,
)


HIDDEN_SIZE = 8
VOCAB_SIZE = 17000


def test_m2_sequence_construction_composes_layout_embeddings_and_mask():
    perception_batch = MockPerceptionCache(
        MockPerceptionConfig(
            batch_size=1,
            history_frames=2,
            object_slots=2,
            text_t2_len=2,
            vq_tokens_per_frame=1,
            seed=31,
        )
    ).generate()
    sequence_batch = SequenceTokenizer().forward(perception_batch)

    sequence_batch = build_sequence_embeddings(
        sequence_batch=sequence_batch,
        perception_batch=perception_batch,
        embed_tokens=_build_embedding_table(),
        slot_adapter=_DeterministicSlotAdapter(hidden_size=HIDDEN_SIZE),
    )
    sequence_batch = build_sequence_attention_mask(sequence_batch)

    batch_size, sequence_length = sequence_batch.input_ids.shape
    assert sequence_batch.token_type_ids.shape == (batch_size, sequence_length)
    assert sequence_batch.inputs_embeds.shape == (batch_size, sequence_length, HIDDEN_SIZE)
    assert sequence_batch.attention_mask_4d.shape == (
        batch_size,
        1,
        sequence_length,
        sequence_length,
    )
    assert sequence_batch.attention_mask_4d.dtype == torch.bool

    first_slot_pos = sequence_batch.slot_positions[0, 0, 0].item()
    assert sequence_batch.token_type_ids[0, first_slot_pos].item() == int(TokenType.SLOT)
    assert torch.allclose(
        sequence_batch.inputs_embeds[0, first_slot_pos],
        sequence_batch.slot_embeds[0, 0, 0],
    )

    act_q_pos = sequence_batch.act_q_position[0].item()
    assert sequence_batch.attention_mask_4d[0, 0, act_q_pos, first_slot_pos].item() is True
    assert sequence_batch.attention_mask_4d[0, 0, first_slot_pos, act_q_pos].item() is False


def _build_embedding_table() -> torch.nn.Embedding:
    embed_tokens = torch.nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
    with torch.no_grad():
        values = torch.arange(VOCAB_SIZE * HIDDEN_SIZE, dtype=torch.float32)
        embed_tokens.weight.copy_(values.view(VOCAB_SIZE, HIDDEN_SIZE) / 1000.0)
    return embed_tokens


class _DeterministicSlotAdapter(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, slot_vector: torch.Tensor) -> torch.Tensor:
        return slot_vector[..., : self.hidden_size].contiguous()
