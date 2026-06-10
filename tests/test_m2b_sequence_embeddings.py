import pytest
import torch

from src.perception import MockPerceptionCache, MockPerceptionConfig

try:
    from src.tokenizer import (
        SLOT_TOKEN_ID,
        SequenceTokenizer,
        TokenType,
        build_sequence_embeddings,
    )
except ImportError:
    SLOT_TOKEN_ID = 16500
    SequenceTokenizer = None
    TokenType = None
    build_sequence_embeddings = None


pytestmark = pytest.mark.xfail(
    SequenceTokenizer is None or build_sequence_embeddings is None,
    reason="M2b sequence embedding scaffold is not available yet",
    strict=False,
)


HIDDEN_SIZE = 8
VOCAB_SIZE = 17000


def test_sequence_embeddings_builds_m2b_fields():
    perception_batch, sequence_batch, embed_tokens, slot_adapter = _build_toy_m2b_inputs()

    result = build_sequence_embeddings(
        sequence_batch=sequence_batch,
        perception_batch=perception_batch,
        embed_tokens=embed_tokens,
        slot_adapter=slot_adapter,
    )

    assert result.inputs_embeds is not None
    assert result.slot_vector is not None
    assert result.slot_embeds is not None


def test_sequence_embeddings_scatter_slot_vectors_to_slot_positions():
    perception_batch, sequence_batch, embed_tokens, slot_adapter = _build_toy_m2b_inputs()

    result = build_sequence_embeddings(
        sequence_batch=sequence_batch,
        perception_batch=perception_batch,
        embed_tokens=embed_tokens,
        slot_adapter=slot_adapter,
    )

    assert result.inputs_embeds.shape == (1, sequence_batch.input_ids.shape[1], HIDDEN_SIZE)
    assert result.slot_vector.shape == (1, 2, 3, 320)
    assert result.slot_embeds.shape == (1, 2, 3, HIDDEN_SIZE)

    for frame_idx in range(2):
        for slot_idx in range(3):
            position = result.slot_positions[0, frame_idx, slot_idx].item()
            assert result.input_ids[0, position].item() == SLOT_TOKEN_ID
            assert result.token_type_ids[0, position].item() == int(TokenType.SLOT)
            assert torch.allclose(
                result.inputs_embeds[0, position],
                result.slot_embeds[0, frame_idx, slot_idx],
            )


def test_sequence_embeddings_preserve_non_slot_token_embeddings():
    perception_batch, sequence_batch, embed_tokens, slot_adapter = _build_toy_m2b_inputs()
    expected_base_embeds = embed_tokens(sequence_batch.input_ids)

    result = build_sequence_embeddings(
        sequence_batch=sequence_batch,
        perception_batch=perception_batch,
        embed_tokens=embed_tokens,
        slot_adapter=slot_adapter,
    )

    non_slot_mask = sequence_batch.token_type_ids.ne(int(TokenType.SLOT))
    assert torch.allclose(result.inputs_embeds[non_slot_mask], expected_base_embeds[non_slot_mask])


def test_sequence_embeddings_reject_slot_adapter_hidden_dim_mismatch():
    perception_batch, sequence_batch, embed_tokens, _ = _build_toy_m2b_inputs()
    bad_slot_adapter = _DeterministicSlotAdapter(hidden_size=HIDDEN_SIZE + 1)

    with pytest.raises(ValueError, match="hidden dim"):
        build_sequence_embeddings(
            sequence_batch=sequence_batch,
            perception_batch=perception_batch,
            embed_tokens=embed_tokens,
            slot_adapter=bad_slot_adapter,
        )


def _build_toy_m2b_inputs():
    perception_batch = MockPerceptionCache(
        MockPerceptionConfig(
            batch_size=1,
            history_frames=2,
            object_slots=2,
            text_t2_len=2,
            vq_tokens_per_frame=0,
            seed=23,
        )
    ).generate()
    sequence_batch = SequenceTokenizer().forward(perception_batch)
    embed_tokens = _build_embedding_table()
    slot_adapter = _DeterministicSlotAdapter(hidden_size=HIDDEN_SIZE)
    return perception_batch, sequence_batch, embed_tokens, slot_adapter


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
