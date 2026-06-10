from types import SimpleNamespace

import pytest
import torch

from src.perception import MockPerceptionCache, MockPerceptionConfig
from src.tokenizer import (
    SequenceBatch,
    SequenceTokenizer,
    build_sequence_attention_mask,
    build_sequence_embeddings,
)

try:
    from src.trunk import SlotAwareTrunk, TrunkOutput, chameleon_attention_mask_adapter
except ImportError:
    SlotAwareTrunk = None
    TrunkOutput = None
    chameleon_attention_mask_adapter = None


pytestmark = pytest.mark.xfail(
    SlotAwareTrunk is None or TrunkOutput is None or chameleon_attention_mask_adapter is None,
    reason="M3 SlotAwareTrunk implementation is not available yet",
    strict=False,
)


def test_slot_aware_trunk_passes_inputs_embeds_and_mask_to_backbone():
    sequence_batch = _build_sequence_batch()
    expected_hidden = _position_encoded_hidden(batch_size=2, sequence_length=15, hidden_size=4)
    adapted_mask = ~sequence_batch.attention_mask_4d
    backbone = _RecordingBackbone(expected_hidden)

    trunk = SlotAwareTrunk(backbone=backbone, mask_adapter=lambda mask: adapted_mask)
    output = trunk(sequence_batch)

    assert isinstance(output, TrunkOutput)
    assert backbone.call["input_ids"] is None
    assert backbone.call["inputs_embeds"] is sequence_batch.inputs_embeds
    assert backbone.call["attention_mask"] is adapted_mask
    assert backbone.call["output_hidden_states"] is True
    assert backbone.call["return_dict"] is True
    assert backbone.call["use_cache"] is False
    assert torch.equal(output.last_hidden_state, expected_hidden)


def test_slot_aware_trunk_gathers_slot_hidden_from_flattened_positions():
    sequence_batch = _build_sequence_batch()
    expected_hidden = _position_encoded_hidden(batch_size=2, sequence_length=15, hidden_size=4)

    output = SlotAwareTrunk(_RecordingBackbone(expected_hidden))(sequence_batch)

    assert output.slot_hidden.shape == (2, 2, 3, 4)
    for batch_idx in range(2):
        for frame_idx in range(2):
            for slot_idx in range(3):
                position = sequence_batch.slot_positions[batch_idx, frame_idx, slot_idx]
                assert torch.equal(
                    output.slot_hidden[batch_idx, frame_idx, slot_idx],
                    expected_hidden[batch_idx, position],
                )


def test_slot_aware_trunk_gathers_act_q_hidden():
    sequence_batch = _build_sequence_batch()
    expected_hidden = _position_encoded_hidden(batch_size=2, sequence_length=15, hidden_size=4)

    output = SlotAwareTrunk(_RecordingBackbone(expected_hidden))(sequence_batch)

    assert output.act_q_hidden.shape == (2, 4)
    assert torch.equal(output.act_q_hidden[0], expected_hidden[0, 13])
    assert torch.equal(output.act_q_hidden[1], expected_hidden[1, 14])


def test_slot_aware_trunk_uses_last_slot_hidden_as_before_reset_in_m3():
    sequence_batch = _build_sequence_batch()
    expected_hidden = _position_encoded_hidden(batch_size=2, sequence_length=15, hidden_size=4)

    output = SlotAwareTrunk(_RecordingBackbone(expected_hidden))(sequence_batch)

    assert torch.equal(output.slot_hidden_before_reset, output.slot_hidden)


def test_slot_aware_trunk_rejects_missing_inputs_embeds():
    sequence_batch = _build_sequence_batch(inputs_embeds=None)
    expected_hidden = _position_encoded_hidden(batch_size=2, sequence_length=15, hidden_size=4)

    with pytest.raises(ValueError, match="inputs_embeds"):
        SlotAwareTrunk(_RecordingBackbone(expected_hidden))(sequence_batch)


def test_slot_aware_trunk_rejects_missing_attention_mask_4d():
    sequence_batch = _build_sequence_batch(attention_mask_4d=None)
    expected_hidden = _position_encoded_hidden(batch_size=2, sequence_length=15, hidden_size=4)

    with pytest.raises(ValueError, match="attention_mask_4d"):
        SlotAwareTrunk(_RecordingBackbone(expected_hidden))(sequence_batch)


def test_chameleon_attention_mask_adapter_converts_true_allow_to_additive_mask():
    attention_mask_4d = torch.tensor(
        [[[[True, False], [True, True]]]],
        dtype=torch.bool,
    )

    additive_mask = chameleon_attention_mask_adapter(attention_mask_4d, dtype=torch.float32)

    assert additive_mask.shape == attention_mask_4d.shape
    assert additive_mask.dtype == torch.float32
    assert additive_mask[0, 0, 0, 0].item() == 0.0
    assert additive_mask[0, 0, 1, 0].item() == 0.0
    assert additive_mask[0, 0, 0, 1].item() < -1e30


def test_slot_aware_trunk_tiny_chameleon_inputs_embeds_smoke():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(11)

    hidden_size = 16
    sequence_batch = _build_sequence_batch(
        hidden_size=hidden_size,
        attention_mask_4d=torch.tril(torch.ones(2, 1, 15, 15, dtype=torch.bool)),
    )
    config = transformers.ChameleonConfig(
        vocab_size=32,
        hidden_size=hidden_size,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        vocabulary_map={"<image>": 16},
    )
    backbone = transformers.ChameleonModel(config)
    backbone.eval()

    trunk = SlotAwareTrunk(
        backbone=backbone,
        mask_adapter=chameleon_attention_mask_adapter,
    )
    with torch.no_grad():
        output = trunk(sequence_batch)

    assert output.last_hidden_state.shape == sequence_batch.inputs_embeds.shape
    assert output.slot_hidden.shape == (2, 2, 3, hidden_size)
    assert output.act_q_hidden.shape == (2, hidden_size)
    assert torch.isfinite(output.last_hidden_state).all()


def test_m2_sequence_batch_runs_through_tiny_chameleon_trunk():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(17)

    hidden_size = 16
    backbone = transformers.ChameleonModel(
        transformers.ChameleonConfig(
            vocab_size=17000,
            hidden_size=hidden_size,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
            vocabulary_map={"<image>": 16},
        )
    )
    backbone.eval()
    perception_batch = MockPerceptionCache(
        MockPerceptionConfig(
            batch_size=1,
            history_frames=2,
            object_slots=2,
            text_t2_len=2,
            vq_tokens_per_frame=1,
            seed=41,
        )
    ).generate()
    sequence_batch = SequenceTokenizer().forward(perception_batch)
    sequence_batch = build_sequence_embeddings(
        sequence_batch=sequence_batch,
        perception_batch=perception_batch,
        embed_tokens=backbone.embed_tokens,
        slot_adapter=_DeterministicSlotAdapter(hidden_size=hidden_size),
    )
    sequence_batch = build_sequence_attention_mask(sequence_batch)

    trunk = SlotAwareTrunk(
        backbone=backbone,
        mask_adapter=chameleon_attention_mask_adapter,
    )
    with torch.no_grad():
        output = trunk(sequence_batch)

    batch_size, sequence_length = sequence_batch.input_ids.shape
    assert output.last_hidden_state.shape == (batch_size, sequence_length, hidden_size)
    assert output.slot_hidden.shape == (batch_size, 2, 3, hidden_size)
    assert output.act_q_hidden.shape == (batch_size, hidden_size)
    assert torch.isfinite(output.last_hidden_state).all()


def _build_sequence_batch(
    *,
    inputs_embeds: torch.Tensor | None | object = ...,
    attention_mask_4d: torch.Tensor | None | object = ...,
    hidden_size: int = 4,
) -> SequenceBatch:
    batch_size = 2
    sequence_length = 15
    slot_positions = torch.tensor(
        [
            [[2, 4, 5], [7, 8, 9]],
            [[1, 3, 6], [10, 11, 12]],
        ],
        dtype=torch.long,
    )
    act_q_position = torch.tensor([13, 14], dtype=torch.long)
    default_inputs_embeds = torch.arange(
        batch_size * sequence_length * hidden_size,
        dtype=torch.float32,
    ).view(batch_size, sequence_length, hidden_size)
    default_attention_mask_4d = torch.ones(
        batch_size,
        1,
        sequence_length,
        sequence_length,
        dtype=torch.bool,
    )

    return SequenceBatch(
        input_ids=torch.zeros(batch_size, sequence_length, dtype=torch.long),
        token_type_ids=torch.zeros(batch_size, sequence_length, dtype=torch.long),
        slot_positions=slot_positions,
        act_q_position=act_q_position,
        f_bos_positions=torch.tensor([[0, 6], [0, 6]], dtype=torch.long),
        s_bos_positions=torch.tensor([[1, 7], [1, 7]], dtype=torch.long),
        special_token_mask=torch.zeros(batch_size, sequence_length, dtype=torch.bool),
        slot_valid_mask=torch.ones(batch_size, 2, 3, dtype=torch.bool),
        slot_is_object=torch.tensor(
            [
                [[False, True, True], [False, True, True]],
                [[False, True, True], [False, True, True]],
            ],
            dtype=torch.bool,
        ),
        inputs_embeds=default_inputs_embeds if inputs_embeds is ... else inputs_embeds,
        attention_mask_4d=(
            default_attention_mask_4d if attention_mask_4d is ... else attention_mask_4d
        ),
    )


def _position_encoded_hidden(
    *,
    batch_size: int,
    sequence_length: int,
    hidden_size: int,
) -> torch.Tensor:
    return torch.arange(
        batch_size * sequence_length * hidden_size,
        dtype=torch.float32,
    ).view(batch_size, sequence_length, hidden_size)


class _RecordingBackbone(torch.nn.Module):
    def __init__(self, last_hidden_state: torch.Tensor) -> None:
        super().__init__()
        self.last_hidden_state = last_hidden_state
        self.call = None

    def forward(
        self,
        *,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        output_hidden_states=None,
        return_dict=None,
        use_cache=None,
    ):
        self.call = {
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
            "use_cache": use_cache,
        }
        return SimpleNamespace(
            last_hidden_state=self.last_hidden_state,
            hidden_states=(inputs_embeds, self.last_hidden_state),
            attentions=None,
        )


class _DeterministicSlotAdapter(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, slot_vector: torch.Tensor) -> torch.Tensor:
        return slot_vector[..., : self.hidden_size].contiguous()
