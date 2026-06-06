import pytest
import torch

from src.perception import (
    MockPerceptionCache,
    MockPerceptionConfig,
    OBJECT_ROLE_ID,
    PADDING_ROLE_ID,
    ROBOT_ROLE_ID,
)
from src.tokenizer import build_slot_vector


def test_slot_vector_shape():
    batch = MockPerceptionCache(
        MockPerceptionConfig(batch_size=2, history_frames=4, object_slots=3, seed=7)
    ).generate()

    slot_vector = build_slot_vector(batch)

    assert slot_vector.shape == (2, 4, 4, 320)
    assert slot_vector.dtype == batch.content.dtype
    assert slot_vector.device == batch.content.device


def test_slot_vector_component_order():
    batch = MockPerceptionCache(
        MockPerceptionConfig(batch_size=1, history_frames=2, object_slots=2, seed=11)
    ).generate()

    slot_vector = build_slot_vector(batch)

    assert torch.equal(slot_vector[..., :32], batch.addr)
    assert torch.equal(slot_vector[..., 32:288], batch.content)
    assert torch.equal(slot_vector[..., 288:304], batch.slot_time_emb)
    assert torch.equal(slot_vector[..., 304:320], batch.slot_role_emb)


def test_mock_cache_role_and_padding_masks():
    batch = MockPerceptionCache(
        MockPerceptionConfig(
            batch_size=2,
            history_frames=3,
            object_slots=4,
            valid_object_counts=(2, 0),
            seed=13,
        )
    ).generate()

    assert batch.slot_role_ids[0, :, 0].eq(ROBOT_ROLE_ID).all()
    assert batch.slot_role_ids[0, :, 1:3].eq(OBJECT_ROLE_ID).all()
    assert batch.slot_role_ids[0, :, 3:].eq(PADDING_ROLE_ID).all()
    assert batch.slot_valid_mask[0, :, :3].all()
    assert not batch.slot_valid_mask[0, :, 3:].any()

    assert batch.slot_role_ids[1, :, 0].eq(ROBOT_ROLE_ID).all()
    assert batch.slot_role_ids[1, :, 1:].eq(PADDING_ROLE_ID).all()
    assert batch.slot_valid_mask[1, :, 0].all()
    assert not batch.slot_valid_mask[1, :, 1:].any()
    assert not batch.slot_is_object[1].any()


def test_addr_temporal_consistency():
    batch = MockPerceptionCache(
        MockPerceptionConfig(batch_size=2, history_frames=5, object_slots=4, seed=17)
    ).generate()

    first_frame_addr = batch.addr[:, :1]

    assert torch.equal(batch.addr, first_frame_addr.expand_as(batch.addr))
    assert torch.equal(batch.object_track_ids[:, :1], batch.object_track_ids[:, 1:2])
    assert torch.equal(batch.object_track_ids[:, :1], batch.object_track_ids[:, -1:])


def test_padding_slots_are_zeroed_in_mock_cache_and_slot_vector():
    batch = MockPerceptionCache(
        MockPerceptionConfig(
            batch_size=1,
            history_frames=3,
            object_slots=4,
            valid_object_counts=(1,),
            seed=19,
        )
    ).generate()

    padding_mask = ~batch.slot_valid_mask
    slot_vector = build_slot_vector(batch)

    assert batch.addr[padding_mask].abs().sum().item() == pytest.approx(0.0)
    assert batch.content[padding_mask].abs().sum().item() == pytest.approx(0.0)
    assert batch.pose[padding_mask].abs().sum().item() == pytest.approx(0.0)
    assert slot_vector[padding_mask].abs().sum().item() == pytest.approx(0.0)


def test_build_slot_vector_rejects_bad_component_width():
    batch = MockPerceptionCache(MockPerceptionConfig()).generate().as_dict()
    batch["addr"] = batch["addr"][..., :31]

    with pytest.raises(ValueError, match="addr last dim"):
        build_slot_vector(batch)

