# Handoff: M1 Mock Cache + Slot Vector

## Done
- Added a typed `PerceptionBatch` schema for mock or cached frozen perception output.
- Added deterministic `MockPerceptionCache` with full M1 `PerceptionBatch` fields.
- Added `build_slot_vector()` to construct `[addr32, content256, time16, role16]` tensors.
- Added M1 tests for shape, component order, role/padding masks, temporal addr consistency, padding zeroing, and invalid component width.
- Added `environment.yml` for the recommended OA-WAM conda environment.

## Artifacts
- `src/perception/schema.py`
- `src/perception/mock_cache.py`
- `src/perception/__init__.py`
- `src/tokenizer/slot_vector.py`
- `src/tokenizer/__init__.py`
- `tests/test_m1_mock_cache_slot_vector.py`
- `environment.yml`
- `specs/00-project-status.md`

## Decisions
- M1 stays on deterministic mock tensors and does not read LIBERO yet.
- Padding slots are zeroed in mock `addr/content/pose/dino_features/slot_vector`.
- Robot slot uses role id `1`, object slots use `2`, padding uses `0`, matching the interface contract.
- `addr` is broadcast across time in mock cache to test the episode-level identity invariant.
- New project env is `oa-wam`; `environment.yml` pins `numpy<2` for PyTorch 2.2.x ABI compatibility.

## Tests
- `conda run -n oa-wam python -c "import numpy, torch, transformers, peft, accelerate; ..."`
- Result: `numpy 1.26.4`, `torch 2.2.2`, `cuda 12.1`, `cuda_available False`, `transformers 4.46.3`, `peft 0.19.1`, `accelerate 1.13.0`
- `conda run -n oa-wam pytest tests/test_m1_mock_cache_slot_vector.py -q`
- Result: `6 passed in 1.18s`

## Blockers
- Real Stage0-mini still needs a LIBERO cached perception pipeline after M1-M6.
- `object_track_ids` remain mock-fixed; real LIBERO cache must provide reliable tracking or oracle IDs.

## Next
- Begin M2: sequence construction (`input_ids`, slot placeholders, `inputs_embeds`, `attention_mask_4d`, `slot_positions`, `act_q_position`).
- Add tests for `test_sequence_slot_positions_match_token_type` and initial attention mask invariants.
