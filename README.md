# OA-WAM Reproduction Workspace

This repository is a learning-oriented reproduction workspace for OA-WAM. The goal is to rebuild the paper's core mechanisms step by step, while keeping the implementation small enough to inspect, test, and understand.

## Current Direction

- First build a mechanism-level MVE: frozen backbone forward, slot vectors, sequence construction, OA attention/reset invariants, and address swap diagnostics.
- Use Stage0-mini as the practical substitute for the unavailable and unaffordable large-scale paper Stage 0.
- Treat approximate reproduction as trend validation, not strict paper-number reproduction.

## Workflow

Before starting work, read:

1. `CLAUDE.md`
2. `specs/00-project-status.md`
3. `specs/04-interface-contracts.md`
4. `docs/agent-workflow.md`

Each milestone such as M0, M1, or M2 follows:

1. Define the scoped interface, shapes, and invariants.
2. Implement the smallest useful module.
3. Add CPU shape/invariant tests.
4. Update status and handoff documents.
5. Review the code with `learning-coach` before moving to the next milestone.

## Environment

Create the recommended environment:

```bash
conda env create -f environment.yml
conda activate oa-wam
```

Run the current M1 tests:

```bash
pytest tests/test_m1_mock_cache_slot_vector.py -q
```

Codex-side validation is CPU-first. The physical training server is assumed to be 8x RTX 4090 48GB, but real GPU training is started by the user in a GPU-visible terminal.

## Repository Layout

- `src/perception/`: perception cache schema and mock cache.
- `src/tokenizer/`: slot vector and sequence construction code.
- `specs/`: executable architecture, training, experiment, and interface specs.
- `docs/`: project workflow and learning support documents.
- `handoffs/`: task handoffs and implementation notes.
- `tests/`: focused CPU tests for shapes and invariants.

## Git Hygiene

Do not commit datasets, generated caches, checkpoints, or experiment outputs. The local `paper/` folder is ignored because paper PDFs and source files are reference material rather than project implementation artifacts.
