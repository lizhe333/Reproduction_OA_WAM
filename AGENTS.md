# OA-WAM Codex Instructions

This repository uses `CLAUDE.md` as the canonical project workflow document.
Before doing any work, read:

1. `CLAUDE.md`
2. `specs/00-project-status.md`
3. `specs/04-interface-contracts.md`

Follow the agent workflow in `docs/agent-workflow.md`.

Core rules:
- Keep tasks small and module-scoped.
- Update specs before changing an interface.
- Add or update tests for tensor shapes and invariants.
- Write a handoff under `handoffs/` after each task.
- Do not commit datasets, caches, checkpoints, or generated experiment outputs.
