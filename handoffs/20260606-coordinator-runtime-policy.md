# Handoff: Runtime Policy

## Done
- Documented that the physical training server is 8x RTX 4090 48GB.
- Documented that Codex sandbox verification defaults to CPU tests and smoke checks.
- Documented that real Stage0-mini / Stage II GPU training is started by the user in a GPU-visible terminal.

## Artifacts
- `CLAUDE.md`
- `specs/02-training-config.md`
- `specs/00-project-status.md`

## Decisions
- `torch.cuda.is_available()==False` inside Codex is not treated as a project environment failure.
- Agent work focuses on code, configs, CPU unit tests, training commands, and log diagnosis.

## Tests
- Documentation-only update; no code tests required.

## Blockers
- None.

## Next
- Continue M2 sequence construction with CPU shape/invariant tests.
