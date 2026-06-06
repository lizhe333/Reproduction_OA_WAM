# Handoff: Learning Workflow Update

## Done
- Added the default milestone workflow: Agent build, guided human review, explain-back, close gate.
- Clarified Agent-first vs Human-first division of labor.
- Added a post-implementation learning-coach prompt.
- Clarified that `docs/learning-log.md` is user-owned and must not be edited by Agent unless explicitly requested.

## Artifacts
- `CLAUDE.md`
- `docs/agent-workflow.md`

## Decisions
- A milestone is not considered closed just because code and tests pass; it also needs guided review.
- `docs/learning-log.md` contains the user's own reflections; Agent may suggest prompts but should not write entries directly.
- Interface/glue modules can be Agent-first.
- Core tensor reasoning modules should be Human-first with Agent tests/scaffold/review.

## Tests
- Documentation-only update; no code tests required.

## Blockers
- None.

## Next
- Continue learning review of M1 files before starting M2.
