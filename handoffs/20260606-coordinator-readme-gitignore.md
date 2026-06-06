# Handoff: README and Paper Ignore

## Done
- Added a project `README.md` with goals, workflow, environment setup, layout, and git hygiene.
- Added `paper/` to `.gitignore`.
- Removed the paper Git strategy item from current blockers and recorded the decision.

## Artifacts
- `README.md`
- `.gitignore`
- `specs/00-project-status.md`

## Decisions
- `paper/` is local reference material and should be ignored for future changes.
- Existing tracked `paper/` files remain tracked until explicitly removed from the Git index.

## Tests
- Documentation-only update; no code tests required.

## Blockers
- None.

## Next
- If desired, run `git rm --cached -r paper` to stop tracking already tracked paper files while keeping them on disk.
