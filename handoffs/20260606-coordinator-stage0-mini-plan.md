# Handoff: Stage0-mini Reproduction Plan

## Done
- Updated the project direction from mechanism-only MVE toward approximate experimental reproduction using Stage0-mini as a replacement for the unavailable large-scale Stage 0.
- Clarified that Stage0-mini is not equivalent to the paper's 384xA100 Stage 0 and must be reported separately from released Stage 0 checkpoints.
- Added required baselines for approximate reproduction: direct Chameleon, Stage0-mini full OA, no-key-mask, no-reset, and no-OA.
- Added Stage0-mini agent orchestration prompts for read-heavy planning and correctness review.

## Artifacts
- `CLAUDE.md`
- `specs/00-project-status.md`
- `specs/02-training-config.md`
- `specs/03-experiment-matrix.md`
- `docs/codex-multi-agent.md`
- `handoffs/20260606-coordinator-stage0-mini-plan.md`

## Decisions
- User confirmed `ckpt-stage0.pt` is not released; Stage0-mini is now the active route, not just a fallback.
- Stage0-mini trains slot adapter, `f_addr`/`f_cnt`, world head, LoRA, and optionally SE(3) bias using `L_world + 0.04 L_vq`.
- Stage0-mini's purpose is to make a public Chameleon-style backbone slot-aware before Stage II, not to claim paper-level Stage 0 reproduction.
- Approximate result claims must include direct-Chameleon or no-OA baselines.

## Tests
- No code tests run; this was a documentation and planning update only.

## Blockers
- Need BackboneIntegration confirmation for the Stage0-mini base (`facebook/chameleon-7b` or another Chameleon-style base): reserved IDs, `inputs_embeds`, `position_ids`, LoRA target names, and VQ/lm_head compatibility.
- Need Stage0-mini data strategy: start LIBERO-only cache, then optionally add OXE/DROID subset.
- Existing tracked PDFs still need a Git policy decision.

## Next
- Before M1 implementation, run the Stage0-mini planning prompt in `docs/codex-multi-agent.md` if multi-agent planning is desired.
- Keep M1 small: mock cache + slot vector shape tests still comes before real Stage0-mini training.
- After M1-M6 pass, implement Stage0-mini training and compare against direct-Chameleon baseline before attempting full Stage II.
