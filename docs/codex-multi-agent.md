# Codex Multi-Agent Setup

## What is supported
Codex subagent workflows can spawn specialized agents in parallel when explicitly requested. Project-scoped custom agents live in `.codex/agents/*.toml`.

Current caveat: subagent activity is surfaced in the Codex app and CLI. IDE extension visibility is documented as coming soon, so VSCode may not show the full subagent thread UI even though the same project instructions and CLI configuration are shared.

## Project agents
- `coordinator`
- `paper-spec`
- `backbone-integration`
- `perception-cache`
- `sequence-tokenizer`
- `model-surgery`
- `heads-training`
- `eval-ablation`
- `test-reviewer`
- `learning-coach`

## Recommended prompts

### Parallel spec review
```text
Use subagents in parallel. Spawn paper-spec, backbone-integration, and test-reviewer.
Ask paper-spec to verify specs/01-architecture.md against paper/*.tex.
Ask backbone-integration to identify what must be known about the current backbone before M1.
Ask test-reviewer to review specs/04-interface-contracts.md for missing invariants.
Wait for all agents and summarize decisions, blockers, and next actions.
```

### Focused implementation review
```text
Use the test-reviewer agent to review my uncommitted changes.
Focus on tensor shapes, mask leakage, gradient paths, and missing tests.
Do not implement fixes yet.
```

### MVE implementation split
```text
Use subagents in parallel for read-heavy planning only.
Spawn sequence-tokenizer to propose the smallest M1 implementation plan.
Spawn model-surgery to propose the OA invariant tests.
Wait for both, then give me a human-coding checklist. Do not edit files.
```

### Learning-focused module session
```text
Use the learning-coach agent.
Teach me the slot_vector module before I implement it.
Explain the concept, list tensor shapes, give me the invariant tests, then ask me 5 check questions.
Do not write the implementation.
```

### Post-implementation teaching review
```text
Use learning-coach and test-reviewer.
Ask learning-coach to check whether my explanation of the module is correct.
Ask test-reviewer to inspect the code for shape, mask, and gradient risks.
Wait for both and separate learning feedback from code findings.
```

## CLI examples
Run these from the repository root:

```bash
codex "Use the paper-spec agent to check whether specs/01-architecture.md matches the LaTeX source."
```

```bash
codex "Use subagents in parallel: paper-spec reviews the paper spec, test-reviewer reviews interface contracts. Wait for both and summarize."
```

## VSCode workflow
In the VSCode Codex extension:
1. Open the `OA-WAM` repository root.
2. Start a new Codex thread.
3. Ask Codex to read `AGENTS.md`.
4. Use the recommended prompts above.

If you need to inspect active subagent threads directly, use the Codex CLI or Codex app until IDE subagent visibility is available.
