# Agent Workflow

本项目的 agent 协作目标是帮助人类开发者练代码能力，而不是让 agent 独立吞掉整个复现。

## Standard Loop
1. 领取一个小任务。
2. 阅读 `CLAUDE.md`、`specs/00-project-status.md`、`specs/04-interface-contracts.md`。
3. 如有接口变化，先改 specs。
4. 写或更新最小测试。
5. 实现最小代码。
6. 运行相关测试。
7. 更新项目状态。
8. 写 handoff。

## Codex Subagents
项目级 Codex custom agents 放在 `.codex/agents/`。使用方法见 `docs/codex-multi-agent.md`。

注意：subagent 适合并行读代码、查论文、做 review、总结测试日志；不适合让多个写代码 agent 同时改同一批文件。

## Task Size Rule
一个任务最多覆盖一个模块：
- good: `实现 slot vector dataclass + shape test`
- good: `实现 addr reset hook + invariant test`
- bad: `实现整个 OA-WAM`
- bad: `把训练和评估都搭完`

## Review Questions
每个核心模块完成后，Reviewer Agent 应至少问：
- 这个模块的输入输出 shape 是否写进 interface contract？
- padding slot 是否会泄漏到 attention 或 loss？
- 哪些 tensor 应该 detach？
- 这个测试能否在 CPU 或小 GPU 上跑？
- 如果关闭该模块的关键开关，diagnostic 是否会变化？

## Learning Coach Loop
当目标是提升代码能力时，优先让 `learning-coach` 参与每个核心模块：
1. 实现前：解释概念、shape、invariant，并给一个小 drill。
2. 实现中：只提示下一步或指出误解，不直接整段代写。
3. 实现后：让你先口头解释，再检查解释是否正确。
4. 复盘时：把困惑、bug、学到的 invariant 写入 `docs/learning-log.md`。

适合交给 `learning-coach` 的 prompt：

```text
Use the learning-coach agent. I am about to implement <module>.
Teach me the concept, expected tensor shapes, invariants to test, and 5 check questions.
Do not write the implementation.
```

## Handoff Template
```md
# Handoff: <task>

## Done
-

## Artifacts
-

## Decisions
-

## Tests
-

## Blockers
-

## Next
-
```
