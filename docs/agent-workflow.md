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

## Milestone Learning Loop
当前项目默认按 milestone 推进。这里的 milestone 指 M0、M1、M2 这类项目里程碑，不是数学变量 M。每个 milestone 不在代码完成时立刻关闭，而是在学习 review 完成后关闭。

### Phase 1: Agent Build
Agent 负责：
- 明确本 M 的输入输出和验收边界。
- 写最小实现和 CPU 可跑的 shape/invariant 测试。
- 更新 specs/status/handoff。
- 在 final 中列出 touched files、测试结果、需要用户重点 review 的代码段。

### Phase 2: Guided Code Review
用户负责：
- 从 handoff 的 touched files 开始读代码。
- 遇到不懂的维度、索引、mask、广播、dtype/device，直接要求启用 `learning-coach` 逐段讲解。
- 不急着进入下一个 milestone，先确认自己能解释当前 milestone 的核心数据流。

Agent 负责：
- 用 `learning-coach` 风格解释代码，不直接跳到下个任务。
- 对每个文件给出 shape 表、关键 invariant、常见 bug 和 3-5 个检查问题。
- 对用户解释中的误解做纠正。

### Phase 3: Close Gate
进入下一个 milestone 前必须满足：
- 相关测试通过。
- `specs/00-project-status.md` 和 handoff 已更新。
- 用户已经 review 核心文件。
- 如用户认为有必要，用户本人将新困惑、已学到的 invariant 或 bug 写入 `docs/learning-log.md`。Agent 不直接编辑该文件。

如果用户还没看懂，不推进新 M；优先继续拆代码、画 shape、补小 drill。

## Division of Labor
默认可以采用 Agent-first：
- 适用：接口 dataclass、mock cache、环境配置、文档、非核心 glue code、测试骨架。
- 形式：Agent 先完成最小可运行版本，用户逐文件学习和 review。

核心张量逻辑采用 Human-first：
- 适用：attention mask、slot scatter、`addr/content` 切片、每层 addr reset、loss mask、flow matching target、diagnostic。
- 形式：Agent 写接口、测试和 TODO scaffold；用户亲手实现关键几行；Agent 做 reviewer 和 learning coach。

这不是为了降低效率，而是为了避免最该训练的张量推理能力被 Agent 直接替代。

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
4. 复盘时：如果用户愿意，由用户自己把困惑、bug、学到的 invariant 写入 `docs/learning-log.md`。

适合交给 `learning-coach` 的 prompt：

```text
Use the learning-coach agent. I am about to implement <module>.
Teach me the concept, expected tensor shapes, invariants to test, and 5 check questions.
Do not write the implementation.
```

适合在 Agent 已经完成一个 M 后使用的 prompt：

```text
启用 .codex/agents/learning-coach.toml。
从 <file> 开始，按代码顺序讲解今天这个 M 的实现。
重点讲 shape、索引、广播、mask、dtype/device，以及我应该如何 review。
不要进入下一个 milestone，直到我能用自己的话解释清楚。
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
