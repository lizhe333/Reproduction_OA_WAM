# OA-WAM Reproduction Workspace

本仓库的目标不是一次性外包复现，而是在复现 OA-WAM 的过程中训练代码能力。Agent 的职责是拆题、审查、补测试和记录决策；关键张量逻辑优先由人类开发者亲手实现。

## Startup Protocol
- 先读本文件，再读 `specs/00-project-status.md` 和 `specs/04-interface-contracts.md`。
- 如果要改接口，必须先更新 `specs/04-interface-contracts.md`，再改代码。
- 如果要改训练、评估或消融范围，必须同步更新对应的 `specs/02-training-config.md` 或 `specs/03-experiment-matrix.md`。
- 每次任务结束必须写 `handoffs/<YYYYMMDD>-<agent>-<task>.md`，并更新 `specs/00-project-status.md`。

## Work Style
- 一次只领取一个子系统或一个小模块，避免跨模块大改。
- 每个模块必须包含：spec 对齐、实现、最小测试、handoff。
- 先写 shape/invariant 测试，再写核心实现；尤其是 mask、slot padding、addr reset、loss mask。
- 不追求一开始复现论文数值，先追求机制正确：forward、OA invariant、address swap diagnostic；机制通过后，用 Stage0-mini 替代不可承受的大规模 Stage 0，目标是近似复现实验趋势和部分结果。
- 不把大数据、缓存、权重、实验输出提交到 Git。

## Human-Learning Rule
以下模块优先由人类开发者亲手写 60%-80%，Agent 主要做 review 和测试：
- attention mask 构造
- `addr/content` 切片逻辑
- 每层 address reset hook
- loss mask 和 padding mask
- flow matching target
- address swap diagnostic

## Current Stage
阶段一：深度理解 + MVE 骨架 + Stage0-mini 方案固化。当前优先级：
1. 论文规格固化为接口契约。
2. 跑通 frozen backbone 的 `inputs_embeds` forward。
3. 跑通 slot adapter + sequence construction。
4. 验证 OA attention/reset 的 invariant。
5. 跑通 address swap diagnostic。
6. 在机制通过后进入 Stage0-mini：用公开 Chameleon-style backbone + LoRA/冻结主干 + cached perception，训练 slot-aware warmup（`L_world + 0.04 L_vq`），再进入 Stage I-lite / Stage II。

## Key Constraints
- 感知栈（SAM3/DINOv3/VQ-GAN/Qwen3-VL）默认冻结，并优先离线缓存。
- Backbone 使用现有 Chameleon-style/VLM backbone，不复现论文 384xA100 规模 Stage 0。
- Stage 0: `ckpt-stage0` 尚未发布，当前正式执行 Stage0-mini 替代路线，不声称严格论文数值复现；未来若 checkpoint 发布可重新评估。
- Stage0-mini: 冻结或 LoRA 化公开 backbone，训练 slot adapter、`f_addr`/`f_cnt`、world head 和必要 LoRA，使主干适应 slot token、OA mask/reset、VQ/world 目标。
- Stage I-lite: 建议保留，用于 slot adapter + world head 对齐或 small-batch overfit。
- Stage II: LoRA + action head + full loss 微调。
- 当前硬件假设：8x RTX 4090 48G；优先 LoRA/QLoRA、gradient accumulation、cached perception。

## Agent Roles
- `Coordinator`: 维护状态、阻塞、handoff 和任务粒度。
- `PaperSpec`: 从 LaTeX 提取可执行规格，不写模型实现。
- `BackboneIntegration`: 对接现有 backbone、reserved token、`inputs_embeds`、LoRA target。
- `PerceptionCache`: 定义并实现冻结感知缓存，可先提供 mock cache。
- `SequenceTokenizer`: 实现 slot vector、masked scatter、block causal mask。
- `ModelSurgery`: 实现 OA key mask、addr reset hook、slot-aware trunk wrapper。
- `HeadsTraining`: 实现 world/action heads、loss、Stage0-mini/Stage I/Stage II loop。
- `EvalAblation`: 实现 LIBERO smoke eval、A1/A2/A3/A4 diagnostics。
- `Reviewer`: 只做 bug/risk/test review，不主动扩大实现范围。
- `LearningCoach`: 解释论文和代码，设计练习，反向提问，帮助人类开发者形成自己的理解；不直接替代实现。

## Handoff Format
每个 handoff 必须包含：
- Done:
- Artifacts:
- Decisions:
- Tests:
- Blockers:
- Next:
