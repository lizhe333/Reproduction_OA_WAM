# Backbone 集成分析

## Done:
- 阅读 `CLAUDE.md` 和 `specs/04-interface-contracts.md`，确认 M1 之前的接口状态
- 收集 Chameleon-7B 的完整架构参数（32层/4096/32头/FFN 11008/vocab 65536/RoPE θ=10000）
- 确认 HuggingFace `ChameleonModel.forward()` 支持 `inputs_embeds` 参数（v4.44+）
- 调研 Lumina-mGPT-7B-768 作为替代 backbone 的可行性
- 分析 OA 机制对 hidden_size 的硬依赖程度（结论：无硬依赖）
- 输出 backbone 信息清单、候选推荐、风险评估、M1 实现建议、mock backbone 最小接口

## Artifacts:
- 分析报告通过 Agent 回复输出，问题已逐项回答
- 无新建代码文件（M1 之前属于分析阶段）

## Decisions:
- 强烈推荐 `facebook/chameleon-7b` 作为目标 backbone
- `hidden_size=4096` 作为默认配置，但所有 adapter/head 从 BackboneConfig 读取（非硬编码）
- M1 可以在 backbone 未确定时推进，只需 slot adapter out_features 可配置
- M2 前需确定 tokenizer 和 reserved token 方案
- M3 前需下载 Chameleon-7B 权重（约 14GB bf16）
- OA 关键机制（key mask on first 32 dims, addr reset）与 D 维度无关，换 backbone 不影响核心 invariant
- 不使用论文同款 backbone 的最大风险是复现可信度，而非架构不兼容

## Tests:
- 无（分析阶段，无代码产出）

## Blockers:
- 无新阻塞

## Next:
- Coordinator 确认 backbone 选择（facebook/chameleon-7b 或替代）
- PerceptionCache Agent 准备 M1（Mock Cache）
- SequenceTokenizer Agent 准备 M1（Slot Vector）
- BackboneIntegration Agent 的下一步是 M3 前准备 backbone wrapper
