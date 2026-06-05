# 项目仪表盘

## North Star
先完成机制级 MVE，再扩展到论文级训练与评估。MVE 成功标准不是 LIBERO 高分，而是：
- frozen backbone 能通过 `inputs_embeds` 完成 forward；
- slot adapter + sequence construction shape 正确；
- OA key mask 和 addr reset hook 的 invariant 有单元测试覆盖；
- address swap diagnostic 能观测到目标绑定变化。

## Milestones
| 里程碑 | 状态 | 验收标准 |
|--------|------|----------|
| M0: Paper spec freeze | ✅ done | `specs/01-architecture.md` 和 `04-interface-contracts.md` 已填实，三个 Agent 审查完成 |
| M1: Mock cache + slot vector | ⬜ pending | 可构造 batch slot cache，并通过 shape/padding 测试 |
| M2: Sequence construction | ⬜ pending | 可输出 `inputs_embeds`, `attention_mask`, `token_type_ids` |
| M3: Frozen backbone forward | ⬜ pending | 单 batch 前向成功，hidden state shape 正确 |
| M4: OA invariant | ⬜ pending | key mask/reset hook 单元测试通过 |
| M5: Heads + losses | ⬜ pending | world/action loss 可 overfit tiny batch |
| M6: Address swap diagnostic | ⬜ pending | 可交换 addr 并记录 action/hidden 改变量 |
| M7: Stage II small run | ⬜ pending | 8 卡或单卡小规模训练日志稳定 |

## 子系统状态
| 子系统 | 状态 | 负责Agent | 最后更新 |
|--------|------|-----------|----------|
| 架构分析 | ✅ done | PaperSpec + Coordinator | 2026-06-05 |
| 训练管线分析 | ✅ done | PaperSpec + Coordinator | 2026-06-05 |
| 实验基准分析 | ✅ done | PaperSpec + Coordinator | 2026-06-05 |
| 接口契约 | ✅ done | TestReviewer + Coordinator | 2026-06-05 |
| Backbone 集成分析 | ✅ done | BackboneIntegration | 2026-06-05 |
| 感知栈 / cache | ⬜ pending | - | - |
| 序列构造 | ⬜ pending | - | - |
| OA主干 | ⬜ pending | - | - |
| 世界头+动作头 | ⬜ pending | - | - |
| 训练循环 | ⬜ pending | - | - |
| 评测管线 | ⬜ pending | - | - |

## 当前阻塞
- 🟡 Backbone 最终确认：推荐 `facebook/chameleon-7b`，待用户确认。M1 可先于确认推进（mock cache + slot vector 不依赖 backbone）。
- 🟡 PDF 文件的 Git 策略：`paper/OA-WAM.pdf` 和 `paper/OA-WAM_cn.pdf` 需移除 Git 跟踪或明确保留。

## 集成状态
- [ ] 感知栈 ↔ 序列构造 接口对齐
- [ ] 序列构造 ↔ 主干 接口对齐
- [ ] 主干 ↔ 头部 接口对齐
- [ ] 数据加载 → 训练循环 接口对齐
- [ ] 训练循环 → 评测管线 接口对齐

## Decision Log
| 日期 | 决策 | 原因 |
|------|------|------|
| 2026-06-05 | 跳过 Stage 0 backbone 预训练 | 8x4090 不适合复现 7B 预训练，且用户计划使用现有 backbone |
| 2026-06-05 | 保留 Stage I-lite | slot adapter/world head 对齐是机制验证的一部分，不等同于 VQ-GAN 重建 |
| 2026-06-05 | 先做机制级 MVE | 更适合训练代码能力，也更快暴露 OA 实现错误 |
| 2026-06-05 | 推荐 facebook/chameleon-7b 为目标 backbone | 架构与论文完全一致（32层/4096/32头），inputs_embeds 接口已确认可用，OA 机制与 D 无硬依赖 |
| 2026-06-05 | M1 可先于 backbone 确定推进 | M1 只涉及 slot vector 构造，不涉及 backbone forward；hidden_size 从配置读取 |
