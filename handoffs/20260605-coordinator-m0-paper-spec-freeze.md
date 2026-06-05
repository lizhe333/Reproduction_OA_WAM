# Handoff: M0 Paper Spec Freeze

## Done
- PaperSpec Agent 完整审查了 `specs/01-architecture.md` 与论文的一致性：零直接冲突，发现 10+ 高优先级缺失项
- BackboneIntegration Agent 确认了 `inputs_embeds` 可行性、OA 机制对 hidden_size 无硬依赖，推荐 `facebook/chameleon-7b`
- TestReviewer Agent 审查了 `specs/04-interface-contracts.md`：发现 7 处矛盾、8 个隐含跨阶段依赖、11 个新测试需求
- 三个 Agent 的发现已合并并应用于规格文件更新

## Artifacts
- `specs/01-architecture.md` — 新增了 Backbone 配置表、特殊 Token ID 表、Slot Adapter 架构、f_addr/f_cnt 生成器、World/Action Head 架构、Loss 配置（含系数调度）、Attention Mask 规则、训练阶段规格（Stage 0/I/II）、精确化 OA Invariants
- `specs/04-interface-contracts.md` — 新增了 RawDemoBatch 扩展字段、PerceptionBatch T1/T2 分离+slot 分量、SequenceBatch 完整字段（含 attention_mask_4d）、TrunkOutput/WorldPred/ActionPred 诊断字段、AddrCache 生命周期接口、Compose 增强接口、Role Loss 接口、SE(3) 几何偏置接口、训练/推理模式区分、18 项必需测试
- `handoffs/20260605-BackboneIntegration-backbone-analysis.md` — BackboneIntegration 子 agent 的详细 handoff

## Decisions
| 日期 | 决策 | 原因 |
|------|------|------|
| 2026-06-05 | 推荐 `facebook/chameleon-7b` 作为目标 backbone | 架构完全匹配论文（32层/4096/32头/无GQA），HuggingFace 原生支持 `inputs_embeds`，权重 ~14GB bf16 可在 8x4090 运行 |
| 2026-06-05 | M1 可在 backbone 未最终确定前推进 | M1 只做 slot vector 拼接和 mock perception，不涉及 backbone forward；只需将 hidden_size 设为可配置参数 |
| 2026-06-05 | MVE 阶段可使用 mock backbone | 用一个简单的 `nn.Linear(D,D)` 或恒等映射代替真实 backbone，先跑通数据流和 shape |
| 2026-06-05 | 接口规格做大幅扩展但不阻塞实现 | 规格新增了大量字段和接口；M1 实现时可先用 mock 值填充，不需要全部字段就位 |
| 2026-06-05 | 保持 slot_role_ids 的 0/1/2 语义为 padding/robot/object | 统一各 Agent 的理解，避免 robot=0 vs padding=0 的歧义 |

## Tests
- 无新测试在本轮执行（M0 为规划阶段）
- `specs/04-interface-contracts.md` 中 Required Tests 从 7 项扩展到 18 项，覆盖了所有关键不变量

## Blockers
- ⬜ **Backbone 最终确定**：需用户确认使用 `facebook/chameleon-7b` 还是其他 VLM backbone
- ⬜ **PDF 文件的 Git 策略**：`paper/OA-WAM.pdf` 和 `paper/OA-WAM_cn.pdf` 已在 .gitignore，但已跟踪的版本需移除
- ⬜ **硬件环境验证**：未在 8x RTX 4090 环境上实际测试 bf16 支持和显存容量

## Next
1. **用户确认 backbone 选择** → 下载或准备 Chameleon-7B 权重
2. **进入 M1: Mock Cache + Slot Vector**
   - LearningCoach → 教 slot_vector = [addr32, content256, time16, role16] 的概念
   - 用户亲手实现 slot_vector 构造 + shape 测试
   - SequenceTokenizer Agent → 规划 M1 最小实现
   - TestReviewer → 审查实现
3. **可选**：先清理 Git 中已跟踪的 PDF 文件
