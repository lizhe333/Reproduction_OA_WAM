# 项目仪表盘

## North Star
先完成机制级 MVE，再用 Stage0-mini 替代不可承受的论文 Stage 0，最后扩展到近似实验复现。MVE 成功标准不是 LIBERO 高分，而是：
- frozen backbone 能通过 `inputs_embeds` 完成 forward；
- slot adapter + sequence construction shape 正确；
- OA key mask 和 addr reset hook 的 invariant 有单元测试覆盖；
- address swap diagnostic 能观测到目标绑定变化。

近似实验复现的当前路线已确定为 Stage0-mini（`ckpt-stage0.pt` 尚未发布）。成功标准不是完全达到论文表格数值，而是：
- 标准 LIBERO small rollout 可稳定执行并产生非随机成功率；
- Stage0-mini 后 world/action loss 比直接 Stage II 更稳定；
- A1 中 full OA 相对 no-key-mask/no-reset 在 A2 swap binding 和几何扰动上出现可测优势；
- LIBERO-Plus Camera/Robot/Layout 至少呈现与论文同方向的 OA 鲁棒性趋势。

## Milestones
| 里程碑 | 状态 | 验收标准 |
|--------|------|----------|
| M0: Paper spec freeze | ✅ done | `specs/01-architecture.md` 和 `04-interface-contracts.md` 已填实，三个 Agent 审查完成 |
| M0.5: Stage0-mini reproduction plan | ✅ done | 明确跳过论文大规模 Stage 0 后的替代训练、评估门槛和 agent 编排 |
| M1: Mock cache + slot vector | ✅ done | 实现和测试已通过；人工 review 已完成核心数据流 |
| M2: Sequence construction | ✅ done | 已提交 `0995c59 complete M2`；`inputs_embeds`, `attention_mask_4d`, `token_type_ids` 输出和 CPU shape/invariant 测试通过 |
| M3: Frozen backbone forward | ✅ done | 已实现 `SlotAwareTrunk` wrapper：传入 `inputs_embeds`/mask、frozen backbone forward、gather `slot_hidden`/`act_q_hidden`；fake backbone、tiny Chameleon、M2→tiny Chameleon CPU smoke、真实 Chameleon-7B 单 batch CPU smoke 均通过 |
| M4: OA invariant | ⬜ pending | key mask/reset hook 单元测试通过 |
| M5: Heads + losses | ⬜ pending | world/action loss 可 overfit tiny batch |
| M6: Address swap diagnostic | ⬜ pending | 可交换 addr 并记录 action/hidden 改变量 |
| M7: Stage0-mini small run | ⬜ pending | cached LIBERO small cache 上 `L_world + 0.04 L_vq` 稳定下降，OA invariant 仍通过 |
| M8: Stage II small run | ⬜ pending | 8 卡或单卡小规模训练日志稳定 |

## 子系统状态
| 子系统 | 状态 | 负责Agent | 最后更新 |
|--------|------|-----------|----------|
| 架构分析 | ✅ done | PaperSpec + Coordinator | 2026-06-05 |
| 训练管线分析 | ✅ done | PaperSpec + Coordinator | 2026-06-05 |
| 实验基准分析 | ✅ done | PaperSpec + Coordinator | 2026-06-05 |
| 接口契约 | ✅ done | TestReviewer + Coordinator | 2026-06-05 |
| Backbone 集成分析 | ✅ done | BackboneIntegration | 2026-06-05 |
| Stage0-mini 复现方案 | ✅ done | Coordinator + LearningCoach | 2026-06-06 |
| 感知栈 / cache | 🟡 partial | PerceptionCache | 2026-06-06 |
| 序列构造 | ✅ done | SequenceTokenizer | 2026-06-10 |
| OA主干 | ✅ M3 done | BackboneIntegration | 2026-06-12 |
| 世界头+动作头 | ⬜ pending | - | - |
| 训练循环 | ⬜ pending | - | - |
| 评测管线 | ⬜ pending | - | - |

## 当前阻塞
- 🟡 Backbone 具体实现确认：Stage0-mini 已确定为当前路线，默认从 `facebook/chameleon-7b` 或等价 Chameleon-style base 热启动；M3 fake/tiny/真实 7B wrapper smoke 已通过，并明确 Chameleon 需要 additive 4D mask（允许为 0，禁止为大负数）；后续仍需确认 reserved IDs、`position_ids`、LoRA target 和 VQ/lm_head 兼容性。
- 🟡 Stage0-mini 数据策略：需确定先用 LIBERO-only cache，还是同时准备 OXE/DROID 子集 cache。建议先 LIBERO-only，机制和训练稳定后再扩。

## Runtime Policy
- 物理训练服务器由用户确认为 8x RTX 4090 48GB。
- Codex sandbox 侧默认只跑 CPU 单元测试和 smoke checks；GPU 不可见时不作为环境失败处理。
- Stage0-mini、Stage II 和 rollout 等 GPU/长时间任务由用户在 GPU 可见终端启动，Agent 提供命令、配置和日志诊断。

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
| 2026-06-06 | 采用 Stage0-mini 作为论文 Stage 0 替代路线 | 用户目标是近似复现实验结果；384xA100 级 Stage 0 不可行，但直接跳过 Stage 0 只适合机制验证 |
| 2026-06-06 | Stage0-mini 的验收重点放在趋势而非论文绝对分数 | 没有 Stage 0 checkpoint 时不能声称严格复现；应验证 full OA 相对消融在 swap binding 和几何扰动上同方向改善 |
| 2026-06-06 | 正式选择 Stage0-mini 当前路线 | `ckpt-stage0.pt` 尚未发布；released checkpoint 仅作为未来可选替换，不阻塞当前实现 |
| 2026-06-06 | M1 先实现 mock perception 和 slot vector，不接 LIBERO | M1-M6 的目标是 shape/invariant；真实 LIBERO cache 等 sequence/trunk/head 基础通过后再接 |
| 2026-06-06 | Codex 侧验证使用 CPU，真实训练由用户在 8x4090 服务器上启动 | 当前 Codex sandbox 不保证挂载 `/dev/nvidia*`；CPU 单测足以覆盖 M1-M6 的 shape/invariant，GPU 训练另行执行 |
| 2026-06-06 | `paper/` 加入 `.gitignore` | 论文 PDF/source 作为本地参考材料，不应继续作为实现产物纳入后续提交 |
| 2026-06-07 | Guided Review 改为 `learning-coach` 逐段讲解并记录 Agent memory | 用户希望每段代码解释后先记录问题和解释，再等待确认后继续；该能力写入 `.codex/agents/learning-coach.toml`，记录写入 `docs/guided-review-memory.md`，不写入用户专属 `docs/learning-log.md` |
| 2026-06-07 | 根据 M1 guided review 增加学习画像驱动的 agent 协作 | 用户在 shape/mask/embedding/Python 容器概念上需要具体例子和逐段 explain-back；新增 `docs/developer-learning-profile.md` 并让 learning-coach/coordinator/sequence-tokenizer/test-reviewer 读取或遵循 |
| 2026-06-08 | M2a 先采用 test-first sequence layout scaffold | 序列构造最容易出现 L 维位置、token type、slot 数量和 ACT_Q 索引错误；先用 toy layout 固化 shape/invariant，再进入实现 |
| 2026-06-09 | M2b 拆为 embedding/scatter helper，scatter 采用 Human-first 实现 | `inputs_embeds` 和 slot scatter 是核心张量逻辑；先固化接口与 invariant 测试，再由用户亲手补关键索引写入 |
| 2026-06-10 | M2c 先采用 boolean 4D attention mask 测试脚手架 | attention mask 是核心张量逻辑；先固化 `[B,1,L,L]` query/key 语义、causal、同帧 slot 双向、padding key 排除和 action/slot 单向规则，再由用户亲手补实现 |
| 2026-06-10 | M2c mask 实现采用分层覆盖顺序 | 先构造 base causal mask，再打开同帧 valid slot 双向可见，最后关闭 padding slot key 列；action/slot 单向性由当前 layout 顺序和 causal 规则覆盖 |
| 2026-06-10 | M3 先做 frozen forward preflight，不立即下载或微调真实 7B | tiny Chameleon 随机配置足以确认 `inputs_embeds`、attention mask 形状和 hidden state 输出契约；真实 `facebook/chameleon-7b` 在 wrapper smoke 通过后接入，LoRA 留到 Stage0-mini/Stage II |
| 2026-06-10 | handoff 统一写入日期目录 | 使用 `handoffs/<YYYYMMDD>/<YYYYMMDD>-<agent>-<task>.md`，避免根目录堆积并便于按天回溯 |
| 2026-06-10 | M3 wrapper 先用 fake backbone 固化接口和 gather invariant | fake backbone 能精确验证 `inputs_embeds` 而非 `input_ids`、mask adapter hook、`last_hidden_state [B,L,D]` 和 slot/action gather；真实 Chameleon 接入留作下一步 smoke |
| 2026-06-10 | Chameleon mask adapter 使用 additive 4D mask | HF Chameleon 4D mask 会原样进入 attention 并加到 logits 上；项目 bool `True=allow` 必须转换为允许位置 `0`、禁止位置 `torch.finfo(dtype).min` |
| 2026-06-10 | M3 tiny Chameleon smoke 通过后再接真实 7B | tiny random Chameleon 已验证 `SlotAwareTrunk` 和 M2 产物可以完成 CPU forward；真实 7B 仍需 license/权重/GPU 环境，作为独立 smoke 处理 |
| 2026-06-12 | 修复 `oa-wam` 环境 Hugging Face 依赖 metadata 污染 | `huggingface_hub-1.19.0.dist-info` 残留导致 transformers 误判版本；升级到 `huggingface-hub==0.36.2` 并移走残留 metadata 后，M3 Chameleon tests 和全量测试恢复通过 |
| 2026-06-12 | M3 真实 Chameleon-7B 单 batch smoke 通过 | 使用本地 HF cache 权重、`inputs_embeds`、Chameleon additive 4D mask 和 `SlotAwareTrunk` 完成 CPU forward，输出 `[1,8,4096]`/`[1,1,2,4096]`/`[1,4096]` 且 finite |
