# 子系统接口契约

所有 agent 改代码前必须先检查本文件。接口改动必须先改本文件，再改实现和测试。

## Naming Conventions
- Batch dict key 使用 snake_case。
- Tensor 默认 batch-first。
- mask 中 `true` 表示有效/可用，除非函数文档明确说明为 additive attention mask。
- 所有模块必须接受显式 `device` 上的 tensor，不在 forward 中偷偷 `.cuda()`。

## Dataclasses / Batch Dicts

### `RawDemoBatch`
由 data loader 输出。
| key | shape | dtype | 说明 |
|-----|-------|-------|------|
| `rgb` | `[B,T,V,3,256,256]` | float32/bfloat16 | range `[0,1]` |
| `rgb_fullres` | `[B,T,V,3,H_full,W_full]` | float32 | 原始分辨率图像，供 SAM3/DINOv3 使用 |
| `language_raw` | `list[str]` | - | 原始语言指令文本（未经 tokenize），T1 和 T2 均从此出发 |
| `language` | `list[str]` or token ids | - | 原始指令或已 tokenized |
| `proprio` | `[B,T,7]` | float32 | EE pose/gripper |
| `action_history` | `[B,T-1,7]` | float32 | 历史动作 |
| `action_target` | `[B,H,7]` | float32 | 未来动作块 |
| `episode_id` | `list[str]` | - | episode 标识，addr cache 生命周期管理依赖 |
| `frame_idx` | `[B,T]` | int64 | 帧在 episode 内的绝对索引 |
| `done` | `[B,T]` | bool | episode 终止标记 |
| `camera_intrinsics` | `[B,V,3,3]` | float32 | 相机内参，SE(3) 偏置计算必需 |
| `camera_extrinsics` | `[B,V,4,4]` | float32 | 相机外参，SE(3) 偏置计算必需 |

### `PerceptionBatch`
由 `src/perception/` 输出，可来自真实冻结感知或 mock cache。
| key | shape | dtype | 说明 |
|-----|-------|-------|------|
| `text_t1_noun_ids` | `[B,L_t1]` | int64 | T1 路径：Qwen3-VL 提取的名词短语 token ids（仅作 SAM3 prompt，也可不进入主干） |
| `text_t2_bpe_ids` | `[B,L_t2]` | int64 | T2 路径：语言指令 BPE token ids，通过 `embed_tokens` 进入主干 |
| `vq_input_ids` | `[B,T,L_vq]` | int64 | 图像 VQ ids，可先 mock |
| `state_input_ids` | `[B,T,7]` | int64 | proprio 每维 256 桶离散 token |
| `action_input_ids` | `[B,T-1,7]` | int64 | history action 每维 256 桶离散 token |
| `addr` | `[B,T,S,32]` | float32/bfloat16 | object identity address。t=0 由 f_addr 计算一次，episode 内跨帧不变（T 维为广播副本） |
| `content` | `[B,T,S,256]` | float32/bfloat16 | object content，每帧由 f_cnt 重新计算 |
| `pose` | `[B,T,S,9]` | float32/bfloat16 | object pose（3D position + 6D rotation） |
| `slot_valid_mask` | `[B,T,S]` | bool | robot + valid object 为 true（padding 为 false） |
| `slot_is_object` | `[B,T,S]` | bool | 仅 object slot 为 true（robot/padding 为 false），世界头 loss 掩码直接来源 |
| `slot_role_ids` | `[B,T,S]` | int64 | 0 padding, 1 robot, 2 object |
| `slot_time_emb` | `[B,T,S,16]` | float32 | slot_vector 的 time16 分量（帧索引正弦嵌入） |
| `slot_role_emb` | `[B,T,S,16]` | float32 | slot_vector 的 role16 分量（三类角色的可学习查表） |
| `dino_features` | `[B,T,S,d_dino]` | float32 | I-B 路径 DINOv3 逐 slot 特征（典型 d_dino=1024） |
| `sam_masks` | `[B,T,S,H_m,W_m]` | bool | I-B 路径 SAM3 逐 slot 分割掩码 |
| `object_track_ids` | `[B,T,S]` | int64 | 跨帧对象追踪 ID，addr 在 t=0 计算后按此 ID 传播 |
| `target_slot_ids` | `[B]` or `[B,T]` | int64 | 可选，用于 role loss/diagnostic |

## 感知栈 → 序列构造
`SequenceTokenizer.forward(perception_batch) -> SequenceBatch`

### `SequenceBatch`
| key | shape | dtype | 说明 |
|-----|-------|-------|------|
| `input_ids` | `[B,L]` | int64 | 包含 `⟨slot⟩` 和 `[ACT_Q]` |
| `inputs_embeds` | `[B,L,D]` | bf16/fp16/fp32 | slot scatter 后 embedding |
| `attention_mask` | backend-specific | bool/float | block causal + slot rules（见下） |
| `attention_mask_4d` | `[B,1,L,L]` | bool/float | 显式 4D mask：block causal + slot intra-frame bidirectional + slot→action unidirectional |
| `token_type_ids` | `[B,L]` | int64 | TEXT_T1/TEXT_T2/VQ/SLOT/STATE/ACTION/SPECIAL |
| `slot_positions` | `[B,T,S]` | int64 | 每个 slot 在 L 维序列的位置 |
| `act_q_position` | `[B]` | int64 | `[ACT_Q]` 位置 |
| `addr_cache` | `[B,T,S,32]` | same as embeds | detached reset source（AddrCache.retrieve 输出） |
| `slot_valid_mask` | `[B,T,S]` | bool | 透传 |
| `slot_is_object` | `[B,T,S]` | bool | 透传，世界头 loss 掩码来源 |
| `slot_vector` | `[B,T,S,320]` | bf16/fp32 | [addr32, content256, time16, role16]，slot adapter 输入 |
| `slot_embeds` | `[B,T,S,D]` | bf16/fp32 | slot adapter 输出（scatter 前） |
| `f_bos_positions` | `[B,T]` | int64 | 每帧 F_BOS token 位置 |
| `s_bos_positions` | `[B,T]` | int64 | 每帧 S_BOS token 位置 |
| `special_token_mask` | `[B,L]` | bool | 特殊 token 位置标记（compose loss 排除干扰） |

## 序列构造 → 主干
`SlotAwareTrunk.forward(sequence_batch) -> TrunkOutput`

### `TrunkOutput`
| key | shape | dtype | 说明 |
|-----|-------|-------|------|
| `last_hidden_state` | `[B,L,D]` | bf16/fp16/fp32 | backbone hidden（最终层） |
| `slot_hidden` | `[B,T,S,D]` | same | 从 `slot_positions` gather（reset 后） |
| `slot_hidden_before_reset` | `[B,T,S,D]` | same | 每层 addr reset 前的 slot hidden（诊断用，可仅最后一层） |
| `act_q_hidden` | `[B,D]` | same | 从 `act_q_position` gather |
| `optional_attentions` | backend-specific | optional | diagnostic only |

## 主干 → 世界头
`WorldHead.forward(slot_hidden, slot_valid_mask) -> WorldPred`

### `WorldPred`
| key | shape | dtype | 说明 |
|-----|-------|-------|------|
| `content_next` | `[B,T,S,256]` | float32/bfloat16 | 预测下一帧 content |
| `pose_next` | `[B,T,S,9]` | float32/bfloat16 | 预测下一帧 pose |
| `object_loss_mask` | `[B,T,S]` | bool | slot_is_object 且下一帧 target 存在为 true |
| `target_exists_mask` | `[B,T]` | bool | 该帧有有效的下一帧 target 为 true |

Loss mask:
- robot slot 不参与 world loss（`object_loss_mask` 中为 false）。
- padding slot 不参与 world loss。
- episode 越界或最后一帧 target 不存在时不参与 world loss（`target_exists_mask` 为 false）。

## 主干 → 动作头
`ActionHead.forward(act_q_hidden, action_target=None) -> ActionPred`

### `ActionPred`
| key | shape | dtype | 说明 |
|-----|-------|-------|------|
| `velocity` | `[B,H,7]` | float32/bfloat16 | flow matching velocity（训练时输出） |
| `action_sample` | `[B,H,7]` | float32/bfloat16 | 推理时 Euler sample（H=16, A=7） |
| `flow_time` | `[B]` | float32 | 训练时采样的 τ ~ U(0,1) |
| `flow_trajectory` | `[B,K,H,7]` | float32 | 推理时 K=4 步 Euler 积分中间轨迹（仅推理模式） |
| `flow_noise` | `[B,H,7]` | float32 | 训练时采样的高斯噪声 ε ~ N(0,I) |

训练/推理模式区分：
- 训练: `forward(act_q_hidden, action_target=ground_truth)` → 采样 τ, ε, 计算 velocity, 返回 `velocity` + `flow_time`
- 推理: `forward(act_q_hidden, action_target=None, num_euler_steps=4)` → 从 N(0,I) 出发 Euler 积分，返回 `action_sample` + `flow_trajectory`

## 数据加载 → 训练循环
`TrainingStep` 需要：
- `RawDemoBatch` 或已缓存的 `PerceptionBatch`
- `SequenceBatch`
- `TrunkOutput`
- `WorldPred`
- `ActionPred`
- `global_step: int`（用于 loss 权重调度）
- `max_steps: int`（用于 loss 权重调度）
- scalar losses: `loss`, `loss_act`, `loss_world`, optional `loss_vq`, `loss_compose`, `loss_role`

## AddrCache 生命周期
`AddrCache` 管理 object addr 的存储和检索。

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `store(t, addr)` | frame_idx int, addr `[B,S,32]` | - | t=0 时计算并缓存 addr（来自 f_addr 输出） |
| `retrieve(t)` | frame_idx int | addr `[B,S,32]` | 检索缓存 addr（detached） |
| `reset(mask)` | `[B]` bool | - | episode 边界清除对应 batch 的缓存 |
| `is_cached(t)` | frame_idx int | `[B]` bool | 检查是否已缓存 |

约束：
- `store` 仅在 t=0 调用
- `retrieve` 返回的 addr 必须 `.detach()`, loss 不应通过 reset path 更新 f_addr
- 每个新 episode 开始（done 为 True 的下一帧）必须调用 `reset`
- addr 在 episode 内跨帧不变，通过 object_track_ids 传播

## Compose 增强接口
`ComposeAugmentation.forward(perception_batch, global_step, max_steps) -> perception_batch_augmented`

### 增强类型
1. **干扰物置换**: 随机置换非目标/非参照 object slot 的顺序（在 S 维上），同步置换成对几何 G 和掩码
2. **干扰物插入**: 从 batch 中另一样本随机采样干扰物 slot，嫁接到当前样本的空 padding slot

### Compose Loss
`compose_loss(slot_hidden_orig, slot_hidden_aug, slot_assignments, action_orig, action_aug) -> scalar`
- KL(α_orig.detach() || α_aug) + ||A_orig.detach() - A_aug||²
- η_compose 在前 30% steps 从 0 线性 ramp 到 0.1

## Role Loss 接口
`RoleAttention.forward(slot_hidden) -> slot_role_logits  # [B,T,S,num_roles]`

`role_loss(slot_role_logits, target_slot_ids, ref_slot_ids) -> scalar`
- KL(α_0, onehot(target)) + KL(α_1, onehot(ref)), 权重 0.05
- 仅在训练前半段（前 50k 步）启用，之后权重置零
- target_slot_ids 和 ref_slot_ids 来自 Path T1 弱标签或数据集

## SE(3) 相对几何偏置
`SE3GeometryBias.compute(pose, camera_intrinsics, camera_extrinsics) -> relative_pose_features [B,T,S,S,D_se3]`
- 输入: pose `[B,T,S,9]`, intrinsics `[B,V,3,3]`, extrinsics `[B,V,4,4]`
- 输出: slot 对之间的相对 SE(3) 变换特征（R³ 平移 + R³ 轴角，6 个标量）
- 注入点: 作为 slot↔slot attention 对的 additive bias（小型 MLP φ_h 每个 head 产生一个偏置）

## Required Tests
- `test_slot_vector_shape`
- `test_padding_slots_do_not_contribute_to_loss`
- `test_padding_slots_receive_zero_attention`
- `test_sequence_slot_positions_match_token_type`
- `test_slot_permutation_equivariance`
- `test_addr_temporal_consistency`
- `test_oa_key_mask_only_reads_addr`
- `test_key_mask_only_depends_on_addr_slice`
- `test_addr_reset_restores_cached_addr`
- `test_addr_reset_applied_every_layer`
- `test_addr_cache_reset_on_episode_boundary`
- `test_slot_to_action_unidirectional_attention`
- `test_action_head_flow_shapes`
- `test_flow_inference_euler_steps`
- `test_address_swap_changes_diagnostic_signal`
- `test_world_loss_excludes_robot_and_padding`
- `test_compose_stability_on_unperturbed_slots`
- `test_role_loss_alpha_schedule`
