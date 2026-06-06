# 训练配置

本项目默认不复现论文 Stage 0 的 7B backbone 大规模预训练。训练目标分为机制级 MVE、Stage0-mini 替代 warmup、Stage I-lite、Stage II 四层递进。

原则：
- 当前确认 `ckpt-stage0.pt` 尚未发布，因此正式采用 Stage0-mini 作为 Stage 0 替代路线。
- 若未来作者发布 `ckpt-stage0.pt`，可作为替换初始化重新评估，但不阻塞当前实现。
- Stage0-mini 不是论文 Stage 0 的等价复现，而是让公开 Chameleon-style backbone 适应 slot token、OA mask/reset 和 world/VQ 目标的工程替代。
- 报告时必须区分“机制复现”“Stage0-mini 近似复现”和“论文完整数值复现”。

## Hardware Assumption
- 物理训练服务器：8x RTX 4090 48GB
- 优先 bf16；若当前 PyTorch/4090 环境不稳定，则使用 fp16 + GradScaler
- 优先 cached perception，避免训练 loop 调 SAM/DINO/Qwen
- 优先 LoRA/QLoRA + gradient accumulation，不做 7B 全参训练

## Runtime Verification Policy
- Codex sandbox 默认只负责 CPU 侧实现验证和单元测试。
- 如果 Codex 内 `nvidia-smi` 或 `/dev/nvidia*` 不可见，不视为训练服务器无 GPU，也不视为 conda 环境失败。
- GPU 训练、multi-GPU launch、长时间 rollout 由用户在 GPU 可见的终端中执行。
- Agent 应提供可复制的训练命令、配置检查和日志诊断，但不要求在 Codex sandbox 内直接运行 GPU 训练。

## Stage 0（论文大规模预训练，不复现）
| 项 | 设置 |
|----|------|
| 论文设置 | Chameleon-7B base 热启动，约 600k steps，384xA100-80GB，VQ-CE + `L_world`，OA mask warmup |
| 本项目处理 | 不执行该规模训练；`ckpt-stage0.pt` 尚未发布，当前正式选择 Stage0-mini |
| 风险 | 若只用公开 Chameleon base，缺少 slot-aware pretraining，Stage II 可能难以达到论文成功率 |
| 验收 | 明确 checkpoint 来源：当前为 `stage0-mini` 或 `direct-chameleon`；未来若有 released Stage 0 再增列 |

## Stage0-mini（Stage 0 替代 warmup）
| 项 | 设置 |
|----|------|
| 目标 | 让公开 Chameleon-style backbone 适应 slot embedding、OA key mask/reset、world/VQ 目标，为近似实验复现提供比“直接 Stage II”更强的初始化 |
| Backbone | 默认冻结 base weights；优先训练 LoRA，而非全参更新 |
| 可训练参数 | slot adapter、`f_addr`/`f_cnt`、world head、LoRA；可选 SE(3) bias MLP |
| Loss | `L_world + 0.04 L_vq`；不使用 action loss、compose loss、role loss |
| 数据阶梯 | mock cache sanity -> LIBERO cached perception -> LIBERO + small OXE/DROID cached subset |
| 推荐规模 | tiny overfit -> 1k-5k debug steps -> 20k-50k practical run；根据 4090 显存和缓存吞吐调整 |
| 验收 | `L_world` 和 VQ loss 稳定下降；OA invariant 测试仍通过；A2 swap diagnostic 比直接 Chameleon base 有更强信号；进入 Stage II 后 action loss 更稳定 |
| 失败判据 | loss NaN/Inf、slot padding 泄漏、addr reset 后不等于 cached addr、full OA 与 no-OA 在 A2 上无差异 |

Stage0-mini 不负责生成最终策略动作；它只建立 slot-aware trunk 的可训练起点。若只使用 LIBERO 数据，它可能过拟合标准分布，因此必须保留 A1/A2 消融来判断鲁棒性是否来自 OA 结构。

## Stage I-lite（建议保留）
| 项 | 设置 |
|----|------|
| 目标 | 在 Stage0-mini 后继续对齐 slot adapter + world head；或至少 overfit tiny batch |
| Backbone | frozen base weights；保留或关闭 Stage0-mini LoRA 由实验决定 |
| 可训练参数 | slot adapter、`f_addr`/`f_cnt`、world head |
| Loss | `L_world`，可选 tiny VQ auxiliary |
| 数据 | LIBERO small cache -> larger cache；mock cache 仅用于单元和 overfit |
| 验收 | world loss 在 tiny batch 上下降，slot/padding mask 正确 |

## Stage II
| 项 | 设置 |
|----|------|
| 目标 | LoRA + action head + full loss 微调 |
| Backbone | frozen base weights + LoRA；当前从 Stage0-mini checkpoint 初始化，未来可评估 released Stage 0 checkpoint |
| LoRA target | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`，按实际 backbone 命名调整 |
| Action horizon | `H=16` |
| Action dim | `A=7` |
| Loss | `L_act + λw L_world + λv L_vq + λc L_compose + λr L_role` |
| 验收 | tiny batch action loss 可下降；A2 swap diagnostic 有响应；small LIBERO rollout 优于未做 Stage0-mini 的直接 Stage II baseline |

## 优化器与调度
初始建议，不是最终论文数值：
- Optimizer: AdamW
- Betas: `(0.9, 0.95)`
- Weight decay: `0.05`
- Grad clip: `1.0`
- LR new modules: `2e-4`
- LR LoRA: `2e-4`
- Warmup: `2k` steps 或总步数 `5%`
- Scheduler: cosine
- EMA: Stage II 可启用 `0.999`
- Global batch: 从小开始，优先稳定；8x4090 可用 grad accumulation 达到等效 batch

## Loss 权重
论文默认：
- `λw = 0.5`
- `λv = 0.04`
- `λc = 0.1`
- `λr = 0.05`
- `λc` 前 30% steps 从 0 ramp 到 0.1
- `λr` 前半训练启用，后半设为 0

## Training Gates
每个 gate 通过后再进入下一步：
1. `single_batch_forward`: 无梯度单 batch forward。
2. `single_batch_backward`: loss backward，无 NaN/Inf。
3. `tiny_overfit_world`: world loss 可下降。
4. `stage0_mini_warmup`: `L_world + 0.04 L_vq` 在 cached LIBERO small cache 上稳定下降。
5. `stage0_mini_compare`: Stage0-mini 初始化优于 direct-Chameleon 初始化（至少 world loss 或 A2 信号更好）。
6. `tiny_overfit_action`: action loss 可下降。
7. `oa_ablation_sanity`: 关闭 key mask/reset 后 A2 指标变化。
8. `small_rollout`: 少量 LIBERO episode smoke eval。

## Baselines Required For Approximate Reproduction
近似复现实验结论时必须至少保留以下对照：
- `direct_chameleon_stage2`: 不做 Stage0-mini，直接 Stage II。
- `stage0_mini_full_oa`: Stage0-mini + Stage II，开启 key mask 和 reset。
- `stage0_mini_no_key_mask`: 关闭 key mask，保留 reset。
- `stage0_mini_no_reset`: 保留 key mask，关闭 reset。
- `stage0_mini_no_oa`: 同时关闭 key mask 和 reset。

只有当 `stage0_mini_full_oa` 在 A2 swap binding 和几何扰动 smoke eval 上优于 no-OA 变体时，才认为近似复现了论文的核心实验趋势。
