# 训练配置

本项目默认不复现论文 Stage 0 的 7B backbone 预训练。训练目标分为机制级 MVE、Stage I-lite、Stage II 三层递进。

## Hardware Assumption
- 8x RTX 4090 48G
- 优先 bf16；若当前 PyTorch/4090 环境不稳定，则使用 fp16 + GradScaler
- 优先 cached perception，避免训练 loop 调 SAM/DINO/Qwen
- 优先 LoRA/QLoRA + gradient accumulation，不做 7B 全参训练

## Stage 0（跳过）
| 项 | 设置 |
|----|------|
| 目标 | 使用现有 Chameleon-style/VLM backbone |
| 训练 | 不训练 |
| 风险 | backbone hidden size、tokenizer、reserved tokens 与论文不完全一致 |
| 验收 | `inputs_embeds` forward 可跑通 |

## Stage I-lite（建议保留）
| 项 | 设置 |
|----|------|
| 目标 | 对齐 slot adapter + world head；或至少 overfit tiny batch |
| Backbone | frozen |
| 可训练参数 | slot adapter、`f_addr`/`f_content`、world head |
| Loss | `L_world`，可选 tiny VQ auxiliary |
| 数据 | mock cache -> LIBERO small cache -> larger cache |
| 验收 | world loss 在 tiny batch 上下降，slot/padding mask 正确 |

## Stage II
| 项 | 设置 |
|----|------|
| 目标 | LoRA + action head + full loss 微调 |
| Backbone | frozen base weights + LoRA |
| LoRA target | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`，按实际 backbone 命名调整 |
| Action horizon | `H=16` |
| Action dim | `A=7` |
| Loss | `L_act + λw L_world + λv L_vq + λc L_compose + λr L_role` |
| 验收 | tiny batch action loss 可下降；A2 swap diagnostic 有响应 |

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
4. `tiny_overfit_action`: action loss 可下降。
5. `oa_ablation_sanity`: 关闭 key mask/reset 后 A2 指标变化。
6. `small_rollout`: 少量 LIBERO episode smoke eval。
