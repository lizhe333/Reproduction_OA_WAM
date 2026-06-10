# 全局架构

本文档把 OA-WAM 论文描述固化为可执行的软件架构。实现时允许用 mock perception 和现有 backbone 先跑通机制，但接口 shape 不应频繁漂移。

## Backbone 配置（参考 Chameleon-7B/Lumina-mGPT-7B-768）

实现从 backbone config 读取以下参数；OA-WAM 论文参考值：
| 参数 | 值 | 说明 |
|------|-----|------|
| `num_hidden_layers` | 32 | Transformer 层数 |
| `hidden_size` | 4096 | 隐藏维度 D |
| `num_attention_heads` | 32 | 注意力头数，无 GQA |
| `intermediate_size` | 11008 | FFN 大小（SwiGLU） |
| `vocab_size` | 65536 | 词表大小 |
| `max_position_embeddings` | 4096 | 最大位置编码 |
| `rope_theta` | 10000 | RoPE 基频 |

OA 机制（key mask 仅读前 32 维、addr reset）不依赖特定 hidden_size，可与不同 backbone 适配。slot adapter 和预测头需从 backbone config 读取 D 以匹配维度。

### 本地 Backbone 路径与加载

| 项目 | 值 |
|------|-----|
| HF 模型名 | `facebook/chameleon-7b` |
| 本地路径 | `/data/cache/models/chameleon-7b` |
| 磁盘占用 | ~14 GB（3 个 safetensors 分片） |
| HF 缓存路径 | `/data/cache/huggingface/hub/models--facebook--chameleon-7b/` |
| Transformers 源码 | `/home/lizhe/.local/lib/python3.10/site-packages/transformers/models/chameleon/modeling_chameleon.py` |

关键源码位置（行号可能随 transformers 版本漂移）：

| 类/函数 | 文件 | 大致行号 |
|---------|------|----------|
| `ChameleonModel.forward` | `modeling_chameleon.py` | L915 |
| `ChameleonForConditionalGeneration.forward` | `modeling_chameleon.py` | L1055 |
| `ChameleonAttention.forward` | `modeling_chameleon.py` | L312 |
| `eager_attention_forward`（mask additive 逻辑） | `modeling_chameleon.py` | L217 |
| `ChameleonDecoderLayer.forward` | `modeling_chameleon.py` | L380 |
| `create_causal_mask` | `transformers/masking_utils.py` | L727 |

验证下载完整性：
```bash
du -sh /data/cache/models/chameleon-7b          # 预期 ~14G
ls /data/cache/models/chameleon-7b/*.safetensors | wc -l  # 预期 3
```

Python 加载示例：
```python
import torch
from transformers import ChameleonForConditionalGeneration

model = ChameleonForConditionalGeneration.from_pretrained(
    "/data/cache/models/chameleon-7b",   # 或 "facebook/chameleon-7b"
    torch_dtype=torch.float16,           # 或 bfloat16
    device_map="auto",
)

# inputs_embeds forward（M3 关键接口）:
# model.model.forward(inputs_embeds=embeds, attention_mask=mask_4d)
#   → BaseModelOutputWithPast(last_hidden_state=[B,L,4096], ...)
#
# mask 格式: additive 4D [B,1,L,L], 0=allow, -inf=block
# 源码位置: eager_attention_forward L231-233
#   attn_weights = attn_weights + causal_mask
```

下载方式（如需重新下载）：
```bash
# 注意: gated model 需要先在 https://huggingface.co/facebook/chameleon-7b 同意许可
# 如果 HF_ENDPOINT 指向镜像站，gated model 鉴权可能失败，需临时切回官方站:
HF_ENDPOINT=https://huggingface.co hf download facebook/chameleon-7b \
  --local-dir /data/cache/models/chameleon-7b
```

## 系统概览
数据流：
1. `Dataset` 读取 RGB、proprio、历史动作、语言、未来动作块。
2. `PerceptionCache` 产生冻结感知缓存：文本短语、VQ image ids、object slots、pose、mask。
3. `SequenceTokenizer` 构造统一 token 序列，并把 slot embedding scatter 到 `<slot>` 占位位置。
4. `SlotAwareTrunk` 包装现有 backbone：非 slot token 走标准注意力，slot token 使用 OA key mask 和 addr reset。
5. `WorldHead` 从 slot hidden 预测下一帧 content/pose。
6. `ActionHead` 从 `[ACT_Q]` hidden 用 flow matching 解码 16 步动作块。
7. `Losses` 汇总 action/world/VQ/compose/role loss。
8. `EvalAblation` 做 rollout、A1、A2、A3、A4。

## 模块清单
| 模块 | 目录 | 说明 |
|------|------|------|
| Perception cache | `src/perception/` | 冻结感知输出或 mock cache |
| Tokenizer / sequence | `src/tokenizer/` | slot vector、special tokens、masked scatter、attention mask |
| Trunk surgery | `src/trunk/` | backbone adapter、OA key mask、addr reset hook |
| Prediction heads | `src/heads/` | world head、flow matching action head、optional role attention |
| Training | `src/training/` | Stage I-lite、Stage II、loss、optim config |
| Eval | `src/eval/` | LIBERO smoke eval、address swap、ablation |
| Tests | `tests/` | shape/invariant/unit tests |

## Tensor Shape 表
默认符号：
- `B`: batch size
- `T`: history frames，默认 4
- `N`: object slot capacity，默认 16
- `S = N + 1`: slot count including robot slot
- `H`: action horizon，默认 16
- `D`: backbone hidden size，论文为 4096；实现从 backbone config 读取
- `A`: action dim，默认 7

| 张量 | Shape | 说明 |
|------|-------|------|
| `rgb` | `[B, T, V, 3, 256, 256]` | `V` 为视角数，可先用 1 |
| `proprio` | `[B, T, 7]` | EE pose/gripper |
| `action_history` | `[B, T-1, 7]` | 输入历史动作 |
| `action_target` | `[B, H, 7]` | 监督动作块 |
| `addr` | `[B, T, S, 32]` | episode 内对象身份地址；同一对象跨时间应一致 |
| `content` | `[B, T, S, 256]` | 时变对象内容 |
| `pose` | `[B, T, S, 9]` | 3D position + 6D rotation |
| `slot_role` | `[B, T, S]` | robot/object/padding |
| `slot_valid_mask` | `[B, T, S]` | padding slot 为 false |
| `slot_vector` | `[B, T, S, 320]` | `[addr32, content256, time16, role16]` |
| `slot_embeds` | `[B, T, S, D]` | slot adapter 输出 |
| `input_ids` | `[B, L]` | 文本/VQ/state/action/special token ids |
| `inputs_embeds` | `[B, L, D]` | slot scatter 后送入 backbone |
| `token_type_ids` | `[B, L]` | TEXT_T1(Qwen3-VL名词)/TEXT_T2(BPE文本)/VQ/SLOT/STATE/ACTION/SPECIAL |
| `attention_mask` | `[B, 1, L, L]` 或 backend 所需格式 | block causal + slot rules |
| `hidden_states` | `[B, L, D]` | backbone 输出 |
| `slot_hidden` | `[B, T, S, D]` | 从 hidden 中 gather slot positions |
| `act_q_hidden` | `[B, D]` | `[ACT_Q]` hidden |
| `world_pred_content` | `[B, T, S, 256]` | 下一帧 content 预测 |
| `world_pred_pose` | `[B, T, S, 9]` | 下一帧 pose 预测 |
| `action_pred` | `[B, H, 7]` | 推理动作块 |

## 特殊 Token ID
复用 Chameleon 保留词表槽位，避免调整 `embed_tokens` 或 `lm_head` 大小：
| Token | 保留 ID | 作用 |
|-------|---------|------|
| `⟨slot⟩` | `<reserved16500>` | slot 嵌入的 `masked_scatter` 占位符 |
| `S_BOS` | `<reserved16501>` | 帧内 slot 块起始标记 |
| `S_EOS` | `<reserved16502>` | 帧内 slot 块结束标记 |
| `F_BOS` | `<reserved16503>` | 时间帧组起始标记 |
| `F_EOS` | `<reserved16504>` | 时间帧组结束标记 |
| `[ACT_Q]` | `<reserved16505>` | 序列末端可学习动作查询 |

## Slot Adapter (f_φ)
- 结构: 2-layer MLP — `LayerNorm(320) → Linear(320→4096) → GELU → Linear(4096→4096)`
- 参数量: ~18.0M
- 输入: `slot_vector` [B, T, S, 320]
- 输出: `slot_embeds` [B, T, S, D]（D=4096 或其他 backbone hidden_size）
- 槽位注入: 通过 LLaVA 风格 `masked_scatter` 将 slot_embeds 写入 input_ids 中 `⟨slot⟩` 占位位置

## 地址/内容生成器
- **f_addr**: ℝ⁵¹² → ℝ¹²⁸ → ℝ³² — 输入 `[language_label_embed(256), initial_DINOv3_feature(256)]`，t=0 计算一次，episode 内不变
- **f_cnt**: ℝ⁵⁴⁰ → ℝ⁵¹² → ℝ²⁵⁶ — 输入 `[DINOv3_feat(256), pose(9), lang_embed(256), shape_descriptor(15)]`，每帧重新计算
- 总参数量: ~0.4M

## 预测头架构
### World Head (h_ψ) — ~5.4M
- Content branch: `Linear(D→1024) → GELU → Linear(1024→256)`
- Pose branch: `Linear(D→256) → GELU → Linear(256→9)`
- 输入: slot_hidden [B,T,S,D]，reshape 为 [B*T*S, D] 后分别通过两个分支
- 注意: robot slot 和 padding slot 不参与 world loss，但 forward 时仍输出预测

### Action Head (h_ξ, Flow Matching MLP) — ~22M
- 条件向量: `c = W_c · H_[ACT_Q]` ∈ ℝ¹⁰²⁴
- 结构: 8 个残差块（每个 `LN → Linear(1024→1024) → GELU → Linear(1024→1024)` + 残差连接）
- 输入: c、展平噪声动作块 A_t^τ、flow time τ 的正弦嵌入
- 输出: velocity v_ξ ∈ ℝ^(H×7)，H=16
- 推理: 4 步前向 Euler 积分，从 N(0,I) 出发
- 可选角色注意力: ~1.2M，产生逐步软槽位分配 α ∈ ℝ^(num_roles × (N+1))

### 辅助 VQ Head
- 复用 Stage 0 预训练 `lm_head`（65536×4096），不新增参数
- 预测下一帧 VQ token，L_vq 用加权交叉熵（图像 VQ 词表范围 ID 3-8194 权重 0.04）

## Loss 配置
### 总损失
```
L_total = L_act + 0.5·L_world + 0.04·L_vq + η_c·L_compose + 0.05·L_role
```
### 系数调度
| 系数 | 调度规则 |
|------|----------|
| η_compose | 前 30% 训练步数中线性从 0 增至 0.1，之后保持 0.1 |
| L_role | 仅在前 50k 步计算，之后权重置零 |

### 排除规则
- robot slot 不参与 L_world
- padding slot 不参与 L_world、L_compose、L_role
- episode 边界越界帧不参与 L_world（target 不存在）

## Attention Mask 规则
- **帧间 (inter-frame)**: block causal，帧 G_t 可关注 G_0,...,G_t，不可关注 G_{t+1},...
- **帧内 slot↔slot**: bidirectional，同一帧内所有 slot 互相可见（置换等变性所需）
- **slot/VQ → action**: 单向，action token 可关注 slot/VQ，反向不可（防止世界侧被动作污染）
- **[ACT_Q]**: 可关注所有位置
- **padding slot**: 被排除为被关注对象（mask 为 -∞），且自身不产生输出

## 训练阶段规格
### Stage 0（论文预训练 — 本仓库不复现）
- Full 7B 主干从 Chameleon-7B base 热启动，在 2.5T tokens 上重训
- OA 约束从 step 5k 开始生效（前 5k 步线性退火）
- ~600k steps, 384×A100-80GB
- `ckpt-stage0.pt` 尚未发布；本仓库当前选择 Stage0-mini，若未来 checkpoint 发布，可作为替换初始化重新评估

### Stage0-mini（本仓库替代 warmup）
- 目的: 在公开 Chameleon-style backbone 上建立 slot-aware 初始化，使其适应 slot embedding、OA key mask/reset、world/VQ 目标
- Backbone: 冻结 base weights，优先训练 LoRA；不做 7B 全参大规模预训练
- 可训练: slot adapter、`f_addr`/`f_cnt`、world head、LoRA、可选 SE(3) bias MLP
- Loss: `L_world + 0.04·L_vq`，不使用 action/compose/role loss
- 数据: mock sanity → LIBERO cached perception → 可选 OXE/DROID 小子集 cache
- 验收: world/VQ loss 稳定下降，OA invariant 仍通过，A2 swap diagnostic 或几何 smoke eval 优于 direct-Chameleon/no-OA baseline
- 报告约束: Stage0-mini 是当前正式工程替代路线，不等价于论文 Stage 0；实验结果必须同时报告 direct baseline 或 no-OA baseline

### Stage I（L_world only — 本仓库实现 Stage I-lite）
- 冻结 released Stage 0 或 Stage0-mini 初始化后的主干，仅训练 slot adapter + world head（~23.8M）
- 仅使用 L_world 损失
- 50k steps（参考），8×A100
- 目的: slot adapter / world head 对齐验证

### Stage II（Full loss + LoRA — 本仓库主要训练阶段）
- 冻结主干 base weights，挂载 LoRA（r=32, α=64, dropout=0.05）
- LoRA target: `{q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj}`
- 训练所有 heads + LoRA，使用完整 L(θ)
- EMA: 0.999, 梯度裁剪: 1.0
- 100k steps（参考），8×A100
- 可训练参数: ~127M（80M LoRA + 47M heads/adapter）

## OA Invariants
- **Key mask**: slot token 的 key projection 输入被 `mask_≤32` 限制为前 32 维（addr 分量）。非 slot token 使用标准注意力路径，不受影响。OA 掩码本身无参数（不可学习的按维度索引选择）。
- **Addr reset**: 每个 Transformer 层之后，在 slot 类型位置执行 `hidden[:, :32] = addr_k.detach()`，覆盖前 32 个输出坐标。addr cache 在 t=0 由 f_addr 计算一次，episode 内不变。
- **Gradient isolation**: `addr` cache 位于 autograd 图之外；任务 loss 不应通过 reset path 反向传播到 f_addr。槽位适配器仅通过输入层 slot_embeds 接收梯度。
- **Padding mask**: padding slot (role=padding) 不可被 attention 读取，不参与 world loss、compose loss、role loss。
- **Permutation equivariance**: 同一帧内所有 object slot 共享相同 RoPE 位置索引，确保 slot 顺序不影响目标绑定。帧内置换等变性是 OA 架构的结构属性。
- **Robot exclusion**: robot slot 不参与 world loss（其未来由正在生成的动作决定）。

## MVE 数据流
MVE 允许先绕开重型 perception：
1. 构造 mock `PerceptionBatch`，随机但可重复地生成 `addr/content/pose/mask`。
2. 用 tiny batch 验证 sequence construction。
3. 接入现有 backbone 的 `inputs_embeds` forward。
4. 注入 OA surgery，并用测试验证 key mask/reset。
5. 接 world/action heads，overfit tiny batch。
6. 做 address swap diagnostic。
