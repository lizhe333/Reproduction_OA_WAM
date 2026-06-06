# 实验矩阵

实验按“机制验证 -> Stage0-mini 替代 warmup -> 小规模训练 -> 论文表格近似复现”递进。早期不要直接追完整 LIBERO 分数；没有 released Stage 0 checkpoint 时，报告重点是同方向趋势和消融差异。

## Phase 0: Unit Diagnostics
| 实验 | 指标 | 目的 |
|------|------|------|
| shape tests | tensor shape/dtype/device | 防止接口漂移 |
| padding tests | padding slot 不被 attention/loss 使用 | 防止 mask 泄漏 |
| addr reset tests | 每层后前 32 维等于 cached addr | 验证 OA invariant |
| key mask tests | slot K 只依赖 addr slice | 验证对象可寻址路由 |

## Phase 1: MVE Diagnostics
| 实验 | 指标 | 通过标准 |
|------|------|----------|
| tiny world overfit | `L_world` | 明显下降 |
| tiny action overfit | `L_act` | 明显下降 |
| address swap | hidden/action delta, swap cosine | full OA 高于 no-OA |
| compose perturbation | perm KL, insertion drift | full OA/compose 更稳定 |

## Phase 2: Stage0-mini Warmup
| 实验 | 指标 | 通过标准 |
|------|------|----------|
| direct Chameleon baseline | `L_world`, A2 swap cosine | 作为未做 Stage0-mini 的下限 |
| Stage0-mini tiny overfit | `L_world + 0.04 L_vq` | tiny cache 上稳定下降，无 NaN/Inf |
| Stage0-mini small cache | world loss, VQ loss, padding leakage checks | LIBERO small cache 上稳定下降，OA invariant 仍通过 |
| Stage0-mini vs direct | world loss, A2 hidden/action delta | Stage0-mini 初始化优于 direct Chameleon |
| Stage0-mini A1 smoke | full OA / no key / no reset / no OA | full OA 的 swap binding 或几何 smoke 指标最高 |

## Phase 3: Stage II Approximate Reproduction
| 实验 | 指标 | 通过标准 |
|------|------|----------|
| Stage II tiny action overfit | `L_act` | tiny batch 明显下降 |
| Stage II small LIBERO | success rate, action loss | Spatial/Object 先产生非随机成功率 |
| Stage II A2 | swap cosine, trajectory delta | full OA 明显高于 no-OA |
| Stage II A1 small | Camera/Robot/Layout smoke success | full OA 在几何扰动上优于 no-OA |

## LIBERO
| 套件 | 指标 | 早期设置 |
|------|------|----------|
| Spatial | success rate | 先 5-10 episodes smoke |
| Object | success rate | 先 5-10 episodes smoke |
| Goal | success rate | 先 5-10 episodes smoke |
| Long | success rate | 最后再跑，长任务调试成本高 |

近似复现阶段先报告 Spatial/Object 小规模结果，再扩到四套件。不要在 Stage0-mini 还未证明优于 direct baseline 时跑完整 LIBERO 表格。

## SimplerEnv
| 任务 | 指标 | 优先级 |
|------|------|--------|
| Spoon on Towel | visual matching success | medium |
| Carrot on Plate | visual matching success | medium |
| Stack Cube | visual matching success | medium |
| Eggplant in Basket | visual matching success | medium |

## LIBERO-Plus
| 扰动轴 | 指标 | 优先级 |
|--------|------|--------|
| Camera | success rate | high，最能验证 OA 几何鲁棒性 |
| Robot init | success rate | high |
| Layout | success rate | high |
| Light | success rate | medium |
| Background | success rate | medium |
| Language | success rate | medium |
| Sensor noise | success rate | low，主要考验感知栈 |

近似复现阶段优先跑 Camera/Robot/Layout 的 smoke eval。目标是验证 OA 结构在几何扰动上相对 no-OA 的差异，而不是立即追论文 `83.9%` 七轴平均。

## 因果槽位干预测试
核心 A2：
1. 选取目标槽位与另一个可见 object slot。
2. 交换两者 `addr`，保持 `content/pose/text/image/proprio` 不变。
3. 比较动作块或 EE residual trajectory 是否朝交换目标偏转。
4. 记录 full OA、no key mask、no reset、mean-pool head 等变体。

## 消融实验
| 消融 | 改动 | 预期 |
|------|------|------|
| direct Chameleon | 不做 Stage0-mini | 作为训练替代路线的必要下限 |
| Stage0-mini full OA | Stage0-mini + key mask + reset | 近似复现实验的主线 |
| A1 no OA | 关闭 key mask + reset hook | camera/layout/robot 鲁棒性下降 |
| A1 no key mask | 关闭 key mask，保留 reset | swap binding 下降 |
| A1 no reset | 保留 key mask，关闭 reset | 高层 addr 污染，swap binding 下降 |
| A3 no world head | 移除 `L_world` | 几何轴和 LIBERO 可能下降 |
| A4 no compose | 移除 distractor permutation/insertion | layout 和 perm KL 变差 |

## Reporting Rule
每个实验记录：
- commit hash
- config path
- checkpoint path（不要提交权重）
- dataset/cache version
- seed
- command
- result table
- failure notes

额外要求：
- 明确 checkpoint lineage：`released-stage0`、`direct-chameleon` 或 `stage0-mini`。
- 对外描述时不得把 Stage0-mini 称作论文 Stage 0。
- 每个接近论文表格的结果必须同时给出 no-OA 或 direct baseline，否则无法证明趋势来自 OA。
