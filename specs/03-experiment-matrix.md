# 实验矩阵

实验按“机制验证 -> 小规模训练 -> 论文表格”递进。早期不要直接追完整 LIBERO 分数。

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

## LIBERO
| 套件 | 指标 | 早期设置 |
|------|------|----------|
| Spatial | success rate | 先 5-10 episodes smoke |
| Object | success rate | 先 5-10 episodes smoke |
| Goal | success rate | 先 5-10 episodes smoke |
| Long | success rate | 最后再跑，长任务调试成本高 |

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

## 因果槽位干预测试
核心 A2：
1. 选取目标槽位与另一个可见 object slot。
2. 交换两者 `addr`，保持 `content/pose/text/image/proprio` 不变。
3. 比较动作块或 EE residual trajectory 是否朝交换目标偏转。
4. 记录 full OA、no key mask、no reset、mean-pool head 等变体。

## 消融实验
| 消融 | 改动 | 预期 |
|------|------|------|
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
