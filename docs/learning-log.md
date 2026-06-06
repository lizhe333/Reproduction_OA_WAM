# Learning Log

记录复现过程中的思考与debug记录

## 2026-06-05

### Today I implemented
- 初始化 OA-WAM 复现工作区骨架。
- 对于backbone中的使用，论文中使用的是Chamelone 7b，他们自己重新微调了，但是对于我们复现，加载预训练好的chept即可。

### I was confused by
- Stage 0 / Stage I / Stage II 的边界需要澄清。

### Invariants I learned
- OA-WAM 的核心不是普通 slot token，而是 key 只读 `addr` 且每层重置 `addr`。

### Bugs I hit
- Git commit 需要配置 `user.name` 和 `user.email`。
- Git push 前本地必须有 commit 和对应分支。

### Next drill
- 跑通数据流
- 写出 `slot_vector = [addr, content, time, role]` 的 shape 测试和构造逻辑。

## 2026-06-06
### Today I implemented
- 确认用微调的方式来处理Chameloen-7b无法识别新维度的token的问题
- 重新组织stage0-mini 来做一个adapter处理上述问题

### I was confused by
- mock-cache.py看到77行，对于slot中的一些特征和维度变化还是没有太搞懂
