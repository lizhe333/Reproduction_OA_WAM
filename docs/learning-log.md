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

## 2026-06-09
### Today I implemented
- 搞懂了具体打包的数据类型中各个tensor的作用

## 2026-06-10
### Today I implemented
- 完成了M2模块，将各个数据类型打包成 sequence_batch 并可以将slot和padding区分，构建了 attention_mask_4d，修改了注意力规则。
- 关键 shape
- input_ids/token_type_ids:[B,L]
- slot_positions:[B,T,S],值是L维当中的位置
- inputs_embeds:[B,L,D]
- attention_mask_4d :[B,1,L,L]
- 关键 invariant：
    - inputs_embeds[b, slot_positions[b,t,s], :] ==        slot_embeds[b,t,s,:]
    - mask[b,0,query_pos,key_pos] 中 True 表示 query 可以读 key
    - padding slot 作为 key 时整列必须为 False
    - action 可以读 slot，但 slot 不能读 action
  1. M2a 产出 sequence layout metadata：input_ids [B,L]、token_type_ids [B,L]、
     slot_positions [B,T,S]、act_q_position [B]。

  2. M2b 产出 embedding：slot_vector [B,T,S,320]、slot_embeds [B,T,S,D]、
     inputs_embeds [B,L,D]。

  3. M2c 产出 attention_mask_4d [B,1,L,L]，True 表示 query 可以读 key。
  4. M3 backbone 核心吃 inputs_embeds 和 attention mask。