# Developer Learning Profile

Agent-maintained profile derived from guided review sessions. This file is used to tune agent behavior for learning efficiency; it is not a performance evaluation and should be updated as the developer's understanding changes.

`docs/learning-log.md` remains user-maintained only.

## 2026-06-07 Snapshot

### Current Strengths
- Can reason through slot role semantics once terms are grounded in examples.
- Correctly identified that `action_input_ids` uses `T-1` because actions connect adjacent frames.
- Correctly identified `dim=-1` as concatenation along the final feature dimension.
- Correctly recognized broadcasting as the mechanism behind expanding `[B,T,S,1,1]` over SAM mask height/width.
- Uses explain-back effectively: misunderstandings are surfaced quickly and corrected.
- Correctly explained the M2a distinction that `input_ids` help produce embeddings, `token_type_ids` support later structural logic such as attention-mask construction, and `inputs_embeds` are the actual vectors consumed by transformer attention.

### Active Gaps To Scaffold
- Python container vocabulary: `dict`, `Mapping`, `.get`, `tuple(...)`, generator expressions, and `_` as an unused loop variable need concrete examples.
- Tensor shape accounting: off-by-one risks around `object_slots` vs total `S=N+1` need repeated checks.
- Embedding concepts: distinguish boolean masks, integer token IDs, category/type IDs, continuous feature vectors, lookup tables, and learned embeddings.
- Sequence layout concepts: continue reinforcing that paper diagrams show abstract concatenated tokens, while implementation uses helper tensors such as `input_ids`, `token_type_ids`, `slot_positions`, and `act_q_position` to construct and validate `inputs_embeds`.
- Broadcasting: understands the word, but should keep practicing exact before/after shapes.
- Spec provenance: wants to know whether implementation details come from standard formulas, project contracts, or paper evidence.

### Recent M2a Questions And Clarifications
- Asked what `input_ids` does. Clarified that `input_ids` is a `[B,L]` integer token-id sequence used for embedding lookup for discrete tokens; slot positions use a `<slot>` placeholder and are later replaced by continuous `slot_embeds`.
- Asked what `token_type_ids` does. Clarified that `token_type_ids` is a `[B,L]` category/type label tensor for downstream structural logic, checks, masks, scatter/gather validation, and loss exclusion; it is not concatenated with `input_ids`.
- Asked why the paper figure does not show `input_ids` and `token_type_ids`. Clarified that the paper describes abstract concatenated tokens, while the implementation represents that sequence through multiple tensors: `input_ids` for token lookup, `token_type_ids` for structure labels, `slot_positions`/`act_q_position` for index bookkeeping, and `inputs_embeds` as the actual transformer input.
- Asked whether `input_ids` and `token_type_ids` are concatenated and attended over. Clarified that they are not concatenated together for transformer attention; `inputs_embeds [B,L,D]` and `attention_mask` are the backbone-facing tensors.
- User explain-back: "`input_ids`用于得到输入指令的embedding，`token_type_ids`用于得到token中不同位置的类型构造attention mask，`inputs_embeds`才是真正进入transformer中进行注意力计算的内容." Correction to preserve: `input_ids` covers all discrete tokens, not only the language instruction tokens.
- User confirmed understanding of the three-way distinction among `input_ids`, `token_type_ids`, and `inputs_embeds`; next M2a coaching should build on this by explaining `slot_positions`, `act_q_position`, and the scatter/gather flow.

### Preferred Coaching Pattern
1. Explain one small code section at a time.
2. Start with plain-language purpose before line-by-line details.
3. Include concrete toy values such as `B=1,T=4,object_slots=4`.
4. For every tensor op, state before/after shape.
5. Separate category IDs, masks, and continuous embeddings explicitly.
6. Ask one explain-back question, then wait.
7. Record questions and confirmations in `docs/guided-review-memory.md`.

### Agent Routing Implications
- Use `learning-coach` automatically for requests like "给我讲解一下 xxx 代码", "解释 xxx", "继续 review", or shape/mask/broadcasting questions.
- Before core tensor implementation, `learning-coach` should provide a micro-drill; implementation agents should leave key lines as TODOs when the module is marked Human-first.
- `test-reviewer` should report not only correctness risks but also which missing tests would catch the developer's current likely mistakes, especially off-by-one, mask leakage, dtype, and broadcasting errors.
- `Coordinator` should not close a learning gate until the guided review memory has no unresolved core-shape or mask questions for that milestone.
