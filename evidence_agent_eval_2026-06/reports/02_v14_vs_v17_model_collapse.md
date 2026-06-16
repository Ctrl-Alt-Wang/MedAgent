# 02 · v14 崩溃 vs v17 可用：证据与分析

## 结论先行

- **v14 = `ebm_agent_14b_grpo_4gpu/global_step_336`（合并为 `grpo_agent_336_hf`）= GRPO 训练崩溃的 checkpoint**，不可用于循证作答。
- **v17 = `grpo_v17_plus_small_step336_hf` = 可用的好模型**，是历史上"grpo-agent-336 效果好"真正对应的权重。
- 两者底座、chat_template、generation_config 完全一致；差异**仅在训练得到的权重**。

---

## 一、关键事实核查

| 检查项 | v14 (`grpo_agent_336_hf`) | v17 (`grpo_v17_plus_small_step336_hf`) |
|---|---|---|
| 来源 checkpoint | `ebm_agent_14b_grpo_4gpu/global_step_336` | `ebm_agent_14b_grpo_4gpu_v17_plus_small/global_step_336` |
| chat_template.jinja | 2507 字节，标准 Qwen2.5 hermes 工具模板 | **逐字节相同** |
| generation_config | temp 0.7 / top_p 0.8 / top_k 20 / rep_penalty 1.05 | 相同 |
| plain chat 连贯性 | ✅ 短文本连贯 | ✅ |
| 强制工具调用 | ✅ 能产出完美 PICO 工具调用 | ✅ |
| **auto 下主动调工具** | ❌ 倾向凭记忆答 | ✅ 稳定调用 |
| **喂真证据后带 `[^..]` 引用** | ❌ 0 条 | ✅ 12–18 条 |
| **长输出稳定性** | ❌ 复读退化 | ✅ 稳定 |
| **是否编造文献** | ❌ 编造作者/期刊 + 加禁止的参考文献列表 | ✅ 真实可追溯 |

> 补充：服务器 `/workspace/models/grpo-agent-336` 仅含 tokenizer/config，**无 safetensors 权重**，不可直接 serve；其 config/chat_template/generation_config 与 `grpo_agent_336_hf` 一致，证明后者就是 v14 的完整权重。

---

## 二、v14 崩溃的实证

### 决定性测试：注入真实证据 + 强采样控制

将真实检索到的 **12981 字证据**直接作为 tool 结果注入，配合 `temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.1` 与完整 system prompt，v14 仍然：

1. **0 个 `[^xx-xxxx-x]` 内嵌引用**（prompt 明确要求的格式，一个都没产出）；
2. **加了 prompt 明令禁止的"参考文献"列表**；
3. **编造文献**——`Li L, Xu J, Wang X, et al. ... Front Med. 2025;10(3):345-357` 等作者/期刊/卷期全是瞎编，不是检索返回的真实 doc ID；
4. **严重退化**——结尾复读"以上为简化的循证结论…"20+ 次。

### 对照：v17 同题、同流程的输出（节选 q08 脓毒症）

```
### 一、直接结论
在脓毒症休克患者中，目前尚无高质量证据显示限制性液体复苏策略能显著降低急性肾损伤
（AKI）或28天病死率……[^04-4eeb-0][^04-7bff-0]。
### 二、循证分层证据总结
#### （一）中文指南证据层 …[^01-5611-0][^01-da93-0][^01-72e9-0]。
#### （二）英文指南证据层 …[^02-ad9a-0][^02-77cf-0]。
#### （三）系统评价/Meta分析证据层 …[^03-c588-0][^03-4888-0]。
#### （四）临床试验证据层 …[^04-4eeb-0][^04-7bff-0]。
### 三、临床实践决策 …[^01-5611-0][^02-ad9a-0][^04-4eeb-0]。
```

结构规范、四层分明、引用真实、零幻觉、无退化。

---

## 三、为什么是"崩溃"而非"权重损坏"

- **plain chat 连贯**证明权重数值本身没坏（FSDP 合并正确）。
- 崩溃只在**长输出 + 需遵循训练任务格式**时显现：模型丢失了训练期学到的"调工具→读证据→分层带引用"行为，退化为凭记忆生成 + 复读。
- 这是 **GRPO / RL 训练 mode collapse / reward hacking** 的典型表现：某个中间 checkpoint（step 336）策略坍缩，输出多样性与任务遵循度崩塌。

> 早期把贪心 `temperature=0` 下的"干预干预干预…"误判为权重损坏；后续用模型自带的采样参数（temp 0.7 + rep_penalty 1.05）复测，确认短文本连贯，纠正了该误判。

---

## 四、复现该结论的最小步骤

1. 分别 serve 两个模型（同样 vLLM 参数：`--tool-call-parser hermes --enable-auto-tool-choice`）。
2. 用同一道题、同一 system prompt，分别：
   - `tool_choice="auto"` 直接问 → 观察 v14 不调工具；
   - 注入相同的真实证据 + 同采样参数 → 观察 v14 不带 `[^..]` 引用且退化、v17 正常。
3. 对比 `chat_template.jinja`（`diff` 应完全相同）以排除模板因素。

---

## 五、行动建议

- **立即停用 v14 `global_step_336`**，所有评测/部署以 **v17** 为基线。
- 训练侧排查 v14 该 run 的 reward 曲线与 KL 控制，定位崩溃步数，避免选用坍缩 checkpoint。
- 选 checkpoint 时除 reward 外，应加**格式遵循 / 引用真实率 / 重复率**等自动指标，及早发现 mode collapse。
