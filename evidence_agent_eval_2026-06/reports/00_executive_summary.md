# 00 · 执行摘要

> 日期：2026-06-16　|　范围：循证智能体（EBM Agent）模型甄别、评测、线上对比、检索规模分析

## 任务背景

最初目标有三个：
1. 在服务器（`<SERVER_IP>:23`，root）上用 vLLM 把 **grpo-agent-336** 模型起起来；
2. 接入 `tools_embedding.py` 检索工具，回答 **10 道临床 PICO 题**（需确认检索为 **top1500 截断 + 每库 2 篇**）；
3. 把这 10 题的回答与**公司线上已上线的循证智能体**做对比。

## 关键发现（按重要性）

### 1. "grpo-agent-336"对应的 v14 checkpoint 已训练崩溃，真正可用的是 v17
- 最初合并/部署的 `grpo_agent_336_hf`（来自 `ebm_agent_14b_grpo_4gpu/global_step_336`，即 v14）**喂了真实检索证据也不按训练格式带 `[^xx-xxxx-x]` 引用、会编造参考文献、长输出严重复读退化**——典型的 **GRPO mode collapse（RL 训练崩溃）**。
- 服务器上的 `/workspace/models/grpo-agent-336` 只有 tokenizer/config，**无 safetensors 权重**，不可直接服务。
- **v17（`grpo_v17_plus_small_step336_hf`）才是好模型**：同一套流程下每题稳定产出 12–18 条真实分层引用，格式规范、零幻觉、无退化。`main_api_openai.py` 里残留的 "grpo-agent-336" 只是历史命名。
- 详见 [02_v14_vs_v17_model_collapse.md](02_v14_vs_v17_model_collapse.md)。

### 2. "模型不调工具"的根因不是框架/检索，而是采样与训练分布
- v14 在 `tool_choice="auto"` 下倾向凭记忆直接答；强制具名工具时能产出**完美** PICO 工具调用 → 说明工具/格式没问题。
- vLLM 0.7.3 **不支持 `tool_choice="required"`**，只认 `"auto"/"none"/具名工具`。
- 关键修正：`ModelSettings` 需设 `tool_choice="search_embedding_db"`（强制检索）+ 显式采样 `temperature=0.7, top_p=0.8, extra_body={repetition_penalty:1.05, top_k:20}`（否则模型退化复读）。
- 详见 [02](02_v14_vs_v17_model_collapse.md) 与 [03](03_eval_pipeline_architecture.md)。

### 3. 评测管线完整复现，v17 跑通 10 题
- 架构：`eval_runner（A2A message/stream）→ :9999 main_api_openai.py（OpenAI Agents SDK）→ tools_embedding.py（4 库检索，Milvus）+ prompt_new.py（system prompt）→ vLLM:8400`。
- v17 结果：10 题全部正常，**平均 14.5 条引用 / 1223 字 / ~30s**，分层规范（01-中文指南 / 02-英文指南 / 03-系统评价Meta / 04-RCT）。
- 详见 [03_eval_pipeline_architecture.md](03_eval_pipeline_architecture.md)、数据 `data/answers_v17.md`。

### 4. 线上生产循证智能体接口已逆向，跑通同 10 题
- 真实接口：`POST https://api.infox-med.com/infoxmed20sse/sseEndpoint`（SSE），`functionId=6`，需 `Token` + `X-Api-Version: 5003016`。
- 响应 messageType：`8`=答案正文、`49`=工具调用状态、`50`=检索文档、`61`=证据卡。
- 引用格式 `[^xx-xxxx-x$$N]`（与本地同体系，多了 `$$显示序号`）。**也用 `search_embedding_db` + PICO**，架构一致。
- 详见 [04_online_agent_reverse_engineering.md](04_online_agent_reverse_engineering.md)、数据 `data/answers_online.md`。

### 5. v17 vs 线上：结论一致，差在颗粒度
| 指标 | 本地 v17 | 线上生产版 |
|---|---|---|
| 平均字数 | 1223 | **2638**（约 2 倍） |
| 平均引用（去重） | 14.5 | 6.3（**100% 可追溯**） |
| 证据卡 | 无 | 平均 4.3 张 |
| 核心结论 | 与线上**一致** | 与本地一致 |
| 响应时间 | ~30s | ~60s |

- 线上更强：具体效应量（RR/CI/HR）、点名具体试验（CLOVERS / REVERSE-AKI / REDUCE）、临床亚组 nuance、决策**表格**、可点击证据卡。
- v17 不输：核心结论正确、格式规范、引用真实、更简洁更快。
- 详见 [05_v17_vs_online_comparison.md](05_v17_vs_online_comparison.md)。

### 6. 差距根因：检索规模，而非架构
- 线上 = **top10 + 不截断**；本地 = **top2 + 1500 字截断** → 线上喂给模型的证据量是本地的 5–20 倍 → 自然更长更细、数字更多。
- 测算（Qwen2.5-14B 原生 32K 上下文）：
  - 12000（现状）→ 支撑 ~7–8 篇不截断；
  - 32768（原生上限）→ ~28–30 篇（≈ 每库 top7–8）；
  - 要完全复刻线上 top10×4≈40 篇不截断 → 需 YaRN 扩到 ~48K。
- **关键风险**：v17 是在 top2+1500截断分布下训练的，直接放大到 30–40 篇是**训练分布外（OOD）**，可能"lost in the middle"或退化。对齐线上的正路是**按 top10/不截断重训**，而非仅推理时放大。
- 详见 [06_context_length_retrieval_analysis.md](06_context_length_retrieval_analysis.md)。

## 最终建议

1. **下线/不要再用 v14 `global_step_336`**，统一以 v17 为基线。
2. 评测/部署 v17 时务必带上 `tool_choice=具名工具` + 正确采样参数。
3. 若要让 v17 逼近线上详细度：**保守甜点区 = `max-model-len 32768` + 每库 top7~8 + 去截断**（原生支持、显存无压力），但需实测 OOD 退化。
4. 认真对齐线上质量：**用 top10/不截断的检索配置重新训练一版**。

详见 [07_conclusions_recommendations.md](07_conclusions_recommendations.md)。
