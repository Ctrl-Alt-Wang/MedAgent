# 03 · 评测管线架构详解

## 整体数据流

```
eval_runner.py
   │  POST /  (A2A JSON-RPC, method="message/stream")
   ▼
[:9999] main_api_openai.py   (FastAPI + OpenAI Agents SDK)
   │  Runner.run(agent, question, max_turns=2)
   │
   ├── system prompt ← prompt_new.py  (SYSTEM_INSTRUCTION)
   ├── tool: search_embedding_db ← tools_embedding.py
   │
   ▼  turn1: 模型抽 PICO → 调 search_embedding_db
[tools_embedding.py]  并发检索 4 个 Milvus 库
   │  https://ai.infox-med.com/UserQueryProcess/GeneralQueryInterface
   │  guideline_zh / guideline_en / systematic_and_meta / RCT
   │  每库 top2 + 单篇 1500 字截断（本地配置）
   ▼  turn2: 模型基于证据分层带 [^xx-xxxx-x] 引用作答
[:8400] vLLM (Qwen2.5-14B, v17)  ← LiteLLM (openai/ 前缀)
   │
   ▼  SSE 流式回传 → eval_runner 收集 → *_answers.json
```

## 各组件要点

### eval_runner.py（评测发题端）
- 通过 **A2A 协议**（JSON-RPC `message/stream`）向 `http://localhost:9999/` 发题。
- 用 `sseclient` 收流，从 `result.artifacts[].parts[].text` 拼接答案。
- 先 `GET /.well-known/agent.json` 探活。

### main_api_openai.py（Agent 服务，:9999）
- **OpenAI Agents SDK**（`agents.Agent` / `Runner`），非 Google ADK。
- 模型：`LitellmModel(model="openai/"+LLM_MODEL, base_url=VLLM_API_URL)`（注意是 `openai/` 前缀，不是 `hosted_vllm/`）。
- `max_turns=2`：turn1 调工具检索，turn2 出最终答案。
- **关键配置（本次修正）**：
  ```python
  ModelSettings(
      max_tokens=1536,
      tool_choice="search_embedding_db",          # 强制检索（硬约束）
      temperature=0.7, top_p=0.8,
      extra_body={"repetition_penalty": 1.05, "top_k": 20},  # 防退化
  )
  ```
- SSE 输出格式：`result.artifacts[].parts[{kind:"text", text:...}]` + `status.state ∈ {completed, failed}`。

### tools_embedding.py（检索工具）
- 工具 `search_embedding_db(P, I, C, O)`：并发检索 4 库。
- 每库按 `P / I / PI / PIC / PIO / PICO` 多组关键词检索后按 id 去重、按 `weight×reranker_score` 排序。
- **本地配置**：每库 `result_data = data[:2]`（top2）；`fuzzy_search` 中 `content[:1500]`（单篇 1500 字截断）。
- 引用 ID 体系：`{库前缀}-{uuid4[:4]}-0`
  - `01-` 中文指南（guideline_zh）
  - `02-` 英文指南（guideline_en）
  - `03-` 系统评价/Meta（systematic_and_meta）
  - `04-` RCT（clinical）
- ⚠️ 引用 ID 的中间四位是**检索时随机 uuid**，不持久化 → 事后无法对已生成答案反查标题，需在生成时同步抓取映射。

### prompt_new.py（system prompt）
- 角色"循证决策专家"；强制：医学问题**必须先调检索工具**、禁止凭记忆答、禁止编造引文、禁止文末参考文献列表。
- 强制输出结构：**一、直接结论 → 二、循证分层证据总结（四层）→ 三、临床实践决策**。
- 严格分层约束：各层只能引用对应前缀 ID（01/02/03/04），禁止跨层混用。

### vLLM（:8400）
- `Qwen2.5-14B`（v17），`--dtype bfloat16 --max-model-len 12000 --enable-auto-tool-choice --tool-call-parser hermes --gpu-memory-utilization 0.90`。
- served-model-name `grpo_v17_336`。

## 部署注意事项（本次实践）

1. 服务器 py312 环境需补：`rapidfuzz`、`sseclient-py`（`agents`/`litellm`/`fastapi`/`uvicorn` 已有）。
2. `.env`：`MODEL_PROVIDER=vllm` / `LLM_MODEL=grpo_v17_336` / `VLLM_API_URL=http://localhost:8400/v1` / `VLLM_API_KEY=none` / `PROJECT_NAME=xunzheng`。
3. 长任务用 `screen` 后台跑，避免 SSH 断连杀进程。
4. Milvus 检索端点 `https://ai.infox-med.com/...` 服务器可达（线上/服务器均可）。

## v17 评测结果（10 题）

| 题 | 引用数 | 字数 | 耗时 |
|---|---|---|---|
| q01 | 18 | 1246 | 31s |
| q02 | 15 | 1000 | 29s |
| q03 | 14 | 1377 | 32s |
| q04 | 12 | 1036 | 27s |
| q05 | 15 | 1524 | 33s |
| q06 | 16 | 1264 | 29s |
| q07 | 13 | 1013 | 26s |
| q08 | 14 | 1047 | 28s |
| q09 | 14 | 1469 | 33s |
| q10 | 14 | 1254 | 30s |

完整答案见 `data/answers_v17.md`。
