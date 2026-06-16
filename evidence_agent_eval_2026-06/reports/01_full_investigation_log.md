# 01 · 完整调查日志（全过程时间线）

本文件按实际发生顺序，完整记录这次调查的每一个阶段、尝试、踩坑与修正，便于复盘与复现。

---

## 阶段 0：任务下达

- 服务器：`<SERVER_IP>:23`（root，密码见环境变量），paramiko 方式连接。
- 目标：vLLM 起 grpo-agent-336 → 用 `tools_embedding.py` 回答 10 道 PICO 题 → 与线上循证智能体对比。
- 约束确认：`tools_embedding.py` 须为 **top1500 截断 + 每库 2 篇**。

### 10 道评测题（临床 PICO）
1. 急性心肌梗死合并糖尿病是否显著增加 5 年内心血管死亡风险？
2. 缺血性脑卒中后，双联抗血小板 vs 单药，能否降低 1 年内复发卒中？
3. 晚期 NSCLC，一线免疫联合化疗 vs 免疫单药，真实世界 OS 获益且不显著增毒？
4. 中重度类风湿关节炎，JAK 抑制剂 vs TNF-α 抑制剂，缓解率是否更优？
5. 妊娠期糖尿病孕妇，孕前 BMI 升高是否增加巨大儿风险？
6. 轻中度抑郁，CBT vs 单纯药物，能否更有效改善症状并降低复发率？
7. 慢性失眠成年人，CBT+褪黑素 vs 单用 CBT，是否更有效改善主观睡眠质量？
8. 脓毒症休克，限制性 vs 传统积极补液，能否降低 AKI 和 28 天病死率？
9. 急诊胸痛，hs-cTn + HEART 评分 vs 单独肌钙蛋白，能否更安全快速识别可出院患者？
10. 疑似肺癌，低剂量螺旋 CT vs 胸部 X 线，能否显著提高早期肺癌检出率？

---

## 阶段 1：模型甄别——发现 v14 是错的

1. 最初把 `ebm_agent_14b_grpo_4gpu/global_step_336`（FSDP checkpoint，4 shard，56GB）合并为 HF 模型 `grpo_agent_336_hf`（7 safetensors，28GB），用 `verl 0.5.0 model_merger`。
2. 用 vLLM 0.7.3 serve（`--tool-call-parser hermes --enable-auto-tool-choice --max-model-len 12000`），走框架推理 → 结果**异常**：
   - 输出里出现原始 JSON tool call 文本、伪造的 `{03-02}` 占位引用；
   - 多处"未检索到相关文献"刷屏、凭记忆作答。
3. 排查 `/workspace/models/grpo-agent-336` → **只有 tokenizer/config，0 个 safetensors**，不是可服务的模型。
4. 用户提示"grpo-agent-336 对应 v14 训练"，但实测 v14 step336 表现不对。

## 阶段 2：工具调用问题深挖（决定性）

逐项排除：
1. **chat_template**：`grpo_agent_336_hf` 与可用的 `grpo_v17_plus_small_step336_hf` 的 `chat_template.jinja` **逐字节相同**（2507 字节，标准 Qwen2.5 hermes 工具模板）。排除模板差异。
2. **vLLM 配置**：启动日志确认已加载该 jinja、`tool_call_parser=hermes`、`enable_auto_tool_choice=True`。排除配置。
3. **原生采样探针**（直连 vLLM）：
   - `temperature=0` 贪心 + 无 rep penalty → 退化复读"干预干预干预…"（这是贪心解码的伪影，不代表权重坏）。
   - **plain chat（无工具）→ 模型连贯**（正确解释脓毒症休克）→ **权重没坏**。
   - **带工具 `auto` → `tool_calls:[]`**，凭记忆答 + 编造"参考文献" → 问题是**不主动调工具**。
   - **强制具名 `tool_choice` → 产出完美 PICO 工具调用** → 模型能调、格式对。
4. **vLLM 0.7.3 限制**：`tool_choice="required"` 报错（只支持 `auto/none/具名工具`）。
5. **决定性测试**：把真实检索到的 12981 字证据**直接注入** + 强采样控制（rep_penalty 1.1）→ v14 仍**0 条 `[^..]` 引用、加禁止的"参考文献"列表、编造作者期刊、复读"以上为简化的循证结论…"20+ 次**。
   - 结论：**v14 step336 = GRPO 训练崩溃**。
6. 全盘搜索发现 v17 等其它合并模型，确认 **v17 才可用**。

## 阶段 3：评测管线复现（本地 v17）

1. 找到评测框架（本地 `D:\evidence_ii\backend\simple_agent\`）：
   - `main_api_openai.py`（FastAPI :9999，OpenAI Agents SDK）
   - `root_agent/sub_agents/tools_embedding.py`（4 库 Milvus 检索）
   - `root_agent/sub_agents/prompt_new.py`（system prompt）
   - `eval_sql_agent/eval_runner.py`（A2A `message/stream` 发题收 SSE）
2. 部署到服务器 `/workspace/eval_run/`，装依赖（`rapidfuzz`、`sseclient-py`），写 `.env`（`LLM_MODEL`、`VLLM_API_URL=:8400`、`PROJECT_NAME=xunzheng`）。
3. **切换到 v17**：停 v14 vLLM，在 :8400 起 `grpo_v17_plus_small_step336_hf`（served 名 `grpo_v17_336`）。
4. **关键修正**：`ModelSettings(max_tokens=1536, tool_choice="search_embedding_db", temperature=0.7, top_p=0.8, extra_body={"repetition_penalty":1.05,"top_k":20})`。
5. 单题验证：retrieval 命中真实文献（ESICM 2025 补液指南、"早期液体复苏对脓毒症死亡率"Meta 等），**19 条真实分层引用**，格式规范。
6. 跑完 10 题（screen 防 SSH 断），结果见 `data/answers_v17.md`。
   - 期间踩坑：同步 `exec_command` 长任务导致 SSH socket 超时关闭 → 改用 `screen` 后台 + 轮询日志解决。

## 阶段 4：线上生产智能体逆向

1. 用户提供线上入口 `https://www.infox-med.com/20/#/aiChat/6`（SPA）。
2. 之前盲试 ~40 种 `api.infox-med.com` 路径全 404 → 改为**从前端 JS bundle 逆向**：
   - 下载 `index-*.js`（4.3MB），grep API 路径。
   - 发现 SSE 走 `fetchEventSource`，专用前缀 `/infoxmed20sse/`，端点 `sseEndpoint`、`cancelSse`、`reconnectSse`。
   - 循证智能体 = `fId:"6"`，header 需 `Token: getToken()` + `X-Api-Version: getAiVersionByFid(fId)`。
3. 用户从浏览器 F12 提供 **Token** 与 **X-Api-Version=5003016**。
4. 单题 smoke：HTTP 200 SSE，解析出 messageType 分工（8=答案/49=工具/50=文档/61=证据卡），引用格式 `[^xx-xxxx-x$$N]`。
5. 跑完 10 题，结果见 `data/answers_online.md`、`data/online_answers.json`。
   - 踩坑：响应 `Content-Encoding: br` + 中文 → 发 `Accept-Encoding: identity` 避免压缩、按 UTF-8 解码。

## 阶段 5：对比分析

1. 修正引用正则以兼容 `$$N` 后缀（否则线上引用数误计为 0）。
2. 生成总览表 + 逐题并排（`data/comparison_v17_vs_online.md`）。
3. 线上引用 100% 可追溯到检索文档（含标题/年份/库/证据卡）。

## 阶段 6：检索规模 × 上下文长度分析

- 用户回忆：线上是 **top10 + 不截断**，本地是 **top2 + 1500 截断** → 这才是回答长度/详细度差异的根因。
- 据此测算 Qwen2.5-14B（原生 32K）在不同 `max-model-len` 下能支撑的不截断文档数，并提示 OOD 训练分布风险（详见 [06](06_context_length_retrieval_analysis.md)）。

---

## 踩坑速查表

| 现象 | 根因 | 解决 |
|---|---|---|
| v14 不调工具、编造引用、复读 | GRPO 训练崩溃（mode collapse） | 改用 v17 |
| `tool_choice="required"` 报错 | vLLM 0.7.3 不支持 | 用具名工具 `search_embedding_db` |
| 输出退化复读 | SDK 未传 rep_penalty/top_k | `extra_body={repetition_penalty:1.05, top_k:20}` |
| `tool_choice=0` 即贪心退化 | temp=0 无惩罚 | 用 temp 0.7 + top_p 0.8 |
| SSH 长任务断连 | 同步 exec_command 超时 | screen 后台 + 轮询日志 |
| 线上 API 全 404 | 盲试路径 | 从 SPA JS bundle 逆向 |
| 线上响应乱码 | br 压缩 + 编码 | `Accept-Encoding: identity` + UTF-8 |
| 线上引用计数=0 | 正则没含 `$$N` | 正则兼容 `(?:\$\$[0-9]+)?` |
