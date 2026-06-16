# 循证智能体评测与对比分析（2026-06）

本目录完整记录了一次围绕 **循证智能体（Evidence-Based Medicine Agent）** 的端到端调查、评测与对比工作，内容涵盖：

1. **模型版本甄别**：发现 v14 (`global_step_336`) 是 GRPO 训练崩溃的 checkpoint，确认 **v17** 才是可用的好模型。
2. **工具调用问题深挖**：定位"模型不调检索工具"的真实根因（采样参数 / `tool_choice` / 训练分布），而非框架或检索的问题。
3. **评测管线复现**：在服务器上完整搭起 `eval_runner → :9999 (A2A) → main_api_openai.py → tools_embedding.py + prompt_new.py → vLLM` 的官方评测流程，跑完 10 道临床 PICO 题。
4. **线上生产智能体逆向**：从前端 SPA 的 JS bundle 还原出线上循证智能体的真实 SSE 接口、请求格式、响应结构与引用体系，并跑完同样 10 题。
5. **逐题对比分析**：本地 v17 vs 线上生产版，从引用数、字数、证据颗粒度、临床洞察、可追溯性等维度对比。
6. **检索规模 × 上下文长度分析**：量化"为什么线上回答更长更细"，给出 `max-model-len ↔ 文档检索数量` 的支撑能力测算与 OOD 风险提示。

> ⚠️ **凭证脱敏说明**：本目录所有脚本中的 **SSH 密码**、**线上鉴权 Token** 均已替换为环境变量占位符（`<SSH_PASSWORD>` / `<ONLINE_TOKEN>`）。请勿在仓库中提交任何明文凭证。

---

## 目录结构

```
evidence_agent_eval_2026-06/
├── README.md                      ← 本文件
├── reports/                       ← 分析报告（按阅读顺序编号）
│   ├── 00_executive_summary.md            执行摘要（先读这个）
│   ├── 01_full_investigation_log.md       完整调查日志（全过程时间线）
│   ├── 02_v14_vs_v17_model_collapse.md    v14 崩溃 vs v17 可用：证据与分析
│   ├── 03_eval_pipeline_architecture.md   评测管线架构详解
│   ├── 04_online_agent_reverse_engineering.md  线上生产智能体接口逆向
│   ├── 05_v17_vs_online_comparison.md     v17 vs 线上 逐题对比分析
│   ├── 06_context_length_retrieval_analysis.md 上下文长度 × 检索规模测算
│   └── 07_conclusions_recommendations.md  结论与后续建议
├── data/                          ← 原始评测数据与产物
│   ├── eval_dataset_10.json               10 道评测题
│   ├── answers_v17.json / .md             v17 模型 10 题回答
│   ├── online_answers.json                线上 10 题回答（含检索文档/证据卡原始数据）
│   ├── answers_online.md                  线上 10 题回答（问答对格式）
│   ├── comparison_v17_vs_online.md        逐题并排对比（含线上引用可追溯来源）
│   └── compare_data.json                  对比的结构化中间数据
└── scripts/                       ← 关键脚本（已脱敏）
    ├── online_eval_runner.py              线上循证智能体批量提问 + SSE 解析
    ├── local_v17_eval_runner.py           本地 v17 评测（paramiko 远程驱动）
    ├── reverse_engineer_online_api.md     线上接口逆向方法（命令记录）
    └── build_comparison.py                生成对比文档
```

## 一句话结论

> 自训练的 14B 模型 **v17 已能产出结论正确、格式规范、引用真实、零幻觉**的循证回答；与线上生产版的差距主要在 **证据颗粒度、临床 nuance、决策表格化与证据卡可视化**，其根因是 **检索规模（线上 top10 不截断 vs 本地 top2 + 1500 截断）** 与可能的**更大模型/更强工程**，而非检索架构或模型基本能力的差距。
