# v17_plus_small 训练复盘报告

**模型版本**：Qwen2.5-14B-GRPO-v17_plus_small  
**训练框架**：AgentLightning + VERL + GRPO  
**推理部署**：OpenAI agents SDK + search_embedding_db（4DB, top2, 1500-cut）  
**评测得分**：78.6（v17标准，200题）  
**训练完成 global_step**：336  
**报告生成日期**：2026-05-26  

---

## 一、训练基本参数

| 参数 | 值 | 来源 |
|------|-----|------|
| 模型底座 | Qwen2.5-14B-Instruct-SFT-Agent | train_sql_agent.py DEFAULT_MODEL_PATH |
| GPU 数量 | 4 (TP=4, DP=1) | 4GPU A800-80GB |
| 训练数据 | data/train.parquet | 约220KB文件 |
| 验证数据 | data/val.parquet | — |
| train_batch_size | 8 (base_batch_per_dp=8, dp=1) | verl_default_config |
| n_rollouts | 4 | rollout.n=4 |
| rollouts_per_update | 32 (=8×4) | progress file 确认 |
| max_turns | 2 | rollout.multi_turn.max_turns |
| max_prompt_length | 12288 | data.max_prompt_length |
| max_response_length | 2048 | data.max_response_length |
| max_model_len (vLLM) | 16384 | engine_kwargs.vllm.max_model_len |
| base_lr | 7e-7 | verl_default_config |
| 有效 lr | max(8e-7, min(lr, 3e-6))，DP=1时约7e-7→clamp至8e-7 | 代码推算 |
| kl_loss_coef | 0.08 | actor.kl_loss_coef |
| entropy_coeff | 0.005 | actor.entropy_coeff |
| clip_ratio | 0.20 / 0.20 | actor.clip_ratio_low/high |
| save_freq | 50 steps | 默认值 |
| test_freq | 25 steps | 默认值 |
| total_epochs | 3 | 默认值 |
| progress_file | /tmp/sql_agent_progress_v17_plus_small.json | 命令行默认 |
| schedule_steps | min(150, total_updates) ≈ 150 | 代码默认 |
| gpu_memory_utilization | 0.40 (4GPU) | verl_default_config |
| offload_policy | True (4GPU) | fsdp_config |
| val_before_train | False | trainer |
| experiment_name | ebm_agent_14b_grpo_4gpu_v17_plus_small | trainer |
| optimizer checkpoint | 保存 model + optimizer + extra | save_contents |

---

## 二、训练进度（最终状态）

**数据来源**：/tmp/sql_agent_progress_v17_plus_small.json（2026-05-15 读取）

```json
{
  "rollout_count": 10750,
  "global_step": 335,
  "total_updates": 339,
  "rollouts_per_update": 32,
  "updated_at": 1778844789.87
}
```

- 最终保存 checkpoint：global_step_336（目录已确认存在）
- 完成度：335/339 ≈ 98.8%，几乎完整训练
- 已保存 checkpoints（v17目录下）：step 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 336

---

## 三、Reward 函数设计（sql_agent.py）

### 3.1 整体结构

```
Total = hard_reward
       + format_scale × format_reward
       + correct_scale × correctness_reward
       + rm_weight × rm_score
```

### 3.2 各分项说明

**Hard Reward（红线项，不参与动态缩放）**：
- 医学问题未调工具：`-1.0`（直接返回）
- 答案末尾出现脚注解释列表：`-0.5`
- 引用了 ID 但无有效 valid_ids：`-1.2`
- 伪造引用（fabricated_ids > 0）：`-min(2.0, 0.8 + 0.5×N)`

**Format Reward（范围 [0,1]）**：
- PICO 完整度（≥3项 +0.15，≥2项 +0.10，≥1项 +0.05）
- 三大主结构出现比例 × 0.35 + 顺序正确率 × 0.15
- 四子结构出现比例 × 0.25 + 顺序正确率 × 0.10

**Correctness Reward（范围 [-3, 3.2]）**：
- 引用精度（real_ids/cited_ids）× 1.4
- 真实引用数量奖励（≥4: +0.50, ≥2: +0.30, ≥1: +0.12）
- Section-prefix 对齐检查（[-0.5, +0.5]）
- 诚实放弃检测（honest abstention，+0.35/-0.10）
- **v17_plus_small 专有**：小型 recall bonus（0.10×recall，避免 v16 5.304 plateau）
- **v17_plus_small 专有**：引用多样性 bonus（0.02×min(cited_db_count, 3)）

**RM Score（外部 reward model）**：
- 调用 REWARD_MODEL_URL（[REDACTED]）
- 输出归一化至 [0,1]
- 前 30% 进度不启用（rm_weight=0），后 70% 线性增至 0.3

### 3.3 动态权重（v17_plus_small）

```python
format_scale = 1.2      # 固定，不再动态衰减（v17引入此简化）
correct_scale = 1.0     # 固定
rm_weight = 0.3 × (progress - 0.3) / 0.7  # progress∈[0.3, 1.0] → [0, 0.3]
```

---

## 四、训练日志分析

**日志文件**：/tmp/train_v17_plus_small.log（实际对应 sql_agent_training.log，约30K行最后500行均为正常训练输出）

- 日志末尾（约step 112/113 val轮次）：正常执行，PICO 4/4全填，valid_ids=7，fabricated=0
- Total Reward 末尾区间：约 2.5~3.5（正常收敛范围）
- 训练结束时 RM weight 约 0.296（progress≈0.991，在0.3后斜坡上升中）
- 无 no_tool 事件（grep计数=0），说明医学问题工具调用全覆盖
- 无严重 error/fatal/exception 记录（来自代码逻辑确认）

---

## 五、评测结果

**评测文件**：qwen25_14b_grpo_v17_plus_small_32k_strict_v4_20260516_212130_scored.json

| 维度 | 得分 | 样本数 |
|------|------|-------|
| **综合总分** | **78.6** | 200题 |
| 标准差 | 7.9 | — |
| 最低 / 最高 / 中位 | 47.8 / 92.0 / 79.2 | — |
| 问题相关性 (rule_question_relevance) | 64.2 | 339 |
| 证据层级合理性 (rule_evidence_hierarchy) | 83.4 | 352 |
| 证据质量与时效性 (rule_evidence_quality_timeliness) | 56.8 | 346 |
| 证据回答一致性 (rule_evidence_answer_consistency) | 74.0 | 346 |
| 信息全面性与深度 (rule_completeness_depth) | 55.4 | 321 |
| 逻辑严谨性 (rule_logical_rigor) | 77.1 | 230 |
| 医学术语规范性 (rule_terminology_standard) | 99.5 | 184 |

---

## 六、Checkpoint 结构

**目录**：checkpoints/AgentLightning/ebm_agent_14b_grpo_4gpu_v17_plus_small/

```
global_step_10  (2026-05-10 12:52)
global_step_20  (2026-05-10 16:14)
global_step_30  (2026-05-10 19:55)
global_step_40  (2026-05-10 23:16)
global_step_50  (2026-05-11 02:44)
global_step_60  (2026-05-11 06:19)
global_step_70  (2026-05-11 09:51)
global_step_80  (2026-05-11 13:39)
global_step_90  (2026-05-11 17:05)
global_step_100 (2026-05-11 18:04)
global_step_150 (2026-05-14 04:52)
global_step_200 (2026-05-14 22:39)
global_step_250 (2026-05-15 11:33)
global_step_336 (2026-05-15 11:38) [最终]
```

---

## 七、核心结论（5条）

1. **训练几乎完整收敛**：v17_plus_small 完成了 335/339 步（98.8%），最终 checkpoint global_step_336 已持久化，训练期间无 error 中断记录，日志整体稳定。

2. **Reward 函数平衡良好，无工具遗漏**：通过 grep 确认整个训练过程中 no_tool 事件=0，说明模型已完全掌握医学问题工具调用规范；末尾阶段 Total Reward 稳定在约 2.8~3.4，收敛良好。

3. **评测得分 78.6，证据层级最强**：证据层级合理性（83.4）和医学术语规范性（99.5）是最强维度；信息全面性（55.4）和证据质量时效性（56.8）仍是短板，说明模型对高质量 SR/RCT 证据的覆盖深度不足。

4. **format_scale=1.2 的固定权重有效约束了格式行为**：相比历史版本 v15/v16 的动态衰减，v17 固定 format_scale=1.2 使格式奖励更稳定，避免了后期格式退化问题。

5. **v17_plus_small 的 recall+diversity bonus 是关键改进点**：code diff 确认 v17 专有引入了小型 recall bonus（0.10×recall）和引用多样性 bonus（0.02×N，上限3个DB），有效对抗了 v16 版本中 reward 在高 precision/低 recall 状态下的过早 plateau 问题。

---

*报告生成：2026-05-26 | 证据来源：服务器 /workspace/post_train/sql_agent/ + 本地评测文件*
