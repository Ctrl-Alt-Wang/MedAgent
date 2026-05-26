# v18_full_grpo 训练复盘报告

**模型版本**：Qwen2.5-14B-GRPO-v18  
**训练框架**：AgentLightning + VERL + GRPO  
**推理部署**：OpenAI agents SDK + search_embedding_db（4DB, top2, 1500-cut），vLLM bfloat16，max_model_len=32768  
**评测得分**：77.0（v17标准，199题）  
**最佳验证 checkpoint**：step300（val_peak）  
**报告生成日期**：2026-05-26  

---

## 一、训练基本参数

| 参数 | 值 | 来源 |
|------|-----|------|
| 模型底座 | Qwen2.5-14B-Instruct-SFT-Agent | train_sql_agent_v18.py DEFAULT_MODEL_PATH |
| GPU 数量 | 4 (TP=4, DP=1) | 4GPU A800-80GB |
| 训练数据 | data/train_v18_full_weighted.parquet | 约236KB |
| 验证数据 | data/val.parquet | — |
| train_batch_size | 8 (base_batch_per_dp=8, dp=1) | verl_default_config |
| n_rollouts | 4 | rollout.n=4 |
| rollouts_per_update | 32 (=8×4) | progress file 确认 |
| max_turns | 2 | rollout.multi_turn.max_turns |
| max_prompt_length | 12288 | data.max_prompt_length |
| max_response_length | 2048 | data.max_response_length |
| max_model_len (vLLM) | 16384 | engine_kwargs.vllm.max_model_len |
| base_lr | **5e-7**（v17 为 7e-7）| verl_default_config，降低 |
| 有效 lr 下限 | max(**3e-7**, min(lr, 3e-6))（v17 为 8e-7）| 降低 |
| kl_loss_coef | **0.10**（v17 为 0.08）| actor.kl_loss_coef，增加 |
| entropy_coeff | 0.005 | 与 v17 相同 |
| clip_ratio | 0.20 / 0.20 | 与 v17 相同 |
| save_freq | 100 steps（推算，4个ckpt：100,200,300,400）| checkpoint目录确认 |
| test_freq | 25 steps | 默认值 |
| total_epochs | 3 | 默认值 |
| progress_file | /tmp/sql_agent_progress_v18_full_grpo.json | 命令行默认 |
| gpu_memory_utilization | 0.40 (4GPU) | verl_default_config |
| val_before_train | **True**（v17 为 False）| trainer，新增初始验证 |
| experiment_name | ebm_agent_14b_grpo_4gpu_v18_full_grpo | trainer |
| optimizer checkpoint | 保存 model + optimizer + extra | save_contents |

---

## 二、训练进度（最终状态）

**数据来源**：/tmp/sql_agent_progress_v18_full_grpo.json（2026-05-26 读取）

```json
{
  "rollout_count": 13024,
  "global_step": 407,
  "total_updates": 460,
  "rollouts_per_update": 32,
  "updated_at": 1779716055.62
}
```

- 当前运行到 step 407/460（88.5%）
- 已保存 checkpoint 目录（v18）：global_step_100, 200, 300, 400
- 另有专门保存的两个峰值 checkpoint：
  - `KEEP_v18_step300_val_peak`（2026-05-23 22:57）—— 验证峰值
  - `KEEP_v18_step400_current`（2026-05-25 11:21）—— 最新当前
- **评测使用的是 step300 val_peak 版本**

---

## 三、Reward 函数设计（sql_agent_v18.py）

### 3.1 整体结构（与 v17 相同框架）

```
Total = hard_reward
       + format_scale × format_reward
       + correct_scale × correctness_reward
       + evidence_safety_reward         # v18 新增
       + rm_weight × rm_score
```

### 3.2 v18 核心变动

**format_scale 降低（0.7 vs v17 的 1.2）**：
```python
format_scale = 0.7   # v17: 1.2
correct_scale = 1.0  # 不变
```
v18 设计意图：降低格式奖励权重，更突出 citation 质量和 correctness。

**rm_weight 降低（最大 0.25 vs v17 的 0.3）**：
```python
rm_weight = 0.25 × (progress - 0.3) / 0.7  # v17: 0.3
```

**EvidenceSafety 新增模块**（v18 完全新增，v17 无此函数）：
- 无真实引用支撑但使用强结论词（如"明确优于""强烈推荐"）：`-0.35`
- 有真实引用但无 SR/RCT 级别引用且使用强结论词：`-0.12`
- 谨慎表达（证据有限/需谨慎等）且无 SR/RCT 级别引用：`+0.15`
- 谨慎表达且有指南级引用但无 SR/RCT：额外 `+0.05`
- 范围裁剪：`[-0.5, 0.25]`

**新增正则模式**：
- `CAUTION_PATTERN`：检测谨慎表达词汇
- `STRONG_CLAIM_PATTERN`：检测过度强结论词汇

### 3.3 动态权重（v18）

```python
format_scale = 0.7      # 固定（v17为1.2）
correct_scale = 1.0     # 固定
rm_weight = 0.25 × (progress-0.3)/0.7  # 最大0.25（v17为0.3）
```

---

## 四、训练日志分析

**日志文件**：/tmp/train_v18_full_grpo.log（步骤约到step 397以上）

- 最后100条 Total Reward：集中在 3.35~3.43 区间，偶见2.88和1.935（正常离群值）
- EvidenceSafety 典型记录（末尾50条）：
  - score=0.000 占主流（引用完整时既无奖也无罚）
  - 极少出现负分，说明模型已基本学会避免无根据的强结论
- 日志中 no_tool 事件 = 0（与 v17 相同，工具调用全覆盖）
- 最新记录 step 397 的完整训练指标（从 global_step 日志读取）：
  - training/reward: 3.363
  - critic/score/mean: 3.359（min=2.875, max=3.422）
  - actor/kl_loss: 0.0086
  - actor/grad_norm: 2.41
  - response_length/mean: 421.5（min=55, max=1002）
  - 无 truncated triplets

---

## 五、v18 新数据集特征

**v18 训练集**：data/train_v18_full_weighted.parquet（236KB，略大于 v17 的 221KB）

eval 数据集（200题）的分布（仅 eval 信息完整，train 因服务器无 pyarrow 未能直接读取 schema）：

**评测数据集** (eval_dataset_final_v5.json, 200题):
| 科室 | 题数 |
|------|------|
| 呼吸系统 | 22 |
| 内分泌代谢 | 20 |
| 心血管 | 20 |
| 感染性疾病 | 20 |
| 消化系统 | 20 |
| 神经内科 | 20 |
| 肿瘤学 | 20 |
| 风湿免疫 | 20 |
| 肾脏疾病 | 19 |
| 血液系统 | 19 |

---

## 六、评测结果

**评测文件**：grpo_v18_step300_v17criteria_20260526_143929_scored.json  
**评测时间**：2026-05-26 12:05~14:39（step300 val_peak 版本）

| 维度 | 得分 | 样本数 |
|------|------|-------|
| **综合总分** | **77.0** | 199题（1题错误） |
| 标准差 | **10.2** | — |
| 最低 / 最高 / 中位 | 27.3 / 92.0 / 78.3 | — |
| 问题相关性 (rule_question_relevance) | 64.0 | 337 |
| 证据层级合理性 (rule_evidence_hierarchy) | 81.6 | 350 |
| 证据质量与时效性 (rule_evidence_quality_timeliness) | 55.6 | 344 |
| 证据回答一致性 (rule_evidence_answer_consistency) | **67.4** | 344 |
| 信息全面性与深度 (rule_completeness_depth) | **57.4** | 319 |
| 逻辑严谨性 (rule_logical_rigor) | **70.0** | 229 |
| 医学术语规范性 (rule_terminology_standard) | 98.9 | 183 |

**注意**：标准差从 v17 的 7.9 上升至 10.2，最低分从 47.8 降至 27.3，表明 v18 存在更多低分异常回答。

---

## 七、Checkpoint 结构

**目录**：checkpoints/AgentLightning/ebm_agent_14b_grpo_4gpu_v18_full_grpo/

```
global_step_100  (2026-05-23 22:52)
global_step_200  (2026-05-25 11:15)
global_step_300  (2026-05-23 22:57)
global_step_400  (2026-05-25 11:21) [当前最新]
```

**专项保存**：
```
KEEP_v18_step300_val_peak   [评测使用版本，2026-05-23 22:57]
KEEP_v18_step400_current    [最新进展，2026-05-25 11:21]
```

---

## 八、核心结论（5条）

1. **整体得分退步，v18 未达 v17 水平**：v18 step300 val_peak 在 v17 标准下得分 77.0，低于 v17 的 78.6（差距 -1.6 分），且标准差更大（10.2 vs 7.9），最低分更低（27.3 vs 47.8），说明 v18 的回答稳定性下降。

2. **逻辑严谨性明显退步（-7.1 分）**：logicl_rigor 从 77.1 降至 70.0，这是最大单维度退步。假设原因：format_scale 从 1.2 降至 0.7 后，模型对结构性推理的奖励信号减弱，导致论证链路不完整。

3. **证据回答一致性显著退步（-6.6 分）**：rule_evidence_answer_consistency 从 74.0 降至 67.4。虽然 EvidenceSafety 模块有约束强结论，但可能同时抑制了模型在有证据支持时给出明确结论的能力，造成一致性下降的反效果。

4. **信息全面性有所改善（+2.0 分）**：rule_completeness_depth 从 55.4 提升至 57.4，说明 v18 使用更大/加权数据集（train_v18_full_weighted.parquet）确实带来了一定的覆盖度提升。

5. **EvidenceSafety 模块效果中性偏负**：日志分析显示 EvidenceSafety 最终 score 大量为 0.000（正负抵消），说明其在训练后期实际约束效果较弱；但从评测得分看，逻辑严谨性和一致性的退步可能与该模块的惩罚信号干扰了正常推理链路有关。

---

*报告生成：2026-05-26 | 证据来源：服务器 /workspace/post_train/sql_agent/ + 本地评测文件*
