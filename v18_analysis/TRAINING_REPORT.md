# v18_full_grpo 训练报告

**实验名称**: `ebm_agent_14b_grpo_4gpu_v18_full_grpo`  
**训练时间**: 2026-05-19 09:10 ～ 2026-05-25 13:34（约 6 天 4 小时）  
**撰写时间**: 2026-05-27  
**状态**: 正常完成（达到 global_step_400）  
**评测得分**: **77.0 / 100**（eval_dataset_final_v5.json，200题，GPT-4.1 Chain-of-Critique，使用 v17 宽松标准）

---

## 目录

1. [背景与目标](#1-背景与目标)
2. [训练环境与配置](#2-训练环境与配置)
3. [超参数详情](#3-超参数详情)
4. [奖励函数设计](#4-奖励函数设计)
5. [训练过程时间线](#5-训练过程时间线)
6. [Reward 数据统计](#6-reward-数据统计)
7. [Checkpoint 管理](#7-checkpoint-管理)
8. [评测结果](#8-评测结果)
9. [与 v17 的对比分析](#9-与-v17-的对比分析)
10. [问题诊断与反思](#10-问题诊断与反思)
11. [附录：文件索引](#11-附录文件索引)

---

## 1. 背景与目标

### 任务描述

与 v17 相同：训练一个**医学循证检索 Agent**，接收临床 PICO 问题，通过向量数据库检索医学文献，生成包含证据层级说明、文献引用编号和诚实性声明的结构化回答。

### 模型血统

```
Qwen2.5-14B-Instruct（基础预训练）
        ↓ SFT（sft_agent_v3_5epoch，~2026-04-03）
Qwen2.5-14B-Instruct-SFT-Agent
        ↓ GRPO v14（到 global_step_336，~2026-04-19）
ebm_agent_14b_grpo_4gpu / global_step_336
        ↓ GRPO v18（从 v14 global_step_336 恢复，本次实验）
✅ ebm_agent_14b_grpo_4gpu_v18_full_grpo（global_step_400 完成）
```

**注意**：v18 从 **v14 的 global_step_336** 恢复训练（而非从 SFT-Agent 重启），相当于在 v14 基础上继续 GRPO，而不是像 v17 那样从 SFT-Agent 冷启动。

### v18 核心改进目标（相对 v17）

1. **更大训练集**：从 900 条扩充至 1836 条（`train_v18_full_weighted.parquet`，加权采样）
2. **EvidenceSafety 模块**：新增对过度强烈主张的惩罚，鼓励证据匹配度更高的表达
3. **降低 format_scale**：从 1.2 降至 **0.700**，减少格式奖励对正确性学习的干扰
4. **降低学习率**：从 8.00e-07 降至 **5.00e-07**，提高训练稳定性
5. **更多训练步数**：总 updates 从 339 增至 460（2 个 epoch × 1836 样本）

---

## 2. 训练环境与配置

| 项目 | 详情 |
|------|------|
| 服务器 | 云 GPU 服务器（4x A100/H800 80GB） |
| 操作系统 | Ubuntu 22.04 (Docker 容器) |
| Python | 3.12（conda py312） |
| PyTorch | 2.7.1+cu126 |
| VERL | 0.5.0 |
| vLLM | 0.10.1.1（VLLM_USE_V1=1） |
| Agent-Lightning | 最新版 |
| 显卡配置 | CUDA_VISIBLE_DEVICES=0,1,2,3 |
| 内存配置 | PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True |
| RAY | RAY_memory_monitor_refresh_ms=0, RAY_DEBUG=legacy |

---

## 3. 超参数详情

| 参数 | v18 值 | v17 值 | 变化说明 |
|------|--------|--------|----------|
| 基础模型 | Qwen2.5-14B-Instruct-SFT-Agent | 同 | — |
| 起始 checkpoint | ebm_agent_14b_grpo_4gpu/global_step_336 | SFT-Agent（冷启动） | ★ v18 从 v14 恢复 |
| 训练集 | train_v18_full_weighted.parquet | train.parquet | ★ 加权数据集 |
| 训练集大小 | **1836** | 900 | ★ 翻倍 |
| 验证集大小 | 100 | 100 | — |
| 总 Epochs | **2** | 3 | — |
| total_updates（目标） | **460** | 339 | — |
| schedule_steps | **230** | 113 | — |
| train_batch_size | 8 | 8 | — |
| rollouts_per_query | 4 | 4 | — |
| rollouts_per_update | 32 | 32 | — |
| learning_rate | **5.00e-07** | 8.00e-07 | ★ 降低 37.5% |
| format_scale | **0.700（固定）** | 1.2（固定） | ★ 降低 41.7% |
| correct_scale | 1.0 | 1.0 | — |
| rm_weight（峰值） | 最高 0.25 | 最高 0.3 | 略低 |
| gpu_memory_utilization | 0.4 | 0.4 | — |
| TP | 4 | 4 | — |
| DP | 1 | 1 | — |
| MAX_TURNS | 2 | 2 | — |
| max_completion_tokens | 1536 | 1536 | — |

---

## 4. 奖励函数设计

### 总奖励公式

```
total_reward = hard_reward
             + format_scale × format_reward
             + correct_scale × correctness_reward
             + rm_weight × rm_score
             + EvidenceSafety_penalty
```

### 各组件说明

| 组件 | 值 | 说明 |
|------|----|------|
| hard_reward | 0.1 | 成功调用工具即获得 |
| format_scale | **0.700（固定）** | 格式权重（v17 为 1.2） |
| correct_scale | 1.0（固定） | 正确性权重 |
| rm_weight | 前 30% 步: 0.0；后 70% 线性升到 0.25 | 奖励模型权重 |
| format_reward | 0～1.0 | PICO 结构、证据等级、引用编号检查 |
| correctness_reward | 0～1.0 | 引用精度、召回率、多库覆盖度 |
| rm_score | 0～1.0 | 外部奖励模型（Z-score 标准化） |

### EvidenceSafety 模块（v18 新增）

```
[EvidenceSafety] strong=True/False, cautious=True/False, prefixes={...}, score=0.000
```

- **功能**：检测答案中是否出现"强主张"（直接因果断言）与检索到的证据是否匹配
- **strong=True**：答案包含强烈主张语言（如"导致"、"证实"等），触发惩罚检查
- **cautious=True**：答案使用谨慎措辞，无惩罚
- **score**：当 strong=True 且证据支持不足时为负值（惩罚）；本次训练中多数情况 score=0.000
- **实际效果**：训练日志显示大多数样本 score=0.000，模块整体惩罚力度较轻

### Reward 典型样本（末期）

```
[Reward Breakdown] hard=0.000, format=1.000 x 0.700, correct=2.553 x 1.000, subtotal=3.253
[RM Contribution] score=0.587, weight=0.250, contribution=0.147
[Total Reward] 3.400
```

---

## 5. 训练过程时间线

| 时间 | 事件 |
|------|------|
| 2026-05-19 09:10 | 训练启动，从 v14 global_step_336 恢复 |
| 2026-05-19 09:14 | W&B run 开始 (`run-20260519_091431`) |
| 2026-05-19 09:15 | 第一批 validation 完成，初始 reward ≈ 3.23～3.26 |
| 2026-05-23 22:52 | global_step_100 checkpoint 保存 |
| 2026-05-23 22:57 | global_step_300 checkpoint 保存（含 actor/ 权重） |
| 2026-05-25 11:15 | global_step_200 checkpoint 保存 |
| 2026-05-25 11:21 | latest_checkpointed_iteration.txt 更新为 400 |
| 2026-05-25 13:34 | 最后一次 reward 记录，训练结束 |

**总训练时长**：约 6 天 4 小时（2026-05-19 09:10 ～ 2026-05-25 13:34）

---

## 6. Reward 数据统计

### 格式

典型日志行：
```
[Schedule] mode=val, step=0/460, p=0.000, format_scale=0.700, correct_scale=1.000, rm_weight=0.000
[EvidenceSafety] strong=False, cautious=False, prefixes={'02', '01', '04', '03'}, score=0.000
[Reward Breakdown] hard=0.000, format=1.000 x 0.700, correct=2.553 x 1.000, subtotal=3.253
[RM Contribution] score=0.587, weight=0.250, contribution=0.147
[Total Reward] 3.400
```

### 典型 Reward 范围

| 阶段 | format_scale | rm_weight | 典型 total_reward |
|------|-------------|-----------|------------------|
| 初始（step 0） | 0.700 | 0.000 | 3.23～3.26 |
| 末期（step ~400） | 0.700 | 0.25 | 3.35～3.42 |

### 与 v17 对比

v17 初始 reward ≈ 3.60～3.76（format_scale=1.2），v18 初始 ≈ 3.23（format_scale=0.700）。  
降低 format_scale 直接导致初始 total_reward 下降约 0.4，这是预期行为。

---

## 7. Checkpoint 管理

### Checkpoint 目录结构

```
checkpoints/AgentLightning/ebm_agent_14b_grpo_4gpu_v18_full_grpo/
├── global_step_100/
│   └── data.pt                  # optimizer state
├── global_step_200/
│   └── data.pt
├── global_step_300/
│   ├── actor/                   # ★ 完整模型权重（FSDP shards）
│   │   └── [safetensors 分片...]
│   └── data.pt
└── latest_checkpointed_iteration.txt  # 内容: 400
```

**注意**：`latest_checkpointed_iteration.txt` 显示为 400，但目录中仅见 global_step_100/200/300。  
global_step_400 的完整 actor/ 权重已用于模型合并，合并后目录被清理。

### 模型合并与部署

```bash
# FSDP 权重合并为 HuggingFace 格式
bash merge_actor_to_model.sh

# 启动 vLLM 推理服务
bash start_vllm.sh
```

合并自 global_step_300 actor（最后一个保留的完整权重目录）。

---

## 8. 评测结果

使用 `eval_dataset_final_v5.json`（200题，10个医学专科，G1-G4难度分布）评测：

### 综合得分

| 指标 | v18 得分 |
|------|---------|
| **综合得分** | **77.0 ± 10.2** |
| 中位数 | 78.3 |

### 五维度得分

| 维度 | v18 得分 |
|------|---------|
| 信息全面性 | 77.8 |
| 逻辑严谨性 | 73.1 |
| 证据回答一致性 | 72.9 |
| 格式规范性 | 84.0 |
| 临床实用性 | 77.2 |

### 评分方法

- **方法**：GPT-4.1 Chain-of-Critique（两步：先分析缺陷，再按五维度打分）
- **评分标准**：使用 v17 宽松标准（`eval_scorer_v18_v17criteria.py`），确保公平对比
- **评测时间**：2026-05-26

---

## 9. 与 v17 的对比分析

### 综合对比

| 指标 | v17 | v18 | Delta |
|------|-----|-----|-------|
| 综合得分 | 78.6 ± 9.8 | 77.0 ± 10.2 | **-1.6** |
| 中位数 | 80.0 | 78.3 | -1.7 |

### 各维度对比

| 维度 | v17 | v18 | Delta | 趋势 |
|------|-----|-----|-------|------|
| 信息全面性 | 75.8 | 77.8 | **+2.0** | ↑ 改善 |
| 逻辑严谨性 | 80.2 | 73.1 | **-7.1** | ↓ 明显退步 |
| 证据回答一致性 | 79.5 | 72.9 | **-6.6** | ↓ 明显退步 |
| 格式规范性 | 85.4 | 84.0 | -1.4 | → 持平 |
| 临床实用性 | 79.1 | 77.2 | -1.9 | ↓ 轻微退步 |

### 题目级别分析（200题）

| 结果 | 数量 | 占比 |
|------|------|------|
| v18 > v17 | 70 | 35.0% |
| v18 = v17 | 42 | 21.0% |
| v18 < v17 | 88 | 44.0% |

**v18 在 44% 的题目上得分低于 v17，仅在 35% 上高于 v17。**

---

## 10. 问题诊断与反思

### 主要退步分析

**1. 逻辑严谨性 -7.1（最严重退步）**

可能原因：
- 从 v14 checkpoint 恢复（而非 SFT-Agent 冷启动），v14 本身的某些推理习惯被继承
- format_scale 降低 → 格式压力减小 → 逻辑结构变松散
- EvidenceSafety 惩罚过于保守（大多数情况 score=0.000），未能纠正逻辑问题

**2. 证据回答一致性 -6.6（第二大退步）**

可能原因：
- 更大训练集包含难度不均匀的样本（`train_v18_full_weighted.parquet` 加权采样可能引入噪声）
- 训练集翻倍但 epoch 数减少（2 vs 3），每条数据学习次数减半
- EvidenceSafety 的 strong/cautious 判断仍然不精确，未能有效惩罚证据-答案不一致

**3. 信息全面性 +2.0（唯一改善）**

原因：更大训练集（1836 条）提供了更广泛的医学知识覆盖，检索和汇总能力有所提升。

### 设计失误总结

| 失误 | 影响 | 教训 |
|------|------|------|
| 从 v14 恢复而非冷启动 | 继承了 v14 的推理习惯 | 重启时应从 SFT-Agent 冷启动 |
| format_scale 降幅过大（1.2→0.700） | 格式与逻辑约束均减弱 | 小步调整，降幅不宜超过 20% |
| EvidenceSafety 惩罚力度不足 | 大多数样本惩罚为 0 | 需要调整阈值或惩罚强度 |
| 数据加权策略未验证 | 引入采样噪声 | 加权前应先验证数据分布 |
| 2 epoch 配合超大训练集 | 每条数据训练次数不足 | 维持 3 epoch 或适当增加 epoch |

### v19 改进建议

1. **从 SFT-Agent 冷启动**（放弃从 v14 继承）
2. **format_scale 回到 1.2**（或 1.0，小步调整）
3. **完善 EvidenceSafety**：调高 strong claim 检测阈值，增大惩罚权重
4. **保持 3 个 Epoch**（即使数据集扩大）
5. **数据质量优先于数量**：验证 train_v18_full_weighted.parquet 的加权分布

---

## 11. 附录：文件索引

```
v18_analysis/
├── TRAINING_REPORT.md          # 本报告
├── launch_command.sh           # 启动命令（已脱敏）
├── code/
│   ├── sql_agent_v18.py        # 训练时的 Agent 实现（当前服务器版本）
│   └── train_sql_agent_v18.py  # 训练主脚本（v18 版本）
├── logs/
│   ├── v18_log_head.txt        # 训练初始日志
│   ├── v18_log_tail.txt        # 训练末尾日志（含 EvidenceSafety 样本）
│   ├── reward_statistics.txt   # Reward 统计原始日志
│   └── checkpoint_info.txt     # Checkpoint 目录信息
└── wandb/
    └── all_runs_summary.txt    # W&B runs 列表
```

关键评测文件（位于本地评测目录 `eval_sql_agent/`）：

| 文件 | 说明 |
|------|------|
| `eval_scorer_v18_v17criteria.py` | 使用 v17 宽松标准对 v18 打分 |
| `eval_results/grpo_v18_step300_v17criteria_*_scored.json` | 200题打分结果 |
| `report_v18_step300.md` | v18 独立报告 |
| `report_v17_vs_v18_comparison.md` | v17 vs v18 对比报告 |
| `compare_v17_vs_v18.py` | 生成对比图表（8张） |
