# MedAgent v18 — GRPO 训练分支

**实验名称**: `ebm_agent_14b_grpo_4gpu_v18_full_grpo`  
**训练时间**: 2026-05-19 09:10 ～ 2026-05-25 13:34（约 6 天 4 小时）  
**基础模型**: Qwen2.5-14B-Instruct-SFT-Agent（从 v14 global_step_336 恢复）  
**评测得分**: **77.0 / 100**（200题，GPT-4.1 Chain-of-Critique，v17 宽松标准）  
**状态**: 正常完成（global_step_400）

---

## 项目简介

本仓库记录医学循证检索 Agent（MedAgent）的 GRPO 强化学习训练过程。该 Agent 接收临床 PICO 问题，通过向量数据库检索医学文献，生成包含证据层级说明和文献引用的结构化回答。

**v18 分支**在 v17 基础上引入了三项主要改进：
1. **EvidenceSafety 模块**：检测答案中过度强烈的主张，对证据支持不足的强断言施加惩罚
2. **更大训练集**：从 900 条扩充至 1836 条（`train_v18_full_weighted.parquet`，加权采样）
3. **调低 format_scale**：从 1.2 降至 0.700，减少格式奖励对正确性学习的干扰

然而评测显示 v18 综合得分（77.0）略低于 v17（78.6），主要退步在逻辑严谨性（-7.1）和证据回答一致性（-6.6）。详细分析见 [v18_analysis/TRAINING_REPORT.md](v18_analysis/TRAINING_REPORT.md)。

---

## 模型血统

```
Qwen2.5-14B-Instruct（基础预训练）
        ↓ SFT（sft_agent_v3_5epoch，~2026-04-03）
Qwen2.5-14B-Instruct-SFT-Agent
        ↓ GRPO v14（到 global_step_336，~2026-04-19）
ebm_agent_14b_grpo_4gpu / global_step_336
        ↓ GRPO v18（从 v14 global_step_336 恢复）
✅ ebm_agent_14b_grpo_4gpu_v18_full_grpo（global_step_400 完成）
```

---

## 训练配置

| 参数 | v18 值 | 与 v17 的变化 |
|------|--------|--------------|
| 基础模型 | Qwen2.5-14B-Instruct-SFT-Agent | — |
| 起始 checkpoint | v14 global_step_336 | ★ v17 从 SFT-Agent 冷启动 |
| 训练数据集 | train_v18_full_weighted.parquet | ★ 加权采样新数据集 |
| 训练集大小 | **1836 条** | v17: 900 条（翻倍） |
| 验证集大小 | 100 条 | — |
| 总 Epochs | **2** | v17: 3 |
| total_updates（目标） | **460** | v17: 339 |
| schedule_steps | **230** | v17: 113 |
| train_batch_size | 8 | — |
| rollouts_per_query | 4 | — |
| rollouts_per_update | 32 | — |
| 学习率 | **5.00e-07** | ★ v17: 8.00e-07（降低 38%） |
| format_scale | **0.700（固定）** | ★ v17: 1.200（降低 42%） |
| correct_scale | 1.0（固定） | — |
| EvidenceSafety | **新增** | v17 无此模块 |
| kl_loss_coef | 0.08 | — |
| GPU 数量 | 4x（TP=4, DP=1） | — |
| GPU 内存利用率 | 40% | — |
| max_completion_tokens | 1536 | — |
| MAX_TURNS | 2 | — |

---

## 奖励函数设计

```
total_reward = hard_reward
             + format_scale × format_reward
             + correct_scale × correctness_reward
             + rm_weight × rm_score
             + EvidenceSafety_penalty
```

| 组件 | 值 | 说明 |
|------|----|------|
| hard_reward | 0.1 | 成功调用工具即获得 |
| format_scale | **0.700（固定）** | 格式权重（v17 为 1.2） |
| correct_scale | 1.0（固定） | 正确性权重 |
| rm_weight | 前 30%: 0.0；后 70% 线性升至 0.25 | 奖励模型权重 |
| EvidenceSafety | 0～负值 | 强主张惩罚（v18 新增） |

### EvidenceSafety 模块（v18 新增）

该模块检测答案中的"强主张"（直接因果断言），当强主张与检索证据不一致时施加惩罚。  
训练日志示例：
```
[EvidenceSafety] strong=True, cautious=False, prefixes={'02', '01', '04', '03'}, score=0.000
[EvidenceSafety] strong=False, cautious=False, prefixes={'02', '01', '04', '03'}, score=0.000
```
实践中大多数样本 score=0.000，惩罚力度较轻，效果有限。

---

## 代码结构

```
v18 分支根目录/
├── sql_agent.py              # Agent 实现 + 奖励计算（v18 版本）
├── train_sql_agent.py        # 训练主脚本（v18 版本）
├── tools.py                  # 数据库查询工具
├── tools_embedding.py        # 向量检索工具
├── sqlagent_test.py          # 独立测试脚本
├── merge_actor_to_model.sh   # FSDP 权重合并脚本
├── start_vllm.sh             # vLLM 服务启动脚本
├── restart_ray.sh            # Ray 集群重启脚本
├── requirements.txt          # 依赖清单
└── v18_analysis/             # v18 实验分析目录
    ├── TRAINING_REPORT.md    # 完整训练报告（11 节，含诊断分析）
    ├── launch_command.sh     # 训练启动命令（已脱敏）
    ├── code/
    │   ├── sql_agent_v18.py        # v18 训练时使用的代码副本
    │   └── train_sql_agent_v18.py  # v18 训练主脚本
    ├── logs/
    │   ├── v18_log_head.txt        # 训练初始日志
    │   ├── v18_log_tail.txt        # 训练末尾日志（含 EvidenceSafety 样本）
    │   ├── reward_statistics.txt   # Reward 统计原始日志
    │   └── checkpoint_info.txt     # Checkpoint 目录信息
    └── wandb/
        └── all_runs_summary.txt    # W&B runs 列表
```

---

## 训练过程概述

### 关键时间节点

| 时间 | 事件 |
|------|------|
| 2026-05-19 09:10 | 训练启动，从 v14 global_step_336 恢复 |
| 2026-05-19 09:15 | 首批 validation 完成，初始 reward ≈ 3.23～3.26 |
| 2026-05-23 22:57 | global_step_300 checkpoint（含完整 actor/ 权重） |
| 2026-05-25 11:21 | latest checkpoint 更新为 400 |
| 2026-05-25 13:34 | 训练正常结束 |

### Checkpoint 结构

```
checkpoints/AgentLightning/ebm_agent_14b_grpo_4gpu_v18_full_grpo/
├── global_step_100/ (data.pt)
├── global_step_200/ (data.pt)
├── global_step_300/ (actor/ + data.pt)  ← 保留完整权重
└── latest_checkpointed_iteration.txt    ← 内容: 400
```

---

## 评测结果

使用 `eval_dataset_final_v5.json`（200题，10个医学专科，G1-G4难度分布）评测：

### v17 vs v18 对比

| 维度 | v17 | v18 | Delta |
|------|-----|-----|-------|
| **综合得分** | **78.6 ± 9.8** | **77.0 ± 10.2** | **-1.6** |
| 信息全面性 | 75.8 | 77.8 | **+2.0** ↑ |
| 逻辑严谨性 | 80.2 | 73.1 | **-7.1** ↓ |
| 证据回答一致性 | 79.5 | 72.9 | **-6.6** ↓ |
| 格式规范性 | 85.4 | 84.0 | -1.4 → |
| 临床实用性 | 79.1 | 77.2 | -1.9 ↓ |

**结论**：v18 信息覆盖面有所提升，但逻辑严谨性和证据一致性出现明显退步，综合得分低于 v17。

---

## 详细分析

完整的训练报告（含失误诊断、设计反思和 v19 改进建议）见：

👉 [v18_analysis/TRAINING_REPORT.md](v18_analysis/TRAINING_REPORT.md)

---

## 快速上手（复现训练）

### 环境准备

```bash
# Docker 环境（推荐）
docker pull modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.6.3-py311-torch2.7.1-vllm0.10.1.1-modelscope1.29.2-swift3.8.3

# 安装依赖
cd agent-lightning
pip install uv
uv pip install --system --no-cache-dir -e .[dev,agent,apo] \
  fastmcp==2.14.1 openai-agents==0.6.3 vllm==0.10.1.1 verl==0.5.0 \
  'litellm[proxy]>=1.78' 'agentops>=0.4.21' 'openai>=2.0.0'

# 修复 omegaconf 依赖
pip uninstall -y antlr4-python3-runtime
pip install antlr4-python3-runtime==4.9.3
```

### 训练命令

完整的启动命令（含所有环境变量）见 [v18_analysis/launch_command.sh](v18_analysis/launch_command.sh)。

```bash
# 简化版（需自行配置环境变量）
cd /workspace/post_train/sql_agent
python train_sql_agent_v18.py \
  --n-gpus 4 \
  --model /workspace/models/Qwen2.5-14B-Instruct-SFT-Agent \
  --train-file data/train_v18_full_weighted.parquet \
  --val-file data/val.parquet \
  --total-epochs 2 \
  --resume-from checkpoints/AgentLightning/ebm_agent_14b_grpo_4gpu/global_step_336 \
  --experiment-name ebm_agent_14b_grpo_4gpu_v18_full_grpo
```

### 合并权重并部署

```bash
# 合并 FSDP 权重为 HuggingFace 格式
bash merge_actor_to_model.sh

# 启动 vLLM 服务
bash start_vllm.sh
```

---

## 注意事项

- `format_scale=0.700` 导致 format reward 对 total reward 贡献减小，需观察是否影响格式质量
- 从 v14 恢复而非冷启动，模型保留了 v14 的推理倾向，可能是逻辑退步的原因之一
- EvidenceSafety 模块的惩罚阈值需在 v19 中调整，当前版本惩罚力度不足
- 训练需要至少 4 张 A100/H100 GPU，显存 ≥ 80GB × 4
- 数据文件未包含在此仓库中（文件过大）
