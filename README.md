# MedAgent v17 — GRPO 训练分支

**实验名称**: `ebm_agent_14b_grpo_4gpu_v17_plus_small`  
**训练时间**: 2026-05-10 02:46 ～ 2026-05-15 11:38（约 5 天 9 小时）  
**基础模型**: Qwen2.5-14B-Instruct-SFT-Agent  
**评测得分**: **78.6 / 100**（200题，GPT-4.1 Chain-of-Critique 评分）  
**状态**: 正常完成（335/339 步，98.8%）

---

## 项目简介

本仓库记录医学循证检索 Agent（MedAgent）的 GRPO 强化学习训练过程。该 Agent 接收临床 PICO 问题，通过向量数据库检索医学文献，生成包含证据层级说明和文献引用的结构化回答。

**v17 分支**是在 v16 崩溃后的重启训练，通过提高 KL 散度惩罚（`kl_loss_coef=0.08`）和固定 `format_scale=1.2` 成功完成了完整的 3 个 Epoch 训练。

---

## 模型血统

```
Qwen2.5-14B-Instruct（基础预训练）
        ↓ SFT（sft_agent_v3_5epoch，~2026-04-03）
Qwen2.5-14B-Instruct-SFT-Agent
        ↓ GRPO v14（global_step_336，~2026-04-19）
ebm_agent_14b_grpo_4gpu / global_step_336
        ↓ GRPO v15（ckpt250，~2026-04-30）
grpo_v15_ckpt250_hf
        ↓ GRPO v16（从 SFT-Agent 重启，Step ~160 崩溃）
❌ 训练崩溃
        ↓ GRPO v17（从 SFT-Agent 重启，解决崩溃问题）
✅ ebm_agent_14b_grpo_4gpu_v17_plus_small（本次实验）
```

---

## 训练配置

| 参数 | 值 |
|------|----|
| 基础模型 | Qwen2.5-14B-Instruct-SFT-Agent |
| 算法 | GRPO (Group Relative Policy Optimization) |
| 框架 | Agent-Lightning + VERL |
| 推理引擎 | vLLM (async, VLLM_USE_V1=1) |
| 分布式策略 | FSDP2 (Tensor Parallel = 4) |
| GPU 数量 | 4x (TP=4, DP=1) |
| GPU 内存利用率 | 40% |
| 训练集大小 | 900 条（医学 EBM PICO 问题） |
| 验证集大小 | 100 条 |
| 总 Epochs | 3（分两次运行：Epoch 1 + Epoch 2-3 续训） |
| train_batch_size | 8 |
| rollouts_per_query | 4 |
| rollouts_per_update | 32 |
| schedule_steps | 113 |
| total_updates（目标） | 339 |
| total_updates（实际） | 335（98.8%）|
| 学习率 | 8.00e-07 |
| kl_loss_coef | 0.08 |
| max_completion_tokens | 1536 |
| MAX_TURNS | 2 |

---

## 奖励函数设计

```
total_reward = hard_reward
             + format_scale × format_reward
             + correct_scale × correctness_reward
             + rm_weight × rm_score
```

| 组件 | 值 | 说明 |
|------|----|------|
| hard_reward | 0.1 | 成功调用工具即获得 |
| format_scale | **1.2（固定）** | 格式奖励权重，不再动态衰减 |
| correct_scale | 1.0（固定） | 正确性奖励权重 |
| rm_weight | 前 30% 步: 0.0；后 70% 线性升到 0.3 | 奖励模型权重 |
| format_reward | 0～1.0 | PICO 结构、证据等级、引用编号等格式检查 |
| correctness_reward | 0～1.0 | GPT-4.1 评估答案正确性 |
| rm_score | 0～1.0 | 外部奖励模型打分 |

**vs v16 关键差异**：`kl_loss_coef` 从 0.02 提高至 0.08，有效防止策略震荡崩溃。

---

## 代码结构

```
v17 分支根目录/
├── sql_agent.py              # Agent 实现 + 奖励计算（v17 版本）
├── train_sql_agent.py        # 训练主脚本（v17 版本）
├── tools.py                  # 数据库查询工具
├── tools_embedding.py        # 向量检索工具
├── sqlagent_test.py          # 独立测试脚本
├── merge_actor_to_model.sh   # FSDP 权重合并脚本
├── start_vllm.sh             # vLLM 服务启动脚本
├── restart_ray.sh            # Ray 集群重启脚本
├── requirements.txt          # 依赖清单
└── v17_analysis/             # v17 实验分析目录
    ├── TRAINING_REPORT.md    # 完整训练报告（11 节，含统计分析）
    ├── launch_command.sh     # 训练启动命令（已脱敏）
    ├── code/
    │   ├── sql_agent.py      # v17 训练时使用的代码副本
    │   └── train_sql_agent.py
    ├── logs/
    │   ├── v17_log_head.txt        # 训练初始日志（前 200 行）
    │   ├── v17_log_tail.txt        # 训练末尾日志（后 300 行）
    │   ├── v17_resume_logs.txt     # 续训阶段日志
    │   ├── reward_statistics.txt   # Reward / Schedule 原始日志摘录
    │   └── checkpoint_info.txt     # Checkpoint 目录树
    └── wandb/
        └── all_runs_summary.txt    # W&B 两次 run 的摘要
```

---

## 训练过程概述

### 两阶段运行

**第一阶段（Epoch 1，global_step 1～112）**
- 启动时间：2026-05-10 02:46
- 初始 reward 范围：3.60～3.76（含 hard_reward）
- 运行至 global_step 112 后终止（用于续训准备）

**第二阶段（Epoch 2-3，global_step 113～335）**
- 从 global_step_112 checkpoint 恢复
- 续训完成后终止于 global_step 335
- 结束时间：2026-05-15 11:38

### Checkpoint 策略

- 每 10 步保存一次 `data.pt`（optimizer state + progress）
- 仅 `global_step_336` 保存了完整 `actor/` 权重目录
- 最终模型通过 `merge_actor_to_model.sh` 合并为 HuggingFace 格式，部署于 vLLM

---

## 评测结果

使用 `eval_dataset_final_v5.json`（200题，10个医学专科，G1-G4难度分布）评测：

| 指标 | v17 得分 |
|------|---------|
| **综合得分** | **78.6 ± 9.8** |
| 中位数 | 80.0 |
| 信息全面性 | 75.8 |
| 逻辑严谨性 | 80.2 |
| 证据回答一致性 | 79.5 |
| 格式规范性 | 85.4 |
| 临床实用性 | 79.1 |

评分方法：GPT-4.1 Chain-of-Critique（两步评分：先分析缺陷，再按维度打分）

---

## 与前代对比

| 版本 | 综合得分 | 说明 |
|------|---------|------|
| GPT-4.1 baseline | ~75.3 | 参考基线 |
| v14（GRPO） | ~76.2 | 早期 GRPO 实验 |
| v17（本分支） | **78.6** | 稳定完成 3 Epochs |
| v18 | 77.0 | 见 v18 分支 |

---

## 详细分析

完整的训练报告（含 Reward 统计、崩溃分析、v16 对比、v18 建议）见：

👉 [v17_analysis/TRAINING_REPORT.md](v17_analysis/TRAINING_REPORT.md)

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

完整的启动命令（含所有环境变量）见 [v17_analysis/launch_command.sh](v17_analysis/launch_command.sh)。

```bash
# 简化版（需自行配置环境变量）
cd /workspace/post_train/sql_agent
python train_sql_agent.py \
  --n-gpus 4 \
  --model /workspace/models/Qwen2.5-14B-Instruct-SFT-Agent \
  --train-file data/train.parquet \
  --val-file data/val.parquet \
  --total-epochs 3 \
  --experiment-name ebm_agent_14b_grpo_4gpu_v17_plus_small
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

- `sql_agent.py` 中的 `MAX_TURNS=2` 必须与 VERL 配置中的 `max_turns` 保持一致
- `kl_loss_coef=0.08` 是防止崩溃的关键参数（v16 为 0.02，导致崩溃）
- 训练需要至少 4 张 A100/H100 GPU，显存 ≥ 80GB × 4
- 数据文件 `data/train.parquet` 和 `data/val.parquet` 未包含在此仓库中（文件过大）
