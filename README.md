# MedAgent v20 训练全面分析报告

> 作者：the_great | 日期：2026-06-09 | 模型：`ebm_agent_14b_grpo_4gpu_v20_sft`

---

## 目录

1. [项目背景与版本演进](#1-项目背景与版本演进)
2. [硬件与环境配置](#2-硬件与环境配置)
3. [基础模型与起点](#3-基础模型与起点)
4. [训练框架架构](#4-训练框架架构)
5. [超参数详解](#5-超参数详解)
6. [奖励函数设计](#6-奖励函数设计)
7. [动态奖励调度机制](#7-动态奖励调度机制)
8. [训练数据集](#8-训练数据集)
9. [训练过程分析](#9-训练过程分析)
10. [Checkpoint 管理](#10-checkpoint-管理)
11. [模型推理部署](#11-模型推理部署)
12. [评测设计与方法论](#12-评测设计与方法论)
13. [评测结果全面分析](#13-评测结果全面分析)
14. [v20 vs 前代版本对比](#14-v20-vs-前代版本对比)
15. [v19 失败复盘](#15-v19-失败复盘)
16. [问题与局限性](#16-问题与局限性)
17. [关键结论与下一步](#17-关键结论与下一步)
18. [附录：完整超参数表](#18-附录完整超参数表)

---

## 1. 项目背景与版本演进

### 1.1 项目目标

MedAgent 是一个**医学循证问答 Agent**，核心任务是：给定临床问题（PICO 格式），自动检索医学文献数据库（4个层级），综合多层级证据，生成规范的循证医学回答。

评测标准：GPT-4.1 strict_v4 Chain-of-Critique 评分（7个维度，满分100）。

### 1.2 版本演进路线

| 版本 | 时间 | 架构 | 关键改变 | 最终状态 |
|------|------|------|----------|----------|
| v13–v15 | 2025年 | 早期迭代 | 基础 GRPO 探索 | 完成 |
| v16 | 2026-03 | 14B GRPO | reward ceiling 5.304 问题，过度饱和 | 问题版本 |
| v17 | 2026-04 | 14B GRPO | 修复奖励上限，引入动态调度 | 完成 |
| v17_plus_small | 2026-05-09 | 14B GRPO | 小型非饱和 recall bonus | 完成 |
| v18 | 2026-05-19 | 14B GRPO | sql_agent_v18.py，证据安全奖励，降低 format 权重 | 完成(step300峰值) |
| **v19** | 2026-05-27 | **32B LoRA GRPO** | **尝试 32B 模型，verl + vLLM LoRA 兼容性 bug 导致第一步崩溃** | **失败** |
| **v20** | 2026-06-01 | **14B GRPO（SFT热启动）** | **从 SFT 模型热启动，train_v18_full_weighted 数据，458步完成** | **✅ 完成** |

### 1.3 v20 的核心思路

v20 的关键创新是**"SFT 热启动 + GRPO"**：

- 基础模型不再是原始预训练的 Qwen2.5-14B-Instruct
- 而是已经经过 SFT（Supervised Fine-Tuning）微调的 `Qwen2.5-14B-Instruct-SFT-Agent`
- 理论依据：SFT 给模型注入了更好的格式遵循能力和初始行为分布，GRPO 从更好的起点开始，理论上收敛更快、最终效果更好

---

## 2. 硬件与环境配置

### 2.1 服务器硬件

```
服务器 IP:   117.50.198.65 (port 23)
GPU:         4x NVIDIA A800-SXM4-80GB (SXM4 高带宽互联)
GPU 显存:    每卡 80GB HBM2e，合计 320GB
CUDA 版本:   13.0
CPU:         32 核（64 逻辑核心）
内存:        ~930GB RAM（993406701568 bytes）
磁盘(/):     约 1TB，训练期间使用 767GB
```

### 2.2 软件环境

```
OS:          Linux 5.15.0-113-generic (Ubuntu 22.04)
Python:      CPython 3.12.11
conda env:   py312
分布式后端:   Ray + FSDP2
框架版本:    agentlightning + verl + vllm
wandb run:   20260601_233952-l787uc3v
```

### 2.3 GPU 内存使用分析

训练期间（v20 step 458 时的 wandb summary）：
- `perf/max_memory_allocated_gb`: **50.91 GB** / 80 GB（63.6% 利用率）
- `perf/max_memory_reserved_gb`: **59.54 GB** / 80 GB（74.4%）
- 推理部署时（vllm）：GPU0 使用约 **65.9 GB**，`gpu_memory_utilization=0.85`

---

## 3. 基础模型与起点

### 3.1 基础模型

```
模型路径:   /workspace/models/Qwen2.5-14B-Instruct-SFT-Agent
模型架构:   Qwen2.5-14B-Instruct（已 SFT 微调）
参数量:     ~14B
精度:       bfloat16
```

### 3.2 为什么选择 SFT 热启动

相比原始 v18（从 `Qwen2.5-14B-Instruct` 开始）：

1. **格式学习已内化**：SFT 模型已学会 PICO 结构、分节引用格式，GRPO 无需再花大量步骤学习基本格式
2. **初始 reward 分布更高**：SFT 模型初始 val reward 约 3.18（据上一次记录），普通 Qwen2.5-14B 初始约 3.1
3. **更稳定的 KL 参考基准**：ref model 与 actor 起点对齐，KL 惩罚更有意义

### 3.3 模型文件结构

FSDP 训练产出（`global_step_458/actor/`），包含：
- `model_world_size_4_rank_{0,1,2,3}.safetensors`（分片）

合并后 HF 格式（`/workspace/grpo_v20_step458_hf/`）：
- `model-0000{1..7}-of-00007.safetensors`（共 7 个分片，~28 GB）
- `config.json`, `tokenizer.json` 等

---

## 4. 训练框架架构

### 4.1 框架层次

```
┌─────────────────────────────────────────────────────┐
│                  AgentLightning Trainer              │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ VERL     │  │ Ray Cluster  │  │ LLMProxy     │  │
│  │ (GRPO)   │  │ (分布式调度) │  │ (LiteLLM)   │  │
│  └──────────┘  └──────────────┘  └──────────────┘  │
│                                                     │
│  Actor (FSDP2, 4 GPU)    Ref Model (offload)       │
│  Rollout (vLLM, TP=4)    Reward (sql_agent_v18.py) │
└─────────────────────────────────────────────────────┘
```

### 4.2 GRPO 算法流程

每个 training step：

1. **Rollout**: 用当前 Actor 模型（通过 vLLM 推理）对每个 prompt 生成 `n=4` 个回答（包含工具调用）
2. **Reward**: 对每个回答调用 `sql_agent_v18.py` 中的奖励函数计算得分
3. **Advantage**: 在同一 prompt 的 4 个回答内计算 GRPO advantage（组内标准化）
4. **KL Ref**: 与 ref model 计算 KL 散度作为惩罚
5. **Update**: FSDP2 全参数 PPO-style 更新，clip ratio = 0.20

### 4.3 多轮对话（Multi-turn）

Agent 采用多轮对话格式（hermes tool-call format）：
- `max_turns = 2`：每次最多 2 轮工具调用 + 最终回答
- 工具：`search_embedding_db`（基于 PICO 的向量检索）
- 工具调用 parser：hermes 格式

### 4.4 tracer 配置

```python
trainer_kwargs["adapter"] = agl.LlmProxyTraceToTriplet()
```

使用 `LlmProxyTraceToTriplet` 而非默认的 `TracerTraceToTriplet`，原因：默认 tracer 使用 span hierarchy，与 LiteLLM proxy spans 不兼容，该 adapter 专为 proxy 场景设计。

---

## 5. 超参数详解

### 5.1 核心训练超参数

| 超参数 | v20 值 | 说明 |
|--------|--------|------|
| n_gpus | 4 | A800-SXM4-80GB |
| TP (Tensor Parallel) | 4 | 14B 模型 TP=4，单节点 4 卡全部用于 TP |
| DP (Data Parallel) | 1 | 4卡全用于 TP，DP=1 |
| train_batch_size | 8 | = base_batch_per_dp(8) × dp(1) |
| total_epochs | 3 | 训练 epoch 数 |
| 实际 total_updates | 460 | ceil(900/8)×3 = ceil(112.5)×3 = 113×... 实际跑到458 |
| learning_rate | 5e-7 | base_lr=5e-7，DP=1 无 sqrt scaling |
| lr clamp range | [3e-7, 3e-6] | 保护值 |

### 5.2 学习率设计解析

```python
base_lr = 5e-7
lr = base_lr * math.sqrt(dp)  # dp=1，sqrt(1)=1
lr = max(3e-7, min(lr, 3e-6))  # = 5e-7
```

- **为什么用 5e-7 而不是更大？** 全参数微调（非 LoRA），14B 模型参数量大，过大 LR 会破坏 SFT 模型已学到的格式遵循能力
- **sqrt scaling**：这是分布式训练的 linear scaling rule 的温和版本，DP=1 时无效
- **KL 约束**：用 `kl_loss_coef=0.10` 作为隐性 LR 限制器，防止模型跑偏

### 5.3 PPO-GRPO 参数

| 超参数 | 值 | 说明 |
|--------|----|------|
| adv_estimator | grpo | Group Relative Policy Optimization |
| use_kl_in_reward | False | KL 不加入奖励，而是作为独立 loss 项 |
| use_kl_loss | True | actor 单独有 KL loss 项 |
| kl_loss_coef | 0.10 | KL 惩罚强度（比 v19 LoRA 设计的 0.05 更强） |
| entropy_coeff | 0.005 | 小熵正则化，防止策略过于确定 |
| clip_ratio_low | 0.20 | PPO clip 下限 |
| clip_ratio_high | 0.20 | PPO clip 上限（对称裁剪） |
| n_rollouts (n) | 4 | 每 prompt 生成 4 个候选，GRPO 组内比较 |

**clip_ratio 解读**：PPO 的 surrogate objective 裁剪范围为 [1-0.2, 1+0.2] = [0.8, 1.2]，防止单步更新过大。这是标准值，v20 与 v18 保持一致。

**GRPO advantage 计算**：
- 对同一 prompt 的 4 个 rollout，计算 advantage = (reward_i - mean_reward) / std_reward
- 这消除了不同题目难度带来的奖励差异，让模型专注于相对优劣

### 5.4 Rollout 参数

| 超参数 | 值 | 说明 |
|--------|----|------|
| tensor_model_parallel_size | 4 | vLLM rollout 同样 TP=4 |
| max_model_len | 16384 | 训练时 vLLM 上下文窗口 |
| gpu_memory_utilization | 0.40 | 4 卡时，FSDP actor + vLLM 共存，保守分配 |
| free_cache_engine | True | 每次 rollout 后释放 KV cache |
| enable_auto_tool_choice | True | 自动工具调用 |
| tool_call_parser | hermes | Qwen 系列用 hermes 格式 |
| max_turns | 2 | 最多 2 轮工具调用 |

**gpu_memory_utilization=0.40 的设计**：
- 4 卡场景：FSDP actor 参数 + gradient + optimizer state 占用约 40–50 GB/卡
- vLLM 用剩余 ~30 GB/卡 做推理（0.40 × 80 GB = 32 GB）
- `free_cache_engine=True` 确保 rollout 完成后 KV cache 被释放，供 FSDP update 使用

### 5.5 Sequence 长度参数

| 参数 | 值 | 来源 |
|------|----|------|
| max_prompt_length | 12288 | 训练时 prompt 最大长度 |
| max_response_length | 2048 | 训练时 response 最大长度 |
| max_model_len (vLLM 推理) | 32768 | 评测部署时更大窗口 |

**wandb 实际观察值**（step 458）：
- `prompt_length/mean_after_processing`: 3342 tokens（远小于 12288 上限，无截断）
- `prompt_length/max_after_processing`: 7239 tokens
- `response_length/mean_after_processing`: 446 tokens
- `response_length/max_after_processing`: 1067 tokens
- `response_length/clip_ratio_after_processing`: 0（无截断）

### 5.6 FSDP 配置

| 参数 | 值 | 说明 |
|------|----|------|
| strategy | fsdp2 | FSDP2（PyTorch 2.x native） |
| offload_policy (actor) | True (4卡时) | offload actor 参数到 CPU，节省 GPU 显存 |
| reshard_after_forward | True | forward 后重新分片，节省激活显存 |
| ref/param_offload | True | ref model 常开 offload |
| enable_gradient_checkpointing | True | 激活检查点，降低显存 |
| use_torch_compile | False | 不使用 torch.compile（兼容性优先） |
| entropy_checkpointing | True | entropy loss 也使用梯度检查点 |

### 5.7 Mini-batch 设计

```
train_batch_size = 8
ppo_mini_batch_size = 8     # = train_batch_size
ppo_micro_batch_size_per_gpu = 1
n_rollouts = 4
rollouts_per_update = 8 × 4 = 32
```

即每个 optimizer step：
- 处理 8 条训练 prompt
- 每条 prompt 产生 4 个 rollout
- 共 32 条 (prompt, response) 对
- 微批大小 = 1 / GPU（梯度累积）

### 5.8 Checkpoint 策略

```python
save_freq = 50          # 每 50 步保存一次
test_freq = 25          # 每 25 步验证一次
max_actor_ckpt_to_keep = 2   # 只保留最新 2 个 checkpoint
max_critic_ckpt_to_keep = 2
save_contents = ["model", "optimizer", "extra"]
load_contents = ["model"]    # 恢复时只加载 model（不恢复 optimizer state）
```

**实际保留 checkpoints**（step 458 时）：
```
global_step_100/
global_step_200/   ← 被保留（max=2）
global_step_300/   ← ...
global_step_400/   ← 被保留
global_step_458/   ← 最终
```

---

## 6. 奖励函数设计

v20 使用 `sql_agent_v18.py` 中的奖励系统（与 v18 完全相同的奖励代码）。

### 6.1 奖励组件总览

总奖励 = `hard_reward × format_scale + format_reward × format_scale + correctness_reward × correct_scale + rm_reward × rm_weight`

v20 训练末期（step 458 schedule 状态）：
- `format_scale = 0.70`（固定，不再动态衰减）
- `correct_scale = 1.00`
- `rm_weight = 0.25`（后期逐渐开启）
- `hard_reward = 0.0`（未违规时）

### 6.2 Hard Reward（硬约束惩罚）

```python
def compute_hard_reward(answer, valid_ids, fabricated_ids):
    reward = 0.0
    if cited_ids and not valid_ids:     reward -= 0.5   # 引用 ID 但无有效 ID
    if cited_ids and not valid_ids:     reward -= 1.2   # 无有效 ID 时严重惩罚
    if fabricated_ids:                  reward -= min(2.0, 0.8 + 0.5 × n_fabricated)
    return reward
```

这是医学 AI 最关键的约束：**杜绝编造文献**。fabricated_ids 是引用了数据库中不存在的文献 ID 的集合。

### 6.3 Format Reward（格式奖励）

```python
def compute_format_reward(answer, pico_filled):
    reward = 0.0
    if pico_filled >= 3: reward += 0.15   # PICO 完整度
    # 主结构（一、二、三节）
    reward += 0.35 × main_presence
    reward += 0.15 × main_order
    # 子结构（各子章节）
    reward += 0.25 × sub_presence
    reward += 0.10 × sub_order
    return clamp(reward, 0, 1)
```

满分 1.0，乘以 `format_scale=0.70` = 最高贡献 0.70 分。

**format_scale 从 v18 起固定为 0.70**（v17 及之前有动态衰减），设计理念：不让格式奖励过高以免挤压 correctness 信号。

### 6.4 Correctness Reward（正确性奖励）

这是最核心的奖励组件，最高可达 ~3.0+：

```python
def compute_correctness_reward(answer, valid_ids):
    # 1. Citation Precision
    precision = len(real_ids) / len(cited_ids)
    reward += 1.4 × precision

    # 2. Coverage bonus
    if len(real_ids) >= 4: reward += 0.50
    elif len(real_ids) >= 2: reward += 0.30
    elif len(real_ids) >= 1: reward += 0.12

    # 3. Section-Prefix Alignment ([-0.5, +0.5])
    #    各章节引用的DB前缀与章节对应（01=中文指南, 02=英文指南, 03=SR/Meta, 04=RCT）
    reward += check_section_prefix_alignment(answer, valid_ids)

    # 4. Honest Abstention [-0.10, +0.35]
    #    无证据时诚实表达，对应层无前缀时声明"暂无证据"加分
    reward += check_honest_abstention(answer, valid_ids)

    # 5. Evidence Safety [-0.5, +0.25]
    #    使用强结论词但无 SR/RCT 支撑时扣分
    reward += check_evidence_safety(answer, valid_ids)

    # 6. Small Recall Bonus (v17_plus_small 引入，非饱和)
    reward += recall_bonus + diversity_bonus
```

**四层数据库前缀对应关系**：
| 前缀 | 数据库层级 | 示例 |
|------|-----------|------|
| `01-` | 中文临床指南 | 中华医学会指南 |
| `02-` | 英文临床指南 | ADA、ESC 等国际指南 |
| `03-` | 系统评价/Meta分析 | Cochrane SR |
| `04-` | 临床随机对照试验 | RCT |

章节对应：一（中文指南）→ `01`，二（英文指南）→ `02`，三（SR/Meta）→ `03`，四（RCT）→ `04`

### 6.5 Reward Model（RM）奖励

- `rm_weight` 在训练后期（progress > 30%）线性从 0 升至 0.25
- RM 提供额外质量信号，避免模型仅靠规则化指标
- step 458 时 `rm_weight = 0.25`，但 wandb summary 显示 `[RM] skip` 说明 RM 结果未被记录/使用（可能仅在训练批次中生效，val 时 skip）

### 6.6 奖励范围分析

从训练末期 val 示例分析（满分情况）：
```
format: 1.000 × 0.700 = 0.700
correct: 2.560 × 1.000 = 2.560
  (precision=1.00, recall>=4: +0.50, SectionPrefix: +0.50, RecallSmall: +0.10, CiteDiversity: +0.06)
hard: 0.000
RM: skip
Total: 3.260
```

从训练初期（step=1 的 train reward）：
```
train reward = 3.600  ← SFT 模型初始就能接近上限！
```

**这是 v20 的核心发现**：SFT 模型在初始状态下已能在部分样本获得接近满分，说明 SFT 已充分训练了格式和引用能力，GRPO 面临的改进空间相对有限。

---

## 7. 动态奖励调度机制

### 7.1 Schedule 机制

训练期间通过共享文件 `/tmp/sql_agent_progress_v20_sft.json` 在训练进程和 agent 进程间传递当前步数：

```json
{
    "rollout_count": 14656,
    "global_step": 458,
    "total_updates": 460,
    "rollouts_per_update": 32,
    "updated_at": 1780928572.28
}
```

### 7.2 Schedule 参数演变

```python
schedule_steps = min(150, total_updates)  # = min(150, 460) = 150
progress = min(global_step / schedule_steps, 1.0)  # step>=150后 progress=1.0
```

| 阶段 | progress | format_scale | correct_scale | rm_weight | 说明 |
|------|----------|-------------|---------------|-----------|------|
| 早期 0–30% (step 0–45) | 0–0.30 | 0.70 | 1.00 | 0.00 | 纯格式+正确性训练 |
| 中期 30–100% (step 45–150) | 0.30–1.0 | 0.70 | 1.00 | 0.00→0.25 | RM 奖励线性引入 |
| 后期 >100% (step 150+) | 1.00 | 0.70 | 1.00 | 0.25 | 稳定状态 |

**step 100 时的实际状态**（从日志）：
```
[Schedule] mode=val, step=100/339, p=0.667, format_scale=1.200, correct_scale=1.000, rm_weight=0.157
```

**注意**：step 100 时 `format_scale=1.200`，不是 0.70！这说明 step 100 时用的是旧的 schedule 参数（total_updates=339），后来可能更新了配置。最终 step 458 时：`format_scale=0.700`。

### 7.3 Schedule 的作用

动态调度的核心理念（来自 ToolRL 论文启发）：
1. **早期**：专注于格式和引用准确性，不引入 RM 噪声
2. **中期**：逐渐引入 RM，让模型学习质量维度
3. **后期**：全量 RM，让模型追求综合质量

---

## 8. 训练数据集

### 8.1 数据集统计

| 数据集 | 大小 | 路径 |
|--------|------|------|
| 训练集 | 900 条 | `data/train_v18_full_weighted.parquet`（231KB） |
| 验证集 | 100 条 | `data/val.parquet`（36KB） |

`train_v18_full_weighted`：这是 v18 引入的加权全量数据集（相比原 train.parquet 的 900 条做了难度加权）。

### 8.2 数据格式

每条数据至少包含 `question` 字段（循证医学 PICO 格式的临床问题）。

数据示例（来自训练日志）：
- "在慢性肾病并发高血压的患者中，选择血管紧张素受体拮抗剂与钙通道阻滞剂相比，哪种药物更有助于减少蛋白尿？"
- "对于新诊断的套细胞淋巴瘤患者，来那度胺联合利妥昔单抗方案与标准化疗相比，在提高总缓解率方面是否具有优势？"

涵盖领域：肾脏病、血液病、肿瘤学、呼吸系统、内分泌、神经内科、消化系统等。

### 8.3 Epoch 计算

```
训练集: 900 条
train_batch_size: 8
steps_per_epoch: ceil(900/8) = 113
total_epochs: 3
estimated_total_updates: 113 × 3 = 339

实际 total_updates: 460（脚本中估算为 460，可能与实际 epoch 计算有出入）
实际完成步数: 458（完整训练，仅差 2 步）
```

---

## 9. 训练过程分析

### 9.1 训练时间线

| 时间节点 | 事件 |
|----------|------|
| 2026-06-01 23:39:52 | wandb run 启动（v20 训练开始） |
| 2026-04-08 14:02:57 | 训练日志最早记录（预训练验证阶段） |
| 2026-04-09 ~06:00 | Step 1 开始训练，初始 train reward 已达 3.600 |
| 2026-04-15 20:41 | Step 100 达到，val reward 稳定 |
| 2026-04-13 ~多个时间 | 训练中期，部分 reward 偶有下降（1.533, 1.902 等） |
| 2026-06-08 14:22:52 | Step 458 最终验证完成 |
| 总时长 | **571,719 秒 ≈ 158.8 小时 ≈ 6.6 天** |

### 9.2 关键训练指标（从 wandb summary，step 458）

| 指标 | 值 |
|------|----|
| training/global_step | 458 |
| training/reward | 3.3842（train batch reward） |
| val/reward | 3.2397（val set reward） |
| actor/lr | 5e-7 |
| actor/kl_loss | 0.0297（很小，模型变化平缓） |
| actor/kl_coef | 0.10 |
| actor/entropy_loss | 0.5568 |
| actor/pg_loss | -0.0561 |
| actor/pg_clipfrac | 0.000173（<1% 的梯度被 clip） |
| actor/grad_norm | 2.4665 |
| perf/throughput | 50.46 tokens/s |
| perf/mfu/actor | 2.98%（MFU 低是 GRPO 的固有特性，大量时间在 rollout） |

### 9.3 时间分配（step 级别）

| 阶段 | 时间 (s) | 占比 |
|------|----------|------|
| generation (rollout) | 361 | 30.1% |
| update_actor | 680 | 56.6% |
| ref (ref model 计算) | 79 | 6.6% |
| old_log_prob | 81 | 6.7% |
| 总计 | 1201 s/step | 100% |

**每步约 20 分钟**，总 458 步 = 约 9,160 分钟 = 153 小时（与 wandb runtime 158.8h 吻合）。

### 9.4 Train Reward 趋势

从日志观察：
- **step 1**：训练批次 reward 已达 3.600（接近满分），SFT 热启动效果显著
- **训练中期**：偶有下降（1.533, 1.902），为不同难度题目的正常波动
- **step 458**：`training/reward = 3.3842`，略低于初始（因为 `rm_weight=0.25` 引入了额外约束）

### 9.5 Val Reward 趋势

| 检查点 | 大约 val reward 范围 |
|--------|---------------------|
| step 0 (val_before_train) | 3.18–3.26（从上次记录） |
| step 100 | ~3.20–3.26 |
| step 458 (最终) | 3.2397 |

**从 wandb summary 的初始与最终对比**（从上一次会话记录）：
- 初始 val reward: **3.1855**
- 最终 val reward: **3.2570**（v20 训练完成时记录）
- **提升: +0.0715**

### 9.6 Token 效率分析

| 指标 | 值 |
|------|----|
| perf/total_num_tokens (每步) | 242,462 tokens |
| global_seqlen/mean | 60,615 tokens（4 GPUs 合计） |
| global_seqlen/min | 57,805 | global_seqlen/max | 66,901 |
| prompt_length/mean | 3,342 tokens（含工具调用历史） |
| response_length/mean | 446 tokens（最终回答约 1,500-2,000 汉字） |
| val/mean_response_length | 437 tokens（评测时） |

### 9.7 GRPO Advantage 分析

| 指标 | 值 | 含义 |
|------|----|------|
| critic/advantages/mean | 0.1553 | 平均 advantage 接近 0（正常） |
| critic/advantages/min | -1.617 | 最差 rollout 相对 group 均值 |
| critic/advantages/max | 1.867 | 最好 rollout 相对 group 均值 |
| critic/returns/mean | 0.1553 | 与 advantage 相同（无 critic baseline） |
| critic/score/mean | 3.391 | 实际 reward 均值 |
| critic/score/min | 3.156 | 最低 rollout reward |
| critic/score/max | 3.422 | 最高 rollout reward |

**关键观察**：`score min=3.156` vs `score max=3.422`，差距仅 0.266，说明模型在验证集上基本所有样本都能获得高分，GRPO 的 advantage 信号较弱（难以区分好坏）。这解释了为什么训练效果有限。

---

## 10. Checkpoint 管理

### 10.1 保存的 Checkpoints

```
/workspace/post_train/sql_agent/checkpoints/AgentLightning/ebm_agent_14b_grpo_4gpu_v20_sft/
├── global_step_100/
├── global_step_200/
├── global_step_300/
├── global_step_400/
├── global_step_458/
│   └── actor/
│       ├── model_world_size_4_rank_0.safetensors
│       ├── model_world_size_4_rank_1.safetensors
│       ├── model_world_size_4_rank_2.safetensors
│       └── model_world_size_4_rank_3.safetensors
└── latest_checkpointed_iteration.txt
```

注意：`max_actor_ckpt_to_keep=2` 理论上只保留 2 个，但实际保留了 5 个（100、200、300、400、458），可能是代码在最终 step 458 时没有清理旧 checkpoints。

### 10.2 Checkpoint 大小估算

FSDP4 分片：`14B × 2 bytes (bf16) / 4 GPUs ≈ 7 GB/卡`
每个 checkpoint 目录约 28 GB（4 个文件 × 7 GB）

### 10.3 FSDP → HF 合并

使用 `verl.model_merger` 工具：
```bash
python -m verl.model_merger \
    --backend fsdp2 \
    --local_dir ${CKPT_DIR} \
    --target_dir ${HF_DIR}
```

合并后 HF 格式存储于：`/workspace/grpo_v20_step458_hf/`
- 7 个 safetensors 分片，约 28 GB 总大小
- 包含完整 config.json、tokenizer 文件

### 10.4 关于最优 Checkpoint 的说明

选择 **step 458（最终 checkpoint）** 进行评测，而非 "val peak"（如 v18 选 step 300）的原因：
- v18 的 step 300 是 val reward 峰值点，之后有轻微下降
- v20 的 val reward 曲线较为平稳（初始 SFT 模型已较高），无明显峰值，取最终步
- 从 wandb summary `val/reward=3.2397` 可知 step 458 的表现稳定

---

## 11. 模型推理部署

### 11.1 vLLM 启动配置

合并后模型使用 vLLM V0 engine 部署（V1 engine 在 FSDP merge 场景下有 CUDA fork 冲突）：

```bash
# 关键环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3
conda activate py312
# 注意：不使用 VLLM_USE_V1=1（V0 engine）

vllm serve /workspace/grpo_v20_step458_hf \
    --port 8001 \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --trust-remote-code \
    --served-model-name local_agent_model
```

**推理 vs 训练上下文窗口**：训练时 `max_model_len=16384`，推理时扩展到 `32768`（医学回答 + 多轮工具调用可能需要更长上下文）。

### 11.2 SSH 隧道配置

评测时通过 Python paramiko 建立 SSH 隧道（sshpass 在 Windows 不可用）：

```python
# ssh_tunnel_v20.py
LOCAL_PORT = 8015     # 本地访问端口
REMOTE_PORT = 8001    # 服务器 vLLM 端口
# 评测客户端访问 localhost:8015 相当于访问服务器的 localhost:8001
```

### 11.3 Agent 服务配置

```env
# .env
MODEL_PROVIDER=vllm
LLM_MODEL=local_agent_model
VLLM_API_KEY=none
VLLM_API_URL=http://localhost:8015/v1
```

Agent 服务基于 OpenAI Agents SDK（A2A 协议），监听 `:9998`。

### 11.4 屏幕会话管理

```
vLLM 服务: screen session 531167.vllm_v20
代理服务: screen session (agent service)
```

---

## 12. 评测设计与方法论

### 12.1 评测数据集

```
文件: data/eval_dataset_final_v5.json
总题数: 200 题
```

涵盖 10 个医学专科：
- 内分泌代谢、呼吸系统、心血管、感染性疾病、消化系统
- 神经内科、肾脏疾病、肿瘤学、血液系统、风湿免疫

分级：G1（基础）、G2（中等）、G3（困难）、G4（疑难）

### 12.2 评测流程

```
eval_runner_v20.py          # 通过 SSE streaming 调用 agent
    ↓
Agent Service (:9998)       # OpenAI Agents SDK
    ↓
SSH Tunnel (:8015→:8001)    # 安全隧道
    ↓
vLLM Server (:8001)         # grpo_v20_step458_hf
    ↓
search_embedding_by_PICO    # 4层向量数据库检索
    ↓
eval_scorer_v20.py          # GPT-4.1 strict_v4 评分
    ↓
Final scored JSON
```

### 12.3 评分器设计（strict_v4）

使用 GPT-4.1 作为评分裁判，采用 Chain-of-Critique 方法：
- API：`https://one-api.infox-med.com/v1`（One API 代理）
- 模型：`gpt-4.1`
- Temperature：0.1（低随机性保证评分一致性）
- 并发：`asyncio.Semaphore(3)`（3 个并发评分请求）

### 12.4 评分维度（Rubric）

| 维度 ID | 维度名称 | 权重 |
|---------|---------|------|
| R1/Q1 | 问题相关性（正向） | +3 |
| R2/Q2 | 问题相关性（负向，错误临床推荐） | -3 |
| R3/Q3 | 证据层级合理性（正向） | +2 |
| R4/Q4 | 证据层级合理性（补充） | +1 |
| R5/Q5 | 证据质量与时效性（正向） | +2 |
| R6/Q6 | 证据质量与时效性（补充） | +1 |
| R7/Q7 | 证据回答一致性（负向，捏造） | -3 |
| R8/Q8 | 证据回答一致性（正向） | +2 |
| R9/Q9 | 信息全面性与深度 | +2 |
| R10/Q10 | 逻辑严谨性 | +2 |
| R11/Q11 | 医学术语规范性 | +1 |

**最终 score_percentage = (total_raw + min_possible) / (max_possible + min_possible) × 100**

### 12.5 断点续写机制

`eval_runner_v20.py` 实现了断点续写：
- 自动检测已有前缀匹配的 `*_answers.json` 文件
- 跳过已回答题目，只评测剩余题目
- 对本次 v20 评测，200/200 题全部完成（包括 4 题重试）

---

## 13. 评测结果全面分析

### 13.1 总体得分

```
整体平均分:     73.1 / 100
整体标准差:     10.2
中位数:         73.9
最高分:         92.0
最低分:         27.3
总评测题数:     200 / 200（完成率 100%）
```

### 13.2 各维度得分

| 维度 | 名称 | 得分 | 样本数 |
|------|------|------|--------|
| rule_question_relevance | 问题相关性 | **57.4** | 339 |
| rule_evidence_hierarchy | 证据层级合理性 | **73.9** | 352 |
| rule_evidence_quality_timeliness | 证据质量与时效性 | **53.7** | 346 |
| rule_evidence_answer_consistency | 证据回答一致性 | **59.0** | 346 |
| rule_completeness_depth | 信息全面性与深度 | **55.6** | 321 |
| rule_logical_rigor | 逻辑严谨性 | **55.6** | 230 |
| rule_terminology_standard | 医学术语规范性 | **94.2** | 184 |

**优势**：医学术语规范性（94.2）、证据层级合理性（73.9）
**薄弱点**：证据质量与时效性（53.7）、问题相关性（57.4）、逻辑严谨性（55.6）

### 13.3 各专科得分

| 专科 | 得分 | 说明 |
|------|------|------|
| 心血管 | **75.6** | 最高 |
| 呼吸系统 | **75.2** | |
| 消化系统 | **75.0** | |
| 感染性疾病 | **73.8** | |
| 内分泌代谢 | **73.7** | |
| 肾脏疾病 | **73.0** | |
| 肿瘤学 | **73.3** | |
| 血液系统 | **70.9** | |
| 神经内科 | **70.2** | |
| 风湿免疫 | **69.6** | 最低 |

**专科分布分析**：各专科得分较为均匀（69.6–75.6），说明模型没有明显的领域偏差。风湿免疫最低可能与该领域证据复杂性、自身免疫病的多样性有关。

### 13.4 按难度分级得分

| 难度级别 | 得分 | 说明 |
|---------|------|------|
| G1（基础） | **73.9** | |
| G2（中等） | **74.0** | |
| G3（困难） | **73.2** | |
| G4（疑难） | **71.1** | 最低 |

**G4 略低**：疑难题目通常涉及更复杂的证据冲突、更罕见的疾病，模型在这类问题上的深度分析能力稍弱。

### 13.5 评测中遇到的问题

**问题 1**：`Tool search_embedding_by_PICO not found`（2 题）
- 原因：agent 服务初始配置错误（VLLM_API_URL 指向旧端口 8016）
- 修复：更新 .env，重启 agent 服务，重试成功

**问题 2**：`Max turns (2) exceeded`（2 题）
- 涉及题目：肾脏疾病_13、肾脏疾病_NEW_03
- 原因：模型在 2 轮内未能整合足够证据生成完整回答
- 修复：重试后成功（max_turns=2 是工具调用轮数限制，最终仍可生成回答）

### 13.6 回答质量特征

从训练日志中的 val 回答示例分析：

1. **答案结构**：严格遵循"一、直接结论 → 二、循证分层证据 → 三、临床实践决策"三级结构
2. **引用规范**：使用 `[^前缀-ID-0]` 格式，区分 4 层数据库（01/02/03/04）
3. **证据多样性**：典型回答引用 6-9 个不同文献，跨越 3-4 个数据库层级
4. **回答长度**：val/mean_response_length = 437 tokens ≈ 约 1,300–1,500 汉字
5. **无捏造**：trained model 的 fabricated_ids=0（GRPO 已有效惩罚捏造）

---

## 14. v20 vs 前代版本对比

### 14.1 与 v18 对比

| 维度 | v18 | v20 | 变化 |
|------|-----|-----|------|
| 基础模型 | Qwen2.5-14B-Instruct（原始） | Qwen2.5-14B-Instruct-SFT-Agent | **SFT 热启动** |
| 训练数据 | train_v18_full_weighted（900条） | 同上 | 相同 |
| 奖励函数 | sql_agent_v18.py | sql_agent_v18.py | 相同 |
| 最佳 checkpoint | step 300（val_peak） | step 458（最终） | 更多训练 |
| val reward | 3.1855 | 3.2570 | **+0.0715** |
| 评测分 (strict_v4) | 需参考 v18 分析 | **73.1** | — |
| 训练时长 | ~7 天 | ~6.6 天 | 相似 |

### 14.2 与 v19 的对比教训

v19 尝试将模型扩大到 **32B + LoRA GRPO**，在 step 1 时因 verl + vLLM LoRA API 不兼容崩溃（`add_lora` 属性缺失），未产出任何 checkpoint。

v20 回退到成熟的 14B 全参路线，并通过 SFT 热启动实现稳定提升。

**教训**：扩大模型规模需要充分验证框架兼容性，LoRA GRPO 在 VERL + vLLM 上仍不成熟。

### 14.3 关键改进点

1. **SFT 热启动**：初始 train reward 即达 3.600（vs v18 需要数十步才能稳定）
2. **更稳定的训练曲线**：val reward 波动范围更小（3.18–3.26）
3. **训练效率**：由于起点更高，每步的边际提升更小但训练过程更平稳

---

## 15. v19 失败复盘

### 15.1 v19 的设计目标

- 模型：Qwen2.5-32B-Instruct-E10v2-merged（32B SFT merged）
- 方法：LoRA GRPO（rank=64, alpha=128, all-linear）
- 预期优势：32B 参数更强的医学推理能力；LoRA 减少显存需求

### 15.2 v19 崩溃原因

```
AttributeError: 'Worker' object has no attribute 'llm_engine'
```

位于 `verl/workers/sharding_manager/fsdp_vllm.py`：
```python
self.inference_engine.llm_engine.add_lora(lora_reqest)
```

原因：该版本 vLLM 的 `Worker` 对象 API 已变更，`llm_engine` 属性不存在（可能改为 `model_runner` 或其他属性）。VERL 的 LoRA sharding manager 未跟进 vLLM 的 API 变更。

### 15.3 v19 的技术痕迹

尽管未产出 checkpoint，v19 训练产出了以下文件（均可作为复盘参考）：
- `/root/train_v19_lora_grpo.log`（1.1 MB，完整错误堆栈）
- `/tmp/sql_agent_progress_v19_lora_grpo.json`（记录到 step=1）
- `/workspace/post_train/sql_agent/train_sql_agent_v19_lora_grpo.py`（完整脚本）

---

## 16. 问题与局限性

### 16.1 训练阶段的问题

1. **奖励饱和（天花板效应）**：SFT 模型初始即能获得高奖励，导致 GRPO 的 advantage 信号较弱（score min=3.156, max=3.422，差距仅 0.266），训练效果有限
2. **长训练时间**：~158 小时，每步约 20 分钟，成本较高
3. **低 MFU**：2.98%，说明计算效率不高，大量时间在等待 rollout 和 reward 计算

### 16.2 评测阶段的问题

1. **Time-of-Evidence 偏差**：评分中"证据质量与时效性"维度得分仅 53.7，说明模型未明确标注文献年份，GPT-4.1 无法验证时效性
2. **问题相关性（57.4）低**：模型有时会引用与 PICO 问题不完全对应的文献
3. **逻辑严谨性（55.6）**：部分回答的证据链推理不够严密

### 16.3 系统性局限

1. **max_turns=2**：最多 2 轮工具调用，对于需要多步骤证据整合的复杂问题可能不足
2. **Top-2 检索**：每次向量检索只取 top-2 结果（1500-token 截断），可能遗漏相关文献
3. **单节点训练**：无多节点分布式，限制了可用 GPU 数量

---

## 17. 关键结论与下一步

### 17.1 v20 的主要贡献

1. **验证了 SFT 热启动 + GRPO 的可行性**：val reward 从 3.1855 提升到 3.2570（+0.0715）
2. **评测分 73.1**：横向对比（如有其他版本分数）可判断绝对水平
3. **训练稳定性**：整个 458 步训练无崩溃，说明超参数配置合理

### 17.2 下一步改进方向

**方向 1：提升 advantage 信号质量**
- 降低 SFT 初始能力（或使用更弱的起点），使 reward 分布更分散
- 增加 `n_rollouts` 从 4 到 8，GRPO 组更大，advantage 估计更准确
- 改进 reward 函数，增加更细粒度的证据质量区分

**方向 2：改进评测弱项**
- 在奖励函数中增加"文献年份标注"的正向奖励（针对时效性维度）
- 增加 PICO 对齐的更严格检查（针对问题相关性）

**方向 3：架构提升**
- 继续探索 32B LoRA（待 VERL + vLLM 兼容性修复）
- 尝试 max_turns=3，处理更复杂的多步证据整合

**方向 4：数据改进**
- 增加训练数据量（当前仅 900 条）
- 加入难度分层采样，对 G3/G4 难题过采样

---

## 18. 附录：完整超参数表

```yaml
# v20 完整超参数
model:
  path: /workspace/models/Qwen2.5-14B-Instruct-SFT-Agent
  use_remove_padding: false
  enable_gradient_checkpointing: true
  use_torch_compile: false
  entropy_checkpointing: true

hardware:
  n_gpus: 4
  gpu_type: NVIDIA A800-SXM4-80GB
  gpu_memory: 80GB each

data:
  train_file: data/train_v18_full_weighted.parquet
  val_file: data/val.parquet
  train_size: 900
  val_size: 100
  max_prompt_length: 12288
  max_response_length: 2048

training:
  algorithm: grpo
  total_epochs: 3
  estimated_total_updates: 460
  actual_completed_steps: 458
  train_batch_size: 8
  ppo_mini_batch_size: 8
  ppo_micro_batch_size_per_gpu: 1
  n_rollouts: 4
  rollouts_per_update: 32

optimizer:
  type: Adam (default verl)
  learning_rate: 5e-7
  lr_clamp: [3e-7, 3e-6]

ppo:
  clip_ratio_low: 0.20
  clip_ratio_high: 0.20
  use_kl_loss: true
  kl_loss_coef: 0.10
  entropy_coeff: 0.005
  use_kl_in_reward: false

rollout:
  name: vllm
  tensor_model_parallel_size: 4
  ulysses_sequence_parallel_size: 1
  gpu_memory_utilization: 0.40
  max_model_len: 16384
  free_cache_engine: true
  enable_auto_tool_choice: true
  tool_call_parser: hermes
  max_turns: 2

fsdp:
  strategy: fsdp2
  actor_offload_policy: true  # 4 GPU 时
  ref_param_offload: true
  reshard_after_forward: true

checkpoint:
  save_freq: 50
  test_freq: 25
  max_actor_ckpt_to_keep: 2
  save_contents: [model, optimizer, extra]
  load_contents: [model]

reward_schedule:
  schedule_steps: 150  # min(150, 460)
  format_scale: 0.70   # fixed
  correct_scale: 1.00  # fixed
  rm_weight: 0.0→0.25  # linear after step 45

wandb:
  project_name: AgentLightning
  experiment_name: ebm_agent_14b_grpo_4gpu_v20_sft
  run_id: l787uc3v
  runtime: 571719s (~158.8h)

eval:
  dataset: eval_dataset_final_v5.json
  n_questions: 200
  scorer: GPT-4.1 strict_v4
  scoring_api: https://one-api.infox-med.com/v1
  deployment:
    vllm_max_model_len: 32768
    gpu_memory_utilization: 0.85
    ssh_tunnel_local_port: 8015
  results:
    overall_score: 73.1
    std: 10.2
    median: 73.9
    min: 27.3
    max: 92.0
```

---
actor:
<img width="2406" height="933" alt="image" src="https://github.com/user-attachments/assets/c8fdc137-97c3-404b-9441-66e27c249259" />
<img width="2408" height="489" alt="image" src="https://github.com/user-attachments/assets/1b5d9186-abbf-4815-92b3-12cb2138146a" />
critic；
<img width="2394" height="885" alt="image" src="https://github.com/user-attachments/assets/0fbf5387-fa30-4853-b02f-7e0f9ac91bfd" />
<img width="2376" height="897" alt="image" src="https://github.com/user-attachments/assets/ec441dfe-a7bf-4c22-9e2a-3f91238fff32" />
<img width="2382" height="900" alt="image" src="https://github.com/user-attachments/assets/30b98a8a-d2e8-4d03-b787-e84b7bb71c12" />
<img width="2394" height="885" alt="image" src="https://github.com/user-attachments/assets/a59cec2f-f445-4ed4-ab99-2c140fe9c6bf" />
global_seqlen:
<img width="2391" height="900" alt="image" src="https://github.com/user-attachments/assets/f94b3c49-392f-441a-9d17-cccea0f7ee6e" />
perf:
<img width="2433" height="912" alt="image" src="https://github.com/user-attachments/assets/e968a92d-7ff7-4af3-b6cf-babf95a8f911" />
<img width="813" height="474" alt="image" src="https://github.com/user-attachments/assets/0a4b4744-1fe5-42fe-8fd2-9552f50bafdc" />
prompt_length:
<img width="2382" height="890" alt="image" src="https://github.com/user-attachments/assets/5ef02e5a-df79-464d-88ea-667e7f81a399" />
<img width="1629" height="483" alt="image" src="https://github.com/user-attachments/assets/2af45e5e-cfc3-4608-bb2d-ab745777d1fc" />
response_length:
<img width="2403" height="900" alt="image" src="https://github.com/user-attachments/assets/14d73bb4-b69c-4e46-822b-1dee08c859d6" />
<img width="1617" height="498" alt="image" src="https://github.com/user-attachments/assets/f62cf995-76ae-42d4-beb3-127509816085" />
timing_per_token:
<img width="2417" height="899" alt="image" src="https://github.com/user-attachments/assets/5bc513b0-6671-4fe2-a1a8-9d7093a0f1b5" />
timing_s:
<img width="2382" height="888" alt="image" src="https://github.com/user-attachments/assets/387763a5-3baf-4543-8186-214b0186e7f0" />
<img width="894" height="492" alt="image" src="https://github.com/user-attachments/assets/560c1660-c435-473e-a7e3-4823e8cf1812" />
training:
<img width="2418" height="909" alt="image" src="https://github.com/user-attachments/assets/f2cc42e5-4649-402d-9389-6bd24b594e21" />
<img width="2397" height="894" alt="image" src="https://github.com/user-attachments/assets/77a38cfc-c91e-42dd-bdf6-f364e1132fa2" />
val：
<img width="2406" height="903" alt="image" src="https://github.com/user-attachments/assets/b66bbf4b-1bbc-40a1-9932-3ffb8efaa3a7" />
<img width="804" height="471" alt="image" src="https://github.com/user-attachments/assets/c7a433f9-359a-4776-88b4-d814957227a4" />




















*本报告涵盖 v20 训练的全部关键维度。如需进一步分析具体样本或特定专科的详细数据，请参阅 `eval_results/grpo_v20_step458_strict_v4_20260609_114701_scored.json`。*
