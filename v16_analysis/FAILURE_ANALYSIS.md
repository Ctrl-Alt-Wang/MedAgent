# v16 训练失败分析报告

**实验名称**: `ebm_agent_14b_grpo_4gpu_v16`  
**训练时间**: 2026-05-07 02:06 ～ 2026-05-09 01:23（约 47 小时）  
**撰写时间**: 2026-05-09  
**状态**: 手动终止（训练崩溃后未恢复）

---

## 目录

1. [背景与目标](#1-背景与目标)
2. [训练环境与配置](#2-训练环境与配置)
3. [超参数详情](#3-超参数详情)
4. [奖励函数设计](#4-奖励函数设计)
5. [训练过程时间线](#5-训练过程时间线)
6. [数据统计分析](#6-数据统计分析)
7. [失败根因分析](#7-失败根因分析)
8. [wandb 记录](#8-wandb-记录)
9. [对比 v15](#9-对比-v15)
10. [改进建议（v17 方向）](#10-改进建议v17-方向)

---

## 1. 背景与目标

### 任务描述

训练一个**医学循证检索 Agent**，能够接收临床问题，通过调用向量数据库检索工具（`search_embedding_db`）检索相关医学文献，并按照 PICO 框架生成结构化的循证医学回答。

### 模型血统

```
Qwen2.5-14B-Instruct (基础)
        ↓ SFT (run: sft_agent_v3_5epoch, ~2026-04-03)
Qwen2.5-14B-Instruct-SFT-Agent
        ↓ GRPO v14 (到 global_step_336, ~2026-04-19)
grpo_model_hf / grpo-agent-336
        ↓ GRPO v15 (到 ~2026-04-30)
grpo_v15_ckpt250_hf
        ↓ GRPO v16 (本次, 从 SFT-Agent 重新开始)
❌ 训练崩溃
```

**注意**: v16 使用 `/workspace/models/Qwen2.5-14B-Instruct-SFT-Agent` 作为初始权重，而非 v15 的 checkpoint，相当于从 SFT 阶段重新开始 GRPO。

### 训练框架

- **强化学习算法**: GRPO (Group Relative Policy Optimization)
- **框架**: Agent-Lightning + VERL
- **推理引擎**: vLLM (async mode)
- **分布式策略**: FSDP2

---

## 2. 训练环境与配置

| 项目 | 值 |
|------|-----|
| 服务器 | 117.50.198.65 (port 23) |
| GPU | 4× NVIDIA A800-SXM4-80GB |
| CPU | 32 物理核 / 64 逻辑核 |
| 内存 | ~993 GB |
| CUDA | 13.0 |
| OS | Linux 5.15.0-113-generic |
| Python | CPython 3.12.11 (conda env: py312) |
| 初始模型 | `/workspace/models/Qwen2.5-14B-Instruct-SFT-Agent` (28GB, 6-shard safetensors) |
| 训练数据 | `data/train.parquet`（900 条医学问答） |
| 验证数据 | `data/val.parquet`（100 条） |
| 启动命令 | 见 `launch_command.sh` |

### 启动命令

```bash
screen -dmS train_v16 bash -c "
  source /usr/local/miniconda3/bin/activate py312
  export VLLM_USE_V1=1
  export RAY_DEBUG=legacy
  export HYDRA_FULL_ERROR=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export WANDB_BASE_URL=http://103.139.212.228:3005
  export WANDB_API_KEY=local-f2ca8cd44276ac92ca0a2c12641a6902beb6847d
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  export RAY_memory_monitor_refresh_ms=0
  unset RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES
  export VLLM_NCCL_SO_PATH=.../libnccl.so.2
  cd /workspace/post_train/sql_agent
  python train_sql_agent.py \
    --n-gpus 4 \
    --model /workspace/models/Qwen2.5-14B-Instruct-SFT-Agent \
    --train-file data/train.parquet \
    --val-file data/val.parquet \
    --total-epochs 3 \
    --save-freq 50 \
    --test-freq 25 \
    2>&1 | tee /tmp/train_v16.log
"
```

---

## 3. 超参数详情

### 自动计算参数（4 GPU 时）

| 参数 | 值 | 说明 |
|------|----|------|
| TP (张量并行) | 4 | `min(n_gpus, 4)` |
| DP (数据并行) | 1 | `max(1, n_gpus // tp)` |
| train_batch_size | 8 | `base_batch_per_dp(8) × dp(1)` |
| **learning_rate** | **1.50e-6** | `base_lr(1.5e-6) × sqrt(dp=1)`, clamped [8e-7, 3e-6] |
| FSDP offload | True | 4 卡时开启 |
| gpu_memory_utilization | 0.40 | 4 卡时 0.40，8 卡时 0.45 |

### 手动指定参数

| 参数 | 值 |
|------|----|
| total_epochs | 3 |
| save_freq | 50 steps |
| test_freq | 25 steps |
| schedule_steps | 150（默认 `min(150, total_updates=339)`） |
| total_updates (预计) | 339（`ceil(900/8) × 3`） |
| rollouts_per_update | 32（`train_batch_size(8) × n_rollouts(4)`） |

### GRPO 核心参数

| 参数 | 值 |
|------|----|
| adv_estimator | grpo |
| n_rollouts | 4 |
| ppo_epochs | 1 |
| ppo_mini_batch_size | 8 |
| ppo_micro_batch_size_per_gpu | 1 |
| clip_ratio_low | 0.20 |
| clip_ratio_high | 0.28 |
| **kl_loss_coef** | **0.02** |
| entropy_coeff | 0.01 |
| grad_clip | 1.0 |
| use_kl_in_reward | False |
| norm_adv_by_std_in_grpo | True |

### vLLM / Rollout 参数

| 参数 | 值 |
|------|----|
| max_prompt_length | 8192 |
| max_response_length | 2048 |
| max_model_len (vLLM) | 16384 |
| rollout temperature | 1.0 |
| val temperature | 0（greedy） |
| multi_turn.max_turns | 2 |
| multi_turn.format | hermes |
| max_tool_response_length | 256 |

### 优化器参数

| 参数 | 值 |
|------|----|
| optimizer | AdamW |
| lr | 1.50e-6 |
| weight_decay | 0.01 |
| warmup_style | constant |
| lr_warmup_steps | -1（无 warmup） |
| min_lr_ratio | 0.0 |

---

## 4. 奖励函数设计

### 总体公式

```
total_reward = hard_reward
             + format_scale × format_reward
             + correct_scale × correctness_reward
             + rm_weight × rm_score
```

最终 clamp 到 `[-6.0, 6.0]`。

### 动态 Schedule

| 参数 | v16 实际值 | 旧设计（已废弃）|
|------|-----------|----------------|
| format_scale | **1.2（固定）** | 2.0 → 1.0（动态衰减） |
| correct_scale | **1.2（固定）** | 2/3 → 1.0（动态增长） |
| rm_weight | **0.0（始终为 0）** | 训练后期逐步增大 |
| schedule_steps | 150 | — |

**关键问题**: `rm_weight` 全程为 0，外部 Reward Model（`http://117.50.48.176:8400/score`）从未被调用（日志始终显示 `[RM] skip`）。这意味着奖励信号缺少语义质量维度，只有格式和引用正确性。

### 子项详解

#### hard_reward（工具调用合规性）

```python
def compute_hard_reward(...):
    reward = 0.0
    if not tool_called:
        reward -= 0.5          # 未调用工具
    if fabricated_ids:
        reward -= 1.2          # 有幻觉引用
        reward -= min(2.0, 0.8 + 0.5 * len(fabricated_ids))  # 按数量加重
    return reward
```

| 情况 | hard_reward |
|------|-------------|
| 正常调用工具，无幻觉 | 0.0 |
| 未调用工具 | -0.5 |
| 1 个幻觉 ID | -2.0 |
| 多个幻觉 ID | 最低 -3.2 |

#### format_reward（答案格式质量，范围 0~1.0）

```
format_reward =
    PICO 填写：+0.15 / +0.10 / +0.05（按填写字段数分段）
  + 主章节存在率 × 0.35
  + 主章节顺序率 × 0.15
  + 子章节存在率 × 0.25
  + 子章节顺序率 × 0.10
```

clamp 到 `[0.0, 1.0]`。  
v16 中模型始终输出完整格式，**format_reward 几乎锁定在 1.0**。

#### correctness_reward（引用正确性，范围 -3.0~5.0）

```
correctness_reward =
    1.4 × precision（引用 ID 精确率）
  + recall 梯度奖励（+0.5 / +0.30 / +0.12，按 recall 区间）
  + recall_bonus（高 recall 额外奖励）
  + diversity_bonus（引用文献多样性）
  + check_section_prefix_alignment()（章节前缀对齐）
  + check_honest_abstention()（诚实不确定性）
  - 0.05 × 多余引用数
  - 0.20（调用了 search 但无检索结果）
```

clamp 到 `[-3.0, 5.0]`。  
v16 中模型几乎锁定输出：`precision=1.0, recall=完整` → **correctness_reward 固定在 3.420**。

#### rm_score（外部 Reward Model）

```python
REWARD_MODEL_URL = "http://117.50.48.176:8400/score"
# 返回 raw_score，经 sigmoid 归一化
z = (raw_score - (-3.4)) / 8.0
score = 1 / (1 + exp(-z))  # 范围 (0, 1)
```

**v16 全程 rm_weight=0.0，该项完全未生效**，相当于只用了结构化指标训练，缺少对答案语义质量的评估。

### 奖励天花板分析

在 v16 配置下（format_scale=1.2, correct_scale=1.2, rm_weight=0.0）：

```
理论最大值 = 0 + 1.2×1.0 + 1.2×5.0 = 1.2 + 6.0 = 7.2 → clamp → 6.0
实际观测最大值 = 0 + 1.2×1.0 + 1.2×3.420 = 1.2 + 4.104 = 5.304
```

**5.304 是模型找到的稳定行为边界**，在这个点上：
- PICO 4/4 字段填写
- 格式章节完整（format=1.0）
- 检索 7~9 条文献，precision=1.0，recall 适度
- correctness=3.420（精确率满分 + 部分 recall 奖励）

---

## 5. 训练过程时间线

### 阶段一：初始饱和（Step 0 ～ Step 120）

- **开始时间**: 2026-05-07 02:06
- **现象**: 训练第 1 个样本就打出 5.304，此后绝大多数样本持续满分
- **原因**: SFT-Agent 模型已经具备完善的工具调用和格式能力，初始权重对 v16 reward 函数已经过拟合
- **GRPO 效果**: 所有样本 reward 相近，advantage 函数标准化后接近 0，实际上没有有效的学习信号

**Step 0 典型日志**:
```
[Schedule] step=0/339, format_scale=1.200, correct_scale=1.200, rm_weight=0.000
[PICO] filled=4/4
[Citations] valid_ids=7, cited_ids=7, fabricated=0
[Reward Breakdown] hard=0.000, format=1.000 x 1.200, correct=3.420 x 1.200, subtotal=5.304
[RM] skip
[Total Reward] 5.304
```

### 阶段二：训练崩溃（Step ～120 ～ Step ～160）

- **崩溃时间**: 约 2026-05-08 上午（从 wandb 图表推算）
- **现象**:
  - `training/reward` 从 ~5 暴跌到 -1，剧烈震荡
  - `training/n_triplets` 从 ~60 骤降到 ~10
  - `training/n_triplets_prompt_too_long` 出现大尖峰
- **机制**: 长期的近零梯度信号导致模型权重在某一更新步骤被推离稳定区域，工具调用格式开始出现大量错误（reward=-1 是格式错误或工具调用失败的标志）

**崩溃前后对比**:
```
# 崩溃前（正常）
[train] reward = 5.304
[train] reward = 5.304
[train] reward = 5.304

# 崩溃后
[train] reward = -1.000
[train] reward = -1.000
[Total Reward] 3.926  ← 极不稳定
[train] reward = 5.304
[train] reward = -1.000
```

### 阶段三：部分恢复但未稳定（Step ～160 之后）

- 崩溃后 reward 有所回升，但波动极大（-1 与 5.304 交替出现）
- `n_triplets` 虽有所回升但未恢复到崩溃前水平
- 继续运行没有实质意义，于 2026-05-09 01:23 手动终止

### 终止时状态

- **训练日志大小**: 349MB（`/tmp/train_v16.log`）
- **运行时长**: ~47 小时
- **已完成步数**: ~160 步（预计 339 步，完成 47%）
- **有效 checkpoint**: 无（v16 的 checkpoint 目录为 `ebm_agent_14b_grpo_4gpu_v16`，save_freq=50，崩溃前可能在 step 50/100 各保存一次，但模型质量与 SFT-Agent 无本质差异）

---

## 6. 数据统计分析

### v16 Reward 分布

| reward 值 | 出现次数 | 占比 |
|-----------|---------|------|
| **5.304**（满分上限） | **7,718** | **77.3%** |
| 5.1xx～5.2xx（正常范围） | ~1,000 | ~10% |
| 0.900 | 60 | 0.6% |
| 1.380～1.440 | ~42 | 0.4% |
| 0.535 | 22 | 0.2% |
| 低值（<0） | — | — |
| **-1.000（工具调用失败）** | **1,514（独立计数）** | ~13% |
| **总 [Total Reward] 样本数** | **9,980** | — |

### v15 对比（健康训练的分布）

| reward 值 | 出现次数 | 特征 |
|-----------|---------|------|
| 4.656（主导值） | 5,454 | ~55%，比 v16 低 0.65，说明模型仍有提升空间 |
| 5.1xx～5.2xx | 广泛分布 | 每个值出现 130～162 次，分布均匀 |
| 满分锁定现象 | 无 | 最高值约 5.204，无绝对天花板 |

**结论**: v15 的 reward 分布健康（均匀覆盖 4.6～5.2），v16 严重偏斜（77% 满分）。

### 日志文件大小

| 文件 | 大小 | 训练时间 |
|------|------|---------|
| `/tmp/train_v14.log` | 811MB | ~2026-04-19 |
| `/tmp/train_v15.log` | 1.2GB | ~2026-04-30 |
| `/tmp/train_v16.log` | 349MB | 2026-05-07 ～ 05-09 |

---

## 7. 失败根因分析

### 根因一：模型起点与 Reward 函数的能力错配（主因）

**问题描述**:  
v16 使用 `Qwen2.5-14B-Instruct-SFT-Agent` 作为初始权重——这个模型已经经过专项 SFT（`sft_agent_v3_5epoch`，5 epoch 监督微调），具备了完善的工具调用、PICO 填写和格式输出能力。对于 v16 的 reward 函数（格式+引用正确性），该模型**开箱即达上限**。

**证据**:
- 第 1 步就输出 5.304（满分）
- PICO 全部 4/4 填写
- Citations precision=1.0，fabricated=0
- format_reward=1.0

**本质**: 用一个已经"毕业"的模型去做 GRPO，相当于让一个满分学生去做他已经掌握的练习题，没有学习空间。

### 根因二：Reward 函数粒度不足，无法区分"满分"内的质量差异

**问题描述**:  
当前 reward 函数上限 5.304 对应的行为模式是：precision=1.0 + 适度 recall + 完整格式。这个上限过低，且一旦达到就无法区分"引用更精准"、"回答更深入"、"检索策略更优"等差异。

**具体表现**:
```
correct=3.420（固定不变）= 1.4×1.0(precision) + 部分recall奖励 + 对齐分
```
无论模型如何改进检索策略或答案质量，只要格式和引用不出错，就是 3.420，没有更高的分数可以追求。

### 根因三：Reward Model 未启用，缺少语义质量信号

**问题描述**:  
代码中设计了外部 Reward Model（`http://117.50.48.176:8400/score`）用于评估答案语义质量，但 v16 的 `rm_weight` 始终为 0，全程 `[RM] skip`。

**影响**:  
纯结构化指标（格式+引用ID正确性）无法评估"这个答案是否真正回答了问题"、"推荐的治疗方案是否合理"等医学内容质量，导致模型可以通过检索任意 9 条相关文献并完整格式化就拿到满分，而不需要真正"理解"问题。

### 根因四：GRPO 自身机制在满分饱和时失效

**问题描述**:  
GRPO 的优势函数计算为：
```
advantage_i = (r_i - mean(r)) / std(r)
```
当所有样本 `r_i ≈ 5.304` 时，`std(r) → 0`，`advantage → NaN 或极大值`。虽然代码有 `norm_adv_by_std_in_grpo=True` 的保护，但近零方差时数值稳定性极差，梯度更新变得不可预测，最终在某个步骤触发崩溃。

### 根因五：KL 约束强度不足（潜在因素）

**问题描述**:  
`kl_loss_coef=0.02` 是一个较小的值，对模型偏离参考策略的约束相对宽松。在有效学习信号缺失的情况下，模型权重可以在随机方向上漂移，最终偏离了能够稳定输出格式的区域，导致 Step ~120 的崩溃。

---

## 8. wandb 记录

### v16 相关 Runs

| Run ID | 开始时间 | 状态 |
|--------|---------|------|
| `run-20260507_011837-dkbckcl0` | 2026-05-07 01:18 | 先期测试，很快结束 |
| `run-20260507_020604-kw3d3jpz` | 2026-05-07 02:06 | **v16 主训练 run** |

### 关键 Metrics 图表（来自 wandb 截图）

| 指标 | 观测现象 | 正常应该 |
|------|---------|---------|
| `training/reward` | Step 0 起锁定 ~5，Step ~120 崩溃至 -1 | 从低到高缓慢上升 |
| `training/n_triplets` | 稳定在 60，Step ~120 后暴跌至 10 | 稳定在高值 |
| `training/n_triplets_prompt_too_long` | Step ~120 后出现大尖峰 | 接近 0 的稳定低值 |
| `training/n_truncated_triplets` | 全程 0 | — |
| `training/n_rollouts_w_trace` | 全程固定 32 | — |
| `training/n_triplets_dropped_remainder` | 全程噪声（0~7） | — |

### 服务器 wandb 存放路径

```
/workspace/post_train/sql_agent/wandb/
├── run-20260507_011837-dkbckcl0/
│   └── files/wandb-metadata.json
└── run-20260507_020604-kw3d3jpz/
    └── files/wandb-metadata.json
```

wandb summary 文件因训练异常终止未完整写入（config.yaml 和 wandb-summary.json 缺失）。

---

## 9. 对比 v15

| 维度 | v15 | v16 |
|------|-----|-----|
| 初始模型 | grpo-agent-336（已经过 GRPO v14）| SFT-Agent（未经 GRPO 训练） |
| 初始 reward | ~4.656（有提升空间） | 5.304（立即满分） |
| Reward 满分率 | ~0% | 77.3% |
| Reward 分布 | 均匀覆盖 4.6～5.2 | 严重偏斜至 5.304 |
| n_triplets | 稳定 60 | 初期 60，Step 120 后崩至 10 |
| 训练完成 | 正常完成 | 崩溃，手动终止 |
| 日志大小 | 1.2GB（正常训练时长） | 349MB（提前终止） |
| 最终 checkpoint | grpo_v15_ckpt250_hf（已转 HF 格式） | 无有效 checkpoint |

**关键差异**: v15 从 grpo-agent-336 出发，模型已经过一轮 GRPO 调整，起点 reward (~4.656) 低于 v16 reward 上限，因此有真实的学习空间。v16 错误地从 SFT-Agent 重新开始，直接进入饱和状态。

---

## 10. 改进建议（v17 方向）

### 建议 1：使用正确的初始权重（最高优先级）

使用 `grpo_v15_ckpt250_hf` 或其他已经过多轮 GRPO 训练的 checkpoint 作为起点，而非回退到 SFT-Agent。

```bash
--model /workspace/grpo_v15_ckpt250_hf
# 或指定最优 checkpoint
--model /workspace/post_train/sql_agent/checkpoints/AgentLightning/ebm_agent_14b_grpo_4gpu_v15/global_step_XXX
```

### 建议 2：提升 Reward 函数上限（重要）

当前上限 5.304 是由 `correct=3.420`（精确率满分时的上限）决定的。改进方向：

- **提高 correctness_reward 上限**：增加对"检索策略质量"的评分（如 PICO 匹配相关性、检索词多样性）
- **启用 Reward Model**：设置 `rm_weight > 0`（如 0.3），让外部 RM 评估答案语义质量
- **细化 precision 评分**：当前 precision=1.0 是单层满分，可以引入检索文献的相关性权重

### 建议 3：启用动态 schedule

将 `format_scale` 和 `correct_scale` 恢复为动态变化，使训练早期侧重格式、后期侧重语义质量：

```python
# 建议恢复动态 schedule
format_scale = 2.0 → 0.8  # 随 progress 衰减
correct_scale = 0.8 → 1.5  # 随 progress 增长
rm_weight = 0.0 → 0.4      # 后期启用 RM
```

### 建议 4：增大 KL 约束（防止崩溃）

将 `kl_loss_coef` 从 0.02 提高到 0.05～0.1，在模型偏离参考策略时提供更强约束：

```python
"kl_loss_coef": 0.05,  # 原 0.02
```

### 建议 5：增加 clip ratio 强度

适当降低 `clip_ratio_high`（从 0.28 到 0.20），限制每次更新的步长，减少崩溃风险：

```python
"clip_ratio_low": 0.15,   # 原 0.20
"clip_ratio_high": 0.20,  # 原 0.28
```

### 建议 6：使用更难的训练数据

900 条训练数据对 SFT-Agent 来说过于简单。建议：
- 筛选 SFT-Agent 答错或得分低的样本
- 增加需要多跳检索的复杂问题
- 引入对抗样本（格式正确但内容错误的干扰样本）

### 建议 7：添加 reward 方差监控

在训练脚本中添加对 reward 标准差的监控，当 `std(batch_rewards) < threshold` 时提前预警：

```python
if np.std(batch_rewards) < 0.1:
    logger.warning("[WARNING] Reward variance too low, possible saturation!")
```

---

## 附录

### 文件索引

```
v16_analysis/
├── FAILURE_ANALYSIS.md              # 本文件
├── code/
│   ├── train_sql_agent.py           # v16 训练脚本（完整）
│   ├── sql_agent.py                 # Agent + Reward 函数（完整）
│   └── tools.py                     # 工具定义
├── logs/
│   ├── v16_reward_lines.txt         # v16 全部 reward 行提取（~3.8MB）
│   ├── v16_log_head.txt             # v16 日志前 500 行（启动配置）
│   ├── v16_log_tail.txt             # v16 日志最后 300 行（终止状态）
│   ├── v16_reward_breakdown_samples.txt  # reward 细分样本
│   ├── reward_statistics.txt        # v16/v15 reward 分布统计
│   ├── checkpoint_info.txt          # checkpoint 目录结构
│   └── agentops_tail.txt            # AgentOps 日志尾部
└── wandb/
    ├── all_runs_summary.txt         # 所有 wandb runs 汇总
    └── run-20260507_*/
        └── wandb-metadata.json      # Run 元数据（GPU、环境信息）
```

### 原始日志位置（服务器）

| 文件 | 路径 | 大小 |
|------|------|------|
| v16 训练日志 | `/tmp/train_v16.log` | 349MB |
| v15 训练日志 | `/tmp/train_v15.log` | 1.2GB |
| v14 训练日志 | `/tmp/train_v14.log` | 811MB |
| SQL Agent 训练日志 | `/workspace/post_train/sql_agent/sql_agent_training.log` | ~250MB |
| AgentOps 日志 | `/workspace/post_train/sql_agent/agentops.log` | ~84KB |
