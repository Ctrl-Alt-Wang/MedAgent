# MedAgent v14 — Training Record

> Branch: `v14`  
> Experiment name: `ebm_agent_14b_grpo_4gpu`  
> Final checkpoint: `global_step_336`  
> **Evaluation score: 76.2 / 100** (GPT-4.1 strict_v4 rubric, 200-question benchmark)

---

## Training Overview

| Item | Value |
|------|-------|
| Base model | Qwen2.5-14B-Instruct |
| Framework | Agent-Lightning + VERL + GRPO |
| GPUs | 4 × NVIDIA A800-80GB |
| Training dates | 2026-04-08 → 2026-04-19 |
| Total steps | 336 (training complete) |
| Train dataset | `data/train.parquet` (~1000 medical Q&A) |
| Val dataset | `data/val.parquet` |

---

## Hyperparameters (`train_sql_agent.py`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `base_lr` | `1.5e-6` | Effective LR for DP=1: **1.5e-6** (clamped [8e-7, 3e-6]) |
| `n_rollouts` | **4** | Rollouts sampled per prompt per update |
| `max_prompt_length` | **8192** | VERL truncation limit for training context |
| `max_response_length` | 2048 | |
| `train_batch_size` | 8 (DP=1) | `base_batch_per_dp=8 × dp=1` |
| `kl_loss_coef` | **0.02** | KL penalty against reference model (loose) |
| `entropy_coeff` | **0.01** | Entropy bonus to encourage exploration |
| `clip_ratio_low` | 0.20 | PPO clip lower bound |
| `clip_ratio_high` | **0.28** | PPO clip upper bound (asymmetric) |
| `max_model_len` (vLLM) | 16384 | vLLM engine max context |
| `TP` | 4 | Tensor parallelism |
| `save_freq` | 50 steps | |
| `total_epochs` | 3 | |
| `gpu_memory_utilization` | 0.40 | |
| KL offload (ref model) | param_offload=True | |
| Actor checkpoint save | model + optimizer + extra | |

---

## Reward Function (`sql_agent.py`)

### Reward Formula

```
total_reward = hard_reward
             + format_scale  × format_reward      # [0, 1.2]
             + correct_scale × correctness_reward  # [-3.6, 3.0]
             + rm_weight     × rm_score            # [0, 0.3]
```

### Component Weights (fixed during training)

| Component | Scale | Range |
|-----------|-------|-------|
| `format_scale` | **1.2** (fixed) | |
| `correct_scale` | **1.0** (fixed) | format > correctness ratio |
| `rm_weight` | 0 → **0.3** (scheduled) | |

### RM Weight Schedule

```python
if progress < 0.3:        # first 30% of training: RM disabled
    rm_weight = 0.0
else:
    rm_weight = 0.3 * (progress - 0.3) / 0.7   # linearly 0.0 → 0.3
```
RM activates only after 30% training progress, reaching maximum **0.3** at 100%.

### hard_reward

```
-1.0   medical question but no tool called
-0.5   reference list pattern detected in answer tail
-1.2   cited IDs exist but none are valid search results
-0.8 to -2.0   fabricated IDs (hallucinated citations)
```

### format_reward — range [0, 1]

| Sub-component | Max pts |
|---------------|---------|
| PICO fields filled (≥3/4) | 0.15 |
| Main section presence | 0.35 |
| Main section order | 0.15 |
| Sub-section presence | 0.25 |
| Sub-section order | 0.10 |

### correctness_reward — range [-3, 3]

| Sub-component | Value |
|---------------|-------|
| Citation precision (`real/cited`) | `1.4 × precision` |
| ≥4 valid cited IDs bonus | +0.50 |
| ≥2 valid cited IDs bonus | +0.30 |
| ≥1 valid cited ID bonus | +0.12 |
| No citations at all | -0.05 |
| Section-prefix alignment | [-0.5, +0.5] |
| Honest abstention (no evidence) | bonus/penalty |
| Valid IDs exist but none cited | -0.20 |

### RM Score Normalization

```python
z = (raw_score - (-3.4)) / 8.0
rm_score = sigmoid(z)          # maps raw RM score to [0, 1]
```
Center point: `-3.4` (scores above -3.4 map to rm_score > 0.5)

### Observed Reward Distribution (training mid-point, ~step 46)

- Typical reward: **2.1 – 3.6**
- Reward variance: **~1.5** (high, GRPO gradients meaningful)
- rm_weight at step 46: ~0.003 (RM barely active)

---

## Key Observations

- Training **ran to completion** at step 336 — the model fully converged.
- Asymmetric PPO clip (`high=0.28`) allowed larger positive gradient steps than negative ones, facilitating faster adaptation.
- Low KL coeff (0.02) gave the model room to deviate from the reference model and specialize.
- `tools_embedding.py` had **no document truncation** — full search result text was included in training context.
- The RM model at `http://117.50.48.176:8400/score` contributed only modestly (max weight 0.3), keeping the training primarily driven by format/correctness rules.

---

## Files in This Branch

| File | Description |
|------|-------------|
| `train_sql_agent.py` | GRPO training launcher (v14 configuration) |
| `sql_agent.py` | Agent + reward function (v14, committed state) |
| `tools_embedding.py` | 4-way vector search tool (no truncation) |
