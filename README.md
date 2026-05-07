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
- `tools_embedding.py` truncated each retrieved document to **1500 characters** — same truncation used during evaluation, so train/eval is consistent.
- The RM model at `http://117.50.48.176:8400/score` contributed only modestly (max weight 0.3), keeping the training primarily driven by format/correctness rules.

---

## Files in This Branch

| File | Description |
|------|-------------|
| `train_sql_agent.py` | GRPO training launcher (v14 configuration) |
| `sql_agent.py` | Agent + reward function (v14, committed state) |
| `tools_embedding.py` | 4-way vector search tool (1500-char truncation per document) |



actor
<img width="2409" height="801" alt="image" src="https://github.com/user-attachments/assets/6677564a-a82d-441d-83e7-edc3006f208b" />
<img width="2436" height="435" alt="image" src="https://github.com/user-attachments/assets/3e442942-2865-4886-b2e3-53576cee1913" />
critic
<img width="2418" height="789" alt="image" src="https://github.com/user-attachments/assets/76496bcc-1fee-494b-b631-18b755e7b18d" />
<img width="2400" height="792" alt="image" src="https://github.com/user-attachments/assets/1b1b19c9-345d-4212-b30a-ba41044ae3ef" />
<img width="2400" height="795" alt="image" src="https://github.com/user-attachments/assets/785f468f-70ed-4273-af9a-5bd5e453fe91" />
<img width="2403" height="801" alt="image" src="https://github.com/user-attachments/assets/2b65b53c-8011-47a4-9abe-20c7760c6ada" />
global_seqlen
<img width="2418" height="804" alt="image" src="https://github.com/user-attachments/assets/01941f53-7a72-4409-852f-ed0b1fe3f19b" />
perf
<img width="2403" height="795" alt="image" src="https://github.com/user-attachments/assets/c1f2cc97-188e-44f6-9148-556934d1659c" />
<img width="849" height="434" alt="image" src="https://github.com/user-attachments/assets/0fdfd51b-4ce2-4625-b2b4-597a8d7481b9" />
prompt_length
<img width="2421" height="792" alt="image" src="https://github.com/user-attachments/assets/1982563f-c0c8-40dd-b4bf-c37ef07ad5b8" />
<img width="1641" height="423" alt="image" src="https://github.com/user-attachments/assets/ff5c3da5-bafb-4a20-9435-a4db800b32e2" />
response_length
<img width="2421" height="801" alt="image" src="https://github.com/user-attachments/assets/dbe09147-2a38-410c-87d0-38d79cebce49" />
<img width="1614" height="423" alt="image" src="https://github.com/user-attachments/assets/4c67a191-e467-4e8d-9b1d-c4e9e8c51d52" />
timing_per_token_ms
<img width="2418" height="807" alt="image" src="https://github.com/user-attachments/assets/06da45c3-2685-4e05-9842-c2c08cd20f27" />
timing_s
<img width="2412" height="798" alt="image" src="https://github.com/user-attachments/assets/629c8803-f6eb-4975-ad57-198ccbd54e20" />
<img width="822" height="434" alt="image" src="https://github.com/user-attachments/assets/27ebc431-2b4b-4e16-92a3-dcb82724c991" />
training
<img width="2424" height="813" alt="image" src="https://github.com/user-attachments/assets/aa3f0621-48f3-47ad-b2a3-355915d85eef" />
<img width="2412" height="786" alt="image" src="https://github.com/user-attachments/assets/57fa8bc5-1407-4db0-9dc0-f3a499de4918" />

val(不一定对，当时设置有点问题)
<img width="2436" height="801" alt="image" src="https://github.com/user-attachments/assets/dc8a6821-4f38-45d9-9da9-fcf594bdd0d9" />
<img width="909" height="435" alt="image" src="https://github.com/user-attachments/assets/6c853627-cd78-40e1-9aa2-1a0f251787c3" />
System
<img width="2408" height="783" alt="image" src="https://github.com/user-attachments/assets/4dbfa0e3-75fe-444e-b1a9-149197e5128e" />
<img width="2403" height="792" alt="image" src="https://github.com/user-attachments/assets/c6cc98cc-0e93-4f15-b46e-a81c0f761897" />
<img width="2388" height="789" alt="image" src="https://github.com/user-attachments/assets/7d834619-b449-447a-8814-87039b04da76" />
<img width="2393" height="813" alt="image" src="https://github.com/user-attachments/assets/db045b38-248d-4355-a097-451b68936fe2" />

















































