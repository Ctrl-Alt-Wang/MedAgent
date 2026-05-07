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
Wandb曲线情况：
actor
<img width="2409" height="789" alt="image" src="https://github.com/user-attachments/assets/52e477a9-62df-437f-8a90-a23329b4e8bd" />
<img width="2426" height="420" alt="image" src="https://github.com/user-attachments/assets/cc1af4e4-93af-4fcd-b374-07b54e2718a3" />
critic
<img width="2405" height="795" alt="image" src="https://github.com/user-attachments/assets/1004b180-5b16-48de-80d8-172b94599c4f" />
<img width="2402" height="795" alt="image" src="https://github.com/user-attachments/assets/f051c302-b125-447f-b109-8bfeab5f51aa" />
<img width="2397" height="795" alt="image" src="https://github.com/user-attachments/assets/6a91a4ec-1620-43e6-80b7-052d25c63b09" />
<img width="2415" height="798" alt="image" src="https://github.com/user-attachments/assets/1a5dcd7b-88d7-4f91-93d8-2cc433e0c8dd" />
global_seqlen
<img width="2412" height="789" alt="image" src="https://github.com/user-attachments/assets/ea0af404-6734-4560-bb0d-3f5361d8b879" />
perf
<img width="2418" height="795" alt="image" src="https://github.com/user-attachments/assets/ca688168-5d6a-4f97-8ed9-c74026b6943a" />
<img width="804" height="420" alt="image" src="https://github.com/user-attachments/assets/4f60afac-8461-4e0e-b1ac-1f98f613676a" />
prompt_length
<img width="2421" height="783" alt="image" src="https://github.com/user-attachments/assets/29bcf7ee-b6b2-42b3-8063-4b9433c42ea9" />
<img width="1605" height="420" alt="image" src="https://github.com/user-attachments/assets/20d833b1-1999-4d55-94cd-269ed8ba1277" />
response_length
<img width="2403" height="789" alt="image" src="https://github.com/user-attachments/assets/83e4f173-7bb4-4638-9182-f34b674c7aba" />
<img width="1605" height="417" alt="image" src="https://github.com/user-attachments/assets/037a2a89-e039-4f51-8a9a-20db6aea23cb" />
timing_per_token_ms
<img width="2427" height="791" alt="image" src="https://github.com/user-attachments/assets/6357cd7d-f477-40ab-9179-2f2b8fb4360d" />
timing_s
<img width="2409" height="810" alt="image" src="https://github.com/user-attachments/assets/bdef9b16-f3ed-42d8-bc35-5e941f26b76a" />
<img width="813" height="416" alt="image" src="https://github.com/user-attachments/assets/514544e9-6635-43c0-9a64-98d843e6a391" />
training
<img width="2418" height="792" alt="image" src="https://github.com/user-attachments/assets/c60133a5-ae68-40a4-b230-0198280e778f" />
<img width="2421" height="783" alt="image" src="https://github.com/user-attachments/assets/ace5f4c2-e8b7-40f9-b75b-9fee0dc14c21" />
val这个好像并未真的验证（先看着，当时设置的有些问题）
<img width="2405" height="777" alt="image" src="https://github.com/user-attachments/assets/a1d52b42-4262-4d10-b3b1-1cf4a7fc67fa" />
<img width="801" height="419" alt="image" src="https://github.com/user-attachments/assets/1de8b2dc-27e1-4fce-8634-04444b923019" />
Sysytem
<img width="2421" height="821" alt="image" src="https://github.com/user-attachments/assets/9b208ed9-2bf9-4bca-9eb3-d7c5070ba7a7" />
<img width="2424" height="806" alt="image" src="https://github.com/user-attachments/assets/4c3f1096-f972-4545-9b3e-00abeae6689c" />
<img width="2436" height="822" alt="image" src="https://github.com/user-attachments/assets/66303a12-6e19-4a6b-bde1-cc709021b12b" />
<img width="2454" height="792" alt="image" src="https://github.com/user-attachments/assets/23c7f83b-95d1-451d-b9e0-d1c6681dea0c" />
























