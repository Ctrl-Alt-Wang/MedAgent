# MedAgent v15 — Training Record

> Branch: `v15`  
> Experiment name: `ebm_agent_14b_grpo_4gpu_v15`  
> Evaluated checkpoint: `global_step_250` (training **not complete**, ~74% progress)  
> **Evaluation score: 72.2 / 100** (GPT-4.1 strict_v4 rubric, 200-question benchmark)

---

## Training Overview

| Item | Value |
|------|-------|
| Base model | Qwen2.5-14B-Instruct |
| Framework | Agent-Lightning + VERL + GRPO |
| GPUs | 4 × NVIDIA A800-80GB |
| Training dates | 2026-04-26 → 2026-04-29 |
| Total steps saved | 250 (estimated full training: ~339 steps) |
| Train dataset | `data/train.parquet` (~1000 medical Q&A) |
| Val dataset | `data/val.parquet` |

> **Note:** v15 score (72.2) vs v14 score (76.2) is **not a fair comparison** — v15 evaluated at step 250 (~74% through training) vs v14 at its final step 336.

---

## Hyperparameters (`train_sql_agent.py`)

| Parameter | v14 value | **v15 value** | Change |
|-----------|-----------|---------------|--------|
| `base_lr` | 1.5e-6 | **7e-7** → clamped to **8e-7** | ↓ ~47% |
| `n_rollouts` | 4 | **8** | ×2 |
| `max_prompt_length` | 8192 | **12288** | +50% |
| `max_response_length` | 2048 | 2048 | unchanged |
| `train_batch_size` | 8 | 8 | unchanged |
| `kl_loss_coef` | 0.02 | **0.08** | ×4 |
| `entropy_coeff` | 0.01 | **0** | removed |
| `clip_ratio_low` | 0.20 | 0.20 | unchanged |
| `clip_ratio_high` | 0.28 | **0.20** | symmetric |
| `max_model_len` (vLLM) | 16384 | 16384 | unchanged |
| `TP` | 4 | 4 | unchanged |
| `save_freq` | 50 steps | 50 steps | unchanged |
| `total_epochs` | 3 | 3 | unchanged |
| `max_actor_ckpt_to_keep` | 1 | 2 | |
| Actor checkpoint save_contents | (default) | `["model"]` only | |

---

## Reward Function (`sql_agent.py`)

### Reward Formula

```
total_reward = hard_reward
             + format_scale  × format_reward        # [0, 1.2]
             + correct_scale × correctness_reward    # [-3.6, 3.6]
             + rm_weight     × rm_score              # [0, 0.8]
             + prefix_coverage_bonus                 # [0, 0.48]  ← NEW
```

### Component Weights vs v14

| Component | v14 | **v15** | Change |
|-----------|-----|---------|--------|
| `format_scale` | 1.2 | 1.2 | unchanged |
| `correct_scale` | 1.0 | **1.2** | +20% |
| `rm_weight` max | 0.3 | **0.8** | ×2.67 |
| RM activation threshold | progress > 0.3 | **progress > 0.2** | earlier |
| `prefix_coverage` bonus | — | **+0.12 per DB (max +0.48)** | NEW |

### RM Weight Schedule

```python
if progress < 0.2:        # first 20% of training: RM disabled
    rm_weight = 0.0
else:
    rm_weight = 0.8 * (progress - 0.2) / 0.8   # linearly 0.0 → 0.8
```
RM activates at 20% (earlier than v14's 30%), reaching maximum **0.8** at 100%.

### NEW: prefix_coverage bonus (in `compute_correctness_reward`)

```python
if real_ids:
    covered_prefixes = {rid[:2] for rid in real_ids}
    db_source_count = len(covered_prefixes & {"01", "02", "03", "04"})
    reward += 0.12 * db_source_count   # max +0.48 for all 4 DBs
```

| DB prefix | Source |
|-----------|--------|
| `01` | Chinese clinical guidelines |
| `02` | English clinical guidelines |
| `03` | Systematic reviews & meta-analyses |
| `04` | RCT studies |

### RM Score Normalization (changed from v14)

```python
# v14: z = (raw_score - (-3.4)) / 8.0
z = (raw_score - (-5.0)) / 8.0    # center shifted left → scores map higher
rm_score = sigmoid(z)
```

### hard_reward — unchanged from v14

```
-1.0   medical question but no tool called
-0.5   reference list pattern detected in answer tail
-1.2   cited IDs exist but none are valid
-0.8 to -2.0   fabricated IDs
```

### format_reward — unchanged from v14, range [0, 1]

| Sub-component | Max pts |
|---------------|---------|
| PICO fields filled (≥3/4) | 0.15 |
| Main section presence | 0.35 |
| Main section order | 0.15 |
| Sub-section presence | 0.25 |
| Sub-section order | 0.10 |

### correctness_reward — unchanged from v14, range [-3, 3]

| Sub-component | Value |
|---------------|-------|
| Citation precision (`real/cited`) | `1.4 × precision` |
| ≥4 valid cited IDs bonus | +0.50 |
| ≥2 valid cited IDs bonus | +0.30 |
| ≥1 valid cited ID bonus | +0.12 |
| No citations at all | -0.05 |
| Section-prefix alignment | [-0.5, +0.5] |
| Honest abstention | bonus/penalty |
| Valid IDs exist but none cited | -0.20 |

### Observed Reward Distribution (step 127, April 26)

- Typical reward: **5.05 – 5.12**
- Reward variance: **~0.07** (collapsed — 20× lower than v14's ~1.5)
- rm_weight at step 127: **0.647** (dominant component)
- RM contribution per rollout: 0.40 – 0.46
- prefix_coverage: **always +0.48** (all 4 DBs hit → constant, zero variance)

---

## Key Observations & Known Issues

### 1. Reward Variance Collapse
By step 127, the total reward range per batch dropped to ~0.07 (vs v14's ~1.5). Root cause:
- `prefix_coverage` always fires at max (+0.48) once the model learns to call tools from all 4 DBs
- High `rm_weight=0.8` with RM scores clustering at 0.63–0.72 → RM contribution also near-constant
- GRPO intra-group advantage ≈ 0 → gradient signal effectively died ~step 127

### 2. Conservative Hyperparams Compound the Issue
- `kl_loss_coef=0.08` (×4 vs v14) + `base_lr=8e-7` (clamped minimum) = very slow effective learning
- With 8 rollouts per prompt (vs v14's 4), the batch requires twice as many inference calls but parameter updates remain the same frequency

### 3. RM Alignment
Increasing `rm_weight` from 0.3 to 0.8 resulted in a **4-point score drop** (76.2 → 72.2) on GPT-4.1 strict_v4 evaluation. This suggests the proxy RM model (`http://117.50.48.176:8400/score`) has weak correlation with the actual evaluation rubric.

### 4. ContextWindowExceeded Errors (validation only)
3,204 errors from validation rollouts during April 8–14 (training with different context settings). These affected **only validation monitoring**, not training gradients. Training rollouts (29,963 successful) were unaffected by VERL's `max_prompt_length=12288` truncation.

---

## Files in This Branch

| File | Description |
|------|-------------|
| `train_sql_agent.py` | GRPO training launcher (v15 configuration) |
| `sql_agent.py` | Agent + reward function (v15, working tree) |
| `tools_embedding.py` | 4-way vector search tool (1500-char truncation per document, same as v14) |

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
