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

## Changes to `tools_embedding.py`

```python
# v15 added (not in v14):
if len(content) > 1500:
    content = content[:1500] + "...(内容已截断)"
```
Each search result document is now truncated to 1500 characters to prevent context overflow during training.

---

## Files in This Branch

| File | Description |
|------|-------------|
| `train_sql_agent.py` | GRPO training launcher (v15 configuration) |
| `sql_agent.py` | Agent + reward function (v15, working tree) |
| `tools_embedding.py` | 4-way vector search tool (with 1500-char truncation) |
