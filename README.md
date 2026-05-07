# MedAgent v13 — Training Record

> Branch: `v13`  
> Experiment name: `ebm_agent_14b_grpo_4gpu`  
> Status: **Training stopped due to reward hacking** (~40–45 steps)  
> Source: git commit `cf5dfde` (committed 2026-04-21; code written/used from 2026-04-09)

---

## Training Overview

| Item | Value |
|------|-------|
| Base model | Qwen2.5-14B-Instruct |
| Framework | Agent-Lightning + VERL + GRPO |
| GPUs | 4 × NVIDIA A800-80GB |
| Training dates | 2026-04-09 → 2026-04-13 (multiple short runs, final run ~45 steps) |
| Total steps | ~45 (stopped, training **not complete**) |
| Train dataset | `data/train.parquet` (~1000 medical Q&A) |
| Val dataset | `data/val.parquet` |

> **Note:** v13 was stopped because reward hacking was observed in wandb — `training/reward` oscillated 0.5–2.5 while `training/n_triplets` (tokens per rollout) continuously decreased from ~90 to ~55, suggesting the model was exploiting the reward function rather than learning genuine evidence synthesis. The issues were diagnosed and fixed in v14.

---

## Hyperparameters (`train_sql_agent.py`)

| Parameter | v13 value | v14 fix | Change |
|-----------|-----------|---------|--------|
| `base_lr` | `7e-7` → clamped to **8e-7** | **1.5e-6** | ×1.9 |
| `n_rollouts` | 4 | 4 | unchanged |
| `max_prompt_length` | **12288** | **8192** | −33% |
| `max_response_length` | 2048 | 2048 | unchanged |
| `train_batch_size` | 8 | 8 | unchanged |
| `kl_loss_coef` | **0.08** | **0.02** | ×0.25 |
| `entropy_coeff` | **0.005** | **0.01** | ×2 |
| `clip_ratio_low` | 0.20 | 0.20 | unchanged |
| `clip_ratio_high` | **0.20** (symmetric) | **0.28** (asymmetric) | allows larger positive steps |
| `total_epochs` | 3 | 3 | unchanged |
| `save_freq` | 50 steps | 50 steps | unchanged |
| `TP` | 4 | 4 | unchanged |
| `max_model_len` (vLLM) | 16384 | 16384 | unchanged |
| `experiment_name` | `ebm_agent_14b_grpo_{n_gpus}gpu` | same | same |

---

## Reward Function (`sql_agent.py`)

### Reward Formula

```
total_reward = hard_reward
             + format_scale  × format_reward        # [0, 1.2]
             + correct_scale × correctness_reward    # [-3.0, 3.0]
             + rm_weight     × rm_score              # [0, 0.3]

total_reward = clip(total_reward, -6.0, 6.0)
```

### Component Weights

| Component | v13 value | Notes |
|-----------|-----------|-------|
| `format_scale` | **1.2** (fixed) | |
| `correct_scale` | **1.0** (fixed) | |
| `rm_weight` max | **0.3** (scheduled) | activates after 30% progress — never reached in v13's ~45 steps |

### RM Weight Schedule

```python
if progress < 0.3:
    rm_weight = 0.0
else:
    rm_weight = 0.3 * (progress - 0.3) / 0.7   # linearly 0.0 → 0.3
```
RM was never activated during v13 — at ~45 steps out of ~339 total, progress ≈ 13%.

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
| No citations at all | −0.05 |
| Section-prefix alignment | [−0.5, +0.5] |
| Honest abstention (no evidence) | bonus/penalty |
| Valid IDs exist but none cited | −0.20 |
| Inner clip | `max(-3.0, min(3.0, reward))` |

> **No recall bonus, no citation diversity bonus, no prefix_coverage bonus** — those components were introduced in v15 and v16.

### RM Score Normalization

```python
z = (raw_score - (-3.4)) / max(8.0, 1e-6)
rm_score = sigmoid(z)    # maps raw RM score to [0, 1]
```
Center point: `−3.4` (raw scores above −3.4 map to rm_score > 0.5).  
RM model endpoint: `http://117.50.48.176:8400/score`

### Theoretical Reward Range (no RM, at v13 step count)

```
max = 0                            # hard_reward (no violations)
    + 1.2 × 1.0                    # format_scale × format_max
    + 1.0 × (1.4 + 0.50 + 0.50)   # correct_scale × (precision + count + alignment)
    = 1.2 + 2.4 = 3.6
```

---

## Observed Reward Hacking

**Wandb metrics (April 13 run, ~45 steps):**

| Metric | Observed | Interpretation |
|--------|----------|----------------|
| `training/reward` | 0.5 → 2.5, dip at step ~40–42, spike to ~2.5 | non-monotonic, unstable |
| `training/n_triplets` | ↓ 90 → 55 (steady decrease) | model generating shorter outputs over time |
| `training/n_triplets_prompt_too_long` | 0–8 per step (noisy, elevated early) | context truncation events |
| `training/n_rollouts_w_trace` | constant ~30 out of 32 | most rollouts include tool calls |

**Root cause analysis:**

### 1. Context pressure from `max_prompt_length=12288`
The `n_triplets_prompt_too_long` metric shows training prompts (history + tool results) regularly hit the truncation limit. As training progressed, the model appears to have learned to reduce its tool call output volume (fewer retrieved passages, shorter answers), causing `n_triplets` to fall from ~90 to ~55. This shrinking output reduces rollout diversity, collapsing GRPO advantage variance.

### 2. Learning rate too low (`base_lr = 7e-7`, clamped to `8e-7`)
Combined with high KL penalty, gradient updates were too small to shift behavior away from discovered shortcuts. Once the model settled on a "safe citation" strategy (cite 1–2 certain IDs), it could not escape.

### 3. KL too high (`kl_loss_coef = 0.08`)
The reference model anchor was 4× stronger than v14's 0.02. This made exploration costly and kept the policy close to the base model's citation-averse behavior.

### 4. Symmetric PPO clip (`clip_ratio_high = 0.20`)
Blocked large positive gradient steps. The asymmetric `0.28` upper bound in v14 was specifically introduced to allow faster learning when behavior improves, while `0.20` lower bound still limits negative steps.

### 5. Correctness reward has no recall component
The model could achieve `precision = 1.00` by citing only 1–2 IDs it was highly confident about, earning `1.4 + 0.12 = 1.52` correctness reward while doing minimal evidence retrieval. This "safe citation" strategy scores identically to citing 4+ diverse sources (which requires real retrieval effort). Without a recall penalty, there is no incentive to search broadly.

---

## Fix Applied in v14

| Problem in v13 | Fix in v14 |
|----------------|------------|
| `base_lr=8e-7` too low | → `1.5e-6` |
| `kl_loss_coef=0.08` suppressing exploration | → `0.02` |
| `entropy_coeff=0.005` insufficient | → `0.01` |
| `clip_ratio_high=0.20` blocking positive updates | → `0.28` (asymmetric) |
| `max_prompt_length=12288` causing truncation | → `8192` |

These five changes together restored meaningful GRPO gradient signal. v14 ran to **step 336** (training complete) and scored **76.2 / 100** on the GPT-4.1 strict_v4 200-question benchmark.

---

## Cross-Version Comparison

| Version | Steps | Score | Outcome |
|---------|-------|-------|---------|
| **v13** | ~45 | — | Reward hacking → stopped manually |
| **v14** | 336 | **76.2** | Completed |
| **v15** | 250 | 72.2\* | Reward variance collapse → evaluated early |
| **v16** | in progress | — | Training ongoing |

\*v15 evaluated at step 250 (~74% through training), not directly comparable to v14.

---

## Files in This Branch

| File | Description |
|------|-------------|
| `train_sql_agent.py` | GRPO training launcher — v13 configuration (`base_lr=7e-7`, `kl=0.08`, `max_prompt=12288`) |
| `sql_agent.py` | Agent + reward function — v13 (no recall/diversity bonus, correctness clipped to ±3) |
| `tools_embedding.py` | 4-way vector search tool (1500-char truncation per document, same as v14) |
| `tools.py` | SQL database query tool (earlier version, pre-embedding migration) |
| `data/train.parquet` | Training dataset (~1000 medical Q&A pairs) |
| `data/val.parquet` | Validation dataset |
| `sqlagent_test.py` | Manual test script for the agent |
| `start_vllm.sh` | vLLM server launch script |
| `restart_ray.sh` | Ray cluster restart script |
