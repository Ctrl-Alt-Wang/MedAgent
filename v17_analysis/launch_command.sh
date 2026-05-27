#!/bin/bash
# v17_plus_small 启动命令（已脱敏）
# 训练时间：2026-05-10 02:46 ～ 2026-05-15 11:38
# 共三阶段：初始训练（epoch 1） + 两次续训（epoch 2～3）

# ===== 第一阶段（epoch 1，total_updates=113）=====
screen -dmS train_v17 bash -c "
  source /usr/local/miniconda3/bin/activate py312
  export VLLM_USE_V1=1
  export RAY_DEBUG=legacy
  export HYDRA_FULL_ERROR=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export WANDB_BASE_URL=http://103.139.212.228:3005
  export WANDB_API_KEY=local-[REDACTED]
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  export RAY_memory_monitor_refresh_ms=0
  unset RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES
  cd /workspace/post_train/sql_agent
  python train_sql_agent.py \
    --n-gpus 4 \
    --model /workspace/models/Qwen2.5-14B-Instruct-SFT-Agent \
    --train-file data/train.parquet \
    --val-file data/val.parquet \
    --total-epochs 1 \
    --progress-file /tmp/sql_agent_progress_v17_plus_small.json \
    --experiment-name ebm_agent_14b_grpo_4gpu_v17_plus_small \
    2>&1 | tee /tmp/train_v17_plus_small.log
"

# ===== 第二阶段（续训，total_updates=339，从 global_step_112 恢复）=====
screen -dmS train_v17_resume bash -c "
  source /usr/local/miniconda3/bin/activate py312
  export VLLM_USE_V1=1
  export RAY_DEBUG=legacy
  export HYDRA_FULL_ERROR=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export WANDB_BASE_URL=http://103.139.212.228:3005
  export WANDB_API_KEY=local-[REDACTED]
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  export RAY_memory_monitor_refresh_ms=0
  unset RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES
  cd /workspace/post_train/sql_agent
  python train_sql_agent.py \
    --n-gpus 4 \
    --model /workspace/models/Qwen2.5-14B-Instruct-SFT-Agent \
    --train-file data/train.parquet \
    --val-file data/val.parquet \
    --total-epochs 3 \
    --progress-file /tmp/sql_agent_progress_v17_plus_small.json \
    --experiment-name ebm_agent_14b_grpo_4gpu_v17_plus_small \
    2>&1 | tee /tmp/train_v17_plus_small_resume.log
"
