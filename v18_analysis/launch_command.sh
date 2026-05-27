#!/bin/bash
# v18_full_grpo 启动命令（已脱敏）
# 训练时间：2026-05-19 09:10 ～ 2026-05-25 13:34
# 从 v14 global_step_336 checkpoint 恢复，完整训练 2 个 Epoch（total_updates=460）

# ===== 训练命令 =====
screen -dmS train_v18 bash -c "
  source /usr/local/miniconda3/bin/activate py312
  export VLLM_USE_V1=1
  export RAY_DEBUG=legacy
  export HYDRA_FULL_ERROR=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export WANDB_BASE_URL=http://[WANDB_SERVER_IP]:3005
  export WANDB_API_KEY=local-[REDACTED]
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  export RAY_memory_monitor_refresh_ms=0
  unset RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES
  cd /workspace/post_train/sql_agent
  python train_sql_agent_v18.py \
    --n-gpus 4 \
    --model /workspace/models/Qwen2.5-14B-Instruct-SFT-Agent \
    --train-file data/train_v18_full_weighted.parquet \
    --val-file data/val.parquet \
    --total-epochs 2 \
    --progress-file /tmp/sql_agent_progress_v18_full_grpo.json \
    --experiment-name ebm_agent_14b_grpo_4gpu_v18_full_grpo \
    --resume-from checkpoints/AgentLightning/ebm_agent_14b_grpo_4gpu/global_step_336 \
    2>&1 | tee /tmp/train_v18_full_grpo.log
"

# ===== 关键参数说明 =====
# --train-file data/train_v18_full_weighted.parquet  # 1836 条加权医学问题
# --total-epochs 2                                    # 2 个完整 epoch
# --resume-from .../global_step_336                   # 从 v14 最后 checkpoint 恢复
#
# 主要超参数变化（相对 v17）：
#   learning_rate: 5.00e-07  (v17: 8.00e-07)
#   format_scale:  0.700     (v17: 1.200，固定)
#   EvidenceSafety module: 新增，惩罚强主张与证据不匹配
#   train_size: 1836         (v17: 900)
