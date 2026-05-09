#!/bin/bash
# v16 训练启动命令（实际执行于 117.50.198.65）

screen -dmS train_v16 bash -c "
  source /usr/local/miniconda3/bin/activate py312
  export VLLM_USE_V1=1
  export RAY_DEBUG=legacy
  export HYDRA_FULL_ERROR=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export WANDB_BASE_URL=http://103.139.212.228:3005
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  export RAY_memory_monitor_refresh_ms=0
  unset RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES
  export VLLM_NCCL_SO_PATH=/usr/local/miniconda3/envs/py312/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2

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
