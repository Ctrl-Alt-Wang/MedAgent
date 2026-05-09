#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GRPO 训练脚本（适用于医学循证 Agent）

特点：
1. 适配 Qwen2.5-14B-Instruct，4/8 卡 A800-80GB
2. 使用 Agent-Lightning + VERL + GRPO
3. 配合 sql_agent.py 中的动态 reward scale
4. 通过 progress file 近似追踪 global step，实现 ToolRL 风格动态 reward 调度

推荐用法：
    # 8卡正式训练
    python train_sql_agent.py \
        --n-gpus 8 \
        --train-file data/train.parquet \
        --val-file data/val.parquet

    # 4卡正式训练
    python train_sql_agent.py \
        --n-gpus 4 \
        --train-file data/train.parquet \
        --val-file data/val.parquet

    # 从已有 warm-start / SFT checkpoint 做 GRPO
    python train_sql_agent.py \
        --n-gpus 8 \
        --model /workspace/models/qwen14b_warmstart \
        --train-file data/train.parquet \
        --val-file data/val.parquet

    # 从 GRPO checkpoint 恢复，并设置恢复时的 global step
    python train_sql_agent.py \
        --n-gpus 8 \
        --resume_checkpoint /path/to/ckpt \
        --resume-global-step 60 \
        --train-file data/train.parquet \
        --val-file data/val.parquet

    # 快速 CI / smoke test
    python train_sql_agent.py \
        --n-gpus 8 \
        --ci-fast \
        --train-file data/train.parquet \
        --val-file data/val.parquet
"""

from __future__ import annotations

import argparse
import json
import math
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, cast

import dotenv
from datasets import Dataset as HuggingFaceDataset

dotenv.load_dotenv()

import agentlightning as agl
from agentlightning.env_var import (
    LightningEnvVar,
    resolve_bool_env_var,
    resolve_str_env_var,
)

from sql_agent import SQLSearchAgent

DEFAULT_MODEL_PATH = os.getenv("DEFAULT_MODEL_PATH", "/workspace/models/Qwen2.5-14B-Instruct")


# ============================================================
# 配置构造
# ============================================================

def get_logger_backends(use_wandb: bool) -> List[str]:
    return ["console", "wandb"] if use_wandb else ["console"]


def verl_default_config(
    n_gpus: int,
    use_wandb: bool,
    total_epochs: int,
    save_freq: int,
    test_freq: int,
    train_batch_size_override: Optional[int] = None,
    lr_override: Optional[float] = None,
) -> Dict[str, Any]:
    """
    根据 GPU 数量生成较稳妥的 GRPO 配置。
    目标：Qwen2.5-14B-Instruct on 4/8 x A800-80GB
    """

    # 14B：TP=4 一般比较合适；4卡时 TP=4, DP=1；8卡时 TP=4, DP=2
    tp = min(n_gpus, 4)
    dp = max(1, n_gpus // tp)

    # batch size：默认 4卡=8, 8卡=16
    base_batch_per_dp = 8
    train_batch_size = train_batch_size_override or (base_batch_per_dp * dp)

    # 学习率：随 DP 做轻微 sqrt scaling
    base_lr = 1.5e-6
    lr = lr_override if lr_override is not None else (base_lr * math.sqrt(dp))
    lr = max(8e-7, min(lr, 3e-6))

    # 4 卡建议开启 offload；8 卡一般可以关掉
    need_offload = n_gpus <= 4

    # rollout
    n_rollouts = 4
    max_turns = 2

    # gpu memory utilization
    gpu_memory_utilization = 0.40 if n_gpus <= 4 else 0.45

    print(f"\n{'=' * 60}")
    print("[自动配置]")
    print(f"  GPUs: {n_gpus}")
    print(f"  TP: {tp}")
    print(f"  DP: {dp}")
    print(f"  train_batch_size: {train_batch_size}")
    print(f"  learning_rate: {lr:.2e}")
    print(f"  need_offload: {need_offload}")
    print(f"  gpu_memory_utilization: {gpu_memory_utilization}")
    print(f"  n_rollouts: {n_rollouts}")
    print(f"{'=' * 60}\n")

    config: Dict[str, Any] = {
        "algorithm": {
            "adv_estimator": "grpo",
            "use_kl_in_reward": False,
        },
        "data": {
            "train_batch_size": train_batch_size,
            "max_prompt_length": 8192,
            "max_response_length": 2048,
        },
        "actor_rollout_ref": {
            "rollout": {
                "tensor_model_parallel_size": tp,
                "ulysses_sequence_parallel_size": 1,
                "n": n_rollouts,
                "log_prob_micro_batch_size_per_gpu": 1,
                "multi_turn": {
                    "format": "hermes",
                    "max_turns": max_turns,
                },
                "name": "vllm",
                "free_cache_engine": True,
                "gpu_memory_utilization": gpu_memory_utilization,
                "engine_kwargs": {
                    "vllm": {
                        "enable_auto_tool_choice": True,
                        "tool_call_parser": "hermes",
                        "max_model_len": 16384,
                    }
                },
            },
            "actor": {
                "strategy": "fsdp2",
                "ppo_mini_batch_size": train_batch_size,
                "ppo_micro_batch_size_per_gpu": 1,
                "optim": {
                    "lr": lr,
                },
                # 医学场景下给一点小 KL，训练更稳
                "use_kl_loss": True,
                "kl_loss_coef": 0.02,
                "entropy_coeff": 0.01,
                "clip_ratio_low": 0.20,
                "clip_ratio_high": 0.28,
                "fsdp_config": {
                    "offload_policy": need_offload,
                    "reshard_after_forward": True,
                },
                "checkpoint": {
                    "save_contents": ["model"],
                    "load_contents": ["model", "optimizer", "extra"],
                },
            },
            "ref": {
                "log_prob_micro_batch_size_per_gpu": 1,
                "fsdp_config": {
                    # ref 常开 offload，省显存
                    "param_offload": True,
                },
            },
            "model": {
                "path": DEFAULT_MODEL_PATH,
                "use_remove_padding": False,
                "enable_gradient_checkpointing": True,
                "use_torch_compile": False,
                "entropy_checkpointing": True,
            },
        },
        "trainer": {
            "n_gpus_per_node": n_gpus,
            "val_before_train": False,
            "critic_warmup": 0,
            "logger": get_logger_backends(use_wandb),
            "project_name": "AgentLightning",
            "experiment_name": f"ebm_agent_14b_grpo_{n_gpus}gpu_v16",
            "nnodes": 1,
            "save_freq": save_freq,
            "test_freq": test_freq,
            "total_epochs": total_epochs,
            "max_actor_ckpt_to_keep": 2,
            "max_critic_ckpt_to_keep": 2,
            
        },
    }
    return config


# ============================================================
# 数据 / 训练步数 / progress 初始化
# ============================================================

def load_parquet_as_list(path: str) -> List[Dict[str, Any]]:
    data = HuggingFaceDataset.from_parquet(path).to_list()
    if not data:
        raise ValueError(f"数据集为空: {path}")
    return cast(List[Dict[str, Any]], data)


def validate_dataset_schema(dataset: List[Dict[str, Any]], dataset_name: str) -> None:
    sample = dataset[0]
    if "question" not in sample:
        raise ValueError(
            f"{dataset_name} 缺少必须字段 'question'。"
            f" 当前字段: {list(sample.keys())}"
        )

    # is_medical 为可选字段；若没有，sql_agent.py 默认当作 True
    if "is_medical" not in sample:
        print(f"[提醒] {dataset_name} 未包含 is_medical 字段，将默认按医学问题处理。")


def print_dataset_preview(train_dataset: List[Dict[str, Any]], val_dataset: List[Dict[str, Any]]) -> None:
    print(f"\n训练集: {len(train_dataset)} 条")
    print(f"验证集: {len(val_dataset)} 条")
    print("训练数据前 3 条：")
    for i, item in enumerate(train_dataset[:3]):
        q = str(item.get("question", ""))
        print(f"  [{i}] {q[:120]}...")


def estimate_total_updates(config: Dict[str, Any], train_dataset_size: int) -> int:
    """
    估计 optimizer update 次数：
    - 如果 trainer 显式给了 total_training_steps，就优先用它
    - 否则按 ceil(len(dataset) / train_batch_size) * epochs 估算
    """
    trainer_cfg = config.get("trainer", {})
    data_cfg = config.get("data", {})

    if "total_training_steps" in trainer_cfg:
        return max(1, int(trainer_cfg["total_training_steps"]))

    batch_size = max(1, int(data_cfg["train_batch_size"]))
    total_epochs = max(1, int(trainer_cfg.get("total_epochs", 1)))
    steps_per_epoch = math.ceil(train_dataset_size / batch_size)
    return max(1, steps_per_epoch * total_epochs)


def initialize_progress_tracking(
    *,
    progress_file: str,
    total_updates: int,
    rollouts_per_update: int,
    schedule_steps: int,
    resume_global_step: int,
) -> None:
    """
    为 sql_agent.py 中的动态 reward scale 初始化共享进度文件。
    """

    os.makedirs(os.path.dirname(progress_file) or ".", exist_ok=True)

    state = {
        "rollout_count": int(resume_global_step * rollouts_per_update),
        "global_step": int(resume_global_step),
        "total_updates": int(total_updates),
        "rollouts_per_update": int(rollouts_per_update),
        "updated_at": datetime.now().timestamp(),
    }

    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)

    # 这些环境变量会被 sql_agent.py 读取
    os.environ["SQL_AGENT_PROGRESS_FILE"] = progress_file
    os.environ["SQL_AGENT_TOTAL_UPDATES"] = str(total_updates)
    os.environ["SQL_AGENT_ROLLOUTS_PER_UPDATE"] = str(rollouts_per_update)
    os.environ["SQL_AGENT_SCHEDULE_STEPS"] = str(schedule_steps)

    print(f"\n[Progress Tracking]")
    print(f"  progress_file: {progress_file}")
    print(f"  total_updates: {total_updates}")
    print(f"  rollouts_per_update: {rollouts_per_update}")
    print(f"  schedule_steps: {schedule_steps}")
    print(f"  resume_global_step: {resume_global_step}\n")


# ============================================================
# 训练主逻辑
# ============================================================

def train(
    *,
    train_file: str,
    val_file: str,
    model: Optional[str],
    llm_proxy: bool,
    ci: bool,
    ci_fast: bool,
    n_runners: int,
    n_gpus: int,
    external_store_address: str,
    trajectory_level: bool,
    weave: bool,
    mongo_uri: Optional[str],
    resume_checkpoint: Optional[str],
    resume_global_step: int,
    use_wandb: bool,
    total_epochs: int,
    save_freq: int,
    test_freq: int,
    progress_file: str,
    schedule_steps: Optional[int],
    train_batch_size_override: Optional[int],
    lr_override: Optional[float],
) -> None:
    # ---------- 加载数据 ----------
    train_dataset = load_parquet_as_list(train_file)
    val_dataset = load_parquet_as_list(val_file)

    validate_dataset_schema(train_dataset, "train_dataset")
    validate_dataset_schema(val_dataset, "val_dataset")
    print_dataset_preview(train_dataset, val_dataset)

    # ---------- 构建配置 ----------
    config = verl_default_config(
        n_gpus=n_gpus,
        use_wandb=use_wandb,
        total_epochs=total_epochs,
        save_freq=save_freq,
        test_freq=test_freq,
        train_batch_size_override=train_batch_size_override,
        lr_override=lr_override,
    )

    # 覆盖模型路径
    if model:
        config["actor_rollout_ref"]["model"]["path"] = model
        print(f"[模型] 使用自定义模型: {model}")
    else:
        print(f"[模型] 使用默认模型: {DEFAULT_MODEL_PATH}")

    # checkpoint 恢复
    if resume_checkpoint is not None:
        if resume_checkpoint == "latest":
            print("[Checkpoint] 从最近 checkpoint 恢复")
            config["trainer"]["load_checkpoint"] = True
        else:
            print(f"[Checkpoint] 从指定 checkpoint 恢复: {resume_checkpoint}")
            config["trainer"]["load_checkpoint"] = resume_checkpoint

    # trajectory-level trace
    if trajectory_level:
        config["agentlightning"] = {
            "trace_aggregator": {
                "level": "trajectory",
                "trajectory_max_prompt_length": 8192,
                "trajectory_max_response_length": 2048,
            }
        }
        print("[Trace] 已启用 trajectory-level trace aggregator")

    # ---------- CI 模式 ----------
    if ci or ci_fast:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_suffix = uuid.uuid4().hex[:8]
        experiment_name = f"ebm_agent_ci_{timestamp}_{random_suffix}"
        project_name = "AgentLightningCI"

        agl_current_role = resolve_str_env_var(LightningEnvVar.AGL_CURRENT_ROLE)
        if agl_current_role != "runner":
            github_output = os.getenv("GITHUB_OUTPUT")
            if github_output:
                with open(github_output, "a", encoding="utf-8") as f:
                    f.write(f"project_name={project_name}\n")
                    f.write(f"run_name={experiment_name}\n")

        config["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] = 0.40
        config["trainer"]["total_epochs"] = 1
        config["trainer"]["total_training_steps"] = 20
        config["trainer"]["test_freq"] = 20
        config["trainer"]["experiment_name"] = experiment_name
        config["trainer"]["project_name"] = project_name
        config["trainer"].pop("save_freq", None)

        if ci_fast:
            config["trainer"]["total_training_steps"] = 1
            config["trainer"]["test_freq"] = 1

        print(f"[CI] experiment_name: {experiment_name}")
        print(f"[CI] total_training_steps: {config['trainer'].get('total_training_steps')}")

    # ---------- 估计总 update 数 ----------
    total_updates = estimate_total_updates(config, len(train_dataset))
    train_batch_size = int(config["data"]["train_batch_size"])
    n_rollouts = int(config["actor_rollout_ref"]["rollout"]["n"])
    rollouts_per_update = train_batch_size * n_rollouts

    # schedule_steps：默认 min(150, total_updates)
    effective_schedule_steps = (
        min(150, total_updates)
        if schedule_steps is None
        else max(1, min(schedule_steps, total_updates))
    )

    initialize_progress_tracking(
        progress_file=progress_file,
        total_updates=total_updates,
        rollouts_per_update=rollouts_per_update,
        schedule_steps=effective_schedule_steps,
        resume_global_step=resume_global_step,
    )

    # ---------- 打印训练摘要 ----------
    print(f"\n{'=' * 60}")
    print("[训练摘要]")
    print(f"  train_file: {train_file}")
    print(f"  val_file: {val_file}")
    print(f"  train_size: {len(train_dataset)}")
    print(f"  val_size: {len(val_dataset)}")
    print(f"  total_updates(estimated): {total_updates}")
    print(f"  train_batch_size: {train_batch_size}")
    print(f"  rollouts_per_query: {n_rollouts}")
    print(f"  rollouts_per_update: {rollouts_per_update}")
    print(f"  schedule_steps: {effective_schedule_steps}")
    print(f"  total_epochs: {config['trainer'].get('total_epochs', 'N/A')}")
    print(f"  total_training_steps: {config['trainer'].get('total_training_steps', 'N/A')}")
    print(f"  learning_rate: {config['actor_rollout_ref']['actor']['optim']['lr']:.2e}")
    print(f"  model_path: {config['actor_rollout_ref']['model']['path']}")
    print(f"  project_name: {config['trainer']['project_name']}")
    print(f"  experiment_name: {config['trainer']['experiment_name']}")
    print(f"{'=' * 60}\n")

    # ---------- 构建算法 ----------
    algorithm = agl.VERL(config)

    # ---------- 构建 store ----------
    if external_store_address:
        store = agl.LightningStoreClient(external_store_address)
        print(f"[Store] 使用外部 store: {external_store_address}")
    elif mongo_uri:
        from agentlightning.store.mongo import MongoLightningStore
        store = MongoLightningStore(mongo_uri=mongo_uri)
        print(f"[Store] 使用 Mongo store: {mongo_uri}")
    else:
        store = None
        print("[Store] 使用内存 store")

    # ---------- 构建 trainer ----------
    trainer_kwargs: Dict[str, Any] = {
        "algorithm": algorithm,
        "n_runners": n_runners,
        "store": store,
    }

    # Always use LlmProxyTraceToTriplet - this script uses LiteLLM proxy for inference.
    # The default TracerTraceToTriplet uses span hierarchy which fails with proxy spans.
    trainer_kwargs["adapter"] = agl.LlmProxyTraceToTriplet()

    if llm_proxy:
        trainer_kwargs["tracer"] = agl.OtelTracer()
        print("[Tracer] 使用 LLM Proxy")
    elif weave:
        from agentlightning.tracer.weave import WeaveTracer
        trainer_kwargs["tracer"] = WeaveTracer()
        print("[Tracer] 使用 Weave")

    trainer = agl.Trainer(**trainer_kwargs)

    # ---------- 开始训练 ----------
    print("[启动] 开始 GRPO 训练...")
    trainer.fit(SQLSearchAgent(), train_dataset, val_dataset=val_dataset)


# ============================================================
# CLI
# ============================================================

def main() -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    parser = argparse.ArgumentParser(
        description="使用 Agent-Lightning + VERL + GRPO 训练医学循证 Agent。"
    )

    parser.add_argument(
        "--train-file",
        type=str,
        default="data/train.parquet",
        help="训练集 parquet 路径",
    )
    parser.add_argument(
        "--val-file",
        type=str,
        default="data/val.parquet",
        help="验证集 parquet 路径",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="模型路径（可为 warm-start / SFT checkpoint 或 HF ID）",
    )
    parser.add_argument(
        "--llm-proxy",
        action="store_true",
        help="启用 LLM Proxy 追踪",
    )
    parser.add_argument(
        "--weave",
        action="store_true",
        help="启用 Weave 追踪",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="CI 模式（减少训练步数）",
    )
    parser.add_argument(
        "--ci-fast",
        action="store_true",
        help="快速 CI 模式（1 step，隐含 --ci）",
    )
    parser.add_argument(
        "--n-runners",
        type=int,
        default=4,
        help="Trainer 的 runner 数量",
    )
    parser.add_argument(
        "--n-gpus",
        type=int,
        default=8,
        help="每节点 GPU 数量，推荐 4 或 8",
    )
    parser.add_argument(
        "--external-store-address",
        type=str,
        default="",
        help="外部 store 地址，如 http://localhost:9999",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用 DEBUG 日志",
    )
    parser.add_argument(
        "--trajectory-level",
        action="store_true",
        help="启用 trajectory-level trace aggregator",
    )
    parser.add_argument(
        "--mongo-uri",
        type=str,
        default=None,
        help="MongoDB URI（持久化 store）",
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        nargs="?",
        const="latest",
        help="从 checkpoint 恢复；不传值=latest，传路径=指定 checkpoint",
    )
    parser.add_argument(
        "--resume-global-step",
        type=int,
        default=0,
        help="若从 checkpoint 恢复，可手动指定恢复时的 global step，用于动态 reward 调度",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="关闭 wandb，仅保留 console logger",
    )
    parser.add_argument(
        "--total-epochs",
        type=int,
        default=3,
        help="总训练 epoch 数",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=50,
        help="checkpoint 保存频率（steps）",
    )
    parser.add_argument(
        "--test-freq",
        type=int,
        default=25,
        help="验证频率（steps）",
    )
    parser.add_argument(
        "--progress-file",
        type=str,
        default="/tmp/sql_agent_progress.json",
        help="动态 reward 调度使用的 progress 文件路径",
    )
    parser.add_argument(
        "--schedule-steps",
        type=int,
        default=None,
        help="动态 reward 调度步数；默认 min(150, total_updates)",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=None,
        help="手动覆盖 train_batch_size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="手动覆盖 actor learning rate",
    )

    args = parser.parse_args()

    if args.external_store_address:
        print(f"[Store] 连接外部存储: {args.external_store_address}")
        if resolve_bool_env_var(LightningEnvVar.AGL_MANAGED_STORE, fallback=True):
            raise ValueError(
                "使用外部存储时，请设置环境变量 AGL_MANAGED_STORE=0，"
                "否则 Trainer 仍会尝试管理 store 生命周期。"
            )

    if args.ci_fast:
        args.ci = True

    agl.setup_logging("DEBUG" if args.debug else "INFO")

    train(
        train_file=args.train_file,
        val_file=args.val_file,
        model=args.model,
        llm_proxy=args.llm_proxy,
        ci=args.ci,
        ci_fast=args.ci_fast,
        n_runners=args.n_runners,
        n_gpus=args.n_gpus,
        external_store_address=args.external_store_address,
        trajectory_level=args.trajectory_level,
        weave=args.weave,
        mongo_uri=args.mongo_uri,
        resume_checkpoint=args.resume_checkpoint,
        resume_global_step=args.resume_global_step,
        use_wandb=(not args.no_wandb),
        total_epochs=args.total_epochs,
        save_freq=args.save_freq,
        test_freq=args.test_freq,
        progress_file=args.progress_file,
        schedule_steps=args.schedule_steps,
        train_batch_size_override=args.train_batch_size,
        lr_override=args.lr,
    )


if __name__ == "__main__":
    main()