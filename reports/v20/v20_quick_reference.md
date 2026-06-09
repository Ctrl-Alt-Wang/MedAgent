# v20 快速参考卡

## 一句话总结

> Qwen2.5-14B-Instruct **SFT 热启动** + GRPO，4×A800-80GB，训练 458 步（~6.6 天），
> val reward 从 **3.1855 → 3.2570（+0.0715）**，GPT-4.1 strict_v4 评测 **73.1/100**。

---

## 核心参数速查

| 类别 | 参数 | 值 |
|------|------|-----|
| 模型 | 基础模型 | Qwen2.5-14B-Instruct-SFT-Agent |
| 模型 | 参数量 | ~14B |
| 硬件 | GPU | 4× A800-SXM4-80GB |
| 训练 | LR | 5e-7 |
| 训练 | Batch Size | 8 |
| 训练 | Rollouts/prompt | 4 |
| 训练 | Steps | 458 / 460 |
| 训练 | Epochs | 3 |
| 训练 | 总时长 | ~158.8 小时 |
| 训练 | KL 系数 | 0.10 |
| 训练 | Clip Ratio | 0.20 (对称) |
| 奖励 | format_scale | 0.70 |
| 奖励 | correct_scale | 1.00 |
| 奖励 | rm_weight 末期 | 0.25 |
| 评测 | 分数 | 73.1/100 |
| 评测 | 最高 | 92.0 |
| 评测 | 最低 | 27.3 |

---

## 关键文件位置

### 服务器 (117.50.198.65 port 23)
```
训练脚本:   /workspace/post_train/sql_agent/train_sql_agent_v20_sft.py
Agent代码:  /workspace/post_train/sql_agent/sql_agent_v18.py
训练日志:   /workspace/post_train/sql_agent/sql_agent_training.log  (596MB)
检查点:     /workspace/post_train/sql_agent/checkpoints/AgentLightning/
              ebm_agent_14b_grpo_4gpu_v20_sft/global_step_458/
HF模型:     /workspace/grpo_v20_step458_hf/  (~28GB)
进度文件:   /tmp/sql_agent_progress_v20_sft.json
```

### 本地 (D:\evidence_ii)
```
SSH隧道:    eval_sql_agent/ssh_tunnel_v20.py
评测Runner: eval_sql_agent/eval_runner_v20.py
评测Scorer: eval_sql_agent/eval_scorer_v20.py
回答文件:   eval_sql_agent/eval_results/grpo_v20_step458_*_answers.json
评分文件:   eval_sql_agent/eval_results/grpo_v20_step458_strict_v4_*_scored.json
Agent配置:  backend/simple_agent/.env
```

---

## 评测结果速查

```
总分:    73.1/100  (std: 10.2)

专科最高: 心血管 75.6  | 最低: 风湿免疫 69.6
难度最高: G2 74.0     | 最低: G4 71.1
维度最高: 术语规范性 94.2% | 最低: 时效性 53.7%
```

---

## 与 v18 的关键区别

| 维度 | v18 | v20 |
|------|-----|-----|
| 起始模型 | 原始 14B-Instruct | **SFT-Agent（已微调）** |
| 初始 val reward | ~3.10 | **~3.18（更高起点）** |
| 最终 val reward | 3.1855 | **3.2570** |
| Δ val reward | — | **+0.0715** |
| 评测最终分 | — | **73.1** |

---

## 快速复现步骤

```bash
# 1. 服务器上部署 vLLM
ssh root@117.50.198.65 -p 23
screen -S vllm_v20
cd /workspace/post_train/sql_agent
# 合并 checkpoint（如需）
python -m verl.model_merger --backend fsdp2 \
    --local_dir checkpoints/AgentLightning/ebm_agent_14b_grpo_4gpu_v20_sft/global_step_458/actor \
    --target_dir /workspace/grpo_v20_step458_hf
# 启动 vLLM
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /workspace/grpo_v20_step458_hf \
    --port 8001 --tensor-parallel-size 4 --dtype bfloat16 \
    --max-model-len 32768 --gpu-memory-utilization 0.85 \
    --enable-auto-tool-choice --tool-call-parser hermes

# 2. 本地建立 SSH 隧道
python D:\evidence_ii\eval_sql_agent\ssh_tunnel_v20.py

# 3. 启动 Agent 服务
cd D:\evidence_ii\backend\simple_agent
python main_api_openai.py  # 监听 :9998

# 4. 运行评测
python D:\evidence_ii\eval_sql_agent\eval_runner_v20.py

# 5. 评分
python D:\evidence_ii\eval_sql_agent\eval_scorer_v20.py
```
