# 训练日志分析说明文档

**报告生成日期**：2026-05-26

---

## 一、日志文件位置（服务器）

| 版本 | 日志路径 | 大小 |
|------|---------|------|
| v17 训练日志 | /tmp/train_v17_plus_small.log | 未单独存在，对应 sql_agent_training.log（456MB） |
| v18 训练日志 | /tmp/train_v18_full_grpo.log | 约 130+MB（tail -500 触发大文件告警） |
| AgentOps 日志 | /workspace/post_train/sql_agent/agentops.log | 1.1MB |
| 综合训练日志 | /workspace/post_train/sql_agent/sql_agent_training.log | 456MB |

**注意**：/tmp/train_v17_plus_small.log 文件存在，tail -500 约 71KB 输出（正常日志内容）。
/tmp/train_v18_full_grpo.log 文件存在，tail -500 约 128KB 输出（含 TaskRunner pid 调试信息）。

---

## 二、关键日志字段说明

### 2.1 Schedule 行（每次 reward 计算时输出）

```
[Schedule] mode=train/val, step=X/Y, p=Z, format_scale=A, correct_scale=B, rm_weight=C
```
- `step`：当前 global_step / total_updates
- `p`：进度比例（progress = global_step / schedule_steps）
- `format_scale`：格式奖励乘数（v17=1.2，v18=0.7）
- `rm_weight`：RM 奖励权重（前30%进度=0，后续线性增至0.3/0.25）

### 2.2 PICO 行

```
[PICO] P=..., I=..., C=..., O=..., filled=N/4
```
- 表示本次检索的 PICO 元素填充情况
- filled=4/4 表示完整 PICO

### 2.3 Citations 行

```
[Citations] valid_ids=N, cited_ids=M, fabricated=K
```
- `valid_ids`：检索工具返回的真实 ID 数量
- `cited_ids`：答案中引用的 ID 数量
- `fabricated`：伪造引用数（应尽量为0）

### 2.4 Reward Breakdown 行

```
[Reward Breakdown] hard=A, format=B x C, correct=D x E, subtotal=F
```
- 各分项细节

### 2.5 Total Reward 行

```
[Total Reward] X.XXX
```
- 最终 reward 值（裁剪到 [-6, 6]）

### 2.6 EvidenceSafety 行（v18 专有）

```
[EvidenceSafety] strong=True/False, cautious=True/False, prefixes={...}, score=Y
```
- `strong`：是否触发强结论检测词
- `cautious`：是否触发谨慎表达检测词
- `prefixes`：本次引用的 DB 类型前缀集合
- `score`：证据安全分（[-0.5, 0.25]）

---

## 三、grep 分析命令备忘

```bash
# 统计 no_tool 事件
grep "no_tool" /tmp/train_v18_full_grpo.log | wc -l

# 查看最新 Total Reward
grep "Total Reward" /tmp/train_v18_full_grpo.log | tail -100

# 查看 RM Contribution
grep "RM Contribution" /tmp/train_v18_full_grpo.log | tail -100

# 查看 EvidenceSafety 分布
grep "EvidenceSafety" /tmp/train_v18_full_grpo.log | tail -50

# 查看 error 情况
grep -i "error\|fatal\|exception" /tmp/train_v18_full_grpo.log | tail -50

# 查看 global_step 训练统计
grep "global_step" /tmp/train_v18_full_grpo.log | tail -20

# 查看 PICO 填充情况统计
grep "filled=" /tmp/train_v18_full_grpo.log | grep -o "filled=[0-9]*/4" | sort | uniq -c
```

---

## 四、v17 末尾日志关键信息（2026-05-11 验证轮次）

- 最后可见的验证轮次：mode=val, step=112/113, p=0.991
- 该轮次 PICO 填充：4/4（完整）
- 该轮次引用情况：valid_ids=7, cited_ids=7, fabricated=0（完美引用）
- format_scale=1.200, correct_scale=1.000, rm_weight=0.296

---

## 五、v18 末尾日志关键信息（2026-05-25 最新）

- 最新训练 step：397（来自 global_step 日志行）
- training/reward: 3.363
- critic/score/mean: 3.359（min=2.875, max=3.422）
- actor/kl_loss: 0.0086
- actor/kl_coef: 0.1
- actor/grad_norm: 2.41
- response_length/mean: 421.5 tokens
- EvidenceSafety 最新50条：score=0.000 占主流，说明训练后期模型已很少触发惩罚
- 无 no_tool 事件

---

*此文档用于指导技术团队进行日志分析和问题排查。服务器访问方式见 scripts/analyze_training_logs.py*
