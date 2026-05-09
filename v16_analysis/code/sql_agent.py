from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, cast

import dotenv
import httpx

# 先加载 .env，再读取环境变量
dotenv.load_dotenv()

try:
    import fcntl  # Linux 环境可用
except Exception:  # pragma: no cover
    fcntl = None

from agentlightning import (
    LLM,
    LitAgent,
    NamedResources,
    Trainer,
    emit_reward,
    setup_logging,
)
from agentlightning.types import AttemptedRollout, Rollout
from agents import Agent, Runner
from agents.extensions.models.litellm_model import LitellmModel
from agents.lifecycle import RunHooks
from agents.model_settings import ModelSettings

from tools_embedding import search_embedding_db

setup_logging()

# ============================================================
# 基础配置
# ============================================================

REWARD_MODEL_URL = os.getenv("REWARD_MODEL_URL", "http://117.50.48.176:8400/score")
MAX_COMPLETION_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", "1536"))
MAX_TURNS = 2

# ============================================================
# 日志配置
# ============================================================

log_file = os.getenv("SQL_AGENT_LOG_FILE", "sql_agent_training.log")
logger = logging.getLogger("sql_agent_training")
logger.setLevel(logging.INFO)

if not logger.handlers:
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

today = datetime.today().strftime("%Y-%m-%d")

# ============================================================
# Prompt
# ============================================================

PICO_RULE = """PICO提取规则：
- 语言：中文
- P（患者/人群）：明确疾病、年龄、性别等关键特征，不要泛化
- I（干预）：具体到药物/剂量/术式/方案
- C（对照）：如用户未提及，可省略
- O（结局）：聚焦用户最关心的临床终点
- 总体原则：精准、简要，不需要过度扩展"""

sql_agent_prompt = f"""重要提醒：今天日期是: {today}
你是「循证决策专家」。你的任务是将医生用户的临床决策问题转化为可检索、可分层总结的循证结论。

---

# 第一步：问题分流（最高优先级）

在执行任何操作之前，先判断用户输入是否与医学、临床或健康咨询相关。

**若为纯粹的非医学问题**（如日常问候"你好"、纯粹的IT编程问题、日常通用对话等）：
- 直接以自然、流畅的语言回答，不需要引用文献，不需要生成分层报告。
- 完全忽略下方所有循证输出要求，回答后即结束。

**若为医学问题**：
- **【红线警告】**：你**绝对不能**凭借自己的预训练记忆或常识直接回答！
- 必须继续执行下方流程，在输出任何文字回答前，**第一步且唯一的动作必须是调用指定的检索工具**。

---

# 工具与检索策略

**可用工具**：文献检索工具（search_embedding_db）

**{PICO_RULE}**

**工具调用策略**：
1. **必然触发原则（最高优先级）**：只要判断为循证/医学类问题，**在输出任何回答前，都必须、强制、毫无例外地调用一次文献检索工具**。绝对不允许不调用工具直接作答！
2. 根据用户问题提取 PICO 后调用文献检索工具。
3. 工具返回为空时，在对应层级注明"未检索到相关文献"或“暂无直接证据”。
4. **引用来源约束（杜绝假引文）**：回答中涉及的任何 `[^xx-xxxx-x]` 引用，**必须精确对应本次工具检索实际返回的文献ID**。严禁编造文献、严禁捏造不存在的引文标注！

---

# ⚠️ 核心输出规则 ⚠️
1. **绝对禁止**在回答末尾添加任何参考文献列表、脚注解释、引用说明
2. 脚注仅用[^xx-xxxx-x]格式内嵌在正文中，**不要在文末解释这些脚注**
3. 回答在最后一个实质性句子结束后立即停止，不添加任何补充内容
4. 不要输出"参考文献"、"引用来源"、"References"等任何形式的标题

# 正确示例
"...该方案在3个月随访中显示良好疗效[^xx-xxxx-x]。对于特殊人群需要调整剂量[^xx-xxxx-x]。"
(回答到此结束，没有任何后续内容)

# 错误示例
"..疗效良好[^xx-xxxx-x]。
[^xx-xxxx-x]: xxx
[^xx-xxxx-x]: xxx
..."
(类似这种格式绝对禁止)

# ⚠️ 强制输出结构（仅针对医学问题，顺序不可更改）

### 一、直接结论
- 针对用户问题给出简明的核心结论（仅2~3句话）
- 句末使用脚注编号标注（如 xxxx[^xx-xxxx-x]）

### 二、循证分层证据总结
#### （一）中文指南证据层
#### （二）英文指南证据层
#### （三）系统评价/Meta分析证据层
#### （四）临床试验证据层

### 三、临床实践决策

⚠️ **严格分层约束**：
- **禁止跨层级使用证据**：各层级仅能引用对应ID前缀的文献，严禁混用。
- **ID前缀映射规则**：
  - 中文指南层：仅限 `01-` 开头的ID
  - 英文指南层：仅限 `02-` 开头的ID
  - 系统评价/Meta分析层：仅限 `03-` 开头的ID
  - 临床试验层：仅限 `04-` 开头的ID

# 回答结束标准
当最后一句临床建议写完后，立即停止输出，不要添加：
- 参考文献列表
- 脚注解释
- 引用说明
- 任何形式的补充内容

# ⚠️不回答政治相关的问题⚠️
"""

# ============================================================
# 正则
# ============================================================

SEARCH_CITE_PATTERN = re.compile(r"引用标识[：:]\s*(\d{2}-[0-9A-Fa-f]{4}-\d+)")
GENERIC_ID_PATTERN = re.compile(r"(\d{2}-[0-9A-Fa-f]{4}-\d+)")
ANSWER_CITE_PATTERN = re.compile(r"\[\^(\d{2}-[0-9A-Fa-f]{4}-\d+)\]")
REF_LIST_PATTERN = re.compile(r"\[\^\d{2}-[0-9A-Fa-f]{4}-\d+\]\s*[：:]")
ABSTAIN_PATTERN = re.compile(r"未检索到相关文献|暂无直接证据|证据不足|暂无高质量证据|目前缺乏直接证据")

MAIN_SECTION_PATTERNS = [
    re.compile(r"一[、\.]?\s*直接结论"),
    re.compile(r"二[、\.]?\s*循证分层证据"),
    re.compile(r"三[、\.]?\s*临床实践决策"),
]

SUB_SECTION_PATTERNS = [
    re.compile(r"中文指南"),
    re.compile(r"英文指南"),
    re.compile(r"系统评价|Meta分析"),
    re.compile(r"临床试验"),
]

SECTION_PREFIX_MAP = OrderedDict(
    [
        ("cn", "01"),
        ("en", "02"),
        ("sr", "03"),
        ("rct", "04"),
    ]
)

SECTION_TEXT_PATTERNS = OrderedDict(
    [
        ("cn", re.compile(r"中文指南.*?(?=英文指南|系统评价|Meta分析|临床试验|临床实践决策|$)", re.S)),
        ("en", re.compile(r"英文指南.*?(?=系统评价|Meta分析|临床试验|临床实践决策|$)", re.S)),
        ("sr", re.compile(r"(?:系统评价|Meta分析).*?(?=临床试验|临床实践决策|$)", re.S)),
        ("rct", re.compile(r"临床试验.*?(?=临床实践决策|$)", re.S)),
    ]
)

# ============================================================
# Progress Tracking
# train_sql_agent.py 会写这些环境变量
# ============================================================

def _progress_file_path() -> str:
    return os.getenv("SQL_AGENT_PROGRESS_FILE", "/tmp/sql_agent_progress.json")


def _total_updates() -> int:
    try:
        return max(1, int(os.getenv("SQL_AGENT_TOTAL_UPDATES", "1")))
    except Exception:
        return 1


def _rollouts_per_update() -> int:
    try:
        return max(1, int(os.getenv("SQL_AGENT_ROLLOUTS_PER_UPDATE", "1")))
    except Exception:
        return 1


def _schedule_steps() -> int:
    total_updates = _total_updates()
    try:
        configured = int(os.getenv("SQL_AGENT_SCHEDULE_STEPS", "150"))
    except Exception:
        configured = 150
    return max(1, min(configured, total_updates))


@contextmanager
def _locked_progress_file(exclusive: bool = True):
    path = _progress_file_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(path, "a+", encoding="utf-8") as f:
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield f
        finally:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _load_progress_state_unlocked(f) -> Dict[str, Any]:
    f.seek(0)
    raw = f.read().strip()
    if not raw:
        return {
            "rollout_count": 0,
            "global_step": 0,
            "total_updates": _total_updates(),
            "rollouts_per_update": _rollouts_per_update(),
            "updated_at": time.time(),
        }

    try:
        state = json.loads(raw)
        if not isinstance(state, dict):
            raise ValueError("progress state is not dict")
        return state
    except Exception:
        return {
            "rollout_count": 0,
            "global_step": 0,
            "total_updates": _total_updates(),
            "rollouts_per_update": _rollouts_per_update(),
            "updated_at": time.time(),
        }


def _save_progress_state_unlocked(f, state: Dict[str, Any]) -> None:
    f.seek(0)
    f.truncate()
    json.dump(state, f, ensure_ascii=False)
    f.flush()


def get_training_schedule(increment_rollout: bool) -> Dict[str, float]:
    """
    近似 global step 追踪：
    - 每个 train rollout 增加一次 rollout_count
    - global_step ≈ rollout_count // rollouts_per_update

    动态缩放：
    - format_scale: 2.0 -> 1.0
    - correct_scale: 2/3 -> 1.0
    - rm_weight: 前 60% 为 0，后 40% 线性升到 0.3
    """
    with _locked_progress_file(exclusive=True) as f:
        state = _load_progress_state_unlocked(f)

        state["total_updates"] = _total_updates()
        state["rollouts_per_update"] = _rollouts_per_update()

        if increment_rollout:
            state["rollout_count"] = int(state.get("rollout_count", 0)) + 1

        rollout_count = int(state.get("rollout_count", 0))
        rollouts_per_update = max(1, int(state.get("rollouts_per_update", 1)))
        global_step = min(
            rollout_count // rollouts_per_update,
            int(state.get("total_updates", 1)),
        )

        state["global_step"] = global_step
        state["updated_at"] = time.time()
        _save_progress_state_unlocked(f, state)

    total_updates = max(1, int(state["total_updates"]))
    schedule_steps = _schedule_steps()
    progress = min(global_step / schedule_steps, 1.0)

    format_scale = 1.2                        # 固定，不再动态衰减
    correct_scale = 1.2                       # 固定，与 format 比 1.2:1

    if progress < 0.3:
        rm_weight = 0.0
    else:
        rm_weight = 0.3 * (progress - 0.3) / 0.7  # v16: 0.0 -> 0.3

    return {
        "global_step": float(global_step),
        "total_updates": float(total_updates),
        "progress": float(progress),
        "format_scale": float(format_scale),
        "correct_scale": float(correct_scale),
        "rm_weight": float(rm_weight),
    }

# ============================================================
# 工具结果 / section / citation 解析
# ============================================================

def stringify_tool_result(obj: Any) -> str:
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)


def extract_ids_from_obj(obj: Any) -> Set[str]:
    """
    尽量从各种 tool return 结构中鲁棒提取 ID。
    """
    ids: Set[str] = set()

    if obj is None:
        return ids

    if isinstance(obj, str):
        strict_ids = set(SEARCH_CITE_PATTERN.findall(obj))
        if strict_ids:
            ids.update(strict_ids)
        else:
            ids.update(GENERIC_ID_PATTERN.findall(obj))
        return ids

    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str):
                ids.update(GENERIC_ID_PATTERN.findall(k))
            ids.update(extract_ids_from_obj(v))
        return ids

    if isinstance(obj, (list, tuple, set)):
        for item in obj:
            ids.update(extract_ids_from_obj(item))
        return ids

    ids.update(GENERIC_ID_PATTERN.findall(str(obj)))
    return ids


def extract_answer_cites(answer: str) -> Set[str]:
    return set(ANSWER_CITE_PATTERN.findall(answer))


def extract_section_texts(answer: str) -> Dict[str, str]:
    sections: Dict[str, str] = {}
    for key, pattern in SECTION_TEXT_PATTERNS.items():
        m = pattern.search(answer)
        if m:
            sections[key] = m.group(0)
    return sections


def ordered_presence_score(answer: str, patterns: List[re.Pattern]) -> Tuple[float, float]:
    """
    返回：
    - presence_ratio: 出现比例
    - order_ratio: 已出现项是否按顺序排列
    """
    positions: List[Optional[int]] = []
    found_count = 0

    for p in patterns:
        m = p.search(answer)
        if m:
            positions.append(m.start())
            found_count += 1
        else:
            positions.append(None)

    presence_ratio = found_count / len(patterns) if patterns else 0.0
    found_positions = [x for x in positions if x is not None]

    if not found_positions:
        return presence_ratio, 0.0
    if len(found_positions) == 1:
        return presence_ratio, 1.0

    ordered_pairs = 0
    total_pairs = len(found_positions) - 1
    for i in range(total_pairs):
        if found_positions[i] < found_positions[i + 1]:
            ordered_pairs += 1

    order_ratio = ordered_pairs / total_pairs if total_pairs > 0 else 1.0
    return presence_ratio, order_ratio

# ============================================================
# Reward
# ============================================================

def compute_hard_reward(
    answer: str,
    valid_ids: Set[str],
    cited_ids: Set[str],
    fabricated_ids: Set[str],
    tool_called: bool,
    is_medical: bool,
) -> float:
    """
    红线项：固定重罚，不参与动态缩放
    """
    reward = 0.0

    if is_medical and not tool_called:
        return -1.0

    tail = answer[-800:] if len(answer) > 800 else answer
    if REF_LIST_PATTERN.search(tail):
        reward -= 0.5

    if cited_ids and not valid_ids:
        reward -= 1.2

    if fabricated_ids:
        reward -= min(2.0, 0.8 + 0.5 * len(fabricated_ids))

    return reward


def compute_format_reward(answer: str, pico_filled: int) -> float:
    """
    format-soft，范围 [0, 1]
    """
    reward = 0.0

    # PICO 完整度
    if pico_filled >= 3:
        reward += 0.15
    elif pico_filled >= 2:
        reward += 0.10
    elif pico_filled >= 1:
        reward += 0.05

    # 主结构
    main_presence, main_order = ordered_presence_score(answer, MAIN_SECTION_PATTERNS)
    reward += 0.35 * main_presence
    reward += 0.15 * main_order

    # 子结构
    sub_presence, sub_order = ordered_presence_score(answer, SUB_SECTION_PATTERNS)
    reward += 0.25 * sub_presence
    reward += 0.10 * sub_order

    return max(0.0, min(reward, 1.0))


def check_section_prefix_alignment(answer: str, valid_ids: Set[str]) -> float:
    """
    只检查真实引用的 section-prefix 对齐。
    返回约 [-0.5, 0.5]
    """
    if not valid_ids:
        return 0.0

    sections = extract_section_texts(answer)
    total_checks = 0
    aligned_checks = 0

    for section_key, expected_prefix in SECTION_PREFIX_MAP.items():
        section_text = sections.get(section_key, "")
        if not section_text:
            continue

        real_section_cites = extract_answer_cites(section_text) & valid_ids
        if not real_section_cites:
            continue

        for cite_id in real_section_cites:
            total_checks += 1
            if cite_id.startswith(expected_prefix):
                aligned_checks += 1

    if total_checks == 0:
        return 0.0

    alignment_rate = aligned_checks / total_checks
    score = alignment_rate * 1.0 - 0.5  # 0 -> -0.5, 1 -> +0.5

    logger.info(
        f"[SectionPrefix] checks={total_checks}, aligned={aligned_checks}, "
        f"rate={alignment_rate:.2f}, score={score:.2f}"
    )
    return max(-0.5, min(score, 0.5))


def check_honest_abstention(answer: str, valid_ids: Set[str]) -> float:
    """
    医学任务关键项：
    - 整体无证据时诚实说明，给少量正分或至少不误罚
    - 某层无对应前缀证据时，如果该层诚实写“未检索到相关文献/暂无直接证据”，给小正分
    """
    reward = 0.0
    sections = extract_section_texts(answer)
    available_prefixes = {vid[:2] for vid in valid_ids}

    if not valid_ids:
        if ABSTAIN_PATTERN.search(answer):
            reward += 0.35
        else:
            reward -= 0.10
        return reward

    for section_key, prefix in SECTION_PREFIX_MAP.items():
        section_text = sections.get(section_key, "")
        if not section_text:
            continue

        if prefix not in available_prefixes:
            if ABSTAIN_PATTERN.search(section_text):
                reward += 0.12
            else:
                reward -= 0.04

    return reward


def compute_correctness_reward(answer: str, valid_ids: Set[str]) -> float:
    """
    correctness-soft:
    - 真实引用 precision
    - 真实引用覆盖
    - section-prefix 对齐
    - honest abstention

    范围裁剪到 [-3, 3]
    """
    reward = 0.0
    cited_ids = extract_answer_cites(answer)
    real_ids = cited_ids & valid_ids

    if cited_ids and valid_ids:
        precision = len(real_ids) / max(1, len(cited_ids))
        reward += 1.4 * precision

        if len(real_ids) >= 4:
            reward += 0.50
        elif len(real_ids) >= 2:
            reward += 0.30
        elif len(real_ids) >= 1:
            reward += 0.12

        logger.info(
            f"[Correctness] cited={len(cited_ids)}, real={len(real_ids)}, precision={precision:.2f}"
        )
    elif not cited_ids:
        # 无引用不直接重罚，交给 honest abstention 处理
        reward -= 0.05

    reward += check_section_prefix_alignment(answer, valid_ids)
    reward += check_honest_abstention(answer, valid_ids)

    # v16: recall bonus — cite more of the available valid evidence
    if valid_ids:
        recall = len(real_ids) / max(len(valid_ids), 1)
        recall_bonus = 0.7 * recall
        reward += recall_bonus
        logger.info(f"[Recall] real={len(real_ids)}, total_valid={len(valid_ids)}, recall={recall:.2f}, bonus={recall_bonus:.2f}")

    # v16: citation diversity — based on cited IDs (not tool calls), avoids degeneracy
    if real_ids:
        cited_db_prefixes = {rid[:2] for rid in real_ids}
        cited_db_count = len(cited_db_prefixes & {"01", "02", "03", "04"})
        diversity_bonus = 0.08 * cited_db_count
        reward += diversity_bonus
        logger.info(f"[CiteDiversity] prefixes={cited_db_prefixes}, db_count={cited_db_count}, bonus={diversity_bonus:.2f}")

    if valid_ids and not real_ids and not ABSTAIN_PATTERN.search(answer):
        reward -= 0.20

    reward = max(-3.0, min(5.0, reward))
    return reward


async def compute_reward_model_score(question: str, answer: str) -> float:
    """
    外部 reward model，输出 [0,1]
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                REWARD_MODEL_URL,
                json={"prompt": question, "response": answer},
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            result = resp.json()
            raw_score = float(result.get("score", 0.0))
            z = (raw_score - (-3.4)) / max(8.0, 1e-6)
            score = 1.0 / (1.0 + math.exp(-z))
            logger.info(f"[RM] raw={raw_score:.4f}, z={z:.2f}, score={score:.3f}")
            return float(score)
    except Exception as e:
        logger.error(f"[RM Error] {e}")
        return 0.0


async def compute_sql_agent_reward(
    question: str,
    answer: str,
    tool_calls: List[Dict[str, Any]],
    is_medical: bool = True,
    mode: str = "train",
    enable_rm_reward: bool = True,
) -> float:
    """
    最终 reward:
    total = hard_reward
          + format_scale * format_reward
          + correct_scale * correctness_reward
          + rm_weight * rm_score
    """
    schedule = get_training_schedule(increment_rollout=(mode == "train"))
    logger.info(
        f"[Schedule] mode={mode}, step={schedule['global_step']:.0f}/{schedule['total_updates']:.0f}, "
        f"p={schedule['progress']:.3f}, format_scale={schedule['format_scale']:.3f}, "
        f"correct_scale={schedule['correct_scale']:.3f}, rm_weight={schedule['rm_weight']:.3f}"
    )

    if not answer or not answer.strip():
        logger.info("[Reward] 空回答 -> -0.5")
        return -0.5

    # ---------- 解析工具调用 ----------
    tool_called = False
    pico_filled = 0
    search_result_obj: Any = None

    for tc in tool_calls:
        if tc.get("name") == "search_embedding_db":
            tool_called = True
            args = tc.get("args", {})
            pico_fields = ["P", "I", "C", "O"]
            pico_filled = sum(1 for f in pico_fields if str(args.get(f, "")).strip())

            logger.info(f"[PICO] P={args.get('P', '')}")
            logger.info(f"[PICO] I={args.get('I', '')}")
            logger.info(f"[PICO] C={args.get('C', '')}")
            logger.info(f"[PICO] O={args.get('O', '')}")
            logger.info(f"[PICO] filled={pico_filled}/4")

            raw = tc.get("result", {})
            if isinstance(raw, dict) and "raw_result" in raw:
                search_result_obj = raw["raw_result"]
            else:
                search_result_obj = raw
            break

    # ---------- 非医学样本 ----------
    if not is_medical:
        reward = 0.0
        if tool_called:
            reward -= 0.20
        else:
            reward += 0.20
        if answer.strip():
            reward += 0.10

        logger.info(f"[Reward] 非医学样本 reward={reward:.3f}")
        return reward

    # ---------- 医学样本 ----------
    search_result_text = stringify_tool_result(search_result_obj)
    valid_ids = extract_ids_from_obj(search_result_obj)
    cited_ids = extract_answer_cites(answer)
    fabricated_ids = cited_ids - valid_ids if valid_ids else cited_ids

    logger.info(
        f"[Citations] valid_ids={len(valid_ids)}, cited_ids={len(cited_ids)}, fabricated={len(fabricated_ids)}"
    )
    if valid_ids:
        logger.info(f"[Citations] sample_valid_ids={list(sorted(valid_ids))[:10]}")

    hard_reward = compute_hard_reward(
        answer=answer,
        valid_ids=valid_ids,
        cited_ids=cited_ids,
        fabricated_ids=fabricated_ids,
        tool_called=tool_called,
        is_medical=is_medical,
    )

    if not tool_called:
        logger.info(f"[Reward] 医学问题未调工具 -> hard_reward={hard_reward:.3f}")
        return hard_reward

    format_reward = compute_format_reward(answer=answer, pico_filled=pico_filled)
    correctness_reward = compute_correctness_reward(answer=answer, valid_ids=valid_ids)

    total_reward = (
        hard_reward
        + schedule["format_scale"] * format_reward
        + schedule["correct_scale"] * correctness_reward
    )

    logger.info(
        f"[Reward Breakdown] hard={hard_reward:.3f}, "
        f"format={format_reward:.3f} x {schedule['format_scale']:.3f}, "
        f"correct={correctness_reward:.3f} x {schedule['correct_scale']:.3f}, "
        f"subtotal={total_reward:.3f}"
    )

    # 后半段逐步引入 RM reward
    if enable_rm_reward and mode == "train" and schedule["rm_weight"] > 0.0:
        rm_score = await compute_reward_model_score(question, answer)
        rm_contribution = schedule["rm_weight"] * rm_score
        total_reward += rm_contribution
        logger.info(
            f"[RM Contribution] score={rm_score:.3f}, weight={schedule['rm_weight']:.3f}, "
            f"contribution={rm_contribution:.3f}"
        )
    else:
        logger.info("[RM] skip")

    # 最终裁剪，防止极端值不稳
    total_reward = max(-6.0, min(6.0, total_reward))
    logger.info(f"[Total Reward] {total_reward:.3f}")

    return float(total_reward)

# ============================================================
# Hooks
# ============================================================

class NavHooks(RunHooks[Any]):
    def __init__(self):
        self.tool_calls: List[Dict[str, Any]] = []

    async def on_tool_end(self, context, agent, tool, result: Any):
        try:
            args = json.loads(context.tool_arguments)
        except Exception:
            args = {"raw": context.tool_arguments}

        self.tool_calls.append(
            {
                "name": getattr(tool, "name", "unknown"),
                "args": args,
                "result": {"raw_result": result},
            }
        )

        tool_name = getattr(tool, "name", "unknown")
        result_str = stringify_tool_result(result)
        found_ids = extract_ids_from_obj(result)

        logger.info(f"\n{'─' * 60}")
        logger.info(f"[工具调用] {tool_name}")
        logger.info(f"  P: {args.get('P', '未填写')}")
        logger.info(f"  I: {args.get('I', '未填写')}")
        logger.info(f"  C: {args.get('C', '未填写')}")
        logger.info(f"  O: {args.get('O', '未填写')}")
        logger.info(f"  [检索返回长度] {len(result_str)}")
        logger.info(f"  [检索摘要] {result_str[:800]}")
        logger.info(f"  [有效ID数] {len(found_ids)}")
        if found_ids:
            logger.info(f"  [部分ID] {list(sorted(found_ids))[:10]}")
        logger.info(f"{'─' * 60}")

# ============================================================
# Agent
# ============================================================

class SQLSearchAgent(LitAgent[Any]):
    def __init__(self, trained_agents=None):
        super().__init__(trained_agents=trained_agents)

    def _build_agent(self, base_url: str, llm: LLM, temperature: float) -> Agent:
        return Agent(
            model=LitellmModel(
                model="hosted_vllm/" + llm.model,
                base_url=base_url,
                api_key=llm.api_key,
            ),
            model_settings=ModelSettings(
                max_tokens=MAX_COMPLETION_TOKENS,
                temperature=temperature,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
            ),
            name="ebm_agent",
            instructions=sql_agent_prompt,
            tools=[search_embedding_db],
        )

    async def _run_agent_with_retry(
        self,
        agent: Agent,
        question: str,
        max_retries: int = 3,
    ) -> Tuple[Optional[str], NavHooks]:
        hooks = NavHooks()
        answer = None

        for attempt in range(max_retries):
            try:
                res = await Runner.run(agent, question, max_turns=MAX_TURNS, hooks=hooks)
                answer = res.final_output
                break
            except Exception as e:
                logger.warning(f"Run failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))
                    hooks = NavHooks()

        return answer, hooks

    async def _compute_and_log_reward(
        self,
        task: Dict[str, Any],
        question: str,
        answer: Optional[str],
        hooks: NavHooks,
        mode: str,
    ) -> Optional[float]:
        if not answer:
            logger.warning(f"[{mode}] Agent 重试后仍失败，跳过: {question}")
            return None

        logger.info(f"\n{'=' * 72}")
        logger.info(f"[{mode}] 问题: {question}")
        logger.info(f"[{mode}] 工具调用次数: {len(hooks.tool_calls)}")
        logger.info(f"[{mode}] 回答长度: {len(answer)} 字")
        logger.info(f"{'-' * 40}")
        logger.info(answer[:2500])
        if len(answer) > 2500:
            logger.info(f"... (已截断，完整长度 {len(answer)} 字)")
        logger.info(f"{'-' * 40}")

        is_medical = bool(task.get("is_medical", True))

        reward = await compute_sql_agent_reward(
            question=question,
            answer=answer,
            tool_calls=hooks.tool_calls,
            is_medical=is_medical,
            mode=mode,
            enable_rm_reward=True,
        )

        logger.info(f"[{mode}] reward = {reward:.3f}")
        logger.info(f"{'=' * 72}\n")
        emit_reward(float(reward))
        return float(reward)

    async def training_rollout_async(
        self,
        task: Dict[str, Any],
        resources: NamedResources,
        rollout: Rollout,
    ) -> float | None:
        llm = cast(LLM, resources.get("main_llm"))
        rollout = cast(AttemptedRollout, rollout)
        base_url = llm.get_base_url(rollout.rollout_id, rollout.attempt.attempt_id)
        question = task.get("question", "")

        logger.info(f"--- [Train] Q: {question} ---")

        agent = self._build_agent(base_url, llm, temperature=0.7)
        answer, hooks = await self._run_agent_with_retry(agent, question)
        return await self._compute_and_log_reward(task, question, answer, hooks, mode="train")

    async def validation_rollout_async(
        self,
        task: Dict[str, Any],
        resources: NamedResources,
        rollout: Rollout,
    ) -> float | None:
        llm = cast(LLM, resources.get("main_llm"))
        rollout = cast(AttemptedRollout, rollout)
        base_url = llm.get_base_url(rollout.rollout_id, rollout.attempt.attempt_id)
        question = task.get("question", "")

        logger.info(f"--- [Val] Q: {question} ---")

        # 验证时 deterministic
        agent = self._build_agent(base_url, llm, temperature=0.0)
        answer, hooks = await self._run_agent_with_retry(agent, question)
        return await self._compute_and_log_reward(task, question, answer, hooks, mode="val")


if __name__ == "__main__":
    Trainer(n_workers=12).fit_v0(SQLSearchAgent(), "http://localhost:9999/")