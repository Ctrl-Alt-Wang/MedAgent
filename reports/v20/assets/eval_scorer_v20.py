#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""评估步骤2：用 GPT-4.1 对 grpo_v20_step458 回答逐条打分（strict v4 - Chain-of-Critique）
标识：grpo_v20_step458
输入：eval_results/grpo_v20_step458_*_answers.json（最新）
输出：eval_results/grpo_v20_step458_strict_v4_<timestamp>_scored.json
"""

import json, os, re, glob, asyncio, time, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from datetime import datetime
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "backend", "simple_agent", ".env"))

API_BASE = "https://one-api.infox-med.com/v1"
API_KEY  = "sk-VzP164mp8fXCt7p2089dD47d35Aa4fA8A730F1E0A61dF77b"
MODEL    = "gpt-4.1"
EVAL_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "eval_results")
FILE_PREFIX      = "grpo_v20_step458"
FENCE = "```"


def compute_question_score_range(rubrics):
    max_score = sum(r["weight"] for r in rubrics if r["weight"] > 0)
    min_score = sum(r["weight"] for r in rubrics if r["weight"] < 0)
    return max_score, min_score


def to_percentage_dynamic(raw_score, max_score, min_score):
    score_range = max_score - min_score
    if score_range == 0:
        return 0.0
    pct = (raw_score - min_score) / score_range * 100
    return round(max(0.0, min(100.0, pct)), 1)


def extract_json_from_text(text):
    match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    match = re.search(r'```\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    matches = list(re.finditer(r'\{[^{}]*"scores"\s*:\s*\[.*?\]\s*\}', text, re.DOTALL))
    if matches:
        return json.loads(matches[-1].group(0))
    brace_start = text.rfind('{"scores"')
    if brace_start == -1:
        brace_start = text.rfind('{')
    if brace_start != -1:
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    return json.loads(text[brace_start:i+1])
    raise ValueError("无法从输出中提取 JSON")


SYSTEM_PROMPT = (
    "你是一位严谨且挑剔的循证医学评审专家。你的核心职责是找出AI回答中的不足和问题，然后据此客观打分。\n\n"
    "## 你的评审原则\n\n"
    "1. **批判性审查优先。** 你的首要任务是找出回答中每个维度的具体不足，而不是肯定其优点。每一分都需要回答用具体内容来挣得。\n\n"
    "2. **满分极其罕见。** 某一项拿满分意味着该维度上找不到任何不足。即使是很好的回答，在大多数维度上也应该能找到改进空间。如果你在某项给了满分，你必须确认自己确实找不到任何不足之处。\n\n"
    "3. **具体性是硬性要求。** 以下情况必须扣分：\n"
    "   - 提到\"有研究表明\"\"有证据显示\"但未注明具体研究名称、类型或年份\n"
    "   - 提到\"指南推荐\"但未给出指南的具体名称或版本\n"
    "   - 仅给出结论但未说明支撑证据的层级和质量\n\n"
    "4. **负权重项需要主动搜寻问题。** 你必须逐条检查：\n"
    "   - 每个引用编号是否可追溯？有无编造嫌疑？\n"
    "   - 引用的证据是否真正支持回答中的对应结论？\n"
    "   - 推理链是否完整，有无逻辑跳跃或前后矛盾？\n\n"
    "5. **理由必须具体。** 必须指出回答中的原文内容或明确指出缺失了什么，禁止使用\"较好\"\"基本符合\"\"覆盖全面\"等模糊评语。\n\n"
    "## 关于引用格式的重要说明\n\n"
    "该AI系统使用专业医学文献数据库进行检索，返回的引用格式为 [^xx-xxxx-x]，其中：\n"
    "- 01开头 = 中文临床指南\n"
    "- 02开头 = 英文临床指南\n"
    "- 03开头 = 系统评价/Meta分析\n"
    "- 04开头 = 随机对照试验(RCT)\n\n"
    "这些编号是数据库的标准文献索引号，其存在本身不构成\"编造\"或\"无法追溯\"。\n"
    "判断是否存在虚构/编造时，应关注以下情况：\n"
    "- 回答中声称的研究结论与引用的文献类型明显不匹配（如引用了指南编号却声称是RCT结论）\n"
    "- 回答中出现了未使用任何引用编号的具体数据或统计结果\n"
    "- 回答中的结论与同领域公认知识明显矛盾且无合理解释\n"
    "- 不能仅因为引用编号是数据库索引格式就判定为\"无法追溯\"或\"编造嫌疑\""
)


def build_scoring_prompt(question, pico, answer, rubrics):
    rubrics_text = ""
    for r in rubrics:
        weight = r["weight"]
        if weight > 0:
            score_range = f"0~{weight}的整数"
            if weight >= 3:
                scoring_guide = (
                    f"0分=完全未涉及或内容无关；"
                    f"1分=仅笼统提及，缺乏具体来源/数据/年份，或仅覆盖单一方面；"
                    f"2分=有一定具体性（如提到具体指南名称或研究类型），但仍有明显遗漏或不够深入；"
                    f"{weight}分=全面且具体，有明确来源、数据支撑，覆盖多个方面，找不到明显不足"
                )
            elif weight == 2:
                scoring_guide = (
                    f"0分=完全未涉及或内容无关；"
                    f"1分=有所涉及但不够具体或有明显遗漏；"
                    f"2分=具体且充分，找不到明显不足"
                )
            else:
                scoring_guide = (
                    f"0分=未涉及或不符合；"
                    f"1分=明确符合且有具体内容支撑"
                )
        else:
            score_range = f"只能是 0 或 {weight}"
            scoring_guide = (
                f"0分=经逐条审查后确认不存在此问题；"
                f"{weight}分=发现存在此问题（须在理由中指出具体位置）"
            )
        rubrics_text += (
            f"\n【{r['rubric_id']}】{r['dimension']}\n"
            f"- 标准：{r['criterion']}\n"
            f"- 权重：{weight}\n"
            f"- 打分范围：{score_range}\n"
            f"- 评分指引：{scoring_guide}\n"
            f"- 验证提示：{r['verification_hint']}\n"
        )

    critique_list = "\n".join(
        f"- {r['rubric_id']}（{r['dimension']}）：该维度有哪些具体不足？缺少了什么？"
        for r in rubrics
    )
    scores_template = ",\n".join(
        '    {"rubric_id": "' + r["rubric_id"] + '", "score": <分数>, "reason": "<基于上述缺陷分析的具体理由>"}'
        for r in rubrics
    )

    return (
        "请对以下AI系统的回答进行严格评审。\n\n"
        "【临床问题】\n"
        f"{question}\n\n"
        "【PICO框架】\n"
        f"- P（人群）：{pico.get('P', '')}\n"
        f"- I（干预）：{pico.get('I', '')}\n"
        f"- C（对照）：{pico.get('C', '')}\n"
        f"- O（结局）：{pico.get('O', '')}\n\n"
        "【AI系统的回答】\n"
        f"{answer[:8000]}\n\n"
        "---\n\n"
        f"【评分标准】（共{len(rubrics)}项）\n"
        f"{rubrics_text}\n"
        "---\n\n"
        "## 请严格按照以下两个步骤完成评审：\n\n"
        "### 第一步：缺陷分析（必须先完成）\n\n"
        "对照PICO框架和每项评分标准，逐一列出回答在每个维度上的具体不足、遗漏或问题。"
        "如果某个维度确实没有明显不足，写\"未发现明显不足\"（但请确认你已仔细检查）。\n\n"
        "请逐项分析：\n"
        f"{critique_list}\n\n"
        "对于负权重项，请特别检查：\n"
        "- 回答中是否有未附引用编号的具体数据或统计结果（可能是编造的）？\n"
        "- 引用编号的文献类型（01指南/02英文指南/03系统评价/04RCT）是否与声称的证据类型一致？\n"
        "- 推理过程是否有逻辑断裂或前后矛盾？\n"
        "- 注意：引用编号[^xx-xxxx-x]本身是数据库标准格式，不能仅因为是编号就判定为编造。\n\n"
        "### 第二步：基于缺陷分析打分\n\n"
        "根据第一步中发现的具体问题，对每项评分标准打分。"
        "**你的打分必须与你在缺陷分析中列出的问题一致——如果你在某项列出了不足，该项就不能给满分。**\n\n"
        f"请将打分结果以如下JSON格式输出（放在{FENCE}json代码块中）：\n\n"
        f"{FENCE}json\n"
        "{\n"
        '  "scores": [\n'
        f"{scores_template}\n"
        "  ]\n"
        "}\n"
        f"{FENCE}\n\n"
        "重要提醒：\n"
        "1. 正权重项：如果缺陷分析中列出了不足，必须相应扣分，不能列了问题却仍给满分\n"
        "2. 负权重项：如果缺陷分析中发现了问题，必须触发扣分\n"
        "3. 如果回答为空或报错，所有正权重项给0分"
    )


async def score_single(client, item, semaphore):
    async with semaphore:
        eval_id = item["eval_id"]
        answer = item.get("answer", "")
        rubrics = item.get("rubrics", [])
        is_error = answer.startswith("[ERROR]")
        prompt = build_scoring_prompt(
            question=item["question"],
            pico=item.get("pico", {}),
            answer=answer,
            rubrics=rubrics
        )
        max_possible, min_possible = compute_question_score_range(rubrics)
        for attempt in range(3):
            try:
                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=4000,
                )
                content = response.choices[0].message.content
                scores_data = extract_json_from_text(content)
                rubric_scores = []
                total_raw = 0
                rubrics_map = {r["rubric_id"]: r for r in rubrics}
                for s in scores_data.get("scores", []):
                    rid = s["rubric_id"]
                    score = s["score"]
                    rubric = rubrics_map.get(rid, {})
                    weight = rubric.get("weight", 0)
                    if weight > 0:
                        score = max(0, min(weight, score))
                    else:
                        score = weight if score != 0 else 0
                    total_raw += score
                    rubric_scores.append({
                        "rubric_id": rid,
                        "dimension": rubric.get("dimension", ""),
                        "dimension_id": rubric.get("dimension_id", ""),
                        "weight": weight,
                        "score": score,
                        "reason": s.get("reason", "")
                    })
                return {
                    "eval_id": eval_id,
                    "is_error": is_error,
                    "rubric_scores": rubric_scores,
                    "total_raw_score": total_raw,
                    "max_possible_score": max_possible,
                    "min_possible_score": min_possible,
                    "score_percentage": to_percentage_dynamic(total_raw, max_possible, min_possible)
                }
            except Exception as e:
                if attempt < 2:
                    print(f"  {eval_id} 打分失败 (尝试 {attempt+1}): {e}, 重试...")
                    await asyncio.sleep(3)
                else:
                    print(f"  {eval_id} 打分最终失败: {e}")
                    return {
                        "eval_id": eval_id,
                        "is_error": True,
                        "rubric_scores": [],
                        "total_raw_score": 0,
                        "max_possible_score": max_possible,
                        "min_possible_score": min_possible,
                        "score_percentage": to_percentage_dynamic(0, max_possible, min_possible),
                        "scoring_error": str(e)
                    }


def compute_summary(scored_results, raw_results):
    results_map = {r["eval_id"]: r for r in raw_results}
    all_scores, dimension_scores, specialty_scores, grade_scores = [], {}, {}, {}
    for sr in scored_results:
        if sr.get("scoring_error"):
            continue
        eval_id = sr["eval_id"]
        raw = results_map.get(eval_id, {})
        pct = sr["score_percentage"]
        all_scores.append(pct)
        for rs in sr.get("rubric_scores", []):
            dim_id = rs.get("dimension_id", "")
            if not dim_id:
                continue
            if dim_id not in dimension_scores:
                dimension_scores[dim_id] = {"name": rs["dimension"], "scores": [], "weights": [], "max_weights": []}
            dimension_scores[dim_id]["scores"].append(rs["score"])
            dimension_scores[dim_id]["weights"].append(rs["weight"])
            if rs["weight"] > 0:
                dimension_scores[dim_id]["max_weights"].append(rs["weight"])
        spec = raw.get("specialty", "未知")
        specialty_scores.setdefault(spec, []).append(pct)
        grade = raw.get("grade", "未知")
        grade_scores.setdefault(grade, []).append(pct)

    dim_pct = {}
    for dim_id, data in dimension_scores.items():
        positive_sum = sum(s for s, w in zip(data["scores"], data["weights"]) if w > 0)
        max_sum = sum(data["max_weights"]) if data["max_weights"] else 1
        negative_sum = sum(s for s, w in zip(data["scores"], data["weights"]) if w < 0)
        dim_score = max(0, (positive_sum + negative_sum) / max_sum * 100) if max_sum > 0 else 0
        dim_pct[dim_id] = {"name": data["name"], "score": round(dim_score, 1), "sample_count": len(data["scores"])}

    n = len(all_scores)
    mean = sum(all_scores) / n if n else 0
    return {
        "overall_score": round(mean, 1),
        "overall_std": round((sum((s - mean)**2 for s in all_scores) / n)**0.5, 1) if n else 0,
        "total_evaluated": n,
        "score_min": round(min(all_scores), 1) if n else 0,
        "score_max": round(max(all_scores), 1) if n else 0,
        "score_median": round(sorted(all_scores)[n // 2], 1) if n else 0,
        "dimension_scores": dim_pct,
        "specialty_scores": {k: round(sum(v)/len(v), 1) for k, v in specialty_scores.items()},
        "grade_scores": {k: round(sum(v)/len(v), 1) for k, v in grade_scores.items()}
    }


async def run_scoring():
    answer_files = glob.glob(os.path.join(EVAL_RESULTS_DIR, f"{FILE_PREFIX}_*_answers.json"))
    if not answer_files:
        print(f"未找到 {FILE_PREFIX} 回答文件，请先运行 eval_runner_v20.py")
        return
    latest_file = max(answer_files, key=os.path.getmtime)
    print(f"加载回答文件: {latest_file}")
    with open(latest_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = data["results"]
    eval_meta = data["eval_meta"]
    print(f"共 {len(results)} 条待打分")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scored_path = os.path.join(EVAL_RESULTS_DIR, f"{FILE_PREFIX}_strict_v4_{timestamp}_scored.json")
    print(f"输出文件: {scored_path}")

    scored_results = []
    client = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE)
    semaphore = asyncio.Semaphore(3)

    print(f"开始打分: {len(results)} 条 (strict v4 - Chain-of-Critique)")
    for i, item in enumerate(results):
        print(f"[{i+1}/{len(results)}] 打分: {item['eval_id']}...")
        start = time.time()
        score_result = await score_single(client, item, semaphore)
        elapsed = time.time() - start
        scored_results.append(score_result)
        print(f"  得分: {score_result['score_percentage']}分 ({elapsed:.1f}s)")
        if (i + 1) % 10 == 0 or (i + 1) == len(results):
            summary = compute_summary(scored_results, results)
            output = {
                "eval_meta": {**eval_meta, "scoring_version": "strict_v4", "scoring_time": timestamp,
                              "file_prefix": FILE_PREFIX},
                "scored_results": scored_results,
                "summary": summary
            }
            with open(scored_path, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            print(f"  已保存 {len(scored_results)} 条 | 当前均分: {summary['overall_score']}")
        await asyncio.sleep(1)

    summary = compute_summary(scored_results, results)
    output = {
        "eval_meta": {**eval_meta, "scoring_version": "strict_v4", "scoring_time": timestamp,
                      "file_prefix": FILE_PREFIX},
        "scored_results": scored_results,
        "summary": summary
    }
    with open(scored_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"打分完成！标识: {FILE_PREFIX}")
    print(f"总体得分: {summary['overall_score']}分 (标准差: {summary['overall_std']})")
    print(f"分数范围: {summary['score_min']} ~ {summary['score_max']}")
    print(f"中位数: {summary['score_median']}")
    print(f"各维度得分:")
    for dim_id, dim_data in summary["dimension_scores"].items():
        print(f"  {dim_data['name']}: {dim_data['score']}分 (n={dim_data['sample_count']})")
    print(f"\n结果保存至: {scored_path}")
    return scored_path


if __name__ == "__main__":
    asyncio.run(run_scoring())
