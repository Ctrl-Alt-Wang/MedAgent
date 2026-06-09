#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""评估：Qwen2.5-14B-GRPO-v20 (step458, final) + search_embedding_db
标识：grpo_v20_step458
模型：/workspace/grpo_v20_step458_hf (FSDP→HF 转换，ebm_agent_14b_grpo_4gpu_v20_sft global_step_458)
      base: Qwen2.5-14B-Instruct, GRPO-SFT 4GPU AgentLightning, step458 (final)
服务：vllm @ server:8001 (via SSH tunnel localhost:8015)
Agent：localhost:9998, main_api_openai.py (OpenAI agents SDK)
"""

import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import uuid, json, requests, os, time, sseclient
from datetime import datetime

EVAL_DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_dataset_final_v5.json")
EVAL_RESULTS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results")
AGENT_URL         = "http://localhost:9998"
TIMEOUT           = 600
REQUEST_INTERVAL  = 3
FILE_PREFIX       = "grpo_v20_step458"
MODEL_DESC        = (
    "Qwen2.5-14B-GRPO-v20 (step458 final, AgentLightning, 4GPU)"
    " + search_embedding_db"
    " (OpenAI agents SDK, main_api_openai.py, 4DB, top2, 1500-cut)"
    " | vllm bfloat16, max_model_len=32768, server:8001 via SSH tunnel local:8015"
)


def send_message_streaming(question: str, eval_id: str) -> dict:
    request_id = uuid.uuid4().hex
    payload = {
        "jsonrpc": "2.0", "id": request_id, "method": "message/stream",
        "params": {"message": {"role": "user", "parts": [{"type": "text", "text": question}],
            "messageId": request_id, "metadata": {"language": "Chinese"}}}
    }
    all_text_parts, artifact_text = [], ""
    try:
        response = requests.post(f"{AGENT_URL}/", json=payload,
            headers={"Accept": "text/event-stream", "Content-Type": "application/json"},
            stream=True, timeout=TIMEOUT)
        if response.status_code != 200:
            return {"answer": f"[ERROR] HTTP {response.status_code}: {response.text[:200]}", "request_id": request_id}
        for event in sseclient.SSEClient(response).events():
            try: data = json.loads(event.data)
            except: continue
            result = data.get("result", {})
            for artifact in result.get("artifacts", []):
                for part in artifact.get("parts", []):
                    if part.get("kind") == "text": artifact_text += part.get("text", "")
            status = result.get("status", {})
            msg = status.get("message", {})
            if isinstance(msg, dict):
                for part in msg.get("parts", []):
                    if part.get("kind") == "text": all_text_parts.append(part.get("text", ""))
            if status.get("state") in ("completed", "failed"): break
    except requests.exceptions.Timeout:
        return {"answer": "[ERROR] 请求超时", "request_id": request_id}
    except Exception as e:
        return {"answer": f"[ERROR] {str(e)}", "request_id": request_id}
    final_answer = artifact_text if artifact_text else "".join(all_text_parts)
    return {"answer": final_answer.replace("\~", "~"), "request_id": request_id}


def run_eval():
    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"加载评估数据集: {len(dataset)} 条")

    try:
        r = requests.get(f"{AGENT_URL}/.well-known/agent.json", timeout=10)
        if r.status_code != 200:
            print(f"服务异常: {r.status_code}"); return
        print(f"连接成功: {r.json().get('name', 'unknown')}")
    except Exception as e:
        print(f"无法连接到智能体服务(port 9998): {e}")
        print("请确认已启动：")
        print("  1. python ssh_tunnel_v20.py  (另一个终端)")
        print("  2. cd ../backend/simple_agent && python main_api_openai.py --port 9998  (另一个终端)")
        return

    os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)
    eval_name = f"{FILE_PREFIX}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_path = os.path.join(EVAL_RESULTS_DIR, f"{eval_name}_answers.json")

    # 断点续写
    completed = {}
    existing_files = [f for f in os.listdir(EVAL_RESULTS_DIR)
                      if f.endswith("_answers.json") and f.startswith(FILE_PREFIX)]
    if existing_files:
        latest = max(existing_files, key=lambda f: os.path.getmtime(os.path.join(EVAL_RESULTS_DIR, f)))
        try:
            with open(os.path.join(EVAL_RESULTS_DIR, latest), "r", encoding="utf-8") as f:
                existing = json.load(f)
            for item in existing.get("results", []):
                if item.get("answer") and not item["answer"].startswith("[ERROR]"):
                    completed[item["eval_id"]] = item
            if completed:
                print(f"断点续写: 已有 {len(completed)} 条有效结果 <- {latest}")
                output_path = os.path.join(EVAL_RESULTS_DIR, latest)
                eval_name = existing.get("eval_meta", {}).get("eval_name", eval_name)
        except: pass

    results = list(completed.values())
    total, errors = len(dataset), 0

    for i, item in enumerate(dataset):
        eval_id, question = item["eval_id"], item["question"]
        if eval_id in completed:
            print(f"[{i+1}/{total}] 跳过(已完成): {eval_id}"); continue
        print(f"[{i+1}/{total}] 评估: {eval_id} - {question[:50]}...")
        start = time.time()
        try:
            response = send_message_streaming(question, eval_id)
            elapsed = time.time() - start
            result = {
                "eval_id": eval_id, "question": question, "pico": item.get("pico", {}),
                "specialty": item.get("specialty", ""), "grade": item.get("grade", ""),
                "question_type": item.get("question_type", ""), "final_eval_id": item.get("final_eval_id", ""),
                "answer": response["answer"], "answer_length": len(response["answer"]),
                "elapsed_seconds": round(elapsed, 1), "request_id": response["request_id"],
                "rubrics": item.get("rubrics", [])
            }
            if response["answer"].startswith("[ERROR]"):
                errors += 1; print(f"  失败 ({elapsed:.1f}s): {response['answer'][:100]}")
            else:
                print(f"  完成 ({elapsed:.1f}s, {len(response['answer'])}字符): {response['answer'][:60].replace(chr(10),' ')}...")
        except Exception as e:
            elapsed = time.time() - start; errors += 1
            result = {"eval_id": eval_id, "question": question, "pico": item.get("pico", {}),
                "specialty": item.get("specialty", ""), "grade": item.get("grade", ""),
                "question_type": item.get("question_type", ""), "final_eval_id": item.get("final_eval_id", ""),
                "answer": f"[ERROR] {str(e)}", "answer_length": 0,
                "elapsed_seconds": round(elapsed, 1), "request_id": "", "rubrics": item.get("rubrics", [])}
            print(f"  异常 ({elapsed:.1f}s): {e}")
        results.append(result)
        if (i + 1) % 5 == 0 or (i + 1) == total:
            valid_count = len([r for r in results if not r["answer"].startswith("[ERROR]")])
            output = {"eval_meta": {"eval_name": eval_name, "file_prefix": FILE_PREFIX,
                "model_name": MODEL_DESC,
                "timestamp": datetime.now().isoformat(), "dataset": "eval_dataset_final_v5.json",
                "total_questions": total, "completed_questions": valid_count, "error_questions": errors},
                "results": results}
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            print(f"  [保存] {valid_count}/{total} -> {os.path.basename(output_path)}")
        time.sleep(REQUEST_INTERVAL)

    valid_count = len([r for r in results if not r["answer"].startswith("[ERROR]")])
    print(f"\n{'='*50}\n评估完成！总题数: {total}, 成功: {valid_count}, 失败: {errors}")
    print(f"结果文件: {output_path}\n{'='*50}")
    return output_path

if __name__ == "__main__":
    run_eval()
