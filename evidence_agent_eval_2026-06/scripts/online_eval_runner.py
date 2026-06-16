# -*- coding: utf-8 -*-
"""
线上生产循证智能体（infox-med，functionId=6）批量提问 + SSE 解析。

用法：
    export ONLINE_TOKEN="<你的登录Token>"        # 从浏览器 F12 请求头 Token 取得
    python online_eval_runner.py                 # 读取 eval_dataset_10.json，输出 online_answers.json

接口契约（逆向自前端 JS bundle，详见 reports/04）：
    POST https://api.infox-med.com/infoxmed20sse/sseEndpoint   (SSE)
    headers: Content-Type/json, Token, X-Api-Version=5003016, Accept-Encoding=identity
    body: {functionId:"6", messageType:1, ts, content, attachment:{}, sessionId:null, isNew:1}
    响应 messageType: 8=答案正文 / 49=工具状态 / 50=检索文档 / 61=证据卡
"""
import os, sys, io, json, time, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import requests

URL = "https://api.infox-med.com/infoxmed20sse/sseEndpoint"
TOKEN = os.environ.get("ONLINE_TOKEN", "<ONLINE_TOKEN>")   # 凭证走环境变量，勿硬编码
HEADERS = {
    "Content-Type": "application/json", "Token": TOKEN, "X-Api-Version": "5003016",
    "Origin": "https://www.infox-med.com", "Referer": "https://www.infox-med.com/",
    "Accept": "text/event-stream", "Accept-Encoding": "identity", "User-Agent": "Mozilla/5.0",
}

def assemble(chunks):
    """messageType=8 片段拼接；兼容累积式与增量式两种流。"""
    if not chunks:
        return ""
    if len(chunks) > 1 and all(len(chunks[i]) <= len(chunks[i+1]) for i in range(len(chunks)-1)):
        return chunks[-1]   # 累积式 -> 取最后一帧
    return "".join(chunks)  # 增量式 -> 拼接

def ask(question):
    body = {"functionId": "6", "messageType": 1, "ts": int(time.time()*1000),
            "content": question, "attachment": {}, "sessionId": None, "isNew": 1}
    r = requests.post(URL, headers=HEADERS, json=body, stream=True, timeout=300)
    if r.status_code != 200:
        return {"answer": f"[ERROR] HTTP {r.status_code}: {r.text[:200]}", "docs": {}, "cards": []}
    ans, docs, cards = [], {}, []
    for bline in r.iter_lines(decode_unicode=False):
        if not bline:
            continue
        line = bline.decode('utf-8', errors='replace')
        if not line.startswith("data:"):
            continue
        try:
            obj = json.loads(line[5:].strip())
        except Exception:
            continue
        mt, content = obj.get("messageType"), obj.get("content", "")
        if mt == 8 and content:
            ans.append(content)
        elif mt == 50 and content:
            try:
                for grp in json.loads(content):
                    for d in grp.get("data", []):
                        for ms in d.get("match_sentences", []):
                            if ms.get("id"):
                                docs[ms["id"]] = {"title": d.get("title"), "key": d.get("key"),
                                                  "pub": d.get("publish_time", "")}
            except Exception:
                pass
        elif mt == 61 and content:
            try:
                for c in json.loads(content):
                    cards.append({"doc_id": c.get("doc_id"), "label": c.get("card_label"),
                                  "title": c.get("title")})
            except Exception:
                pass
    return {"answer": assemble(ans), "docs": docs, "cards": cards}

def main():
    ds = json.load(open("eval_dataset_10.json", encoding="utf-8"))
    out = []
    for i, item in enumerate(ds):
        q = item["question"]; eid = item.get("eval_id", f"q{i+1}")
        t0 = time.time()
        try:
            res = ask(q)
        except Exception as e:
            res = {"answer": f"[ERROR] {e}", "docs": {}, "cards": []}
        el = round(time.time()-t0, 1)
        cites = list(dict.fromkeys(re.findall(r"\[\^([0-9]{2}-[0-9a-z]+-[0-9]+)(?:\$\$[0-9]+)?\]", res["answer"])))
        out.append({"eval_id": eid, "question": q, "answer": res["answer"],
                    "answer_length": len(res["answer"]), "elapsed_seconds": el,
                    "citations_used": cites, "all_docs": res["docs"], "cards": res["cards"]})
        print(f"[{i+1}/{len(ds)}] {eid} {el}s {len(res['answer'])}chars cites={len(cites)} docs={len(res['docs'])}")
        json.dump(out, open("online_answers.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        time.sleep(3)
    print("DONE -> online_answers.json")

if __name__ == "__main__":
    main()
