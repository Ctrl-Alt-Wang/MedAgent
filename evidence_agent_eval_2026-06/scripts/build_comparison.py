# -*- coding: utf-8 -*-
"""
对比与产物生成（无凭证）。输入 answers_v17.json + online_answers.json，输出：
  - answers_online.md            线上问答对（同 v17 格式）
  - comparison_v17_vs_online.md  逐题并排 + 线上引用可追溯来源 + 证据卡
  - compare_data.json            结构化中间数据

用法：在含上述两个输入文件的目录下 `python build_comparison.py`
"""
import sys, io, json, re, statistics as st
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

CITE = re.compile(r'\[\^([0-9]{2}-[0-9a-z]+-[0-9]+)(?:\$\$[0-9]+)?\]')   # 兼容线上 $$N 后缀
KEY = {'search_guidelinezh_db': '中文指南', 'search_guideline_db': '英文指南',
       'search_meta_db': '系统评价/Meta', 'search_clinical_db': 'RCT'}

online = json.load(open('online_answers.json', encoding='utf-8'))
v17 = json.load(open('answers_v17.json', encoding='utf-8'))['results']
v17map = {r['eval_id']: r for r in v17}

# ---------- answers_online.md ----------
L = ['# 线上生产循证智能体 — 10题评测答案\n',
     '> 来源：infox-med 生产循证智能体（functionId=6）　|　共 %d 题\n' % len(online)]
for i, r in enumerate(online, 1):
    ans = r['answer'].replace('[stop]', '').rstrip()
    nc = len(dict.fromkeys(CITE.findall(ans)))
    L.append("\n---\n\n## %d. %s\n" % (i, r['question']))
    L.append("*引用数 %d　|　%d 字　|　%ss　|　证据卡 %d 张*\n" % (nc, r['answer_length'], r['elapsed_seconds'], len(r.get('cards', []))))
    L.append(ans + "\n")
open('answers_online.md', 'w', encoding='utf-8').write("\n".join(L))

# ---------- comparison + compare_data ----------
rows, detail = [], []
for o in online:
    eid, q = o['eval_id'], o['question']
    ocids = list(dict.fromkeys(CITE.findall(o['answer'])))
    cmap = {c: o.get('all_docs', {}).get(c) for c in ocids}
    omatched = sum(1 for v in cmap.values() if v)
    vr = v17map.get(eid, {})
    rows.append((eid, q, vr.get('num_citations', 0), vr.get('answer_length', 0),
                 len(ocids), omatched, o['answer_length'], len(o.get('cards', []))))
    detail.append((eid, q, vr.get('answer', ''), vr.get('num_citations', 0),
                   o['answer'], ocids, cmap, o.get('cards', [])))

C = ["# 循证智能体对比：线上生产版 vs 本地 v17（同 10 题）\n",
     "## 一、总览对比\n",
     "| 题 | v17 引用数 | v17 字数 | 线上引用数 | 线上可追溯 | 线上字数 | 线上证据卡 |",
     "|----|-----------|---------|-----------|-----------|---------|-----------|"]
for r in rows:
    C.append(f"| {r[0]} | {r[2]} | {r[3]} | {r[4]} | {r[5]}/{r[4]} | {r[6]} | {r[7]} |")
C.append("")
C.append(f"- v17 平均：引用 {st.mean([r[2] for r in rows]):.1f} 条 / {st.mean([r[3] for r in rows]):.0f} 字")
C.append(f"- 线上平均：引用 {st.mean([r[4] for r in rows]):.1f} 条 / {st.mean([r[6] for r in rows]):.0f} 字 / {st.mean([r[7] for r in rows]):.1f} 张证据卡")
C.append("\n## 二、逐题对比\n")
for d in detail:
    C.append(f"\n---\n\n### {d[0].upper()}. {d[1]}\n")
    C.append(f"#### 🟦 本地 v17（引用 {d[3]} 条）\n\n{d[2].strip()}\n")
    C.append(f"#### 🟩 线上生产版（引用 {len(d[5])} 条）\n\n{d[4].strip()}\n")
    if any(d[6].values()):
        C.append("\n**线上引用来源（可追溯）：**\n")
        for cid in d[5]:
            v = d[6].get(cid)
            if v:
                C.append(f"- `[^{cid}]` 〔{KEY.get(v.get('key'), v.get('key',''))}〕{v.get('title','')}（{v.get('pub','') or '—'}）")
    if d[7]:
        C.append("\n**线上证据卡（mt=61）：**\n")
        for c in d[7]:
            C.append(f"- [{c.get('label','')}] {c.get('title','')}（doc_id {c.get('doc_id','')}）")
    C.append("")
open('comparison_v17_vs_online.md', 'w', encoding='utf-8').write("\n".join(C))

json.dump({"rows": rows, "detail": [{"eid": d[0], "q": d[1], "v17_answer": d[2], "v17_cites": d[3],
           "online_answer": d[4], "online_cids": d[5], "online_cmap": d[6], "cards": d[7]} for d in detail]},
          open('compare_data.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

print("wrote answers_online.md / comparison_v17_vs_online.md / compare_data.json")
