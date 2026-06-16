# -*- coding: utf-8 -*-
"""
本地 v17 模型评测：paramiko 远程驱动服务器上的评测管线，跑完 10 题。

前置（服务器上已就绪，详见 reports/03）：
  - vLLM serve grpo_v17_plus_small_step336_hf 到 :8400（hermes 工具解析）
  - /workspace/eval_run/ 下部署 main_api_openai.py + sub_agents/ + eval_dataset_10.json
  - main_api_openai.py 的 ModelSettings 须含:
        tool_choice="search_embedding_db",
        temperature=0.7, top_p=0.8,
        extra_body={"repetition_penalty":1.05, "top_k":20}
  - :9999 Agent 服务在 screen 中运行

用法：
  export SSH_HOST=...  SSH_PORT=23  SSH_USER=root  SSH_PASSWORD=...
  python local_v17_eval_runner.py

注意：凭证全部走环境变量，勿硬编码。长任务在服务器端用 screen 跑，避免 SSH 断连。
"""
import os, sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import paramiko

HOST = os.environ.get("SSH_HOST", "<SERVER_IP>")
PORT = int(os.environ.get("SSH_PORT", "23"))
USER = os.environ.get("SSH_USER", "root")
PW   = os.environ.get("SSH_PASSWORD", "<SSH_PASSWORD>")
PY   = "/usr/local/miniconda3/envs/py312/bin/python"
REMOTE_DIR = "/workspace/eval_run"

# 服务器端跑的批量评测脚本（通过 A2A message/stream 打 :9999，收 SSE）
REMOTE_RUNNER = r'''
import json, uuid, requests, sseclient, sys, io, time, re
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8",errors="replace")
AGENT="http://localhost:9999"
ds=json.load(open("/workspace/eval_run/eval_dataset_10.json",encoding="utf-8"))
results=[]
for i,item in enumerate(ds):
    q=item["question"]; eid=item.get("eval_id",f"q{i+1}"); rid=uuid.uuid4().hex
    payload={"jsonrpc":"2.0","id":rid,"method":"message/stream","params":{"message":{
        "role":"user","parts":[{"type":"text","text":q}],"messageId":rid,
        "metadata":{"user_data":eid,"language":"Chinese"}}}}
    t0=time.time(); ans=""
    try:
        r=requests.post(AGENT+"/",json=payload,headers={"Accept":"text/event-stream",
            "Content-Type":"application/json"},stream=True,timeout=240)
        for ev in sseclient.SSEClient(r).events():
            try: d=json.loads(ev.data)
            except: continue
            res=d.get("result",{})
            for a in res.get("artifacts",[]):
                for p in a.get("parts",[]):
                    if p.get("kind")=="text": ans+=p.get("text","")
            if res.get("status",{}).get("state","") in ("completed","failed"): break
    except Exception as e: ans=f"[ERROR] {e}"
    cites=re.findall(r"\[\^[0-9]{2}-[0-9a-z]+-[0-9]+\]",ans)
    results.append({"eval_id":eid,"question":q,"answer":ans,"answer_length":len(ans),
                    "elapsed_seconds":round(time.time()-t0,1),"num_citations":len(cites)})
    print(f"[{i+1}/{len(ds)}] {eid} {results[-1]['elapsed_seconds']}s {len(ans)}chars cites={len(cites)}")
    json.dump({"model":"grpo_v17_336","results":results},
              open("/workspace/eval_run/answers_v17.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
print("DONE, saved answers_v17.json")
'''

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, port=PORT, username=USER, password=PW, timeout=30)
    ssh.get_transport().set_keepalive(15)
    with ssh.open_sftp().open(f"{REMOTE_DIR}/run_all.py", "w") as f:
        f.write(REMOTE_RUNNER)
    # 在 screen 后台跑，避免 SSH 断连杀进程
    ssh.exec_command("screen -S run10 -X quit 2>/dev/null || true"); time.sleep(2)
    ssh.exec_command(f"cd {REMOTE_DIR} && screen -dmS run10 bash -c "
                     f"'{PY} run_all.py > {REMOTE_DIR}/run10.log 2>&1'")
    print("已在 screen 'run10' 启动；轮询 /workspace/eval_run/run10.log 查看进度，"
          "完成后用 sftp 取 answers_v17.json")
    ssh.close()

if __name__ == "__main__":
    main()
