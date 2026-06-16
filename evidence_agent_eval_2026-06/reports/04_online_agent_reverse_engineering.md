# 04 · 线上生产循证智能体接口逆向

## 背景

线上循证智能体入口是 SPA：`https://www.infox-med.com/20/#/aiChat/6`。提问接口由前端 JS 发起，无法靠盲试路径找到（之前试 ~40 种 `api.infox-med.com` 路径全 404，只有只读的 `getPromptExamples` 能通）。解决办法：**从前端 JS bundle 逆向**。

## 逆向步骤

1. 取 SPA 入口 HTML，找到 JS 资源：
   ```
   GET https://www.infox-med.com/20/  →  ./assets/index-*.js (4.3MB)
   ```
2. 下载 JS bundle，grep API 模式：
   - API base：`baseURL = isTest ? "https://devapi.infox-med.com" : "https://api.infox-med.com"`
   - 普通接口前缀 `/infoxmed20/`（如 `getPromptExamples`、`getChatHistoryList3` 等）。
   - **SSE 专用前缀 `/infoxmed20sse/`**：`sseEndpoint`、`cancelSse`、`reconnectSse`。
3. 找到 SSE 发送逻辑：
   ```js
   aiBaseUrl = `${baseURL}/infoxmed20sse/sseEndpoint`
   sseApi → fetchEventSource(`${baseUrl}${url}`, {
     method: "POST",
     body: JSON.stringify(data),
     headers: { "Content-Type":"application/json", Token: getToken(), ...,
                "X-API-Version": getAiVersionByFid(fId) }
   })
   ```
4. 循证智能体标识：`{ fId:"6", title:"循证智能体", path:"/aiChat/6" }`。
5. 版本号：`getAiVersionByFid(t)=aiVersionMaps[t]||"5002018"`；实际抓包得 fId=6 → `X-Api-Version: 5003016`。
6. 鉴权：`Token: getToken()`（登录后令牌），从浏览器 F12 请求头取得。

## 完整接口契约

**Endpoint**：`POST https://api.infox-med.com/infoxmed20sse/sseEndpoint`（SSE 流）

**Headers**：
```
Content-Type: application/json
Token: <ONLINE_TOKEN>            # 登录令牌（脱敏）
X-Api-Version: 5003016           # fId=6 循证智能体
Origin: https://www.infox-med.com
Referer: https://www.infox-med.com/
Accept-Encoding: identity        # 避免 br 压缩，便于流式读取
```

**请求体**：
```json
{
  "functionId": "6",
  "messageType": 1,
  "ts": 1781596907000,
  "content": "<临床问题>",
  "attachment": {},
  "sessionId": null,
  "isNew": 1
}
```

## 响应（SSE）结构

每条 `data:{...}` 是 JSON，关键字段 `messageType`：

| messageType | 含义 | 内容 |
|---|---|---|
| (首条) | 建立连接 | `{linkId}` |
| **8** | **答案正文**（流式片段） | Markdown 文本，含 `[^xx-xxxx-x$$N]` 引用 |
| **49** | 工具调用状态 | `search_embedding_db` 的 Working/Done + PICO 参数 |
| **50** | 检索文档 | 各库 `data[]`：`id/title/key/match_sentences[{id,...}]` |
| **61** | 证据卡 | `{doc_id, card_label(RCT/指南/...), title, user_question}` |
| (ping) | 保活 | `event:ping` |

> 解析要点：响应是 `Content-Encoding: br` 且中文 → 发 `Accept-Encoding: identity` 取消压缩、按 **UTF-8** 解码；答案为 messageType=8 的片段拼接。

## 与本地架构的一致性

线上循证智能体**同样**：
- 调 `search_embedding_db` 抽 **PICO** 检索；
- 检索 **4 库**（中文指南/英文指南/系统评价Meta/RCT）；
- 用 **`[^xx-xxxx-x]` 同体系引用**（仅多 `$$N` 显示序号）；
- 输出**分层结构**（直接结论 → 分层证据 → 临床决策）。

差异主要在**检索规模**（线上 top10 不截断 vs 本地 top2+1500截断）与**证据卡可视化**（messageType=61），详见 [05](05_v17_vs_online_comparison.md)、[06](06_context_length_retrieval_analysis.md)。

## 复现脚本

见 `scripts/online_eval_runner.py`（已脱敏，Token 走环境变量）。
