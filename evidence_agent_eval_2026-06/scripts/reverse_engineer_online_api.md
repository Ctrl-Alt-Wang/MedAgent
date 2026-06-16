# 线上循证智能体接口逆向 · 方法记录

> 目标：在只知道前端入口 `https://www.infox-med.com/20/#/aiChat/6`（SPA）的情况下，
> 还原出真实的提问接口、请求格式与响应结构。结论见 reports/04。

## 为什么不能盲试

SPA 的提问接口由 JS 在运行时调用，且需鉴权 Token。盲试 `api.infox-med.com` 下 ~40 种常见路径
（`/chat`、`/query`、`/ask`、`/stream`、`/agent` …）全部 404，只有只读的 `/infoxmed20/getPromptExamples` 能通。
正确做法是**从前端 JS bundle 逆向**。

## 步骤（命令记录）

```bash
# 1) 取 SPA 入口，找 JS 资源
curl -s "https://www.infox-med.com/20/" -o idx.html
grep -oE 'src="[^"]*\.js"' idx.html            # → ./assets/index-*.js

# 2) 下载 JS bundle（~4.3MB）
curl -s "https://www.infox-med.com/20/assets/index-XXXX.js" -o app.js

# 3) 找 API base 与端点
grep -oE 'baseURL=[^,;]*'            app.js     # → https://api.infox-med.com
grep -oE '/infoxmed20[a-z]*/[A-Za-z0-9_]+' app.js | sort -u

# 4) 找 SSE 发送机制
grep -oE '.{200}fetchEventSource.{260}' app.js  # → sseApi: POST ${baseUrl}${url}, headers{Token, X-API-Version}
grep -oE '/infoxmed20sse/[A-Za-z0-9_]+'  app.js  # → sseEndpoint / cancelSse / reconnectSse
grep -oE 'aiBaseUrl=`[^`]*`'             app.js  # → ${baseURL}/infoxmed20sse/sseEndpoint

# 5) 找循证智能体标识与版本号
grep -oE '.{0,40}循证.{0,80}'            app.js  # → fId:"6", title:"循证智能体", path:"/aiChat/6"
grep -oE 'getAiVersionByFid=[^;]*'       app.js  # → aiVersionMaps[t]||"5002018"

# 6) 找请求体字段
grep -oE 'functionId:[a-zA-Z.]+[,}].{0,120}' app.js
#   → {functionId, messageType:1, ts, content:"", attachment:{}, sessionId:sId||null, isNew:sId?0:1}
```

## 从浏览器补齐（无法静态拿到的）

- **Token**：F12 → Network → 任一 `sseEndpoint` 请求 → Headers → `Token: ...`（或 Application → Local Storage）。
- **X-Api-Version**：同上请求头，fId=6 实际为 `5003016`。

## 最终契约

见 [reports/04_online_agent_reverse_engineering.md](../reports/04_online_agent_reverse_engineering.md)。
复现脚本见 [online_eval_runner.py](online_eval_runner.py)。

## 解析坑

- 响应 `Content-Encoding: br` + 中文 → 请求发 `Accept-Encoding: identity` 取消压缩，按 **UTF-8** 解码。
- 答案为 `messageType=8` 的流式片段；`49`=工具状态、`50`=检索文档、`61`=证据卡。
- 引用格式 `[^xx-xxxx-x$$N]`，统计引用时正则需兼容 `$$N` 后缀：`\[\^([0-9]{2}-[0-9a-z]+-[0-9]+)(?:\$\$[0-9]+)?\]`。
