# Docision HTTP API 与 MCP 接入

[English](API_MCP.md) · [简体中文](API_MCP.zh-CN.md)

本地工作台、HTTP 客户端和 MCP 共用同一个模型进程。HTTP 仅监听 `127.0.0.1`；MCP 使用官方 Python SDK 1.30.0 的 stdio 传输，不开放额外网络端口。MCP 桥接环境不安装模型依赖。

## 1. 启动 HTTP 服务

需要完整 Docision 仓库、Python 3.10+ 和本地完整模型。在 `deployment` 内执行：

```bash
./start.sh --port 7860 --no-browser --model /absolute/path/to/model --device cuda:0
```

也可以省略 `--model`，打开终端打印的完整链接，在工作台加载模型。MCP 的 `load_model` 工具仅能加载其启动配置中 `--model` 指定的**HTTP 主机上的**目录。

首次启动创建 `.runtime/api-token`，以后复用；Linux/macOS 文件权限为 0600。不要提交或分享凭证文件。`--token-file /absolute/path` 可指定另一份凭证。网页链接中的会话值与 API 使用同一凭证。服务重启保留凭证，但清空所有内存任务。修改或删除凭证文件后需要重启 HTTP 服务，并同步更新 MCP 主机的凭证副本。

已运行的旧版本需要 Ctrl+C 后重新启动；只刷新网页不能更新后端。

## 2. HTTP 调用

请求使用 `Authorization: Bearer <凭证文件内容>`；网页兼容 `X-Docision-Token`。Host/Origin 必须对应服务的 `127.0.0.1:端口`。不存在跨域访问设置。

以下 Python 示例在与服务同一主机运行，从允许读取的本地路径取文件并上传。无需安装 HTTP 客户端依赖：

```python
import base64
import json
import time
from pathlib import Path
from http_client import DocisionClient  # 在 deployment 内运行

client = DocisionClient("http://127.0.0.1:7860", ".runtime/api-token")
print(client.request("GET", "health"))
file = Path("/absolute/path/to/document.pdf")
job = client.request("POST", "jobs", {
    "name": file.name,
    "data": base64.b64encode(file.read_bytes()).decode(),
    "questions": [
        {"question": "这一页是否包含表格？", "candidates": ["包含", "不包含", "无法判断"]},
        {"question": "这一页是否有可见签名？", "candidates": ["有", "无", "无法判断"]}
    ]
})
while True:
    status = client.request("GET", "jobs/" + job["id"])
    if status["status"] in ("completed", "failed"):
        break
    time.sleep(0.5)
if status["status"] == "failed":
    raise RuntimeError(status["error"])
print(client.request("GET", "jobs/" + job["id"] + "/results?limit=20"))
# 使用新文件名，避免覆盖已有文件
with Path("docision-output.json").open("xb") as out:
    out.write(client.request("GET", "jobs/" + job["id"] + "/export?format=json", raw=True))
```

### 路由

所有 API 路由都要求凭证。JSON 请求需 `Content-Type: application/json`。

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/api/v1/health` | 服务版本、模型是否就绪、是否忙碌 |
| GET / POST | `/api/v1/model` | 模型状态 / 加载 `{model, device}` |
| GET | `/api/v1/templates` | 问题模板 |
| POST | `/api/v1/score` | 同步推理，负载与 jobs 相同；忙碌返回 409 |
| POST | `/api/v1/jobs` | 提交单文件多问题，立即返回 202 和任务 ID |
| GET | `/api/v1/jobs/{id}` | queued / running / completed / failed 与页面进度 |
| GET | `/api/v1/jobs/{id}/results?offset=0&limit=50` | 分页结果；继续使用 `next_offset`，直到 null |
| GET | `/api/v1/jobs/{id}/export?format=json` | 导出 JSON；format 也可为 csv |
| DELETE | `/api/v1/jobs/{id}` | 删除已完成/失败任务的内存结果；活动任务返回 409 |
| POST | `/api/v1/csv` | 将 `{rows: [...]}` 转成 CSV；工作台人工修正也使用它 |
| GET | `/api/v1/openapi.json` | OpenAPI 3.0.3 契约 |

任务完成不代表每题都成功：查看 `summary.errors` 和每行的 `error`。结果每行保留文件名、页码、问题编号、候选答案、模型判断、实际批量和回退记录。任务 API 的导出读取服务端原始结果；工作台中的人工修正留在浏览器，通过工作台导出，或把修正后的 rows 发送给 `/csv`。

错误格式为 `{"error":{"code":"...","message":"..."}}`。常见状态：400 输入错误，403 凭证/Host/Origin 不匹配，409 模型未加载、忙碌或结果尚未就绪，413 过大，415 内容类型错误，429 队列已满，404 任务不存在/过期。

### 边界与保留期限

- 单文件不超过 25 MiB；JSON 请求不超过 36 MiB；PDF 不超过 100 页。
- 每轮 1–20 题，每题 2–26 个不重复候选答案。上传一次即可逐页回答全部问题。
- 单 GPU 工作线程与网页同步推理共享锁；最多 4 个等待任务，队列满时返回 429，不重载模型。
- 最多保留 32 个任务；完成结果最多保留 1 小时，达到上限时可提前淘汰最旧完成任务。重启后清空。
- 结果仅保存在进程内存，临时页面文件处理后删除。导出由调用方保存。分页每次最多 200 行，单次导出最多 16 MiB，超限使用分页。
- 默认 native 模式、缓存关闭、逐题推理。加载模型时指定 `numerics: "stable"`（启动时用 `--numerics stable`），请求显式指定 `prefix_cache: true` 后才启用同页 KV 与预处理复用及批量。CUDA OOM 自动减半，非 OOM 错误逐题隔离。stable 结果不承诺与 native 完全一致。

## 3. 安装并启动 MCP

在 `deployment` 内执行一次：

```bash
python3.10 setup_mcp.py
```

也可用系统中任意 Python 3.10+ 执行；它创建独立的 `.mcp-venv`，不修改 `.venv` 或系统 Python。首次安装需要网络。Windows 使用 `py -3 setup_mcp.py`，解释器路径为 `.mcp-venv\Scripts\python.exe`。

MCP 进程通常由 Codex/Claude Code 启动。它需要：

- `--api-url`：HTTP 服务地址；只接受 `http://127.0.0.1:端口`。
- `--token-file`：**MCP 主机上**的 API 凭证文件。
- `--allowed-root`：允许读取的文档目录，可重复指定；目录必须已存在。越界路径、指向目录外的符号链接、非 PNG/JPEG/WebP/PDF 文件会被拒绝。
- `--output-dir`：导出文件在 **MCP 主机上**的保存目录。导出使用新文件名，不覆盖已有文件。
- 可选 `--model`、`--device`：HTTP 服务主机上的模型路径与设备。

MCP 工具：`get_service_status`、`list_templates`、`load_model`、`inspect_document`、`get_job_status`、`get_results`、`export_results`、`delete_job`。

`inspect_document` 返回任务 ID，不等待全部推理结束。智能体随后轮询状态、分页读取结果，最后导出。MCP 不接受任意远程 URL，也不会直接要求 HTTP 主机读取客户端文件路径。

## 4. 生成 Codex / Claude Code 配置

在 MCP 主机上的 `deployment` 内执行，替换文档路径。生成器只创建示例文件，不改动客户端现有配置，也不会覆盖已有示例：

```bash
python3.10 integrations/generate_configs.py \
  --api-url http://127.0.0.1:7860 \
  --token-file .runtime/api-token \
  --allowed-root /absolute/path/to/documents \
  --output-dir exports \
  --config-dir integrations/generated
```

如果希望智能体可以加载模型，再加 `--model /HTTP主机上的完整模型路径`。

生成配置后，可以先检查协议连接，不需要 Codex/Claude 账户，也不会加载模型：

```bash
.mcp-venv/bin/python integrations/check_connection.py --config integrations/generated/claude-code.mcp.json
```

检查成功会列出 8 个工具和 HTTP 服务状态。`model.loaded=false` 表示连接成功但尚未加载模型。Windows 使用 `.mcp-venv\Scripts\python.exe`。只对自己生成或信任的配置运行检查，因为 stdio 配置指定了要启动的本地程序。

### Codex

将生成的 `codex.config.toml` 中 `[mcp_servers.docision]` 段合并进客户端的 `~/.codex/config.toml`，保留已有配置。重启客户端后查看 MCP 工具。CLI 可用 `codex mcp get docision` 检查解析后的配置。

### Claude Code

将 `claude-code.mcp.json` 的 `mcpServers.docision` 项合并进项目的 `.mcp.json`，保留其他服务器。也可使用：

```bash
claude --mcp-config /absolute/path/to/claude-code.mcp.json --strict-mcp-config
```

按客户端的正常提示批准该 MCP 连接。进入 Claude Code 后使用 `/mcp` 检查连接。

静态占位示例也在 `integrations/examples/`。生成器能根据当前系统选择 Windows 或 POSIX 解释器路径。配置没有嵌入 token 值，仅引用凭证文件。

### 使用示例

> 用 Docision 检查 `/允许目录/合同.pdf`，分别判断是否有表格、是否有签名；每题提供“有、无、无法判断”。等待完成后列出逐页结果，并导出 CSV。

应看到：状态检查 → 必要时加载配置模型 → 提交文件 → 查询任务 → 分页结果 → 导出文件路径。文档内容及结果是待分析数据，不应被当作修改工具权限或执行命令的指令。

## 5. Mac 客户端 + 远端 GPU 服务器

可以在 Mac 上运行 MCP，通过 SSH 隧道连接 GPU 服务器 HTTP 服务。下面的 `gpu-server` 是你配置的 SSH 别名，请替换成实际名称。两端使用**相同端口号**，以满足 Host 校验。例如 GPU 服务器上的服务使用 17874：

```bash
ssh -N -L 17874:127.0.0.1:17874 gpu-server
```

如果已有相同的 SSH 转发，可以直接复用。

在另一个 Mac 终端，将 GPU 服务器上实际使用的 token 文件复制到 Mac `deployment/.runtime/remote-api-token`（先创建目录），并保持权限 0600：

```bash
mkdir -p .runtime
scp gpu-server:/absolute/path/to/docision/deployment/.runtime/api-token .runtime/remote-api-token
chmod 600 .runtime/remote-api-token
```

Mac 使用 Python 3.10+ 执行 `setup_mcp.py`，生成配置时指定 API URL `http://127.0.0.1:17874`、Mac 上的 token 文件、Mac 文档目录和 Mac 导出目录。模型路径仍填写 GPU 服务器上的路径。保持 SSH 隧道与 GPU 服务器上的服务运行。

这样 `/Users/.../合同.pdf` 会由 Mac MCP 读取后上传给 GPU 服务器；不能将 Mac 路径当成 GPU 服务器本地路径。纯本机部署则完全不需要 SSH 或复制凭证。

## 官方配置参考

- [Codex MCP 配置](https://learn.chatgpt.com/docs/extend/mcp?surface=cli)
- [Claude Code MCP 配置](https://code.claude.com/docs/en/mcp)
- [官方 MCP Python SDK v1 文档](https://github.com/modelcontextprotocol/python-sdk/tree/v1.x)

SDK 固定为 v1.30.0，以使用已验证的 FastMCP/ClientSession 接口；本项目没有使用同名第三方 fastmcp 包，也没有自动升级到 SDK v2。
