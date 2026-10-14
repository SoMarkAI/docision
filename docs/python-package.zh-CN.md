# Python 软件包、API Server 与本地 Playground

需要 Python 3.10+。请提供完整的本地 Qwen3-VL 模型；软件包不包含权重，也不会自动下载权重。完整模型推理需要合适的内存和硬件，建议使用支持 BF16 的 NVIDIA GPU。macOS 使用 CPU，不使用 MPS。

## 安装

在完整源码仓库中，创建新的虚拟环境并安装：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[server,mcp]'
```

如果系统 Python 缺少 `venv`/`ensurepip`，可以改用 PyPA 引导工具创建环境，以下示例使用 Python 3.10：

```bash
mkdir -p .cache/bootstrap
curl -fL https://bootstrap.pypa.io/virtualenv/3.10/virtualenv.pyz -o .cache/bootstrap/virtualenv.pyz
python3.10 .cache/bootstrap/virtualenv.pyz --no-periodic-update .venv
source .venv/bin/activate
python -m pip install '.[server,mcp]'
```

在网络访问受限的主机上，可通过 `PIP_NO_INDEX=1` 和 `PIP_FIND_LINKS=/path/to/wheels` 提供兼容的 wheel 包目录。目录中应包含全部间接依赖，并检查其哈希。这仍然是在新环境中安装，但应记为辅助安装，而不是全新联网下载。

也可以安装提供的 wheel，无需保留源码目录：

```bash
python -m pip install '/absolute/path/to/docision-0.2.1rc2-py3-none-any.whl[server,mcp]'
```

`server` 可选依赖组表示受支持的服务端安装方式；HTTP 服务使用标准库，无需额外框架。`mcp` 会安装官方 Python SDK。两者复用同一个已安装的 Docision 软件包。这些命令不代表软件包已发布到 PyPI；请使用经过审阅的源码或提供的 wheel。

## Python

导入名称为首字母大写的 `Docision`，沿用现有包名。

```python
from docision import Docision

questions = [
    {"question": "Does this page contain a table?", "candidates": ["Yes", "No", "Cannot determine"]},
    {"question": "Does this page have a signature?", "candidates": ["Yes", "No", "Cannot determine"]},
]
with Docision("/absolute/path/to/model", device="cuda:0") as model:
    rows = model.inspect("contract.pdf", questions=questions)
    for row in rows:
        print(row["page"], row["question_id"], row.get("result", row.get("error")))
    Docision.export(rows, "results.json")
    Docision.export(rows, "results.csv")
```

`inspect()` 为每个“页面／问题”返回一行，先按页码、再按问题顺序排列。每行包含 `result` 或 `error`；判断文档是否处理成功前，应检查这两种情况。JSON/CSV 导出拒绝覆盖已有文件。

一个模型实例可以检查多个文件；调用 `close()` 或使用上下文管理器可以释放模型引用。为保证 PDF 渲染安全，同一进程中的调用会串行执行。不同模型实例不共享模型状态。已有的底层 `from docision import DocisionEngine` 接口仍然可用。

支持 PNG/JPEG/WebP/PDF，单文件最多 25 MiB，PDF 最多 100 页，每次 1–20 个问题，每题 2–26 个互不重复的候选答案。同页 KV 与预处理复用、整页批量推理，以及 CUDA 显存不足时批量减半的逻辑，与 HTTP/MCP/Playground 共用。候选概率是相对评分，不是经过校准的置信度。

## 一条命令启动 HTTP API

```bash
docision serve --model /absolute/path/to/model --device cuda:0 --port 7860
```

这条命令会加载模型，并在 `127.0.0.1` 提供 `/api/v1/` 服务。它不会打开浏览器，也不会开放公网监听。同一服务还通过 `/` 提供本地 Playground。

启动链接包含本地 token；API 调用使用 `Authorization: Bearer TOKEN`。token 文件默认为 `~/.cache/docision/.runtime/api-token`，可以通过 `--token-file` 指定其他路径。运行数据默认保存在 `~/.cache/docision`，可通过 `--state-dir` 或 `DOCISION_STATE_DIR` 配置。软件包安装目录仅提供只读资源，不用于存储运行数据；不会在已安装的代码旁写入文件。

API 提供服务健康状态、模型、模板、同步推理、异步任务、任务状态、分页结果、JSON/CSV 导出和任务删除。需要认证的 `/api/v1/openapi.json` 包含完整的请求与响应契约。已完成任务在内存中最多保留一小时，任务数量上限为 32；重启后清空。

## 本地 Playground

```bash
docision playground --model /absolute/path/to/model --device cuda:0 --port 7860
```

该命令使用同一服务并打开浏览器。在没有图形界面的 GPU 服务器上，添加 `--no-browser`，并使用本地与远端端口一致的 SSH 隧道。保持服务和隧道运行。项目不提供在线托管的 Playground。

## MCP

安装 `mcp` 可选依赖后，配置客户端启动：

```bash
docision mcp --api-url http://127.0.0.1:7860 \
  --token-file /absolute/path/to/api-token \
  --allowed-root /absolute/path/to/documents \
  --output-dir /absolute/path/to/exports
```

这个 stdio 桥接进程调用已经运行的 HTTP 服务。文件路径与导出目录属于 MCP 主机，模型属于 HTTP 主机。Mac 客户端可以通过 SSH 连接 GPU 主机，并需要一份该服务 token 的副本。服务器本身不会读取任意 Mac 路径。`docision mcp --help` 可列出选项。

八个工具分别为 get_service_status、list_templates、load_model、inspect_document、get_job_status、get_results、export_results 和 delete_job。使用 load_model 时，需要由操作者在桥接进程的启动参数中提供可选的 `--model`。

## 构建与验收

```bash
python -m pip wheel --no-deps . -w dist
```

将生成的 wheel 安装到新环境中。取消设置 PYTHONPATH，并从仓库外的目录运行，以确认导入的模块和资源来自已安装的软件包。验证 Python 图片/PDF 调用与导出，再验证 `docision serve`、HTTP/MCP 请求和 Playground HTML。`docision serve --help` 与 `docision playground --help` 不会加载权重。

命令行使用小写 `docision`。底层命令为 `docision-score`、`docision-eval` 和 `docision-serve`；`Docision-serve` 保留为独立图片服务的兼容别名。源码中的 `deployment/start.sh` 启动器仍然可用，并调用同一套实现。

## 推理模式

默认 native 模式，关闭缓存，每次前向处理一道题。需要加速时，使用 `Docision(model_path, numerics="stable")` 并调用 `inspect(..., prefix_cache=True)`。HTTP/Playground 使用 `--numerics stable` 启动，再显式启用缓存。MCP 上传默认 `prefix_cache=False`，仅在 HTTP 模型为 stable 时可设为 true；native 缓存请求会被拒绝。stable 的概率或判断可能与 native 不同。结果行记录 `numerics`、`cache_status`，以及适用时的实际批量信息。
