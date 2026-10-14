# 本地部署

[English](README.md) · [简体中文](README.zh-CN.md)

需要完整 Docision 仓库、Python 3.10+ 和已下载的模型。推荐支持 BF16 的 NVIDIA GPU；macOS 使用 CPU。

## 启动

在 `deployment/` 中操作：

- Linux：`./start.sh`
- macOS：`bash start.sh`
- Windows：双击 `start.bat`

首次启动会在 `.venv/` 安装依赖并打开浏览器。填写模型目录、选择设备并加载模型。模型权重需单独下载。

```bash
./start.sh --model /absolute/path/to/model --device cuda:0
```

可选参数：`--no-browser`、`--port 7861`、`--torch-backend cpu|cu128|cu130`；修复依赖使用 `--repair`。打开终端打印的完整链接，按 Ctrl+C 退出。

## 使用

1. 添加 1–20 个问题，每题设置 2–26 个不重复候选答案。
2. 上传 PNG/JPEG/WebP 图片或 PDF：每个文件最多 25 MiB，PDF 最多 100 页。
3. 查看逐页结果，按需人工修正，导出 CSV 或 JSON。

默认 native 模式逐题推理、不启用缓存。需要同页缓存与批量时，选择 stable 模式加载模型，再勾选缓存。stable 结果可能与 native 不同。开始新任务前请先导出已有结果。

## MCP 与运行文件

Codex、Claude Code 及远端 GPU 接入见 [MCP 配置](integrations/API_MCP.zh-CN.md)。

- `.venv/`：模型环境；`.mcp-venv/`：MCP 环境。
- `.runtime/api-token`：本地凭证，请勿公开。
- `.cache/`：下载缓存和临时文件。
- 服务监听 `127.0.0.1`。API 结果完成后最多保留一小时，重启清空。

## 检查

```bash
.venv/bin/python -B -m unittest -v test_app.py test_service_api.py test_release.py
```

[发布检查](release/README.zh-CN.md) · [安装包用法](../docs/python-package.zh-CN.md) · [许可证](LICENSE) · [第三方说明](THIRD_PARTY_NOTICES.zh-CN.md)
