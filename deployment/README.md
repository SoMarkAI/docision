# Local deployment

[English](README.md) · [简体中文](README.zh-CN.md)

Requires the complete Docision repository, Python 3.10+, and a downloaded model. An NVIDIA GPU with BF16 support is recommended; macOS uses CPU.

## Start

Run inside `deployment/`:

- Linux: `./start.sh`
- macOS: `bash start.sh`
- Windows: double-click `start.bat`

The first launch installs dependencies in `.venv/` and opens the browser. Enter the model directory, select a device, and load the model. Model weights are downloaded separately.

```bash
./start.sh --model /absolute/path/to/model --device cuda:0
```

Options: `--no-browser`, `--port 7861`, `--torch-backend cpu|cu128|cu130`, and `--repair` to repair dependencies. Use the complete URL printed in the terminal. Press Ctrl+C to stop.

## Use

1. Add 1–20 questions, each with 2–26 distinct candidates.
2. Upload PNG/JPEG/WebP images or PDFs: up to 25 MiB per file and 100 pages per PDF.
3. Review page-level answers, make corrections if needed, and export CSV or JSON.

Native mode processes questions individually without caching. For same-page caching and batching, load in stable mode and enable the cache option. Stable results may differ from native results. Export before starting a new run.

## MCP and runtime files

See [MCP setup](integrations/API_MCP.md) for Codex, Claude Code, and remote GPU connections.

- `.venv/`: model environment; `.mcp-venv/`: MCP environment.
- `.runtime/api-token`: local credential; keep private.
- `.cache/`: downloads and temporary files.
- The service listens on `127.0.0.1`. API results expire within one hour after completion and are cleared on restart.

## Checks

```bash
.venv/bin/python -B -m unittest -v test_app.py test_service_api.py test_release.py
```

[Release checks](release/README.md) · [Package usage](../docs/python-package.md) · [License](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md)
