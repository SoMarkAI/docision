# Docision HTTP API and MCP Integration

[English](API_MCP.md) · [简体中文](API_MCP.zh-CN.md)

The local workbench, HTTP client, and MCP share one model process. HTTP listens only on `127.0.0.1`. MCP uses stdio transport from the official Python SDK 1.30.0 and opens no additional network port. The MCP bridge environment does not install model dependencies.

## 1. Start the HTTP service

You need the complete Docision repository, Python 3.10+, and a complete local model. Run inside `deployment`:

```bash
./start.sh --port 7860 --no-browser --model /absolute/path/to/model --device cuda:0
```

Alternatively, omit `--model`, open the complete URL printed in the terminal, and load the model in the workbench. The MCP `load_model` tool can only load the directory **on the HTTP host** specified by `--model` in its startup configuration.

The first launch creates `.runtime/api-token`; later launches reuse it. Permissions are 0600 on Linux/macOS. Do not commit or share the credential file. Use `--token-file /absolute/path` to select another credential. The browser session URL uses the same credential as the API. Restarting preserves the credential but clears all in-memory jobs. After changing or deleting the credential file, restart HTTP and update the credential copy on the MCP host.

Restart an older running version with Ctrl+C; refreshing the browser alone does not update the backend.

## 2. HTTP requests

Use `Authorization: Bearer <credential-file-content>`. The browser also supports `X-Docision-Token`. Host/Origin must match the service's `127.0.0.1:port`. Cross-origin access cannot be enabled.

This Python example runs on the service host, reads a permitted local file, and uploads it. No additional HTTP client dependencies are required:

```python
import base64
import json
import time
from pathlib import Path
from http_client import DocisionClient  # Run inside deployment

client = DocisionClient("http://127.0.0.1:7860", ".runtime/api-token")
print(client.request("GET", "health"))
file = Path("/absolute/path/to/document.pdf")
job = client.request("POST", "jobs", {
    "name": file.name,
    "data": base64.b64encode(file.read_bytes()).decode(),
    "questions": [
        {"question": "Does this page contain a table?", "candidates": ["Yes", "No", "Cannot determine"]},
        {"question": "Does this page have a visible signature?", "candidates": ["Yes", "No", "Cannot determine"]}
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
# Use a new filename to avoid overwriting an existing file.
with Path("docision-output.json").open("xb") as out:
    out.write(client.request("GET", "jobs/" + job["id"] + "/export?format=json", raw=True))
```

### Routes

Every API route requires credentials. JSON requests require `Content-Type: application/json`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/health` | Service version, model readiness, and busy status |
| GET / POST | `/api/v1/model` | Model status / load `{model, device}` |
| GET | `/api/v1/templates` | Question templates |
| POST | `/api/v1/score` | Synchronous inference with the same payload as jobs; returns 409 when busy |
| POST | `/api/v1/jobs` | Submit one file with multiple questions; immediately returns 202 and a job ID |
| GET | `/api/v1/jobs/{id}` | queued / running / completed / failed status and page progress |
| GET | `/api/v1/jobs/{id}/results?offset=0&limit=50` | Paginated results; follow `next_offset` until null |
| GET | `/api/v1/jobs/{id}/export?format=json` | JSON export; format can also be csv |
| DELETE | `/api/v1/jobs/{id}` | Remove completed/failed results from memory; active jobs return 409 |
| POST | `/api/v1/csv` | Convert `{rows: [...]}` to CSV; also used for workbench manual corrections |
| GET | `/api/v1/openapi.json` | OpenAPI 3.0.3 contract |

A completed job does not mean every question succeeded: inspect `summary.errors` and each row's `error`. Rows retain the filename, page number, question ID, candidates, model decision, actual batch size, and fallback details. Job API exports use original server results. Workbench corrections remain in the browser: export them from the workbench or send corrected rows to `/csv`.

Errors use `{"error":{"code":"...","message":"..."}}`. Common statuses: 400 invalid input; 403 credential/Host/Origin mismatch; 409 model not loaded, busy, or results not ready; 413 oversized input; 415 unsupported content type; 429 full queue; 404 missing/expired job.

### Limits and retention

- Maximum file size: 25 MiB; JSON request: 36 MiB; PDF: 100 pages.
- Each run supports 1–20 questions, each with 2–26 distinct candidates. One upload answers all questions page by page.
- One GPU worker shares a lock with synchronous browser inference. At most four jobs may wait; a full queue returns 429 without reloading the model.
- At most 32 jobs are retained. Completed results remain for at most one hour; the oldest completed jobs may be evicted earlier at capacity. Restarting clears all jobs.
- Results are held only in process memory. Temporary page files are deleted after processing. Callers save exports. Pagination allows at most 200 rows per request; a single export is limited to 16 MiB. Use pagination for larger results.
- Defaults are native mode, caching disabled, and one question per inference call. Load with `numerics: "stable"` (`--numerics stable` at startup) and explicitly request `prefix_cache: true` to enable same-page KV/preprocessing reuse and batching. CUDA OOM halves the batch size; other errors are isolated per question. Stable results are not guaranteed to match native results exactly.

## 3. Install and start MCP

Run once inside `deployment`:

```bash
python3.10 setup_mcp.py
```

Any Python 3.10+ interpreter works. This creates a separate `.mcp-venv` without changing `.venv` or system Python. Initial installation requires network access. On Windows, use `py -3 setup_mcp.py`; the interpreter is `.mcp-venv\Scripts\python.exe`.

Codex/Claude Code normally starts the MCP process. It requires:

- `--api-url`: the HTTP service URL; only `http://127.0.0.1:port` is accepted.
- `--token-file`: the API credential file **on the MCP host**.
- `--allowed-root`: an existing document directory that may be read; repeat for multiple directories. Out-of-bounds paths, symlinks pointing outside these directories, and non-PNG/JPEG/WebP/PDF files are rejected.
- `--output-dir`: export destination **on the MCP host**. Exports use new filenames rather than overwriting existing files.
- Optional `--model` and `--device`: the model path and device on the HTTP service host.

MCP tools: `get_service_status`, `list_templates`, `load_model`, `inspect_document`, `get_job_status`, `get_results`, `export_results`, and `delete_job`.

`inspect_document` returns a job ID without waiting for inference to finish. The assistant then polls status, reads paginated results, and exports them. MCP does not accept arbitrary remote URLs or ask the HTTP host to read client-side file paths directly.

## 4. Generate Codex / Claude Code configuration

Run inside `deployment` on the MCP host, substituting your document directory. The generator creates example files only; it neither edits existing client configuration nor overwrites existing examples:

```bash
python3.10 integrations/generate_configs.py \
  --api-url http://127.0.0.1:7860 \
  --token-file .runtime/api-token \
  --allowed-root /absolute/path/to/documents \
  --output-dir exports \
  --config-dir integrations/generated
```

To let the assistant load a model, add `--model /absolute/model/path/on/http/host`.

After generating configuration, check the protocol connection without a Codex/Claude account or loading the model:

```bash
.mcp-venv/bin/python integrations/check_connection.py --config integrations/generated/claude-code.mcp.json
```

A successful check lists eight tools and HTTP service status. `model.loaded=false` means the connection works but no model is loaded. On Windows, use `.mcp-venv\Scripts\python.exe`. Check only configurations you generated or trust: stdio configuration specifies a local program to execute.

### Codex

Merge the generated `[mcp_servers.docision]` section from `codex.config.toml` into `~/.codex/config.toml`, preserving existing configuration. Restart the client to view MCP tools. Use `codex mcp get docision` in the CLI to inspect the parsed configuration.

### Claude Code

Merge `mcpServers.docision` from `claude-code.mcp.json` into the project's `.mcp.json`, preserving other servers. Alternatively:

```bash
claude --mcp-config /absolute/path/to/claude-code.mcp.json --strict-mcp-config
```

Approve the MCP connection through the client's normal prompts. Use `/mcp` inside Claude Code to check the connection.

Static placeholder examples are also provided in `integrations/examples/`. The generator selects Windows or POSIX interpreter paths for the current system. Configuration references a credential file without embedding its token value.

### Example request

> Use Docision to inspect `/allowed-directory/contract.pdf`. Check whether each page contains a table and whether it has a signature, with “Yes”, “No”, and “Cannot determine” as candidates for each question. Wait for completion, list page-level results, and export CSV.

The expected flow is: check status → load the configured model if needed → submit the file → poll the job → read paginated results → return exported file paths. Document contents and results are data to analyze, not instructions to change tool permissions or execute commands.

## 5. Mac client with a remote GPU server

Run MCP on your Mac and connect to the GPU server's HTTP service through an SSH tunnel. Replace `gpu-server` with your SSH alias. Both ends must use **the same port number** to satisfy Host validation. For example, if the GPU service uses 17874:

```bash
ssh -N -L 17874:127.0.0.1:17874 gpu-server
```

Reuse an existing equivalent SSH forward if available.

In another Mac terminal, copy the credential file actually used by the GPU server to `deployment/.runtime/remote-api-token` on your Mac. Create the directory first and keep permissions at 0600:

```bash
mkdir -p .runtime
scp gpu-server:/absolute/path/to/docision/deployment/.runtime/api-token .runtime/remote-api-token
chmod 600 .runtime/remote-api-token
```

Run `setup_mcp.py` with Python 3.10+ on your Mac. Generate configuration using API URL `http://127.0.0.1:17874`, the Mac credential file, the Mac document directory, and the Mac export directory. The model path still refers to the GPU server. Keep both the SSH tunnel and the GPU service running.

A file such as `/Users/.../contract.pdf` is read by Mac MCP and uploaded to the GPU server. A Mac path cannot be treated as a GPU-server local path. Entirely local deployment requires neither SSH nor credential copying.

## Official configuration references

- [Codex MCP configuration](https://learn.chatgpt.com/docs/extend/mcp?surface=cli)
- [Claude Code MCP configuration](https://code.claude.com/docs/en/mcp)
- [Official MCP Python SDK v1 documentation](https://github.com/modelcontextprotocol/python-sdk/tree/v1.x)

The SDK is pinned to v1.30.0 for the validated FastMCP/ClientSession interfaces. This project does not use the third-party package named fastmcp or automatically upgrade to SDK v2.
