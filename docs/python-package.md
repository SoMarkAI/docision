# Python package, API Server and local Playground

Python 3.10+ is required. Supply a complete local Qwen3-VL model; weights are not
included or automatically downloaded. Full-model inference requires appropriate
memory and hardware; NVIDIA BF16 is recommended. macOS uses CPU, not MPS.

## Install

From the complete source repository, install into a new virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[server,mcp]'
```

If the system Python lacks `venv`/`ensurepip`, create the environment with the
PyPA bootstrap tool instead (this example uses Python 3.10):

```bash
mkdir -p .cache/bootstrap
curl -fL https://bootstrap.pypa.io/virtualenv/3.10/virtualenv.pyz -o .cache/bootstrap/virtualenv.pyz
python3.10 .cache/bootstrap/virtualenv.pyz --no-periodic-update .venv
source .venv/bin/activate
python -m pip install '.[server,mcp]'
```

On a host with restricted network access, a compatible wheelhouse can be supplied
using `PIP_NO_INDEX=1` and `PIP_FIND_LINKS=/path/to/wheels`. Include all transitive
dependencies and check their hashes. This is still a new environment installation,
but should be reported as an assisted installation, not a fresh online download.

Or install the supplied wheel without keeping the source checkout:

```bash
python -m pip install '/absolute/path/to/docision-0.2.1rc2-py3-none-any.whl[server,mcp]'
```

The `server` extra names the supported server installation; the HTTP server uses
the standard library and needs no additional framework. `mcp` adds the official
Python SDK. Both reuse the same installed Docision package. No PyPI publication
is implied by these commands; use the reviewed source or supplied wheel.

## Python

The import is capitalized `Docision`, preserving the existing package name.

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

`inspect()` returns one row per page/question, in page then question order.
Each row has either `result` or `error`; inspect both before treating a document
as successfully processed. JSON/CSV export refuses to overwrite an existing file.
A model instance can inspect many files; `close()` or the context manager releases
its model references. Calls in one process are serialized for PDF rendering safety.
Model instances do not share their model state. The existing low-level
`from docision import DocisionEngine` interface remains available.

PNG/JPEG/WebP/PDF, at most 25 MiB per file, 100 PDF pages, and 1–20 questions
with 2–26 distinct candidates are supported. Same-page KV and preprocessing reuse,
whole-page batching and CUDA OOM halving are shared with HTTP/MCP/Playground.
Candidate probabilities are relative scores, not calibrated confidence.

## One-command HTTP API

```bash
docision serve --model /absolute/path/to/model --device cuda:0 --port 7860
```

This loads the model and serves `/api/v1/` on `127.0.0.1`. It does not open a browser
or expose a public listener. The same service serves the local Playground at `/`.
The startup URL includes a local token; API calls use `Authorization: Bearer TOKEN`.
The token defaults to `~/.cache/docision/.runtime/api-token`; use `--token-file`
to choose another path. Runtime data defaults to `~/.cache/docision`, configurable
with `--state-dir` or `DOCISION_STATE_DIR`. Package installation directories are
read-only assets, not runtime storage. No files are written beside installed code.

The API exposes health/model/templates, synchronous score, asynchronous jobs,
status, paginated results, JSON/CSV export, and task deletion. The authenticated
`/api/v1/openapi.json` contains the full request/response contract. Completed jobs
remain in memory for up to one hour, capped at 32; restart clears them.

## Local Playground

```bash
docision playground --model /absolute/path/to/model --device cuda:0 --port 7860
```

This uses the same service and opens the browser. On a headless GPU server add
`--no-browser` and use an SSH tunnel with matching local/remote ports. Keep the
service and tunnel running. There is no online hosted Playground.

## MCP

With the `mcp` extra installed, configure the client to launch:

```bash
docision mcp --api-url http://127.0.0.1:7860 \
  --token-file /absolute/path/to/api-token \
  --allowed-root /absolute/path/to/documents \
  --output-dir /absolute/path/to/exports
```

This stdio bridge calls an already running HTTP server. File paths and exports
belong to the MCP host; the model belongs to the HTTP host. Mac clients can connect
to a GPU host through SSH and need a copy of that service's token. The server does
not read arbitrary Mac paths itself. `docision mcp --help` lists options.

The eight tools are get_service_status, list_templates, load_model,
inspect_document, get_job_status, get_results, export_results and delete_job.
load_model requires the optional operator-supplied `--model` on the bridge.

## Build and acceptance

```bash
python -m pip wheel --no-deps . -w dist
```

Install the resulting wheel in a new environment. Run from a directory outside
the repository, with PYTHONPATH unset, to prove imports and assets come from the
installed package. Verify Python image/PDF calls and export, then `docision serve`,
HTTP/MCP requests and the Playground HTML. `docision serve --help` and
`docision playground --help` do not load weights.

The command line uses lowercase `docision`. Low-level commands are
`docision-score`, `docision-eval` and `docision-serve`.
`Docision-serve` remains a compatibility alias for the separate image service. The source `deployment/start.sh`
launcher remains available and routes to the same implementation.

## Inference modes

The default is native numerics, no prefix cache and one question per forward. To opt in, construct `Docision(model_path, numerics="stable")` and call `inspect(..., prefix_cache=True)`. For HTTP/Playground, start with `--numerics stable` and explicitly enable caching. MCP uploads default to `prefix_cache=False`; set it to true only for a stable-mode HTTP model. Native cache requests are rejected. Stable results may differ from native. Rows record `numerics`, `cache_status`, and actual batch information when applicable.
