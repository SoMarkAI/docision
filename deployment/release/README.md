# Build and validate

[English](README.md) · [简体中文](README.zh-CN.md)

Run these commands inside `deployment/`. The source overlay requires the complete repository. For wheel/sdist builds, see [package instructions](../../docs/python-package.md).

## Build

```bash
python3 release/audit_source.py --repo .. --output validation/source-audit.json
python3 release/build_overlay.py --output release-output/deployment-source.tar.gz
tar -tzf release-output/deployment-source.tar.gz
```

Review audit findings before publishing. `manifest.json` selects shipped files and excludes runtime data, credentials, environments, and weights. The builder creates a SHA-256 manifest and refuses to overwrite an existing archive.

## Clean installation

Use a fresh repository copy without environments or caches, Python 3.10+, and a compatible NVIDIA driver.

```bash
PIP_NO_CACHE_DIR=1 PYTHONNOUSERSITE=1 ./start.sh --no-browser --help
PIP_NO_CACHE_DIR=1 PYTHONNOUSERSITE=1 python3.10 setup_mcp.py
.venv/bin/python -m pip check
.mcp-venv/bin/python -m pip check
.venv/bin/python -B -m unittest -v test_app.py test_service_api.py test_release.py
```

The first command installs dependencies and displays help without loading a model. Replace `python3.10` with another Python 3.10+ interpreter if needed. For offline installation, set `PIP_NO_INDEX=1` and `PIP_FIND_LINKS` to a directory containing all required wheels.

## Full-model validation

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 .mcp-venv/bin/python release/smoke.py \
  --model /absolute/path/to/full-model \
  --port 17886 \
  --output validation/release-smoke
```

Use an unused port and a new output directory. The test starts a service, processes synthetic images/PDFs through HTTP and MCP, compares results, checks pagination and CSV/JSON exports, tests invalid inputs and path restrictions, then stops the service.

Success prints `FULL_MODEL_RELEASE_SMOKE_PASS`. This is a functional check, not an accuracy evaluation. Outputs contain local paths and credentials; keep them private.

Full-model checks above target Linux/NVIDIA. For a Mac connected to a GPU server, see [MCP setup](../integrations/API_MCP.md).
