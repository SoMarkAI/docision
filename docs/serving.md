# HF image scoring service

[Document guide](../README.md) &nbsp;&nbsp; [中文](serving.zh-CN.md)

Serve independent image requests with **Transformers and PyTorch**. No vLLM,
custom C++/CUDA build, NCCL process group, or `generate()` is required.
Qwen3-VL-based Docision checkpoints use this image-only service.

Each configured device has one model replica in its own **spawn** process.
Workers collect requests for a short interval and perform a real cross-image
batch forward. Image preprocessing, positions, questions and candidate counts
remain independent. Results are returned in input order, even if workers finish
out of order. Different images never share visual prefix KV.

## Start

Install the document extra as described in [installation](../README.md#start), and
download a complete local checkpoint. Single NVIDIA GPU:

```bash
python -m docision.service --model models/Docision --devices cuda:0 --max-batch-size 4 --numerics stable --port 8000
```

Two independent GPU replicas:

```bash
python -m docision.service --model models/Docision --devices cuda:0,cuda:1 --max-batch-size 4 --numerics stable --batch-wait-ms 10 --max-pending 128
```

The installed `docision-serve` command starts this image service; `Docision-serve` is a compatibility alias. It is separate from the token-authenticated `docision serve` Playground API.
The default listener is `127.0.0.1:8000`; specify `--host 0.0.0.0` to listen on
all interfaces. `GET /health` returns 503 while models load and 200 once every
worker is ready. This lightweight server has no built-in authentication.

No tensor parallelism is used: **each checkpoint must fit on each selected
device**. Start with batch 1 on limited VRAM, and benchmark batch 2/4/8. A larger
batch improves throughput only if the workload and device benefit from it;
waiting to form a batch can increase individual-request latency.

## Submit existing examples

In another terminal:

```bash
python examples/service_client.py --inputs examples/single.json examples/multiple.json --workers 2 --output outputs/service.json
```

Use several JSON files pointing to different images for cross-image concurrency.
The client sends PNG data URLs and strips reference labels. It requires no SDK.
`--url http://SERVER:8000` targets another machine.

`POST /v1/score` accepts the existing document shape:

```json
{
  "image": "data:image/png;base64,...",
  "context": "Inspect the supplied media carefully.",
  "questions": [
    {"id": "q1", "question": "Which document category fits this page?", "candidates": {"A": "Invoice", "B": "Receipt", "C": "Quotation"}}
  ]
}
```

Use `{"documents": [document1, document2]}` for multiple images in one call.
`question.instructions`/`question.criteria` is accepted too. The response keeps
document/question IDs and returns every candidate's label, text and probability,
the decision, worker ID and actual service batch size. Uploaded image copies
live only until their inference completes, including after an HTTP timeout.

Local server paths are disabled by default. With `--media-root /path/to/media`,
`image` can be a path under that directory. The server snapshots it into the same
request-scoped temporary storage, so files changing later cannot change a queued
request. Neither uploaded images nor reference labels modify the checkpoint.

## Numerical profiles and limits

The default is one question per forward, causal attention, full LM Head readout, native BF16, no prefix
cache, 4,000 input tokens and 501,760 maximum image pixels. `--mode isolated`
enables the existing candidate-isolation mask. `--numerics stable` uses the
existing FP32 text/shape-stable attention profile when comparing batched and
serial scoring. Native low-precision results can vary with batch shape; neither
profile is silently substituted. Record the profile when reporting results.

The same-image prefix-cache API also remains available in `DocisionEngine`.
With `--prefix-cache`, this service also retains worker-local GPU KV between requests.

The queue has at most `--max-pending` outstanding questions. Overload or unready
workers returns 503; request expiry returns 504. Invalid model inputs return
422 with a structured error. Batch errors are recursively
isolated to individual jobs. OOM reduces the batch, never the input resolution
or candidate set. An individual inference error is returned explicitly.
Worker death fails outstanding requests instead of leaving them waiting forever.

## Windows native image inference

The service uses standard-library HTTP, queues and `multiprocessing.spawn`; it
does not depend on Unix signals, fork, Bash, NCCL or vLLM. PyTorch provides native
Windows CUDA wheels. The image-only extra avoids video/audio dependencies.

PowerShell, from the checkout (Python 3.10+ and a compatible NVIDIA driver):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e ".[Docision]"
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
.\.venv\Scripts\python.exe -m docision.service --model models/Docision --devices cuda:0 --max-batch-size 4 --numerics stable
```

Choose a driver-compatible PyTorch 2.11 wheel from the [official installation
guide](https://pytorch.org/get-started/locally/) if CUDA 12.8 is unsuitable.
Native BF16 GPU inference needs BF16-capable hardware. CPU fallback is possible
with `--devices cpu --dtype float32`, but full 4B CPU latency is not measured.

Cross-image scoring is checked with actual tiny random-weight Qwen3-VL CPU
forwards; spawn/HTTP scheduling is checked with test workers. Windows CUDA,
full-weight online-service parity and production throughput require separate
hardware validation. Existing Linux benchmark scores are not a Windows-service
performance claim.

## Dynamic batches with prefix reuse

```bash
python -m docision.service --model models/Docision --devices cuda:0,cuda:1 \
  --max-batch-size 4 --numerics stable --prefix-cache \
  --cache-max-entries 8 --cache-max-tokens 8192
```

Cache keys use the image file content SHA-256 and shared context, not temporary
filenames or questions/candidates. Model, processor settings and numerical mode
must remain fixed during the worker lifetime. Independent KV rows are left-padded
and masked for a real mixed-prefix batch; native per-row positions are retained.
Reuse across HTTP requests requires reaching the same worker. There is no
cross-device cache sharing or cache-affinity scheduling.

Each worker evicts least recently used entries to respect both retained-entry and
prefix-token limits. Oversized entries are reused only within the current batch.
These budgets exclude temporary batch KV copies, model weights and activations;
they are not hard VRAM limits. OOM clears retained entries and splits the batch.
`prefix_cache_hit` indicates a prefix was already available for that question.
`num_cached_tokens` is the prefix length used for suffix scoring, including a
newly constructed prefix whose hit flag is false.

Caching and batch sizes above 1 require explicit `--numerics stable`. Use the same stable profile for comparisons; native BF16 remains sensitive to
batch/cache shapes. Real tiny random-weight Qwen3-VL tests cover FP32/BF16 weights
in stable mode, causal/isolated masks, unequal prefixes and candidate counts,
ordering, eviction and error splitting. Full 4B GPU throughput and accuracy have
not been validated for this combined service path.
