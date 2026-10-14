# HF 多图片推理服务

[文档模型](../README.zh-CN.md) &nbsp;&nbsp; [English](serving.md)

使用 Transformers＋PyTorch 接收多个独立图片请求，不需要 vLLM、自定义 C++／CUDA
编译或 NCCL。适用于基于 Qwen3-VL 的 Docision 图文模型。

每张指定 GPU 启动一个 `spawn` 模型进程。请求队列短暂收集输入后，组成真正的跨图片
batch；每张图片保留自己的问题、候选项和位置编码。结果按原始输入顺序回传。
不同图片的视觉 KV 相互隔离。加 `--prefix-cache` 可在每个 worker 内复用同图同上下文
的前缀，并把不同长度前缀对应的问题后缀组成同一个 batch。

## 启动与调用

按[安装说明](../README.zh-CN.md#start)安装 `.[Docision]`，准备完整的本地模型目录：

```bash
python -m docision.service --model models/Docision --devices cuda:0 --max-batch-size 4 --numerics stable --port 8000
```

两张卡各一份模型：

```bash
python -m docision.service --model models/Docision --devices cuda:0,cuda:1 --max-batch-size 4 --numerics stable --batch-wait-ms 10 --max-pending 128
```

安装后的 `docision-serve` 是图片服务入口，`Docision-serve` 为兼容别名；它不同于带 token 认证的 `docision serve` Playground API。默认监听 `127.0.0.1:8000`；
指定 `--host 0.0.0.0` 可监听所有网卡。本轻量服务没有内置认证。
`GET /health` 在模型加载时返回503，所有进程准备好后返回200。

另开终端调用现有示例：

```bash
python examples/service_client.py --inputs examples/single.json examples/multiple.json --workers 2 --output outputs/service.json
```

将输入 JSON 指向不同图片即可测试跨图片并发。客户端自动编码图片，移除参考答案，
不依赖额外 SDK。`--url http://服务器地址:8000` 可以访问远程服务。

`POST /v1/score` 沿用已有 `image + context + questions[]`，图片使用 PNG／JPEG／WebP
的 base64 data URL；也支持 `{"documents": [文档1, 文档2]}`。
问题可以采用 `question + candidates` 或 `question.instructions + question.criteria`。
输出保留 ID，返回候选标签、文本、概率和最终决策、worker ID 和实际 batch 大小。
临时图片在推理完成后清理，客户端超时也不会提前删掉模型仍在使用的文件。

本地路径需显式指定 `--media-root`，仅接受该目录内的图片，并在入队前保存独立快照。
每个模型必须单独放得进指定设备；这里不使用 tensor parallel。

## 参数与评分协议

默认 causal、full LM Head、native BF16、不使用 prefix cache，输入上限4,000 tokens、
图片像素上限501,760。可通过 `--mode isolated` 开启已有候选隔离 mask，
通过 `--numerics stable` 使用更重的 FP32／形状稳定计算。
native 低精度结果可能随 batch 形状出现数值差异；做准确率比较时记录实际参数。

`--max-pending` 限制待处理问题数；队列满返回503，请求超时返回504。
OOM 会拆小 batch，不会偷偷降分辨率、删候选或截断输入。无效输入被隔离到具体任务，
worker 崩溃会明确返回失败。显存较小时从 batch 1 开始，实测2／4／8的吞吐和延迟，
不能把请求并发数直接等同于提速倍数。

## 动态 batch ＋前缀缓存

```bash
python -m docision.service --model models/Docision --devices cuda:0,cuda:1 \
  --max-batch-size 4 --numerics stable --prefix-cache \
  --cache-max-entries 8 --cache-max-tokens 8192
```

缓存按图片文件内容的 SHA-256 和公共 context 匹配，而非临时文件名；不同问题与候选
不进入缓存键。模型、processor 配置和数值模式必须在 worker 生命周期内保持不变。
不同图片的 KV 左侧补齐后合批，补齐位置被 mask 屏蔽，每题保留原生位置编码。
同图缓存可跨 HTTP 请求复用，但仅限命中同一 worker；没有跨 GPU 共享或缓存亲和调度。

每个 worker 最多保留指定条目数和前缀 token 总数，超出时按 LRU 淘汰。
单项超过 token 预算时只在当前批次复用，不长期保留。预算不包含当前 batch 的 KV 副本、
模型和激活，因此不是显存硬上限。OOM 清空保留缓存并拆小 batch 重试。
`prefix_cache_hit` 表示该题准备时已有可用前缀，`num_cached_tokens` 报告后缀评分使用的
前缀长度；首次构建前缀也会报告长度，但 hit 为 false。

默认批量为 1，缓存关闭；缓存或批量大于 1 时必须显式指定 `--numerics stable`。推荐用 `stable` 核对与串行无缓存的结果；native BF16 的 batch/cache
数值差异仍存在。跨图片缓存已用真实随机小型 Qwen3-VL 在 FP32/BF16 权重、stable
模式下验证 causal/isolated、不同前缀长度、候选数量、顺序、缓存淘汰和异常拆批。
这是实现检查，不是完整 4B GPU 吞吐或精度认证。

## Windows 原生使用

HTTP、队列和 `spawn` 多进程不依赖 Linux。PyTorch 提供 Windows CUDA 安装包，
图片专用的 `.[Docision]` 不要求 vLLM、FlashAttention 编译、FFmpeg 或音视频组件。

PowerShell，项目根目录下执行（Python 3.10+＋兼容的 NVIDIA 驱动）：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e ".[Docision]"
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
.\.venv\Scripts\python.exe -m docision.service --model models/Docision --devices cuda:0 --max-batch-size 4 --numerics stable
```

CUDA 安装包需与驱动兼容，其他选择见 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)。
BF16 GPU 推理需要支持 BF16 的硬件；`--devices cpu --dtype float32` 可走 CPU，
但未测量完整4B的 CPU 延迟。
