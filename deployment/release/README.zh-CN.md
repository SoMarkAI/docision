# 构建与验证

[English](README.md) · [简体中文](README.zh-CN.md)

以下命令在 `deployment/` 中执行。源码补充包需配合完整仓库使用；wheel/sdist 构建见[安装包说明](../../docs/python-package.zh-CN.md)。

## 构建

```bash
python3 release/audit_source.py --repo .. --output validation/source-audit.json
python3 release/build_overlay.py --output release-output/deployment-source.tar.gz
tar -tzf release-output/deployment-source.tar.gz
```

发布前检查扫描结果。`manifest.json` 指定打包文件，排除运行数据、凭证、环境和权重。构建时生成 SHA-256 清单，不覆盖已有压缩包。

## 全新环境安装

使用不含环境和缓存的仓库副本，准备 Python 3.10+ 和兼容的 NVIDIA 驱动。

```bash
PIP_NO_CACHE_DIR=1 PYTHONNOUSERSITE=1 ./start.sh --no-browser --help
PIP_NO_CACHE_DIR=1 PYTHONNOUSERSITE=1 python3.10 setup_mcp.py
.venv/bin/python -m pip check
.mcp-venv/bin/python -m pip check
.venv/bin/python -B -m unittest -v test_app.py test_service_api.py test_release.py
```

第一条命令安装依赖并显示帮助，不加载模型。`python3.10` 可替换为其他 Python 3.10+ 解释器。离线安装时，设置 `PIP_NO_INDEX=1`，并将 `PIP_FIND_LINKS` 指向包含全部依赖 wheel 的目录。

## 完整模型验证

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 .mcp-venv/bin/python release/smoke.py \
  --model /absolute/path/to/full-model \
  --port 17886 \
  --output validation/release-smoke
```

使用空闲端口和不存在的输出目录。测试会启动服务，通过 HTTP 和 MCP 处理合成图片/PDF，对比结果，检查分页、CSV/JSON 导出、损坏输入和路径限制，最后关闭服务。

成功显示 `FULL_MODEL_RELEASE_SMOKE_PASS`，表示功能检查通过，不是准确率评测。输出包含本地路径和凭证，请勿公开。

上述完整模型检查面向 Linux/NVIDIA；Mac 连接 GPU 服务器见 [MCP 配置](../integrations/API_MCP.zh-CN.md)。
