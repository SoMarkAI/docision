# 许可与第三方组件

deployment 源码依据仓库的 Apache License 2.0 分发；`LICENSE` 是上一级仓库许可证的未修改副本。这一声明涵盖应用源码，不涵盖用户文档、模型权重、训练数据、商标或独立分发的依赖。分发衍生作品时，应保留适用的版权、许可和署名声明。

## 安装的依赖

源码补充包不内置 Python wheel 或原生库；安装器会分别获取这些依赖。主要组件如下：

| 组件 | 上游许可证／审阅说明 |
|---|---|
| PyTorch 和 Torchvision | BSD 类许可证；其 wheel 内含其他第三方声明 |
| Transformers、Accelerate、Safetensors | Apache-2.0 |
| Pillow | MIT-CMU；二进制 wheel 可能包含其他库 |
| pypdfium2 | Apache-2.0 / BSD-3-Clause；内置 PDFium 及其依赖各有自己的声明 |
| 官方 MCP Python SDK | MIT |
| PyPA virtualenv 引导工具 | MIT；内置的种子包保留各自许可证 |

完整安装环境还包含间接依赖。例如，certifi 使用 MPL-2.0，tqdm 的元数据声明为 MPL-2.0 AND MIT，regex 声明为 Apache-2.0 AND CNRI-Python。CUDA 安装包含适用专有许可证的 NVIDIA 组件。不要把整个打包运行时标为 Apache-2.0。

安装依赖的源码发布，与重新分发这些二进制文件是不同的情形。在分发内置 wheel 的安装器、Docker 镜像或完整环境之前，应分别审阅每个实际分发版本的 LICENSE/NOTICE 文件，以及 NVIDIA 的再分发条款。

使用以下命令，为实际安装环境生成元数据清单：

```bash
.venv/bin/python release/dependency_inventory.py > model-licenses.json
.mcp-venv/bin/python release/dependency_inventory.py > mcp-licenses.json
```

元数据可能缺失或经过简写。这些清单帮助定位许可证文件，不能替代对文件本身的阅读。审查输出应保存在 `validation/` 中，不应放入源码包。

## 模型与数据

模型下载: [SoMarkAI/Docision-1.0](https://huggingface.co/SoMarkAI/Docision-1.0)

`release/smoke.py` 生成的验收素材是脚本绘制的合成图片和 PDF，不包含真实合同、签名或人物。上一级仓库中的视觉素材和现有示例文档需要单独审阅来源；它们不包含在 deployment 补充包中。
