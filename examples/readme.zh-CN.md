# 使用示例

[English](readme.md) · [简体中文](readme.zh-CN.md) · [项目首页](../README.zh-CN.md)

一张表格图片，四对中英文问题。

| 文件 | 内容 |
|---|---|
| `validation-table.jpg` | 原始图片 |
| `single.json` | 一道问题，四个候选答案 |
| `multiple.json` | 八道问题，即四对中英文问题 |
| `recorded-output.json` | 历史 native BF16 结果，可选字段可能与当前版本不同 |

在仓库根目录运行：

```bash
python demo_docision.py --model models/docision-1.0 --input examples/single.json
python demo_docision.py --model models/docision-1.0 --input examples/multiple.json
```

同页缓存与批量推理添加 `--numerics stable --prefix-cache --question-batch-size 2`。

模型：[SoMarkAI/Docision-1.0](https://huggingface.co/SoMarkAI/Docision-1.0)

## 图片来源

表 3 来自 Digant Gupta、Carolyn A Lammersfeld、Pankaj G Vashi、Sadie L Dahlk 和 Christopher G Lis 的论文 *Can subjective global assessment of nutritional status predict survival in ovarian cancer?*，发表于 **Journal of Ovarian Research 1, 5 (2008)**。© 2008 Gupta 等；被许可方为 BioMed Central Ltd.

[原始文章](https://link.springer.com/article/10.1186/1757-2215-1-5) · [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/)

图片为 Infinity 提供的表格裁剪图，未作修改。图片适用 CC BY 2.0，代码适用 Apache-2.0。
