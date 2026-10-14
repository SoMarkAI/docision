# Examples

[English](readme.md) · [简体中文](readme.zh-CN.md) · [Project](../README.md)

One table image with four Chinese/English question pairs.

| File | Contents |
|---|---|
| `validation-table.jpg` | Source image |
| `single.json` | One question with four candidates |
| `multiple.json` | Eight questions: four Chinese/English pairs |
| `recorded-output.json` | Historical native BF16 results; optional fields may differ from the current version |

Run from the repository root:

```bash
python demo_docision.py --model models/docision-1.0 --input examples/single.json
python demo_docision.py --model models/docision-1.0 --input examples/multiple.json
```

For same-page caching and batching, add `--numerics stable --prefix-cache --question-batch-size 2`.

Model: [SoMarkAI/Docision-1.0](https://huggingface.co/SoMarkAI/Docision-1.0)

## Image attribution

Table 3 from Digant Gupta, Carolyn A Lammersfeld, Pankaj G Vashi, Sadie L Dahlk and Christopher G Lis, *Can subjective global assessment of nutritional status predict survival in ovarian cancer?*, **Journal of Ovarian Research 1, 5 (2008)**. © 2008 Gupta et al.; licensee BioMed Central Ltd.

[Original article](https://link.springer.com/article/10.1186/1757-2215-1-5) · [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/)

Table crop supplied by Infinity, unchanged. The image uses CC BY 2.0; the code uses Apache-2.0.
