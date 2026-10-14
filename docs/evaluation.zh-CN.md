# 评测

[项目首页](../README.zh-CN.md)

## 运行

```bash
python -m docision.evaluate --model models/Qwen3-VL-4B-Instruct \
  --input examples/multiple.json --protocol circular --output outputs/evaluation.json
```

输入可以是带参考标签的文档 JSON，也可以是每行一份文档、每份文档包含 `questions[]` 的 JSONL。扁平的文档记录也受支持。图片路径相对于输入文件解析。评测器不会把参考标签、语言或元数据作为提示词文本发送给模型。

输出报告包含原始顺序指标、全部排列变体的指标、按语言分类的结果、标签直方图和语义一致性。配套的 `.predictions.jsonl` 记录顺序映射、概率和语义预测。本地图片或服务器路径不会复制到预测记录中。

离线重新计分不需要模型：

```bash
python -m docision.evaluate --predictions outputs/evaluation.predictions.jsonl \
  --output outputs/recomputed.json
```

## doc-decision-bench

doc-decision-bench 包含 608 张图片、2617 对双语题目和 5234 条语言记录。将 `grouped.jsonl` 作为 `--input`，并把下载的 `images/` 目录放在它旁边。

使用 `--protocol none` 计算原始候选顺序下的准确率，或使用 `--protocol circular` 单独进行候选顺序实验。

## 指标

| 指标 | 定义 |
| :--- | :--- |
| Accuracy（准确率） | 概率最高的候选项与参考答案一致的比例 |
| Permutation Consistency（排列一致性） | 在所有测试顺序中都选择同一个语义候选项的问题比例 |
| All-variants Accuracy（全部变体均正确率） | 在每一种测试顺序下都回答正确的问题比例 |
| Identity Agreement（与原始顺序的一致率） | 非原始顺序的预测与原始顺序预测选择同一语义候选项的比例 |

一致性不等于准确率：始终选择同一个错误候选项，会得到 100% 的一致性和 0% 的全部变体均正确率。只评测原始顺序时，没有对顺序进行干预，因此一致性为 1 是自然结果，不应被用作稳健性的证据。

## 候选顺序

- `--protocol none`：仅使用原始顺序，并按原始顺序计分。
- `--protocol circular`：使用全部 K 种循环移位，包括原始顺序。每个语义候选项都会出现在每一个位置。这并不是全部 K! 种排列。
- `--protocol random --random-count 4 --seed 42`：原始顺序，加上最多四种由固定随机种子生成的不同随机顺序。候选集合较小时，可能先用尽所有可用排列。

每种顺序都会重新分配标签。在计算一致性或与参考答案比较之前，预测会映射回原始候选项的从零开始的索引。原始问题和候选答案字符串始终保持不变。

## 比较原始权重与训练后权重

对两个完整的本地检查点使用同一份带参考标签的 JSON/JSONL 输入：

```bash
python -m docision.compare --base-model models/Qwen3-VL-4B-Instruct \
  --trained-model /path/to/Docision \
  --input examples/multiple.json --out outputs/Docision-comparison
```

比较程序在同一张可见 GPU 上依次运行两个模型。它固定图片和输入文件的哈希，检查实际处理器张量和提示词是否一致，保存候选概率，并重新计算配对准确率。运行采用 causal/native/full 投影、温度 1、原始候选顺序，不使用前缀缓存。失败时会保留部分文件，但不会生成 `_SUCCESS`；重新运行时请使用新的输出目录。如果输入包含私密媒体或问题，应将本地输出保持私有。

软件包、Python API 和 CLI 均使用 Docision 名称。

## 结果

| 模型 | 文档理解与判断（3,696 条记录） | 跨行业分类（1,538 条记录） |
|---|---:|---:|
| Qwen3-VL-4B-Instruct | 82.85% | 96.23% |
| **Docision 4B** | **95.13%** | **98.70%** |
| JEV-27B-VL | 94.07% | 99.67% |

## 离线覆盖率校验

在线评测在推理前生成配套的 `.predictions.manifest.json`，包含预期题目及变体清单。离线重算自动读取清单，也可使用 `--manifest PATH` 指定。缺题、缺变体、候选数量不一致和非原始顺序冒充变体 0 均会被拒绝。旧预测没有清单时，必须显式指定 `--allow-unverified-coverage`，报告标明 `coverage.verified=false`。汇总 JSON 本身不能证明评测完整。
