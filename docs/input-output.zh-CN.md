# 输入与输出

[项目首页](../README.zh-CN.md)

## 文档输入

一个 JSON 对象描述一张本地文档图片，以及一个或多个问题：

```json
{
  "image": "validation-table.jpg",
  "context": "Inspect the supplied media carefully.",
  "questions": [
    {
      "id": "Docision:c5bc6f4e69f7b8bf_r0q04:en",
      "language": "en",
      "question": "Are the 95% CIs for moderately malnourished and stage at diagnosis the same?",
      "candidates": {
        "A": "Different",
        "B": "The same"
      },
      "label": "A"
    }
  ]
}
```

`image` 或 `media_path` 相对于 JSON 文件解析。此底层接口仅支持图片媒体。`context` 默认使用上面示例中的文本。问题 ID 应保持唯一；评测器要求唯一 ID 和参考标签。`language` 可选，仅用于报告。`label` 是参考候选标签，不是提示词指令。

只有一个问题时，可以省略 `questions`，把 `question`、`candidates` 和可选的 `label` 放在顶层。支持 `{"A":"Different","B":"The same"}` 这样的有序映射。兼容的嵌套问题结构也受支持：

```json
{
  "image": "validation-table.jpg",
  "question": {
    "instructions": "Are the 95% CIs for moderately malnourished and stage at diagnosis the same?",
    "criteria": {
      "A": "Different",
      "B": "The same"
    }
  },
  "label": "A"
}
```

程序不会自动改写或扩展问题。候选文本必须非空且互不重复。输入限制为 2–128 个候选项，同时还受实际单 token 标签能力的约束；这不代表每个 tokenizer 都支持 128 个标签。问题、上下文和候选答案中出现聊天控制 token 时，输入会被拒绝。

## 输出

输出是 JSON 数组，按输入顺序为每个问题提供一个结果。结构如下：

```text
{
  id,
  backend: "transformers",
  model_family: "qwen3_vl",
  mode: "causal" | "isolated",
  projection: "full",
  numerics: "native" | "stable",
  candidates: [{label, text, probability}, ...],
  decision: {label, text, probability,
             index, tied_labels, tie_policy},
  probability_sum,
  prompt_tokens,
  num_cached_tokens,
  cache_note
}
```

`probability` 仅在所提供的标签之间计算 softmax。`decision.index` 从零开始；概率并列时选择输入顺序中的第一个候选项。token 数量包括处理器展开后的完整图片输入。`--check-only` 不加载权重，只返回准备好的 token 区间、标签 token ID 和 token 数量。

`examples/recorded-output.json` 展示了 Docision 在留出的验证表格上运行原生 BF16 推理得到的真实输出。

## Python API

```python
from docision import DocisionEngine
from docision.io import load_document

engine = DocisionEngine("models/Qwen3-VL-4B-Instruct")
results = engine.score_document(load_document("examples/multiple.json"))
```

显式复用跨调用的公共上下文：

```python
engine = DocisionEngine("models/Qwen3-VL-4B-Instruct", numerics="stable")
document = load_document("examples/multiple.json")
cache = engine.prepare_context(document["image"], context=document["context"])
results = engine.score_questions(cache, document["questions"],
                                 projection="full", batch_size=2)
del cache  # release persistent KV references
```

缓存仅属于创建它的引擎。更改公共上下文、媒体或元数据后，需要建立新缓存。不要修改缓存字段。在引擎缓存的整个生命周期内，应保持同一种数值模式。
