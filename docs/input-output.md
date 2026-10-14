# Input and output

[Project](../README.md)

## Document input

A JSON object describes one local document image and one or more questions:

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

`image` or `media_path` resolves relative to the JSON file. Supported media is images only in this low-level interface. `context` defaults to the text above. Question IDs should be unique; the evaluator requires unique IDs and reference labels. `language` is optional and used only for reporting. `label` is the reference candidate label; it is not a prompt instruction.

For one question, omit `questions` and put `question`, `candidates` and optional `label` at the top level. Ordered mappings `{"A":"Different","B":"The same"}` are accepted. The compatible nested question structure also works:

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

No question is rewritten or expanded automatically. Candidate text must be nonempty and distinct. The input limit is 2–128 candidates, further constrained by actual single-token labels. This is not a promise that every tokenizer supports 128 labels. Chat control tokens in question/context/candidates are rejected.

## Output

A JSON array contains one result per question, in input order. Its schema is:

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

`probability` is softmax over supplied labels only. `decision.index` is zero-based; ties choose the first candidate in input order. Token counts include the full processor-expanded image input. `--check-only` returns prepared spans, label token IDs and token counts without loading weights.

See `examples/recorded-output.json` for a real Docision native BF16 output on the held-out validation table.

## Python API

```python
from docision import DocisionEngine
from docision.io import load_document

engine = DocisionEngine("models/Qwen3-VL-4B-Instruct")
results = engine.score_document(load_document("examples/multiple.json"))
```

Explicit shared context reuse across calls:

```python
engine = DocisionEngine("models/Qwen3-VL-4B-Instruct", numerics="stable")
document = load_document("examples/multiple.json")
cache = engine.prepare_context(document["image"], context=document["context"])
results = engine.score_questions(cache, document["questions"],
                                 projection="full", batch_size=2)
del cache  # release persistent KV references
```

Caches are engine-local. Changing public context, media or metadata requires a new cache. Never mutate cache fields. Keep one numerical profile for an engine's cache lifetime.
