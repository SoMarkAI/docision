# Evaluation

[Project](../README.md)

## Run

```bash
python -m docision.evaluate --model models/Qwen3-VL-4B-Instruct \
  --input examples/multiple.json --protocol circular --output outputs/evaluation.json
```

Inputs can be a labelled document JSON or JSONL with one document per line, each carrying `questions[]`. Flat document records also work. Images resolve relative to the input file. The evaluator never sends the reference label, language or metadata as prompt text.

The output report contains original-order metrics, all-variant metrics, language breakdowns, label histograms and semantic consistency. A companion `.predictions.jsonl` records order mappings, probabilities and semantic predictions. Local image/server paths are not copied into prediction records.

Offline rescoring requires no model:

```bash
python -m docision.evaluate --predictions outputs/evaluation.predictions.jsonl \
  --output outputs/recomputed.json
```

## doc-decision-bench

doc-decision-bench contains 608 images, 2617 bilingual pairs and 5234 language records. Use `grouped.jsonl` as `--input`, with the downloaded `images/` directory alongside it.

Use `--protocol none` for original-order accuracy or `--protocol circular` for a separate candidate-order experiment. 

## Metrics

| Metric | Definition |
| :--- | :--- |
| Accuracy | Fraction whose highest-probability candidate matches the reference |
| Permutation Consistency | Fraction of questions selecting the same semantic candidate in every tested order |
| All-variants Accuracy | Fraction correct for every tested order |
| Identity Agreement | Fraction of nonidentity predictions matching their original-order semantic prediction |


Consistency is not accuracy: consistently selecting the wrong candidate scores 100% consistency and 0% all-variants accuracy. Original-only evaluations contain no order intervention, so their consistency of 1 is trivial and should not be reported as robustness evidence.

## Candidate order

- `--protocol none`: original order only; original-order scoring.
- `--protocol circular`: all K cyclic rotations, including identity. Every semantic candidate visits every position. This is not all K! permutations.
- `--protocol random --random-count 4 --seed 42`: identity plus up to four distinct seeded random orders. Small candidate sets may exhaust the available permutations first.

Labels are reassigned for each order. Predictions are mapped back to original zero-based candidate indices before consistency or reference comparisons. The original question and candidate strings never change.

## Compare original and trained weights

Use the same labelled JSON/JSONL input for both complete local checkpoints:

```bash
python -m docision.compare --base-model models/Qwen3-VL-4B-Instruct \
  --trained-model /path/to/Docision \
  --input examples/multiple.json --out outputs/Docision-comparison
```

The comparison runs each model sequentially on one visible GPU. It freezes image/input hashes, checks every actual processor tensor and prompt for equality, saves candidate probabilities, and recomputes paired Accuracy. It uses causal/native/full projection, temperature 1, original candidate order and no prefix cache. A failure preserves partial files without `_SUCCESS`; use a fresh output directory for a new run. Keep local outputs private when the input contains private media or questions.

The package, Python API and CLI use the Docision name.

## Results

| Model | Document understanding & decisions (3,696 records) | Cross-industry classification (1,538 records) |
|---|---:|---:|
| Qwen3-VL-4B-Instruct | 82.85% | 96.23% |
| **Docision 4B** | **95.13%** | **98.70%** |
| JEV-27B-VL | 94.07% | 99.67% |

## Offline coverage verification

Live scoring writes a companion `.predictions.manifest.json` before inference, containing the expected questions and variants. Offline scoring reads it automatically or accepts `--manifest PATH`. Missing questions/variants, inconsistent candidate counts, and nonidentity variant 0 are rejected. Legacy predictions without a manifest require explicit `--allow-unverified-coverage` and report `coverage.verified=false`. A summary alone cannot prove completeness.
