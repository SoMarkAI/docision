"""Compare complete base/trained checkpoints on one frozen labelled image cohort."""

import argparse
import gc
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .evaluate import load_benchmark, score_benchmark, summarize, build_manifest
from .io import write_json


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def signature(batch, spec):
    import torch

    tensors = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().contiguous()
            tensors[key] = {
                "shape": list(value.shape), "dtype": str(value.dtype),
                "sha256": hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest(),
            }
        else:
            tensors[key] = value
    fields = {key: spec[key] for key in (
        "text", "raw_ids", "spans", "labels", "label_ids", "prefix_length",
        "prompt_tokens", "candidate_texts",
    )}
    return {"tensors": tensors, "prompt": fields}


def paired_summary(base, trained):
    if len(base) != len(trained) or not base:
        raise ValueError("Models must score the same nonempty cohort")
    groups = defaultdict(list)
    for before, after in zip(base, trained, strict=True):
        for key in ("id", "language", "variant", "order", "semantic_target", "target_index"):
            if before[key] != after[key]:
                raise ValueError(f"Paired reference mismatch: {key}")
        pair = (before["semantic_prediction"] == before["semantic_target"],
                after["semantic_prediction"] == after["semantic_target"])
        groups["overall"].append(pair)
        groups["language:" + before["language"]].append(pair)
    result = {}
    for group, values in groups.items():
        n = len(values)
        b, t = sum(x for x, _ in values), sum(y for _, y in values)
        result[group] = {
            "records": n, "base_correct": b, "trained_correct": t,
            "base_accuracy": b / n, "trained_accuracy": t / n,
            "delta_percentage_points": 100 * (t - b) / n,
            "wrong_to_right": sum(not x and y for x, y in values),
            "right_to_wrong": sum(x and not y for x, y in values),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--trained-model", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Use a fresh output directory; partial runs are preserved")
    questions = load_benchmark(args.input)
    image_hashes = {q["image"]: digest(q["image"]) for q in questions}
    input_hash = digest(args.input)
    args.out.mkdir(parents=True)
    write_json(args.out / "cohort.json", {
        "input_sha256": input_hash, "image_hashes": image_hashes,
        "questions": questions,
    })
    manifest = build_manifest(questions, input_hash)
    write_json(args.out / "evaluation.manifest.json", manifest)
    import torch
    from .engine import DocisionEngine

    all_rows, all_inputs = {}, {}
    for name, model in (("base", args.base_model), ("trained", args.trained_model)):
        if digest(args.input) != input_hash or any(digest(p) != h for p, h in image_hashes.items()):
            raise ValueError("Frozen input or media changed")
        torch.manual_seed(0)
        engine = DocisionEngine(model, numerics="native", device_map="cuda:0")
        captures = []
        original = engine.prepare

        def prepare(*values, **kwargs):
            batch, spec, decoded = original(*values, **kwargs)
            captures.append(signature(batch, spec))
            return batch, spec, decoded

        engine.prepare = prepare
        with torch.inference_mode():
            rows = score_benchmark(engine, questions, protocol="none", mode="causal",
                                   prefix_cache=False, batch_size=1)
        if len(rows) != len(questions) or len(captures) != len(rows):
            raise ValueError("Incomplete prediction or input witness coverage")
        report = summarize(rows, manifest)
        write_json(args.out / f"{name}.json", report)
        write_json(args.out / f"{name}.inputs.json", captures)
        (args.out / f"{name}.predictions.jsonl").write_text(
            "".join(json.dumps(r, allow_nan=False) + "\n" for r in rows)
        )
        all_rows[name], all_inputs[name] = rows, captures
        print(json.dumps({"model": name, **report["original_order"]}), flush=True)
        # prepare is an instance attribute containing a bound method; remove the
        # closure before releasing the engine so a second model fits on one GPU.
        del engine.prepare
        del prepare, original, engine
        gc.collect()
        torch.cuda.empty_cache()
    if all_inputs["base"] != all_inputs["trained"]:
        raise ValueError("Actual processor/template inputs differ between models")
    comparison = {
        "models": {"base": Path(args.base_model).name, "trained": Path(args.trained_model).name},
        "input_sha256": input_hash, "actual_inputs_match": True,
        "protocol": {"attention": "causal", "numerics": "native", "projection": "full",
                     "temperature": 1, "prefix_cache": False, "batch_size": 1},
        "groups": paired_summary(all_rows["base"], all_rows["trained"]),
    }
    write_json(args.out / "comparison.json", comparison)
    (args.out / "_SUCCESS").write_text("Both models completed; actual input witnesses match.\n")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
