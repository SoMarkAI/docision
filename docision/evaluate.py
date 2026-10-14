"""Score labelled document JSONL, or evaluate saved predictions offline."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from .io import load_document, normalize_document, write_json
from .metrics import permutation_metrics, probability_metrics
from .permutations import orders
from docision._runtime.inputs import candidate_labels


def load_benchmark(path):
    path = Path(path)
    if path.suffix == ".json":
        documents = [load_document(path)]
    else:
        documents = [
            normalize_document(json.loads(line), path.parent)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    questions, identifiers = [], set()
    for doc_index, document in enumerate(documents):
        for index, question in enumerate(document["questions"]):
            identifier = question.get("id", f"{doc_index}:{index}")
            if not isinstance(identifier, str) or not identifier or identifier in identifiers:
                raise ValueError("Question IDs must be unique nonempty strings")
            identifiers.add(identifier)
            if "label" not in question:
                raise ValueError("Benchmark questions require reference labels")
            questions.append(
                {
                    **question,
                    "id": identifier,
                    "image": document["image"],
                    "context": document["context"],
                }
            )
    if not questions:
        raise ValueError("Benchmark is empty")
    return questions


def score_benchmark(
    engine,
    questions,
    protocol="none",
    random_count=4,
    seed=0,
    mode="causal",
    prefix_cache=False,
    batch_size=1,
):
    groups = defaultdict(list)
    for question in questions:
        groups[(question["image"], question["context"])].append(question)
    results = []
    for (image, context), values in groups.items():
        requests, metadata = [], []
        for question in values:
            labels = candidate_labels(len(question["candidates"]))
            target = labels.index(question["label"])
            question_seed = seed + int.from_bytes(
                hashlib.sha256(question["id"].encode()).digest()[:8], byteorder="big"
            )
            for variant, order in enumerate(
                orders(len(labels), protocol, random_count, question_seed)
            ):
                requests.append(
                    {
                        "id": f"{question['id']}@{variant}",
                        "question": question["question"],
                        "candidates": [question["candidates"][i] for i in order],
                    }
                )
                metadata.append((question, variant, order, target))
        scored = engine.score_many(
            image,
            requests,
            context=context,
            mode=mode,
            projection="full",
            use_prefix_cache=prefix_cache,
            batch_size=batch_size,
        )
        if len(scored) != len(metadata):
            raise ValueError("Model returned the wrong number of predictions")
        for result, (question, variant, order, target) in zip(scored, metadata, strict=True):
            rows = result["candidates"]
            results.append(
                {
                    "id": question["id"],
                    "language": question.get("language", "unspecified"),
                    "variant": variant,
                    "order": list(order),
                    "target_index": order.index(target),
                    "semantic_target": target,
                    "semantic_prediction": order[result["decision"]["index"]],
                    "probabilities": [row["probability"] for row in rows],
                    "num_cached_tokens": result["num_cached_tokens"],
                    "question_sha256": question_fingerprint(question),
                }
            )
    return results


def question_fingerprint(question):
    payload = {key: question.get(key, "") for key in ("question", "context")}
    payload["candidates"] = question["candidates"]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def build_manifest(questions, input_sha256, protocol="none", random_count=4, seed=0):
    entries = []
    for question in questions:
        count = len(question["candidates"])
        question_seed = seed + int.from_bytes(
            hashlib.sha256(question["id"].encode()).digest()[:8], byteorder="big")
        entries.append({
            "id": question["id"], "candidate_count": count,
            "semantic_target": candidate_labels(count).index(question["label"]),
            "language": question.get("language", "unspecified"),
            "question_sha256": question_fingerprint(question),
            "orders": [list(order) for order in orders(count, protocol, random_count, question_seed)],
        })
    return {"schema_version": 1, "input_sha256": input_sha256, "protocol": protocol,
            "random_count": random_count, "seed": seed, "questions": entries}


def validate_predictions(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("Need nonempty predictions")
    groups = defaultdict(list)
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError("Prediction IDs must be nonempty strings")
        variant = row.get("variant")
        if type(variant) is not int or variant < 0:
            raise ValueError("Variant must be a nonnegative integer")
        probability_metrics([row])
        count = len(row["probabilities"])
        order = row["order"]
        if (not isinstance(order, list) or any(type(i) is not int for i in order)
                or sorted(order) != list(range(count))):
            raise ValueError("Saved candidate order is not a permutation")
        if variant == 0 and order != list(range(count)):
            raise ValueError("Variant 0 must use original candidate order")
        target = row["target_index"]
        predicted = max(range(count), key=row["probabilities"].__getitem__)
        if (type(row["semantic_target"]) is not int
                or type(row["semantic_prediction"]) is not int
                or row["semantic_target"] != order[target]
                or row["semantic_prediction"] != order[predicted]):
            raise ValueError("Saved semantic mapping disagrees with candidate probabilities")
        groups[row["id"]].append(row)
    for identifier, values in groups.items():
        if len({len(row["probabilities"]) for row in values}) != 1:
            raise ValueError(f"Candidate counts differ between variants: {identifier}")
        if len({row.get("language", "unspecified") for row in values}) != 1:
            raise ValueError(f"Languages differ between variants: {identifier}")
        if len({tuple(row["order"]) for row in values}) != len(values):
            raise ValueError(f"Repeated candidate order: {identifier}")
    # Also checks identity presence, variant uniqueness and semantic targets.
    permutation_metrics(rows)


def validate_coverage(rows, manifest):
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Unsupported evaluation manifest")
    digest = manifest.get("input_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Manifest requires input SHA-256")
    entries = manifest.get("questions")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest requires a complete question list")
    expected = {}
    for entry in entries:
        identifier = entry.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in expected:
            raise ValueError("Manifest IDs must be unique nonempty strings")
        count = entry.get("candidate_count")
        target = entry.get("semantic_target")
        if type(count) is not int or count < 2 or type(target) is not int or not 0 <= target < count:
            raise ValueError("Invalid manifest candidates or target")
        question_seed = manifest["seed"] + int.from_bytes(
            hashlib.sha256(identifier.encode()).digest()[:8], byteorder="big")
        expected_orders = [list(o) for o in orders(count, manifest["protocol"],
                                                   manifest["random_count"], question_seed)]
        if entry.get("orders") != expected_orders:
            raise ValueError("Manifest orders disagree with its protocol")
        expected[identifier] = entry
    actual = defaultdict(dict)
    for row in rows:
        actual[row["id"]][row["variant"]] = row
    missing, extra = sorted(expected.keys() - actual.keys()), sorted(actual.keys() - expected.keys())
    if missing or extra:
        raise ValueError(f"Evaluation coverage mismatch: missing {len(missing)} questions {missing}; extra {len(extra)} questions {extra}")
    for identifier, entry in expected.items():
        values = actual[identifier]
        if set(values) != set(range(len(entry["orders"]))):
            raise ValueError(f"Evaluation variant coverage mismatch: {identifier}")
        for variant, row in values.items():
            if (len(row["probabilities"]) != entry["candidate_count"]
                    or row["semantic_target"] != entry["semantic_target"]
                    or row["order"] != entry["orders"][variant]
                    or row.get("language", "unspecified") != entry["language"]
                    or row.get("question_sha256") != entry["question_sha256"]):
                raise ValueError(f"Prediction does not match manifest: {identifier} variant {variant}")
    return {"verified": True, "basis": "supplied_manifest", "input_sha256": digest,
            "expected_questions": len(expected), "observed_questions": len(actual),
            "expected_variants": sum(len(e["orders"]) for e in entries),
            "observed_variants": len(rows)}


def summarize(rows, manifest=None):
    validate_predictions(rows)
    coverage = (validate_coverage(rows, manifest) if manifest is not None else
                {"verified": False, "reason": "No independent expected-question manifest supplied"})
    identity = [row for row in rows if row["variant"] == 0]
    languages = sorted({row.get("language", "unspecified") for row in identity})
    return {
        "coverage": coverage,
        "original_order": probability_metrics(identity),
        "all_variants": probability_metrics(rows),
        "permutation": permutation_metrics(rows),
        "languages": {
            lang: probability_metrics(
                [r for r in identity if r.get("language", "unspecified") == lang]
            )
            for lang in languages
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Labelled JSON or JSONL benchmark")
    source.add_argument("--predictions", type=Path, help="Saved Docision evaluation JSONL")
    parser.add_argument("--manifest", type=Path, help="Expected evaluation manifest for offline coverage checks")
    parser.add_argument("--allow-unverified-coverage", action="store_true",
                        help="Explicitly rescore old predictions without proving completeness")
    parser.add_argument("--model", help="Local official model or trained checkpoint")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--protocol", choices=["none", "circular", "random"], default="none")
    parser.add_argument("--random-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--numerics", choices=["native", "stable"], default="native")
    parser.add_argument("--mode", choices=["causal", "isolated"], default="causal")
    parser.add_argument("--prefix-cache", action="store_true")
    parser.add_argument("--question-batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.question_batch_size < 1 or args.random_count < 1:
        parser.error("batch size and random count must be positive")
    manifest = None
    if args.predictions:
        manifest_path = args.manifest or args.predictions.with_suffix(".manifest.json")
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        elif args.manifest or not args.allow_unverified_coverage:
            parser.error("Offline scoring requires --manifest or a companion .manifest.json; use --allow-unverified-coverage only for explicitly unverified legacy results")
        rows = [
            json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        metadata = {
            "source": "saved_predictions",
            "prediction_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
        }
    else:
        if not args.model:
            parser.error("--model is required when scoring a benchmark")
        if (args.prefix_cache or args.question_batch_size > 1) and args.numerics != "stable":
            parser.error("Cache/batch mode requires --numerics stable")
        from .engine import DocisionEngine

        questions = load_benchmark(args.input)
        manifest = build_manifest(questions, hashlib.sha256(args.input.read_bytes()).hexdigest(),
                                  args.protocol, args.random_count, args.seed)
        manifest_path = args.output.with_suffix(".predictions.manifest.json")
        write_json(manifest_path, manifest)
        engine = DocisionEngine(args.model, numerics=args.numerics)
        rows = score_benchmark(
            engine,
            questions,
            args.protocol,
            args.random_count,
            args.seed,
            args.mode,
            args.prefix_cache,
            args.question_batch_size,
        )
        prediction_path = args.output.with_suffix(".predictions.jsonl")
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
        labels = Counter(question["label"] for question in questions)
        metadata = {
            "source": "live_scoring",
            "model_name": Path(args.model).name,
            "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
            "protocol": args.protocol,
            "seed": args.seed,
            "mode": args.mode,
            "numerics": args.numerics,
            "prefix_cache": args.prefix_cache,
            "question_batch_size": args.question_batch_size,
            "label_histogram": dict(labels),
            "label_histogram_by_candidate_count": {
                str(count): dict(
                    Counter(q["label"] for q in questions if len(q["candidates"]) == count)
                )
                for count in sorted({len(q["candidates"]) for q in questions})
            },
        }
    if manifest is not None:
        metadata["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    report = {**metadata, **summarize(rows, manifest)}
    write_json(args.output, report)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
