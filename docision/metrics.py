"""Candidate accuracy and semantic permutation consistency."""

import math
from collections import defaultdict


def probabilities_from_logits(logits):
    if not isinstance(logits, (list, tuple)) or len(logits) < 2:
        raise ValueError("Need at least two candidate logits")
    if any(
        isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
        for x in logits
    ):
        raise ValueError("Logits must be finite numbers")
    peak = max(logits)
    weights = [math.exp(x - peak) for x in logits]
    total = sum(weights)
    return [x / total for x in weights]


def probability_metrics(rows):
    if not rows:
        raise ValueError("Need nonempty predictions")
    correct_sum = 0
    for row in rows:
        p = row["probabilities"]
        target = row["target_index"]
        if len(p) < 2 or any(
            isinstance(x, bool)
            or not isinstance(x, (int, float))
            or not math.isfinite(x)
            or not 0 <= x <= 1
            for x in p
        ):
            raise ValueError("Invalid candidate probabilities")
        if abs(sum(p) - 1) > 1e-5:
            raise ValueError("Candidate probabilities must sum to one")
        if isinstance(target, bool) or not isinstance(target, int) or not 0 <= target < len(p):
            raise ValueError("Target is outside the candidate set")
        predicted = max(range(len(p)), key=p.__getitem__)
        correct = int(predicted == target)
        correct_sum += correct
        if "raw_logits" in row:
            logits = row["raw_logits"]
            recomputed = probabilities_from_logits(logits)
            if len(logits) != len(p) or max(abs(a - b) for a, b in zip(p, recomputed)) > 1e-5:
                raise ValueError("Probabilities disagree with raw logits")
    count = len(rows)
    return {"count": count, "accuracy": correct_sum / count}


def permutation_metrics(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["id"]].append(row)
    if not groups:
        raise ValueError("Empty permutation cohort")
    consistent = all_correct = agreement = comparisons = 0
    for values in groups.values():
        targets = {row["semantic_target"] for row in values}
        if len(targets) != 1 or len({row["variant"] for row in values}) != len(values):
            raise ValueError("Permutation targets or variant IDs are inconsistent")
        identity = [row for row in values if row["variant"] == 0]
        if len(identity) != 1:
            raise ValueError("Each question requires exactly one identity variant")
        decisions = [row["semantic_prediction"] for row in values]
        consistent += len(set(decisions)) == 1
        all_correct += all(decision == next(iter(targets)) for decision in decisions)
        agreement += sum(
            row["semantic_prediction"] == identity[0]["semantic_prediction"]
            for row in values
            if row["variant"] != 0
        )
        comparisons += len(values) - 1
    return {
        "questions": len(groups),
        "variants": len(rows),
        "permutation_consistency": consistent / len(groups),
        "all_variants_accuracy": all_correct / len(groups),
        "identity_agreement": agreement / comparisons if comparisons else None,
    }
