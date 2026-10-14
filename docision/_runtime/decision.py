"""Deterministic argmax on unmodified candidate logits."""
import math

def candidate_probabilities(logits):
    """Compute candidate softmax in float64 without changing logit decisions."""
    return logits.double().softmax(dim=-1)


def choose(candidates):
    if not candidates or any(not math.isfinite(c['raw_logit']) for c in candidates):
        raise ValueError('Expected nonempty finite candidate logits')
    winner = max(range(len(candidates)), key=lambda i: candidates[i]['raw_logit'])
    ties = [c['label'] for c in candidates if c['raw_logit'] == candidates[winner]['raw_logit']]
    return {**candidates[winner], 'index': winner, 'tied_labels': ties,
            'tie_policy': 'first_in_input_order', 'tie_basis': 'raw_logits'}
