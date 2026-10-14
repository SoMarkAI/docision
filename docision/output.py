"""Public result formatting; never mutate internal scoring results."""
from functools import wraps


def public_output(value):
    """Keep choices/probabilities and operational metadata, hide scoring internals."""
    if isinstance(value, dict):
        return {key: public_output(item) for key, item in value.items()
                if key not in {"raw_logit", "raw_logits", "token_id"}}
    if isinstance(value, list):
        return [public_output(item) for item in value]
    return value


def public_scores(method):
    @wraps(method)
    def wrapped(*args, **kwargs):
        return public_output(method(*args, **kwargs))
    return wrapped
