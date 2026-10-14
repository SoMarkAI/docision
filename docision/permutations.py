"""Reorder candidates and remap answers without changing semantic content."""

import random


def orders(count, protocol="none", random_count=4, seed=0):
    if count < 2:
        raise ValueError("Need at least two candidates")
    identity = tuple(range(count))
    if protocol == "none":
        return [identity]
    if protocol == "circular":
        return [identity[shift:] + identity[:shift] for shift in range(count)]
    if protocol != "random" or random_count < 1:
        raise ValueError("Use none, circular, or random with positive random count")
    # Bounded unique sampling: never spin indefinitely when K! is exhausted.
    rng = random.Random(seed)
    found = [identity]
    seen = {identity}
    for _ in range(max(100, random_count * 30)):
        value = list(identity)
        rng.shuffle(value)
        order = tuple(value)
        if order not in seen:
            found.append(order)
            seen.add(order)
        if len(found) == random_count + 1:
            break
    return found
