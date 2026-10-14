"""Bounded page-local question batches with per-question error recovery."""
import time


def _is_cuda_oom(error):
    import torch
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and 'cuda out of memory' in str(error).lower())


def _release_unused_cuda_memory():
    # Called only after the failed exception scope has exited. Live model/page
    # KV remains referenced; only unused allocations in this process are freed.
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def score_cached_page(engine, cache, specs, batch_size=None):
    """Start with all questions; shrink the page's batch limit only on CUDA OOM."""
    if batch_size is not None and (isinstance(batch_size, bool) or
                                  not isinstance(batch_size, int) or batch_size < 1):
        raise ValueError('batch_size must be a positive integer or None')
    if not specs:
        return []
    limit = len(specs) if batch_size is None else batch_size
    outcomes, reductions = [], []
    offset = 0
    retry_seconds = 0.0
    while offset < len(specs):
        chunk = specs[offset:offset+limit]
        started = time.perf_counter()
        warning = None
        oom = False
        try:
            values = engine.score_questions(cache, chunk, mode='causal',
                                            projection='full', batch_size=len(chunk))
            if len(values) != len(chunk):
                raise RuntimeError('缓存推理返回的结果数量与问题数量不一致。')
        except Exception as error:
            warning = f'{type(error).__name__}: {error}'
            oom = _is_cuda_oom(error)
        elapsed = time.perf_counter()-started
        if warning is None:
            seconds = (elapsed + retry_seconds) / len(chunk)
            outcomes.extend({'result': value, 'elapsed_seconds': round(seconds, 6),
                             'timing_scope': 'batch_average_including_retry',
                             'inference_batch_size': len(chunk),
                             'batch_reductions': list(reductions)} for value in values)
            offset += len(chunk)
            retry_seconds = 0.0
            continue
        if oom:
            cleanup_started = time.perf_counter()
            _release_unused_cuda_memory()
            retry_seconds += elapsed + time.perf_counter()-cleanup_started
            if len(chunk) > 1:
                new_limit = max(1, len(chunk)//2)
                reductions.append({'from': len(chunk), 'to': new_limit, 'reason': warning})
                limit = new_limit
                # Retry these same questions; retain the smaller size for the page.
                continue
        # Non-OOM batch failures keep per-question isolation. A failed singleton
        # is already isolated and must not loop/retry indefinitely.
        if len(chunk) == 1:
            outcomes.append({'error': warning, 'batch_warning': warning,
                             'inference_batch_size': 1, 'batch_reductions': list(reductions),
                             'elapsed_seconds': round(retry_seconds if oom else elapsed+retry_seconds, 6),
                             'timing_scope': 'question_including_retry'})
        else:
            failed_share = (elapsed + retry_seconds) / len(chunk)
            for spec in chunk:
                # Single-question path also releases unused memory if it OOMs.
                row = score_cached_page(engine, cache, [spec], batch_size=1)[0]
                row['batch_warning'] = warning
                row['batch_reductions'] = list(reductions) + row['batch_reductions']
                row['elapsed_seconds'] = round(row['elapsed_seconds']+failed_share, 6)
                outcomes.append(row)
        offset += len(chunk)
        retry_seconds = 0.0
    return outcomes


from contextlib import contextmanager, nullcontext
from unittest.mock import patch


def _settings_key(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_settings_key(x) for x in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _settings_key(v)) for k, v in value.items()))
    raise TypeError('Non-scalar processor setting: skip reuse')


def _image_key(images):
    from PIL import Image
    if isinstance(images, Image.Image):
        return id(images)
    if isinstance(images, (list, tuple)):
        return tuple(_image_key(image) for image in images)
    raise TypeError('Unsupported media object: skip reuse')


@contextmanager
def reuse_page_preprocessing(engine, enabled=True):
    """Reuse only identical in-memory page images and processor settings.

    The caller owns this scope for one page. Tensors are read-only; outer
    mappings/lists are copied. Restore the processor even if inference fails.
    """
    processor = getattr(getattr(engine, 'builder', None), 'processor', None)
    original = getattr(processor, '_process_images', None)
    if not enabled or original is None:
        yield
        return
    memo = {}
    def process(images, **kwargs):
        try:
            key = (_image_key(images), _settings_key(kwargs))
        except TypeError:
            return original(images, **kwargs)
        if key not in memo:
            processed, replacements = original(images, **kwargs)
            # Retain images so their identities cannot be recycled in this scope.
            memo[key] = (images, dict(processed), list(replacements))
        _, processed, replacements = memo[key]
        return dict(processed), list(replacements)
    try:
        with getattr(engine, 'lock', nullcontext()), patch.object(processor, '_process_images', process):
            yield
    finally:
        memo.clear()
