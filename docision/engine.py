"""Document-only API over the pinned native HF runtime."""

from pathlib import Path
from PIL import Image
from .io import DEFAULT_CONTEXT
from .output import public_scores
from docision._runtime.families import VL, detect
from docision._runtime.hf import HFDocisionEngine
from docision._runtime.prompt import PromptBuilder


class DocisionEngine(HFDocisionEngine):
    # The runtime computes logits and selects the winner before formatting.
    @public_scores
    def score_prepared(self, batch, spec, *, mode="causal", projection="full", cache=None):
        return super().score_prepared(batch, spec, mode=mode, projection=projection, cache=cache)

    @public_scores
    def score_many(self, media, questions, *, modality="image", context=DEFAULT_CONTEXT,
                   mode="causal", video_options=None, projection="full",
                   use_prefix_cache=False, batch_size=1):
        return super().score_many(media, questions, modality=modality, context=context,
                                  mode=mode, video_options=video_options, projection=projection,
                                  use_prefix_cache=use_prefix_cache, batch_size=batch_size)

    @public_scores
    def score_questions(self, cache, questions, *, mode="causal", projection="full", batch_size=1):
        return super().score_questions(cache, questions, mode=mode,
                                       projection=projection, batch_size=batch_size)

    score_requests = public_scores(HFDocisionEngine.score_requests)

    def __init__(
        self,
        model_path,
        *,
        numerics="native",
        device_map="auto",
        dtype="bfloat16",
        max_input_tokens=4000,
        min_pixels=3136,
        max_pixels=501760,
    ):
        if detect(model_path) != VL:
            raise ValueError("Docision requires a Qwen3-VL checkpoint")
        validate_limits(max_input_tokens, min_pixels, max_pixels)
        super().__init__(
            model_path,
            numerics=numerics,
            device_map=device_map,
            dtype=dtype,
            max_input_tokens=max_input_tokens,
        )
        self.builder.processor.image_processor.size = {
            "shortest_edge": min_pixels,
            "longest_edge": max_pixels,
        }

    def prepare(
        self,
        media,
        question,
        candidates,
        *,
        modality="image",
        context=DEFAULT_CONTEXT,
        video_options=None,
        decoded=None,
    ):
        if modality != "image" or video_options:
            raise ValueError("Docision supports document images only")
        # Decode document images as PIL RGB without implicit EXIF rotation.
        if decoded is None:
            with Image.open(Path(media)) as image:
                decoded = {"images": [image.convert("RGB")]}
        return super().prepare(
            media, question, candidates, modality="image", context=context, decoded=decoded
        )

    @public_scores
    def score_document(self, document, *, mode="causal", prefix_cache=False, batch_size=1):
        return self.score_many(
            document["image"],
            document["questions"],
            context=document["context"],
            mode=mode,
            projection="full",
            use_prefix_cache=prefix_cache,
            batch_size=batch_size,
        )


def validate_limits(tokens, minimum, maximum):
    if (
        any(
            isinstance(v, bool) or not isinstance(v, int) or v < 1
            for v in (tokens, minimum, maximum)
        )
        or minimum > maximum
    ):
        raise ValueError("Input limits must be positive integers; min_pixels <= max_pixels")


def preflight(model_path, document, *, max_input_tokens=4000, min_pixels=3136, max_pixels=501760):
    """Check the actual processor/template/token labels without loading model weights."""
    validate_limits(max_input_tokens, min_pixels, max_pixels)
    if detect(model_path) != VL:
        raise ValueError("Docision requires a Qwen3-VL checkpoint")
    engine = DocisionEngine.__new__(DocisionEngine)
    engine.builder = PromptBuilder(model_path)
    engine.max_input_tokens = max_input_tokens
    engine.builder.processor.image_processor.size = {
        "shortest_edge": min_pixels,
        "longest_edge": max_pixels,
    }
    results, decoded = [], None
    for question in document["questions"]:
        _, spec, decoded = engine.prepare(
            document["image"],
            question["question"],
            question["candidates"],
            context=document["context"],
            decoded=decoded,
        )
        results.append(
            {key: spec[key] for key in ("prompt_tokens", "spans", "labels", "label_ids")}
        )
    return results
