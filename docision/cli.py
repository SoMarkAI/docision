"""Run native HF document scoring or processor-only preflight."""

import argparse
import json
from pathlib import Path
from .io import load_document, write_json


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--model", required=True, help="Local official model or trained checkpoint")
    value.add_argument("--input", required=True, type=Path)
    value.add_argument("--output", type=Path)
    value.add_argument("--mode", choices=["causal", "isolated"], default="causal")
    value.add_argument("--numerics", choices=["native", "stable"], default="native")
    value.add_argument("--device-map", default="auto")
    value.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    value.add_argument("--max-input-tokens", type=int, default=4000)
    value.add_argument("--min-pixels", type=int, default=3136)
    value.add_argument("--max-pixels", type=int, default=501760)
    value.add_argument("--prefix-cache", action="store_true")
    value.add_argument("--question-batch-size", type=int, default=1)
    value.add_argument("--check-only", action="store_true")
    return value


def main():
    arguments = parser()
    args = arguments.parse_args()
    if args.question_batch_size < 1:
        arguments.error("question batch size must be positive")
    if (args.prefix_cache or args.question_batch_size > 1) and args.numerics != "stable":
        arguments.error("Cache/batch mode requires --numerics stable for reproducible comparisons")
    from .engine import DocisionEngine, preflight

    document = load_document(args.input)
    options = dict(
        max_input_tokens=args.max_input_tokens,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    if args.check_only:
        results = preflight(args.model, document, **options)
    else:
        engine = DocisionEngine(
            args.model,
            device_map=args.device_map,
            dtype=args.dtype,
            numerics=args.numerics,
            **options,
        )
        results = engine.score_document(
            document,
            mode=args.mode,
            prefix_cache=args.prefix_cache,
            batch_size=args.question_batch_size,
        )
    if args.output:
        write_json(args.output, results)
    print(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
