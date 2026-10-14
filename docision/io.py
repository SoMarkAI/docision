"""One document, independent questions. Reference labels never enter the prompt."""

import json
from pathlib import Path
from docision._runtime.inputs import candidate_labels, normalize_questions

DEFAULT_CONTEXT = "Inspect the supplied media carefully."


def normalize_document(value, base_dir="."):
    if not isinstance(value, dict):
        raise ValueError("A document must be a JSON object")
    if value.get("modality", "image") != "image":
        raise ValueError("Docision supports document images only")
    image = value.get("image", value.get("media_path"))
    if not isinstance(image, str) or not image.strip():
        raise ValueError("image or media_path must be a nonempty local path")
    path = (Path(base_dir) / image).resolve()
    if not path.is_file():
        raise ValueError(f"Image does not exist: {path}")
    context = value.get("context", DEFAULT_CONTEXT)
    if not isinstance(context, str):
        raise ValueError("context must be text")
    raw_questions = value.get("questions", [value])
    # Accept the compatible question.instructions / criteria representation.
    converted = []
    for raw in raw_questions:
        if not isinstance(raw, dict):
            raise ValueError("Each question must be an object")
        raw = dict(raw)
        if isinstance(raw.get("question"), dict):
            nested = raw["question"]
            raw["question"] = nested.get("instructions")
            raw["candidates"] = nested.get("criteria")
        converted.append(raw)
    questions = normalize_questions(converted)
    for question in questions:
        if "label" in question and question["label"] not in candidate_labels(
            len(question["candidates"])
        ):
            raise ValueError("Reference label is outside the candidate set")
    return {"image": str(path), "context": context, "questions": questions}


def load_document(path):
    path = Path(path)
    return normalize_document(json.loads(path.read_text(encoding="utf-8")), path.parent)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    temp.replace(path)
