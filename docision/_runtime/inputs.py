"""Shared, model-independent validation for CLI and scoring inputs."""


def candidate_labels(count):
    labels = []
    for i in range(count):
        label, value = '', i + 1
        while value:
            value, remainder = divmod(value - 1, 26)
            label = chr(65 + remainder) + label
        labels.append(label)
    return labels


def normalize_question(question):
    if not isinstance(question, dict):
        raise ValueError('Each question must be an object')
    text = question.get('question')
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Question must be nonempty text')
    candidates = question.get('candidates')
    if isinstance(candidates, dict):
        if list(candidates) != candidate_labels(len(candidates)):
            raise ValueError('Candidate mapping must use ordered A, B, ... labels')
        candidates = list(candidates.values())
    if not isinstance(candidates, list) or not 2 <= len(candidates) <= 128:
        raise ValueError('Require a list of 2..128 candidates')
    if not all(isinstance(value, str) and value.strip() for value in candidates):
        raise ValueError('Candidates must be nonempty strings')
    if len(set(candidates)) != len(candidates):
        raise ValueError('Candidates must be distinct nonempty strings')
    return {**question, 'candidates': list(candidates)}


def normalize_questions(questions):
    if not isinstance(questions, list) or not questions:
        raise ValueError('questions must be a nonempty list')
    return [normalize_question(question) for question in questions]
