"""Official chat rendering; candidate spans tracked after image expansion."""
from transformers import AutoProcessor
from .inputs import candidate_labels, normalize_question

class PromptBuilder:
    def __init__(self, model):
        self.model_path = model
        self.processor = AutoProcessor.from_pretrained(model, local_files_only=True)
        self.tokenizer = self.processor.tokenizer

    labels = staticmethod(candidate_labels)

    def build(self, image, context, question, candidates, mode='causal', debug=False,
              *, modality='image', video_options=None):
        if mode not in ('isolated', 'causal', 'stock'):
            raise ValueError('Unknown attention mode')
        if not isinstance(context, str) or not isinstance(question, str) or not question.strip():
            raise ValueError('Context must be text and question must be nonempty text')
        candidates = normalize_question({'question': question, 'candidates': candidates})['candidates']
        labels = self.labels(len(candidates))
        common = (context + '\nQuestion: ' + question +
                  '\nChoose exactly one candidate and answer with its label.\nCandidates:\n')
        chunks = [f'[{label}] {text}\n' for label, text in zip(labels, candidates)]
        message = common + ''.join(chunks)
        if modality != 'image' or video_options:
            raise ValueError('Docision supports document images only')
        media_item = {'type': 'image', 'image': image}
        messages = [{'role': 'user', 'content': [media_item, {'type': 'text', 'text': message}]}]
        rendered = self.processor.apply_chat_template(messages, tokenize=False,
                                                       add_generation_prompt=True) + 'Answer:\n'
        # Guard against duplicated boundary strings or embedded chat control tokens.
        if rendered.count(message) != 1:
            raise ValueError('Ambiguous prompt boundary')
        for field in [context, question, *candidates]:
            if any(t in field for t in self.tokenizer.all_special_tokens):
                raise ValueError('Special tokens are not allowed inside task text')
        encoded = self.tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        raw, offsets = encoded['input_ids'], encoded['offset_mapping']
        # Stop before question-dependent text, backing off a crossing BPE token.
        context_end = rendered.index(message) + len(context)
        prefix_length = next((i for i, (a, b) in enumerate(offsets) if b > context_end), len(raw))
        char_start = rendered.index(message) + len(common)
        boundaries = [char_start]
        for chunk in chunks:
            boundaries.append(boundaries[-1] + len(chunk))
        token_boundaries = []
        for boundary in boundaries:
            matches = [i for i, (a, b) in enumerate(offsets) if a == boundary and b > a]
            if not matches:
                raise ValueError(f'Candidate boundary crosses tokenizer token at {boundary}')
            token_boundaries.append(matches[0])
        spans = list(map(list, zip(token_boundaries[:-1], token_boundaries[1:])))
        label_ids = []
        for label in labels:
            joint = self.tokenizer.encode(rendered + label, add_special_tokens=False)
            if len(joint) != len(raw) + 1 or joint[:-1] != raw:
                raise ValueError(f'Label {label!r} is not one token after this Answer: prefix')
            label_ids.append(joint[-1])
        if len(set(label_ids)) != len(label_ids):
            raise ValueError('Label tokens must be distinct')
        return {"text": rendered, "messages": messages, "raw_ids": raw, "spans": spans, "labels": labels, "label_ids": label_ids, "prefix_length": prefix_length}
