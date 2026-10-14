"""Small model-family boundary; native Transformers modules remain unchanged."""
from transformers import AutoConfig

VL = 'qwen3_vl'

def family(config):
    if config.model_type != VL:
        raise ValueError(f'Unsupported model_type {config.model_type!r}; expected Qwen3-VL')
    return VL


def detect(path):
    return family(AutoConfig.from_pretrained(path, local_files_only=True))


def text_model(model):
    return model.model.language_model


def multimodal_model(model):
    return model.model


def check_modality(kind, modality):
    if kind != VL or modality != 'image':
        raise ValueError('Docision supports document images only; unsupported media is never silently discarded')


def positions(model, batch):
    device = model.get_input_embeddings().weight.device
    def get(key):
        value = batch.get(key)
        return value.to(device) if hasattr(value, 'to') else value
    import torch
    types = get('mm_token_type_ids')
    if types is None:
        types = torch.zeros_like(get('input_ids'))
    return model.model.get_rope_index(
        get('input_ids'), types, image_grid_thw=get('image_grid_thw'),
        attention_mask=get('attention_mask'))[0]
