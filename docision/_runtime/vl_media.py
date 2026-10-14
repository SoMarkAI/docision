"""Native Qwen3-VL document image processing."""
from transformers.image_utils import load_image

def prepare_media(processor, media, modality, video_options=None):
    if modality != 'image' or video_options:
        raise ValueError('Docision supports document images only')
    return {'images': [load_image(media)]}
