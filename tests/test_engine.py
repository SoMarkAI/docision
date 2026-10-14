from types import SimpleNamespace
import pytest
from docision import DocisionEngine
from docision._runtime.families import family

def test_document_engine_rejects_other_media_before_loading():
    engine = DocisionEngine.__new__(DocisionEngine)
    for modality in ('video', 'audio', 'audio_video'):
        with pytest.raises(ValueError, match='document images only'):
            engine.prepare('unused.png', 'Question?', ['Yes', 'No'], modality=modality)

def test_runtime_rejects_non_document_model_family():
    with pytest.raises(ValueError, match='expected Qwen3-VL'):
        family(SimpleNamespace(model_type='qwen3_omni_moe'))
