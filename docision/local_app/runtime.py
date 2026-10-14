"""Shared image/PDF inference for Python, HTTP, MCP and the local UI."""
import base64
import json
import os
from pathlib import Path
import tempfile
import time
import threading
from functools import wraps
from .core import question_specs
from .cached_inference import score_cached_page, reuse_page_preprocessing


_RENDER_LOCK = threading.RLock()

def serialized(function):
    @wraps(function)
    def run(*args, **kwargs):
        with _RENDER_LOCK:
            return function(*args, **kwargs)
    return run


def default_state_dir():
    return Path(os.environ.get('DOCISION_STATE_DIR', Path.home()/'.cache'/'docision')).expanduser().resolve()


def configure_cache(root):
    root = Path(root)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    for key, child in [('HF_HOME','huggingface'), ('TORCHINDUCTOR_CACHE_DIR','torchinductor'),
                       ('TRITON_CACHE_DIR','triton'), ('TORCH_HOME','torch'),
                       ('PYTORCH_KERNEL_CACHE_PATH','torch-kernels'), ('CUDA_CACHE_PATH','cuda')]:
        os.environ.setdefault(key, str(root/'.cache'/child))
    (root/'.cache').mkdir(parents=True, exist_ok=True)


def load_model(body, engine=None, model=None):
    path = Path(body.get('model', '')).expanduser().resolve()
    if not (path / 'config.json').is_file():
        raise ValueError('请选择包含 config.json 和权重文件的本地完整模型目录。')
    device = body.get('device', 'cuda:0')
    numerics = body.get('numerics', 'native')
    if numerics not in ('native', 'stable'):
        raise ValueError('numerics must be native or stable')
    if not isinstance(device, str) or (device != 'cpu' and not (device.startswith('cuda:') and device[5:].isdigit())):
        raise ValueError('设备应为 cpu 或 cuda:0 等 CUDA 编号。')
    if engine is not None:
        if model == (str(path), device) and getattr(engine, '_numerics_mode', 'native') == numerics:
            return engine, model, {'message': '模型已就绪。'}
        raise ValueError('切换模型、设备或计算模式前，请关闭并重新启动应用。')
    index_path = path / 'model.safetensors.index.json'
    if index_path.is_file():
        weights = set(json.loads(index_path.read_text())['weight_map'].values())
        missing = sorted(name for name in weights if not (path / name).is_file())
        if missing:
            raise ValueError('模型权重不完整，缺少：' + ', '.join(missing))
    elif not (path / 'model.safetensors').is_file():
        raise ValueError('模型目录缺少完整 safetensors 权重。')
    import torch
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    if device.startswith('cuda:'):
        if not torch.cuda.is_available() or int(device[5:]) >= torch.cuda.device_count():
            raise ValueError('所选 NVIDIA GPU 不可用，请检查驱动或选择 CPU。')
        with torch.cuda.device(int(device[5:])):
            if not torch.cuda.is_bf16_supported():
                raise ValueError('所选 GPU 不支持 BF16，请选择支持 BF16 的设备或 CPU。')
    started = time.perf_counter()
    from docision.engine import DocisionEngine
    engine = DocisionEngine(str(path), device_map=device, numerics=numerics,
                            dtype='float32' if device == 'cpu' else 'bfloat16')
    model = (str(path), device)
    return engine, model, {'message': f'模型已就绪（{time.perf_counter()-started:.1f} 秒），文件将在本机处理。'}


@serialized
def score_file(body, engine, root, progress=None, batch_size=None,
               score_page=score_cached_page, preprocessor=reuse_page_preprocessing):
    if engine is None:
        raise ValueError('请先加载本地模型。')
    specs = question_specs(body)
    use_cache = body.get('prefix_cache', False)
    if not isinstance(use_cache, bool):
        raise ValueError('prefix_cache 必须为 true 或 false。')
    numerics = getattr(engine, '_numerics_mode', 'native')
    if use_cache and numerics != 'stable':
        raise ValueError('Prefix cache requires a model loaded with numerics=stable')
    file_started = time.perf_counter()
    cache_pages = []
    name = str(body.get('name', 'document'))
    raw = base64.b64decode(body.get('data', ''), validate=True)
    if not raw or len(raw) > 25 * 1024 * 1024:
        raise ValueError('单个文件大小需在 0–25 MB 之间。')
    from PIL import Image
    from docision.io import DEFAULT_CONTEXT
    rows = []

    def new_row(spec, page):
        return {'file': name, 'page': page, 'question_id': spec['id'],
                'question': spec['question'], 'candidates': list(spec['candidates'].values()),
                'correction': '', 'numerics': numerics, 'prefix_cache_requested': use_cache}

    def failed_page(page, error):
        for spec in specs:
            rows.append(dict(new_row(spec, page), error=f'{type(error).__name__}: {error}'))
    (root / '.cache').mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root / '.cache') as directory:
        folder = Path(directory)
        pdf = None
        try:
            if name.lower().endswith('.pdf'):
                import pypdfium2 as pdfium
                pdf = pdfium.PdfDocument(raw)
                count = len(pdf)
                if not 1 <= count <= 100:
                    raise ValueError('每个 PDF 需为 1–100 页，请拆分后处理。')
            else:
                count = 1
            if progress:
                progress({'pages_done':0,'pages_total':count,'rows_done':0})
            for index in range(count):
                try:
                    image_path = folder / f'{index}.png'
                    if pdf is not None:
                        page = pdf[index]
                        try:
                            width, height = page.get_size()
                            bitmap = page.render(scale=min(2.0, 2048 / max(width, height)))
                            try:
                                bitmap.to_pil().convert('RGB').save(image_path)
                            finally:
                                bitmap.close()
                        finally:
                            page.close()
                    else:
                        import io
                        with Image.open(io.BytesIO(raw)) as source:
                            if source.width * source.height > 40_000_000:
                                raise ValueError('图片超过 4000 万像素，请缩小后重试。')
                            source.convert('RGB').save(image_path)
                except Exception as error:
                    failed_page(index+1, error)
                    if progress:
                        progress({'pages_done':index+1,'pages_total':count,'rows_done':len(rows)})
                    continue
                # A context belongs only to this page; each question clones its KV.
                cache = None
                cache_info = {'page': index+1, 'status': 'disabled' if not use_cache else 'single_question',
                              'prefix_tokens': 0, 'build_seconds': 0.0}
                with preprocessor(engine, enabled=use_cache and len(specs)>1):
                    try:
                        if use_cache and len(specs) > 1:
                            cache_started = time.perf_counter()
                            try:
                                cache = engine.prepare_context(str(image_path), context=DEFAULT_CONTEXT)
                                cache_info.update(status='reused', prefix_tokens=cache.prefix_length)
                            except Exception as error:
                                cache_info.update(status='fallback', reason=f'{type(error).__name__}: {error}')
                            cache_info['build_seconds'] = round(time.perf_counter()-cache_started, 6)
                        cache_pages.append(cache_info)
                        cached_outcomes = score_page(engine, cache, specs, batch_size) if cache is not None else None
                        for question_index, spec in enumerate(specs):
                            row = new_row(spec, index+1)
                            row['cache_status'] = cache_info['status']
                            if 'reason' in cache_info:
                                row['cache_warning'] = cache_info['reason']
                            if cached_outcomes is not None:
                                row.update(cached_outcomes[question_index])
                                rows.append(row)
                                continue
                            started = time.perf_counter()
                            try:
                                row['inference_batch_size'] = 1
                                if cache is None:
                                    result = engine.score_document({'image': str(image_path), 'context': DEFAULT_CONTEXT,
                                                                    'questions': [spec]})
                                else:
                                    result = engine.score_questions(cache, [spec], mode='causal',
                                                                    projection='full', batch_size=1)
                                row['result'] = result[0]
                            except Exception as error:
                                row['error'] = f'{type(error).__name__}: {error}'
                            row['elapsed_seconds'] = round(time.perf_counter() - started, 3)
                            rows.append(row)
                    finally:
                        # Drop page KV before rendering the next page, also on errors.
                        cache = None
                if progress:
                    progress({'pages_done':index+1,'pages_total':count,'rows_done':len(rows)})
        except Exception as error:
            if rows:
                raise
            failed_page(1, error)
        finally:
            if pdf is not None:
                pdf.close()
    return {'rows': rows, 'cache_pages': cache_pages,
            'elapsed_seconds': round(time.perf_counter()-file_started, 6)}
