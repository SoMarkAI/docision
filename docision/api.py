"""Stable local document interface. No HTTP service required."""
import base64
import gc
import json
from pathlib import Path
import threading
from .local_app.core import csv_export, question_specs
from .local_app.runtime import configure_cache, default_state_dir, load_model, score_file


class Docision:
    """Load a full local Qwen3-VL checkpoint and inspect images or PDFs.

    Each instance owns its engine and serializes its own calls. inspect() returns
    the same page/question rows as HTTP/MCP, including per-row errors. Candidate
    probabilities are relative scores, not calibrated confidence estimates.
    """
    def __init__(self, model, *, device='cuda:0', state_dir=None, numerics='native'):
        self._lock = threading.RLock()
        self._engine = None
        self._state_dir = Path(state_dir).expanduser().resolve() if state_dir else default_state_dir()
        configure_cache(self._state_dir)
        self._engine, self._model, _ = load_model({'model': str(model), 'device': device, 'numerics': numerics})

    def inspect(self, file_path, *, questions, prefix_cache=False):
        """Return page/question rows; validate all questions before reading input.

        questions: [{"question": "...", "candidates": ["yes", "no"]}, ...]
        """
        question_specs({'questions': questions})
        if not isinstance(prefix_cache, bool):
            raise ValueError('prefix_cache must be bool')
        path = Path(file_path).expanduser().resolve(strict=True)
        if not path.is_file() or path.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.webp', '.pdf'):
            raise ValueError('Expected a PNG/JPEG/WebP/PDF file')
        with self._lock:
            if self._engine is None:
                raise RuntimeError('Docision instance is closed')
            with path.open('rb') as stream:
                raw = stream.read(25*1024*1024+1)
            if not 0 < len(raw) <= 25*1024*1024:
                raise ValueError('File must be 1 byte to 25 MiB')
            body = {'name': path.name, 'data': base64.b64encode(raw).decode(),
                    'questions': questions, 'prefix_cache': prefix_cache}
            return score_file(body, self._engine, self._state_dir)['rows']

    @staticmethod
    def export(rows, path):
        """Write JSON or CSV to a new file; never silently overwrite an export."""
        path = Path(path).expanduser()
        if path.suffix.lower() not in ('.json', '.csv'):
            raise ValueError('Export path must end in .json or .csv')
        if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
            raise ValueError('rows must be a list of dictionaries')
        text = (json.dumps(rows, ensure_ascii=False, allow_nan=False, indent=2)
                if path.suffix.lower()=='.json' else csv_export(rows))
        with path.open('x', encoding='utf-8', newline='') as stream:
            stream.write(text)
        return path.resolve()

    def close(self):
        """Release this instance's model references; repeated calls are safe."""
        with self._lock:
            self._engine = None
            gc.collect()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
