"""Compatibility entry point; implementation lives in docision.local_app."""
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from docision.local_app import cached_inference as implementation
sys.modules[__name__] = implementation
