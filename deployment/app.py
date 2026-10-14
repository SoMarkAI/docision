"""Compatibility entry point; implementation lives in docision.local_app."""
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault('DOCISION_STATE_DIR', str(Path(__file__).resolve().parent))
from docision.local_app import server as implementation
if __name__ == '__main__':
    implementation.main(open_browser=True)
else:
    sys.modules[__name__] = implementation
