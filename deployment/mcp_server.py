"""Compatibility entry point; implementation lives in docision.local_app."""
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from docision.local_app import mcp_server as implementation
if __name__ == '__main__':
    implementation.main()
else:
    sys.modules[__name__] = implementation
