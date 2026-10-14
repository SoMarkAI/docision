"""Install only the MCP bridge dependencies, isolated from the model runtime."""
import os
from pathlib import Path
import subprocess
import sys
from bootstrap import ensure_environment

ROOT = Path(__file__).resolve().parent
if __name__ == '__main__':
    if sys.version_info < (3,10):
        raise SystemExit('Python 3.10+ required')
    env = ROOT/'.mcp-venv'
    python = env/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    ensure_environment(env,python)
    environment = dict(os.environ,PIP_CACHE_DIR=str(ROOT/'.cache/pip'))
    subprocess.run([str(python),'-m','pip','install','--disable-pip-version-check','-r',str(ROOT/'requirements-mcp.txt')],env=environment,check=True)
    subprocess.run([str(python),'-m','pip','check'],check=True)
    print(f'MCP Python: {python}')
