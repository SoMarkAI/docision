"""Create an isolated environment once, then open the local workbench."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import venv
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent


def has_pip(python):
    if not python.is_file():
        return False
    result = subprocess.run([str(python), '-I', '-m', 'pip', '--version'],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def ensure_environment(env, python):
    if has_pip(python):
        return
    print('正在创建应用独立环境……', flush=True)
    if not python.exists() and importlib.util.find_spec('ensurepip') is not None:
        try:
            venv.EnvBuilder(with_pip=True).create(env)
        except subprocess.CalledProcessError:
            # Debian/Ubuntu may ship Python without ensurepip.
            pass
    if has_pip(python):
        return
    print('当前 Python 缺少可用的 venv/pip，正在准备项目内的 virtualenv……', flush=True)
    cache = ROOT / '.cache' / 'bootstrap'
    cache.mkdir(parents=True, exist_ok=True)
    version = f'{sys.version_info.major}.{sys.version_info.minor}'
    archive = cache / f'virtualenv-{version}.pyz'
    if not archive.is_file():
        url = f'https://bootstrap.pypa.io/virtualenv/{version}/virtualenv.pyz'
        partial = archive.with_suffix('.part')
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                partial.write_bytes(response.read())
            if not zipfile.is_zipfile(partial):
                raise RuntimeError('virtualenv 下载内容不是有效的 zipapp。')
            partial.replace(archive)
        finally:
            partial.unlink(missing_ok=True)
    subprocess.run([sys.executable, str(archive), '--app-data', str(ROOT / '.cache' / 'virtualenv'),
                    '--no-periodic-update', str(env)], check=True)
    if not has_pip(python):
        raise RuntimeError('独立环境未能初始化 pip，请检查 Python 安装和目录权限。')


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--torch-backend', choices=['pypi', 'cu128', 'cu130', 'cpu'],
                        default='pypi')
    parser.add_argument('--repair', action='store_true')
    options, app_args = parser.parse_known_args()
    if sys.version_info < (3, 10):
        raise SystemExit('请安装 Python 3.10 或更高版本，再运行启动器。')
    if not (ROOT.parent / 'docision' / 'engine.py').is_file():
        raise SystemExit('请将 deployment 保留在完整 Docision 仓库内。')
    env = ROOT / '.venv'
    python = env / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    stamp = env / '.docision-installed'
    requirements = ROOT / 'requirements.txt'
    expected = json.dumps({'requirements': requirements.read_text(), 'backend': options.torch_backend,
                           'bootstrap': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, sort_keys=True)
    ensure_environment(env, python)
    if options.repair or not stamp.exists() or stamp.read_text() != expected:
        print('首次启动需要联网安装依赖。后续推理只读取本地文件。', flush=True)
        os.environ['PIP_CACHE_DIR'] = str(ROOT / '.cache' / 'pip')
        pip = [str(python), '-m', 'pip', 'install', '--disable-pip-version-check']
        if sys.platform == 'darwin' or options.torch_backend == 'pypi':
            subprocess.run(pip + ['torch==2.11.0', 'torchvision==0.26.0'], check=True)
        else:
            subprocess.run(pip + ['--no-deps', 'torch==2.11.0+' + options.torch_backend,
                                  'torchvision==0.26.0+' + options.torch_backend, '--index-url',
                                  'https://download.pytorch.org/whl/' + options.torch_backend], check=True)
        subprocess.run(pip + ['-r', str(requirements)], check=True)
        subprocess.run([str(python), '-m', 'pip', 'check'], check=True)
        stamp.write_text(expected)
    envvars = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    return subprocess.call([str(python), '-B', str(ROOT / 'app.py'), *app_args], cwd=ROOT, env=envvars)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (subprocess.CalledProcessError, OSError, RuntimeError) as error:
        raise SystemExit(f'启动未完成，请检查 Python、venv 和网络环境后重试：{error}')
