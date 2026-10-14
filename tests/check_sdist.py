"""Check an sdist's runnable examples, README resources and private-file boundary.

Usage: python tests/check_sdist.py path/to/docision-VERSION.tar.gz
Reads the archive without extracting or executing its contents.
"""
import argparse
import hashlib
import json
from pathlib import PurePosixPath
import re
import tarfile
from urllib.parse import unquote, urlsplit


def check(path):
    with tarfile.open(path, 'r:gz') as archive:
        members = archive.getmembers()
        roots = {PurePosixPath(m.name).parts[0] for m in members}
        assert len(roots) == 1, 'Expected one distribution root'
        files = {}
        for member in members:
            parts = PurePosixPath(member.name).parts
            assert not member.name.startswith('/') and '..' not in parts, member.name
            assert member.isdir() or member.isfile(), 'Unexpected link or special file: ' + member.name
            if member.isfile():
                name = str(PurePosixPath(*parts[1:]))
                assert name not in files, 'Duplicate archive path: ' + name
                files[name] = archive.extractfile(member).read()
        for name in ('demo_docision.py', 'docision/cli.py', 'LICENSE', 'NOTICE',
                     'examples/service_client.py', 'examples/readme.md',
                     'examples/recorded-output.json',
                     'docision/local_app/assets/index.html', 'docision/local_app/assets/openapi.json'):
            assert name in files, 'Missing release file: ' + name
        for name in ('README.md', 'README.zh-CN.md'):
            text = files[name].decode('utf-8')
            references = re.findall(r'(?:src|href)=["\']([^"\']+)', text)
            references += re.findall(r'\]\(([^)]+)\)', text)
            for reference in references:
                parsed = urlsplit(reference)
                if parsed.scheme or parsed.netloc or not parsed.path:
                    continue
                target = unquote(parsed.path)
                assert target in files, f'{name} references missing file: {target}'
        for name in ('single.json', 'multiple.json'):
            document = json.loads(files['examples/' + name])
            image = 'examples/' + document['image']
            assert image in files, f'{name} references missing image: {image}'
        assert hashlib.sha256(files['examples/validation-table.jpg']).hexdigest() == 'c5bc6f4e69f7b8bf446c1ae35e2e74904bc3dac2e2dcf58e2c7dba67c53fb595', 'Example media changed'
        forbidden = {'.git', '.venv', '.mcp-venv', '.runtime', '.cache', 'validation',
                     'inputs', 'exports', 'profile_output', 'release-output', '__pycache__'}
        for name in files:
            assert not (set(PurePosixPath(name).parts) & forbidden), 'Private runtime file: ' + name
            assert not name.endswith(('.safetensors', '.ckpt', '.pem', '.key', '.log', '.pyc')), name
            assert not PurePosixPath(name).name.startswith('.env'), name
    print(json.dumps({'status': 'SDIST_CONTENT_PASS', 'files': len(files),
                      'bilingual_readme_links': True, 'example_media_hash': True,
                      'private_runtime_excluded': True}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive')
    check(parser.parse_args().archive)
