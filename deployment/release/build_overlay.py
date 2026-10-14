"""Build a source-only deployment overlay from an explicit reviewed manifest.

Run from any directory. This archive must be unpacked in a complete Docision
repository; it deliberately excludes weights, environments, user files and logs.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def build(output):
    manifest = json.loads((ROOT / 'release/manifest.json').read_text())
    files = manifest['files']
    if len(files) != len(set(files)):
        raise ValueError('Duplicate manifest entry')
    payload = []
    for name in sorted(files):
        path = ROOT / name
        if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink():
            raise ValueError('Unsafe manifest path: ' + name)
        path.resolve().relative_to(ROOT.resolve())
        payload.append((name, path.read_bytes()))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream, tarfile.open(fileobj=stream, mode='w:gz') as tar:
        for name, data in payload:
            info = tarfile.TarInfo('deployment/' + name)
            info.size = len(data)
            info.mode = 0o755 if name in ('start.sh', 'start.command') else 0o644
            info.mtime = 0
            tar.addfile(info, io.BytesIO(data))
    receipt = {'archive': output.name, 'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
               'files': {n: hashlib.sha256(b).hexdigest() for n, b in payload}}
    output.with_suffix(output.suffix + '.manifest.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'files': len(payload), 'archive': str(output), 'sha256': receipt['sha256']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    build(parser.parse_args().output)
