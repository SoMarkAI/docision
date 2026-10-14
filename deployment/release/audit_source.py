"""Read-only source/history checks. Reports locations, never matched values.

Pattern matches need review; a clean report is not a proof that all secrets or
third-party material have been identified. --repo scans tracked HEAD and all
locally reachable history; it never fetches or changes Git state.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    'credential_pattern': r'(?:hf_[A-Za-z0-9]{24,}|glpat-[A-Za-z0-9_-]{15,}|sk-[A-Za-z0-9_-]{24,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)',
    'private_absolute_path': r'/(?:Users|home|data[0-9]*)/(?!user(?:/|\b)|example(?:/|\b))[^\s\"<>]+',
    'service_url_review': r'https?://(?:gitlab\.[A-Za-z0-9.-]+|[A-Za-z0-9-]+\.feishu\.cn)(?:/|$)',
}


def scan(data, path, scope, findings):
    if b'\0' in data:
        return
    for number, line in enumerate(data.decode('utf-8', errors='replace').splitlines(), 1):
        for kind, expression in PATTERNS.items():
            if re.search(expression, line):
                findings.append({'scope': scope, 'path': path, 'line': number, 'kind': kind})


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    findings = []
    files = json.loads((ROOT / 'release/manifest.json').read_text())['files']
    digests = {}
    for name in files:
        data = (ROOT / name).read_bytes()
        digests[name] = hashlib.sha256(data).hexdigest()
        scan(data, 'deployment/' + name, 'overlay', findings)
    report = {'overlay_files': digests, 'findings': findings,
              'limitations': ['Pattern-based text scan; binary artwork and sample data need provenance review.',
                             'History scope is locally reachable Git refs, not remote deleted/unreachable objects.',
                             'License metadata does not establish ownership of training data or branding.']}
    if args.repo:
        report['head'] = git(args.repo, 'rev-parse', 'HEAD').decode().strip()
        tracked = git(args.repo, 'ls-tree', '-r', '-z', 'HEAD').split(b'\0')
        for entry in tracked:
            if not entry:
                continue
            meta, path = entry.split(b'\t', 1)
            mode, kind, oid = meta.decode().split()
            if kind == 'blob':
                scan(git(args.repo, 'cat-file', 'blob', oid), path.decode(), 'HEAD', findings)
        objects = git(args.repo, 'rev-list', '--objects', '--all').decode().splitlines()
        count = 0
        for item in objects:
            oid, _, path = item.partition(' ')
            if git(args.repo, 'cat-file', '-t', oid).strip() != b'blob':
                continue
            data = git(args.repo, 'cat-file', 'blob', oid)
            scan(data, path, 'history:' + oid[:12], findings)
            count += 1
        report['history_blobs_scanned'] = count
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    counts = {}
    for item in findings:
        key = item['scope'].split(':')[0] + ':' + item['kind']
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps({'report': str(args.output), 'counts': counts}))


if __name__ == '__main__':
    main()
