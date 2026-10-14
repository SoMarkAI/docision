#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
for candidate in python3.12 python3.11 python3.13 python3.10 python3; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
    exec "$candidate" -B bootstrap.py "$@"
  fi
done
printf '%s\n' '请先安装 Python 3.10 或更高版本。'
exit 1
