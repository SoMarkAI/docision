"""Persistent local API credential shared with configured local clients."""
import os
from pathlib import Path
import secrets


def load_token(path):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError('Token file must not be a symlink')
    try:
        fd = os.open(path, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
    except FileExistsError:
        pass
    else:
        with os.fdopen(fd,'w') as f:
            f.write(secrets.token_urlsafe(32)+'\n')
    if os.name != 'nt':
        os.chmod(path,0o600)
    token = path.read_text().strip()
    if len(token)<32 or not token.isascii() or any(c.isspace() for c in token):
        raise ValueError('Token must contain at least 32 non-whitespace characters')
    return token
