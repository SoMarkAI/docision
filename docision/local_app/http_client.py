"""Small loopback-only HTTP client; no model/PyTorch imports."""
import json
from pathlib import Path
import re
import urllib.error
import urllib.parse
import urllib.request


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class DocisionClient:
    def __init__(self, base_url, token_file):
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme!='http' or parsed.hostname!='127.0.0.1' or parsed.username or parsed.password or parsed.path not in ('','/') or parsed.query or parsed.fragment:
            raise ValueError('Use http://127.0.0.1:PORT; remote deployments require an SSH tunnel')
        if not parsed.port:
            raise ValueError('API URL requires an explicit port')
        self.base_url = base_url.rstrip('/')
        self.token_file = Path(token_file).expanduser().resolve()
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())

    def request(self, method, path, body=None, raw=False, timeout=30):
        token = self.token_file.read_text().strip()
        if len(token)<32 or not token.isascii() or any(c.isspace() for c in token):
            raise ValueError('Invalid API token file')
        data = None if body is None else json.dumps(body,ensure_ascii=False).encode('utf-8')
        request = urllib.request.Request(self.base_url+'/api/v1/'+path,data=data,method=method,
            headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
        try:
            with self.opener.open(request,timeout=timeout) as response:
                payload = response.read(16*1024*1024+1)
        except urllib.error.HTTPError as error:
            detail = error.read(8192).decode('utf-8',errors='replace')
            try:
                detail = json.loads(detail)['error']
                if isinstance(detail,dict):detail=detail.get('message',str(detail))
            except (ValueError,KeyError,TypeError):
                pass
            raise RuntimeError(f'Docision HTTP {error.code}: {detail}') from None
        if len(payload)>16*1024*1024:
            raise ValueError('Response exceeds 16 MiB; retrieve fewer result rows')
        return payload if raw else json.loads(payload)

    @staticmethod
    def job_path(job_id):
        if not isinstance(job_id,str) or not re.fullmatch('[0-9a-f]{32}',job_id):
            raise ValueError('Invalid job ID')
        return 'jobs/'+job_id
