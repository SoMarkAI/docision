"""Loopback-only document workbench. No hosted services or remote inference."""
import argparse
import base64
import hmac
import json
import os
from pathlib import Path
import secrets
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import webbrowser
from urllib.parse import urlsplit, parse_qs
from .service_api import APIError, JobManager, validate_document
from .token_store import load_token
from .core import TEMPLATES, question_specs, csv_export
from .cached_inference import score_cached_page, reuse_page_preprocessing
CACHE_BATCH_SIZE = None  # Whole page first; CUDA OOM halves the batch size.

from .runtime import default_state_dir, configure_cache, load_model, score_file as runtime_score_file
from importlib.resources import files

ROOT = default_state_dir()
TOKEN = secrets.token_urlsafe(32)
LOCK = threading.Lock()
ENGINE = None
MODEL = None
MAX_BODY = 36 * 1024 * 1024


def load_engine(body):
    global ENGINE, MODEL
    configure_cache(ROOT)
    ENGINE, MODEL, message = load_model(body, ENGINE, MODEL)
    return message


def score_file(body, progress=None):
    return runtime_score_file(body, ENGINE, ROOT, progress, CACHE_BATCH_SIZE,
                              score_cached_page, reuse_page_preprocessing)


JOBS = JobManager(score_file, LOCK, lambda: ENGINE is not None)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def reply(self, value, status=200, content_type='application/json; charset=utf-8'):
        raw = json.dumps(value, ensure_ascii=False, allow_nan=False).encode() if isinstance(value, (dict, list)) else value
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(raw)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(raw)

    def valid_host(self):
        return self.headers.get('Host') == f'127.0.0.1:{self.server.server_port}'

    def do_GET(self):
        if not self.valid_host():
            return self.reply({'error': {'code':'invalid_host','message':'Invalid host'}}, 403)
        if self.path == '/':
            return self.reply(files(__package__).joinpath('assets/index.html').read_bytes(), content_type='text/html; charset=utf-8')
        if not self.authorized():
            return self.reply({'error': {'code':'unauthorized','message':'Use the launcher URL or a valid API token'}}, 403)
        if self.path == '/api/templates':
            return self.reply(TEMPLATES)
        if self.path.startswith('/api/v1/'):
            return self.versioned('GET')
        self.reply({'error': 'Not found'}, 404)

    def authorized(self):
        origin = self.headers.get('Origin')
        supplied = self.headers.get('X-Docision-Token', '') or self.headers.get('Authorization', '').removeprefix('Bearer ')
        return (self.valid_host() and (not origin or origin == f'http://127.0.0.1:{self.server.server_port}')
                and supplied.isascii() and hmac.compare_digest(supplied, TOKEN))

    def do_POST(self):
        if not self.authorized():
            return self.reply({'error': {'code':'unauthorized','message':'Invalid local session'}}, 403)
        if self.path.startswith('/api/v1/'):
            return self.versioned('POST')
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= MAX_BODY:
                return self.reply({'error': '请求过大或为空。'}, 413)
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError('请求必须是 JSON 对象。')
            if self.path == '/api/csv':
                return self.reply({'csv': csv_export(body['rows'])})
            if self.path not in ('/api/load', '/api/score'):
                return self.reply({'error': 'Not found'}, 404)
            if not LOCK.acquire(blocking=False):
                return self.reply({'error': '本机正在处理任务，请稍后重试。'}, 409)
            try:
                result = load_engine(body) if self.path == '/api/load' else score_file(body)
            finally:
                LOCK.release()
            self.reply(result)
        except Exception as error:
            self.reply({'error': f'{type(error).__name__}: {error}'}, 400)

    def do_DELETE(self):
        if not self.authorized():
            return self.reply({'error': {'code':'unauthorized','message':'Invalid local session'}}, 403)
        return self.versioned('DELETE')

    def versioned(self, method):
        try:
            path = urlsplit(self.path).path
            query = parse_qs(urlsplit(self.path).query)
            body = None
            if method == 'POST':
                if self.headers.get('Content-Type','').split(';')[0].strip() != 'application/json':
                    raise APIError(415,'unsupported_media_type','Use application/json')
                length = int(self.headers.get('Content-Length','0'))
                if not 0 < length <= MAX_BODY:
                    raise APIError(413,'request_too_large','Request must be 1 byte to 36 MiB')
                body = json.loads(self.rfile.read(length))
                if not isinstance(body,dict):
                    raise APIError(400,'invalid_input','Request must be a JSON object')
            if method == 'GET' and path == '/api/v1/health':
                return self.reply({'status':'ok','api_version':'1','model_loaded':ENGINE is not None,
                                   'busy':LOCK.locked(),'job_ttl_seconds':JOBS.ttl})
            if method == 'GET' and path == '/api/v1/templates':
                return self.reply(TEMPLATES)
            if method == 'GET' and path == '/api/v1/openapi.json':
                return self.reply(files(__package__).joinpath('assets/openapi.json').read_bytes())
            if method == 'GET' and path == '/api/v1/model':
                return self.reply({'loaded':ENGINE is not None,'model':MODEL[0] if MODEL else None,
                                   'device':MODEL[1] if MODEL else None,
                                   'numerics':getattr(ENGINE, '_numerics_mode', 'native'),'busy':LOCK.locked()})
            if method == 'POST' and path == '/api/v1/jobs':
                return self.reply(JOBS.submit(body),202)
            if method == 'POST' and path == '/api/v1/csv':
                if not isinstance(body.get('rows'),list) or any(not isinstance(r,dict) for r in body['rows']):
                    raise APIError(400,'invalid_input','rows must be a list of result objects')
                return self.reply({'csv':csv_export(body['rows'])})
            if method == 'POST' and path in ('/api/v1/model','/api/v1/score'):
                if path.endswith('/score'):
                    validate_document(body)
                if not LOCK.acquire(blocking=False):
                    raise APIError(409,'busy','GPU is busy; submit an async job or retry later')
                try:
                    result = load_engine(body) if path.endswith('/model') else score_file(body)
                finally:
                    LOCK.release()
                return self.reply(result)
            parts = path.strip('/').split('/')
            if len(parts) in (4,5) and parts[:3] == ['api','v1','jobs']:
                job_id = parts[3]
                if method == 'GET' and len(parts)==4:
                    return self.reply(JOBS.get(job_id))
                if method == 'DELETE' and len(parts)==4:
                    return self.reply(JOBS.delete(job_id))
                if method == 'GET' and len(parts)==5 and parts[4]=='results':
                    return self.reply(JOBS.results(job_id,int(query.get('offset',['0'])[0]),int(query.get('limit',['50'])[0])))
                if method == 'GET' and len(parts)==5 and parts[4]=='export':
                    format = query.get('format',['json'])[0]
                    if format not in ('json','csv'):
                        raise APIError(400,'invalid_format','format must be json or csv')
                    rows = JOBS.export(job_id)
                    raw = (json.dumps(rows,ensure_ascii=False,allow_nan=False,indent=2) if format=='json' else csv_export(rows)).encode('utf-8')
                    if len(raw)>16*1024*1024:
                        raise APIError(413,'export_too_large','Export exceeds 16 MiB; retrieve paginated results')
                    return self.reply(raw,content_type='application/json; charset=utf-8' if format=='json' else 'text/csv; charset=utf-8')
            raise APIError(404,'not_found','Unknown API route')
        except APIError as error:
            self.reply({'error':{'code':error.code,'message':str(error)}},error.status)
        except (ValueError,TypeError,KeyError) as error:
            self.reply({'error':{'code':'invalid_input','message':str(error)}},400)
        except Exception as error:
            self.reply({'error':{'code':'internal_error','message':f'{type(error).__name__}: {error}'}},500)


def main(argv=None, *, open_browser=False):
    global TOKEN, ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--state-dir', default=str(ROOT))
    parser.add_argument('--token-file')
    parser.add_argument('--model', help='Load a local model before serving requests')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--numerics',choices=['native','stable'],default='native')
    args = parser.parse_args(argv)
    ROOT = Path(args.state_dir).expanduser().resolve()
    TOKEN = load_token(args.token_file or ROOT/'.runtime/api-token')
    if args.model:
        load_engine({'model':args.model,'device':args.device,'numerics':args.numerics})
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    url = f'http://127.0.0.1:{server.server_port}/#{TOKEN}'
    print(f'Docision Local\n{url}\n按 Ctrl+C 关闭。', flush=True)
    if open_browser and not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
