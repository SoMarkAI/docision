"""Bounded in-memory async jobs shared by HTTP clients and the workbench."""
import base64
import copy
import queue
import threading
import time
import uuid
from .core import question_specs


class APIError(Exception):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status, self.code = status, code


def validate_document(body):
    question_specs(body)
    if not isinstance(body.get('name'), str) or not body['name'].strip():
        raise APIError(400, 'invalid_document', 'name is required')
    if not isinstance(body.get('data'), str):
        raise APIError(400, 'invalid_document', 'data must be base64 text')
    try:
        data = base64.b64decode(body['data'], validate=True)
    except (ValueError, TypeError) as error:
        raise APIError(400, 'invalid_document', 'data is not valid base64') from error
    if not 0 < len(data) <= 25*1024*1024:
        raise APIError(413, 'file_too_large', 'File must be 1 byte to 25 MiB')
    if not isinstance(body.get('prefix_cache', False), bool):
        raise APIError(400, 'invalid_input', 'prefix_cache must be a boolean')


class JobManager:
    """One GPU worker, at most four waiting uploads and 32 retained jobs."""
    def __init__(self, score, inference_lock, ready, ttl=3600, max_jobs=32, queue_size=4):
        self.score, self.inference_lock, self.ready = score, inference_lock, ready
        self.ttl, self.max_jobs = ttl, max_jobs
        self.jobs = {}
        self.lock = threading.RLock()
        self.queue = queue.Queue(maxsize=queue_size)
        self.worker = None
        self.expiry_worker = None

    def _expire(self):
        now = time.time()
        for key, job in list(self.jobs.items()):
            if job['status'] in ('completed', 'failed') and now-job['finished_at'] > self.ttl:
                del self.jobs[key]

    def _expire_loop(self):
        # Separate from GPU work so completed results expire during long inference.
        interval = max(0.01, min(1.0, self.ttl / 2))
        while True:
            time.sleep(interval)
            with self.lock:
                self._expire()
                if not self.jobs:
                    self.expiry_worker = None
                    return

    def _prune(self):
        self._expire()
        while len(self.jobs) >= self.max_jobs:
            done = [j for j in self.jobs.values() if j['status'] in ('completed','failed')]
            if not done:
                raise APIError(429, 'queue_full', 'Too many active jobs; retry later')
            del self.jobs[min(done, key=lambda j:j['finished_at'])['id']]

    def submit(self, body):
        validate_document(body)
        if not self.ready():
            raise APIError(409, 'model_not_loaded', 'Load the local model first')
        with self.lock:
            if self.queue.full():
                raise APIError(429, 'queue_full', 'GPU job queue is full; retry later')
            self._prune()
            job_id = uuid.uuid4().hex
            job = {'id':job_id, 'status':'queued', 'created_at':time.time(),
                   'progress':{'pages_done':0,'pages_total':None,'rows_done':0},
                   'body':copy.deepcopy(body), 'result':None}
            self.jobs[job_id] = job
            self.queue.put_nowait(job_id)
            if self.expiry_worker is None:
                self.expiry_worker = threading.Thread(target=self._expire_loop, daemon=True,
                                                      name='docision-expiry')
                self.expiry_worker.start()
            if self.worker is None or not self.worker.is_alive():
                self.worker = threading.Thread(target=self._run, daemon=True, name='docision-jobs')
                self.worker.start()
            return self._public(job)

    @staticmethod
    def _public(job):
        result = {k:copy.deepcopy(v) for k,v in job.items() if k not in ('body','result')}
        if job['result'] is not None:
            rows = job['result']['rows']
            result['summary'] = {'rows':len(rows),'errors':sum('error' in r for r in rows)}
        return result

    def _get(self, job_id):
        job = self.jobs.get(job_id)
        if job is None or (job['status'] in ('completed','failed') and time.time()-job['finished_at']>self.ttl):
            self.jobs.pop(job_id, None)
            raise APIError(404, 'job_not_found', 'Job not found or expired')
        return job

    def get(self, job_id):
        with self.lock:
            return self._public(self._get(job_id))

    def results(self, job_id, offset=0, limit=50):
        if offset < 0 or not 1 <= limit <= 200:
            raise APIError(400, 'invalid_pagination', 'offset >= 0; limit 1..200')
        with self.lock:
            job = self._get(job_id)
            if job['status'] != 'completed':
                raise APIError(409, 'results_not_ready', f"Job status: {job['status']}")
            result = job['result'];rows = result['rows']
            end = min(len(rows),offset+limit)
            return {'job_id':job_id,'total':len(rows),'offset':offset,
                    'next_offset':end if end<len(rows) else None,
                    'rows':copy.deepcopy(rows[offset:end])}

    def export(self, job_id):
        with self.lock:
            job = self._get(job_id)
            if job['status'] != 'completed':
                raise APIError(409, 'results_not_ready', f"Job status: {job['status']}")
            return copy.deepcopy(job['result']['rows'])

    def delete(self, job_id):
        with self.lock:
            job = self._get(job_id)
            if job['status'] not in ('completed','failed'):
                raise APIError(409, 'job_active', 'Cannot delete an active job')
            del self.jobs[job_id]
        return {'deleted':job_id}

    def _run(self):
        while True:
            job_id = self.queue.get()
            try:
                with self.inference_lock:
                    with self.lock:
                        job = self.jobs[job_id]
                        job.update(status='running', started_at=time.time())
                        body = job.pop('body')
                    def progress(value):
                        with self.lock:
                            job['progress'] = dict(value)
                    try:
                        result = self.score(body, progress=progress)
                        with self.lock:
                            job.update(status='completed',result=result,finished_at=time.time())
                    except Exception as error:
                        with self.lock:
                            job.update(status='failed',error=f'{type(error).__name__}: {error}',finished_at=time.time())
                    finally:
                        body = None
            finally:
                job = result = body = progress = None
                self.queue.task_done()
