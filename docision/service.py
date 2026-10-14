"""Bounded HF image-scoring service; spawn replicas, dynamic batches, no vLLM.

The server never imports CUDA in the parent process. Each worker owns one model
and one device. HTTP requests carry images and tasks, never training labels.
"""
import argparse
import base64
import concurrent.futures
import io
import json
import math
import multiprocessing as mp
import queue
import tempfile
import threading
import time
import uuid
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image
from docision._runtime.inputs import normalize_question
from .output import public_output
from .io import DEFAULT_CONTEXT


class ServiceBusy(RuntimeError):
    pass


class InferenceFailure(RuntimeError):
    def __init__(self,details):
        self.details=details
        super().__init__(details['message'])


def create_engine(config, device):
    # Lazy imports are important for Windows spawn and CUDA fork safety.
    import torch
    from .engine import DocisionEngine
    if device.startswith('cuda:'):
        index=int(device.split(':')[1])
        if not torch.cuda.is_available() or index>=torch.cuda.device_count():
            raise ValueError(f'CUDA device unavailable: {device}')
        torch.cuda.set_device(index)
    torch.set_num_threads(config['cpu_threads'])
    return DocisionEngine(config['model'],device_map=device,dtype=config['dtype'],
        numerics=config['numerics'],max_input_tokens=config['max_input_tokens'],
        min_pixels=config['min_pixels'],max_pixels=config['max_pixels'])


def _run_batch(engine, jobs, config, worker_id, outbox, prefix_cache=None):
    try:
        values=(prefix_cache or engine).score_requests([j['row'] for j in jobs],mode=config['mode'],
            projection='full',batch_size=len(jobs))
        if len(values)!=len(jobs):raise RuntimeError('Model returned wrong batch length')
    except Exception as error:
        # Isolate malformed inputs and reduce an OOM batch. Never retry forever,
        # drop requests silently, truncate inputs or change the pixel budget.
        error_type=type(error).__name__;error_message=str(error)
        error.__traceback__=None
        if prefix_cache is not None:prefix_cache.clear()
        if len(jobs)>1:
            if error_type=='OutOfMemoryError':
                import gc
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():torch.cuda.empty_cache()
                except ImportError:pass
            mid=len(jobs)//2
            _run_batch(engine,jobs[:mid],config,worker_id,outbox,prefix_cache)
            _run_batch(engine,jobs[mid:],config,worker_id,outbox,prefix_cache)
        else:
            outbox.put({'kind':'result','job_id':jobs[0]['job_id'],'error':
                {'type':error_type,'message':error_message}})
        return
    for job,value in zip(jobs,values):
        value={**public_output(value),'worker_id':worker_id,'service_batch_size':len(jobs)}
        outbox.put({'kind':'result','job_id':job['job_id'],'value':value})


def _worker(worker_id, device, config, inbox, outbox, stopping, factory):
    try:
        engine=factory(config,device)
        prefix_cache=None
        if config.get('prefix_cache',False):
            from docision._runtime.hf_request_cache import RequestPrefixCache
            prefix_cache=RequestPrefixCache(engine,max_entries=config.get('cache_max_entries',8),
                                            max_tokens=config.get('cache_max_tokens',8192))
        outbox.put({'kind':'ready','worker_id':worker_id,'device':device})
    except Exception as error:
        outbox.put({'kind':'fatal','worker_id':worker_id,'error':str(error)})
        return
    while not stopping.is_set():
        try:first=inbox.get(timeout=.2)
        except queue.Empty:continue
        jobs=[first];deadline=time.monotonic()+config['batch_wait_ms']/1000
        while len(jobs)<config['max_batch_size'] and not stopping.is_set():
            remaining=deadline-time.monotonic()
            if remaining<=0:break
            try:jobs.append(inbox.get(timeout=remaining))
            except queue.Empty:break
        live=[]
        for job in jobs:
            if time.monotonic()>job['deadline']:
                outbox.put({'kind':'result','job_id':job['job_id'],'error':
                    {'type':'TimeoutError','message':'Request expired before inference'}})
            else:live.append(job)
        if live:_run_batch(engine,live,config,worker_id,outbox,prefix_cache)


class ReplicaPool:
    """CPU-side coordinator. One spawn child per explicitly selected device."""
    def __init__(self,config,devices,*,engine_factory=create_engine):
        if not devices or len(set(devices))!=len(devices):
            raise ValueError('devices must be nonempty and distinct')
        if config['max_batch_size']<1 or config['max_pending']<config['max_batch_size'] or config['batch_wait_ms']<0:
            raise ValueError('Invalid batch size, queue capacity or batch wait')
        self.config=dict(config);self.devices=list(devices);self.context=mp.get_context('spawn')
        self.inbox=self.context.Queue();self.outbox=self.context.Queue();self.stopping=self.context.Event()
        self.lock=threading.RLock();self.pending={};self.workers={};self.broken=None;self.closed=False
        self.processes=[self.context.Process(target=_worker,args=(i,d,self.config,
            self.inbox,self.outbox,self.stopping,engine_factory),name=f'Docision-hf-{i}') for i,d in enumerate(devices)]
        try:
            for process in self.processes:process.start()
        except BaseException:
            self.close();raise
        self.collector=threading.Thread(target=self._collect,name='Docision-results',daemon=True)
        self.collector.start()

    def _break(self,message):
        with self.lock:
            if self.broken:return
            self.broken=message;self.stopping.set();pending=list(self.pending.values());self.pending.clear()
        for future in pending:
            if not future.done():future.set_exception(RuntimeError(message))

    def _collect(self):
        while not self.closed:
            try:item=self.outbox.get(timeout=.2)
            except queue.Empty:
                if not self.stopping.is_set():
                    dead=[p.name for p in self.processes if p.exitcode is not None]
                    if dead:self._break('Model worker exited: '+', '.join(dead))
                continue
            if item['kind']=='ready':
                with self.lock:self.workers[item['worker_id']]=item['device']
            elif item['kind']=='fatal':self._break('Model worker failed to load: '+item['error'])
            else:
                with self.lock:future=self.pending.pop(item['job_id'],None)
                if future is None or future.done():continue
                if 'error' in item:future.set_exception(InferenceFailure(item['error']))
                else:future.set_result(item['value'])

    @property
    def ready(self):
        with self.lock:return not self.closed and not self.broken and len(self.workers)==len(self.devices)

    def health(self):
        with self.lock:return {'status':'ready' if self.ready else 'failed' if self.broken else 'loading',
            'backend':'transformers','workers':dict(self.workers),'expected_workers':len(self.devices),
            'pending_questions':len(self.pending),'max_pending':self.config['max_pending'],
            'max_batch_size':self.config['max_batch_size'],'batch_wait_ms':self.config['batch_wait_ms'],
            'numerics':self.config['numerics'],'mode':self.config['mode'],
            'prefix_cache':self.config.get('prefix_cache',False),'error':self.broken}

    def wait_ready(self,timeout=900):
        deadline=time.monotonic()+timeout
        while not self.ready:
            if self.broken:raise RuntimeError(self.broken)
            if time.monotonic()>deadline:raise TimeoutError('Model workers did not become ready')
            time.sleep(.02)

    def submit_many(self,rows,timeout):
        if not rows:raise ValueError('Empty request')
        futures=[];jobs=[]
        with self.lock:
            if not self.ready:raise ServiceBusy('Model workers are not ready')
            if len(self.pending)+len(rows)>self.config['max_pending']:
                raise ServiceBusy('Question queue is full; retry later')
            for row in rows:
                jid=uuid.uuid4().hex;future=concurrent.futures.Future();self.pending[jid]=future
                futures.append(future);jobs.append({'job_id':jid,'row':row,'deadline':time.monotonic()+timeout})
            for job in jobs:self.inbox.put(job)
        return futures

    def close(self):
        with self.lock:
            if self.closed:return
            self.closed=True;self.stopping.set();pending=list(self.pending.values());self.pending.clear()
        for process in self.processes:
            if process.pid is not None:
                process.join(timeout=5)
                if process.is_alive():process.terminate();process.join(timeout=5)
        for future in pending:
            if not future.done():future.set_exception(RuntimeError('Service stopped'))
        collector=getattr(self,'collector',None)
        if collector and collector is not threading.current_thread():collector.join(timeout=1)
        # Feeder threads must not block shutdown with queued work after a crash.
        for q in [self.inbox,self.outbox]:q.cancel_join_thread();q.close()


def _document_image(value,directory,index,media_root,max_image_bytes):
    if not isinstance(value,str) or not value:raise ValueError('image must be a path or image data URL')
    if value.startswith('data:'):
        header,separator,encoded=value.partition(',')
        if not separator or header not in ['data:image/png;base64','data:image/jpeg;base64','data:image/webp;base64']:
            raise ValueError('Use a PNG, JPEG or WebP base64 image data URL')
        try:data=base64.b64decode(encoded,validate=True)
        except (ValueError,base64.binascii.Error) as error:raise ValueError('Invalid image base64') from error
        if len(data)>max_image_bytes:raise ValueError('Image upload exceeds byte limit')
        source=io.BytesIO(data)
    else:
        if media_root is None:raise ValueError('Local image paths require --media-root; otherwise send a data URL')
        root=Path(media_root).resolve();source=(root/value).resolve()
        if not source.is_relative_to(root) or not source.is_file():raise ValueError('Image must be a file under media-root')
        if source.stat().st_size>max_image_bytes:raise ValueError('Image exceeds byte limit')
    target=Path(directory)/f'{index}.png'
    with warnings.catch_warnings():
        warnings.simplefilter('error',Image.DecompressionBombWarning)
        try:
            with Image.open(source) as image:
                if image.width*image.height>40_000_000:raise ValueError('Image exceeds 40 megapixels')
                image.convert('RGB').save(target)
        except (OSError,Image.DecompressionBombWarning,Image.DecompressionBombError) as error:
            raise ValueError('Invalid or oversized image') from error
    return str(target)


def prepare_payload(body,directory,*,media_root=None,max_image_bytes=20*1024**2,max_questions=128):
    if not isinstance(body,dict):raise ValueError('Request must be a JSON object')
    docs=body.get('documents',[body])
    if not isinstance(docs,list) or not docs:raise ValueError('documents must be a nonempty list')
    flat=[];groups=[]
    for i,doc in enumerate(docs):
        if not isinstance(doc,dict) or doc.get('modality','image')!='image':raise ValueError('Each document must be an image task')
        context=doc.get('context',DEFAULT_CONTEXT)
        if not isinstance(context,str):raise ValueError('context must be text')
        qs=doc.get('questions',[doc])
        if not isinstance(qs,list) or not qs:raise ValueError('questions must be a nonempty list')
        start=len(flat)
        # Validate tasks before decoding an image. Reference labels are discarded.
        for q in qs:
            if not isinstance(q,dict):raise ValueError('Each question must be an object')
            text=q.get('question');candidates=q.get('candidates')
            if isinstance(text,dict):candidates=text.get('criteria');text=text.get('instructions')
            normalized=normalize_question({'question':text,'candidates':candidates})
            if len(text)>32000 or sum(map(len,normalized['candidates']))>64000:raise ValueError('Task text exceeds character limit')
            flat.append({'id':q.get('id'),'question':normalized['question'],'candidates':normalized['candidates'],'context':context})
            if len(flat)>max_questions:raise ValueError('Too many questions in request')
        image=_document_image(doc.get('image',doc.get('media_path')),directory,i,media_root,max_image_bytes)
        for row in flat[start:]:row['image']=image
        groups.append({'id':doc.get('id'),'start':start,'end':len(flat)})
    return flat,groups,'documents' in body


class ScoringServer(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self,address,pool,*,media_root=None,request_timeout=300,max_body_bytes=32*1024**2):
        self.pool=pool;self.media_root=media_root;self.request_timeout=request_timeout;self.max_body_bytes=max_body_bytes
        self.request_slots=threading.BoundedSemaphore(min(32,pool.config['max_pending']))
        super().__init__(address,ScoringHandler)


class ScoringHandler(BaseHTTPRequestHandler):
    def _send(self,status,body):
        data=json.dumps(body,ensure_ascii=False,allow_nan=False).encode()
        try:
            self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        except (BrokenPipeError,ConnectionResetError):pass
    def do_GET(self):
        if self.path not in ['/health','/healthz']:self._send(404,{'error':'Unknown endpoint'});return
        self._send(200 if self.server.pool.ready else 503,self.server.pool.health())
    def do_POST(self):
        if self.path!='/v1/score':self._send(404,{'error':'Unknown endpoint'});return
        if not self.server.request_slots.acquire(blocking=False):
            self._send(503,{'error':'Too many in-flight requests; retry later'});return
        temporary=None;futures=[]
        try:
            if self.headers.get('Transfer-Encoding'):raise ValueError('Chunked uploads are not supported')
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=self.server.max_body_bytes:self._send(413,{'error':'Body exceeds configured limit or has no length'});return
            self.connection.settimeout(30)
            data=self.rfile.read(length)
            if len(data)!=length:raise ValueError('Incomplete request body')
            body=json.loads(data);temporary=tempfile.TemporaryDirectory(prefix='Docision-request-')
            rows,groups,multi=prepare_payload(body,temporary.name,media_root=self.server.media_root,
                max_questions=self.server.pool.config['max_pending'])
            started=time.monotonic();futures=self.server.pool.submit_many(rows,self.server.request_timeout)
            # Preserve input order regardless of worker completion order.
            results=[]
            for future in futures:
                remaining=self.server.request_timeout-(time.monotonic()-started)
                results.append(future.result(timeout=max(0,remaining)))
            docs=[{'id':g['id'],'questions':results[g['start']:g['end']]} for g in groups]
            value={'documents':docs} if multi else docs[0]
            self._send(200,{'backend':'transformers','elapsed_seconds':time.monotonic()-started,**value})
        except ServiceBusy as error:self._send(503,{'error':str(error)})
        except concurrent.futures.TimeoutError:self._send(504,{'error':'Request timed out; queued jobs have an expiry deadline'})
        except InferenceFailure as error:self._send(422 if error.details['type']=='ValueError' else 500,{'error':error.details})
        except (ValueError,TypeError,KeyError) as error:self._send(400,{'error':str(error)})
        except Exception as error:self._send(500,{'error':str(error)})
        finally:
            if temporary:
                if not futures or all(f.done() for f in futures):temporary.cleanup()
                else:
                    # A timed-out/disconnected client must not remove a file
                    # that an in-flight GPU worker still needs.
                    lock=threading.Lock();state={'cleaned':False}
                    def cleanup(_):
                        with lock:
                            if not state['cleaned'] and all(f.done() for f in futures):
                                state['cleaned']=True;temporary.cleanup()
                    for f in futures:f.add_done_callback(cleanup)
            self.server.request_slots.release()


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True);p.add_argument('--devices',default='cuda:0',help='Comma-separated devices: cuda:0,cuda:1; one replica each')
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8000)
    p.add_argument('--max-batch-size',type=int,default=1);p.add_argument('--batch-wait-ms',type=float,default=10)
    p.add_argument('--max-pending',type=int,default=128);p.add_argument('--request-timeout',type=float,default=300)
    p.add_argument('--numerics',choices=['native','stable'],default='native');p.add_argument('--mode',choices=['causal','isolated'],default='causal')
    p.add_argument('--dtype',choices=['bfloat16','float32'],default='bfloat16');p.add_argument('--cpu-threads',type=int,default=2)
    p.add_argument('--max-input-tokens',type=int,default=4000);p.add_argument('--min-pixels',type=int,default=3136);p.add_argument('--max-pixels',type=int,default=501760)
    p.add_argument('--prefix-cache',action='store_true',help='Worker-local image/context KV cache')
    p.add_argument('--cache-max-entries',type=int,default=8)
    p.add_argument('--cache-max-tokens',type=int,default=8192)
    p.add_argument('--media-root',type=Path,help='Optional root for trusted local image paths; data URLs work without it')
    return p


def main():
    mp.freeze_support()
    p=parser();a=p.parse_args();devices=[x.strip() for x in a.devices.split(',')]
    if (a.prefix_cache or a.max_batch_size>1) and a.numerics!='stable':p.error('Cache/batch mode requires --numerics stable')
    if any(x!='cpu' and not (x.startswith('cuda:') and x[5:].isdigit()) for x in devices):p.error('Use cpu or explicit cuda:N devices')
    if a.cpu_threads<1 or not math.isfinite(a.request_timeout) or a.request_timeout<=0 or a.max_batch_size<1 or a.max_pending<a.max_batch_size or not math.isfinite(a.batch_wait_ms) or a.batch_wait_ms<0:p.error('Invalid worker/batch/timeout configuration')
    if min(a.max_input_tokens,a.min_pixels,a.max_pixels)<1 or a.min_pixels>a.max_pixels:
        p.error('Input limits must be positive; min-pixels <= max-pixels')
    if min(a.cache_max_entries,a.cache_max_tokens)<1:p.error('Cache limits must be positive')
    config={k:getattr(a,k) for k in ['model','dtype','numerics','mode','cpu_threads','max_input_tokens','min_pixels','max_pixels','max_batch_size','max_pending','batch_wait_ms','prefix_cache','cache_max_entries','cache_max_tokens']}
    pool=ReplicaPool(config,devices)
    try:
        server=ScoringServer((a.host,a.port),pool,media_root=a.media_root,request_timeout=a.request_timeout)
        print(json.dumps({'listening':f'http://{a.host}:{a.port}','devices':devices,'backend':'transformers','numerics':a.numerics,'status':'loading'}),flush=True)
        try:server.serve_forever(poll_interval=.2)
        finally:server.server_close()
    except KeyboardInterrupt:pass
    finally:pool.close()


if __name__=='__main__':main()
