"""Spawn/HTTP routing and bounded-queue tests; fake engine, no accuracy claims."""
import base64
import concurrent.futures
import io
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from PIL import Image
from docision.service import ReplicaPool, ScoringServer, ServiceBusy, prepare_payload


class FakeEngine:
    def score_requests(self,rows,**kwargs):
        assert all('label' not in r and 'answer' not in r for r in rows)
        if any(r['question']=='crash' for r in rows):os._exit(12)
        if any(r['question']=='slow' for r in rows):time.sleep(.3)
        if any(r['question']=='invalid' for r in rows):raise ValueError('Bad test input')
        return [{'id':r['id'],'question':r['question'],'image_seen':r['image'],
                 'image_exists':Path(r['image']).exists(),'fake_pid':os.getpid(),
                 'probability_sum':1.0,'candidate_count':len(r['candidates'])} for r in rows]


def fake_factory(config,device):return FakeEngine()


def config(**kw):
    return dict(max_batch_size=4,max_pending=16,batch_wait_ms=40,numerics='native',mode='causal',**kw)


def image_url(colour='red'):
    b=io.BytesIO();Image.new('RGB',(24,32),colour).save(b,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()


def body(question='ok',identifier='q'):
    return {'image':image_url(),'questions':[{'id':identifier,'question':question,'candidates':['yes','no'],'label':'A'}]}


@pytest.fixture
def pool():
    value=ReplicaPool(config(),['cpu'],engine_factory=fake_factory)
    value.wait_ready(timeout=15)
    try:yield value
    finally:value.close()


@pytest.fixture
def http(pool):
    server=ScoringServer(('127.0.0.1',0),pool)
    t=threading.Thread(target=server.serve_forever,daemon=True);t.start()
    try:yield server,f'http://127.0.0.1:{server.server_port}'
    finally:server.shutdown();server.server_close();t.join(timeout=2)


def post(url,value):
    req=urllib.request.Request(url+'/v1/score',data=json.dumps(value).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=8) as r:return json.load(r)


def test_payload_preserves_schema_order_and_discards_answers(tmp_path):
    data={'documents':[body(identifier='q1'),{'id':'doc2','image':image_url('blue'),
          'question':{'instructions':'second','criteria':{'A':'left','B':'middle','C':'right'}},'label':'C','answer':'right'}]}
    rows,groups,multi=prepare_payload(data,tmp_path)
    assert multi and [r['id'] for r in rows]==['q1','doc2']
    assert [len(r['candidates']) for r in rows]==[2,3]
    assert rows[0]['image']!=rows[1]['image']
    assert groups==[{'id':None,'start':0,'end':1},{'id':'doc2','start':1,'end':2}]
    assert all('label' not in r and 'answer' not in r for r in rows)


def test_payload_local_path_containment_and_limits(tmp_path):
    root=tmp_path/'allowed';root.mkdir();Image.new('RGB',(24,32)).save(tmp_path/'outside.png')
    with pytest.raises(ValueError,match='media-root'):
        prepare_payload({'image':'../outside.png','question':'q','candidates':['a','b']},tmp_path,media_root=root)
    with pytest.raises(ValueError,match='base64'):
        prepare_payload({'image':'data:image/png;base64,@@','question':'q','candidates':['a','b']},tmp_path)
    with pytest.raises(ValueError,match='Too many'):
        prepare_payload({'image':image_url(),'questions':[body()['questions'][0]]*3},tmp_path,max_questions=2)


def test_real_spawn_processes_and_dynamic_batch(pool,tmp_path):
    rows=[{'id':str(i),'image':str(tmp_path),'question':'ok','candidates':['a','b']} for i in range(4)]
    result=[f.result(timeout=5) for f in pool.submit_many(rows,5)]
    assert [r['id'] for r in result]==['0','1','2','3']
    assert all(r['service_batch_size']==4 and r['fake_pid']!=os.getpid() for r in result)
    assert pool.context.get_start_method()=='spawn'


def test_multiple_replicas_start_without_ddp():
    value=ReplicaPool(config(),['cpu','cuda:7'],engine_factory=fake_factory)
    try:
        value.wait_ready(15);assert value.health()['workers']=={0:'cpu',1:'cuda:7'}
        assert len({p.pid for p in value.processes})==2
    finally:value.close()


def test_queue_backpressure_and_input_error_isolation(tmp_path):
    c=config();c.update(max_batch_size=4,max_pending=4)
    value=ReplicaPool(c,['cpu'],engine_factory=fake_factory)
    try:
        value.wait_ready(15)
        rows=[{'id':str(i),'image':str(tmp_path),'question':'slow','candidates':['a','b']} for i in range(4)]
        pending=value.submit_many(rows,5)
        with pytest.raises(ServiceBusy):value.submit_many(rows[:1],5)
        assert all(f.result(5)['id']==str(i) for i,f in enumerate(pending))
        rows[0]['question']='invalid'
        pending=value.submit_many(rows[:2],5)
        with pytest.raises(RuntimeError,match='Bad test input'):pending[0].result(5)
        assert pending[1].result(5)['id']=='1'
    finally:value.close()


def test_http_cross_image_requests_and_cleanup(http):
    server,url=http
    data={'documents':[body(identifier='first'),body(identifier='second')]}
    result=post(url,data)
    questions=[d['questions'][0] for d in result['documents']]
    assert [q['id'] for q in questions]==['first','second']
    assert all(q['image_exists'] and q['service_batch_size']==2 for q in questions)
    # Handler cleans its request files after the response.
    deadline=time.monotonic()+2
    while any(Path(q['image_seen']).exists() for q in questions) and time.monotonic()<deadline:time.sleep(.01)
    assert all(not Path(q['image_seen']).exists() for q in questions)
    with urllib.request.urlopen(url+'/health') as r:assert json.load(r)['status']=='ready'


def test_concurrent_http_requests_are_really_batched(http):
    _,url=http
    barrier=threading.Barrier(2)
    def call(i):barrier.wait();return post(url,body(identifier=str(i)))['questions'][0]
    with concurrent.futures.ThreadPoolExecutor(2) as executor:results=list(executor.map(call,range(2)))
    assert [x['id'] for x in results]==['0','1']
    assert all(x['service_batch_size']==2 for x in results)


def test_http_timeout_keeps_files_until_worker_finishes(http):
    server,url=http;server.request_timeout=.08
    with pytest.raises(urllib.error.HTTPError) as e:post(url,body('slow'))
    assert e.value.code==504
    deadline=time.monotonic()+3
    while server.pool.health()['pending_questions'] and time.monotonic()<deadline:time.sleep(.02)
    assert server.pool.health()['pending_questions']==0


def test_http_reports_individual_model_input_error(http):
    _,url=http
    with pytest.raises(urllib.error.HTTPError) as e:post(url,body('invalid'))
    assert e.value.code==422
    assert json.load(e.value)['error']=={'type':'ValueError','message':'Bad test input'}


def test_example_client_does_not_send_reference_labels(tmp_path):
    import importlib.util
    script=Path(__file__).resolve().parents[1]/'examples/service_client.py'
    spec=importlib.util.spec_from_file_location('Docision_example_client',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    Image.new('RGB',(24,32),'blue').save(tmp_path/'source.png')
    source={'image':'source.png','questions':[{'id':'q','question':{'instructions':'q','criteria':{'A':'a','B':'b'}},'label':'B','answer':'b','metadata':{'secret':'omit'}}]}
    p=tmp_path/'input.json';p.write_text(json.dumps(source))
    result=module.payload(p)
    assert result['image'].startswith('data:image/png;base64,')
    assert set(result['questions'][0])=={'id','question'}
    assert result['questions'][0]['question']==source['questions'][0]['question']


def test_worker_crash_fails_pending_without_hanging(pool,tmp_path):
    futures=pool.submit_many([{'id':'crash','image':str(tmp_path),'question':'crash','candidates':['a','b']}],5)
    with pytest.raises(RuntimeError,match='worker exited'):futures[0].result(5)
    assert pool.health()['status']=='failed'
