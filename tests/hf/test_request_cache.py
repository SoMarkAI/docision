"""Real tiny Qwen3-VL forwards through the cached request/service path."""
from copy import deepcopy
from pathlib import Path
import queue
import torch
import pytest
from test_qwen3_vl import vl, prepared, values
from docision._runtime.hf_request_cache import RequestPrefixCache
from docision.service import _run_batch


@pytest.fixture
def inputs(vl,tmp_path,monkeypatch):
    a=tmp_path/'a.png';b=tmp_path/'b.png';alias=tmp_path/'alias.png'
    a.write_bytes(b'image-a');b.write_bytes(b'image-b');alias.write_bytes(a.read_bytes())
    base={key:prepared('image') for key in [b'image-a',b'image-b']}
    def prepare(media,question,candidates,*,context='',decoded=None,**kw):
        x,s=deepcopy(base[Path(media).read_bytes()])
        prefix=5
        # Distinct shared context, without changing the image, must miss cache.
        if context=='long':
            x['input_ids']=torch.cat([x['input_ids'][:,:5],torch.tensor([[17,18]]),x['input_ids'][:,5:]],1)
            x['attention_mask']=torch.ones_like(x['input_ids'])
            x['mm_token_type_ids']=torch.cat([x['mm_token_type_ids'],torch.zeros((1,2),dtype=torch.long)],1)
            prefix+=2;s['spans']=[[a+2,b+2] for a,b in s['spans']];s['prompt_tokens']+=2
        s['prefix_length']=prefix
        if question=='changed':x['input_ids'][0,-1]=23
        return x,s,{}
    monkeypatch.setattr(vl,'prepare',prepare)
    count=[];prefill=vl._prefill
    def counted(*args):count.append(1);return prefill(*args)
    monkeypatch.setattr(vl,'_prefill',counted)
    rows=[{'id':'a','image':str(a),'question':'q','candidates':['yes','no']},
          {'id':'b','image':str(b),'question':'q','candidates':['yes','no'],'context':'long'},
          {'id':'a2','image':str(alias),'question':'changed','candidates':['yes','no']}]
    return rows,count


@pytest.mark.parametrize('mode',['causal','isolated'])
def test_request_cache_hits_parity_and_service_integration(vl,inputs,mode, assert_public_output):
    rows,count=inputs;cache=RequestPrefixCache(vl)
    expected=vl.score_requests(rows,mode=mode,batch_size=3)
    actual=cache.score_requests(rows,mode=mode,batch_size=3)
    assert len(count)==2
    assert [v['prefix_cache_hit'] for v in actual]==[False,False,True]
    assert [v['num_cached_tokens'] for v in actual]==[5,7,5]
    assert all(v['batch_size']==3 for v in actual)
    for a,b in zip(actual,expected):torch.testing.assert_close(values(a),values(b),atol=3e-6,rtol=1e-5)
    outbox=queue.Queue()
    _run_batch(vl,[{'job_id':str(i),'row':r} for i,r in enumerate(rows[::-1])],
               {'mode':mode},3,outbox,cache)
    result=[outbox.get_nowait()['value'] for _ in rows]
    assert len(count)==2 and all(v['prefix_cache_hit'] for v in result)
    assert [v['id'] for v in result]==['a2','b','a']
    assert all(v['worker_id']==3 and v['service_batch_size']==3 for v in result)
    for a,b in zip(result,expected[::-1]):
        torch.testing.assert_close(torch.tensor([c['probability'] for c in a['candidates']]),torch.tensor([c['probability'] for c in b['candidates']]),atol=3e-6,rtol=1e-5)
        assert_public_output(a)


def test_lru_context_change_file_change_and_limits(vl,inputs):
    rows,count=inputs;cache=RequestPrefixCache(vl,max_entries=1,max_tokens=6)
    cache.score_requests(rows,batch_size=3)
    assert len(cache.entries)==1 and cache.tokens==5
    changed={**rows[0],'context':'long'}
    result=cache.score_requests([changed])
    assert not result[0]['prefix_cache_hit'] and len(count)==3
    # A longer-than-budget prefix is usable but never retained.
    cache.score_requests([changed]);assert len(count)==4 and cache.tokens==5
    Path(rows[0]['image']).write_bytes(b'image-b')
    result=cache.score_requests([rows[0]])
    assert not result[0]['prefix_cache_hit'] and len(count)==5
    cache.score_requests([rows[2]]);assert len(count)==6  # LRU evicted old content.
    cache.clear();assert cache.tokens==0 and not cache.entries
    with pytest.raises(ValueError,match='positive'):RequestPrefixCache(vl,max_tokens=0)
    with pytest.raises(ValueError,match='positive'):cache.score_requests(rows,batch_size=0)


def test_cached_oom_split_and_bad_input_isolation(vl,inputs,monkeypatch):
    rows,count=inputs;cache=RequestPrefixCache(vl)
    original=cache.score_requests;attempts=[]
    def constrained(rows,**kwargs):
        attempts.append(len(rows))
        if len(rows)>1:raise torch.OutOfMemoryError('test memory limit')
        if rows[0]['question']=='invalid':raise ValueError('invalid question')
        return original(rows,**kwargs)
    monkeypatch.setattr(cache,'score_requests',constrained)
    rows[1]['question']='invalid';out=queue.Queue()
    _run_batch(vl,[{'job_id':str(i),'row':r} for i,r in enumerate(rows)],{'mode':'causal'},0,out,cache)
    results=[out.get_nowait() for _ in rows]
    assert attempts==[3,1,2,1,1]
    assert 'value' in results[0] and 'value' in results[2]
    assert results[1]['error']['type']=='ValueError'
    assert not vl.model.model.language_model._forward_pre_hooks


def tiny_service_factory(config,device):
    # Real model forward; handcrafted token/patch inputs stand in for processor.
    from types import SimpleNamespace
    from functools import partial
    engine=vl.__wrapped__(SimpleNamespace(param=torch.float32))
    engine.prepare=partial(tiny_service_prepare,engine)
    return engine


def tiny_service_prepare(engine,media,question,candidates,*,context='',decoded=None,**kw):
    import hashlib
    seed=int.from_bytes(hashlib.sha256(Path(media).read_bytes()).digest()[:4],'big')
    with torch.random.fork_rng():
        torch.manual_seed(seed);x,s=prepared('image')
    s['prefix_length']=5
    if question=='changed':x['input_ids'][0,-1]=23
    return x,s,{}


def test_cached_service_real_model_spawn_http_and_temp_cleanup(assert_public_output):
    import base64
    import io
    import json
    import threading
    import urllib.request
    from PIL import Image
    from docision.service import ReplicaPool,ScoringServer
    def image(colour):
        stream=io.BytesIO();Image.new('RGB',(24,32),colour).save(stream,format='PNG')
        return 'data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()
    def document(colour):
        return {'image':image(colour),'questions':[
            {'id':colour+'-1','question':'q','candidates':['yes','no']},
            {'id':colour+'-2','question':'changed','candidates':['yes','no']}]}
    config={'max_batch_size':4,'max_pending':8,'batch_wait_ms':50,'numerics':'stable',
            'mode':'causal','prefix_cache':True,'cache_max_entries':8,'cache_max_tokens':64}
    pool=ReplicaPool(config,['cpu'],engine_factory=tiny_service_factory)
    server=None;thread=None
    try:
        pool.wait_ready(20)
        server=ScoringServer(('127.0.0.1',0),pool)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        payload={'documents':[document('red'),document('blue')]}
        def post():
            req=urllib.request.Request(f'http://127.0.0.1:{server.server_port}/v1/score',
                data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=20) as response:
                return [q for doc in json.load(response)['documents'] for q in doc['questions']]
        first=post();second=post()  # Independent snapshot filenames on the next HTTP call.
        assert [v['id'] for v in first]==['red-1','red-2','blue-1','blue-2']
        assert [v['prefix_cache_hit'] for v in first]==[False,True,False,True]
        assert all(v['prefix_cache_hit'] and v['service_batch_size']==4 for v in second)
        for a,b in zip(first,second):
            torch.testing.assert_close(torch.tensor([c['probability'] for c in a['candidates']]),torch.tensor([c['probability'] for c in b['candidates']]),atol=3e-6,rtol=1e-5)
            assert_public_output(a)
            assert abs(a['probability_sum']-1)<1e-6
        assert pool.health()['prefix_cache']
    finally:
        if server:server.shutdown();server.server_close()
        if thread:thread.join(timeout=2)
        pool.close()
