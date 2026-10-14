"""Actual tiny Qwen3-VL forwards with different images/grids/candidate counts."""
from copy import deepcopy
import pytest
import torch
from test_qwen3_vl import vl, prepared, values
from docision._runtime.hf_batch import score_batch


@pytest.mark.parametrize('mode',['causal','isolated'])
@pytest.mark.parametrize('projection',['full','selected'])
def test_cross_image_padding_positions_and_independence(vl,mode,projection):
    b,s=prepared('image');b2,s2=prepared('image')
    # Second image has twice as many patches and image tokens, plus a third option.
    b2['input_ids']=torch.tensor([[1,100,105,105,101,2,3,4,5,6,7,8,9,10]])
    b2['attention_mask']=torch.ones_like(b2['input_ids'])
    b2['mm_token_type_ids']=torch.zeros_like(b2['input_ids']);b2['mm_token_type_ids'][0,2:4]=1
    b2['image_grid_thw']=torch.tensor([[1,2,4]])
    b2['pixel_values']=torch.randn(8,3*2*16*16)
    s2.update(spans=[[6,8],[8,10],[10,12]],label_ids=[10,11,12],labels=['A','B','C'],candidate_texts=['left','middle','right'],prompt_tokens=14)
    baseline=[vl.score_prepared(x,y,mode=mode,projection=projection) for x,y in [(b,s),(b2,s2)]]
    actual,frames=score_batch(vl,[(b,s),(b2,s2)],mode=mode,projection=projection)
    assert frames[0]['batch_size']==2 and frames[0]['valid_query_lengths']==[11,14]
    for a,e in zip(actual,baseline):
        torch.testing.assert_close(values(a),values(e),atol=3e-6,rtol=1e-5)
        assert abs(a['probability_sum']-1)<1e-6
    reverse,_=score_batch(vl,[(b2,s2),(b,s)],mode=mode,projection=projection)
    for a,e in zip(actual,reversed(reverse)):torch.testing.assert_close(values(a),values(e),atol=3e-6,rtol=1e-5)
    changed=deepcopy(b);changed['pixel_values']=torch.randn_like(b['pixel_values'])*3
    altered,_=score_batch(vl,[(changed,s),(b2,s2)],mode=mode,projection=projection)
    torch.testing.assert_close(values(actual[1]),values(altered[1]),atol=3e-6,rtol=1e-5)
    assert not torch.equal(values(actual[0]),values(altered[0]))


def test_cross_image_public_api_order(vl,tmp_path,monkeypatch):
    p1=tmp_path/'one.png';p2=tmp_path/'two.png';p1.write_bytes(b'1');p2.write_bytes(b'2')
    a=prepared('image');b=prepared('image');b[0]['pixel_values']*=2
    def prepare(image,question,candidates,**kwargs):
        x,s=deepcopy(a if image==str(p1) else b)
        return x,s,{'decoded':image}
    monkeypatch.setattr(vl,'prepare',prepare)
    rows=[{'id':'image-one','image':p1,'question':'one','candidates':['yes','no']},
          {'id':'image-two','image':p2,'question':'two','candidates':['yes','no']}]
    result=vl.score_requests(rows,batch_size=2)
    assert [x['id'] for x in result]==['image-one','image-two']
    for result,(batch,spec) in zip(result,[a,b]):
        expected=vl.score_prepared(batch,spec,projection='full')
        torch.testing.assert_close(values(result),values(expected),atol=3e-6,rtol=1e-5)

@pytest.mark.parametrize('mode',['causal','isolated'])
@pytest.mark.parametrize('projection',['full','selected'])
def test_mixed_prefix_batch_matches_uncached_and_preserves_caches(vl,mode,projection):
    from docision._runtime.hf import HFContext
    a,sa=prepared('image');sa['prefix_length']=5
    b,sb=deepcopy(a),deepcopy(sa)
    # Different image size, longer context, longer question and three candidates.
    b['input_ids']=torch.tensor([[1,100,105,105,101,2,20,21,3,4,5,6,7,8,9,10]])
    b['attention_mask']=torch.ones_like(b['input_ids'])
    b['mm_token_type_ids']=torch.zeros_like(b['input_ids']);b['mm_token_type_ids'][0,2:4]=1
    b['image_grid_thw']=torch.tensor([[1,2,4]])
    b['pixel_values']=torch.randn(8,3*2*16*16)
    sb.update(prefix_length=8,spans=[[8,10],[10,12],[12,14]],label_ids=[10,11,12],
              labels=['A','B','C'],candidate_texts=['left','middle','right'],prompt_tokens=16)
    rows=[(a,sa),(b,sb),(a,sa)]
    cache=[]
    for x,s in rows[:2]:
        past,pos=vl._prefill(x,s['prefix_length'])
        cache.append(HFContext(vl,None,'image','',None,None,x,past,pos,s['prefix_length']))
    cache.append(cache[0])
    before=[[(l.keys.clone(),l.values.clone()) for l in c.past_key_values.layers] for c in cache]
    baseline=[vl.score_prepared(x,s,mode=mode,projection=projection) for x,s in rows]
    actual,frames=score_batch(vl,rows,mode=mode,projection=projection,cache=cache)
    assert frames[0]['batch_size']==3 and frames[0]['query_tokens']==8
    assert [r['num_cached_tokens'] for r in actual]==[5,8,5]
    for x,y in zip(actual,baseline):
        torch.testing.assert_close(values(x),values(y),atol=3e-6,rtol=1e-5)
        assert abs(x['probability_sum']-1)<1e-6
    order=[1,2,0]
    reordered,_=score_batch(vl,[rows[i] for i in order],mode=mode,projection=projection,cache=[cache[i] for i in order])
    for x,y in zip([actual[i] for i in order],reordered):
        torch.testing.assert_close(values(x),values(y),atol=3e-6,rtol=1e-5)
    for c,snapshot in zip(cache,before):
        for layer,(k,v) in zip(c.past_key_values.layers,snapshot):
            assert torch.equal(layer.keys,k) and torch.equal(layer.values,v)
    with pytest.raises(ValueError,match='One cache'):
        score_batch(vl,rows,cache=cache[:1])
    with pytest.raises(ValueError,match='Token prefix changed'):
        score_batch(vl,rows,cache=[cache[1],cache[0],cache[0]])
