"""Native tiny Qwen3-VL witnesses; random weights, not pretrained accuracy."""
from copy import deepcopy
from threading import RLock
import pytest
import torch
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
from docision._runtime.hf import HFDocisionEngine, HFContext, scoring_hooks
from docision._runtime.hf_batch import score_batch
from docision._runtime.families import text_model, check_modality, VL

@pytest.fixture(params=[torch.float32,torch.bfloat16])
def vl(request):
    torch.manual_seed(37)
    config=Qwen3VLConfig(text_config=dict(vocab_size=128,hidden_size=32,intermediate_size=64,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8,
        rope_parameters={'rope_type':'default','rope_theta':10000.,'mrope_section':[1,1,2]}),
        vision_config=dict(depth=2,hidden_size=32,intermediate_size=64,num_heads=4,
            out_hidden_size=32,deepstack_visual_indexes=[0],num_position_embeddings=16),
        image_token_id=105,video_token_id=106,vision_start_token_id=100,vision_end_token_id=101)
    config._attn_implementation='sdpa'
    model=Qwen3VLForConditionalGeneration(config).to(request.param).eval()
    model.generate=lambda *a,**k:(_ for _ in ()).throw(AssertionError('generate forbidden'))
    engine=HFDocisionEngine.__new__(HFDocisionEngine)
    engine.model=model;engine.lock=RLock();engine._numerics_mode='stable'
    return engine


def prepared(kind):
    token=105 if kind=='image' else 106
    ids=torch.tensor([[1,100,token,101,2,3,4,5,6,7,8]])
    types=torch.zeros_like(ids);types[0,2]=1 if kind=='image' else 2
    batch=dict(input_ids=ids,attention_mask=torch.ones_like(ids),mm_token_type_ids=types)
    grid=torch.tensor([[1,2,2]]);pixels=torch.randn(4,3*2*16*16)
    if kind=='image':batch.update(pixel_values=pixels,image_grid_thw=grid)
    else:batch.update(pixel_values_videos=pixels,video_grid_thw=grid)
    spec=dict(spans=[[5,7],[7,9]],label_ids=[10,11],labels=['A','B'],candidate_texts=['yes','no'],prompt_tokens=11)
    return batch,spec


def values(result):return torch.tensor([r['raw_logit'] for r in result['candidates']])

@pytest.mark.parametrize('kind',['image'])
@pytest.mark.parametrize('mode',['causal','isolated'])
def test_vl_real_forward_batch_cache_and_unchanged_weights(vl,kind,mode):
    b,s=prepared(kind);before={k:v.clone() for k,v in vl.model.state_dict().items()}
    baseline=vl.score_prepared(b,s,mode=mode,projection='full')
    torch.testing.assert_close(values(vl.score_prepared(b,s,mode=mode)),values(baseline),atol=1e-6,rtol=1e-5)
    past,pos=vl._prefill(b,5)
    cache=HFContext(vl,None,kind,'',None,None,b,past,pos,5)
    s['prefix_length']=5
    b2=deepcopy(b);s2=deepcopy(s)
    b2['input_ids']=torch.cat([b2['input_ids'],torch.tensor([[12,13]])],1)
    b2['attention_mask']=torch.ones_like(b2['input_ids'])
    b2['mm_token_type_ids']=torch.cat([b2['mm_token_type_ids'],torch.zeros((1,2),dtype=torch.long)],1)
    s2['prompt_tokens']+=2;s2['spans'].append([9,10]);s2['labels'].append('C');s2['label_ids'].append(12);s2['candidate_texts'].append('other')
    baseline2=vl.score_prepared(b2,s2,mode=mode,projection='full')
    for c in (None,cache):
        result,frames=score_batch(vl,[(b,s),(b2,s2)],mode=mode,cache=c)
        assert frames[0]['batch_size']==2
        for actual,expected in zip(result,[baseline,baseline2]):
            torch.testing.assert_close(values(actual),values(expected),atol=3e-6,rtol=1e-5)
            assert abs(actual['probability_sum']-1)<1e-6
        reverse,_=score_batch(vl,[(b2,s2),(b,s)],mode=mode,cache=c)
        for actual,expected in zip(result,reversed(reverse)):
            torch.testing.assert_close(values(actual),values(expected),atol=3e-6,rtol=1e-5)
    assert past.get_seq_length()==5
    for k,v in vl.model.state_dict().items():assert torch.equal(v,before[k])


def test_vl_mask_isolation_and_hook_cleanup(vl):
    b,s=prepared('image');changed=deepcopy(b);changed['input_ids'][0,5:7]=torch.tensor([21,22]);states=[]
    h=text_model(vl.model).register_forward_hook(lambda m,a,o:states.append(o.last_hidden_state.clone()))
    try:
        for mode in ['isolated','causal']:
            for batch in [b,changed]:vl.score_prepared(batch,s,mode=mode)
    finally:h.remove()
    torch.testing.assert_close(states[0][:,7:9],states[1][:,7:9],atol=1e-6,rtol=1e-5)
    assert not torch.equal(states[0][:,-1],states[1][:,-1])
    assert not torch.equal(states[2][:,7:9],states[3][:,7:9])
    with pytest.raises(RuntimeError):
        with scoring_hooks(vl.model,s['spans'],s['label_ids'],'isolated'):raise RuntimeError('probe')
    assert not text_model(vl.model)._forward_pre_hooks

@pytest.mark.parametrize('modality',['audio','audio_video','video'])
def test_vl_rejects_audio(modality):
    with pytest.raises(ValueError,match='never silently discarded'):check_modality(VL,modality)


def test_vl_official_class_checkpoint_load(vl,tmp_path):
    from types import SimpleNamespace
    vl.model.save_pretrained(tmp_path)
    loaded=HFDocisionEngine(tmp_path,device_map='cpu',dtype='float32',processor=SimpleNamespace(tokenizer=None))
    assert type(loaded.model) is Qwen3VLForConditionalGeneration
    assert not loaded.loading_info['missing_keys']
    for k,v in vl.model.state_dict().items():assert torch.equal(v,loaded.model.state_dict()[k])
