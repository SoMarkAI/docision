"""Public formatting preserves real tiny-model scoring and cached batches."""
from copy import deepcopy
import json
import torch
import pytest
from test_qwen3_vl import vl, prepared
from docision.engine import DocisionEngine

@pytest.mark.parametrize('batch_size',[1,2])
def test_real_public_scores_preserve_probabilities_and_decisions(vl,tmp_path,batch_size, assert_public_output):
    image=tmp_path/'page.png';image.write_bytes(b'test-image')
    batch,spec=prepared('image');spec['prefix_length']=5
    def prepare(*args,**kwargs):return deepcopy(batch),deepcopy(spec),{}
    vl.prepare=prepare
    public=DocisionEngine.__new__(DocisionEngine)
    public.__dict__.update(vl.__dict__)
    questions=[{'id':str(i),'question':'q','candidates':['yes','no']} for i in range(2)]
    raw=vl.score_many(image,questions,projection='full',batch_size=batch_size)
    outputs=[public.score_document({'image':image,'questions':questions,'context':''},batch_size=batch_size),
             public.score_many(image,questions,projection='full',batch_size=batch_size),
             public.score_requests([dict(q,image=image) for q in questions],batch_size=batch_size),
             public.score_questions(public.prepare_context(image),questions,projection='full',batch_size=batch_size)]
    for results in outputs:
        assert_public_output(results)
        for result,expected in zip(results,raw):
            assert result['decision']['label']==expected['decision']['label']
            assert result['decision']['tied_labels']==expected['decision']['tied_labels']
            torch.testing.assert_close(torch.tensor([c['probability'] for c in result['candidates']]),
                                       torch.tensor([c['probability'] for c in expected['candidates']]),atol=3e-6,rtol=1e-5)
    assert all('raw_logit' in c for r in raw for c in r['candidates'])
