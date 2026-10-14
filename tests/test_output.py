from copy import deepcopy
from queue import Queue
import json
from pathlib import Path
import pytest
from docision.output import public_output
from docision._runtime.decision import choose
from docision._runtime.hf import HFDocisionEngine
from docision.engine import DocisionEngine
from docision.service import _run_batch


def internal_result():
    # Probability rounding may tie even when logits differ: retain logit winner.
    candidates = [dict(label="A", text="first", token_id=32, raw_logit=1.0, probability=0.5),
                  dict(label="B", text="second", token_id=33, raw_logit=1.0000001, probability=0.5)]
    return dict(id="q", candidates=candidates, decision=choose(candidates), num_cached_tokens=3)


def test_public_result_preserves_choice_probabilities_and_source(assert_public_output):
    raw=internal_result();before=deepcopy(raw);result=public_output(raw)
    assert raw==before
    assert result['decision']['label']=='B'
    assert [c['probability'] for c in result['candidates']]==[0.5,0.5]
    assert all(set(c)=={'label','text','probability'} for c in result['candidates'])
    assert_public_output(result)
    assert result['num_cached_tokens']==3


def test_equal_logits_keep_first_input_tie_policy():
    raw=internal_result()
    raw['candidates'][1]['raw_logit']=1.0
    raw['decision']=choose(raw['candidates'])
    result=public_output(raw)
    assert result['decision']['label']=='A'
    assert result['decision']['tied_labels']==['A','B']


def test_document_api_omits_internal_scores(monkeypatch, assert_public_output):
    monkeypatch.setattr(DocisionEngine,'score_many',lambda *a,**k: [internal_result()])
    # Test the actual score_document boundary independently of lower-level mocks.
    engine=DocisionEngine.__new__(DocisionEngine)
    result=engine.score_document({'image':'unused','questions':[],'context':''})
    assert_public_output(result)


@pytest.mark.parametrize('cached',[False,True])
def test_service_masks_both_cache_paths(cached, assert_public_output):
    class Engine:
        def score_requests(self,*a,**k):return [internal_result()]
    engine=Engine();out=Queue()
    _run_batch(engine,[{'row':{},'job_id':'job'}],{'mode':'causal'},0,out,engine if cached else None)
    result=out.get_nowait()['value']
    assert result['decision']['label']=='B'
    assert_public_output(result)


def test_recorded_output_has_public_candidates(assert_public_output):
    path=Path(__file__).resolve().parents[1]/'examples/recorded-output.json'
    rows=json.loads(path.read_text())
    assert_public_output(rows)
    assert all(set(c)=={'label','text','probability'} for r in rows for c in r['candidates'])
