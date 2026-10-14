"""Functional regressions; no benchmark data or metric changes."""
import base64
import inspect
import threading
import time
import unittest
from unittest.mock import patch
import torch
from docision.evaluate import score_benchmark, summarize
from docision.engine import DocisionEngine
from docision._runtime.hf import HFDocisionEngine
from docision._runtime.decision import choose
from docision.local_app.service_api import JobManager

class FunctionalRegressions(unittest.TestCase):
    def test_python310_evaluation_and_compare_shared_path(self):
        class Engine:
            def score_many(self, image, requests, **kwargs):
                return [{'candidates':[{'probability':.25},{'probability':.75}],
                         'decision':{'index':1},'num_cached_tokens':0} for _ in requests]
        question={'id':'q','image':'fixture','context':'context','question':'q',
                  'candidates':['a','b'],'label':'B'}
        rows=score_benchmark(Engine(),[question],protocol='none')
        self.assertEqual(summarize(rows)['original_order'],{'count':1,'accuracy':1.0})
        from docision.compare import score_benchmark as compare_score, paired_summary
        self.assertIs(compare_score,score_benchmark)
        self.assertEqual(paired_summary(rows,rows)['overall']['delta_percentage_points'],0)
        self.assertEqual(score_benchmark(Engine(),[question],protocol='random',seed=42),
                         score_benchmark(Engine(),[question],protocol='random',seed=42))

    def test_near_tie_probability_preserves_logit_winner_and_exact_ties(self):
        from docision._runtime.decision import candidate_probabilities
        for logits in ([0.,1e-8],[1e-8,0.],[0.,0.],[10000.,-10000.]):
            tensor=torch.tensor(logits,dtype=torch.float32)
            probs=candidate_probabilities(tensor).tolist()
            rows=[{'label':str(i),'raw_logit':float(tensor[i]),'probability':p}
                  for i,p in enumerate(probs)]
            decision=choose(rows)
            self.assertEqual(decision['index'],max(range(len(probs)),key=probs.__getitem__))
            self.assertAlmostEqual(sum(probs),1.)
            self.assertEqual(decision['tie_basis'],'raw_logits')
        self.assertEqual(choose([{'label':'A','raw_logit':0.},{'label':'B','raw_logit':0.}])['tied_labels'],['A','B'])

    def test_public_projection_defaults_are_full_and_forwarded(self):
        engine=DocisionEngine.__new__(DocisionEngine)
        for method,args in [('score_prepared',({},{})),('score_many',('x',[])),('score_questions',(object(),[]))]:
            self.assertEqual(inspect.signature(getattr(DocisionEngine,method)).parameters['projection'].default,'full')
            with patch.object(HFDocisionEngine,method,return_value=[]) as call:
                getattr(engine,method)(*args)
                self.assertEqual(call.call_args.kwargs['projection'],'full')
                getattr(engine,method)(*args,projection='selected')
                self.assertEqual(call.call_args.kwargs['projection'],'selected')

    def test_expiry_without_access_or_new_submission_during_busy_job(self):
        release=threading.Event();busy=threading.Event()
        def score(body,progress):
            if body['name']=='slow.png':
                busy.set()
                if not release.wait(3):raise RuntimeError('test timeout')
            return {'rows':[{'result':'ok'}]}
        jobs=JobManager(score,threading.Lock(),lambda:True,ttl=.08)
        def body(name):return {'name':name,'data':base64.b64encode(b'x').decode(),
            'questions':[{'question':'q','candidates':['yes','no']}]}
        first=jobs.submit(body('fast.png'));jobs.queue.join()
        second=jobs.submit(body('slow.png'));self.assertTrue(busy.wait(1))
        try:
            deadline=time.monotonic()+1
            while time.monotonic()<deadline:
                with jobs.lock:
                    if first['id'] not in jobs.jobs:break
                time.sleep(.01)
            with jobs.lock:
                self.assertNotIn(first['id'],jobs.jobs)
                self.assertEqual(jobs.jobs[second['id']]['status'],'running')
        finally:release.set();jobs.queue.join()

if __name__=='__main__':unittest.main()
