import base64,json,threading,time,unittest,tempfile
from pathlib import Path
from unittest.mock import patch
import app
from service_api import JobManager,APIError
from token_store import load_token
import test_app

BODY={'name':'a.png','data':base64.b64encode(b'test').decode(),'questions':[{'question':'table?','candidates':['yes','no']}]}

class QueueTests(unittest.TestCase):
    def test_bounded_queue_progress_pagination_delete(self):
        running=threading.Event();release=threading.Event()
        def score(body,progress):
            running.set();release.wait(5);progress({'pages_done':1,'pages_total':1,'rows_done':2})
            return {'rows':[{'file':body['name'],'page':1},{'file':body['name'],'page':2}]}
        jobs=JobManager(score,threading.Lock(),lambda:True,queue_size=1)
        first=jobs.submit(BODY);self.assertTrue(running.wait(2))
        second=jobs.submit(BODY)
        with self.assertRaises(APIError) as error:jobs.submit(BODY)
        self.assertEqual(error.exception.status,429)
        with self.assertRaises(APIError):jobs.delete(first['id'])
        release.set();jobs.queue.join()
        self.assertEqual(jobs.get(first['id'])['status'],'completed')
        self.assertNotIn('body',jobs.get(first['id']))
        self.assertEqual(jobs.get(first['id'])['progress']['pages_done'],1)
        self.assertEqual(jobs.results(first['id'],0,1)['next_offset'],1)
        self.assertIsNone(jobs.results(first['id'],1,1)['next_offset'])
        jobs.delete(first['id'])
        with self.assertRaises(APIError):jobs.get(first['id'])
        self.assertEqual(jobs.get(second['id'])['summary']['rows'],2)

    def test_failure_expiry_and_unloaded_model(self):
        jobs=JobManager(lambda body,progress:(_ for _ in ()).throw(ValueError('broken')),threading.Lock(),lambda:True,ttl=1)
        job=jobs.submit(BODY);jobs.queue.join()
        self.assertEqual(jobs.get(job['id'])['status'],'failed')
        with self.assertRaises(APIError):jobs.results(job['id'])
        jobs.jobs[job['id']]['finished_at']-=2
        with self.assertRaises(APIError):jobs.get(job['id'])
        unloaded=JobManager(None,threading.Lock(),lambda:False)
        with self.assertRaises(APIError) as error:unloaded.submit(BODY)
        self.assertEqual(error.exception.status,409)

    def test_validation_happens_before_queueing(self):
        jobs=JobManager(None,threading.Lock(),lambda:True)
        for bad in [dict(BODY,data='??'),dict(BODY,data=''),dict(BODY,questions=[]),dict(BODY,prefix_cache='yes')]:
            with self.assertRaises((APIError,ValueError)):jobs.submit(bad)
        self.assertFalse(jobs.jobs)

class V1HTTPTests(unittest.TestCase):
    setUpClass = classmethod(test_app.HTTPTests.setUpClass.__func__)
    tearDownClass = classmethod(test_app.HTTPTests.tearDownClass.__func__)
    request = test_app.HTTPTests.request
    def test_v1_auth_contract_and_async(self):
        headers={'Authorization':'Bearer '+app.TOKEN,'Content-Type':'application/json'}
        status,raw=self.request('GET','/api/v1/health',**headers)
        self.assertEqual(status,200);self.assertEqual(json.loads(raw)['api_version'],'1')
        self.assertEqual(self.request('GET','/api/v1/health')[0],403)
        self.assertEqual(self.request('POST','/api/v1/jobs',BODY,**{'Authorization':'Bearer '+app.TOKEN})[0],415)
        jobs=JobManager(lambda body,progress:{'rows':[{'file':body['name'],'page':1,'question':'x','correction':''}]},threading.Lock(),lambda:True)
        with patch.object(app,'JOBS',jobs):
            status,raw=self.request('POST','/api/v1/jobs',BODY,**headers)
            self.assertEqual(status,202);job=json.loads(raw);jobs.queue.join()
            self.assertEqual(self.request('GET',f"/api/v1/jobs/{job['id']}/results?limit=1",**headers)[0],200)
            status,raw=self.request('GET',f"/api/v1/jobs/{job['id']}/export?format=csv",**headers)
            self.assertEqual(status,200);self.assertIn('人工修正',raw.decode('utf-8'))
            self.assertEqual(self.request('GET',f"/api/v1/jobs/{job['id']}/results?limit=0",**headers)[0],400)
            self.assertEqual(self.request('DELETE',f"/api/v1/jobs/{job['id']}",**headers)[0],200)
        self.assertEqual(self.request('GET','/api/v1/openapi.json',**headers)[0],200)

class CredentialTests(unittest.TestCase):
    def test_token_persists_and_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'token';a=load_token(path);self.assertEqual(a,load_token(path));self.assertGreaterEqual(len(a),32)
            link=Path(temp)/'link';link.symlink_to(path)
            with self.assertRaises(ValueError):load_token(link)

if __name__=='__main__':unittest.main()
