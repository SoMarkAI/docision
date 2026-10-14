import base64
import http.client
import io
import json
from pathlib import Path
import threading
import unittest
from unittest.mock import patch
import sys
import types
import app
from core import question_spec, csv_export

class CoreTests(unittest.TestCase):
    def test_question_validation(self):
        self.assertEqual(question_spec(' test ', ['yes','no'])['candidates'], {'A':'yes','B':'no'})
        for choices in [['yes','yes'], ['one'], [1,2], [' ']*3]:
            with self.assertRaises(ValueError): question_spec('test', choices)

    def test_csv_preserves_corrections_and_prevents_formula(self):
        value = csv_export([{'file':'=SUM(A1)', 'page':2, 'correction':'人工确认', 'error':'problem'}])
        self.assertIn("'=SUM(A1)", value)
        self.assertIn('人工确认', value)

class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=app.ThreadingHTTPServer(('127.0.0.1',0),app.Handler)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()
    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.thread.join()
    def request(self, method, path, body=None, **headers):
        connection=http.client.HTTPConnection('127.0.0.1',self.server.server_port)
        connection.request(method,path,json.dumps(body) if body is not None else None,headers)
        response=connection.getresponse();result=(response.status,response.read());connection.close();return result
    def test_ui_and_authenticated_templates(self):
        self.assertEqual(self.request('GET','/')[0],200)
        self.assertEqual(self.request('GET','/api/templates')[0],403)
        status,data=self.request('GET','/api/templates',**{'X-Docision-Token':app.TOKEN})
        self.assertEqual(status,200);self.assertIn('文件分类',json.loads(data))
    def test_cross_origin_and_foreign_host_rejected(self):
        self.assertEqual(self.request('GET','/',Host='attacker.example')[0],403)
        self.assertEqual(self.request('POST','/api/load',{},**{'X-Docision-Token':app.TOKEN,'Origin':'https://attacker.example'})[0],403)
    def test_model_and_input_errors(self):
        status,data=self.request('POST','/api/load',{'model':'/nonexistent'},**{'X-Docision-Token':app.TOKEN})
        self.assertEqual(status,400);self.assertIn('config.json',json.loads(data)['error'])
    def test_busy_response(self):
        with app.LOCK:
            self.assertEqual(self.request('POST','/api/load',{},**{'X-Docision-Token':app.TOKEN})[0],409)
    def test_export_endpoint(self):
        status,data=self.request('POST','/api/csv',{'rows':[]},**{'X-Docision-Token':app.TOKEN})
        self.assertEqual(status,200);self.assertIn('人工修正',json.loads(data)['csv'])

class FileTests(unittest.TestCase):
    def test_image_and_pdf_real_rendering_with_stub_engine(self):
        initial_temp_paths = set((app.ROOT/'.cache').glob('tmp*'))
        from PIL import Image
        import pypdfium2 as pdfium
        class Engine:
            def score_document(self, document):
                with Image.open(document['image']) as image:
                    self.assertion=image.width>0
                return [{'decision':{'text':'包含'},'candidates':[{'text':'包含','probability':1.0}]}]
        fake_io=types.ModuleType('docision.io');fake_io.DEFAULT_CONTEXT='Inspect document.'
        data=io.BytesIO();Image.new('RGB',(100,100),'white').save(data,format='PNG')
        pdf=pdfium.PdfDocument.new()
        page=pdf.new_page(100,100);page.close()
        pdf_data=io.BytesIO();pdf.save(pdf_data);pdf.close()
        with patch.dict(sys.modules, {'docision.io':fake_io}),patch.object(app,'ENGINE',Engine()):
            for name,raw in [('test.png',data.getvalue()),('test.pdf',pdf_data.getvalue())]:
                result=app.score_file({'name':name,'data':base64.b64encode(raw).decode(),'question':'有表格吗？','candidates':['包含','不包含']})
                self.assertEqual(result['rows'][0]['result']['decision']['text'],'包含')
                self.assertEqual(result['rows'][0]['page'],1)
        self.assertEqual(set((app.ROOT/'.cache').glob('tmp*')), initial_temp_paths)


class BootstrapTests(unittest.TestCase):
    def test_warm_start_skips_install(self):
        import bootstrap
        import tempfile
        import hashlib
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'deployment';root.mkdir()
            (root.parent/'docision').mkdir();(root.parent/'docision'/'engine.py').touch()
            env=root/'.venv';(env/'bin').mkdir(parents=True);(env/'bin'/'python').touch()
            (root/'requirements.txt').write_text('test-spec')
            expected=json.dumps({'requirements':'test-spec','backend':'pypi',
                'bootstrap':hashlib.sha256(Path(bootstrap.__file__).read_bytes()).hexdigest()},sort_keys=True)
            (env/'.docision-installed').write_text(expected)
            with patch.object(bootstrap,'ROOT',root),patch.object(bootstrap,'ensure_environment'),patch.object(sys,'argv',['bootstrap.py','--port','9999','--no-browser']),patch.object(bootstrap.subprocess,'run') as install,patch.object(bootstrap.subprocess,'call',return_value=0) as start:
                self.assertEqual(bootstrap.main(),0)
                install.assert_not_called()
                self.assertIn('--no-browser',start.call_args.args[0])

    def test_incomplete_model_rejected_before_loading(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'config.json').write_text('{}')
            (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'a':'missing.safetensors'}}))
            with patch.object(app,'ENGINE',None),self.assertRaisesRegex(ValueError,'missing.safetensors'):
                app.load_engine({'model':temporary,'device':'cpu'})

class BackendSwitchTests(unittest.TestCase):
    def test_explicit_backend_requires_matching_build(self):
        import bootstrap
        import tempfile
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'deployment';root.mkdir()
            (root.parent/'docision').mkdir();(root.parent/'docision'/'engine.py').touch()
            env=root/'.venv';(env/'bin').mkdir(parents=True);(env/'bin'/'python').touch()
            (root/'requirements.txt').write_text('test-spec')
            with patch.object(bootstrap,'ROOT',root),patch.object(bootstrap,'ensure_environment'),patch.object(sys,'argv',['bootstrap.py','--torch-backend','cpu']),patch.object(bootstrap.subprocess,'run') as install,patch.object(bootstrap.subprocess,'call',return_value=0):
                bootstrap.main()
                command=install.call_args_list[0].args[0]
                self.assertIn('torch==2.11.0+cpu',command)
                self.assertIn('torchvision==0.26.0+cpu',command)
                self.assertIn('https://download.pytorch.org/whl/cpu',command)

class PythonVersionTests(unittest.TestCase):
    def test_python39_is_rejected_before_environment_changes(self):
        import bootstrap
        with patch.object(sys,'argv',['bootstrap.py']),patch.object(sys,'version_info',(3,9)),patch.object(bootstrap,'ensure_environment') as create:
            with self.assertRaisesRegex(SystemExit,'Python 3.10'):
                bootstrap.main()
            create.assert_not_called()

class EnvironmentCreationTests(unittest.TestCase):
    def test_seeded_environment_is_reused_without_network(self):
        import bootstrap
        with patch.object(bootstrap,'has_pip',return_value=True),patch.object(bootstrap.venv,'EnvBuilder') as builder,patch.object(bootstrap.urllib.request,'urlopen') as download:
            bootstrap.ensure_environment(Path('unused'),Path('unused/bin/python'))
            builder.assert_not_called()
            download.assert_not_called()

    def test_missing_ensurepip_uses_only_project_local_fallback(self):
        import bootstrap
        import tempfile
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);cache=root/'.cache/bootstrap';cache.mkdir(parents=True)
            (cache/f'virtualenv-{sys.version_info.major}.{sys.version_info.minor}.pyz').touch()
            with patch.object(bootstrap,'ROOT',root),patch.object(bootstrap,'has_pip',side_effect=[False,False,True]),patch.object(bootstrap.importlib.util,'find_spec',return_value=None),patch.object(bootstrap.venv,'EnvBuilder') as builder,patch.object(bootstrap.urllib.request,'urlopen') as download,patch.object(bootstrap.subprocess,'run') as run:
                bootstrap.ensure_environment(root/'.venv',root/'.venv/bin/python')
                builder.assert_not_called()
                download.assert_not_called()
                command=run.call_args.args[0]
                self.assertIn(str(root/'.cache/virtualenv'),command)
                self.assertIn('--no-periodic-update',command)
                self.assertEqual(command[-1],str(root/'.venv'))


class MultiQuestionTests(unittest.TestCase):
    questions = [
        {'question': '是否有表格？', 'candidates': ['包含', '不包含']},
        {'question': '页面语言？', 'candidates': ['中文', '英文', '混合']},
        {'question': '是否清晰？', 'candidates': ['清晰', '模糊']},
    ]

    def test_validate_all_questions_before_inference(self):
        from core import question_specs
        specs = question_specs({'questions': self.questions})
        self.assertEqual([q['id'] for q in specs], ['q1', 'q2', 'q3'])
        self.assertEqual(specs[1]['candidates'], {'A': '中文', 'B': '英文', 'C': '混合'})
        for invalid in [[], None, {}, self.questions * 7, [None],
                        [self.questions[0], {'question': 'x', 'candidates': ['a', 'a']}]]:
            with self.assertRaises(ValueError):
                question_specs({'questions': invalid})
        with patch.object(app, 'ENGINE') as engine:
            with self.assertRaisesRegex(ValueError, '问题 2'):
                app.score_file({'questions': [self.questions[0], {'question': ''}]})
            engine.score_document.assert_not_called()

    def test_page_question_mapping_and_error_isolation(self):
        import pypdfium2 as pdfium
        from PIL import Image
        import csv
        pdf = pdfium.PdfDocument.new()
        for _ in range(2):
            page = pdf.new_page(100, 100); page.close()
        raw = io.BytesIO(); pdf.save(raw); pdf.close()
        calls = []
        class Engine:
            def score_document(self, document):
                spec = document['questions'][0]
                calls.append((document['image'], spec))
                if spec['id'] == 'q2':
                    raise RuntimeError('isolated failure')
                return [{'id': spec['id'], 'decision': {'text': list(spec['candidates'].values())[0]},
                         'candidates': [{'text': c, 'probability': 1.0 if i == 0 else 0.0}
                                        for i, c in enumerate(spec['candidates'].values())]}]
        fake_io = types.ModuleType('docision.io'); fake_io.DEFAULT_CONTEXT = 'Inspect.'
        initial = set((app.ROOT / '.cache').glob('tmp*'))
        original_save = Image.Image.save
        renders = []
        def save(image, path, *args, **kwargs):
            renders.append(str(path)); return original_save(image, path, *args, **kwargs)
        with patch.dict(sys.modules, {'docision.io': fake_io}), patch.object(app, 'ENGINE', Engine()), patch.object(Image.Image, 'save', save):
            rows = app.score_file({'name': 'two.pdf', 'data': base64.b64encode(raw.getvalue()).decode(),
                                   'questions': self.questions})['rows']
            broken = app.score_file({'name': 'broken.pdf', 'data': base64.b64encode(b'broken').decode(),
                                     'questions': self.questions})['rows']
        self.assertEqual(len(renders), 2)
        self.assertEqual(len(calls), 6)
        self.assertEqual([(r['page'], r['question_id']) for r in rows],
                         [(p, q) for p in (1, 2) for q in ('q1', 'q2', 'q3')])
        self.assertEqual([r['question_id'] for r in rows if 'error' in r], ['q2', 'q2'])
        self.assertEqual(rows[2]['result']['decision']['text'], '清晰')
        self.assertEqual(rows[1]['candidates'], ['中文', '英文', '混合'])
        self.assertEqual(len(broken), 3)
        self.assertTrue(all('error' in r for r in broken))
        rows[2]['correction'] = '人工修正,保留\n换行'
        exported = list(csv.DictReader(io.StringIO(csv_export(rows).lstrip('\ufeff'))))
        self.assertEqual(len(exported), 6)
        self.assertEqual(exported[2]['问题编号'], 'q3')
        self.assertEqual(exported[2]['人工修正'], rows[2]['correction'])
        self.assertEqual(exported[1]['候选答案'], '中文 / 英文 / 混合')
        self.assertEqual(exported[1]['错误'], rows[1]['error'])
        self.assertEqual(set((app.ROOT / '.cache').glob('tmp*')), initial)


class PrefixCacheTests(unittest.TestCase):
    def exercise(self, *, enabled=True, count=3, fail_build=False, fail_question=False):
        import pypdfium2 as pdfium
        import weakref
        pdf=pdfium.PdfDocument.new()
        for _ in range(2):
            page=pdf.new_page(80,80); page.close()
        raw=io.BytesIO();pdf.save(raw);pdf.close()
        builds=[];cached=[];uncached=[];references=[];calls=[];live_before_build=[]
        class Context:
            prefix_length=42
            def __init__(self, media): self.media=media
        class Engine:
            _numerics_mode = 'stable'
            def prepare_context(self, media, **kwargs):
                live_before_build.append(any(ref() is not None for ref in references))
                builds.append(media)
                if fail_build: raise RuntimeError('cache unavailable')
                context=Context(media);references.append(weakref.ref(context));return context
            def score_questions(self, cache, questions, **kwargs):
                calls.append((dict(kwargs), len(questions)))
                cached.extend((cache.media, question['id']) for question in questions)
                if fail_question and any(q['id']=='q2' for q in questions):
                    raise ValueError('bad question')
                return [{'id':q['id'],'num_cached_tokens':42,'decision':{'text':'yes'}}
                        for q in questions]
            def score_document(self, document):
                uncached.append(document['questions'][0]['id'])
                return [{'num_cached_tokens':0,'decision':{'text':'yes'}}]
        fake_io=types.ModuleType('docision.io');fake_io.DEFAULT_CONTEXT='context'
        with patch.dict(sys.modules,{'docision.io':fake_io}),patch.object(app,'ENGINE',Engine()):
            result=app.score_file({'name':'pages.pdf','data':base64.b64encode(raw.getvalue()).decode(),
                'questions':[{'question':f'Question {i}?','candidates':['yes','no']} for i in range(count)],'prefix_cache':enabled})
        self.assertTrue(all(ref() is None for ref in references))
        self.assertFalse(any(live_before_build), 'previous page cache leaked')
        for kwargs, size in calls:
            self.assertEqual(kwargs, {'mode':'causal','projection':'full','batch_size':size})
        return result,builds,cached,uncached

    def test_page_cache_reused_and_released_even_after_question_error(self):
        result,builds,cached,uncached=self.exercise(fail_question=True)
        self.assertEqual(len(builds),2);self.assertEqual(len(cached),12);self.assertEqual(uncached,[])
        self.assertNotEqual(builds[0],builds[1])
        self.assertEqual([r['question_id'] for r in result['rows'] if 'error' in r],['q2','q2'])
        self.assertTrue(all(r['result']['num_cached_tokens']==42 for r in result['rows'] if 'result' in r))
        self.assertTrue(all(p['status']=='reused' for p in result['cache_pages']))

    def test_whole_page_batch_succeeds_without_fallback(self):
        result,builds,cached,uncached=self.exercise()
        self.assertEqual(len(builds),2)
        self.assertEqual(len(cached),6)
        self.assertEqual(uncached,[])
        self.assertEqual(len(result['rows']),6)
        for row in result['rows']:
            self.assertNotIn('error',row)
            self.assertNotIn('batch_warning',row)
            self.assertEqual(row['inference_batch_size'],3)

    def test_disabled_and_single_question_skip_cache(self):
        for options in [{'enabled':False},{'count':1}]:
            result,builds,cached,uncached=self.exercise(**options)
            self.assertEqual(builds,[]);self.assertEqual(cached,[])
            self.assertEqual(len(uncached),len(result['rows']))
            self.assertTrue(all(r['result']['num_cached_tokens']==0 for r in result['rows']))

    def test_build_failure_falls_back_with_explicit_metadata(self):
        result,builds,cached,uncached=self.exercise(fail_build=True)
        self.assertEqual(len(builds),2);self.assertEqual(cached,[]);self.assertEqual(len(uncached),6)
        self.assertTrue(all(r['cache_status']=='fallback' and 'cache unavailable' in r['cache_warning'] for r in result['rows']))

    def test_cache_flag_requires_boolean(self):
        with patch.object(app,'ENGINE',object()),self.assertRaisesRegex(ValueError,'prefix_cache'):
            app.score_file({'question':'x','candidates':['a','b'],'prefix_cache':'false'})


class CachedBatchTests(unittest.TestCase):
    def test_batch_failure_retries_independently_and_keeps_order(self):
        from cached_inference import score_cached_page
        calls=[]
        class Engine:
            def score_questions(self, cache, specs, **kwargs):
                calls.append([s['id'] for s in specs])
                if any(s['id']=='q2' for s in specs):raise RuntimeError('q2 failed')
                return [{'id':s['id'],'decision':{'text':s['id']}} for s in specs]
        specs=[{'id':f'q{i}'} for i in range(1,7)]
        values=score_cached_page(Engine(),object(),specs,batch_size=4)
        self.assertEqual(len(values),6)
        self.assertEqual([x['result']['id'] if 'result' in x else 'ERROR' for x in values],['q1','ERROR','q3','q4','q5','q6'])
        self.assertEqual(calls,[['q1','q2','q3','q4'],['q1'],['q2'],['q3'],['q4'],['q5','q6']])
        self.assertEqual(values[-1]['inference_batch_size'],2)
        self.assertIn('batch_warning',values[0])

    def test_preprocessing_scope_checks_identity_settings_and_restores(self):
        from cached_inference import reuse_page_preprocessing
        from PIL import Image
        calls=[]
        class Processor:
            def _process_images(self, images, **kwargs):
                calls.append((images,kwargs));return {'pixels':object()},['image']
        processor=Processor();engine=types.SimpleNamespace(builder=types.SimpleNamespace(processor=processor),lock=threading.RLock())
        image=Image.new('RGB',(5,5),'white');other=Image.new('RGB',(5,5),'white')
        original=processor._process_images
        with self.assertRaisesRegex(RuntimeError,'exit'):
            with reuse_page_preprocessing(engine):
                first,repl=processor._process_images([image],size={'width':5})
                pixels=first.pop('pixels');repl.clear()
                second,repl2=processor._process_images([image],size={'width':5})
                self.assertIs(second['pixels'],pixels);self.assertEqual(repl2,['image'])
                processor._process_images([image],size={'width':6})
                processor._process_images([other],size={'width':5})
                self.assertEqual(len(calls),3)
                raise RuntimeError('exit')
        self.assertEqual(processor._process_images,original)
        with reuse_page_preprocessing(engine):processor._process_images([image],size={'width':5})
        self.assertEqual(len(calls),4)


class AdaptiveBatchTests(unittest.TestCase):
    def test_whole_page_first_then_halves_and_keeps_smaller_limit(self):
        import torch
        import cached_inference as ci
        calls=[]
        class Engine:
            def score_questions(self, cache, specs, **kwargs):
                calls.append([s['id'] for s in specs])
                if len(specs)>4:raise torch.cuda.OutOfMemoryError('test CUDA out of memory')
                return [{'id':s['id']} for s in specs]
        specs=[{'id':str(i)} for i in range(20)]
        with patch.object(ci,'_release_unused_cuda_memory') as release:
            rows=ci.score_cached_page(Engine(),object(),specs)
        self.assertEqual([len(c) for c in calls], [20,10,5]+[2]*10)
        self.assertEqual([r['result']['id'] for r in rows],[str(i) for i in range(20)])
        self.assertEqual(release.call_count,3)
        self.assertEqual([(x['from'],x['to']) for x in rows[-1]['batch_reductions']],[(20,10),(10,5),(5,2)])
        self.assertTrue(all(r['inference_batch_size']==2 for r in rows))

    def test_singleton_oom_terminates_and_next_question_is_processed(self):
        import torch
        import cached_inference as ci
        calls=[]
        class Engine:
            def score_questions(self, cache, specs, **kwargs):
                calls.append([s['id'] for s in specs])
                if any(s['id']=='bad' for s in specs):raise torch.cuda.OutOfMemoryError('test')
                return [{'id':s['id']} for s in specs]
        with patch.object(ci,'_release_unused_cuda_memory'):
            rows=ci.score_cached_page(Engine(),object(),[{'id':'bad'},{'id':'good'}])
        self.assertEqual(calls,[['bad','good'],['bad'],['good']])
        self.assertIn('error',rows[0]);self.assertEqual(rows[1]['result']['id'],'good')

    def test_full_page_success_does_not_split(self):
        from cached_inference import score_cached_page
        calls=[]
        class Engine:
            def score_questions(self, cache, specs, **kwargs):
                calls.append(kwargs['batch_size']);return [{'id':s['id']} for s in specs]
        rows=score_cached_page(Engine(),object(),[{'id':str(i)} for i in range(20)])
        self.assertEqual(calls,[20]);self.assertTrue(all(r['inference_batch_size']==20 and not r['batch_reductions'] for r in rows))

if __name__ == '__main__':
    unittest.main()
