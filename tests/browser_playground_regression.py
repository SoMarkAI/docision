"""Browser regression for model readiness. Requires playwright + chromium.
Run: python tests/browser_playground_regression.py
No GPU or model is required; the HTTP fixture supplies controlled model states.
"""
import json
import os
from pathlib import Path
import threading
import unittest
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from playwright.sync_api import sync_playwright

HTML=Path(__file__).resolve().parents[1]/'docision/local_app/assets/index.html'

class StartupTests(unittest.TestCase):
    def test_startup_states(self):
        cases=[('stable_preloaded',{'loaded':True,'model':'/ready/model','device':'cuda:0','numerics':'stable'},None),('preloaded',{'loaded':True,'model':'/ready/model','device':'cuda:0'},None),
               ('stale_saved_model',{'loaded':True,'model':'/ready/model','device':'cuda:7'},'/old/model'),
               ('unloaded',{'loaded':False,'model':None,'device':None},'/saved/model'),
               ('status_error',None,None)]
        with sync_playwright() as pw:
            browser=pw.chromium.launch(headless=True, executable_path=os.environ.get("DOCISION_CHROMIUM_EXECUTABLE"))
            try:
                for label,model_state,saved in cases:
                    with self.subTest(case=label):
                        calls=[]
                        class Handler(BaseHTTPRequestHandler):
                            def log_message(self,*_):pass
                            def do_GET(self):
                                calls.append(self.path)
                                code=200
                                if self.path=='/':data=HTML.read_bytes();kind='text/html'
                                else:
                                    kind='application/json'
                                    if self.path.endswith('/templates'):value={'文件分类':['Table?',['Yes','No']]}
                                    elif model_state is None:code=503;value={'error':{'message':'status unavailable'}}
                                    else:value=model_state
                                    data=json.dumps(value).encode()
                                self.send_response(code);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
                        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
                        context=browser.new_context()
                        if saved:context.add_init_script('localStorage.setItem("docision.model",'+json.dumps(saved)+');')
                        try:
                            page=context.new_page();page.goto(f'http://127.0.0.1:{server.server_port}/#test-token');page.wait_for_load_state('networkidle')
                            self.assertIn('/api/v1/model',calls)
                            if model_state and model_state['loaded']:
                                self.assertTrue(page.locator('#run').is_enabled())
                                self.assertEqual(page.locator('#numerics').input_value(),model_state.get('numerics','native'))
                                self.assertFalse(page.locator('#prefixCache').is_checked())
                                self.assertEqual(page.locator('#prefixCache').is_enabled(),model_state.get('numerics')=='stable')
                                self.assertEqual(page.locator('#model').input_value(),model_state['model'])
                                self.assertEqual(page.locator('#device').input_value(),model_state['device'])
                            else:
                                self.assertTrue(page.locator('#run').is_disabled())
                                if model_state:self.assertEqual(page.locator('#model').input_value(),saved)
                                else:self.assertIn('status unavailable',page.locator('#status').inner_text())
                        finally:context.close();server.shutdown();server.server_close();thread.join()
            finally:browser.close()

if __name__=='__main__':unittest.main()
