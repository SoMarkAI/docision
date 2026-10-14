"""Full-model source-release acceptance; creates only synthetic input files.
Run with .mcp-venv/bin/python release/smoke.py --model /path/to/full-model.
"""
import argparse
import asyncio
import base64
import csv
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from http_client import DocisionClient


def unpack(result):
    if result.isError:
        raise RuntimeError(str(result.content))
    return result.structuredContent or json.loads(result.content[0].text)


async def run(args, out, client):
    params = StdioServerParameters(command=sys.executable,
        args=[str(ROOT/'mcp_server.py'), '--api-url', f'http://127.0.0.1:{args.port}',
              '--token-file', str(out/'api-token'), '--allowed-root', str(out/'inputs'),
              '--output-dir', str(out/'exports'), '--model', args.model], env=dict(os.environ))
    questions = [dict(question='Does this page contain a table?', candidates=['Yes','No','Cannot determine']),
                 dict(question='Does this page contain a visible signature?', candidates=['Yes','No','Cannot determine'])]
    receipt = {}
    async with stdio_client(params) as (read, write):
        async with ClientSession(read,write) as session:
            receipt['protocol'] = (await session.initialize()).protocolVersion
            receipt['tools'] = [t.name for t in (await session.list_tools()).tools]
            assert len(receipt['tools']) == 8
            status = unpack(await session.call_tool('get_service_status',{}))
            assert not status['model']['loaded']
            unpack(await session.call_tool('load_model',{}))
            assert unpack(await session.call_tool('get_service_status',{}))['model']['loaded']
            async def complete(job_id):
                deadline = time.monotonic()+600
                while time.monotonic()<deadline:
                    state=unpack(await session.call_tool('get_job_status',{'job_id':job_id}))
                    if state['status']=='failed':raise RuntimeError(state)
                    if state['status']=='completed':return state
                    await asyncio.sleep(.2)
                raise TimeoutError(job_id)
            async def rows_for(job_id):
                rows=[]; offset=0
                while True:
                    page=unpack(await session.call_tool('get_results',{'job_id':job_id,'offset':offset,'limit':1}))
                    rows+=page['rows']
                    if page['next_offset'] is None:return rows
                    offset=page['next_offset']
            job=unpack(await session.call_tool('inspect_document',{'file_path':str(out/'inputs/synthetic.pdf'),'questions':questions}))
            state=await complete(job['id']);rows=await rows_for(job['id'])
            assert state['summary']=={'rows':4,'errors':0},state
            assert [(r['page'],r['question_id']) for r in rows]==[(p,q) for p in (1,2) for q in ('q1','q2')]
            assert all(r['result']['decision']['text'] in questions[i%2]['candidates'] for i,r in enumerate(rows))
            assert all(r['result']['num_cached_tokens']==0 for r in rows)
            assert all(r['numerics']=='native' and not r['prefix_cache_requested'] for r in rows)
            direct=client.request('POST','jobs',{'name':'synthetic.pdf','data':base64.b64encode((out/'inputs/synthetic.pdf').read_bytes()).decode(),'questions':questions})
            await complete(direct['id']);direct_rows=await rows_for(direct['id'])
            assert [r['result'] for r in direct_rows]==[r['result'] for r in rows]
            exports={}
            for fmt in ('json','csv'):
                exports[fmt]=unpack(await session.call_tool('export_results',{'job_id':job['id'],'format':fmt}))
            assert json.loads(Path(exports['json']['path']).read_text())==rows
            csv_rows=list(csv.DictReader(io.StringIO(Path(exports['csv']['path']).read_text(encoding='utf-8-sig'))))
            assert len(csv_rows)==4
            assert [r['模型判断'] for r in csv_rows]==[r['result']['decision']['text'] for r in rows]
            image_job=unpack(await session.call_tool('inspect_document',{'file_path':str(out/'inputs/synthetic.png'),'questions':questions}))
            assert (await complete(image_job['id']))['summary']=={'rows':2,'errors':0}
            corrupt=unpack(await session.call_tool('inspect_document',{'file_path':str(out/'inputs/broken.pdf'),'questions':questions}))
            assert (await complete(corrupt['id']))['summary']['errors']==2
            denied=await session.call_tool('inspect_document',{'file_path':str(out/'outside.pdf'),'questions':questions})
            assert denied.isError
            if os.name!='nt':
                link=out/'inputs/outside-link.pdf';link.symlink_to(out/'outside.pdf')
                assert (await session.call_tool('inspect_document',{'file_path':str(link),'questions':questions})).isError
            unpack(await session.call_tool('delete_job',{'job_id':job['id']}))
            assert (await session.call_tool('get_job_status',{'job_id':job['id']})).isError
            receipt.update(status='FULL_MODEL_RELEASE_SMOKE_PASS',pdf_rows=rows,image_rows=2,
                http_mcp_results_identical=True,pagination_verified=True,csv_json_verified=True,
                malformed_pdf_isolated=True,outside_root_denied=True,delete_verified=True,exports=exports,
                accuracy_claim='None: synthetic functional acceptance, not a benchmark.')
    (out/'ACCEPTANCE.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(receipt['status'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True)
    parser.add_argument('--port',type=int,default=17886)
    parser.add_argument('--output',default='validation/release-smoke')
    args=parser.parse_args()
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'inputs').mkdir()
    python=ROOT/'.venv'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    fixture_code='''import sys,torch,json
from pathlib import Path
from PIL import Image,ImageDraw
out=Path(sys.argv[1]);torch.manual_seed(0)
x=torch.randn(8,8,device="cuda:0"); y=x@x
assert y.isfinite().all();print("CUDA_WITNESS",tuple(y.shape),torch.cuda.get_device_name())
a=Image.new("RGB",(600,400),"white");d=ImageDraw.Draw(a)
d.text((30,20),"SYNTHETIC TEST DOCUMENT - NO REAL PERSON",fill="black")
for x in (30,200,400,560):d.line((x,80,x,280),fill="black",width=2)
for y in (80,130,180,230,280):d.line((30,y,560,y),fill="black",width=2)
b=Image.new("RGB",(600,400),"white");ImageDraw.Draw(b).text((30,40),"SYNTHETIC BLANK FORM. Signature: __________",fill="black")
a.save(out/"inputs/synthetic.png");a.save(out/"inputs/synthetic.pdf",save_all=True,append_images=[b])
(out/"inputs/broken.pdf").write_bytes(b"not a PDF")
(out/"outside.pdf").write_bytes(b"not a PDF")
'''
    subprocess.run([str(python),'-c',fixture_code,str(out)],check=True)
    client=DocisionClient(f'http://127.0.0.1:{args.port}',str(out/'api-token'))
    with (out/'app.log').open('w') as log:
        proc=subprocess.Popen([str(python),'-B',str(ROOT/'bootstrap.py'),'--port',str(args.port),'--no-browser',
                               '--token-file',str(out/'api-token')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            for _ in range(600):
                if proc.poll() is not None:raise RuntimeError('Launcher exited; inspect private app.log')
                try:
                    if client.request('GET','health')['status']=='ok':break
                except Exception:pass
                time.sleep(.2)
            else:raise TimeoutError('Launcher readiness')
            asyncio.run(run(args,out,client))
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=30)


if __name__=='__main__':
    main()
