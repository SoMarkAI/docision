"""Submit existing document JSON files concurrently; no SDK or vLLM required."""
import argparse
import base64
import concurrent.futures
import io
import json
import urllib.request
from pathlib import Path
from PIL import Image


def payload(path):
    path=Path(path);source=json.loads(path.read_text(encoding='utf-8'))
    documents=source.get('documents',[source]);result=[]
    for doc in documents:
        image=doc.get('image',doc.get('media_path'))
        if not image.startswith('data:'):
            with Image.open(path.parent/image) as im:
                buffer=io.BytesIO();im.convert('RGB').save(buffer,format='PNG')
            image='data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()
        # Reference answers and unrelated metadata are never sent to inference.
        questions=[{k:q[k] for k in ['id','question','candidates'] if k in q}
                   for q in doc.get('questions',[doc])]
        result.append({'id':doc.get('id'),'image':image,'context':doc.get('context','Inspect the supplied media carefully.'),'questions':questions})
    return {'documents':result} if 'documents' in source else result[0]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',default='http://127.0.0.1:8000');p.add_argument('--inputs',nargs='+',required=True,type=Path)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--timeout',type=float,default=300);p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.workers<1 or a.timeout<=0:p.error('workers and timeout must be positive')
    def score(path):
        data=json.dumps(payload(path),ensure_ascii=False).encode()
        req=urllib.request.Request(a.url.rstrip('/')+'/v1/score',data=data,headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=a.timeout) as r:return {'input':str(path),'result':json.load(r)}
    with concurrent.futures.ThreadPoolExecutor(a.workers) as pool:results=list(pool.map(score,a.inputs))
    text=json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False)
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text+'\n',encoding='utf-8')
    print(text)


if __name__=='__main__':main()
