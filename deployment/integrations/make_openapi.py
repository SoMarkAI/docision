"""Generate the bundled OpenAPI contract using only the standard library."""
import json
from pathlib import Path
Q={'type':'object','required':['question','candidates'],'properties':{'question':{'type':'string','minLength':1,'maxLength':8000},'candidates':{'type':'array','minItems':2,'maxItems':26,'uniqueItems':True,'items':{'type':'string','minLength':1,'maxLength':2000}}}}
D={'type':'object','required':['name','data','questions'],'properties':{'name':{'type':'string'},'data':{'type':'string','format':'byte','description':'Base64 document bytes; decoded size <=25 MiB'},'questions':{'type':'array','minItems':1,'maxItems':20,'items':{'$ref':'#/components/schemas/Question'}},'prefix_cache':{'type':'boolean','default':False}}}
def operation(name,body=None,code='200',description='Success'):
 x={'operationId':name,'responses':{code:{'description':description},'400':{'description':'Invalid input'},'403':{'description':'Invalid credential, Host or Origin'},'409':{'description':'Busy, model not loaded or results not ready'},'413':{'description':'Size limit exceeded'},'429':{'description':'Queue full'}}}
 if body:x['requestBody']={'required':True,'content':{'application/json':{'schema':body}}}
 return x
paths={}
for route,method,name,body,code in [
 ('health','get','health',None,'200'),('model','get','model_status',None,'200'),
 ('model','post','load_model',{'type':'object','required':['model'],'properties':{'model':{'type':'string','description':'Path on HTTP server host'},'device':{'type':'string','default':'cuda:0'},'numerics':{'type':'string','enum':['native','stable'],'default':'native'}}},'200'),
 ('templates','get','templates',None,'200'),('score','post','score_sync',{'$ref':'#/components/schemas/Document'},'200'),
 ('jobs','post','submit_job',{'$ref':'#/components/schemas/Document'},'202'),
 ('jobs/{job_id}','get','job_status',None,'200'),('jobs/{job_id}','delete','delete_completed_job',None,'200'),
 ('jobs/{job_id}/results','get','job_results',None,'200'),('jobs/{job_id}/export','get','job_export',None,'200'),
 ('csv','post','rows_to_csv',{'type':'object','required':['rows'],'properties':{'rows':{'type':'array','items':{'type':'object'}}}},'200'),
 ('openapi.json','get','openapi',None,'200')]:
 op=operation(name,body,code)
 if '{job_id}' in route:op['parameters']=[{'name':'job_id','in':'path','required':True,'schema':{'type':'string','pattern':'^[0-9a-f]{32}$'}}]
 if route.endswith('/results'):op['parameters'] += [{'name':'offset','in':'query','schema':{'type':'integer','minimum':0,'default':0}},{'name':'limit','in':'query','schema':{'type':'integer','minimum':1,'maximum':200,'default':50}}]
 if route.endswith('/export'):op['parameters'] += [{'name':'format','in':'query','schema':{'type':'string','enum':['json','csv'],'default':'json'}}]
 paths.setdefault('/api/v1/'+route,{})[method]=op
spec={'openapi':'3.0.3','info':{'title':'Docision Local HTTP API','version':'1.0.0','description':'Loopback-only. Async jobs are in memory for at most one hour / latest 32 completed jobs; process restart clears jobs. One worker, 4 pending uploads. JSON requests <=36 MiB. Exports <=16 MiB; use pagination for larger results. Completed jobs may contain per-question errors.'},'servers':[{'url':'http://127.0.0.1:7860'}],'security':[{'bearerAuth':[]},{'sessionToken':[]}],'paths':paths,'components':{'securitySchemes':{'bearerAuth':{'type':'http','scheme':'bearer'},'sessionToken':{'type':'apiKey','in':'header','name':'X-Docision-Token'}},'schemas':{'Question':Q,'Document':D}}}
schemas=spec['components']['schemas']
schemas['Row']={'type':'object','properties':{'file':{'type':'string'},'page':{'type':'integer'},'question_id':{'type':'string'},'question':{'type':'string'},'candidates':{'type':'array','items':{'type':'string'}},'correction':{'type':'string'},'result':{'type':'object','description':'Decision, candidate probabilities and runtime metadata'},'error':{'type':'string'}}}
schemas['Job']={'type':'object','required':['id','status','created_at','progress'],'properties':{'id':{'type':'string'},'status':{'type':'string','enum':['queued','running','completed','failed']},'created_at':{'type':'number'},'started_at':{'type':'number'},'finished_at':{'type':'number'},'progress':{'type':'object','properties':{'pages_done':{'type':'integer'},'pages_total':{'type':'integer','nullable':True},'rows_done':{'type':'integer'}}},'summary':{'type':'object','properties':{'rows':{'type':'integer'},'errors':{'type':'integer'}}},'error':{'type':'string'}}}
schemas['Error']={'type':'object','required':['error'],'properties':{'error':{'type':'object','required':['code','message'],'properties':{'code':{'type':'string'},'message':{'type':'string'}}}}}
responses={
'health':{'type':'object','properties':{'status':{'type':'string'},'api_version':{'type':'string'},'model_loaded':{'type':'boolean'},'busy':{'type':'boolean'},'job_ttl_seconds':{'type':'integer'}}},
'model_status':{'type':'object','properties':{'loaded':{'type':'boolean'},'model':{'type':'string','nullable':True},'device':{'type':'string','nullable':True},'numerics':{'type':'string','enum':['native','stable']},'busy':{'type':'boolean'}}},
'load_model':{'type':'object','properties':{'message':{'type':'string'}}},
'job_status':{'$ref':'#/components/schemas/Job'},'submit_job':{'$ref':'#/components/schemas/Job'},
'job_results':{'type':'object','properties':{'job_id':{'type':'string'},'total':{'type':'integer'},'offset':{'type':'integer'},'next_offset':{'type':'integer','nullable':True},'rows':{'type':'array','items':{'$ref':'#/components/schemas/Row'}}}},
'score_sync':{'type':'object','properties':{'rows':{'type':'array','items':{'$ref':'#/components/schemas/Row'}},'cache_pages':{'type':'array','items':{'type':'object'}},'elapsed_seconds':{'type':'number'}}},
'rows_to_csv':{'type':'object','properties':{'csv':{'type':'string'}}},
'delete_completed_job':{'type':'object','properties':{'deleted':{'type':'string'}}},
'job_export':{'type':'array','items':{'$ref':'#/components/schemas/Row'}},'templates':{'type':'object','additionalProperties':{'type':'array','items':{}}},'openapi':{'type':'object'}}
for methods in paths.values():
 for op in methods.values():
  code='202' if op['operationId']=='submit_job' else '200'
  op['responses'][code]['content']={'application/json':{'schema':responses[op['operationId']]}}
  if op['operationId']=='job_export':op['responses'][code]['content']['text/csv']={'schema':{'type':'string'}}
  for error_code in ['400','403','404','409','413','415','429','500']:
   op['responses'].setdefault(error_code,{'description':'API error'})['content']={'application/json':{'schema':{'$ref':'#/components/schemas/Error'}}}
(Path(__file__).resolve().parents[2]/'docision/local_app/assets/openapi.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n')
