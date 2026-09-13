"""Temporary single-GPU evaluation; saved predictions and guaranteed cleanup attempt.
Run with .venv/bin/python (the existing evaluation environment). No training.
"""
import argparse,csv,hashlib,json,signal,sys,time
from pathlib import Path
import requests
from fireworks_dedicated import ROOT,API,load_key,read_config,local_plan,scrub
from evaluate import SMOKE,normalize,summarize
from prepare_data import messages

RECORD=ROOT/'results/fireworks-inference-session.json'


def prompt(text):
    return '<|begin_of_text|>'+''.join('<|start_header_id|>'+m['role']+'<|end_header_id|>\n\n'+m['content']+'<|eot_id|>' for m in messages(text))+'<|start_header_id|>assistant<|end_header_id|>\n\n'


def stopped(api, resource):
    for _ in range(24):
        try:
            code,j=api.call('GET',resource)
            if code==404 or (code==200 and j.get('state')=='DELETED' and j.get('replicaCount',0)==0):
                return True
            if code==200: api.call('DELETE',resource+'?ignoreChecks=true')
        except Exception: pass
        time.sleep(5)
    return False


def evaluate(session,model,split,folder):
    output=ROOT/'results'/folder
    output.mkdir(exist_ok=False)
    if split=='smoke':tickets=[{'ticket_id':f'smoke-{i}','text':t,'label':l} for i,(t,l) in enumerate(SMOKE)]
    else:
        with (ROOT/'data'/f'{split}.csv').open() as f:tickets=list(csv.DictReader(f))
    signature={'backend':'fireworks','model':model,'split':split,'format':'raw Llama3 training renderer; no date insertion','prompt_sha256':hashlib.sha256(prompt('').encode()).hexdigest(),'dataset_sha256':hashlib.sha256(json.dumps(tickets,sort_keys=True).encode()).hexdigest(),'generation':{'temperature':0,'max_tokens':16},'sequential_requests':True,'warmup_excluded':True}
    (output/'run_config.json').write_text(json.dumps(signature,indent=2)+'\n')
    def infer(text):
        r=session.post('https://api.fireworks.ai/inference/v1/completions',json={'model':model,'prompt':prompt(text),'temperature':0,'max_tokens':16,'stream':False,'stop':['<|eot_id|>','<|end_of_text|>']},timeout=(15,60),allow_redirects=False)
        if r.status_code!=200:raise RuntimeError(f'Inference HTTP {r.status_code}: '+scrub(r.text,load_key())[:800])
        j=r.json();return j['choices'][0]['text'],j.get('usage',{})
    infer('Please install Adobe Acrobat on my computer.')
    rows=[]
    with (output/'predictions.jsonl').open('x') as f:
        for ticket in tickets:
            t=time.monotonic();raw,usage=infer(ticket['text'])
            row={**ticket,'raw_output':raw,'prediction':normalize(raw),'seconds':time.monotonic()-t,'usage':usage}
            rows.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
            if len(rows)%20==0:print(f'{folder}: {len(rows)}/{len(tickets)}',flush=True)
    result={**signature,**summarize(rows),'parsing':'case-insensitive exact label after whitespace trim; invalids are errors'}
    (output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'run':folder,**{k:result[k] for k in ('n','accuracy','macro_f1','invalid_outputs','latency_median_seconds')}}),flush=True)
    return result


def main():
    cfg=read_config();local_plan(cfg)
    spec=json.loads((ROOT/'configs/fireworks_inference.json').read_text())
    key=load_key();api=API(key);resource=f"accounts/{cfg['account']}/deployments/{spec['deployment_id']}"
    if RECORD.exists():raise ValueError('Session record exists; inspect it before any repeat deployment')
    code,_=api.call('GET',resource)
    if code!=404:raise ValueError('Deployment absence not established')
    # This signal bounds startup and all sequential inference requests; cleanup runs in finally.
    def deadline(*_):raise TimeoutError('Evaluation runtime deadline reached')
    signal.signal(signal.SIGALRM,deadline);signal.signal(signal.SIGTERM,deadline)
    signal.alarm(spec['max_runtime_seconds'])
    state={'deployment':resource,'status':'creating','started_at_unix':time.time(),'payload':spec['payload']}
    RECORD.write_text(json.dumps(state,indent=2)+'\n');attempted=False
    try:
        attempted=True
        code,j=api.call('POST',f"accounts/{cfg['account']}/deployments?deploymentId={spec['deployment_id']}&disableSpeculativeDecoding=true",spec['payload'])
        state['create_http_status']=code
        if code==409:attempted=False
        if code not in (200,201):raise RuntimeError(f'Deployment create HTTP {code}: {j}')
        if j.get('name')!=resource:raise ValueError('Unexpected deployment name')
        for field in ['acceleratorType','acceleratorCount','precision','maxReplicaCount']:
            if j.get(field)!=spec['payload'][field]:raise ValueError('Deployment configuration changed: '+field)
        print('Temporary one-H200 BF16 deployment accepted.',flush=True)
        start=time.monotonic();last=None
        while time.monotonic()-start<900:
            code,j=api.call('GET',resource)
            if code!=200:raise RuntimeError(f'Deployment status HTTP {code}')
            if j.get('state')!=last:print('Deployment: '+str(j.get('state')),flush=True);last=j.get('state')
            if j.get('state')=='READY':break
            if j.get('state') in ('FAILED','DELETED'):raise RuntimeError(str(j.get('status')))
            time.sleep(10)
        else:raise TimeoutError('GPU readiness timed out')
        state['ready_deployment']={k:j[k] for k in ('state','acceleratorType','acceleratorCount','precision','replicaCount') if k in j}
        tuned=f"accounts/{cfg['account']}/models/{cfg['output_model_id']}"
        code,j=api.call('POST',f"accounts/{cfg['account']}/deployedModels",{'model':tuned,'deployment':resource,'public':False,'serverless':False})
        if code not in (200,201):raise RuntimeError(f'Load adapter HTTP {code}: {j}')
        addon=j['name'];state['addon']=addon;start=time.monotonic()
        while time.monotonic()-start<300:
            code,j=api.call('GET',addon)
            if code==200 and j.get('state')=='DEPLOYED':break
            if code!=200:raise RuntimeError(f'Adapter status HTTP {code}')
            time.sleep(5)
        else:raise TimeoutError('Adapter loading timed out')
        print('Adapter loaded. Starting saved smoke, validation, and test evaluations.',flush=True)
        with requests.Session() as session:
            session.headers['Authorization']='Bearer '+key
            for label,model in [('baseline',cfg['base_model']),('tuned',tuned)]:
                smoke=evaluate(session,model+'#'+resource,'smoke',f'fireworks-{label}-smoke')
                # Record smoke errors as evidence; never rewrite labels or tune against test.
                if label=='tuned' and smoke['invalid_outputs']==5:raise ValueError('All smoke outputs invalid; inspect formatting before evaluation')
                evaluate(session,model+'#'+resource,'valid',f'fireworks-{label}-valid')
                evaluate(session,model+'#'+resource,'test',f'fireworks-{label}-test')
        state['status']='evaluation_complete'
    except BaseException as e:
        state['status']='failed';state['error']=scrub(str(e),key);raise
    finally:
        signal.alarm(0)
        state['cleanup_confirmed']=stopped(api,resource) if attempted else False
        state['elapsed_seconds']=time.time()-state['started_at_unix']
        RECORD.write_text(json.dumps(state,indent=2)+'\n')
        print('Deployment cleanup confirmed: '+str(state['cleanup_confirmed']),flush=True)
        if not state['cleanup_confirmed']:print('ACTION REQUIRED: inspect deployment '+resource,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');a=p.parse_args()
    if a.run:
        try:main()
        except BaseException as e:print(scrub(type(e).__name__+': '+str(e),load_key()),file=sys.stderr);sys.exit(1)
    else:print('Use --run to provision temporary inference and evaluate the saved models.')
