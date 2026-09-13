"""Identical prompts and scoring for local/Fireworks baseline and tuned models."""
import argparse,csv,hashlib,json,os,time
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score
from prepare_data import ROOT,LABELS,messages
INVALID='INVALID'
SMOKE=[('Please create a new user account for the new employee starting Monday.','Active Directory'),('I cannot access the shared network folder — I keep getting Access Denied.','Fileservice'),("Outlook keeps asking me to re-enter my password and Teams won't sync.",'O365'),("We're ready to retire [SERVER] — please remove it from monitoring.",'EOL'),('Adobe Acrobat fails to install, error code 1603.','Software')]

def normalize(raw):
    return {l.casefold():l for l in LABELS}.get(raw.strip().casefold(),INVALID)

def summarize(rows):
    truth=[r['label'] for r in rows]; pred=[r['prediction'] for r in rows]
    latency=[r['seconds'] for r in rows]
    return {'n':len(rows),'accuracy':accuracy_score(truth,pred),
            'macro_f1':f1_score(truth,pred,labels=LABELS,average='macro',zero_division=0),
            'invalid_outputs':pred.count(INVALID),'latency_median_seconds':float(np.median(latency)),
            'latency_p95_seconds':float(np.percentile(latency,95)),
            'classification_report':classification_report(truth,pred,labels=LABELS,target_names=LABELS,output_dict=True,zero_division=0),
            'confusion_labels':LABELS+[INVALID],
            'confusion_matrix':confusion_matrix(truth,pred,labels=LABELS+[INVALID]).tolist()}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--backend',choices=['mlx','fireworks'],required=True)
    p.add_argument('--model',required=True); p.add_argument('--adapter'); p.add_argument('--split',choices=['valid','test','smoke'],required=True)
    p.add_argument('--output',required=True); p.add_argument('--limit',type=int); p.add_argument('--resume',action='store_true'); a=p.parse_args(); os.chdir(ROOT)
    output=Path(a.output)
    if output.exists() and not a.resume: raise SystemExit(f'Refusing to overwrite {output}; use --resume only for the same run.')
    output.mkdir(parents=True,exist_ok=True)
    if a.split=='smoke': tickets=[{'ticket_id':f'smoke-{i}','text':t,'label':l} for i,(t,l) in enumerate(SMOKE)]
    else:
        with (ROOT/'data'/f'{a.split}.csv').open() as f: tickets=list(csv.DictReader(f))
    if a.limit: tickets=tickets[:a.limit]
    signature={k:v for k,v in vars(a).items() if k not in ('resume','output')}
    signature['dataset_sha256']=hashlib.sha256(json.dumps(tickets,sort_keys=True).encode()).hexdigest()
    signature['prompt_sha256']=hashlib.sha256(json.dumps(messages('')).encode()).hexdigest()
    signature['generation']={'max_tokens':16,'temperature':0}
    if a.adapter: signature['adapter_sha256']=hashlib.sha256((Path(a.adapter)/'adapters.safetensors').read_bytes()).hexdigest()
    signature_path=output/'run_config.json'
    if signature_path.exists():
        assert json.loads(signature_path.read_text())==signature, 'Resume inputs/configuration differ'
    elif a.resume and (output/'predictions.jsonl').exists():
        raise SystemExit('Cannot resume legacy output without a configuration signature')
    else: signature_path.write_text(json.dumps(signature,indent=2))
    rows=[]
    predictions=output/'predictions.jsonl'
    if a.resume and predictions.exists():
        rows=[json.loads(line) for line in predictions.read_text().splitlines()]
        assert len(rows)<=len(tickets)
        for old,expected in zip(rows,tickets):
            assert all(old[k]==expected[k] for k in expected), 'Resume ticket order/content differs'
    if a.backend=='mlx':
        import mlx.core as mx
        from mlx_lm import load,stream_generate
        from mlx_lm.sample_utils import make_sampler
        mx.set_memory_limit(9*2**30); mx.set_cache_limit(256*2**20)
        model,tok=load(a.model,adapter_path=a.adapter,tokenizer_config={'trust_remote_code':False})
        def infer(text):
            prompt=tok.apply_chat_template(messages(text),tokenize=False,add_generation_prompt=True)
            chunks=list(stream_generate(model,tok,prompt=prompt,max_tokens=16,sampler=make_sampler(0)))
            last=chunks[-1]
            return ''.join(c.text for c in chunks),{'prompt_tokens':last.prompt_tokens,'completion_tokens':last.generation_tokens}
    else:
        import requests
        key=os.environ.get('FIREWORKS_API_KEY')
        if not key and (ROOT/'.env').exists():
            for line in (ROOT/'.env').read_text().splitlines():
                if line.startswith('FIREWORKS_API_KEY='): key=line.split('=',1)[1].strip().strip('\"\'')
        if not key: raise SystemExit('Set FIREWORKS_API_KEY in the environment or local .env.')
        session=requests.Session(); session.headers['Authorization']=f'Bearer {key}'
        def infer(text):
            response=session.post('https://api.fireworks.ai/inference/v1/chat/completions',json={'model':a.model,'messages':messages(text),'temperature':0,'max_tokens':16,'stream':False},timeout=(20,90))
            if response.status_code!=200: raise RuntimeError(f'Fireworks inference HTTP {response.status_code}; no retry or extra spend.')
            result=response.json(); return result['choices'][0]['message']['content'],result.get('usage',{})
    # One identical synthetic warmup, excluded from reported latency.
    infer('Please install Adobe Acrobat on my computer.')
    completed=len(rows)
    with predictions.open('a' if a.resume else 'w') as f:
        for i,ticket in enumerate(tickets[completed:],completed):
            start=time.monotonic(); raw,usage=infer(ticket['text']); seconds=time.monotonic()-start
            row={**ticket,'raw_output':raw,'prediction':normalize(raw),'seconds':seconds,'usage':usage}
            f.write(json.dumps(row,ensure_ascii=False)+'\n'); f.flush(); rows.append(row)
            if (i+1)%10==0: print(f'{i+1}/{len(tickets)} evaluated',flush=True)
    result={**vars(a),**summarize(rows),'max_new_tokens':16,'temperature':0,'parsing':'case-insensitive exact label after whitespace trim; invalids are errors'}
    (output/'metrics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('classification_report','confusion_matrix')},indent=2))

if __name__=='__main__': main()
