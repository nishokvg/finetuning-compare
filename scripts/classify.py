"""Route one ticket with the standalone local model (no cloud/API key)."""
import argparse,json,os,time
import mlx.core as mx
from mlx_lm import load,stream_generate
from mlx_lm.sample_utils import make_sampler
from prepare_data import ROOT,messages
from evaluate import normalize
p=argparse.ArgumentParser();p.add_argument('ticket');p.add_argument('--model',default='models/local-router-merged-8bit');a=p.parse_args()
os.chdir(ROOT)
mx.set_memory_limit(9*2**30);mx.set_cache_limit(256*2**20)
model,tok=load(a.model,tokenizer_config={'trust_remote_code':False})
prompt=tok.apply_chat_template(messages(a.ticket),tokenize=False,add_generation_prompt=True)
start=time.monotonic()
raw=''.join(r.text for r in stream_generate(model,tok,prompt=prompt,max_tokens=16,sampler=make_sampler(0)))
print(json.dumps({'category':normalize(raw),'raw_output':raw,'seconds_excluding_model_load':round(time.monotonic()-start,3)},indent=2))
