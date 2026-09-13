"""Fuse the selected adapter into a standalone 4-bit MLX model; re-evaluate it."""
import argparse,json,os,time
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
from mlx_lm import load
from mlx_lm.utils import save
from prepare_data import ROOT
p=argparse.ArgumentParser(); p.add_argument('--adapter',required=True);p.add_argument('--output',required=True);p.add_argument('--bits',type=int,choices=[4,8],default=4);a=p.parse_args()
os.chdir(ROOT)
if Path(a.output).exists(): raise SystemExit('Output exists; refusing to overwrite')
mx.set_memory_limit(9*2**30);mx.set_cache_limit(256*2**20)
start=time.monotonic()
model,tok,config=load('models/qwen3-4b-4bit',adapter_path=a.adapter,return_config=True,tokenizer_config={'trust_remote_code':False})
pending=[(name,layer) for name,layer in model.named_modules() if hasattr(layer,'fuse')]
count=len(pending)
assert count, 'No adapters found'
# Materialize and replace one projection at a time to bound merge memory.
while pending:
    name,layer=pending.pop()
    if a.bits==4:
        fused=layer.fuse(dequantize=False)
    else:
        linear=layer.fuse(dequantize=True)
        assert isinstance(linear,nn.Linear)
        fused=nn.QuantizedLinear.from_linear(linear,group_size=64,bits=8,mode='affine')
        config['quantization'][name]={'bits':8,'group_size':64,'mode':'affine'}
        del linear
    mx.eval(fused.parameters())
    model.update_modules(tree_unflatten([(name,fused)]))
    del layer,fused
    mx.clear_cache()
config['quantization_config']=config['quantization']
save(Path(a.output),'models/qwen3-4b-4bit',model,tok,config,donate_model=False)
Path(a.output,'merge_manifest.json').write_text(json.dumps({'adapter':a.adapter,'source_model':'models/qwen3-4b-4bit','requantized':True,'projection_bits':a.bits,'unchanged_embedding_bits':4,'group_size':64,'fused_modules':count,'seconds':time.monotonic()-start,'peak_mlx_gib':mx.get_peak_memory()/2**30},indent=2))
print('Saved standalone model:',a.output)
