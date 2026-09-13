"""Train reproducibly on development splits only, with memory and timing evidence."""
import argparse, json, os, time, types
from pathlib import Path
import numpy as np
import psutil
import mlx.core as mx
from mlx_lm import load
from mlx_lm.lora import CONFIG_DEFAULTS, train_model
from mlx_lm.tuner.datasets import ChatDataset
from mlx_lm.tuner.callbacks import TrainingCallback
ROOT=Path(__file__).resolve().parents[1]

class Recorder(TrainingCallback):
    def __init__(self,path):
        self.path=path; self.start=time.monotonic(); self.swap=psutil.swap_memory().used
    def record(self,kind,info):
        event={'kind':kind,**info,'wall_seconds':time.monotonic()-self.start,
               'mlx_peak_gib':mx.get_peak_memory()/2**30,
               'process_rss_gib':psutil.Process().memory_info().rss/2**30,
               'available_gib':psutil.virtual_memory().available/2**30,
               'swap_growth_gib':(psutil.swap_memory().used-self.swap)/2**30}
        with self.path.open('a') as f: f.write(json.dumps(event)+'\n')
        if event['mlx_peak_gib']>9 or event['swap_growth_gib']>3:
            raise RuntimeError('Memory guard stopped training; inspect metrics before retrying.')
    def on_train_loss_report(self,info): self.record('train',info)
    def on_val_loss_report(self,info): self.record('validation',info)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); args=p.parse_args()
    os.chdir(ROOT)
    config={**CONFIG_DEFAULTS,**json.loads(Path(args.config).read_text())}
    output=Path(config['adapter_path'])
    if output.exists(): raise SystemExit(f'Refusing to overwrite {output}; choose a new run directory.')
    output.mkdir(parents=True)
    recorder=Recorder(output/'metrics.jsonl')
    mx.set_cache_limit(256*2**20); mx.set_memory_limit(9*2**30)
    np.random.seed(config['seed']); mx.random.seed(config['seed'])
    model,tokenizer=load(config['model'],tokenizer_config={'trust_remote_code':False})
    datasets=[]; lengths={}
    for split in ('train','valid'):
        records=[json.loads(line) for line in (ROOT/'data'/f'{split}.jsonl').read_text().splitlines()]
        dataset=ChatDataset(records,tokenizer,mask_prompt=True)
        processed=[dataset.process(r) for r in records]
        assert all(0<offset<len(tokens) for tokens,offset in processed), 'Invalid assistant masking'
        lengths[split]={'rows':len(records),'max_tokens':max(len(t) for t,o in processed),
                       'total_tokens':sum(len(t) for t,o in processed)}
        assert lengths[split]['max_tokens']<=config['max_seq_length'], 'Would truncate tickets'
        datasets.append(dataset)
    (output/'token_counts.json').write_text(json.dumps(lengths,indent=2))
    print('Token counts:',json.dumps(lengths),flush=True)
    train_model(types.SimpleNamespace(**config),model,*datasets,training_callback=recorder)
    recorder.record('complete',{'microsteps':config['iters'],
                    'optimizer_updates':config['iters']//config['grad_accumulation_steps'],
                    'epochs':config['iters']*config['batch_size']/lengths['train']['rows']})

if __name__=='__main__': main()
