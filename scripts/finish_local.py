"""Validate the fixed three-epoch adapter, fuse it, then evaluate the untouched test."""
import json,subprocess,sys
from pathlib import Path
from prepare_data import ROOT

def run(*args):
    subprocess.run([sys.executable,*args],cwd=ROOT,check=True)

def evaluate(name,model='models/qwen3-4b-4bit',adapter=None,split='valid'):
    out=ROOT/'results'/name
    if not (out/'metrics.json').exists():
        args=['scripts/evaluate.py','--backend','mlx','--model',model,'--split',split,'--output',str(out)]
        if adapter: args+=['--adapter',str(adapter)]
        if out.exists(): args+=['--resume']
        run(*args)
    return json.loads((out/'metrics.json').read_text())

train=ROOT/'adapters/local-full'
metrics=[json.loads(line) for line in (train/'metrics.jsonl').read_text().splitlines()]
assert metrics[-1]['kind']=='complete', 'Full training must complete before evaluation'
# Predeclared comparison uses the final three-epoch model on both platforms.
# Validation is diagnostic; no test-guided checkpoint selection.
best={'epoch':3,'adapter':'adapters/local-full'}
evaluate('local-full-valid',adapter=best['adapter'])
(ROOT/'results/local-selection.json').write_text(json.dumps({'rule':'Fixed final three-epoch checkpoint, matching the planned cloud run.','selected':best},indent=2))
if not (ROOT/'models/local-router-merged').exists(): run('scripts/merge_local.py','--adapter',best['adapter'],'--output','models/local-router-merged')
merged=evaluate('local-merged-valid',model='models/local-router-merged')
evaluate('local-merged-smoke',model='models/local-router-merged',split='smoke')
if not (ROOT/'models/local-router-merged-8bit').exists(): run('scripts/merge_local.py','--adapter',best['adapter'],'--output','models/local-router-merged-8bit','--bits','8')
evaluate('local-merged-8bit-valid',model='models/local-router-merged-8bit')
evaluate('local-merged-8bit-smoke',model='models/local-router-merged-8bit',split='smoke')
# All configuration and selection are now frozen; retain both adapter and merged
# scores so merging/requantization effects are visible, without tuning on test.
evaluate('local-baseline-test',split='test')
evaluate('local-tuned-test',adapter=best['adapter'],split='test')
evaluate('local-merged-test',model='models/local-router-merged',split='test')
evaluate('local-merged-8bit-test',model='models/local-router-merged-8bit',split='test')
print('Local evaluation complete; see results/local-selection.json and *-test/metrics.json.')
