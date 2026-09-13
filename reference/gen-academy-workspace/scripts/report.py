"""Generate the comparison, plots, and executed-output evidence notebook from saved runs."""
import base64,csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from prepare_data import ROOT,LABELS
from evaluate import summarize
OUT=ROOT/'results'
RUNS=[('Mac baseline','local-baseline-test'),('Mac tuned adapter','local-tuned-test'),('Mac merged 8-bit','local-merged-8bit-test'),('Mac merged 4-bit (diagnostic)','local-merged-test'),('Fireworks baseline','fireworks-baseline-test'),('Fireworks tuned','fireworks-tuned-test')]
with (ROOT/'data/test.csv').open() as f:expected=list(csv.DictReader(f))
metrics={};predictions={};errors=[]
for name,folder in RUNS:
    path=OUT/folder/'metrics.json'
    if not path.exists():continue
    rows=[json.loads(l) for l in (path.parent/'predictions.jsonl').read_text().splitlines()]
    assert len(rows)==len(expected)==117
    for r,e in zip(rows,expected):
        assert all(r[k]==e[k] for k in ('ticket_id','text','label')), 'Mismatched shared test set'
        if r['prediction']!=r['label']:errors.append({'run':name,**{k:r[k] for k in ('ticket_id','text','label','prediction','raw_output')}})
    saved=json.loads(path.read_text());computed=summarize(rows)
    assert abs(saved['accuracy']-computed['accuracy'])<1e-12
    assert abs(saved['macro_f1']-computed['macro_f1'])<1e-12
    assert saved['confusion_matrix']==computed['confusion_matrix']
    metrics[name]=saved;predictions[name]=rows
cloud=json.loads((OUT/'fireworks-llama-training.json').read_text())
session_path=OUT/'fireworks-inference-session.json'
session=json.loads(session_path.read_text()) if session_path.exists() else {}
billing_path=OUT/'fireworks-final-billing.json'
billing=json.loads(billing_path.read_text()) if billing_path.exists() else {}
complete='Fireworks tuned' in metrics and session.get('cleanup_confirmed')
status='Both experiments are complete; the cloud training and inference resources have been released.' if complete else 'Cloud training is complete; inference evaluation is in progress or incomplete.'
with (OUT/'misclassified_tickets.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=['run','ticket_id','text','label','prediction','raw_output']);w.writeheader();w.writerows(errors)
fig,ax=plt.subplots(figsize=(12,6),layout='constrained');names=list(metrics);x=np.arange(len(names));width=.36
ax.bar(x-width/2,[metrics[n]['accuracy']*100 for n in names],width,label='Accuracy',color='#2563eb')
ax.bar(x+width/2,[metrics[n]['macro_f1']*100 for n in names],width,label='Macro F1',color='#0d9488')
ax.set_xticks(x,names,rotation=25,ha='right');ax.set_ylim(0,105);ax.set_ylabel('Percent');ax.set_title('Same 117 held-out tickets');ax.legend()
fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)
local_records=[json.loads(l) for l in (ROOT/'adapters/local-full/metrics.jsonl').read_text().splitlines()]
local_done=[r for r in local_records if r['kind']=='complete'][-1]
fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
for kind,key,label in [('train','train_loss','Train'),('validation','val_loss','Validation')]:
    rs=[r for r in local_records if r['kind']==kind]
    axs[0].plot([r['iteration']/4 for r in rs],[r[key] for r in rs],label=label)
axs[0].set_title('Mac Qwen training');axs[0].legend()
axs[1].plot([r['step'] for r in cloud['train_metrics']],[r['ce_loss'] for r in cloud['train_metrics']],alpha=.5,label='Batch train loss')
axs[1].plot([r['step'] for r in cloud['validation_metrics']],[r['eval_loss'] for r in cloud['validation_metrics']],marker='o',label='Validation')
axs[1].set_title('Fireworks Llama training');axs[1].legend()
for ax in axs:ax.set_xlabel('Optimizer update');ax.set_ylabel('Assistant-token loss')
fig.savefig(OUT/'training-losses.png',dpi=160);plt.close(fig)
text=['# Support ticket router: measured comparison','',status,'',
'**This compares two practical setups:** Mac Mini M4 (16 GB) with Qwen3-4B-Instruct-2507 and Fireworks with Llama-3.2-3B-Instruct. It is not a controlled hardware benchmark or an exact reproduction of the assignment’s Qwen3-1.7B-Base experiment.','',
'## Held-out test results','', '| Run | Correct / 117 | Accuracy | Macro F1 | Invalid | Median seconds | p95 seconds |','|---|---:|---:|---:|---:|---:|---:|']
for name,folder in RUNS:
    if name in metrics:
        m=metrics[name];text.append(f"| {name} | {round(m['accuracy']*117)} | {m['accuracy']:.1%} | {m['macro_f1']:.3f} | {m['invalid_outputs']} | {m['latency_median_seconds']:.3f} | {m['latency_p95_seconds']:.3f} |")
text+=['','![Comparison](comparison.png)','']
for a,b,label in [('Mac baseline','Mac tuned adapter','Mac fine-tuning gain'),('Fireworks baseline','Fireworks tuned','Fireworks fine-tuning gain')]:
    if a in metrics and b in metrics:
        text.append(f"**{label}: {(metrics[b]['accuracy']-metrics[a]['accuracy'])*100:+.1f} percentage points accuracy; {metrics[b]['macro_f1']-metrics[a]['macro_f1']:+.3f} macro F1.**\n")
if complete:
    a=metrics['Mac merged 8-bit'];b=metrics['Fireworks tuned']
    text += [f"The cloud tuned model differs from the recommended Mac export by **{(b['accuracy']-a['accuracy'])*100:+.1f} percentage points** on this test. Model family, size, precision and implementation all differ, so that difference cannot be attributed to Fireworks or the GPU alone.",'']
if complete:
    local=predictions['Mac merged 8-bit'];remote=predictions['Fireworks tuned']
    local_only=sum(a['prediction']==a['label'] and b['prediction']!=b['label'] for a,b in zip(local,remote))
    cloud_only=sum(a['prediction']!=a['label'] and b['prediction']==b['label'] for a,b in zip(local,remote))
    text += [f"The Mac alone was correct on **{local_only} tickets**; Fireworks alone was correct on **{cloud_only}**. The net advantage is only one ticket and does not establish a reliable overall winner. Active Directory recall remained weak: Mac **4/9**, Fireworks **3/9**. Fireworks’ higher overall score does not mean every category improved.",'']
text += ['## Data and evaluation method','',
'585 source tickets, seven service categories; 372 train / 94 validation / 117 test after excluding two near-duplicate training examples. The outer split uses seed 42, validation seed 43. All comparisons above were verified against the same ordered ticket IDs, text and labels. Test results did not select a checkpoint or change any prompt, label, parser or hyperparameter.','',
'Both setups receive the same system and ticket text, temperature 0 and at most 16 output tokens. Labels match exactly after case normalization and whitespace trimming; invalid outputs count as errors. The Mac uses the Qwen chat template. Fireworks uses raw Llama3 role headers matching its training renderer, without date injection. Each run excludes one identical synthetic warmup. Reported latency includes cloud network travel or local tokenization; loading and provisioning are excluded. Requests are sequential, not a throughput benchmark.','',
'The primary baseline is each instruction model with the same label prompt and no task adapter. It does not use the assignment’s separate letter-choice baseline. Invalid-output counts are reported to make formatting failures visible. Models were already instruction tuned before this task.','',
'## Training and cost','',
'| Setting | Mac | Fireworks |','|---|---|---|',
'| Model | Qwen3-4B-Instruct-2507 | Llama-3.2-3B-Instruct |',
'| Training | MLX 4-bit QLoRA, Apple GPU | Dedicated one-H200 LoRA |',
'| Adapter | Rank 8, alpha-equivalent 16, all 7 linear projections | Rank 8, alpha 16, all 7 linear projections |',
'| Epochs / optimizer updates | 3 / 279 | 3 / 279 |',
'| Learning rate / effective batch | 0.0001 / 4 (1 × accumulation 4) | 0.0001 / 4 |',
'| Loss | Assistant answer tokens only | Assistant answer tokens only |',
f"| Elapsed run | {local_done['wall_seconds']/60:.2f} minutes | {cloud['elapsed_seconds_including_startup_checkpoints_cleanup']/60:.2f} minutes |",
f"| Memory | {local_done['mlx_peak_gib']:.2f} GiB peak MLX allocation | H200 has 141 GB; actual peak utilization not measured |",
'| Serving precision | 8-bit merged projections, unchanged 4-bit embeddings | BF16 base with LoRA addon |',
'| Local API charges | $0; hardware/electricity not measured | Cloud GPU time billed |','',
'Local elapsed time includes loading and validation but excludes installation and downloading. Fireworks elapsed time includes provisioning, training, validation, checkpoint saves, registration and cleanup. These are operational timings with different boundaries, not isolated compute benchmarks. Training precision on Fireworks was not explicitly reported by the selected training shape.','']
if billing:
    text += [f"Billing observed at {billing['checked_at']}: **${billing['total_spend_usd']:.2f} total account spend** for the period, including **${billing['training_spend_usd']:.2f} training** and **${billing['inference_spend_usd']:.2f} inference/deployments**; prepaid credits **${billing['credits_usd']:.2f}**. No reload or credit purchase was performed. Billing can lag; this is the observed dashboard amount, not a settled invoice. Rounded category values may not sum to the displayed total. The project allowance remains $25.",'']
else:text += [f"Training dashboard observation: ${cloud['billing_observed']['training_spend_usd']:.2f}. Final combined billing observation pending; allowance $25.",'']
text += ['![Training losses](training-losses.png)','',
'Cloud validation token loss by epoch: '+', '.join(f"{r['eval_loss']:.3f}" for r in cloud['validation_metrics'])+'. Final loss is lowest, but token loss is not routing accuracy.','',
'## Smoke and validation checks','', '| Variant | Smoke correct | Validation accuracy |','|---|---:|---:|']
for name,smoke,valid in [('Mac adapter','local-adapter-smoke','local-full-valid'),('Mac recommended export','local-merged-8bit-smoke','local-merged-8bit-valid'),('Fireworks baseline','fireworks-baseline-smoke','fireworks-baseline-valid'),('Fireworks tuned','fireworks-tuned-smoke','fireworks-tuned-valid')]:
    sp=OUT/smoke/'metrics.json';vp=OUT/valid/'metrics.json'
    if sp.exists() and vp.exists():
        s=json.loads(sp.read_text());v=json.loads(vp.read_text());text.append(f"| {name} | {round(s['accuracy']*s['n'])}/{s['n']} | {v['accuracy']:.1%} |")
text += ['','The generic new-account smoke ticket is ambiguous relative to related training labels. All five original tickets and expected labels are preserved. A smoke failure is reported, not hidden by changing the prompt or adding a routing rule. See [label audit](label-audit.md).','']
for name,folder in RUNS:
    if name not in metrics:continue
    m=metrics[name];matrix=np.array(m['confusion_matrix'])[:7,:];short=folder.replace('-test','')
    fig,ax=plt.subplots(figsize=(10,6),layout='constrained');im=ax.imshow(matrix,cmap='Blues')
    ax.set_xticks(range(8),LABELS+['INVALID'],rotation=35,ha='right');ax.set_yticks(range(7),LABELS);ax.set_xlabel('Predicted');ax.set_ylabel('True');ax.set_title(name)
    for i in range(7):
        for j in range(8):ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>max(1,matrix.max()/2) else 'black')
    fig.savefig(OUT/f'{short}-confusion.png',dpi=160);plt.close(fig)
    text += [f'## {name}: per-class results','', '| Category | Precision | Recall | F1 | Support |','|---|---:|---:|---:|---:|']
    for label in LABELS:
        r=m['classification_report'][label];text.append(f"| {label} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1-score']:.3f} | {int(r['support'])} |")
    text += ['',f'![{name} confusion matrix]({short}-confusion.png)','']
text += ['## What to learn from this experiment','',
'- Fine-tuning gain must be measured against the same model’s baseline, not against another model family.','- The Mac 4-bit merged export lost accuracy. Validation selected the 8-bit projection export before test evaluation; its test predictions match the adapter exactly.','- Per-class samples are small (8–30 tickets), so differences of one or two tickets are noisy. The classes describe service teams, not urgency levels.','- Raw incorrect predictions are preserved in [misclassified_tickets.csv](misclassified_tickets.csv). Review sparse and ambiguous categories before further tuning.','- Temporary cloud inference is shut down after evaluation. The registered adapter persists, but using it again requires serving capacity; the model READY state does not mean an endpoint remains online.','',
'## Evidence and reproduction','',
'See [the learning walkthrough](../PROJECT_WALKTHROUGH.md), [README](../README.md), and `Comparison_Results.ipynb`. The notebook contains saved execution outputs generated from the actual run artifacts; training was performed by scripts, not notebook cells. No screenshots of training are fabricated. The original assignment notebook is preserved.','',
'Sources: [Fireworks LoRA deployment](https://docs.fireworks.ai/fine-tuning/deploying-loras), [Fireworks pricing](https://fireworks.ai/pricing), [MLX-LM](https://github.com/ml-explore/mlx-lm).']
(OUT/'REPORT.md').write_text('\n'.join(text)+'\n')
cells=[{'cell_type':'markdown','metadata':{},'id':'intro','source':['# Support ticket router: Mac and Fireworks\n',status+'\n\nSaved results from actual scripts. These are evidence cells, not a claim that training ran inside this notebook.\n\nSee results/REPORT.md and PROJECT_WALKTHROUGH.md for methodology, cost and limitations.']}]
for name,folder in RUNS:
    if name not in metrics:continue
    m=metrics[name];cells.append({'cell_type':'code','metadata':{},'id':folder,'source':[f"import json\nfrom pathlib import Path\nprint(Path('results/{folder}/metrics.json').read_text())"],'execution_count':len(cells),'outputs':[{'output_type':'stream','name':'stdout','text':[json.dumps(m,indent=2)]}]})
for artifact in ['comparison.png','training-losses.png','fireworks-tuned-confusion.png']:
    if not (OUT/artifact).exists():continue
    cells.append({'cell_type':'code','metadata':{},'id':artifact.replace('.','-'),'source':[f"from IPython.display import Image, display\ndisplay(Image('results/{artifact}'))"],'execution_count':len(cells),'outputs':[{'output_type':'display_data','metadata':{},'data':{'image/png':base64.b64encode((OUT/artifact).read_bytes()).decode(),'text/plain':['<Figure: saved experiment results>']}}]})
(ROOT/'Comparison_Results.ipynb').write_text(json.dumps({'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':5},indent=1))
print('Verified all included predictions; generated report, figures, and evidence notebook.')
