# Support ticket router: Mac MLX and Fireworks

Fine-tune a seven-category IT ticket router and measure the improvement against each platform's own untrained baseline. This extends the Gen Academy Week 5 assignment. The original Colab notebook and source CSV remain available.

## Completed results

| Setup | Baseline accuracy | Tuned accuracy | Tuned macro F1 |
|---|---:|---:|---:|
| Mac Qwen 4B | 44.4% | 82.9% (97/117) | 0.794 |
| Fireworks Llama 3B | 41.0% | 83.8% (98/117) | 0.810 |

Both use the same 117 held-out tickets. The one-ticket difference does not establish a reliable winner. Both tuned models pass 4/5 original smoke tests, with the ambiguous account-creation ticket still failing; see the [label audit](results/label-audit.md). Training took 24.96 minutes locally and 9.93 minutes on Fireworks, with different timing boundaries documented in the report. The cloud training and inference resources are deleted; the registered adapter is retained.

**Start with [PROJECT_WALKTHROUGH.md](PROJECT_WALKTHROUGH.md)** for every local and cloud step. The [measured comparison](results/REPORT.md) includes all scores, per-class results, cost observations and limitations.

Start with [the local demo guide](LOCAL_DEMO.md) to use the model already installed on this Mac. Open [the measured report](results/REPORT.md) or `Comparison_Results.ipynb` for the confusion matrices and per-class metrics. Run `.venv/bin/python -m unittest discover -s tests -v` to check the evaluator.

## Models and comparison

The comparison uses **Qwen3-4B-Instruct-2507 locally and Llama-3.2-3B-Instruct on Fireworks**:

| Setting | Mac Mini M4, 16 GB | Fireworks |
|---|---|---|
| Source | `Qwen/Qwen3-4B-Instruct-2507` | `accounts/fireworks/models/llama-v3p2-3b-instruct` |
| Local source revision | `cdbee75f17c01a7cc42f958dc650907174af0554` | Provider-managed revision; record when available |
| Weights | 4-bit affine base, group size 64 | BF16 inference base; training precision not explicitly reported |
| Training | MLX QLoRA | Dedicated one-H200 LoRA SFT |
| Epochs / rank / learning rate | 3 / 8 / 0.0001 | 3 / 8 / 0.0001 |
| Effective batch | 1 ticket × 4 accumulation steps | 4 samples per optimizer step |
| Context limit | 1024 | 1024 |
| Loss | Assistant response only | Assistant-only mask verified on all 466 train/validation examples |
| Adapter details | All linear projections in all 36 blocks; MLX scale 2, dropout 0 | All 7 projection types; rank 8, alpha 16 |

This is a comparison of practical training/serving setups. Quantization, optimizer implementation, batching order, adapter details, and provider model revision can still explain differences. It is not a controlled GPU-hardware benchmark. The assignment's original `Qwen/Qwen3-1.7B-Base` is a different, smaller base checkpoint; do not call the 4B experiment a reproduction of that original model.

## Shared data

`data/manifest.json` records the source hash, split hashes, seeds, prompt, labels, and near-duplicate audit.

- Source: 585 labeled tickets, seven classes.
- Train: **372**, validation: **94**, final test: **117**.
- The 117-ticket outer split reproduces the notebook's shuffled stratified 80/20 split with seed 42. Validation is carved from development with seed 43.
- Two training examples are almost identical to a test example. They are excluded from training on **both** platforms. Connected groups with character-TFIDF similarity ≥0.90 are kept out of cross-split evaluation by excluding lower-priority examples (`test > valid > train`).
- Upload only `train.jsonl` and `valid.jsonl` to the training workflow. `test.jsonl` is never a training/evaluation dataset for checkpoint selection.
- The fixed three-epoch checkpoint is used for the primary comparison; test results do not guide model selection.

## Run locally

Commands below run from this repository. This run uses Homebrew Python 3.14 on macOS 26.6.2. Installation and downloading are required only once.

```sh
/opt/homebrew/bin/python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/prepare_data.py
```

The current workspace already has the official source model in `models/qwen3-4b-source` and its converted version in `models/qwen3-4b-4bit`. These large generated files are excluded from Git. To download the pinned source and rebuild the quantized model:

```sh
.venv/bin/hf download Qwen/Qwen3-4B-Instruct-2507 --revision cdbee75f17c01a7cc42f958dc650907174af0554 --local-dir models/qwen3-4b-source
.venv/bin/mlx_lm.convert --hf-path models/qwen3-4b-source --mlx-path models/qwen3-4b-4bit -q --q-bits 4 --q-group-size 64
```

Train and finish the local experiment:

```sh
.venv/bin/python scripts/train_local.py --config configs/local_pilot.json
.venv/bin/python scripts/train_local.py --config configs/local_full.json
.venv/bin/python scripts/finish_local.py
.venv/bin/python scripts/report.py
```

The scripts preserve the original 4-bit export for diagnosis and also build the higher-precision export. Training refuses to overwrite existing adapter directories. To repeat an experiment, copy a config and change `adapter_path`. `finish_local.py` reuses completed evaluation outputs; use fresh output directories if changing any input. Interrupted evaluations can use `--resume`; input/configuration signatures are checked before resuming. Preserve the original run evidence.

The pilot is 80 microsteps = 20 optimizer updates, about 0.215 epochs. The full run is 1,116 microsteps = 279 updates = exactly three epochs over 372 tickets. MLX's reported “trained tokens” count assistant-loss tokens; total input tokens are recorded separately in `token_counts.json`.

`train_local.py` writes loss, wall time, MLX peak memory, available RAM, and system swap growth. It stops at report boundaries if MLX peak exceeds 9 GiB or system swap grows by more than 3 GiB. This is a conservative guard, not an operating-system memory reservation. The trainer only reads train/validation files.

## Try your own ticket

After the full run and merge complete, classification uses the recommended 8-bit projection export entirely on your Mac:

```sh
.venv/bin/python scripts/classify.py "Please install Adobe Acrobat on my computer."
```

For a short Loom demo, show this command, the five smoke-test results, and `Comparison_Results.ipynb`. Explain the shared split and compare the measured baseline and tuned accuracy. See the report for the completed cloud metrics and resource state. [PROJECT_WALKTHROUGH.md](PROJECT_WALKTHROUGH.md) explains each local and cloud step.

## Fireworks training and inference

The cloud experiment uses Llama after the Qwen B200 quota restriction and managed Llama SFT rejection. The dedicated Training API succeeded; the failed managed submission is retained as history, not the final status.

- Training configuration: `configs/fireworks_llama_dedicated.json`.
- Tokenizer: authorized Meta download, locally converted and checked; see `results/meta-tokenizer-conversion.json`.
- Training runner: `scripts/fireworks_dedicated.py`, isolated `.venv-fireworks` dependencies.
- Training results: `results/fireworks-llama-training.json` (279 updates, three epochs).
- Final adapter: `accounts/nishokvg-myac5qmcmir/models/support-router-llama3b-dedicated-01`.
- Inference runner: `scripts/finish_fireworks.py`, using the existing `.venv` evaluation dependencies.
- Inference configuration: one H200, BF16, LoRA addons, one replica maximum, 40-minute deadline, deletion after evaluation.
- Inference lifecycle: `results/fireworks-inference-session.json`.

The training GPU has been released. A READY adapter is stored model data, not a running inference endpoint. The inference script likewise deletes its temporary serving deployment after completion. View saved evidence without deploying anything:

```sh
.venv/bin/python scripts/report.py
```

**Budget: $25 in existing credits for training and evaluation combined.** No credit purchase or auto-reload was performed. The report records observed charges and explains billing lag. This used dedicated GPU-hour pricing, not the earlier abandoned $0.058 managed-SFT estimate.

Scripts read the API key from Git-ignored `.env` without printing it. Never publish that file, signed links, model weights or raw SDK artifacts. `results/fireworks-dedicated/` is ignored for that reason. The public results JSON files contain selected metrics and resource identifiers only.

The job IDs and output directories already have saved evidence. To repeat an experiment, use new IDs and output directories and preserve the old run. Do not execute `--launch` or `--run` merely to open results. [The walkthrough](PROJECT_WALKTHROUGH.md) gives the complete procedure and explains the earlier access issues.

## Evaluation and evidence

Every variant receives the same system/user text, temperature 0, and at most 16 generated tokens. Mac uses the Qwen chat template; Fireworks uses raw Llama3 headers matching training, with no injected date text. Labels are matched exactly after whitespace trimming and case normalization. An unrecognized answer is an `INVALID` error, never silently mapped to Support general. Each run has one identical synthetic warmup excluded from its reported latency.

Saved outputs contain ticket IDs, true labels, raw answers, parsed predictions, elapsed request times, token usage, accuracy, macro-F1, per-class precision/recall/F1, and a confusion matrix with an invalid-output column. Label ordering is explicit. This fixes the original notebook's classification-report ordering issue without altering the original notebook.

Compare:

1. Local untrained → local tuned adapter: local training gain.
2. Fireworks untrained → Fireworks tuned: cloud training gain.
3. Local tuned → Fireworks tuned: practical setup comparison.
4. Local tuned adapter → local merged models: effect of merging and requantization.

The recommended standalone export is `models/local-router-merged-8bit`: merged projections use 8-bit weights and unchanged embeddings remain 4-bit. Validation accuracy matched the adapter at 90.4%. The first 4-bit export fell to 71.3% validation accuracy, so it is retained only as a diagnostic. This export decision was based on validation, not the final test.

There are only 8–30 test tickets per class, so small per-class differences are noisy. The dataset labels are service categories, not urgency levels; do not invent a “high urgency” class from these labels. The supplied smoke tests are diagnostics, not a representative accuracy estimate.

For assignment submission, include measured results, screenshots of the successful run, and the methodology/caveats. The custom-project route accepts a GitHub link with assets and a Loom walkthrough. Model weights and credentials should stay out of the Git repository.

## Sources

- [Original assignment repository](https://github.com/The-Gen-Academy/5A-Fine-Tune-a-Support-Ticket-Router)
- [Official Qwen checkpoint](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- [MLX-LM](https://github.com/ml-explore/mlx-lm)
- [Fireworks dedicated training](https://docs.fireworks.ai/fine-tuning/training-api/dedicated)
- [Fireworks pricing](https://fireworks.ai/pricing)
- [Fireworks evaluation deployments](https://docs.fireworks.ai/fine-tuning/evaluating-fine-tuned-models)
