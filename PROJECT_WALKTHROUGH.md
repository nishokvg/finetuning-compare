# How this project was completed

Read this alongside [the measured comparison](results/REPORT.md). You trained two ticket routers: Qwen on your Mac and Llama on Fireworks. A router reads a ticket and returns one of seven team labels. The models generate text, but we score their answers as a classification task.

The final files are already saved. Opening the report or notebook does not start a GPU or spend credits. Commands below explain the workflow; training and deployment commands should not be repeated just to view the results.

## 1. Understand the assignment and choose feasible models

The assignment starts from `Qwen/Qwen3-1.7B-Base` in Colab/LLaMA Factory. This custom project instead uses:

- **Mac:** `Qwen/Qwen3-4B-Instruct-2507`, with MLX on the Apple GPU.
- **Fireworks:** `accounts/fireworks/models/llama-v3p2-3b-instruct`, with a LoRA adapter on one H200.

The final local run is 4B, not the initially discussed 1.7B. Fireworks Qwen training required unavailable B200 quota. Llama’s managed SFT endpoint also rejected submission despite advertised eligibility; its separate dedicated Training API worked. These changes mean this is a comparison of two practical setups, not an experiment isolating local versus cloud hardware.

MLX supports Apple unified memory. Quantizing the Qwen base weights to 4 bits reduced memory use enough for this run on the 16 GB Mac. LoRA trains small extra matrices while keeping the base weights frozen. This saves optimizer memory; it does not mean every model or context length fits in 16 GB.

## 2. Prepare one shared dataset

**Input:** `support_tickets.csv`, 585 labeled tickets.

**Script:** `scripts/prepare_data.py`.

The script checks for empty text, missing labels, unexpected categories and duplicates. It adds stable ticket IDs, makes stratified splits so each category is represented, and checks for nearly identical tickets across splits.

| Split | Tickets | Purpose |
|---|---:|---|
| Train | 372 | Update the adapter |
| Validation | 94 | Check learning and export choices |
| Test | 117 | Measure final performance once choices are fixed |

The outer split uses seed 42, validation uses seed 43. Two near-duplicate training examples were excluded to avoid giving the model near copies of test examples. The test set was preserved. `data/manifest.json` records hashes, seeds, labels and the duplicate audit.

Each example becomes a conversation:

```text
System: You are an IT helpdesk ticket routing assistant ... [seven allowed labels]
User: Support Ticket: Adobe Acrobat fails to install ...
Assistant: Software
```

Training predicts the assistant label. The system and user messages provide context, but their tokens do not contribute to the training loss. The same data and system/user text are used on both platforms.

## 3. Set up the Mac environment and download the base model

The local Python environment is `.venv`; versions are in `requirements.txt`. The source Qwen checkpoint was pinned to revision `cdbee75f17c01a7cc42f958dc650907174af0554` and saved to `models/qwen3-4b-source`.

`mlx_lm.convert` produced `models/qwen3-4b-4bit` using 4-bit affine quantization with group size 64. The source download and converted weights are large generated files and are ignored by Git.

Quantization compresses weight storage. It can change predictions, so the experiment evaluates the actual quantized baseline rather than assuming it behaves exactly like the full-precision source.

## 4. Run a small local pilot

**Script:** `scripts/train_local.py --config configs/local_pilot.json`.

The pilot used 80 microsteps, equal to 20 optimizer updates. It checked that the data loader, loss mask, adapter and Apple GPU worked before the full run. Pilot results are retained in `results/local-pilot-smoke/` and the pilot logs. They are not presented as the final result.

## 5. Train the final local adapter

**Configuration:** `configs/local_full.json`.

| Choice | Value | Meaning |
|---|---|---|
| Epochs | 3 | Three passes over 372 tickets |
| Batch / accumulation | 1 / 4 | Process one ticket at a time; update after four |
| Updates | 279 | 1,116 microsteps divided by four |
| Learning rate | 0.0001 | Step size for adapter updates |
| LoRA rank / scale | 8 / 2 | Adapter capacity; equivalent alpha is 16 |
| Target layers | All seven linear projections in all 36 blocks | Adapter learns across attention and MLP projections |
| Context | 1,024 tokens | Maximum input length used for training |
| Loss mask | Assistant only | Learn the target routing answer |
| Gradient checkpointing | On | Recompute some activations to save memory |

The selected checkpoint was the fixed final three-epoch adapter, not a checkpoint chosen using test performance. The run took about **25 minutes**, with **3.03 GiB peak MLX allocation**. That allocation is not the Mac’s total memory usage. Training logs are in `adapters/local-full/metrics.jsonl`.

## 6. Merge the local adapter and check quantization damage

A LoRA adapter needs its base model for inference. Merging folds the learned changes into the base model’s linear weights, producing a standalone local export.

The first 4-bit merged export lost substantial validation accuracy: **71.3%**, versus **90.4%** for the adapter. The workflow therefore produced an 8-bit projection export and evaluated it on validation before the final test. It matched the adapter’s **90.4%** validation accuracy.

The recommended export is `models/local-router-merged-8bit`. Its merged projections use 8-bit weights; untouched embeddings remain 4-bit. The original 4-bit export is kept as a diagnostic. On test, the adapter and recommended export both got **97/117 correct (82.9%)**, with identical predictions. The baseline got **52/117 (44.4%)**.

This demonstrates why merging should be checked, especially when requantizing the result.

## 7. Resolve cloud access and tokenizer formatting

The Fireworks SDK and pinned training cookbook were installed separately in `.venv-fireworks`, using Python 3.13. `requirements-fireworks.txt` and its lock file record the versions. This kept the local MLX environment separate.

Hugging Face initially denied access to the gated Meta model. You supplied an authorized direct Meta download link. Only the small tokenizer/configuration files were downloaded; their checksums were verified against Meta’s checklist. No Llama model weights were downloaded to the Mac for cloud training.

`scripts/convert_meta_router_tokenizer.py` converted the tokenizer into the format the training recipe expects. It checked all 256 special token IDs, 1,403 text cases and all 466 train/validation conversations against the original vocabulary. The assistant-only loss masks also matched. The longest conversation was 183 tokens, below the 1,024 limit.

The recipe uses explicit Llama3 role headers without adding knowledge-date text. Cloud inference uses that same format. Conversion evidence is in `results/meta-tokenizer-conversion.json`. The signed download link and API key are not stored in the published evidence.

## 8. Train on Fireworks

**Script:** `scripts/fireworks_dedicated.py`.

With no arguments, this script only validates the data and prints a plan. `--check-runtime` renders the examples locally. `--launch` is the paid training path and has already run.

The dedicated API accepted one trainer, using one H200. Settings were three epochs, batch four, rank eight, alpha 16, learning rate 0.0001 and assistant-only loss. The Mac ran the control loop; Fireworks ran the neural-network computations.

All **279 updates completed** in **595.76 seconds**, including startup, validation, checkpoint saving, registration and cleanup. Validation token losses were **0.221 → 0.226 → 0.191**. Loss measures prediction error on expected answer tokens; it is not a percentage accuracy score.

The output adapter is:

```text
accounts/nishokvg-myac5qmcmir/models/support-router-llama3b-dedicated-01
```

It was registered READY. The trainer was deleted, and Fireworks explicitly confirmed resource release and no further billing. The first observed training charge was $0.81; see the report for the final combined billing observation.

## 9. Evaluate baseline and adapter on temporary inference capacity

**Script:** `scripts/finish_fireworks.py`.

The trained adapter is not an always-running endpoint. The evaluation script creates a temporary BF16 Llama deployment with LoRA support, limited to one H200 and one replica. It loads the adapter on that deployment and addresses the baseline and adapter separately.

For each model, the script saves:

1. The original five smoke-ticket predictions.
2. Predictions on all 94 validation tickets.
3. Predictions on all 117 test tickets.

Generation uses temperature zero and a maximum of 16 tokens. Parsing accepts only exact category names after trimming whitespace and normalizing case. `AD` is not silently treated as `Active Directory`; an unrecognized answer is an error. Each run includes one identical warmup whose latency is excluded.

A generic new-account smoke ticket was already ambiguous in the local experiment. Its original expected label remains unchanged. Smoke outcomes are reported transparently; they do not justify rewriting test labels or adding special rules.

The script records every raw answer and elapsed request time. Its deadline bounds startup and evaluation, and cleanup deletes the deployment after success or failure. `results/fireworks-inference-session.json` records the observed cleanup result. Check [the report](results/REPORT.md) for the completed scores and resource state.

## 10. Compare and interpret the measured results

**Script:** `scripts/report.py`.

The report generator verifies the ticket IDs, text and labels across runs and recomputes metrics from the saved predictions. It produces accuracy, macro F1, per-class precision/recall/F1, confusion matrices, latency, training curves and an error CSV.

Read the results in this order:

1. Mac baseline versus Mac tuned: what local fine-tuning added.
2. Fireworks baseline versus Fireworks tuned: what cloud fine-tuning added.
3. Mac tuned versus Fireworks tuned: which practical setup performed better in this experiment.
4. Per-class recall: which teams still receive incorrectly routed tickets.
5. Time and cost: operational tradeoffs, with measurement boundaries stated.

There are only 8–30 test tickets per class. These labels identify service teams, not urgency levels. The two model families and precisions differ, so a difference in accuracy cannot establish that one platform trains better in general.

## 11. Run a local demo and submit evidence

```sh
cd /Users/nishokmini/project/finetuning-compare
.venv/bin/python scripts/classify.py "Please install Adobe Acrobat on my computer."
```

The command uses the already installed local export; it does not require Fireworks. Read [LOCAL_DEMO.md](LOCAL_DEMO.md), open `Comparison_Results.ipynb`, and show the comparison figure, a confusion matrix and one misclassified example.

The evidence notebook includes saved outputs from real script runs. It does not claim that training ran inside notebook cells. The original Colab notebook remains available as `Finetune_Support_Ticket_Classifier_Qwen3.ipynb`.

For the assignment, use a screenshot of the results notebook in a Google Doc, or the custom-project route with a GitHub repository and your Loom explanation. A suggested narration is in [SUBMISSION_GUIDE.md](SUBMISSION_GUIDE.md). Credentials, signed links, model weights and raw SDK artifacts are excluded from submission assets. The project is published at https://github.com/nishokvg/finetuning-compare. Your personal Loom recording and course submission remain separate steps.
