# Use the completed local router

The local phase is complete. The Fireworks experiment is documented in [the comparison report](results/REPORT.md). No API key, payment details, or cloud service is needed for this demo.

## Classify a ticket

The environment and trained model are already installed on this Mac. Open Terminal and run:

```sh
cd /Users/nishokmini/project/finetuning-compare
.venv/bin/python scripts/classify.py "Please install Adobe Acrobat on my computer."
```

The saved demo returned `Software`. Replace the quoted sentence with your own ticket. The command returns the category, raw model answer, and inference time; timing excludes loading the model.

## What is complete

- Reproducible data preparation: 372 training, 94 validation, and 117 held-out test tickets, with near-duplicate training examples excluded.
- Qwen3-4B-Instruct-2507 trained locally using MLX QLoRA: three epochs, 25.0 minutes, 3.03 GiB peak MLX allocation.
- Baseline and trained-model evaluations on the same test tickets: accuracy **44.4% → 82.9%**, macro-F1 **0.401 → 0.794**.
- Standalone export at `models/local-router-merged-8bit`, with 8-bit merged projections and unchanged 4-bit embeddings. Its 117 test predictions match the adapter exactly.
- [Measured report](results/REPORT.md), confusion matrices, raw predictions, and [saved evidence notebook](Comparison_Results.ipynb).

## Known limitation

The adapter and recommended export pass **4 of 5** supplied smoke tests. A generic new-account request is routed to Fileservice instead of the expected Active Directory. Related training examples carry inconsistent labels; see the [label audit](results/label-audit.md). This is documented, not hidden by a routing rule or a changed test. The original 4-bit merged export is retained only for diagnosis because merging reduced its accuracy.

## Record a walkthrough

1. Show the classification command and its output.
2. Open the evidence notebook or measured report and explain the baseline-to-trained improvement.
3. Show the smoke-test limitation and the Fireworks comparison; explain that the models and serving precision differ.

The notebook contains saved evaluation evidence; it is not a record of training executed inside notebook cells. Training logs are in `adapters/local-full/metrics.jsonl`. See [README.md](README.md) for reproduction commands and the original assignment’s submission options. No retraining is needed to use this completed run.
