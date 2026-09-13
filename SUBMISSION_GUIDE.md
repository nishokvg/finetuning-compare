# Submission and demo guide

The project assets are ready to review locally. The project repository is https://github.com/nishokvg/finetuning-compare. The PowerPoint and matching narration are in `presentation/`. Recording your personal Loom and sending the course submission remain your final steps.

## Show the evidence

1. Open `Comparison_Results.ipynb` in VS Code or Jupyter. It contains saved outputs from the actual experiments. Capture the results table or chart as a screenshot for a Google Doc submission.
2. Open `results/REPORT.md` for per-class metrics, confusion matrices, training curves and cost.
3. Run the existing Mac demo from the repository:

```sh
.venv/bin/python scripts/classify.py "Please install Adobe Acrobat on my computer."
```

4. Show one error from `results/misclassified_tickets.csv`. Explain what you would improve next instead of claiming perfect accuracy.

The notebook is a results-evidence notebook. Training was executed by scripts; do not describe the displayed cells as Colab training logs. The original assignment notebook remains unchanged.

## Suggested three-minute narration

**0:00–0:30 — Problem:** “I built a router for seven IT support teams. I used 372 training tickets, 94 validation tickets and 117 test tickets. I removed two near-duplicate training examples to avoid leakage.”

**0:30–1:10 — Method:** “On my 16 GB Mac Mini M4, I used MLX with a 4-bit Qwen 4B base and a small LoRA adapter. On Fireworks, I used Llama 3.2 3B with a dedicated H200. Both ran for three epochs and 279 updates. The model choices differ, so this compares practical setups rather than hardware alone.”

**1:10–2:10 — Results:** Show the report. Read each model’s baseline and tuned accuracy from the table, followed by macro F1 and the weakest class recall. Explain that validation loss is not classification accuracy. Show the measured time and observed cost, including the billing-lag caveat.

**2:10–2:40 — Demo and limitation:** Run the local command. Explain that 4-bit merging damaged accuracy, so validation selected the 8-bit projection export. Show the ambiguous new-account smoke ticket and its unchanged expected label.

**2:40–3:00 — Operations:** Show saved Fireworks model and resource states. Training and inference GPUs were released after use. Explain that the next improvement would be better labels and more examples for weak categories, evaluated on new held-out data.

## Assets for the custom GitHub route

Include README, walkthrough, configs, scripts, requirements, tests, prepared data as permitted by the assignment, sanitized result JSON/predictions, figures, and the evidence notebook. Add your Loom link after recording it.

Exclude `.env`, `.venv*`, downloaded model weights, local adapters/exports, signed download URLs, and raw `results/fireworks-dedicated/` SDK artifacts. Do not commit anything merely to take a screenshot. See `.gitignore`.

The project intentionally extends the assignment with different models and MLX/API execution. Describe this clearly and use the custom-project submission option if required by your course. A successful run does not imply an evaluator has already accepted the submission.

## Submission links

GitHub: https://github.com/nishokvg/finetuning-compare

Loom: add the link after you record the presentation.

Use the custom-project submission route. The original model and Colab workflow differ from this MLX/Fireworks extension.
