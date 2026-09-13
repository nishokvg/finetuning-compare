# Dedicated Llama training: review before launch

**Status: training completed successfully on 2026-09-12 UTC.** The user approved one attempt within a $6 allowance and $25 project budget. All 279 updates completed. The final adapter is READY, and Fireworks reports the trainer DELETED with resources released and no further billing. The attempt took 595.76 seconds including startup, checkpoints and cleanup. The refreshed billing page shows $0.81 in H200 training spend and $25.19 in credits; billing can lag. Inference evaluation subsequently completed on a temporary BF16 H200 deployment: baseline 48/117 (41.0%), tuned 98/117 (83.8%), with zero tuned invalid labels. The inference deployment is DELETED with zero replicas. See results/REPORT.md for the full comparison and current observed billing.

| Validation after pass | Token prediction loss |
|---|---:|
| 1 | 0.220949 |
| 2 | 0.225947 |
| 3 | 0.190766 |

The final checkpoint has the lowest observed validation loss. This is not classification accuracy. The output is `accounts/nishokvg-myac5qmcmir/models/support-router-llama3b-dedicated-01` (LoRA adapter). Sanitized results and all 279 batch metrics are in `results/fireworks-llama-training.json`. The following sections document the reviewed configuration and execution procedure; the launch command has already been executed and must not be repeated. The 117 test tickets remain unused for cloud inference.

The earlier managed SFT submission was rejected with `model does not support tuning`, despite the model's `supervisedLoraTunable: true` metadata. This draft uses the separate dedicated Training API. Its published Llama configuration uses one H200, and the account has H200 quota. The dedicated job was accepted with HTTP 200 and completed; the earlier managed API rejection does not apply to this successful dedicated route.

## What you are reviewing

| Setting | Proposed value | What it means |
|---|---|---|
| Base model | `accounts/fireworks/models/llama-v3p2-3b-instruct` | Llama 3.2, 3 billion parameters, already instruction tuned |
| Method | LoRA, rank 8, alpha 16 | Learn a small adapter; keep base weights frozen. Alpha 16 is explicit in this draft. |
| Examples | 372 train; 94 validation | Same saved split as the local run |
| Test set | 117 tickets, kept out of the recipe | Reserved for the later baseline and tuned-model comparison |
| Epochs / batch | 3 / 4 | Three passes, four examples per update: 279 updates |
| Learning rate | 0.0001, constant, no warmup | Same nominal rate and schedule as local |
| Optimizer | Recipe Adam defaults with beta2 0.999 and weight decay 0 | Explicit beta2 avoids the cookbook's 0.95 default |
| Input limit | Validate every example at 1,024 tokens | Refuse rows the actual renderer drops or splits; the GPU shape itself has a larger capacity |
| Hardware | One H200, one replica | Abort if the pinned configuration advertises different hardware |
| Deadline | 30 minutes from the create request | Includes provisioning, training, validation loss and final checkpoint registration |
| Inactivity fallback | 10 minutes | Ask Fireworks to stop the trainer after tracked activity ceases |
| Checkpoints | Every 93 updates and at the end | Save progress each epoch; register the final model if training completes |
| Serving | No inference deployment | Evaluation by generated labels is a later approved step |

Source: [dedicated training](https://docs.fireworks.ai/fine-tuning/training-api/dedicated), [training shapes](https://docs.fireworks.ai/fine-tuning/training-api/training-shapes), and the [pinned SFT recipe](https://github.com/fw-ai/cookbook/blob/a8c8586f025b87f9fd72fceacb74ed949447cf3c/training/recipes/sft_loop.py).

## Cost proposal

At the published **$8 per H200-hour**, 30 minutes is **$4** of trainer time. Propose **$6 of the existing $25 budget** for this attempt, leaving $19 for subsequent work. This is a planning allowance, not a provider-enforced dollar cap or a prediction of runtime. Recheck pricing and remaining credits before launch. Inference deployments are charged separately. [Fireworks pricing](https://fireworks.ai/pricing)

The local supervisor stops its worker process group at the deadline and requests trainer deletion, then checks for a stopped or absent trainer. It also cleans up after ordinary failure and Ctrl-C. If the Mac sleeps, loses its network, or the supervisor is forcibly killed, local cleanup may not run. The cloud inactivity timeout is a fallback, not a maximum lifetime: heartbeats and active operations count as activity. Shutdown delays or failures can exceed the allowance. Keep the Mac awake and check the recorded trainer in Fireworks if cleanup is unconfirmed.

## Step by step: what the script would do

1. **Check the data locally.** Confirm file hashes, row counts and message structure. With no arguments, the script ends here after printing the plan. It reads no API key and makes no network calls on the successful default path.
2. **Check dependencies and tokenization before renting a GPU.** Import the pinned SDK/cookbook and render all training and validation examples using the real Llama tokenizer and assistant-only loss mask. Missing dependencies, changed local tokenizer checksums, or dropped/split rows stop this step. The script uses verified local files, so this step no longer needs Hugging Face access. Source model/revision fields are retained as provenance; the local tokenizer path is used at runtime.
3. **Check the destination.** Confirm the fixed job ID and output model do not already exist. Fetch the pinned training shape and verify one H200. A repeat run does not automatically resume, retry training, or choose another GPU.
4. **Create one trainer.** Send one POST to `accounts/nishokvg-myac5qmcmir/rlorTrainerJobs`, with reservations off and the requested inactivity timeout. The resource name says RLOR, but this recipe performs supervised learning, not reinforcement learning. Save the known job ID before creating it. An uncertain POST is never blindly retried.
5. **Train through a supervised worker process.** Your Mac runs the Python control loop; Fireworks performs the model computations. The recipe attaches to this trainer, learns from the 372 examples and measures validation loss on the 94 held-out examples after each epoch. Loss measures how well the model predicts the expected response tokens; it is not routing accuracy.
6. **Save the learned adapter.** Save periodic resumable and promotable checkpoints. On successful completion, register the final output model. Registration does not deploy it. A timeout may leave a partial checkpoint; it does not count as a completed run. There is no automatic retry or resume.
7. **Stop the GPU and verify.** Stop the local worker, request deletion if needed, and read the trainer state. A deletion acknowledgment alone is not treated as proof of shutdown. Report cleanup failure prominently. Preserve a sanitized supervisor record and worker log locally.
8. **Pause for your review.** After a successful run, inspect training/validation loss, checkpoint and cost. Later, separately approve inference setup, five smoke tickets, and baseline/tuned evaluation on the same test set.

## Files and commands

- `scripts/fireworks_dedicated.py`: default offline plan, runtime check, supervised launch and targeted cleanup.
- `configs/fireworks_llama_dedicated.json`: proposed settings, pinned model/tokenizer/shape revisions, split hashes and budget reference.
- `requirements-fireworks.txt`: isolated Fireworks dependencies. Direct versions and cookbook commit are pinned. `requirements-fireworks.lock.txt` now records the installed transitive versions.
- `tests/test_fireworks_dedicated.py`: offline tests for changed data/hardware, duplicate prevention, timeout termination, uncertain submission and cleanup failures.

The completed local MLX environment remains separate. The environment setup below has completed. **Launch has now completed; do not run it again.** The runtime check passed using the converted local tokenizer.

Safe offline review:

```sh
cd <ORIGINAL_PROJECT_ROOT>
python3 scripts/fireworks_dedicated.py
python3 -m unittest discover -s tests -p 'test_fireworks_dedicated.py' -v
```

Setup commands already executed using Python 3.13:

```sh
python3.13 -m venv .venv-fireworks
.venv-fireworks/bin/python -m pip install -r requirements-fireworks.txt
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv-fireworks/bin/python scripts/fireworks_dedicated.py --check-runtime
```

Only after reviewing that result and separately approving the paid run:

```sh
.venv-fireworks/bin/python scripts/fireworks_dedicated.py --launch
```

If an approved run later needs manual cleanup, this targets only the trainer matching its saved ownership record:

```sh
python3 scripts/fireworks_dedicated.py --cleanup
```

Logs would appear in `results/fireworks-dedicated/llama3b-01/`. This directory is ignored by Git because SDK artifacts can contain private dataset previews or signed storage links. The supervisor records only selected settings/status, and worker console output is redacted. Do not publish raw recipe artifacts without inspection.

## How the eventual comparison will be described

| Aspect | Completed Mac run | Proposed Fireworks run |
|---|---|---|
| Model | Qwen3-4B-Instruct-2507 | Llama-3.2-3B-Instruct |
| Training | MLX, 4-bit base plus LoRA | Dedicated H200 LoRA; shape does not explicitly declare base-weight precision |
| Data | Same 372 / 94 / 117 split | Same split |
| Test result | Baseline 44.4%; trained adapter / selected 8-bit merged model 82.9% | Not yet measured |
| Interpretation | Local practical setup | Cloud practical setup |

Model family, size, quantization and training implementation differ. Compare each model against its own untuned baseline, and then compare accuracy, macro F1, per-class recall, latency, elapsed training time and cost across setups. Do not present the result as a controlled comparison of hardware alone.

## Verification limits

The offline plan and lifecycle tests can run without training dependencies or API credentials. They test the wrapper, not Fireworks' backend. Dependency installation and package consistency have passed. Actual Llama rendering has passed locally. Job creation, final model registration and GPU resource release have now been verified. Routing accuracy has now been measured; final settled billing may lag the dashboard observation. The exact setup result is recorded in `results/fireworks-runtime-check.json`.

## Runtime check finding: explicit Llama formatter

The pinned cookbook did not infer a formatter for this exact model name. Both preparation and training now explicitly select its `llama3` renderer. This renderer uses Llama role headers and end-of-turn tokens, and intentionally omits the knowledge-date text injected by Hugging Face's stock template. When inference is prepared later, align its formatting with training or document the difference. All 466 train/validation examples now render successfully. The longest conversation is 183 tokens, below the 1,024-token limit. The conversion script and report are `scripts/convert_meta_router_tokenizer.py` and `results/meta-tokenizer-conversion.json`. The signed Meta download URL is not stored in these files. Model weights were not downloaded.
