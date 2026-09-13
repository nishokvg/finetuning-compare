# Attribution and project lineage

This project extends [The Gen Academy support-ticket router assignment](https://github.com/The-Gen-Academy/5A-Fine-Tune-a-Support-Ticket-Router).

The original `Finetune_Support_Ticket_Classifier_Qwen3.ipynb` and `support_tickets.csv` come from that repository. Their original content is preserved. No upstream license file was present in the checked-out source, and this repository does not relicense those materials.

The added MLX and Fireworks scripts, shared split, measured evaluations, comparison report, walkthrough and presentation document Nishok's assignment experiment, developed with Codex assistance. Model weights and their licenses are separate from the code and are not distributed here.

The original assignment uses Qwen3-1.7B-Base in Colab. This custom experiment uses Qwen3-4B-Instruct-2507 locally and Llama-3.2-3B-Instruct on Fireworks. Model family and precision differ, so the experiment compares practical setups rather than isolating hardware performance.

Source checkout commit: `05f7ea9643495741bda4e272bd1977105c3ad529`.
