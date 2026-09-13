# Video narration script

Approximately six minutes at a comfortable pace, plus a short demo pause. Use the PowerPoint speaker notes or read the passages below. Pause on slide 9 for the local terminal demo.

## Slide 1: Support ticket routing

Hi, I’m Nishok. This project compares two ways to fine-tune a support-ticket router: on my 16 GB Mac Mini M4 using MLX, and on Fireworks using a cloud GPU. Both approaches improved substantially over their own baselines. On the same 117 test tickets, the Mac model got 97 correct and the cloud model got 98. I’ll explain the workflow, the evidence, and why that small difference does not establish a clear winner.

**On screen:** show this slide.

## Slide 2: The routing task

The input is a short IT support ticket. The output is exactly one team label. For example, an Adobe Acrobat installation failure should route to Software. The dataset has seven service categories, including general support, file services, Microsoft 365, end of life, Active Directory, and computer services. These labels identify teams, not urgency. The assignment originally used a smaller Qwen base model in Colab. This project extends it with two different instruction models and deployment methods.

**On screen:** show this slide.

## Slide 3: One shared split

I began with 585 labeled tickets. The shared split contains 372 training tickets, 94 validation tickets, and 117 final test tickets. Two training examples were nearly identical to test examples, so the preparation script removed them. Stable ticket IDs and file hashes keep both experiments aligned. Training updates use only the training split. Validation checks learning and the model export. The test set measures final performance after the choices are fixed.

**On screen:** show this slide.

## Slide 4: Two training setups

On the Mac, MLX trained a small LoRA adapter on a four-bit Qwen 4B base. Four-bit storage reduced memory use. I processed one ticket at a time and accumulated four gradients before each update. On Fireworks, a dedicated H200 trained a Llama 3B adapter with batches of four. Both runs used rank eight, a learning rate of 0.0001, three epochs, and 279 optimizer updates. Only the assistant’s expected answer contributed to the loss. The model family and precision differ, which limits a direct platform comparison.

**On screen:** show this slide.

## Slide 5: Merging needs a validation check

After local training, I merged the adapter into the base model for a standalone demo. The first four-bit merged export lost substantial validation accuracy: it fell from 90.4 percent to 71.3 percent. I then produced an eight-bit projection export. It matched the adapter at 90.4 percent validation accuracy, so I selected it before evaluating the test set. Its test predictions match the adapter exactly. The lesson is that a successful merge operation does not guarantee preserved model quality.

**On screen:** show this slide.

## Slide 6: Fine-tuning improved both models

On the shared test set, the Mac baseline scored 44.4 percent accuracy and the trained model scored 82.9 percent. Fireworks improved from 41.0 percent to 83.8 percent. That is 97 correct tickets locally and 98 in the cloud. The tuned macro F1 scores were 0.794 and 0.810. We used the same prompt text, temperature zero, a limit of sixteen output tokens, and strict category parsing. The cloud baseline produced two invalid labels. Both tuned models produced none.

**On screen:** show this slide.

## Slide 7: One ticket separates the final scores

The cloud model was correct on seven tickets the Mac missed. The Mac was correct on six tickets the cloud missed. That leaves a net difference of one ticket, which is too small to establish a reliable overall winner from this experiment. Active Directory remains a weakness: the Mac identified four of nine tickets, while Fireworks identified three. Both models passed four of the five smoke tests and missed the generic new-account example. Related labels are ambiguous, so I preserved the failures rather than changing the answers.

**On screen:** show this slide.

## Slide 8: Time, latency and cloud resources

The local training run took about 25 minutes and reported 3.03 gibibytes of peak MLX allocation. Fireworks took about 9.9 minutes including startup, checkpoint saves and cleanup. The timing boundaries differ, so these are operational timings rather than pure compute benchmarks. Median test latency was about 0.511 seconds locally and 0.218 seconds on Fireworks, including network travel. At the saved billing check, Fireworks showed 87 cents total spend, with reporting lag possible. Both temporary cloud GPUs are deleted, and the trained adapter remains stored.

**On screen:** show this slide.

## Slide 9: Local demo and saved evidence

For the demo, I can use the trained model directly on my Mac. I run the classification command with a ticket asking to install Adobe Acrobat, and the model returns Software. No cloud deployment is needed for that local command. The repository also contains the raw predictions, per-class metrics, confusion matrices, training records, and a results notebook. The notebook displays evidence from real script runs. It does not claim that training ran inside notebook cells.

**On screen:** switch to Terminal, run the displayed command, then return to the deck.

## Slide 10: What I learned

My main conclusion is that a small adapter can substantially improve this routing task, both locally and in the cloud. The Mac was capable of the full workflow within its memory limits, while Fireworks provided a shorter operational training run. Evaluation mattered at every stage: it caught damage from merging, exposed formatting failures, and identified weak category recall. My next improvement would be to review ambiguous labels and collect more Active Directory examples, then test on new held-out data. The submission includes reproducible code, the measured report, and the limitations alongside the results.

**On screen:** show this slide.

## Recording checklist

Open the PPTX in PowerPoint or Keynote and use presentation mode. Record the slides with your microphone in Loom. Keep the API key and billing account details off screen. Use the repository README for the final link. Paste the Loom URL into SUBMISSION_GUIDE.md after recording.