# Task 2 - Generative AI: Domain-Specific Fine-Tuning Pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GimsaraK/CDAZZDEV-MLE-Gimsara_Elgiriyage/blob/main/task2_genai/task2_finetuning.ipynb)

An end-to-end QLoRA fine-tuning pipeline for a domain-specific task, covering use case definition, synthetic dataset engineering with a teacher model, 4-bit fine-tuning on Colab free tier, and a rigorous baseline comparison.

## Links

| Item | Link |
|---|---|
| Fine-tuned (merged) model | [huggingface.co/GimsaraK/northwind-compliance-qwen2.5-1.5b](https://huggingface.co/GimsaraK/northwind-compliance-qwen2.5-1.5b) (public) |
| Experiment tracking | Not used. Loss is logged per epoch and plotted with matplotlib in the notebook |

## Contents

| Path | Description |
|---|---|
| `task2_finetuning.ipynb` | Main notebook, executed with outputs visible |
| `prompts/` | Teacher-model system prompt and evaluation / judge prompts |
| `data/` | Generated dataset and `train` / `val` / `test` JSONL splits |
| `src/` | Data generation, training, and evaluation helpers |
| `outputs/` | Loss curves, diversity plots, evaluation results |
| `requirements.txt` | Python dependencies |

## 2A - Use Case and Dataset

### Problem Statement

Closed-book assistant for the fictional company Northwind Components. The user turn is a scenario only. The manual is not pasted in, so the model has to apply the rules it was trained on. The manual is fictional and is not legal advice.

| | |
|---|---|
| Use case | Compliance policy assistance |
| Input | One employee scenario (who did what, and what they want) |
| Output | JSON with `policy_ids`, `required_action`, and `rationale` |
| Correct response | Every id is in the manual, or `["NONE"]` when no rule applies, and the action is the one that clause states |
| Incorrect response | An unknown id, the wrong clause, an action the clause does not allow, or an invented rule |

### Dataset

| Item | Value |
|---|---|
| Teacher model | Groq `openai/gpt-oss-120b` (OpenRouter only if validation fails) |
| Student (base) model | `Qwen/Qwen2.5-1.5B-Instruct` (chat template only in 2A; training is 2B) |
| Total examples | 200 accepted after schema and duplicate checks |
| Train / Val / Test (80 / 10 / 10) | 160 / 20 / 20 |

Eight policy topics are crossed with five situation types (clear breach, borderline, compliant, not covered, two-policy), 25 examples each. Scenario length runs from 40 to 149 words (median 64). Exact-duplicate drops were 0 and character 5-gram near-duplicate drops were 0 (Jaccard threshold 0.80). Two teacher responses needed a repair call. Diversity charts are in `outputs/`.

### Teacher System Prompt

The full system prompt, including the policy manual, is in [`prompts/teacher_system.txt`](prompts/teacher_system.txt) and in appendix A of the notebook. The student system prompt is a different file, [`prompts/student_system.txt`](prompts/student_system.txt).

## 2B - Fine-Tuning

QLoRA with 4-bit NF4 quantization and PEFT LoRA adapters, merged with `merge_and_unload()`.

| Parameter | Value | Reason |
|---|---|---|
| 4-bit load | true | QLoRA keeps the base weights in 4-bit so a 1.5B model fits a free T4. |
| Quant type | nf4 | NF4 is the QLoRA 4-bit type. It is not the default int4. |
| Double quantization | true | A second quantization of the quantization constants saves more T4 memory. |
| Compute dtype | float16 | T4 has no bfloat16. Forward and backward math stays in float16. |
| Load dtype / adapter dtype | float16 / float32 | Unquantized layers load in float16, not Qwen's bfloat16 default. LoRA weights stay float32 because the float16 GradScaler cannot unscale bfloat16 or float16 gradients. |
| LoRA rank (r) | 16 | Rank 16 is enough for 160 short JSON answers without a large adapter. |
| LoRA alpha | 32 | Alpha is 2x rank, the usual QLoRA scale, so the update is not tiny. |
| LoRA dropout | 0.05 | 0.05 limits memorizing the wording of 160 training scenarios. |
| LoRA bias | none | QLoRA does not train bias terms. The adapter is the weight update only. |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | Attention and MLP projections. Attention alone under-uses a 1.5B model on this task. |
| Learning rate | 2e-4 | Standard QLoRA rate for a small instruct model. Higher overfits 160 rows. |
| LR scheduler | cosine | Cosine decay after warmup. A constant rate keeps stepping hard at the end of 3 epochs. |
| Warmup ratio | 0.03 | 3 percent of steps ramp the rate so the first updates are not full size. |
| Epochs | 3 | Three epochs give validation loss more than one step in which to fall. |
| Batch size | 1 | One example per step. A larger microbatch is the first thing that OOMs on a T4. |
| Gradient accumulation steps | 8 | Effective batch of 8. One epoch is about 20 optimizer steps. |
| Max sequence length | 512 | 512 covers a 149-word scenario plus the prompt. A longer tokenized train row raises the cap to 768. An out-of-memory retry drops it to 384. |
| Pad token | tokenizer pad (EOS if none) | Qwen2.5 ships endoftext as its pad token. A tokenizer without one pads with EOS. Padded positions are masked from attention and loss. |
| Optimizer | paged_adamw_8bit | 8-bit paged AdamW keeps optimizer state off the T4 when memory spikes. |
| Weight decay | 0.01 | Light decay. The set is small, so zero decay memorizes more easily. |
| Gradient checkpointing | true | Recomputes activations. This is what makes the MLP adapters fit. |
| Packing | false | Rows stay separate. Packing would glue two scenarios into one sequence. |
| Loss mask | assistant tokens only | The system and user turns are context. The loss is the JSON answer. The mask is built in qlora.py because TRL 0.20 dropped its completion-only collator. |
| Eval and save | every epoch | One validation number per epoch, which is what the loss rubric asks for. |
| Best checkpoint | lowest eval loss | The merge uses that checkpoint, not whatever the last epoch happened to be. |
| Seed | 42 | Fixed seed so the same notebook run is repeatable. |
| Merge dtype | float16 base, not 4-bit | merge_and_unload on a 4-bit model is unreliable. The base is reloaded in float16. |

The same rows are printed from `src/train_config.py` in the notebook. The trainer reads `train.jsonl` and `val.jsonl` only.

### Training Loss

| Epoch | Train loss | Val loss |
|---|---|---|
| 1 | 1.495 | 0.673 |
| 2 | 0.592 | 0.417 |
| 3 | 0.337 | 0.361 |

Validation loss fell every epoch, from 0.673 to 0.361. The gap to train loss widens in epoch 3 (0.337 vs 0.361), so a fourth epoch would likely start to overfit 160 rows. Logged by the Colab T4 run in `outputs/loss_history.json` and plotted in `outputs/loss_curve.png`. The merged model card is on the Hub.

### Out-of-Memory Log

If the T4 runs out of memory, section 7 retries in this order and records which step finished:

1. Max length 512 (768 only if a train row does not fit), LoRA on attention and MLP.
2. Max length 384, same modules.
3. Max length 384, attention projections only.

Result of the Colab T4 run: no out-of-memory error. Step 0 finished (max length 512, all 7 target modules). The longest train row is 372 tokens, so nothing was truncated.

## 2C - Evaluation

Both models answer the same 20 held-out rows from `test.jsonl`, with the same system and user turns. The base model gets the system prompt and no fine-tuning. Both run in float16 with plain greedy decoding (max 256 new tokens, repetition penalty off). The fine-tuned model is loaded from the Hub repo, or from the merged folder of section 8 if the push did not happen. Per-row answers are saved to `outputs/eval_predictions.jsonl`.

| Metric (test set, n=20) | Base model + system prompt | Fine-tuned model |
|---|---|---|
| ROUGE-L F1, whole answer (canonical JSON) | 0.254 | **0.581** |
| ROUGE-L F1, `required_action` field | 0.171 | **0.582** |
| ROUGE-L F1, `rationale` field | 0.180 | **0.425** |
| BERTScore F1 (`roberta-large`) | 0.885 | **0.941** |
| LLM judge total, mean of 8 | 2.15 | **5.20** |
| Judge: `policy_correct`, mean of 2 | 0.30 | **1.50** |
| Judge: `action_faithful`, mean of 2 | 0.35 | **1.20** |
| Judge: `rationale_grounded`, mean of 2 | 0.70 | **1.20** |
| Judge: `no_invention`, mean of 2 | 0.80 | **1.30** |
| Valid JSON, strict (%) | 0 | **100** |
| `policy_ids` exact match (%) | 20 | **80** |
| Answers with an id not in the manual (%) | **0** | 5 |
| `NONE` on not-covered rows (%), n=4 | **100** | 75 |

The base model wraps every answer in a markdown fence, so none is strict JSON. Its fields are still scored after the fence is stripped. Without the manual it answers `["NONE"]` on all 20 rows, which is why it scores 100% on the four not-covered rows and 20% on exact ids (those same four rows). The fine-tuned model names the right policy on 80% of rows, but one answer uses an id that is not in the manual and one not-covered row gets a real policy.

- **ROUGE-L** is computed on the answer in canonical form (compact JSON with sorted keys, the form the gold answers use), so key order and whitespace do not cost points. It is also computed on the two text fields alone.
- **BERTScore F1** uses `roberta-large` on the same canonical text.
- **LLM judge:** `google/gemma-4-31b-it:free` on OpenRouter, with `nvidia/nemotron-3-super-120b-a12b:free` (also OpenRouter) grading only the rows Gemma could not, after one repair attempt. Each verdict records its grader.
  - Neither judge is the Qwen student or the gpt-oss teacher that wrote the references. Gemma is listed as the 2A backup teacher but wrote none of the dataset (`fallback_calls` is 0 in `outputs/generation_report.json`).
  - Groq's free catalogue only offers gpt-oss and Qwen models, so both judges run on OpenRouter and share its free-tier limits. Saved verdicts are reused, so a re-run only grades the missing rows.
  - In the saved run, Gemma returned HTTP 429 (rate limited) on every call, so Nemotron graded all 40 answers (20 base, 20 fine-tuned). No answer was left unjudged. The grader of each verdict is in `outputs/judge_scores.jsonl`.
  - The judge sees the manual, the scenario, the reference, and one answer, and is not told which model wrote it.
  - It returns four 0-2 scores as Pydantic-validated JSON: `policy_correct`, `action_faithful`, `rationale_grounded`, `no_invention`. The total out of 8 is summed in code.
  - Full prompt: [`prompts/judge_system.txt`](prompts/judge_system.txt). Per-row verdicts: `outputs/judge_scores.jsonl`.
- **Domain checks** need no model. They read the answer JSON and compare it with the gold label and the manual.

### Manual Review and Hallucination Rate

All 20 fine-tuned answers are reviewed by hand against the policy manual (the spec minimum is 10). The manual wins over the expected answer, because the expected answers were written by the teacher and can be wrong. The notebook suggests a label from the domain checks and the judge; the reviewer sets every final label, and the notebook table marks each override. Labels are in `outputs/manual_review.json`.

| Label | Rule |
|---|---|
| correct | The `policy_ids` are the right ones under the manual, and the action is one the clause allows |
| partially correct | A real, relevant policy, but the action is incomplete or slightly off, or one of two ids is missing. Nothing is invented |
| hallucinated | An id not in the manual, an invented rule, limit, or deadline, a wrong policy stated as applying, `NONE` when a policy applies, a policy cited for an uncovered case, or no usable JSON |

| Manual review | Value |
|---|---|
| Responses reviewed | 20 of 20 fine-tuned answers |
| Correct / Partially correct / Hallucinated | 7 / 4 / 9 |
| Hallucination rate | **45%** (9 of 20) |

- Correct: rows 1, 3, 10, 11, 14, 15, 18. Partially correct: rows 5, 8, 9, 12. Hallucinated: rows 0, 2, 4, 6, 7, 13, 16, 17, 19.
- The review overrode the suggested label on 5 rows (4, 7, 9, 11, 16). Each row's label and note are in `outputs/manual_review.json` and in the section 12 table.
- The rubric is strict: an answer with the right policy and action still counts as hallucinated if it adds a rule, deadline, or consequence the manual does not contain (rows 6 and 7).

Qualitative analysis (two paragraphs, citing specific test rows) is in section 13 of the notebook. In short: fine-tuning taught the model the manual (exact policy ids 20% to 80%, valid JSON 0% to 100%), and the remaining errors are confusion between look-alike clauses, one clause's action leaking into another, and invented specifics in the rationale.

## How to Run

1. Open the notebook with the Colab badge above and select a **T4 GPU** runtime.
2. Add `HF_TOKEN`, `GROQ_API_KEY`, and `OPENROUTER_API_KEY` in the Colab Secrets panel.
3. Run all cells.
