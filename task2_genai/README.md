# Task 2 - Generative AI: Domain-Specific Fine-Tuning Pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GimsaraK/CDAZZDEV-MLE-Gimsara_Elgiriyage/blob/main/task2_genai/task2_finetuning.ipynb)

An end-to-end QLoRA fine-tuning pipeline for a domain-specific task, covering use case definition, synthetic dataset engineering with a teacher model, 4-bit fine-tuning on Colab free tier, and a rigorous baseline comparison.

## Links

| Item | Link |
|---|---|
| Fine-tuned (merged) model | _TBD - Hugging Face Hub_ |
| W&B run (if used) | _TBD_ |

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

| | |
|---|---|
| Use case | _TBD_ |
| Input | _TBD_ |
| Output | _TBD_ |
| Correct response | _TBD_ |
| Incorrect response | _TBD_ |

### Dataset

| Item | Value |
|---|---|
| Teacher model | _TBD_ |
| Student (base) model | _TBD_ |
| Total examples | _TBD_ |
| Train / Val / Test (80 / 10 / 10) | _TBD_ / _TBD_ / _TBD_ |

Diversity analysis (prompt length distribution, keyword / topic frequency) is in the notebook and `outputs/`.

### Teacher System Prompt

The full system prompt used for data generation is in [`prompts/`](prompts/) and reproduced in the notebook appendix.

## 2B - Fine-Tuning

QLoRA with 4-bit NF4 quantization and PEFT LoRA adapters, merged with `merge_and_unload()`.

| Parameter | Value | Reason |
|---|---|---|
| LoRA rank (r) | _TBD_ | _TBD_ |
| LoRA alpha | _TBD_ | _TBD_ |
| Target modules | _TBD_ | _TBD_ |
| Learning rate | _TBD_ | _TBD_ |
| LR scheduler | _TBD_ | _TBD_ |
| Epochs | _TBD_ | _TBD_ |
| Batch size | _TBD_ | _TBD_ |
| Gradient accumulation steps | _TBD_ | _TBD_ |
| Max sequence length | _TBD_ | _TBD_ |

The full table, including every non-default parameter, is in the notebook.

### Training Loss

| Epoch | Train loss | Val loss |
|---|---|---|
| _TBD_ | | |

### Out-of-Memory Log

_Document any OOM errors, what was attempted, and the fix applied._

## 2C - Evaluation

| Metric (test set) | Base model + system prompt | Fine-tuned model |
|---|---|---|
| ROUGE-L | _TBD_ | _TBD_ |
| _BERTScore F1 / LLM-as-judge_ | _TBD_ | _TBD_ |

| Manual review | Value |
|---|---|
| Responses reviewed | _TBD_ |
| Correct / Partially correct / Hallucinated | _TBD_ |
| Hallucination rate | _TBD_ % |

Qualitative analysis is in the notebook.

## Bonus - RAG Fallback

_TBD_

## How to Run

1. Open the notebook with the Colab badge above and select a **T4 GPU** runtime.
2. Add `HF_TOKEN`, `GROQ_API_KEY` / `OPENROUTER_API_KEY`, and optionally `WANDB_API_KEY` in the Colab Secrets panel.
3. Run all cells.
