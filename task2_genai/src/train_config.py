"""QLoRA settings for the Colab T4 run.

Every value the trainer uses is listed here with a reason. The notebook prints
this table. Nothing is left at a library default without that sentence.
Training itself runs on a GPU. This module does not import torch.
"""
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'Implement the Task 2B QLoRA plan', Date: 2026-10-08 (see CITATIONS.md Entry 10)

from . import config

# Public repo. Change the owner if the Hugging Face username is not GimsaraK.
HF_REPO_ID = "GimsaraK/northwind-compliance-qwen2.5-1.5b"
ENV_HF_TOKEN = "HF_TOKEN"

# Marker already written into each row's ChatML text. Loss starts after it.
RESPONSE_TEMPLATE = "<|im_start|>assistant\n"

# Train and validation only. The test split is reserved for Task 2C.
TRAIN_FILE = config.TRAIN_PATH.name
VAL_FILE = config.VAL_PATH.name

LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
ATTENTION_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")

LEARNING_RATE = 2e-4
LR_SCHEDULER = "cosine"
WARMUP_RATIO = 0.03
NUM_EPOCHS = 3
OPTIMIZER = "paged_adamw_8bit"
WEIGHT_DECAY = 0.01
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
MAX_SEQ_LENGTH = 512
MAX_SEQ_LENGTH_FALLBACK = 384
COMPUTE_DTYPE = "float16"
SEED = 42

# If CUDA runs out of memory, section 7 walks this list in order.
# Step 0 is the intended run. Later steps are smaller, not a different experiment.
OOM_STEPS = (
    {"step": 0, "max_seq_length": MAX_SEQ_LENGTH, "target_modules": TARGET_MODULES},
    {"step": 1, "max_seq_length": MAX_SEQ_LENGTH_FALLBACK, "target_modules": TARGET_MODULES},
    {"step": 2, "max_seq_length": MAX_SEQ_LENGTH_FALLBACK, "target_modules": ATTENTION_MODULES},
)

# Parameter, value, reason. The names match the rubric's list.
HYPERPARAMETERS = (
    ("4-bit load", "true", "QLoRA keeps the base weights in 4-bit so a 1.5B model fits a free T4."),
    ("Quant type", "nf4", "NF4 is the QLoRA 4-bit type. It is not the default int4."),
    ("Double quantization", "true", "A second quantization of the quantization constants saves more T4 memory."),
    ("Compute dtype", COMPUTE_DTYPE, "T4 has no bfloat16. Forward and backward math stays in float16."),
    ("Load dtype / adapter dtype", "float16 / float32", "Unquantized layers load in float16, not Qwen's bfloat16 default. LoRA weights stay float32 because the float16 GradScaler cannot unscale bfloat16 or float16 gradients."),
    ("LoRA rank (r)", str(LORA_RANK), "Rank 16 is enough for 160 short JSON answers without a large adapter."),
    ("LoRA alpha", str(LORA_ALPHA), "Alpha is 2x rank, the usual QLoRA scale, so the update is not tiny."),
    ("LoRA dropout", str(LORA_DROPOUT), "0.05 limits memorizing the wording of 160 training scenarios."),
    ("LoRA bias", "none", "QLoRA does not train bias terms. The adapter is the weight update only."),
    ("Target modules", ", ".join(TARGET_MODULES), "Attention and MLP projections. Attention alone under-uses a 1.5B model on this task."),
    ("Learning rate", "2e-4", "Standard QLoRA rate for a small instruct model. Higher overfits 160 rows."),
    ("LR scheduler", LR_SCHEDULER, "Cosine decay after warmup. A constant rate keeps stepping hard at the end of 3 epochs."),
    ("Warmup ratio", str(WARMUP_RATIO), "3 percent of steps ramp the rate so the first updates are not full size."),
    ("Epochs", str(NUM_EPOCHS), "Three epochs give validation loss more than one step in which to fall."),
    ("Batch size", str(BATCH_SIZE), "One example per step. A larger microbatch is the first thing that OOMs on a T4."),
    ("Gradient accumulation steps", str(GRAD_ACCUM_STEPS), "Effective batch of 8. One epoch is about 20 optimizer steps."),
    ("Max sequence length", str(MAX_SEQ_LENGTH), "512 covers a 149-word scenario plus the prompt. A longer tokenized train row raises the cap to 768. An out-of-memory retry drops it to 384."),
    ("Pad token", "tokenizer pad (EOS if none)", "Qwen2.5 ships endoftext as its pad token. A tokenizer without one pads with EOS. Padded positions are masked from attention and loss."),
    ("Optimizer", OPTIMIZER, "8-bit paged AdamW keeps optimizer state off the T4 when memory spikes."),
    ("Weight decay", str(WEIGHT_DECAY), "Light decay. The set is small, so zero decay memorizes more easily."),
    ("Gradient checkpointing", "true", "Recomputes activations. This is what makes the MLP adapters fit."),
    ("Packing", "false", "Rows stay separate. Packing would glue two scenarios into one sequence."),
    ("Loss mask", "assistant tokens only", "The system and user turns are context. The loss is the JSON answer. The mask is built in qlora.py because TRL 0.20 dropped its completion-only collator."),
    ("Eval and save", "every epoch", "One validation number per epoch, which is what the loss rubric asks for."),
    ("Best checkpoint", "lowest eval loss", "The merge uses that checkpoint, not whatever the last epoch happened to be."),
    ("Seed", str(SEED), "Fixed seed so the same notebook run is repeatable."),
    ("Merge dtype", "float16 base, not 4-bit", "merge_and_unload on a 4-bit model is unreliable. The base is reloaded in float16."),
)

REQUIRED_PARAMETERS = (
    "LoRA rank (r)",
    "LoRA alpha",
    "Target modules",
    "Learning rate",
    "LR scheduler",
    "Epochs",
    "Batch size",
    "Gradient accumulation steps",
    "Max sequence length",
    "LoRA dropout",
    "Warmup ratio",
    "Optimizer",
    "Weight decay",
    "Compute dtype",
    "Double quantization",
    "Gradient checkpointing",
    "Eval and save",
    "Seed",
    "Quant type",
    "4-bit load",
)


def reason_table():
    """Rows of (parameter, value, reason). A blank reason is a bug."""
    return tuple(HYPERPARAMETERS)


def training_files():
    """The two JSONL names the trainer may read. Test is intentionally absent."""
    return (TRAIN_FILE, VAL_FILE)


def losses_by_epoch(log_history):
    """One train loss and one validation loss per epoch from a Trainer log.

    A logging step stores `loss`. An eval step stores `eval_loss`. Both carry
    `epoch`. Steps that are neither are ignored.
    """
    grouped = {}
    for row in log_history:
        if "epoch" not in row:
            continue
        # The Trainer logs fractional epochs (e.g. 0.98 or 1.0); rounding files each entry under its epoch.
        epoch = int(round(float(row["epoch"])))
        slot = grouped.setdefault(epoch, {})
        # When an epoch has several training-loss logs, the later one overwrites the earlier one,
        # so the table keeps the loss at the end of the epoch.
        if "eval_loss" in row:
            slot["val_loss"] = float(row["eval_loss"])
        elif "loss" in row:
            slot["train_loss"] = float(row["loss"])
    table = []
    for epoch in sorted(grouped):
        table.append(
            {
                "epoch": epoch,
                "train_loss": grouped[epoch].get("train_loss"),
                "val_loss": grouped[epoch].get("val_loss"),
            }
        )
    return table


def validation_loss_fell(table):
    """True only when the last validation loss is lower than the first.

    Missing values return False. The notebook must say so instead of claiming a drop.
    """
    values = [row["val_loss"] for row in table if row.get("val_loss") is not None]
    if len(values) < 2:
        return False
    return values[-1] < values[0]


MODEL_CARD = """---
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
tags:
- qlora
- compliance
---

# Northwind compliance assistant (Qwen2.5-1.5B, QLoRA)

Fine-tuned for a technical assessment. The company and the policy manual are fictional. This is not legal advice.

## Base model

`Qwen/Qwen2.5-1.5B-Instruct`, merged with a LoRA adapter (`merge_and_unload` onto a float16 copy, not the 4-bit copy).

## Dataset

200 synthetic employee scenarios written by a different teacher model (`openai/gpt-oss-120b`). Split 160 train / 20 validation / 20 test. The test split was not used for training. Each row is a Northwind policy question. The label is JSON: `policy_ids`, `required_action`, `rationale`.

## Training

4-bit NF4 QLoRA on a Colab T4. Rank 16, alpha 32, dropout 0.05. Target modules are the attention and MLP projections. Learning rate 2e-4, cosine schedule, 3 epochs, microbatch 1, gradient accumulation 8, max length 512. Loss is on the assistant tokens only. Optimizer `paged_adamw_8bit`. Compute dtype float16 because a T4 has no bfloat16.
"""
