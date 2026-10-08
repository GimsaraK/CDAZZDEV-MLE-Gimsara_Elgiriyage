"""QLoRA load, train, plot, and merge helpers for the Colab notebook.

torch, transformers, peft, and trl are imported inside the functions so the
offline tests can import the package on a machine without a GPU stack.
"""
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'Implement the Task 2B QLoRA plan', Date: 2026-10-08 (see CITATIONS.md Entry 10)

import inspect
from pathlib import Path

from . import config
from . import train_config as tc


def float16_kwargs(torch) -> dict:
    """Load-dtype argument for from_pretrained. transformers 4.56 renamed torch_dtype to dtype.

    Without it, newer transformers loads Qwen2.5 in its config dtype (bfloat16),
    which a T4 cannot train with float16 mixed precision.
    """
    import transformers
    from packaging.version import Version

    if Version(transformers.__version__) >= Version("4.56.0"):
        return {"dtype": torch.float16}
    return {"torch_dtype": torch.float16}


def upcast_trainable_to_float32(model) -> int:
    """Cast every trainable weight to float32 and return how many were changed.

    float16 mixed precision unscales gradients with a GradScaler, which fails on
    bfloat16 or float16 gradients. The LoRA weights are the only trainable ones, so
    keeping them in float32 costs little memory. The 4-bit base stays frozen.
    """
    import torch

    changed = 0
    for param in model.parameters():
        if param.requires_grad and param.dtype != torch.float32:
            param.data = param.data.to(torch.float32)
            changed += 1
    return changed


def load_student(target_modules):
    """4-bit NF4 base model plus a LoRA adapter. Caller must be on CUDA.

    Compute dtype is float16 because a T4 does not support bfloat16.
    Returns the peft model and the tokenizer. The pad token is the EOS token
    when the tokenizer has none, which Qwen2.5 often does.
    """
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # NF4 and double quantization are set explicitly. They are not left as library defaults.
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.STUDENT_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.STUDENT_MODEL,
        quantization_config=quantization,
        device_map="auto",
        # The layers that stay unquantized load in float16, not the config's bfloat16.
        **float16_kwargs(torch),
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    adapter = LoraConfig(
        r=tc.LORA_RANK,
        lora_alpha=tc.LORA_ALPHA,
        lora_dropout=tc.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(target_modules),
    )
    model = get_peft_model(model, adapter)
    upcast_trainable_to_float32(model)
    return model, tokenizer


IGNORE_INDEX = -100


def completion_labels(input_ids, template_ids):
    """Labels for one row: -100 up to and including the last assistant header.

    Only the answer after that header is scored. A row whose header was
    truncated away gets no scored tokens at all.
    """
    input_ids = [int(token) for token in input_ids]
    template_ids = [int(token) for token in template_ids]
    labels = [IGNORE_INDEX] * len(input_ids)
    width = len(template_ids)
    for start in range(len(input_ids) - width, -1, -1):
        if input_ids[start:start + width] == template_ids:
            begin = start + width
            labels[begin:] = input_ids[begin:]
            break
    return labels


def _template_ids(tokenizer):
    return tokenizer(tc.RESPONSE_TEMPLATE, add_special_tokens=False)["input_ids"]


def make_completion_collator(tokenizer, max_length):
    """Pad a batch and score only the assistant answer.

    TRL 0.20 removed DataCollatorForCompletionOnlyLM, and Colab installs the
    latest TRL, so the masking lives here instead. Newer TRL hands the collator
    input_ids without an attention mask, so the mask is rebuilt from row lengths.
    """
    import torch

    template_ids = _template_ids(tokenizer)
    pad_id = tokenizer.pad_token_id

    def collate(features):
        rows = [[int(token) for token in feature["input_ids"]][:max_length] for feature in features]
        width = max(len(row) for row in rows)
        input_ids, attention, labels = [], [], []
        for row in rows:
            gap = width - len(row)
            input_ids.append(row + [pad_id] * gap)
            attention.append([1] * len(row) + [0] * gap)
            labels.append(completion_labels(row, template_ids) + [IGNORE_INDEX] * gap)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


def resolve_max_length(tokenizer, requested):
    """Return (cap, longest train row). The cap becomes 768 only if a row does not fit.

    Validation and test are not measured. Later out-of-memory steps keep their
    own shorter cap and do not call this again.
    """
    from datasets import load_dataset

    rows = load_dataset("json", data_files=str(config.TRAIN_PATH), split="train")
    longest = 0
    for text in rows["text"]:
        count = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        if count > longest:
            longest = count
    if longest > requested:
        return 768, longest
    return requested, longest


def make_trainer(model, tokenizer, max_seq_length, output_dir: Path):
    """SFT trainer on train.jsonl and val.jsonl. Test is not loaded.

    Loss is masked to the assistant span. The response marker is the ChatML
    assistant header already stored in each row's text field.
    """
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(config.TRAIN_PATH),
            "validation": str(config.VAL_PATH),
        },
    )
    # Keep only the rendered ChatML. Newer TRL sees a `messages` column and
    # switches to conversational handling, which ignores dataset_text_field.
    dataset = dataset.select_columns(["text"])
    # Stop here if the assistant header is not found. Otherwise every label is
    # masked and the run trains on nothing.
    first = tokenizer(dataset["train"][0]["text"], add_special_tokens=False)["input_ids"][:max_seq_length]
    if all(label == IGNORE_INDEX for label in completion_labels(first, _template_ids(tokenizer))):
        raise ValueError("Assistant header %r not found in the first train row." % tc.RESPONSE_TEMPLATE)
    collator = make_completion_collator(tokenizer, max_seq_length)
    # TRL has renamed arguments across versions. Keep only keys this install accepts.
    config_params = inspect.signature(SFTConfig.__init__).parameters
    kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": tc.NUM_EPOCHS,
        "per_device_train_batch_size": tc.BATCH_SIZE,
        "per_device_eval_batch_size": tc.BATCH_SIZE,
        "gradient_accumulation_steps": tc.GRAD_ACCUM_STEPS,
        "learning_rate": tc.LEARNING_RATE,
        "lr_scheduler_type": tc.LR_SCHEDULER,
        "warmup_ratio": tc.WARMUP_RATIO,
        "optim": tc.OPTIMIZER,
        "weight_decay": tc.WEIGHT_DECAY,
        "fp16": True,
        "bf16": False,
        "gradient_checkpointing": True,
        "logging_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "seed": tc.SEED,
        "report_to": "none",
        "packing": False,
        "dataset_text_field": "text",
    }
    # The eval-strategy and max-length argument names changed between TRL releases.
    if "eval_strategy" in config_params:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    # Older TRL puts the length on the trainer. Newer TRL puts it on SFTConfig.
    length_on_config = False
    if "max_length" in config_params:
        kwargs["max_length"] = max_seq_length
        length_on_config = True
    elif "max_seq_length" in config_params:
        kwargs["max_seq_length"] = max_seq_length
        length_on_config = True
    accepts_any = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in config_params.values())
    if accepts_any:
        args = SFTConfig(**kwargs)
    else:
        args = SFTConfig(**{key: value for key, value in kwargs.items() if key in config_params})

    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset["validation"],
        "data_collator": collator,
    }
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    if "dataset_text_field" in trainer_params:
        trainer_kwargs["dataset_text_field"] = "text"
    if not length_on_config:
        if "max_seq_length" in trainer_params:
            trainer_kwargs["max_seq_length"] = max_seq_length
        elif "max_length" in trainer_params:
            trainer_kwargs["max_length"] = max_seq_length
    trainer = SFTTrainer(**trainer_kwargs)
    # Some TRL versions recast the adapter while building the trainer. Check again just before training.
    upcast_trainable_to_float32(trainer.model)
    return trainer


def save_loss_plot(table, path: Path):
    """Train and validation loss on one chart. `table` is the list from losses_by_epoch."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in table]
    fig, axis = plt.subplots(figsize=(6, 3.5))
    train = [row["train_loss"] for row in table]
    val = [row["val_loss"] for row in table]
    if any(value is not None for value in train):
        axis.plot(epochs, train, marker="o", label="train")
    if any(value is not None for value in val):
        axis.plot(epochs, val, marker="o", label="validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title("QLoRA train and validation loss")
    axis.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def merge_and_push(adapter_dir: Path, save_dir: Path, token: str) -> str:
    """Merge the adapter into a fresh float16 base model and push that model.

    The 4-bit training model is not merged. merge_and_unload on a quantized
    model drops or corrupts weights. `token` is used for the upload and is not logged.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.STUDENT_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # float16, not 4-bit. This is a second load, independent of the trainer model.
    base = AutoModelForCausalLM.from_pretrained(
        config.STUDENT_MODEL,
        device_map="auto",
        **float16_kwargs(torch),
    )
    merged = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = merged.merge_and_unload()
    save_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    (save_dir / "README.md").write_text(tc.MODEL_CARD, encoding="utf-8", newline="\n")
    # Upload the saved folder so the model card goes up with the weights.
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(tc.HF_REPO_ID, token=token, exist_ok=True, private=False)
    api.upload_folder(folder_path=str(save_dir), repo_id=tc.HF_REPO_ID, token=token)
    return "https://huggingface.co/" + tc.HF_REPO_ID
