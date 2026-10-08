"""Offline checks for the QLoRA settings. No GPU and no model download."""
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'Implement the Task 2B QLoRA plan', Date: 2026-10-08 (see CITATIONS.md Entry 10)

from task2_genai.src.qlora import IGNORE_INDEX, completion_labels
from task2_genai.src.train_config import (
    ATTENTION_MODULES,
    OOM_STEPS,
    REQUIRED_PARAMETERS,
    TARGET_MODULES,
    losses_by_epoch,
    reason_table,
    training_files,
    validation_loss_fell,
)


def test_every_required_parameter_has_a_value_and_a_reason():
    rows = {name: (value, reason) for name, value, reason in reason_table()}
    missing = [name for name in REQUIRED_PARAMETERS if name not in rows]
    assert missing == []
    for name, (value, reason) in rows.items():
        assert str(value).strip()
        assert len(reason.strip()) > 20, name


def test_trainer_reads_train_and_val_not_test():
    files = training_files()
    assert files == ("train.jsonl", "val.jsonl")
    assert "test.jsonl" not in files


def test_oom_ladder_shrinks_length_then_drops_mlp():
    assert [step["step"] for step in OOM_STEPS] == [0, 1, 2]
    assert OOM_STEPS[0]["max_seq_length"] == 512
    assert OOM_STEPS[1]["max_seq_length"] == 384
    assert OOM_STEPS[1]["target_modules"] == TARGET_MODULES
    assert OOM_STEPS[2]["target_modules"] == ATTENTION_MODULES
    assert "gate_proj" not in OOM_STEPS[2]["target_modules"]


def test_epoch_table_keeps_train_and_val_and_notices_a_drop():
    history = [
        {"epoch": 1.0, "loss": 1.5},
        {"epoch": 1.0, "eval_loss": 1.4},
        {"epoch": 2.0, "loss": 1.1},
        {"epoch": 2.0, "eval_loss": 1.2},
        {"step": 3, "learning_rate": 0.0},
    ]
    table = losses_by_epoch(history)
    assert table == [
        {"epoch": 1, "train_loss": 1.5, "val_loss": 1.4},
        {"epoch": 2, "train_loss": 1.1, "val_loss": 1.2},
    ]
    assert validation_loss_fell(table) is True
    flat = [
        {"epoch": 1, "train_loss": 1.0, "val_loss": 1.0},
        {"epoch": 2, "train_loss": 0.5, "val_loss": 1.2},
    ]
    assert validation_loss_fell(flat) is False


def test_only_tokens_after_the_assistant_header_are_scored():
    header = [7, 8, 9]
    # system / user context, then the header, then a 3-token answer.
    row = [1, 2, 3, 7, 8, 4, 7, 8, 9, 50, 51, 52]
    labels = completion_labels(row, header)
    assert labels[:9] == [IGNORE_INDEX] * 9
    assert labels[9:] == [50, 51, 52]


def test_row_without_the_header_scores_nothing():
    labels = completion_labels([1, 2, 3, 7, 8], [7, 8, 9])
    assert labels == [IGNORE_INDEX] * 5
