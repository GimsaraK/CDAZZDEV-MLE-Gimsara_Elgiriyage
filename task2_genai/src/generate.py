"""Ask the teacher for one cell at a time, keep rows that pass schema and dedup, then split.

A cell is one topic crossed with one situation. The run resumes from
data/accepted.jsonl, so a second start does not call the teacher again for
cells that already have 5 accepted rows.
"""
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'Implement the Task 2A dataset plan', Date: 2026-10-08 (see CITATIONS.md Entry 8)

import json
import logging
import re
from collections import Counter
from typing import Dict, List, Optional

from pydantic import ValidationError

from . import config
from .client import ChatClient, build_clients, load_api_keys
from .dedup import DuplicateIndex
from .format import training_row
from .manual import load_manual, topic_policy_id, topics
from .prompts import teacher_system, teacher_user, write_prompt_files
from .schema import TeacherExample, validate_example
from .diversity import write_diversity_outputs
from .split import split_rows

logger = logging.getLogger("task2.generate")

# Matches an opening ``` or ```json at the start, or a closing ``` at the end, so both can be removed.
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _strip_fence(text: str) -> str:
    """Some models wrap JSON in a fence even when JSON mode was requested."""
    return _FENCE.sub("", text.strip())


def _parse_examples(raw: str) -> List[dict]:
    """Pull the examples array out of the teacher JSON. A missing key is an error."""
    payload = json.loads(_strip_fence(raw))
    examples = payload.get("examples")
    if not isinstance(examples, list):
        raise ValueError("teacher JSON has no examples array")
    return examples


def _validate_batch(raw_items: List[dict], manual_ids: set, topic: str, situation: str, primary_id: str):
    """Split a batch into accepted models and human-readable rejection strings."""
    accepted: List[TeacherExample] = []
    errors: List[str] = []
    for index, item in enumerate(raw_items):
        try:
            accepted.append(validate_example(item, manual_ids, topic, situation, primary_id))
        except (ValidationError, ValueError) as exc:
            errors.append(f"item {index}: {exc}")
    return accepted, errors


def _repair_message(system: str, user: str, raw: str, errors: List[str]) -> List[dict]:
    """Second user turn. The model sees the validation errors and must return the JSON again."""
    repair = (
        "The previous JSON failed validation. Fix every item and return the same JSON shape.\n"
        "Errors:\n- " + "\n- ".join(errors) + "\n"
        "Previous response:\n" + raw[:4000]
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "user", "content": repair},
    ]


def _from_provider(client: ChatClient, messages, manual_ids, topic, situation, primary_id, stats):
    """One completion plus one repair. Returns schema-valid examples, before dedup."""
    system = messages[0]["content"]
    user = messages[1]["content"]
    # A network error from complete() is not caught here: it goes up to generate_cell, which
    # moves on to the fallback teacher. Only bad *content* is handled in this function.
    raw = client.complete(messages)
    try:
        items = _parse_examples(raw)
        accepted, errors = _validate_batch(items, manual_ids, topic, situation, primary_id)
    except (json.JSONDecodeError, ValueError) as exc:
        # Unparseable JSON: treat the whole response as one error so the repair turn can fix it.
        items, accepted, errors = [], [], [str(exc)]
    if errors:
        # The same provider gets one chance to fix its own JSON. This is not a retry of a network error.
        stats["repair_calls"] += 1
        try:
            raw = client.complete(_repair_message(system, user, raw, errors))
            items = _parse_examples(raw)
            accepted, errors = _validate_batch(items, manual_ids, topic, situation, primary_id)
        except Exception as exc:  # noqa: BLE001 - repair failure just yields no rows from this provider
            logger.warning("Repair failed for %s/%s via %s: %s", topic, situation, client.provider, exc)
            return []
    stats["returned"] += len(items)
    stats["schema_rejected"] += len(errors)
    return accepted


def generate_cell(
    client: ChatClient,
    manual,
    topic: str,
    situation: str,
    index: DuplicateIndex,
    stats: Dict[str, int],
    fallback: Optional[ChatClient] = None,
) -> List[TeacherExample]:
    """Groq call, one repair, then dedup. OpenRouter is used only when Groq yields no valid row."""
    manual_ids = {policy["id"] for policy in manual["policies"]}
    primary_id = topic_policy_id(manual)[topic]
    system = teacher_system(manual)
    user = teacher_user(manual, topic, situation)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    try:
        accepted = _from_provider(client, messages, manual_ids, topic, situation, primary_id, stats)
    except Exception as exc:  # noqa: BLE001 - provider/transport failure falls through to the other teacher
        logger.warning("%s failed for %s/%s: %s", client.provider, topic, situation, exc)
        accepted = []
        stats["provider_failed"] += 1

    # A validation failure on Groq is the only reason to spend an OpenRouter call.
    if not accepted and fallback is not None and fallback.provider != client.provider:
        stats["fallback_calls"] += 1
        try:
            accepted = _from_provider(fallback, messages, manual_ids, topic, situation, primary_id, stats)
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s fallback failed for %s/%s: %s", fallback.provider, topic, situation, exc)
            return []

    # Dedup runs last, against every scenario accepted so far (in this cell and all earlier ones).
    # consider() also adds each kept scenario to the index, so two near-copies in one batch are caught.
    kept: List[TeacherExample] = []
    for example in accepted:
        if index.consider(example.scenario) is None:
            kept.append(example)
    return kept


def _read_jsonl(path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8", newline="\n")


def _cell_counts(rows: List[dict]) -> Counter:
    return Counter((row["topic"], row["situation"]) for row in rows)


def generate_dataset(clients: Optional[Dict[str, ChatClient]] = None) -> Dict[str, int]:
    """Fill each taxonomy cell to 5 accepted rows, then write the three split files.

    Returns the generation counters. Does not call the teacher when accepted.jsonl
    already holds the target.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    manual = load_manual()
    write_prompt_files(manual)
    rows = _read_jsonl(config.ACCEPTED_PATH)
    index = DuplicateIndex()
    for row in rows:
        # Rebuild the index from the user turn so a resumed run still catches copies.
        user = next(message["content"] for message in row["messages"] if message["role"] == "user")
        index.add_existing(user)

    stats = {
        "returned": 0,
        "schema_rejected": 0,
        "exact_dropped": 0,
        "near_dropped": 0,
        "accepted": len(rows),
        "repair_calls": 0,
        "fallback_calls": 0,
        "provider_failed": 0,
    }
    if len(rows) >= config.TARGET_ACCEPTED:
        # A finished run already wrote the report, including repair counts. Do not replace it with zeros.
        logger.info("Loaded %d accepted rows; teacher was not called", len(rows))
        if not (config.TRAIN_PATH.exists() and config.GENERATION_REPORT_PATH.exists()):
            _write_splits(rows[: config.TARGET_ACCEPTED], stats, index)
        return stats

    if clients is None:
        load_api_keys()
        clients = build_clients()
    # Groq is the teacher when its key is set; OpenRouter is then only the fallback. With only an
    # OpenRouter key, OpenRouter becomes the teacher and there is no fallback.
    primary = clients.get(config.LLM_PROVIDER_GROQ) or clients.get(config.LLM_PROVIDER_OPENROUTER)
    if primary is None:
        raise RuntimeError("No teacher API key is set")
    fallback = clients.get(config.LLM_PROVIDER_OPENROUTER) if primary.provider == config.LLM_PROVIDER_GROQ else None

    # Cells that are still short. Order is stable so a resumed run continues at the first gap.
    for topic in topics(manual):
        for situation in config.SITUATIONS:
            calls = 0
            # Keep asking the teacher until this cell has its quota, but never more than
            # MAX_CALLS_PER_CELL times: a cell the teacher keeps failing must not loop forever.
            while _cell_counts(rows)[(topic, situation)] < config.TARGET_PER_CELL and calls < config.MAX_CALLS_PER_CELL:
                if len(rows) >= config.TARGET_ACCEPTED:
                    break
                calls += 1
                fresh = generate_cell(primary, manual, topic, situation, index, stats, fallback)
                for example in fresh:
                    # A batch can return more rows than the cell still needs; extra rows are discarded
                    # so every cell ends with exactly TARGET_PER_CELL rows (balanced topics).
                    if _cell_counts(rows)[(topic, situation)] >= config.TARGET_PER_CELL:
                        break
                    if len(rows) >= config.TARGET_ACCEPTED:
                        break
                    rows.append(training_row(example))
                # Saved after every call, so an interrupted run resumes from here without repeating calls.
                _write_jsonl(config.ACCEPTED_PATH, rows)
                logger.info(
                    "Cell %s/%s now %d; accepted %d",
                    topic,
                    situation,
                    _cell_counts(rows)[(topic, situation)],
                    len(rows),
                )

    stats["accepted"] = len(rows)
    stats["exact_dropped"] = index.exact_dropped
    stats["near_dropped"] = index.near_dropped
    if len(rows) != config.TARGET_ACCEPTED:
        raise RuntimeError(f"Accepted {len(rows)} rows; expected {config.TARGET_ACCEPTED}")
    _write_splits(rows, stats, index)
    return stats


def _write_splits(rows: List[dict], stats: Dict[str, int], index: DuplicateIndex) -> None:
    """Write train/val/test and the generation report. Diversity plots are a separate step."""
    stats["exact_dropped"] = index.exact_dropped
    stats["near_dropped"] = index.near_dropped
    stats["accepted"] = len(rows)
    train, val, test = split_rows(
        rows, config.TRAIN_SIZE, config.VAL_SIZE, config.TEST_SIZE, config.SPLIT_SEED
    )
    _write_jsonl(config.TRAIN_PATH, train)
    _write_jsonl(config.VAL_PATH, val)
    _write_jsonl(config.TEST_PATH, test)
    # Charts and the length / keyword report cover the full 200, not one split.
    write_diversity_outputs(rows)
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    config.GENERATION_REPORT_PATH.write_text(
        json.dumps(
            {
                **stats,
                "train": len(train),
                "val": len(val),
                "test": len(test),
                "teacher_model": config.TEACHER_MODEL_GROQ,
                "student_model": config.STUDENT_MODEL,
            },
            indent=2,
        ),
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    """Entry point: python -m task2_genai.src.generate"""
    stats = generate_dataset()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
