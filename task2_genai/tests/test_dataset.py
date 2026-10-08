"""Offline checks for the Task 2A dataset rules. No network and no teacher model."""
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'Implement the Task 2A dataset plan', Date: 2026-10-08 (see CITATIONS.md Entry 8)

import json

import pytest

from task2_genai.src import config
from task2_genai.src.dedup import DuplicateIndex, jaccard, char_ngrams
from task2_genai.src.diversity import diversity_report
from task2_genai.src.format import render_qwen_chatml, training_row
from task2_genai.src.generate import generate_cell
from task2_genai.src.manual import load_manual, topic_policy_id
from task2_genai.src.schema import TeacherExample, validate_example
from task2_genai.src.split import split_rows


def _words(n: int, salt: str = "alpha") -> str:
    """A scenario long enough to pass the word gate, and distinct when salt changes."""
    return " ".join(f"{salt}{i}" for i in range(n))


def _payload(situation="clear_breach", policy_ids=None, scenario=None, topic="leave"):
    return {
        "scenario": scenario or _words(50),
        "policy_ids": policy_ids or ["NW-LEAVE"],
        "required_action": "Do the thing the clause names.",
        "rationale": "The rule covers this fact.",
        "topic": topic,
        "situation": situation,
    }


@pytest.fixture(scope="module")
def manual():
    return load_manual()


def test_schema_rejects_an_unknown_policy_id(manual):
    ids = {policy["id"] for policy in manual["policies"]}
    with pytest.raises(ValueError, match="unknown policy"):
        validate_example(_payload(policy_ids=["NW-MADE-UP"]), ids, "leave", "clear_breach", "NW-LEAVE")


def test_schema_rejects_a_short_scenario(manual):
    ids = {policy["id"] for policy in manual["policies"]}
    with pytest.raises(ValueError, match="words"):
        validate_example(_payload(scenario=_words(10)), ids, "leave", "clear_breach", "NW-LEAVE")


def test_not_covered_must_be_none_and_two_policy_needs_both_ids(manual):
    ids = {policy["id"] for policy in manual["policies"]}
    with pytest.raises(ValueError, match="NONE"):
        validate_example(
            _payload(situation="not_covered", policy_ids=["NW-LEAVE"]),
            ids,
            "leave",
            "not_covered",
            "NW-LEAVE",
        )
    example = validate_example(
        _payload(situation="two_policy", policy_ids=["NW-LEAVE", "NW-GIFT"]),
        ids,
        "leave",
        "two_policy",
        "NW-LEAVE",
    )
    assert example.policy_ids == ["NW-LEAVE", "NW-GIFT"]


def test_near_duplicate_is_dropped_and_exact_duplicate_is_dropped():
    index = DuplicateIndex()
    original = _words(50, "beta")
    assert index.consider(original) is None
    assert index.consider(original) == "exact"
    # One token differs, so the character 5-gram overlap stays above the 0.80 cut.
    near = original.replace("beta0", "betaX")
    assert jaccard(char_ngrams(original), char_ngrams(near)) >= config.NEAR_DUP_JACCARD
    assert index.consider(near) == "near"
    assert index.exact_dropped == 1
    assert index.near_dropped == 1


def test_split_is_80_10_10_and_every_topic_is_in_each_split():
    rows = []
    for topic in range(8):
        for number in range(10):
            rows.append({"topic": f"t{topic}", "situation": "clear_breach", "n": number})
    train, val, test = split_rows(rows, 64, 8, 8, seed=42)
    assert (len(train), len(val), len(test)) == (64, 8, 8)
    for part in (train, val, test):
        assert {row["topic"] for row in part} == {f"t{topic}" for topic in range(8)}


def test_chatml_contains_the_three_roles_and_closes_the_assistant_turn(manual):
    ids = {policy["id"] for policy in manual["policies"]}
    example = validate_example(_payload(), ids, "leave", "clear_breach", topic_policy_id(manual)["leave"])
    row = training_row(example)
    text = row["text"]
    assert text == render_qwen_chatml(row["messages"])
    assert "<|im_start|>system\n" in text
    assert "<|im_start|>user\n" in text
    assert "<|im_start|>assistant\n" in text
    assert text.endswith("<|im_end|>\n")
    label = json.loads(row["messages"][2]["content"])
    assert label["policy_ids"] == ["NW-LEAVE"]
    # The user turn is the scenario only. The manual is not pasted in.
    assert "NW-LEAVE" not in row["messages"][1]["content"]


def test_diversity_report_counts_topics_and_keywords():
    example = TeacherExample.model_validate(_payload(scenario=_words(40, "receipt")))
    row = training_row(example)
    report = diversity_report([row, row])
    assert report["topic_counts"] == {"leave": 2}
    assert report["length_words"]["n"] == 2
    assert any(item["token"] == "receipt0" for item in report["top_keywords"])


class _Scripted:
    """Teacher stand-in. Each complete() call returns the next canned string."""

    def __init__(self, payloads, provider="groq"):
        self._payloads = list(payloads)
        self.provider = provider

    def complete(self, messages):
        return self._payloads.pop(0)


def test_generate_cell_repairs_invalid_json_then_keeps_the_valid_row(manual):
    ids = {policy["id"] for policy in manual["policies"]}
    good = _payload()
    valid = json.dumps({"examples": [good]})
    client = _Scripted(["{not json", valid])
    stats = {"returned": 0, "schema_rejected": 0, "repair_calls": 0, "fallback_calls": 0, "provider_failed": 0}
    kept = generate_cell(client, manual, "leave", "clear_breach", DuplicateIndex(), stats, fallback=None)
    assert stats["repair_calls"] == 1
    assert len(kept) == 1
    assert kept[0].policy_ids == ["NW-LEAVE"]
    assert ids  # the manual ids are what the cell is checked against
