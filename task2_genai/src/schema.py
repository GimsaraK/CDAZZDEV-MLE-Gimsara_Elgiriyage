"""Pydantic check for one teacher example. Invalid rows are not written to JSONL."""
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'Implement the Task 2A dataset plan', Date: 2026-10-08 (see CITATIONS.md Entry 8)

from typing import List

from pydantic import BaseModel, Field, field_validator, model_validator

from . import config
from .textutil import to_ascii, word_count


class TeacherExample(BaseModel):
    """One scenario plus the gold JSON the student is trained to emit."""

    scenario: str
    policy_ids: List[str] = Field(min_length=1)
    required_action: str
    rationale: str
    topic: str
    situation: str

    @field_validator("scenario", "required_action", "rationale", "topic", "situation")
    @classmethod
    def _ascii_text(cls, value: str) -> str:
        cleaned = to_ascii(value).strip()
        if not cleaned:
            raise ValueError("field is blank")
        return cleaned

    @field_validator("policy_ids")
    @classmethod
    def _ascii_ids(cls, value: List[str]) -> List[str]:
        cleaned = []
        for item in value:
            text = to_ascii(str(item)).strip()
            if not text:
                raise ValueError("policy id is blank")
            cleaned.append(text)
        return cleaned

    @field_validator("scenario")
    @classmethod
    def _scenario_length(cls, value: str) -> str:
        # Outside this band the row is too thin, or it is an essay the later model cannot learn.
        count = word_count(value)
        if count < config.MIN_SCENARIO_WORDS or count > config.MAX_SCENARIO_WORDS:
            raise ValueError(
                f"scenario has {count} words; expected {config.MIN_SCENARIO_WORDS}-{config.MAX_SCENARIO_WORDS}"
            )
        return value


def validate_example(payload: dict, manual_ids: set, topic: str, situation: str, primary_id: str) -> TeacherExample:
    """Parse one object and enforce the cell's id rule.

    The model is told the cell in the user message. We still reject a row that
    names a different topic, a different situation, or an id the manual does not contain.
    """
    example = TeacherExample.model_validate(payload)
    if example.topic != topic or example.situation != situation:
        raise ValueError(f"cell mismatch: got {example.topic}/{example.situation}, expected {topic}/{situation}")
    unknown = [policy_id for policy_id in example.policy_ids if policy_id not in manual_ids and policy_id != config.NONE_POLICY_ID]
    if unknown:
        raise ValueError(f"unknown policy ids: {unknown}")
    if len(example.policy_ids) != len(set(example.policy_ids)):
        raise ValueError("policy_ids contains a duplicate")

    # The id rule is the definition of a correct label for this situation.
    if situation == "not_covered":
        if example.policy_ids != [config.NONE_POLICY_ID]:
            raise ValueError("not_covered must use policy_ids ['NONE']")
    elif situation == "two_policy":
        if len(example.policy_ids) != 2 or primary_id not in example.policy_ids:
            raise ValueError("two_policy must be the cell policy plus one other manual id")
        if config.NONE_POLICY_ID in example.policy_ids:
            raise ValueError("two_policy cannot use NONE")
    else:
        if example.policy_ids != [primary_id]:
            raise ValueError(f"{situation} must use policy_ids ['{primary_id}']")
    return example
