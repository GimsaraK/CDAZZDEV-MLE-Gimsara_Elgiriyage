"""Turn an accepted example into a Qwen2.5 ChatML training row.

The string matches the Qwen2.5-Instruct template when the first message is the
system turn and add_generation_prompt is false: each turn is
<|im_start|>role\\ncontent<|im_end|>\\n
The notebook checks one row against the tokenizer. Tests use this function so
they do not download the model.
"""
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'Implement the Task 2A dataset plan', Date: 2026-10-08 (see CITATIONS.md Entry 8)

import json
from typing import Dict, List

from .prompts import STUDENT_SYSTEM
from .schema import TeacherExample


def assistant_json(example: TeacherExample) -> str:
    """Gold label the student must emit. Compact and ASCII so the split files stay stable."""
    payload = {
        "policy_ids": example.policy_ids,
        "rationale": example.rationale,
        "required_action": example.required_action,
    }
    # sort_keys + compact separators give one canonical string per label: the same answer always
    # serialises identically, so the model learns one exact format and ROUGE compares like with like.
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def messages_for(example: TeacherExample) -> List[Dict[str, str]]:
    """System, user, assistant. The user turn is only the scenario, not the manual."""
    return [
        {"role": "system", "content": STUDENT_SYSTEM.strip()},
        {"role": "user", "content": example.scenario},
        {"role": "assistant", "content": assistant_json(example)},
    ]


def render_qwen_chatml(messages: List[Dict[str, str]]) -> str:
    """Qwen2.5 ChatML with the assistant turn closed. No generation prompt is added."""
    parts = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        # Each turn is wrapped in Qwen's special tokens; the tokenizer turns these markers into single ids.
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    return "".join(parts)


def training_row(example: TeacherExample) -> Dict:
    """JSONL object: the three turns plus the rendered template string."""
    messages = messages_for(example)
    return {
        "messages": messages,
        "text": render_qwen_chatml(messages),
        "topic": example.topic,
        "situation": example.situation,
    }
