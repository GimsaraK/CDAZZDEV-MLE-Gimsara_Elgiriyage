"""Load the fictional Northwind policy manual. The teacher may not go beyond this file."""
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'Implement the Task 2A dataset plan', Date: 2026-10-08 (see CITATIONS.md Entry 8)

import json
from typing import Dict, List

from . import config


def load_manual(path=None) -> Dict:
    """Return the manual dict. Raises FileNotFoundError if the JSON is missing."""
    manual_path = path or config.POLICY_MANUAL_PATH
    return json.loads(manual_path.read_text(encoding="utf-8"))


def policy_by_id(manual: Dict) -> Dict[str, Dict]:
    """Map policy id to its record. NONE is not a policy in the file."""
    return {policy["id"]: policy for policy in manual["policies"]}


def topics(manual: Dict) -> List[str]:
    """Topic name for each policy, in manual order. This is the diversity taxonomy."""
    return [policy["topic"] for policy in manual["policies"]]


def topic_policy_id(manual: Dict) -> Dict[str, str]:
    """Map a topic name to the one policy id that owns it."""
    return {policy["topic"]: policy["id"] for policy in manual["policies"]}


def manual_block(manual: Dict) -> str:
    """Plain-text manual embedded in the teacher system prompt."""
    lines = [f"Company: {manual['company']}", ""]
    for policy in manual["policies"]:
        lines.append(f"{policy['id']} | {policy['title']} | topic={policy['topic']}")
        lines.append(f"Rule: {policy['rule']}")
        lines.append(f"Required action when this policy applies: {policy['required_action']}")
        lines.append("")
    lines.append(
        f"{config.NONE_POLICY_ID}: no policy in this manual applies. "
        "Required action: tell the employee the manual does not cover this and to ask the compliance officer. "
        "Do not invent a rule."
    )
    return "\n".join(lines)
