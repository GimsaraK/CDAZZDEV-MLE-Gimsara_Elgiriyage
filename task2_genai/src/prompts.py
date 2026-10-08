"""Teacher and student prompts.

The teacher prompt is the data-generation instruction and includes the manual.
The student prompt is the system turn stored in every training row. They are
not the same text, and the student model is not the teacher model.
"""
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'Implement the Task 2A dataset plan', Date: 2026-10-08 (see CITATIONS.md Entry 8)

from . import config
from .manual import load_manual, manual_block, topic_policy_id

# Stable instruction. The manual text is appended by teacher_system().
_TEACHER_PREAMBLE = """You write training rows for a compliance assistant at Northwind Components, a fictional company.
Use only the policy manual below. Do not invent policies, dollar limits, deadlines, or actions that the manual does not state.

Return a JSON object with one key, "examples". Its value is an array of exactly 5 objects.
Each object has these keys and no others:
- scenario: the employee situation, in the word range given in the user message. Name a role and concrete facts. Do not mention a policy id in the scenario.
- policy_ids: array of policy ids, following the user message.
- required_action: the action that clause requires, in one or two sentences, using only the manual's required action.
- rationale: one or two sentences that cite the rule. Do not add a new rule.
- topic: copy the topic string from the user message.
- situation: copy the situation string from the user message.

The 5 scenarios must differ in role, workplace, and facts. Do not reuse the same person names or the same numbers.

Policy manual:
"""

STUDENT_SYSTEM = """You are the compliance assistant for Northwind Components. Answer only from Northwind policy you were trained on. Do not invent rules, limits, or deadlines.
Return a JSON object with exactly these keys:
- policy_ids: an array of policy ids, or ["NONE"] when no Northwind policy applies
- required_action: what the employee or the manager must do
- rationale: one or two sentences citing the rule
Do not wrap the JSON in markdown.
"""

# The cell (topic, situation, id rule, length) is the user turn, so the system prompt stays one document.
TEACHER_USER = """Write 5 examples for this cell.
topic: {topic}
situation: {situation}
policy_ids rule: {id_rule}
scenario length: {length_hint}
"""


# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'Implement the Task 2C evaluation plan' (see CITATIONS.md Entry 12)
# Task 2C judge. It grades one answer at a time and is not told which model wrote it.
_JUDGE_PREAMBLE = """You grade answers from a compliance assistant for Northwind Components, a fictional company.
Grade only against the policy manual below. The reference answer was written from the same manual but can contain mistakes. When the reference and the manual disagree, the manual wins.

Score each criterion 0, 1, or 2:
- policy_correct: 2 = the policy_ids are exactly the ones the manual applies to this scenario (["NONE"] when no rule covers it). 1 = partly right, for example one of two ids, or a correct id plus a wrong one. 0 = wrong, missing, or not in the manual.
- action_faithful: 2 = the required_action is what the applicable clause requires. 1 = in the right direction but incomplete or partly wrong. 0 = contradicts the clause, or no action is given.
- rationale_grounded: 2 = the rationale states the actual rule from the manual. 1 = vague or partly accurate. 0 = cites a rule that does not exist or does not apply.
- no_invention: 2 = no invented rule, limit, deadline, amount, or policy id. 1 = one minor unsupported detail. 0 = an invented rule, limit, deadline, amount, or id.

If the answer is not valid JSON, grade what it says. Score 0 for any criterion it does not address.

Return only a JSON object with exactly these keys: policy_correct, action_faithful, rationale_grounded, no_invention.
Each value is an object {"score": 0, 1 or 2, "reason": "one sentence"}. No other keys and no markdown.

Policy manual:
"""

JUDGE_USER = """Scenario:
{scenario}

Reference answer:
{reference}

Answer to grade:
{candidate}
"""


def judge_system(manual=None) -> str:
    """Full judge system prompt, including the manual."""
    manual = manual if manual is not None else load_manual()
    return _JUDGE_PREAMBLE + manual_block(manual)


def judge_user(scenario: str, reference: str, candidate: str) -> str:
    """One answer to grade. An empty answer is shown as such, not as a blank line."""
    return JUDGE_USER.format(
        scenario=scenario,
        reference=reference,
        candidate=candidate.strip() if (candidate or "").strip() else "(empty answer)",
    )


def write_judge_prompt(path, manual=None) -> None:
    """Save the judge system prompt for the README and the notebook."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(judge_system(manual), encoding="utf-8", newline="\n")


def teacher_system(manual=None) -> str:
    """Full teacher system prompt, including the manual. This is the text sent to the model."""
    manual = manual if manual is not None else load_manual()
    return _TEACHER_PREAMBLE + manual_block(manual)


def id_rule(manual, topic: str, situation: str) -> str:
    """Tell the teacher which ids are legal for this cell. The schema checks the same rule."""
    primary = topic_policy_id(manual)[topic]
    if situation == "not_covered":
        return (
            f'Use exactly ["{config.NONE_POLICY_ID}"]. The scenario should look related to {topic} '
            "but must not actually be covered by any rule in the manual."
        )
    if situation == "two_policy":
        return (
            f'Use exactly two ids: "{primary}" and one other id from the manual, not {config.NONE_POLICY_ID}. '
            "The scenario must need both rules at once."
        )
    if situation == "compliant":
        return (
            f'Use exactly ["{primary}"]. The scenario must follow that rule, '
            "so the required action is the allowed outcome, not a penalty."
        )
    if situation == "borderline":
        return (
            f'Use exactly ["{primary}"]. The facts should sit close to the rule boundary '
            "but still fall on the breach side of that one policy."
        )
    return f'Use exactly ["{primary}"]. The scenario is a clear breach of that one policy.'


def teacher_user(manual, topic: str, situation: str) -> str:
    """User turn for one taxonomy cell."""
    return TEACHER_USER.format(
        topic=topic,
        situation=situation,
        id_rule=id_rule(manual, topic, situation),
        length_hint=config.LENGTH_HINTS[situation],
    )


def write_prompt_files(manual=None) -> None:
    """Save the prompts the notebook appendix and the README point at."""
    config.PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    manual = manual if manual is not None else load_manual()
    config.TEACHER_PROMPT_PATH.write_text(teacher_system(manual), encoding="utf-8", newline="\n")
    config.TEACHER_USER_PATH.write_text(TEACHER_USER, encoding="utf-8", newline="\n")
    config.STUDENT_PROMPT_PATH.write_text(STUDENT_SYSTEM, encoding="utf-8", newline="\n")
