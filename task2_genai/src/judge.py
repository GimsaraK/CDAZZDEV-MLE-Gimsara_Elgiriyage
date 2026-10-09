"""LLM-as-judge for Task 2C. Four 0-2 criteria per answer, validated with Pydantic.

Gemma grades first. Nemotron grades only the rows Gemma could not, and each
verdict records its grader. Neither is the teacher or the student. A row that no judge could grade
is recorded as unjudged with the reason, and the run continues.
"""
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'Implement the Task 2C evaluation plan' (see CITATIONS.md Entry 12)

import json
import logging
import time
from typing import Callable, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from . import eval_config as ec
from .generate import _read_jsonl, _strip_fence, _write_jsonl
from .prompts import judge_system, judge_user

logger = logging.getLogger("task2.judge")


class Criterion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: int = Field(ge=0, le=ec.JUDGE_SCORE_MAX)
    reason: str = Field(min_length=1)


class JudgeVerdict(BaseModel):
    """The judge returns the four scores. The total is added up here, not asked of the model."""

    model_config = ConfigDict(extra="forbid")

    policy_correct: Criterion
    action_faithful: Criterion
    rationale_grounded: Criterion
    no_invention: Criterion

    @property
    def total(self) -> int:
        return sum(getattr(self, name).score for name in ec.JUDGE_CRITERIA)


def parse_verdict(raw: str) -> JudgeVerdict:
    """Raises json.JSONDecodeError or ValidationError when the reply does not fit the rubric."""
    return JudgeVerdict.model_validate(json.loads(_strip_fence(raw)))


def _repair_messages(system: str, user: str, raw: str, error: str) -> List[Dict[str, str]]:
    """One second chance for the same judge. It sees its reply and the validation error."""
    repair = (
        "Your previous reply did not match the required JSON. Return only the JSON object with the keys "
        + ", ".join(ec.JUDGE_CRITERIA)
        + ', each {"score": 0, 1 or 2, "reason": "..."}.\nError: '
        + error[:1000]
        + "\nPrevious reply:\n"
        + (raw or "")[:2000]
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "user", "content": repair},
    ]


def _grader_name(client) -> str:
    return "%s:%s" % (client.provider, client.model)


def judge_one(clients, system: str, user: str) -> Dict:
    """Try each judge in order, each with one repair. Never raises."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    errors = []
    for client in clients:
        raw = ""
        try:
            raw = client.complete(messages)
            verdict = parse_verdict(raw)
        # Two failure kinds, handled differently:
        #  - the judge answered but the JSON is wrong -> give the same judge one repair turn;
        #  - the call itself failed (rate limit, outage)  -> move straight on to the next judge.
        except (json.JSONDecodeError, ValidationError) as exc:
            try:
                raw = client.complete(_repair_messages(system, user, raw, str(exc)))
                verdict = parse_verdict(raw)
            except Exception as repair_exc:  # noqa: BLE001 - the next judge gets the row
                errors.append("%s: %s" % (_grader_name(client), type(repair_exc).__name__))
                logger.warning("Judge %s failed after repair: %s", _grader_name(client), repair_exc)
                continue
        except Exception as exc:  # noqa: BLE001 - rate limit, outage, or a rejected key
            errors.append("%s: %s" % (_grader_name(client), exc))
            logger.warning("Judge %s failed: %s", _grader_name(client), exc)
            continue
        return {
            "status": "judged",
            "grader": _grader_name(client),
            "scores": verdict.model_dump(),
            "total": verdict.total,
        }
    return {"status": "unjudged", "error": "; ".join(errors) or "no judge configured"}


def judge_all(
    records: List[dict],
    clients,
    manual=None,
    path=None,
    sleep: Callable[[float], None] = time.sleep,
    interval: Optional[float] = None,
) -> List[dict]:
    """Grade every model's answer on every test row.

    Judged rows already in the file are reused, so a re-run does not spend calls
    again. Unjudged rows are tried again. The file is rewritten after each call.
    """
    target = path or ec.JUDGE_SCORES_PATH
    pause = ec.JUDGE_MIN_INTERVAL_SECONDS if interval is None else interval
    system = judge_system(manual)
    # Results are keyed by (test row id, which model answered), e.g. (7, "finetuned").
    # Only successfully judged rows are reused; unjudged ones get another try on a re-run.
    cached = {(row["id"], row["model"]): row for row in _read_jsonl(target) if row.get("status") == "judged"}
    rows = dict(cached)
    # The output order is fixed (row by row, base then fine-tuned) however many rows were cached.
    order = [(record["id"], key) for record in records for key in ec.MODEL_KEYS if key in record]
    called = False
    for record in records:
        for key in ec.MODEL_KEYS:
            if key not in record or (record["id"], key) in cached:
                continue
            # Pause between calls (not before the first) to stay under the free tier's requests-per-minute.
            if called:
                sleep(pause)
            called = True
            outcome = judge_one(clients, system, judge_user(record["scenario"], record["gold"], record[key]))
            rows[(record["id"], key)] = {"id": record["id"], "model": key, **outcome}
            # Cached rows are kept in the file too, so a crash part-way loses nothing already graded.
            _write_jsonl(target, [rows[item] for item in order if item in rows])
    results = [rows[item] for item in order if item in rows]
    _write_jsonl(target, results)
    return results
